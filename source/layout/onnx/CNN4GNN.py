"""CNN4GNN.py

ONNX-based inference class for the CNN4GNN layout analysis model.

Mirrors the BoxRFDGNN interface so it can be used as a drop-in replacement
for document layout analysis pipelines.

Typical usage:

    from .CNN4GNN import CNN4GNN   # inside the layout.model package

    # All paths are optional: the bundled default model is used when omitted.
    model = CNN4GNN()

    # Or with explicit paths (stage1 path is derived from the stage2 path):
    model = CNN4GNN(
        model_cfg_path  = '/path/to/model.yaml',
        stage2_onnx_path = '/path/to/cnn4gnn_stage2.onnx',
        device_id = 0,       # -1 (default) = CPU only, 0 or more = CUDA device
        verbose   = True,    # False (default) = no print output
    )

    # page: pymupdf.Page. Same return value as BoxRFDGNN.predict.
    det_result = model.predict(page)          # [[x1, y1, x2, y2, class], ...]
    result = model.predict(page, return_raw=True)
    for g in result['groups']:
        print(g['class_name'], g['group_bbox'])
"""

import os
import math
from pathlib import Path

import anyconfig
import cv2
from ..common_util import resize_image, to_gray
import numpy as np
import pymupdf
import onnxruntime as ort

from .DefaultSorter import DefaultSorter
from .TableGridExtractorV5 import TableGridExtractorV5
from ..common_util import (
    get_text_pattern,
    get_edge_by_directional_nn,
    get_edge_matrix,
    group_node_by_edge,
)

# ---------------------------------------------------------------------------
# Default model files, used when the matching constructor arguments are None.
# SCRIPT_DIR is the layout package directory (parent of this file's folder),
# the same base the table grid model is looked up from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_CFG_PATH = (
    f'{SCRIPT_DIR}/resources/onnx/cnn4gnn_layout_medium.yaml')
DEFAULT_STAGE1_ONNX_PATH = (
    f'{SCRIPT_DIR}/resources/onnx/cnn4gnn_layout_medium_stage1.onnx')
DEFAULT_STAGE2_ONNX_PATH = (
    f'{SCRIPT_DIR}/resources/onnx/cnn4gnn_layout_medium_stage2.onnx')


# ---------------------------------------------------------------------------
# Default table grid profile (same entry as V5-SH100-4L in grid_profiles.yaml).
# Used when the matching constructor arguments are left at their defaults.
# The model file is looked up under <layout package>/resources/onnx/.
# ---------------------------------------------------------------------------
TABLE_GRID_DEFAULT_PATH = 'default'     # sentinel: use the profile filename
TABLE_GRID_PROFILE = {
    'extractor_class': TableGridExtractorV5,
    'filename'       : 'table_grid_model_v5_sh100_4L.onnx',
    'h_on_threshold' : 0.15,
    'v_on_threshold' : 0.3,
    'nms_min_dist'   : 0.01,
}


# ---------------------------------------------------------------------------
# Directional nearest-neighbour feature helpers
# Self-contained copies from CNN4GNNDataset.py -- no common_util dependency.
# Keep in sync with get_directional_margin_features and get_nn_geo_features
# in CNN4GNNDataset.py when either is changed.
# ---------------------------------------------------------------------------

def _get_directional_margin_features(bboxes_img, img_w, img_h,
                                     vertical_gap=0.3):
    """4-directional nearest-neighbour margin features, normalised by page size.

    Returns (N, 4) float32: [margin_l, margin_r, margin_t, margin_b].
    1.0 when no neighbour exists in that direction.
    """
    N = len(bboxes_img)
    if N == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if N == 1:
        return np.ones((1, 4), dtype=np.float32)

    bbox     = bboxes_img.astype(np.float32)
    center_y = (bbox[:, 1] + bbox[:, 3]) / 2.0
    height   = bbox[:, 3] - bbox[:, 1]

    cy_i = center_y[:, None].repeat(N, axis=1)
    y0_j = (bbox[:, 1] - height * vertical_gap)[None, :].repeat(N, axis=0)
    y1_j = (bbox[:, 3] + height * vertical_gap)[None, :].repeat(N, axis=0)
    y_center_flag = (cy_i < y0_j) | (cy_i > y1_j)
    eye = np.eye(N, dtype=bool)

    x0_i  = bbox[:, 0][:, None].repeat(N, axis=1)
    x1_i  = bbox[:, 2][:, None].repeat(N, axis=1)
    y0_i  = bbox[:, 1][:, None].repeat(N, axis=1)
    x0_j  = bbox[:, 0][None, :].repeat(N, axis=0)
    x1_j  = bbox[:, 2][None, :].repeat(N, axis=0)
    y0_jr = bbox[:, 1][None, :].repeat(N, axis=0)
    y1_im = bbox[:, 3][:, None].repeat(N, axis=1)
    y1_jr = bbox[:, 3][None, :].repeat(N, axis=0)

    hor_dist = (x0_i - x1_j) ** 2 + (y0_i - y0_jr) ** 2
    ver_dist = (x0_i - x0_j) ** 2 + (y0_i - y1_jr) ** 2

    INF = math.inf

    def masked(dist, *masks):
        d = dist.copy()
        m = np.zeros((N, N), dtype=bool)
        for mk in masks:
            m |= mk
        d[m] = INF
        return d

    left_d  = masked(hor_dist, y_center_flag, eye, x0_i >= x0_j)
    right_d = masked(hor_dist, y_center_flag, eye, x1_i <= x1_j)
    up_d    = masked(ver_dist, eye, y0_i  >= y0_jr)
    down_d  = masked(ver_dist, eye, y1_im <= y1_jr)

    def nearest(dm, norm):
        mins = np.min(dm, axis=1)
        res  = np.where(mins == INF, 1.0, np.sqrt(mins) / max(norm, 1.0))
        return np.clip(res, 0.0, 1.0).astype(np.float32)

    return np.stack([
        nearest(left_d,  img_w),
        nearest(right_d, img_w),
        nearest(up_d,    img_h),
        nearest(down_d,  img_h),
    ], axis=1)  # (N, 4)


def _get_nn_geo_features(bboxes_img, vertical_gap=0.3):
    """4-directional nearest-neighbour edge-geo features per node.

    Returns (N, 72) float32: [left_18, right_18, up_18, down_18].
    Zeros for directions with no neighbour.
    """
    N = len(bboxes_img)
    if N <= 1:
        return np.zeros((N, 72), dtype=np.float32)

    bbox     = bboxes_img.astype(np.float32)
    eps      = 1e-6
    center_y = (bbox[:, 1] + bbox[:, 3]) / 2.0
    height   = bbox[:, 3] - bbox[:, 1]

    cy_i = center_y[:, None].repeat(N, axis=1)
    y0_j = (bbox[:, 1] - height * vertical_gap)[None, :].repeat(N, axis=0)
    y1_j = (bbox[:, 3] + height * vertical_gap)[None, :].repeat(N, axis=0)
    y_center_flag = (cy_i < y0_j) | (cy_i > y1_j)
    eye = np.eye(N, dtype=bool)

    x0_i  = bbox[:, 0][:, None].repeat(N, axis=1)
    x1_i  = bbox[:, 2][:, None].repeat(N, axis=1)
    y0_i  = bbox[:, 1][:, None].repeat(N, axis=1)
    x0_j  = bbox[:, 0][None, :].repeat(N, axis=0)
    x1_j  = bbox[:, 2][None, :].repeat(N, axis=0)
    y0_jr = bbox[:, 1][None, :].repeat(N, axis=0)
    y1_im = bbox[:, 3][:, None].repeat(N, axis=1)
    y1_jr = bbox[:, 3][None, :].repeat(N, axis=0)

    hor_dist = (x0_i - x1_j) ** 2 + (y0_i - y0_jr) ** 2
    ver_dist = (x0_i - x0_j) ** 2 + (y0_i - y1_jr) ** 2

    INF = math.inf

    def masked(dist, *masks):
        d = dist.copy()
        m = np.zeros((N, N), dtype=bool)
        for mk in masks:
            m |= mk
        d[m] = INF
        return d

    dir_matrices = [
        masked(hor_dist, y_center_flag, eye, x0_i >= x0_j),  # left
        masked(hor_dist, y_center_flag, eye, x1_i <= x1_j),  # right
        masked(ver_dist, eye, y0_i  >= y0_jr),                # up
        masked(ver_dist, eye, y1_im <= y1_jr),                # down
    ]

    def edge_geo_np(si, oi):
        sw = max(si[2]-si[0], eps); sh = max(si[3]-si[1], eps)
        ow = max(oi[2]-oi[0], eps); oh = max(oi[3]-oi[1], eps)
        rx1 = min(si[0],oi[0]); ry1 = min(si[1],oi[1])
        rx2 = max(si[2],oi[2]); ry2 = max(si[3],oi[3])
        rw  = max(rx2-rx1, eps); rh  = max(ry2-ry1, eps)
        return np.array([
            (si[0]-oi[0])/sw, (si[1]-oi[1])/sh,
            (oi[0]-si[0])/ow, (oi[1]-si[1])/oh,
            np.log(sw/ow),    np.log(sh/oh),
            (si[0]-rx1)/sw,   (si[1]-ry1)/sh,
            (rx1-si[0])/rw,   (ry1-si[1])/rh,
            np.log(sw/rw),    np.log(sh/rh),
            (oi[0]-rx1)/ow,   (oi[1]-ry1)/oh,
            (rx1-oi[0])/rw,   (ry1-oi[1])/rh,
            np.log(ow/rw),    np.log(oh/rh),
        ], dtype=np.float32)

    result = np.zeros((N, 72), dtype=np.float32)
    for d_idx, dm in enumerate(dir_matrices):
        mins = np.min(dm, axis=1)
        nns  = np.argmin(dm, axis=1)
        for i in range(N):
            if mins[i] == INF:
                continue
            j   = int(nns[i])
            geo = edge_geo_np(bbox[i], bbox[j])
            geo = np.nan_to_num(geo, nan=0.0, posinf=1e5, neginf=-1e5)
            result[i, d_idx*18: d_idx*18+18] = geo
    return result


# ---------------------------------------------------------------------------
# FCOS decoding helper (mirrors _decode_det_boxes in train_cnn4gnn.py)
# ---------------------------------------------------------------------------

def _nms_boxes(boxes, scores, iou_thresh=0.5):
    """Greedy NMS. Returns kept (x1,y1,x2,y2) tuples."""
    if not boxes:
        return []
    order  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept   = []
    while order:
        i = order.pop(0)
        kept.append(boxes[i])
        def iou(a, b):
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            ua    = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            return inter / ua if ua > 0 else 0.0
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thresh]
    return kept


def decode_fcos(det_cls, det_reg, det_ctr, img_h, img_w,
                det_class_names, score_thresh=0.3, nms_iou=0.5):
    """Decode FCOS head outputs to detection boxes.

    Args:
        det_cls          : (num_det_classes, fH, fW) numpy float32
        det_reg          : (4, fH, fW) numpy float32, ltrb in [0,1]
        det_ctr          : (1, fH, fW) numpy float32, centerness logit
        img_h, img_w     : original image size
        det_class_names  : list of class name strings
        score_thresh     : minimum sigmoid(cls)*sigmoid(ctr) to keep
        nms_iou          : NMS IoU threshold

    Returns:
        list of dicts: {'bbox': [x1,y1,x2,y2], 'class_name': str, 'score': float}
    """
    fH, fW    = det_reg.shape[1], det_reg.shape[2]
    ctr_score = 1.0 / (1.0 + np.exp(-det_ctr[0]))    # (fH, fW)
    cls_probs = 1.0 / (1.0 + np.exp(-det_cls))        # (num_det_classes, fH, fW)

    results = []
    for ci, cls_name in enumerate(det_class_names):
        score_map = cls_probs[ci] * ctr_score          # (fH, fW)
        ys, xs    = np.where(score_map > score_thresh)
        raw_boxes, raw_scores = [], []
        for y, x in zip(ys, xs):
            l  = float(det_reg[0, y, x]) * img_w
            t  = float(det_reg[1, y, x]) * img_h
            r  = float(det_reg[2, y, x]) * img_w
            b  = float(det_reg[3, y, x]) * img_h
            cx = (x + 0.5) / fW * img_w
            cy = (y + 0.5) / fH * img_h
            x1, y1 = max(0, cx - l), max(0, cy - t)
            x2, y2 = min(img_w, cx + r), min(img_h, cy + b)
            if x2 > x1 and y2 > y1:
                raw_boxes.append((x1, y1, x2, y2))
                raw_scores.append(float(score_map[y, x]))
        for box in _nms_boxes(raw_boxes, raw_scores, nms_iou):
            results.append({
                'bbox'      : list(box),
                'class_name': cls_name,
                'score'     : raw_scores[raw_boxes.index(box)]
                              if box in raw_boxes else 0.0,
            })
    return results


# ---------------------------------------------------------------------------
# cv_pred features: stage1 FCOS detections as GNN input
#
# ONE implementation shared by save_stage1_lmdb (training lmdb) and the
# inference class, so both build exactly the same numbers.
# ---------------------------------------------------------------------------

_DEFAULT_DET_CLASSES = ['picture', 'table', 'formula']


def cv_pred_settings(model_section):
    """FCOS decoding settings used for the cv_pred features.

    model_section: the 'model' dict of the config (has 'stage1', 'stage2').
    Keys under model.stage2.cv_pred: score_thresh (default 0.3), nms_iou
    (default 0.5). The class list is model.stage1.det_classes (the order of
    the model's det_cls channels). These are independent of the inference
    class's det_score_thresh / det_nms_iou, which only affect the returned
    detections: the GNN input must not change with them.
    """
    s1 = model_section.get('stage1', {})
    cp = model_section.get('stage2', {}).get('cv_pred', {}) or {}
    return {
        'det_classes' : list(s1.get('det_classes', _DEFAULT_DET_CLASSES)),
        'score_thresh': float(cp.get('score_thresh', 0.3)),
        'nms_iou'     : float(cp.get('nms_iou', 0.5)),
    }


def compute_cv_pred_features(det_results, bboxes_img, edge_index,
                             det_class_names):
    """Node and edge features from the stage1 FCOS detections.

    Args:
        det_results     : decode_fcos() output, boxes in model-input pixels.
        bboxes_img      : (N, 4) node boxes in the same pixel coordinates.
        edge_index      : (2, E) int edge list.
        det_class_names : class names in det_cls channel order.

    Returns:
        node_feat (N, C) float32: per FCOS class, the highest score among
                  the detected boxes of that class that contain the node's
                  bbox center (0.0 when no such box).
        edge_feat (E, 1) float32: 1.0 when a single detected box (any class)
                  contains the centers of both bboxes of the edge, else 0.0.

    "Contains" is the bbox CENTER lying inside the detected box (borders
    included). The values are plain predictions of the frozen stage1, not
    trainable and never given a gradient.
    """
    bboxes_img = np.asarray(bboxes_img, dtype=np.float32).reshape(-1, 4)
    edge_index = np.asarray(edge_index).reshape(2, -1)
    N, E, C = bboxes_img.shape[0], edge_index.shape[1], len(det_class_names)
    node_feat = np.zeros((N, C), dtype=np.float32)
    edge_feat = np.zeros((E, 1), dtype=np.float32)
    if N == 0 or not det_results:
        return node_feat, edge_feat

    cls_idx = {n: i for i, n in enumerate(det_class_names)}
    unknown = sorted({d['class_name'] for d in det_results} - set(cls_idx))
    if unknown:
        raise ValueError(f'[cv_pred] detections of classes {unknown} are not '
                         f'in det_class_names {list(det_class_names)}')
    boxes  = np.array([d['bbox'] for d in det_results], dtype=np.float32)  # (K,4)
    scores = np.array([d['score'] for d in det_results], dtype=np.float32)  # (K,)
    kcls   = np.array([cls_idx[d['class_name']] for d in det_results])      # (K,)

    cx = (bboxes_img[:, 0] + bboxes_img[:, 2]) / 2.0                         # (N,)
    cy = (bboxes_img[:, 1] + bboxes_img[:, 3]) / 2.0
    inside = ((cx[None, :] >= boxes[:, 0:1]) & (cx[None, :] <= boxes[:, 2:3])
              & (cy[None, :] >= boxes[:, 1:2]) & (cy[None, :] <= boxes[:, 3:4]))  # (K,N)

    for c in range(C):
        m = inside & (kcls == c)[:, None]                                    # (K,N)
        if m.any():
            node_feat[:, c] = np.where(m, scores[:, None], 0.0).max(axis=0)
    if E > 0:
        both = inside[:, edge_index[0]] & inside[:, edge_index[1]]           # (K,E)
        edge_feat[:, 0] = both.any(axis=0).astype(np.float32)
    return node_feat, edge_feat


# ---------------------------------------------------------------------------
# OCR delegation
#
# CNN4GNN does not contain an OCR engine. Whenever it needs text for an image
# (an image input, or a PDF page without a text layer) it calls ocr_func:
#
#     ocr_func(bgr_img) -> (bboxes, texts)
#
#   bgr_img : np.ndarray (H, W, 3) uint8, BGR
#   bboxes  : N boxes [x1, y1, x2, y2] in pixel coordinates of bgr_img
#             (list of lists or np.ndarray (N, 4))
#   texts   : list of N str
#
# The boxes must be LINE level, like the text layer lines of a PDF.
# DefaultOCR (RapidOCR, then Tesseract) is used when no ocr_func is given.
# ---------------------------------------------------------------------------

class DefaultOCR:
    """Built-in OCR: RapidOCR first, then Tesseract (lazy imports).

    Raises RuntimeError with install instructions when neither is available.
    Returns (bboxes, texts) as described in the OCR delegation note above.
    """

    def __init__(self, verbose=False):
        self._rapidocr = None
        self.verbose   = verbose

    def __call__(self, bgr_img):
        # --- RapidOCR ---
        try:
            from rapidocr_onnxruntime import RapidOCR
            if self._rapidocr is None:
                self._rapidocr = RapidOCR()
            result, _ = self._rapidocr(bgr_img)
            bboxes, texts = [], []
            if result:
                for item in result:
                    quad, text = item[0], item[1]
                    if not str(text).strip():
                        continue
                    xs = [pt[0] for pt in quad]
                    ys = [pt[1] for pt in quad]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    if x2 > x1 and y2 > y1:
                        bboxes.append([float(x1), float(y1),
                                       float(x2), float(y2)])
                        texts.append(str(text))
            return bboxes, texts
        except ImportError:
            if self.verbose:
                print('[CNN4GNN] rapidocr-onnxruntime not found. '
                      'Install with: pip install rapidocr-onnxruntime')

        # --- Tesseract ---
        try:
            import pytesseract
            from collections import defaultdict
            data = pytesseract.image_to_data(
                bgr_img,
                config='--psm 3',
                output_type=pytesseract.Output.DICT,
            )
            lines = defaultdict(
                lambda: {'words': [], 'x1': [], 'y1': [], 'x2': [], 'y2': []})
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i]
                if conf < 0 or not text.strip():
                    continue
                key = (data['block_num'][i],
                       data['par_num'][i],
                       data['line_num'][i])
                x, y, w, h = (data['left'][i], data['top'][i],
                              data['width'][i], data['height'][i])
                lines[key]['words'].append(text)
                lines[key]['x1'].append(x)
                lines[key]['y1'].append(y)
                lines[key]['x2'].append(x + w)
                lines[key]['y2'].append(y + h)
            bboxes, texts = [], []
            for v in lines.values():
                line_text = ' '.join(v['words']).strip()
                if not line_text:
                    continue
                x1, y1 = min(v['x1']), min(v['y1'])
                x2, y2 = max(v['x2']), max(v['y2'])
                if x2 > x1 and y2 > y1:
                    bboxes.append([float(x1), float(y1),
                                   float(x2), float(y2)])
                    texts.append(line_text)
            return bboxes, texts
        except ImportError:
            if self.verbose:
                print('[CNN4GNN] pytesseract not found. '
                      'Install with: pip install pytesseract  '
                      'and the Tesseract binary: sudo apt install tesseract-ocr')

        raise RuntimeError(
            '[CNN4GNN] No OCR engine available. '
            'Install at least one of:\n'
            '  pip install rapidocr-onnxruntime\n'
            '  pip install pytesseract  +  sudo apt install tesseract-ocr')


def normalize_ocr_result(result, img_hw):
    """Validate and clean the return value of an ocr_func.

    Applied to every ocr_func, so a custom function gets the same cleaning as
    the built-in one: boxes are clipped to the image, and entries with a
    blank text, a non-finite value or an empty box are dropped.

    Args:
        result : the value returned by ocr_func, (bboxes, texts).
        img_hw : (height, width) of the image given to ocr_func.

    Returns:
        (boxes, texts): np.ndarray (N, 4) float32 and list of N str.

    Raises:
        ValueError when the result does not follow the ocr_func contract.
    """
    try:
        bboxes, texts = result
    except (TypeError, ValueError):
        raise ValueError(
            '[CNN4GNN] ocr_func must return a pair (bboxes, texts)') from None
    texts = [str(t) for t in texts]
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.size == 0:
        if texts:
            raise ValueError(
                f'[CNN4GNN] ocr_func returned no bboxes but {len(texts)} texts')
        return np.zeros((0, 4), dtype=np.float32), []
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f'[CNN4GNN] ocr_func bboxes must have shape (N, 4), '
            f'got {boxes.shape}')
    if len(texts) != len(boxes):
        raise ValueError(
            f'[CNN4GNN] ocr_func returned {len(boxes)} bboxes but '
            f'{len(texts)} texts')

    h, w = img_hw
    finite = np.isfinite(boxes).all(axis=1)
    boxes = np.nan_to_num(boxes)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    keep = (finite
            & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            & np.array([bool(t.strip()) for t in texts]))
    return (np.ascontiguousarray(boxes[keep]),
            [t for t, k in zip(texts, keep) if k])


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

def softmax(x):
    """Numerically stable softmax over the last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CNN4GNN:
    """ONNX-based CNN4GNN layout analysis model.

    The network is deployed as two ONNX models. Stage 1 (backbone +
    decoders) runs once per page.  Stage 2 (GNN heads) receives the
    intermediate tensors from Stage 1 directly, so the backbone is never
    re-executed (see create_data / is_ocr_needed / predict).

    Parameters
    ----------
    model_cfg_path    : str or None -- path to the model YAML config.
                        None uses DEFAULT_MODEL_CFG_PATH.
    stage1_onnx_path  : str or None -- stage-1 ONNX path. When None and
                        stage2_onnx_path is given, derived from it
                        (x_stage2.onnx -> x_stage1.onnx). When both are None,
                        DEFAULT_STAGE1_ONNX_PATH is used.
    stage2_onnx_path  : str or None -- stage-2 ONNX path. None uses
                        DEFAULT_STAGE2_ONNX_PATH.
    edge_threshold    : float -- minimum edge probability to classify as connected.
    det_score_thresh  : float -- minimum FCOS detection score.
    det_nms_iou       : float -- NMS IoU threshold for FCOS detections.
    device_id         : int   -- CUDA device index. -1 (default) = GPU is not
                        used (CPU only). 0 or more = use that CUDA device
                        when available.
    verbose           : bool  -- False (default) = the constructor and the
                        built-in OCR print no messages.
    table_grid_path   : str or None -- table grid ONNX. Default: the profile
                        file resources/onnx/table_grid_model_v5_sh100_4L.onnx.
                        None disables table grid extraction. The grid model
                        always runs on CPU.
    table_grid_extractor_cls, table_h_on_threshold, table_v_on_threshold,
    table_nms_min_dist : grid extractor settings. None means the value of
                        TABLE_GRID_PROFILE (V5-SH100-4L).
    use_sort          : bool -- True (default) = predict() puts the groups in
                        reading order with DefaultSorter (page-header first,
                        page-footer last, body by column and position). The
                        per call keyword do_sort=False turns it off. False =
                        groups stay in the order of their first text node.
    ocr_func          : callable or None -- OCR used for image inputs and for
                        PDF pages without a text layer:
                        ocr_func(bgr_img) -> (bboxes, texts), with bgr_img a
                        BGR uint8 (H, W, 3) array, bboxes N line level boxes
                        [x1, y1, x2, y2] in pixels of bgr_img and texts N
                        str. None uses the built-in DefaultOCR (RapidOCR,
                        then Tesseract). Exceptions raised by ocr_func are
                        not caught. Can be replaced later through the
                        attribute self.ocr_func.
    """

    def __init__(self,
                 model_cfg_path    = None,
                 stage1_onnx_path  = None,
                 stage2_onnx_path  = None,
                 edge_threshold    = 0.55,
                 det_score_thresh  = 0.3,
                 det_nms_iou       = 0.5,
                 device_id         = -1,
                 table_grid_path   = TABLE_GRID_DEFAULT_PATH,
                 table_grid_extractor_cls = None,
                 table_h_on_threshold = None,
                 table_v_on_threshold = None,
                 table_nms_min_dist   = None,
                 ocr_func             = None,
                 use_sort             = True,
                 verbose              = False):

        self.verbose = verbose

        # Default paths for everything that is not given.
        if model_cfg_path is None:
            model_cfg_path = DEFAULT_MODEL_CFG_PATH

        if stage2_onnx_path is None:
            stage2_onnx_path = DEFAULT_STAGE2_ONNX_PATH
            if stage1_onnx_path is None:
                stage1_onnx_path = DEFAULT_STAGE1_ONNX_PATH
        elif stage1_onnx_path is None:
            # Auto-derive stage1 path from the given stage2 path.
            # Convention: stage2 path ends with '_stage2.onnx',
            #             stage1 path ends with '_stage1.onnx'.
            if '_stage2.onnx' in stage2_onnx_path:
                stage1_onnx_path = stage2_onnx_path.replace(
                    '_stage2.onnx', '_stage1.onnx')
                self._log(f'[CNN4GNN] Auto-derived stage1 path: '
                          f'{stage1_onnx_path}')
            else:
                raise ValueError(
                    'stage2_onnx_path does not contain "_stage2.onnx". '
                    'Please provide stage1_onnx_path explicitly.')

        # Optional hook that can replace the stage2 logits before decoding
        # (see _apply_logits_hook). None = the model output is used as is.
        self.logits_hook      = None

        self.edge_threshold   = edge_threshold
        self.det_score_thresh = det_score_thresh
        self.det_nms_iou      = det_nms_iou
        self.use_sort         = use_sort
        self.sorter           = DefaultSorter()

        # ------------------------------------------------------------------
        # Load model config
        # ------------------------------------------------------------------
        with open(model_cfg_path, 'rb') as f:
            self.model_cfg = anyconfig.load(f)

        # Config is rooted under 'train_cnn4gnn' key
        train_cfg = self.model_cfg.get('train_cnn4gnn', self.model_cfg)

        data_cfg = train_cfg.get('data', {})
        self.class_names = data_cfg.get('class_list', [])
        self.class_map   = {n: i for i, n in enumerate(self.class_names)}
        # Alias kept so that code written against BoxRFDGNN keeps working.
        self.data_class_names = self.class_names
        self.data_class_map   = self.class_map

        self.class_priority_list = data_cfg.get('class_priority', [])
        self.det_classes = data_cfg.get('det_classes', ['picture', 'table', 'formula'])

        ds_cfg = train_cfg.get('dataset', {})
        self.vertical_gap = float(ds_cfg.get('vertical_gap',
                                   data_cfg.get('vertical_gap', 0.3)))
        self.max_nodes    = int(ds_cfg.get('max_nodes', 5000))
        # Render scale of the SOURCE page image (1.0 = page size in points,
        # 72 dpi). The training data (PKL images) are page-size renders; the
        # model input is that image resized to img_w x img_h by resize_image,
        # exactly as CNN4GNNDataset does. Must equal the scale the PKL
        # images were made with (render_pkl_images.py --scale).
        self.render_scale = float(ds_cfg.get('render_scale', 1.0))

        # stage1_feat_type and feature_set determine which tensors stage2 needs
        model_s2 = train_cfg.get('model', {}).get('stage2', {})
        self.stage1_feat_type = model_s2.get('stage1_feat_type', 'result')
        self.feature_set      = set(model_s2.get('feature_set', []) or [])
        # cv_pred_*: GNN inputs built from the stage1 FCOS output (see
        # compute_cv_pred_features); needs the split stage1 / stage2 models.
        self.cv_node = 'cv_pred_node' in self.feature_set
        self.cv_edge = 'cv_pred_edge' in self.feature_set
        self.cv_pred = cv_pred_settings(train_cfg.get('model', {}))

        # ------------------------------------------------------------------
        # ONNX sessions
        # ------------------------------------------------------------------
        # device_id < 0 (default) means CPU only, 0 or more selects a CUDA device.
        device_id = int(device_id)
        providers = ([('CUDAExecutionProvider', {'device_id': device_id}),
                      'CPUExecutionProvider']
                     if device_id >= 0 else ['CPUExecutionProvider'])
        ort.set_default_logger_severity(3)

        self.session_s1   = ort.InferenceSession(stage1_onnx_path,
                                                 providers=providers)
        self.session_s2   = ort.InferenceSession(stage2_onnx_path,
                                                 providers=providers)
        # Stage1 outputs / stage2 inputs are addressed BY NAME, never by
        # position or hard-coded key: the exported tensor set changes
        # whenever the model's export signature does (e.g. the 9-output
        # layout with mask_pred became the 11-output layout with
        # region_pred/text_pred), and positional indices silently break.
        self._s1_out_names = [o.name for o in self.session_s1.get_outputs()]
        self._s2_in_names  = [i.name for i in self.session_s2.get_inputs()]
        self._s2_out_names = [o.name for o in self.session_s2.get_outputs()]
        self._log(f'[CNN4GNN] Stage 1: {stage1_onnx_path}')
        self._log(f'[CNN4GNN] Stage 2: {stage2_onnx_path}')

        # ------------------------------------------------------------------
        # Image size: inferred from ONNX input shape (most reliable).
        # stage1 input: (1, 1, H, W). If exported with dynamic H/W axes the
        # shape values are strings -- fall back to config or safe default.
        # ------------------------------------------------------------------
        def _infer_img_size(session):
            shape = session.get_inputs()[0].shape
            if len(shape) >= 4:
                h, w = shape[2], shape[3]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return w, h
            return None

        inferred = _infer_img_size(self.session_s1)
        if inferred is not None:
            self.img_w, self.img_h = inferred
        else:
            ds_sz = ds_cfg.get('image_size', None)
            if ds_sz is None:
                raise KeyError(
                    "[CNN4GNN] 'data.image_size' is required in the model config "
                    "but was not found. Please set it explicitly "
                    "(e.g. data.image_size: [300, 300]).")
            self.img_w, self.img_h = int(ds_sz[0]), int(ds_sz[1])

        self.ocr_func = (ocr_func if ocr_func is not None
                         else DefaultOCR(verbose=verbose))

        # ------------------------------------------------------------------
        # Table grid extractor (enabled by default with the default profile;
        # disabled when table_grid_path is None). The grid model always runs on CPU: BoxRFDGNN does the same
        # and GPU numerics can move lines near the thresholds.
        # conn_onnx_path is None, so connectivity is not predicted.
        # ------------------------------------------------------------------
        self.table_grid_extractor = None
        if table_grid_path == TABLE_GRID_DEFAULT_PATH:
            table_grid_path = (f'{SCRIPT_DIR}/resources/onnx/'
                               f'{TABLE_GRID_PROFILE["filename"]}')
        # Unset options fall back to the default profile.
        if table_grid_extractor_cls is None:
            table_grid_extractor_cls = TABLE_GRID_PROFILE['extractor_class']
        if table_h_on_threshold is None:
            table_h_on_threshold = TABLE_GRID_PROFILE['h_on_threshold']
        if table_v_on_threshold is None:
            table_v_on_threshold = TABLE_GRID_PROFILE['v_on_threshold']
        if table_nms_min_dist is None:
            table_nms_min_dist = TABLE_GRID_PROFILE['nms_min_dist']

        if table_grid_path is not None:
            if not os.path.exists(table_grid_path):
                raise FileNotFoundError(
                    f'[CNN4GNN] table grid model not found: {table_grid_path}')
            if 'table' not in self.class_names:
                self._log('[CNN4GNN] Warning: "table" is not in class_list, '
                          'table grid extraction is disabled.')
            else:
                self.table_grid_extractor = table_grid_extractor_cls(
                    table_grid_path,
                    conn_onnx_path = None,
                    h_on_threshold = table_h_on_threshold,
                    v_on_threshold = table_v_on_threshold,
                    nms_min_dist   = table_nms_min_dist,
                )
                self._log(f'[CNN4GNN] Table grid: {table_grid_path}')

        self._log(f'[CNN4GNN] Classes   : {self.class_names}')
        self._log(f'[CNN4GNN] Image size: {self.img_w}x{self.img_h}')
        self._log(f'[CNN4GNN] Source render scale: {self.render_scale} '
                  f'(page image resized to the model input size, as in the '
                  f'training dataset)')

    def _log(self, msg):
        """print(msg) only when verbose is True."""
        if getattr(self, 'verbose', False):
            print(msg)

    # ------------------------------------------------------------------
    # Internal: page -> image array
    # ------------------------------------------------------------------

    def _render_source(self, page, scale=None):
        """Render the page like the training data: COLOR, page size
        (render_scale), no resizing yet. Returns a BGR uint8 array
        (src_h, src_w, 3). scale overrides render_scale (used for the
        scale 1 table crops).

        The PKL image is this color render stored as JPEG; the training
        dataset decodes it with IMREAD_COLOR (BGR), resizes it with
        resize_image and only then applies to_gray (see
        CNN4GNNDataset._process_raw). _source_to_image does the same here.
        """
        s   = self.render_scale if scale is None else float(scale)
        mat = pymupdf.Matrix(s, s)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3)
        return np.ascontiguousarray(rgb[:, :, ::-1])      # RGB -> BGR

    def _source_to_image(self, src):
        """BGR uint8 source image -> model input (H, W) float32 in [0, 1]:
        resize_image to (img_w, img_h), to_gray, min-max, as the dataset."""
        arr = resize_image(src, (self.img_w, self.img_h))
        arr = to_gray(arr).astype(np.float32) / 255.0
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr  # (H, W) float32

    def _page_to_image(self, page):
        """Render a pymupdf.Page to the model input (H, W) float32."""
        return self._source_to_image(self._render_source(page))

    def _prepare_page(self, page):
        """Everything predict() derives from a PDF page before the model.

        Returns a dict: image (H, W) float32, bboxes_px (N,4) image-pixel
        coords, bboxes_pt (N,4) PDF points, bboxes_src (N,4) coordinates of
        the SOURCE image (the coordinate system the training dataset builds
        its edges in), texts.
        """
        src = self._render_source(page)
        sh, sw = src.shape[:2]
        bboxes_px, bboxes_pt, texts, bboxes_src = self._extract_bboxes_full(
            page, src_wh=(sw, sh))
        return {'image': self._source_to_image(src), 'src': src,
                'bboxes_px': bboxes_px,
                'bboxes_pt': bboxes_pt, 'bboxes_src': bboxes_src,
                'texts': texts}

    # ------------------------------------------------------------------
    # Internal: extract PDF text bboxes and texts
    # ------------------------------------------------------------------

    def _extract_bboxes(self, page):
        """Extract line-level bboxes and text from PDF text layer.

        Returns
        -------
        bboxes_px : np.ndarray (N, 4) float32 -- image-pixel coords (CNN input scale)
        bboxes_pt : np.ndarray (N, 4) float32 -- PDF point coords (original page scale)
        texts     : list[str] of length N
        """
        bboxes_px, bboxes_pt, texts, _ = self._extract_bboxes_full(page)
        return bboxes_px, bboxes_pt, texts

    def _extract_bboxes_full(self, page, src_wh=None):
        """_extract_bboxes plus bboxes_src: the boxes in pixels of the
        source image of size src_wh = (src_w, src_h) (None -> zeros (0,4)
        placeholder equal to bboxes_px). Same scaling the dataset uses:
        pt * (src_size / page_size)."""
        pw, ph = page.rect.width, page.rect.height
        sx = self.img_w / pw
        sy = self.img_h / ph
        if src_wh is not None:
            ux, uy = src_wh[0] / pw, src_wh[1] / ph

        page_dict = page.get_text('dict',
                                  flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        bboxes_px, bboxes_pt, texts, bboxes_src = [], [], [], []
        for block in page_dict.get('blocks', []):
            if block.get('type') != 0:
                continue
            for line in block.get('lines', []):
                bbox = line.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                text = ''.join(s.get('text', '') for s in line.get('spans', []))
                # Skip lines that are empty or whitespace-only (matches
                # CNN4GNNDataset filtering so train/inference inputs are consistent)
                if not text.strip():
                    continue
                bboxes_pt.append([x1,    y1,    x2,    y2   ])  # PDF point coords
                bboxes_px.append([x1*sx, y1*sy, x2*sx, y2*sy])  # CNN input coords
                if src_wh is not None:
                    bboxes_src.append([x1*ux, y1*uy, x2*ux, y2*uy])
                texts.append(text)

        if not bboxes_px:
            z = np.zeros((0, 4), dtype=np.float32)
            return z, z.copy(), [], z.copy()
        px = np.array(bboxes_px, dtype=np.float32)
        src = (np.array(bboxes_src, dtype=np.float32)
               if src_wh is not None else px)
        return px, np.array(bboxes_pt, dtype=np.float32), texts, src

    # ------------------------------------------------------------------
    # Internal: build edges
    # ------------------------------------------------------------------

    def run_stage1(self, image_np):
        """Run stage 1 only.

        image_np (1,1,H,W) float32. Returns a dict output name -> array, as
        exported (batch dimension already removed), for example region_pred,
        text_pred, det_cls. Pass it to run_onnx(stage1=...) to skip stage 1.
        """
        s1_out = self.session_s1.run(None, {'image': image_np})
        return dict(zip(self._s1_out_names, s1_out))

    def run_onnx(self, image_np, bboxes_px, union_bboxes, edge_index,
                 text_patterns, margin_features=None, nn_geo_features=None,
                 return_features=False, stage1=None):
        """Run the ONNX model(s) on prepared arrays.

        image_np (1,1,H,W) float32; bboxes_px (N,4); union_bboxes (E,4);
        edge_index (2,E) int64; text_patterns (N,42); margin_features /
        nn_geo_features only when the model's feature_set needs them.
        Returns (node_logits, edge_logits, det_cls, det_reg, det_ctr).

        return_features=True appends a sixth element, a dict
        {'node_head_in': (N, D) or None, 'edge_head_in': (E, D) or None}:
        the inputs of the classification heads. They are None when the
        stage2 ONNX was exported without them (export2onnx.return_features).

        stage1: dict returned by run_stage1() for the same image. When
        given, stage 1 is not run again.
        """
        features = {'node_head_in': None, 'edge_head_in': None}
        # ---- Stage 1: backbone + decoders (skipped when given) ----
        s1 = stage1 if stage1 is not None else self.run_stage1(image_np)

        def _s1_get(*names):
            """First stage1 output present under any of `names`
            (later names are legacy aliases)."""
            for n in names:
                if n in s1:
                    return s1[n]
            raise KeyError(
                f'[CNN4GNN] stage1 ONNX has no output among {names}; '
                f'it provides {self._s1_out_names}')

        region_pred_np     = _s1_get('region_pred', 'mask_pred')  # (C, H, W)
        border_pred_np     = _s1_get('border_pred')               # (1, H, W)
        # 'dec-all' exports mask_dec_feat_0..n-1 instead of a single
        # mask_dec_feat; every such output is forwarded by name.
        mask_dec_feats_np  = {n: v for n, v in s1.items()
                              if n == 'mask_dec_feat'
                              or n.startswith('mask_dec_feat_')}
        border_dec_feat_np = _s1_get('border_dec_feat') # (F, H, W) or (0,)
        det_cls_np         = _s1_get('det_cls')
        det_reg_np         = _s1_get('det_reg')
        det_ctr_np         = _s1_get('det_ctr')

        # ---- Stage 2: GNN heads (no backbone re-execution) ----
        # Feed exactly the inputs this stage2 graph declares. The same
        # tensor is offered under its legacy name too (mask_pred), so
        # ONNX files exported before the region_pred rename still run.
        s2_candidates = {
            'region_pred'    : region_pred_np,
            'mask_pred'      : region_pred_np,
            'border_pred'    : border_pred_np,
            'border_dec_feat': border_dec_feat_np,
            'bboxes'         : bboxes_px,
            'union_bboxes'   : union_bboxes,
            'edge_index'     : edge_index,
            'text_patterns'  : text_patterns,
        }
        s2_candidates.update(mask_dec_feats_np)
        # 'image-line-edge': page profiles are stage1 outputs.
        for _ln in ('line_row', 'line_col'):
            if _ln in s1:
                s2_candidates[_ln] = s1[_ln]
        if margin_features is not None:
            s2_candidates['margin_features'] = margin_features
        if nn_geo_features is not None:
            s2_candidates['nn_geo_features'] = nn_geo_features
        if self.cv_node or self.cv_edge:
            cv_n, cv_e = self._cv_pred_features(
                det_cls_np, det_reg_np, det_ctr_np, bboxes_px, edge_index)
            s2_candidates['cv_node_features'] = cv_n
            s2_candidates['cv_edge_features'] = cv_e

        missing = [n for n in self._s2_in_names if n not in s2_candidates]
        if missing:
            raise RuntimeError(
                f'[CNN4GNN] stage2 ONNX needs inputs {missing} that '
                f'inference cannot supply. Its inputs: '
                f'{self._s2_in_names}; available: '
                f'{sorted(s2_candidates)}. If these are margin_features/'
                f'nn_geo_features, model.stage2.feature_set in the '
                f'config does not match the exported model.')
        s2_inputs = {n: s2_candidates[n] for n in self._s2_in_names}

        s2_out = self.session_s2.run(None, s2_inputs)
        node_logits, edge_logits = s2_out[0], s2_out[1]
        _s2o = dict(zip(self._s2_out_names, s2_out))
        for _k in features:
            features[_k] = _s2o.get(_k)
        if return_features:
            return (node_logits, edge_logits,
                    det_cls_np, det_reg_np, det_ctr_np, features)
        return node_logits, edge_logits, det_cls_np, det_reg_np, det_ctr_np

    def _apply_logits_hook(self, hook, ctx):
        """Let `hook` replace the stage2 logits; return (node, edge).

        hook(ctx) is called with a dict (see predict) and returns None (keep
        the model output) or a dict with optional 'node_logits' /
        'edge_logits' arrays replacing the respective output. A replacement
        must have the shape of the original and be finite; anything else is
        an error, never silently ignored.
        """
        node_logits, edge_logits = ctx['node_logits'], ctx['edge_logits']
        res = hook(ctx)
        if res is None:
            return node_logits, edge_logits
        if not isinstance(res, dict):
            raise TypeError(
                f'[CNN4GNN] logits hook must return None or a dict with '
                f"'node_logits' / 'edge_logits', got {type(res).__name__}")
        unknown = set(res) - {'node_logits', 'edge_logits'}
        if unknown:
            raise KeyError(
                f'[CNN4GNN] logits hook returned unknown keys {sorted(unknown)}')
        out = {'node_logits': node_logits, 'edge_logits': edge_logits}
        for k in ('node_logits', 'edge_logits'):
            v = res.get(k)
            if v is None:
                continue
            v = np.asarray(v, dtype=np.float32)
            if v.shape != out[k].shape:
                raise ValueError(
                    f'[CNN4GNN] logits hook returned {k} of shape {v.shape}, '
                    f'expected {out[k].shape}')
            if not np.isfinite(v).all():
                raise ValueError(
                    f'[CNN4GNN] logits hook returned non-finite {k}')
            out[k] = v
        return out['node_logits'], out['edge_logits']

    def _cv_pred_features(self, det_cls_np, det_reg_np, det_ctr_np,
                          bboxes_px, edge_index):
        """cv_node_features (N, C) and cv_edge_features (E, 1) from the
        stage1 FCOS outputs, in model-input pixel coordinates."""
        names = self.cv_pred['det_classes']
        if det_cls_np.shape[0] != len(names):
            raise ValueError(
                f'[CNN4GNN] stage1 det_cls has {det_cls_np.shape[0]} classes '
                f'but model.stage1.det_classes lists {len(names)}: {names}')
        dets = decode_fcos(det_cls_np, det_reg_np, det_ctr_np,
                           self.img_h, self.img_w, names,
                           score_thresh=self.cv_pred['score_thresh'],
                           nms_iou=self.cv_pred['nms_iou'])
        return compute_cv_pred_features(dets, bboxes_px, edge_index, names)

    def _build_stage2_inputs(self, bboxes_px, texts, bboxes_src=None):
        """Everything stage2 needs besides the stage1 outputs and the boxes.

        Returns (edge_index (2,E) int64, union_bboxes (E,4) float32,
        text_patterns (N,42) float32, margin_features (N,4) or None,
        nn_geo_features (N,72) or None). bboxes_src: boxes in source image
        pixels for the edge search (see _build_edges). Used by predict() and by the
        pipeline-gap diagnosis so both build these inputs by the same code.
        """
        # ------------------------------------------------------------------
        # Edges
        # ------------------------------------------------------------------
        edge_index, union_bboxes = self._build_edges(bboxes_px, bboxes_src)

        # ------------------------------------------------------------------
        # Text pattern features (N, 42)
        # ------------------------------------------------------------------
        def _content_features(text):
            n = len(text)
            if n == 0:
                return [0.0, 0.0]
            num_ratio = sum(c.isdigit() for c in text) / n
            sym_ratio = sum(
                not c.isalnum() and not c.isspace() for c in text) / n
            return [float(num_ratio), float(sym_ratio)]

        pattern_vecs  = [get_text_pattern(t, return_vector=True) for t in texts]
        content_vecs  = [_content_features(t) for t in texts]
        text_patterns = np.array(
            [p + c for p, c in zip(pattern_vecs, content_vecs)],
            dtype=np.float32)  # (N, 42)

        # ------------------------------------------------------------------
        # Optional extra node features (margin, nn_geo)
        # ------------------------------------------------------------------
        margin_features = None
        nn_geo_features = None

        if 'margin' in self.feature_set:
            margin_features = _get_directional_margin_features(
                bboxes_px, img_w=self.img_w, img_h=self.img_h,
                vertical_gap=self.vertical_gap,
            )  # (N, 4) float32

        if 'nn_geo' in self.feature_set:
            nn_geo_features = _get_nn_geo_features(
                bboxes_px,
                vertical_gap=self.vertical_gap,
            )  # (N, 72) float32
        return (edge_index, union_bboxes, text_patterns,
                margin_features, nn_geo_features)

    def _build_edges(self, bboxes_px, bboxes_edge=None):
        """Build directed edges using get_edge_by_directional_nn.

        bboxes_edge: boxes the nearest-neighbour search runs on. The training
        dataset searches in SOURCE image coordinates (bboxes_src), which is
        not the same graph as in resized image coordinates when the source
        is not square. None -> bboxes_px. The union boxes always come from
        bboxes_px (image coordinates), as in the dataset.

        Returns
        -------
        edge_index   : np.ndarray (2, E) int64
        union_bboxes : np.ndarray (E, 4) float32
        """
        N = len(bboxes_px)
        if N == 0:
            return (np.zeros((2, 0), dtype=np.int64),
                    np.zeros((0, 4),  dtype=np.float32))

        edge_list, _ = get_edge_by_directional_nn(
            bboxes_px if bboxes_edge is None else bboxes_edge,
            dist_threshold=self.max_nodes,
            vertical_gap=self.vertical_gap,
        )
        if not edge_list:
            return (np.zeros((2, 0), dtype=np.int64),
                    np.zeros((0, 4),  dtype=np.float32))

        # Add both directions
        edge_pairs = edge_list + [(j, i) for i, j in edge_list]
        edge_index = np.array(edge_pairs, dtype=np.int64).T  # (2, E)

        src, dst = edge_index[0], edge_index[1]
        union_bboxes = np.stack([
            np.minimum(bboxes_px[src, 0], bboxes_px[dst, 0]),
            np.minimum(bboxes_px[src, 1], bboxes_px[dst, 1]),
            np.maximum(bboxes_px[src, 2], bboxes_px[dst, 2]),
            np.maximum(bboxes_px[src, 3], bboxes_px[dst, 3]),
        ], axis=1).astype(np.float32)

        return edge_index, union_bboxes

    # ------------------------------------------------------------------
    # Internal: OCR-based bbox extraction (delegated to self.ocr_func)
    # ------------------------------------------------------------------

    def _run_ocr(self, bgr_img, ocr_func=None):
        """Call ocr_func (default self.ocr_func) on a BGR image and clean the
        result.

        Returns (boxes, texts): boxes (N, 4) float32 in pixels of bgr_img.
        Exceptions of ocr_func are not caught.
        """
        fn = ocr_func if ocr_func is not None else self.ocr_func
        return normalize_ocr_result(fn(bgr_img), bgr_img.shape[:2])

    def _extract_bboxes_ocr(self, bgr_img, ocr_func=None):
        """Line level bboxes and texts of an image input, via ocr_func.

        The image itself is the page, so the page units are image pixels.

        Returns
        -------
        bboxes_px : np.ndarray (N, 4) float32 -- CNN input scale coords
        bboxes_pt : np.ndarray (N, 4) float32 -- original image pixel coords
        texts     : list[str] of length N
        """
        H_orig, W_orig = bgr_img.shape[:2]
        sx = self.img_w / W_orig
        sy = self.img_h / H_orig

        bboxes_pt, texts = self._run_ocr(bgr_img, ocr_func)
        if not texts:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32),
                    [])
        bboxes_px = bboxes_pt * np.array([sx, sy, sx, sy],  # CNN input scale
                                         dtype=np.float32)
        return bboxes_px, bboxes_pt, texts

    def _extract_bboxes_ocr_page(self, src, page_size, ocr_func=None):
        """Same as _extract_bboxes_ocr for a PDF page without a text layer.

        OCR runs on the full resolution source render (src, render_scale), not
        on the downscaled model input. The boxes are converted back to the
        coordinate systems of a normal PDF page: PDF points, model input
        pixels and source pixels.

        Returns
        -------
        bboxes_px, bboxes_pt, texts, bboxes_src  (same order as
        _extract_bboxes_full)
        """
        sh, sw = src.shape[:2]
        pw, ph = page_size
        boxes_src, texts = self._run_ocr(src, ocr_func)
        if not texts:
            z = np.zeros((0, 4), dtype=np.float32)
            return z, z.copy(), [], z.copy()
        to_pt = np.array([pw / sw, ph / sh, pw / sw, ph / sh],
                         dtype=np.float32)
        to_px = np.array([self.img_w / sw, self.img_h / sh,
                          self.img_w / sw, self.img_h / sh], dtype=np.float32)
        return boxes_src * to_px, boxes_src * to_pt, texts, boxes_src

    # ------------------------------------------------------------------
    # Internal: image array preparation (image path)
    # ------------------------------------------------------------------

    def _image_to_array(self, bgr_img):
        """Convert a BGR uint8 ndarray to the normalised float32 input
        expected by the model: (H, W) float32 in [0, 1], resized to
        (self.img_h, self.img_w).
        """
        gray = to_gray(bgr_img)
        gray = resize_image(gray, (self.img_w, self.img_h))
        arr  = gray.astype(np.float32) / 255.0
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr  # (H, W) float32

    # ------------------------------------------------------------------
    # Public: postprocess (logits -> groups), usable without ONNX sessions
    # ------------------------------------------------------------------

    @classmethod
    def for_postprocess(cls, model_cfg_path, edge_threshold=0.55):
        """Instance that can only run postprocess() (no ONNX sessions).

        Used to turn cached stage2 logits into detections without exporting
        or running any ONNX model.
        """
        self = object.__new__(cls)
        self.verbose = False
        self.edge_threshold = edge_threshold
        with open(model_cfg_path, 'rb') as f:
            self.model_cfg = anyconfig.load(f)
        train_cfg = self.model_cfg.get('train_cnn4gnn', self.model_cfg)
        data_cfg = train_cfg.get('data', {})
        self.class_names         = data_cfg.get('class_list', [])
        self.class_map           = {n: i for i, n in enumerate(self.class_names)}
        self.data_class_names    = self.class_names
        self.data_class_map      = self.class_map
        self.class_priority_list = data_cfg.get('class_priority', [])
        self.table_grid_extractor = None
        return self

    def postprocess(self, node_logits, edge_logits, edge_index, bboxes_pt,
                    edge_threshold=None):
        """Stage2 logits -> groups (each with 'class_name').

        node_logits : (N, num_node_classes) float
        edge_logits : (E, num_edge_classes) float
        edge_index  : (2, E) int64
        bboxes_pt   : (N, 4) node bboxes in PDF point coordinates
        """
        return self._postprocess_full(node_logits, edge_logits, edge_index,
                                      bboxes_pt, edge_threshold)[0]

    def _postprocess_full(self, node_logits, edge_logits, edge_index,
                          bboxes_pt, edge_threshold=None):
        """postprocess() plus the intermediate predictions.

        Returns (groups, node_labels (N,), node_scores (N,),
        edge_labels (E,)).
        """
        if edge_threshold is None:
            edge_threshold = self.edge_threshold
        N = node_logits.shape[0]
        if N == 0:
            return ([], np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.int64))

        # Node classification
        node_probs           = softmax(node_logits)
        predicted_node_cls   = node_probs.argmax(axis=1)
        predicted_node_score = node_probs[np.arange(N), predicted_node_cls]

        # Edge classification
        E = edge_index.shape[1] if edge_index.ndim == 2 else 0
        if E > 0 and edge_logits.size > 0:
            edge_probs            = softmax(edge_logits)
            predicted_edge_labels = (edge_probs[:, 1] > edge_threshold
                                     ).astype(np.int64)
        else:
            predicted_edge_labels = np.zeros(E, dtype=np.int64)

        # Group nodes by edges (original-scale bboxes)
        edge_matrix = get_edge_matrix(N, edge_index, predicted_edge_labels)
        groups = group_node_by_edge(
            predicted_node_cls, predicted_node_score,
            edge_matrix, np.asarray(bboxes_pt).tolist(),
            self.class_priority_list,
        )
        for g in groups:
            cls_idx = int(g['group_class'])
            g['class_name'] = (self.class_names[cls_idx]
                               if 0 <= cls_idx < len(self.class_names)
                               else str(cls_idx))
        return (groups, predicted_node_cls, predicted_node_score,
                predicted_edge_labels)

    # ------------------------------------------------------------------
    # data_dict: per page container shared by is_ocr_needed() and predict()
    # ------------------------------------------------------------------

    def create_data(self, page, use_ocr=False, verbose=False, ocr_func=None):
        """Build the per page data_dict used by is_ocr_needed() and predict().

        Everything that is derived from the page before the neural network
        is stored here, together with (lazily) the stage 1 outputs. Passing
        the same data_dict to is_ocr_needed() and then to predict() runs the
        stage 1 network only once per page.

        Keys follow the BoxRFDGNN data_dict where they exist:
            bboxes       (N, 4) float32 line bboxes in page units
                         (PDF points for a PDF page, pixels for an image)
            text         list of N line texts
            image        BGR uint8 source image the crops are cut from
            image_scale  pixels of image per page unit (render_scale)
            page_size    (w, h) of the page in the units of bboxes
        CNN4GNN specific keys:
            model_image  (H, W) float32 model input
            bboxes_px    (N, 4) bboxes in model input pixels
            bboxes_src   (N, 4) bboxes in source image pixels, or None
            source_size  (w, h) of the page or image, used to check that a
                         data_dict belongs to the page given to predict()
            stage1       dict of stage 1 outputs, None until computed

        The data_dict is meant to live for one page only: stage1 holds the
        full set of stage 1 tensors.

        Args:
            page    : pymupdf.Page or np.ndarray (H, W, 3) uint8 BGR image.
                      An image is always read with OCR (self.ocr_func).
            ocr_func: OCR for this call only, same contract as the
                      constructor argument. None uses self.ocr_func.
            use_ocr : for a PDF page, fall back to OCR (self.ocr_func) when
                      the page has no text layer. The OCR input is the full
                      resolution source render; the boxes are converted back
                      to PDF points, so the result looks like a page with a
                      text layer.
        """
        if isinstance(page, np.ndarray):
            bgr_img = page
            bboxes_px, bboxes_pt, texts = self._extract_bboxes_ocr(
                bgr_img, ocr_func)
            h_img, w_img = bgr_img.shape[:2]
            return {
                'bboxes'     : bboxes_pt,
                'text'       : texts,
                'image'      : bgr_img,
                'image_scale': 1.0,
                'page_size'  : (float(w_img), float(h_img)),
                'model_image': self._image_to_array(bgr_img),
                'bboxes_px'  : bboxes_px,
                'bboxes_src' : None,
                'source_size': (float(w_img), float(h_img)),
                'stage1'     : None,
            }

        # Normalize /Rotate so coordinates and rendering are upright
        # (PDF only; same call as in the dataset class).
        if getattr(page, 'rotation', 0):
            page.remove_rotation()
        prep = self._prepare_page(page)
        page_size = (float(page.rect.width), float(page.rect.height))
        data = {
            'bboxes'     : prep['bboxes_pt'],
            'text'       : prep['texts'],
            'image'      : prep['src'],
            'image_scale': self.render_scale,
            'page_size'  : page_size,
            'model_image': prep['image'],
            'bboxes_px'  : prep['bboxes_px'],
            'bboxes_src' : prep['bboxes_src'],
            'source_size': page_size,
            'stage1'     : None,
        }

        # OCR fallback: when the PDF has no text layer and use_ocr=True.
        # OCR runs on the full resolution source render and the boxes are
        # converted to the usual coordinates, so everything else in data
        # (page_size, image, image_scale, model_image) stays as it is.
        # Errors of ocr_func are not caught.
        if len(data['bboxes_px']) == 0 and use_ocr:
            if verbose:
                print('[CNN4GNN] No text layer found, falling back to OCR ...')
            bboxes_px, bboxes_pt, texts, bboxes_src = (
                self._extract_bboxes_ocr_page(prep['src'], page_size,
                                              ocr_func))
            data.update({
                'bboxes'    : bboxes_pt,
                'text'      : texts,
                'bboxes_px' : bboxes_px,
                'bboxes_src': bboxes_src,
            })
        return data

    def _check_data_matches(self, data, page):
        """Raise ValueError when data_dict was not made from this page."""
        if isinstance(page, np.ndarray):
            size = (float(page.shape[1]), float(page.shape[0]))
        else:
            size = (float(page.rect.width), float(page.rect.height))
        sw, sh = data['source_size']
        if abs(sw - size[0]) > 1e-3 or abs(sh - size[1]) > 1e-3:
            raise ValueError(
                f'[CNN4GNN] data_dict was made for a {sw:g}x{sh:g} page but '
                f'predict() got a {size[0]:g}x{size[1]:g} page')

    # ------------------------------------------------------------------
    # OCR need detection (text segmentation of stage 1 vs line bboxes)
    # ------------------------------------------------------------------

    def _ensure_stage1(self, data):
        """Stage 1 outputs of this data_dict, running stage 1 once if needed."""
        s1 = data.get('stage1')
        if s1 is None:
            image_np = np.asarray(data['model_image'],
                                  dtype=np.float32)[np.newaxis, np.newaxis]
            s1 = self.run_stage1(image_np)
            data['stage1'] = s1
        return s1

    def text_mask_iou(self, data_dict, text_thresh=0.5):
        """IoU between the line bbox mask and the text segmentation mask.

        Both masks are built at page resolution (one pixel per page unit):
        the bbox mask from data_dict['bboxes'], the text mask from the stage 1
        output text_pred (sigmoid probability, (1, H, W)) thresholded at
        text_thresh and resized with nearest neighbour. Both masks empty
        gives 1.0.

        Runs stage 1 when data_dict has no stage1 yet.

        Returns:
            float IoU, or None when the stage 1 ONNX has no text_pred
            output.
        """
        s1 = self._ensure_stage1(data_dict)
        text_pred = s1.get('text_pred')
        if text_pred is None or np.asarray(text_pred).size == 0:
            return None

        prob = np.asarray(text_pred, dtype=np.float32)
        prob = prob.reshape(-1, prob.shape[-2], prob.shape[-1])[0]

        page_w, page_h = data_dict['page_size']
        W = max(1, int(np.ceil(page_w)))
        H = max(1, int(np.ceil(page_h)))

        bbox_mask = np.zeros((H, W), dtype=np.bool_)
        for x0, y0, x1, y1 in np.asarray(
                data_dict['bboxes'], dtype=np.float32).reshape(-1, 4):
            r0 = max(0, int(y0))
            r1 = min(H, int(np.ceil(y1)))
            c0 = max(0, int(x0))
            c1 = min(W, int(np.ceil(x1)))
            if r1 > r0 and c1 > c0:
                bbox_mask[r0:r1, c0:c1] = True

        feat_h, feat_w = prob.shape
        if feat_h != H or feat_w != W:
            row_idx = (np.arange(H) * feat_h / H).astype(np.int32)
            col_idx = (np.arange(W) * feat_w / W).astype(np.int32)
            prob = prob[np.ix_(row_idx, col_idx)]
        text_mask = prob >= text_thresh

        intersection = np.count_nonzero(bbox_mask & text_mask)
        union        = np.count_nonzero(bbox_mask | text_mask)
        return float(intersection / union) if union > 0 else 1.0

    def is_ocr_needed(self, data_dict, iou_threshold=0.2):
        """Estimate whether the page needs OCR.

        Compares the PDF line bboxes with the text segmentation of the model
        (see text_mask_iou). Also catches pages whose text layer is only
        partly extracted.

        Runs stage 1 when data_dict has no stage1 yet; predict() reuses it.

        Args:
            data_dict     : from create_data() for this page.
            iou_threshold : mask IoU below this value means OCR is needed.

        Returns:
            True  -- low overlap, the page likely needs OCR.
            False -- enough overlap.
            None  -- unavailable (the stage 1 ONNX has no text_pred).
        """
        iou = self.text_mask_iou(data_dict)
        if iou is None:
            return None
        return bool(iou < iou_threshold)

    def is_page_ocr_needed(self, data_dict=None, iou_threshold=0.2):
        """Same as is_ocr_needed(); the name used by BoxRFDGNN."""
        if data_dict is None:
            return None
        return self.is_ocr_needed(data_dict, iou_threshold)

    def is_image_page(self, page):
        """Compatibility stub for the BoxRFDGNN interface.

        CNN4GNN has no separate image page detector, so this always returns
        False.
        """
        return False

    # ------------------------------------------------------------------
    # Table grid
    # ------------------------------------------------------------------

    def _sort_groups(self, groups, page_width):
        """Groups in reading order (DefaultSorter).

        page_width is the page width in the units of group_bbox (points), so
        image inputs work too: DefaultSorter only reads page.rect.width.
        """
        if not groups:
            return groups
        sorter = getattr(self, 'sorter', None)
        if sorter is None:
            sorter = self.sorter = DefaultSorter()

        class _PageWidth:
            rect = type('R', (), {'width': float(page_width)})()

        order = sorter.sort(_PageWidth, groups, groups)
        return [groups[i] for i in order]

    def _extract_tables(self, groups, bboxes_pt, texts, get_page_image):
        """Run the table grid extractor on every table group, in place.

        Sets group['table_grid'] (GridPrediction or None) and
        group['table_cells'] (list of CellInfo) on each group whose class is
        table. The crop uses the final group_bbox, and the line bboxes are
        moved to crop space, the same way BoxRFDGNN does it.

        Args:
            groups         : group dicts from postprocess().
            bboxes_pt      : (N, 4) node bboxes. Their coordinates must match
                             the pixels of the image returned by
                             get_page_image().
            texts          : list of N line texts.
            get_page_image : callable returning the BGR uint8 image to crop
                             from. Called at most once, and only when the
                             page has at least one table group.
        """
        if self.table_grid_extractor is None:
            return

        page_image = None
        bboxes_arr = np.asarray(bboxes_pt, dtype=np.float32).reshape(-1, 4)

        for group in groups:
            if group.get('class_name') != 'table':
                continue

            group['table_grid']  = None
            group['table_cells'] = []

            indices = list(group.get('indicies', []))
            if not indices:
                continue

            if page_image is None:
                page_image = get_page_image()

            crop_x0, crop_y0, crop_x1, crop_y1 = group['group_bbox'][:4]

            crop_y0_int = max(0, int(crop_y0))
            crop_y1_int = min(page_image.shape[0], int(crop_y1))
            crop_x0_int = max(0, int(crop_x0))
            crop_x1_int = min(page_image.shape[1], int(crop_x1))

            crop_img = page_image[crop_y0_int:crop_y1_int,
                                  crop_x0_int:crop_x1_int]
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue

            crop_bboxes = bboxes_arr[indices].copy()
            crop_bboxes[:, 0] -= crop_x0
            crop_bboxes[:, 2] -= crop_x0
            crop_bboxes[:, 1] -= crop_y0
            crop_bboxes[:, 3] -= crop_y0
            group_texts = [texts[i] for i in indices]

            grid, cells = self.table_grid_extractor.predict(
                crop_img, crop_bboxes, texts=group_texts)
            group['table_grid']  = grid
            group['table_cells'] = cells

    # ------------------------------------------------------------------
    # Public: predict
    # ------------------------------------------------------------------

    def predict(self, page=None, verbose=False, data_dict=None, **kwargs):
        """Run layout analysis on a page or a BGR image.

        The call signature and the return values follow BoxRFDGNN.predict.

        Parameters
        ----------
        page          : pymupdf.Page  or  np.ndarray (H, W, 3) uint8 BGR.
                        The old keyword name page_or_image is still accepted.
        verbose       : bool
        data_dict     : result of create_data(page) for this same page. When
                        it already holds the stage 1 outputs (for example
                        after is_ocr_needed), stage 1 is not run again. None
                        builds a new one. A data_dict made for a page of a
                        different size raises ValueError.
        return_raw    : bool (kwarg) -- return a dict, see Returns.
        return_groups : bool (kwarg) -- return the group dicts instead of
                        det_result (the behaviour of the old CNN4GNN default).
        return_det    : bool (kwarg) -- also return the FCOS detections,
                        as a tuple (primary_result, fcos_results). With
                        return_raw the FCOS detections are stored under the
                        key fcos_results instead.
        return_logits : bool (kwarg) -- with return_raw, add node_logits,
                        edge_logits and bboxes_pt to the dict.
        edge_threshold: float (kwarg) -- override the instance default.
        ocr_func      : callable (kwarg) -- OCR for this call, instead of
                        self.ocr_func (used when data_dict is None).
        use_ocr       : bool (kwarg) -- OCR fallback when the PDF has no
                        text layer (used when data_dict is None).
        check_ocr_need : bool (kwarg) -- compute needs_ocr with
                        is_ocr_needed().
        ocr_need_iou_threshold : float (kwarg) -- IoU threshold for
                        check_ocr_need, default 0.2.
        logits_hook   : callable (kwarg) -- per-call override of
                        self.logits_hook.
        do_sort       : bool (kwarg, default True) -- with use_sort=True (the
                        constructor default) the groups are put in reading
                        order; False keeps the order of the first text node.
        filter_img_bboxes, expand_by_vectors : accepted for
                        BoxRFDGNN compatibility. They have no effect here:
                        the geometry post-processing is not applied.

        Returns
        -------
        return_raw=False : det_result, a list of [x1, y1, x2, y2, class_name]
                           (or the group dicts when return_groups=True).
        return_raw=True  : dict with the keys
                           det_result, groups, needs_ocr (None unless
                           check_ocr_need is set and available),
                           node_feat, edge_feat (inputs of the node and edge
                           heads, None when the ONNX does not export them;
                           not comparable with the BoxRFDGNN embeddings),
                           node_bboxes, node_labels, node_scores,
                           edge_index (2, E), predicted_edge_labels (E,).
        Each table group has table_grid and table_cells when a table grid
        model is loaded. Every group has bboxes, the line bboxes it contains.
        """
        if page is None:
            page = kwargs.pop('page_or_image', None)
        else:
            kwargs.pop('page_or_image', None)
        if page is None:
            raise TypeError('[CNN4GNN] predict() needs a page or an image')

        return_det     = kwargs.get('return_det',     False)
        return_raw     = kwargs.get('return_raw',     False)
        return_groups  = kwargs.get('return_groups',  False)
        edge_threshold = kwargs.get('edge_threshold', self.edge_threshold)
        use_ocr        = kwargs.get('use_ocr',        False)
        check_ocr_need = kwargs.get('check_ocr_need', False)
        ocr_need_iou_threshold = kwargs.get('ocr_need_iou_threshold', 0.2)
        # logits_hook: per-call override of self.logits_hook (see
        # _apply_logits_hook); None keeps the model output.
        logits_hook    = kwargs.get('logits_hook', self.logits_hook)

        if verbose and (kwargs.get('filter_img_bboxes')
                        or kwargs.get('expand_by_vectors')):
            print('[CNN4GNN] filter_img_bboxes and expand_by_vectors are '
                  'ignored by this model')

        is_image_input = isinstance(page, np.ndarray)

        # ------------------------------------------------------------------
        # 1. Page data (image, boxes, texts, cached stage 1)
        # ------------------------------------------------------------------
        if data_dict is None:
            data = self.create_data(page, use_ocr=use_ocr, verbose=verbose,
                                    ocr_func=kwargs.get('ocr_func'))
        else:
            data = data_dict
            self._check_data_matches(data, page)

        bboxes_px  = data['bboxes_px']
        bboxes_pt  = data['bboxes']
        texts      = data['text']
        bboxes_src = data['bboxes_src']
        image_np   = np.asarray(data['model_image'],
                                dtype=np.float32)[np.newaxis, np.newaxis]
        N = len(bboxes_px)

        def get_page_image():
            # The crop source must have one pixel per bboxes unit. Reuse the
            # source render when it does, otherwise render the page at 1.
            if data.get('image_scale', 1.0) == 1.0:
                return data['image']
            return self._render_source(page, scale=1.0)

        # OCR need (runs stage 1 once and caches it in data)
        needs_ocr = None
        if check_ocr_need:
            needs_ocr = self.is_ocr_needed(data, ocr_need_iou_threshold)

        if N == 0:
            if return_raw:
                return {
                    'det_result':  [],
                    'groups':      [],
                    'needs_ocr':   needs_ocr,
                    'node_feat':   np.empty((0, 0), dtype=np.float32),
                    'edge_feat':   np.empty((0, 0), dtype=np.float32),
                    'node_bboxes': [],
                    'node_labels': np.zeros(0, dtype=np.int64),
                    'node_scores': np.zeros(0, dtype=np.float32),
                    'edge_index':  np.zeros((2, 0), dtype=np.int64),
                    'predicted_edge_labels': np.zeros(0, dtype=np.int64),
                }
            if return_det:
                return [], []
            return []

        # ------------------------------------------------------------------
        # 2-3b. Edges, text pattern features, optional margin / nn_geo
        #       (shared with diagnose_gap through _build_stage2_inputs)
        # ------------------------------------------------------------------
        (edge_index, union_bboxes, text_patterns,
         margin_features, nn_geo_features) = self._build_stage2_inputs(
            bboxes_px, texts, bboxes_src)

        # ------------------------------------------------------------------
        # 4. ONNX inference (see run_onnx). Stage 1 comes from the
        #    data_dict cache and runs only if it is not there yet.
        # ------------------------------------------------------------------
        stage1 = self._ensure_stage1(data)
        (node_logits, edge_logits,
         det_cls_np, det_reg_np, det_ctr_np, _feats) = self.run_onnx(
            image_np, bboxes_px, union_bboxes, edge_index, text_patterns,
            margin_features, nn_geo_features, return_features=True,
            stage1=stage1)

        # Optional replacement of the logits by a downstream model.
        if logits_hook is not None:
            node_logits, edge_logits = self._apply_logits_hook(logits_hook, {
                'node_logits'     : node_logits,
                'edge_logits'     : edge_logits,
                'node_head_in'    : _feats['node_head_in'],
                'edge_head_in'    : _feats['edge_head_in'],
                'edge_index'      : edge_index,
                'bboxes_px'       : bboxes_px,
                'bboxes_pt'       : bboxes_pt,
                'union_bboxes'    : union_bboxes,
                'texts'           : texts,
                'text_patterns'   : text_patterns,
                'margin_features' : margin_features,
                'nn_geo_features' : nn_geo_features,
                'det_cls'         : det_cls_np,
                'det_reg'         : det_reg_np,
                'det_ctr'         : det_ctr_np,
                'page'            : page,
                'img_size'        : (self.img_w, self.img_h),
            })

        if verbose:
            print(f'[CNN4GNN] {N} nodes, {edge_index.shape[1]} edges')

        # ------------------------------------------------------------------
        # 5-7. Node / edge decoding and grouping (shared with the cached
        #      logits path, see postprocess())
        # ------------------------------------------------------------------
        (groups, node_labels, node_scores,
         edge_labels) = self._postprocess_full(
            node_logits, edge_logits, edge_index, bboxes_pt,
            edge_threshold=edge_threshold)

        if verbose:
            print(f'[CNN4GNN] {len(groups)} groups')

        # Attach the raw line bboxes merged into each group (same as
        # BoxRFDGNN), then the table grids.
        # Reading order (groups only; node level outputs are not affected).
        if getattr(self, 'use_sort', False) and kwargs.get('do_sort', True):
            groups = self._sort_groups(groups, data['page_size'][0])

        for g in groups:
            g['bboxes'] = [[float(v) for v in bboxes_pt[i]]
                           for i in g['indicies']]

        self._extract_tables(groups, bboxes_pt, texts, get_page_image)

        det_result = [g['group_bbox'][:4] + [g['class_name']] for g in groups]

        # ------------------------------------------------------------------
        # 8. FCOS detection (only when requested)
        # ------------------------------------------------------------------
        fcos_results = None
        if return_det:
            fcos_results = decode_fcos(
                det_cls_np, det_reg_np, det_ctr_np,
                self.img_h, self.img_w,
                self.det_classes,
                score_thresh = self.det_score_thresh,
                nms_iou      = self.det_nms_iou,
            )
            # Scale detection bboxes to original page / image coordinates
            if is_image_input:
                H_orig, W_orig = page.shape[:2]
                sx_inv = W_orig / self.img_w
                sy_inv = H_orig / self.img_h
            else:
                sx_inv = page.rect.width  / self.img_w
                sy_inv = page.rect.height / self.img_h
            for d in fcos_results:
                b = d['bbox']
                d['bbox'] = [b[0]*sx_inv, b[1]*sy_inv,
                             b[2]*sx_inv, b[3]*sy_inv]

        # ------------------------------------------------------------------
        # Return
        # ------------------------------------------------------------------
        if return_raw:
            out = {
                'det_result':  det_result,
                'groups':      groups,
                'needs_ocr':   needs_ocr,
                # Inputs of the classification heads (None when the stage2
                # ONNX does not export them).
                'node_feat':   _feats['node_head_in'],
                'edge_feat':   _feats['edge_head_in'],
                'node_bboxes': np.asarray(bboxes_pt).tolist(),
                'node_labels': node_labels,
                'node_scores': node_scores,
                # edge_index: (2, E). predicted_edge_labels: (E,), 1 means
                # connected.
                'edge_index':  np.asarray(edge_index, dtype=np.int64),
                'predicted_edge_labels': edge_labels,
            }
            if fcos_results is not None:
                out['fcos_results'] = fcos_results
            if kwargs.get('return_logits', False):
                # Raw stage2 outputs, for cache-vs-onnx diagnosis.
                out['node_logits'] = np.asarray(node_logits, dtype=np.float32)
                out['edge_logits'] = np.asarray(edge_logits, dtype=np.float32)
                out['bboxes_pt']   = np.asarray(bboxes_pt, dtype=np.float32)
            return out

        primary = groups if return_groups else det_result
        if return_det:
            return primary, fcos_results
        return primary
