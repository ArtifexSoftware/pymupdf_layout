"""CNN4GNN.py

ONNX-based inference class for the CNN4GNN layout analysis model.

Mirrors the BoxRFDGNN interface so it can be used as a drop-in replacement
for document layout analysis pipelines.

Typical usage:

    from source.layout.model.CNN4GNN import CNN4GNN

    model = CNN4GNN(
        model_cfg_path  = '/path/to/model.yaml',
        onnx_path       = '/path/to/cnn4gnn.onnx',
    )

    # page: pymupdf.Page
    groups = model.predict(page)
    for g in groups:
        print(g['class_name'], g['group_bbox'])
"""

import os
import math
from pathlib import Path

import anyconfig
import cv2
import numpy as np
import pymupdf
import onnxruntime as ort

from source.layout.common_util import (
    get_text_pattern,
    get_edge_by_directional_nn,
    get_edge_matrix,
    group_node_by_edge,
)


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
# Main inference class
# ---------------------------------------------------------------------------

class CNN4GNN:
    """ONNX-based CNN4GNN layout analysis model.

    Parameters
    ----------
    model_cfg_path : str
        Path to the model YAML config (same file used for training).
    onnx_path : str
        Path to the exported .onnx file.
    edge_threshold : float
        Minimum edge probability (softmax[:, 1]) to classify an edge as
        connected.  Default 0.55, matching BoxRFDGNN.
    det_score_thresh : float
        Minimum FCOS detection score (sigmoid(cls)*sigmoid(ctr)) to keep a
        detection box.  Default 0.3.
    det_nms_iou : float
        NMS IoU threshold for FCOS detections.  Default 0.5.
    use_gpu : bool
        If True, use CUDA execution provider when available.
    """

    def __init__(self,
                 model_cfg_path,
                 onnx_path,
                 edge_threshold   = 0.55,
                 det_score_thresh = 0.3,
                 det_nms_iou      = 0.5,
                 use_gpu          = False):

        self.edge_threshold   = edge_threshold
        self.det_score_thresh = det_score_thresh
        self.det_nms_iou      = det_nms_iou

        # ------------------------------------------------------------------
        # Load model config
        # ------------------------------------------------------------------
        with open(model_cfg_path, 'rb') as f:
            self.model_cfg = anyconfig.load(f)

        data_cfg = self.model_cfg.get('data', {})
        self.class_names = data_cfg.get('class_list', [])
        self.class_map   = {n: i for i, n in enumerate(self.class_names)}

        # Class priority list for group_node_by_edge tie-breaking
        self.class_priority_list = data_cfg.get('class_priority', [])

        # Detection class names (FCOS head)
        self.det_classes = self.model_cfg.get('model', {}).get(
            'det_classes', ['picture', 'table', 'formula'])

        # Image size (H, W) for CNN input
        img_size_cfg = self.model_cfg.get('model', {}).get('image_size', {})
        if isinstance(img_size_cfg, dict):
            sz = img_size_cfg.get('size', [512, 512])
        else:
            sz = [512, 512]
        self.img_w, self.img_h = int(sz[0]), int(sz[1])

        # Edge construction params
        ds_cfg = self.model_cfg.get('data', {})
        self.vertical_gap = float(ds_cfg.get('vertical_gap', 0.3))
        self.max_nodes    = int(ds_cfg.get('max_nodes', 5000))

        # ------------------------------------------------------------------
        # ONNX session
        # ------------------------------------------------------------------
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if use_gpu else ['CPUExecutionProvider']
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Cache input/output names
        self._input_names  = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

        print(f'[CNN4GNN] Loaded ONNX: {onnx_path}')
        print(f'[CNN4GNN] Inputs : {self._input_names}')
        print(f'[CNN4GNN] Outputs: {self._output_names}')
        print(f'[CNN4GNN] Classes: {self.class_names}')
        print(f'[CNN4GNN] Image size: {self.img_w}x{self.img_h}')

    # ------------------------------------------------------------------
    # Internal: page → image array
    # ------------------------------------------------------------------

    def _page_to_image(self, page):
        """Render a pymupdf.Page to a grayscale float32 array in [0,1]."""
        # Render at a scale that gives approximately the target resolution
        scale = max(self.img_w / page.rect.width,
                    self.img_h / page.rect.height)
        mat   = pymupdf.Matrix(scale, scale)
        pix   = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
        arr   = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width)
        arr   = cv2.resize(arr, (self.img_w, self.img_h),
                           interpolation=cv2.INTER_LINEAR)
        arr   = arr.astype(np.float32) / 255.0
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return arr  # (H, W) float32

    # ------------------------------------------------------------------
    # Internal: extract PDF text bboxes and texts
    # ------------------------------------------------------------------

    def _extract_bboxes(self, page):
        """Extract line-level bboxes and text from PDF text layer.

        Returns
        -------
        bboxes_px : np.ndarray (N, 4) float32 in image-pixel coords
        texts     : list[str] of length N
        """
        pw, ph = page.rect.width, page.rect.height
        sx = self.img_w / pw
        sy = self.img_h / ph

        page_dict = page.get_text('dict',
                                  flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
        bboxes_px, texts = [], []
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
                bboxes_px.append([x1*sx, y1*sy, x2*sx, y2*sy])
                texts.append(''.join(
                    s.get('text', '') for s in line.get('spans', [])))

        if not bboxes_px:
            return np.zeros((0, 4), dtype=np.float32), []
        return np.array(bboxes_px, dtype=np.float32), texts

    # ------------------------------------------------------------------
    # Internal: build edges
    # ------------------------------------------------------------------

    def _build_edges(self, bboxes_px):
        """Build directed edges using get_edge_by_directional_nn.

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
            bboxes_px,
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
    # Public: predict
    # ------------------------------------------------------------------

    def predict(self, page, verbose=False, **kwargs):
        """Run layout analysis on a single pymupdf.Page.

        Parameters
        ----------
        page    : pymupdf.Page
        verbose : bool -- print debug info

        Returns
        -------
        groups : list[dict]
            Each dict has:
                'group_bbox'  : [x1, y1, x2, y2]
                'group_class' : int class index
                'class_name'  : str class name
                'score'       : float confidence
        det_results : list[dict] (keyword: return_det=True)
            FCOS detection results for picture/table/formula regions.
            Each dict: {'bbox': [x1,y1,x2,y2], 'class_name': str, 'score': float}
        """
        return_det     = kwargs.get('return_det',     False)
        edge_threshold = kwargs.get('edge_threshold', self.edge_threshold)

        # ------------------------------------------------------------------
        # 1. Image
        # ------------------------------------------------------------------
        img_arr   = self._page_to_image(page)       # (H, W) float32
        image_np  = img_arr[np.newaxis, np.newaxis]  # (1, 1, H, W)

        # ------------------------------------------------------------------
        # 2. Bboxes + texts
        # ------------------------------------------------------------------
        bboxes_px, texts = self._extract_bboxes(page)
        N = len(bboxes_px)

        if N == 0:
            if return_det:
                return [], []
            return []

        # ------------------------------------------------------------------
        # 3. Edges
        # ------------------------------------------------------------------
        edge_index, union_bboxes = self._build_edges(bboxes_px)

        # ------------------------------------------------------------------
        # 4. Text pattern features (N, 40)
        # ------------------------------------------------------------------
        text_patterns = np.array(
            [get_text_pattern(t, return_vector=True) for t in texts],
            dtype=np.float32)

        # ------------------------------------------------------------------
        # 5. ONNX inference
        # ------------------------------------------------------------------
        onnx_inputs = {
            'image'        : image_np,
            'bboxes'       : bboxes_px,
            'union_bboxes' : union_bboxes,
            'edge_index'   : edge_index,
            'text_patterns': text_patterns,
        }

        if verbose:
            print('[CNN4GNN] ONNX inputs:')
            for k, v in onnx_inputs.items():
                print(f'  {k}: shape={v.shape}, dtype={v.dtype}')

        ort_outputs = self.session.run(None, onnx_inputs)
        # node_logits, edge_logits, det_cls, det_reg, det_ctr, mask_logits
        node_logits, edge_logits, det_cls, det_reg, det_ctr, mask_logits = ort_outputs

        # ------------------------------------------------------------------
        # 6. Node classification
        # ------------------------------------------------------------------
        def softmax(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)

        node_probs          = softmax(node_logits)                   # (N, C)
        predicted_node_cls  = node_probs.argmax(axis=1)              # (N,)
        predicted_node_score= node_probs[np.arange(N), predicted_node_cls]

        # ------------------------------------------------------------------
        # 7. Edge classification
        # ------------------------------------------------------------------
        E = edge_index.shape[1] if edge_index.ndim == 2 else 0
        if E > 0 and edge_logits.size > 0:
            edge_probs             = softmax(edge_logits)            # (E, 2)
            predicted_edge_labels  = (edge_probs[:, 1] > edge_threshold
                                      ).astype(np.int64)
        else:
            predicted_edge_labels = np.zeros(E, dtype=np.int64)

        # ------------------------------------------------------------------
        # 8. Group nodes by edges
        # ------------------------------------------------------------------
        edge_matrix = get_edge_matrix(N, edge_index, predicted_edge_labels)
        groups = group_node_by_edge(
            predicted_node_cls, predicted_node_score,
            edge_matrix, bboxes_px.tolist(),
            self.class_priority_list,
        )
        for g in groups:
            cls_idx = int(g['group_class'])
            g['class_name'] = (self.class_names[cls_idx]
                               if 0 <= cls_idx < len(self.class_names)
                               else str(cls_idx))

        if verbose:
            print(f'[CNN4GNN] {N} nodes, {E} edges, {len(groups)} groups')

        if not return_det:
            return groups

        # ------------------------------------------------------------------
        # 9. FCOS detection (picture / table / formula)
        # ------------------------------------------------------------------
        det_results = decode_fcos(
            det_cls, det_reg, det_ctr,
            self.img_h, self.img_w,
            self.det_classes,
            score_thresh = self.det_score_thresh,
            nms_iou      = self.det_nms_iou,
        )
        # Scale detection bboxes back to page (PDF point) coordinates
        pw, ph = page.rect.width, page.rect.height
        sx_inv = pw / self.img_w
        sy_inv = ph / self.img_h
        for d in det_results:
            b = d['bbox']
            d['bbox'] = [b[0]*sx_inv, b[1]*sy_inv, b[2]*sx_inv, b[3]*sy_inv]

        return groups, det_results
