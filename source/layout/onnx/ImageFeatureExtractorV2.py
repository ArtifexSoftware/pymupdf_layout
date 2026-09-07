"""
ImageFeatureExtractorV2
Purpose: ONNX-based image feature extraction and FCOS-based bbox detection
         using a UNetThin2 model exported with output_name='dec_all'.

Differences from V1 (UNetThin + CCL)
--------------------------------------
Feature map :  unchanged -- 'combined' decoder output (1, 5*F, H, W) is
               forwarded to the GNN pipeline exactly as before, via
               get_feature_map(). Class logits (1, C, H, W) are exposed
               separately via get_class_logits() -- the two are no longer
               concatenated (see get_feature_map()/get_class_logits()
               docstrings for why: they have different statistical
               character and should be pooled differently downstream).

Detection   :  CCL replaced by FCOS heads.
               - coarse head (H/8) : table, picture
               - fine   head (H/4) : all remaining foreground classes
               Each head's reg/ctr outputs are decoded directly from the
               ONNX session; no morphological post-processing is needed.

ONNX output layout varies by training flags (dec_all branch):

  use_fcos=False, use_text_seg=False:
    [0] combined   -- (1, 5*F, H, W)
    [1] logits     -- (1, C,   H, W)

  use_fcos=True, use_text_seg=False:
    [0] combined   -- (1, 5*F, H, W)
    [1] reg_coarse -- (1, 4,   H/8, W/8)
    [2] ctr_coarse -- (1, 1,   H/8, W/8)
    [3] reg_fine   -- (1, 4,   H/4, W/4)
    [4] ctr_fine   -- (1, 1,   H/4, W/4)
    [5] logits     -- (1, C,   H, W)

  use_fcos=True, use_text_seg=True, use_db_thresh=False:
    [0..5] same as above
    [6] text_logits -- (1, 2, H, W)  raw bg/text class logits

  use_fcos=True, use_text_seg=True, use_db_thresh=True:
    [0..5] same as above
    [6] text_logits -- (1, 2, H, W)
    [7] thresh_map  -- (1, 1, H, W)  adaptive threshold in [0, 1]
    [8] db_map      -- (1, 1, H, W)  differentiable binary map in [0, 1]

Calling contract:
    predict(page_img) must be called explicitly by the caller for every page,
    exactly once per page, BEFORE is_image_page() / is_ocr_needed() are used.
    Neither method runs inference itself -- they only read the state set by
    the most recent predict() call, plus a data_dict (from
    create_input_data_from_page()) for the page-level signals they
    additionally need:
        predict(page_img)
        is_image_page(data_dict)   # or: is_ocr_needed(data_dict)
    This guarantees predict() runs exactly once per page and that both query
    methods always observe the current page's state (see BoxRFDGNN.predict()
    for the call site that owns this sequencing).
"""

import numpy as np

try:
    from ..common_util import resize_image, to_gray, softmax_numpy, sigmoid_numpy
except ImportError:
    def softmax_numpy(x: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Numerically stable softmax along the given axis.
        """
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


    def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.
        """
        return 1.0 / (1.0 + np.exp(-x))


    def resize_image(image: np.ndarray, new_size: tuple) -> np.ndarray:
        """
        Resize an image using bilinear interpolation with numpy (vectorized).
        Equivalent to cv2.resize(..., INTER_LINEAR).

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W) or (H, W, C).
        new_size : tuple
            (new_width, new_height)

        Returns
        -------
        np.ndarray
            Resized image.
        """
        h, w = image.shape[:2]
        new_w, new_h = new_size

        # Generate grid of coordinates in output image
        x = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
        y = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5

        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dx = (x - x0)[None, :]  # shape (1, new_w)
        dy = (y - y0)[:, None]  # shape (new_h, 1)

        # Clip to valid range
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)

        if image.ndim == 3:
            # Expand dimensions for broadcasting
            Ia = image[y0[:, None], x0[None, :]]  # top-left
            Ib = image[y0[:, None], x1[None, :]]  # top-right
            Ic = image[y1[:, None], x0[None, :]]  # bottom-left
            Id = image[y1[:, None], x1[None, :]]  # bottom-right
        else:
            Ia = image[np.ix_(y0, x0)]
            Ib = image[np.ix_(y0, x1)]
            Ic = image[np.ix_(y1, x0)]
            Id = image[np.ix_(y1, x1)]

        # Bilinear interpolation
        wa = (1 - dx) * (1 - dy)
        wb = dx * (1 - dy)
        wc = (1 - dx) * dy
        wd = dx * dy

        out = (wa[..., None] * Ia + wb[..., None] * Ib +
               wc[..., None] * Ic + wd[..., None] * Id) if image.ndim == 3 else \
            (wa * Ia + wb * Ib + wc * Ic + wd * Id)

        return np.clip(out, 0, 255).astype(image.dtype)


    def to_gray(image: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to grayscale using numpy only.
        Equivalent to cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).

        Parameters
        ----------
        image : np.ndarray
            Input image in BGR format, shape (H, W, 3).

        Returns
        -------
        np.ndarray
            Grayscale image, shape (H, W).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must be a BGR image with shape (H, W, 3).")

        # Extract B, G, R channels
        B = image[:, :, 0].astype(np.float32)
        G = image[:, :, 1].astype(np.float32)
        R = image[:, :, 2].astype(np.float32)

        # Apply standard BGR ¡æ Gray conversion formula
        gray = 0.114 * B + 0.587 * G + 0.299 * R

        # Clip to valid range and convert back to uint8
        return np.clip(gray, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Class index layout (must match the training class_list in cfg)
# Index 0 is background; foreground indices start at 1.
# ---------------------------------------------------------------------------
_CLASS_NAMES = [
    'background',       # 0
    'text',             # 1
    'title',            # 2
    'picture',          # 3
    'table',            # 4
    'list-item',        # 5
    'page-header',      # 6
    'page-footer',      # 7
    'section-header',   # 8
    'footnote',         # 9
    'caption',          # 10
    'formula',          # 11
]

_BACKGROUND_CLASS = 'background'
_PICTURE_CLASS    = 'picture'

# Foreground class index sets (1-based, matching DocumentJsonDataset convention)
_COARSE_CLASS_NAMES = {'table', 'picture'}
_COARSE_CLASS_IDX   = sorted(
    i for i, name in enumerate(_CLASS_NAMES)
    if name in _COARSE_CLASS_NAMES
)  # e.g. [3, 4]

_FINE_CLASS_IDX = [
    i for i, name in enumerate(_CLASS_NAMES)
    if i > 0 and name not in _COARSE_CLASS_NAMES
]  # e.g. [1, 2, 5, 6, 7, 8, 9, 10, 11]

_PICTURE_CLASS_IDX = [
    i for i, name in enumerate(_CLASS_NAMES) if name == _PICTURE_CLASS
]  # [3]

_NON_PICTURE_COARSE_IDX = [
    i for i in _COARSE_CLASS_IDX if _CLASS_NAMES[i] != _PICTURE_CLASS
]  # coarse foreground classes excluding picture (i.e. table only)

_NON_PICTURE_FINE_IDX = _FINE_CLASS_IDX  # fine head carries no picture class

# Detection hyperparameters
_SCORE_COARSE_THR = 0.3   # coarse head (table, picture) -- large objects
_SCORE_FINE_THR   = 0.3   # fine   head (text, title, ...) -- dense objects
_NMS_IOU          = 0.25


# ---------------------------------------------------------------------------
# Pure-numpy FCOS decoder (no PyTorch dependency at inference time)
# ---------------------------------------------------------------------------

def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on (N, 4) boxes sorted by descending score.

    Returns:
        keep : (K,) int array of kept indices.
    """
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1  = np.maximum(x1[i], x1[rest])
        yy1  = np.maximum(y1[i], y1[rest])
        xx2  = np.minimum(x2[i], x2[rest])
        yy2  = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thr]

    return np.array(keep, dtype=np.int32)


def _fcos_decode_numpy(
        seg_logits: np.ndarray,
        reg:        np.ndarray,
        ctr:        np.ndarray,
        score_thr:  float,
        nms_iou:    float,
        active_class_indices: list | None,
        orig_h: int,
        orig_w: int,
) -> list[dict]:
    """Decode one FCOS head's ONNX outputs into bbox dicts (numpy-only).

    Normalisation convention (DocumentJsonDataset):
        l_norm = l_fmap_px / fmap_w
        Decoded: l_orig = l_norm * orig_w
               = (l_fmap_px / fmap_w) * orig_w
        Grid:    grid_x_orig = (i + 0.5) * (orig_w / fmap_w)
        => x1 = grid_x_orig - l_orig  correctly recovers orig-space coordinate.

    Args:
        seg_logits : (1, C, H, W) -- raw segmentation logits at fmap resolution.
        reg        : (1, 4, H, W) -- normalised (l, t, r, b) per DocumentJsonDataset.
        ctr        : (1, 1, H, W) -- centerness logits.
        score_thr  : minimum score (seg_prob x centerness) to keep.
        nms_iou    : IoU threshold for class-wise NMS.
        active_class_indices : foreground indices this head is responsible for.
                               None -> all foreground classes.
        orig_h, orig_w : output coordinate space (page image size).

    Returns:
        List with one element (batch=1):
            {'boxes':  (N, 4) float32  -- (x1, y1, x2, y2) in orig image px,
             'labels': (N,)   int32,
             'scores': (N,)   float32}
    """
    _, C, H, W = seg_logits.shape

    # Softmax over class axis (numpy-side, not baked into the ONNX graph;
    # see common_util.softmax_numpy for the shared implementation).
    seg_probs = softmax_numpy(seg_logits[0], axis=0)      # (C, H, W)

    ctr_scores = sigmoid_numpy(ctr[0, 0])                 # (H, W)

    # Pixel-centre coordinate grids in orig image space
    scale_x = orig_w / W
    scale_y = orig_h / H
    ys = (np.arange(H, dtype=np.float32) + 0.5) * scale_y   # (H,)
    xs = (np.arange(W, dtype=np.float32) + 0.5) * scale_x   # (W,)
    grid_x, grid_y = np.meshgrid(xs, ys)                     # (H, W)

    # Recover orig-space ltrb: l_norm * orig_w (matches fcos_decode PyTorch)
    l  = reg[0, 0] * orig_w   # (H, W)
    t  = reg[0, 1] * orig_h
    r  = reg[0, 2] * orig_w
    bv = reg[0, 3] * orig_h

    cls_range = active_class_indices if active_class_indices is not None \
                else list(range(1, C))

    all_boxes, all_labels, all_scores = [], [], []

    for cls in cls_range:
        if cls <= 0 or cls >= C:
            continue

        score = seg_probs[cls] * ctr_scores   # (H, W)
        mask  = score > score_thr
        if not mask.any():
            continue

        x1 = np.clip((grid_x - l)[mask], 0, orig_w)
        y1 = np.clip((grid_y - t)[mask], 0, orig_h)
        x2 = np.clip((grid_x + r)[mask], 0, orig_w)
        y2 = np.clip((grid_y + bv)[mask], 0, orig_h)

        boxes  = np.stack([x1, y1, x2, y2], axis=1)   # (M, 4)
        scores = score[mask]                            # (M,)

        keep = _nms_numpy(boxes, scores, nms_iou)
        if keep.size == 0:
            continue

        all_boxes.append(boxes[keep])
        all_labels.append(np.full(keep.size, cls, dtype=np.int32))
        all_scores.append(scores[keep])

    if all_boxes:
        return [{'boxes':  np.concatenate(all_boxes,  axis=0),
                 'labels': np.concatenate(all_labels, axis=0),
                 'scores': np.concatenate(all_scores, axis=0)}]
    return [{'boxes':  np.empty((0, 4), dtype=np.float32),
             'labels': np.empty(0,       dtype=np.int32),
             'scores': np.empty(0,       dtype=np.float32)}]


def _merge_detections(
        dets_coarse: list[dict],
        dets_fine:   list[dict],
        nms_iou:     float,
) -> dict:
    """Concatenate coarse + fine detections and apply a per-class NMS pass.

    Class-agnostic NMS would incorrectly suppress valid detections of
    different classes that happen to overlap (e.g. a section-header and the
    text block immediately below it).  Per-class NMS only suppresses
    duplicate candidates within the same class, which matches the convention
    used in fcos_decode (PyTorch path).

    Args:
        dets_coarse, dets_fine : each a single-element list (batch=1) as
                                 returned by _fcos_decode_numpy.
        nms_iou : IoU threshold for per-class NMS.

    Returns:
        Single dict with keys 'boxes', 'labels', 'scores'.
    """
    dc, df = dets_coarse[0], dets_fine[0]

    all_boxes  = np.concatenate([dc['boxes'],  df['boxes']],  axis=0)
    all_labels = np.concatenate([dc['labels'], df['labels']], axis=0)
    all_scores = np.concatenate([dc['scores'], df['scores']], axis=0)

    if all_boxes.shape[0] == 0:
        return {'boxes': all_boxes, 'labels': all_labels, 'scores': all_scores}

    kept_boxes, kept_labels, kept_scores = [], [], []
    for cls in np.unique(all_labels):
        mask   = all_labels == cls
        keep   = _nms_numpy(all_boxes[mask], all_scores[mask], nms_iou)
        kept_boxes.append(all_boxes[mask][keep])
        kept_labels.append(all_labels[mask][keep])
        kept_scores.append(all_scores[mask][keep])

    return {
        'boxes':  np.concatenate(kept_boxes,  axis=0),
        'labels': np.concatenate(kept_labels, axis=0),
        'scores': np.concatenate(kept_scores, axis=0),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageFeatureExtractorV2:
    """
    Wraps a UNetThin2 ONNX session (exported with output_name='dec_all',
    use_fcos=True) to produce:

      - a combined decoder feature map  (forwarded to the GNN pipeline)
      - picture-class bbox detections   (lazy, via get_picture_detections())
      - non-picture detections          (lazy, via is_image_page())

    The public interface is fully compatible with ImageFeatureExtractorV1
    so existing callers (BoxRFDGNN, image_feature_extraction_task) require
    no changes.

    Internal changes vs. V1:
      - CCL (connected-component labelling) is replaced by FCOS decoding.
      - ONNX session produces 5 outputs instead of 1; all are stored.
      - Detection results are split by head assignment (coarse / fine) and
        merged with a final NMS pass.
    """

    def __init__(self, onnx_session):
        """
        Args:
            onnx_session : onnxruntime.InferenceSession for a UNetThin2 model
                           exported with output_name='dec_all', use_fcos=True.
                           Expected output order:
                             [0] combined, [1] reg_coarse, [2] ctr_coarse,
                             [3] reg_fine, [4] ctr_fine
        """
        self._session = onnx_session

        # Raw ONNX outputs for FCOS decoding and GNN feature input
        self._combined   = None    # (1, 5*F, H, W)
        self._reg_coarse = None    # (1, 4, H/8, W/8)
        self._ctr_coarse = None    # (1, 1, H/8, W/8)
        self._reg_fine   = None    # (1, 4, H/4, W/4)
        self._ctr_fine   = None    # (1, 1, H/4, W/4)
        self._logits     = None    # (1, C, H, W) segmentation logits

        # Text segmentation outputs (present only when model was trained with
        # use_text_seg=True; None otherwise)
        self._text_logits = None   # (1, 2, H, W) raw bg/text logits
        self._thresh_map  = None   # (1, 1, H, W) adaptive threshold in [0,1]
        self._db_map      = None   # (1, 1, H, W) differentiable binary map

        # Single-use cache flag (BoxRFDGNN + image_feature_extraction_task)
        self._cached = False

        # Lazy detection results -- None means not yet computed this cycle
        self._detections_picture     = None   # dict | None
        self._detections_non_picture = None   # dict | None

        # Page / model resolution -- set during predict()
        self._page_h   = 0
        self._page_w   = 0
        self._target_h = 0
        self._target_w = 0

    # ------------------------------------------------------------------
    # Cache protocol (identical to V1)
    # ------------------------------------------------------------------

    def mark_cached(self):
        """Signal that predict() results should be reused once."""
        self._cached = True

    def consume_cache(self):
        """
        Return True (and clear the flag) if a cached result exists.
        Called by image_feature_extraction_task() to skip predict().
        """
        if self._cached:
            self._cached = False
            return True
        return False

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, page_img, aug_fetmap=None):
        """
        Run ONNX inference and store all five head outputs.
        FCOS decoding is NOT performed here -- results are computed lazily.

        Calling predict() invalidates all previously cached detection results
        so that get_picture_detections() and is_image_page() always reflect
        the current page.

        Args:
            page_img   : np.ndarray (H, W, C) uint8
            aug_fetmap : optional extra channel map concatenated before inference

        Side effects:
            self._combined / _logits / _reg_coarse / _ctr_coarse /
            _reg_fine / _ctr_fine    <- set from ONNX outputs
            self._detections_picture / _non_picture <- reset to None
        """
        self._page_h, self._page_w = page_img.shape[:2]

        input_shape = self._session.get_inputs()[0].shape
        self._target_h, self._target_w = input_shape[2], input_shape[3]

        # Preprocess: resize -> grayscale -> normalise -> add batch dim
        img_resized = resize_image(page_img, (self._target_w, self._target_h))
        img_gray    = to_gray(img_resized).astype(np.float32)

        min_val, max_val = img_gray.min(), img_gray.max()
        img_gray = (img_gray - min_val) / (max_val - min_val + 1e-8)

        nn_input = np.expand_dims(img_gray, axis=0)    # (1, H, W)
        if aug_fetmap is not None:
            nn_input = np.concatenate([nn_input, aug_fetmap], axis=0)
        nn_input = np.expand_dims(nn_input, axis=0)    # (1, C_in, H, W)

        # ONNX inference
        input_name  = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]
        ort_outputs = self._session.run(None, {input_name: nn_input})

        # Build a name->array map so unpacking is robust to output count.
        out_map = dict(zip(output_names, ort_outputs))

        self._combined   = out_map.get('combined')
        self._reg_coarse = out_map.get('reg_coarse')
        self._ctr_coarse = out_map.get('ctr_coarse')
        self._reg_fine   = out_map.get('reg_fine')
        self._ctr_fine   = out_map.get('ctr_fine')
        self._logits     = out_map.get('logits')

        # Text segmentation outputs (None when not exported)
        self._text_logits = out_map.get('text_logits')   # (1, 2, H, W) or None
        self._thresh_map  = out_map.get('thresh_map')    # (1, 1, H, W) or None
        self._db_map      = out_map.get('db_map')        # (1, 1, H, W) or None

        # Invalidate stale detection results from the previous predict() cycle
        self._detections_picture     = None
        self._detections_non_picture = None
    # ------------------------------------------------------------------
    # Lazy FCOS detection helpers
    # ------------------------------------------------------------------

    def _logits_for_fmap(self, fmap_h: int, fmap_w: int) -> np.ndarray:
        """Resize the real segmentation logits to the given feature-map size.

        Uses self._logits, which is the actual seg_head output (index 5 from
        the ONNX session).  This is identical to the logits used in
        check_unethin2_result, so FCOS score computation is consistent between
        the PyTorch and ONNX inference paths.

        Args:
            fmap_h, fmap_w : target spatial size matching the reg/ctr tensors
                             (H/8 for coarse head, H/4 for fine head).

        Returns:
            np.ndarray (1, C, fmap_h, fmap_w) float32
        """
        _, C, H, W = self._logits.shape
        if H == fmap_h and W == fmap_w:
            return self._logits

        try:
            import cv2
            out = np.zeros((1, C, fmap_h, fmap_w), dtype=np.float32)
            for c in range(C):
                out[0, c] = cv2.resize(
                    self._logits[0, c],
                    (fmap_w, fmap_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            return out
        except ImportError:
            ys = (np.arange(fmap_h) * H / fmap_h).astype(int)
            xs = (np.arange(fmap_w) * W / fmap_w).astype(int)
            return self._logits[:, :, ys[:, None], xs[None, :]]


    def _ensure_picture_detections(self):
        """Run picture FCOS decoding if not yet computed this predict() cycle."""
        if self._detections_picture is not None:
            return

        fmap_h_c = self._reg_coarse.shape[2]
        fmap_w_c = self._reg_coarse.shape[3]
        logits_c = self._logits_for_fmap(fmap_h_c, fmap_w_c)

        dets = _fcos_decode_numpy(
            seg_logits=logits_c,
            reg=self._reg_coarse,
            ctr=self._ctr_coarse,
            score_thr=_SCORE_COARSE_THR,
            nms_iou=_NMS_IOU,
            active_class_indices=_PICTURE_CLASS_IDX,
            orig_h=self._page_h,
            orig_w=self._page_w,
        )
        # Wrap in the same dict structure as _merge_detections for consistency
        self._detections_picture = dets[0]

    def _ensure_non_picture_detections(self):
        """Run non-picture FCOS decoding if not yet computed this predict() cycle.

        Non-picture classes span both heads:
          - coarse head : table (non-picture coarse class)
          - fine   head : text, title, list-item, page-header, ...
        Both results are merged with a final NMS pass.
        """
        if self._detections_non_picture is not None:
            return

        # Coarse head: non-picture coarse classes (table only by default)
        fmap_h_c = self._reg_coarse.shape[2]
        fmap_w_c = self._reg_coarse.shape[3]
        logits_c = self._logits_for_fmap(fmap_h_c, fmap_w_c)

        dets_coarse = _fcos_decode_numpy(
            seg_logits=logits_c,
            reg=self._reg_coarse,
            ctr=self._ctr_coarse,
            score_thr=_SCORE_COARSE_THR,
            nms_iou=_NMS_IOU,
            active_class_indices=_NON_PICTURE_COARSE_IDX or None,
            orig_h=self._page_h,
            orig_w=self._page_w,
        )

        # Fine head: all fine-head foreground classes
        fmap_h_f = self._reg_fine.shape[2]
        fmap_w_f = self._reg_fine.shape[3]
        logits_f = self._logits_for_fmap(fmap_h_f, fmap_w_f)

        dets_fine = _fcos_decode_numpy(
            seg_logits=logits_f,
            reg=self._reg_fine,
            ctr=self._ctr_fine,
            score_thr=_SCORE_FINE_THR,
            nms_iou=_NMS_IOU,
            active_class_indices=_NON_PICTURE_FINE_IDX or None,
            orig_h=self._page_h,
            orig_w=self._page_w,
        )

        self._detections_non_picture = _merge_detections(
            dets_coarse, dets_fine, _NMS_IOU
        )

    # ------------------------------------------------------------------
    # Public API  (V1-compatible)
    # ------------------------------------------------------------------

    def get_feature_map(self):
        """
        Return the combined decoder feature map from the last predict() call.
        This is the continuous embedding output only -- it no longer
        includes the class logits (see get_class_logits() for those).

        Returns:
            np.ndarray (1, 5*F, H, W) float32, or None if predict() not called.
        """
        return self._combined

    def get_class_logits(self):
        """
        Return the per-class segmentation logits from the last predict()
        call. Raw (pre-softmax) scores; apply softmax before pooling ops
        that expect a probability distribution (e.g. entropy, margin --
        see roi_pooling.extract_bbox_features_by_roi_pooling).

        Returns:
            np.ndarray (1, C, H, W) float32, or None if predict() not called.
        """
        return self._logits

    def is_image_page(self, data_dict) -> bool:
        """
        Determine whether the page is an image-only PDF page that requires OCR.

        Returns True only when ALL three conditions hold:
          1. The page contains at least one raster image.
          2. The page contains no embedded selectable text.
          3. The FCOS model detects at least one non-picture region
             (text, table, header, etc.) with score > _SCORE_FINE_THR.

        Contract: predict() must already have been called for this page
        before calling this method (this method does not run inference
        itself). Callers should follow the pattern:
            feature_extractor.predict(page_img)
            feature_extractor.is_image_page(data_dict)
        Picture detection is deferred to get_picture_detections() if needed.

        Args:
            data_dict: output of create_input_data_from_page(), providing
                       'has_raster_image' and 'has_embedded_text' (raw,
                       input_type-independent page-level signals).

        Returns:
            bool
        """
        # Condition 1: page must contain at least one raster image
        if not data_dict['has_raster_image']:
            return False

        # Condition 2: page must have no embedded selectable text
        if data_dict['has_embedded_text']:
            return False

        # Condition 3: FCOS must detect at least one non-picture region
        self._ensure_non_picture_detections()

        return self._detections_non_picture['boxes'].shape[0] > 0

    def is_ocr_needed(self, data_dict, iou_threshold: float = 0.2) -> bool | None:
        """
        Estimate whether the page likely needs OCR by comparing PDF-extracted
        bbox positions against the text segmentation probability map.

        Unlike is_image_page() -- which detects fully scanned pages with no
        embedded text at all -- this method also catches pages where PDF
        parsing has only partially extracted the text (some regions missing).

        Contract: predict() must already have been called for this page
        before calling this method (this method does not run inference
        itself). Callers should follow the pattern:
            feature_extractor.predict(page_img)
            feature_extractor.is_ocr_needed(data_dict)

        Args:
            data_dict     : output of create_input_data_from_page() for this
                            same page (provides 'image' and 'bboxes').
            iou_threshold : mask IoU below this value triggers True.
                            0.0 = no overlap at all (definitely needs OCR).
                            1.0 = perfect overlap  (definitely does not).
                            Default 0.2.

        Returns:
            True  -- low overlap; page likely needs OCR.
            False -- sufficient overlap; OCR probably not needed.
            None  -- model was not exported with use_text_seg=True;
                     capability unavailable.
        """
        if self._text_logits is None:
            # Model does not support text segmentation.
            return None

        try:
            page_image = data_dict['image']
            img_h, img_w = page_image.shape[:2]

            # Build bbox binary mask at page image resolution.
            # Only include bboxes whose box_type is 'text'; skip all others
            # (e.g. 'table_img_line') because they are not PDF-extracted text
            # and would produce spurious mask coverage that inflates IoU.
            box_types = data_dict.get('box_type', [None] * len(data_dict['bboxes']))
            bbox_mask = np.zeros((img_h, img_w), dtype=np.bool_)
            for (x0, y0, x1, y1), box_type in zip(data_dict['bboxes'], box_types):
                if not isinstance(box_type, str) or box_type != 'text':
                    continue
                r0 = max(0, int(y0))
                r1 = min(img_h, int(np.ceil(y1)))
                c0 = max(0, int(x0))
                c1 = min(img_w, int(np.ceil(x1)))
                if r1 > r0 and c1 > c0:
                    bbox_mask[r0:r1, c0:c1] = True

            # Resize text_prob to page image resolution and binarize.
            text_seg  = self.get_text_segmentation()
            text_prob = text_seg['text_prob']   # (H_feat, W_feat) float32
            feat_h, feat_w = text_prob.shape
            if feat_h != img_h or feat_w != img_w:
                row_idx = (np.arange(img_h) * feat_h / img_h).astype(np.int32)
                col_idx = (np.arange(img_w) * feat_w / img_w).astype(np.int32)
                text_prob = text_prob[np.ix_(row_idx, col_idx)]
            text_mask = text_prob >= 0.5

            # Both masks empty: no text in PDF or image -> no OCR needed.
            intersection = np.count_nonzero(bbox_mask & text_mask)
            union        = np.count_nonzero(bbox_mask | text_mask)
            iou = intersection / union if union > 0 else 1.0
            result = bool(iou < iou_threshold)

            return result

        except Exception:
            return None

    def get_picture_detections(self) -> list:
        """
        Return picture-class detections from the last predict() call in
        page pixel coordinates.

        Runs picture FCOS decoding on the first call after each predict().
        Subsequent calls within the same predict() cycle are free (cached).

        Returns:
            list of [x1, y1, x2, y2] in page pixel space (float).

        Raises:
            RuntimeError if predict() has not been called.
        """
        if self._combined is None:
            raise RuntimeError(
                "predict() must be called before get_picture_detections()"
            )

        self._ensure_picture_detections()

        det = self._detections_picture
        if det['boxes'].shape[0] == 0:
            return []

        return det['boxes'].tolist()

    # ------------------------------------------------------------------
    # Extended API (V2 additions -- not present in V1)
    # ------------------------------------------------------------------

    def get_layout_detections(self) -> dict:
        """
        Return all FCOS layout detections (both heads, merged) from the last
        predict() call in page pixel coordinates.

        Useful for downstream consumers that need the full detection set
        rather than the picture-only subset.

        Returns:
            dict with keys:
                'boxes'  : list of [x1, y1, x2, y2]
                'labels' : list of int  (1-based foreground class indices)
                'scores' : list of float
                'names'  : list of str  (class name for each detection)

        Raises:
            RuntimeError if predict() has not been called.
        """
        if self._combined is None:
            raise RuntimeError(
                "predict() must be called before get_layout_detections()"
            )

        self._ensure_picture_detections()
        self._ensure_non_picture_detections()

        pic  = self._detections_picture
        nopx = self._detections_non_picture

        all_boxes  = np.concatenate([pic['boxes'],  nopx['boxes']],  axis=0)
        all_labels = np.concatenate([pic['labels'], nopx['labels']], axis=0)
        all_scores = np.concatenate([pic['scores'], nopx['scores']], axis=0)

        if all_boxes.shape[0] == 0:
            return {'boxes': [], 'labels': [], 'scores': [], 'names': []}

        return {
            'boxes':  all_boxes.tolist(),
            'labels': all_labels.tolist(),
            'scores': all_scores.tolist(),
            'names':  [
                _CLASS_NAMES[lbl] if lbl < len(_CLASS_NAMES) else str(lbl)
                for lbl in all_labels
            ],
        }

    def get_layout_detections_refined(self, db_thresh: float = 0.5,
                                      min_area: int = 10) -> dict:
        """
        Refine layout detections by grouping text boxes into layout regions.

        Algorithm:
          1. For each text detection box, check which layout bbox contains
             its centre point.
          2. If it belongs to a layout region, add it to that region's group.
             The final bbox for that region is the union of all its text boxes.
             The original layout class and score are preserved.
             picture regions are excluded from this process (kept as-is).
          3. Text boxes whose centre does not fall inside any layout region
             are assigned as independent 'text' class detections.
          4. Layout regions that received no text boxes keep their original
             FCOS bbox unchanged.

        Requires use_text_seg=True export.  Falls back to
        get_layout_detections() when text outputs are absent.

        Args:
            db_thresh : binarisation threshold passed to get_text_detection().
            min_area  : minimum contour area passed to get_text_detection().

        Returns:
            Same dict structure as get_layout_detections().

        Raises:
            RuntimeError if predict() has not been called.
        """
        if self._combined is None:
            raise RuntimeError(
                "predict() must be called before get_layout_detections_refined()"
            )

        layout = self.get_layout_detections()
        if not layout['boxes']:
            return layout

        try:
            text_boxes = self.get_text_detection(db_thresh=db_thresh,
                                                 min_area=min_area)
        except RuntimeError:
            return layout

        if not text_boxes:
            return layout

        picture_idx = _PICTURE_CLASS_IDX[0] if _PICTURE_CLASS_IDX else -1
        text_idx    = next((i for i, n in enumerate(_CLASS_NAMES)
                            if n == 'text'), 1)

        n_layout = len(layout['boxes'])
        tb       = np.array(text_boxes, dtype=np.float32)   # (M, 4)
        cx       = (tb[:, 0] + tb[:, 2]) * 0.5              # (M,)
        cy       = (tb[:, 1] + tb[:, 3]) * 0.5              # (M,)

        # For each text box: index of the layout region it belongs to (-1 = none)
        assignment = np.full(len(text_boxes), -1, dtype=np.int32)

        for i, (box, lbl) in enumerate(zip(layout['boxes'], layout['labels'])):
            if lbl == picture_idx:
                continue
            x1, y1, x2, y2 = box
            inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
            # Only assign unassigned boxes (first-match wins)
            unassigned_inside = inside & (assignment == -1)
            assignment[unassigned_inside] = i

        # Build refined results
        out_boxes  = []
        out_labels = []
        out_scores = []
        out_names  = []

        for i, (box, lbl, score, name) in enumerate(zip(
                layout['boxes'], layout['labels'],
                layout['scores'], layout['names'])):

            if lbl == picture_idx:
                out_boxes.append(box)
                out_labels.append(lbl)
                out_scores.append(score)
                out_names.append(name)
                continue

            members = tb[assignment == i]
            if members.shape[0] == 0:
                # No text boxes fell inside -- drop this region (not picture)
                continue

            # Union of all member text boxes
            out_boxes.append([
                float(members[:, 0].min()),
                float(members[:, 1].min()),
                float(members[:, 2].max()),
                float(members[:, 3].max()),
            ])
            out_labels.append(lbl)
            out_scores.append(score)
            out_names.append(name)

        # Orphan text boxes -> independent 'text' detections
        orphans = tb[assignment == -1]
        for row in orphans:
            out_boxes.append(row.tolist())
            out_labels.append(text_idx)
            out_scores.append(0.0)
            out_names.append('text')

        return {
            'boxes':  out_boxes,
            'labels': out_labels,
            'scores': out_scores,
            'names':  out_names,
        }


    def get_text_detection(self, db_thresh: float = 0.3,
                           min_area: int = 10) -> list:
        """
        Extract text bounding boxes from the DB binary map.

        The DB map is a near-binary probability map computed in forward() as
            db_map = sigmoid(k * (text_prob - thresh_map)),  k=50
        so values are clustered near 0 or 1.  Thresholding at db_thresh and
        running contour extraction yields tight word/line-level text boxes.

        Falls back to text_prob (softmax class-1 of text_logits) when the
        model was exported without use_db_thresh (db_map is None).

        Args:
            db_thresh : binarization threshold applied to db_map or text_prob.
                        Default 0.5 works well for db_map; increase to ~0.6-0.7
                        when falling back to text_prob to reduce false positives.
            min_area  : minimum contour area in feature-map pixels.
                        Contours smaller than this are discarded as noise.

        Returns:
            list of [x1, y1, x2, y2] in page pixel coordinates (float).

        Raises:
            RuntimeError if predict() has not been called.
            RuntimeError if the model was not exported with use_text_seg=True.
        """
        if self._combined is None:
            raise RuntimeError(
                "predict() must be called before get_text_detection()"
            )
        if self._text_logits is None:
            raise RuntimeError(
                "The ONNX model was not exported with use_text_seg=True. "
                "Re-export with use_text_seg=True to use this method."
            )

        try:
            import cv2 as _cv2
        except ImportError:
            raise RuntimeError("opencv-python is required for get_text_detection()")

        # Select the best available probability map.
        # db_map is preferred: k=50 steep sigmoid makes it nearly binary,
        # giving cleaner contours than the raw softmax text_prob.
        if self._db_map is not None:
            prob = self._db_map[0, 0]           # (H_feat, W_feat)  in [0, 1]
        else:
            tl   = self._text_logits[0]         # (2, H_feat, W_feat)
            prob = softmax_numpy(tl, axis=0)[1]  # (H_feat, W_feat)

        feat_h, feat_w = prob.shape
        scale_x = self._page_w / feat_w
        scale_y = self._page_h / feat_h

        # Binarize and extract contours
        binary = (prob >= db_thresh).astype(np.uint8) * 255
        contours, _ = _cv2.findContours(
            binary, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            if _cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = _cv2.boundingRect(cnt)
            x1 = float(np.clip(x * scale_x,           0, self._page_w))
            y1 = float(np.clip(y * scale_y,           0, self._page_h))
            x2 = float(np.clip((x + w) * scale_x,     0, self._page_w))
            y2 = float(np.clip((y + h) * scale_y,     0, self._page_h))
            boxes.append([x1, y1, x2, y2])

        return boxes

    def get_text_segmentation(self) -> dict:
        """
        Return text segmentation outputs from the last predict() call.

        Available only when the model was exported with use_text_seg=True.
        Returns raw numpy arrays so the caller can apply any threshold strategy.

        Output keys:
            'text_prob'  : (H, W) float32  -- per-pixel text probability in [0, 1].
                           Derived from text_logits via softmax class-1 channel.
                           Always present when use_text_seg=True.
            'thresh_map' : (H, W) float32  -- adaptive threshold map in [0, 1].
                           Present only when use_db_thresh=True; None otherwise.
            'db_map'     : (H, W) float32  -- differentiable binary map in [0, 1].
                           Present only when use_db_thresh=True; None otherwise.

        Returns:
            dict with the keys above.

        Raises:
            RuntimeError if predict() has not been called.
            RuntimeError if the model was not exported with use_text_seg=True.
        """
        if self._combined is None:
            raise RuntimeError(
                "predict() must be called before get_text_segmentation()"
            )
        if self._text_logits is None:
            raise RuntimeError(
                "The ONNX model was not exported with use_text_seg=True. "
                "Re-export with use_text_seg=True to use this method."
            )

        # Softmax over the 2-class axis, take class-1 (text) channel
        tl = self._text_logits[0]                      # (2, H, W)
        text_prob = softmax_numpy(tl, axis=0)[1]        # (H, W)

        thresh_map = self._thresh_map[0, 0] if self._thresh_map is not None else None
        db_map     = self._db_map[0, 0]     if self._db_map     is not None else None

        return {
            'text_prob':  text_prob,
            'thresh_map': thresh_map,
            'db_map':     db_map,
        }

# ---------------------------------------------------------------------------
# Standalone visualisation entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    import argparse
    import cv2
    import numpy as np
    import onnxruntime as ort
    import pymupdf

    parser = argparse.ArgumentParser(
        description="Visualise ImageFeatureExtractorV2 results on PDF files."
    )
    parser.add_argument("onnx_path", help="Path to the exported .onnx model file.")
    parser.add_argument("pdf_dir",   help="Directory containing PDF files to process.")
    parser.add_argument("--dpi",     type=int, default=96,
                        help="Rendering DPI for PDF pages (default: 96).")
    parser.add_argument("--score_thr", type=float, default=0.3,
                        help="Layout detection score threshold (default: 0.3).")
    parser.add_argument("--db_thr",    type=float, default=0.5,
                        help="DB map binarisation threshold (default: 0.5).")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_path):
        print(f"ERROR: ONNX file not found: {args.onnx_path}")
        sys.exit(1)
    if not os.path.isdir(args.pdf_dir):
        print(f"ERROR: PDF directory not found: {args.pdf_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Colormap: one BGR colour per class
    # ------------------------------------------------------------------
    _rng = np.random.RandomState(42)
    _class_colors = np.vstack([
        np.array([[30, 30, 30]], dtype=np.uint8),
        _rng.randint(60, 220, size=(len(_CLASS_NAMES) - 1, 3), dtype=np.uint8),
    ])  # (num_classes, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scale_disp(img, max_w=1800, max_h=900):
        scale = min(max_w / max(img.shape[1], 1),
                    max_h / max(img.shape[0], 1), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    def _draw_layout_boxes(canvas, detections):
        """Draw get_layout_detections() results."""
        out = canvas.copy()
        for (x1, y1, x2, y2), lbl, score, name in zip(
                detections['boxes'], detections['labels'],
                detections['scores'], detections['names']):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = tuple(int(c) for c in
                          _class_colors[min(lbl, len(_class_colors) - 1)])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f'{name} {score:.2f}'
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2),
                          color, -1)
            cv2.putText(out, label, (x1 + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _draw_text_boxes(canvas, boxes):
        """Draw get_text_detection() results (green rectangles)."""
        out = canvas.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 200, 0), 1)
        return out

    def _map_to_jet(arr, h, w):
        """Render a float32 (H,W) map in [0,1] as a JET colormap BGR image."""
        resized = cv2.resize(arr.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
        uint8 = (np.clip(resized, 0.0, 1.0) * 255).astype(np.uint8)
        return cv2.applyColorMap(uint8, cv2.COLORMAP_JET)

    def _seg_to_color(logits, h, w):
        """Render segmentation logits (1,C,Hf,Wf) as a colour class map."""
        pred = np.argmax(logits[0], axis=0).astype(np.int32)   # (Hf, Wf)
        out  = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c in range(len(_CLASS_NAMES)):
            out[pred == c] = _class_colors[min(c, len(_class_colors) - 1)]
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    def _label(img, text):
        """Stamp a label in the top-left corner."""
        out = img.copy()
        cv2.putText(out, text, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(out, text, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _process_page(extractor, bgr_orig, score_thr, db_thr):
        """Run inference and display results in separate windows per row.

        Window 1 - Layout Detection:
            [ original | layout detection ]

        Window 2 - Segmentation Maps:
            [ layout seg | text prob | thresh map | db map ]
            (text panels only when use_text_seg=True)

        Window 3 - Text Detection:
            [ original | text detection ]
            (only when use_text_seg=True)

        predict() internally resizes the image to model input size, runs ONNX,
        then decodes coordinates back into the original page pixel space using
        _page_h/_page_w.  So all returned coordinates are already in bgr_orig
        space with maximum precision.

        Returns n_layout (int).
        """
        orig_h, orig_w = bgr_orig.shape[:2]

        extractor._SCORE_FINE_THR   = score_thr
        extractor._SCORE_COARSE_THR = score_thr
        extractor.predict(bgr_orig)   # _page_h/w = orig_h/w; coords decoded in orig space

        # ---- Window 1: layout detection ----
        layout_dets   = extractor.get_layout_detections_refined(db_thresh=db_thr)
        layout_canvas = _draw_layout_boxes(bgr_orig, layout_dets)
        cv2.imshow('1. Layout Detection',
                   _scale_disp(_label(layout_canvas, 'layout detection (refined)')))

        # ---- Window 2: segmentation maps ----
        seg_color = _seg_to_color(extractor._logits, orig_h, orig_w)
        seg_panel = _label(seg_color, 'layout seg')

        try:
            text_seg  = extractor.get_text_segmentation()
            tp_panel  = _label(_map_to_jet(text_seg['text_prob'], orig_h, orig_w),
                               'text prob')

            if text_seg['thresh_map'] is not None:
                thr_panel = _label(_map_to_jet(text_seg['thresh_map'], orig_h, orig_w),
                                   'thresh map')
                db_panel  = _label(_map_to_jet(text_seg['db_map'],     orig_h, orig_w),
                                   'db map')
            else:
                blank     = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                thr_panel = blank.copy()
                db_panel  = blank.copy()
                cv2.putText(thr_panel, 'thresh: N/A', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.putText(db_panel,  'db: N/A',     (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            win2 = np.concatenate([seg_panel, tp_panel, thr_panel, db_panel], axis=1)
            cv2.imshow('2. Segmentation Maps', _scale_disp(win2))

            # ---- Window 3: text detection ----
            text_boxes  = extractor.get_text_detection(db_thresh=db_thr)
            text_canvas = _draw_text_boxes(bgr_orig, text_boxes)
            cv2.imshow('3. Text Detection',
                       _scale_disp(_label(text_canvas,
                                          f'text detection  n={len(text_boxes)}')))

        except RuntimeError:
            # Model exported without use_text_seg -- seg map only
            cv2.imshow('2. Segmentation Maps', _scale_disp(seg_panel))

        return len(layout_dets['boxes'])


    # ------------------------------------------------------------------
    # Load ONNX session
    # ------------------------------------------------------------------
    print(f"Loading ONNX: {args.onnx_path}")
    session   = ort.InferenceSession(
        args.onnx_path,
        providers=['CPUExecutionProvider'],
    )
    extractor = ImageFeatureExtractorV2(session)

    print(f"  outputs: {[o.name for o in session.get_outputs()]}")

    # ------------------------------------------------------------------
    # Collect PDF files
    # ------------------------------------------------------------------
    pdf_files = sorted(
        os.path.join(args.pdf_dir, f)
        for f in os.listdir(args.pdf_dir)
        if f.lower().endswith('.pdf')
    )
    if not pdf_files:
        print(f"No PDF files found in {args.pdf_dir}")
        sys.exit(0)
    print(f"Found {len(pdf_files)} PDF file(s). Press ESC to stop.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    stop = False
    for pdf_path in pdf_files:
        if stop:
            break
        doc    = pymupdf.open(pdf_path)
        n_pages = len(doc)
        doc.close()
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{pdf_name}  ({n_pages} pages)")

        for page_idx in range(n_pages):
            print(f"  page {page_idx} ...", end=' ', flush=True)

            doc  = pymupdf.open(pdf_path)
            page = doc[page_idx]
            mat  = pymupdf.Matrix(args.dpi / 72, args.dpi / 72)
            pix  = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
            doc.close()
            bgr_orig = cv2.cvtColor(
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3),
                cv2.COLOR_RGB2BGR,
            )

            try:
                n_layout = _process_page(extractor, bgr_orig,
                                         args.score_thr, args.db_thr)
            except Exception as e:
                print(f"failed: {e}")
                import traceback; traceback.print_exc()
                continue

            print(f"layout={n_layout} boxes")

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                print("ESC -- stopping.")
                stop = True
                break

    cv2.destroyAllWindows()
