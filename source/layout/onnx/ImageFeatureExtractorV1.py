"""
ImageFeatureExtractorV1
Purpose: ONNX-based image feature extraction and segmentation-based bbox detection.

Calling contract:
    predict(page_img) must be called explicitly by the caller for every page,
    exactly once per page, BEFORE is_image_page() / is_ocr_needed() /
    get_feature_map() / get_class_logits() / get_picture_detections() are
    used. Neither is_image_page() nor is_ocr_needed() runs inference
    themselves -- they only read the state set by the most recent predict()
    call, plus a data_dict (from create_input_data_from_page()) for the
    page-level signals they additionally need:
        predict(page_img)
        is_image_page(data_dict)   # or: is_ocr_needed(data_dict)
    This guarantees predict() runs exactly once per page and that both query
    methods always observe the current page's state (see BoxRFDGNN.predict()
    for the call site that owns this sequencing).

CCL execution strategy (lazy):
    predict()             -> ONNX inference only. CCL is NOT run.
    is_image_page()        -> non-picture CCL only (predict() already done).
    get_picture_detections()-> runs picture CCL on first call after predict().

This minimises redundant work across three usage patterns:

  Pattern A  is_image_page() -> False (common case)
             ONNX: 1,  CCL: 1 (non-picture only)

  Pattern B  is_image_page() -> True -> get_picture_detections()
             ONNX: 1,  CCL: 2 (non-picture + picture, each once)

  Pattern C  predict() -> get_feature_map() only
             ONNX: 1,  CCL: 0

Cache protocol (single-use):
    mark_cached() / consume_cache() -> let the caller's explicit predict()
    call (see calling contract above) signal that ONNX inference has already
    run for this page, so the next image_feature_extraction_task() call
    skips a redundant predict().

ONNX output contract (matches ImageFeatureExtractorV2):
    'combined' -- (1, 5*F, H, W) concatenated decoder feature maps
    'logits'   -- (1, C,   H, W) raw per-class segmentation logits (no
                  softmax baked into the graph; CCL uses argmax, which does
                  not need it, and softmax is computed in numpy elsewhere
                  only where an actual probability is required).
Both outputs are read by name via out_map, never by output position, so the
extraction code here is robust to output ordering the same way V2 is.

get_feature_map() / get_class_logits() contract:
    These two outputs are kept SEPARATE (not concatenated) because they have
    different statistical character: 'combined' is a continuous embedding
    space best summarized by mean/max pooling, while 'logits' is a per-class
    score that should be turned into a probability (softmax) before pooling
    with ops like mean/min/max/entropy/margin. Callers that need both
    (e.g. GNN node/edge image features) pool each separately with
    roi_pooling.extract_bbox_features_by_roi_pooling and concatenate the
    results themselves; see roi_pooling.DEFAULT_FEATURE_MAP_POOLING_OPS and
    DEFAULT_CLASS_LOGITS_POOLING_OPS.
"""

import numpy as np

from ..common_util import (
    resize_image,
    to_gray,
    extract_bboxes_from_segmentation,
    extract_bboxes_from_segmentation_numpy,
)


# Class names expected from the segmentation head (index-aligned with channel axis)
_CLASS_NAMES = [
    'background', 'text', 'title', 'picture', 'table',
    'list-item', 'page-header', 'page-footer',
    'section-header', 'footnote', 'caption', 'formula',
]

_BACKGROUND_CLASS    = 'background'
_PICTURE_CLASS       = 'picture'
_NON_PICTURE_CLASSES = [c for c in _CLASS_NAMES
                        if c not in (_BACKGROUND_CLASS, _PICTURE_CLASS)]

_DETECTION_SCORE_THRESHOLD = 0.5
_MIN_COMPONENT_AREA        = 10
_MORPHOLOGY_KERNEL_SIZE    = 3


class ImageFeatureExtractorV1:
    """
    Wraps an ONNX inference session to produce:
      - a feature map
      - picture-class bbox detections      (lazy, via get_picture_detections())
      - non-picture-class detections       (lazy, via is_image_page())

    All CCL work is deferred until the result is actually needed, and cached
    so repeated calls within the same predict() cycle are free.
    """

    def __init__(self, onnx_session):
        """
        Args:
            onnx_session: onnxruntime.InferenceSession (or any object with
                          get_inputs() / run() matching the ORT interface)
        """
        self._session      = onnx_session
        self._combined     = None  # (1, 5*F, H, W) float32 -- decoder feature map, GNN input
        self._logits       = None  # (1, C,   H, W) float32 -- per-class seg logits, GNN input
        self._raw_outputs  = None  # alias for _logits; kept for CCL callers
        self._cached       = False  # cache flag for image_feature_extraction_task()

        # Lazy CCL results ? None means "not yet computed for this predict() cycle"
        self._detections_picture     = None  # list[dict] | None
        self._detections_non_picture = None  # list[dict] | None

        # Page / model resolution ? set during predict()
        self._page_h   = 0
        self._page_w   = 0
        self._target_h = 0
        self._target_w = 0

    # ------------------------------------------------------------------
    # Cache protocol (BoxRFDGNN + image_feature_extraction_task)
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
        Run ONNX inference and store the outputs.
        CCL is NOT performed here -- results are computed lazily on demand.

        Calling predict() invalidates all previously cached CCL results so
        that get_picture_detections() and is_image_page() always reflect the
        current page.

        Args:
            page_img:   np.ndarray (H, W, C), uint8
            aug_fetmap: optional extra channel map concatenated before inference

        Side effects:
            self._combined / self._logits    <- set from ONNX outputs by name
            self._raw_outputs                <- alias for self._logits (CCL input)
            self._detections_picture         <- reset to None
            self._detections_non_picture     <- reset to None
        """
        self._page_h, self._page_w = page_img.shape[:2]

        input_shape = self._session.get_inputs()[0].shape
        self._target_h, self._target_w = input_shape[2], input_shape[3]

        # Preprocess
        img_resized = resize_image(page_img, (self._target_w, self._target_h))
        img_gray    = to_gray(img_resized).astype(np.float32)

        min_val, max_val = img_gray.min(), img_gray.max()
        if max_val > min_val:
            img_gray = (img_gray - min_val) / (max_val - min_val)
        else:
            img_gray = np.zeros_like(img_gray, dtype=np.float32)

        nn_input = np.expand_dims(img_gray, axis=0)    # (1, H, W)
        if aug_fetmap is not None:
            nn_input = np.concatenate([nn_input, aug_fetmap], axis=0)
        nn_input = np.expand_dims(nn_input, axis=0)    # (1, C_in, H, W)

        # ONNX inference -- read outputs by name (out_map), never by
        # position, so this stays correct regardless of output ordering or
        # future additional outputs (same pattern as ImageFeatureExtractorV2).
        input_name   = self._session.get_inputs()[0].name
        output_names = [o.name for o in self._session.get_outputs()]
        ort_outputs  = self._session.run(output_names, {input_name: nn_input})
        out_map      = dict(zip(output_names, ort_outputs))

        self._combined = out_map['combined']
        self._logits   = out_map['logits']
        self._raw_outputs = self._logits

        # Invalidate stale CCL results from the previous predict() cycle
        self._detections_picture     = None
        self._detections_non_picture = None

    # ------------------------------------------------------------------
    # Lazy CCL helpers
    # ------------------------------------------------------------------

    def _ensure_picture_detections(self):
        """Run picture CCL if not yet computed for the current predict() cycle."""
        if self._detections_picture is not None:
            return
        self._detections_picture = extract_bboxes_from_segmentation_numpy(
            seg_logits=self._logits,
            class_names=_CLASS_NAMES,
            target_class=[_PICTURE_CLASS],
            min_component_area=_MIN_COMPONENT_AREA,
        )

    def _ensure_non_picture_detections(self):
        """Run non-picture CCL if not yet computed for the current predict() cycle."""
        if self._detections_non_picture is not None:
            return
        self._detections_non_picture = extract_bboxes_from_segmentation_numpy(
            seg_logits=self._logits,
            class_names=_CLASS_NAMES,
            target_class=_NON_PICTURE_CLASSES,
            min_component_area=_MIN_COMPONENT_AREA,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_feature_map(self):
        """
        Return the decoder feature map ('combined') from the last predict()
        call. This is the continuous embedding output only -- it no longer
        includes the class logits (see get_class_logits() for those).

        Returns:
            np.ndarray of shape (1, 5*F, H, W), or None if predict() not
            yet called.
        """
        return self._combined

    def get_class_logits(self):
        """
        Return the per-class segmentation logits ('logits') from the last
        predict() call. Raw (pre-softmax) scores; apply softmax before
        pooling ops that expect a probability distribution (e.g. entropy,
        margin -- see roi_pooling.extract_bbox_features_by_roi_pooling).

        Returns:
            np.ndarray of shape (1, C, H, W), or None if predict() not yet
            called.
        """
        return self._logits

    def is_image_page(self, data_dict):
        """
        Determine whether the page is an image-only PDF page that requires OCR.

        Returns True only when ALL three conditions hold:
          1. The page contains at least one raster image.
          2. The page contains no embedded selectable text.
          3. The segmentation model detects at least one non-picture region
             (text, table, header, etc.) inside the page image.

        Condition 1 + 2 confirm the PDF structure is image-only.
        Condition 3 confirms there is recoverable content inside the image.
        All three must hold for OCR to be both necessary and worthwhile.

        Contract: predict() must already have been called for this page
        before calling this method (this method does not run inference
        itself). Callers should follow the pattern:
            feature_extractor.predict(page_img)
            feature_extractor.is_image_page(data_dict)
        picture CCL is deferred to get_picture_detections() if needed.

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

        # Condition 3: segmentation model must detect non-picture content
        self._ensure_non_picture_detections()

        return any(
            d['score'] > _DETECTION_SCORE_THRESHOLD
            for d in self._detections_non_picture
        )


    def is_ocr_needed(self, data_dict, iou_threshold: float = 0.4) -> bool | None:
        """
        Estimate whether the page likely needs OCR by comparing PDF-extracted
        bbox positions against a content mask derived from layout segmentation.

        Unlike is_image_page() -- which detects fully scanned pages with no
        embedded text at all -- this method also catches pages where PDF
        parsing has only partially extracted the text (some regions missing).

        Uses _logits (layout segmentation) instead of a dedicated text
        segmentation head (V2). The content mask treats all non-background,
        non-picture classes as content, filtered by a confidence threshold to
        suppress false positives in blank regions where any class wins by a
        slim margin.

        Contract: predict() must already have been called for this page
        before calling this method (this method does not run inference
        itself). Callers should follow the pattern:
            feature_extractor.predict(page_img)
            feature_extractor.is_ocr_needed(data_dict)

        Args:
            data_dict     : output of create_input_data_from_page() for this
                            same page (provides 'image' and 'bboxes').
            iou_threshold : mask IoU below this value triggers True.
                            Default 0.4 (higher than V2's 0.2 because the
                            layout seg mask is coarser than a text-seg head).

        Returns:
            True  -- low overlap; page likely needs OCR.
            False -- sufficient overlap; OCR probably not needed.
            None  -- predict() not yet called or logits unavailable.
        """
        if self._logits is None:
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

            # Derive content mask from layout segmentation logits.
            # Apply softmax to get per-class probabilities.
            logits = self._logits[0]                          # (C, H, W)
            logits_shifted = logits - logits.max(axis=0, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)  # (C, H, W)

            confidence = probs.max(axis=0)                    # (H, W)
            pred_class = probs.argmax(axis=0)                 # (H, W)

            # Exclude background and picture; keep all other content classes.
            bg_idx      = _CLASS_NAMES.index(_BACKGROUND_CLASS)
            picture_idx = _CLASS_NAMES.index(_PICTURE_CLASS)
            non_content = {bg_idx, picture_idx}
            content_mask_feat = (
                ~np.isin(pred_class, list(non_content)) &
                (confidence >= _DETECTION_SCORE_THRESHOLD)
            )

            # Resize content mask from model resolution to page image resolution.
            feat_h, feat_w = content_mask_feat.shape
            if feat_h != img_h or feat_w != img_w:
                row_idx = (np.arange(img_h) * feat_h / img_h).astype(np.int32)
                col_idx = (np.arange(img_w) * feat_w / img_w).astype(np.int32)
                content_mask = content_mask_feat[np.ix_(row_idx, col_idx)]
            else:
                content_mask = content_mask_feat

            # Both masks empty: no text in PDF or image -> no OCR needed.
            intersection = np.count_nonzero(bbox_mask & content_mask)
            union        = np.count_nonzero(bbox_mask | content_mask)
            iou = intersection / union if union > 0 else 1.0
            return bool(iou < iou_threshold)

        except Exception:
            return None

    def get_picture_detections(self):
        """
        Return picture-class detections from the last predict() call,
        rescaled from model resolution to page pixel coordinates.

        Runs picture CCL on the first call after each predict() (lazy).
        Subsequent calls within the same predict() cycle are free.

        Returns:
            list of [x1, y1, x2, y2] in page pixel space.

        Raises:
            RuntimeError if predict() has not been called.
        """
        if self._combined is None:
            raise RuntimeError("predict() must be called before get_picture_detections()")

        self._ensure_picture_detections()

        resize_x = self._page_w / self._target_w
        resize_y = self._page_h / self._target_h

        bboxes = []
        for det in self._detections_picture:
            if det['score'] > _DETECTION_SCORE_THRESHOLD:
                b = det['bbox']
                bboxes.append([
                    b[0] * resize_x,
                    b[1] * resize_y,
                    b[2] * resize_x,
                    b[3] * resize_y,
                ])
        return bboxes
