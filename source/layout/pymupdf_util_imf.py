"""
Image Model Features (IMF) extraction
Maintained by: AI Researchers
Purpose: Extract image-based features using neural network (ONNX)

image_feature_extraction_task() is the stable interface for pymupdf_util_ext.py.
Inference and detection logic lives in ImageFeatureExtractorV1 / V2.

feature_map / class_logits are returned SEPARATELY (not concatenated):
they have different statistical character (continuous embedding vs.
per-class score) and are meant to be pooled differently downstream -- see
roi_pooling.DEFAULT_FEATURE_MAP_POOLING_OPS / DEFAULT_CLASS_LOGITS_POOLING_OPS.

Detector-based bbox helpers (image-only path):
    _ensure_predicted(feature_extractor, page_img)
        Shared predict() / cache protocol used by apply_dtext_boxes(),
        apply_dimage_boxes(), apply_dimage_rect_boxes(), and
        apply_detector_table_lines()/apply_detector_table_rect() (which
        import it from here). Canonical home is this module because the
        protocol is not specific to tables -- it is the general contract
        for all feature_extractor consumers.

    apply_dtext_boxes(data_dict, feature_extractor, db_thresh)
        Append detector-based text bboxes (empty text, no OCR).
        Requires model exported with use_text_seg=True.
        Image-only counterpart of the 'text' input_type.

    apply_dimage_boxes(data_dict, feature_extractor)
        Append detector-based picture bboxes from get_layout_detections()
        'picture' class. Image-only counterpart of the 'image' input_type.

    apply_dimage_rect_boxes(page, data_dict, feature_extractor, tol_px=10)
        Handles 'dimage_rect'. PDF-path only (unlike apply_dtext_boxes/
        apply_dimage_boxes above, which are image-only-path helpers).
        Same detector source as apply_dimage_boxes
        (get_picture_detections()), but instead of adding the raw
        detector box unconditionally, tightens it to the union of
        embedded-PDF-image, vector-line, and already-extracted bbox
        evidence found inside it, and DROPS the region entirely if
        nothing is found inside -- treating an empty detection as a
        false positive. Embedded-image and vector-line evidence are
        fetched independent of whether 'image'/'vec_line' were
        requested (see its own docstring for why). Same "evidence
        required" contract as apply_detector_table_rect() ('dtable_rect',
        pymupdf_util_table.py) applied to picture regions.
"""

from .pymupdf_util_base import BOX_IMAGE, BOX_TEXT, get_vector_lines, merge_lines


# ---------------------------------------------------------------------------
# Shared predict() / cache protocol
# ---------------------------------------------------------------------------

def _ensure_predicted(feature_extractor, page_img):
    """
    Ensure feature_extractor.predict() has been run for this page before
    reading any detection results, calling it here if necessary.

    Two calling patterns exist across callers of create_input_data_from_page()
    and create_input_data_from_image(), and this function handles both:

    1. Inference (e.g. BoxRFDGNN.predict()): the caller already calls
       feature_extractor.predict(page_img) and mark_cached() BEFORE
       create_input_data_from_page(), typically to reuse a rasterization
       it needed for its own purposes. Here, consume_cache() returns True
       -- predict() must NOT be called again.

    2. Training data generation (e.g. DocumentJsonDataset): the caller
       passes feature_extractor straight through without ever calling
       predict() itself. Here, consume_cache() returns False, so predict()
       is called here.

    After resolving either case, mark_cached() is (re-)set so that a
    later consumer within the same pipeline call can likewise detect
    "already predicted for this page" via its own consume_cache() check.

    This function is the canonical home for the protocol; it was previously
    duplicated in pymupdf_util_table.py, which now imports it from here.
    """
    if not feature_extractor.consume_cache():
        feature_extractor.predict(page_img)
    feature_extractor.mark_cached()


# ---------------------------------------------------------------------------
# Detector-based bbox helpers (image-only path)
# ---------------------------------------------------------------------------

def apply_dtext_boxes(data_dict, feature_extractor, db_thresh=0.3, txt_func=None):
    """
    Append detector-based text bboxes to data_dict.

    Two operating modes, selected by txt_func:

    Default mode (txt_func=None):
        Uses ImageFeatureExtractorV2.get_text_detection() to locate text
        regions. Text content is left empty (no OCR). Requires feature_extractor
        exported with use_text_seg=True.

    Custom mode (txt_func provided):
        Calls txt_func(page_img) instead of get_text_detection(). The function
        is responsible for both detection AND recognition, returning bboxes with
        text content. feature_extractor.predict() is still called for the image
        feature map (Step 2 / IMF) if feature_extractor is not None; otherwise
        only the txt_func results are added and predict() is skipped entirely.

        Signature:
            txt_func(page_img: np.ndarray) -> list of [x1, y1, x2, y2, text]
                page_img: (H, W, C) uint8, same array as data_dict['image'].
                Return coordinates must be in page pixel space (same coordinate
                system as get_text_detection() output). text is a str; may be
                empty if the caller has detection but no recognition result.

    In both modes, pixel coordinates are converted to PDF point coordinates
    using page_width / page_height from data_dict (1 pt == 1 px on the
    image-only path), and each accepted region gets box_type BOX_TEXT so
    downstream consumers (make_custom_feature, is_ocr_needed) treat it
    identically to a PDF-extracted text line.

    Args:
        data_dict:         data dict with 'image', 'page_width',
                           'page_height', 'bboxes', 'text', 'box_type'.
                           Mutated in place.
        feature_extractor: ImageFeatureExtractorV2 instance, or None only
                           when txt_func is provided (in that case predict()
                           is skipped and no image feature map is produced
                           at this step; Step 2 will also be skipped).
        db_thresh:         binarization threshold forwarded to
                           get_text_detection() in default mode. Ignored
                           when txt_func is provided.
        txt_func:          Optional callable -- see above. When None, default
                           mode (get_text_detection) is used.

    Raises:
        RuntimeError: (default mode only) if the model was not exported with
                      use_text_seg=True (propagated from get_text_detection()).
    """
    page_width  = data_dict['page_width']
    page_height = data_dict['page_height']
    page_img    = data_dict['image']
    img_h, img_w = page_img.shape[:2]
    scale_x = page_width  / img_w
    scale_y = page_height / img_h
    box_type = data_dict['box_type']

    if txt_func is not None:
        # Custom mode: txt_func supplies both bbox and text content.
        # Still call predict() for the image feature map if feature_extractor
        # is available, so Step 2 (IMF) can reuse the cached result.
        if feature_extractor is not None:
            _ensure_predicted(feature_extractor, page_img)

        raw_results = txt_func(page_img)

        for entry in raw_results:
            px1, py1, px2, py2, text = entry
            x1 = float(px1) * scale_x
            y1 = float(py1) * scale_y
            x2 = float(px2) * scale_x
            y2 = float(py2) * scale_y

            if not (0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height):
                continue

            bbox = [x1, y1, x2, y2]
            if bbox not in data_dict['bboxes']:
                data_dict['bboxes'].append(bbox)
                data_dict['text'].append(text)
                box_type.append(BOX_TEXT)

    else:
        # Default mode: get_text_detection() supplies bboxes; text is empty.
        _ensure_predicted(feature_extractor, page_img)

        try:
            pixel_boxes = feature_extractor.get_text_detection(db_thresh=db_thresh)
        except RuntimeError:
            # Model not exported with use_text_seg=True -- silently skip.
            # Callers that require this capability should check in advance
            # (e.g. by inspecting feature_extractor._text_logits is not None).
            return

        for (px1, py1, px2, py2) in pixel_boxes:
            x1 = float(px1) * scale_x
            y1 = float(py1) * scale_y
            x2 = float(px2) * scale_x
            y2 = float(py2) * scale_y

            if not (0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height):
                continue

            bbox = [x1, y1, x2, y2]
            if bbox not in data_dict['bboxes']:
                data_dict['bboxes'].append(bbox)
                data_dict['text'].append('')
                box_type.append(BOX_TEXT)


def apply_dimage_boxes(data_dict, feature_extractor):
    """
    Append detector-based picture bboxes to data_dict.

    Uses ImageFeatureExtractorV2.get_picture_detections() to locate picture
    regions in the rasterized page image. Pixel coordinates are converted to
    PDF point coordinates using page_width / page_height from data_dict.

    This is the image-only counterpart of the 'image' input_type (which
    reads embedded raster images from the PDF page object via
    page.get_image_info()). Use 'dimage' when no PDF page object is
    available (create_input_data_from_image()) or when the page is a
    scanned image and get_image_info() would return the whole-page scan
    rather than individual picture regions.

    Args:
        data_dict:         data dict with 'image', 'page_width',
                           'page_height', 'bboxes', 'text', 'box_type'.
                           Mutated in place.
        feature_extractor: ImageFeatureExtractorV2 instance. predict() will
                           be called automatically if not already done
                           (see _ensure_predicted()).
    """
    _ensure_predicted(feature_extractor, data_dict['image'])

    page_width  = data_dict['page_width']
    page_height = data_dict['page_height']
    img_h, img_w = data_dict['image'].shape[:2]
    scale_x = page_width  / img_w
    scale_y = page_height / img_h

    box_type = data_dict['box_type']

    for (px1, py1, px2, py2) in feature_extractor.get_picture_detections():
        x1 = float(px1) * scale_x
        y1 = float(py1) * scale_y
        x2 = float(px2) * scale_x
        y2 = float(py2) * scale_y

        if not (0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height):
            continue

        bbox = [x1, y1, x2, y2]
        if bbox not in data_dict['bboxes']:
            data_dict['bboxes'].append(bbox)
            data_dict['text'].append('')
            box_type.append(BOX_IMAGE)


def apply_dimage_rect_boxes(page, data_dict, feature_extractor, tol_px=10):
    """
    Append 'dimage_rect' bboxes to data_dict: for each picture region
    detected by feature_extractor.get_picture_detections(), tighten the
    raw detector box to the union of whatever content is actually found
    inside it -- embedded PDF images, vector lines, and any bbox already
    present in data_dict['bboxes'] (text, other images, lines, ...)
    whose center falls inside the (tolerance-expanded) detected region.
    A detected region with nothing found inside it is treated as a
    false positive and is NOT added.

    PDF-path only (requires a real page, unlike apply_dtext_boxes/
    apply_dimage_boxes which are image-only-path helpers). This is the
    same "evidence required" contract as apply_detector_table_rect()
    ('dtable_rect', pymupdf_util_table.py) applied to picture regions
    instead of table regions, and it belongs on the PDF path for the
    same reason that function's whole point is being applied there
    too: it validates a detector's region guess against REAL,
    PDF-extracted content (actual embedded images, actual vector
    paths). On the image-only path the only available "evidence" would
    itself be other detector output (dtext/dimage/dtable_img_line) --
    a weak, circular validation of one detector's guess against
    another's -- so this input_type is intentionally NOT offered there
    (see _IMAGE_ONLY_VALID_INPUT_TYPES in pymupdf_util.py).

    Embedded-image evidence (page.get_image_info()) is ALWAYS fetched
    fresh here, regardless of whether 'image' is in input_type -- this
    matters because an actual embedded raster image at that location is
    the single strongest, most direct signal that a detected region
    really is a picture. If this evidence were only read from
    data_dict['bboxes'] (i.e. only present when 'image' was ALSO
    requested by the caller), a genuine picture region with no nearby
    text or vector lines would end up with zero evidence whenever
    'image' wasn't separately requested, and get wrongly dropped as a
    false positive purely because of an unrelated input_type choice.
    Vector-line evidence is similarly made independent of whether
    'vec_line'/'table_vec_line'/etc. was requested: reused from
    data_dict['vector_lines'] if already cached by an earlier call in
    this same create_input_data_from_page() call, otherwise computed
    fresh here (same fallback as apply_detector_table_rect()). Text
    evidence is the one exception left as opt-in: it is only used if
    'text' was ALSO requested and already populated data_dict['bboxes']
    -- independently re-parsing PDF text spans here would duplicate
    Step 1's stext_page work, which is out of scope for this function.

    Args:
        page:              PyMuPDF page object. Required -- this
                           input_type is PDF-path only.
        data_dict:         data dict with 'image', 'page_width',
                           'page_height', 'bboxes', 'text', 'box_type'.
                           Mutated in place.
        feature_extractor: ImageFeatureExtractorV2 instance. predict() will
                           be called automatically if not already done
                           (see _ensure_predicted()).
        tol_px:            float, pixel-space tolerance (converted to
                           point space via scale_x/scale_y) by which the
                           detected region is expanded before checking
                           what falls inside it -- absorbs the layout
                           detection model's regression noise on the
                           detected box edges, same rationale as
                           apply_detector_table_lines' tolerance.
    """
    _ensure_predicted(feature_extractor, data_dict['image'])

    page_width  = data_dict['page_width']
    page_height = data_dict['page_height']
    img_h, img_w = data_dict['image'].shape[:2]
    scale_x = page_width  / img_w
    scale_y = page_height / img_h

    tol_x = tol_px * scale_x
    tol_y = tol_px * scale_y

    box_type = data_dict['box_type']

    # Vector-line evidence: reuse cached lines if an earlier
    # 'vec_line'/'table_vec_line'/etc. call in this same
    # create_input_data_from_page() call already computed them,
    # otherwise compute fresh -- independent of whether any
    # vector-related input_type was actually requested.
    if 'vector_lines' in data_dict:
        h_lines, v_lines = data_dict['vector_lines']
    else:
        h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
        h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
        v_lines = merge_lines(v_lines, orientation='v', tolerance=3)
        data_dict['vector_lines'] = (h_lines, v_lines)
    evidence_lines = [(r.x0, r.y0, r.x1, r.y1) for r in list(h_lines) + list(v_lines)]

    # Embedded-image evidence: always fetched fresh from the PDF page
    # itself, independent of whether 'image' was requested -- see
    # docstring above for why this can't be left to data_dict['bboxes'].
    evidence_images = [tuple(itm["bbox"]) for itm in page.get_image_info()]

    for (px1, py1, px2, py2) in feature_extractor.get_picture_detections():
        rx0 = float(px1) * scale_x
        ry0 = float(py1) * scale_y
        rx1 = float(px2) * scale_x
        ry1 = float(py2) * scale_y

        ex0, ey0 = rx0 - tol_x, ry0 - tol_y
        ex1, ey1 = rx1 + tol_x, ry1 + tol_y

        found_x0, found_y0, found_x1, found_y1 = [], [], [], []

        # Evidence 1: vector lines overlapping the expanded region.
        for (lx0, ly0, lx1, ly1) in evidence_lines:
            lx0, lx1 = sorted((lx0, lx1))
            ly0, ly1 = sorted((ly0, ly1))
            if lx1 < ex0 or lx0 > ex1 or ly1 < ey0 or ly0 > ey1:
                continue
            found_x0.append(lx0); found_x1.append(lx1)
            found_y0.append(ly0); found_y1.append(ly1)

        # Evidence 2: embedded PDF images overlapping the expanded region.
        for (ix0, iy0, ix1, iy1) in evidence_images:
            ix0, ix1 = sorted((ix0, ix1))
            iy0, iy1 = sorted((iy0, iy1))
            if ix1 < ex0 or ix0 > ex1 or iy1 < ey0 or iy0 > ey1:
                continue
            found_x0.append(ix0); found_x1.append(ix1)
            found_y0.append(iy0); found_y1.append(iy1)

        # Evidence 3: any bbox already extracted (text, other images,
        # lines, ...) whose center falls inside the expanded region.
        for (bx0, by0, bx1, by1) in data_dict['bboxes']:
            cx = (bx0 + bx1) / 2.0
            cy = (by0 + by1) / 2.0
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                found_x0.append(bx0); found_x1.append(bx1)
                found_y0.append(by0); found_y1.append(by1)

        if not found_x0:
            # Nothing found inside the detected region -- treat as a
            # false-positive detection and skip it entirely.
            continue

        x1v = max(0.0, min(found_x0))
        y1v = max(0.0, min(found_y0))
        x2v = min(page_width, max(found_x1))
        y2v = min(page_height, max(found_y1))

        if not (0 <= x1v < x2v <= page_width and 0 <= y1v < y2v <= page_height):
            continue

        bbox = [x1v, y1v, x2v, y2v]
        if bbox not in data_dict['bboxes']:
            data_dict['bboxes'].append(bbox)
            data_dict['text'].append('')
            box_type.append(BOX_IMAGE)


# ---------------------------------------------------------------------------
# Step 2 interface (used by pymupdf_util_ext.py)
# ---------------------------------------------------------------------------

def image_feature_extraction_task(page_img, feature_extractor, input_type, aug_fetmap=None):
    """
    Extract image-based features and segmentation-derived bboxes.

    Args:
        page_img:         np.ndarray (H, W, C), uint8 — page raster image
        feature_extractor: ImageFeatureExtractorV1 / V2 instance
        input_type:       tuple of element types; 'seg-image' enables bbox detection
        aug_fetmap:       optional extra channel map passed to predict()

    Returns:
        feature_map:      decoder embedding output, shape (1, 5*F, H, W)
        class_logits:     per-class segmentation logits, shape (1, C, H, W)
        bboxes_to_add:    list of [x1, y1, x2, y2] in page pixel space
        box_types_to_add: list of box type strings (parallel to bboxes_to_add)
    """
    # Skip inference if BoxRFDGNN.is_image_page() already ran predict() for this page.
    if not feature_extractor.consume_cache():
        feature_extractor.predict(page_img, aug_fetmap=aug_fetmap)

    feature_map = feature_extractor.get_feature_map()
    class_logits = feature_extractor.get_class_logits()
    bboxes_to_add = []
    box_types_to_add = []

    if 'seg-image' in input_type:
        for bbox in feature_extractor.get_picture_detections():
            bboxes_to_add.append(bbox)
            box_types_to_add.append(BOX_IMAGE)

    return feature_map, class_logits, bboxes_to_add, box_types_to_add
