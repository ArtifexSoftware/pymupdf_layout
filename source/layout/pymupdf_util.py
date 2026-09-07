"""
PDF processing utilities - Main integration module
This module combines base PDF parsing and experimental feature extraction.

Interface:
    create_input_data_from_page(page, options=None)
    create_input_data_from_image(page_img, options=None)
    create_input_data_by_pymupdf(pdf_path=None, document=None, page_no=0, options=None)

Options dict keys (with defaults):
    input_type          : tuple   = ('text',)       — Element types to extract. Two categories:

                                                       PDF-path types (create_input_data_from_page only):
                                                         'text'            PDF-extracted text lines with content.
                                                         'image'           Embedded raster images via get_image_info().
                                                         'picture_clusters' Clustered vector+image drawings.
                                                         'vec_line'        All detected PDF vector lines.
                                                         'img_line'        All lines detected via Sobel edge detection on
                                                                           the rasterized page image -- the image-based
                                                                           counterpart of 'vec_line'. Useful for PDFs where
                                                                           lines are encoded as filled rects or embedded
                                                                           images rather than explicit vector drawings.
                                                         'table_vec_line'  Vector lines belonging to heuristic table grids
                                                                           (closed-border + open-sided tables, i.e.
                                                                           table_type='all').
                                                         'table_vec_line_full'    Same, closed-border tables only.
                                                         'table_vec_line_partial' Same, open-sided tables only.
                                                         'table_img_line'  Sobel-detected lines in heuristic table grids
                                                                           (closed-border + open-sided tables, i.e.
                                                                           table_type='all').
                                                         'table_img_line_full'    Same, closed-border tables only.
                                                         'table_img_line_partial' Same, open-sided tables only.
                                                         'table_rect_full' ONE bbox per closed-border table (the
                                                                           table's outer bbox only, no internal
                                                                           border/divider lines) -- fewer graph
                                                                           nodes than table_vec_line_full.
                                                         'img_table_rect_full' Same as table_rect_full, sourced
                                                                           from Sobel image-based lines instead
                                                                           of PDF vector paths.
                                                         'stable_img_line' Hybrid: heuristic table regions
                                                                           (Sobel + find_table_grids, same as
                                                                           table_img_line) filtered by the model's
                                                                           per-pixel table-class softmax purity.
                                                                           Only regions whose mean table probability
                                                                           >= stable_img_line_purity (default 0.5)
                                                                           have their img_lines extracted. Combines
                                                                           OOD generalization of heuristics with
                                                                           in-distribution precision of the model.
                                                                           Requires feature_extractor. PDF-path only.
                                                         'stable_img_line_full'    Same, closed-border tables only.
                                                         'stable_img_line_partial' Same, open-sided tables only.
                                                         'dimage_rect'     One bbox per detected picture region
                                                                           (from feature_extractor.get_picture_detections()),
                                                                           tightened to embedded-image/vector-line/
                                                                           already-extracted-bbox content actually
                                                                           found inside it; empty regions are dropped
                                                                           as false-positive detections. Needs a real
                                                                           page (unlike 'dimage' below) since it
                                                                           validates against REAL PDF content.

                                                       Detector-based types (both paths; image-only path preferred):
                                                         'dtext'           Text bboxes from get_text_detection() (no OCR,
                                                                           empty content). Requires use_text_seg=True model.
                                                                           Image-only counterpart of 'text'.
                                                         'dimage'          Picture bboxes from get_layout_detections()
                                                                           'picture' class. Image-only counterpart of 'image'.
                                                         'seg-image'       Picture bboxes from get_picture_detections().
                                                         'dtable_vec_line' Vector lines in detector-based table regions.
                                                                           No-op when page=None (image-only path).
                                                         'dtable_img_line' Sobel lines in detector-based table regions.
                                                         'dtable_rect'     One bbox per detected table region,
                                                                           tightened to whatever content is actually
                                                                           found inside it; empty regions are
                                                                           dropped as false-positive detections.

    feature_set_name    : str     = 'rf+imf'        — Features to extract: 'rf', 'imf', 'yf', 'jf' (combined with '+')
    max_image_num       : int     = 500             — Maximum number of images to extract
    max_vec_line_num    : int     = 200             — Maximum number of vector lines to extract
    feature_extractor   : object  = None            — ONNX model used for image-based feature extraction (Step 2),
                                                       AND for all detector-based input types (Step 1.5).
                                                       The caller does NOT need to have called predict() beforehand
                                                       -- Step 1.5 calls it automatically if needed (see
                                                       pymupdf_util_imf._ensure_predicted()). If the caller DID
                                                       already call predict() (e.g. BoxRFDGNN), that cached result
                                                       is reused instead of re-predicting.
    stable_img_line_purity : float = 0.5            — Purity threshold for 'stable_img_line*': minimum mean
                                                       softmax probability of the 'table' class over a
                                                       heuristic region's crop in the model's logit map.
                                                       Regions below this threshold are discarded. Only used
                                                       when input_type contains a 'stable_img_line*' value.
    bbox_num_threshold  : int     = 0               — Return early if bbox count exceeds this (0 = disabled)
    page_img            : ndarray = None            — Optional pre-rasterized (H, W, C) uint8 page image.
                                                       create_input_data_from_page() only: pass this when the
                                                       caller already rasterized the page to avoid rasterizing twice.
                                                       create_input_data_from_image() always receives page_img as
                                                       its first positional argument instead.
    txt_func            : callable = None           — create_input_data_from_image() + 'dtext' only.
                                                       When provided, called instead of get_text_detection() to
                                                       produce text regions WITH content (e.g. an OCR engine).
                                                       Signature: txt_func(page_img) -> list of
                                                       [x1, y1, x2, y2, text] in page pixel coordinates.
                                                       When None (default), get_text_detection() is used and
                                                       text content is left empty.
"""

import os
import numpy as np
import pymupdf

from .pymupdf_util_base import extract_base_elements, make_custom_feature
from .pymupdf_util_ext import apply_feature_extractors
from .pymupdf_util_imf import apply_dtext_boxes, apply_dimage_boxes, apply_dimage_rect_boxes
from .pymupdf_util_table import (
    apply_detector_table_lines, apply_detector_table_rect,
    apply_detector_lsd_lines,
    apply_stable_img_line,
    TABLE_INPUT_TYPE_EXCLUSIVE_GROUP,
)


def create_input_data_from_page(page, options=None):
    """
    Create input data from a PDF page with optional feature extraction.

    Pipeline:
    1. Extract base PDF elements (text, images, vectors, checkboxes)
    2. Optionally apply feature extractors (RF, IMF, YF, JF)

    Args:
        page: PyMuPDF page object
        options: Configuration dict. See module docstring for available keys.

    Returns:
        data_dict with guaranteed keys:
            bboxes, text, custom_features, page_width, page_height, image, file_path
        Optional keys (depending on options):
            stext_page, feature_map, box_type
    """
    if options is None:
        options = {}
    input_type = options.get('input_type', ('text',))
    feature_set_name = options.get('feature_set_name', 'rf+imf')
    max_image_num = options.get('max_image_num', 500)
    max_vec_line_num = options.get('max_vec_line_num', 200)
    feature_extractor = options.get('feature_extractor', None)
    bbox_num_threshold = options.get('bbox_num_threshold', 0)
    page_img = options.get('page_img', None)
    stable_img_line_purity = options.get('stable_img_line_purity', 0.5)

    # Validate input_type: reject any unrecognized values early so callers get
    # a clear error instead of silently producing no output for that type.
    unknown_types = [t for t in input_type if t not in _PAGE_VALID_INPUT_TYPES]
    if unknown_types:
        raise ValueError(
            f"create_input_data_from_page() received unrecognized input_type "
            f"value(s): {unknown_types}. Valid values: {sorted(_PAGE_VALID_INPUT_TYPES)}"
        )

    _check_table_input_type_exclusivity(input_type)

    # Step 1: Extract base PDF elements
    data_dict = extract_base_elements(
        page=page,
        input_type=input_type,
        max_image_num=max_image_num,
        max_vec_line_num=max_vec_line_num,
        page_img=page_img,
    )

    # Step 1.5: Detector-based table line extraction ('dtable_vec_line' /
    # 'dtable_img_line'). Implementation lives in pymupdf_util_table.py
    # (alongside 'table_vec_line'/'table_img_line', dispatched from
    # within extract_base_elements() itself) rather than here or in
    # pymupdf_util_base.py, since it depends on feature_extractor (a CV
    # model) and pymupdf_util_base.py stays model-agnostic, stable
    # PDF-parsing-only code (see its own docstring). No-op if neither
    # input_type value is requested. feature_extractor.predict() is
    # called automatically here if the caller hasn't already done so
    # (see pymupdf_util_table._ensure_predicted()) -- this supports both
    # inference callers (which pre-call predict()) and training data
    # generation callers (which don't).
    apply_detector_table_lines(
        page=page,
        data_dict=data_dict,
        input_type=input_type,
        max_vec_line_num=max_vec_line_num,
        feature_extractor=feature_extractor,
    )

    # Step 1.5 (continued): 'dtable_rect' -- detector-based table region,
    # tightened to whatever content is actually found inside it, with
    # empty regions dropped as false positives. Separate function from
    # apply_detector_table_lines because it produces a single region bbox
    # per table rather than filtered lines, and its own false-positive
    # check depends on data_dict['bboxes'] already containing whatever
    # 'text'/'image'/vec_line/etc. elements Step 1 extracted. No-op if
    # 'dtable_rect' is not requested.
    apply_detector_table_rect(
        page=page,
        data_dict=data_dict,
        input_type=input_type,
        feature_extractor=feature_extractor,
    )

    # Step 1.5 (continued): 'dtable_lsd_line' -- detector-based table line
    # extraction using LSD instead of Sobel. Same region source as
    # 'dtable_img_line' (feature_extractor layout detections), different line
    # extraction algorithm. No full/partial variant (see apply_detector_lsd_lines).
    # cv2 is required; ImportError propagates to the caller if not installed.
    # No-op if 'dtable_lsd_line' is not in input_type.
    apply_detector_lsd_lines(
        page=page,
        data_dict=data_dict,
        input_type=input_type,
        max_vec_line_num=max_vec_line_num,
        feature_extractor=feature_extractor,
    )

    # Step 1.5 (continued): 'stable_img_line*' -- hybrid heuristic+model
    # table line extraction. Must run after extract_base_elements (Step 1)
    # since it needs data_dict['box_type'] as a list. No-op if none of
    # the 'stable_img_line*' values are in input_type.
    apply_stable_img_line(
        page=page,
        data_dict=data_dict,
        input_type=input_type,
        max_vec_line_num=max_vec_line_num,
        feature_extractor=feature_extractor,
        purity_threshold=stable_img_line_purity,
    )

    # Step 1.5 (continued): 'dimage_rect' -- detector-based picture
    # region, tightened to whatever content is actually found inside it
    # (embedded PDF images, vector lines, already-extracted bboxes), with
    # empty regions dropped as false positives. PDF-path only, unlike
    # 'dtext'/'dimage' above (those are image-only-path helpers) -- see
    # apply_dimage_rect_boxes()'s own docstring for why this one needs a
    # real page. No-op if 'dimage_rect' is not requested.
    if 'dimage_rect' in input_type:
        if feature_extractor is None:
            raise ValueError("'dimage_rect' input_type requires a feature_extractor")
        apply_dimage_rect_boxes(page, data_dict, feature_extractor)

    # Early return check
    if len(data_dict['bboxes']) > bbox_num_threshold > 0:
        _ensure_custom_features(data_dict)
        return data_dict

    debug_extraction = False
    if debug_extraction:
        print("EXTRACTED BBOXES")
        list = zip(data_dict['bboxes'], data_dict['box_type'], data_dict['text'])
        for i, info in enumerate(list):
            bbox = info[0]
            t = info[1]
            txt = info[2]
            print(f'{i}: [{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}] {t} "{txt}"')
        print("END OF BBOXES")

    # Step 2: Apply feature extractors (if requested)
    if feature_set_name:
        # Prepare page_dict for YF features if needed
        page_dict = None
        if 'yf' in feature_set_name or 'ymf' in feature_set_name:
            stext_page = data_dict.get('stext_page')
            if stext_page is not None:
                page_dict = page.get_text("dict", textpage=stext_page)
                page_dict['width'] = data_dict['page_width']
                page_dict['height'] = data_dict['page_height']

        data_dict = apply_feature_extractors(
            data_dict=data_dict,
            feature_set_name=feature_set_name,
            feature_extractor=feature_extractor,
            input_type=input_type,
            page=page,
            page_dict=page_dict,
        )
    else:
        _ensure_custom_features(data_dict)

    export_markedup_image = False
    if export_markedup_image:
        import pprint
        pprint.pp(data_dict)

        newdoc = pymupdf.open(page.parent.name)
        newpage = newdoc[page.number]
        data = zip(data_dict['bboxes'], data_dict['box_type'])
        RED=(1,0,0)
        for i, info in enumerate(data):
            bbox = info[0]
            name = info[1]
            newpage.draw_rect(bbox, color=RED, width=1)
            newpage.insert_text(bbox[2:], f"{i}: {name}", color=RED, fontsize=5)
        newdoc.ez_save("layout_input.pdf")

    return data_dict


def _check_table_input_type_exclusivity(input_type):
    """
    Raise ValueError if more than one mutually-exclusive table input_type
    is requested simultaneously.

    All table-oriented input_type values (table_vec_line*, table_img_line*,
    table_rect_full, img_table_rect_full, dtable_vec_line, dtable_img_line,
    dtable_rect, stable_img_line*) serve the same purpose -- representing
    table structure as line or region nodes in the GNN graph -- via different
    detection strategies. Combining more than one produces duplicate or
    near-duplicate bboxes for the same table, which pollutes the graph
    without adding useful signal. The full exclusive group is defined in
    pymupdf_util_table.TABLE_INPUT_TYPE_EXCLUSIVE_GROUP.
    """
    active = [t for t in input_type if t in TABLE_INPUT_TYPE_EXCLUSIVE_GROUP]
    if len(active) > 1:
        raise ValueError(
            f"input_type contains more than one mutually-exclusive table "
            f"input_type value: {sorted(active)}. "
            f"Choose exactly one table detection strategy. "
            f"All exclusive values: {sorted(TABLE_INPUT_TYPE_EXCLUSIVE_GROUP)}"
        )


def _ensure_custom_features(data_dict):
    """
    Ensure custom_features exists in data_dict for backward compatibility.
    Legacy code expects custom_features to always be present.
    """
    if 'custom_features' not in data_dict:
        data_dict['custom_features'] = []
        box_type = data_dict.get('box_type', [])

        for row_idx in range(len(data_dict['bboxes'])):
            bt = box_type[row_idx] if row_idx < len(box_type) else 'unknown'
            text = data_dict['text'][row_idx]
            data_dict['custom_features'].append(make_custom_feature(bt, text))


def _build_image_only_data_dict(page_img):
    """
    Build a minimal data_dict skeleton from a rasterized page image.

    This is the image-only equivalent of extract_base_elements()'s
    initialization block: it sets the same guaranteed keys so the rest of
    the pipeline (apply_dtext_boxes, apply_dimage_boxes,
    apply_detector_table_lines, apply_feature_extractors) can consume
    data_dict without knowing whether it came from a PDF page or a raw image.

    Fixed values for image-only pages:
        has_raster_image  = True  (the whole page IS the raster image)
        has_embedded_text = False (no PDF text layer; OCR not done here)
        file_path         = None  (no source PDF)

    Args:
        page_img: np.ndarray (H, W, C) uint8, page raster image.

    Returns:
        data_dict with keys: bboxes, text, box_type, page_width, page_height,
        image, has_raster_image, has_embedded_text, file_path.
    """
    img_h, img_w = page_img.shape[:2]
    return {
        'bboxes':            [],
        'text':              [],
        'box_type':          [],
        'page_width':        float(img_w),
        'page_height':       float(img_h),
        'image':             page_img,
        'has_raster_image':  True,
        'has_embedded_text': False,
        'file_path':         None,
    }


# All input_type values recognized by create_input_data_from_page().
# Any value NOT in this set raises ValueError so callers get a clear error
# instead of silently getting no output (the old behavior was to ignore
# unknown values, which made typos and stale type names very hard to debug).
_PAGE_VALID_INPUT_TYPES = frozenset({
    # PDF-native element types
    'text',
    'text_pm',
    'image',
    'picture_clusters',
    'vec_line',
    'img_line',
    # Heuristic table types (vec-based)
    'table_vec_line',
    'table_vec_line_full',
    'table_vec_line_partial',
    'table_rect_full',
    # Heuristic table types (img-based)
    'table_img_line',
    'table_img_line_full',
    'table_img_line_partial',
    'img_table_rect_full',
    # Detector-based types (require feature_extractor)
    'dtext',
    'dimage',
    'seg-image',
    'dimage_rect',
    'dtable_vec_line',
    'dtable_img_line',
    'dtable_lsd_line',
    'dtable_rect',
    # Hybrid heuristic+model types (require feature_extractor, PDF-path only)
    'stable_img_line',
    'stable_img_line_full',
    'stable_img_line_partial',
})

# Image-only input types that are valid for create_input_data_from_image().
# PDF-dependent types ('text', 'image', 'picture_clusters', 'vec_line',
# 'table_vec_line', 'table_vec_line_full', 'table_vec_line_partial',
# 'table_img_line', 'table_img_line_full', 'table_img_line_partial',
# 'table_rect_full', 'img_table_rect_full', 'dimage_rect', 'text_pm')
# are not in this set and will raise ValueError if requested on the
# image-only path. Note 'table_img_line*'/'img_table_rect_full' are
# Sobel image-based, not PDF-vector-based, but they are still page-only:
# they are wired through extract_base_elements() (Step 1), which only
# runs on the create_input_data_from_page() path. 'dimage_rect' is
# PDF-path only for a different reason -- it validates a detected
# picture region against REAL PDF content (embedded images, vector
# lines), which only exists on the PDF path; see
# apply_dimage_rect_boxes()'s own docstring. 'dtable_rect' below is the
# detector-based table-region counterpart and DOES work on both paths
# (its own evidence sources -- Sobel lines / already-extracted bboxes --
# are meaningful on the image-only path too).
_IMAGE_ONLY_VALID_INPUT_TYPES = frozenset({
    'dtext',
    'dimage',
    'seg-image',
    'dtable_vec_line',   # no-op when page=None, but not an error
    'dtable_img_line',
    'dtable_lsd_line',
    'dtable_rect',
})


def create_input_data_from_image(page_img, options=None):
    """
    Create input data from a rasterized page image (no PDF page object).

    Use this function when the document page is available only as a raster
    image -- e.g. a scanned document, an image converted to PDF with no text
    layer, or a direct image file. All element extraction is detector-based
    (feature_extractor required for most useful input types).

    Pipeline:
    1. Build a data_dict skeleton from the image shape (no PDF parsing).
    2. Apply detector-based bbox extractors ('dtext', 'dimage', 'seg-image').
    3. Apply detector-based table extractors ('dtable_vec_line' is a
       no-op here since there are no PDF vector paths; 'dtable_img_line'
       and 'dtable_rect' work normally).
    4. Apply feature extractors (RF, IMF, YF, JF) -- same Step 2 as
       create_input_data_from_page(), fully reused.

    Supported input_type values:
        'dtext'           Text bboxes from get_text_detection() (empty content).
                          Requires model exported with use_text_seg=True.
        'dimage'          Picture bboxes from get_layout_detections() 'picture'
                          class. Detector-based counterpart of 'image'.
        'seg-image'       Picture bboxes from get_picture_detections().
        'dtable_vec_line' No-op (no PDF vector paths); accepted without error.
        'dtable_img_line' Sobel lines filtered to detector-based table regions.
        'dtable_rect'     One bbox per detected table region, tightened to
                          whatever content (Sobel lines / already-extracted
                          bboxes) is actually found inside it; regions with
                          nothing inside are dropped as false positives.

    Unsupported types ('text', 'image', 'picture_clusters', 'vec_line',
    'table_vec_line', 'table_vec_line_full', 'table_vec_line_partial',
    'table_img_line', 'table_img_line_full', 'table_img_line_partial',
    'table_rect_full', 'img_table_rect_full', 'dimage_rect', 'text_pm')
    raise ValueError.

    Args:
        page_img: np.ndarray (H, W, C) uint8 -- rasterized page image.
                  Coordinate space for all returned bboxes: image pixels
                  re-expressed as "PDF points" where 1 point == 1 pixel
                  (page_width = img_w, page_height = img_h).
        options:  Configuration dict. See module docstring for available keys.
                  'page_img' option key is ignored here (use the argument).

    Returns:
        data_dict with guaranteed keys:
            bboxes, text, box_type, custom_features,
            page_width, page_height, image, file_path.
        Optional keys (depending on options):
            feature_map, stext_page (absent on this path).
    """
    if options is None:
        options = {}

    input_type         = options.get('input_type', ('dtext',))
    feature_set_name   = options.get('feature_set_name', 'rf+imf')
    max_vec_line_num   = options.get('max_vec_line_num', 200)
    feature_extractor  = options.get('feature_extractor', None)
    bbox_num_threshold = options.get('bbox_num_threshold', 0)
    txt_func           = options.get('txt_func', None)

    # Validate: reject PDF-only input types early with a clear message.
    unsupported = [t for t in input_type if t not in _IMAGE_ONLY_VALID_INPUT_TYPES]
    if unsupported:
        raise ValueError(
            f"create_input_data_from_image() does not support PDF-dependent "
            f"input_type(s): {unsupported}. Use create_input_data_from_page() "
            f"for PDF pages, or replace with detector-based equivalents: "
            f"'text' -> 'dtext', 'image' -> 'dimage'."
        )

    # Step 1: build data_dict skeleton from image (no PDF parsing)
    data_dict = _build_image_only_data_dict(page_img)

    # Step 1.5a: detector-based text bboxes
    if 'dtext' in input_type:
        if txt_func is None and feature_extractor is None:
            raise ValueError(
                "'dtext' input_type requires either a feature_extractor "
                "(default detector path) or a txt_func (custom OCR path)"
            )
        apply_dtext_boxes(data_dict, feature_extractor, txt_func=txt_func)

    # Step 1.5b: detector-based picture bboxes
    if 'dimage' in input_type:
        if feature_extractor is None:
            raise ValueError("'dimage' input_type requires a feature_extractor")
        apply_dimage_boxes(data_dict, feature_extractor)

    # Step 1.5c: detector-based table lines
    # page=None -> 'dtable_vec_line' is a no-op; 'dtable_img_line' works normally.
    apply_detector_table_lines(
        page=None,
        data_dict=data_dict,
        input_type=input_type,
        max_vec_line_num=max_vec_line_num,
        feature_extractor=feature_extractor,
    )

    # Step 1.5d: LSD-based table lines ('dtable_lsd_line').
    # page=None is fine -- LSD is image-based, same as dtable_img_line.
    # cv2 ImportError propagates to the caller if not installed.
    apply_detector_lsd_lines(
        page=None,
        data_dict=data_dict,
        input_type=input_type,
        max_vec_line_num=max_vec_line_num,
        feature_extractor=feature_extractor,
    )

    # Step 1.5e: detector-based table region ('dtable_rect'). Works
    # normally with page=None -- unlike 'dtable_vec_line' it does not
    # depend on PDF vector paths, only on already-extracted bboxes (Step
    # 1.5a/b results) and, if present, cached Sobel lines from
    # 'dtable_img_line' above.
    apply_detector_table_rect(
        page=None,
        data_dict=data_dict,
        input_type=input_type,
        feature_extractor=feature_extractor,
    )

    # Early return check (same pattern as create_input_data_from_page)
    if len(data_dict['bboxes']) > bbox_num_threshold > 0:
        _ensure_custom_features(data_dict)
        return data_dict

    # Step 2: apply feature extractors (fully reused from PDF path)
    if feature_set_name:
        data_dict = apply_feature_extractors(
            data_dict=data_dict,
            feature_set_name=feature_set_name,
            feature_extractor=feature_extractor,
            input_type=input_type,
            page=None,       # no PDF page; YF/JF features will skip page-dependent channels
            page_dict=None,  # no stext_page on image-only path
        )
    else:
        _ensure_custom_features(data_dict)

    return data_dict


def create_input_data_by_pymupdf(pdf_path=None, document=None, page_no=0, options=None):
    """
    Create input data from a PDF file or document object.

    Args:
        pdf_path: Path to PDF file (if document is None)
        document: PyMuPDF document object (if provided, pdf_path is ignored for opening)
        page_no: Page number to process
        options: Configuration dict. See module docstring for available keys.

    Returns:
        data_dict with guaranteed keys:
            bboxes, text, custom_features, page_width, page_height, image, file_path
    """
    if document is None:
        if not os.path.exists(pdf_path):
            raise Exception(f'{pdf_path} does not exist!')
        doc = pymupdf.open(pdf_path)
        page = doc[page_no]
        should_close = True
    else:
        doc = document
        page = doc[page_no]
        should_close = False

    data_dict = create_input_data_from_page(page=page, options=options)
    data_dict['file_path'] = pdf_path

    if should_close:
        doc.close()

    return data_dict
