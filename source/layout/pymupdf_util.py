"""
PDF processing utilities - Main integration module
This module combines base PDF parsing and experimental feature extraction.

Interface:
    create_input_data_from_page(page, options=None)
    create_input_data_by_pymupdf(pdf_path=None, document=None, page_no=0, options=None)

Options dict keys (with defaults):
    input_type          : tuple   = ('text',)       — Element types: 'text', 'image', 'picture_clusters', 'vec_line', 'seg-image'
    feature_set_name    : str     = 'rf+imf'        — Features to extract: 'rf', 'imf', 'yf', 'jf' (combined with '+')
    max_image_num       : int     = 500             — Maximum number of images to extract
    max_vec_line_num    : int     = 200             — Maximum number of vector lines to extract
    feature_extractor   : object  = None            — ONNX model for image-based feature extraction
    bbox_num_threshold  : int     = 0               — Return early if bbox count exceeds this (0 = disabled)
    remove_watermarks   : bool    = None            — Leave watermark text out of the input (None = module default)
"""

import os
import pymupdf

from .pymupdf_util_base import extract_base_elements, make_custom_feature
from .pymupdf_util_ext import apply_feature_extractors


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
    remove_watermarks = options.get('remove_watermarks', None)

    # Step 1: Extract base PDF elements
    data_dict = extract_base_elements(
        page=page,
        input_type=input_type,
        max_image_num=max_image_num,
        max_vec_line_num=max_vec_line_num,
        remove_watermarks=remove_watermarks,
    )

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
