"""
PDF parsing utilities - Base module
Maintained by: PDF Engineers
Purpose: Stable PDF element extraction (text, images, vectors, checkboxes)
"""

import copy
import numpy as np
import pymupdf
from collections import defaultdict


def get_vector_lines(page, omit_invisible=True):
    """Return horizontal and vertical vector lines on a page.

    Optionally, omit vectors that have the color of the background.
    The background color is determined as the most frequent color on a pixel
    level as determined from a rasterized image (pixmap) of the page.
    """
    # determine background color:
    # disable anti-aliasing for more accurate color count
    # We assume the most frequent color to be the background.
    pymupdf.TOOLS.set_aa_level(0)
    pix = page.get_pixmap()
    portion, color = pix.color_topusage()
    # sRGB integer represents an RGB color tripel.
    sRGB = color[0] << 16 | color[1] << 8 | color[2]
    # convert to a tripel of three floats in 0..1 range.
    bg_color = pymupdf.sRGB_to_pdf(sRGB)

    # all potentially significant paths
    paths = [
        p for p in page.get_drawings() if p["rect"].height > 5 or p["rect"].width > 5
    ]

    h_lines = []
    v_lines = []

    for p in paths:
        if (
                1
                and omit_invisible
                and (
                p["fill"] == bg_color
                and p["color"] in (None, bg_color)
                or p["color"] == bg_color
                and p["fill"] in (None, bg_color)
        )
        ):
            continue
        # total rectangle already is "line-like" ...
        if p["rect"].height < 3 and p["rect"].width > 30:
            h_lines.append(p["rect"])
            continue
        elif p["rect"].width < 3 and p["rect"].height > 30:
            v_lines.append(p["rect"])
            continue

        # cover other cases by looking at single draw commands (items)
        # ".normalize()" is required because invalid rectangles are not drawn
        for item in p["items"]:
            # Handle line items. Must be axis-parallel
            if item[0] == "l":  # a line
                p1, p2 = item[1:]
                if abs(p1.y - p2.y) <= 1 and abs(p2.x - p1.x) > 30:  # horizontal line
                    h_lines.append(pymupdf.Rect(p1, p2).normalize())
                elif abs(p1.x - p2.x) <= 1 and abs(p2.y - p1.y) > 30:  # vertical line
                    v_lines.append(pymupdf.Rect(p1, p2).normalize())
                continue
            if item[0] != "re":  # skip non-rectangles
                continue
            r = item[1]  # the rectangle of this item
            h_lines.append(pymupdf.Rect(r.tl, r.tr).normalize())  # top horizontal
            h_lines.append(pymupdf.Rect(r.bl, r.br).normalize())  # bottom horizontal
            v_lines.append(pymupdf.Rect(r.tl, r.bl).normalize())  # left vertical
            v_lines.append(pymupdf.Rect(r.tr, r.br).normalize())  # right vertical

    return h_lines, v_lines

# ---------------------------------------------------------------------------------------------
"""
Optimized get_picture_clusters with identical semantics to the original.

Optimization:
- Faster containment filtering before clustering using a point-query grid over
  already-accepted larger rectangles.

We intentionally keep PyMuPDF's `page.cluster_drawings()` for the final
clustering step because it is already cheap on the profiled worst-case pages and
preserves exact output semantics.
"""

# Default tolerances matching PyMuPDF Page.cluster_drawings()
X_TOLERANCE = 3.0
Y_TOLERANCE = 3.0

# When union touches more than this many cells (incl. adjacent), use full scan (avoids grid overhead on worst-case pages)
MAX_CELLS_FOR_GRID = 64

# Grid resolution for the pre-clustering containment filter.
CONTAINMENT_GRID_CELLS = 32

def _are_neighbors(r1, r2, delta_x, delta_y):
    """True if r1 and r2 are within tolerance (expanded-rect overlap)."""
    rr1_x0, rr1_x1 = (r1.x0, r1.x1) if r1.x1 > r1.x0 else (r1.x1, r1.x0)
    rr1_y0, rr1_y1 = (r1.y0, r1.y1) if r1.y1 > r1.y0 else (r1.y1, r1.y0)
    rr2_x0, rr2_x1 = (r2.x0, r2.x1) if r2.x1 > r2.x0 else (r2.x1, r2.x0)
    rr2_y0, rr2_y1 = (r2.y0, r2.y1) if r2.y1 > r2.y0 else (r2.y1, r2.y0)
    if (
        rr1_x1 < rr2_x0 - delta_x
        or rr1_x0 > rr2_x1 + delta_x
        or rr1_y1 < rr2_y0 - delta_y
        or rr1_y0 > rr2_y1 + delta_y
    ):
        return False
    return True


def _expanded_bounds(rect, delta_x, delta_y):
    x0 = rect.x0 - delta_x
    x1 = rect.x1 + delta_x
    y0 = rect.y0 - delta_y
    y1 = rect.y1 + delta_y
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)


def _normalized_bounds(rect):
    """Return normalized rect bounds as (x0, y0, x1, y1)."""
    x0, x1 = (rect.x0, rect.x1) if rect.x1 >= rect.x0 else (rect.x1, rect.x0)
    y0, y1 = (rect.y0, rect.y1) if rect.y1 >= rect.y0 else (rect.y1, rect.y0)
    return (x0, y0, x1, y1)


def _grid_cell_for_point(px, py, page_rect, cell_w, cell_h):
    """Map a point to a containment-grid cell, clamped to page bounds."""
    max_x = CONTAINMENT_GRID_CELLS - 1
    max_y = CONTAINMENT_GRID_CELLS - 1
    cx = int((px - page_rect.x0) // cell_w)
    cy = int((py - page_rect.y0) // cell_h)
    if cx < 0:
        cx = 0
    elif cx > max_x:
        cx = max_x
    if cy < 0:
        cy = 0
    elif cy > max_y:
        cy = max_y
    return (cx, cy)


def _candidate_cells_for_point(px, py, page_rect, cell_w, cell_h):
    """
    Return the point cell plus its immediate neighbors.

    This preserves correctness for degenerate rects (zero width / height) whose
    center may lie exactly on a grid boundary while the accepted container was
    indexed into the adjacent cell.
    """
    cx, cy = _grid_cell_for_point(px, py, page_rect, cell_w, cell_h)
    max_x = CONTAINMENT_GRID_CELLS - 1
    max_y = CONTAINMENT_GRID_CELLS - 1
    cells = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx <= max_x and 0 <= ny <= max_y:
                cells.append((nx, ny))
    return cells


def _grid_keys_for_rect(rect, page_rect, cell_w, cell_h):
    """Yield containment-grid cells touched by rect."""
    x0, y0, x1, y1 = _normalized_bounds(rect)
    max_x = CONTAINMENT_GRID_CELLS - 1
    max_y = CONTAINMENT_GRID_CELLS - 1
    cx0 = int((x0 - page_rect.x0) // cell_w)
    cx1 = int((x1 - page_rect.x0) // cell_w)
    cy0 = int((y0 - page_rect.y0) // cell_h)
    cy1 = int((y1 - page_rect.y0) // cell_h)
    if cx0 < 0:
        cx0 = 0
    elif cx0 > max_x:
        cx0 = max_x
    if cx1 < 0:
        cx1 = 0
    elif cx1 > max_x:
        cx1 = max_x
    if cy0 < 0:
        cy0 = 0
    elif cy0 > max_y:
        cy0 = max_y
    if cy1 < 0:
        cy1 = 0
    if cy1 > max_y:
        cy1 = max_y
    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            yield (cx, cy)


def _filter_contained_paths_fast(all_pictures, page_rect, limit=1000):
    """
    Faster equivalent of:
        if not any(p["rect"] in f["rect"] for f in filtered_paths):
            filtered_paths.append(p)

    Because candidates are processed in descending area, any containing rectangle
    is already in filtered_paths. A rectangle that contains the candidate must
    also contain the candidate's center point, so we query only the accepted
    rectangles whose coverage includes that point's cell.
    """
    if not all_pictures:
        return []

    cell_w = max(page_rect.width / CONTAINMENT_GRID_CELLS, 1.0)
    cell_h = max(page_rect.height / CONTAINMENT_GRID_CELLS, 1.0)
    grid = {}  # (cell_x, cell_y) -> [accepted rect indices]
    filtered_paths = []

    for p in all_pictures[:limit]:
        rect = p["rect"]
        x0, y0, x1, y1 = _normalized_bounds(rect)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        contained = False
        for cell in _candidate_cells_for_point(cx, cy, page_rect, cell_w, cell_h):
            for accepted_idx in grid.get(cell, ()):
                # Use PyMuPDF's exact Rect containment semantics here, especially
                # for degenerate / invalid rectangles near page boundaries.
                if rect in filtered_paths[accepted_idx]["rect"]:
                    contained = True
                    break
            if contained:
                break
        if contained:
            continue

        accepted_idx = len(filtered_paths)
        filtered_paths.append(p)
        for key in _grid_keys_for_rect(rect, page_rect, cell_w, cell_h):
            if key not in grid:
                grid[key] = []
            grid[key].append(accepted_idx)

    return filtered_paths


def get_picture_clusters(page, x_tolerance=X_TOLERANCE, y_tolerance=Y_TOLERANCE):
    """Identify and group connected vectors on a page (optimized implementation)."""
    page_rect = page.rect

    paths = [page_rect & p["rect"] for p in page.get_drawings()]
    images = [page_rect & img["bbox"] for img in page.get_image_info()]

    all_pictures = sorted(
        [{"rect": r} for r in paths + images if r.width >= 3 or r.height >= 3],
        key=lambda x: x["rect"].width * x["rect"].height,
        reverse=True,
    )

    filtered_paths = _filter_contained_paths_fast(all_pictures, page_rect, limit=1000)

    if not filtered_paths:
        return []

    return page.cluster_drawings(
        drawings=filtered_paths,
        x_tolerance=x_tolerance,
        y_tolerance=y_tolerance,
    )
# ---------------------------------------------------------------------------------------------


def merge_lines(lines, orientation='h', tolerance=3):
    """
    Merge fragmented lines into longer continuous lines.

    lines: list of line objects with x0, y0, x1, y1 attributes
    orientation: 'h' for horizontal, 'v' for vertical
    tolerance: allowable gap or offset to consider lines as mergeable
    """
    # Sort lines by position: horizontal by y, vertical by x
    if orientation == 'h':
        lines.sort(key=lambda r: (r.y0, r.x0))
    else:
        lines.sort(key=lambda r: (r.x0, r.y0))

    merged = []
    for line in lines:
        x0, y0, x1, y1 = line.x0, line.y0, line.x1, line.y1
        matched = False

        for m in merged:
            if orientation == 'h':
                # Check if y is close and x ranges overlap or touch
                if abs(m.y0 - y0) <= tolerance and not (x1 < m.x0 - tolerance or x0 > m.x1 + tolerance):
                    m.x0 = min(m.x0, x0)
                    m.x1 = max(m.x1, x1)
                    matched = True
                    break
            else:
                # Check if x is close and y ranges overlap or touch
                if abs(m.x0 - x0) <= tolerance and not (y1 < m.y0 - tolerance or y0 > m.y1 + tolerance):
                    m.y0 = min(m.y0, y0)
                    m.y1 = max(m.y1, y1)
                    matched = True
                    break

        if not matched:
            # No match found, add as new merged line
            merged.append(copy.deepcopy(line))

    return merged


def find_closed_border_rects(h_lines, v_lines, tol=0.75):
    """
    Find fully-closed border rectangles (e.g. full-border tables) using
    exact coordinate matching -- NOT proximity/tolerance-based clustering
    like page.cluster_drawings(). See design discussion: clustering's
    tolerance is itself an unvalidated heuristic, and real tables are built
    in inconsistent ways (lines-only, filled cells, text-wrapping vectors),
    so no single tolerance is guaranteed to generalize. Exact-match closure
    instead only asserts something narrower and more defensible: "these
    lines were clearly drawn to form a rectangle" -- true regardless of how
    close together two SEPARATE closed rectangles happen to sit.

    Core idea: a full-border rectangle's outer boundary is formed by two
    horizontal lines sharing the exact same x-span, and two vertical lines
    sharing the exact same y-span (the horizontal pair's y-values),
    positioned at exactly those two x-coordinates.

    Args:
        h_lines: list of objects with x0, y0, x1, y1 (horizontal lines,
            y0 == y1 up to float noise). As returned by
            get_vector_lines()/merge_lines().
        v_lines: list of objects with x0, y0, x1, y1 (vertical lines,
            x0 == x1 up to float noise).
        tol: float. Coordinate-snapping tolerance to absorb float rounding
            noise ONLY (e.g. two lines both meant to be drawn at x=100.0
            landing at 100.0 and 100.03) -- NOT a proximity threshold for
            deciding whether two nearby-but-distinct coordinates should be
            treated as "close enough" structurally. Keep well below the
            smallest real gap you'd expect between two genuinely different
            table/cell edges.

    Returns:
        list of (x0, y_top, x1, y_bottom) tuples: one closed rectangle per
        matching horizontal-line x-span/vertical-line y-span pair found.
        Checks every pair of horizontal lines sharing an x-span (not just
        the global min/max y for that span), so two separate rectangles
        that happen to share the same width don't get silently merged into
        one spanning both.
    """
    def snap(v):
        return round(v / tol) * tol

    h_by_span = defaultdict(list)
    for r in h_lines:
        x0, x1 = sorted((r.x0, r.x1))
        y = (r.y0 + r.y1) / 2.0
        h_by_span[(snap(x0), snap(x1))].append(y)

    v_by_span = defaultdict(set)
    for r in v_lines:
        y0, y1 = sorted((r.y0, r.y1))
        x = (r.x0 + r.x1) / 2.0
        v_by_span[(snap(y0), snap(y1))].add(snap(x))

    closed_rects = []
    for (x0k, x1k), ys in h_by_span.items():
        if x0k == x1k or len(ys) < 2:
            continue

        ys_sorted = sorted(set(ys))
        for a in range(len(ys_sorted)):
            for b in range(a + 1, len(ys_sorted)):
                y_top, y_bottom = ys_sorted[a], ys_sorted[b]
                v_xs = v_by_span.get((snap(y_top), snap(y_bottom)))
                if v_xs and x0k in v_xs and x1k in v_xs:
                    closed_rects.append((x0k, y_top, x1k, y_bottom))

    return closed_rects


def get_rect_member_id(bbox, closed_rects):
    """
    Which closed rect (from find_closed_border_rects) contains this bbox's
    center, if any. Returns the index into closed_rects, or -1 if the bbox
    isn't inside any of them. Used to build the 'same_rect_member' edge
    feature (see pymupdf_util_edge.get_edge_transform_yf): two nodes with
    the same (non -1) rect member id sit inside the same closed
    rectangle -- e.g. two cells of the same table -- while two nodes in
    different (or no) closed rectangles don't, even if they happen to be
    geometrically close (e.g. cells from two separate, adjacent tables).
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    for idx, (x0, y0, x1, y1) in enumerate(closed_rects):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            return idx
    return -1


def merge_boxes(boxes, iou_threshold=0.5):
    """Merge overlapping bounding boxes based on IoU threshold."""
    from .common_util import compute_iou

    merged = []
    while boxes:
        base = boxes.pop(0)
        has_merged = False
        for i, other in enumerate(boxes):
            iou = compute_iou(base, other)
            if iou > iou_threshold:
                new_box = [
                    min(base[0], other[0]),
                    min(base[1], other[1]),
                    max(base[2], other[2]),
                    max(base[3], other[3])
                ]
                boxes[i] = new_box
                has_merged = True
                break
        if not has_merged:
            merged.append(base)
    return merged


def text_extract(data_dict, box_type, page_width, page_height, blocks):
    """Extract text blocks from PDF page."""
    for block in blocks:
        if block["type"] == 2:
            text_extract(data_dict, box_type, page_width, page_height, block["blocks"])
            continue
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            x1 = line['bbox'][0]
            y1 = line['bbox'][1]
            x2 = line['bbox'][2]
            y2 = line['bbox'][3]

            txt = []
            for span in line['spans']:
                txt.append(span['text'])

            txt = ' '.join(txt).strip()

            # Filter Empty text
            if txt != '' and 0 <= x2 and x1 <= x2 and x1 <= page_width and 0 <= y2 and y1 <= y2 and y1 <= page_height:
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append(txt)
                    box_type.append(BOX_TEXT)


def create_stext_page(page, flags):
    return page.get_textpage(flags=flags)


def extract_base_elements(page, input_type=('text',), feature_set_name='rf',
                          max_image_num=500, max_vec_line_num=200, page_img=None):
    """
    Extract basic PDF elements without any feature extraction.

    Args:
        page: PyMuPDF page object
        input_type: Tuple of element types to extract ('text', 'image',
            'picture_clusters', 'vec_line', 'img_line', 'table_vec_line',
            'table_vec_line_full', 'table_vec_line_partial',
            'table_img_line', 'table_img_line_full',
            'table_img_line_partial', 'table_rect_full',
            'img_table_rect_full'). 'vec_line'
            adds every detected PDF vector line as an element;
            'img_line' is the page-wide image-based counterpart: it
            runs Sobel edge detection on the rasterized page image and
            adds every detected horizontal/vertical line as a bbox --
            useful for PDFs where lines are encoded as filled rects,
            background colors, or embedded images rather than explicit
            vector drawings, which 'vec_line' misses entirely. Raw Sobel
            output is cached (data_dict['image_lines_raw']) and merged
            lines cached (data_dict['image_lines']) for reuse by
            'table_img_line*', 'img_table_rect_full', and
            'dtable_img_line' in the same pipeline call.
            'table_vec_line' instead adds only the vec_line lines that
            belong to a validated table structure (closed border, or
            border open on one side, always with at least one internal
            row/column divider) -- both shapes together.
            'table_vec_line_full' restricts this to closed-
            border tables only, 'table_vec_line_partial' to open-sided
            tables only (see find_table_grids' table_type parameter in
            pymupdf_util_table.py). 'table_rect_full' also restricts to
            closed-border tables only, but adds ONE bbox per table (the
            table's outer bbox) instead of the individual border/divider
            lines -- see apply_table_rect_full() in
            pymupdf_util_table.py; intended to test whether GNN
            performance holds up with far fewer nodes per table than
            'table_vec_line_full'. 'img_table_rect_full' is the same
            idea sourced from Sobel image-based lines instead of PDF
            vector paths -- the image-based counterpart of
            'table_rect_full', exactly as 'table_img_line' is to
            'table_vec_line' (see apply_img_table_rect_full()).
            'table_img_line'/'table_img_line_full'/'table_img_line_partial'
            mirror the table_vec_line family but built on top of
            img_line's raw output, with looser tolerances to account for
            image-detection noise. The implementation of the
            table_vec_line*/table_img_line* family (find_table_grids and
            the extraction blocks themselves) lives in
            pymupdf_util_table.py, not here -- this function only
            dispatches to it, to keep this (model-agnostic,
            PDF-engineer-owned) module free of table-grid-heuristic
            details. Their detector-based counterparts,
            'dtable_vec_line'/'dtable_img_line', are NOT handled here at
            all -- since they require feature_extractor (a CV model),
            they are dispatched separately as a "Step 1.5" by
            pymupdf_util.py, after this function returns (see
            pymupdf_util_table.apply_detector_table_lines()). All of
            these are independent and can be combined, though for most
            GNN layout use cases picking one line source (vec vs. img)
            and one granularity (raw vs. table-filtered, full vs.
            partial, heuristic vs. detector) is preferable to avoid
            duplicate/near-duplicate line bboxes for the same table.
        max_image_num: Maximum number of images to extract
        max_vec_line_num: Maximum number of vector lines to extract
        page_img: Optional pre-rasterized (H, W, C) uint8 page image. When
            provided, page.get_pixmap() is not called again here -- pass this
            whenever the caller already rasterized the page (e.g. BoxRFDGNN
            already needed it for feature_extractor.predict()), to avoid
            redundant rasterization of the same page.

    Returns:
        data_dict: Dictionary with 'bboxes', 'text', 'box_type', 'page_width',
            'page_height', 'image', 'stext_page', 'has_raster_image',
            'has_embedded_text'
    """
    data_dict = {
        'bboxes': [],
        'text': [],
        'page_width': 0,
        'page_height': 0,
    }
    box_type = []

    page_width, page_height = page.rect[2], page.rect[3]
    data_dict['page_width'] = page_width
    data_dict['page_height'] = page_height

    # Raw page-level signals used by ImageFeatureExtractorV1/V2.is_image_page()
    # and is_ocr_needed(). These must be computed unconditionally, independent
    # of input_type, since they answer questions about the raw PDF page itself
    # ("does it contain a raster image", "does it have embedded selectable
    # text") rather than about which element types the caller asked to
    # extract into bboxes/text.
    data_dict['has_raster_image'] = bool(page.get_image_info())
    data_dict['has_embedded_text'] = bool(page.get_text("text").strip())

    # Extract page image (reuse caller-provided rasterization if available)
    if page_img is None:
        pix = page.get_pixmap()
        bytes_data = np.frombuffer(pix.samples, dtype=np.uint8)
        page_img = bytes_data.reshape(pix.height, pix.width, pix.n)
    data_dict['image'] = page_img

    # Create structured text page
    """Create structured text page with optimal flags."""
    stext_flags = (
            0
            | pymupdf.TEXT_PRESERVE_IMAGES
            | pymupdf.TEXT_PRESERVE_WHITESPACE
            | pymupdf.TEXT_PRESERVE_LIGATURES
            | pymupdf.TEXT_ACCURATE_BBOXES
            | pymupdf.TEXT_COLLECT_VECTORS
            | pymupdf.TEXT_COLLECT_STYLES
            | pymupdf.TEXT_SEGMENT
            | pymupdf.TEXT_PARAGRAPH_BREAK
            | pymupdf.TEXT_COLLECT_STRUCTURE
            | pymupdf.TEXT_TABLE_HUNT
    )
    if pymupdf.mupdf_version_tuple >= (1, 27, 1):
        stext_flags |= pymupdf.TEXT_LAZY_VECTORS
        stext_flags |= pymupdf.TEXT_CLIP
    else:
        # For versions of PyMuPDF that didnt expose CLIP, but are built
        # on versions of MuPDF that support it.
        stext_flags |= 64  # CLIP
    if pymupdf.mupdf_version_tuple >= (1, 27, 2):
        stext_flags |= pymupdf.TEXT_FUZZY_VECTORS

    # For image feature set only, omit some flags for speed up extraction of bboxes
    if feature_set_name == 'imf':
        stext_flags = (
                0
                | pymupdf.TEXT_PRESERVE_WHITESPACE
                | pymupdf.TEXT_PRESERVE_LIGATURES
                | pymupdf.TEXT_INHIBIT_SPACES
                | pymupdf.TEXT_ACCURATE_BBOXES
                #| pymupdf.TEXT_COLLECT_VECTORS
                #| pymupdf.TEXT_COLLECT_STYLES
                #| pymupdf.TEXT_SEGMENT
                | pymupdf.TEXT_PARAGRAPH_BREAK
                | pymupdf.TEXT_COLLECT_STRUCTURE
                #| pymupdf.TEXT_TABLE_HUNT
        )


    stext_page = create_stext_page(page, stext_flags)
    data_dict['stext_page'] = stext_page

    if False:
        import pprint
        pprint.pp(input_type)
        pprint.pp(stext_page.extractDICT())

    # Extract images
    if 'image' in input_type:
        img_bboxes = [itm["bbox"] for itm in page.get_image_info()]
        img_bboxes = merge_boxes(img_bboxes, iou_threshold=0.5)
        img_bboxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        img_bboxes = img_bboxes[:max_image_num]

        for rect in img_bboxes:
            x1 = max(0, rect[0])
            y1 = max(0, rect[1])
            x2 = max(0, rect[2])
            y2 = max(0, rect[3])
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append('')
                    box_type.append(BOX_IMAGE)

    # Extract picture clusters
    if 'picture_clusters' in input_type:
        pic_bboxes = get_picture_clusters(page)
        pic_bboxes = sorted(
            pic_bboxes,
            key=lambda r: r.width * r.height,
            reverse=True
        )[:max_image_num]

        for rect in pic_bboxes:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                if y1 == y2:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append('')
                    box_type.append(BOX_IMAGE)

    # Extract vector lines
    if 'vec_line' in input_type:
        h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
        h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
        v_lines = merge_lines(v_lines, orientation='v', tolerance=3)

        # Cache the merged (pre-truncation) result so other consumers of the
        # same page (e.g. pymupdf_util_yf.create_feature_map's
        # vector_margin_* channels) can reuse it instead of re-parsing
        # page.get_drawings() and re-rasterizing page.get_pixmap() (the
        # latter purely to sample a background color) a second time.
        data_dict['vector_lines'] = (h_lines, v_lines)

        # Also cache closed (full-border) rectangles derived from these same
        # lines, for the 'same_rect_member' edge feature (see
        # pymupdf_util_edge.get_edge_transform_yf) -- cheap to compute here
        # since h_lines/v_lines are already in hand, and avoids recomputing
        # per edge_type='YF' call.
        data_dict['closed_rects'] = find_closed_border_rects(h_lines, v_lines)

        h_lines = sorted(h_lines, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]
        for rect in h_lines:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height:
                if y1 == y2:
                    y2 = y1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append('')
                    box_type.append(BOX_HLINE)

        v_lines = sorted(v_lines, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]
        for rect in v_lines:
            x1 = rect.x0
            y1 = rect.y0
            x2 = rect.x1
            y2 = rect.y1
            if 0 <= x1 <= x2 <= page_width and 0 <= y1 < y2 <= page_height:
                if x1 == x2:
                    x2 = x1 + 1
                bbox = [x1, y1, x2, y2]
                if bbox not in data_dict['bboxes']:
                    data_dict['bboxes'].append(bbox)
                    data_dict['text'].append('')
                    box_type.append(BOX_VLINE)

    # Extract image-based lines ('img_line'): Sobel edge detection over the
    # rasterized page image, page-wide -- the image-based counterpart of
    # 'vec_line'. Raw Sobel output is cached in data_dict['image_lines_raw']
    # and merged lines in data_dict['image_lines'] so that downstream
    # consumers ('table_img_line*', 'img_table_rect_full', 'dtable_img_line')
    # can reuse them without re-running Sobel on the same page image.
    if 'img_line' in input_type:
        from .pymupdf_util_table import apply_img_line
        apply_img_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num)

    # Extract table lines ('table_vec_line'/'table_vec_line_full'/
    # 'table_vec_line_partial', 'table_img_line'/'table_img_line_full'/
    # 'table_img_line_partial') and their detector-based counterparts
    # ('dtable_vec_line'/'dtable_img_line' are dispatched separately,
    # from pymupdf_util.py, since they need feature_extractor -- see
    # pymupdf_util_table.py's module docstring). Implementation lives in
    # pymupdf_util_table.py, not here, so this (model-agnostic) module
    # stays free of table-grid-heuristic details. Imported lazily to
    # avoid a circular import at module load time: pymupdf_util_table
    # imports get_vector_lines/merge_lines/BOX_HLINE/BOX_VLINE from this
    # module at its own module top.
    #
    # The plain 'table_vec_line'/'table_img_line' values map to
    # table_type='all' (fully closed + open-sided tables), matching the
    # original (pre-table_type) behavior. The '_full'/'_partial' suffixed
    # values request only one table shape. More than one of these can be
    # requested at once (e.g. both '_full' and '_partial') -- each
    # present value triggers its own call, and results accumulate in
    # data_dict rather than the last call overwriting the previous one
    # (see apply_table_vec_line/apply_table_img_line). The key set here
    # is duplicated from pymupdf_util_table.VEC_LINE_TABLE_TYPES /
    # IMG_LINE_TABLE_TYPES (just the input_type strings, not the mapping
    # itself) so this check does not need to import that module -- the
    # module is only imported lazily below, once we know it is actually
    # needed, to avoid the circular import noted above.
    #
    # 'table_rect_full' is a separate, simpler sibling: it adds ONE bbox
    # per fully closed table (find_table_grids' outer bbox) instead of
    # every individual border/divider line, to test whether a single
    # coarse table-region node keeps GNN performance while using far
    # fewer nodes than table_vec_line_full for the same table.
    _table_line_input_types = (
        'table_vec_line', 'table_vec_line_full', 'table_vec_line_partial',
        'table_img_line', 'table_img_line_full', 'table_img_line_partial',
    )
    if any(k in input_type for k in _table_line_input_types):
        from .pymupdf_util_table import (
            apply_table_vec_line, apply_table_img_line,
            VEC_LINE_TABLE_TYPES, IMG_LINE_TABLE_TYPES,
        )

        for key, table_type in VEC_LINE_TABLE_TYPES.items():
            if key in input_type:
                apply_table_vec_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num, table_type=table_type)

        for key, table_type in IMG_LINE_TABLE_TYPES.items():
            if key in input_type:
                apply_table_img_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num, table_type=table_type)

    if 'table_rect_full' in input_type:
        from .pymupdf_util_table import apply_table_rect_full

        apply_table_rect_full(page, data_dict, box_type, page_width, page_height)

    if 'img_table_rect_full' in input_type:
        from .pymupdf_util_table import apply_img_table_rect_full

        apply_img_table_rect_full(page, data_dict, box_type, page_width, page_height)

    # Extract text
    if 'text' in input_type or 'text_pm' in input_type:
        if 'text_pm' in input_type:
            blocks = page.get_text("dict")["blocks"]
        else:
            blocks = page.get_text("dict", textpage=stext_page)["blocks"]
        text_extract(data_dict, box_type, page_width, page_height, blocks)

    # Extract checkboxes
    checkboxes = [(widget.rect, widget.field_value) for widget in page.widgets() if
                  widget.field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX]
    for bbox, value in checkboxes:
        x1 = bbox.x0
        y1 = bbox.y0
        x2 = bbox.x1
        y2 = bbox.y1
        if 0 <= x1 < x2 <= page_width and 0 <= y1 < y2 <= page_height:
            bbox = [x1, y1, x2, y2]
            if bbox not in data_dict['bboxes']:
                data_dict['bboxes'].append(bbox)
                data_dict['text'].append('')
                box_type.append(BOX_CHECKBOX)

    data_dict['box_type'] = box_type
    return data_dict


# ── Box type constants (single source of truth) ──────────────────────
BOX_TEXT = 'text'
BOX_IMAGE = 'image'
BOX_HLINE = 'h-line-vector'
BOX_VLINE = 'v-line-vector'
BOX_CHECKBOX = 'check-box'
BOX_TABLE = 'table'


def make_custom_feature(box_type_str, text=''):
    """
    Create a custom_feature dict from a box_type string.

    This is the SINGLE SOURCE OF TRUTH for box_type → flag mapping.
    All modules (pymupdf_util.py, pymupdf_util_ext.py) must use this
    instead of duplicating the logic.
    """
    is_text = 1 if box_type_str == BOX_TEXT else 0
    is_image = 1 if box_type_str == BOX_IMAGE else 0
    is_hline_vector = 1 if box_type_str == BOX_HLINE else 0
    is_vline_vector = 1 if box_type_str == BOX_VLINE else 0
    is_table = 1 if box_type_str == BOX_TABLE else 0

    if len(text) > 0:
        num_count = sum(1 for c in text if c.isdigit())
        num_ratio = num_count / len(text)
    else:
        num_ratio = 0.0

    return {
        'box_type': box_type_str,
        'num_ratio': num_ratio,
        'is_text': is_text,
        'is_image': is_image,
        'is_hline_vector': is_hline_vector,
        'is_vline_vector': is_vline_vector,
        'is_table': is_table,
        'is_line': is_text,                             # legacy alias
        'is_vector': is_hline_vector + is_vline_vector, # legacy alias
    }
