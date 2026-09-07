"""
Table-related line extraction.
Maintained by: PDF Engineers (heuristic table_vec_line/table_img_line) +
               CV team (detector-based dtable_vec_line/dtable_img_line)

All logic for the table-oriented input_type values lives here, kept out
of pymupdf_util_base.py so that file stays a small, stable,
model-agnostic core (general PDF element extraction only). This module
is allowed to depend on the image-based layout detector (feature_extractor)
for its dtable_* half; the heuristic table_vec_line/table_img_line half
does not.

input_type values handled here (see VEC_LINE_TABLE_TYPES /
IMG_LINE_TABLE_TYPES for the exact mapping):
    'table_vec_line'         All table shapes (fully closed + open-sided).
    'table_vec_line_full'    Fully closed tables only.
    'table_vec_line_partial' Open-sided tables only.
    'table_img_line'         All table shapes (fully closed + open-sided).
    'table_img_line_full'    Fully closed tables only.
    'table_img_line_partial' Open-sided tables only.
    'table_rect_full'        ONE bbox per fully closed table (the table's
        outer bbox only -- no individual border/divider lines added).
        See apply_table_rect_full().
    'img_table_rect_full'    Same as 'table_rect_full', but sourced from
        Sobel image-based lines instead of PDF vector paths -- the
        image-based counterpart, exactly as 'table_img_line' is to
        'table_vec_line'. See apply_img_table_rect_full().
    'dtable_vec_line' / 'dtable_img_line' -- detector-based, unaffected
        by the full/partial distinction (see apply_detector_table_lines).
    'dtable_lsd_line' -- detector-based, same region source as dtable_img_line
        but uses LSD (Line Segment Detector) instead of Sobel for line
        extraction. No full/partial variant: the detector supplies the table
        region, so find_table_grids() structural validation is not applied and
        the full/partial distinction has no meaning here. Requires cv2
        (opencv-python or opencv-python-headless); raises ImportError at
        runtime if cv2 is not installed. See apply_detector_lsd_lines().
    'dtable_rect' -- detector-based table region, tightened to the union
        of whatever content is actually found inside it; regions with
        nothing inside are dropped as false positives. See
        apply_detector_table_rect().
    'stable_img_line' / 'stable_img_line_full' / 'stable_img_line_partial'
        -- hybrid: heuristic table regions (same as table_img_line family)
        filtered by the model's segmentation map purity for the 'table'
        class. Only regions whose mean table-class softmax probability
        meets a configurable threshold (stable_img_line_purity option,
        default 0.5) have their Sobel img_lines extracted. This combines
        the OOD generalization of heuristic detection with the precision
        of the learned model. Requires feature_extractor. PDF-path only
        (same restriction as table_img_line: depends on extract_base_elements
        Sobel cache path). See apply_stable_img_line() and
        STABLE_IMG_LINE_TABLE_TYPES.

Mutual-exclusion rule -- all table-line input_type values belong to one
    group; requesting more than one simultaneously raises ValueError:
        table_vec_line*, table_img_line*, table_rect_full, img_table_rect_full,
        dtable_vec_line, dtable_img_line, dtable_lsd_line, dtable_rect,
        stable_img_line*
    Rationale: all these types serve the same purpose (representing table
    structure as line or region nodes) via different detection strategies.
    Combining strategies produces duplicate or near-duplicate bboxes for
    the same table, which pollutes the graph without adding useful signal.
    The check is enforced in create_input_data_from_page() (pymupdf_util.py)
    before any extraction runs (see _check_table_input_type_exclusivity()).

Interface:
    apply_table_vec_line(page, data_dict, box_type, page_width, page_height,
                          max_vec_line_num, table_type='all')
        Must be called from WITHIN extract_base_elements() (Step 1),
        passing the same box_type list that extract_base_elements() will
        later assign to data_dict['box_type'] -- that key does not exist
        yet at this point. table_type controls which table shapes
        find_table_grids() looks for; see VEC_LINE_TABLE_TYPES for how
        extract_base_elements() maps each input_type value to it.

    apply_table_img_line(page, data_dict, box_type, page_width, page_height,
                          max_vec_line_num)
        Same calling convention as apply_table_vec_line.

    apply_table_rect_full(page, data_dict, box_type, page_width, page_height)
        Same calling convention as apply_table_vec_line, but adds one
        bbox per fully closed table grid instead of its member lines.
        Handles 'table_rect_full' only -- always table_type='full',
        no table_type parameter (no 'table_rect'/'table_rect_partial'
        input_type values exist yet; add them the same way as
        VEC_LINE_TABLE_TYPES if needed later).

    apply_img_table_rect_full(page, data_dict, box_type, page_width, page_height)
        Same calling convention and behavior as apply_table_rect_full,
        but sourced from Sobel image-based lines (like
        apply_table_img_line) instead of PDF vector paths. Handles
        'img_table_rect_full' only.

    apply_detector_table_lines(page, data_dict, input_type, max_vec_line_num,
                                feature_extractor)
        Must be called AFTER extract_base_elements() has returned (Step
        1.5), since data_dict['box_type'] must already exist as a list.
        feature_extractor.predict() does NOT need to have been called by
        the caller beforehand -- see contract below. No-op if neither
        'dtable_vec_line' nor 'dtable_img_line' is requested.

    apply_detector_table_rect(page, data_dict, input_type, feature_extractor)
        Same calling convention as apply_detector_table_lines (Step
        1.5). No-op if 'dtable_rect' is not requested. Unlike
        apply_detector_table_lines, drops a detected region entirely if
        nothing is found inside it -- see its own docstring.

    apply_stable_img_line(page, data_dict, input_type, max_vec_line_num,
                           feature_extractor, purity_threshold=0.5)
        Must be called AFTER extract_base_elements() (Step 1.5), same
        convention as apply_detector_table_lines. No-op if no
        'stable_img_line*' value is in input_type. Requires
        feature_extractor and a real page (PDF-path only). See
        STABLE_IMG_LINE_TABLE_TYPES for the input_type -> table_type mapping.

Contract for apply_detector_table_lines() (see feature_extractor.predict()
and its mark_cached()/consume_cache() protocol):
    The caller does NOT need to have called feature_extractor.predict()
    beforehand. apply_detector_table_lines() ensures it itself (see
    _ensure_predicted()), using data_dict['image'] as the page_img, via
    the mark_cached()/consume_cache() protocol -- this correctly supports
    both known callers of create_input_data_from_page():
      - Inference (e.g. BoxRFDGNN.predict()): already calls predict() +
        mark_cached() beforehand, typically to reuse a rasterization it
        needed for its own purposes -- that cached result is reused, NOT
        recomputed.
      - Training data generation (e.g. DocumentJsonDataset): passes
        feature_extractor straight through without ever calling
        predict() itself, the same way pymupdf_util_ext.apply_feature_extractors
        already handles 'imf'-family features in Step 2 -- predict() is
        called here, lazily, the first time it's needed.
    Either way, mark_cached() is re-set afterward so a later same-call
    consumer (Step 2's apply_feature_extractors, if 'imf' is also
    requested) can detect "already predicted" too and skip re-predicting.

This module reuses pymupdf_util_base's public helpers (get_vector_lines,
merge_lines, BOX_HLINE, BOX_VLINE) rather than duplicating or modifying
them -- import is one-directional (this module imports from
pymupdf_util_base, never the other way at module load time; see
pymupdf_util_base.extract_base_elements()'s lazy import of
apply_table_vec_line/apply_table_img_line for why).
"""

import pymupdf

from .pymupdf_util_base import get_vector_lines, merge_lines, BOX_HLINE, BOX_VLINE, BOX_TABLE
from .pymupdf_util_imf import _ensure_predicted
from .common_util import softmax_numpy



# Valid values for find_table_grids()'s table_type parameter.
TABLE_TYPE_FULL = 'full'
TABLE_TYPE_PARTIAL = 'partial'
TABLE_TYPE_ALL = 'all'
_VALID_TABLE_TYPES = frozenset({TABLE_TYPE_FULL, TABLE_TYPE_PARTIAL, TABLE_TYPE_ALL})

# input_type string -> table_type value forwarded to find_table_grids().
# 'table_vec_line' / 'table_img_line' keep the original (pre-table_type)
# behavior -- both fully-closed and open-sided tables -- by mapping to
# 'all'. The '_full' / '_partial' suffixed variants let a caller request
# only one kind. Used by extract_base_elements() (pymupdf_util_base.py)
# to dispatch each requested input_type to apply_table_vec_line() /
# apply_table_img_line() with the right table_type; a caller can request
# more than one of these at once (e.g. both '_full' and '_partial'),
# which results in multiple calls whose bboxes/table_grids accumulate
# rather than overwrite (see apply_table_vec_line/apply_table_img_line).
VEC_LINE_TABLE_TYPES = {
    'table_vec_line':         TABLE_TYPE_ALL,
    'table_vec_line_full':    TABLE_TYPE_FULL,
    'table_vec_line_partial': TABLE_TYPE_PARTIAL,
}
IMG_LINE_TABLE_TYPES = {
    'table_img_line':         TABLE_TYPE_ALL,
    'table_img_line_full':    TABLE_TYPE_FULL,
    'table_img_line_partial': TABLE_TYPE_PARTIAL,
}
STABLE_IMG_LINE_TABLE_TYPES = {
    'stable_img_line':         TABLE_TYPE_ALL,
    'stable_img_line_full':    TABLE_TYPE_FULL,
    'stable_img_line_partial': TABLE_TYPE_PARTIAL,
}

# input_type values for LSD-based detector line extraction.
# Unlike the heuristic IMG_LINE_TABLE_TYPES, there is no full/partial variant:
# the table region comes from the layout detector, not from find_table_grids(),
# so structural full/partial validation is not applied and the distinction has
# no meaning. LSD_LINE_TABLE_TYPES is a set (not a dict) used only for
# membership tests in apply_detector_lsd_lines().
LSD_LINE_TABLE_TYPES = {'dtable_lsd_line'}

# All table-oriented input_type values that are mutually exclusive with
# each other. Requesting more than one from this set simultaneously raises
# ValueError in create_input_data_from_page() (see pymupdf_util.py).
TABLE_INPUT_TYPE_EXCLUSIVE_GROUP = frozenset(
    list(VEC_LINE_TABLE_TYPES)
    + list(IMG_LINE_TABLE_TYPES)
    + list(STABLE_IMG_LINE_TABLE_TYPES)
    + list(LSD_LINE_TABLE_TYPES)
    + ['table_rect_full', 'img_table_rect_full',
       'dtable_vec_line', 'dtable_img_line', 'dtable_rect']
)

# Index of the 'table' class in _CLASS_NAMES (ImageFeatureExtractorV2).
# Defined here so apply_stable_img_line does not import from the extractor
# module (that would create a circular dependency). Value must be kept in
# sync with ImageFeatureExtractorV2._CLASS_NAMES.
_TABLE_CLASS_IDX = 4


def find_table_grids(h_lines, v_lines, border_tol=3, divider_tol=3, table_type=TABLE_TYPE_FULL):
    """
    Find full-border-table vector-line structures: rectangular regions
    formed by "table-spanning" lines (lines that run the full width or
    height of the table, never partial/cell-level segments) that are
    unambiguously a real grid table -- i.e. structurally show >= 2 rows
    or >= 2 columns -- as opposed to a simple line or closed rectangle
    with no internal structure (decorative frame, image border, single
    cell, section-underline).

    Covers four border configurations, all requiring the SAME strength
    of evidence used by the strictest (fully-closed) case: at least 3
    table-spanning lines forming an L/U/rectangle shape, sharing
    endpoints within border_tol/divider_tol of each other. A single
    matching line pair is never enough on its own -- that would also
    match arbitrary decorative lines with a coincidentally shared span.

    1. Fully closed: 2 h-lines (top/bottom, matching x-span) + 2 v-lines
       (left/right, matching y-span) + >= 1 internal divider (h or v).
       Selected when table_type == 'full'.
    2. Left/right open: 2 h-lines (top/bottom, matching x-span) + >= 1
       internal v-line spanning the full y-span between them. No side
       borders required, but the interior divider must run the table's
       full height, exactly like a "real" column divider would.
       Selected when table_type == 'partial'.
    3. Top open: 2 v-lines (left/right) whose BOTTOM endpoints match
       within tolerance, + a closing h-line at that bottom y spanning
       (left, right), + >= 1 internal v-line sharing that same bottom
       endpoint (evidence of >= 2 columns). The open (top) end has no
       required closing line; its bbox edge is the shortest verified
       extent among left/right/divider (conservative: never claims
       more structure than is actually drawn).
       Selected when table_type == 'partial'.
    4. Bottom open: mirror of (3), anchored on the TOP endpoints.
       Selected when table_type == 'partial'.

    Args:
        h_lines: list of objects with x0, y0, x1, y1 (horizontal lines,
            y0 == y1 up to float noise). As returned by
            get_vector_lines()/merge_lines().
        v_lines: list of objects with x0, y0, x1, y1 (vertical lines,
            x0 == x1 up to float noise).
        border_tol: float. Max allowed gap (in points) between two
            lines' endpoints when deciding they form the same table
            border -- absorbs drawing imprecision (a border line drawn
            a couple of points short of, or past, where it "should"
            end), NOT a proximity threshold for deciding whether two
            structurally different, merely nearby, tables should be
            treated as one. Kept tight (default 3pt, same magnitude as
            merge_lines' tolerance) so two separate tables that happen
            to sit close together are not fused.
        divider_tol: float. Same tolerance, applied to an internal
            divider line's endpoints against the (already confirmed)
            border. Kept as a separate parameter from border_tol in
            case future tuning needs them to diverge, though they
            share the same default and rationale.
        table_type: str, 'full' (default), 'partial', or 'all'.
            'full'    Only case 1 (fully closed border: both left and
                      right vertical borders present, matching y-span,
                      plus >= 1 internal divider). open_side is always
                      None in the results.
            'partial' Only cases 2, 3, 4 (one side left undetermined:
                      left_right / top / bottom open_side). Case 1 is
                      skipped entirely, so a fully-closed table will
                      NOT be reported when table_type == 'partial' even
                      though its lines would also satisfy case 2/3/4 --
                      use 'full' or 'all' to catch fully-closed tables.
            'all'     All four cases together (case 1 plus cases 2/3/4)
                      -- equivalent to calling find_table_grids() once
                      with table_type='full' and once with 'partial'
                      and concatenating the results. This matches the
                      original (pre-table_type) behavior of this
                      function.
            'full' and 'partial' are mutually exclusive; use 'all' when
            both kinds are needed instead of calling find_table_grids()
            twice and combining with dedup_table_grids() yourself.

    Raises:
        ValueError: if table_type is not 'full' or 'partial'.

    Returns:
        list of dicts, one per validated table candidate:
            {
                "bbox": (x0, y0, x1, y1),
                "h_members": [line, ...],  # border + divider h-lines
                "v_members": [line, ...],  # border + divider v-lines
                "open_side": None | "left_right" | "top" | "bottom",
            }
        Note: candidates may still be nested or overlapping (e.g. a
        merged-cell table can genuinely contain a smaller full-grid
        sub-region). Callers that only want the outermost table per
        grid should follow up with dedup_table_grids().
    """
    if table_type not in _VALID_TABLE_TYPES:
        raise ValueError(
            f"table_type must be one of {sorted(_VALID_TABLE_TYPES)}, got {table_type!r}"
        )

    def close(a, b, t=border_tol):
        return abs(a - b) <= t

    tables = []

    # ---- Cases 1 & 2: anchored on a pair of horizontal (top/bottom) lines ----
    n_h = len(h_lines)
    for i in range(n_h):
        r1 = h_lines[i]
        x0_1, x1_1 = sorted((r1.x0, r1.x1))
        y_1 = (r1.y0 + r1.y1) / 2.0

        for j in range(i + 1, n_h):
            r2 = h_lines[j]
            x0_2, x1_2 = sorted((r2.x0, r2.x1))
            y_2 = (r2.y0 + r2.y1) / 2.0

            if y_1 == y_2:
                continue
            # top/bottom horizontal borders must agree on both x
            # endpoints within tolerance -- this is the same drawing-
            # imprecision issue as the vertical borders below, just on
            # the other axis (e.g. a table's bottom border drawn a
            # touch narrower than its top border).
            if not (close(x0_1, x0_2) and close(x1_1, x1_2)):
                continue

            x0k, x1k = min(x0_1, x0_2), max(x1_1, x1_2)
            if x1k - x0k < 30:  # guard against degenerate/near-zero-width matches
                continue
            y_top, y_bottom = sorted((y_1, y_2))
            top_line, bottom_line = (r1, r2) if y_1 < y_2 else (r2, r1)

            # Verticals spanning the full (y_top, y_bottom) height,
            # tolerating a couple of points of overshoot/undershoot on
            # either endpoint rather than requiring exact match.
            full_verticals = [
                r for r in v_lines
                if close(min(r.y0, r.y1), y_top)
                and close(max(r.y0, r.y1), y_bottom)
            ]
            left_border = [r for r in full_verticals if close((r.x0 + r.x1) / 2.0, x0k)]
            right_border = [r for r in full_verticals if close((r.x0 + r.x1) / 2.0, x1k)]
            interior_v = [
                r for r in full_verticals
                if x0k < (r.x0 + r.x1) / 2.0 < x1k
                and r not in left_border
                and r not in right_border
            ]

            h_dividers = [
                r for r in h_lines
                if r is not top_line and r is not bottom_line
                and y_top < (r.y0 + r.y1) / 2.0 < y_bottom
                and close(min(r.x0, r.x1), x0k, divider_tol)
                and close(max(r.x0, r.x1), x1k, divider_tol)
            ]

            if left_border and right_border:
                # Case 1: fully closed border. Still requires >= 1
                # internal divider to reject plain undivided rectangles.
                # Only reported when table_type is 'full' or 'all'.
                if table_type in (TABLE_TYPE_FULL, TABLE_TYPE_ALL) and (h_dividers or interior_v):
                    tables.append({
                        "bbox": (x0k, y_top, x1k, y_bottom),
                        "h_members": [top_line, bottom_line] + h_dividers,
                        "v_members": left_border + right_border + interior_v,
                        "open_side": None,
                    })
            elif interior_v:
                # Case 2: left/right open. The full-height interior
                # divider is the only remaining evidence of >= 2
                # columns without side borders.
                # Only reported when table_type is 'partial' or 'all'.
                if table_type in (TABLE_TYPE_PARTIAL, TABLE_TYPE_ALL):
                    tables.append({
                        "bbox": (x0k, y_top, x1k, y_bottom),
                        "h_members": [top_line, bottom_line] + h_dividers,
                        "v_members": interior_v,
                        "open_side": "left_right",
                    })

    # ---- Cases 3 & 4: anchored on a pair of vertical (left/right) lines,
    # closed on only one of top/bottom.
    # Both cases are open-sided by definition, so this whole block only
    # runs when table_type is 'partial' or 'all'. ----
    if table_type in (TABLE_TYPE_PARTIAL, TABLE_TYPE_ALL):
        n_v = len(v_lines)
        for i in range(n_v):
            r1 = v_lines[i]
            y0_1, y1_1 = sorted((r1.y0, r1.y1))
            x_1 = (r1.x0 + r1.x1) / 2.0

            for j in range(i + 1, n_v):
                r2 = v_lines[j]
                y0_2, y1_2 = sorted((r2.y0, r2.y1))
                x_2 = (r2.x0 + r2.x1) / 2.0

                if x_1 == x_2:
                    continue
                x0k, x1k = min(x_1, x_2), max(x_1, x_2)
                if x1k - x0k < 5:
                    continue

                for is_top_closed in (True, False):
                    if is_top_closed:
                        if not close(y0_1, y0_2):
                            continue
                        anchor = min(y0_1, y0_2)
                        open_end_1, open_end_2 = y1_1, y1_2
                    else:
                        if not close(y1_1, y1_2):
                            continue
                        anchor = max(y1_1, y1_2)
                        open_end_1, open_end_2 = y0_1, y0_2

                    # Closing horizontal line at the anchor y, spanning
                    # (x0k, x1k) -- this is the one drawn edge of the table.
                    closing_h = [
                        r for r in h_lines
                        if close((r.y0 + r.y1) / 2.0, anchor)
                        and close(min(r.x0, r.x1), x0k)
                        and close(max(r.x0, r.x1), x1k)
                    ]
                    if not closing_h:
                        continue

                    # Internal vertical divider sharing the same anchor
                    # endpoint as left/right -- the evidence of >= 2
                    # columns that justifies treating this as a table
                    # rather than e.g. a two-sided text-block border.
                    interior_v = []
                    open_ends = [open_end_1, open_end_2]
                    for r in v_lines:
                        x = (r.x0 + r.x1) / 2.0
                        if not (x0k < x < x1k):
                            continue
                        r_y0, r_y1 = sorted((r.y0, r.y1))
                        r_anchor = r_y0 if is_top_closed else r_y1
                        if close(r_anchor, anchor, divider_tol):
                            interior_v.append(r)
                            open_ends.append(r_y1 if is_top_closed else r_y0)

                    if not interior_v:
                        continue

                    # Open-end bbox coordinate: the shortest verified
                    # extent among left/right/divider, so the reported bbox
                    # never claims more structure than is actually drawn.
                    if is_top_closed:
                        y_top, y_bottom = anchor, min(open_ends)
                    else:
                        y_top, y_bottom = max(open_ends), anchor

                    tables.append({
                        "bbox": (x0k, y_top, x1k, y_bottom),
                        "h_members": closing_h,
                        "v_members": [r1, r2] + interior_v,
                        "open_side": "bottom" if is_top_closed else "top",
                    })

    return tables


def dedup_table_grids(tables, tol=3):
    """
    Keep only maximal grid table candidates: drop any candidate whose
    bbox is contained within another candidate's bbox (within tol).

    find_table_grids can, in rare cases, report multiple nested or
    overlapping candidates for what is really a single table:
    - a genuinely distinct full-grid sub-region that happens to sit
      inside a larger merged-cell table.
    - a fully-closed reading AND an open-sided reading of the SAME
      lines (e.g. a table's border is closed on every side, but the
      open-sided matcher still fires because it doesn't know that).
      The open-sided candidate's bbox can end up slightly LARGER than
      the closed one's on the "open" edge, because that edge falls
      back to an unverified line endpoint instead of the confirmed
      border coordinate -- so sorting by area alone would keep the
      WEAKER (open-sided) reading over the STRONGER (fully-closed)
      one, and exact containment would fail to recognize them as the
      same table at all (the open-sided bbox isn't strictly inside
      the closed one).

    To handle this, candidates are ranked primarily by evidence
    strength (fully-closed first, then by member count, then area as
    a final tie-breaker), and containment is checked with the same
    magnitude of tolerance (tol, default 3pt) used for border/divider
    matching in find_table_grids, so a same-table open-sided reading
    that overshoots the closed reading by a couple of points is still
    recognized as a duplicate and dropped.
    """
    def rank_key(t):
        is_closed = t.get("open_side") is None
        member_count = len(t["h_members"]) + len(t["v_members"])
        return (is_closed, member_count, _bbox_area(t["bbox"]))

    ordered = sorted(tables, key=rank_key, reverse=True)

    kept = []
    for cand in ordered:
        if any(_bbox_contains(kept_t["bbox"], cand["bbox"], tol) for kept_t in kept):
            continue
        kept.append(cand)
    return kept


def _bbox_area(bbox):
    x0, y0, x1, y1 = bbox
    return (x1 - x0) * (y1 - y0)


def _bbox_contains(outer, inner, tol=0):
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        ox0 - tol <= ix0
        and oy0 - tol <= iy0
        and ox1 + tol >= ix1
        and oy1 + tol >= iy1
    )

def apply_table_vec_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num, table_type=TABLE_TYPE_ALL):
    """
    Append 'table_vec_line' bboxes to data_dict: only vec_line lines that
    belong to a validated table structure (heuristic grid detection via
    find_table_grids). Unlike 'vec_line' (every detected vector line),
    this only keeps table-spanning lines that belong to a validated
    table structure (fully closed border, left/right open, or
    top/bottom open -- always with at least one internal row/column
    divider, see find_table_grids) -- i.e. an actual table -- which
    avoids feeding GNN-based layout detection unstructured/decorative
    vector lines that caused GT-assignment and misclassification issues.

    Must be called from WITHIN extract_base_elements() (Step 1): pass
    the same box_type list that extract_base_elements() will later
    assign to data_dict['box_type'] -- that key does not exist yet at
    this point.

    Args:
        table_type: forwarded to find_table_grids() -- 'full' (closed
            border only), 'partial' (open-sided only), or 'all' (both,
            default). Callers that want 'full' and 'partial' as
            SEPARATE input_type values (see VEC_LINE_TABLE_TYPES) call
            this function once per requested value; each call's tables
            are appended to data_dict['table_grids'] rather than
            replacing it, so results from multiple calls accumulate
            instead of the last call's clobbering the previous one's.
    """
    # Reuse h_lines/v_lines already computed for 'vec_line' if
    # present in this same call, to avoid re-parsing page.get_drawings()
    # and re-rasterizing the page a second time.
    if 'vector_lines' in data_dict:
        table_h_lines, table_v_lines = data_dict['vector_lines']
    else:
        table_h_lines, table_v_lines = get_vector_lines(page, omit_invisible=True)
        table_h_lines = merge_lines(table_h_lines, orientation='h', tolerance=3)
        table_v_lines = merge_lines(table_v_lines, orientation='v', tolerance=3)
        data_dict['vector_lines'] = (table_h_lines, table_v_lines)

    tables = find_table_grids(table_h_lines, table_v_lines, table_type=table_type)
    tables = dedup_table_grids(tables)
    data_dict.setdefault('table_grids', []).extend(tables)

    table_h_members = []
    table_v_members = []
    for t in tables:
        table_h_members.extend(t['h_members'])
        table_v_members.extend(t['v_members'])

    table_h_members = sorted(table_h_members, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]
    table_v_members = sorted(table_v_members, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]

    _add_line_bboxes(data_dict, box_type, table_h_members, page_width, page_height, True, BOX_HLINE)
    _add_line_bboxes(data_dict, box_type, table_v_members, page_width, page_height, False, BOX_VLINE)


def apply_img_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num):
    """
    Append 'img_line' bboxes to data_dict: the page-wide Sobel image-based
    counterpart of 'vec_line'.

    'vec_line' extracts every PDF vector line from page.get_drawings();
    'img_line' instead runs Sobel edge detection on the rasterized page image
    and adds every detected horizontal/vertical line as a bbox -- useful for
    PDFs where lines are encoded as filled rects, background colors, or embedded
    images rather than explicit vector drawings, which are missed entirely by
    'vec_line'.

    Raw Sobel output is converted to pymupdf.Rect objects and then passed
    through merge_lines() (tolerance=3, same as 'vec_line') to collapse
    fragmented collinear segments into longer continuous lines before adding
    them to data_dict.

    The raw Sobel result is cached in data_dict['image_lines_raw'] so that
    subsequent callers in the same pipeline step ('table_img_line',
    'img_table_rect_full', 'dtable_img_line') can reuse it without re-running
    Sobel on the same page image.

    Also caches data_dict['image_lines'] as (h_lines, v_lines) of merged
    pymupdf.Rect objects (parallel to data_dict['vector_lines'] for 'vec_line'),
    so apply_detector_table_lines() can reuse the merged result when
    'dtable_img_line' is also requested.

    Must be called from WITHIN extract_base_elements() (Step 1): pass
    the same box_type list that extract_base_elements() will later
    assign to data_dict['box_type'] -- that key does not exist yet at
    this point.
    """
    from .common_util import extract_sobel_line_from_image

    img_line_tol = 3  # same as vec_line's merge_lines tolerance

    if 'image_lines_raw' in data_dict:
        raw_img_h_lines, raw_img_v_lines = data_dict['image_lines_raw']
    else:
        raw_img_h_lines, raw_img_v_lines = extract_sobel_line_from_image(
            data_dict['image'], page_width, page_height,
        )
        data_dict['image_lines_raw'] = (raw_img_h_lines, raw_img_v_lines)

    img_h_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_h_lines],
        orientation='h', tolerance=img_line_tol,
    )
    img_v_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_v_lines],
        orientation='v', tolerance=img_line_tol,
    )

    # Cache merged lines for apply_detector_table_lines() to reuse when
    # 'dtable_img_line' is also requested in the same pipeline call.
    data_dict['image_lines'] = (img_h_lines, img_v_lines)

    img_h_sorted = sorted(img_h_lines, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]
    img_v_sorted = sorted(img_v_lines, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]

    _add_line_bboxes(data_dict, box_type, img_h_sorted, page_width, page_height, True, BOX_HLINE)
    _add_line_bboxes(data_dict, box_type, img_v_sorted, page_width, page_height, False, BOX_VLINE)


def apply_table_img_line(page, data_dict, box_type, page_width, page_height, max_vec_line_num, table_type=TABLE_TYPE_ALL):
    """
    Append 'table_img_line' bboxes to data_dict: same idea as
    apply_table_vec_line, but sourced from Sobel image-based line
    detection (extract_sobel_line_from_image) instead of PDF vector
    paths -- useful for PDFs where table lines are encoded as filled
    rects/background colors/embedded images rather than explicit vector
    drawings, which get missed entirely by 'vec_line'/'table_vec_line'.

    Tolerances are doubled relative to apply_table_vec_line's default
    (border_tol=6, divider_tol=6, dedup tol=6 vs. 3): image-based line
    detection has a different, generally larger, error profile than PDF
    vector paths -- on top of the same kind of fragmentation
    merge_lines() already fixes, Sobel edge localization can place a
    line a couple of pixels off from its true position (anti-aliasing /
    thresholding), which is coordinate noise vector paths don't have. If
    page.get_pixmap() is ever rendered at higher zoom than the current
    default, these tolerances (in page-point units, same coordinate
    system as vec_line since extract_sobel_line_from_image already
    returns page-point coordinates) should be re-checked against real
    samples again.

    Must be called from WITHIN extract_base_elements() (Step 1): pass
    the same box_type list that extract_base_elements() will later
    assign to data_dict['box_type'] -- that key does not exist yet at
    this point.

    Args:
        table_type: forwarded to find_table_grids() -- 'full' (closed
            border only), 'partial' (open-sided only), or 'all' (both,
            default). Callers that want 'full' and 'partial' as
            SEPARATE input_type values (see IMG_LINE_TABLE_TYPES) call
            this function once per requested value; each call's tables
            are appended to data_dict['table_img_grids'] rather than
            replacing it, so results from multiple calls accumulate
            instead of the last call's clobbering the previous one's.
    """
    from .common_util import extract_sobel_line_from_image

    img_table_tol = 6  # 2x table_vec_line's default (3), see note above

    if 'image_lines_raw' in data_dict:
        raw_img_h_lines, raw_img_v_lines = data_dict['image_lines_raw']
    else:
        raw_img_h_lines, raw_img_v_lines = extract_sobel_line_from_image(
            data_dict['image'], page_width, page_height,
        )
        data_dict['image_lines_raw'] = (raw_img_h_lines, raw_img_v_lines)

    table_img_h_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_h_lines],
        orientation='h', tolerance=img_table_tol,
    )
    table_img_v_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_v_lines],
        orientation='v', tolerance=img_table_tol,
    )

    img_tables = find_table_grids(
        table_img_h_lines, table_img_v_lines,
        border_tol=img_table_tol, divider_tol=img_table_tol,
        table_type=table_type,
    )
    img_tables = dedup_table_grids(img_tables, tol=img_table_tol)
    data_dict.setdefault('table_img_grids', []).extend(img_tables)

    img_table_h_members = []
    img_table_v_members = []
    for t in img_tables:
        img_table_h_members.extend(t['h_members'])
        img_table_v_members.extend(t['v_members'])

    img_table_h_members = sorted(img_table_h_members, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]
    img_table_v_members = sorted(img_table_v_members, key=lambda r: r.width * r.height, reverse=True)[:max_vec_line_num]

    _add_line_bboxes(data_dict, box_type, img_table_h_members, page_width, page_height, True, BOX_HLINE)
    _add_line_bboxes(data_dict, box_type, img_table_v_members, page_width, page_height, False, BOX_VLINE)


def apply_table_rect_full(page, data_dict, box_type, page_width, page_height):
    """
    Append 'table_rect_full' bboxes to data_dict: the OUTER bbox of each
    heuristically validated, fully-closed table grid (find_table_grids
    with table_type='full') -- ONE box per table, unlike
    apply_table_vec_line which adds every individual h/v border/divider
    line as its own box. The internal border/divider lines themselves
    are NOT added.

    Rationale: adding every internal line as a separate node inflates
    the GNN's node count for a table-heavy page. This input_type tests
    whether a single coarse table-region node preserves downstream
    performance while cutting node count and therefore compute.

    Must be called from WITHIN extract_base_elements() (Step 1): pass
    the same box_type list that extract_base_elements() will later
    assign to data_dict['box_type'] -- that key does not exist yet at
    this point.
    """
    # Reuse h_lines/v_lines already computed for 'vec_line'/
    # 'table_vec_line'/etc. if present in this same call, to avoid
    # re-parsing page.get_drawings() and re-rasterizing the page again.
    if 'vector_lines' in data_dict:
        h_lines, v_lines = data_dict['vector_lines']
    else:
        h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
        h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
        v_lines = merge_lines(v_lines, orientation='v', tolerance=3)
        data_dict['vector_lines'] = (h_lines, v_lines)

    tables = find_table_grids(h_lines, v_lines, table_type=TABLE_TYPE_FULL)
    tables = dedup_table_grids(tables)
    data_dict.setdefault('table_rect_grids', []).extend(tables)

    for t in tables:
        x0, y0, x1, y1 = t['bbox']
        x1v = max(0.0, x0)
        y1v = max(0.0, y0)
        x2v = min(page_width, x1)
        y2v = min(page_height, y1)
        if not (0 <= x1v < x2v <= page_width and 0 <= y1v < y2v <= page_height):
            continue
        bbox = [x1v, y1v, x2v, y2v]
        if bbox not in data_dict['bboxes']:
            data_dict['bboxes'].append(bbox)
            data_dict['text'].append('')
            box_type.append(BOX_TABLE)


def apply_img_table_rect_full(page, data_dict, box_type, page_width, page_height):
    """
    Append 'img_table_rect_full' bboxes to data_dict: same idea as
    apply_table_rect_full (ONE outer bbox per fully-closed table, no
    internal border/divider lines added), but sourced from Sobel
    image-based line detection (extract_sobel_line_from_image) instead
    of PDF vector paths -- the image-based counterpart of
    'table_rect_full', exactly as apply_table_img_line is to
    apply_table_vec_line. Useful for PDFs where table lines are encoded
    as filled rects/background colors/embedded images rather than
    explicit vector drawings, which get missed entirely by
    'vec_line'/'table_vec_line'/'table_rect_full'.

    Tolerances are doubled relative to apply_table_rect_full's default
    (border_tol=6, divider_tol=6, dedup tol=6 vs. 3), for the same
    reason as apply_table_img_line: image-based line detection has a
    larger, different error profile than PDF vector paths (Sobel edge
    localization noise on top of the fragmentation merge_lines() already
    fixes).

    Must be called from WITHIN extract_base_elements() (Step 1): pass
    the same box_type list that extract_base_elements() will later
    assign to data_dict['box_type'] -- that key does not exist yet at
    this point.
    """
    from .common_util import extract_sobel_line_from_image

    img_table_tol = 6  # 2x table_rect_full's default (3), see note above

    if 'image_lines_raw' in data_dict:
        raw_img_h_lines, raw_img_v_lines = data_dict['image_lines_raw']
    else:
        raw_img_h_lines, raw_img_v_lines = extract_sobel_line_from_image(
            data_dict['image'], page_width, page_height,
        )
        data_dict['image_lines_raw'] = (raw_img_h_lines, raw_img_v_lines)

    img_h_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_h_lines],
        orientation='h', tolerance=img_table_tol,
    )
    img_v_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_v_lines],
        orientation='v', tolerance=img_table_tol,
    )

    tables = find_table_grids(
        img_h_lines, img_v_lines,
        border_tol=img_table_tol, divider_tol=img_table_tol,
        table_type=TABLE_TYPE_FULL,
    )
    tables = dedup_table_grids(tables, tol=img_table_tol)
    data_dict.setdefault('img_table_rect_grids', []).extend(tables)

    for t in tables:
        x0, y0, x1, y1 = t['bbox']
        x1v = max(0.0, x0)
        y1v = max(0.0, y0)
        x2v = min(page_width, x1)
        y2v = min(page_height, y1)
        if not (0 <= x1v < x2v <= page_width and 0 <= y1v < y2v <= page_height):
            continue
        bbox = [x1v, y1v, x2v, y2v]
        if bbox not in data_dict['bboxes']:
            data_dict['bboxes'].append(bbox)
            data_dict['text'].append('')
            box_type.append(BOX_TABLE)


def _add_line_bboxes(data_dict, box_type, rects, page_width, page_height, is_horizontal, box_type_str):
    """
    Append line rects to data_dict as bboxes, matching the base module's
    dedup / clamping conventions.

    box_type is taken as an explicit list parameter (rather than reading
    data_dict['box_type']) because the two calling contexts differ:
    apply_table_vec_line/apply_table_img_line run from WITHIN
    extract_base_elements(), before data_dict['box_type'] exists (it is
    only assigned at the very end of that function); callers there pass
    the function's local box_type list. apply_detector_table_lines runs
    AFTER extract_base_elements() returns, so data_dict['box_type']
    already exists there and can be passed directly.
    """
    for rect in rects:
        x1, y1, x2, y2 = rect.x0, rect.y0, rect.x1, rect.y1
        if is_horizontal:
            if not (0 <= x1 < x2 <= page_width and 0 <= y1 <= y2 <= page_height):
                continue
            if y1 == y2:
                y2 = y1 + 1
        else:
            if not (0 <= x1 <= x2 <= page_width and 0 <= y1 < y2 <= page_height):
                continue
            if x1 == x2:
                x2 = x1 + 1
        bbox = [x1, y1, x2, y2]
        if bbox not in data_dict['bboxes']:
            data_dict['bboxes'].append(bbox)
            data_dict['text'].append('')
            box_type.append(box_type_str)


def _pixel_to_point_scale(page_img, page_width, page_height):
    """
    Return (scale_x, scale_y): pixel-per-point ratios for page_img
    relative to the PDF page size in points. Shared by
    get_detected_table_regions() (to convert detector boxes into point
    space) and apply_detector_table_lines() (to convert a pixel-space
    tolerance into the matching point-space tolerance) so both use the
    exact same scale.
    """
    img_h, img_w = page_img.shape[:2]
    return img_w / page_width, img_h / page_height


def get_detected_table_regions(feature_extractor, page_img, page_width, page_height):
    """
    Return detected 'table' class regions, converted to PDF point
    coordinates, from the last feature_extractor.predict() call.

    feature_extractor.get_layout_detections() reports boxes in the pixel
    coordinate space of the page_img that was passed to predict() (i.e.
    orig_h/orig_w == page_img.shape[:2] at predict() time). This function
    itself does NOT call predict() -- it only reads existing detection
    results, and expects the caller (apply_detector_table_lines(), via
    _ensure_predicted()) to have already ensured predict() was run, with
    a page_img of the same pixel resolution as the page_img given here.

    Args:
        feature_extractor: object with get_layout_detections(), e.g.
            ImageFeatureExtractorV2. predict() must already have been
            called on it for the current page.
        page_img: (H, W, C) uint8 array. Must have the same pixel
            resolution as the page_img used for feature_extractor.predict().
        page_width, page_height: PDF page size in points (page.rect[2:4]).

    Returns:
        list of (x0, y0, x1, y1) tuples in PDF point coordinates, one per
        detected 'table' class box (raw get_layout_detections() output --
        no additional score filtering is applied here beyond what
        get_layout_detections() already bakes in).

    Raises:
        RuntimeError: propagated from get_layout_detections() if
            feature_extractor.predict() has not been called yet.
    """
    detections = feature_extractor.get_layout_detections()

    scale_x, scale_y = _pixel_to_point_scale(page_img, page_width, page_height)

    regions = []
    for box, name in zip(detections['boxes'], detections['names']):
        if name != 'table':
            continue
        px_x0, px_y0, px_x1, px_y1 = box
        regions.append((
            px_x0 / scale_x,
            px_y0 / scale_y,
            px_x1 / scale_x,
            px_y1 / scale_y,
        ))
    return regions


def _line_in_table_regions(rect, regions, tol_x, tol_y):
    """
    Overlap test: does this line's bbox overlap (any part of it) with
    any detected table region, expanded by tol_x/tol_y on every side
    (up/down/left/right)?

    Switched from a center-point test to a full bbox-overlap test:
    when detection accuracy drops, a detector box can come out
    significantly UNDERSIZED relative to the real table (not just
    shifted by a few pixels). A long border line's midpoint can then
    land outside the region even after expansion, even though the line
    clearly runs along (part of) that table's edge -- a center-point
    test would wrongly reject it. An overlap test only requires ANY
    part of the line to fall within the expanded region, which is more
    robust to this kind of detector undersizing while a plain
    center-point check would still work fine for it as a special case
    (a rect always "overlaps" a region that contains its center).

    tol_x/tol_y absorb the layout detection model's regression noise on
    the detected box edges -- unlike find_table_grids's border/divider
    matching (which validates exact structural evidence), a detector box
    is a single noisy bbox with no such internal cross-check, so a
    border line can legitimately land outside the raw detected box (or
    a bit inside it) even when it visually belongs to that table.
    Separate x/y tolerances (rather than one scalar) because they are
    derived from a fixed PIXEL tolerance converted through possibly
    different x/y pixel-per-point scales -- see apply_detector_table_lines.
    """
    rx0, rx1 = sorted((rect.x0, rect.x1))
    ry0, ry1 = sorted((rect.y0, rect.y1))
    for (x0, y0, x1, y1) in regions:
        ex0, ex1 = x0 - tol_x, x1 + tol_x
        ey0, ey1 = y0 - tol_y, y1 + tol_y
        # Standard axis-aligned rect overlap: NOT separated on either axis.
        if rx1 < ex0 or rx0 > ex1 or ry1 < ey0 or ry0 > ey1:
            continue
        return True
    return False


# _ensure_predicted() has been moved to pymupdf_util_imf (canonical home:
# the protocol is not table-specific but shared by all feature_extractor
# consumers). It is imported at the top of this module and re-exported so
# that any existing external callers that import it from here continue to work.


def _line_length(rect):
    """
    Rank key for truncating candidate lines to max_vec_line_num: how
    long the line runs, not its area.

    A border/divider line is a thin rectangle (width*height is small
    regardless of how long the line actually is, since the "thickness"
    dimension is near-zero noise, not signal). Sorting by area instead
    of length under-ranks long, thin, important border lines relative
    to shorter-but-slightly-thicker noise lines -- which matters more
    now that the overlap-based table-region test (see
    _line_in_table_regions) accepts a larger candidate pool than the
    previous center-point test did: a bigger pool competing for the
    same max_vec_line_num cutoff makes a bad ranking key more likely to
    push out a real border line in favor of noise.
    """
    return max(rect.width, rect.height)


def _sobel_lines_in_regions(page_img, page_width, page_height, table_regions, min_length_ratio=0.03):
    """
    Run extract_sobel_line_from_image() separately on each detected
    table region (cropped from page_img, with generous padding) instead
    of on the whole page, then translate the returned line coordinates
    back into full-page PDF point space.

    Rationale: get_detected_table_regions() already tells us WHERE on
    the page a table might be, and apply_detector_table_lines()'s
    'dtable_img_line' branch only keeps lines that fall inside one of
    these (typically much smaller than the full page) regions anyway.
    Running Sobel edge detection on the ENTIRE page when the caller only
    cares about lines inside these regions wastes computation
    proportional to (page area - total table area). This crops each
    region and offsets the result back instead, so cost scales with
    total table area rather than page area. Callers should skip calling
    this entirely when table_regions is empty (the filtered result would
    be empty either way -- see apply_detector_table_lines()).

    Padding: deliberately larger than 'dtable_img_line's own
    _line_in_table_regions() matching tolerance (dtable_tol_px=10),
    because that tolerance assumes the FULL, untruncated line is
    available to test against -- a naive crop at exactly
    region+dtable_tol_px could cut off a border line's far end before
    that comparison ever happens, silently changing results relative to
    full-page processing. get_detected_table_regions()'s own docs
    already note detector boxes can come out significantly UNDERSIZED
    relative to the real table, so a real border can legitimately
    extend well past the raw detected edge. Padding here is the larger
    of a fixed floor (20pt) and 15% of the region's own width/height, to
    scale with probable undersizing on larger tables -- NOT validated
    against real samples yet; tune against real data before relying on
    this in production, same as every other tolerance in this file.

    Region handling: each detected table region is cropped and
    processed SEPARATELY (one extract_sobel_line_from_image() call per
    region), not merged into one bounding-box crop -- keeps cost low
    even when tables are scattered far apart on the page, at the cost of
    one Sobel call per region (fine for the typical few-tables-per-page
    case; degrades if a page has many small scattered table detections,
    in which case a single bounding-box crop would trade accuracy for
    fewer calls).

    min_length_ratio correction (VERIFIED against common_util.py):
    extract_sobel_line_from_image() filters out any candidate run
    shorter than `min_length_ratio * img_w` pixels (h-lines) /
    `min_length_ratio * img_h` pixels (v-lines) -- i.e. relative to
    the pixel size of WHATEVER image it is given, not the true full
    page. Called naively on a small crop, that threshold shrinks along
    with the crop, so a short fragment that full-page processing would
    have discarded as noise could pass here instead -- a real behavior
    change, not just a coordinate-offset issue. Since crop_width_pt <=
    page_width and crop_height_pt <= page_height always (see clamping
    below), the crop call's own internal filter is always AT LEAST as
    permissive as the full-page filter would be, so nothing that
    full-page processing would keep gets dropped early -- but the
    reverse isn't true, so this function re-applies the exact
    full-page-equivalent absolute length threshold (min_length_ratio *
    page_width / page_height, using the TRUE page dimensions) as a
    post-filter after each crop call, before offsetting. This makes the
    cropped result's line-length filtering equivalent to what
    extract_sobel_line_from_image(page_img, page_width, page_height,
    min_length_ratio=min_length_ratio) would have produced on the whole
    page.

    ASSUMPTION ON extract_sobel_line_from_image()'S COORDINATE CONTRACT
    (VERIFIED against common_util.py): the function converts pixel-space
    line detections to point space using ONLY the given image's own
    pixel dimensions and the given page_width/page_height arguments
    (sx = page_width / img_w, sy = page_height / img_h -- no reference
    to any "original full page" beyond what it is told). So passing it
    a cropped sub-image together with that crop's own real-world
    width/height yields lines in point space relative to the CROP's
    top-left corner, which this function then offsets back by the
    crop's own page-space origin. Confirmed correct as of this review.

    Args:
        page_img: (H, W, C) uint8 full-page raster image.
        page_width, page_height: PDF page size in points.
        table_regions: list of (x0, y0, x1, y1) tuples in PDF point
            coordinates, as returned by get_detected_table_regions().
            Must be non-empty (callers should skip calling this
            otherwise).
        min_length_ratio: forwarded to extract_sobel_line_from_image()
            AND used to compute the full-page-equivalent post-filter
            threshold described above -- must be kept in sync with
            whatever value the caller would otherwise pass for a
            full-page call (default matches
            extract_sobel_line_from_image()'s own default of 0.03).

    Returns:
        (raw_h_lines, raw_v_lines): same shape/format as
        extract_sobel_line_from_image()'s own return value -- lists of
        (x1, y1, x2, y2) tuples in full-page PDF point coordinates,
        concatenated across all regions. May contain near-duplicate/
        overlapping detections if two regions' padded crops overlap --
        downstream merge_lines() already absorbs this the same way it
        absorbs raw Sobel fragmentation.
    """
    from .common_util import extract_sobel_line_from_image

    img_h, img_w = page_img.shape[:2]
    px_per_pt_x = img_w / page_width
    px_per_pt_y = img_h / page_height

    # Full-page-equivalent absolute minimum line length, in points --
    # see "min_length_ratio correction" above. Applied identically to
    # every region's crop regardless of that crop's own (looser)
    # internal filter.
    min_h_len_pt = min_length_ratio * page_width
    min_v_len_pt = min_length_ratio * page_height

    all_h_lines = []
    all_v_lines = []

    for (rx0, ry0, rx1, ry1) in table_regions:
        pad_x_pt = max(20.0, 0.15 * (rx1 - rx0))
        pad_y_pt = max(20.0, 0.15 * (ry1 - ry0))

        ex0_pt = max(0.0, rx0 - pad_x_pt)
        ey0_pt = max(0.0, ry0 - pad_y_pt)
        ex1_pt = min(page_width, rx1 + pad_x_pt)
        ey1_pt = min(page_height, ry1 + pad_y_pt)

        crop_width_pt = ex1_pt - ex0_pt
        crop_height_pt = ey1_pt - ey0_pt
        if crop_width_pt <= 0 or crop_height_pt <= 0:
            continue

        px0 = max(0, min(int(ex0_pt * px_per_pt_x), img_w - 1))
        py0 = max(0, min(int(ey0_pt * px_per_pt_y), img_h - 1))
        px1 = max(px0 + 1, min(int(round(ex1_pt * px_per_pt_x)), img_w))
        py1 = max(py0 + 1, min(int(round(ey1_pt * px_per_pt_y)), img_h))

        crop_img = page_img[py0:py1, px0:px1]

        crop_h_lines, crop_v_lines = extract_sobel_line_from_image(
            crop_img, crop_width_pt, crop_height_pt,
            min_length_ratio=min_length_ratio,
        )

        # Re-apply the full-page-equivalent absolute length threshold --
        # see "min_length_ratio correction" above. h_lines are
        # [x1, y1, x2, y2] with x2 > x1 (length along x); v_lines have
        # length along y.
        crop_h_lines = [ln for ln in crop_h_lines if (ln[2] - ln[0]) >= min_h_len_pt]
        crop_v_lines = [ln for ln in crop_v_lines if (ln[3] - ln[1]) >= min_v_len_pt]

        for (x1, y1, x2, y2) in crop_h_lines:
            all_h_lines.append((x1 + ex0_pt, y1 + ey0_pt, x2 + ex0_pt, y2 + ey0_pt))
        for (x1, y1, x2, y2) in crop_v_lines:
            all_v_lines.append((x1 + ex0_pt, y1 + ey0_pt, x2 + ex0_pt, y2 + ey0_pt))

    return all_h_lines, all_v_lines


def apply_detector_table_lines(page, data_dict, input_type, max_vec_line_num, feature_extractor):
    """
    Append 'dtable_vec_line' and/or 'dtable_img_line' bboxes to data_dict,
    whichever are present in input_type. No-op if neither is present.

    Detector-based counterparts of apply_table_vec_line/apply_table_img_line:
    instead of locating the table region with the vector/grid heuristic
    (find_table_grids), the table region comes from
    feature_extractor.get_layout_detections() ('table' class boxes from
    the image-based layout detection model). Only the underlying
    vec_line / img_line lines that overlap a detected table region
    (expanded by tolerance) are kept.

    Must be called AFTER extract_base_elements() (Step 1) OR after
    _build_image_only_data_dict() (image-only path), since it requires
    data_dict['box_type'] to already exist as a list. It reuses
    data_dict['vector_lines'] / data_dict['image_lines_raw'] when already
    cached by 'vec_line'/'table_vec_line' or 'img_line'/'table_img_line'.

    Cost note for 'dtable_img_line': since table_regions is already known
    before Sobel edge detection would run, running it on the WHOLE page
    when the caller only needs lines inside those (typically much
    smaller) regions would waste computation. This function avoids that:
    if table_regions is empty, Sobel is skipped entirely (the filtered
    result is guaranteed empty either way); otherwise, unless a full-page
    raw-line cache already exists from an earlier 'img_line'/
    'table_img_line' call, Sobel is run PER REGION on a cropped (+
    padded) sub-image and the results are offset back into page space
    (see _sobel_lines_in_regions()). This region-scoped result is cached
    under data_dict['dtable_region_lines_raw'] (a SEPARATE key from
    data_dict['image_lines_raw'], which means "full-page" to every other
    consumer) and shared with apply_detector_table_rect() if it runs in
    the same pipeline call.

    Args:
        page: PyMuPDF page object, or None for the image-only path
            (create_input_data_from_image()). When None, 'dtable_vec_line'
            is a no-op (no PDF vector paths available); 'dtable_img_line'
            works normally since it only needs data_dict['image'].
        data_dict: dict mutated in place.
        input_type: tuple of requested element types.
        max_vec_line_num: max lines to keep per orientation per block.
        feature_extractor: image-based layout detection model. predict()
            does NOT need to have been called by the caller -- this
            function calls it automatically if needed (see
            _ensure_predicted()), using data_dict['image'] as the
            page_img. If the caller already called predict() (e.g.
            BoxRFDGNN, to reuse its own rasterization) and marked it
            cached, that result is reused instead of re-predicting.

    Raises:
        ValueError: if 'dtable_vec_line' or 'dtable_img_line' is requested
            but feature_extractor is None.
    """
    wants_vec = 'dtable_vec_line' in input_type
    wants_img = 'dtable_img_line' in input_type
    if not wants_vec and not wants_img:
        return

    if feature_extractor is None:
        raise ValueError(
            "'dtable_vec_line'/'dtable_img_line' input_type requires a feature_extractor"
        )

    page_width = data_dict['page_width']
    page_height = data_dict['page_height']
    box_type = data_dict['box_type']

    _ensure_predicted(feature_extractor, data_dict['image'])

    # Table regions are shared between the two blocks -- compute once.
    table_regions = get_detected_table_regions(
        feature_extractor, data_dict['image'], page_width, page_height,
    )

    # Tolerance for the layout detector's box regression noise, expressed
    # in PIXELS (the detector's own coordinate space) rather than PDF
    # points, since detection-box imprecision is a property of the
    # detector's input resolution, not of the page's point size. A line
    # whose center lands within this many pixels of the detected table
    # boundary -- on either side, inward or outward -- is accepted.
    # Applies uniformly to both 'dtable_vec_line' and 'dtable_img_line':
    # the noise here comes from the detector's box regression, not from
    # the line source (vector vs. Sobel), so there is no reason for the
    # two to differ. Converted to point-space per axis using the same
    # scale as get_detected_table_regions(), since x/y pixel-per-point
    # ratios can differ if the rasterization isn't perfectly isotropic.
    # Re-check against real samples if the detector's input resolution
    # changes significantly.
    dtable_tol_px = 10
    scale_x, scale_y = _pixel_to_point_scale(data_dict['image'], page_width, page_height)
    tol_x = dtable_tol_px / scale_x
    tol_y = dtable_tol_px / scale_y

    if wants_vec:
        if 'vector_lines' in data_dict:
            h_lines, v_lines = data_dict['vector_lines']
        elif page is None:
            # Image-only path: no PDF vector lines available.
            # 'dtable_vec_line' is silently a no-op when page=None.
            h_lines, v_lines = [], []
        else:
            h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
            h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
            v_lines = merge_lines(v_lines, orientation='v', tolerance=3)
            data_dict['vector_lines'] = (h_lines, v_lines)

        h_members = [r for r in h_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]
        v_members = [r for r in v_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]

        h_members = sorted(h_members, key=_line_length, reverse=True)[:max_vec_line_num]
        v_members = sorted(v_members, key=_line_length, reverse=True)[:max_vec_line_num]

        _add_line_bboxes(data_dict, box_type, h_members, page_width, page_height, True, BOX_HLINE)
        _add_line_bboxes(data_dict, box_type, v_members, page_width, page_height, False, BOX_VLINE)

    if wants_img:
        # Same tolerance as apply_table_img_line's img_table_tol (2x the
        # vec-line default): image-based line detection has fragmentation
        # AND coordinate noise (Sobel edge localization) beyond what PDF
        # vector paths have, so merging needs a looser tolerance than
        # merge_lines' 3pt default. See apply_table_img_line's docstring.
        img_merge_tol = 6

        if 'image_lines_raw' in data_dict:
            # Full-page raw lines already computed by an earlier
            # 'img_line'/'table_img_line'/'img_table_rect_full' call in
            # this same pipeline call -- reuse directly. No benefit to
            # cropping since the (more expensive) full-page work is
            # already done and cached.
            raw_h_lines, raw_v_lines = data_dict['image_lines_raw']
        elif not table_regions:
            # No table detected anywhere on the page -- filtering by
            # table region would drop every line regardless of what
            # Sobel finds, so the result is guaranteed empty either way.
            # Skip Sobel processing entirely rather than run it on the
            # whole page for that guaranteed-empty result.
            raw_h_lines, raw_v_lines = [], []
        elif 'dtable_region_lines_raw' in data_dict:
            # Already computed for these same table_regions -- e.g. by
            # apply_detector_table_rect() running earlier in this same
            # pipeline call. Reuse.
            raw_h_lines, raw_v_lines = data_dict['dtable_region_lines_raw']
        else:
            # Crop-and-translate: run Sobel only on the detected table
            # regions (+ padding), not the whole page -- see
            # _sobel_lines_in_regions()'s own docstring for the padding
            # rationale and the coordinate-contract assumption this
            # relies on. Deliberately NOT cached into
            # data_dict['image_lines_raw'] -- that key means "full-page
            # raw lines" to every other consumer ('img_line',
            # 'table_img_line', 'img_table_rect_full', 'dimage_rect'); a
            # region-cropped result must never be mistaken for that.
            # Cached under its own key instead, shared with
            # apply_detector_table_rect() (same table_regions, same
            # relevance to it).
            raw_h_lines, raw_v_lines = _sobel_lines_in_regions(
                data_dict['image'], page_width, page_height, table_regions,
            )
            data_dict['dtable_region_lines_raw'] = (raw_h_lines, raw_v_lines)

        # NOTE: raw_h_lines/raw_v_lines are RAW Sobel detections -- typically
        # fragmented (a single visual line broken into several short
        # segments) and can contain near-duplicate parallel detections a
        # few pixels apart (e.g. both edges of a thick stroke). Unlike the
        # wants_vec branch above, these must be merge_lines()'d before use,
        # or _line_in_table_regions' region-overlap filter passes ALL of
        # these fragments/duplicates straight through (they're trivially
        # inside the detected table region), producing many duplicate/
        # near-duplicate line bboxes for what is visually one line.
        img_h_lines = merge_lines(
            [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_h_lines],
            orientation='h', tolerance=img_merge_tol,
        )
        img_v_lines = merge_lines(
            [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_v_lines],
            orientation='v', tolerance=img_merge_tol,
        )

        img_h_members = [r for r in img_h_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]
        img_v_members = [r for r in img_v_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]

        img_h_members = sorted(img_h_members, key=_line_length, reverse=True)[:max_vec_line_num]
        img_v_members = sorted(img_v_members, key=_line_length, reverse=True)[:max_vec_line_num]

        _add_line_bboxes(data_dict, box_type, img_h_members, page_width, page_height, True, BOX_HLINE)
        _add_line_bboxes(data_dict, box_type, img_v_members, page_width, page_height, False, BOX_VLINE)


def apply_detector_lsd_lines(page, data_dict, input_type, max_vec_line_num, feature_extractor):
    """
    Append 'dtable_lsd_line' bboxes to data_dict. No-op if 'dtable_lsd_line'
    is not in input_type.

    Detector-based counterpart of apply_detector_table_lines()'s
    'dtable_img_line' branch: the table region source is identical
    (feature_extractor.get_layout_detections() 'table' class boxes), but
    line extraction uses LSD (Line Segment Detector via cv2) instead of
    Sobel + RLE. LSD differences relevant here:
      - Returns complete line segments directly; no per-row/column RLE step.
      - A thick filled-rect line produces ONE segment instead of two
        near-duplicate edge responses, reducing merge_lines() dependence.
      - Handles faint or blurry lines robustly through gradient-field
        region growing rather than a global pixel threshold.
    merge_lines() is still applied after LSD for the same reason as
    dtable_img_line: dashed or interrupted lines produce multiple short
    segments that should be bridged before the region-overlap filter runs.

    No full/partial variant: unlike the heuristic table_img_line family,
    this function does not call find_table_grids() -- the table region comes
    directly from the layout detector, so structural full/partial validation
    has no meaning here. All lines found inside detected table regions are
    extracted regardless of border configuration.

    Cache key: 'dtable_lsd_region_lines_raw'. Intentionally separate from
    'dtable_region_lines_raw' (Sobel) and 'image_lines_raw' (full-page
    Sobel) -- LSD output has a different format and is NOT compatible with
    consumers that expect Sobel raw tuples. This key is not shared with
    apply_detector_table_rect(), which reads Sobel lines for its own
    tightening step; mixing the two would produce incorrect results.

    cv2 availability: cv2 is an optional dependency. If not installed,
    extract_lsd_line_from_image() raises ImportError with install
    instructions. The error is allowed to propagate -- the caller
    (create_input_data_from_page) is in the best position to decide
    whether to abort, skip, or fall back.

    Args:
        page: PyMuPDF page object, or None for the image-only path.
            Not used for line extraction (LSD is image-based), but
            kept for API symmetry with apply_detector_table_lines().
        data_dict: dict mutated in place.
        input_type: tuple of requested element types.
        max_vec_line_num: max line segments to keep per orientation.
        feature_extractor: image-based layout detection model. predict()
            does NOT need to have been called by the caller -- this
            function calls it automatically if needed via _ensure_predicted().

    Raises:
        ValueError: if 'dtable_lsd_line' is requested but feature_extractor
            is None.
        ImportError: propagated from extract_lsd_line_from_image() when
            cv2 is not installed.
    """
    if 'dtable_lsd_line' not in input_type:
        return

    if feature_extractor is None:
        raise ValueError(
            "'dtable_lsd_line' input_type requires a feature_extractor"
        )

    from .common_util import extract_lsd_line_from_image

    page_width  = data_dict['page_width']
    page_height = data_dict['page_height']
    box_type    = data_dict['box_type']

    _ensure_predicted(feature_extractor, data_dict['image'])

    table_regions = get_detected_table_regions(
        feature_extractor, data_dict['image'], page_width, page_height,
    )

    # Same detector-noise tolerance as apply_detector_table_lines: the
    # imprecision comes from the detector's box regression, not from
    # the line source, so the same 10px figure applies here.
    dtable_tol_px = 10
    scale_x, scale_y = _pixel_to_point_scale(data_dict['image'], page_width, page_height)
    tol_x = dtable_tol_px / scale_x
    tol_y = dtable_tol_px / scale_y

    # LSD merge tolerance: same 6pt as dtable_img_line (2x the vec-line
    # default). LSD already returns longer segments than Sobel RLE, so
    # in practice fewer fragments reach merge_lines(); the tolerance is
    # kept equal so dashed/interrupted lines are still bridged reliably.
    lsd_merge_tol = 6

    # LSD raw lines: cached under a dedicated key.
    # Unlike dtable_region_lines_raw (Sobel crop-and-translate output),
    # this is NOT reused by apply_detector_table_rect() -- that function
    # reads Sobel lines for its tightening step and must not receive LSD
    # output in their place.
    if not table_regions:
        # No table region detected -- result is guaranteed empty.
        # Skip LSD entirely (it would run on the whole image otherwise,
        # which is expensive and produces only lines that the region filter
        # would immediately discard).
        raw_h_lines, raw_v_lines = [], []
    elif 'dtable_lsd_region_lines_raw' in data_dict:
        raw_h_lines, raw_v_lines = data_dict['dtable_lsd_region_lines_raw']
    else:
        # Run LSD on each detected table region (cropped + padded), same
        # crop-and-translate strategy as _sobel_lines_in_regions() for
        # dtable_img_line. LSD is called per region to avoid running it on
        # the full page when only a small table area is relevant -- LSD's
        # cost scales with image area.
        raw_h_lines, raw_v_lines = _lsd_lines_in_regions(
            data_dict['image'], page_width, page_height, table_regions,
        )
        data_dict['dtable_lsd_region_lines_raw'] = (raw_h_lines, raw_v_lines)

    lsd_h_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_h_lines],
        orientation='h', tolerance=lsd_merge_tol,
    )
    lsd_v_lines = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_v_lines],
        orientation='v', tolerance=lsd_merge_tol,
    )

    lsd_h_members = [r for r in lsd_h_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]
    lsd_v_members = [r for r in lsd_v_lines if _line_in_table_regions(r, table_regions, tol_x, tol_y)]

    lsd_h_members = sorted(lsd_h_members, key=_line_length, reverse=True)[:max_vec_line_num]
    lsd_v_members = sorted(lsd_v_members, key=_line_length, reverse=True)[:max_vec_line_num]

    _add_line_bboxes(data_dict, box_type, lsd_h_members, page_width, page_height, True,  BOX_HLINE)
    _add_line_bboxes(data_dict, box_type, lsd_v_members, page_width, page_height, False, BOX_VLINE)


def _lsd_lines_in_regions(page_img, page_width, page_height, table_regions,
                           min_length_ratio=0.03):
    """
    Run extract_lsd_line_from_image() on each detected table region (cropped
    from page_img with padding) and translate results back into full-page PDF
    point space. Mirrors _sobel_lines_in_regions() in structure and contract.

    Rationale: same as _sobel_lines_in_regions() -- LSD on the full page when
    only small table areas are relevant is wasteful; crop-and-translate limits
    cost to total table area rather than full page area.

    Padding and coordinate contract are identical to _sobel_lines_in_regions():
    pad = max(20pt, 15% of region dimension) to avoid cutting border lines
    whose far end extends past the raw detected edge. The crop is passed to
    extract_lsd_line_from_image() with its own real-world dimensions so that
    returned coordinates are relative to the crop's top-left, which are then
    offset back into full-page space.

    min_length_ratio is forwarded to extract_lsd_line_from_image() and also
    used to compute a full-page-equivalent post-filter (same correction as
    _sobel_lines_in_regions() -- crop-local filtering is looser than
    full-page filtering for the same ratio value, so short fragments that
    full-page processing would have discarded can pass crop-local filtering).

    Args:
        page_img: (H, W, C) uint8 full-page raster image.
        page_width, page_height: PDF page size in points.
        table_regions: list of (x0, y0, x1, y1) in PDF point coordinates.
            Must be non-empty (callers should skip calling this otherwise).
        min_length_ratio: forwarded to extract_lsd_line_from_image(). Must
            match the default used by that function (0.03) for the post-filter
            to be equivalent to a full-page call.

    Returns:
        (raw_h_lines, raw_v_lines): lists of [x1, y1, x2, y2] tuples in
        full-page PDF point coordinates, same format as
        extract_lsd_line_from_image()'s return value, concatenated across
        all regions.
    """
    from .common_util import extract_lsd_line_from_image

    img_h, img_w = page_img.shape[:2]
    px_per_pt_x = img_w / page_width
    px_per_pt_y = img_h / page_height

    # Full-page-equivalent minimum line lengths in points -- same correction
    # logic as _sobel_lines_in_regions; see its docstring for the rationale.
    min_h_len_pt = min_length_ratio * page_width
    min_v_len_pt = min_length_ratio * page_height

    all_h_lines = []
    all_v_lines = []

    for (rx0, ry0, rx1, ry1) in table_regions:
        pad_x_pt = max(20.0, 0.15 * (rx1 - rx0))
        pad_y_pt = max(20.0, 0.15 * (ry1 - ry0))

        ex0_pt = max(0.0, rx0 - pad_x_pt)
        ey0_pt = max(0.0, ry0 - pad_y_pt)
        ex1_pt = min(page_width,  rx1 + pad_x_pt)
        ey1_pt = min(page_height, ry1 + pad_y_pt)

        crop_width_pt  = ex1_pt - ex0_pt
        crop_height_pt = ey1_pt - ey0_pt
        if crop_width_pt <= 0 or crop_height_pt <= 0:
            continue

        px0 = max(0, min(int(ex0_pt * px_per_pt_x), img_w - 1))
        py0 = max(0, min(int(ey0_pt * px_per_pt_y), img_h - 1))
        px1 = max(px0 + 1, min(int(round(ex1_pt * px_per_pt_x)), img_w))
        py1 = max(py0 + 1, min(int(round(ey1_pt * px_per_pt_y)), img_h))

        crop_img = page_img[py0:py1, px0:px1]

        crop_h_lines, crop_v_lines = extract_lsd_line_from_image(
            crop_img, crop_width_pt, crop_height_pt,
            min_length_ratio=min_length_ratio,
        )

        # Re-apply full-page-equivalent absolute length threshold --
        # see "min_length_ratio correction" in _sobel_lines_in_regions.
        crop_h_lines = [ln for ln in crop_h_lines if (ln[2] - ln[0]) >= min_h_len_pt]
        crop_v_lines = [ln for ln in crop_v_lines if (ln[3] - ln[1]) >= min_v_len_pt]

        for (x1, y1, x2, y2) in crop_h_lines:
            all_h_lines.append((x1 + ex0_pt, y1 + ey0_pt, x2 + ex0_pt, y2 + ey0_pt))
        for (x1, y1, x2, y2) in crop_v_lines:
            all_v_lines.append((x1 + ex0_pt, y1 + ey0_pt, x2 + ex0_pt, y2 + ey0_pt))

    return all_h_lines, all_v_lines


def apply_detector_table_rect(page, data_dict, input_type, feature_extractor):
    """
    Append 'dtable_rect' bboxes to data_dict: for each table region
    detected by feature_extractor.get_layout_detections() ('table'
    class), tighten the raw detector box to the union of whatever
    content is actually found inside it -- vector lines (PDF path or
    Sobel, whichever is already available) AND any bbox already present
    in data_dict['bboxes'] (text, image, other lines, ...) whose center
    falls inside the (tolerance-expanded) detected region. A detected
    region with nothing found inside it is treated as a false positive
    and is NOT added -- this is the main difference from
    'dtable_vec_line'/'dtable_img_line', which trust the detector's
    table classification unconditionally and only filter which
    already-known lines belong to it.

    No-op if 'dtable_rect' is not in input_type.

    Must be called AFTER extract_base_elements() (Step 1) OR after
    _build_image_only_data_dict() (image-only path), since it inspects
    data_dict['bboxes'] for already-extracted content and requires
    data_dict['box_type'] to already exist as a list. Works on both
    paths: page=None (image-only) simply skips the PDF-vector-line
    evidence source and relies on data_dict['bboxes'] (plus cached Sobel
    lines in data_dict['image_lines_raw'] and/or
    data_dict['dtable_region_lines_raw'], if 'img_line'/'table_img_line'/
    'dtable_img_line' already populated either in this same call -- see
    apply_detector_table_lines()'s docstring for the latter).

    Args:
        page: PyMuPDF page object, or None for the image-only path.
        data_dict: dict mutated in place.
        input_type: tuple of requested element types.
        feature_extractor: image-based layout detection model. predict()
            does NOT need to have been called by the caller -- this
            function calls it automatically if needed (see
            _ensure_predicted()), using data_dict['image'] as the
            page_img.

    Raises:
        ValueError: if 'dtable_rect' is requested but feature_extractor
            is None.
    """
    if 'dtable_rect' not in input_type:
        return

    if feature_extractor is None:
        raise ValueError("'dtable_rect' input_type requires a feature_extractor")

    page_width = data_dict['page_width']
    page_height = data_dict['page_height']
    box_type = data_dict['box_type']

    _ensure_predicted(feature_extractor, data_dict['image'])

    table_regions = get_detected_table_regions(
        feature_extractor, data_dict['image'], page_width, page_height,
    )

    # Same detector-noise tolerance as apply_detector_table_lines: a line
    # or bbox a few detector-pixels outside the raw box still counts as
    # "inside" the table region.
    dtable_tol_px = 10
    scale_x, scale_y = _pixel_to_point_scale(data_dict['image'], page_width, page_height)
    tol_x = dtable_tol_px / scale_x
    tol_y = dtable_tol_px / scale_y

    # Vector-line evidence: PDF vector lines if a page is available,
    # otherwise whatever Sobel lines are already cached from an earlier
    # 'img_line'/'table_img_line'/'dtable_img_line' call in this same
    # create_input_data_from_*() call. Neither is computed fresh here --
    # if nothing is cached and page is None, this evidence source is
    # simply empty and detection falls back to data_dict['bboxes'] alone.
    evidence_lines = []
    if 'vector_lines' in data_dict:
        h_lines, v_lines = data_dict['vector_lines']
        evidence_lines = list(h_lines) + list(v_lines)
    elif page is not None:
        h_lines, v_lines = get_vector_lines(page, omit_invisible=True)
        h_lines = merge_lines(h_lines, orientation='h', tolerance=3)
        v_lines = merge_lines(v_lines, orientation='v', tolerance=3)
        data_dict['vector_lines'] = (h_lines, v_lines)
        evidence_lines = list(h_lines) + list(v_lines)
    elif 'image_lines_raw' in data_dict:
        raw_h_lines, raw_v_lines = data_dict['image_lines_raw']
        evidence_lines = [
            pymupdf.Rect(x1, y1, x2, y2)
            for (x1, y1, x2, y2) in list(raw_h_lines) + list(raw_v_lines)
        ]

    # Additional evidence: region-cropped Sobel lines, if
    # apply_detector_table_lines()'s 'dtable_img_line' branch already
    # computed them for these same table_regions in this same call (see
    # _sobel_lines_in_regions()). This is a DIFFERENT, more targeted
    # source than data_dict['image_lines_raw'] above (full-page) and is
    # additive to it, not exclusive -- whichever ran, both are useful
    # here.
    if 'dtable_region_lines_raw' in data_dict:
        raw_h_lines, raw_v_lines = data_dict['dtable_region_lines_raw']
        evidence_lines = evidence_lines + [
            pymupdf.Rect(x1, y1, x2, y2)
            for (x1, y1, x2, y2) in list(raw_h_lines) + list(raw_v_lines)
        ]

    for region in table_regions:
        rx0, ry0, rx1, ry1 = region
        ex0, ey0 = rx0 - tol_x, ry0 - tol_y
        ex1, ey1 = rx1 + tol_x, ry1 + tol_y

        found_x0, found_y0, found_x1, found_y1 = [], [], [], []

        # Evidence 1: vector lines overlapping the expanded region.
        for r in evidence_lines:
            lx0, lx1 = sorted((r.x0, r.x1))
            ly0, ly1 = sorted((r.y0, r.y1))
            if lx1 < ex0 or lx0 > ex1 or ly1 < ey0 or ly0 > ey1:
                continue
            found_x0.append(lx0); found_x1.append(lx1)
            found_y0.append(ly0); found_y1.append(ly1)

        # Evidence 2: any bbox already extracted (text, image, other
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
            box_type.append(BOX_TABLE)


def _table_region_purity(logits, table_class_idx, region_pt, page_width, page_height):
    """
    Compute the mean softmax probability of table_class_idx inside a
    heuristic table region, using the model's segmentation logit map.

    The logit map (1, C, H_feat, W_feat) lives in model-input pixel space,
    which may differ from the page image resolution. Coordinates are
    converted from PDF point space to logit-map pixel space using the
    logit map's own spatial dimensions, NOT page_img dimensions, so the
    scale is computed here independently.

    Args:
        logits:          np.ndarray (1, C, H_feat, W_feat), raw pre-softmax
                         logits from feature_extractor.get_class_logits().
        table_class_idx: int, channel index for the 'table' class.
        region_pt:       (x0, y0, x1, y1) in PDF point coordinates,
                         as produced by find_table_grids/dedup_table_grids.
        page_width:      PDF page width in points.
        page_height:     PDF page height in points.

    Returns:
        float in [0, 1]: mean table-class softmax probability over the
        region's pixels in the logit map. Returns 0.0 if the crop is
        empty (degenerate region or rounding collapses it to zero size).
    """
    _, C, feat_h, feat_w = logits.shape

    # Scale: logit-map pixels per PDF point (independent of page_img size).
    scale_x = feat_w / page_width
    scale_y = feat_h / page_height

    rx0, ry0, rx1, ry1 = region_pt
    px0 = int(max(0, rx0 * scale_x))
    py0 = int(max(0, ry0 * scale_y))
    px1 = int(min(feat_w, rx1 * scale_x + 0.999))
    py1 = int(min(feat_h, ry1 * scale_y + 0.999))

    # Guard against degenerate crops (tiny table, coarse logit resolution).
    if px1 <= px0 or py1 <= py0:
        return 0.0

    # Softmax over the class axis on the cropped region only -- cheaper
    # than softmaxing the full map when only one region is needed.
    crop_logits = logits[0, :, py0:py1, px0:px1]   # (C, crop_h, crop_w)
    crop_probs  = softmax_numpy(crop_logits, axis=0) # (C, crop_h, crop_w)

    table_prob_crop = crop_probs[table_class_idx]    # (crop_h, crop_w)
    return float(table_prob_crop.mean())


def apply_stable_img_line(page, data_dict, input_type, max_vec_line_num,
                           feature_extractor, purity_threshold=0.5):
    """
    Append 'stable_img_line*' bboxes to data_dict.

    Hybrid strategy: heuristic table regions (same source as table_img_line)
    filtered by the model's segmentation map purity for the 'table' class,
    then Sobel img_lines extracted only for the regions that pass the filter.

    Motivation:
        table_img_line  -- heuristic Sobel + find_table_grids: good OOD
                           generalization, but produces false positives on
                           non-table vector patterns.
        dtable_img_line -- detector-based regions: near-zero false positives
                           on in-distribution data, but weaker OOD.
        stable_img_line -- takes the heuristic regions (OOD strength) and
                           confirms each with the model's per-pixel table
                           probability (precision), rejecting regions that
                           the model classifies as something other than table
                           with high confidence.

    Pipeline:
        A. Heuristic table regions:
           Reuse data_dict['table_img_grids'] if already populated by a
           prior table_img_line call in this same pipeline invocation;
           otherwise run Sobel + find_table_grids + dedup_table_grids
           directly (without adding those lines to data_dict -- that
           would duplicate bbox output). Sobel raw output is cached
           under data_dict['image_lines_raw'] exactly as apply_img_line
           does, so later consumers reuse it for free.
        B. Purity filter:
           For each heuristic region, compute the mean softmax probability
           of the 'table' class within the corresponding crop of the
           model's logit map (see _table_region_purity()). Regions whose
           purity < purity_threshold are discarded.
        C. Sobel img_line extraction:
           For passing regions, run _sobel_lines_in_regions() (cropped
           Sobel, same as dtable_img_line) to obtain lines, then filter
           with _line_in_table_regions() and append via _add_line_bboxes().
           Raw cropped lines are cached under
           data_dict['stable_region_lines_raw'] (separate from
           data_dict['image_lines_raw'] which always means full-page).

    Calling convention: Step 1.5 (after extract_base_elements() returns),
    same as apply_detector_table_lines(). data_dict['box_type'] must
    already exist as a list. PDF-path only -- 'stable_img_line*' is not
    in _IMAGE_ONLY_VALID_INPUT_TYPES (the heuristic Sobel path is wired
    through extract_base_elements Step 1 for the image-only path, and
    combining that with the Step 1.5 purity filter here would require
    additional refactoring not yet justified by use cases).

    Args:
        page:              PyMuPDF page object. Required for Sobel raster
                           (page.get_pixmap()) when no cached lines exist.
        data_dict:         dict mutated in place. Must have 'box_type' as
                           a list (i.e. called after extract_base_elements).
        input_type:        tuple of requested element types.
        max_vec_line_num:  max lines per orientation to keep per call.
        feature_extractor: ImageFeatureExtractorV2 instance. predict() is
                           called automatically if not already cached (see
                           _ensure_predicted()). get_class_logits() is used
                           to read the segmentation map after prediction.
        purity_threshold:  float in [0, 1]. Minimum mean table-class
                           softmax probability for a heuristic region to
                           be accepted. Default 0.5. Exposed as
                           'stable_img_line_purity' in the options dict
                           (resolved by the caller in
                           create_input_data_from_page()).

    Raises:
        ValueError: if any 'stable_img_line*' value is in input_type but
                    feature_extractor is None.
    """
    active_types = {k for k in STABLE_IMG_LINE_TABLE_TYPES if k in input_type}
    if not active_types:
        return

    if feature_extractor is None:
        raise ValueError(
            "'stable_img_line'/'stable_img_line_full'/'stable_img_line_partial' "
            "input_type requires a feature_extractor"
        )

    page_width  = data_dict['page_width']
    page_height = data_dict['page_height']
    box_type    = data_dict['box_type']

    # Step A: ensure model inference is done.
    _ensure_predicted(feature_extractor, data_dict['image'])

    # Step A (continued): heuristic table regions via find_table_grids.
    # Reuse already-merged Sobel lines if any table_img_line* call in this
    # same pipeline invocation already ran Sobel (image_lines_raw) or merged
    # them (image_lines). Otherwise run Sobel now and cache both.
    img_table_tol = 6   # same as apply_table_img_line -- see its docstring

    if 'image_lines_raw' in data_dict:
        raw_img_h_lines, raw_img_v_lines = data_dict['image_lines_raw']
    else:
        from .common_util import extract_sobel_line_from_image
        raw_img_h_lines, raw_img_v_lines = extract_sobel_line_from_image(
            data_dict['image'], page_width, page_height,
        )
        data_dict['image_lines_raw'] = (raw_img_h_lines, raw_img_v_lines)

    if 'image_lines' in data_dict:
        img_h_lines, img_v_lines = data_dict['image_lines']
    else:
        img_h_lines = merge_lines(
            [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_h_lines],
            orientation='h', tolerance=img_table_tol,
        )
        img_v_lines = merge_lines(
            [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_img_v_lines],
            orientation='v', tolerance=img_table_tol,
        )
        # Cache for dtable_img_line / apply_img_line reuse within same call.
        data_dict['image_lines'] = (img_h_lines, img_v_lines)

    # Step A (continued): find and dedup heuristic table grids.
    # Process each requested stable_img_line* variant independently (same
    # pattern as apply_table_img_line -- results accumulate in
    # 'stable_img_grids' rather than overwriting each other).
    # Reuse 'table_img_grids' if a table_img_line* call already populated
    # it AND the table_type matches. In practice callers request either
    # stable_img_line alone or alongside non-table input_types; if
    # table_img_line* was ALSO requested the exclusivity check in
    # create_input_data_from_page() would already have raised ValueError
    # before reaching here. So 'table_img_grids' reuse is a defensive
    # optimization for future callers that bypass the exclusivity check.
    all_passing_regions = []   # (x0, y0, x1, y1) in PDF points, post-filter

    logits = feature_extractor.get_class_logits()   # (1, C, H_feat, W_feat)

    for key, table_type in STABLE_IMG_LINE_TABLE_TYPES.items():
        if key not in input_type:
            continue

        # Try to reuse heuristic grids already computed this call.
        cached_grids = data_dict.get('table_img_grids')
        if cached_grids is not None:
            # table_img_grids may contain grids from a prior call with a
            # different table_type. Filter to those whose open_side matches.
            if table_type == TABLE_TYPE_ALL:
                candidate_grids = cached_grids
            elif table_type == TABLE_TYPE_FULL:
                candidate_grids = [g for g in cached_grids if g['open_side'] is None]
            else:  # TABLE_TYPE_PARTIAL
                candidate_grids = [g for g in cached_grids if g['open_side'] is not None]
        else:
            candidate_grids = find_table_grids(
                img_h_lines, img_v_lines,
                border_tol=img_table_tol, divider_tol=img_table_tol,
                table_type=table_type,
            )
            candidate_grids = dedup_table_grids(candidate_grids, tol=img_table_tol)
            data_dict.setdefault('stable_img_grids', []).extend(candidate_grids)

        # Step B: purity filter -- discard regions the model disagrees with.
        for grid in candidate_grids:
            purity = _table_region_purity(
                logits, _TABLE_CLASS_IDX, grid['bbox'], page_width, page_height,
            )
            if purity >= purity_threshold:
                all_passing_regions.append(grid['bbox'])

    if not all_passing_regions:
        return

    # Deduplicate passing regions (same region can appear from multiple
    # table_type variants if e.g. both 'full' and 'all' were somehow both
    # in active_types -- shouldn't happen given exclusivity check, but
    # defensive dedup costs nothing).
    seen = set()
    unique_regions = []
    for r in all_passing_regions:
        key = tuple(int(round(c)) for c in r)
        if key not in seen:
            seen.add(key)
            unique_regions.append(r)

    # Step C: Sobel line extraction scoped to passing regions.
    # Reuse full-page raw lines if already cached (image_lines_raw); otherwise
    # crop-and-translate (same strategy as dtable_img_line). Cached under
    # 'stable_region_lines_raw' -- a distinct key from 'image_lines_raw'
    # (full-page) and 'dtable_region_lines_raw' (detector regions) so other
    # consumers are never misled about the source.
    if 'image_lines_raw' in data_dict:
        raw_h, raw_v = data_dict['image_lines_raw']
    elif 'stable_region_lines_raw' in data_dict:
        raw_h, raw_v = data_dict['stable_region_lines_raw']
    else:
        raw_h, raw_v = _sobel_lines_in_regions(
            data_dict['image'], page_width, page_height, unique_regions,
        )
        data_dict['stable_region_lines_raw'] = (raw_h, raw_v)

    # Tolerance for _line_in_table_regions: heuristic region boundaries are
    # exact (drawn lines, not a detector estimate), so a smaller tolerance
    # than dtable's 10px detector-noise tolerance is appropriate. We use the
    # same img_table_tol (6pt) as the grid-matching tolerances above.
    tol_x = img_table_tol
    tol_y = img_table_tol

    img_h_lines_merged = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_h],
        orientation='h', tolerance=img_table_tol,
    )
    img_v_lines_merged = merge_lines(
        [pymupdf.Rect(x1, y1, x2, y2) for (x1, y1, x2, y2) in raw_v],
        orientation='v', tolerance=img_table_tol,
    )

    img_h_members = [
        r for r in img_h_lines_merged
        if _line_in_table_regions(r, unique_regions, tol_x, tol_y)
    ]
    img_v_members = [
        r for r in img_v_lines_merged
        if _line_in_table_regions(r, unique_regions, tol_x, tol_y)
    ]

    img_h_members = sorted(img_h_members, key=_line_length, reverse=True)[:max_vec_line_num]
    img_v_members = sorted(img_v_members, key=_line_length, reverse=True)[:max_vec_line_num]

    _add_line_bboxes(data_dict, box_type, img_h_members, page_width, page_height, True,  BOX_HLINE)
    _add_line_bboxes(data_dict, box_type, img_v_members, page_width, page_height, False, BOX_VLINE)
