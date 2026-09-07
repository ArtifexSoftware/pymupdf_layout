"""
TableGridExtractorV5.py

Loads exported GridModelV5 and ConnClassifier ONNX models and predicts
table grid structure (line positions + cell connectivity) from a cropped
table image.

Post-processing pipeline
------------------------
1. Resize input image to fixed GridModelV5 input size.
2. Run GridModelV5 ONNX inference.
   Outputs: h_on_logit, h_offset, v_on_logit, v_offset, feature_map,
            h_heatmap, v_heatmap, h_score, v_score
3. Decode anchor outputs using h_score > h_on_threshold (and v_score >
   v_on_threshold). Score-aware 1D NMS applied to candidates.
   line_pos = anchor + offset * anchor_step
4. Convert normalized line positions to input image pixel coordinates.
5. Optionally filter empty lines and snap to bbox gaps.
6. Run ConnClassifier ONNX inference on cell features.

Key difference from V2
-----------------------
V2 applies h_on_threshold / v_on_threshold directly to sigmoid(on_logit).
V5 instead thresholds score = sigmoid(on_logit) * heatmap, where heatmap
is an independent tent-on-gap confidence map (TableGridDatasetV5
_make_h_heatmap / _make_v_heatmap) gating false positives in
cell-interior regions. The per-axis threshold parameters
(h_on_threshold, v_on_threshold) are retained from V2, now applied to
score instead of sigmoid(on_logit).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .table_grid_types import GridPrediction, CellInfo


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_ensemble_score(
    on_logit: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
    pre_filter_ratio: float = 0.8,
) -> np.ndarray:
    """
    Compute ensemble score from on_logit and heatmap.

    Only anchors where sigmoid(on_logit) > threshold * pre_filter_ratio
    are considered. Their score is the average of sigmoid(on_logit) and
    heatmap. All other anchors get score 0.

    Parameters
    ----------
    on_logit          : (max_n,) float32, raw logit output from model
    heatmap           : (max_n,) float32, tent-on-gap heatmap in [0, 1]
    threshold         : final threshold (h_on_threshold / v_on_threshold)
    pre_filter_ratio  : fraction of threshold used as pre-filter gate

    Returns
    -------
    score : (max_n,) float32, ensemble score in [0, 1]
    """
    sigmoid = 1.0 / (1.0 + np.exp(-on_logit.astype(np.float64))).astype(np.float32)
    score   = np.zeros_like(sigmoid)
    mask    = sigmoid > (threshold * pre_filter_ratio)
    score[mask] = (sigmoid[mask] + heatmap[mask]) / 2.0
    return score


def _decode_anchors_score(
    score: np.ndarray,
    offsets: np.ndarray,
    nms_min_dist: float = 0.0,
    score_thresh: float = 0.5,
) -> tuple:
    """
    Decode anchor-based line predictions from a pre-computed score map.

    Parameters
    ----------
    score        : (max_n,) float32, ensemble score in [0, 1].
                   score[i] > score_thresh means anchor i predicts a line.
    offsets      : (max_n,) float32, anchor-step-normalized offsets.
    nms_min_dist : minimum distance between kept lines (normalized [0, 1]).
    score_thresh : threshold for candidate selection.

    Returns
    -------
    positions : (K,) float32, sorted normalized line positions
    scores    : (max_n,) float32, score values (used as confidence for NMS)
    """
    max_n       = len(score)
    anchors     = np.linspace(0.0, 1.0, max_n, dtype=np.float32)
    anchor_step = 1.0 / (max_n - 1) if max_n > 1 else 1.0

    mask       = score > score_thresh
    candidates = (anchors + offsets * anchor_step)[mask]
    scores_sel = score[mask]

    if nms_min_dist > 0.0 and len(candidates) > 0:
        order      = np.argsort(-scores_sel)
        suppressed = np.zeros(len(candidates), dtype=bool)
        keep       = []
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            for j in range(len(candidates)):
                if not suppressed[j] and j != i:
                    if abs(float(candidates[j]) - float(candidates[i])) < nms_min_dist:
                        suppressed[j] = True
        keep_arr  = np.array(keep, dtype=np.int32)
        positions = np.sort(candidates[keep_arr])
    else:
        positions = np.sort(candidates)

    return positions, score



def _extract_cell_features(feature_map, h_lines_norm, v_lines_norm):
    """Pool feature_map regions defined by detected line positions."""
    C, H, W = feature_map.shape
    N = len(h_lines_norm)
    M = len(v_lines_norm)

    if N < 2 or M < 2:
        return np.zeros((max(N - 1, 1), max(M - 1, 1), C), dtype=np.float32)

    ys = np.clip((h_lines_norm * H).astype(np.int32), 0, H)
    xs = np.clip((v_lines_norm * W).astype(np.int32), 0, W)

    cell_feat = np.zeros((N - 1, M - 1, C), dtype=np.float32)
    for i in range(N - 1):
        y1 = ys[i]
        y2 = max(y1 + 1, ys[i + 1])
        y2 = min(y2, H)
        if y1 >= H:
            continue
        for j in range(M - 1):
            x1 = xs[j]
            x2 = max(x1 + 1, xs[j + 1])
            x2 = min(x2, W)
            if x1 >= W:
                continue
            region = feature_map[:, y1:y2, x1:x2]
            if region.size == 0:
                continue
            cell_feat[i, j] = region.mean(axis=(1, 2))

    return cell_feat


# ---------------------------------------------------------------------------
# Base extractor: shared post-processing logic
# ---------------------------------------------------------------------------

class _TableGridExtractorBase:
    """
    Base class for V5 and V6 extractors.
    Holds all post-processing logic that is identical across versions:
    _filter_empty_lines, _snap_lines_to_bbox_gaps, _post_process_grid, predict.
    Subclasses implement predict_grid and _load_grid_session.
    """

    def __init__(
        self,
        grid_onnx_path,
        conn_onnx_path,
        conn_threshold: float = 0.2,
        nms_min_dist: float = 0.02,
        h_on_threshold: float = 0.25,
        v_on_threshold: float = 0.2,
        filter_empty_lines: bool = True,
        snap_to_bbox_gaps: bool = False,
        providers=None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.grid_onnx_path    = Path(grid_onnx_path)
        self.conn_onnx_path    = Path(conn_onnx_path) if conn_onnx_path else None
        self.conn_threshold    = conn_threshold
        self.nms_min_dist      = nms_min_dist
        self.h_on_threshold    = h_on_threshold
        self.v_on_threshold    = v_on_threshold
        self.filter_empty_lines = filter_empty_lines
        self.snap_to_bbox_gaps  = snap_to_bbox_gaps
        self._providers         = providers

        self._grid_sess = ort.InferenceSession(
            str(self.grid_onnx_path), providers=providers
        )
        self._conn_sess = (
            ort.InferenceSession(str(self.conn_onnx_path), providers=providers)
            if self.conn_onnx_path and self.conn_onnx_path.exists() else None
        )

        # Resolve output indices by name for robustness across export versions
        output_names           = [o.name for o in self._grid_sess.get_outputs()]
        self._idx_h_on_logit   = output_names.index("h_on_logit")
        self._idx_h_offset     = output_names.index("h_offset")
        self._idx_v_on_logit   = output_names.index("v_on_logit")
        self._idx_v_offset     = output_names.index("v_offset")
        self._idx_feature_map  = output_names.index("feature_map")
        self._idx_h_heatmap    = output_names.index("h_heatmap")
        self._idx_v_heatmap    = output_names.index("v_heatmap")

        # Derive max_h and max_v from a dummy forward pass
        self._init_shape()

    def _init_shape(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Post-processing helpers (identical to V2)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_empty_lines(lines, centers, image_size):
        if not lines:
            return lines

        current = sorted(lines)
        changed = True

        while changed:
            changed = False
            if len(current) < 2:
                break
            result = [current[0]]
            for i in range(1, len(current)):
                lo_y, lo_score, lo_cls = result[-1]
                hi_y, hi_score, hi_cls = current[i]
                has_center = any(lo_y <= c <= hi_y for c in centers)
                if not has_center:
                    if hi_score >= lo_score:
                        result[-1] = (hi_y, hi_score, hi_cls)
                    changed = True
                else:
                    result.append(current[i])
            current = result

        return current

    @staticmethod
    def _snap_lines_to_bbox_gaps(h_lines, bboxes_crop, snap_threshold=0.0):
        if len(bboxes_crop) == 0 or len(h_lines) == 0:
            return h_lines

        bottoms = np.sort(bboxes_crop[:, 3])
        tops    = np.sort(bboxes_crop[:, 1])

        gap_lines = []
        for bot in bottoms:
            candidates = tops[tops > bot]
            if len(candidates) > 0:
                gap_lines.append((bot + float(candidates[0])) / 2.0)

        if not gap_lines:
            return h_lines

        gap_arr  = np.array(
            sorted(set(round(g, 4) for g in gap_lines)), dtype=np.float32
        )
        used_gaps = set()
        result    = []
        for y in h_lines:
            crosses = np.any((bboxes_crop[:, 1] < y) & (y < bboxes_crop[:, 3]))
            if crosses:
                dists = np.abs(gap_arr - y)
                order = np.argsort(dists)
                snapped = False
                for idx in order:
                    nearest = float(gap_arr[idx])
                    dist    = float(dists[idx])
                    if nearest in used_gaps:
                        continue
                    if snap_threshold <= 0.0 or dist <= snap_threshold:
                        result.append(nearest)
                        used_gaps.add(nearest)
                        snapped = True
                        break
                if not snapped:
                    result.append(y)
            else:
                result.append(y)
        return result

    def _post_process_grid(self, bboxes_page, grid, span_threshold):
        if grid.h_lines:
            last_h    = sorted(grid.h_lines)[-1]
            row_edges = [0.0] + sorted(grid.h_lines) + [max(last_h * 2, last_h + 1)]
        else:
            row_edges = [0.0, 1.0]

        if grid.v_lines:
            last_v    = sorted(grid.v_lines)[-1]
            col_edges = [0.0] + sorted(grid.v_lines) + [max(last_v * 2, last_v + 1)]
        else:
            col_edges = [0.0, 1.0]

        def find_cell_idx(pos, edges):
            for i in range(len(edges) - 1):
                if edges[i] <= pos < edges[i + 1]:
                    return i
            return max(0, len(edges) - 2)

        results = []
        for i, bbox in enumerate(bboxes_page):
            x0 = float(bbox[0])
            y0 = float(bbox[1])
            x1 = float(bbox[2])
            y1 = float(bbox[3])
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0

            base_row  = find_cell_idx(cy, row_edges)
            base_col  = find_cell_idx(cx, col_edges)
            row_start = base_row
            row_end   = base_row + 1
            col_start = base_col
            col_end   = base_col + 1

            r = base_row
            while r > 0:
                h = row_edges[r] - row_edges[r - 1]
                if h > 0 and (row_edges[r] - y0) / h > span_threshold:
                    row_start = r - 1; r -= 1
                else:
                    break
            r = base_row
            while r < len(row_edges) - 2:
                h = row_edges[r + 1] - row_edges[r]
                if h > 0 and (y1 - row_edges[r + 1]) / h > span_threshold:
                    row_end = r + 2; r += 1
                else:
                    break
            c = base_col
            while c > 0:
                w = col_edges[c] - col_edges[c - 1]
                if w > 0 and (col_edges[c] - x0) / w > span_threshold:
                    col_start = c - 1; c -= 1
                else:
                    break
            c = base_col
            while c < len(col_edges) - 2:
                w = col_edges[c + 1] - col_edges[c]
                if w > 0 and (x1 - col_edges[c + 1]) / w > span_threshold:
                    col_end = c + 2; c += 1
                else:
                    break

            results.append(CellInfo(
                bbox_idx=i, row_start=row_start, row_end=row_end,
                col_start=col_start, col_end=col_end,
                row=base_row, col=base_col,
            ))

        return results

    # ------------------------------------------------------------------
    # Shared predict logic
    # ------------------------------------------------------------------

    def predict(
        self,
        image_bgr: np.ndarray,
        bboxes=None,
        texts=None,
        span_threshold: float = 0.1,
    ) -> tuple:
        """
        Predict grid boundaries and optionally assign bboxes to grid cells.

        Parameters
        ----------
        image_bgr      : cropped table BGR image (any size)
        bboxes         : (N, 4) array-like of [x0, y0, x1, y1] in crop space.
                         If None or empty, only grid prediction is performed.
        texts          : list of text strings aligned with bboxes.
        span_threshold : fractional overlap to trigger span expansion.

        Returns
        -------
        (GridPrediction, list[CellInfo])
        """
        grid = self.predict_grid(image_bgr)

        if bboxes is None or len(bboxes) == 0:
            return grid, []

        bboxes_arr  = np.asarray(bboxes, dtype=np.float32)
        bboxes_crop = bboxes_arr
        crop_h      = float(image_bgr.shape[0])
        crop_w      = float(image_bgr.shape[1])

        cx_list = sorted((float(b[0]) + float(b[2])) / 2.0 for b in bboxes_arr)
        cy_list = sorted((float(b[1]) + float(b[3])) / 2.0 for b in bboxes_arr)

        # Build (pos, score, cls) tuples for filter
        orig_h    = float(image_bgr.shape[0])
        orig_w    = float(image_bgr.shape[1])
        max_h     = len(grid.h_on_prob)
        max_v     = len(grid.v_on_prob)
        anchors_h = np.linspace(0.0, 1.0, max_h, dtype=np.float32)
        anchors_v = np.linspace(0.0, 1.0, max_v, dtype=np.float32)

        h_cls_arr = grid.h_cls if grid.h_cls is not None else np.ones(
            len(grid.h_lines), dtype=np.int32
        )
        h_tuples = []
        for y, c in zip(grid.h_lines, h_cls_arr.tolist()):
            y_norm = y / orig_h
            idx    = int(np.argmin(np.abs(anchors_h - y_norm)))
            score  = float(grid.h_on_prob[idx])
            h_tuples.append((y, score, c))

        v_tuples = []
        for x in grid.v_lines:
            x_norm = x / orig_w
            idx    = int(np.argmin(np.abs(anchors_v - x_norm)))
            score  = float(grid.v_on_prob[idx])
            v_tuples.append((x, score, 0))

        if self.filter_empty_lines:
            h_tuples = self._filter_empty_lines(h_tuples, cy_list, crop_h)
            v_tuples = self._filter_empty_lines(v_tuples, cx_list, crop_w)
            while h_tuples and not any(
                0.0 <= c <= h_tuples[0][0] for c in cy_list
            ):
                h_tuples = h_tuples[1:]
            while v_tuples and not any(
                0.0 <= c <= v_tuples[0][0] for c in cx_list
            ):
                v_tuples = v_tuples[1:]
            while h_tuples and not any(
                h_tuples[-1][0] <= c <= crop_h for c in cy_list
            ):
                h_tuples = h_tuples[:-1]
            while v_tuples and not any(
                v_tuples[-1][0] <= c <= crop_w for c in cx_list
            ):
                v_tuples = v_tuples[:-1]

        filtered_h     = [y for y, _, _ in h_tuples]
        filtered_v     = [x for x, _, _ in v_tuples]
        filtered_h_cls = np.array([c for _, _, c in h_tuples], dtype=np.int32)

        if self.snap_to_bbox_gaps:
            snapped    = self._snap_lines_to_bbox_gaps(filtered_h, bboxes_crop)
            h_tuples   = [(sy, sc, c) for sy, (_, sc, c) in zip(snapped, h_tuples)]
            filtered_h = [y for y, _, _ in h_tuples]

        filtered_h_norm = np.array(
            [y / orig_h for y in filtered_h], dtype=np.float32
        )
        filtered_v_norm = np.array(
            [x / orig_w for x in filtered_v], dtype=np.float32
        )

        grid = GridPrediction(
            h_lines=sorted(filtered_h),
            v_lines=sorted(filtered_v),
            h_on_prob=grid.h_on_prob,
            v_on_prob=grid.v_on_prob,
            h_lines_norm=filtered_h_norm,
            v_lines_norm=filtered_v_norm,
            h_cls=filtered_h_cls,
            connectivity=grid.connectivity,
        )

        cells = self._post_process_grid(
            bboxes_page=bboxes_arr,
            grid=grid,
            span_threshold=span_threshold,
        )

        if texts is not None:
            for cell in cells:
                if 0 <= cell.bbox_idx < len(texts):
                    cell.text = texts[cell.bbox_idx]

        return grid, cells

    # ------------------------------------------------------------------
    # Shared ConnClassifier inference
    # ------------------------------------------------------------------

    def _run_conn(self, feature_map, h_lines_norm, v_lines_norm):
        """
        Run ConnClassifier ONNX inference and return connectivity map.
        Returns None when conn_sess is unavailable or inputs are degenerate.
        """
        if self._conn_sess is None:
            return None
        if len(h_lines_norm) < 2 or len(v_lines_norm) < 2:
            return None

        cell_feat = _extract_cell_features(feature_map, h_lines_norm, v_lines_norm)
        if not np.isfinite(cell_feat).all():
            return None

        cell_inp    = cell_feat.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        conn_out    = self._conn_sess.run(None, {"cell_features": cell_inp})
        conn_logits = conn_out[0][0]
        conn_prob   = 1.0 / (1.0 + np.exp(-conn_logits.astype(np.float64)))
        return conn_prob.transpose(1, 2, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# V5 extractor: fixed input size, per-anchor threshold
# ---------------------------------------------------------------------------

class TableGridExtractorV5(_TableGridExtractorBase):
    """
    ONNX-based table grid extractor for GridModelV5.

    Thresholds score = sigmoid(on_logit) * heatmap against h_on_threshold /
    v_on_threshold (same per-axis threshold parameters as V2, now applied
    to score instead of sigmoid(on_logit)). heatmap is an independent
    tent-on-gap confidence map that gates false positives in cell-interior
    regions.
    Input image is resized to the fixed size baked into the ONNX model.
    """

    def __init__(
        self,
        grid_onnx_path,
        conn_onnx_path=None,
        conn_threshold: float = 0.2,
        nms_min_dist: float = 0.02,
        h_on_threshold: float = 0.25,
        v_on_threshold: float = 0.2,
        filter_empty_lines: bool = True,
        snap_to_bbox_gaps: bool = False,
        providers=None,
    ):
        super().__init__(
            grid_onnx_path=grid_onnx_path,
            conn_onnx_path=conn_onnx_path,
            conn_threshold=conn_threshold,
            nms_min_dist=nms_min_dist,
            h_on_threshold=h_on_threshold,
            v_on_threshold=v_on_threshold,
            filter_empty_lines=filter_empty_lines,
            snap_to_bbox_gaps=snap_to_bbox_gaps,
            providers=providers,
        )

    def _init_shape(self):
        """Derive fixed input size and max_h / max_v from a dummy forward pass."""
        inp            = self._grid_sess.get_inputs()[0]
        self._input_h  = int(inp.shape[2])
        self._input_w  = int(inp.shape[3])

        dummy   = np.zeros(
            (1, 3, self._input_h, self._input_w), dtype=np.float32
        )
        outputs      = self._grid_sess.run(None, {"image": dummy})
        self._max_h  = int(outputs[self._idx_h_heatmap].shape[1])
        self._max_v  = int(outputs[self._idx_v_heatmap].shape[1])

    def predict_grid(self, image_bgr: np.ndarray) -> GridPrediction:
        orig_h, orig_w = image_bgr.shape[:2]

        # Resize to fixed model input size
        img_resized = cv2.resize(
            image_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )
        img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        inp     = img_rgb.transpose(2, 0, 1)[np.newaxis]

        outputs     = self._grid_sess.run(None, {"image": inp})
        h_on_logit  = outputs[self._idx_h_on_logit][0]  # (max_h,)
        h_offset    = outputs[self._idx_h_offset][0]     # (max_h,)
        v_on_logit  = outputs[self._idx_v_on_logit][0]  # (max_v,)
        v_offset    = outputs[self._idx_v_offset][0]     # (max_v,)
        feature_map = outputs[self._idx_feature_map][0]  # (C, H', W')
        h_heatmap   = outputs[self._idx_h_heatmap][0]   # (max_h,)
        v_heatmap   = outputs[self._idx_v_heatmap][0]   # (max_v,)

        h_score = _compute_ensemble_score(h_on_logit, h_heatmap, self.h_on_threshold)
        v_score = _compute_ensemble_score(v_on_logit, v_heatmap, self.v_on_threshold)

        h_lines_norm, h_on_prob = _decode_anchors_score(
            h_score, h_offset, self.nms_min_dist, self.h_on_threshold
        )
        v_lines_norm, v_on_prob = _decode_anchors_score(
            v_score, v_offset, self.nms_min_dist, self.v_on_threshold
        )

        h_cls = np.ones(len(h_lines_norm), dtype=np.int32)

        h_lines = [float(y) * orig_h for y in h_lines_norm]
        v_lines = [float(x) * orig_w for x in v_lines_norm]

        connectivity = self._run_conn(feature_map, h_lines_norm, v_lines_norm)

        return GridPrediction(
            h_lines=sorted(h_lines),
            v_lines=sorted(v_lines),
            h_on_prob=h_on_prob,
            v_on_prob=v_on_prob,
            h_lines_norm=h_lines_norm,
            v_lines_norm=v_lines_norm,
            h_cls=h_cls,
            connectivity=connectivity,
        )
