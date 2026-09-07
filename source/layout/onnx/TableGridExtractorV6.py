"""
TableGridExtractorV6.py

Loads exported GridModelV6 and ConnClassifier ONNX models and predicts
table grid structure (line positions + cell connectivity) from a cropped
table image.

Post-processing pipeline
------------------------
1. Extract candidate rows from cell bboxes using extract_candidate_grid.
2. Count candidate rows and map to a resize stage using row_num_boundaries:
       n_rows <= row_num_boundaries[0] -> input_sizes[0]
       n_rows <= row_num_boundaries[1] -> input_sizes[1]
       ...
       n_rows >  row_num_boundaries[-1] -> input_sizes[-1]
   Fallback (no bboxes or no candidates): input_sizes[0] (= V5 behavior).
3. Run GridModelV6 ONNX inference (dynamic H/W input).
   Outputs: h_on_logit, h_offset, v_on_logit, v_offset, feature_map,
            h_heatmap, v_heatmap
4. Compute ensemble score: candidates where sigmoid(on_logit) >
   threshold * 0.8; score = (sigmoid + heatmap) / 2.
5. Decode using score > h_on_threshold / v_on_threshold + score-aware NMS.
6. Convert normalized line positions to original image pixel coordinates.
7. Optionally filter empty lines and snap to bbox gaps.
8. Run ConnClassifier ONNX inference on cell features.

Key difference from V5
-----------------------
V5 resizes all images to a fixed input size baked into the ONNX model.
V6 resizes each image to one of N fixed sizes (input_sizes) selected by
the estimated row density via row_num_boundaries. The ONNX model is
exported with dynamic height and width axes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import cv2
import numpy as np
import onnxruntime as ort

from .TableGridExtractorV5 import (
    _TableGridExtractorBase,
    _compute_ensemble_score,
    _decode_anchors_score,
    _extract_cell_features,
)
from .TableGridExtractorV1A import extract_candidate_grid
from .table_grid_types import GridPrediction, CellInfo


class TableGridExtractorV6(_TableGridExtractorBase):
    """
    ONNX-based table grid extractor for GridModelV6.

    Resizes the input image to one of N fixed sizes (input_sizes) selected
    by estimated row density (row_num_boundaries), enabling dense tables
    to be processed at higher resolution.

    Parameters
    ----------
    grid_onnx_path    : path to exported GridModelV6 .onnx file
    conn_onnx_path    : path to exported ConnClassifier .onnx file (optional)
    input_sizes       : list of N [H, W] target sizes, ascending order.
                        input_sizes[0] is also the fallback when no bboxes
                        are provided (matches V5 behavior).
    row_num_boundaries: list of N-1 integers. Candidate row count is compared
                        sequentially; the first boundary exceeded determines
                        the stage index.
    grid_margin_px    : alignment tolerance for candidate grid extraction
    conn_threshold    : confidence threshold for connectivity prediction
    nms_min_dist        : minimum distance between kept lines in pixels
                        (converted to normalized space per sample at inference)
    h_on_threshold    : threshold applied to h ensemble score
    v_on_threshold    : threshold applied to v ensemble score
    filter_empty_lines: remove lines with no bbox center between them
    snap_to_bbox_gaps : snap h_lines to nearest inter-row gap center
    providers         : ONNX Runtime execution providers
    """

    def __init__(
        self,
        grid_onnx_path,
        conn_onnx_path=None,
        input_sizes: list | None = None,
        row_num_boundaries: list | None = None,
        grid_margin_px: float = 5.0,
        conn_threshold: float = 0.2,
        nms_min_dist: float = 8.0,
        h_on_threshold: float = 0.25,
        v_on_threshold: float = 0.2,
        filter_empty_lines: bool = True,
        snap_to_bbox_gaps: bool = False,
        providers=None,
    ):
        if input_sizes is not None and row_num_boundaries is not None:
            assert len(input_sizes) == len(row_num_boundaries) + 1, (
                f"input_sizes has {len(input_sizes)} entries but "
                f"row_num_boundaries has {len(row_num_boundaries)} entries; "
                f"expected len(input_sizes) == len(row_num_boundaries) + 1"
            )

        self.input_sizes        = input_sizes
        self.row_num_boundaries = row_num_boundaries
        self.grid_margin_px     = grid_margin_px
        self.nms_min_px         = nms_min_dist  # pixels; converted per-sample in predict_grid

        super().__init__(
            grid_onnx_path=grid_onnx_path,
            conn_onnx_path=conn_onnx_path,
            conn_threshold=conn_threshold,
            nms_min_dist=0.0,  # not used in V6; per-sample pixel conversion applied
            h_on_threshold=h_on_threshold,
            v_on_threshold=v_on_threshold,
            filter_empty_lines=filter_empty_lines,
            snap_to_bbox_gaps=snap_to_bbox_gaps,
            providers=providers,
        )

    @classmethod
    def from_model_yaml(
        cls,
        grid_onnx_path,
        model_yaml_path=None,
        conn_onnx_path=None,
        input_sizes: list | None = None,
        row_num_boundaries: list | None = None,
        grid_margin_px: float = 5.0,
        conn_threshold: float = 0.2,
        nms_min_px: float = 8.0,
        h_on_threshold: float = 0.25,
        v_on_threshold: float = 0.2,
        filter_empty_lines: bool = True,
        snap_to_bbox_gaps: bool = False,
        providers=None,
    ):
        """
        Construct a TableGridExtractorV6, optionally reading input_sizes
        and row_num_boundaries from a model.yaml saved during V6 training,
        ensuring train/inference consistency.

        When model_yaml_path is None, parameters fall back to the
        explicitly passed arguments (or their defaults). Values present
        in model.yaml take precedence when model_yaml_path is given.
        """
        if model_yaml_path is not None:
            with open(model_yaml_path, "rb") as f:
                model_cfg = yaml.safe_load(f)["model"]
            input_sizes        = model_cfg.get("input_sizes",        input_sizes)
            row_num_boundaries = model_cfg.get("row_num_boundaries", row_num_boundaries)

        return cls(
            grid_onnx_path=grid_onnx_path,
            conn_onnx_path=conn_onnx_path,
            input_sizes=input_sizes,
            row_num_boundaries=row_num_boundaries,
            grid_margin_px=grid_margin_px,
            conn_threshold=conn_threshold,
            nms_min_dist=nms_min_px,
            h_on_threshold=h_on_threshold,
            v_on_threshold=v_on_threshold,
            filter_empty_lines=filter_empty_lines,
            snap_to_bbox_gaps=snap_to_bbox_gaps,
            providers=providers,
        )

    def _init_shape(self):
        """
        Derive max_h and max_v from a small dummy forward pass.
        Input size is dynamic in V6, so we use a minimal dummy.
        """
        dummy        = np.zeros((1, 3, 64, 64), dtype=np.float32)
        outputs      = self._grid_sess.run(None, {"image": dummy})
        self._max_h  = int(outputs[self._idx_h_heatmap].shape[1])
        self._max_v  = int(outputs[self._idx_v_heatmap].shape[1])

    # ------------------------------------------------------------------
    # N-stage resize helpers
    # ------------------------------------------------------------------

    def _compute_stage(self, cand_h_lines: list) -> int:
        """
        Determine resize stage index from candidate row count.

        Compared sequentially against row_num_boundaries; the first
        boundary not exceeded determines the stage index.
        Falls back to stage 0 when no candidates or boundaries not set.
        """
        if self.row_num_boundaries is None or self.input_sizes is None:
            return 0
        n_rows = len(cand_h_lines)
        for i, boundary in enumerate(self.row_num_boundaries):
            if n_rows <= boundary:
                return i
        return len(self.input_sizes) - 1

    def _stage_to_size(self, stage: int) -> list | None:
        """
        Map stage index to target [H, W] from input_sizes.
        Returns None when input_sizes is not configured.
        """
        if self.input_sizes is None:
            return None
        stage = max(0, min(stage, len(self.input_sizes) - 1))
        return list(self.input_sizes[stage])

    # ------------------------------------------------------------------
    # Grid prediction
    # ------------------------------------------------------------------

    def predict_grid(
        self,
        image_bgr: np.ndarray,
        bboxes: list | None = None,
    ) -> GridPrediction:
        """
        Predict grid lines from image using N-stage resolution selection.

        Parameters
        ----------
        image_bgr : cropped table BGR image (any size)
        bboxes    : list of [x1, y1, x2, y2] cell bbox pixel coordinates
                    used to estimate row density for stage selection.
                    When None or empty, falls back to stage 0 (input_sizes[0]).

        Returns
        -------
        GridPrediction with line positions in original image pixel coordinates.
        """
        orig_h, orig_w = image_bgr.shape[:2]

        if bboxes and len(bboxes) > 0:
            cand_h, _, _ = extract_candidate_grid(
                bboxes, orig_h, orig_w, self.grid_margin_px
            )
            stage = self._compute_stage(cand_h)
        else:
            # No bboxes: fallback to stage 0 (input_sizes[0] = V5 behavior)
            stage = 0

        target_size = self._stage_to_size(stage)
        if target_size is not None:
            new_h, new_w = target_size
        else:
            new_h, new_w = orig_h, orig_w

        img_resized = cv2.resize(
            image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        inp     = img_rgb.transpose(2, 0, 1)[np.newaxis]

        # Convert pixel NMS distance to normalized [0,1] space for this image size
        nms_dist_h = self.nms_min_px / float(new_h) if new_h > 0 else 0.012
        nms_dist_v = self.nms_min_px / float(new_w) if new_w > 0 else 0.012

        outputs     = self._grid_sess.run(None, {"image": inp})
        h_on_logit  = outputs[self._idx_h_on_logit][0]
        h_offset    = outputs[self._idx_h_offset][0]
        v_on_logit  = outputs[self._idx_v_on_logit][0]
        v_offset    = outputs[self._idx_v_offset][0]
        feature_map = outputs[self._idx_feature_map][0]
        h_heatmap   = outputs[self._idx_h_heatmap][0]
        v_heatmap   = outputs[self._idx_v_heatmap][0]

        h_score = _compute_ensemble_score(h_on_logit, h_heatmap, self.h_on_threshold)
        v_score = _compute_ensemble_score(v_on_logit, v_heatmap, self.v_on_threshold)

        h_lines_norm, h_on_prob = _decode_anchors_score(
            h_score, h_offset, nms_dist_h, self.h_on_threshold
        )
        v_lines_norm, v_on_prob = _decode_anchors_score(
            v_score, v_offset, nms_dist_v, self.v_on_threshold
        )

        h_cls = np.ones(len(h_lines_norm), dtype=np.int32)

        # Convert normalized positions back to original image pixel space
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

    def predict(
        self,
        image_bgr: np.ndarray,
        bboxes=None,
        texts=None,
        span_threshold: float = 0.1,
    ) -> tuple:
        """
        Predict grid boundaries and optionally assign bboxes to grid cells.

        Overrides base predict to pass bboxes into predict_grid for
        stage selection before delegating to base post-processing.
        """
        bboxes_list = list(bboxes) if bboxes is not None else None
        grid = self.predict_grid(image_bgr, bboxes=bboxes_list)

        if bboxes is None or len(bboxes) == 0:
            return grid, []

        bboxes_arr  = np.asarray(bboxes, dtype=np.float32)
        bboxes_crop = bboxes_arr
        crop_h      = float(image_bgr.shape[0])
        crop_w      = float(image_bgr.shape[1])

        cx_list = sorted((float(b[0]) + float(b[2])) / 2.0 for b in bboxes_arr)
        cy_list = sorted((float(b[1]) + float(b[3])) / 2.0 for b in bboxes_arr)

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
            h_tuples   = [
                (sy, sc, c) for sy, (_, sc, c) in zip(snapped, h_tuples)
            ]
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
