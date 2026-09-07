import os
import math
import numpy as np

from collections import Counter
from typing import Optional, List, Dict, Tuple


def softmax_numpy(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Numerically stable softmax along the given axis.

    Args:
        x: input array of any shape.
        axis: axis along which softmax is computed.

    Returns:
        np.ndarray of the same shape as x, summing to 1.0 along axis.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.

    Args:
        x: input array of any shape.

    Returns:
        np.ndarray of the same shape as x, values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def get_boxes_transform(bboxes_list):
    """
    Normalizes bounding box coordinates (x1, y1, x2, y2) and
    calculates center coordinates and dimensions.

    Args:
        bboxes_list (list of list): A list of bounding boxes,
                                     e.g., [[x1, y1, x2, y2], ...].
                                     It is assumed that the coordinates are
                                     in the format (top-left, bottom-right).

    Returns:
        numpy.ndarray: A NumPy array where each row contains the normalized
                       (x1, y1, x2, y2), center coordinates (ctr_x, ctr_y),
                       width (box_w), and height (box_h).
                       Shape: [num_boxes, 8]
    """
    if not bboxes_list:
        return np.empty((0, 8), dtype=np.float32) # Return empty array for no boxes

    # Convert the input list of lists to a NumPy array
    # Assume cell_box_data holds the entire bounding box feature map.
    bbox_data = np.array(bboxes_list, dtype=np.float32)

    # 1. Calculate the minimum coordinates of the entire set of bounding boxes
    # Find the minimum x from (x1, x2) across all boxes.
    min_x = bbox_data[:, [0, 2]].min()
    # Find the minimum y from (y1, y2) across all boxes.
    min_y = bbox_data[:, [1, 3]].min()

    # 2. Calculate the width and height of the entire bounding box region
    # Calculate the width based on the maximum x from (x1, x2) and min_x.
    cell_box_w = bbox_data[:, [0, 2]].max() - min_x
    # Calculate the height based on the maximum y from (y1, y2) and min_y.
    cell_box_h = bbox_data[:, [1, 3]].max() - min_y

    # Handle ZeroDivisionError by clamping width/height to a minimum value (like epsilon)
    # If width/height is 0, it would cause division by zero. This prevents that.
    cell_box_w = np.maximum(cell_box_w, 1e-6) # Equivalent to torch.clamp(value, min=1e-6)
    cell_box_h = np.maximum(cell_box_h, 1e-6) # Equivalent to torch.clamp(value, min=1e-6)

    # 3. Normalize bounding box coordinates
    # Normalize x coordinates (x1, x2) by subtracting min_x and dividing by cell_box_w.
    bbox_data[:, [0, 2]] = (bbox_data[:, [0, 2]] - min_x) / cell_box_w
    # Normalize y coordinates (y1, y2) by subtracting min_y and dividing by cell_box_h.
    bbox_data[:, [1, 3]] = (bbox_data[:, [1, 3]] - min_y) / cell_box_h

    # Renaming for clarity (consistent with original logic)
    boxes = bbox_data

    # 4. Calculate center coordinates and dimensions (width, height)
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    ctr_x = (boxes[:, 2] + boxes[:, 0]) / 2
    ctr_y = (boxes[:, 3] + boxes[:, 1]) / 2

    # 5. Stack all features into a single array
    # Output format: [normalized x1, normalized y1, normalized x2, normalized y2, ctr_x, ctr_y, box_w, box_h]
    boxes_feat = np.stack((boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
                           ctr_x, ctr_y, box_w, box_h), axis=1)

    return boxes_feat


def get_edge_by_directional_nn(bbox: np.ndarray, dist_threshold: int, vertical_gap: float = 0.3) -> tuple[
    list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    """
    NOTE (direction-label fix): this function's internal direction names used
    to be swapped -- what was called 'left_dis_matrix' actually finds each
    box's nearest neighbor to its RIGHT, 'right_dis_matrix' actually finds
    the neighbor to its LEFT, 'up_dis_matrix' actually finds the neighbor
    BELOW, and 'down_dis_matrix' actually finds the neighbor ABOVE (verified
    empirically: e.g. for 3 boxes in a row, the matrix named "left" matched
    each box to the one on its right). All of the variable names and
    direction_edges dict keys below have been corrected to match what is
    actually computed. The underlying distance formulas and filter
    conditions are UNCHANGED from before this fix, so the final `edge_index`
    returned by this function (the only thing build_edge_index actually
    uses -- it discards direction_edges) is bit-for-bit identical to before.
    This matters for reproducibility with any model already trained on
    graphs built by the previous version of this function.

    Only the *direction_edges* dict (previously unused by any caller in this
    codebase) is affected in practice: code that starts relying on
    direction_edges['left'] meaning "genuinely left of this node" would have
    silently gotten right-neighbors before this fix.

    Open question, NOT resolved here (see down_neighbor_y_dis_matrix below):
    the y-only-distance matrix historically named 'down_y' reused what was
    then called 'up_dis_matrix's filter condition. Since that matrix's TRUE
    behavior is "find the neighbor below", down_y ends up being a
    y-distance-only variant that's largely redundant with the (corrected)
    down_neighbor_dis_matrix, just with an added x-alignment constraint and
    coarser (y-only, not x+y) ranking. Whether this was an intentional
    distinct signal or a copy-paste of the wrong direction's filter is
    unconfirmed -- flagged for future review rather than changed, since its
    actual behavior (like everything else here) is left untouched by this fix.
    """
    bbox_num = len(bbox)
    eye_matrix = np.eye(bbox_num, dtype=bool)  # eye_matrix를 명시적으로 bool 타입으로 생성

    # 중심 좌표 및 높이 계산
    center_y_box = (bbox[:, 1] + bbox[:, 3]) / 2
    center_x_box = (bbox[:, 0] + bbox[:, 2]) / 2
    bbox_height = bbox[:, 3] - bbox[:, 1]

    # 각 bbox의 중심이 다른 bbox의 수직 범위(vertical_gap 확장) 밖에 있는지 플래그
    y_center_flag = np.logical_or(
        center_y_box.repeat(bbox_num).reshape((bbox_num, -1)).T < (bbox[:, 1] - bbox_height * vertical_gap).repeat(
            bbox_num).reshape((bbox_num, -1)),
        center_y_box.repeat(bbox_num).reshape((bbox_num, -1)).T > (bbox[:, 3] + bbox_height * vertical_gap).repeat(
            bbox_num).reshape((bbox_num, -1))
    )

    # 각 bbox의 중심이 다른 bbox의 수평 범위 밖에 있는지 플래그
    x_center_flag = np.logical_or(
        center_x_box.repeat(bbox_num).reshape((bbox_num, -1)).T < bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)),
        center_x_box.repeat(bbox_num).reshape((bbox_num, -1)).T > bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1))
    )

    # 유클리드 거리 행렬 계산 (이 부분도 원본과 동일하게 repeat().reshape().T 패턴 적용)
    hor_x_dis_matrix = (bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 2].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2
    hor_y_dis_matrix = (bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 1].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2

    ver_x_dis_matrix = (bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 0].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2
    ver_y_dis_matrix = (bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T - bbox[:, 3].repeat(bbox_num).reshape(
        (bbox_num, -1))) ** 2

    # --- get_filtered_distance_matrix 함수 ---
    def get_filtered_distance_matrix(base_distance_matrix, *conditions):
        filtered_matrix = np.copy(base_distance_matrix)

        all_conditions = []
        for i, cond in enumerate(conditions):
            if cond is not None:
                all_conditions.append(cond)
            else:
                all_conditions.append(np.full(base_distance_matrix.shape, False, dtype=bool))

        if not all_conditions:
            filter_mask = np.full(base_distance_matrix.shape, False, dtype=bool)
        else:
            filter_mask = np.logical_or.reduce(all_conditions)

        filtered_matrix[filter_mask] = math.inf
        return filtered_matrix

    # 각 방향별 거리 행렬 및 필터링 (변수명은 실제로 계산되는 방향에 맞게 수정됨;
    # 수식/필터 자체는 수정 전과 완전히 동일 -- 위 NOTE 참고)
    right_neighbor_dis_matrix = get_filtered_distance_matrix(
        hor_x_dis_matrix + hor_y_dis_matrix,
        y_center_flag,
        eye_matrix,
        bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 2].repeat(bbox_num).reshape((bbox_num, -1))
    )

    left_neighbor_dis_matrix = get_filtered_distance_matrix(
        hor_x_dis_matrix + hor_y_dis_matrix,
        y_center_flag,
        eye_matrix,
        bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1)).T >= bbox[:, 0].repeat(bbox_num).reshape((bbox_num, -1))
    )

    down_neighbor_dis_matrix = get_filtered_distance_matrix(
        ver_x_dis_matrix + ver_y_dis_matrix,
        eye_matrix,
        bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1))
    )

    up_neighbor_dis_matrix = get_filtered_distance_matrix(
        ver_x_dis_matrix + ver_y_dis_matrix,
        eye_matrix,
        bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1)).T >= bbox[:, 1].repeat(bbox_num).reshape((bbox_num, -1))
    )

    # y-only-distance variant -- reuses down_neighbor_dis_matrix's filter
    # condition (see open question in the function docstring above).
    down_neighbor_y_dis_matrix = np.copy(ver_y_dis_matrix)
    _down_neighbor_y_filter_condition = np.logical_or(
        eye_matrix,
        bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1)).T <= bbox[:, 3].repeat(bbox_num).reshape((bbox_num, -1))
    )

    down_neighbor_y_filter_mask = np.logical_or(_down_neighbor_y_filter_condition, x_center_flag)
    down_neighbor_y_dis_matrix[down_neighbor_y_filter_mask] = math.inf

    # 각 방향에서 가장 가까운 이웃 찾기
    def get_nearest_indices(distance_matrix, matrix_name="Unnamed"):
        min_distances = np.min(distance_matrix, axis=1)
        min_indices = np.argmin(distance_matrix, axis=1)
        valid_mask = min_distances != math.inf

        indices = [(int(i), int(min_indices[i])) for i in range(bbox_num) if valid_mask[i]]
        return indices

    # 각 방향별 원본 엣지들을 저장할 딕셔너리
    direction_edges = {}

    left_edges = get_nearest_indices(left_neighbor_dis_matrix, "left_neighbor_dis_matrix")
    direction_edges['left'] = left_edges

    right_edges = get_nearest_indices(right_neighbor_dis_matrix, "right_neighbor_dis_matrix")
    direction_edges['right'] = right_edges

    up_edges = get_nearest_indices(up_neighbor_dis_matrix, "up_neighbor_dis_matrix")
    direction_edges['up'] = up_edges

    down_edges = get_nearest_indices(down_neighbor_dis_matrix, "down_neighbor_dis_matrix")
    direction_edges['down'] = down_edges

    down_y_edges = get_nearest_indices(down_neighbor_y_dis_matrix, "down_neighbor_y_dis_matrix")
    direction_edges['down_y'] = down_y_edges

    # 수직 방향으로 두 번째 가까운 이웃 찾기
    down_neighbor_dis_matrix_copy = np.copy(down_neighbor_dis_matrix)
    up_neighbor_dis_matrix_copy = np.copy(up_neighbor_dis_matrix)

    # 첫 번째 이웃을 무한대로 설정하여 두 번째 이웃 찾기
    for i in range(bbox_num):
        if np.min(down_neighbor_dis_matrix_copy[i]) != math.inf:
            down_neighbor_dis_matrix_copy[i, np.argmin(down_neighbor_dis_matrix_copy[i])] = math.inf
        if np.min(up_neighbor_dis_matrix_copy[i]) != math.inf:
            up_neighbor_dis_matrix_copy[i, np.argmin(up_neighbor_dis_matrix_copy[i])] = math.inf

    second_down_edges = get_nearest_indices(down_neighbor_dis_matrix_copy, "second_down_neighbor_dis_matrix")
    direction_edges['second_down'] = second_down_edges

    second_up_edges = get_nearest_indices(up_neighbor_dis_matrix_copy, "second_up_neighbor_dis_matrix")
    direction_edges['second_up'] = second_up_edges

    # 모든 원본 엣지들을 하나의 리스트로 합치기 (중복 제거 및 정규화 전)
    all_raw_edges = []
    for edges in direction_edges.values():
        all_raw_edges.extend(edges)

    # 중복 제거 및 정규화 (항상 (작은 인덱스, 큰 인덱스) 형태로 유지)
    ori_edge_index_list = list(
        set([(min(item), max(item)) for item in all_raw_edges])
    )
    # dist_threshold 필터링
    final_edge_list = [
        item for item in ori_edge_index_list if abs(item[1] - item[0]) < dist_threshold
    ]
    final_edge_list_sorted = sorted(final_edge_list, key=lambda x: (x[0], x[1]))

    return final_edge_list, direction_edges



def get_edge_by_alignment(bbox: np.ndarray, dist_threshold: float = 0.001, center_limit: float = 10000) -> list[tuple[int, int]]:
    """
    Vectorized (broadcasting-based, matching get_edge_by_directional_nn's
    style). Previously a pure-Python O(N^2) double loop with a per-pair
    `any(...)` generator call -- the only unvectorized edge-construction
    function in this module. Not currently used by build_edge_index's
    default '4D' path (only by the experimental '4D+AL' path, not currently
    active), but fixed for consistency and to remove the latent bottleneck.

    Returns the exact same edge set, in the exact same order, as the
    previous implementation (both iterate implicitly in row-major (i, j)
    order over i < j pairs).
    """
    bbox_num = len(bbox)
    if bbox_num < 2:
        return []

    lefts = bbox[:, 0]
    tops = bbox[:, 1]
    rights = bbox[:, 2]
    bottoms = bbox[:, 3]

    centers_x = (lefts + rights) / 2
    centers_y = (tops + bottoms) / 2

    # Pairwise center distance
    center_dist = np.sqrt(
        (centers_x[:, None] - centers_x[None, :]) ** 2
        + (centers_y[:, None] - centers_y[None, :]) ** 2
    )

    hor_aligned = np.zeros((bbox_num, bbox_num), dtype=bool)
    for h in (lefts, rights, centers_x):
        hor_aligned |= np.abs(h[:, None] - h[None, :]) <= dist_threshold

    ver_aligned = np.zeros((bbox_num, bbox_num), dtype=bool)
    for v in (tops, bottoms, centers_y):
        ver_aligned |= np.abs(v[:, None] - v[None, :]) <= dist_threshold

    valid = (center_dist <= center_limit) & (hor_aligned | ver_aligned)
    valid = np.triu(valid, k=1)  # i < j only, matching the original loop's range(i+1, bbox_num)

    i_idx, j_idx = np.nonzero(valid)
    return list(zip(i_idx.tolist(), j_idx.tolist()))


def get_edge_by_knn(bboxes: np.ndarray, k: int) -> np.ndarray:
    """
    Computes a k-nearest neighbor graph based on the center points of bounding boxes,
    returning the result as a numpy array.

    Args:
        bboxes (np.ndarray): A numpy array of shape (N, 4) where each row is a bounding box
                             in the format [x1, y1, x2, y2].
        k (int): The number of nearest neighbors for each node.

    Returns:
        np.ndarray: A numpy array of shape (E, 2) representing the directed edge_index.
    """
    if bboxes.shape[0] < 2:
        return np.empty((0, 2), dtype=np.int64)

    # Calculate center points of each bounding box
    center_points = np.zeros((bboxes.shape[0], 2))
    center_points[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    center_points[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

    # Initialize edge list
    edges = []

    # Compute distances and find k-nearest neighbors for each point
    for i in range(len(center_points)):
        # Calculate Euclidean distances from point i to all points
        distances = np.sqrt(np.sum((center_points - center_points[i]) ** 2, axis=1))
        # Set distance to self as infinity to exclude it
        distances[i] = np.inf
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        # Add directed edges (i -> j)
        for j in nearest_indices:
            edges.append([i, j])

    # Convert edges to numpy array
    edge_index = np.array(edges, dtype=np.int64)

    return edge_index

def get_edge_transform_bbox(bboxes: np.ndarray, edge_index: list):
    """
    Vectorized edge feature computation using NumPy.

    Args:
        bboxes (np.ndarray): (N, 4) array in [x_min, y_min, x_max, y_max] format.
        edge_index (list): (E, 2) list of (src_idx, dst_idx) tuples representing edges.

    Returns:
        np.ndarray: (E, 18) array of edge features.
    """
    if len(edge_index) == 0:
        return np.empty((0, 18), dtype=np.float32)

    # Convert edge list to NumPy array
    edge_index_np = np.array(edge_index)

    # Extract source and target bounding boxes
    S = bboxes[edge_index_np[:, 0]]
    O = bboxes[edge_index_np[:, 1]]

    epsilon = 1e-6  # Minimum safe value to avoid division by zero or log(0)

    # Compute width and height of source and target boxes
    sw = S[:, 2] - S[:, 0]
    sh = S[:, 3] - S[:, 1]
    ow = O[:, 2] - O[:, 0]
    oh = O[:, 3] - O[:, 1]

    # Compute enclosing box coordinates
    R_x1 = np.minimum(S[:, 0], O[:, 0])
    R_y1 = np.minimum(S[:, 1], O[:, 1])
    R_x2 = np.maximum(S[:, 2], O[:, 2])
    R_y2 = np.maximum(S[:, 3], O[:, 3])
    R = np.stack([R_x1, R_y1, R_x2, R_y2], axis=1)

    rw = R[:, 2] - R[:, 0]
    rh = R[:, 3] - R[:, 1]

    # Replace invalid values with epsilon to ensure safe division and log
    safe_sw = np.where(sw <= 0, epsilon, sw)
    safe_sh = np.where(sh <= 0, epsilon, sh)
    safe_ow = np.where(ow <= 0, epsilon, ow)
    safe_oh = np.where(oh <= 0, epsilon, oh)
    safe_rw = np.where(rw <= 0, epsilon, rw)
    safe_rh = np.where(rh <= 0, epsilon, rh)

    features = []

    # Features 1?6: relative positions and scale between S and O
    features.append((S[:, 0] - O[:, 0]) / safe_sw)
    features.append((S[:, 1] - O[:, 1]) / safe_sh)
    features.append((O[:, 0] - S[:, 0]) / safe_ow)
    features.append((O[:, 1] - S[:, 1]) / safe_oh)
    features.append(np.log(safe_sw / safe_ow))
    features.append(np.log(safe_sh / safe_oh))

    # Features 7?12: relative to enclosing box R from S
    features.append((S[:, 0] - R[:, 0]) / safe_sw)
    features.append((S[:, 1] - R[:, 1]) / safe_sh)
    features.append((R[:, 0] - S[:, 0]) / safe_rw)
    features.append((R[:, 1] - S[:, 1]) / safe_rh)
    features.append(np.log(safe_sw / safe_rw))
    features.append(np.log(safe_sh / safe_rh))

    # Features 13?18: relative to enclosing box R from O
    features.append((O[:, 0] - R[:, 0]) / safe_ow)
    features.append((O[:, 1] - R[:, 1]) / safe_oh)
    features.append((R[:, 0] - O[:, 0]) / safe_rw)
    features.append((R[:, 1] - O[:, 1]) / safe_rh)
    features.append(np.log(safe_ow / safe_rw))
    features.append(np.log(safe_oh / safe_rh))

    # Stack features and sanitize invalid values
    edge_attr = np.stack(features, axis=1)
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1e5, neginf=-1e5)
    return edge_attr

def get_edge_transform_bbox_add_alignment(bboxes: np.ndarray, edge_index: list):
    """
    Vectorized edge feature computation using NumPy,
    Args:
        bboxes (np.ndarray): (N, 4) NumPy array in [x_min, y_min, x_max, y_max] format.
        edge_index (list): (E, 2) list of tuples, where each tuple (src_idx, dst_idx)

    Returns:
        np.ndarray: (E, 24) array of edge features (18 relation + 6 alignment).
    """
    if len(edge_index) == 0:
        # 24, not 18: this function returns 18 relation features + 6
        # alignment features (unlike get_edge_transform_bbox, which only has
        # the 18). The empty-case guard previously returned (0, 18), copied
        # from that other function without updating the column count for
        # this one's actual 24-column output -- a real shape mismatch bug
        # whenever edge_index happened to be empty (e.g. a page with a
        # single isolated element and no other nodes to connect to).
        return np.empty((0, 24), dtype=np.float32)

    # Convert edge_index list of tuples to numpy array
    # edge_index_np will be (E, 2)
    edge_index_np = np.array(edge_index)

    # Extract source (S) and object (O) bboxes using numpy indexing
    # These will be (E, 4) numpy arrays: [x_min, y_min, x_max, y_max]
    S = bboxes[edge_index_np[:, 0]]
    O = bboxes[edge_index_np[:, 1]]

    delta = 1e-10  # Prevent division by zero
    # Calculate widths and heights for S and O
    sw = S[:, 2] - S[:, 0]  # S_width
    sh = S[:, 3] - S[:, 1]  # S_height
    ow = O[:, 2] - O[:, 0]  # O_width
    oh = O[:, 3] - O[:, 1]  # O_height

    # Calculate enclosing bounding box (R)
    # R: [out_x_min, out_y_min, out_x_max, out_y_max]
    R_x1 = np.minimum(S[:, 0], O[:, 0])
    R_y1 = np.minimum(S[:, 1], O[:, 1])
    R_x2 = np.maximum(S[:, 2], O[:, 2])
    R_y2 = np.maximum(S[:, 3], O[:, 3])
    R = np.stack([R_x1, R_y1, R_x2, R_y2], axis=1)  # (E, 4)

    rw = R[:, 2] - R[:, 0]  # R_width
    rh = R[:, 3] - R[:, 1]  # R_height

    # Construct the 18 features exactly as in get_relation_feature
    features = []

    # Features 1-6: Relative to S and O
    features.append((S[:, 0] - O[:, 0]) / (sw + delta))  # 1. (x1_min - x2_min) / width1
    features.append(
        (S[:, 1] - O[:, 1]) / (sh + delta)
    )  # 2. (y1_min - y2_min) / height1
    features.append((O[:, 0] - S[:, 0]) / (ow + delta))  # 3. (x2_min - x1_min) / width2
    features.append(
        (O[:, 1] - S[:, 1]) / (oh + delta)
    )  # 4. (y2_min - y1_min) / height2
    features.append(np.log(sw / (ow + delta)))  # 5. log(width1 / width2)
    features.append(np.log(sh / (oh + delta)))  # 6. log(height1 / height2)

    # Features 7-12: Relative to S and R (out_box)
    features.append(
        (S[:, 0] - R[:, 0]) / (sw + delta)
    )  # 7. (x1_min - out_x_min) / width1
    features.append(
        (S[:, 1] - R[:, 1]) / (sh + delta)
    )  # 8. (y1_min - out_y_min) / height1
    features.append(
        (R[:, 0] - S[:, 0]) / (rw + delta)
    )  # 9. (out_x_min - x1_min) / out_width
    features.append(
        (R[:, 1] - S[:, 1]) / (rh + delta)
    )  # 10. (out_y_min - y1_min) / out_height
    features.append(np.log(sw / (rw + delta)))  # 11. log(width1 / out_width)
    features.append(np.log(sh / (rh + delta)))  # 12. log(height1 / out_height)

    # Features 13-18: Relative to O and R (out_box)
    features.append(
        (O[:, 0] - R[:, 0]) / (ow + delta)
    )  # 13. (x2_min - out_x_min) / width2
    features.append(
        (O[:, 1] - R[:, 1]) / (oh + delta)
    )  # 14. (y2_min - out_y_min) / height2
    features.append(
        (R[:, 0] - O[:, 0]) / (rw + delta)
    )  # 15. (out_x_min - x2_min) / out_width
    features.append(
        (R[:, 1] - O[:, 1]) / (rh + delta)
    )  # 16. (out_y_min - y2_min) / out_height
    features.append(np.log(ow / (rw + delta)))  # 17. log(width2 / out_width)
    features.append(np.log(oh / (rh + delta)))  # 18. log(height2 / out_height)

    # Alignment checks between S and O
    is_left_aligned = (np.abs(S[:, 0] - O[:, 0]) <= 1e-5).astype(np.float32)
    is_center_aligned = (np.abs((S[:, 0] + S[:, 2]) / 2 - (O[:, 0] + O[:, 2]) / 2) <= 1e-5).astype(np.float32)
    is_right_aligned = (np.abs(S[:, 2] - O[:, 2]) <= 1e-5).astype(np.float32)
    is_top_aligned = (np.abs(S[:, 1] - O[:, 1]) <= 1e-5).astype(np.float32)
    is_middle_aligned = (np.abs((S[:, 1] + S[:, 3]) / 2 - (O[:, 1] + O[:, 3]) / 2) <= 1e-5).astype(np.float32)
    is_bottom_aligned = (np.abs(S[:, 3] - O[:, 3]) <= 1e-5).astype(np.float32)

    features.append(is_left_aligned)  # 19. is_left_aligned
    features.append(is_center_aligned)  # 20. is_center_aligned
    features.append(is_right_aligned)  # 21. is_right_aligned
    features.append(is_top_aligned)  # 22. is_top_aligned
    features.append(is_middle_aligned)  # 23. is_middle_aligned
    features.append(is_bottom_aligned)  # 24. is_bottom_aligned

    # Stack all features column-wise
    edge_attr = np.stack(features, axis=1)  # (E, 18)
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=1e5, neginf=-1e5)
    return edge_attr

def get_text_pattern(text, text_pattern=None, return_vector=True):
    max_len = 40
    symbols = ['•', '-', '*', '+', '<', '>', '(', ')', '→', '✓', '#', '□', '■', '‣', '◦', '▪', '.', ':', '※', '']
    if text_pattern is None:
        pattern = ''
        for c in text:
            if c in symbols:
                pattern += c
            elif c.isspace():
                pattern += 'W'
            elif c.isdigit():
                pattern += 'D'
            elif not c.isalnum():
                pattern += 'S'
            else:
                pattern += 'C'

        compressed_pattern = ''
        prev_c = ''
        prev_repeat = 1
        for c_idx, c in enumerate(pattern):
            if c_idx == 0:
                compressed_pattern += c
            else:
                if prev_c == c:
                    prev_repeat += 1
                    continue
                else:
                    compressed_pattern += str(prev_repeat)
                    compressed_pattern += c
                    prev_repeat = 1
            prev_c = c
        if prev_c != '':
            compressed_pattern += str(prev_repeat)
        text_pattern = compressed_pattern

    if return_vector:
        vector = []
        for c in text_pattern:
            if c.isdigit():
                vector.append(int(c))
            elif c == 'C':
                vector.append(10)
            elif c == 'D':
                vector.append(11)
            elif c == 'S':
                vector.append(12)
            elif c == 'W':
                vector.append(13)
            elif c in symbols:
                vector.append(14 + symbols.index(c))
            else:
                raise Exception(f'Invalid pattern character: {c}')
        encoded = [format(v, '02o') for v in vector]
        encoded_vector = [0.0] * max_len
        cur_idx = 0
        for code in encoded:
            for c in code:
                encoded_vector[cur_idx] = int(c) / 8
                cur_idx += 1
                if cur_idx >= len(encoded_vector):
                    break
            if cur_idx >= len(encoded_vector):
                break
        return encoded_vector
    else:
        return text_pattern

def get_edge_matrix(num_node, edge_index, edge_label):
    """
    edge_index와 edge_label을 사용하여 n x n 근접이웃 행렬을 반환합니다.

    Args:
      edge_index: 연결의 인덱스를 나타내는 2D 리스트.
      edge_label: 연결 레이블을 나타내는 1D 리스트. 1은 연결이 있음을, 0은 연결이 없음을 의미합니다.

    Returns:
      n x n 근접이웃 행렬 (numpy array).
    """
    adj_matrix = np.zeros((num_node, num_node), dtype=np.int64)  # 초기 0으로 채워진 n x n 행렬

    row, col = edge_index
    for i in range(len(row)):
        adj_matrix[row[i]][col[i]] = int(edge_label[i])

    return adj_matrix


def _make_union_find(num_nodes: int) -> np.ndarray:
    """Initialize Union-Find parent array where each node is its own root."""
    return np.arange(num_nodes, dtype=np.int64)


def _find(parent: np.ndarray, x: int) -> int:
    """Find root with path compression."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


def _union(parent: np.ndarray, x: int, y: int) -> None:
    """Union two sets by linking their roots."""
    rx, ry = _find(parent, x), _find(parent, y)
    if rx != ry:
        parent[rx] = ry

def graph_connected_components(num_nodes: int, rows: np.ndarray, cols: np.ndarray):
    """
    Find connected components via Union-Find.

    Returns:
        labels : (N,) array where labels[i] is the component id of node i
        n_components : total number of components
    """
    parent = _make_union_find(num_nodes)

    for x, y in zip(rows.tolist(), cols.tolist()):
        _union(parent, x, y)

    # Normalize roots so labels are contiguous 0-based integers
    root_to_comp = {}
    labels = np.empty(num_nodes, dtype=np.int64)
    for i in range(num_nodes):
        root = _find(parent, i)
        if root not in root_to_comp:
            root_to_comp[root] = len(root_to_comp)
        labels[i] = root_to_comp[root]

    return labels, len(root_to_comp)


def group_node_by_edge(
        node_cls: np.ndarray,
        node_score: np.ndarray,
        edge_matrix: np.ndarray,
        bboxes: List[List[float]],
        label_priority_list: List[int],
) -> List[Dict]:
    """
    Drop-in replacement for the NetworkX-based implementation using pure numpy.
    No scipy dependency ? connected components are found via Union-Find.

    Optimizations over the original:
        1. O(N^2) double for loop -> np.triu + np.where (vectorized edge extraction)
        2. NetworkX connected_components -> Union-Find (no external dependency)
        3. bbox min/max -> np.min/max on masked array slice (vectorized)
        4. majority vote + priority tie-break -> np.bincount + priority lookup array
        5. tie_class reset per group (bug fix from original)

    Args / Returns: identical to the original function.
    """
    num_nodes = len(node_cls)

    if len(bboxes) != num_nodes or len(node_score) != num_nodes:
        raise ValueError("bboxes and node_score must have the same length as node_cls and edge_matrix.")

    bboxes_arr = np.array(bboxes, dtype=np.float32)  # (N, 4)

    # ------------------------------------------------------------------ #
    # 1. Extract edges from upper triangle (vectorized, no double for loop)
    # ------------------------------------------------------------------ #
    upper = np.triu(edge_matrix, k=1).astype(bool)  # (N, N) upper triangle mask
    rows, cols = np.where(upper)  # all edge pairs (i, j)

    # ------------------------------------------------------------------ #
    # 2. Find connected components via Union-Find (no scipy required)
    # ------------------------------------------------------------------ #
    labels, n_components = graph_connected_components(num_nodes, rows, cols)

    # Build priority lookup array: priority_lookup[cls_id] = priority index.
    # Lower index = higher priority. Classes not in the list get inf.
    max_cls = int(node_cls.max()) + 1 if num_nodes > 0 else 0
    lookup_size = max(max_cls, len(label_priority_list) + 1)
    priority_lookup = np.full(lookup_size, fill_value=np.inf)
    for priority, cls_id in enumerate(label_priority_list):
        if cls_id < lookup_size:
            priority_lookup[cls_id] = priority

    # ------------------------------------------------------------------ #
    # 3. Process each connected component
    # ------------------------------------------------------------------ #
    groups = []

    for comp_id in range(n_components):
        # tie_class is reset per group so each group tracks its own tie state
        tie_class = -1

        mask = labels == comp_id  # (N,) boolean mask
        indices = np.where(mask)[0].tolist()  # sorted node indices

        # Majority vote
        comp_cls = node_cls[mask]  # (M,)
        counts = np.bincount(comp_cls, minlength=int(comp_cls.max()) + 1)
        max_count = counts.max()
        tied = np.where(counts == max_count)[0]  # class ids with highest count

        if len(tied) == 1:
            group_class = int(tied[0])
        else:
            # Tie-break: pick class with highest priority in label_priority_list
            tied_priorities = priority_lookup[tied]
            best = tied[np.argmin(tied_priorities)]
            group_class = int(best)
            tie_class = group_class

        # Bounding box (vectorized)
        if len(indices) == 0:
            group_bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            comp_bboxes = bboxes_arr[mask]  # (M, 4)
            x1 = float(comp_bboxes[:, 0].min())
            y1 = float(comp_bboxes[:, 1].min())
            x2 = float(comp_bboxes[:, 2].max())
            y2 = float(comp_bboxes[:, 3].max())
            group_bbox = [x1, y1, x2, y2]

        groups.append({
            'indicies': indices,
            'group_class': group_class,
            'group_bbox': group_bbox,
            'tie_class': tie_class,
        })

    return groups


def extract_bbox_features_by_roi_align(
        features: np.ndarray,
        bboxes: List[Tuple[float, float, float, float]],
        page_width: int,
        page_height: int,
        roi_align_output_size: int = 2,  # New Parameter
        apply_softmax: bool = False,
        concat_mean_max: bool = True,
        add_uncertainty: bool = False
) -> np.ndarray:
    """
    Extract per-bbox features with RoIAlign-like sampling and mean/max pooling,
    and uncertainty metrics. This version uses bilinear interpolation for
    more accurate feature extraction.

    Args:
        features (np.ndarray): Feature map tensor (N, C, H, W). Typically N=1.
        bboxes (List[Tuple[float, float, float, float]]): Boxes (x1, y1, x2, y2)
            defined on the same coordinate system as page_width/page_height.
        page_width (int): Width of the image used to produce 'features'.
        page_height (int): Height of the image used to produce 'features'.
        roi_align_output_size (int): The target height and width (P) for the
            sampled feature map (P x P). Default is 7.
        apply_softmax (bool): If True, apply channel-wise softmax per spatial location.
        concat_mean_max (bool): If True, output [mean; max] -> 2C dims; else only mean -> C dims.
        add_uncertainty (bool): If True, append entropy_mean and margin_mean -> +2 dims.

    Returns:
        np.ndarray: Shape (len(bboxes), C or 2C plus optional +2 for uncertainty).
    """
    # Validate shapes
    assert features.ndim == 4, f"features must be (N,C,H,W), got {features.shape}"
    N, C, H, W = features.shape
    assert N == 1, "Only batch size 1 is supported."
    fmap = features[0]  # (C, H, W)

    # Use the new parameter
    P = roi_align_output_size
    assert P >= 1, "roi_align_output_size must be a positive integer."

    # --- Prepare feature maps (Raw vs Probs) ---
    if apply_softmax:
        # Calculate probabilities from logits (fmap)
        f = fmap.reshape(C, -1)
        f = f - f.max(axis=0, keepdims=True)  # numerical stability
        np.exp(f, out=f)
        denom = f.sum(axis=0, keepdims=True) + 1e-12
        probs = (f / denom).reshape(C, H, W)  # (C, H, W)
        fmap_used = probs  # use probs for pooling
    else:
        fmap_used = fmap  # use raw features/logits for pooling
        # For uncertainty metrics we still need probabilities
        g = fmap.reshape(C, -1)
        g = g - g.max(axis=0, keepdims=True)
        np.exp(g, out=g)
        denom_g = g.sum(axis=0, keepdims=True) + 1e-12
        probs = (g / denom_g).reshape(C, H, W)  # (C, H, W)

    # --- Output Dimension Setup ---
    base_dim = C * 2 if concat_mean_max else C
    out_dim = base_dim + (2 if add_uncertainty else 0)
    out = np.zeros((len(bboxes), out_dim), dtype=np.float32)

    # --- RoIAlign Helper: Bilinear Interpolation ---
    def bilinear_interpolate(feature_map: np.ndarray, x: float, y: float) -> np.ndarray:
        """
        Perform bilinear interpolation on the feature map at fractional coordinates (x, y).
        The feature map is (C, H, W). Coordinates are 0-indexed, ranging from 0 to W-1 or H-1.
        """
        C_f, H_f, W_f = feature_map.shape

        # Clamp fractional coordinates to boundary
        x = np.clip(x, 0, W_f - 1 - 1e-6)
        y = np.clip(y, 0, H_f - 1 - 1e-6)

        # Get the four surrounding integer grid points
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        # Calculate interpolation weights (deltas)
        dx = x - x0
        dy = y - y0

        # Clamp x1, y1 to boundaries
        x1 = np.clip(x1, 0, W_f - 1)
        y1 = np.clip(y1, 0, H_f - 1)

        # Feature values at the four surrounding points (C,)
        f00 = feature_map[:, y0, x0]
        f10 = feature_map[:, y0, x1]
        f01 = feature_map[:, y1, x0]
        f11 = feature_map[:, y1, x1]

        # Bilinear interpolation
        interp_val = (
                f00 * (1 - dx) * (1 - dy) +
                f10 * dx * (1 - dy) +
                f01 * (1 - dx) * dy +
                f11 * dx * dy
        )
        return interp_val

    # --- Process each bounding box ---
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # 1. Map bbox from image space to feature map space (floating point)
        scale_x = W / float(page_width)
        scale_y = H / float(page_height)

        fx1, fy1 = x1 * scale_x, y1 * scale_y
        fx2, fy2 = x2 * scale_x, y2 * scale_y

        roi_width = fx2 - fx1
        roi_height = fy2 - fy1

        # 2. Define sampling grid (P x P)
        bin_size_x = roi_width / P
        bin_size_y = roi_height / P

        sample_points = []
        for p_y in range(P):
            y_center = fy1 + (p_y + 0.5) * bin_size_y
            for p_x in range(P):
                x_center = fx1 + (p_x + 0.5) * bin_size_x
                sample_points.append((x_center, y_center))

        if not sample_points:
            continue

        # 3. Sample features using Bilinear Interpolation
        sampled_features = []
        sampled_probs = []

        for sx, sy in sample_points:
            # Features for pooling (fmap_used)
            interp_feat = bilinear_interpolate(fmap_used, sx, sy)
            sampled_features.append(interp_feat)

            # Probabilities for uncertainty (probs)
            if add_uncertainty:
                interp_prob = bilinear_interpolate(probs, sx, sy)
                sampled_probs.append(interp_prob)

        # Stack to (C, P*P)
        patch_feat = np.stack(sampled_features, axis=1)  # (C, P*P)

        # 4. Pooling over the sampled features

        # Mean pooling over spatial dims (P*P)
        mean_vec = patch_feat.mean(axis=1)  # (C,)

        if concat_mean_max:
            max_vec = patch_feat.max(axis=1)  # (C,)
            pooled = np.concatenate([mean_vec, max_vec], axis=0)  # (2C,)
        else:
            pooled = mean_vec  # (C,)

        # 5. Add Uncertainty Metrics (from sampled probabilities)
        if add_uncertainty:
            patch_prob = np.stack(sampled_probs, axis=1)  # (C, P*P)

            # Entropy: -sum_c p_c log p_c, averaged over the patch
            eps = 1e-12
            ent_map_flat = -(patch_prob * np.log(patch_prob + eps)).sum(axis=0)  # (P*P,)
            entropy_mean = ent_map_flat.mean()

            # Margin: top1 - top2, averaged over the patch
            C_, num_samples = patch_prob.shape

            # Get indices of the two largest probability channels at each location (P*P)
            idx_part = np.argpartition(patch_prob, -2, axis=0)[-2:]  # (2, P*P)
            # Get the actual top two probability values
            top_two = np.take_along_axis(patch_prob, idx_part, axis=0)  # (2, P*P)

            top_two.sort(axis=0)

            # Margin = top1 - top2 for each location
            margins = top_two[1] - top_two[0]  # (P*P,)
            margin_mean = margins.mean()

            # Append [entropy_mean, margin_mean]
            pooled = np.concatenate([pooled, np.array([entropy_mean, margin_mean], dtype=np.float32)], axis=0)

        out[i, :pooled.shape[0]] = pooled.astype(np.float32)

    return out

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


def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def extract_sobel_line_from_image(page_img, page_width, page_height,
                                  threshold=30, min_length_ratio=0.03,
                                  max_thickness=1):
    """
    Detect horizontal and vertical lines from a rasterized page image using
    Sobel edge detection + run-length encoding. Pure numpy, no OpenCV dependency.

    Intended as an image-processing-based alternative to vec_line extraction
    (get_vector_lines), which relies on PDF drawing objects and is sensitive
    to how the PDF author chose to render lines (vector path, filled rect,
    embedded image, etc.).

    Pipeline per direction:
        1. Convert page_img to grayscale (luminance-weighted).
        2. Apply 1-D Sobel kernel along the axis perpendicular to the line
           direction to detect edges (e.g. horizontal Sobel dY for horizontal
           lines: high response at top/bottom edge of a dark horizontal band).
        3. Threshold the absolute response to get a binary edge map.
        4. For each row (horizontal) or column (vertical), run run-length
           encoding to find contiguous foreground runs.
        5. For runs long enough (>= min_length_ratio * page dimension), record
           the enclosing bbox in PDF coordinate space (scaled from pixel space).

    Coordinate conversion:
        Pixel coords are scaled back to PDF point space using page_width /
        page_height divided by the image pixel dimensions, matching the
        same coordinate space used by get_vector_lines() and
        extract_base_elements().

    Args:
        page_img:         np.ndarray (H, W, C) uint8 -- page raster image as
                          stored in data_dict['image'].
        page_width:       float -- PDF page width in points (data_dict['page_width']).
        page_height:      float -- PDF page height in points (data_dict['page_height']).
        threshold:        int -- Sobel response magnitude cutoff [0, 255].
                          Lower values catch faint lines but increase false
                          positives from text strokes. 30 is a reasonable
                          starting point for clean PDF renders.
        min_length_ratio: float -- minimum line length as a fraction of the
                          page dimension (page_width for h-lines, page_height
                          for v-lines). 0.03 = 3% of page dimension, which
                          filters out short text-stroke artifacts while keeping
                          table ruling lines.
        max_thickness:    int -- maximum pixel thickness of a detected line
                          bbox. Sobel response can span a few pixels on either
                          side of a stroke; this caps how tall/wide a returned
                          bbox can be in the perpendicular axis, keeping line
                          bboxes thin and consistent with vec_line output.

    Returns:
        h_lines: list of [x1, y1, x2, y2] in PDF point space (horizontal lines).
        v_lines: list of [x1, y1, x2, y2] in PDF point space (vertical lines).
    """
    img_h, img_w = page_img.shape[0], page_img.shape[1]

    # Step 1: grayscale (luminance-weighted, no cv2)
    if page_img.ndim == 3 and page_img.shape[2] >= 3:
        gray = (0.299 * page_img[:, :, 0].astype(np.float32)
                + 0.587 * page_img[:, :, 1].astype(np.float32)
                + 0.114 * page_img[:, :, 2].astype(np.float32))
    else:
        gray = page_img[:, :, 0].astype(np.float32) if page_img.ndim == 3 else page_img.astype(np.float32)

    # Scale factors: pixel -> PDF points
    sx = page_width / img_w
    sy = page_height / img_h

    # Step 2: 1-D Sobel kernels (3-tap: [-1, 0, 1] along the perpendicular axis)
    # For horizontal lines: detect edges in Y direction (dY)
    # For vertical lines:   detect edges in X direction (dX)
    sobel_y = np.array([[-1], [0], [1]], dtype=np.float32)  # (3,1) kernel
    sobel_x = np.array([[-1, 0, 1]], dtype=np.float32)       # (1,3) kernel

    def _convolve_rows(img, kernel):
        """Apply a (K,1) kernel along axis-0 via sliding window (no scipy)."""
        k = kernel.shape[0]
        pad = k // 2
        padded = np.pad(img, ((pad, pad), (0, 0)), mode='edge')
        out = np.zeros_like(img)
        for i, w in enumerate(kernel[:, 0]):
            if w != 0:
                out += w * padded[i:i + img.shape[0], :]
        return out

    def _convolve_cols(img, kernel):
        """Apply a (1,K) kernel along axis-1 via sliding window (no scipy)."""
        k = kernel.shape[1]
        pad = k // 2
        padded = np.pad(img, ((0, 0), (pad, pad)), mode='edge')
        out = np.zeros_like(img)
        for i, w in enumerate(kernel[0, :]):
            if w != 0:
                out += w * padded[:, i:i + img.shape[1]]
        return out

    edge_y = np.abs(_convolve_rows(gray, sobel_y))  # (H, W) -- responds to h-lines
    edge_x = np.abs(_convolve_cols(gray, sobel_x))  # (H, W) -- responds to v-lines

    # Step 3: threshold -> binary maps
    bin_y = (edge_y >= threshold).astype(np.uint8)  # foreground = horizontal edge
    bin_x = (edge_x >= threshold).astype(np.uint8)  # foreground = vertical edge

    min_h_px = max(1, int(min_length_ratio * img_w))
    min_v_px = max(1, int(min_length_ratio * img_h))

    # Step 4+5: run-length encoding per row/column -> bboxes
    def _rle_runs(row):
        """Return list of (start, end) inclusive pixel indices of foreground runs."""
        runs = []
        in_run = False
        start = 0
        for x, v in enumerate(row):
            if v and not in_run:
                start = x
                in_run = True
            elif not v and in_run:
                runs.append((start, x - 1))
                in_run = False
        if in_run:
            runs.append((start, len(row) - 1))
        return runs

    h_lines = []
    for row_idx in range(img_h):
        runs = _rle_runs(bin_y[row_idx])
        for x_start, x_end in runs:
            if x_end - x_start + 1 < min_h_px:
                continue
            # Clamp bbox thickness in Y to max_thickness
            y_px1 = max(0, row_idx - max_thickness // 2)
            y_px2 = min(img_h - 1, row_idx + max_thickness // 2)
            x1 = x_start * sx
            y1 = y_px1 * sy
            x2 = (x_end + 1) * sx
            y2 = (y_px2 + 1) * sy
            h_lines.append([x1, y1, x2, y2])

    v_lines = []
    for col_idx in range(img_w):
        runs = _rle_runs(bin_x[:, col_idx])
        for y_start, y_end in runs:
            if y_end - y_start + 1 < min_v_px:
                continue
            x_px1 = max(0, col_idx - max_thickness // 2)
            x_px2 = min(img_w - 1, col_idx + max_thickness // 2)
            x1 = x_px1 * sx
            y1 = y_start * sy
            x2 = (x_px2 + 1) * sx
            y2 = (y_end + 1) * sy
            v_lines.append([x1, y1, x2, y2])

    return h_lines, v_lines


def extract_lsd_line_from_image(page_img, page_width, page_height,
                                angle_threshold=10.0, min_length_ratio=0.03):
    """
    Detect horizontal and vertical lines from a rasterized page image using
    OpenCV's Line Segment Detector (LSD). Requires cv2; raises ImportError
    with install instructions if not available.

    LSD differs from Sobel + RLE in several important ways:
      - Operates directly on gradient fields via region growing; line segments
        are returned as first-class geometric objects rather than per-row/column
        edge responses that must be stitched together afterward.
      - A thick filled-rect line produces ONE segment (the center line), not
        two near-duplicate edge responses -- eliminates the near-duplicate
        problem that makes dtable_img_line dependent on merge_lines for
        correctness rather than just consolidation.
      - Faint or blurry lines: LSD accumulates gradient evidence across the
        full extent of a region, so a line that is too faint to cross a global
        threshold row-by-row is still detected if it is geometrically coherent.
      - Broken/dashed lines: each unbroken segment is returned separately.
        Callers that need to bridge gaps should still pass results through
        merge_lines(), but the input fragment count is much lower than Sobel's.

    Only horizontal and vertical segments are returned. Diagonal segments
    (which LSD also detects) are discarded via angle_threshold.

    Coordinate output format is identical to extract_sobel_line_from_image():
    list of [x1, y1, x2, y2] in PDF point space. This allows callers to
    treat both functions interchangeably.

    Args:
        page_img:         np.ndarray (H, W, C) uint8 -- page raster image as
                          stored in data_dict['image'].
        page_width:       float -- PDF page width in points (data_dict['page_width']).
        page_height:      float -- PDF page height in points (data_dict['page_height']).
        angle_threshold:  float -- max deviation from 0/90 degrees (in degrees)
                          for a segment to be classified as horizontal or
                          vertical. 10.0 absorbs slight perspective / rendering
                          tilt while keeping diagonals out. Tune down (e.g. 5.0)
                          for very clean PDF renders; tune up (e.g. 15.0) for
                          scanned documents with registration skew.
        min_length_ratio: float -- minimum segment length as a fraction of the
                          relevant page dimension (page_width for h-lines,
                          page_height for v-lines). Matches the default used by
                          extract_sobel_line_from_image() (0.03 = 3%) so that
                          the two functions produce comparable output when used
                          as drop-in alternatives.

    Returns:
        h_lines: list of [x1, y1, x2, y2] in PDF point space (horizontal segments).
        v_lines: list of [x1, y1, x2, y2] in PDF point space (vertical segments).

    Raises:
        ImportError: if cv2 (opencv-python or opencv-python-headless) is not installed.
        AttributeError: if the installed OpenCV build does not include
            cv2.createLineSegmentDetector (very old or stripped builds).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "cv2 (OpenCV) is required for extract_lsd_line_from_image but is not installed. "
            "Install it with: pip install opencv-python-headless"
        )

    img_h, img_w = page_img.shape[0], page_img.shape[1]

    # Step 1: grayscale (luminance-weighted, same as extract_sobel_line_from_image)
    if page_img.ndim == 3 and page_img.shape[2] >= 3:
        gray = (0.299 * page_img[:, :, 0].astype(np.float32)
                + 0.587 * page_img[:, :, 1].astype(np.float32)
                + 0.114 * page_img[:, :, 2].astype(np.float32)).astype(np.uint8)
    else:
        gray = page_img[:, :, 0] if page_img.ndim == 3 else page_img

    # Step 2: LSD detection.
    # LSD_REFINE_STD: standard refinement -- merges nearly-collinear segments
    # that belong to the same edge into one. Better for long table ruling lines
    # than LSD_REFINE_NONE (too fragmented) or LSD_REFINE_ADV (slower, marginal
    # gain on axis-aligned lines).
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    except AttributeError:
        raise AttributeError(
            "cv2.createLineSegmentDetector is not available in this OpenCV build. "
            "Install opencv-python or opencv-python-headless >= 3.4."
        )
    detected, *_ = lsd.detect(gray)
    # detected shape varies by OpenCV build:
    #   (N, 1, 4) float32 -- most OpenCV 4.x builds
    #   (N, 4)    float32 -- some builds / versions
    # Normalize to (N, 4) so the per-segment loop works regardless.
    if detected is None:
        return [], []

    detected = detected.reshape(-1, 4)

    # Scale factors: pixel -> PDF points (same convention as Sobel counterpart)
    sx = page_width / img_w
    sy = page_height / img_h

    min_h_len_px = min_length_ratio * img_w   # absolute pixel length floor for h-lines
    min_v_len_px = min_length_ratio * img_h   # absolute pixel length floor for v-lines

    h_lines = []
    v_lines = []

    for seg in detected:                       # iterate over (N, 4)
        x1_px, y1_px, x2_px, y2_px = seg

        dx = x2_px - x1_px
        dy = y2_px - y1_px
        length_px = (dx * dx + dy * dy) ** 0.5
        if length_px == 0:
            continue

        # Angle from horizontal, in degrees: 0 = right, 90 = down.
        angle_deg = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))

        if angle_deg <= angle_threshold:
            # Horizontal segment: length runs along x.
            if length_px < min_h_len_px:
                continue
            # Normalise: ensure x1 <= x2; y is the mean of both endpoints
            # (LSD already returns them in draw order, but may vary).
            lx1, lx2 = sorted((x1_px, x2_px))
            ly = (y1_px + y2_px) / 2.0
            h_lines.append([
                lx1 * sx,
                ly * sy,
                lx2 * sx,
                ly * sy,
            ])

        elif angle_deg >= (90.0 - angle_threshold):
            # Vertical segment: length runs along y.
            if length_px < min_v_len_px:
                continue
            ly1, ly2 = sorted((y1_px, y2_px))
            lx = (x1_px + x2_px) / 2.0
            v_lines.append([
                lx * sx,
                ly1 * sy,
                lx * sx,
                ly2 * sy,
            ])
        # else: diagonal -- discard

    return h_lines, v_lines


def _find_connected_component_regions(binary_mask, connectivity=8, min_area_threshold=10):
    """
    Find connected components in a binary mask and return their bounding boxes.

    Replaces the previous two-step pipeline:
        segmentation_connected_components()  (Python BFS)
        _get_bboxes_from_labeled_image()     (np.argwhere per label)

    Both steps are now handled by cv2.connectedComponentsWithStats in a single
    C++ call, which is ~100x faster than the Python BFS and eliminates the
    O(N * num_labels) argwhere scans.

    Args:
        binary_mask (np.ndarray): 2D uint8 array with 0 (background) and 1 (foreground).
        connectivity (int): 4 or 8.
        min_area_threshold (int): Components smaller than this pixel count are ignored.

    Returns:
        list[dict]: Each dict has keys 'label', 'bbox' (min_row, min_col, max_row, max_col),
                    and 'area'. Same schema as the old _get_bboxes_from_labeled_image().
    """
    import cv2

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity, ltype=cv2.CV_32S
    )

    # stats shape: (num_labels, 5)
    # columns: CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA
    regions = []
    for label_id in range(1, num_labels):   # skip label 0 (background)
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_threshold:
            continue

        x      = int(stats[label_id, cv2.CC_STAT_LEFT])
        y      = int(stats[label_id, cv2.CC_STAT_TOP])
        width  = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # bbox in (min_row, min_col, max_row, max_col) — same convention as before
        regions.append({
            'label': label_id,
            'bbox': (y, x, y + height, x + width),
            'area': area,
        })
    return regions


def _binary_opening_numpy(binary_image, kernel_size):
    """
    Performs binary opening (erosion followed by dilation) on a 2D binary image
    using a square kernel of ones.

    Args:
        binary_image (np.ndarray): A 2D uint8 array (H, W) with 0s and 1s.
        kernel_size (int): Side length of the square kernel. Must be an odd number.

    Returns:
        np.ndarray: The opened binary image (same dtype as input).
    """
    import cv2

    if kernel_size <= 1:
        return binary_image
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size for morphological operations must be an odd number.")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    # BORDER_CONSTANT with value=0 matches the original numpy implementation's
    # zero-padding behavior. cv2's default (BORDER_REFLECT_101) produces
    # different results at image edges.
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

def extract_bboxes_from_segmentation(
    seg_logits: np.ndarray,
    class_names: list[str],
    target_class: list[str] | None = None,
    min_component_area: int = 10,
    morphology_kernel_size: int = 0,
) -> list[dict]:
    """
    Extract bounding boxes only for the classes you care about.

    Args:
        seg_logits (np.ndarray):
            Per-class segmentation logits, shape (1, len(class_names), H, W).
            The caller must slice out exactly the class channels before
            calling this function; no implicit channel slicing happens here.
            Detection uses argmax, which is invariant to whether the values
            are raw logits or post-softmax probabilities, so either can be
            passed in.
        class_names (list[str]):
            Ordered list of class names (index i corresponds to channel i).
        target_class (list[str] | None):
            If given, only these classes are processed.  None means process
            every class except background (class-id 0).
        min_component_area (int):
            Connected components smaller than this many pixels are ignored.
        morphology_kernel_size (int):
            Size of the square morphological opening kernel (must be odd).
            0 or 1 disables opening.

    Returns:
        list[dict]:
            [{'bbox': [x1, y1, x2, y2], 'class': <name>, 'score': <float>}, ...]
            (x1,y1) is top-left, (x2,y2) is bottom-right in image coordinates.
            score = (# pixels of predicted class inside bbox) / (bbox area).
    """
    n_classes = len(class_names)
    assert seg_logits.shape[1] == n_classes, (
        f"seg_logits channel count ({seg_logits.shape[1]}) must equal "
        f"len(class_names) ({n_classes}); caller must pass only the "
        f"class-logit channels, not a larger feature tensor."
    )
    seg_maps = seg_logits[0]          # (n_classes, H, W)

    # build quick lookup & filter requested classes
    if target_class is None:
        active_ids = np.arange(1, n_classes, dtype=np.intp)  # skip background
    else:
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
        active_ids = np.array([name_to_id[n] for n in target_class if n in name_to_id],
                              dtype=np.intp)

    # single arg-max over all channels
    class_map = np.argmax(seg_maps, axis=0)               # (H, W)

    result: list[dict] = []

    for class_id in active_ids:
        # binary mask for current class
        mask = (class_map == class_id).astype(np.uint8)

        # optional morphological opening
        if morphology_kernel_size > 1:
            mask = _binary_opening_numpy(mask, morphology_kernel_size)

        # connected-component labelling
        regions = _find_connected_component_regions(
            mask,
            connectivity=8,
            min_area_threshold=min_component_area,
        )

        cls_name = class_names[class_id]

        for reg in regions:
            min_row, min_col, max_row, max_col = reg["bbox"]

            # slice bbox area directly
            roi = class_map[min_row:max_row, min_col:max_col]
            pix_cls = np.count_nonzero(roi == class_id)
            score = pix_cls / roi.size if roi.size else 0.0

            result.append({
                "bbox": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "class": cls_name,
                "score": float(score),
            })

    return result


def _resize_class_map_nearest(class_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize a 2-D integer label map using nearest-neighbor interpolation.
    Pure numpy; no scipy/cv2 dependency.

    Args:
        class_map (np.ndarray): (H, W) integer array of class IDs.
        target_h (int): Output height.
        target_w (int): Output width.

    Returns:
        np.ndarray: (target_h, target_w) integer array, same dtype as input.
    """
    src_h, src_w = class_map.shape

    # Map each output pixel to the nearest input pixel via index broadcasting.
    row_idx = (np.arange(target_h) * src_h / target_h).astype(np.intp)
    col_idx = (np.arange(target_w) * src_w / target_w).astype(np.intp)

    return class_map[np.ix_(row_idx, col_idx)]


def _find_connected_components_numpy(binary_mask: np.ndarray, min_area: int) -> list[dict]:
    """
    BFS-based connected-component labeling on a 2-D binary uint8 mask.
    8-connectivity.  Pure numpy + Python; no cv2/scipy.

    Args:
        binary_mask (np.ndarray): (H, W) uint8 array, 0 = background, 1 = foreground.
        min_area (int): Components with fewer pixels than this are discarded.

    Returns:
        list[dict]: Each entry has 'bbox' (min_row, min_col, max_row, max_col)
                    and 'area' (pixel count).
    """
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    # Pre-compute foreground pixel coordinates to avoid scanning every cell.
    fg_rows, fg_cols = np.nonzero(binary_mask)

    # 8-connected neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]

    for start_r, start_c in zip(fg_rows.tolist(), fg_cols.tolist()):
        if visited[start_r, start_c]:
            continue

        # BFS
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        min_r = max_r = start_r
        min_c = max_c = start_c
        area = 0

        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            area += 1

            if r < min_r: min_r = r
            if r > max_r: max_r = r
            if c < min_c: min_c = c
            if c > max_c: max_c = c

            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and binary_mask[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        if area >= min_area:
            regions.append({
                "bbox": (min_r, min_c, max_r + 1, max_c + 1),
                "area": area,
            })

    return regions


def extract_bboxes_from_segmentation_numpy(
        seg_logits: np.ndarray,
        class_names: list[str],
        target_class: list[str] | None = None,
        min_component_area: int = 10,
        downscale_size: int = 100,
) -> list[dict]:
    """
    Numpy-only equivalent of extract_bboxes_from_segmentation().

    To compensate for the slower Python BFS, detection is performed on a
    downscaled class map (longest side capped at `downscale_size`), and the
    resulting bounding boxes are scaled back to original resolution.
    Morphological opening is omitted; callers should apply post-processing if
    needed.

    Args:
        seg_logits (np.ndarray):
            Per-class segmentation logits, shape (1, len(class_names), H, W).
            The caller must slice out exactly the class channels before
            calling this function; no implicit channel slicing happens here.
            Detection uses argmax, which is invariant to whether the values
            are raw logits or post-softmax probabilities, so either can be
            passed in.
        class_names (list[str]):
            Ordered list of class names (index i corresponds to channel i).
        target_class (list[str] | None):
            Classes to process.  None processes all non-background classes
            (class-id >= 1).
        min_component_area (int):
            Minimum component size in *original* pixels.  Scaled
            proportionally before BFS so that the threshold is consistent
            regardless of downscale_size.
        downscale_size (int):
            The downscaled map's longest dimension.  Smaller values are
            faster but may merge or lose small regions.

    Returns:
        list[dict]:
            [{'bbox': [x1, y1, x2, y2], 'class': <name>, 'score': <float>}]
            (x1, y1) = top-left, (x2, y2) = bottom-right in original image
            coordinates.  score = fraction of bbox pixels predicted as this
            class in the *original* class map.
    """
    n_classes = len(class_names)
    assert seg_logits.shape[1] == n_classes, (
        f"seg_logits channel count ({seg_logits.shape[1]}) must equal "
        f"len(class_names) ({n_classes}); caller must pass only the "
        f"class-logit channels, not a larger feature tensor."
    )
    seg_maps = seg_logits[0]  # (n_classes, H, W)
    orig_h, orig_w = seg_maps.shape[1], seg_maps.shape[2]

    # Argmax on full-resolution map (cheap; single vectorized op)
    class_map_full = np.argmax(seg_maps, axis=0)  # (H, W)

    # Build downscaled class map via nearest-neighbor
    scale = downscale_size / max(orig_h, orig_w)
    ds_h = max(1, int(round(orig_h * scale)))
    ds_w = max(1, int(round(orig_w * scale)))
    class_map_ds = _resize_class_map_nearest(class_map_full, ds_h, ds_w)

    # Scale area threshold proportionally to downscaled pixel count
    area_scale = (ds_h * ds_w) / (orig_h * orig_w)
    min_area_ds = max(1, int(np.ceil(min_component_area * area_scale)))

    # Inverse scale factors for bbox recovery
    scale_r = orig_h / ds_h
    scale_c = orig_w / ds_w

    # Determine which class IDs to process
    if target_class is None:
        active_ids = np.arange(1, n_classes, dtype=np.intp)
    else:
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
        active_ids = np.array(
            [name_to_id[n] for n in target_class if n in name_to_id],
            dtype=np.intp,
        )

    result: list[dict] = []

    for class_id in active_ids:
        mask_ds = (class_map_ds == class_id).astype(np.uint8)

        regions = _find_connected_components_numpy(mask_ds, min_area=min_area_ds)

        cls_name = class_names[class_id]

        for reg in regions:
            ds_min_r, ds_min_c, ds_max_r, ds_max_c = reg["bbox"]

            # Map bbox corners back to original resolution
            orig_min_r = int(ds_min_r * scale_r)
            orig_min_c = int(ds_min_c * scale_c)
            orig_max_r = min(orig_h, int(np.ceil(ds_max_r * scale_r)))
            orig_max_c = min(orig_w, int(np.ceil(ds_max_c * scale_c)))

            # Score: pixel purity in the original full-resolution roi
            roi = class_map_full[orig_min_r:orig_max_r, orig_min_c:orig_max_c]
            pix_cls = int(np.count_nonzero(roi == class_id))
            score = pix_cls / roi.size if roi.size else 0.0

            result.append({
                "bbox": [orig_min_c, orig_min_r, orig_max_c, orig_max_r],
                "class": cls_name,
                "score": float(score),
            })

    return result


if __name__ == "__main__":
    print(get_text_pattern('(A) This is... A', return_vector=True))
