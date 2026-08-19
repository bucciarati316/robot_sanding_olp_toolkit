"""刀路真实分层语义到 PROCESS 段和过渡请求的转换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.schemas import (
    PathSegmentType,
    ProcessSegment,
    ToolpathResult,
    TransitionKind,
    TransitionRequest,
)


@dataclass(frozen=True)
class ProcessSegmentation:
    """模块 B 向模块 C/G 提供的稳定分段结果。"""

    layer_ids: np.ndarray
    segment_ids: np.ndarray
    segment_types: np.ndarray
    process_segments: tuple[ProcessSegment, ...]
    transition_requests: tuple[TransitionRequest, ...]

    def __post_init__(self) -> None:
        layers = np.asarray(self.layer_ids, dtype=int)
        segments = np.asarray(self.segment_ids, dtype=int)
        kinds = np.asarray(self.segment_types, dtype=object)
        if layers.ndim != 1 or segments.shape != layers.shape or kinds.shape != layers.shape:
            raise ValueError("逐点 layer/segment/type 数组长度必须一致")
        object.__setattr__(self, "layer_ids", layers.copy())
        object.__setattr__(self, "segment_ids", segments.copy())
        object.__setattr__(self, "segment_types", kinds.copy())


def resolve_tcp_matrices(
    result: ToolpathResult,
    sequenced_matrices: np.ndarray,
    ordered_original_indices: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, str]:
    """Resolve the declared authoritative TCP source after path ordering."""
    result.validate()
    sequenced = np.asarray(sequenced_matrices, dtype=float)
    order = np.asarray(ordered_original_indices, dtype=int)
    expected_shape = (len(order), 4, 4)
    if sequenced.shape != expected_shape:
        raise ValueError("sequenced_matrices must have shape (N,4,4)")
    if order.shape != (len(result.points),):
        raise ValueError("ordered_original_indices must contain every input point")
    if len(order) and (
        np.min(order) < 0
        or np.max(order) >= len(result.points)
        or len(np.unique(order)) != len(order)
    ):
        raise ValueError("ordered_original_indices must be a permutation")

    if result.tcp_matrices_authoritative:
        matrices = np.asarray(result.matrices, dtype=float)[order]
        source = result.tcp_matrix_source or "algorithm_authoritative"
    else:
        matrices = sequenced.copy()
        source = "path_sequencer"
    return matrices.copy(), source


def normalize_layer_ids(layer_ids: Sequence[int] | np.ndarray | None, point_count: int) -> np.ndarray:
    """把缺省层语义规范为单层，同时拒绝隐式固定点数推算。"""
    if point_count < 0:
        raise ValueError("point_count 不能为负数")
    if layer_ids is None:
        return np.zeros(point_count, dtype=np.int32)
    raw_values = np.asarray(layer_ids)
    if raw_values.size == 0:
        return np.zeros(point_count, dtype=np.int32)
    if raw_values.shape != (point_count,):
        raise ValueError("layer_ids 必须是与路点等长的逐点层编号")
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("layer_ids 必须是有限整数") from exc
    integer_limits = np.iinfo(np.int32)
    if (
        not np.all(np.isfinite(values))
        or not np.all(values == np.rint(values))
        or np.any(values < integer_limits.min)
        or np.any(values > integer_limits.max)
    ):
        raise ValueError("layer_ids 必须是有限整数")
    return values.astype(np.int32)


def contiguous_layer_runs(layer_ids: Sequence[int] | np.ndarray) -> list[tuple[int, int, int]]:
    """返回 ``(layer_id, start, stop)``，并要求每层只出现一个连续区间。"""
    layers = np.asarray(layer_ids, dtype=int)
    if layers.ndim != 1:
        raise ValueError("layer_ids 必须是一维数组")
    if len(layers) == 0:
        return []
    boundaries = np.flatnonzero(np.r_[True, layers[1:] != layers[:-1], True])
    runs = [
        (int(layers[start]), int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    run_ids = [layer_id for layer_id, _, _ in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("同一 layer_id 不得出现在多个不连续区间")
    return runs


def zigzag_order_indices(layer_ids: Sequence[int] | np.ndarray) -> np.ndarray:
    """按真实层区间反转奇数序号层，保留原 layer_id 与原始点索引。"""
    runs = contiguous_layer_runs(layer_ids)
    pieces: list[np.ndarray] = []
    for ordinal, (_, start, stop) in enumerate(runs):
        indices = np.arange(start, stop, dtype=int)
        pieces.append(indices[::-1] if ordinal % 2 else indices)
    return np.concatenate(pieces) if pieces else np.array([], dtype=int)


def build_process_segmentation(
    tcp_poses: np.ndarray,
    layer_ids: Sequence[int] | np.ndarray | None,
    *,
    original_indices: Sequence[int] | np.ndarray | None = None,
) -> ProcessSegmentation:
    """把显式逐点层语义转换为 PROCESS 段和相邻层过渡请求。"""
    poses = np.asarray(tcp_poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("tcp_poses 必须为 (N,4,4)")
    layers = normalize_layer_ids(layer_ids, len(poses))
    originals = (
        np.arange(len(poses), dtype=int)
        if original_indices is None
        else np.asarray(original_indices, dtype=int)
    )
    if originals.shape != (len(poses),):
        raise ValueError("original_indices 长度必须与 tcp_poses 一致")

    runs = contiguous_layer_runs(layers)
    segment_ids = np.empty(len(poses), dtype=int)
    segment_types = np.full(len(poses), PathSegmentType.PROCESS, dtype=object)
    process_segments: list[ProcessSegment] = []
    for segment_id, (layer_id, start, stop) in enumerate(runs):
        segment_ids[start:stop] = segment_id
        process_segments.append(
            ProcessSegment(
                segment_id=segment_id,
                layer_id=layer_id,
                tcp_poses=poses[start:stop],
                original_indices=originals[start:stop],
            )
        )

    transition_requests: list[TransitionRequest] = []
    for current, following in zip(process_segments[:-1], process_segments[1:]):
        transition_requests.append(
            TransitionRequest(
                kind=TransitionKind.LAYER_TRANSITION,
                start_segment_id=current.segment_id,
                goal_segment_id=following.segment_id,
                start_layer_id=current.layer_id,
                goal_layer_id=following.layer_id,
                start_pose=current.end_pose,
                goal_pose=following.start_pose,
            )
        )

    return ProcessSegmentation(
        layer_ids=layers,
        segment_ids=segment_ids,
        segment_types=segment_types,
        process_segments=tuple(process_segments),
        transition_requests=tuple(transition_requests),
    )
