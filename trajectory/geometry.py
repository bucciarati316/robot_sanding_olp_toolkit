"""TCP 几何路径清理、弧长参数化与自适应采样。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation, Slerp

from core.schemas import GeometricTrajectory, PathSegmentType, ProcessParameters


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    """把含少量数值误差的矩阵投影到 SO(3)。"""
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float((Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)).magnitude())


def _point_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    valid = denom > 1e-20
    ratios = np.zeros(len(vectors))
    ratios[valid] = np.einsum("ij,ij->i", point - starts, vectors)[valid] / denom[valid]
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * vectors
    return float(np.min(np.linalg.norm(projections - point, axis=1)))


def unwrap_revolute_trajectory(positions: np.ndarray) -> np.ndarray:
    """消除连续旋转关节跨越 ±π 产生的假跳变。"""
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2:
        raise ValueError("positions 必须为二维数组")
    return np.unwrap(q, axis=0)


@dataclass
class _SegmentCurve:
    path_s: np.ndarray
    poses: np.ndarray
    position_curve: object
    orientation_curve: Slerp
    use_linear_position: bool = False

    def evaluate(self, s: float) -> np.ndarray:
        result = np.eye(4)
        if self.use_linear_position:
            for axis in range(3):
                result[axis, 3] = np.interp(s, self.path_s, self.poses[:, axis, 3])
        else:
            result[:3, 3] = np.asarray(self.position_curve(s), dtype=float)
        result[:3, :3] = self.orientation_curve([s]).as_matrix()[0]
        return result


class GeometricPathBuilder:
    """构建误差受控的连续 TCP 路径和自适应规划网格。"""

    def __init__(self, parameters: ProcessParameters):
        self.parameters = parameters

    def build(
        self,
        tcp_poses: np.ndarray,
        segment_types: Optional[Iterable[PathSegmentType]] = None,
        segment_ids: Optional[np.ndarray] = None,
        layer_ids: Optional[np.ndarray] = None,
    ) -> GeometricTrajectory:
        poses, kinds, ids, layers, originals = self._sanitize(
            tcp_poses,
            segment_types,
            segment_ids,
            layer_ids,
        )
        samples_s: list[float] = []
        samples_pose: list[np.ndarray] = []
        samples_kind: list[PathSegmentType] = []
        samples_id: list[int] = []
        samples_layer: list[int] = []
        samples_original: list[int] = []
        offset = 0.0

        boundaries = np.flatnonzero(
            np.r_[
                True,
                (ids[1:] != ids[:-1])
                | (kinds[1:] != kinds[:-1])
                | (layers[1:] != layers[:-1]),
                True,
            ]
        )
        for boundary_index in range(len(boundaries) - 1):
            start = int(boundaries[boundary_index])
            stop = int(boundaries[boundary_index + 1])
            if stop - start < 2:
                # Preserve a one-point semantic marker exactly.  It is not a
                # fit candidate and must never borrow a point from an adjacent
                # layer/segment.
                samples_s.append(offset + (1e-9 if samples_s else 0.0))
                samples_pose.append(poses[start].copy())
                samples_kind.append(kinds[start])
                samples_id.append(int(ids[start]))
                samples_layer.append(int(layers[start]))
                samples_original.append(int(originals[start]))
                offset = samples_s[-1]
                continue
            seg_poses = poses[start:stop]
            local_s = self._arc_length(seg_poses[:, :3, 3])
            if local_s[-1] <= 0:
                continue
            curve = self._make_curve(local_s, seg_poses)
            grid = self._adaptive_grid(curve, local_s)
            segment_offset = offset + (1e-9 if samples_s else 0.0)
            for s in grid:
                samples_s.append(segment_offset + float(s))
                samples_pose.append(curve.evaluate(float(s)))
                nearest = int(np.argmin(np.abs(local_s - s))) + start
                samples_kind.append(kinds[nearest])
                samples_id.append(int(ids[nearest]))
                samples_layer.append(int(layers[nearest]))
                samples_original.append(int(originals[nearest]))
            offset = samples_s[-1]

        if len(samples_s) < 2:
            raise ValueError("清理后没有可构建的有效路径")
        return GeometricTrajectory(
            path_s=np.asarray(samples_s),
            tcp_poses=np.asarray(samples_pose),
            segment_types=np.asarray(samples_kind, dtype=object),
            segment_ids=np.asarray(samples_id, dtype=int),
            layer_ids=np.asarray(samples_layer, dtype=int),
            original_indices=np.asarray(samples_original, dtype=int),
            metadata={
                "sampling": "adaptive_arc_length",
                "chord_tolerance_m": self.parameters.chord_tolerance_m,
                "orientation_tolerance_rad": self.parameters.orientation_tolerance_rad,
            },
        )

    def _sanitize(self, tcp_poses, segment_types, segment_ids, layer_ids):
        poses = np.asarray(tcp_poses, dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
            raise ValueError("tcp_poses 必须为至少两个 (4,4) 位姿")
        if not np.all(np.isfinite(poses)):
            raise ValueError("tcp_poses 包含非有限值")
        poses = poses.copy()
        for pose in poses:
            pose[:3, :3] = _project_rotation(pose[:3, :3])
            pose[3] = [0.0, 0.0, 0.0, 1.0]
        kinds = (
            [PathSegmentType.PROCESS] * len(poses)
            if segment_types is None
            else list(segment_types)
        )
        if len(kinds) != len(poses):
            raise ValueError("segment_types 长度错误")
        kinds = [kind if isinstance(kind, PathSegmentType) else PathSegmentType(str(kind)) for kind in kinds]
        ids = np.zeros(len(poses), dtype=int) if segment_ids is None else np.asarray(segment_ids, dtype=int)
        if ids.shape != (len(poses),):
            raise ValueError("segment_ids 长度错误")
        layers = np.zeros(len(poses), dtype=int) if layer_ids is None else np.asarray(layer_ids, dtype=int)
        if layers.shape != (len(poses),):
            raise ValueError("layer_ids 长度错误")

        keep = [0]
        for index in range(1, len(poses)):
            translation_change = np.linalg.norm(poses[index, :3, 3] - poses[keep[-1], :3, 3])
            rotation_change = _rotation_angle(poses[keep[-1], :3, :3], poses[index, :3, :3])
            semantic_boundary = (
                ids[index] != ids[keep[-1]]
                or kinds[index] != kinds[keep[-1]]
                or layers[index] != layers[keep[-1]]
            )
            if semantic_boundary or translation_change > 1e-10 or rotation_change > 1e-9:
                keep.append(index)
        if len(keep) < 2:
            raise ValueError("所有 TCP 位姿都重复")
        return (
            poses[keep],
            np.asarray(kinds, dtype=object)[keep],
            ids[keep],
            layers[keep],
            np.asarray(keep, dtype=int),
        )

    @staticmethod
    def _arc_length(points: np.ndarray) -> np.ndarray:
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]

    def _make_curve(self, local_s: np.ndarray, poses: np.ndarray) -> _SegmentCurve:
        rotations = Rotation.from_matrix(poses[:, :3, :3])
        quaternions = rotations.as_quat()
        for index in range(1, len(quaternions)):
            if np.dot(quaternions[index - 1], quaternions[index]) < 0:
                quaternions[index] *= -1
        orientation_curve = Slerp(local_s, Rotation.from_quat(quaternions))
        degree = min(3, len(local_s) - 1)
        position_curve = make_interp_spline(local_s, poses[:, :3, 3], k=degree, axis=0)
        probe_s = np.linspace(local_s[0], local_s[-1], max(25, 10 * len(local_s)))
        max_deviation = max(
            _point_polyline_distance(np.asarray(position_curve(s)), poses[:, :3, 3])
            for s in probe_s
        )
        return _SegmentCurve(
            path_s=local_s,
            poses=poses,
            position_curve=position_curve,
            orientation_curve=orientation_curve,
            use_linear_position=max_deviation > self.parameters.chord_tolerance_m,
        )

    def _adaptive_grid(self, curve: _SegmentCurve, anchors: np.ndarray) -> np.ndarray:
        result = [float(anchors[0])]
        max_depth = 16

        def subdivide(a: float, b: float, depth: int) -> None:
            pa = curve.evaluate(a)
            pb = curve.evaluate(b)
            mid = 0.5 * (a + b)
            pm = curve.evaluate(mid)
            chord_mid = 0.5 * (pa[:3, 3] + pb[:3, 3])
            chord_error = np.linalg.norm(pm[:3, 3] - chord_mid)
            # Orientation is represented by exact spherical interpolation
            # (SLERP) between the source poses. Subdividing solely because the
            # endpoint rotation is larger than a display tolerance duplicates
            # the same continuous arc without adding information and used to
            # turn 900 sanding waypoints into tens of thousands of IK targets.
            # Keep the original poses as the orientation control polygon; only
            # a genuine positional chord error may refine the IK grid.
            if depth < max_depth and chord_error > self.parameters.chord_tolerance_m:
                subdivide(a, mid, depth + 1)
                subdivide(mid, b, depth + 1)
            else:
                result.append(float(b))

        for left, right in zip(anchors[:-1], anchors[1:]):
            subdivide(float(left), float(right), 0)
        return np.unique(np.asarray(result))
