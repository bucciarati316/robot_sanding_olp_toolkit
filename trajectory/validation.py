"""物理轨迹硬约束、路径偏差与碰撞验证。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from core.schemas import (
    PathSegmentType,
    ProcessParameters,
    TimeParameterizedTrajectory,
    TrajectoryValidationItem,
    TrajectoryValidationReport,
)


def _worst_ratio(values: np.ndarray, limits: np.ndarray) -> tuple[float, tuple[int, int]]:
    ratio = np.abs(values) / limits[None, :]
    flat = int(np.argmax(ratio))
    return float(ratio.flat[flat]), np.unravel_index(flat, ratio.shape)


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    denom = np.einsum("ij,ij->i", vectors, vectors)
    alpha = np.zeros(len(vectors))
    valid = denom > 1e-20
    alpha[valid] = np.einsum("ij,ij->i", point - starts, vectors)[valid] / denom[valid]
    projection = starts + np.clip(alpha, 0.0, 1.0)[:, None] * vectors
    return float(np.min(np.linalg.norm(projection - point, axis=1)))


class TrajectoryValidator:
    def __init__(self, parameters: ProcessParameters, *, relative_tolerance: float = 1e-5):
        self.parameters = parameters
        self.relative_tolerance = relative_tolerance

    def validate(
        self,
        trajectory: TimeParameterizedTrajectory,
        *,
        lower_position_limits: Optional[np.ndarray] = None,
        upper_position_limits: Optional[np.ndarray] = None,
        reference_tcp_positions: Optional[np.ndarray] = None,
        collision_free: Optional[Callable[[np.ndarray], bool]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> TrajectoryValidationReport:
        items: list[TrajectoryValidationItem] = []
        def progress(current: int, total: int, stage: str) -> None:
            if is_cancelled is not None and is_cancelled():
                raise TrajectoryValidationCancelled("用户取消了轨迹验证")
            if progress_callback is not None:
                progress_callback(current, total, stage)

        progress(0, 100, "时间戳与数组结构")
        items.append(TrajectoryValidationItem(
            name="timestamps",
            passed=bool(trajectory.timestamps[0] == 0 and np.all(np.diff(trajectory.timestamps) > 0)),
            message="时间戳从零开始且严格递增",
        ))
        self._validate_dynamic_limit(
            items, "joint_velocity", trajectory.velocities,
            self.parameters.max_joint_velocity, trajectory.timestamps,
        )
        progress(20, 100, "关节动力学约束")
        self._validate_dynamic_limit(
            items, "joint_acceleration", trajectory.accelerations,
            self.parameters.max_joint_acceleration, trajectory.timestamps,
        )
        self._validate_dynamic_limit(
            items, "joint_jerk", trajectory.jerks,
            self.parameters.max_joint_jerk, trajectory.timestamps,
        )
        self._validate_joint_positions(
            items, trajectory, lower_position_limits, upper_position_limits
        )
        progress(35, 100, "关节限位与安全裕度")
        self._validate_tcp_feed(items, trajectory)
        progress(50, 100, "TCP 进给速度")
        self._validate_path_deviation(items, trajectory, reference_tcp_positions)
        progress(60, 100, "几何路径偏差")
        self._validate_collision(
            items, trajectory, collision_free,
            progress_callback=lambda current, total: progress(
                60 + int(40 * current / max(total, 1)), 100, "碰撞区间检查"
            ),
            is_cancelled=is_cancelled,
        )
        progress(100, 100, "验证完成")
        return TrajectoryValidationReport(
            items=items,
            generated_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "method": trajectory.method,
                "duration_s": trajectory.duration_s,
                "sample_count": len(trajectory.timestamps),
            },
        )

    def _validate_dynamic_limit(self, items, name, values, limits, timestamps):
        if limits is None:
            items.append(TrajectoryValidationItem(
                name=name,
                passed=False,
                message="缺少显式限制，无法验证",
            ))
            return
        limits = np.asarray(limits, dtype=float)
        ratio, (time_index, joint_index) = _worst_ratio(values, limits)
        measured = float(abs(values[time_index, joint_index]))
        limit = float(limits[joint_index])
        items.append(TrajectoryValidationItem(
            name=name,
            passed=ratio <= 1.0 + self.relative_tolerance,
            measured=measured,
            limit=limit,
            time_s=float(timestamps[time_index]),
            joint_index=int(joint_index),
            message=f"最大利用率 {ratio * 100:.2f}%",
        ))

    def _validate_joint_positions(self, items, trajectory, lower, upper):
        if lower is None or upper is None:
            items.append(TrajectoryValidationItem(
                name="joint_position_margin",
                passed=False,
                message="缺少关节位置上下限，无法验证安全裕度",
            ))
            return
        lower = np.asarray(lower, dtype=float) + self.parameters.minimum_joint_margin_rad
        upper = np.asarray(upper, dtype=float) - self.parameters.minimum_joint_margin_rad
        q = trajectory.positions
        violation = np.maximum(lower[None, :] - q, q - upper[None, :])
        flat = int(np.argmax(violation))
        time_index, joint_index = np.unravel_index(flat, violation.shape)
        worst = float(violation[time_index, joint_index])
        margin = np.minimum(q - lower[None, :], upper[None, :] - q)
        items.append(TrajectoryValidationItem(
            name="joint_position_margin",
            passed=worst <= 1e-12,
            measured=float(np.min(margin)),
            limit=0.0,
            time_s=float(trajectory.timestamps[time_index]),
            joint_index=int(joint_index),
            message="测量值为应用安全裕度后的最小剩余裕度(rad)",
        ))

    def _validate_tcp_feed(self, items, trajectory):
        mask = trajectory.segment_types == PathSegmentType.PROCESS
        if not np.any(mask):
            items.append(TrajectoryValidationItem(
                name="tcp_feed_rate",
                passed=True,
                hard_constraint=False,
                message="轨迹中没有加工段",
            ))
            return
        masked_indices = np.flatnonzero(mask)
        local = int(np.argmax(trajectory.tcp_speeds_mps[mask]))
        index = int(masked_indices[local])
        peak = float(trajectory.tcp_speeds_mps[index])
        limit = self.parameters.tcp_feed_rate_mps
        items.append(TrajectoryValidationItem(
            name="tcp_feed_rate",
            passed=peak <= limit * (1.0 + self.relative_tolerance),
            measured=peak,
            limit=limit,
            time_s=float(trajectory.timestamps[index]),
            segment_id=int(trajectory.segment_ids[index]),
            message="仅对 PROCESS 加工段施加进给速度上限",
        ))

    def _validate_path_deviation(self, items, trajectory, reference):
        if reference is None or trajectory.tcp_poses is None:
            items.append(TrajectoryValidationItem(
                name="tcp_path_deviation",
                passed=False,
                message="未提供参考 TCP 路径，无法完成几何硬约束验证",
            ))
            return
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
            raise ValueError("reference_tcp_positions 必须为 (N,3)")
        # Approach/retract/rapid transitions are intentionally outside the
        # machining reference. Applying the chord tolerance to them makes every
        # explicit start transition fail even when all PROCESS samples are exact.
        process_indices = np.flatnonzero(
            trajectory.segment_types == PathSegmentType.PROCESS
        )
        if not len(process_indices):
            items.append(TrajectoryValidationItem(
                name="tcp_path_deviation", passed=True, hard_constraint=False,
                message="轨迹中没有加工段",
            ))
            return
        deviations = np.asarray([
            _distance_to_polyline(trajectory.tcp_poses[index, :3, 3], reference)
            for index in process_indices
        ])
        local_index = int(np.argmax(deviations))
        index = int(process_indices[local_index])
        peak = float(deviations[local_index])
        items.append(TrajectoryValidationItem(
            name="tcp_path_deviation",
            passed=peak <= self.parameters.chord_tolerance_m,
            measured=peak,
            limit=self.parameters.chord_tolerance_m,
            time_s=float(trajectory.timestamps[index]),
            segment_id=int(trajectory.segment_ids[index]),
            message="仅对 PROCESS 加工段施加弦误差上限",
        ))

    def _validate_collision(
        self, items, trajectory, collision_free,
        progress_callback=None, is_cancelled=None,
    ):
        if collision_free is None:
            items.append(TrajectoryValidationItem(
                name="collision",
                passed=False,
                message="未连接碰撞服务，无法完成碰撞硬约束验证",
            ))
            return
        total = len(trajectory.positions)
        progress_step = max(1, total // 200)

        def check(q, segment_type) -> bool:
            # The legacy callback contract is ``callback(q)``.  The new D
            # collision service declares this opt-in marker so PROCESS-only
            # contact rules cannot accidentally be used for RAPID segments.
            if getattr(collision_free, "supports_segment_type", False):
                return bool(collision_free(q, segment_type=segment_type))
            return bool(collision_free(q))

        for index, q in enumerate(trajectory.positions):
            if is_cancelled is not None and is_cancelled():
                raise TrajectoryValidationCancelled("用户取消了碰撞验证")
            if progress_callback is not None and (index % progress_step == 0 or index + 1 == total):
                progress_callback(index + 1, total)
            segment_type = trajectory.segment_types[index]
            if not check(q, segment_type):
                items.append(TrajectoryValidationItem(
                    name="collision",
                    passed=False,
                    time_s=float(trajectory.timestamps[index]),
                    segment_id=int(trajectory.segment_ids[index]),
                    message="执行采样点发生碰撞",
                ))
                return
            if index + 1 < len(trajectory.positions):
                midpoint = 0.5 * (q + trajectory.positions[index + 1])
                if not check(midpoint, segment_type):
                    items.append(TrajectoryValidationItem(
                        name="collision",
                        passed=False,
                        time_s=float(0.5 * (trajectory.timestamps[index] + trajectory.timestamps[index + 1])),
                        segment_id=int(trajectory.segment_ids[index]),
                        message="相邻执行采样点之间发生碰撞",
                    ))
                    return
        items.append(TrajectoryValidationItem(
            name="collision", passed=True, message="采样点及区间中点均无碰撞"
        ))


class TrajectoryValidationCancelled(RuntimeError):
    """用户取消后台验证，不应被解释为规划或约束失败。"""
