"""全本地 TOPP-RA 时间参数化与 Ruckig jerk 后处理。"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from core.schemas import PathSegmentType, ProcessParameters, TimeParameterizedTrajectory
from .compression import select_adaptive_keyframes
from .geometry import unwrap_revolute_trajectory


class TrajectoryPlanningError(RuntimeError):
    """轨迹生成失败；调用方不得把失败结果作为已验证轨迹继续使用。"""


def _as_limit(values: Optional[np.ndarray], dof: int, name: str) -> np.ndarray:
    if values is None:
        raise TrajectoryPlanningError(f"缺少 {name}，请在机器人配置或 UI 中明确设置")
    array = np.asarray(values, dtype=float)
    if array.shape != (dof,) or np.any(array <= 0) or not np.all(np.isfinite(array)):
        raise TrajectoryPlanningError(f"{name} 必须是长度为 {dof} 的有限正数数组")
    return array


def _sample_times(duration: float, period: float) -> np.ndarray:
    if duration <= 0:
        raise TrajectoryPlanningError("轨迹时长必须为正数")
    sample_count = int(np.ceil(duration / period)) + 1
    if sample_count > 500_000:
        raise TrajectoryPlanningError(
            f"执行轨迹预计产生 {sample_count:,} 个采样点（{duration:.1f}s / {period:.4f}s），"
            "超过 500,000 点安全上限。请检查 IK 分支跳变、jerk 限制或增大控制周期。"
        )
    times = np.arange(0.0, duration, period)
    if len(times) == 0 or duration - times[-1] > 1e-12:
        times = np.r_[times, duration]
    else:
        times[-1] = duration
    return times


class TrajectoryPlanner:
    """从关节几何路径生成固定控制周期、物理时间一致的轨迹。"""

    def __init__(
        self,
        parameters: ProcessParameters,
        *,
        backend: str = "toppra",
        apply_ruckig: bool = True,
        max_retries: int = 4,
    ):
        if backend not in {"toppra", "auto", "spline_fallback"}:
            raise ValueError("backend 必须为 toppra、auto 或 spline_fallback")
        self.parameters = parameters
        self.backend = backend
        self.apply_ruckig = apply_ruckig
        self.max_retries = max_retries

    def plan(
        self,
        joint_waypoints: np.ndarray,
        *,
        tcp_waypoint_poses: Optional[np.ndarray] = None,
        segment_ids: Optional[np.ndarray] = None,
        segment_types: Optional[Sequence[PathSegmentType]] = None,
        transition_kinds: Optional[Sequence[Optional[str]]] = None,
        fk_provider: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        continuous_joint_mask: Optional[np.ndarray] = None,
        initial_joint_position: Optional[np.ndarray] = None,
    ) -> TimeParameterizedTrajectory:
        joint_waypoints = np.asarray(joint_waypoints, dtype=float)
        tcp_input = None if tcp_waypoint_poses is None else np.asarray(tcp_waypoint_poses, dtype=float)
        ids_input = None if segment_ids is None else np.asarray(segment_ids, dtype=int)
        kinds_input = None if segment_types is None else np.asarray(segment_types, dtype=object)
        transition_kinds_input = (
            None if transition_kinds is None else np.asarray(transition_kinds, dtype=object)
        )
        initial_transition_added = False
        if initial_joint_position is not None:
            if joint_waypoints.ndim != 2 or joint_waypoints.shape[1] == 0:
                raise TrajectoryPlanningError("joint_waypoints must be a non-empty 2D array")
            initial_q = np.asarray(initial_joint_position, dtype=float)
            if initial_q.shape != (joint_waypoints.shape[1],):
                raise TrajectoryPlanningError("initial_joint_position 与关节自由度不一致")
            if np.linalg.norm(initial_q - joint_waypoints[0]) > 1e-8:
                raise TrajectoryPlanningError(
                    "initial_joint_position no longer creates a direct edge; "
                    "assemble a validated stable2C transition before time parameterization"
                )

        q = joint_waypoints.copy()
        if continuous_joint_mask is not None:
            mask = np.asarray(continuous_joint_mask, dtype=bool)
            if mask.shape != (q.shape[1],):
                raise TrajectoryPlanningError("continuous_joint_mask 长度与关节自由度不一致")
            q[:, mask] = unwrap_revolute_trajectory(q[:, mask])
        if q.ndim != 2 or len(q) < 2 or not np.all(np.isfinite(q)):
            raise TrajectoryPlanningError("joint_waypoints 必须为至少两行的有限二维数组")
        q, retained = self._remove_duplicate_waypoints(q)
        if len(q) < 2:
            raise TrajectoryPlanningError("关节路径清理后不足两个不同路点")
        dof = q.shape[1]
        velocity_limits = _as_limit(self.parameters.max_joint_velocity, dof, "最大关节速度")
        acceleration_limits = _as_limit(self.parameters.max_joint_acceleration, dof, "最大关节加速度")
        jerk_limits = _as_limit(self.parameters.max_joint_jerk, dof, "最大关节 jerk")

        tcp_waypoints = None
        if tcp_waypoint_poses is not None:
            tcp_all = tcp_input
            if tcp_all.shape[0] != len(joint_waypoints) or tcp_all.shape[1:] != (4, 4):
                raise TrajectoryPlanningError("tcp_waypoint_poses 与关节路点数量不一致")
            tcp_waypoints = tcp_all[retained]
        ids_all = np.zeros(len(joint_waypoints), dtype=int) if ids_input is None else ids_input
        kinds_all = (
            np.full(len(joint_waypoints), PathSegmentType.PROCESS, dtype=object)
            if kinds_input is None
            else kinds_input
        )
        transition_kinds_all = (
            np.full(len(joint_waypoints), None, dtype=object)
            if transition_kinds_input is None
            else transition_kinds_input
        )
        if (
            ids_all.shape != (len(joint_waypoints),)
            or kinds_all.shape != (len(joint_waypoints),)
            or transition_kinds_all.shape != (len(joint_waypoints),)
        ):
            raise TrajectoryPlanningError("逐路点语义长度错误")
        ids = ids_all[retained]
        kinds = kinds_all[retained]
        waypoint_transition_kinds = transition_kinds_all[retained]
        kinds = np.asarray([
            item if isinstance(item, PathSegmentType) else PathSegmentType(str(item))
            for item in kinds
        ], dtype=object)
        branch_edges = np.array([], dtype=int)
        q, tcp_waypoints, ids, kinds, waypoint_transition_kinds, refinement_count = self._refine_process_joint_path(
            q, tcp_waypoints, ids, kinds, waypoint_transition_kinds, fk_provider,
            tolerance_m=self.parameters.chord_tolerance_m * 0.5,
        )

        method = ""
        try:
            if self.backend == "spline_fallback":
                raise ImportError("显式使用 fallback")
            base = self._retime_with_toppra(q, velocity_limits, acceleration_limits)
            method = "toppra"
        except (ImportError, ModuleNotFoundError) as exc:
            if self.backend == "toppra":
                raise TrajectoryPlanningError(
                    "TOPP-RA 未安装，生产轨迹生成已停止；请安装锁定依赖 toppra==0.6.3"
                ) from exc
            base = self._retime_with_spline(q, velocity_limits, acceleration_limits)
            method = "constraint_scaled_cubic_fallback"

        timestamps, positions, velocities, accelerations = base
        if self.apply_ruckig:
            try:
                piece_ranges, stop_edges = self._build_jerk_pieces(q, ids)
                if len(piece_ranges) == 1:
                    timestamps, positions, velocities, accelerations = self._smooth_with_ruckig(
                        timestamps, positions, velocities, accelerations,
                        velocity_limits, acceleration_limits, jerk_limits,
                        knot_positions=q,
                    )
                    method += "+ruckig"
                else:
                    pieces = []
                    for start, stop in piece_ranges:
                        local_q = q[start:stop]
                        local_base = self._retime_with_toppra(
                            local_q, velocity_limits, acceleration_limits
                        )
                        pieces.append(self._smooth_with_ruckig(
                            *local_base,
                            velocity_limits, acceleration_limits, jerk_limits,
                            knot_positions=local_q,
                        ))
                    timestamps, positions, velocities, accelerations = self._join_pieces(pieces)
                    method += "+ruckig_piecewise_stops"
            except (ImportError, ModuleNotFoundError) as exc:
                if self.backend == "toppra":
                    raise TrajectoryPlanningError(
                        "Ruckig 未安装，jerk 受限后处理已停止；请安装锁定依赖 ruckig==0.19.4"
                    ) from exc
                method += "+no_ruckig"

        jerks = np.gradient(accelerations, timestamps, axis=0, edge_order=1)
        tcp_poses = self._compute_tcp_poses(
            positions, q, tcp_waypoints=tcp_waypoints, fk_provider=fk_provider
        )
        tcp_speeds = self._tcp_speeds(tcp_poses, timestamps)
        sample_ids, sample_kinds, sample_transition_kinds = self._map_waypoint_metadata(
            positions, q, ids, kinds, waypoint_transition_kinds
        )

        # TCP 进给是加工段硬上限。通过统一时间缩放保留几何路径和连续性。
        process_mask = sample_kinds == PathSegmentType.PROCESS
        if np.any(process_mask):
            peak = float(np.max(tcp_speeds[process_mask]))
            if peak > self.parameters.tcp_feed_rate_mps * (1.0 + 1e-9):
                # 留出小幅数值裕量，避免固定周期重采样的线性插值/数值微分
                # 把理论峰值推到硬上限之外。
                scale = (peak / self.parameters.tcp_feed_rate_mps) * 1.002
                scaled_times = timestamps * scale
                timestamps, positions, velocities, accelerations = self._resample_samples(
                    scaled_times,
                    positions,
                    velocities / scale,
                    accelerations / (scale * scale),
                    self.parameters.control_period_s,
                )
                jerks = np.gradient(accelerations, timestamps, axis=0, edge_order=1)
                tcp_poses = self._compute_tcp_poses(
                    positions, q, tcp_waypoints=tcp_waypoints, fk_provider=fk_provider
                )
                tcp_speeds = self._tcp_speeds(tcp_poses, timestamps)
                sample_ids, sample_kinds, sample_transition_kinds = self._map_waypoint_metadata(
                    positions, q, ids, kinds, waypoint_transition_kinds
                )
                method += "+tcp_feed_scaled"

        dense_sample_count = len(timestamps)
        compression = None
        if self.parameters.adaptive_keyframes and dense_sample_count > 2:
            compression = select_adaptive_keyframes(
                timestamps, positions, velocities, accelerations, jerks,
                joint_tolerance_rad=self.parameters.joint_keyframe_tolerance_rad,
                max_interval_s=self.parameters.max_keyframe_interval_s,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                jerk_limits=jerk_limits,
                tcp_poses=tcp_poses,
                chord_tolerance_m=self.parameters.chord_tolerance_m,
                orientation_tolerance_rad=self.parameters.orientation_tolerance_rad,
                segment_ids=sample_ids,
                segment_types=sample_kinds,
            )
            selected = compression.indices
            timestamps = timestamps[selected]
            positions = positions[selected]
            velocities = velocities[selected]
            accelerations = accelerations[selected]
            jerks = jerks[selected]
            tcp_poses = None if tcp_poses is None else tcp_poses[selected]
            tcp_speeds = tcp_speeds[selected]
            sample_ids = sample_ids[selected]
            sample_kinds = sample_kinds[selected]
            sample_transition_kinds = sample_transition_kinds[selected]
            method += "+adaptive_keyframes"

        force_channel = np.full(len(timestamps), self.parameters.normal_force_setpoint_n)
        feed_channel = np.where(
            sample_kinds == PathSegmentType.PROCESS,
            self.parameters.tcp_feed_rate_mps,
            self.parameters.tcp_feed_rate_mps * self.parameters.rapid_speed_ratio,
        )
        return TimeParameterizedTrajectory(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
            tcp_poses=tcp_poses,
            tcp_speeds_mps=tcp_speeds,
            segment_ids=sample_ids,
            segment_types=sample_kinds,
            transition_kinds=sample_transition_kinds,
            process_channels={
                "tcp_feed_setpoint_mps": feed_channel,
                "normal_force_setpoint_n": force_channel,
            },
            method=method,
            metadata={
                "control_period_s": self.parameters.control_period_s,
                "backend": self.backend,
                "local_only": True,
                "ruckig_intermediate_waypoints": False,
                "jerk_stop_boundaries": stop_edges.tolist() if self.apply_ruckig else [],
                "ik_branch_transition_edges": branch_edges.tolist(),
                "tcp_error_refinement_points": refinement_count,
                "initial_transition_added": initial_transition_added,
                "dense_sample_count": dense_sample_count,
                "keyframe_count": len(timestamps),
                "compression_ratio": dense_sample_count / len(timestamps),
                "keyframe_interpolation": "quintic_hermite" if compression is not None else "linear",
                "max_keyframe_interval_s": (
                    compression.max_interval_s if compression is not None else self.parameters.control_period_s
                ),
                "max_joint_reconstruction_error_rad": (
                    compression.max_joint_error_rad if compression is not None else 0.0
                ),
                "max_tcp_reconstruction_error_m": (
                    compression.max_tcp_chord_error_m if compression is not None else 0.0
                ),
                "max_orientation_reconstruction_error_rad": (
                    compression.max_orientation_error_rad if compression is not None else 0.0
                ),
            },
        )

    @staticmethod
    def _label_branch_transitions(q, segment_ids, segment_types):
        """Promote large IK branch changes to explicit non-process transitions."""
        if len(q) < 2:
            return segment_ids, segment_types, np.array([], dtype=int)
        deltas = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        positive = deltas[deltas > 1e-10]
        typical = float(np.median(positive)) if positive.size else 0.0
        edges = np.flatnonzero(deltas > max(0.75, typical * 8.0))
        if not len(edges):
            return segment_ids, segment_types, edges
        ids = np.asarray(segment_ids, dtype=int).copy()
        kinds = np.asarray(segment_types, dtype=object).copy()
        next_id = int(np.max(ids)) + 1
        for offset, edge in enumerate(edges):
            edge = int(edge)
            transition_id = next_id + offset
            transition_start = max(0, edge - 1)
            ids[transition_start:edge + 2] = transition_id
            kinds[transition_start:edge + 1] = PathSegmentType.RETRACT
            kinds[edge + 1] = PathSegmentType.RAPID
            if edge + 2 < len(kinds) and kinds[edge + 2] == PathSegmentType.PROCESS:
                kinds[edge + 2] = PathSegmentType.APPROACH
        return ids, kinds, edges

    @staticmethod
    def _refine_process_joint_path(
        q, tcp_waypoints, segment_ids, segment_types, transition_kinds, fk_provider,
        *, tolerance_m: float, max_depth: int = 7, max_points: int = 20_000,
    ):
        """Locally refine joint chords whose FK midpoint violates TCP chord error."""
        if fk_provider is None or len(q) < 2:
            return q, tcp_waypoints, segment_ids, segment_types, transition_kinds, 0
        q = np.asarray(q, dtype=float)
        poses = np.asarray([fk_provider(item) for item in q], dtype=float)
        output_q = [q[0]]
        output_pose = [poses[0]]
        output_ids = [int(segment_ids[0])]
        output_kinds = [segment_types[0]]
        output_transition_kinds = [transition_kinds[0]]

        def point_segment_distance(point, start, stop):
            vector = stop - start
            denom = float(np.dot(vector, vector))
            if denom <= 1e-20:
                return float(np.linalg.norm(point - start))
            ratio = float(np.clip(np.dot(point - start, vector) / denom, 0.0, 1.0))
            return float(np.linalg.norm(point - (start + ratio * vector)))

        def append_interval(
            qa, qb, pose_a, pose_b, edge_id, edge_kind, edge_transition_kind, depth
        ):
            if len(output_q) >= max_points:
                output_q.append(qb)
                output_pose.append(pose_b)
                output_ids.append(edge_id)
                output_kinds.append(edge_kind)
                output_transition_kinds.append(edge_transition_kind)
                return
            midpoint_q = 0.5 * (qa + qb)
            midpoint_pose = np.asarray(fk_provider(midpoint_q), dtype=float)
            error = point_segment_distance(
                midpoint_pose[:3, 3], pose_a[:3, 3], pose_b[:3, 3]
            )
            # Four baseline bisections suppress global cubic-spline ringing;
            # deeper refinement remains strictly driven by measured TCP error.
            if depth < max_depth and (depth < 4 or error > tolerance_m):
                append_interval(
                    qa, midpoint_q, pose_a, midpoint_pose, edge_id, edge_kind,
                    edge_transition_kind, depth + 1
                )
                append_interval(
                    midpoint_q, qb, midpoint_pose, pose_b, edge_id, edge_kind,
                    edge_transition_kind, depth + 1
                )
            else:
                output_q.append(qb)
                output_pose.append(pose_b)
                output_ids.append(edge_id)
                output_kinds.append(edge_kind)
                output_transition_kinds.append(edge_transition_kind)

        for edge in range(len(q) - 1):
            left_kind = segment_types[edge]
            right_kind = segment_types[edge + 1]
            is_process = (
                left_kind == PathSegmentType.PROCESS
                and right_kind == PathSegmentType.PROCESS
                and segment_ids[edge] == segment_ids[edge + 1]
            )
            if is_process:
                append_interval(
                    q[edge], q[edge + 1], poses[edge], poses[edge + 1],
                    int(segment_ids[edge]), PathSegmentType.PROCESS,
                    transition_kinds[edge], 0,
                )
            else:
                output_q.append(q[edge + 1])
                output_pose.append(poses[edge + 1])
                output_ids.append(int(segment_ids[edge + 1]))
                output_kinds.append(right_kind)
                output_transition_kinds.append(transition_kinds[edge + 1])
        refined_q = np.asarray(output_q)
        return (
            refined_q,
            np.asarray(output_pose) if tcp_waypoints is not None else None,
            np.asarray(output_ids, dtype=int),
            np.asarray(output_kinds, dtype=object),
            np.asarray(output_transition_kinds, dtype=object),
            int(len(refined_q) - len(q)),
        )

    @staticmethod
    def _build_jerk_pieces(q: np.ndarray, segment_ids: np.ndarray):
        """在语义边界或异常关节跳变处插入停靠，避免局部尖点拖慢整条路径。"""
        deltas = np.linalg.norm(np.diff(q, axis=0), axis=1)
        positive = deltas[deltas > 1e-10]
        typical = float(np.median(positive)) if positive.size else 0.0
        jump_threshold = max(0.75, typical * 8.0)
        stop_edges = np.flatnonzero(
            (segment_ids[1:] != segment_ids[:-1]) | (deltas > jump_threshold)
        )
        if not len(stop_edges):
            return [(0, len(q))], stop_edges

        # 每条突变边单独作为 rest-to-rest 过渡段；其前后连续加工段也以零速
        # 接入，避免在尖角处强行保持非零速度/加速度。
        ranges = []
        cursor = 0
        for edge in stop_edges:
            edge = int(edge)
            if edge + 1 - cursor >= 2:
                ranges.append((cursor, edge + 1))
            ranges.append((edge, edge + 2))
            cursor = edge + 1
        if len(q) - cursor >= 2:
            ranges.append((cursor, len(q)))
        return ranges, stop_edges

    @staticmethod
    def _join_pieces(pieces):
        output_t, output_q, output_qd, output_qdd = [], [], [], []
        time_offset = 0.0
        for piece_index, (times, q, qd, qdd) in enumerate(pieces):
            start = 0 if piece_index == 0 else 1
            output_t.extend((times[start:] + time_offset).tolist())
            output_q.extend(q[start:])
            output_qd.extend(qd[start:])
            output_qdd.extend(qdd[start:])
            time_offset += float(times[-1])
        return tuple(map(np.asarray, (output_t, output_q, output_qd, output_qdd)))

    @staticmethod
    def _remove_duplicate_waypoints(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-12]
        retained = np.flatnonzero(keep)
        return q[retained], retained

    def _retime_with_toppra(self, q, velocity_limits, acceleration_limits):
        import toppra as ta
        import toppra.algorithm as algo
        import toppra.constraint as constraint

        delta = np.linalg.norm(np.diff(q, axis=0), axis=1)
        path_s = np.r_[0.0, np.cumsum(delta)]
        path_s /= path_s[-1]
        path = ta.SplineInterpolator(path_s, q)
        constraints = [
            constraint.JointVelocityConstraint(np.c_[-velocity_limits, velocity_limits]),
            constraint.JointAccelerationConstraint(
                np.c_[-acceleration_limits, acceleration_limits],
                discretization_scheme=constraint.DiscretizationType.Interpolation,
            ),
        ]
        instance = algo.TOPPRA(
            constraints,
            path,
            parametrizer="ParametrizeSpline",
            gridpt_max_err_threshold=1e-3,
            gridpt_min_nb_points=max(100, len(q) * 3),
        )
        trajectory = instance.compute_trajectory(0.0, 0.0)
        if trajectory is None:
            raise TrajectoryPlanningError("TOPP-RA 无法在当前约束下完成时间参数化")
        times = _sample_times(float(trajectory.duration), self.parameters.control_period_s)
        positions = np.asarray(trajectory(times, 0))
        velocities = np.asarray(trajectory(times, 1))
        accelerations = np.asarray(trajectory(times, 2))
        return times, positions, velocities, accelerations

    def _retime_with_spline(self, q, velocity_limits, acceleration_limits):
        differences = np.abs(np.diff(q, axis=0))
        velocity_time = np.max(differences / velocity_limits, axis=1)
        acceleration_time = np.max(np.sqrt(6.0 * differences / acceleration_limits), axis=1)
        durations = np.maximum.reduce([
            velocity_time * 1.8,
            acceleration_time,
            np.full(len(differences), self.parameters.control_period_s * 2.0),
        ])
        knots = np.r_[0.0, np.cumsum(durations)]
        for _ in range(self.max_retries + 1):
            spline = CubicSpline(knots, q, axis=0, bc_type=((1, np.zeros(q.shape[1])), (1, np.zeros(q.shape[1]))))
            times = _sample_times(float(knots[-1]), self.parameters.control_period_s)
            positions = spline(times, 0)
            velocities = spline(times, 1)
            accelerations = spline(times, 2)
            v_ratio = float(np.max(np.abs(velocities) / velocity_limits))
            a_ratio = float(np.max(np.sqrt(np.abs(accelerations) / acceleration_limits)))
            scale = max(1.0, v_ratio, a_ratio)
            if scale <= 1.000001:
                return times, positions, velocities, accelerations
            knots *= scale * 1.02
        raise TrajectoryPlanningError("约束缩放样条在最大重试次数内未收敛")

    def _smooth_with_ruckig(self, times, q, qd, qdd, vmax, amax, jmax, *, knot_positions):
        from ruckig import InputParameter, Ruckig, Trajectory

        dof = q.shape[1]
        knots = np.asarray(knot_positions, dtype=float)
        distances = np.linalg.norm(np.diff(knots, axis=0), axis=1)
        path_s = np.r_[0.0, np.cumsum(distances)]
        if path_s[-1] <= 1e-12:
            raise TrajectoryPlanningError("Ruckig 输入路径长度为零")
        path_s /= path_s[-1]
        path = CubicSpline(path_s, knots, axis=0, bc_type="natural")

        # 只让本地 Ruckig 规划单一的全路径进度 s(t)，避免对数百个微段逐段
        # state-to-state 时累计 jerk 边界开销（这正是数小时假轨迹的根因）。
        probe_s = np.linspace(0.0, 1.0, min(10_000, max(500, len(knots) * 5)))
        d1 = np.abs(path(probe_s, 1))
        d2 = np.abs(path(probe_s, 2))
        d3 = np.abs(path(probe_s, 3))
        eps = 1e-12
        progress_vmax = float(np.min(vmax[None, :] / np.maximum(d1, eps)))
        # 曲率项 q_ss*s_dot^2 和 q_sss*s_dot^3 各预留一半预算。
        curve_a_vmax = float(np.min(np.sqrt(0.5 * amax[None, :] / np.maximum(d2, eps))))
        curve_j_vmax = float(np.min(np.cbrt(0.5 * jmax[None, :] / np.maximum(d3, eps))))
        progress_vmax = max(eps, min(progress_vmax, curve_a_vmax, curve_j_vmax))
        progress_amax = max(eps, float(np.min(0.45 * amax[None, :] / np.maximum(d1, eps))))
        progress_jmax = max(eps, float(np.min(0.35 * jmax[None, :] / np.maximum(d1, eps))))

        otg = Ruckig(1)
        inp = InputParameter(1)
        inp.current_position = [0.0]
        inp.current_velocity = [0.0]
        inp.current_acceleration = [0.0]
        inp.target_position = [1.0]
        inp.target_velocity = [0.0]
        inp.target_acceleration = [0.0]
        inp.max_velocity = [progress_vmax]
        inp.max_acceleration = [progress_amax]
        inp.max_jerk = [progress_jmax]

        requested_duration = float(times[-1])
        last_result = None
        for retry in range(self.max_retries + 2):
            inp.minimum_duration = requested_duration
            scalar_trajectory = Trajectory(1)
            last_result = otg.calculate(inp, scalar_trajectory)
            if int(last_result) < 0:
                requested_duration *= 1.5
                continue
            if scalar_trajectory.duration > float(times[-1]) * 25.0:
                raise TrajectoryPlanningError(
                    f"Ruckig 将轨迹从 {times[-1]:.2f}s 膨胀到 {scalar_trajectory.duration:.2f}s，"
                    "已中止以防止生成超大轨迹；请检查 IK 跳变或 jerk 配置。"
                )
            output_t = _sample_times(
                float(scalar_trajectory.duration), self.parameters.control_period_s
            )
            states = [scalar_trajectory.at_time(float(t)) for t in output_t]
            progress = np.asarray([state[0][0] for state in states])
            progress_d = np.asarray([state[1][0] for state in states])
            progress_dd = np.asarray([state[2][0] for state in states])
            output_q = path(progress, 0)
            path_d1 = path(progress, 1)
            path_d2 = path(progress, 2)
            output_qd = path_d1 * progress_d[:, None]
            output_qdd = (
                path_d2 * (progress_d * progress_d)[:, None]
                + path_d1 * progress_dd[:, None]
            )
            output_jerk = np.gradient(output_qdd, output_t, axis=0, edge_order=1)
            ratios = (
                float(np.max(np.abs(output_qd) / vmax[None, :])),
                float(np.max(np.abs(output_qdd) / amax[None, :])),
                float(np.max(np.abs(output_jerk) / jmax[None, :])),
            )
            worst_time_scale = max(ratios[0], np.sqrt(ratios[1]), np.cbrt(ratios[2]))
            if worst_time_scale <= 1.00001:
                return output_t, output_q, output_qd, output_qdd
            requested_duration = float(scalar_trajectory.duration) * worst_time_scale * 1.02

        raise TrajectoryPlanningError(
            f"Ruckig 全路径进度规划经 {self.max_retries + 2} 次受控降速仍未满足约束，"
            f"错误码 {int(last_result) if last_result is not None else 'unknown'}"
        )

    @staticmethod
    def _resample_state(times, positions, period):
        times = np.asarray(times, dtype=float)
        positions = np.asarray(positions, dtype=float)
        new_times = _sample_times(float(times[-1]), period)
        spline = CubicSpline(times, positions, axis=0)
        return new_times, spline(new_times, 0), spline(new_times, 1), spline(new_times, 2)

    @staticmethod
    def _resample_samples(times, positions, velocities, accelerations, period):
        new_times = _sample_times(float(times[-1]), period)
        def interpolate(values):
            return np.column_stack([
                np.interp(new_times, times, values[:, joint])
                for joint in range(values.shape[1])
            ])
        return (
            new_times,
            interpolate(np.asarray(positions)),
            interpolate(np.asarray(velocities)),
            interpolate(np.asarray(accelerations)),
        )

    @staticmethod
    def _compute_tcp_poses(positions, waypoints, *, tcp_waypoints, fk_provider):
        if fk_provider is not None:
            poses = np.asarray([fk_provider(q) for q in positions], dtype=float)
            if poses.shape != (len(positions), 4, 4):
                raise TrajectoryPlanningError("fk_provider 必须返回 4x4 TCP 位姿")
            return poses
        if tcp_waypoints is None:
            return None
        progress_waypoints = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))]
        progress_samples = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        if progress_samples[-1] > 0:
            progress_samples *= progress_waypoints[-1] / progress_samples[-1]
        from scipy.spatial.transform import Rotation, Slerp
        poses = np.repeat(np.eye(4)[None, :, :], len(positions), axis=0)
        for axis in range(3):
            poses[:, axis, 3] = np.interp(progress_samples, progress_waypoints, tcp_waypoints[:, axis, 3])
        rotations = Rotation.from_matrix(tcp_waypoints[:, :3, :3])
        poses[:, :3, :3] = Slerp(progress_waypoints, rotations)(progress_samples).as_matrix()
        return poses

    @staticmethod
    def _tcp_speeds(tcp_poses, timestamps):
        if tcp_poses is None:
            return np.zeros(len(timestamps))
        positions = tcp_poses[:, :3, 3]
        velocity = np.gradient(positions, timestamps, axis=0, edge_order=1)
        return np.linalg.norm(velocity, axis=1)

    @staticmethod
    def _map_waypoint_metadata(samples, waypoints, ids, kinds, transition_kinds):
        waypoint_progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))]
        sample_progress = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(samples, axis=0), axis=1))]
        if sample_progress[-1] > 0:
            sample_progress *= waypoint_progress[-1] / sample_progress[-1]
        indices = np.searchsorted(waypoint_progress, sample_progress, side="right") - 1
        indices = np.clip(indices, 0, len(waypoints) - 1)
        return (
            ids[indices].astype(int),
            kinds[indices].astype(object),
            transition_kinds[indices].astype(object),
        )
