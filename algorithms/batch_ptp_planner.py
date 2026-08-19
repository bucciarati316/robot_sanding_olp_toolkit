"""Offline multi-pose IK branch selection for surface-processing programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence
import time

import numpy as np
import pinocchio as pin

from algorithms.ik_SLSQP_update import SLSQPMultiSolver
from collision.collision_distance_service import CollisionDistanceService
from core.schemas import JointTrajectory


@dataclass(frozen=True)
class BatchPTPPlannerConfig:
    candidate_count: int = 4
    max_solver_attempts_per_pose: int = 12
    roll_candidates_deg: tuple[float, ...] = (-180.0, -90.0, 0.0, 90.0, 180.0)
    safety_margin: float = 0.03
    pose_position_tolerance: float = 0.003
    pose_rotation_tolerance: float = 0.03
    window_size: int = 256
    overlap: int = 8
    max_joint_velocity: float = 1.0
    max_joint_acceleration: float = 2.0
    random_seed: int = 41


@dataclass(frozen=True)
class BatchPTPPlanResult:
    success: bool
    trajectory: Optional[JointTrajectory]
    selected_candidate_indices: np.ndarray
    candidate_counts: np.ndarray
    failure_index: Optional[int]
    failure_reason: Optional[str]
    minimum_sdf_distance: float
    statistics: dict[str, float] = field(default_factory=dict)


class BatchPTPPlanner:
    """Select a continuous joint trajectory instead of solving each pose greedily."""

    def __init__(
        self,
        solver: SLSQPMultiSolver,
        *,
        collision_service: Optional[CollisionDistanceService] = None,
        config: Optional[BatchPTPPlannerConfig] = None,
    ):
        self.solver = solver
        self.kinematics = solver._kin_engine
        self.collision_service = collision_service
        self.config = config or BatchPTPPlannerConfig()

    def plan(
        self,
        tcp_poses: np.ndarray,
        *,
        segment_modes: Optional[Sequence[str]] = None,
        initial_q: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[float, int, str], None]] = None,
    ) -> BatchPTPPlanResult:
        poses = np.asarray(tcp_poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("tcp_poses must have shape (N, 4, 4)")
        if len(poses) == 0:
            return BatchPTPPlanResult(True, JointTrajectory(np.empty((0, self.kinematics.nq))), np.empty(0, int), np.empty(0, int), None, None, float("inf"))
        modes = list(segment_modes or ["process"] * len(poses))
        if len(modes) != len(poses):
            raise ValueError("segment_modes must match tcp_poses")
        lower = np.asarray(self.kinematics.q_min, dtype=np.float64)
        upper = np.asarray(self.kinematics.q_max, dtype=np.float64)
        rng = np.random.default_rng(self.config.random_seed)
        seed = np.asarray(initial_q if initial_q is not None else np.zeros(self.kinematics.nq), dtype=np.float64)
        candidates: list[list[np.ndarray]] = []
        warm_q = seed.copy()
        started_at = time.perf_counter()

        for index, (pose, mode) in enumerate(zip(poses, modes)):
            pose_options = self._pose_options(pose, mode)
            seeds = [warm_q, seed]
            # 先从上一个可行构型的腕翻转/中间关节变体开始；均匀随机种子仅作为补充。
            if hasattr(self.kinematics, "generate_alternative_seeds"):
                seeds.extend(self.kinematics.generate_alternative_seeds(warm_q))
            seeds.extend(rng.uniform(lower, upper) for _ in range(max(0, self.config.candidate_count - len(seeds))))
            accepted: list[np.ndarray] = []
            solver_successes = 0
            pose_rejections = 0
            sdf_rejections = 0
            min_position_error = float("inf")
            min_rotation_error = float("inf")
            attempt_limit = max(1, int(self.config.max_solver_attempts_per_pose))
            attempt_total = min(attempt_limit, len(pose_options) * len(seeds))
            attempts = 0
            # Seed-major round robin: with a small interactive attempt budget,
            # every roll candidate is tried before repeatedly consuming seeds on
            # only the original orientation.
            attempt_pairs = [
                (option_index, seed_index)
                for seed_index in range(len(seeds))
                for option_index in range(len(pose_options))
            ]
            for option_index, seed_index in attempt_pairs[:attempt_limit]:
                option = pose_options[option_index]
                q_seed = seeds[seed_index]
                attempts += 1
                if progress_callback is not None and (
                    attempts == 1 or attempts % 3 == 0
                ):
                    completed = index + (attempts - 1) / max(attempt_total, 1)
                    elapsed = time.perf_counter() - started_at
                    progress_callback(
                        completed, len(poses),
                        f"点 {index + 1}/{len(poses)}，roll {option_index + 1}/{len(pose_options)}，"
                        f"种子 {seed_index + 1}/{len(seeds)}，已用 {elapsed:.1f}s",
                    )
                ok, q = self.solver.solve(option, q_seed)
                if ok:
                    solver_successes += 1
                position_error, rotation_error = self._pose_error(q, option)
                min_position_error = min(min_position_error, position_error)
                min_rotation_error = min(min_rotation_error, rotation_error)
                # SLSQP may report a non-success termination after reaching a
                # valid bound-constrained pose.  The geometric tolerance is
                # authoritative; do not discard a valid candidate only due to
                # the optimizer status flag.
                if (position_error > self.config.pose_position_tolerance
                        or rotation_error > self.config.pose_rotation_tolerance):
                    pose_rejections += 1
                    continue
                if self.collision_service is not None and self.collision_service.min_distance(q).distance < -1e-5:
                    sdf_rejections += 1
                    continue
                if not any(np.linalg.norm(q - other) < 1e-4 for other in accepted):
                    accepted.append(q)
                if len(accepted) >= self.config.candidate_count:
                    break
            if not accepted:
                reason = (
                    "no feasible IK candidate "
                    f"(solver_success={solver_successes}, "
                    f"pose_rejected={pose_rejections}, sdf_rejected={sdf_rejections})"
                    f", min_position_error={min_position_error:.6f}m"
                    f", min_rotation_error={min_rotation_error:.6f}rad"
                )
                return BatchPTPPlanResult(False, None, np.empty(0, int), np.asarray([len(x) for x in candidates]), index, reason, float("-inf"))
            warm_q = accepted[0]
            candidates.append(accepted)
            if progress_callback is not None:
                elapsed = time.perf_counter() - started_at
                progress_callback(
                    index + 1, len(poses),
                    f"点 {index + 1}/{len(poses)} 完成，候选 {len(accepted)}，累计 {elapsed:.1f}s",
                )

        selected = self._select_in_windows(candidates)
        if selected is None:
            return BatchPTPPlanResult(False, None, np.empty(0, int), np.asarray([len(x) for x in candidates]), 0, "no collision-safe candidate edge", float("-inf"))
        q_path = np.vstack([candidates[i][candidate_index] for i, candidate_index in enumerate(selected)])
        timestamps, velocities = self._time_parameterize(q_path)
        min_distance = float("inf")
        if self.collision_service is not None:
            report = self.collision_service.trajectory_collision_report(q_path, margin=0.0, segment_samples=3)
            min_distance = report.min_distance
            if report.colliding_indices:
                # 保留已完成的候选轨迹；密集验证失败不应丢弃数十分钟的 IK 结果。
                partial = JointTrajectory(
                    q_path, velocities=velocities, timestamps=timestamps,
                    method="batch_ptp_partial_dense_collision",
                )
                return BatchPTPPlanResult(
                    False, partial, np.asarray(selected),
                    np.asarray([len(x) for x in candidates]),
                    report.colliding_indices[0], "dense SDF collision", min_distance,
                    {"colliding_indices": float(len(report.colliding_indices))},
                )
        return BatchPTPPlanResult(
            True,
            JointTrajectory(q_path, velocities=velocities, timestamps=timestamps, method="batch_ptp_dynamic_programming"),
            np.asarray(selected, dtype=np.int32),
            np.asarray([len(x) for x in candidates], dtype=np.int32),
            None,
            None,
            min_distance,
            {
                "waypoints": float(len(q_path)),
                "duration_s": float(timestamps[-1]),
                "max_joint_step": float(np.max(np.abs(np.diff(q_path, axis=0)), initial=0.0)),
                "max_joint_velocity": float(np.max(np.abs(velocities), initial=0.0)),
                "max_joint_acceleration": float(
                    np.max(
                        np.abs(np.diff(velocities, axis=0))
                        / np.maximum(np.diff(timestamps)[:, None], 1e-9),
                        initial=0.0,
                    )
                ),
            },
        )

    def _pose_options(self, pose: np.ndarray, mode: str) -> list[np.ndarray]:
        if mode != "process":
            return [pose]
        options = []
        roll_degrees = sorted(set(self.config.roll_candidates_deg), key=lambda value: (abs(value), value))
        for degrees in roll_degrees:
            angle = np.deg2rad(degrees)
            Rz = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
            option = pose.copy()
            option[:3, :3] = pose[:3, :3] @ Rz
            options.append(option)
        return options

    def _pose_ok(self, q: np.ndarray, target: np.ndarray) -> bool:
        position_error, rotation_error = self._pose_error(q, target)
        return bool(position_error <= self.config.pose_position_tolerance
                    and rotation_error <= self.config.pose_rotation_tolerance)

    def _pose_error(self, q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        pin.framesForwardKinematics(self.kinematics.model, self.kinematics.data, q)
        current = self.kinematics.data.oMf[self.kinematics.tool_frame_id]
        err = pin.log6(current.actInv(pin.SE3(target))).vector
        return float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def _select_in_windows(self, candidates: list[list[np.ndarray]]) -> Optional[list[int]]:
        # Windows bound working memory for thousand-point programs.  Each
        # emitted window retains an overlap so the next window is anchored to
        # an already-selected collision-safe state.
        selection: list[int] = []
        start = 0
        while start < len(candidates):
            end = min(len(candidates), start + self.config.window_size)
            indices = self._dynamic_program(candidates[start:end], selection[-1] if selection else None, candidates[start - 1] if start else None)
            if indices is None:
                return None
            emit = len(indices) if end == len(candidates) else max(1, len(indices) - self.config.overlap)
            selection.extend(indices[:emit])
            start += emit
        return selection[:len(candidates)]

    def _dynamic_program(self, candidates: list[list[np.ndarray]], anchor_index: Optional[int], anchor_candidates: Optional[list[np.ndarray]]) -> Optional[list[int]]:
        costs = np.full(len(candidates[0]), np.inf)
        if anchor_index is None:
            costs[:] = [self._node_cost(q) for q in candidates[0]]
        else:
            anchor = anchor_candidates[anchor_index]
            for j, q in enumerate(candidates[0]):
                costs[j] = self._edge_cost(anchor, q) + self._node_cost(q)
        parents: list[np.ndarray] = []
        for layer in candidates[1:]:
            next_costs = np.full(len(layer), np.inf)
            parent = np.full(len(layer), -1, dtype=np.int32)
            for j, q in enumerate(layer):
                transition = np.asarray([self._edge_cost(previous, q) for previous in candidates[len(parents)]])
                total = costs + transition + self._node_cost(q)
                best = int(np.argmin(total))
                if np.isfinite(total[best]):
                    next_costs[j], parent[j] = total[best], best
            parents.append(parent)
            costs = next_costs
        if not np.any(np.isfinite(costs)):
            return None
        chosen = [int(np.argmin(costs))]
        for parent in reversed(parents):
            chosen.append(int(parent[chosen[-1]]))
        return list(reversed(chosen))

    def _node_cost(self, q: np.ndarray) -> float:
        if self.collision_service is None:
            return 0.0
        distance = self.collision_service.min_distance(q).distance
        return max(0.0, self.config.safety_margin - distance) ** 2 * 100.0

    def _edge_cost(self, qa: np.ndarray, qb: np.ndarray) -> float:
        if self.collision_service is not None:
            midpoint = (qa + qb) * 0.5
            if self.collision_service.min_distance(midpoint).distance < 0.0:
                return float("inf")
        return float(np.dot(qb - qa, qb - qa))

    def _time_parameterize(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(positions) == 1:
            return np.zeros(1), np.zeros_like(positions)
        steps = np.abs(np.diff(positions, axis=0))
        velocity_dt = np.max(steps / max(self.config.max_joint_velocity, 1e-6), axis=1)
        acceleration_dt = np.sqrt(
            2.0 * np.max(steps, axis=1) / max(self.config.max_joint_acceleration, 1e-6)
        )
        dt = np.maximum(np.maximum(velocity_dt, acceleration_dt), 1e-3)
        # Enforce acceleration at internal knots by expanding adjacent intervals.
        for _ in range(12):
            velocity = np.diff(positions, axis=0) / dt[:, None]
            if len(velocity) < 2:
                break
            acceleration = np.abs(np.diff(velocity, axis=0)) / (
                0.5 * (dt[:-1] + dt[1:])[:, None]
            )
            ratios = np.max(acceleration, axis=1) / max(self.config.max_joint_acceleration, 1e-6)
            violating = np.flatnonzero(ratios > 1.0 + 1e-9)
            if len(violating) == 0:
                break
            for index in violating:
                scale = float(np.sqrt(ratios[index]))
                dt[index] *= scale
                dt[index + 1] *= scale
        timestamps = np.r_[0.0, np.cumsum(dt)]
        velocities = np.zeros_like(positions)
        velocities[1:] = np.diff(positions, axis=0) / dt[:, None]
        return timestamps, velocities


__all__ = ["BatchPTPPlanner", "BatchPTPPlannerConfig", "BatchPTPPlanResult"]
