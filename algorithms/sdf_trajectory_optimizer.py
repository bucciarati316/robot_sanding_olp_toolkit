"""
SDF 距离场引导的关节轨迹后优化器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from collision.collision_distance_service import (
    CollisionDistanceService,
    TrajectoryCollisionReport,
)


@dataclass(frozen=True)
class SDFTrajectoryOptimizationResult:
    """SDF 轨迹优化结果"""
    success: bool
    positions: np.ndarray
    initial_report: TrajectoryCollisionReport
    final_report: TrajectoryCollisionReport
    iterations: int
    message: str


@dataclass(frozen=True)
class SDFTrajectoryOptimizerConfig:
    """SDF 轨迹优化器配置参数"""
    safety_margin: float = 0.03
    w_collision: float = 10.0
    w_reference: float = 1.0
    w_smooth: float = 0.2
    w_acceleration: float = 0.05
    max_iterations: int = 80
    segment_samples: int = 3
    fix_endpoints: bool = True


class SDFTrajectoryOptimizer:
    """基于 SDF 距离场优化关节轨迹的优化器"""

    def __init__(
        self,
        collision_service: CollisionDistanceService,
        *,
        config: Optional[SDFTrajectoryOptimizerConfig] = None,
        lower_limits: Optional[np.ndarray] = None,
        upper_limits: Optional[np.ndarray] = None,
    ):
        """
        初始化 SDF 轨迹优化器。

        参数:
            collision_service: 碰撞距离场服务
            config: 优化配置，默认为标准配置
            lower_limits: 关节下限
            upper_limits: 关节上限
        """
        self.collision_service = collision_service
        self.config = config or SDFTrajectoryOptimizerConfig()
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

    def optimize(self, positions: np.ndarray) -> SDFTrajectoryOptimizationResult:
        """
        优化轨迹以避开障碍物。

        使用 SLSQP 优化方法，在保持轨迹接近原轨迹的同时最小化碰撞代价。
        """
        Q0 = np.asarray(positions, dtype=np.float64)
        if Q0.ndim != 2:
            raise ValueError(f"positions must have shape (N, nq), got {Q0.shape}")

        initial_report = self.collision_service.trajectory_collision_report(
            Q0,
            margin=self.config.safety_margin,
            segment_samples=self.config.segment_samples,
        )
        variable_indices = self._variable_indices(len(Q0))
        if not variable_indices:
            return SDFTrajectoryOptimizationResult(True, Q0.copy(), initial_report, initial_report, 0, "no free variables")

        x0 = Q0[variable_indices].reshape(-1)
        bounds = self._bounds(len(variable_indices), Q0.shape[1])

        def objective(x: np.ndarray) -> float:
            """优化目标函数：参考轨迹代价 + 平滑代价 + 碰撞代价"""
            Q = Q0.copy()
            Q[variable_indices] = x.reshape(len(variable_indices), Q0.shape[1])
            return self._objective(Q, Q0)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.config.max_iterations, "ftol": 1e-5, "disp": False},
        )

        Q_final = Q0.copy()
        Q_final[variable_indices] = result.x.reshape(len(variable_indices), Q0.shape[1])
        final_report = self.collision_service.trajectory_collision_report(
            Q_final,
            margin=self.config.safety_margin,
            segment_samples=self.config.segment_samples,
        )
        return SDFTrajectoryOptimizationResult(
            success=bool(result.success),
            positions=Q_final,
            initial_report=initial_report,
            final_report=final_report,
            iterations=int(getattr(result, "nit", 0)),
            message=str(result.message),
        )

    def _objective(self, Q: np.ndarray, Q_ref: np.ndarray) -> float:
        cfg = self.config
        ref = float(np.sum((Q - Q_ref) ** 2))
        smooth = float(np.sum(np.diff(Q, axis=0) ** 2))
        acc = 0.0
        if len(Q) >= 3:
            acc = float(np.sum((Q[2:] - 2.0 * Q[1:-1] + Q[:-2]) ** 2))

        collision = 0.0
        for q in Q:
            collision += self.collision_service.collision_cost(q, margin=cfg.safety_margin)
        for i in range(len(Q) - 1):
            qa, qb = Q[i], Q[i + 1]
            for j in range(1, max(1, cfg.segment_samples) + 1):
                t = j / (cfg.segment_samples + 1)
                collision += self.collision_service.collision_cost((1.0 - t) * qa + t * qb, margin=cfg.safety_margin)

        return (
            cfg.w_reference * ref
            + cfg.w_smooth * smooth
            + cfg.w_acceleration * acc
            + cfg.w_collision * collision
        )

    def _variable_indices(self, n_points: int) -> list[int]:
        """
        确定哪些轨迹点是可优化的变量。

        如果 fix_endpoints 为 True，则首尾点固定不动。
        """
        if not self.config.fix_endpoints:
            return list(range(n_points))
        if n_points <= 2:
            return []
        return list(range(1, n_points - 1))

    def _bounds(self, n_variables: int, nq: int):
        """
        生成关节限位边界约束。

        返回每个优化变量的 [下限, 上限] 边界对列表。
        """
        if self.lower_limits is None or self.upper_limits is None:
            return None
        lower = np.asarray(self.lower_limits, dtype=np.float64)
        upper = np.asarray(self.upper_limits, dtype=np.float64)
        if lower.size != nq or upper.size != nq:
            return None
        return [(float(lower[j]), float(upper[j])) for _ in range(n_variables) for j in range(nq)]


__all__ = [
    "SDFTrajectoryOptimizationResult",
    "SDFTrajectoryOptimizer",
    "SDFTrajectoryOptimizerConfig",
]
