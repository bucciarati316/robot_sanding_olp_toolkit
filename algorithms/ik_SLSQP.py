#序贯最小二乘法 Sequential Least SQuares Programming

"""
ik_SLSQP - SLSQP 非线性优化 IK 求解器插件
================================================

基于 scipy.optimize.minimize 的 SLSQP 方法的带边界约束逆运动学求解器。
使用 Pinocchio SE(3) 对数误差的二范数作为目标函数，
以关节上下限作为边界约束。

原理:
    1. 目标函数: f(q) = ||log6(T_current^{-1} @ T_target)||^2
    2. 边界约束: q_lower <= q <= q_upper（从 URDF 提取）
    3. scipy.optimize SLSQP 优化

Author: AI Architect
Date: 2026-06-04
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

from ._utils import _import_core
BaseAlgorithm, ParamDef, ParamType = _import_core()

__all__ = ["SLSQPSolver"]


class SLSQPSolver(BaseAlgorithm):
    """
    SLSQP 非线性优化 IK 求解器（带关节边界约束）

    算法流程：
    1. 构建目标函数: f(q) = ||log6(T_current^{-1} @ T_target)||^2
    2. 提取 URDF 关节边界作为约束
    3. SLSQP 优化求解
    4. 可选翻腕备用构型搜索
    """

    NAME = "SLSQP"
    DESC = "SLSQP"
    SUPPORTED_EXTS = []

    def __init__(self, kinematics_engine=None, **kwargs):
        super().__init__()
        self._kin_engine = kinematics_engine
        self._tolerance = kwargs.get('tolerance', 1e-5)
        self._max_iterations = kwargs.get('max_iterations', 1000)
        self._verbose = kwargs.get('verbose', False)

    def get_parameters(self):
        return [
            ParamDef(
                id="tolerance",
                label="收敛容差",
                ptype=ParamType.FLOAT,
                default=1e-5,
                min_val=1e-10,
                max_val=1e-2,
                step=1e-5,
                desc="SLSQP 优化的目标函数值收敛容差。越小越精确"
            ),
            ParamDef(
                id="max_iterations",
                label="最大迭代次数",
                ptype=ParamType.INT,
                default=1000,
                min_val=100,
                max_val=10000,
                desc="SLSQP 优化器的最大迭代次数。数值越大越可能收敛"
            ),
            ParamDef(
                id="verbose",
                label="详细输出",
                ptype=ParamType.BOOL,
                default=False,
                desc="是否打印优化过程的详细信息"
            ),
        ]

    def load_geometry(self, filepath: str) -> bool:
        return True

    def solve(self, target_pose_rcf: np.ndarray, q_init: np.ndarray):
        """
        基于 SLSQP 的带边界约束逆运动学求解

        参数：
            target_pose_rcf: 目标 SE(3) 位姿（4x4 齐次矩阵），须已在 RCF 下
            q_init: 初始关节角度种子 (nq,)

        返回：
            tuple: (success: bool, q_solution: np.ndarray)
        """
        if self._kin_engine is None:
            raise RuntimeError("SLSQPSolver 未绑定 KinematicsEngine")

        kin = self._kin_engine
        assert target_pose_rcf.shape == (4, 4), "target_pose 必须是 4x4 矩阵"
        assert len(q_init) == kin.nq, f"q_init 长度 {len(q_init)} 与模型不匹配 {kin.nq}"

        target_SE3 = pin.SE3(target_pose_rcf)

        def objective(q: np.ndarray) -> float:
            pin.framesForwardKinematics(kin.model, kin.data, q)
            current_SE3 = kin.data.oMf[kin.tool_frame_id]
            err = pin.log6(current_SE3.actInv(target_SE3)).vector
            return float(np.linalg.norm(err) ** 2)

        lower = kin.model.lowerPositionLimit
        upper = kin.model.upperPositionLimit
        if lower is None or len(lower) == 0:
            lower = -np.pi * np.ones(kin.nq)
        if upper is None or len(upper) == 0:
            upper = np.pi * np.ones(kin.nq)

        bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]

        result = minimize(
            objective,
            q_init,
            method='SLSQP',
            bounds=bounds,
            options={
                'ftol': self._tolerance,
                'maxiter': self._max_iterations,
                'disp': self._verbose,
            }
        )

        if self._verbose:
            print(f"  [SLSQP] success={result.success}, "
                  f"nit={result.nit}, nfev={result.nfev}, "
                  f"fun={result.fun:.6e}")

        return bool(result.success), result.x.copy()

    def generate(self, **kwargs):
        raise NotImplementedError("SLSQPSolver 不支持 generate() 方法，请使用 solve()")


def solve(target_pose_rcf: np.ndarray, kinematics_engine, q_init: np.ndarray,
          tolerance: float = 1e-5, max_iterations: int = 1000,
          verbose: bool = False) -> tuple:
    """
    便捷函数：使用 SLSQP 求解单个位姿的 IK。

    参数：
        target_pose_rcf: 目标 SE(3) 位姿（4x4 齐次矩阵），须已在 RCF 下
        kinematics_engine: KinematicsEngine 实例
        q_init: 初始关节角度种子 (nq,)
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        verbose: 是否打印优化过程信息

    返回：
        tuple: (success: bool, q_solution: np.ndarray)
    """
    solver = SLSQPSolver(
        kinematics_engine=kinematics_engine,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose
    )
    return solver.solve(target_pose_rcf, q_init)
