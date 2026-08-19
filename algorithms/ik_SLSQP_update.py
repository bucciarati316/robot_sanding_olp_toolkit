# 序贯最小二乘法 Sequential Least SQuares Programming

"""
ik_SLSQP_update - 多目标 SLSQP 非线性优化 IK 求解器插件
==========================================================

基于 scipy.optimize.minimize 的 SLSQP 方法的带边界约束逆运动学求解器。
目标函数升级为四目标加权代价：

    f(q) = w_pose * E_pose
         + w_disp * E_disp
         + w_center * E_center
         + w_manip * E_manip

其中：
    E_pose  = ||log6(T_current^{-1} @ T_target)||^2   (完整末端位姿误差)
    E_disp  = ||q - q_init||^2                        (最小关节位移)
    E_center= sum_i ((q_i - q_mid,i) / q_range,i)^2   (关节趋中，归一化)
    E_manip = -sqrt(det(J @ J^T))                      (吉川可操作度，SVD 计算)

该求解核心恢复自已验证提交 3d3ddb32bc9dd80d99f35a8e775d672a8aaa859a：
所有目标封装在单个 objective(q) 中，由 SLSQP 在完整 6D 位姿约束下求解。

防呆设计：
    - 可操作度用 SVD 奇异值乘积替代 det()，避免奇异点数值崩溃
    - 所有关节均参与 E_center 计算（joint_range <= 0 的无效关节除外）
    - 关节边界作为 bounds 约束传入 SLSQP

Author: AI Architect
Date: 2026-06-09
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

from ._utils import _import_core
BaseAlgorithm, ParamDef, ParamType = _import_core()

__all__ = ["SLSQPMultiSolver"]


class SLSQPMultiSolver(BaseAlgorithm):
    """
    多目标 SLSQP 非线性优化 IK 求解器（带关节边界约束）

    在保证末端位姿精度的前提下，通过加权多目标优化实现：
        1. 最小关节位移（轨迹平滑性）
        2. 关节趋中（增加工作空间鲁棒性）
        3. 最大化可操作度（规避奇异构型）

    算法流程：
        1. 构建目标函数: f(q) = w_pose*E_pose + w_disp*E_disp + w_center*E_center + w_manip*E_manip
        2. 提取 URDF 关节边界作为 bounds 约束
        3. SLSQP 优化求解
    """

    NAME = "SLSQP_update"
    DESC = "SLSQP_Multi"
    SUPPORTED_EXTS = []

    def __init__(self, kinematics_engine=None, **kwargs):
        super().__init__()
        self._kin_engine = kinematics_engine

        self._tolerance     = kwargs.get('tolerance', 1e-5)
        self._max_iterations = kwargs.get('max_iterations', 1000)
        self._verbose       = kwargs.get('verbose', False)

        self._w_pose   = kwargs.get('w_pose', 1.0)
        self._w_disp   = kwargs.get('w_disp', 0.01)
        self._w_center = kwargs.get('w_center', 0.01)
        self._w_manip  = kwargs.get('w_manip', 0.001)
        self._w_joints = kwargs.get('w_joints', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        if self._w_joints is None:
            self._w_joints = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self._collision_distance_service = kwargs.get('collision_distance_service', None)
        self._w_collision = kwargs.get('w_collision', 0.2)
        self._safety_margin = kwargs.get('safety_margin', 0.03)

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
            ParamDef(
                id="w_pose",
                label="位姿误差权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.001,
                max_val=10.0,
                step=0.01,
                desc="主任务权重，保证末端到达精度。必须最大以维持优先级"
            ),
            ParamDef(
                id="w_disp",
                label="关节位移权重",
                ptype=ParamType.FLOAT,
                default=0.01,
                min_val=1e-6,
                max_val=1.0,
                step=1e-4,
                desc="最小化与上一路点的关节位移，平滑轨迹"
            ),
            ParamDef(
                id="w_center",
                label="关节趋中权重",
                ptype=ParamType.FLOAT,
                default=0.01,
                min_val=1e-6,
                max_val=1.0,
                step=1e-4,
                desc="将关节引导至限位中心，增加工作空间鲁棒性"
            ),
            ParamDef(
                id="w_manip",
                label="可操作度权重",
                ptype=ParamType.FLOAT,
                default=0.001,
                min_val=1e-6,
                max_val=0.1,
                step=1e-5,
                desc="最大化吉川可操作度，规避奇异构型"
            ),
            ParamDef(
                id="w_j1",
                label="J1 权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=200.0,
                step=0.1,
                desc="关节 1 位移惩罚权重（打磨建议 1.0）"
            ),
            ParamDef(
                id="w_j2",
                label="J2 权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=200.0,
                step=0.1,
                desc="关节 2 位移惩罚权重（打磨建议 1.0）"
            ),
            ParamDef(
                id="w_j3",
                label="J3 权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=200.0,
                step=0.1,
                desc="关节 3 位移惩罚权重（打磨建议 1.0）"
            ),
            ParamDef(
                id="w_j4",
                label="J4 权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=200.0,
                step=0.1,
                desc="关节 4 位移惩罚权重（打磨建议 1.0）"
            ),
            ParamDef(
                id="w_j5",
                label="J5 权重",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=200.0,
                step=0.1,
                desc="关节 5 位移惩罚权重（打磨建议 1.0）"
            ),
            ParamDef(
                id="w_j6",
                label="J6 权重",
                ptype=ParamType.FLOAT,
                default=1,
                min_val=0.01,
                max_val=200.0,
                step=0.5,
                desc="关节 6 位移惩罚权重（打磨建议 50.0，冗余关节）"
            ),
        ]

    def load_geometry(self, filepath: str) -> bool:
        return True

    def solve(self, target_pose_rcf: np.ndarray, q_init: np.ndarray):
        """
        基于多目标 SLSQP 的带边界约束逆运动学求解

        参数：
            target_pose_rcf: 目标 SE(3) 位姿（4x4 齐次矩阵），须已在 RCF 下
            q_init: 初始关节角度种子 (nq,)

        返回：
            tuple: (success: bool, q_solution: np.ndarray)
        """
        if self._kin_engine is None:
            raise RuntimeError("SLSQPMultiSolver 未绑定 KinematicsEngine")

        kin = self._kin_engine
        assert target_pose_rcf.shape == (4, 4), "target_pose 必须是 4x4 矩阵"
        assert len(q_init) == kin.nq, f"q_init 长度 {len(q_init)} 与模型不匹配 {kin.nq}"

        target_SE3 = pin.SE3(target_pose_rcf)

        w_pose   = self._w_pose
        w_disp   = self._w_disp
        w_center = self._w_center
        w_manip  = self._w_manip
        w_collision = self._w_collision
        safety_margin = self._safety_margin
        collision_service = self._collision_distance_service
        W_q      = np.array(self._w_joints, dtype=float)
        if W_q.size != kin.nq:
            W_q = np.resize(W_q, kin.nq)

        def objective(q: np.ndarray) -> float:
            pin.framesForwardKinematics(kin.model, kin.data, q)
            current_SE3 = kin.data.oMf[kin.tool_frame_id]

            # 已验证基线：完整 SE(3) 位姿误差，位置和三个姿态自由度均不得释放。
            err = pin.log6(current_SE3.actInv(target_SE3)).vector
            E_pose = float(np.linalg.norm(err) ** 2)

            E_disp = float(np.sum(W_q * (q - q_init) ** 2))

            lower = kin.model.lowerPositionLimit
            upper = kin.model.upperPositionLimit
            E_center = 0.0
            for i in range(kin.nq):
                joint_range = float(upper[i] - lower[i])
                if joint_range <= 0:
                    continue
                q_mid = float(upper[i] + lower[i]) / 2.0
                normalized = (q[i] - q_mid) / joint_range
                E_center += normalized ** 2

            J = pin.computeFrameJacobian(
                kin.model, kin.data, q,
                kin.tool_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            # Pinocchio 3/4 对单自由度模型的 Python 返回形状不同；统一为 6×nv。
            J = np.asarray(J, dtype=float).reshape(6, kin.model.nv)
            _, s, _ = np.linalg.svd(J, full_matrices=False)
            nonzero_s = s[s > 1e-9]
            manipulability = float(np.prod(nonzero_s)) if nonzero_s.size > 0 else 0.0
            E_manip = -manipulability
            E_collision = 0.0
            if collision_service is not None and w_collision > 0.0:
                try:
                    E_collision = float(
                        collision_service.collision_cost(q, margin=safety_margin)
                    )
                except Exception:
                    E_collision = 0.0

            return (w_pose * E_pose + w_disp * E_disp
                    + w_center * E_center + w_manip * E_manip
                    + w_collision * E_collision)

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
            print(f"  [SLSQP_Multi] success={result.success}, "
                  f"nit={result.nit}, nfev={result.nfev}, "
                  f"fun={result.fun:.6e}")

        return bool(result.success), result.x.copy()

    def generate(self, **kwargs):
        raise NotImplementedError("SLSQPMultiSolver 不支持 generate() 方法，请使用 solve()")


def solve(target_pose_rcf: np.ndarray, kinematics_engine, q_init: np.ndarray,
          tolerance: float = 1e-5, max_iterations: int = 1000,
          verbose: bool = False,
          w_pose: float = 1.0, w_disp: float = 0.01,
          w_center: float = 0.01, w_manip: float = 0.001,
          w_joints=None, collision_distance_service=None,
          w_collision: float = 0.2, safety_margin: float = 0.03) -> tuple:
    """
    便捷函数：使用多目标 SLSQP 求解单个位姿的 IK。

    参数：
        target_pose_rcf: 目标 SE(3) 位姿（4x4 齐次矩阵），须已在 RCF 下
        kinematics_engine: KinematicsEngine 实例
        q_init: 初始关节角度种子 (nq,)
        tolerance: 收敛容差
        max_iterations: 最大迭代次数
        verbose: 是否打印优化过程信息
        w_pose: 位姿误差权重（主任务，默认 1.0）
        w_disp: 关节位移权重（默认 0.01）
        w_center: 关节趋中权重（默认 0.01）
        w_manip: 可操作度权重（默认 0.001）
        w_joints: 各关节位移惩罚权重数组（默认全部为 1）
    返回：
        tuple: (success: bool, q_solution: np.ndarray)
    """
    solver = SLSQPMultiSolver(
        kinematics_engine=kinematics_engine,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
        w_pose=w_pose,
        w_disp=w_disp,
        w_center=w_center,
        w_manip=w_manip,
        w_joints=w_joints,
        collision_distance_service=collision_distance_service,
        w_collision=w_collision,
        safety_margin=safety_margin,
    )
    return solver.solve(target_pose_rcf, q_init)
