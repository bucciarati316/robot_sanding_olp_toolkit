#伪逆迭代法 Pseudo Inverse Iteration Method
#直接伪逆求解法，基于 Pinocchio SE(3)/log6 误差的阻尼最小二乘迭代 IK

"""
ik_PILM - 直接伪逆求解器插件
================================================

基于阻尼最小二乘的伪逆雅可比 IK 求解器。
使用 Pinocchio 的 SE(3) 对数误差（log6）计算位姿误差，
通过迭代伪逆更新关节角度直至收敛。

原理:
    1. OCF 目标位姿经 _compute_robot_target_pose() 转为 RCF
    2. 目标法兰盘位姿 = RCF目标 @ T_flange_tcp^{-1}
    3. 阻尼最小二乘迭代: delta_q = alpha * pinv(J) @ err
    4. Pinocchio integrate() 安全处理流形空间加法
    5. 可选关节限位钳制

Author: AI Architect
Date: 2026-06-04
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from ._utils import _import_core
BaseAlgorithm, ParamDef, ParamType = _import_core()

__all__ = ["PseudoInverseSolver"]


class PseudoInverseSolver(BaseAlgorithm):
    """
    直接伪逆 IK 求解器（阻尼最小二乘法）

    算法流程：
    1. OCF → RCF 坐标系变换
    2. 目标法兰盘 = RCF目标 @ T_flange_tcp^{-1}
    3. 迭代伪逆雅可比更新关节（Pinocchio log6 误差）
    4. 可选关节限位钳制
    """

    NAME = "PILM"
    DESC = "直接计算法"
    SUPPORTED_EXTS = []

    def __init__(self, kinematics_engine=None, **kwargs):
        super().__init__()
        self._kin_engine = kinematics_engine
        self._max_iterations = kwargs.get('max_iterations', 500)
        self._tolerance = kwargs.get('tolerance', 1e-4)
        self._alpha = kwargs.get('alpha', 1.0)
        self._joint_limits = kwargs.get('joint_limits', True)
        self._damping = kwargs.get('damping', 0.001)
        self._verbose = kwargs.get('verbose', False)

    def get_parameters(self):
        return [
            ParamDef(
                id="max_iterations",
                label="最大迭代次数",
                ptype=ParamType.INT,
                default=500,
                min_val=10,
                max_val=5000,
                desc="IK 求解的最大迭代次数。数值越大越可能收敛，但计算时间增加"
            ),
            ParamDef(
                id="tolerance",
                label="收敛容差",
                ptype=ParamType.FLOAT,
                default=1e-4,
                min_val=1e-8,
                max_val=1e-1,
                step=1e-5,
                desc="位姿误差（log6 向量范数）收敛阈值。越小越精确"
            ),
            ParamDef(
                id="alpha",
                label="步长因子",
                ptype=ParamType.FLOAT,
                default=1.0,
                min_val=0.01,
                max_val=2.0,
                step=0.05,
                desc="关节更新步长 (0, 2]。越小越稳定但收敛慢，越大越快但可能震荡"
            ),
            ParamDef(
                id="damping",
                label="阻尼因子",
                ptype=ParamType.FLOAT,
                default=0.001,
                min_val=1e-6,
                max_val=0.1,
                step=1e-4,
                desc="阻尼因子 lambda^2。用于提高近奇异构型稳定性，越大越稳定但收敛越慢"
            ),
            ParamDef(
                id="joint_limits",
                label="启用关节限位",
                ptype=ParamType.BOOL,
                default=True,
                desc="求解后是否将关节角度钳制到 URDF 定义的物理限位内"
            ),
        ]

    def load_geometry(self, filepath: str) -> bool:
        return True

    def solve(self, target_pose_ocf: np.ndarray, q_init: np.ndarray = None):
        """
        逆运动学求解 - 直接伪逆阻尼最小二乘法

        参数：
            target_pose_ocf: 目标位姿在工件坐标系(OCF)下的 4x4 齐次变换矩阵
            q_init: 初始关节角度种子（Seed），用于保证连续性。
                    如果为 None，使用中立姿态作为初始值

        返回：
            tuple: (success, q_solution, final_error)
            - success: bool, 求解是否收敛
            - q_solution: np.ndarray, 求解得到的关节角度
            - final_error: float, 最终误差范数（log6 向量范数）
        """
        if self._kin_engine is None:
            raise RuntimeError("PseudoInverseSolver 未绑定 KinematicsEngine")

        kin = self._kin_engine
        assert target_pose_ocf.shape == (4, 4), "目标位姿必须是 4x4 矩阵"

        T_robot_target_tcp = kin._compute_robot_target_pose(target_pose_ocf)
        T_robot_target_flange = T_robot_target_tcp @ np.linalg.inv(kin.effective_tcp)
        target_SE3 = pin.SE3(T_robot_target_flange)

        end_effector_name = kin.model.names[-1]
        frame_id = kin.model.getFrameId(end_effector_name)

        if q_init is not None:
            assert len(q_init) == kin.nq, f"q_init 长度 {len(q_init)} 与模型不匹配 {kin.nq}"
            q = q_init.copy()
        else:
            q = np.zeros(kin.nq)

        kin.current_q = q.copy()
        prev_error = float('inf')
        alpha = self._alpha

        for iteration in range(self._max_iterations):
            pin.forwardKinematics(kin.model, kin.data, q)
            pin.updateFramePlacements(kin.model, kin.data)
            current_SE3 = kin.data.oMf[frame_id]

            dMi = current_SE3.actInv(target_SE3)
            err = pin.log6(dMi).vector
            error_norm = np.linalg.norm(err)

            if self._verbose and iteration % 100 == 0:
                print(f"  Iter {iteration}: error = {error_norm:.6f}")

            if error_norm < self._tolerance:
                kin.current_q = q.copy()
                return (True, q.copy(), error_norm)

            if error_norm > prev_error * 1.5 and iteration > 10:
                alpha = alpha * 0.5
            prev_error = error_norm

            J = pin.computeFrameJacobian(
                kin.model, kin.data, q, frame_id,
                pin.ReferenceFrame.LOCAL
            )
            J_pinv = np.linalg.pinv(J, rcond=1e-6)
            delta_q = J_pinv @ err
            q = pin.integrate(kin.model, q, alpha * delta_q)

            if self._joint_limits:
                q = kin._apply_joint_limits(q)

        pin.forwardKinematics(kin.model, kin.data, q)
        pin.updateFramePlacements(kin.model, kin.data)
        current_SE3_final = kin.data.oMf[frame_id]
        dMi_final = current_SE3_final.actInv(target_SE3)
        final_error = np.linalg.norm(pin.log6(dMi_final).vector)

        kin.current_q = q.copy()
        return (False, q.copy(), final_error)

    def generate(self, **kwargs):
        raise NotImplementedError("PseudoInverseSolver 不支持 generate() 方法，请使用 solve()")


def solve(target_pose_ocf: np.ndarray, kinematics_engine, q_init: np.ndarray = None,
          max_iterations: int = 500, tolerance: float = 1e-4,
          alpha: float = 1.0, joint_limits: bool = True,
          damping: float = 0.001, verbose: bool = False) -> tuple:
    """
    便捷函数：使用伪逆直接求解法求解单个位姿的 IK。

    参数：
        target_pose_ocf: 目标位姿在工件坐标系(OCF)下的 4x4 齐次变换矩阵
        kinematics_engine: KinematicsEngine 实例
        q_init: 初始关节角度种子
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
        alpha: 步长因子
        joint_limits: 是否启用关节限位
        damping: 阻尼因子
        verbose: 是否打印迭代过程

    返回：
        tuple: (success, q_solution, final_error)
    """
    solver = PseudoInverseSolver(
        kinematics_engine=kinematics_engine,
        max_iterations=max_iterations,
        tolerance=tolerance,
        alpha=alpha,
        joint_limits=joint_limits,
        damping=damping,
        verbose=verbose
    )
    return solver.solve(target_pose_ocf, q_init)
