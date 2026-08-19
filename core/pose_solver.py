"""
PoseSolver - 机器人位姿求解器

功能：
    读取 toolpath_pose.csv (x,y,z,qw,qx,qy,qz)，通过 IK 求解器计算关节轨迹

工作流程：
    1. 读取位姿 CSV
    2. 设置坐标系变换 (T_wcf_rcf, T_wcf_ocf)
    3. 批量 IK 求解（种子姿态连续性）
    4. 导出手柄轨迹 CSV
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional, Dict
import os

from .kinematics_engine import KinematicsEngine
from .schemas import FlangeToolParams, CoordinateFrames
from .coordinate_transforms import (
    ocf_to_rcf, ocf_to_wcf,
    wcf_to_rcf,
    transform_points_with_normals as _transform_points_with_normals,
    is_identity,
)
from .auto_manager import get_manager


class CoordinateTransformer:
    """
    坐标系变换器（向后兼容包装类）。

    内部使用 coordinate_transforms.py 中的纯函数，
    同时提供类实例状态接口以保持与 render_engine.py 等现有代码的兼容性。
    """

    def __init__(self):
        self.T_wcf_rcf = np.eye(4)
        self.T_wcf_ocf = np.eye(4)

    def set_transforms(self, T_wcf_rcf: Optional[np.ndarray] = None,
                      T_wcf_ocf: Optional[np.ndarray] = None) -> None:
        if T_wcf_rcf is not None:
            self.T_wcf_rcf = T_wcf_rcf
        if T_wcf_ocf is not None:
            self.T_wcf_ocf = T_wcf_ocf

    def ocf_to_wcf(self, T_ocf: np.ndarray) -> np.ndarray:
        return ocf_to_wcf(T_ocf, self.T_wcf_ocf)

    def ocf_to_wcf_batch(self, matrices_ocf: List[np.ndarray]) -> List[np.ndarray]:
        return [self.ocf_to_wcf(T) for T in matrices_ocf]

    def ocf_to_rcf(self, T_ocf: np.ndarray) -> np.ndarray:
        return ocf_to_rcf(T_ocf, self.T_wcf_rcf, self.T_wcf_ocf)

    def ocf_to_rcf_batch(self, matrices_ocf: List[np.ndarray]) -> List[np.ndarray]:
        return [self.ocf_to_rcf(T) for T in matrices_ocf]

    def wcf_to_rcf(self, T_wcf: np.ndarray) -> np.ndarray:
        return wcf_to_rcf(T_wcf, self.T_wcf_rcf)

    def wcf_to_rcf_batch(self, matrices_wcf: List[np.ndarray]) -> List[np.ndarray]:
        return [self.wcf_to_rcf(T) for T in matrices_wcf]

    def transform_points_with_normals(self, points: np.ndarray,
                                     normals: Optional[np.ndarray] = None,
                                     apply_ocf: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if apply_ocf:
            return _transform_points_with_normals(points, normals, self.T_wcf_ocf)
        return points, normals

    @staticmethod
    def is_identity(T: np.ndarray, tol: float = 1e-9) -> bool:
        return is_identity(T, tol)


class PoseSolver:
    """
    机器人位姿求解器 - 批量 IK 求解（ABB RAPID 架构版本）。

    核心改进：
        - 持有 ToolLibrary 引用（SSOT），而非独立的 T_flange_tcp 副本
        - 通过共享同一个 ToolLibrary 实例确保与 KinematicsEngine 的刀具数据一致
        - IK 求解直接委托给 KinematicsEngine，Engine 内部以 TCP Frame 进行计算
    """

    def __init__(self, urdf_path: str, pose_csv: str,
                 tool_library=None, verbose: bool = True):
        """
        初始化位姿求解器

        参数:
            urdf_path: 机器人 URDF 文件路径
            pose_csv: 输入位姿 CSV 路径 (x,y,z,qw,qx,qy,qz)
            tool_library: ToolLibrary 实例。如果为 None，创建一个内部实例。
            verbose: 是否打印详细信息
        """
        from .tool_library import ToolLibrary
        self.urdf_path = urdf_path
        self.pose_csv = pose_csv
        self.verbose = verbose

        # 工具库引用（SSOT）
        self._tool_library = tool_library if tool_library is not None else ToolLibrary()

        # 运动学引擎（共享同一个 ToolLibrary）
        self.engine = KinematicsEngine(
            urdf_path=urdf_path,
            tool_library=self._tool_library,
            verbose=False
        )

        # 坐标系变换矩阵（仅 T_wcf_rcf / T_wcf_ocf，T_flange_tcp 由 ToolLibrary 管理）
        self.T_wcf_rcf = np.eye(4)  # 世界 -> 机器人基座
        self.T_wcf_ocf = np.eye(4)  # 世界 -> 工件

        # 求解统计
        self.solve_stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'max_error': 0.0,
            'failed_indices': []
        }

    def set_transforms(
        self,
        T_wcf_rcf: Optional[np.ndarray] = None,
        T_wcf_ocf: Optional[np.ndarray] = None,
        T_flange_tcp: Optional[np.ndarray] = None
    ) -> None:
        """
        设置坐标系变换矩阵

        参数:
            T_wcf_rcf: 世界坐标系到机器人基座的变换 (4x4)
            T_wcf_ocf: 世界坐标系到工件坐标系的变换 (4x4)
            T_flange_tcp: 已废弃，请使用 engine.set_tool() 或 tool_library.set_current_tool()
        """
        if T_wcf_rcf is not None:
            self.T_wcf_rcf = T_wcf_rcf
        if T_wcf_ocf is not None:
            self.T_wcf_ocf = T_wcf_ocf

        # 应用到运动学引擎
        self.engine.set_transforms(self.T_wcf_rcf, self.T_wcf_ocf)

        if T_flange_tcp is not None:
            import warnings
            warnings.warn(
                "T_flange_tcp parameter in set_transforms is deprecated. "
                "Use engine.set_tcp_transform() or tool_library.set_current_tool() instead.",
                DeprecationWarning,
                stacklevel=2
            )
            self.engine.set_tcp_transform(T_flange_tcp)

        if self.verbose:
            print("[PoseSolver] 坐标系变换已更新")
            print(f"  T_wcf_rcf (世界->机器人基座):\n{self.T_wcf_rcf}")

    def set_tool_library(self, tool_library) -> None:
        """
        设置/替换 ToolLibrary 引用。

        同时更新 KinematicsEngine 的引用，确保所有组件使用同一 SSOT。
        """
        self._tool_library = tool_library
        self.engine._tool_library = tool_library
        self.engine._mount_tcp_frame()

    def get_tool_library(self):
        """获取当前 ToolLibrary 引用"""
        return self._tool_library

    def _generate_safe_seed(self) -> np.ndarray:
        """
        生成安全的初始关节姿态

        策略：
            - 连续关节（限位范围 > 2π）：设为 0
            - 普通关节：设为限位范围的中值
        """
        lower = self.engine.model.lowerPositionLimit
        upper = self.engine.model.upperPositionLimit
        q_init = np.zeros(self.engine.nq)

        for i in range(self.engine.nq):
            lower_i = lower[i] if i < len(lower) else -np.pi
            upper_i = upper[i] if i < len(upper) else np.pi

            if upper_i - lower_i > 2 * np.pi:
                q_init[i] = 0.0  # 连续关节
            else:
                q_init[i] = (upper_i + lower_i) / 2.0  # 中间值

        if self.verbose:
            print(f"[PoseSolver] 生成安全种子姿态: {q_init}")

        return q_init

    def _build_transform(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        从位置和四元数构建 4x4 齐次变换矩阵

        参数:
            pos: 位置 [x, y, z]
            quat: 四元数 [qx, qy, qz, qw]

        返回:
            4x4 变换矩阵
        """
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        return T

    def _compute_swing_angle(
        self,
        current_pos: np.ndarray,
        next_pos: np.ndarray,
        current_normal: np.ndarray
    ) -> Dict[str, float]:
        """
        计算行进方向的摆角信息（用于监控关节角度变化）

        参数:
            current_pos: 当前点位置
            next_pos: 下一个点位置
            current_normal: 当前点法向量

        返回:
            包含摆角信息的字典
        """
        # 切线方向
        tangent = next_pos - current_pos
        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm < 1e-6:
            return {'angle_rad': 0.0, 'direction': 'none'}

        tangent = tangent / tangent_norm

        # 计算与法向量垂直平面内各轴的夹角
        # 简化计算：返回切线在 XY 平面的投影角度
        angle_xy = np.arctan2(tangent[1], tangent[0])
        angle_pitch = np.arcsin(np.clip(tangent[2], -1.0, 1.0))

        return {
            'angle_rad': angle_xy,
            'pitch_rad': angle_pitch,
            'direction': 'forward' if tangent_norm > 0.001 else 'none'
        }

    def solve_trajectory(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        alpha: float = 1.0,
        check_joint_limits: bool = True
    ) -> List[np.ndarray]:
        """
        批量 IK 求解

        参数:
            max_iterations: 每个点最大迭代次数
            tolerance: 收敛阈值
            alpha: 步长因子
            check_joint_limits: 是否检查关节限位

        返回:
            List[np.ndarray]: 关节角度轨迹列表
        """
        if self.verbose:
            print(f"[PoseSolver] 加载位姿数据: {self.pose_csv}")

        df = pd.read_csv(self.pose_csv)
        n_points = len(df)

        if self.verbose:
            print(f"[PoseSolver] 共 {n_points} 个位姿待求解")

        # 初始化种子姿态
        q_current = self._generate_safe_seed()
        trajectory_q: List[np.ndarray] = []
        self.solve_stats = {
            'total': n_points,
            'success': 0,
            'failed': 0,
            'max_error': 0.0,
            'failed_indices': []
        }

        # 转换为 numpy 数组便于处理
        pts = df[['x', 'y', 'z']].values
        quats = df[['qx', 'qy', 'qz', 'qw']].values

        for idx in range(n_points):
            pos = pts[idx]
            quat = quats[idx]

            # 构建目标变换矩阵
            T_target = self._build_transform(pos, quat)

            # IK 求解
            manager = get_manager()
            SolverClass = manager.get_algorithm_class('伪逆直接求解法', prefix='ik')
            solver = SolverClass(
                kinematics_engine=self.engine,
                max_iterations=max_iterations,
                tolerance=tolerance,
                alpha=alpha,
                joint_limits=check_joint_limits,
                damping=0.001,
                verbose=self.verbose,
            )
            success, q_sol, error = solver.solve(T_target, q_init=q_current)

            if success:
                trajectory_q.append(q_sol.copy())
                q_current = q_sol.copy()
                self.solve_stats['success'] += 1
                self.solve_stats['max_error'] = max(
                    self.solve_stats['max_error'], error
                )
            else:
                # 求解失败：使用上一帧姿态（保持连续性）
                trajectory_q.append(q_current.copy())
                self.solve_stats['failed'] += 1
                self.solve_stats['failed_indices'].append(idx)

                if self.verbose and idx < 10:
                    print(f"  警告: 点 {idx} 求解失败, 误差: {error:.6f}")

        # 打印统计
        success_rate = 100.0 * self.solve_stats['success'] / n_points if n_points > 0 else 0
        if self.verbose:
            print(f"[PoseSolver] IK 求解完成:")
            print(f"  - 成功率: {self.solve_stats['success']}/{n_points} ({success_rate:.1f}%)")
            print(f"  - 最大误差: {self.solve_stats['max_error']:.6f}")
            if self.solve_stats['failed_indices']:
                failed_count = len(self.solve_stats['failed_indices'])
                print(f"  - 失败点索引 ({failed_count} 个): {self.solve_stats['failed_indices'][:5]}")

        return trajectory_q

    def export_trajectory(
        self,
        trajectory_q: List[np.ndarray],
        output_path: str = "joint_trajectory.csv"
    ) -> None:
        """
        导出手柄轨迹为 CSV

        参数:
            trajectory_q: 关节角度轨迹列表
            output_path: 输出 CSV 路径
        """
        if not trajectory_q:
            print("[PoseSolver] 警告: 轨迹为空，无数据可导出")
            return

        nq = len(trajectory_q[0])
        n_frames = len(trajectory_q)

        # 生成列名
        col_names = ','.join([f'j{i+1}' for i in range(nq)])

        # 生成数据行
        rows = []
        for q in trajectory_q:
            row = ','.join(f'{q[i]:.8f}' for i in range(nq))
            rows.append(row)

        # 写入文件
        with open(output_path, 'w') as f:
            f.write(col_names + '\n')
            f.write('\n'.join(rows))
            f.write('\n')

        if self.verbose:
            print(f"[PoseSolver] 关节轨迹已导出: {output_path}")
            print(f"  - 帧数: {n_frames}")
            print(f"  - 关节数: {nq}")
            print(f"  - 单位: radians")

    def solve_and_export(
        self,
        output_path: str = "joint_trajectory.csv",
        **solve_kwargs
    ) -> List[np.ndarray]:
        """
        一步完成求解和导出

        参数:
            output_path: 输出 CSV 路径
            **solve_kwargs: 传递给 solve_trajectory 的参数

        返回:
            List[np.ndarray]: 关节角度轨迹
        """
        trajectory_q = self.solve_trajectory(**solve_kwargs)
        self.export_trajectory(trajectory_q, output_path)
        return trajectory_q

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节限位"""
        return self.engine.get_joint_limits()

    def get_robot_info(self) -> Dict:
        """获取机器人信息"""
        return {
            'urdf_path': self.urdf_path,
            'nq': self.engine.nq,
            'nv': self.engine.nv,
            'joint_names': self.engine.joint_names
        }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="机器人位姿求解器 - 批量 IK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m core.pose_solver --urdf robot.urdf --input toolpath_pose.csv
  python -m core.pose_solver --urdf robot.urdf --input toolpath_pose.csv --output joint_traj.csv
  python -m core.pose_solver --urdf robot.urdf --input toolpath_pose.csv --tolerance 1e-5
        """
    )

    parser.add_argument(
        "--urdf", "-u",
        required=True,
        help="机器人 URDF 文件路径"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入位姿 CSV 路径 (x,y,z,qw,qx,qy,qz)"
    )
    parser.add_argument(
        "--output", "-o",
        default="joint_trajectory.csv",
        help="输出关节轨迹 CSV 路径 (默认: joint_trajectory.csv)"
    )
    parser.add_argument(
        "--max-iter", "-m",
        type=int,
        default=100,
        help="最大迭代次数 (默认: 100)"
    )
    parser.add_argument(
        "--tolerance", "-t",
        type=float,
        default=1e-4,
        help="收敛阈值 (默认: 1e-4)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印详细信息"
    )

    args = parser.parse_args()

    # 验证输入文件
    if not os.path.exists(args.urdf):
        print(f"错误: URDF 文件不存在: {args.urdf}")
        return

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return

    # 创建求解器
    solver = PoseSolver(
        urdf_path=args.urdf,
        pose_csv=args.input,
        verbose=not args.quiet
    )

    # 打印机器人信息
    info = solver.get_robot_info()
    print(f"\n机器人信息:")
    print(f"  URDF: {info['urdf_path']}")
    print(f"  关节数: {info['nq']}")
    print(f"  关节名称: {info['joint_names']}")

    # 求解并导出
    trajectory_q = solver.solve_and_export(
        output_path=args.output,
        max_iterations=args.max_iter,
        tolerance=args.tolerance
    )

    print(f"\n处理完成!")
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  轨迹长度: {len(trajectory_q)} 帧")


if __name__ == "__main__":
    main()
