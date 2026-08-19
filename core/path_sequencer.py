"""
PathSequencer - 宏观轨迹规划器

功能：
    将刀路点 (x,y,z,nx,ny,nz) 按宏观规划策略转换为四元数位姿 (x,y,z,qw,qx,qy,qz)

宏观规划策略：
    策略 0 - 螺旋型规划器：按原始顺序遍历各层刀路点（同方向连续运动）
    策略 1 - 锯齿型规划器：奇数层顺时针，偶数层逆时针，减少关节限位触发

奇异点检测：
    1. 法向量长度接近零
    2. 切向量与法向量平行（Gram-Schmidt 退化为零）
    3. 相邻点距离过小

使用方法：
    from core.path_sequencer import PathSequencer
    from core.schemas import PathSequencingConfig

    # 输入 CSV 必须包含逐点 ``layer_id``。无分层刀路应显式写为 0。
    config = PathSequencingConfig(strategy=1)
    sequencer = PathSequencer(
        "toolpath_n.csv",
        "toolpath_pose.csv",
        tool_library=tool_library,
        config=config,
    )
    df = sequencer.process()
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional

from trajectory.segmentation import (
    contiguous_layer_runs,
    normalize_layer_ids,
    zigzag_order_indices,
)


# =============================================================================
# 宏观轨迹规划策略
# =============================================================================

def spiral_plan(
    positions: np.ndarray,
    normals: np.ndarray,
    layer_point_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    螺旋型规划器：保持原始顺序。

    所有层按同一方向（顺时针/逆时针）依次遍历，
    适合层间连续加工场景。

    参数:
        positions: (N, 3) 刀路点位置数组
        normals: (N, 3) 刀路点法向量数组
        layer_point_count: 每层刀路点数

    返回:
        (reordered_positions, reordered_normals)
    """
    return positions, normals


def zigzag_plan(
    positions: np.ndarray,
    normals: np.ndarray,
    layer_point_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    锯齿型规划器：偶数层保持顺序，奇数层反向。

    偶数层（0-indexed，即 layer=0,2,4...）：保持原始顺序 [0 .. n-1]
    奇数层（0-indexed，即 layer=1,3,5...）：反向顺序 [n-1 .. 0]

    首尾相接，仅调换顺序，不增加空程移动。

    参数:
        positions: (N, 3) 刀路点位置数组
        normals: (N, 3) 刀路点法向量数组
        layer_point_count: 每层刀路点数

    返回:
        (reordered_positions, reordered_normals)
    """
    if layer_point_count <= 0:
        return positions, normals

    positions = positions.copy()
    normals = normals.copy()
    n_layers = len(positions) // layer_point_count

    for layer in range(n_layers):
        start = layer * layer_point_count
        end = start + layer_point_count
        if layer % 2 == 1:
            positions[start:end] = positions[start:end][::-1]
            normals[start:end] = normals[start:end][::-1]

    return positions, normals


# =============================================================================
# 纯函数：TCP 位姿构建
# =============================================================================

def build_orientation_from_geometry(
    normal: np.ndarray,
    next_pos: np.ndarray,
    pos: np.ndarray,
    end_effector_axis: str = 'z'
) -> Tuple[np.ndarray, bool]:
    """
    从法向量和切向量动态构建刀具方向（表面加工模式，纯函数）。

    刀具方向构建规则：
        - Z_tool = -normal（与法向量反向，刀具指向零件表面）
        - X_tool = 切向量在法向量垂直平面上的投影
        - Y_tool = Z × X（右手坐标系）

    参数:
        normal: 曲面法向量 (3,)
        next_pos: 下一个刀路点位置（用于计算切向）
        pos: 当前刀路点位置
        end_effector_axis: 末端关节旋转轴 ('x', 'y' 或 'z')

    返回:
        (rotation_matrix, is_singular)
    """
    is_singular = False

    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-6:
        tool_dir = np.array([0.0, 0.0, -1.0])
        is_singular = True
    else:
        tool_dir = -normal / normal_norm

    tangent = next_pos - pos
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-6:
        tangent = np.array([1.0, 0.0, 0.0])
        is_singular = True
    else:
        tangent = tangent / tangent_norm

    if end_effector_axis == 'x':
        X_t = tool_dir
        Z_t = tangent - np.dot(tangent, X_t) * X_t
        Z_norm = np.linalg.norm(Z_t)
        if Z_norm < 1e-6:
            Z_t = np.array([0.0, 0.0, 1.0])
            Z_t = Z_t - np.dot(Z_t, X_t) * X_t
            Z_norm = np.linalg.norm(Z_t)
            if Z_norm < 1e-6:
                Z_t = np.array([0.0, 1.0, 0.0])
        Z_t = Z_t / Z_norm
        Y_t = np.cross(Z_t, X_t)

    elif end_effector_axis == 'y':
        Y_t = tool_dir
        X_t = tangent - np.dot(tangent, Y_t) * Y_t
        X_norm = np.linalg.norm(X_t)
        if X_norm < 1e-6:
            X_t = np.array([1.0, 0.0, 0.0])
            X_t = X_t - np.dot(X_t, Y_t) * Y_t
            X_norm = np.linalg.norm(X_t)
            if X_norm < 1e-6:
                X_t = np.array([0.0, 0.0, 1.0])
        X_t = X_t / X_norm
        Z_t = np.cross(X_t, Y_t)

    else:
        Z_t = tool_dir
        X_t = tangent - np.dot(tangent, Z_t) * Z_t
        X_norm = np.linalg.norm(X_t)
        if X_norm < 1e-6:
            X_t = np.array([1.0, 0.0, 0.0])
            X_t = X_t - np.dot(X_t, Z_t) * Z_t
            X_norm = np.linalg.norm(X_t)
            if X_norm < 1e-6:
                X_t = np.array([0.0, 1.0, 0.0])
        X_t = X_t / X_norm
        Y_t = np.cross(Z_t, X_t)

    Y_norm = np.linalg.norm(Y_t)
    if Y_norm < 1e-6:
        Y_t = np.cross(Z_t, X_t)
    else:
        Y_t = Y_t / Y_norm

    return np.column_stack((X_t, Y_t, Z_t)), is_singular


def build_tcp_pose(
    pos: np.ndarray,
    normal: np.ndarray,
    next_pos: np.ndarray,
    end_effector_axis: str = 'z',
    process_geometry: bool = True,
    tool_library=None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    构建刀具坐标系（TCP Frame，纯函数）。

    ABB RAPID 架构：TCP 位置和方向均由本函数输出，
    不再计算法兰盘偏移量。法兰盘偏移由 KinematicsEngine
    通过 Frame 机制自动处理。

    参数:
        pos: 刀路点位置（世界坐标系）
        normal: 曲面法向量
        next_pos: 下一个刀路点位置（用于计算切向）
        end_effector_axis: 末端关节旋转轴 ('x', 'y' 或 'z')
        process_geometry: True=表面加工模式，False=工具标定模式
        tool_library: 工具库实例（标定模式下使用）

    返回:
        (tcp_pos, tcp_rot, is_singular)
            tcp_pos: TCP 位置（等于输入 pos）
            tcp_rot: TCP 旋转矩阵 3x3
            is_singular: 是否为奇异点
    """
    tcp_pos = pos.copy()
    is_singular = False

    if process_geometry:
        tcp_rot, is_singular = build_orientation_from_geometry(
            normal, next_pos, pos, end_effector_axis
        )
    else:
        if tool_library is not None:
            tcp_rot = tool_library.get_current_tool().tcp_rotation
        else:
            tcp_rot = np.eye(3)

    return tcp_pos, tcp_rot, is_singular


def normals_to_quaternions(
    positions: np.ndarray,
    normals: np.ndarray,
    end_effector_axis: str = 'z',
    process_geometry: bool = True,
    tool_library=None,
    minimize_tool_roll: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    纯函数：将位置+法向量数组转换为四元数位姿数组。

    参数:
        positions: (N, 3) 位置数组
        normals: (N, 3) 法向量数组
        end_effector_axis: 末端轴方向
        process_geometry: 是否从几何形状动态构建方向
        tool_library: 工具库实例

    返回:
        (result_array, singular_indices)
            result_array: (N, 7) 数组，每行 [x, y, z, qw, qx, qy, qz]
            singular_indices: 奇异点索引列表
    """
    n_points = len(positions)
    result = np.zeros((n_points, 7))
    singular_indices = []
    previous_rotation = None

    for i in range(n_points):
        pos = positions[i]
        normal = normals[i]
        next_pos = positions[i + 1] if i < n_points - 1 else pos + (pos - positions[i - 1])

        tcp_pos, tcp_rot, is_singular = build_tcp_pose(
            pos, normal, next_pos,
            end_effector_axis=end_effector_axis,
            process_geometry=process_geometry,
            tool_library=tool_library,
        )
        if process_geometry and minimize_tool_roll and previous_rotation is not None:
            tcp_rot, transported = _parallel_transport_tool_roll(
                normal, previous_rotation, tcp_rot, end_effector_axis
            )
            is_singular = is_singular or not transported

        if is_singular:
            singular_indices.append(i)

        quat = R.from_matrix(tcp_rot).as_quat()
        qx, qy, qz, qw = quat
        result[i] = [tcp_pos[0], tcp_pos[1], tcp_pos[2], qw, qx, qy, qz]
        previous_rotation = tcp_rot

    return result, singular_indices


def smooth_path_normals(
    normals: np.ndarray,
    layer_point_count: int = 0,
    window: int = 9,
    layer_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smooth noisy nearest-vertex normals without mixing separate layers."""
    values = np.asarray(normals, dtype=float)
    if window < 3 or len(values) < 3:
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(lengths, 1e-12)
    window = int(window) | 1
    result = values.copy()
    from scipy.signal import savgol_filter
    if layer_ids is not None:
        runs = contiguous_layer_runs(normalize_layer_ids(layer_ids, len(values)))
    elif layer_point_count > 0:
        raise ValueError(
            "smooth_path_normals 不再依据 layer_point_count 推断层边界；"
            "请传入逐点 layer_ids"
        )
    else:
        # 调用方明确没有分层概念时，整个序列是一个单层语义区间。
        runs = [(0, 0, len(values))]
    for _, start, stop in runs:
        segment = values[start:stop]
        if len(segment) and np.allclose(segment, segment[0], atol=1e-12):
            result[start:stop] = segment
            continue
        local_window = min(window, len(segment) if len(segment) % 2 else len(segment) - 1)
        if local_window < 3:
            continue
        mode = 'wrap' if len(runs) > 1 else 'interp'
        result[start:stop] = savgol_filter(
            segment, window_length=local_window, polyorder=min(2, local_window - 1),
            axis=0, mode=mode,
        )
    lengths = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(lengths, 1e-12)


def _parallel_transport_tool_roll(
    normal: np.ndarray,
    previous_rotation: np.ndarray,
    fallback_rotation: np.ndarray,
    end_effector_axis: str,
) -> Tuple[np.ndarray, bool]:
    """Keep the contact axis on the normal while minimizing free-axis roll.

    Sanding discs are rotationally symmetric about the contact axis. Projecting
    the previous transverse axis into the new tangent plane is a discrete
    parallel transport step: it avoids an artificial 360-degree wrist turn on
    every closed contour while preserving the surface-normal constraint.
    """
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return fallback_rotation, False
    tool_direction = -np.asarray(normal, dtype=float) / norm

    if end_effector_axis == 'x':
        x_axis = tool_direction
        z_axis = previous_rotation[:, 2] - np.dot(previous_rotation[:, 2], x_axis) * x_axis
        if np.linalg.norm(z_axis) < 1e-8:
            return fallback_rotation, False
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
    elif end_effector_axis == 'y':
        y_axis = tool_direction
        x_axis = previous_rotation[:, 0] - np.dot(previous_rotation[:, 0], y_axis) * y_axis
        if np.linalg.norm(x_axis) < 1e-8:
            return fallback_rotation, False
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
    else:
        z_axis = tool_direction
        x_axis = previous_rotation[:, 0] - np.dot(previous_rotation[:, 0], z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-8:
            return fallback_rotation, False
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] *= -1.0
    return rotation, True


# =============================================================================
# 主类：PathSequencer
# =============================================================================

class PathSequencer:
    """
    宏观轨迹规划器：将法向量按策略转换为四元数位姿。

    支持螺旋型（strategy=0）和锯齿型（strategy=1）两种规划策略，
    通过 PathSequencingConfig 配置。

    内部调用纯函数 build_tcp_pose() 和 build_orientation_from_geometry()，
    不再持有内部状态。保留向后兼容的 __init__ 和 process() 接口。
    """

    STRATEGY_SPIRAL = 0
    STRATEGY_ZIGZAG = 1

    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        kinematics_engine=None,
        tool_library=None,
        config=None,
    ):
        """
        初始化宏观轨迹规划器。

        参数:
            input_csv: 输入法向量CSV路径 (x,y,z,nx,ny,nz,layer_id)
            output_csv: 输出位姿CSV路径 (x,y,z,qw,qx,qy,qz)
            kinematics_engine: 运动学引擎实例（已废弃，仅保留兼容性）
            tool_library: ToolLibrary 实例（持有 T_flange_tcp 的唯一真实数据源）
            config: PathSequencingConfig 实例（策略配置）
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.singular_indices: List[int] = []
        self._pts: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._kinematics_engine = kinematics_engine
        self._config = config

        if tool_library is not None:
            self._tool_library = tool_library
        else:
            from .tool_library import ToolLibrary
            self._tool_library = ToolLibrary()

    def set_tool_library(self, tool_library) -> None:
        self._tool_library = tool_library

    def process(self) -> pd.DataFrame:
        """
        主处理流程：读取法向量数据 → 宏观规划排序 → 构建 TCP 位姿 → 导出四元数。

        返回:
            pd.DataFrame: 处理后的位姿数据 (x,y,z,qw,qx,qy,qz)
        """
        print(f"正在读取原始刀路数据: {self.input_csv} ...")
        df = pd.read_csv(self.input_csv)

        required_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV 必须包含列: {required_cols}")

        if "layer_id" not in df.columns:
            legacy_count = getattr(self._config, 'layer_point_count', 0) if self._config else 0
            legacy_hint = (
                "；`layer_point_count` 不能再用于反推层号"
                if legacy_count > 0 else ""
            )
            raise ValueError(
                "CSV 缺少必需的逐点 `layer_id` 列。正式刀路只接受显式分层；"
                "无分层刀路请写入全 0 的 `layer_id` 后再执行"
                f"{legacy_hint}"
            )

        self._pts = df[['x', 'y', 'z']].values
        self._normals = df[['nx', 'ny', 'nz']].values
        source_layer_ids = normalize_layer_ids(df["layer_id"].to_numpy(), len(df))
        source_indices = np.arange(len(df), dtype=int)

        end_axis = 'z'
        if self._kinematics_engine is not None:
            end_axis = self._kinematics_engine.get_end_effector_axis()
            print(f"检测到末端执行器轴: {end_axis}")
        else:
            print(f"未提供运动学引擎，使用默认末端轴: {end_axis}")

        # 宏观规划策略排序
        n_points = len(self._pts)
        strategy = getattr(self._config, 'strategy', 0) if self._config else 0
        process_geometry = getattr(self._config, 'process_geometry', True) if self._config else True

        order = (
            zigzag_order_indices(source_layer_ids)
            if strategy == self.STRATEGY_ZIGZAG
            else np.arange(n_points, dtype=int)
        )
        ordered_pts = self._pts[order]
        ordered_normals = self._normals[order]
        ordered_layer_ids = source_layer_ids[order]
        ordered_source_indices = source_indices[order]
        print(
            f"宏观规划: {'锯齿型' if strategy == self.STRATEGY_ZIGZAG else '螺旋型'}，"
            f"使用显式逐点层语义，共 {len(contiguous_layer_runs(ordered_layer_ids))} 层"
        )

        smoothing_window = getattr(self._config, 'normal_smoothing_window', 9) if self._config else 9
        ordered_normals = smooth_path_normals(
            ordered_normals,
            layer_point_count=0,
            window=smoothing_window,
            layer_ids=ordered_layer_ids,
        )

        # 四元数转换
        result_parts = []
        singular_indices = []
        for _, start, stop in contiguous_layer_runs(ordered_layer_ids):
            part, local_singular = normals_to_quaternions(
                ordered_pts[start:stop],
                ordered_normals[start:stop],
                end_effector_axis=end_axis,
                process_geometry=process_geometry,
                tool_library=self._tool_library,
                minimize_tool_roll=getattr(self._config, 'minimize_tool_roll', True) if self._config else True,
            )
            result_parts.append(part)
            singular_indices.extend(start + int(index) for index in local_singular)
        result_array = np.vstack(result_parts) if result_parts else np.empty((0, 7))
        self.singular_indices = singular_indices

        # 转换为 DataFrame
        n_points = len(ordered_pts)
        pose_data = []
        for i in range(n_points):
            pose_data.append({
                'x': result_array[i, 0],
                'y': result_array[i, 1],
                'z': result_array[i, 2],
                'qw': result_array[i, 3],
                'qx': result_array[i, 4],
                'qy': result_array[i, 5],
                'qz': result_array[i, 6],
                'layer_id': int(ordered_layer_ids[i]),
                'original_index': int(ordered_source_indices[i]),
            })

        out_df = pd.DataFrame(pose_data)
        out_df.to_csv(self.output_csv, index=False)

        singular_count = len(singular_indices)
        print(f"宏观轨迹规划完成: {n_points} 点, 末端轴: {end_axis}")
        print(f"  - 奇异点数量: {singular_count}")
        if singular_count > 0 and singular_count <= 10:
            print(f"  - 奇异点索引: {singular_indices}")
        elif singular_count > 10:
            print(f"  - 奇异点索引 (前5个): {singular_indices[:5]} ...")
        print(f"  - 输出文件: {self.output_csv}")

        return out_df

    def get_pose_at_index(self, index: int) -> Optional[dict]:
        """获取指定索引的 TCP 位姿数据。"""
        if self._pts is None or index < 0 or index >= len(self._pts):
            return None

        n = self._normals[index]
        pos = self._pts[index]
        next_pos = self._pts[index + 1] if index < len(self._pts) - 1 else pos

        end_axis = 'z'
        if self._kinematics_engine is not None:
            end_axis = self._kinematics_engine.get_end_effector_axis()

        tcp_pos, tcp_rot, _ = build_tcp_pose(
            pos, n, next_pos,
            end_effector_axis=end_axis,
            process_geometry=getattr(self._config, 'process_geometry', True) if self._config else True,
            tool_library=self._tool_library,
        )
        quat = R.from_matrix(tcp_rot).as_quat()
        qx, qy, qz, qw = quat

        return {
            'x': tcp_pos[0], 'y': tcp_pos[1], 'z': tcp_pos[2],
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz
        }

    def validate_poses(self, df: pd.DataFrame) -> List[int]:
        """验证位姿数据的有效性。"""
        invalid_indices = []
        for idx, row in df.iterrows():
            quat = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
            if abs(np.linalg.norm(quat) - 1.0) > 1e-3:
                invalid_indices.append(idx)
        return invalid_indices


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="宏观轨迹规划器")
    parser.add_argument(
        "--input", "-i",
        default="toolpath_n.csv",
        help="输入法向量CSV路径，必须包含 x,y,z,nx,ny,nz,layer_id (默认: toolpath_n.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="toolpath_pose.csv",
        help="输出位姿CSV路径 (默认: toolpath_pose.csv)"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=int,
        default=0,
        choices=[0, 1],
        help="规划策略: 0=螺旋型, 1=锯齿型 (默认: 0)"
    )
    args = parser.parse_args()

    from .schemas import PathSequencingConfig
    config = PathSequencingConfig(strategy=args.strategy)
    sequencer = PathSequencer(args.input, args.output, config=config)
    sequencer.process()

    print("\n使用说明:")
    print("  1. 将输出的 toolpath_pose.csv 作为 PoseSolver 的输入")
    print("  2. 奇异点已自动处理（使用插值/备用向量）")
    print("  3. 可通过 sequencer.singular_indices 查看奇异点位置")


if __name__ == "__main__":
    main()
