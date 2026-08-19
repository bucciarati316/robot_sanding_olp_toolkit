"""
schemas.py - 类型化数据 Schema 定义

本模块定义所有跨模块边界的数据类型，作为项目的统一数据契约。
所有类型均为 frozen dataclass，确保数据不可变，防止意外修改。

类型层级：
    配置类（Configuration） — 初始化后通常不变化
    数据类（Data）           — 沿流水线流动的数据
    结果类（Result）         — 模块计算返回值
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Mapping

import numpy as np


# =============================================================================
# 机器人配置
# =============================================================================

@dataclass(frozen=True)
class RobotConfig:
    """
    机器人配置数据类。

    属性:
        name: 机器人名称（如 "Demo Six Axis"）
        urdf_path: URDF 文件相对路径
        base_link: 基座 link 名称
        end_effector_link: 末端执行器 link 名称
        default_q_init: 默认初始关节角度（numpy 数组）
        workspace_center: 工作空间中心位置 [x, y, z]
        tool_orientation: 默认刀具姿态（3x3 旋转矩阵）
        joint_limits: 关节限位，shape: (n_joints, 2) = [[lower_i, upper_i], ...]
    """
    name: str
    urdf_path: str
    base_link: str = "base_link"
    end_effector_link: str = "tool0"
    default_q_init: np.ndarray = field(default_factory=lambda: np.zeros(6))
    workspace_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.5]))
    tool_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    joint_limits: Optional[np.ndarray] = None  # shape: (n_joints, 2)

    def to_dict(self) -> dict:
        """序列化为字典（用于 JSON 持久化）。"""
        def _to_list(arr):
            if arr is None:
                return None
            return arr.tolist() if isinstance(arr, np.ndarray) else arr
        return {
            'name': self.name,
            'urdf_path': self.urdf_path,
            'base_link': self.base_link,
            'end_effector_link': self.end_effector_link,
            'default_q_init': _to_list(self.default_q_init),
            'workspace_center': _to_list(self.workspace_center),
            'tool_orientation': _to_list(self.tool_orientation),
            'joint_limits': _to_list(self.joint_limits),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RobotConfig:
        """从字典反序列化（用于从 JSON 加载）。"""
        def _to_array(arr):
            if arr is None:
                return None
            return np.array(arr)
        return cls(
            name=data['name'],
            urdf_path=data['urdf_path'],
            base_link=data.get('base_link', 'base_link'),
            end_effector_link=data.get('end_effector_link', 'tool0'),
            default_q_init=_to_array(data.get('default_q_init')),
            workspace_center=_to_array(data.get('workspace_center')),
            tool_orientation=_to_array(data.get('tool_orientation')),
            joint_limits=_to_array(data.get('joint_limits')),
        )


# =============================================================================
# 工具/法兰盘参数
# =============================================================================

@dataclass
class FlangeToolParams:
    """
    法兰盘与刀具偏移参数（单位: xyz = m, rpy = rad）。

    变换链: 法兰盘 → 刀尖(TCP)
    """
    flange_xyz: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    flange_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    tool_xyz: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    tool_rpy: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


# =============================================================================
# 坐标系变换
# =============================================================================

@dataclass(frozen=True)
class CoordinateFrames:
    """
    世界坐标系下的坐标系变换矩阵集合。

    坐标系说明:
        WCF (World Coordinate Frame):   世界坐标系
        RCF (Robot Control Frame):      机器人基座坐标系
        OCF (Object Control Frame):     工件（加工对象）坐标系

    变换链:
        RCF中的目标点 = T_wcf_rcf^-1 @ WCF中的点
        WCF中的目标点 = T_wcf_ocf @ OCF中的点
        RCF中的目标点 = T_rcf_ocf @ OCF中的点
              其中 T_rcf_ocf = T_wcf_rcf^-1 @ T_wcf_ocf
    """
    T_wcf_rcf: np.ndarray  # 4x4, 世界→机器人基座 (SE3 齐次变换)
    T_wcf_ocf: np.ndarray  # 4x4, 世界→工件坐标系 (SE3 齐次变换)
    T_rcf_ocf: Optional[np.ndarray] = None  # 4x4, 由前两者推导得出

    def __post_init__(self):
        if self.T_rcf_ocf is None and self.T_wcf_rcf is not None and self.T_wcf_ocf is not None:
            # 通过 object.__setattr__ 修改 frozen dataclass 的字段
            object.__setattr__(self, 'T_rcf_ocf',
                np.linalg.inv(self.T_wcf_rcf) @ self.T_wcf_ocf)


# =============================================================================
# 运动学配置
# =============================================================================

@dataclass(frozen=True)
class KinematicsConfig:
    """
    运动学引擎初始化配置。

    包含加载 URDF、设置坐标系、设置工具偏移所需的全部参数。
    """
    urdf_path: str
    flange_tool_params: FlangeToolParams
    coordinate_frames: CoordinateFrames


# =============================================================================
# 点云数据
# =============================================================================

@dataclass(frozen=True)
class PointCloud:
    """
    通用点云数据结构。

    属性:
        points:   Nx3 点坐标数组
        normals:  Nx3 法向量数组（可选）
        colors:   Nx3 RGB 颜色数组（可选）
    """
    points: np.ndarray          # shape: (N, 3)
    normals: Optional[np.ndarray] = None  # shape: (N, 3)
    colors: Optional[np.ndarray] = None   # shape: (N, 3), uint8


# =============================================================================
# 刀轨数据
# =============================================================================

@dataclass(frozen=True)
class Toolpath:
    """
    刀轨数据结构，包含位置、法向量和 TCP 位姿。

    属性:
        positions:  Mx3 位置数组
        normals:    Mx3 法向量数组
        matrices:   Mx4x4 TCP 齐次变换矩阵数组（可选）
        metadata:   附加信息字典（如算法名、时间戳等）
    """
    positions: np.ndarray                # shape: (M, 3)
    normals: np.ndarray                  # shape: (M, 3)
    matrices: Optional[np.ndarray] = None  # shape: (M, 4, 4)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 关节轨迹
# =============================================================================

@dataclass(frozen=True)
class JointTrajectory:
    """
    关节空间轨迹数据。

    属性:
        positions:   Kxn_joints 关节角度数组
        velocities:  Kxn_joints 关节速度数组（可选）
        timestamps:  K 时间戳数组（可选）
        method:      求解器名称（如 "pseudo_inverse"、"SLSQP"）
    """
    positions: np.ndarray                # shape: (K, n_joints)
    velocities: Optional[np.ndarray] = None  # shape: (K, n_joints)
    timestamps: Optional[np.ndarray] = None   # shape: (K,)
    method: str = "unknown"


# =============================================================================
# 物理轨迹与工艺参数
# =============================================================================

class PathSegmentType(Enum):
    """刀路段语义。不同语义的路径不得被无条件跨段平滑。"""

    PROCESS = "process"
    RAPID = "rapid"
    RETRACT = "retract"
    APPROACH = "approach"
    BLEND = "blend"


class TransitionKind(Enum):
    """Named non-process connections that must be geometrically planned."""

    CURRENT_TO_PROCESS = "current_to_process"
    CONFIGURATION_SWITCH = "configuration_switch"
    LAYER_TRANSITION = "layer_transition"
    RETURN_HOME = "return_home"


@dataclass(frozen=True)
class ProcessSegment:
    """一个连续加工层对应的 PROCESS 几何段。"""

    segment_id: int
    layer_id: int
    tcp_poses: np.ndarray
    original_indices: np.ndarray

    def __post_init__(self) -> None:
        poses = np.asarray(self.tcp_poses, dtype=float)
        indices = np.asarray(self.original_indices, dtype=int)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
            raise ValueError("ProcessSegment.tcp_poses 必须为非空 (N,4,4)")
        if indices.shape != (len(poses),):
            raise ValueError("ProcessSegment.original_indices 长度错误")
        object.__setattr__(self, "tcp_poses", poses.copy())
        object.__setattr__(self, "original_indices", indices.copy())

    @property
    def start_pose(self) -> np.ndarray:
        return self.tcp_poses[0].copy()

    @property
    def end_pose(self) -> np.ndarray:
        return self.tcp_poses[-1].copy()


@dataclass(frozen=True)
class TransitionRequest:
    """相邻 PROCESS 段之间尚待模块 C 规划的非加工连接。"""

    kind: TransitionKind
    start_segment_id: int
    goal_segment_id: int
    start_layer_id: int
    goal_layer_id: int
    start_pose: np.ndarray
    goal_pose: np.ndarray

    def __post_init__(self) -> None:
        start = np.asarray(self.start_pose, dtype=float)
        goal = np.asarray(self.goal_pose, dtype=float)
        if start.shape != (4, 4) or goal.shape != (4, 4):
            raise ValueError("TransitionRequest 端点必须为 4x4 TCP 位姿")
        object.__setattr__(self, "start_pose", start.copy())
        object.__setattr__(self, "goal_pose", goal.copy())


class ParameterCapability(Enum):
    """工艺参数当前在系统中的真实能力等级。"""

    ACTIVE_CONSTRAINT = "active_constraint"
    DERIVED = "derived"
    SETPOINT_ONLY = "setpoint_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ProcessParameters:
    """机器人磨削工艺与轨迹约束，内部一律使用 SI 单位。"""

    tcp_feed_rate_mps: float = 0.05
    tcp_acceleration_mps2: float = 0.25
    normal_force_setpoint_n: float = 0.0
    stepover_m: float = 0.005
    tool_tilt_rad: float = 0.0
    corner_blend_radius_m: float = 0.002
    effective_contact_width_m: Optional[float] = None
    minimum_joint_margin_rad: float = np.deg2rad(5.0)
    max_joint_velocity: Optional[np.ndarray] = None
    max_joint_acceleration: Optional[np.ndarray] = None
    max_joint_jerk: Optional[np.ndarray] = None
    control_period_s: float = 0.01
    chord_tolerance_m: float = 0.0002
    orientation_tolerance_rad: float = np.deg2rad(0.5)
    rapid_speed_ratio: float = 1.0
    adaptive_keyframes: bool = True
    max_keyframe_interval_s: float = 0.1
    joint_keyframe_tolerance_rad: float = np.deg2rad(0.05)
    schema_version: int = 2

    def __post_init__(self) -> None:
        positive = {
            "tcp_feed_rate_mps": self.tcp_feed_rate_mps,
            "tcp_acceleration_mps2": self.tcp_acceleration_mps2,
            "stepover_m": self.stepover_m,
            "control_period_s": self.control_period_s,
            "chord_tolerance_m": self.chord_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "rapid_speed_ratio": self.rapid_speed_ratio,
            "max_keyframe_interval_s": self.max_keyframe_interval_s,
            "joint_keyframe_tolerance_rad": self.joint_keyframe_tolerance_rad,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if self.normal_force_setpoint_n < 0:
            raise ValueError("normal_force_setpoint_n 不能为负数")
        if self.corner_blend_radius_m < 0 or self.minimum_joint_margin_rad < 0:
            raise ValueError("圆角半径和关节裕度不能为负数")
        if self.effective_contact_width_m is not None and self.effective_contact_width_m <= 0:
            raise ValueError("effective_contact_width_m 必须为正数或 None")
        for name in ("max_joint_velocity", "max_joint_acceleration", "max_joint_jerk"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                    raise ValueError(f"{name} 必须是一维有限正数数组")
                object.__setattr__(self, name, arr.copy())

    @property
    def overlap_ratio(self) -> Optional[float]:
        """由接触宽度与横向步距派生重叠率；模型未知时返回 None。"""
        if self.effective_contact_width_m is None:
            return None
        return float(np.clip(1.0 - self.stepover_m / self.effective_contact_width_m, 0.0, 1.0))

    @property
    def capabilities(self) -> Mapping[str, ParameterCapability]:
        contact_cap = (
            ParameterCapability.DERIVED
            if self.effective_contact_width_m is not None
            else ParameterCapability.UNAVAILABLE
        )
        return {
            "tcp_feed_rate_mps": ParameterCapability.ACTIVE_CONSTRAINT,
            "tcp_acceleration_mps2": ParameterCapability.ACTIVE_CONSTRAINT,
            "normal_force_setpoint_n": ParameterCapability.SETPOINT_ONLY,
            "stepover_m": ParameterCapability.ACTIVE_CONSTRAINT,
            "tool_tilt_rad": ParameterCapability.ACTIVE_CONSTRAINT,
            "corner_blend_radius_m": ParameterCapability.ACTIVE_CONSTRAINT,
            "minimum_joint_margin_rad": ParameterCapability.ACTIVE_CONSTRAINT,
            "effective_contact_width_m": contact_cap,
            "overlap_ratio": contact_cap,
        }


@dataclass(frozen=True)
class GeometricTrajectory:
    """按累计弧长参数化的离散 SE(3) 几何路径。"""

    path_s: np.ndarray
    tcp_poses: np.ndarray
    segment_types: np.ndarray
    original_indices: np.ndarray
    segment_ids: Optional[np.ndarray] = None
    layer_ids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        s = np.asarray(self.path_s, dtype=float)
        poses = np.asarray(self.tcp_poses, dtype=float)
        kinds = np.asarray(self.segment_types, dtype=object)
        original = np.asarray(self.original_indices, dtype=int)
        if s.ndim != 1 or len(s) < 2 or np.any(np.diff(s) <= 0):
            raise ValueError("path_s 必须包含至少两个严格递增值")
        if poses.shape != (len(s), 4, 4):
            raise ValueError("tcp_poses 必须为 (N, 4, 4)")
        if kinds.shape != (len(s),) or original.shape != (len(s),):
            raise ValueError("segment_types/original_indices 长度必须与 path_s 一致")
        ids = np.zeros(len(s), dtype=int) if self.segment_ids is None else np.asarray(self.segment_ids, dtype=int)
        if ids.shape != (len(s),):
            raise ValueError("segment_ids 长度必须与 path_s 一致")
        layers = np.zeros(len(s), dtype=int) if self.layer_ids is None else np.asarray(self.layer_ids, dtype=int)
        if layers.shape != (len(s),):
            raise ValueError("layer_ids 长度必须与 path_s 一致")
        if not np.all(np.isfinite(poses)):
            raise ValueError("tcp_poses 包含非有限值")
        object.__setattr__(self, "path_s", s.copy())
        object.__setattr__(self, "tcp_poses", poses.copy())
        object.__setattr__(self, "segment_types", kinds.copy())
        object.__setattr__(self, "original_indices", original.copy())
        object.__setattr__(self, "segment_ids", ids.copy())
        object.__setattr__(self, "layer_ids", layers.copy())

    @property
    def length_m(self) -> float:
        return float(self.path_s[-1] - self.path_s[0])

    def evaluate(self, path_s: float) -> np.ndarray:
        """对位置做线性插值、对姿态做最短路径四元数 SLERP。"""
        from scipy.spatial.transform import Rotation, Slerp

        s = float(np.clip(path_s, self.path_s[0], self.path_s[-1]))
        right = int(np.searchsorted(self.path_s, s, side="right"))
        if right == 0:
            return self.tcp_poses[0].copy()
        if right >= len(self.path_s):
            return self.tcp_poses[-1].copy()
        left = right - 1
        span = self.path_s[right] - self.path_s[left]
        ratio = (s - self.path_s[left]) / span
        result = np.eye(4)
        result[:3, 3] = (
            (1.0 - ratio) * self.tcp_poses[left, :3, 3]
            + ratio * self.tcp_poses[right, :3, 3]
        )
        rotations = Rotation.from_matrix(self.tcp_poses[[left, right], :3, :3])
        result[:3, :3] = Slerp([0.0, 1.0], rotations)([ratio]).as_matrix()[0]
        return result


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    tcp_pose: Optional[np.ndarray]
    tcp_speed_mps: float
    segment_id: int
    segment_type: PathSegmentType
    transition_kind: Optional[str] = None


@dataclass(frozen=True)
class TimeParameterizedTrajectory:
    """可按物理时间求值的统一轨迹接口。"""

    timestamps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    jerks: np.ndarray
    tcp_poses: Optional[np.ndarray] = None
    tcp_speeds_mps: Optional[np.ndarray] = None
    segment_ids: Optional[np.ndarray] = None
    segment_types: Optional[np.ndarray] = None
    # ``None`` marks a PROCESS sample.  A non-process sample keeps the
    # stable2C request family that produced it.
    transition_kinds: Optional[np.ndarray] = None
    process_channels: Dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.timestamps, dtype=float)
        q = np.asarray(self.positions, dtype=float)
        qd = np.asarray(self.velocities, dtype=float)
        qdd = np.asarray(self.accelerations, dtype=float)
        qddd = np.asarray(self.jerks, dtype=float)
        if t.ndim != 1 or len(t) < 2 or abs(t[0]) > 1e-12 or np.any(np.diff(t) <= 0):
            raise ValueError("timestamps 必须从 0 开始并严格递增")
        if q.ndim != 2 or q.shape[0] != len(t):
            raise ValueError("positions 必须为 (N, dof)")
        for name, arr in (("velocities", qd), ("accelerations", qdd), ("jerks", qddd)):
            if arr.shape != q.shape:
                raise ValueError(f"{name} 形状必须与 positions 一致")
        if not all(np.all(np.isfinite(arr)) for arr in (t, q, qd, qdd, qddd)):
            raise ValueError("轨迹包含非有限值")
        tcp = None if self.tcp_poses is None else np.asarray(self.tcp_poses, dtype=float)
        if tcp is not None and tcp.shape != (len(t), 4, 4):
            raise ValueError("tcp_poses 必须为 (N, 4, 4)")
        speeds = np.zeros(len(t)) if self.tcp_speeds_mps is None else np.asarray(self.tcp_speeds_mps, dtype=float)
        ids = np.zeros(len(t), dtype=int) if self.segment_ids is None else np.asarray(self.segment_ids, dtype=int)
        kinds = (
            np.full(len(t), PathSegmentType.PROCESS, dtype=object)
            if self.segment_types is None
            else np.asarray(self.segment_types, dtype=object)
        )
        transition_kinds = (
            np.full(len(t), None, dtype=object)
            if self.transition_kinds is None
            else np.asarray(self.transition_kinds, dtype=object)
        )
        if (
            speeds.shape != (len(t),)
            or ids.shape != (len(t),)
            or kinds.shape != (len(t),)
            or transition_kinds.shape != (len(t),)
        ):
            raise ValueError("逐采样元数据长度必须与 timestamps 一致")
        channels = {}
        for name, values in self.process_channels.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (len(t),):
                raise ValueError(f"工艺通道 {name} 长度错误")
            channels[name] = arr.copy()
        for name, value in (("timestamps", t), ("positions", q), ("velocities", qd),
                            ("accelerations", qdd), ("jerks", qddd),
                            ("tcp_speeds_mps", speeds), ("segment_ids", ids),
                            ("segment_types", kinds),
                            ("transition_kinds", transition_kinds)):
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "tcp_poses", None if tcp is None else tcp.copy())
        object.__setattr__(self, "process_channels", channels)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1])

    @property
    def dof(self) -> int:
        return int(self.positions.shape[1])

    def sample_at(self, time_s: float) -> TrajectorySample:
        t = float(np.clip(time_s, 0.0, self.duration_s))
        right = int(np.searchsorted(self.timestamps, t, side="right"))
        if right <= 0:
            left = right = 0
            ratio = 0.0
        elif right >= len(self.timestamps):
            left = right = len(self.timestamps) - 1
            ratio = 0.0
        else:
            left = right - 1
            ratio = (t - self.timestamps[left]) / (self.timestamps[right] - self.timestamps[left])

        def lerp(values: np.ndarray) -> np.ndarray:
            return values[left].copy() if left == right else (1.0 - ratio) * values[left] + ratio * values[right]

        use_quintic = (
            left != right
            and self.metadata.get("keyframe_interpolation") == "quintic_hermite"
        )
        if use_quintic:
            h = float(self.timestamps[right] - self.timestamps[left])
            u = float(ratio)
            q0, q1 = self.positions[left], self.positions[right]
            v0, v1 = self.velocities[left], self.velocities[right]
            a0, a1 = self.accelerations[left], self.accelerations[right]
            displacement = q1 - q0
            c0 = q0
            c1 = v0 * h
            c2 = 0.5 * a0 * h * h
            c3 = 10.0 * displacement - (6.0 * v0 + 4.0 * v1) * h - (1.5 * a0 - 0.5 * a1) * h * h
            c4 = -15.0 * displacement + (8.0 * v0 + 7.0 * v1) * h + (1.5 * a0 - a1) * h * h
            c5 = 6.0 * displacement - 3.0 * (v0 + v1) * h - 0.5 * (a0 - a1) * h * h
            position = c0 + c1 * u + c2 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
            velocity = (c1 + 2.0 * c2 * u + 3.0 * c3 * u**2 + 4.0 * c4 * u**3 + 5.0 * c5 * u**4) / h
            acceleration = (2.0 * c2 + 6.0 * c3 * u + 12.0 * c4 * u**2 + 20.0 * c5 * u**3) / (h * h)
            jerk = (6.0 * c3 + 24.0 * c4 * u + 60.0 * c5 * u**2) / (h**3)
        else:
            position = lerp(self.positions)
            velocity = lerp(self.velocities)
            acceleration = lerp(self.accelerations)
            jerk = lerp(self.jerks)

        tcp_pose = None
        if self.tcp_poses is not None:
            if left == right:
                tcp_pose = self.tcp_poses[left].copy()
            else:
                from scipy.spatial.transform import Rotation, Slerp
                tcp_pose = np.eye(4)
                tcp_pose[:3, 3] = lerp(self.tcp_poses[:, :3, 3])
                rots = Rotation.from_matrix(self.tcp_poses[[left, right], :3, :3])
                tcp_pose[:3, :3] = Slerp([0.0, 1.0], rots)([ratio]).as_matrix()[0]
        nearest = left if left == right or ratio < 0.5 else right
        kind = self.segment_types[nearest]
        if not isinstance(kind, PathSegmentType):
            kind = PathSegmentType(str(kind))
        return TrajectorySample(
            time_s=t,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            tcp_pose=tcp_pose,
            tcp_speed_mps=float(np.interp(t, self.timestamps, self.tcp_speeds_mps)),
            segment_id=int(self.segment_ids[nearest]),
            segment_type=kind,
            transition_kind=(
                None
                if self.transition_kinds[nearest] is None
                else str(self.transition_kinds[nearest])
            ),
        )


@dataclass(frozen=True)
class TrajectoryValidationItem:
    name: str
    passed: bool
    measured: Optional[float] = None
    limit: Optional[float] = None
    time_s: Optional[float] = None
    joint_index: Optional[int] = None
    segment_id: Optional[int] = None
    message: str = ""
    hard_constraint: bool = True


@dataclass(frozen=True)
class TrajectoryValidationReport:
    items: List[TrajectoryValidationItem] = field(default_factory=list)
    generated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.items if item.hard_constraint)

    @property
    def failures(self) -> List[TrajectoryValidationItem]:
        return [item for item in self.items if not item.passed]


# =============================================================================
# 仿真帧
# =============================================================================

@dataclass(frozen=True)
class SimulationFrame:
    """
    单帧仿真状态。

    用于轨迹回放时存储每帧的关节角度和位姿信息。
    """
    q: np.ndarray           # 关节角度, shape: (n_joints,)
    T_flange: np.ndarray   # 法兰盘 4x4 位姿
    T_tcp: np.ndarray      # 刀具 TCP 4x4 位姿
    frame_index: int       # 帧序号


# =============================================================================
# IK 求解结果
# =============================================================================

@dataclass(frozen=True)
class IKSolution:
    """
    单次 IK 求解结果。
    """
    success: bool
    q: Optional[np.ndarray]      # 关节角度解
    error: float                 # 位置误差（m）
    iterations: int              # 迭代次数
    method: str                  # 求解方法


@dataclass(frozen=True)
class IKBatchResult:
    """
    批量 IK 求解结果。
    """
    solutions: List[IKSolution]
    valid_count: int
    total_count: int
    method: str


# =============================================================================
# 插件系统数据类型（来自 core_algorithm.py）
# =============================================================================

class ParamType(Enum):
    """参数类型枚举，决定 UI 控件类型。"""
    FLOAT = "float"   # 浮点数 → 滑块或数值输入
    INT = "int"       # 整数   → 整数滑块或数值输入
    BOOL = "bool"     # 布尔   → 复选框或开关
    CHOICE = "choice"  # 选项   → 下拉菜单


@dataclass(frozen=True, slots=True)
class ParamDef:
    """
    算法参数定义，包含完整元数据用于自动生成 UI 控件。

    属性:
        id:       参数唯一标识符，generate() 调用时的 kwargs 键名
        label:    UI 中显示的友好名称
        ptype:    ParamType 枚举，决定生成何种 UI 控件
        default:  参数默认值
        min_val:  数值类参数的最小值（用于滑块范围）
        max_val:  数值类参数的最大值
        step:     步长（用于滑块/微调框）
        options:  CHOICE 类型的选择列表
        desc:     鼠标悬停提示文本
    """
    id: str
    label: str
    ptype: ParamType
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    desc: str = ""

    def __post_init__(self) -> None:
        if not self.id.isidentifier():
            raise ValueError(f"ParamDef.id '{self.id}' 必须是合法的 Python 标识符")
        if self.ptype is ParamType.CHOICE and not self.options:
            raise ValueError(f"ParamDef '{self.id}' 为 CHOICE 类型但未定义 options")
        if (
            self.ptype in (ParamType.FLOAT, ParamType.INT)
            and self.min_val is not None
            and self.max_val is not None
            and self.min_val > self.max_val
        ):
            raise ValueError(
                f"ParamDef '{self.id}': min_val ({self.min_val}) > "
                f"max_val ({self.max_val})"
            )


class DataType(Enum):
    """可视化数据类型枚举，映射到对应的 3D 渲染图元。"""
    POINT_CLOUD = "point_cloud"   # 散点/小球
    POLYLINE = "polyline"          # 连线段（刀轨拟合前）
    CURVE = "curve"                # 光滑曲线（拟合后刀轨）
    VECTOR_FIELD = "vector_field"   # 箭头场（每个点的方向）


@dataclass(frozen=True, slots=True)
class DebugItem:
    """
    算法中间结果标注，用于可视化调试。

    DebugItem 捕获中间结果的语义标签，使可视化层能够渲染合适的 3D 表达，
    而无需了解算法内部细节。

    属性:
        label:   唯一名称，用作可视化面板中的图层/分组名称
        dtype:    DataType 枚举，决定数据如何渲染
        data:     numpy 数组，实际几何数据
                  - POINT_CLOUD: (N, 3) 三维坐标
                  - POLYLINE:    (N, 3) 按遍历顺序排列的顶点
                  - CURVE:       (N, 3) 曲线的密集采样
                  - VECTOR_FIELD:(N, 6) 每行 [px, py, pz, dx, dy, dz]（位置+方向向量）
        color:    HEX 颜色字符串（默认白色）
        visible:  是否在查看器中默认显示
    """
    label: str
    dtype: DataType
    data: np.ndarray
    color: str = "#FFFFFF"
    visible: bool = True


@dataclass(slots=True)
class ToolpathResult:
    """
    算法输出标准容器：刀轨数据 + 调试可视化数据。

    此数据类确保算法与下游系统（机器人执行、路径优化等）之间的接口一致，
    同时保留中间状态用于可视化调试。

    属性:
        points:      (N, 3) 刀轨上各点的三维位置
        normals:     (N, 3) 各点处的表面法向量（用于姿态定向）
        matrices:    (N, 4, 4) 各路点处的完整位姿齐次变换矩阵（含位置和姿态）
        debug_items: 字典，键为标签，值为 DebugItem 对象，表示中间算法状态
        layer_indices: 每个路点所属的真实层编号；无分层算法统一归入第 0 层
        tcp_matrices_authoritative: matrices 是否可直接作为下游 TCP 权威输入
        tcp_matrix_source: matrices 的产生者，便于下游明确选择而不是静默丢弃
    """
    points: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    normals: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    matrices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    debug_items: Dict[str, DebugItem] = field(default_factory=dict)
    layer_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    tcp_matrices_authoritative: bool = False
    tcp_matrix_source: str = "path_sequencer"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ToolpathResult":
        """规范化并校验跨模块数组；非分层算法自动成为单层。"""
        points = np.asarray(self.points, dtype=float)
        normals = np.asarray(self.normals, dtype=float)
        matrices = np.asarray(self.matrices, dtype=float)
        if points.size == 0:
            points = np.empty((0, 3), dtype=float)
        if normals.size == 0:
            normals = np.empty((0, 3), dtype=float)
        if matrices.size == 0:
            matrices = np.empty((0, 4, 4), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("ToolpathResult.points 必须为 (N,3)")
        if normals.shape != points.shape:
            raise ValueError("ToolpathResult.normals 必须与 points 同为 (N,3)")
        if matrices.shape not in {(0, 4, 4), (len(points), 4, 4)}:
            raise ValueError("ToolpathResult.matrices 必须为空或为 (N,4,4)")
        raw_layers = np.asarray(self.layer_indices)
        if raw_layers.size == 0:
            layers = np.zeros(len(points), dtype=np.int32)
        elif raw_layers.shape != (len(points),):
            raise ValueError("ToolpathResult.layer_indices 必须是逐点 (N,) 层编号")
        else:
            try:
                numeric_layers = np.asarray(raw_layers, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("ToolpathResult.layer_indices 必须是有限整数") from exc
            integer_limits = np.iinfo(np.int32)
            if (
                not np.all(np.isfinite(numeric_layers))
                or not np.all(numeric_layers == np.rint(numeric_layers))
                or np.any(numeric_layers < integer_limits.min)
                or np.any(numeric_layers > integer_limits.max)
            ):
                raise ValueError("ToolpathResult.layer_indices 必须是有限整数")
            layers = numeric_layers.astype(np.int32)
        if self.tcp_matrices_authoritative and len(matrices) != len(points):
            raise ValueError("权威 TCP matrices 不得为空且必须与 points 等长")
        if not all(np.all(np.isfinite(array)) for array in (points, normals, matrices, layers)):
            raise ValueError("ToolpathResult 包含非有限值")
        self.points = points.copy()
        self.normals = normals.copy()
        self.matrices = matrices.copy()
        self.layer_indices = layers.copy()
        return self

    @property
    def layer_ids(self) -> np.ndarray:
        """逐点层编号的规范名称；layer_indices 保留为兼容字段。"""
        return self.layer_indices.copy()

    def add_debug_data(
        self,
        label: str,
        dtype: DataType,
        data: np.ndarray,
        color: str = "#FFFFFF",
        visible: bool = True
    ) -> None:
        """向调试集合中添加中间状态。"""
        self.debug_items[label] = DebugItem(
            label=label, dtype=dtype, data=data, color=color, visible=visible
        )

    @property
    def has_toolpath(self) -> bool:
        """判断结果是否包含有效的刀轨点。"""
        return self.points.size > 0

    @property
    def num_waypoints(self) -> int:
        """返回刀轨路点数量。"""
        return len(self.points)


# =============================================================================
# 平滑/约束配置
# =============================================================================

@dataclass(frozen=True)
class PathSequencingConfig:
    """
    宏观轨迹规划器配置。

    用于 PathSequencer，支持螺旋型和锯齿型两种遍历策略。
    """
    strategy: int = 0              # 0=螺旋型, 1=锯齿型
    layer_point_count: int = 0     # 旧配置兼容字段；正式路径不再据此推断层号
    end_effector_axis: str = 'z'  # 末端执行器轴方向
    process_geometry: bool = True  # 是否从几何形状动态构建方向
    minimize_tool_roll: bool = True  # 沿法向平行运输横轴，避免闭环腕部整圈旋转
    normal_smoothing_window: int = 9  # 分层平滑网格法向，抑制最近邻法向尖点


@dataclass(frozen=True)
class SmoothingConfig:
    """
    轨迹平滑处理器配置。

    用于 TrajectorySmoother 和 ThreeSegmentSmoother。
    """
    max_jump_threshold: float = 0.3   # 最大跳变阈值系数
    velocity_threshold: float = 2.0   # 速度阈值系数
    interpolation_degree: int = 5     # 多项式插值次数（5=五次多项式，3=三次多项式）
    retract_distance: float = 0.1     # 退刀距离（米）
    sign_back: int = -1                # 退刀方向: -1=沿Z轴负方向, 1=沿Z轴正方向
