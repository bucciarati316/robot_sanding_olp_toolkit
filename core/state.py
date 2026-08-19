"""
state.py - 中心化仿真状态容器

SimulationState 是全项目唯一的全局状态对象。
所有跨模块的数据（配置、数据流、仿真状态）都集中存储在此，
避免数据分散在各个模块内部导致的状态不一致。

设计原则:
    - 配置类字段（robot_config 等）在会话期间通常不变化
    - 数据类字段（workpiece_cloud 等）沿流水线流动更新
    - 引擎引用由 pipeline 创建和销毁，不长期持有
    - UI 状态（is_playing 等）保留在 main_app 中，不放入此容器
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, Dict
from uuid import uuid4

import numpy as np

from .schemas import (
    RobotConfig,
    FlangeToolParams,
    CoordinateFrames,
    PointCloud,
    Toolpath,
    ToolpathResult,
    JointTrajectory,
    IKBatchResult,
    ProcessParameters,
    ProcessSegment,
    PathSequencingConfig,
    TimeParameterizedTrajectory,
    TransitionRequest,
    TrajectoryValidationReport,
)

APPLICATION_STATE_SCHEMA_VERSION = "1.0"


def _create_tool_library():
    """延迟创建刀具库，避免 schemas/state 导入链产生循环依赖。"""
    from .tool_library import ToolLibrary

    return ToolLibrary()


@dataclass
class SimulationState:
    """
    全项目仿真全局状态容器。

    字段分为三层:
        配置层 — 每个会话设置一次，通常不变化
        数据层 — 沿流水线流动，每阶段更新
        引用层 — 引擎/工具库等共享对象的弱引用

    配置层:
        robot_config:       机器人配置（URDF 路径、关节限位等）
        flange_tool_params: 法兰盘/刀具偏移参数
        coordinate_frames:  坐标系变换矩阵集合（WCF/RCF/OCF）

    数据层:
        workpiece_cloud:   原始工件点云
        processed_cloud:   预处理后的点云
        toolpath:          生成的刀轨（位置+法向量+TCP矩阵）
        joint_trajectory:  IK 求解后的关节轨迹

    IK 结果:
        ik_results:   IK 求解详细结果列表
        valid_frames: 可视化用有效帧索引列表

    CAD 数据:
        cad_filepath: CAD 模型文件路径
        cad_points:   CAD 顶点数据

    仿真状态:
        is_simulating: 是否正在仿真
        current_frame: 当前播放帧索引
        camera_state:  相机状态字典（用于跨 Tab 持久化）

    工作线程:
        is_playing:    是否正在播放轨迹回放
    """

    # ---- 配置层 ----
    robot_config: Optional[RobotConfig] = None
    flange_tool_params: Optional[FlangeToolParams] = field(default_factory=FlangeToolParams)
    coordinate_frames: Optional[CoordinateFrames] = None
    kinematics_engine: Optional[Any] = None
    tool_library: Any = field(default_factory=_create_tool_library)
    path_seq_config: PathSequencingConfig = field(default_factory=PathSequencingConfig)
    layer_cc_point_count: int = 0
    T_wcf_rcf: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_wcf_ocf: np.ndarray = field(default_factory=lambda: np.eye(4))

    # ---- 数据层 ----
    workpiece_cloud: Optional[PointCloud] = None
    processed_cloud: Optional[PointCloud] = None
    toolpath: Optional[Toolpath] = None
    toolpath_result: Optional[ToolpathResult] = None
    # IK、外部导入和播放器共享的唯一离散关节轨迹契约。
    joint_trajectory: Optional[JointTrajectory] = None
    # Preserve original PROCESS IK after physical time parameterization.
    process_joint_trajectory: Optional[JointTrajectory] = None
    process_parameters: ProcessParameters = field(default_factory=ProcessParameters)
    physical_trajectory: Optional[TimeParameterizedTrajectory] = None
    trajectory_validation: Optional[TrajectoryValidationReport] = None
    physical_trajectory_stale: bool = True
    geometric_trajectory: Optional[Any] = None
    process_segments: List[ProcessSegment] = field(default_factory=list)
    transition_requests: List[TransitionRequest] = field(default_factory=list)
    ik_path_complete: bool = False

    # ---- IK 结果 ----
    ik_results: List[Dict[str, Any]] = field(default_factory=list)
    valid_frames: List[Dict[str, Any]] = field(default_factory=list)
    ik_batch_result: Optional[IKBatchResult] = None

    # ---- 旧 GUI 字段（迁入中心状态，等待各业务模块逐步类型化）----
    raw_points: Optional[np.ndarray] = None
    cropped_points: Optional[np.ndarray] = None
    processed_points: Optional[np.ndarray] = None
    toolpath_points: Optional[np.ndarray] = None
    toolpath_normals: Optional[np.ndarray] = None
    toolpath_preview_normals: Optional[np.ndarray] = None
    target_matrices: Optional[List[np.ndarray]] = None
    joint_trajectory_path: Optional[str] = None

    # ---- CAD 数据 ----
    cad_filepath: Optional[str] = None
    cad_points: Optional[np.ndarray] = None
    tool_filepath: Optional[str] = None
    tool_stl_path: Optional[str] = None
    env_objects: List[dict] = field(default_factory=list)

    # ---- 仿真状态 ----
    is_simulating: bool = False
    current_frame: int = 0
    camera_state: Optional[Dict[str, Any]] = None
    is_playing: bool = False
    trajectory_viz_sample_step: int = 15
    trajectory_viz_axis_length: float = 0.015
    trajectory_viz_point_size: int = 25

    # ---- 共享服务/缓存 ----
    collision_manager: Optional[Any] = None
    # Immutable, actor-free collision input captured on the GUI thread.
    # Workers rebuild their own FCL services from this snapshot.
    collision_scene_snapshot: Optional[Any] = None
    collision_scene_hash: Optional[str] = None
    config_metadata: Dict[str, Any] = field(default_factory=dict)
    _ik_solver_classes_cache: Dict[str, Any] = field(default_factory=dict)
    _ik_solver_names_cache: List[str] = field(default_factory=list)

    # ---- 生命周期审计 ----
    schema_version: str = APPLICATION_STATE_SCHEMA_VERSION
    task_id: str = field(default_factory=lambda: uuid4().hex)
    state_version: int = 0
    scene_version: int = 0
    toolpath_version: int = 0
    trajectory_version: int = 0
    validation_version: int = 0
    current_stage: str = "initialized"
    stage_status: Dict[str, str] = field(default_factory=dict)
    lifecycle_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_stage(
        self,
        stage: str,
        *,
        status: str = "completed",
        details: Optional[Dict[str, Any]] = None,
        version_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """记录生产状态流转，并按数据域递增版本号。"""
        valid_statuses = {"started", "completed", "failed", "stale"}
        if status not in valid_statuses:
            raise ValueError(f"不支持的阶段状态: {status}")

        if status == "completed" and version_domain:
            field_name = f"{version_domain}_version"
            if field_name not in {
                "scene_version",
                "toolpath_version",
                "trajectory_version",
                "validation_version",
            }:
                raise ValueError(f"不支持的版本域: {version_domain}")
            setattr(self, field_name, int(getattr(self, field_name)) + 1)

        self.state_version += 1
        self.current_stage = stage
        self.stage_status[stage] = status
        event = {
            "task_id": self.task_id,
            "state_version": self.state_version,
            "stage": stage,
            "status": status,
            "scene_version": self.scene_version,
            "toolpath_version": self.toolpath_version,
            "trajectory_version": self.trajectory_version,
            "validation_version": self.validation_version,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "details": dict(details or {}),
        }
        self.lifecycle_history.append(event)
        return event

    def mark_collision_scene_changed(
        self,
        source: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        stage: str = "collision_scene_changed",
    ) -> Dict[str, Any]:
        """Invalidate scene-bound planning/validation before incrementing identity."""
        had_physical_trajectory = self.physical_trajectory is not None
        self.physical_trajectory_stale = True
        self.trajectory_validation = None
        return self.record_stage(
            stage,
            details={
                "source": str(source),
                "invalidated_physical_trajectory": bool(had_physical_trajectory),
                **dict(details or {}),
            },
            version_domain="scene",
        )

    def to_summary(self) -> Dict[str, Any]:
        """返回可序列化的中心状态摘要，不包含网格和轨迹大数组。"""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "state_version": self.state_version,
            "scene_version": self.scene_version,
            "toolpath_version": self.toolpath_version,
            "trajectory_version": self.trajectory_version,
            "validation_version": self.validation_version,
            "current_stage": self.current_stage,
            "stage_status": dict(self.stage_status),
            "history_length": len(self.lifecycle_history),
        }
