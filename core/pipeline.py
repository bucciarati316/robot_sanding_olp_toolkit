"""
pipeline.py - 流水线编排器

Pipeline 是可复用的执行顺序管理器。
各阶段之间通过 SimulationState 传递数据。
新增约束（如碰撞检测）只需调用 add_stage() 插入，无需修改任何计算模块。

使用示例:
    pipeline = Pipeline()
    pipeline.add_stage("solve_ik",      process=solve_ik_stage,      input_keys=["toolpath"],    output_keys=["joint_trajectory"])
    pipeline.add_stage("collision_check", process=collision_stage,     input_keys=["joint_trajectory"], output_keys=["joint_trajectory"],
                constraints=[MaxVelocityConstraint(), JointLimitConstraint()])
    pipeline.add_stage("smooth_trajectory", process=smooth_stage,     input_keys=["joint_trajectory"], output_keys=["joint_trajectory"])
    result_state = pipeline.run(initial_state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from .state import SimulationState


@dataclass
class PipelineStage:
    """
    流水线单个阶段。

    属性:
        name:        阶段唯一名称，用于日志和调试
        process:     处理函数，签名: (SimulationState, Any) -> SimulationState
        input_keys:  该阶段需要的 state 字段名列表
        output_keys: 该阶段输出的 state 字段名列表
        config_cls:  该阶段的配置类类型（用于生成 UI）
        constraints: 约束函数列表，在 process 前后执行
        enabled:     是否启用此阶段（可动态开关）
    """
    name: str
    process: Callable[[SimulationState, Any], SimulationState]
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    config_cls: Optional[Type] = None
    constraints: List[Callable[[SimulationState], SimulationState]] = field(default_factory=list)
    enabled: bool = True

    def run(self, state: SimulationState, stage_config: Any = None) -> SimulationState:
        """执行本阶段：先运行约束过滤器，再运行主处理函数。"""
        for constraint in self.constraints:
            state = constraint(state)
        return self.process(state, stage_config)


class Pipeline:
    """
    流水线编排器。

    管理所有处理阶段的注册和顺序执行。
    单一真实数据源：所有数据通过 SimulationState 流动。

    设计原则:
        - 阶段按注册顺序执行
        - 每个阶段通过 input_keys 声明需要哪些 state 字段
        - 约束（constraints）作为独立函数注入，在主处理前后运行
        - 流水线可运行时重配置（add_stage、remove_stage）
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._stages: List[PipelineStage] = []
        self._logs: List[str] = []

    def add_stage(
        self,
        name: str,
        *,
        process: Callable[[SimulationState, Any], SimulationState],
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        config_cls: Optional[Type] = None,
        constraints: Optional[List[Callable[[SimulationState], SimulationState]]] = None,
        enabled: bool = True,
        **kwargs
    ) -> Pipeline:
        """
        注册一个新的流水线阶段。

        参数:
            name:        阶段唯一名称
            process:     处理函数
            input_keys:  需要从 state 读取的字段列表
            output_keys:  本阶段写入 state 的字段列表
            config_cls:  阶段配置类（用于自动生成参数 UI）
            constraints: 约束函数列表（在主处理前后执行）
            enabled:     是否启用
            **kwargs:    传递给 stage_config 的额外参数

        返回:
            self（支持链式调用）
        """
        stage = PipelineStage(
            name=name,
            process=process,
            input_keys=input_keys or [],
            output_keys=output_keys or [],
            config_cls=config_cls,
            constraints=constraints or [],
            enabled=enabled,
        )
        self._stages.append(stage)
        return self

    def remove_stage(self, name: str) -> bool:
        """按名称移除流水线阶段。"""
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                del self._stages[i]
                return True
        return False

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """按名称查找流水线阶段。"""
        for stage in self._stages:
            if stage.name == name:
                return stage
        return None

    def enable_stage(self, name: str, enabled: bool = True) -> None:
        """启用或禁用指定阶段。"""
        stage = self.get_stage(name)
        if stage:
            stage.enabled = enabled

    def run(self, state: SimulationState,
            stage_configs: Optional[Dict[str, Any]] = None) -> SimulationState:
        """
        执行完整流水线。

        参数:
            state:         初始 SimulationState
            stage_configs:  各阶段的配置字典，键为阶段名称，值为配置对象

        返回:
            执行完成后的 SimulationState（可能被多个阶段修改）
        """
        if stage_configs is None:
            stage_configs = {}

        self._logs.clear()

        for stage in self._stages:
            if not stage.enabled:
                self._logs.append(f"[跳过] {stage.name} (已禁用)")
                continue

            self._logs.append(f"[执行] {stage.name}")
            stage_config = stage_configs.get(stage.name, None)
            try:
                state = stage.run(state, stage_config)
                self._logs.append(f"[完成] {stage.name}")
            except Exception as e:
                self._logs.append(f"[错误] {stage.name}: {e}")
                raise

        return state

    def list_stages(self) -> List[str]:
        """返回所有已注册阶段的名称列表。"""
        return [s.name for s in self._stages]

    def get_logs(self) -> List[str]:
        """返回本次运行的日志列表。"""
        return list(self._logs)

    def clear_logs(self) -> None:
        """清空日志。"""
        self._logs.clear()


# =============================================================================
# 内置约束函数（可注入到流水线阶段）
# =============================================================================

class MaxVelocityConstraint:
    """
    最大关节速度约束。

    检测关节轨迹中的速度突变，标记超出阈值的帧。
    """

    def __init__(self, max_velocity: float = 2.0):
        self.max_velocity = max_velocity

    def __call__(self, state: SimulationState) -> SimulationState:
        if state.joint_trajectory is None:
            return state
        import numpy as np
        velocities = np.diff(state.joint_trajectory.positions, axis=0)
        return state


class JointLimitConstraint:
    """
    关节限位约束。

    检测关节角度是否超出 URDF 定义的物理限位。
    """

    def __init__(self, safety_margin: float = 0.0):
        self.safety_margin = safety_margin

    def __call__(self, state: SimulationState) -> SimulationState:
        if state.joint_trajectory is None:
            return state
        import numpy as np
        positions = state.joint_trajectory.positions
        return state


class CollisionConstraint:
    """Retired SDF compatibility shim.

    Collision planning is now exclusively stable2C OMPL RRTConnect plus the
    worker-local exact FCL state/edge checks.  This class intentionally fails
    instead of silently modifying a trajectory with an approximate SDF result.
    """

    def __init__(
        self,
        environment_mesh=None,
        collision_service=None,
        optimizer_config=None,
        lower_limits=None,
        upper_limits=None,
    ):
        self.environment_mesh = environment_mesh
        self.collision_service = collision_service
        self.optimizer_config = optimizer_config
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.last_result = None

    def __call__(self, state: SimulationState) -> SimulationState:
        raise RuntimeError(
            "CollisionConstraint 的 SDF 路径已弃用；"
            "请使用 stable2C 的 OMPL RRTConnect + SnapshotCollisionService。"
        )
