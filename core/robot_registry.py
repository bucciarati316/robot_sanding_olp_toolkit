"""
机器人配置注册表
提供统一接口获取各种机器人的配置信息

使用方法:
    from core.robot_registry import create_default_registry, RobotConfig

    registry = create_default_registry()
    config = registry.get("Demo Six Axis")
    print(config.name, config.urdf_path, config.default_q_init)
"""

import json
import os
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class RobotConfig:
    """
    机器人配置数据类

    属性:
        name: 机器人名称 (如 "Demo Six Axis")
        urdf_path: URDF 文件相对路径
        base_link: 基座 link 名称
        end_effector_link: 末端执行器 link 名称
        default_q_init: 默认初始关节角度 (6维 numpy 数组)
        workspace_center: 工作空间中心位置 [x, y, z]
        tool_orientation: 默认刀具姿态 (3x3 旋转矩阵)
        joint_limits: 关节限位 [[lower], [upper]]
    """
    name: str
    urdf_path: str
    base_link: str = "base_link"
    end_effector_link: str = "tool0"
    default_q_init: np.ndarray = field(default_factory=lambda: np.zeros(6))
    workspace_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.5, 0.5]))
    tool_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    joint_limits: Optional[np.ndarray] = None  # shape: (6, 2) = [[lower_i, upper_i], ...]

    def to_dict(self) -> dict:
        """
        将 RobotConfig 转换为字典以便 JSON 序列化。

        Returns:
            dict: 可序列化的字典,其中的 numpy 数组已转换为 list。
        """
        def _to_list(arr):
            if arr is None:
                return None
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

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
    def from_dict(cls, data: dict) -> 'RobotConfig':
        """
        根据从 JSON 加载的字典创建 RobotConfig。

        Args:
            data: 包含机器人配置数据的字典。

        Returns:
            RobotConfig: 新创建的 RobotConfig 实例。
        """
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


class RobotRegistry:
    """
    机器人注册表

    管理机器人的注册、查询和枚举
    """

    def __init__(self):
        self._robots: Dict[str, RobotConfig] = {}
        self._builtin_names: set = set()
        self._register_builtin_robots()
        self.load_custom_robots()

    def _register_builtin_robots(self):
        """注册内置机器人配置"""

        # The public repository deliberately ships an authored, mesh-free URDF
        # instead of redistributing a vendor robot package.  Resolve the path
        # from this file so the GUI works from any current directory.
        demo_urdf = (
            Path(__file__).resolve().parents[1]
            / "examples" / "assets" / "urdf" / "demo_six_axis.urdf"
        )
        demo_robot = RobotConfig(
            name="Demo Six Axis",
            urdf_path=str(demo_urdf),
            base_link="base_link",
            end_effector_link="tool0",
            workspace_center=np.array([0.0, 0.0, 0.6]),
            tool_orientation=np.eye(3),
            # joint_limits and default_q_init are parsed from the URDF.
        )
        self.register(demo_robot)

        # 把所有已注册机器人标记为内置
        self._builtin_names = set(self._robots.keys())

    def save_custom_robots(self, filepath: str = "custom_robots.json") -> None:
        """
        把所有非内置机器人保存到 JSON 文件。

        Args:
            filepath: 要保存到的 JSON 文件路径。
        """
        custom_robots = [
            config.to_dict()
            for name, config in self._robots.items()
            if name not in self._builtin_names
        ]

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(custom_robots, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Failed to save custom robots: {e}")

    def load_custom_robots(self, filepath: str = "custom_robots.json") -> None:
        """
        从 JSON 文件加载自定义机器人并注册它们。

        Args:
            filepath: 要加载的 JSON 文件路径。
        """
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                custom_robots = json.load(f)

            for robot_data in custom_robots:
                try:
                    config = RobotConfig.from_dict(robot_data)
                    # 仅当尚未注册时才进行注册
                    if config.name not in self._robots:
                        self.register(config)
                except Exception as e:
                    print(f"[WARN] Failed to load robot '{robot_data.get('name', 'unknown')}': {e}")
        except Exception as e:
            print(f"[ERROR] Failed to load custom robots: {e}")

    def register(self, config: RobotConfig) -> None:
        """
        注册机器人配置

        参数:
            config: RobotConfig 实例
        """
        if not isinstance(config, RobotConfig):
            raise TypeError("config 必须是 RobotConfig 类型")

        self._robots[config.name] = config

    def get(self, name: str) -> Optional[RobotConfig]:
        """
        获取机器人配置

        参数:
            name: 机器人名称

        返回:
            RobotConfig 或 None (如果不存在)
        """
        return self._robots.get(name)

    def list_robots(self) -> List[str]:
        """
        列出所有注册的机器人名称

        返回:
            机器人名称列表
        """
        return list(self._robots.keys())

    def get_default(self) -> Optional[RobotConfig]:
        """
        获取默认机器人配置 (返回第一个注册的)

        返回:
            RobotConfig 或 None
        """
        if self._robots:
            return next(iter(self._robots.values()))
        return None

    def remove(self, name: str) -> bool:
        """
        从注册表中移除指定的机器人配置。

        参数:
            name: 机器人名称

        返回:
            bool: 如果成功移除返回 True，如果机器人为内置或不存在返回 False
        """
        if name in self._builtin_names:
            return False
        if name in self._robots:
            del self._robots[name]
            return True
        return False


# ==================== 全局注册表实例 ====================

_registry: Optional[RobotRegistry] = None


def create_default_registry() -> RobotRegistry:
    """
    创建默认注册表 (单例模式)

    返回:
        全局 RobotRegistry 实例
    """
    global _registry
    if _registry is None:
        _registry = RobotRegistry()
    return _registry


# ==================== 便捷函数 ====================

def get_robot_config(name: str) -> Optional[RobotConfig]:
    """
    便捷函数：获取指定机器人配置

    参数:
        name: 机器人名称

    返回:
        RobotConfig 或 None
    """
    return create_default_registry().get(name)


def list_available_robots() -> List[str]:
    """
    便捷函数：列出所有可用的机器人

    返回:
        机器人名称列表
    """
    return create_default_registry().list_robots()


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("机器人配置注册表测试")
    print("=" * 60)

    # 创建注册表
    registry = create_default_registry()

    # 列出所有机器人
    print(f"\n注册的机器人: {registry.list_robots()}")

    # 获取公开示例机器人配置
    config = registry.get("Demo Six Axis")

    if config:
        print(f"\n机器人名称: {config.name}")
        print(f"URDF 路径: {config.urdf_path}")
        print(f"默认初始关节角度: {config.default_q_init}")
        print(f"工作空间中心: {config.workspace_center}")
        if config.joint_limits is not None:
            print(f"关节限位:\n{config.joint_limits}")

    # 测试便捷函数
    print(f"\n便捷函数测试:")
    print(f"  list_available_robots() = {list_available_robots()}")
    print(f"  get_robot_config('UR5') = {get_robot_config('UR5')}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
