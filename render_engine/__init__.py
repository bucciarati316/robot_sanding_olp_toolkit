"""
render_engine/__init__.py - 渲染引擎包

整合 PyVista 渲染、机器人 URDF、Open3D 点云处理。

子模块:
    _robot_visualizer: RobotVisualizer 机器人可视化组件
    _pointcloud_processor: PointCloudProcessor 点云处理器
    _render_utils: URDF 解析和几何工具函数

主类:
    RenderEngine: facade，组合上述组件提供完整渲染 API
"""

from __future__ import annotations

from .render_engine import RenderEngine
from ._robot_visualizer import RobotVisualizer
from ._pointcloud_processor import PointCloudProcessor
from ._render_utils import (
    parse_urdf_package_path,
    parse_urdf_meshes,
    build_rotation_to_align,
)

__all__ = [
    "RenderEngine",
    "RobotVisualizer",
    "PointCloudProcessor",
    "parse_urdf_package_path",
    "parse_urdf_meshes",
    "build_rotation_to_align",
]
