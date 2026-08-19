"""
Algorithms Package - Toolpath Generation Plugin Package
====================================================
算法包 - 刀轨生成插件包

本包包含所有可用的刀轨生成算法插件。

新增算法步骤:
1. 在本目录下新建 .py 文件(例如 algo_my_algorithm.py)
2. 创建继承 BaseAlgorithm 的类
3. 设置类的 NAME 与 SUPPORTED_EXTS 属性
4. 实现 load_geometry()、get_parameters() 与 generate() 方法
5. 重启应用,插件管理器会自动发现并注册新算法

可用算法:
    - MeshSlicingAlgorithm: 基于纸张的网格切片,带等弧长重采样

Author: AI Architect
Date: 2026-04-29
"""

from __future__ import annotations

# 导入所有算法类
from .algo_mesh_slicing import MeshSlicingAlgorithm
from .ik_SLSQP import SLSQPSolver
from .ik_PILM import PseudoInverseSolver
from .ik_SLSQP_update import SLSQPMultiSolver
from .batch_ptp_planner import BatchPTPPlanner, BatchPTPPlannerConfig, BatchPTPPlanResult

# Package version
__version__ = "2.0.0"

# Public exports
__all__ = [
    "MeshSlicingAlgorithm",
    "SLSQPSolver",
    "PseudoInverseSolver",
    "SLSQPMultiSolver",
    "BatchPTPPlanner",
    "BatchPTPPlannerConfig",
    "BatchPTPPlanResult",
]
