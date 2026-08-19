"""
algorithms/_utils.py - 算法包共享工具函数
"""

from __future__ import annotations


def _import_core():
    """
    导入核心算法框架。

    Returns:
        tuple: (BaseAlgorithm, ParamDef, ParamType)

    Raises:
        ImportError: 如果 core_algorithm 不可用
    """
    try:
        from core.core_algorithm import BaseAlgorithm, ParamDef, ParamType
        return BaseAlgorithm, ParamDef, ParamType
    except ImportError:
        raise ImportError("core_algorithm required for IK algorithms")
