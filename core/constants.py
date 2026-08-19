"""
core/constants.py - 全局命名常量

集中管理所有魔法数字，避免代码中出现未命名常量。
使用 Final 类型注解确保运行时不可修改。
"""

from __future__ import annotations

from math import pi
from typing import Final

# =============================================================================
# 单位转换
# =============================================================================

MM_TO_M: Final[float] = 0.001        # 毫米 → 米 (1mm = 0.001m)
M_TO_MM: Final[float] = 1000.0      # 米 → 毫米
DEG_TO_RAD: Final[float] = pi / 180.0  # 度 → 弧度
RAD_TO_DEG: Final[float] = 180.0 / pi  # 弧度 → 度

# =============================================================================
# UI 默认值
# =============================================================================

DEFAULT_SPHERE_THETA: Final[int] = 30
DEFAULT_SPHERE_PHI: Final[int] = 30
DEFAULT_TRAJECTORY_SAMPLE_STEP: Final[int] = 15

# =============================================================================
# 算法权重
# =============================================================================

IK_WEIGHT_DISP: Final[float] = 0.01
IK_WEIGHT_CENTER: Final[float] = 0.01
