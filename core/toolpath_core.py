"""刀路位姿工具函数。

刀路插件的唯一行为协议是 :class:`core_algorithm.BaseAlgorithm`，跨模块
结果类型是 :class:`schemas.ToolpathResult`。旧的 ``BaseToolpathGenerator``
平行继承链未被生产代码使用，已移除。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def check_pose_matrix(matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """检查矩阵是否为合法的 4x4 SE(3) 齐次变换。"""
    if matrix.shape != (4, 4):
        return False

    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=tolerance):
        return False
    if abs(np.linalg.det(rotation) - 1.0) > tolerance:
        return False
    return bool(
        np.allclose(
            matrix[3, :],
            np.array([0.0, 0.0, 0.0, 1.0]),
            atol=tolerance,
        )
    )


def create_pose_from_position_and_normal(
    position: np.ndarray,
    normal: np.ndarray,
    up_hint: Optional[np.ndarray] = None,
) -> np.ndarray:
    """从位置和表面法向构造右手 SE(3) 位姿矩阵。"""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-12:
        raise ValueError("法向量不能为零向量")
    z_axis = normal / normal_norm

    reference = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up_hint is None
        else np.asarray(up_hint, dtype=np.float64).reshape(3)
    )
    if abs(np.dot(z_axis, reference)) >= 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis_raw = reference - np.dot(reference, z_axis) * z_axis
    x_axis_norm = np.linalg.norm(x_axis_raw)
    x_axis = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if x_axis_norm < 1e-12
        else x_axis_raw / x_axis_norm
    )
    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = position
    return transform
