"""
coordinate_transforms.py - 纯函数形式的坐标系变换

将 OCF/WCF/RCF 坐标系变换逻辑提取为无状态的纯函数。
各函数接收变换矩阵作为参数，不持有任何内部状态。

坐标系说明:
    WCF (World Coordinate Frame):   世界坐标系
    RCF (Robot Control Frame):      机器人基座坐标系
    OCF (Object Control Frame):     工件（加工对象）坐标系

变换链（逆向，从右到左应用）:
    T_rcf_target = T_wcf_rcf^-1 @ T_wcf_ocf @ T_ocf_target
"""

from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# 单矩阵变换
# =============================================================================

def ocf_to_wcf(T_ocf: np.ndarray, T_wcf_ocf: np.ndarray) -> np.ndarray:
    """
    将位姿矩阵从 OCF（工件坐标系）变换到 WCF（世界坐标系）。

    公式: T_wcf = T_wcf_ocf @ T_ocf
    """
    return T_wcf_ocf @ T_ocf


def ocf_to_rcf(T_ocf: np.ndarray,
                T_wcf_rcf: np.ndarray,
                T_wcf_ocf: np.ndarray) -> np.ndarray:
    """
    将位姿矩阵从 OCF（工件坐标系）变换到 RCF（机器人基座坐标系）。

    公式: T_rcf = T_wcf_rcf^-1 @ T_wcf_ocf @ T_ocf
    """
    return np.linalg.inv(T_wcf_rcf) @ T_wcf_ocf @ T_ocf


def wcf_to_rcf(T_wcf: np.ndarray, T_wcf_rcf: np.ndarray) -> np.ndarray:
    """
    将位姿矩阵从 WCF（世界坐标系）变换到 RCF（机器人基座坐标系）。

    公式: T_rcf = T_wcf_rcf^-1 @ T_wcf
    """
    return np.linalg.inv(T_wcf_rcf) @ T_wcf


def rcf_to_wcf(T_rcf: np.ndarray, T_wcf_rcf: np.ndarray) -> np.ndarray:
    """
    将位姿矩阵从 RCF（机器人基座坐标系）变换到 WCF（世界坐标系）。

    公式: T_wcf = T_wcf_rcf @ T_rcf
    """
    return T_wcf_rcf @ T_rcf


def wcf_to_ocf(T_wcf: np.ndarray, T_wcf_ocf: np.ndarray) -> np.ndarray:
    """
    将位姿矩阵从 WCF（世界坐标系）变换到 OCF（工件坐标系）。

    公式: T_ocf = T_wcf_ocf^-1 @ T_wcf
    """
    return np.linalg.inv(T_wcf_ocf) @ T_wcf


# =============================================================================
# 批量矩阵变换
# =============================================================================

def ocf_to_wcf_batch(matrices_ocf: List[np.ndarray],
                      T_wcf_ocf: np.ndarray) -> List[np.ndarray]:
    """批量将位姿矩阵从 OCF 变换到 WCF。"""
    return [ocf_to_wcf(T, T_wcf_ocf) for T in matrices_ocf]


def ocf_to_rcf_batch(matrices_ocf: List[np.ndarray],
                      T_wcf_rcf: np.ndarray,
                      T_wcf_ocf: np.ndarray) -> List[np.ndarray]:
    """批量将位姿矩阵从 OCF 变换到 RCF。"""
    return [ocf_to_rcf(T, T_wcf_rcf, T_wcf_ocf) for T in matrices_ocf]


def wcf_to_rcf_batch(matrices_wcf: List[np.ndarray],
                      T_wcf_rcf: np.ndarray) -> List[np.ndarray]:
    """批量将位姿矩阵从 WCF 变换到 RCF。"""
    return [wcf_to_rcf(T, T_wcf_rcf) for T in matrices_wcf]


def rcf_to_wcf_batch(matrices_rcf: List[np.ndarray],
                     T_wcf_rcf: np.ndarray) -> List[np.ndarray]:
    """批量将位姿矩阵从 RCF 变换到 WCF。"""
    return [rcf_to_wcf(T, T_wcf_rcf) for T in matrices_rcf]


# =============================================================================
# 点和法向量变换（矩阵形式，非齐次坐标）
# =============================================================================

def transform_points_with_normals(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    transform: np.ndarray,
    apply_rotation: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    将点集和法向量集通过给定的变换矩阵进行变换。

    参数:
        points:         Nx3 点坐标数组
        normals:        Nx3 法向量数组（可选）
        transform:      4x4 齐次变换矩阵
        apply_rotation: 是否旋转法向量（默认 True）

    返回:
        (变换后的点, 变换后的法向量或 None)
    """
    R = transform[:3, :3]
    t = transform[:3, 3]

    transformed_points = points @ R.T + t

    if normals is not None and len(normals) > 0:
        if apply_rotation:
            transformed_normals = normals @ R.T
        else:
            transformed_normals = normals
    else:
        transformed_normals = None

    return transformed_points, transformed_normals


def transform_points_only(
    points: np.ndarray,
    transform: np.ndarray
) -> np.ndarray:
    """
    仅变换点坐标，不处理法向量。
    比 transform_points_with_normals 效率更高（无需处理法向量分支）。
    """
    R = transform[:3, :3]
    t = transform[:3, 3]
    return points @ R.T + t


# =============================================================================
# 工具方法
# =============================================================================

def is_identity(T: np.ndarray, tol: float = 1e-9) -> bool:
    """判断 4x4 矩阵是否接近单位矩阵（恒等变换）。"""
    return np.allclose(T, np.eye(4), atol=tol)


def compose_transform(translation: np.ndarray,
                     rotation: np.ndarray) -> np.ndarray:
    """
    由平移向量和旋转矩阵合成 4x4 齐次变换矩阵。

    参数:
        translation: 3x1 平移向量
        rotation:   3x3 旋转矩阵

    返回:
        4x4 齐次变换矩阵
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation.flatten() if translation.ndim > 1 else translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    计算 4x4 齐次变换矩阵的逆矩阵。

    对于 SE(3) 变换矩阵，逆矩阵为:
        | R^T  -R^T @ t |
        | 0       1     |
    """
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
