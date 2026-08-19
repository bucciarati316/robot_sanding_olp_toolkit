"""
render_engine/_pointcloud_processor.py - Open3D 点云处理器

从 render_engine.py 提取的 PointCloudProcessor 类，
封装 ICP 配准、统计滤波、下采样等功能。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


class PointCloudProcessor:
    """
    Open3D 点云处理器
    封装 ICP 配准、统计滤波、下采样等功能
    """

    def __init__(self):
        self._source_pcd: Optional[o3d.geometry.PointCloud] = None
        self._target_pcd: Optional[o3d.geometry.PointCloud] = None
        self._result_pcd: Optional[o3d.geometry.PointCloud] = None
        self._transformation = np.eye(4)

    def load_source(self, points: np.ndarray) -> bool:
        """加载源点云"""
        if not OPEN3D_AVAILABLE:
            return False

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            self._source_pcd = pcd
            return True
        except Exception as e:
            print(f"[PointCloudProcessor] 加载源点云失败: {e}")
            return False

    def load_target(self, mesh_or_points) -> bool:
        """加载目标点云/网格"""
        if not OPEN3D_AVAILABLE:
            return False

        try:
            if isinstance(mesh_or_points, np.ndarray):
                # 点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(mesh_or_points)
                self._target_pcd = pcd
            else:
                # 网格采样
                self._target_pcd = mesh_or_points.sample_points_poisson_disk(5000)
            return True
        except Exception as e:
            print(f"[PointCloudProcessor] 加载目标失败: {e}")
            return False

    def icp_refine(self, init_transform: Optional[np.ndarray] = None,
                   max_correspondence_distance: float = 0.1,
                   max_iterations: int = 50) -> Tuple[bool, np.ndarray]:
        """
        ICP 精配准

        返回:
            (success, transformation_matrix)
        """
        if not OPEN3D_AVAILABLE or self._source_pcd is None or self._target_pcd is None:
            return False, np.eye(4)

        try:
            if init_transform is None:
                init_transform = np.eye(4)

            result = o3d.pipelines.registration.registration_icp(
                self._source_pcd,
                self._target_pcd,
                max_correspondence_distance,
                init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iterations
                )
            )

            self._transformation = result.transformation
            self._result_pcd = self._source_pcd.transform(result.transformation)

            return result.fitness > 0.01, result.transformation

        except Exception as e:
            print(f"[PointCloudProcessor] ICP 失败: {e}")
            return False, np.eye(4)

    def statistical_outlier_removal(self, nb_neighbors: int = 20,
                                    std_ratio: float = 2.0) -> bool:
        """
        统计滤波去除孤立噪点

        参数:
            nb_neighbors: 邻域点数
            std_ratio: 标准差倍数阈值
        """
        if not OPEN3D_AVAILABLE or self._source_pcd is None:
            return False

        try:
            cl, ind = self._source_pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            self._source_pcd = self._source_pcd.select_by_index(ind)
            return True
        except Exception as e:
            print(f"[PointCloudProcessor] 统计滤波失败: {e}")
            return False

    def voxel_downsample(self, voxel_size: float = 0.001) -> bool:
        """体素下采样"""
        if not OPEN3D_AVAILABLE or self._source_pcd is None:
            return False

        try:
            self._source_pcd = self._source_pcd.voxel_down_sample(voxel_size)
            return True
        except Exception as e:
            print(f"[PointCloudProcessor] 下采样失败: {e}")
            return False

    def get_result_points(self) -> Optional[np.ndarray]:
        """获取结果点云"""
        if self._result_pcd is not None:
            return np.asarray(self._result_pcd.points)
        elif self._source_pcd is not None:
            return np.asarray(self._source_pcd.points)
        return None

    def get_transformed_source(self) -> Optional[np.ndarray]:
        """获取变换后的源点云"""
        if self._source_pcd is None:
            return None
        points = np.asarray(self._source_pcd.points)
        return self.apply_transform(points)

    def apply_transform(self, points: np.ndarray) -> np.ndarray:
        """应用当前变换到点云"""
        if points is None or len(points) == 0:
            return points if points is not None else np.empty((0, 3))
        ones = np.ones((points.shape[0], 1))
        homogeneous = np.hstack([points, ones])
        transformed = (self._transformation @ homogeneous.T).T
        return transformed[:, :3]
