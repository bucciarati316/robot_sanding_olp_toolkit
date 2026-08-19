"""
SDF 距离场碰撞服务，用于 IK 和轨迹优化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pinocchio as pin

from collision.distance_field import EnvironmentDistanceField
from collision.link_proxy import LinkProxyModel


@dataclass(frozen=True)
class MinDistanceResult:
    """最小距离结果"""
    distance: float
    link_name: Optional[str]
    point_world: Optional[np.ndarray]
    backend: str


@dataclass(frozen=True)
class TrajectoryCollisionReport:
    """轨迹碰撞报告"""
    min_distance: float
    colliding_indices: list[int]
    samples_checked: int
    backend: str


class CollisionDistanceService:
    """
    组合 Pinocchio 正向运动学、连杆代理点和环境距离场。
    """

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        link_proxies: LinkProxyModel,
        distance_field: EnvironmentDistanceField,
        *,
        base_transform: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.data = data
        self.link_proxies = link_proxies
        self.distance_field = distance_field
        self.base_transform = np.eye(4, dtype=np.float64) if base_transform is None else np.asarray(base_transform, dtype=np.float64)
        self._frame_ids: Dict[str, int] = {}
        for link_name in link_proxies.link_names:
            try:
                fid = model.getFrameId(link_name)
                if fid < len(model.frames):
                    self._frame_ids[link_name] = fid
            except Exception:
                continue

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        env_mesh,
        *,
        max_points_per_link: int = 64,
        proxy_radius: float = 0.01,
        prefer_open3d: bool = True,
    ) -> "CollisionDistanceService":
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        proxies = LinkProxyModel.from_urdf(
            urdf_path,
            max_points_per_link=max_points_per_link,
            proxy_radius=proxy_radius,
        )
        field = EnvironmentDistanceField(env_mesh, prefer_open3d=prefer_open3d)
        return cls(model, data, proxies, field)

    def min_distance(self, q: np.ndarray) -> MinDistanceResult:
        """计算当前关节构型下机器人与环境之间的最小距离"""
        all_points: list[np.ndarray] = []
        point_links: list[str] = []
        radii: list[float] = []
        link_poses = self._link_world_poses(q)

        for link_name, proxy in self.link_proxies.items():
            T = link_poses.get(link_name)
            if T is None:
                continue
            points = self._transform_points(proxy.local_points, T)
            all_points.append(points)
            point_links.extend([link_name] * len(points))
            radii.extend([proxy.radius] * len(points))

        if not all_points:
            return MinDistanceResult(float("inf"), None, None, self.distance_field.backend)

        points_world = np.vstack(all_points)
        distances = self.distance_field.distance(points_world) - np.asarray(radii, dtype=np.float64)
        idx = int(np.argmin(distances))
        return MinDistanceResult(
            float(distances[idx]),
            point_links[idx],
            points_world[idx].copy(),
            self.distance_field.backend,
        )

    def collision_cost(self, q: np.ndarray, margin: float = 0.03) -> float:
        """
        计算碰撞惩罚代价（二次形式）。

        当距离小于安全裕量时产生惩罚，越接近碰撞惩罚越大。
        """
        result = self.min_distance(q)
        violation = max(0.0, float(margin) - result.distance)
        return float(violation * violation)

    def trajectory_collision_report(
        self,
        positions: np.ndarray,
        *,
        margin: float = 0.03,
        segment_samples: int = 3,
    ) -> TrajectoryCollisionReport:
        """
        对整条轨迹进行碰撞检测。

        对每个采样点以及相邻点之间的插值点进行碰撞检测，
        返回最小距离、碰撞索引和检测样本数。
        """
        Q = np.asarray(positions, dtype=np.float64)
        min_distance = float("inf")
        colliding: set[int] = set()
        checked = 0

        for i, q in enumerate(Q):
            d = self.min_distance(q).distance
            checked += 1
            min_distance = min(min_distance, d)
            if d < margin:
                colliding.add(i)

        for i in range(len(Q) - 1):
            qa, qb = Q[i], Q[i + 1]
            for j in range(1, max(1, segment_samples) + 1):
                t = j / (segment_samples + 1)
                q = (1.0 - t) * qa + t * qb
                d = self.min_distance(q).distance
                checked += 1
                min_distance = min(min_distance, d)
                if d < margin:
                    colliding.add(i)
                    colliding.add(i + 1)

        return TrajectoryCollisionReport(
            min_distance=float(min_distance),
            colliding_indices=sorted(colliding),
            samples_checked=checked,
            backend=self.distance_field.backend,
        )

    def _link_world_poses(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """计算所有连杆的世界位姿"""
        q_arr = np.asarray(q, dtype=np.float64).flatten()
        pin.forwardKinematics(self.model, self.data, q_arr)
        pin.updateFramePlacements(self.model, self.data)
        poses: Dict[str, np.ndarray] = {}
        for link_name, fid in self._frame_ids.items():
            poses[link_name] = self.base_transform @ self.data.oMf[fid].homogeneous
        return poses

    @staticmethod
    def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """将局部坐标系的点变换到世界坐标系"""
        homo = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
        return (homo @ T.T)[:, :3]


__all__ = [
    "CollisionDistanceService",
    "MinDistanceResult",
    "TrajectoryCollisionReport",
]
