"""
连杆代理点生成模块，用于距离场碰撞代价计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from collision.convex_hull import (
    auto_convert_mesh_to_meters,
    make_transform,
    parse_urdf_collision_meshes,
)


@dataclass(frozen=True)
class LinkProxy:
    """单个连杆的代理点数据"""
    link_name: str
    local_points: np.ndarray
    radius: float


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """对点云进行降采样以限制数量"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points.copy()
    idx = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[idx].copy()


def _mesh_proxy_points(mesh, max_points: int) -> np.ndarray:
    """从 mesh 提取代理点，包括顶点和面心"""
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        tri = mesh.triangulate()
        ug = tri.cast_to_unstructured_grid()
        cells = np.asarray(ug.cells)
        centroids = []
        idx = 0
        for _ in range(ug.n_cells):
            n_verts = int(cells[idx])
            idx += 1
            if n_verts == 3:
                face = np.asarray([cells[idx], cells[idx + 1], cells[idx + 2]], dtype=int)
                centroids.append(np.asarray(ug.points[face], dtype=np.float64).mean(axis=0))
            idx += n_verts
        if centroids:
            points = np.vstack([points, np.asarray(centroids, dtype=np.float64)])
    except Exception:
        pass
    return _downsample_points(points, max_points)


class LinkProxyModel:
    """连杆代理点集合，管理所有连杆的本地代理点。"""

    def __init__(self, proxies: Iterable[LinkProxy]):
        self._proxies: Dict[str, LinkProxy] = {p.link_name: p for p in proxies}

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        *,
        max_points_per_link: int = 64,
        proxy_radius: float = 0.01,
    ) -> "LinkProxyModel":
        """从 URDF 文件解析并生成所有连杆的代理点"""
        proxies = []
        for link_name, mesh_path, pose_info in parse_urdf_collision_meshes(urdf_path):
            try:
                mesh = auto_convert_mesh_to_meters(mesh_path, pose_info)
                points = _mesh_proxy_points(mesh, max_points_per_link)
                if points.size == 0:
                    continue
                local_T = make_transform(
                    pose_info.get("xyz", [0.0, 0.0, 0.0]),
                    pose_info.get("rpy", [0.0, 0.0, 0.0]),
                    pose_info.get("scale", [1.0, 1.0, 1.0]),
                )
                homo = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
                local_points = (homo @ local_T.T)[:, :3]
                proxies.append(LinkProxy(link_name, local_points, float(proxy_radius)))
            except Exception:
                continue
        return cls(proxies)

    @property
    def link_names(self) -> list[str]:
        """返回所有连杆名称列表"""
        return list(self._proxies.keys())

    def get(self, link_name: str) -> Optional[LinkProxy]:
        """获取指定连杆的代理点"""
        return self._proxies.get(link_name)

    def items(self):
        """返回连杆名称到代理点的迭代器"""
        return self._proxies.items()

    def __len__(self) -> int:
        """返回连杆数量"""
        return len(self._proxies)


__all__ = ["LinkProxy", "LinkProxyModel"]
