"""
render_engine/_render_utils.py - 渲染引擎共享工具函数

从 render_engine.py 提取的 URDF 解析和几何辅助函数，
被 render_engine.py facade 及各子模块共同使用。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional

import numpy as np


def parse_urdf_package_path(urdf_path: str, mesh_path: str) -> Optional[str]:
    """
    解析 URDF 中的 package:// 路径，转换为本地文件系统路径

    参数:
        urdf_path: URDF 文件路径
        mesh_path: URDF 中引用的 mesh 路径 (如 package://xxx/meshes/...)

    返回:
        本地文件路径，如果解析失败返回 None
    """
    urdf_path = os.path.abspath(urdf_path)

    if mesh_path.startswith('package://'):
        remainder = mesh_path[10:]
        parts = remainder.split('/', 1)

        if len(parts) >= 2:
            package_name = parts[0]
            relative_path = parts[1]

            urdf_dir = os.path.dirname(urdf_path)

            search_dir = urdf_dir
            for _ in range(10):
                pkg_path = os.path.join(search_dir, package_name)
                if os.path.isdir(pkg_path):
                    full_path = os.path.normpath(os.path.join(pkg_path, relative_path))
                    return full_path

                pkg_xml = os.path.join(search_dir, 'package.xml')
                if os.path.exists(pkg_xml):
                    try:
                        tree = ET.parse(pkg_xml)
                        root = tree.getroot()
                        name_elem = root.find('name')
                        if name_elem is not None and name_elem.text == package_name:
                            full_path = os.path.normpath(os.path.join(search_dir, relative_path))
                            return full_path
                    except Exception:
                        pass

                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent

        relative = mesh_path[10:]
        full_path = os.path.normpath(os.path.join(os.path.dirname(urdf_path), relative.replace('/', os.sep)))
        return full_path

    if os.path.isabs(mesh_path):
        return os.path.normpath(mesh_path)

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    return os.path.normpath(os.path.join(urdf_dir, mesh_path))


def parse_urdf_meshes(urdf_path: str) -> List[Tuple[str, str, dict]]:
    """
    解析 URDF 文件，提取所有视觉 mesh 信息

    参数:
        urdf_path: URDF 文件路径

    返回:
        List of (link_name, mesh_path, origin_pose) tuples
    """
    meshes = []
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('.//link'):
            link_name = link.get('name', '')

            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                if geometry is None:
                    continue

                mesh_elem = geometry.find('mesh')
                if mesh_elem is None:
                    continue

                filename = mesh_elem.get('filename', '')
                if not filename:
                    continue

                origin_elem = visual.find('origin')
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]

                if origin_elem is not None:
                    xyz_str = origin_elem.get('xyz', '0 0 0')
                    rpy_str = origin_elem.get('rpy', '0 0 0')
                    xyz = [float(x) for x in xyz_str.split()]
                    rpy = [float(x) for x in rpy_str.split()]

                scale = [1.0, 1.0, 1.0]
                scale_str = mesh_elem.get('scale')
                if scale_str:
                    scale = [float(x) for x in scale_str.split()]

                color = '#4472C4'
                material = visual.find('material')
                if material is not None:
                    color_elem = material.find('color')
                    if color_elem is not None:
                        rgba_str = color_elem.get('rgba', '1 1 1 1')
                        rgba = [float(x) for x in rgba_str.split()]
                        if len(rgba) >= 3:
                            r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                            color = f'#{r:02x}{g:02x}{b:02x}'

                full_mesh_path = parse_urdf_package_path(urdf_path, filename)

                meshes.append((link_name, full_mesh_path, {
                    'xyz': xyz,
                    'rpy': rpy,
                    'scale': scale,
                    'color': color
                }))

    except Exception as e:
        print(f"[URDF Parser] 解析失败: {e}")

    return meshes


def build_rotation_to_align(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """
    计算将 from_vec 旋转到 to_vec 的旋转矩阵（Rodrigues 公式）。

    参数:
        from_vec: 源向量（会被归一化）
        to_vec: 目标向量（会被归一化）

    返回:
        3x3 旋转矩阵
    """
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)

    if s < 1e-6:
        return np.eye(3) if c > 0 else np.diag([-1, -1, 1])

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return R
