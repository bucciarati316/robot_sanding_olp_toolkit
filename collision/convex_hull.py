"""
collision/convex_hull.py - 凸包碰撞体构建与 PyVista 可视化

本模块提供从 URDF collision mesh 构建 Qhull 凸包的核心能力：
  - 解析 URDF 中的 collision mesh 路径
  - 构建凸包（支持安全膨胀系数）
  - 磁盘缓存（同名 _hull.stl）
  - PyVista 线框叠加可视化

注意：粗/精碰撞检测和轨迹重规划的逻辑尚未实现。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Set

import numpy as np

# Trimesh (loaded lazily)

import pyvista as pv

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    pin = None

# ---------------------------------------------------------------------------
# 单位制自动检测与处理
# ---------------------------------------------------------------------------

# 常见单位制阈值（用于判断 mesh 是 mm 还是 m）
# 如果 mesh 的 AABB 尺寸 > 这个值，认为是 mm 单位
UNIT_DETECTION_THRESHOLD_METERS = 1.0  # 1 米作为分界线


def detect_mesh_unit_system(mesh_path: str, scale: List[float] = None) -> str:
    """
    自动检测 mesh 文件使用的单位制（米 或 毫米）。

    检测逻辑：
    1. 如果 URDF 中指定了非 1.0 的 scale，假设 mesh 是 mm 单位（设计者已标记）
    2. 否则，计算 mesh 的原始 AABB 尺寸
    3. 如果任意轴尺寸 > 1 米，认为是 mm 单位（因为机器人通常不会超过这个尺寸）

    参数:
        mesh_path: mesh 文件路径
        scale: URDF 中指定的 scale，None 表示 1.0

    返回:
        "meters" 或 "millimeters"
    """
    if not os.path.exists(mesh_path):
        return "unknown"

    # 如果 URDF 中有非 1.0 的 scale，说明设计者已经标记了单位转换
    if scale is not None and scale != [1.0, 1.0, 1.0]:
        return "millimeters"  # URDF 设计者标记为需要转换

    # 没有 scale 或 scale=1.0，检测 mesh 原始尺寸
    try:
        mesh = pv.read(mesh_path)
        points = mesh.points

        # 计算 AABB 尺寸
        bounds = np.ptp(points, axis=0)  # peak-to-peak (max - min)
        max_dimension = np.max(bounds)

        # 如果尺寸 > 1 米，判断为 mm 单位
        if max_dimension > UNIT_DETECTION_THRESHOLD_METERS:
            return "millimeters"
        else:
            return "meters"
    except Exception:
        return "unknown"


def auto_convert_mesh_to_meters(mesh_path: str, pose_info: dict) -> pv.PolyData:
    """
    自动检测 mesh 单位制并转换为米制。

    工作流程：
    1. 检测 mesh 的单位制
    2. 如果是 mm 单位，应用 scale 并缩放顶点
    3. 如果是 m 单位，直接返回原 mesh

    参数:
        mesh_path: mesh 文件路径
        pose_info: URDF pose 信息，包含 xyz, rpy, scale

    返回:
        已转换为米制的 PyVista PolyData
    """
    scale = pose_info.get('scale', [1.0, 1.0, 1.0])

    # 检测单位制
    unit_system = detect_mesh_unit_system(mesh_path, scale)

    # 读取 mesh
    mesh = pv.read(mesh_path)

    if unit_system == "millimeters":
        # mm 单位：应用 scale 并转换为米
        if scale != [1.0, 1.0, 1.0]:
            # 已经指定了 scale，直接应用
            mesh.points[:, 0] *= scale[0]
            mesh.points[:, 1] *= scale[1]
            mesh.points[:, 2] *= scale[2]
        else:
            # 没有指定 scale，mesh 本身是 mm，需要整体缩放
            mesh.points *= 0.001
    # else: m 单位，无需转换

    return mesh


def detect_urdf_unit_system(urdf_path: str) -> str:
    """
    通过采样 URDF 中的 collision mesh 来检测整个 URDF 的单位制。

    策略：检查前几个 mesh 的尺寸，如果超过阈值就判定为 mm 单位。

    参数:
        urdf_path: URDF 文件路径

    返回:
        "meters", "millimeters", 或 "unknown"
    """
    meshes = parse_urdf_collision_meshes(urdf_path)
    if not meshes:
        return "unknown"

    mm_count = 0
    m_count = 0

    # 采样检查（最多 5 个）
    for link_name, mesh_path, pose_info in meshes[:5]:
        if mesh_path is None or not os.path.exists(mesh_path):
            continue

        unit = detect_mesh_unit_system(mesh_path, pose_info.get('scale'))
        if unit == "millimeters":
            mm_count += 1
        elif unit == "meters":
            m_count += 1

    # 多数投票
    if mm_count > m_count:
        return "millimeters"
    elif m_count > mm_count:
        return "meters"
    else:
        # 平票或都是 unknown，假设为米（更常见）
        return "meters"


# ---------------------------------------------------------------------------
# URDF 解析
# ---------------------------------------------------------------------------

def parse_urdf_package_path(urdf_path: str, mesh_path: str) -> Optional[str]:
    """
    解析 URDF 中的 package:// 路径或相对路径，转换为本地文件系统路径。

    参数:
        urdf_path: URDF 文件路径
        mesh_path: URDF 中引用的 mesh 路径

    返回:
        本地文件路径，解析失败返回 None
    """
    urdf_path = os.path.abspath(urdf_path)

    if mesh_path.startswith('package://'):
        remainder = mesh_path[10:]
        parts = remainder.split('/', 1)

        if len(parts) >= 2:
            package_name = parts[0]
            relative_path = parts[1]
            search_dir = os.path.dirname(urdf_path)

            for _ in range(10):
                pkg_path = os.path.join(search_dir, package_name)
                if os.path.isdir(pkg_path):
                    return os.path.normpath(os.path.join(pkg_path, relative_path))

                pkg_xml = os.path.join(search_dir, 'package.xml')
                if os.path.exists(pkg_xml):
                    try:
                        tree = ET.parse(pkg_xml)
                        root = tree.getroot()
                        name_elem = root.find('name')
                        if name_elem is not None and name_elem.text == package_name:
                            return os.path.normpath(os.path.join(search_dir, relative_path))
                    except Exception:
                        pass

                parent = os.path.dirname(search_dir)
                if parent == search_dir:
                    break
                search_dir = parent

    if os.path.isabs(mesh_path):
        return os.path.normpath(mesh_path)

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    return os.path.normpath(os.path.join(urdf_dir, mesh_path))


def parse_urdf_collision_meshes(urdf_path: str) -> List[Tuple[str, str, dict]]:
    """
    解析 URDF 文件，提取所有 collision mesh 信息。

    参数:
        urdf_path: URDF 文件路径

    返回:
        List of (link_name, mesh_full_path, origin_info) tuples.
        origin_info = {'xyz': List[float], 'rpy': List[float], 'scale': List[float]}
    """
    meshes: List[Tuple[str, str, dict]] = []

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for link in root.findall('.//link'):
            link_name = link.get('name', '')

            for collision in link.findall('collision'):
                geometry = collision.find('geometry')
                if geometry is None:
                    continue

                mesh_elem = geometry.find('mesh')
                if mesh_elem is None:
                    continue

                filename = mesh_elem.get('filename', '')
                if not filename:
                    continue

                origin_elem = collision.find('origin')
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

                full_mesh_path = parse_urdf_package_path(urdf_path, filename)
                if full_mesh_path is None:
                    continue

                meshes.append((link_name, full_mesh_path, {
                    'xyz': xyz,
                    'rpy': rpy,
                    'scale': scale,
                }))

    except Exception as e:
        print(f"[parse_urdf_collision_meshes] 解析 URDF 失败: {e}")

    return meshes


# ---------------------------------------------------------------------------
# 凸包构建
# ---------------------------------------------------------------------------

def build_convex_hull(mesh_path: str, margin: float = 0.0) -> Optional[trimesh.Trimesh]:
    """
    从 mesh 文件构建 Qhull 凸包。

    参数:
        mesh_path: collision mesh 文件路径（支持 STL/DAE/OBJ）
        margin: 安全膨胀系数（米），正值向外扩张，负值向内收缩

    返回:
        trimesh.Trimesh 凸包对象，失败返回 None
    """
    if not TRIMESH_AVAILABLE:
        print("[build_convex_hull] trimesh 未安装")
        return None

    mesh_path = os.path.normpath(mesh_path)
    if not os.path.exists(mesh_path):
        print(f"[build_convex_hull] 文件不存在: {mesh_path}")
        return None

    try:
        original = trimesh.load(mesh_path, force='mesh')
    except Exception as e:
        print(f"[build_convex_hull] trimesh.load 失败 ({mesh_path}): {e}")
        return None

    try:
        hull = original.convex_hull
    except Exception as e:
        print(f"[build_convex_hull] convex_hull 失败 ({mesh_path}): {e}")
        return None

    if hull is None:
        print(f"[build_convex_hull] convex_hull 返回 None ({mesh_path})")
        return None

    if margin != 0.0:
        try:
            normals = hull.face_normals
            centroid = hull.vertices.mean(axis=0)
            vertices = hull.vertices - centroid
            lengths = np.linalg.norm(vertices, axis=1, keepdims=True)
            lengths = np.maximum(lengths, 1e-9)
            directions = vertices / lengths
            hull.vertices = hull.vertices + directions * margin
        except Exception as e:
            print(f"[build_convex_hull] margin 偏移失败: {e}")

    return hull


def cache_convex_hull(
    collision_mesh_path: str,
    output_path: Optional[str] = None,
    margin: float = 0.0,
    force_rebuild: bool = False,
) -> Optional[str]:
    """
    生成凸包并缓存到磁盘。

    参数:
        collision_mesh_path: 原始 collision mesh 路径
        output_path: 输出 STL 路径，默认 = collision_mesh 同目录 + "_hull.stl"
        margin: 安全膨胀系数（米）
        force_rebuild: True 则强制重建，忽略已有缓存

    返回:
        缓存文件路径，失败返回 None
    """
    collision_mesh_path = os.path.normpath(collision_mesh_path)

    if output_path is None:
        base_dir = os.path.dirname(collision_mesh_path)
        base_name = os.path.splitext(os.path.basename(collision_mesh_path))[0]
        ext = os.path.splitext(collision_mesh_path)[1]
        if ext.lower() in ('.stl', '.dae', '.obj', '.ply'):
            output_path = os.path.join(base_dir, f"{base_name}_hull.stl")
        else:
            output_path = os.path.join(base_dir, f"{base_name}_hull.stl")

    output_path = os.path.normpath(output_path)

    if not force_rebuild and os.path.exists(output_path):
        return output_path

    hull = build_convex_hull(collision_mesh_path, margin=margin)
    if hull is None:
        return None

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        hull.export(output_path)
        return output_path
    except Exception as e:
        print(f"[cache_convex_hull] 导出失败 ({output_path}): {e}")
        return None


# ---------------------------------------------------------------------------
# FK Provider
# ---------------------------------------------------------------------------

def build_link_fk_provider(
    urdf_path: str,
    base_transform: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], Dict[str, 'pin.SE3']]:
    """
    构建一个 link FK 闭包，供 ConvexHullVisualizer 使用。

    参数:
        urdf_path: URDF 文件路径
        base_transform: 基座变换矩阵（4x4），默认 np.eye(4)

    返回:
        一个可调用对象 fk_provider(q) -> Dict[link_name, pin.SE3]
    """
    if base_transform is None:
        base_transform = np.eye(4)

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    child_to_joint: Dict[str, str] = {}
    for joint_elem in root.findall('./joint'):
        child_elem = joint_elem.find('child')
        parent_elem = joint_elem.find('parent')
        if child_elem is None or parent_elem is None:
            continue
        child_link = child_elem.get('link', '')
        joint_name = joint_elem.get('name', '')
        parent_link = parent_elem.get('link', '')
        if child_link and joint_name:
            child_to_joint[child_link] = joint_name

    root_link = None
    for joint_elem in root.findall('./joint'):
        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')
        if parent_elem is None or child_elem is None:
            continue
        parent_link = parent_elem.get('link', '')
        child_link = child_elem.get('link', '')
        if child_link not in child_to_joint and parent_link and parent_link != 'world':
            root_link = parent_link
            break
    if root_link is None:
        for joint_elem in root.findall('./joint'):
            parent_elem = joint_elem.find('parent')
            if parent_elem is not None:
                candidate = parent_elem.get('link', '')
                if candidate and candidate != 'world':
                    root_link = candidate
                    break
    if root_link is None:
        for joint_elem in root.findall('./joint'):
            child_elem = joint_elem.find('child')
            parent_elem = joint_elem.find('parent')
            if child_elem is not None and parent_elem is not None:
                parent_link = parent_elem.get('link', '')
                child_link = child_elem.get('link', '')
                if (child_link and parent_link
                        and child_link in child_to_joint
                        and parent_link not in child_to_joint):
                    root_link = parent_link
                    break

    def fk_provider(q: np.ndarray) -> Dict[str, 'pin.SE3']:
        q = np.asarray(q, dtype=np.float64).flatten()
        if len(q) < model.nq:
            q = np.pad(q, (0, model.nq - len(q)))

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        joint_world_se3: Dict[str, 'pin.SE3'] = {}
        for jid, jname in enumerate(model.names):
            if jname == 'universe':
                continue
            try:
                joint_world_se3[jname] = data.oMi[jid]
            except Exception:
                pass

        link_fk: Dict[str, 'pin.SE3'] = {}
        visited: set = set()

        def accumulate(link_name: str, world_se3: 'pin.SE3'):
            if link_name in visited:
                return
            visited.add(link_name)
            link_fk[link_name] = world_se3
            if link_name not in child_to_joint:
                return
            joint_name = child_to_joint[link_name]
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is not None:
                accumulate(link_name, joint_se3)

        root_se3 = pin.SE3(np.eye(3), np.zeros(3))
        accumulate(root_link, root_se3)

        for child_link, joint_name in child_to_joint.items():
            if child_link in visited:
                continue
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is not None:
                accumulate(child_link, joint_se3)

        return link_fk

    return fk_provider


def make_transform(xyz: List[float], rpy: List[float], scale: List[float] = None) -> np.ndarray:
    """
    创建 xyz + rpy 变换矩阵（严格遵循 URDF 规范：固定轴外旋 XYZ）。

    注意：scale 参数已废弃，请在加载 mesh 时应用 scale。
    如果传入 scale，将被忽略以避免重复缩放。
    """
    from scipy.spatial.transform import Rotation

    R_mat = np.eye(4, dtype=np.float64)
    if any(rpy):
        rot = Rotation.from_euler('XYZ', rpy)
        R_mat[:3, :3] = rot.as_matrix()

    Txyz = np.eye(4, dtype=np.float64)
    Txyz[:3, 3] = xyz

    return Txyz @ R_mat


# ---------------------------------------------------------------------------
# CollisionMeshVisualizer（精检测：原 collision mesh 黄色线框）
# ---------------------------------------------------------------------------

class CollisionMeshVisualizer:
    """
    精检测碰撞体可视化器 + fcl 实时自碰撞检测。

    直接加载 URDF 中每个 link 的 collision mesh，
    以 PyVista 黄色线框（wireframe）叠加渲染。
    基于 python-fcl 做实时自碰撞检测：检测到碰撞时线框变为红色高亮。

    使用方式::

        fk_provider = build_link_fk_provider(urdf_path)
        fine_viz = CollisionMeshVisualizer(urdf_path, plotter, fk_provider)
        fine_viz.update_joints(q)
        colliding = fine_viz.check_self_collision(q)
        fine_viz.apply_collision_highlight(colliding)
    """

    def __init__(
        self,
        urdf_path: str,
        plotter: 'pv.Plotter',
        link_fk_provider: Callable[[np.ndarray], Dict[str, 'pin.SE3']],
        wireframe_color: str = '#888888',
        wireframe_width: int = 2,
        opacity: float = 0.8,
        enable_collision_check: bool = True,
        render_collision_mesh: bool = True,
    ):
        self._urdf_path = urdf_path
        self._plotter = plotter
        self._fk_provider = link_fk_provider
        self._wireframe_color = wireframe_color
        self._wireframe_width = wireframe_width
        self._opacity = opacity
        self._default_color = wireframe_color
        self._collision_color = '#FF3333'
        self._render_collision_mesh = bool(render_collision_mesh)

        self._actors: Dict[str, pv.Actor] = {}
        self._local_poses: Dict[str, dict] = {}
        self._link_transforms: Dict[str, np.ndarray] = {}
        self._base_transform = np.eye(4)
        self._current_q: np.ndarray | None = None
        self._self_collision_links: set = set()   # 自碰撞高亮的 links
        self._env_collision_links: set = set()    # 环境碰撞高亮的 links

        # ---- fcl 碰撞检测 ----
        self._fcl_objects: Dict[str, 'fcl.CollisionObject'] = {}  # link_name -> fcl CollisionObject
        self._adjacent_pairs: set = set()  # frozenset({link_a, link_b})
        self._fcl_available = False
        self._fcl_req = None
        self._fcl_collision_result = None

        collision_infos = parse_urdf_collision_meshes(urdf_path)
        print(f"[CollisionMeshVisualizer] 找到 {len(collision_infos)} 个 collision mesh")

        # ---- 解析 URDF：建立相邻 link 对（排除自碰撞） ----
        child_to_parent_link: Dict[str, str] = {}
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint_elem in root.findall('./joint'):
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get('link', '')
                child_link = child_elem.get('link', '')
                if parent_link and child_link:
                    child_to_parent_link[child_link] = parent_link
                    # 相邻父子 link 不做碰撞检测
                    self._adjacent_pairs.add(frozenset([parent_link, child_link]))

        print(f"[CollisionMeshVisualizer] 相邻 link 对（排除）: {len(self._adjacent_pairs)} 对")

        # ---- 检测 URDF 单位制 ----
        urdf_unit = detect_urdf_unit_system(urdf_path)
        print(f"[CollisionMeshVisualizer] URDF 单位制: {urdf_unit}")

        # ---- 加载 visual mesh（PyVista）+ fcl mesh（碰撞检测） ----
        for link_name, mesh_path, pose_info in collision_infos:
            if mesh_path is None or not os.path.exists(mesh_path):
                print(f"[CollisionMeshVisualizer] 跳过 (文件不存在): {link_name} -> {mesh_path}")
                continue

            # 自动检测并转换单位制（mm -> m）
            try:
                pv_mesh = auto_convert_mesh_to_meters(mesh_path, pose_info)
            except Exception as e:
                print(f"[CollisionMeshVisualizer] 加载失败 ({mesh_path}): {e}")
                continue

            print(f"  {link_name}: collision mesh 顶点数={pv_mesh.n_points}")
            self._local_poses[link_name] = pose_info

            if self._render_collision_mesh:
                actor = self._plotter.add_mesh(
                    pv_mesh,
                    name=f"fine_{link_name}",
                    color=wireframe_color,
                    style='wireframe',
                    line_width=wireframe_width,
                    opacity=opacity,
                )
                self._actors[link_name] = actor

            # ---- fcl BVHModel（从 PyVista PolyData 提取三角网格） ----
            if enable_collision_check:
                fcl_obj = _pv_to_fcl_collision_object(pv_mesh)
                if fcl_obj is not None:
                    self._fcl_objects[link_name] = fcl_obj
                else:
                    print(f"  [警告] fcl BVHModel 加载失败 ({link_name})")

        # ---- fcl 碰撞请求初始化 ----
        if self._fcl_objects:
            try:
                import fcl
                self._fcl_req = fcl.CollisionRequest(num_max_contacts=1, enable_contact=False)
                self._fcl_available = True
                print(f"[CollisionMeshVisualizer] fcl 就绪（{len(self._fcl_objects)} 个物体）")
            except ImportError:
                print("[CollisionMeshVisualizer] python-fcl 未安装，碰撞检测已禁用")
                self._fcl_available = False

        print(
            f"[CollisionMeshVisualizer] FCL 模型 {len(self._fcl_objects)} 个，"
            f"碰撞网格 actor {len(self._actors)} 个"
        )

    def set_base_transform(self, T: np.ndarray) -> None:
        """设置基座变换矩阵（世界坐标系）"""
        self._base_transform = np.asarray(T, dtype=np.float64)

    def _compute_link_transform(self, link_name: str, link_fk: Dict[str, 'pin.SE3']) -> np.ndarray:
        """计算单个 link 的世界变换矩阵（4x4）"""
        pose_info = self._local_poses.get(link_name, {
            'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'scale': [1, 1, 1]
        })
        local_T = make_transform(pose_info['xyz'], pose_info['rpy'], pose_info['scale'])

        world_se3 = link_fk.get(link_name)
        if world_se3 is not None:
            return self._base_transform @ world_se3.homogeneous @ local_T
        else:
            return self._base_transform @ local_T

    def update_joints(self, q: np.ndarray) -> None:
        """
        根据关节角度更新所有碰撞体线框的变换矩阵。
        同时更新 fcl CollisionObject 的变换供碰撞检测使用。
        """
        self._current_q = np.asarray(q, dtype=np.float64).flatten()
        link_fk = self._fk_provider(self._current_q)

        for link_name in self._local_poses:
            total_T = self._compute_link_transform(link_name, link_fk)
            self._link_transforms[link_name] = total_T

            actor = self._actors.get(link_name)
            if actor is not None:
                actor.user_matrix = total_T

            # ---- 更新 fcl CollisionObject 变换 ----
            if self._fcl_available and link_name in self._fcl_objects:
                R = total_T[:3, :3].astype(np.float64)
                t = total_T[:3, 3].astype(np.float64)
                try:
                    import fcl
                    self._fcl_objects[link_name].setTransform(fcl.Transform(R, t))
                except Exception:
                    pass

    def check_self_collision(self, q: np.ndarray = None) -> List[Tuple[str, link_name]]:
        """
        基于 python-fcl 检测连杆自碰撞（O(N^2) 成对检测）。

        参数:
            q: 关节角度数组（可选，如已通过 update_joints 更新则传 None）

        返回:
            List[Tuple[str, str]]: 发生碰撞的 link 对列表，如 [('link_2', 'link_5'), ...]
        """
        if not self._fcl_available:
            return []

        if q is not None:
            self.update_joints(q)

        import fcl
        link_names = list(self._fcl_objects.keys())
        n = len(link_names)

        colliding_pairs: List[Tuple[str, str]] = []
        for i in range(n):
            for j in range(i + 1, n):
                key = frozenset([link_names[i], link_names[j]])
                if key in self._adjacent_pairs:
                    continue  # 跳过相邻关节对

                res = fcl.CollisionResult()
                ret = fcl.collide(
                    self._fcl_objects[link_names[i]],
                    self._fcl_objects[link_names[j]],
                    self._fcl_req,
                    res,
                )
                if ret > 0:
                    colliding_pairs.append((link_names[i], link_names[j]))

        return colliding_pairs

    def apply_env_collision_highlight(self, colliding_links: Set[str]) -> None:
        """
        高亮与外部环境发生碰撞的 link 线框为红色。
        支持多个 link 同时与环境碰撞的情况。
        与自碰撞高亮状态独立，互不覆盖。
        """
        self._env_collision_links = colliding_links.copy()
        self._update_highlight_colors()

    def apply_collision_highlight(self, colliding_pairs: List[Tuple[str, str]]) -> None:
        """
        根据自碰撞检测结果更新线框颜色：正常=黄色，碰撞中=红色。
        与环境碰撞高亮状态独立，互不覆盖。
        """
        highlighted_links = set()
        for link_a, link_b in colliding_pairs:
            highlighted_links.add(link_a)
            highlighted_links.add(link_b)

        self._self_collision_links = highlighted_links
        self._update_highlight_colors()

    def _update_highlight_colors(self) -> None:
        """根据自碰撞 + 环境碰撞状态统一更新所有 actor 颜色"""
        all_colliding = self._self_collision_links | self._env_collision_links
        for link_name, actor in self._actors.items():
            actor.prop.color = (
                self._collision_color if link_name in all_colliding
                else self._default_color
            )

    def get_current_colliding_links(self) -> set:
        """返回当前所有碰撞状态（自碰撞 + 环境碰撞）的 link 名称集合"""
        return self._self_collision_links | self._env_collision_links

    def get_link_transforms(self) -> Dict[str, np.ndarray]:
        """返回最新的 collision link 世界变换，不依赖 PyVista actor。"""
        return {
            link_name: transform.copy()
            for link_name, transform in self._link_transforms.items()
        }

    def set_visible(self, visible: bool) -> None:
        """切换碰撞体线框的可见性"""
        for actor in self._actors.values():
            actor.visibility = visible

    def clear(self) -> None:
        """移除所有碰撞体 actor"""
        for link_name, actor in list(self._actors.items()):
            self._plotter.remove_actor(actor, reset_camera=False)
        self._actors.clear()
        self._local_poses.clear()
        self._link_transforms.clear()
        self._fcl_objects.clear()


# ---------------------------------------------------------------------------
# 共享工具：PyVista -> fcl
# ---------------------------------------------------------------------------

def _pv_to_fcl_collision_object(pv_mesh: 'pv.PolyData') -> Optional['fcl.CollisionObject']:
    """
    从 PyVista PolyData 构建 fcl.CollisionObject（BVHModel）。

    流程:
        1. triangulate() 确保全为三角形
        2. cast_to_unstructured_grid() 提取 cells 数组
        3. 解析 UnstructuredGrid cells 格式：[n_verts, v0, v1, ...] 重复 N 次
        4. 构建 fcl.BVHModel 并 addSubModel
        5. 返回 fcl.CollisionObject（初始 transform = identity at origin）

    返回:
        fcl.CollisionObject 实例，失败时返回 None
    """
    try:
        import fcl
        tri_mesh = pv_mesh.triangulate()
        ug = tri_mesh.cast_to_unstructured_grid()

        cells_arr = ug.cells
        tris = []
        idx = 0
        for _ in range(ug.n_cells):
            n_verts = int(cells_arr[idx])
            idx += 1
            if n_verts == 3:
                tris.append([cells_arr[idx], cells_arr[idx + 1], cells_arr[idx + 2]])
            idx += n_verts
        tris = np.array(tris, dtype=np.int32)

        verts = np.ascontiguousarray(ug.points.astype(np.float64))
        bvh = fcl.BVHModel()
        bvh.beginModel(len(verts), len(tris))
        bvh.addSubModel(verts, tris)
        bvh.endModel()

        return fcl.CollisionObject(bvh, fcl.Transform(np.eye(3), np.zeros(3)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# WireframeVisualizerBase（4 类线框的抽象基类）
# ---------------------------------------------------------------------------

class WireframeVisualizerBase:
    """
    4 类（机器人/CAD/环境/刀具）线框的抽象基类。

    统一接口：
      - apply_wireframe_highlight(is_red: bool)  切换线框颜色（True=红色高亮，False=恢复默认）
      - reset_color()                            恢复默认颜色
      - set_visible(visible: bool)               切换显隐
    """

    @property
    def default_wireframe_color(self) -> str:
        return self._wireframe_color

    @property
    def collision_color(self) -> str:
        return '#FF3333'

    def apply_wireframe_highlight(self, is_red: bool) -> None:
        """切换线框颜色：True=红，False=默认"""
        target = self.collision_color if is_red else self._wireframe_color
        for actor in self._wireframe_actors:
            if actor is None:
                continue
            try:
                actor.prop.color = target
            except Exception:
                pass
        if is_red and hasattr(self, '_solid_actors'):
            for actor in self._solid_actors:
                try:
                    actor.prop.color = self.collision_color
                except Exception:
                    pass
        if getattr(self, '_highlight_source_actor', False):
            source_actor = getattr(self, '_source_actor', None)
            if source_actor is not None:
                try:
                    source_actor.prop.color = (
                        self.collision_color if is_red
                        else self._source_default_color
                    )
                except Exception:
                    pass

    def reset_color(self) -> None:
        """恢复默认颜色"""
        self.apply_wireframe_highlight(False)

    def set_visible(self, visible: bool) -> None:
        """切换显隐（子类可重写以同时控制实体）"""
        for actor in getattr(self, '_wireframe_actors', ()):
            if actor is not None:
                actor.visibility = visible


# ---------------------------------------------------------------------------
# EnvironmentMeshVisualizer（环境碰撞体：外部 STL + 静态变换）
# ---------------------------------------------------------------------------

class EnvironmentMeshVisualizer(WireframeVisualizerBase):
    """
    通用静态网格可视化器：任意 STL + 4x4 变换 + fcl 碰撞模型。

    本质上表示"一个放置在世界中某位置、不随机器人运动的静态碰撞体"，
    历史上被环境物体、刀具（CollisionWidget）、CAD 数模复用。**所有用法
    走同一个类，仅靠 wireframe_color 区分视觉**。

    视觉特征：
        - 实体（半透明 opacity=0.5）
        - 外扩线框（与机器人灰色碰撞线框区分）
        - 同一 mesh 同时渲染两层
        - 碰撞时整体变红

    使用方式::

        env_viz = EnvironmentMeshVisualizer(
            mesh_path='environment.stl',
            plotter=plotter,
            transform=ENV_T_wcf,   # 4x4 世界位姿
            wireframe_color='#1E90FF',  # 默认蓝
        )
        # 后续修改位姿
        env_viz.set_world_transform(ENV_T_wcf)
    """

    def __init__(
        self,
        mesh_path: str = None,
        plotter: 'pv.Plotter' = None,
        transform: np.ndarray = None,
        color: str = 'white',
        wireframe_color: str = '#888888',
        wireframe_width: int = 2,
        opacity: float = 0.5,
        name_prefix: str = 'env',
        wireframe_only: bool = False,
        source_actor: 'pv.Actor' = None,
        preloaded_mesh: 'pv.PolyData' = None,
        render_collision_mesh: bool = True,
    ):
        self._plotter = plotter
        self._color = color
        self._wireframe_color = wireframe_color
        self._wireframe_width = wireframe_width
        self._opacity = opacity
        self._name_prefix = name_prefix
        self._wireframe_only = wireframe_only
        self._source_actor = source_actor  # 已存在的实体 actor（其 user_matrix 即 SOoT）
        self._render_collision_mesh = bool(render_collision_mesh)
        self._highlight_source_actor = (
            not self._render_collision_mesh and source_actor is not None
        )
        try:
            self._source_default_color = source_actor.prop.color
        except Exception:
            self._source_default_color = color
        self._mesh_path = mesh_path  # 可能为 None（当 preloaded_mesh 提供时）

        self._world_transform = (
            np.eye(4) if transform is None
            else np.asarray(transform, dtype=np.float64)
        )

        # ---- 优先使用调用者传入的 mesh，否则从文件加载 ----
        if preloaded_mesh is not None:
            self._pv_mesh = preloaded_mesh
            print(f"[EnvironmentMeshVisualizer] 使用预加载 mesh（{name_prefix}）"
                  f": {preloaded_mesh.n_points} pts, {preloaded_mesh.n_cells} cells")
        else:
            if mesh_path is None or not os.path.exists(mesh_path):
                raise FileNotFoundError(
                    f"[EnvironmentMeshVisualizer] 必须提供 mesh_path 或 preloaded_mesh"
                )
            # 单位制检测与换算
            from collision.convex_hull import detect_mesh_unit_system, auto_convert_mesh_to_meters
            unit = detect_mesh_unit_system(mesh_path)
            if unit == "millimeters":
                print(f"[EnvironmentMeshVisualizer] 检测到 mm 单位，换算到 m: {os.path.basename(mesh_path)}")
                self._pv_mesh = auto_convert_mesh_to_meters(mesh_path, {})
            else:
                self._pv_mesh = pv.read(mesh_path)

        # ---- 渲染：生产模式可完全跳过 collision actor，仅保留 FCL 模型 ----
        if not self._render_collision_mesh:
            if source_actor is None:
                raise ValueError(
                    "[EnvironmentMeshVisualizer] 不渲染碰撞网格时必须提供 source_actor"
                )
            self._solid_actor = None
            self._wireframe_actor = None
            self._solid_actors = ()
            self._wireframe_actors = ()
        elif wireframe_only:
            # wireframe-only 模式：不创建实体，复用 source_actor 的 user_matrix
            if source_actor is None:
                raise ValueError(
                    f"[EnvironmentMeshVisualizer] wireframe_only=True 时必须提供 source_actor"
                )
            self._solid_actor = None
            self._solid_actors = ()
            self._wireframe_actor = self._plotter.add_mesh(
                self._pv_mesh,
                name=self._name_prefix + "_wireframe",
                color=wireframe_color,
                style='wireframe',
                line_width=wireframe_width,
                opacity=1.0,
            )
            # 初始变换：直接拷贝 source_actor 的 user_matrix
            try:
                self._wireframe_actor.user_matrix = np.asarray(source_actor.user_matrix)
            except Exception:
                self._wireframe_actor.user_matrix = self._world_transform
        else:
            # 传统双层模式：实体 + 线框
            self._solid_actor = self._plotter.add_mesh(
                self._pv_mesh,
                name=self._name_prefix + "_solid",
                color=color,
                opacity=opacity,
            )
            self._wireframe_actor = self._plotter.add_mesh(
                self._pv_mesh,
                name=self._name_prefix + "_wireframe",
                color=wireframe_color,
                style='wireframe',
                line_width=wireframe_width,
                opacity=1.0,
            )
            # 初始变换
            for actor in (self._solid_actor, self._wireframe_actor):
                actor.user_matrix = self._world_transform

        # 兼容基类：基类期望 _wireframe_actors / _solid_actors 列表
        if self._render_collision_mesh:
            self._wireframe_actors = (self._wireframe_actor,)
            if self._solid_actor is not None:
                self._solid_actors = (self._solid_actor,)
            else:
                self._solid_actors = ()

        # ---- 构造时立即构建 fcl BVHModel ----
        self._fcl_obj = _pv_to_fcl_collision_object(self._pv_mesh)
        if self._fcl_obj is None:
            print(f"[EnvironmentMeshVisualizer] 警告: fcl BVHModel 构建失败")
        else:
            self._apply_fcl_transform()
            print(f"[EnvironmentMeshVisualizer] fcl BVHModel 就绪")

    def _apply_fcl_transform(self) -> None:
        """将 self._world_transform 应用到 fcl CollisionObject"""
        if self._fcl_obj is None:
            return
        try:
            import fcl
            R = self._world_transform[:3, :3].astype(np.float64)
            t = self._world_transform[:3, 3].astype(np.float64)
            self._fcl_obj.setTransform(fcl.Transform(R, t))
        except Exception:
            pass

    def set_world_transform(self, T: np.ndarray) -> None:
        """
        设置世界坐标系下的 4x4 变换矩阵，并同步更新所有可视化与碰撞模型。

        参数:
            T: 4x4 齐次变换矩阵
        """
        self._world_transform = np.asarray(T, dtype=np.float64)
        if self._source_actor is not None and (
            self._wireframe_only or not self._render_collision_mesh
        ):
            # wireframe-only 模式：以 source_actor 的 user_matrix 为准（单一事实源）
            try:
                self._world_transform = np.asarray(self._source_actor.user_matrix)
                if self._wireframe_actor is not None:
                    self._wireframe_actor.user_matrix = self._world_transform
            except Exception:
                if self._wireframe_actor is not None:
                    self._wireframe_actor.user_matrix = self._world_transform
        else:
            for actor in (self._solid_actor, self._wireframe_actor):
                if actor is not None:
                    actor.user_matrix = self._world_transform
        self._apply_fcl_transform()

    def sync_from_source(self) -> None:
        """
        wireframe-only 模式下，从 source_actor 拉取最新 user_matrix 并应用到 wireframe actor。

        应在每帧调用（在 render_engine 更新 source actor user_matrix 后）。
        """
        if (
            self._source_actor is None
            or (not self._wireframe_only and self._render_collision_mesh)
        ):
            return
        try:
            T_src = np.asarray(self._source_actor.user_matrix, dtype=np.float64)
        except Exception:
            return
        self._world_transform = T_src
        if self._wireframe_actor is not None:
            try:
                self._wireframe_actor.user_matrix = T_src
            except Exception:
                pass
        self._apply_fcl_transform()

    def set_translation(self, xyz) -> None:
        """仅修改平移分量（保留当前旋转）"""
        T = self._world_transform.copy()
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        self.set_world_transform(T)

    def set_rotation_matrix(self, R_mat) -> None:
        """仅修改旋转矩阵（3x3，保留当前平移）"""
        T = self._world_transform.copy()
        T[:3, :3] = np.asarray(R_mat, dtype=np.float64)
        self.set_world_transform(T)

    def set_pose(self, xyz, quat) -> None:
        """
        从位置 + 四元数（qx, qy, qz, qw）设置位姿（与 pose_solver.PoseSolver._build_transform 接口一致）。
        """
        from scipy.spatial.transform import Rotation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = np.asarray(xyz, dtype=np.float64)
        self.set_world_transform(T)

    def get_world_transform(self) -> np.ndarray:
        """返回当前世界变换矩阵（4x4）"""
        return self._world_transform.copy()

    def get_fcl_object(self) -> Optional['fcl.CollisionObject']:
        """返回 fcl CollisionObject（供环境↔机器人碰撞检测使用）"""
        return self._fcl_obj

    def set_visible(self, visible: bool) -> None:
        """切换环境物体的可见性（实体+线框）"""
        if self._solid_actor is not None:
            self._solid_actor.visibility = visible
        if self._wireframe_actor is not None:
            self._wireframe_actor.visibility = visible

    def apply_env_collision_highlight(self, is_colliding: bool) -> None:
        """
        兼容旧 API：转发到基类统一接口 `apply_wireframe_highlight`。
        """
        self.apply_wireframe_highlight(is_colliding)

    def clear(self) -> None:
        """从 plotter 中移除环境物体 actor（仅移除线框 actor，source_actor 留给 RenderEngine 管）"""
        self.reset_color()
        if self._wireframe_actor is not None:
            try:
                self._plotter.remove_actor(self._wireframe_actor, reset_camera=False)
            except Exception:
                pass
        if self._solid_actor is not None:
            try:
                self._plotter.remove_actor(self._solid_actor, reset_camera=False)
            except Exception:
                pass
        self._fcl_obj = None


# ---------------------------------------------------------------------------
# CADWireframeVisualizer（CAD 数模线框：绿色实线）
# ---------------------------------------------------------------------------

class CADWireframeVisualizer(EnvironmentMeshVisualizer):
    """
    CAD 数模的线框可视化：默认灰色实线。

    **不创建新 mesh**，完全复用 RenderEngine._cad_model 的 mesh 和 actor：
        - mesh: 已通过 render_engine.get_cad_mesh() 获取（mm -> m 已转换）
        - source_actor: RenderEngine._cad_model（其实体的 user_matrix 为唯一事实源）
    本类只额外叠加一个 wireframe actor，每帧通过 sync_from_source() 跟随 source actor。
    """

    def __init__(
        self,
        plotter: 'pv.Plotter',
        source_actor: 'pv.Actor',
        preloaded_mesh: 'pv.PolyData',
        transform: np.ndarray = None,
        wireframe_color: str = '#888888',
        wireframe_width: int = 2,
        name_prefix: str = 'cad',
        render_collision_mesh: bool = True,
    ):
        super().__init__(
            mesh_path=None,
            plotter=plotter,
            transform=transform if transform is not None else np.eye(4),
            color=wireframe_color,
            wireframe_color=wireframe_color,
            wireframe_width=wireframe_width,
            opacity=0.0,  # 不创建实体
            name_prefix=name_prefix,
            wireframe_only=True,
            source_actor=source_actor,
            preloaded_mesh=preloaded_mesh,
            render_collision_mesh=render_collision_mesh,
        )
        # CAD 无独立 solid actor
        self._solid_actor = None
        self._solid_actors = ()

    def set_visible(self, solid_visible: bool, wireframe_visible: bool = True) -> None:
        """
        独立控制实体/线框的显隐。

        参数:
            solid_visible: 是否显示 CAD 实体（一般留给 RenderEngine 管）
            wireframe_visible: 是否显示线框
        """
        try:
            self._solid_actor.visibility = solid_visible
        except Exception:
            pass
        try:
            self._wireframe_actor.visibility = wireframe_visible
        except Exception:
            pass
        self._solid_visible_flag = solid_visible


# ---------------------------------------------------------------------------
# ToolWireframeVisualizer（刀具线框：黄色实线 + 末端 FK 驱动）
# ---------------------------------------------------------------------------

class ToolWireframeVisualizer(EnvironmentMeshVisualizer):
    """
    刀具线框可视化器：默认白色实线。

    **不创建新 mesh**，完全复用 RenderEngine._tool_actor 的 mesh 和 actor：
        - mesh: render_engine.get_tool_mesh()（mm -> m + 几何对齐已转换）
        - source_actor: RenderEngine._tool_actor
    每帧通过 sync_from_source() 跟随 source actor 的 user_matrix（即末端 FK）。
    """

    DEFAULT_TOOL_WIRE_COLOR = '#FFFFFF'

    def __init__(
        self,
        plotter: 'pv.Plotter',
        source_actor: 'pv.Actor',
        preloaded_mesh: 'pv.PolyData',
        wireframe_color: str = '#FFFFFF',
        wireframe_width: int = 2,
        name_prefix: str = 'tool',
        render_collision_mesh: bool = True,
    ):
        super().__init__(
            mesh_path=None,
            plotter=plotter,
            transform=np.eye(4),
            color=wireframe_color,
            wireframe_color=wireframe_color,
            wireframe_width=wireframe_width,
            opacity=0.0,
            name_prefix=name_prefix,
            wireframe_only=True,
            source_actor=source_actor,
            preloaded_mesh=preloaded_mesh,
            render_collision_mesh=render_collision_mesh,
        )
        # 工具无独立 solid actor
        self._solid_actor = None
        self._solid_actors = ()

    def update_transform_from_flange(self, T_flange: np.ndarray) -> None:
        """
        每帧调用：推入末端 flange FK 得到的 4x4 变换。

        参数:
            T_flange: 4x4 齐次变换矩阵（世界坐标系）
        """
        self.set_world_transform(T_flange)


# ---------------------------------------------------------------------------
# ConvexHullVisualizer（粗检测：Qhull 凸包线框）
# ---------------------------------------------------------------------------

class ConvexHullVisualizer:
    """
    凸包线框可视化器。

    加载 URDF 中每个 link 的 collision mesh，生成 Qhull 凸包，
    以 PyVista 线框（wireframe）叠加在已有的 visual mesh 之上。
    支持关节角度更新时同步变换凸包。

    使用方式::

        fk_provider = build_link_fk_provider(urdf_path)
        hull_viz = ConvexHullVisualizer(urdf_path, plotter, fk_provider)
        hull_viz.update_joints(q)
    """

    def __init__(
        self,
        urdf_path: str,
        plotter: 'pv.Plotter',
        link_fk_provider: Callable[[np.ndarray], Dict[str, 'pin.SE3']],
        margin: float = 0.0,
        wireframe_color: str = 'cyan',
        wireframe_width: int = 2,
        opacity: float = 0.85,
    ):
        self._urdf_path = urdf_path
        self._plotter = plotter
        self._fk_provider = link_fk_provider
        self._margin = margin
        self._wireframe_color = wireframe_color
        self._wireframe_width = wireframe_width
        self._opacity = opacity

        self._actors: Dict[str, pv.Actor] = {}
        self._local_poses: Dict[str, dict] = {}
        self._base_transform = np.eye(4)

        collision_infos = parse_urdf_collision_meshes(urdf_path)
        print(f"[ConvexHullVisualizer] 找到 {len(collision_infos)} 个 collision mesh")

        for link_name, mesh_path, pose_info in collision_infos:
            if mesh_path is None or not os.path.exists(mesh_path):
                print(f"[ConvexHullVisualizer] 跳过 (文件不存在): {link_name} -> {mesh_path}")
                continue

            cached = cache_convex_hull(mesh_path, margin=margin)
            if cached is None:
                print(f"[ConvexHullVisualizer] 凸包缓存失败: {link_name} -> {mesh_path}")
                continue

            try:
                pv_mesh = pv.read(cached)
            except Exception as e:
                print(f"[ConvexHullVisualizer] PyVista 读取失败 ({cached}): {e}")
                continue

            n_vertices = pv_mesh.n_points
            n_original = 0
            try:
                original_mesh = pv.read(mesh_path)
                n_original = original_mesh.n_points
            except Exception:
                pass

            print(f"  {link_name}: collision顶点数={n_original}, 凸包顶点数={n_vertices}, 缓存={os.path.basename(cached)}")

            self._local_poses[link_name] = pose_info

            actor = self._plotter.add_mesh(
                pv_mesh,
                name=f"hull_{link_name}",
                color=wireframe_color,
                style='wireframe',
                line_width=wireframe_width,
                opacity=opacity,
            )
            self._actors[link_name] = actor

        print(f"[ConvexHullVisualizer] 已渲染 {len(self._actors)} 个凸包线框")

    def set_base_transform(self, T: np.ndarray) -> None:
        """设置基座变换矩阵（世界坐标系）"""
        self._base_transform = np.asarray(T, dtype=np.float64)
        for link_name in self._actors:
            self._force_update_actor(link_name, self._current_q if hasattr(self, '_current_q') else np.zeros(6))

    def update_joints(self, q: np.ndarray) -> None:
        """
        根据关节角度更新所有凸包线框的变换矩阵。

        参数:
            q: 关节角度数组（长度应 >= nq）
        """
        self._current_q = np.asarray(q, dtype=np.float64).flatten()

        link_fk = self._fk_provider(self._current_q)

        for link_name, actor in self._actors.items():
            pose_info = self._local_poses.get(link_name, {'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'scale': [1, 1, 1]})
            local_T = make_transform(
                pose_info['xyz'],
                pose_info['rpy'],
                pose_info['scale'],
            )

            world_se3 = link_fk.get(link_name)
            if world_se3 is not None:
                total_T = self._base_transform @ world_se3.homogeneous @ local_T
            else:
                total_T = self._base_transform @ local_T

            actor.user_matrix = total_T

    def _force_update_actor(self, link_name: str, q: np.ndarray) -> None:
        """强制更新单个 actor"""
        pose_info = self._local_poses.get(link_name, {'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'scale': [1, 1, 1]})
        local_T = make_transform(pose_info['xyz'], pose_info['rpy'], pose_info['scale'])

        link_fk = self._fk_provider(np.asarray(q, dtype=np.float64).flatten())
        world_se3 = link_fk.get(link_name)
        if world_se3 is not None:
            total_T = self._base_transform @ world_se3.homogeneous @ local_T
        else:
            total_T = self._base_transform @ local_T

        if link_name in self._actors:
            self._actors[link_name].user_matrix = total_T

    def set_visible(self, visible: bool) -> None:
        """切换凸包线框的可见性"""
        for actor in self._actors.values():
            actor.visibility = visible

    def clear(self) -> None:
        """移除所有凸包 actor"""
        for link_name, actor in list(self._actors.items()):
            self._plotter.remove_actor(actor, reset_camera=False)
        self._actors.clear()
        self._local_poses.clear()
