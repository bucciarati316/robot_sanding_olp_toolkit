"""
render_engine/_robot_visualizer.py - 机器人可视化组件

从 render_engine.py 提取的 RobotVisualizer 类，
管理 URDF 模型的加载和实时关节更新。
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

# PyVista
import pyvista as pv

# 共享工具函数
from ._render_utils import parse_urdf_meshes, build_rotation_to_align

if TYPE_CHECKING:
    from pyvistaqt import QtInteractor


class RobotVisualizer:
    """
    机器人可视化组件
    管理 URDF 模型的加载和实时关节更新

    渲染策略：Actor 复用 + 矩阵缓存
    - 初始化时一次性加载所有 mesh 并创建 Actor
    - 关节更新时计算变换矩阵，直接更新 Actor
    - mesh.transform() 在原位变换顶点
    """

    def __init__(self, urdf_path: str, plotter: 'QtInteractor', decimate: float = 0.0):
        """
        初始化机器人可视化

        参数:
            urdf_path: URDF 文件路径
            plotter: PyVista QtInteractor 实例
            decimate: mesh 降采样比例（0.0 = 不降采样, 0.5 = 砍掉 50% 三角面）
                高性能模式建议 0.5~0.7，单帧 draw call 减半
        """
        self._urdf_path = urdf_path
        self._plotter = plotter
        self._decimate = decimate
        self._actors: dict = {}      # link_name -> actor
        self._meshes: dict = {}      # link_name -> 已变换的 mesh
        self._original_meshes: dict = {}  # link_name -> 原始 mesh (未变换)
        self._local_poses: dict = {} # link_name -> pose_info
        self._link_fk: dict = {}     # link_name -> pin.SE3 (world frame)

        # Pinocchio 模型
        self._pinocchio_model = None
        self._pinocchio_data = None
        self._current_q = None
        self._nq = 0
        self.base_transform = np.eye(4)

        # 缓存的 URDF 解析结果（避免重复解析）
        self._child_to_joint: Dict[str, str] = {}
        self._root_link: Optional[str] = None

        self._setup_pinocchio()
        self._load_visuals()

    def _setup_pinocchio(self):
        """初始化 Pinocchio"""
        try:
            import pinocchio as pin
            self._pinocchio_model = pin.buildModelFromUrdf(self._urdf_path)
            self._pinocchio_data = self._pinocchio_model.createData()
            self._nq = self._pinocchio_model.nq
            self._current_q = np.zeros(self._nq)
            print(f"[RobotVisualizer] Pinocchio loaded: {self._nq} joints")

        except Exception as e:
            print(f"[RobotVisualizer] Pinocchio failed: {e}")
            self._pinocchio_model = None
            self._nq = 6

    def _make_transform(self, xyz: List[float], rpy: List[float], scale: List[float] = None) -> np.ndarray:
        """
        创建 xyz + rpy + scale 变换矩阵

        严格遵循 URDF 规范：
        - RPY 使用固定轴外旋（大写 'XYZ'）
        - Scale 应用到变换矩阵的对角线
        """
        from scipy.spatial.transform import Rotation

        T = np.eye(4, dtype=np.float64)

        # 构建缩放矩阵 S
        S = np.eye(4, dtype=np.float64)
        if scale is not None:
            S[0, 0] = scale[0]
            S[1, 1] = scale[1]
            S[2, 2] = scale[2]

        # 构建旋转矩阵 R (固定轴外旋 - URDF 规范)
        R = np.eye(4, dtype=np.float64)
        if any(rpy):
            rot = Rotation.from_euler('XYZ', rpy)
            R[:3, :3] = rot.as_matrix()

        # 构建平移矩阵 Txyz
        Txyz = np.eye(4, dtype=np.float64)
        Txyz[:3, 3] = xyz

        # 正确顺序: T_total = Txyz @ R @ S
        # 即先缩放(mesh 单位→米)，再旋转，再平移
        T = Txyz @ R @ S

        return T

    def _load_visuals(self):
        """一次性加载所有视觉组件并创建 Actor"""
        mesh_infos = parse_urdf_meshes(self._urdf_path)

        if not mesh_infos:
            print("[RobotVisualizer] No visual meshes found, creating simplified model")
            self._create_simple_robot()
            return

        if self._pinocchio_model is None:
            print("[RobotVisualizer] No Pinocchio model — using simplified mode")
            self._create_simple_robot()
            return

        import pinocchio as pin

        # ---- Parse XML: build child_link → joint_name mapping ----
        # 注意: 使用 ./joint 而非 .//joint，避免匹配到 <transmission> 内嵌套的同名 <joint> 元素
        tree = ET.parse(self._urdf_path)
        root = tree.getroot()
        for joint_elem in root.findall('./joint'):
            child_elem = joint_elem.find('child')
            if child_elem is None:
                continue
            child_link = child_elem.get('link', '')
            joint_name = joint_elem.get('name', '')
            if child_link and joint_name:
                self._child_to_joint[child_link] = joint_name

        # ---- Find root link (link never appearing as a child, exclude 'world') ----
        for joint_elem in root.findall('./joint'):
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            if parent_elem is None or child_elem is None:
                continue
            parent_link = parent_elem.get('link', '')
            child_link = child_elem.get('link', '')
            # 排除 'world' (它是虚拟根节点，不是真正的机器人 link)
            if child_link not in self._child_to_joint and parent_link and parent_link != 'world':
                self._root_link = parent_link
                break
        if self._root_link is None:
            # 回退: 找第一个 parent 不是 world 的关节的 parent
            for joint_elem in root.findall('./joint'):
                parent_elem = joint_elem.find('parent')
                if parent_elem is not None:
                    candidate = parent_elem.get('link', '')
                    if candidate and candidate != 'world':
                        self._root_link = candidate
                        break
            # 再次回退: 直接用 child_to_joint 中的第一个 link (其 parent link 就是 root)
            if self._root_link is None:
                for joint_elem in root.findall('./joint'):
                    child_elem = joint_elem.find('child')
                    parent_elem = joint_elem.find('parent')
                    if child_elem is not None and parent_elem is not None:
                        parent_link = parent_elem.get('link', '')
                        child_link = child_elem.get('link', '')
                        if child_link and parent_link and child_link in self._child_to_joint and parent_link not in self._child_to_joint:
                            self._root_link = parent_link
                            break

        print(f"[RobotVisualizer] Root link: {self._root_link}")

        # ---- Run initial FK: neutral pose ----
        q0 = np.zeros(self._nq)
        pin.forwardKinematics(self._pinocchio_model, self._pinocchio_data, q0)
        self._current_q = q0.copy()

        # ---- Build joint_name → SE3(in world) from data.oMi ----
        joint_world_se3 = {}
        for jid, jname in enumerate(self._pinocchio_model.names):
            if jname == 'universe':
                continue
            try:
                joint_world_se3[jname] = self._pinocchio_data.oMi[jid]
            except Exception:
                pass

        # ---- Traverse kinematic chain from root ----
        visited = set()

        def accumulate(link_name: str, world_se3: pin.SE3):
            if link_name in visited:
                return
            visited.add(link_name)
            self._link_fk[link_name] = world_se3

            if link_name not in self._child_to_joint:
                return
            joint_name = self._child_to_joint[link_name]
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is None:
                print(f"[RobotVisualizer] No FK for joint '{joint_name}'")
                return
            accumulate(link_name, joint_se3)

        # 从根链接开始遍历
        root_se3 = pin.SE3(np.eye(3), np.zeros(3))
        accumulate(self._root_link, root_se3)

        # 遍历每个尚未访问的子链接
        for child_link, joint_name in self._child_to_joint.items():
            if child_link in visited:
                continue
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is None:
                continue
            accumulate(child_link, joint_se3)

        print(f"[RobotVisualizer] FK chain: {len(self._link_fk)} links mapped")

        # ---- Load each mesh and create actor ----
        for link_name, mesh_path, pose_info in mesh_infos:
            if mesh_path is None:
                continue

            mesh_path = os.path.normpath(mesh_path)
            if not os.path.exists(mesh_path):
                print(f"[RobotVisualizer] Mesh not found: {mesh_path}")
                continue

            try:
                original_mesh = pv.read(mesh_path)
                n_tris_before = original_mesh.n_faces_strict if hasattr(original_mesh, 'n_faces_strict') else original_mesh.n_cells

                # ========== 性能档: mesh 降采样 ==========
                if self._decimate > 0.0 and original_mesh.n_points > 500:
                    try:
                        decimated = original_mesh.decimate(self._decimate, inplace=False)
                        if decimated is not None and decimated.n_points > 100:
                            original_mesh = decimated
                    except Exception:
                        pass  # decimate 失败时保留原始 mesh

                self._original_meshes[link_name] = original_mesh

                # 构建局部变换(xyz + rpy + scale)
                local_T = self._make_transform(
                    pose_info['xyz'],
                    pose_info['rpy'],
                    pose_info.get('scale', [1.0, 1.0, 1.0])
                )

                # World-frame link transform
                world_se3 = self._link_fk.get(link_name)
                if world_se3 is not None:
                    total_T = world_se3.homogeneous @ local_T
                else:
                    total_T = local_T

                # 在原点处添加 actor,并通过 user_matrix 施加变换
                actor = self._plotter.add_mesh(
                    original_mesh,
                    name=f"robot_{link_name}",
                    color=pose_info['color'],
                    opacity=1.0,
                    show_edges=False
                )
                actor.user_matrix = self.base_transform @ total_T

                self._actors[link_name] = actor
                self._meshes[link_name] = original_mesh
                self._local_poses[link_name] = pose_info

                print(f"[RobotVisualizer] Loaded: {link_name} -> {os.path.basename(mesh_path)} (cells={n_tris_before}->{original_mesh.n_cells})")

            except Exception as e:
                print(f"[RobotVisualizer] Failed to load {link_name}: {e}")

        if not self._actors:
            print("[RobotVisualizer] No meshes loaded, creating simplified model")
            self._create_simple_robot()

    def _create_simple_robot(self):
        """创建简化的机器人模型（用于测试）"""
        print("[RobotVisualizer] Creating simplified robot model")

        # 简化的基座和连杆
        base = pv.Cylinder(center=(0, 0, 0.1), direction=(0, 0, 1), radius=0.2, height=0.2)
        self._actors['base_link'] = self._plotter.add_mesh(base, name='robot_base_link', color='#4472C4')
        self._original_meshes['base_link'] = base
        self._local_poses['base_link'] = {'xyz': [0, 0, 0.1], 'rpy': [0, 0, 0], 'scale': [1, 1, 1], 'color': '#4472C4'}

        for i, (height, radius, z_pos) in enumerate([(0.5, 0.1, 0.3), (0.4, 0.08, 0.7), (0.3, 0.06, 1.1)]):
            link = pv.Cylinder(center=(0, 0, z_pos), direction=(0, 1, 0), radius=radius, height=height)
            name = f'link_{i+1}'
            self._actors[name] = self._plotter.add_mesh(link, name=f'robot_{name}', color='#4472C4')
            self._original_meshes[name] = link
            self._local_poses[name] = {'xyz': [0, 0, z_pos], 'rpy': [0, 1.57, 0], 'scale': [1, 1, 1], 'color': '#4472C4'}

    def update_joints(self, q: np.ndarray):
        """
        更新关节角度

        参数:
            q: 关节角度数组
        """
        q = np.asarray(q).flatten()
        if len(q) < self._nq:
            q = np.pad(q, (0, self._nq - len(q)))

        self._current_q = q.copy()

        if self._pinocchio_model is None or not self._actors or not self._original_meshes:
            return

        import pinocchio as pin

        # 用新的关节值重新计算正运动学
        pin.forwardKinematics(self._pinocchio_model, self._pinocchio_data, q)
        pin.updateFramePlacements(self._pinocchio_model, self._pinocchio_data)  # 核心修复:更新固定坐标系(oMf),否则 tool0 无法获取正确位姿

        # 根据 data.oMi 重建 joint_name → SE3(世界坐标系下)
        joint_world_se3 = {}
        for jid, jname in enumerate(self._pinocchio_model.names):
            if jname == 'universe':
                continue
            try:
                joint_world_se3[jname] = self._pinocchio_data.oMi[jid]
            except Exception:
                pass

        # Build/retrieve child_to_joint and root_link from cached instance variables
        # 注意: 使用 ./joint 而非 .//joint，避免匹配到 <transmission> 内嵌套的同名 <joint> 元素
        if not self._child_to_joint:
            tree = ET.parse(self._urdf_path)
            root = tree.getroot()

            for joint_elem in root.findall('./joint'):
                child_elem = joint_elem.find('child')
                if child_elem is None:
                    continue
                child_link = child_elem.get('link', '')
                joint_name = joint_elem.get('name', '')
                if child_link and joint_name:
                    self._child_to_joint[child_link] = joint_name

            # 寻找根链接:其子链接出现在 child_to_joint 中,
            # 但该关节的父链接不是 'world'(world 是虚拟根节点)
            self._root_link = None
            for joint_elem in root.findall('./joint'):
                parent_elem = joint_elem.find('parent')
                child_elem = joint_elem.find('child')
                if parent_elem is None or child_elem is None:
                    continue
                parent_link = parent_elem.get('link', '')
                child_link = child_elem.get('link', '')
                if child_link not in self._child_to_joint and parent_link and parent_link != 'world':
                    self._root_link = parent_link
                    break
            if self._root_link is None:
                for joint_elem in root.findall('./joint'):
                    parent_elem = joint_elem.find('parent')
                    if parent_elem is not None:
                        candidate = parent_elem.get('link', '')
                        if candidate and candidate != 'world':
                            self._root_link = candidate
                            break
            if self._root_link is None:
                for joint_elem in root.findall('./joint'):
                    child_elem = joint_elem.find('child')
                    parent_elem = joint_elem.find('parent')
                    if child_elem is not None and parent_elem is not None:
                        parent_link = parent_elem.get('link', '')
                        child_link = child_elem.get('link', '')
                        if child_link and parent_link and child_link in self._child_to_joint and parent_link not in self._child_to_joint:
                            self._root_link = parent_link
                            break

        # Re-traverse kinematic chain from root
        self._link_fk.clear()
        visited = set()

        def accumulate(link_name: str, world_se3: pin.SE3):
            if link_name in visited:
                return
            visited.add(link_name)
            self._link_fk[link_name] = world_se3
            if link_name not in self._child_to_joint:
                return
            joint_name = self._child_to_joint[link_name]
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is None:
                return
            accumulate(link_name, joint_se3)

        root_se3 = pin.SE3(np.eye(3), np.zeros(3))
        accumulate(self._root_link, root_se3)

        for child_link, joint_name in self._child_to_joint.items():
            if child_link in visited:
                continue
            joint_se3 = joint_world_se3.get(joint_name)
            if joint_se3 is None:
                continue
            accumulate(child_link, joint_se3)

        # 更新每个 actor 的网格
        for link_name, actor in self._actors.items():
            try:
                original_mesh = self._original_meshes.get(link_name)
                if original_mesh is None:
                    continue

                pose_info = self._local_poses.get(link_name, {
                    'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'scale': [1, 1, 1]
                })
                local_T = self._make_transform(
                    pose_info['xyz'],
                    pose_info['rpy'],
                    pose_info.get('scale', [1.0, 1.0, 1.0])
                )

                world_se3 = self._link_fk.get(link_name)
                if world_se3 is not None:
                    total_T = world_se3.homogeneous @ local_T
                else:
                    total_T = local_T

                # 通过 user_matrix 更新 actor 变换 — 无需复制网格
                actor.user_matrix = self.base_transform @ total_T

            except Exception:
                pass  # silent

        self._plotter.render()

    def set_base_transform(self, T: np.ndarray):
        self.base_transform = T
        if self._current_q is not None:
            self.update_joints(self._current_q)

    def get_end_effector_pose(self) -> np.ndarray:
        """获取法兰盘位姿 (4x4 矩阵)"""
        if self._pinocchio_model is None or self._current_q is None:
            return np.eye(4)

        import pinocchio as pin
        try:
            # 优先使用 flange frame（正确包含 URDF joint_6-flange origin 偏移）
            # 不要用 last_joint_id (joint_6) 来获取法兰位置，因为 joint_6-flange
            # 的 origin 偏移记录在 flange frame 中，不在 joint_6 中。
            # joint_6-flange origin xyz="0 0 0.125" 会使 flange frame placement
            # 比 joint_6 的 placement 多 +0.125m。
            if self._pinocchio_model.existFrame("flange"):
                frame_id = self._pinocchio_model.getFrameId("flange")
                pose = self._pinocchio_data.oMf[frame_id]
                return pose.homogeneous
            elif self._pinocchio_model.existFrame("tool0"):
                frame_id = self._pinocchio_model.getFrameId("tool0")
                pose = self._pinocchio_data.oMf[frame_id]
                return pose.homogeneous
            else:
                last_joint_id = len(self._pinocchio_model.names) - 1
                pose = self._pinocchio_data.oMi[last_joint_id]
                return pose.homogeneous
        except:
            return np.eye(4)
