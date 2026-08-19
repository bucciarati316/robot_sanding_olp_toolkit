"""
render_engine/render_engine.py - 渲染引擎 Facade

本文件重构自 render_engine.py (P0 phase)：
- URDF 解析工具 → _render_utils.py
- RobotVisualizer → _robot_visualizer.py
- PointCloudProcessor → _pointcloud_processor.py
- RenderEngine facade → 本文件
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Callable, Dict, Any
import warnings
import os
import time

# PyVista
import pyvista as pv
from pyvistaqt import QtInteractor

# Trimesh
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

# Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

# SciPy KDTree
from scipy.spatial import cKDTree

# 坐标系变换
from core.pose_solver import CoordinateTransformer

# 共享工具（从子模块导入，不再重复定义）
from ._render_utils import parse_urdf_package_path, parse_urdf_meshes, build_rotation_to_align

# 子模块
from ._robot_visualizer import RobotVisualizer
from ._pointcloud_processor import PointCloudProcessor

# pvACTOR 类型别名
try:
    pvACTOR = pv.Actor
except AttributeError:
    pvACTOR = Any


class RenderEngine:
    """
    增强版渲染引擎
    整合 PyVista 渲染、机器人 URDF、Open3D 点云处理
    """

    def __init__(self, plotter: 'QtInteractor',
                 high_performance: bool = True,
                 robot_decimate: float = 0.5):
        """
        参数:
            plotter: PyVista QtInteractor
            high_performance: 启动时套用 high-performance rendering settings (默认 True)
            robot_decimate: URDF mesh 降采样比例, 0=不降采样, 0.5=砍掉 50% 三角面
        """
        self._plotter = plotter
        self._high_performance = high_performance
        self._robot_decimate = robot_decimate
        self._robot: Optional[RobotVisualizer] = None
        self._pc_processor = PointCloudProcessor()

        # ========== 渲染性能优化 ==========
        # 节流渲染：避免在上一帧渲染完成前再次触发渲染
        self._render_pending = False
        self._last_render_time = 0.0  # 上次渲染时间戳（秒）
        self._min_render_interval_ms = 0  # 最小渲染间隔（毫秒），0=不限制

        # 性能统计
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_update = 0.0

        # 点云数据
        self._point_cloud: Optional[pv.PolyData] = None
        self._original_points: Optional[np.ndarray] = None
        self._workpiece_cloud: Optional[pv.PolyData] = None
        self._workpiece_cloud_actor: Optional[Any] = None  # actor 引用，用于动态更新颜色
        self._rgba_colors: Optional[np.ndarray] = None  # RGBA uint8 颜色数组，与 stage_2_pyside6_app.py 一致
        self._toolpath_points: Optional[pv.PolyData] = None
        self._toolpath_actor_names: List[str] = []
        self._toolpath_display_points = np.empty((0, 3), dtype=float)
        self._toolpath_layer_ids = np.empty(0, dtype=int)
        self._toolpath_segment_ids = np.empty(0, dtype=int)
        self._toolpath_segment_types = np.empty(0, dtype=object)
        self._execution_path_actor_names: List[str] = []
        self._flange_trail_points: List[np.ndarray] = []
        self._flange_trail_actor = None
        self._cad_model: Optional[pvACTOR] = None
        self._cad_mesh: Optional[pv.PolyData] = None  # 缓存已单位转换的 CAD mesh（供 CollisionManager 复用）

        # 点云额外偏移（用于 Step 4 独立调整）
        self._pointcloud_extra_offset: Optional[np.ndarray] = None

        # 刀具
        self._tool_actor = None
        self._tool_mesh: Optional[pv.PolyData] = None  # 缓存已单位转换+几何对齐后的刀具 mesh（供 CollisionManager 复用）
        self._tool_radius = 0.005
        self._end_effector_axis = np.array([0.0, 0.0, 1.0])  # 末端执行器轴方向（从法兰旋转矩阵提取）

        # 刀具参数（由 SimulationState 通过 set_flange_tool_params 传入，或由 tool_library 共享引用）
        self._flange_tool_params = None  # FlangeToolParams
        self._tool_library = None  # ToolLibrary 共享引用（优先级高于 _flange_tool_params）

        # 刀具包围盒
        self._bbox_visible = False
        self._bbox_actors: List[Any] = []  # 包围盒 actor 列表
        self._bbox_mesh_local: Optional[Any] = None  # 圆柱 mesh（局部坐标系，未变换）
        self._trajectory_axes_visible = False  # 轨迹坐标系常态化显示标志
        self._trajectory_axes_actors: List[Any] = []  # 轨迹坐标系持久 actor 列表
        self._kdtree: Optional[cKDTree] = None
        self._active_indices: Optional[np.ndarray] = None
        self._status: Optional[np.ndarray] = None

        # Task 2: Box 裁剪状态
        self._current_box_bounds: Optional[tuple] = None

        # 坐标系可视化
        self._flange_axes_actor = None
        self._flange_x_actor = None
        self._flange_y_actor = None
        self._flange_z_actor = None
        self._tool_axes_actor = None
        self._show_coordinate_frames = True

        # 点云坐标系可视化（锚定在点云坐标系原点）
        self._pc_axes_x_actor = None
        self._pc_axes_y_actor = None
        self._pc_axes_z_actor = None

        # CAD 坐标系可视化（锚定在 CAD 模型坐标系原点，随 CAD actor 的 user_matrix 变换）
        self._cad_axes_x_actor = None
        self._cad_axes_y_actor = None
        self._cad_axes_z_actor = None

        # C0 刀具底面中心坐标系可视化（三色坐标轴）
        self._c0_axes_x_actor = None
        self._c0_axes_y_actor = None
        self._c0_axes_z_actor = None
        self._c0_axes_visible = False   # 记录复选框状态

        # 切削体积可视化（底面圆盘 + 顶面圆盘 + 侧面轮廓）
        self._cutting_volume_visible = False
        self._cutting_vol_bottom_actor = None
        self._cutting_vol_top_actor = None
        self._cutting_vol_body_actor = None

        # flange / tool0 / tool 参数
        self._flange_tool_params = None

        # 刀具 STL 几何参数（用于精确包围盒和切削体积）
        self._tool_tip_radius = 0.005   # 刀具底面（切削端）半径（米）
        self._tool_tip_height = 0.1     # 刀具从底面到顶面的高度（米）

        # 坐标系变换器（单一真实数据源）
        self._transformer = CoordinateTransformer()

        # 网格地面 actor
        self._grid_major_actor = None
        self._grid_minor_actor = None

        # 环境物体 actor 字典：name -> {'actor': pvACTOR, 'mesh': pv.PolyData}
        self._env_object_actors: Dict[str, dict] = {}

    def set_transforms(self, T_wcf_rcf: np.ndarray, T_wcf_ocf: np.ndarray):
        """同步坐标系变换到渲染引擎"""
        self._transformer.set_transforms(T_wcf_rcf, T_wcf_ocf)

    # ==================== flange/tool 参数 ====================

    def _build_se3(self, xyz, rpy) -> np.ndarray:
        """将 xyz (m) + rpy (rad) 转换为 4x4 SE3 矩阵"""
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', rpy).as_matrix()
        t = np.array(xyz).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:] = t
        return T

    def _get_tool_length(self) -> float:
        """
        获取刀具总长度。

        优先级：1. STL 测量高度 _tool_tip_height；
                2. FlangeToolParams 三段 Z 偏移之和；
                3. fallback 0.1m。
        """
        # 优先使用 STL 测量值（用户已加载刀具 STL）
        if self._tool_tip_height > 0:
            return self._tool_tip_height
        # fallback：使用 FlangeToolParams
        if self._flange_tool_params is not None:
            length = (
                abs(self._flange_tool_params.flange_xyz[2]) +
                abs(self._flange_tool_params.tool_xyz[2])
            )
            if length > 0:
                return length
        return 0.1

    def _compute_tcp_transform(self) -> np.ndarray:
        """根据 _flange_tool_params 计算 T_flange_tcp"""
        # 优先使用 ToolLibrary（SSOT）
        if self._tool_library is not None:
            return self._tool_library.T_flange_tcp
        if self._flange_tool_params is None:
            return np.eye(4)
        params = self._flange_tool_params
        T = (self._build_se3(params.flange_xyz, params.flange_rpy) @
             self._build_se3(params.tool_xyz,   params.tool_rpy))
        return T

    def set_flange_tool_params(self, params):
        """接收并存储来自 SimulationState 的 18 参数"""
        self._flange_tool_params = params
        print(f"[RenderEngine] flange/tool 参数已同步")

    def set_tool_library(self, tool_library):
        """
        设置共享的 ToolLibrary 引用。

        设置后，刀具可视化会通过 tool_library 实时读取当前刀具偏置，
        保证 Step 5 修改刀具后立即在 3D 预览中反映。
        """
        self._tool_library = tool_library
        print(f"[RenderEngine] ToolLibrary 已绑定: 当前工具={tool_library.get_current_tool().name}")

    def set_bounding_box_visible(self, visible: bool):
        """
        显示/隐藏刀具圆柱包围盒（含 C0 坐标轴和圆柱体）。

        包围盒为沿刀具轴向的圆柱体：刀尖为起点，沿轴向延伸 tool_length，
        半径为 tool_radius。黄色边线透明度 1.0，曲面/平面透明度 0.8。
        """
        self._bbox_visible = visible

        if visible:
            self._update_bounding_box()
        else:
            self._remove_bounding_box()
            self.clear_c0_axes()

    def _update_bounding_box(self):
        """创建/渲染刀具圆柱包围盒（mesh 保存在局部坐标系）"""
        self._remove_bounding_box()

        if self._flange_tool_params is None:
            return

        radius = self._tool_radius
        if radius <= 0:
            radius = 0.01

        tool_length = self._get_tool_length()

        # 圆柱在局部坐标系：底面在原点，顶面在 +Z 方向，中心对齐 +Z 轴
        if not TRIMESH_AVAILABLE:
            return
        cylinder_trimesh = trimesh.creation.cylinder(
            radius=radius, height=tool_length, sections=32
        )
        # 保存局部坐标系 mesh（不变换，后续每帧用 user_matrix 定位）
        self._bbox_mesh_local = pv.wrap(cylinder_trimesh)

        bbox_color = 'yellow'
        self._bbox_actors.append(self._plotter.add_mesh(
            self._bbox_mesh_local, color=bbox_color, opacity=0.8,
            show_edges=True, edge_color=bbox_color, line_width=2.0,
            name="bbox_body"
        ))

        # 立即同步到当前法兰盘位姿
        self.update_bounding_box()
        self._plotter.render()

    def update_bounding_box(self):
        """
        每帧调用：根据当前法兰盘位姿实时更新包围盒变换。
        圆柱底面贴在法兰原点，轴向沿刀具轴向。
        """
        if not self._bbox_visible or len(self._bbox_actors) == 0:
            return
        if self._flange_tool_params is None:
            return
        if not self._robot:
            return

        params = self._flange_tool_params
        T_flange_tcp = (
            self._build_se3(params.flange_xyz, params.flange_rpy) @
            self._build_se3(params.tool_xyz,   params.tool_rpy)
        )
        tool_axis_local = T_flange_tcp[:3, 2]

        base_T = self._robot.base_transform
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            last_id = len(model.names) - 1
            flange_se3 = data.oMi[last_id]

        flange_T = base_T @ flange_se3.homogeneous
        flange_world = flange_T[:3, 3]
        axis_world = flange_T[:3, :3] @ T_flange_tcp[:3, :3] @ tool_axis_local
        axis_world = axis_world / np.linalg.norm(axis_world)

        radius = self._tool_radius
        if radius <= 0:
            radius = 0.01

        tool_length = self._get_tool_length()

        # 旋转 +Z → axis_world，底面移到 flange_world
        R = build_rotation_to_align(np.array([0.0, 0.0, 1.0]), axis_world)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = flange_world + axis_world * (tool_length / 2.0)

        for actor in self._bbox_actors:
            actor.user_matrix = T

        # 同步更新 C0 坐标轴（显示刀具底面中心的世界坐标系）
        if self._c0_axes_x_actor is not None:
            self.update_c0_axes(flange_world, axis_world)

        # 同步更新切削体积（底面圆盘 + 顶面圆盘 + 侧面轮廓线）
        if self._cutting_volume_visible and self._cutting_vol_bottom_actor is not None:
            self._update_cutting_volume_from_pose(flange_world, axis_world, R,
                                                    radius, tool_length)

    def update_tool_visualization(self):
        """
        每帧调用（独立于包围盒可见性）：更新 C0 坐标轴和切削体积。

        仅在对应 actor 已创建时才更新，实现复选框独立控制。
        若 _flange_tool_params 未设置，则使用法兰盘默认 +Z 轴方向。
        """
        if not self._robot:
            return

        base_T = self._robot.base_transform
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            last_id = len(model.names) - 1
            flange_se3 = data.oMi[last_id]

        flange_T = base_T @ flange_se3.homogeneous
        flange_world = flange_T[:3, 3]

        # 轴方向：优先用 _flange_tool_params，否则用法兰盘默认 +Z
        if self._flange_tool_params is not None:
            params = self._flange_tool_params
            T_flange_tcp = (
                self._build_se3(params.flange_xyz, params.flange_rpy) @
                self._build_se3(params.tool_xyz,   params.tool_rpy)
            )
            tool_axis_local = T_flange_tcp[:3, 2]
            axis_world = flange_T[:3, :3] @ T_flange_tcp[:3, :3] @ tool_axis_local
        else:
            axis_world = flange_T[:3, 2]   # 法兰盘默认 +Z

        axis_world = axis_world / np.linalg.norm(axis_world)

        radius = self._tool_radius
        if radius <= 0:
            radius = 0.01

        tool_length = self._get_tool_length()

        R = build_rotation_to_align(np.array([0.0, 0.0, 1.0]), axis_world)

        # 更新包围盒位置（仅在 bbox 已创建时生效）
        if len(self._bbox_actors) > 0:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = flange_world + axis_world * (tool_length / 2.0)
            for actor in self._bbox_actors:
                actor.user_matrix = T

        # 更新 C0 坐标轴
        if self._c0_axes_x_actor is not None:
            self.update_c0_axes(flange_world, axis_world)

        # 更新切削体积
        if self._cutting_volume_visible and self._cutting_vol_bottom_actor is not None:
            self._update_cutting_volume_from_pose(flange_world, axis_world, R,
                                                    radius, tool_length)

    def _update_cutting_volume_from_pose(self, c0_pos: np.ndarray, axis_world: np.ndarray,
                                         R: np.ndarray, radius: float, tool_length: float):
        """根据已知位姿参数更新切削体积几何体（供每帧调用）"""
        if self._cutting_vol_bottom_actor is None:
            return

        # 底面圆盘
        bottom_transform = np.eye(4)
        bottom_transform[:3, :3] = R
        bottom_transform[:3, 3] = c0_pos
        bottom_disk = pv.Disk(nrad=32, rmin=0.0, rmax=radius)
        bottom_disk.transform(bottom_transform, inplace=True)

        # 顶面圆盘
        top_center = c0_pos + axis_world * tool_length
        top_transform = np.eye(4)
        top_transform[:3, :3] = R
        top_transform[:3, 3] = top_center
        top_disk = pv.Disk(nrad=32, rmin=0.0, rmax=radius)
        top_disk.transform(top_transform, inplace=True)

        # 侧面轮廓线（四根母线 + 底面/顶面边缘圆环）
        n_circle = 32
        line_pts, line_segs = [], []
        seg_idx = 0
        for i in range(4):
            angle = i * (np.pi / 4)
            dir_xy = R @ np.array([np.cos(angle), np.sin(angle), 0.0])
            p_bottom = c0_pos + dir_xy * radius
            p_top = top_center + dir_xy * radius
            line_pts.extend([p_bottom, p_top])
            line_segs.append([2, seg_idx, seg_idx + 1])
            seg_idx += 2

        circle_pts, circle_segs = [], []
        ci = 0
        for i in range(n_circle):
            a0, a1 = i * (2 * np.pi / n_circle), (i + 1) * (2 * np.pi / n_circle)
            for c0 in [c0_pos, top_center]:
                dir0 = R @ np.array([np.cos(a0), np.sin(a0), 0.0])
                dir1 = R @ np.array([np.cos(a1), np.sin(a1), 0.0])
                circle_pts.extend([c0 + dir0 * radius, c0 + dir1 * radius])
                circle_segs.append([2, ci, ci + 1])
                ci += 2

        all_pts = np.array(line_pts + circle_pts)
        all_segs = np.array(line_segs + circle_segs, dtype=np.int64)
        lines_poly = pv.PolyData(all_pts)
        lines_poly.lines = all_segs

        self._cutting_vol_bottom_actor.mapper.SetInputData(bottom_disk)
        self._cutting_vol_top_actor.mapper.SetInputData(top_disk)
        self._cutting_vol_body_actor.mapper.SetInputData(lines_poly)
        self._cutting_vol_bottom_actor.mapper.Update()
        self._cutting_vol_top_actor.mapper.Update()
        self._cutting_vol_body_actor.mapper.Update()

    def _remove_bounding_box(self):
        """删除所有包围盒 actor 和 C0 坐标轴"""
        for actor in self._bbox_actors:
            self._plotter.remove_actor(actor, reset_camera=False)
        self._bbox_actors.clear()
        self._bbox_mesh_local = None
        self.clear_c0_axes()
        self._plotter.render()

    # ==================== 切削体积可视化 ====================

    def set_cutting_volume_visible(self, visible: bool):
        """
        显示/隐藏切削体积（底面圆盘 + 顶面圆盘 + 侧面轮廓），黄色。

        切削体积由当前 C0、轴向、刀具半径和长度决定。
        """
        self._cutting_volume_visible = visible

        if visible:
            self._update_cutting_volume()
        else:
            self._remove_cutting_volume()

    def _update_cutting_volume(self):
        """根据当前刀具参数创建/更新切削体积几何体"""
        self._remove_cutting_volume()

        if self._flange_tool_params is None or not self._robot:
            return

        params = self._flange_tool_params
        radius = self._tool_radius if self._tool_radius > 0 else 0.01

        tool_length = self._get_tool_length()

        # 计算法兰盘当前世界位姿
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data
        base_T = self._robot.base_transform

        T_flange_tcp = (
            self._build_se3(params.flange_xyz, params.flange_rpy) @
            self._build_se3(params.tool_xyz, params.tool_rpy)
        )
        tool_axis_local = T_flange_tcp[:3, 2]

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            last_id = len(model.names) - 1
            flange_se3 = data.oMi[last_id]

        flange_T = base_T @ flange_se3.homogeneous
        c0_pos = flange_T[:3, 3]
        axis_world = flange_T[:3, :3] @ T_flange_tcp[:3, :3] @ tool_axis_local
        axis_world = axis_world / np.linalg.norm(axis_world)

        # 旋转轴：从局部 +Z 旋转到 axis_world
        R = build_rotation_to_align(np.array([0.0, 0.0, 1.0]), axis_world)

        # 底面圆盘
        bottom_center = c0_pos
        bottom_transform = np.eye(4)
        bottom_transform[:3, :3] = R
        bottom_transform[:3, 3] = bottom_center
        bottom_disk = pv.Disk(nrad=32, rmin=0.0, rmax=radius)
        bottom_disk.transform(bottom_transform, inplace=True)

        # 顶面圆盘
        top_center = c0_pos + axis_world * tool_length
        top_transform = np.eye(4)
        top_transform[:3, :3] = R
        top_transform[:3, 3] = top_center
        top_disk = pv.Disk(nrad=32, rmin=0.0, rmax=radius)
        top_disk.transform(top_transform, inplace=True)

        # 侧面：四根轮廓线（圆柱的母线，围成切面轮廓）
        angle_step = np.pi / 4
        line_pts = []
        line_segs = []
        seg_idx = 0
        for i in range(4):
            angle = i * angle_step
            dir_xy = np.array([np.cos(angle), np.sin(angle), 0.0])
            dir_world = R @ dir_xy
            p_bottom = c0_pos + dir_world * radius
            p_top = top_center + dir_world * radius
            line_pts.extend([p_bottom, p_top])
            line_segs.append([2, seg_idx, seg_idx + 1])
            seg_idx += 2

        # 底面和顶面边缘圆环线（用多个圆弧段近似）
        n_circle = 32
        circle_pts = []
        circle_segs = []
        ci = 0
        for i in range(n_circle):
            angle0 = i * (2 * np.pi / n_circle)
            angle1 = (i + 1) * (2 * np.pi / n_circle)
            dir0 = R @ np.array([np.cos(angle0), np.sin(angle0), 0.0])
            dir1 = R @ np.array([np.cos(angle1), np.sin(angle1), 0.0])
            circle_pts.extend([c0_pos + dir0 * radius, c0_pos + dir1 * radius])
            circle_segs.append([2, ci, ci + 1])
            ci += 2

        ci2 = len(circle_pts)
        for i in range(n_circle):
            angle0 = i * (2 * np.pi / n_circle)
            angle1 = (i + 1) * (2 * np.pi / n_circle)
            dir0 = R @ np.array([np.cos(angle0), np.sin(angle0), 0.0])
            dir1 = R @ np.array([np.cos(angle1), np.sin(angle1), 0.0])
            circle_pts.extend([top_center + dir0 * radius, top_center + dir1 * radius])
            circle_segs.append([2, ci2, ci2 + 1])
            ci2 += 2

        all_pts = np.array(line_pts + circle_pts)
        all_segs = np.array(line_segs + circle_segs, dtype=np.int64)
        lines_poly = pv.PolyData(all_pts)
        lines_poly.lines = all_segs

        self._cutting_vol_bottom_actor = self._plotter.add_mesh(
            bottom_disk, color='yellow', opacity=0.5, name="cutting_vol_bottom"
        )
        self._cutting_vol_top_actor = self._plotter.add_mesh(
            top_disk, color='yellow', opacity=0.5, name="cutting_vol_top"
        )
        self._cutting_vol_body_actor = self._plotter.add_mesh(
            lines_poly, color='yellow', line_width=1.5, opacity=0.8, name="cutting_vol_body"
        )
        self._plotter.render()

    def _remove_cutting_volume(self):
        """删除切削体积所有 actor"""
        for actor, name in [
            (self._cutting_vol_bottom_actor, "cutting_vol_bottom"),
            (self._cutting_vol_top_actor, "cutting_vol_top"),
            (self._cutting_vol_body_actor, "cutting_vol_body"),
        ]:
            if actor is not None:
                self._plotter.remove_actor(name, reset_camera=False)
        self._cutting_vol_bottom_actor = None
        self._cutting_vol_top_actor = None
        self._cutting_vol_body_actor = None

    # ==================== 网格地面 ====================

    def setup_scene(self,
                     background: str = "#2b2b2b",
                     high_performance: bool = True):
        """
        初始化 3D 场景

        参数:
            background: 背景色
            high_performance: True 表示追求最高帧率（默认）
                - 抗锯齿: 'none'（最高 FPS）
                - 光照: 简化环境光，光源数清零
                - 网格地面: minor_grid 自动隐藏以减少 draw call
            False 表示画质优先（保留 FXAA + 双层网格）
        """
        self._high_performance = high_performance

        self._plotter.set_background(background)
        self._plotter.enable_terrain_style(mouse_wheel_zooms=True)
        self._plotter.enable_trackball_style()

        # ========== 性能优化配置 ==========
        # 关闭光照计算以提升性能（机器人可视化不需要复杂光照）
        try:
            # 禁用所有光源，提升实时渲染性能
            self._plotter.renderer.light_actors = []
            # 使用简化的环境光
            self._plotter.renderer.ambient = 0.3
        except Exception:
            pass

        # 禁用平滑 shading，直接使用 flat shading（更快）
        try:
            self._plotter.enable_eye_dome_lighting(False)
        except Exception:
            pass

        # 抗锯齿：高 FPS 档直接关；画质档再开 FXAA
        try:
            if high_performance:
                self._plotter.enable_anti_aliasing('none')
            else:
                self._plotter.enable_anti_aliasing('fxaa')
        except Exception:
            pass

        # 关闭 depth peeling（当前未启用，保持代码一致性）
        # 注意：不启用 depth peeling — 它会导致半透明网格线被错误剔除

        # 高性能档：地面只画主网格，省掉 minor grid 的 draw call
        if high_performance:
            self._add_grid_floor(minor_step=0.0)
        else:
            self._add_grid_floor()

    def set_high_performance(self, enabled: bool):
        """
        运行时切换性能档位（不需要重新初始化 plotter）
        """
        self._high_performance = enabled
        try:
            self._plotter.enable_anti_aliasing('none' if enabled else 'fxaa')
        except Exception:
            pass
        # 切回性能档时，隐藏次网格
        if enabled and self._grid_minor_actor is not None:
            try:
                self._plotter.remove_actor(self._grid_minor_actor, reset_camera=False)
                self._grid_minor_actor = None
            except Exception:
                pass
            try:
                self._plotter.render()
            except Exception:
                pass

    def _add_grid_floor(self, size: float = 5.0, major_step: float = 1.0,
                        minor_step: float = 0.2,
                        major_color: str = '#606060',
                        minor_color: str = '#404040',
                        major_opacity: float = 1.0,
                        minor_opacity: float = 1.0):
        """
        添加 Mujoco/PyBullet 风格的双层网格地面。

        参数:
            size: 地面半尺寸（米），实际地面为 2*size x 2*size
            major_step: 主网格线间距（米）
            minor_step: 次网格线间距（米）
            major_color: 主网格线颜色
            minor_color: 次网格线颜色
            major_opacity: 主网格线透明度
            minor_opacity: 次网格线透明度
        """
        self._grid_major_actor = None
        self._grid_minor_actor = None

        def _build_grid_lines(half: float, step: float):
            pts, lns = [], []
            idx = 0
            n = int(2 * half / step) + 1
            for i in range(n):
                t = -half + i * step
                # 横向线
                pts.extend([[-half, t, 0], [half, t, 0]])
                lns.append([2, idx, idx + 1])
                idx += 2
                # 纵向线
                pts.extend([[t, -half, 0], [t, half, 0]])
                lns.append([2, idx, idx + 1])
                idx += 2
            pts_arr = np.array(pts)
            lns_arr = np.hstack(lns) if lns else np.array([], dtype=np.int64)
            return pv.PolyData(pts_arr, lines=lns_arr)

        # 次网格（先渲染，层级靠下）
        if minor_step > 0:
            grid_minor = _build_grid_lines(size, minor_step)
            self._grid_minor_actor = self._plotter.add_mesh(
                grid_minor,
                name="grid_minor",
                color=minor_color,
                line_width=0.5,
                opacity=minor_opacity,
                pickable=False
            )

        # 主网格（后渲染，层级靠上）
        grid_major = _build_grid_lines(size, major_step)
        self._grid_major_actor = self._plotter.add_mesh(
            grid_major,
            name="grid_major",
            color=major_color,
            line_width=1.5,
            opacity=major_opacity,
            pickable=False
        )

    def toggle_grid(self, visible: bool = None):
        """
        切换或设置网格地面可见性。

        参数:
            visible: True 显示，False 隐藏，None 切换当前状态

        返回:
            bool: 当前可见性状态
        """
        if visible is None:
            visible = not self._is_grid_visible()

        if self._grid_major_actor is not None:
            self._grid_major_actor.SetVisibility(visible)
        if self._grid_minor_actor is not None:
            self._grid_minor_actor.SetVisibility(visible)
        self._plotter.render()
        return visible

    def _is_grid_visible(self) -> bool:
        """返回网格当前可见性状态"""
        if self._grid_major_actor is not None:
            return bool(self._grid_major_actor.GetVisibility())
        return False

    def add_axes(self, interactive: bool = True):
        """添加坐标轴"""
        self._plotter.add_axes(interactive=interactive)

    def add_camera_control(self):
        """添加相机控制面板"""
        self._plotter.add_camera_position_widget()

    # ==================== 机器人渲染 ====================

    def load_robot(self, urdf_path: str, decimate: float = None) -> bool:
        """
        加载机器人 URDF

        参数:
            urdf_path: URDF 文件路径
            decimate: mesh 降采样比例（None 表示使用 self._robot_decimate 默认值；0 表示不降采样）

        返回:
            bool: 加载是否成功
        """
        try:
            # 清除旧机器人
            if self._robot:
                self.clear_robot()

            # 优先使用实参，否则使用引擎全局默认值
            d = self._robot_decimate if decimate is None else decimate
            self._robot = RobotVisualizer(urdf_path, self._plotter, decimate=d)

            # 应用快速渲染优化到所有机器人 actor
            self._apply_fast_mode_to_actors()

            # 重置相机并渲染（使用 force_render 确保立即生效）
            self._plotter.reset_camera()
            self.force_render()

            print(f"[RenderEngine] 机器人加载完成: {urdf_path}")
            return True
        except Exception as e:
            print(f"[RenderEngine] 加载机器人失败: {e}")
            return False

    def _apply_fast_mode_to_actors(self):
        """为所有 actor 应用快速渲染模式"""
        if not hasattr(self._plotter, 'renderer'):
            return
        try:
            renderer = self._plotter.renderer
            for actor in renderer.actors.values():
                # 禁用 interactor 的拾取以提升性能
                if hasattr(actor, 'PickableOff'):
                    actor.PickableOff()
                # 使用低精度渲染（如果支持）
                if hasattr(actor, 'GetMapper'):
                    mapper = actor.GetMapper()
                    if hasattr(mapper, 'SetResolveCoincidentTopologyToOff'):
                        mapper.SetResolveCoincidentTopologyToOff()
        except Exception:
            pass

    def clear_robot(self):
        """清除机器人模型"""
        if self._robot:
            for name, actor in self._robot._actors.items():
                self._plotter.remove_actor(f"robot_{name}")
            self._robot = None

    def update_robot_joints(self, q: np.ndarray, render: bool = True):
        """更新机器人关节角度"""
        if self._robot:
            self._robot.update_joints(q)
            # 让刀具跟随末端法兰
            if self._tool_actor and len(self._robot._pinocchio_model.names) > 0:
                model = self._robot._pinocchio_model
                data = self._robot._pinocchio_data

                # 优先获取法兰盘坐标系 (tool0)，确保方向包含了固定端面偏移
                if model.existFrame("tool0"):
                    frame_id = model.getFrameId("tool0")
                    flange_se3 = data.oMf[frame_id]
                elif model.existFrame("flange"):
                    frame_id = model.getFrameId("flange")
                    flange_se3 = data.oMf[frame_id]
                else:
                    last_joint_id = len(model.names) - 1
                    flange_se3 = data.oMi[last_joint_id]

                # 更新末端执行器轴方向（旋转矩阵的第三列对应 Z 轴）
                self._end_effector_axis = flange_se3.rotation[:, 2].copy()

                # 使用参数化方式计算 TCP 变换
                T_flange_tcp = self._compute_tcp_transform()

                # 乘以机器人的基座 WCF 变换
                flange_T = self._robot.base_transform @ flange_se3.homogeneous
                # 刀具 STL 始终渲染在 tool0 关节处（不包含 TCP 偏移）
                self.update_tool_transform(flange_T)
                # 法兰坐标系可视化（X=红, Y=绿, Z=蓝），常态跟随法兰盘
                self.update_coordinate_frames(flange_T)
                if render:
                    self.render()

    def _sync_tool_to_flange(self):
        """
        将刀具绑定到当前法兰盘（tool0）位姿。
        提取 tool0 帧的方向向量重置 _end_effector_axis，
        再通过 user_matrix 同步刀具变换。
        """
        if not self._tool_actor or not self._robot:
            return
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            # fallback: joint_6 的位置（不包含 flange offset）
            last_joint_id = len(model.names) - 1
            flange_se3 = data.oMi[last_joint_id]

        self._end_effector_axis = flange_se3.rotation[:, 2].copy()
        flange_T = self._robot.base_transform @ flange_se3.homogeneous
        self.update_tool_transform(flange_T)

    def get_robot_ee_pose(self) -> np.ndarray:
        """获取机器人末端位姿"""
        if self._robot:
            return self._robot.get_end_effector_pose()
        return np.eye(4)

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        返回关节位置上下限。

        Returns
        -------
        lower : np.ndarray
            各关节下限。
        upper : np.ndarray
            各关节上限。
        """
        if self._robot:
            return self._robot.get_joint_limits()
        import numpy as np
        return -np.pi * np.ones(6), np.pi * np.ones(6)

    # ==================== CAD 模型 ====================

    def load_cad_model(self, filepath: str, opacity: float = 0.3) -> bool:
        """
        加载 CAD 模型 (STL/STEP)

        参数:
            filepath: 文件路径
            opacity: 透明度

        返回:
            bool: 加载是否成功
        """
        try:
            import trimesh

            # 加载网格
            scene = trimesh.load(filepath, force='mesh')

            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.Scene):
                # 合并所有网格
                meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(meshes)
            else:
                print(f"[RenderEngine] 不支持的 CAD 格式")
                return False

            pv_mesh = pv.wrap(mesh)

            # --- 尺寸单位修正 ---
            # 假设 CAD 软件导出默认使用毫米(mm)，这里统一缩小 1000 倍转化为 URDF 的标准单位米(m)
            pv_mesh.points *= 0.001

            # 添加半透明显示
            if self._cad_model:
                self._plotter.remove_actor(self._cad_model)

            self._cad_model = self._plotter.add_mesh(
                pv_mesh,
                name="cad_model",
                color='#90EE90',  # 浅绿色
                opacity=opacity,
                show_edges=True,
                edge_color='#228B22'
            )

            # 在 CAD 模型坐标系原点渲染三色坐标轴（X=红, Y=绿, Z=蓝）
            self._update_cad_axes(pv_mesh)

            # 重置相机并渲染
            self._plotter.reset_camera()
            self._plotter.render()

            print(f"[RenderEngine] CAD 模型加载完成: {filepath}")
            self._cad_mesh = pv_mesh  # 缓存已单位转换的 mesh（供 CollisionManager 复用）
            return True

        except Exception as e:
            print(f"[RenderEngine] 加载 CAD 失败: {e}")
            return False

    def remove_cad_model(self):
        """移除 CAD 模型"""
        if self._cad_model:
            self._plotter.remove_actor(self._cad_model)
            self._cad_model = None
        self._cad_mesh = None
        self._clear_cad_axes()

    def _update_cad_axes(self, pv_mesh: 'pv.PolyData'):
        """
        在 CAD 模型坐标系原点渲染/更新三色坐标轴。

        CAD 原点 = 文件原始坐标系 (0, 0, 0)，与 mesh bounds center 无关。
        坐标轴长度按 CAD bounds 对角线的 7.5% 自适应计算，最小 0.02m。
        随 CAD actor 的 user_matrix 一起变换，确保跟随 CAD 模型位姿移动。
        """
        if pv_mesh is None:
            return

        # 计算自适应轴长：包围盒对角线的 7.5%，最小 0.02m
        bounds = pv_mesh.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0]) ** 2 +
            (bounds[3] - bounds[2]) ** 2 +
            (bounds[5] - bounds[4]) ** 2
        )
        axis_length = max(diagonal * 0.075, 0.02)

        # CAD 坐标系原点（文件原始坐标 (0, 0, 0)）
        origin = np.array([0.0, 0.0, 0.0])

        # X 轴（红色）, Y 轴（绿色）, Z 轴（蓝色）
        x_end = origin + np.array([axis_length, 0.0, 0.0])
        y_end = origin + np.array([0.0, axis_length, 0.0])
        z_end = origin + np.array([0.0, 0.0, axis_length])

        x_line = pv.Line(origin, x_end)
        y_line = pv.Line(origin, y_end)
        z_line = pv.Line(origin, z_end)

        if self._cad_axes_x_actor is None:
            # 首次创建
            self._cad_axes_x_actor = self._plotter.add_mesh(
                x_line, name="cad_axes_x", color='red',
                line_width=3, opacity=1.0, pickable=False
            )
            self._cad_axes_y_actor = self._plotter.add_mesh(
                y_line, name="cad_axes_y", color='green',
                line_width=3, opacity=1.0, pickable=False
            )
            self._cad_axes_z_actor = self._plotter.add_mesh(
                z_line, name="cad_axes_z", color='blue',
                line_width=3, opacity=1.0, pickable=False
            )
        else:
            # 复用已有 actor：更新几何数据
            self._cad_axes_x_actor.mapper.SetInputData(x_line)
            self._cad_axes_y_actor.mapper.SetInputData(y_line)
            self._cad_axes_z_actor.mapper.SetInputData(z_line)
            self._cad_axes_x_actor.mapper.Update()
            self._cad_axes_y_actor.mapper.Update()
            self._cad_axes_z_actor.mapper.Update()

        # 坐标轴跟随 CAD actor 的 user_matrix（自动跟随 CAD 模型位姿变换）
        if self._cad_model is not None and hasattr(self._cad_model, 'user_matrix'):
            T = self._cad_model.user_matrix
            for actor in (self._cad_axes_x_actor, self._cad_axes_y_actor, self._cad_axes_z_actor):
                if actor is not None:
                    actor.user_matrix = T

    def _clear_cad_axes(self):
        """清除 CAD 坐标系可视化 actor"""
        for actor in (self._cad_axes_x_actor, self._cad_axes_y_actor, self._cad_axes_z_actor):
            if actor is not None:
                self._plotter.remove_actor(actor, reset_camera=False)
        self._cad_axes_x_actor = None
        self._cad_axes_y_actor = None
        self._cad_axes_z_actor = None

    # ==================== 环境物体 ====================

    def load_env_object(self, filepath: str, name: str, transform: np.ndarray = None) -> bool:
        """
        加载环境物体 STL 文件

        参数:
            filepath: STL 文件路径
            name: 物体名称（唯一标识）
            transform: 初始 4x4 变换矩阵，默认单位矩阵

        返回:
            bool: 加载是否成功
        """
        try:
            import trimesh

            if transform is None:
                transform = np.eye(4)

            # 如果已存在同名物体，先移除
            if name in self._env_object_actors:
                self.remove_env_object(name)

            # 加载网格
            scene = trimesh.load(filepath, force='mesh')

            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.Scene):
                meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(meshes)
            else:
                print(f"[RenderEngine] 不支持的环境物体格式")
                return False

            pv_mesh = pv.wrap(mesh)

            # STL 单位修正（假设毫米转米）
            pv_mesh.points *= 0.001

            # 添加到渲染器
            actor = self._plotter.add_mesh(
                pv_mesh,
                name=f"env_object_{name}",
                color='#87CEEB',  # 天蓝色
                opacity=0.9,
                show_edges=True,
                edge_color='#4682B4'
            )
            actor.user_matrix = transform

            # 存储 actor 和 mesh
            self._env_object_actors[name] = {
                'actor': actor,
                'mesh': pv_mesh,
                'filepath': filepath,
                'transform': transform.copy()
            }

            print(f"[RenderEngine] 环境物体加载完成: {name} <- {filepath}")
            self._plotter.render()
            return True

        except Exception as e:
            print(f"[RenderEngine] 加载环境物体失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_env_object_transform(self, name: str, transform: np.ndarray):
        """
        更新环境物体的变换矩阵

        参数:
            name: 物体名称
            transform: 4x4 变换矩阵
        """
        if name not in self._env_object_actors:
            return

        actor = self._env_object_actors[name]['actor']
        actor.user_matrix = transform
        self._env_object_actors[name]['transform'] = transform.copy()
        self._plotter.render()

    def get_env_object_transform(self, name: str) -> Optional[np.ndarray]:
        """获取环境物体的当前变换矩阵"""
        if name not in self._env_object_actors:
            return None
        return self._env_object_actors[name]['transform'].copy()

    def remove_env_object(self, name: str):
        """移除环境物体"""
        if name not in self._env_object_actors:
            return

        actor = self._env_object_actors[name]['actor']
        self._plotter.remove_actor(actor, reset_camera=False)
        del self._env_object_actors[name]
        self._plotter.render()
        print(f"[RenderEngine] 环境物体已移除: {name}")

    def clear_env_objects(self):
        """清除所有环境物体"""
        for name in list(self._env_object_actors.keys()):
            self.remove_env_object(name)

    def get_env_object_names(self) -> List[str]:
        """获取所有环境物体名称列表"""
        return list(self._env_object_actors.keys())

    def get_env_object_mesh(self, name: str):
        """
        获取环境物体的已转换 mesh（mm -> m 后），供 CollisionManager 复用。
        """
        if name not in self._env_object_actors:
            return None
        return self._env_object_actors[name].get('mesh')

    def get_cad_mesh(self):
        """获取 CAD 模型已转换 mesh，供 CollisionManager 复用"""
        return getattr(self, '_cad_mesh', None)

    def get_tool_mesh(self):
        """获取刀具已单位转换+几何对齐后的 mesh，供 CollisionManager 复用"""
        return getattr(self, '_tool_mesh', None)

    def _sync_cad_axes_to_model(self):
        """将 CAD 坐标轴的 user_matrix 同步到 CAD model actor"""
        if self._cad_axes_x_actor is None:
            return
        if self._cad_model is not None and hasattr(self._cad_model, 'user_matrix'):
            T = self._cad_model.user_matrix
            for actor in (self._cad_axes_x_actor, self._cad_axes_y_actor, self._cad_axes_z_actor):
                if actor is not None:
                    actor.user_matrix = T

    # ==================== C0 刀具底面坐标系 ====================

    def update_c0_axes(self, c0_world_pos: np.ndarray, axis_world: np.ndarray, axis_length: float = 0.06):
        """
        在 C0 刀具底面中心处渲染/更新三色坐标轴。

        参数:
            c0_world_pos: C0 底面中心的世界坐标 (3,)
            axis_world:    刀具轴向单位向量（世界坐标系）(3,)
            axis_length:  坐标轴长度（米）
        """
        try:
            R_local_to_world = build_rotation_to_align(
                np.array([0.0, 0.0, 1.0]), axis_world
            )
            local_x = R_local_to_world @ np.array([axis_length, 0.0, 0.0])
            local_y = R_local_to_world @ np.array([0.0, axis_length, 0.0])
            local_z = R_local_to_world @ np.array([0.0, 0.0, axis_length])
            origin = c0_world_pos

            x_line = pv.Line(origin, origin + local_x)
            y_line = pv.Line(origin, origin + local_y)
            z_line = pv.Line(origin, origin + local_z)

            if self._c0_axes_x_actor is None:
                self._c0_axes_x_actor = self._plotter.add_mesh(
                    x_line, name="c0_axes_x", color='red',
                    line_width=4, opacity=1.0, pickable=False
                )
                self._c0_axes_y_actor = self._plotter.add_mesh(
                    y_line, name="c0_axes_y", color='green',
                    line_width=4, opacity=1.0, pickable=False
                )
                self._c0_axes_z_actor = self._plotter.add_mesh(
                    z_line, name="c0_axes_z", color='blue',
                    line_width=4, opacity=1.0, pickable=False
                )
            else:
                self._c0_axes_x_actor.mapper.SetInputData(x_line)
                self._c0_axes_y_actor.mapper.SetInputData(y_line)
                self._c0_axes_z_actor.mapper.SetInputData(z_line)
                self._c0_axes_x_actor.mapper.Update()
                self._c0_axes_y_actor.mapper.Update()
                self._c0_axes_z_actor.mapper.Update()
        except Exception as e:
            print(f"[RenderEngine] C0 坐标轴更新失败: {e}")

    def clear_c0_axes(self):
        """清除 C0 坐标轴"""
        for actor in (self._c0_axes_x_actor, self._c0_axes_y_actor, self._c0_axes_z_actor):
            if actor is not None:
                self._plotter.remove_actor(actor, reset_camera=False)
        self._c0_axes_x_actor = None
        self._c0_axes_y_actor = None
        self._c0_axes_z_actor = None

    def set_c0_axes_visible(self, visible: bool):
        """
        设置 C0 坐标轴可见性。

        开启时：若 actor 未创建，则用当前法兰盘位姿创建坐标轴；
        关闭时：直接隐藏（保留 actor，避免重复创建开销）。
        """
        self._c0_axes_visible = visible
        if visible:
            if self._c0_axes_x_actor is None and self._robot is not None:
                self._ensure_c0_axes_created()
            elif self._c0_axes_x_actor is not None:
                for actor in (self._c0_axes_x_actor, self._c0_axes_y_actor, self._c0_axes_z_actor):
                    if actor is not None:
                        actor.SetVisibility(True)
        else:
            for actor in (self._c0_axes_x_actor, self._c0_axes_y_actor, self._c0_axes_z_actor):
                if actor is not None:
                    actor.SetVisibility(False)

    def _ensure_c0_axes_created(self):
        """
        用当前法兰盘位姿创建 C0 坐标轴。

        从 _flange_tool_params 获取刀具轴向，从 _robot 获取法兰盘世界坐标。
        若 _flange_tool_params 未配置则用默认 (+Z)。
        """
        if self._robot is None:
            return
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data
        base_T = self._robot.base_transform

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            last_id = len(model.names) - 1
            flange_se3 = data.oMi[last_id]

        flange_T = base_T @ flange_se3.homogeneous
        c0_pos = flange_T[:3, 3]

        if self._flange_tool_params is not None:
            params = self._flange_tool_params
            T_flange_tcp = (
                self._build_se3(params.flange_xyz, params.flange_rpy) @
                self._build_se3(params.tool_xyz, params.tool_rpy)
            )
            tool_axis_local = T_flange_tcp[:3, 2]
            axis_world = flange_T[:3, :3] @ T_flange_tcp[:3, :3] @ tool_axis_local
            axis_world = axis_world / np.linalg.norm(axis_world)
        else:
            axis_world = flange_T[:3, 2]

        self.update_c0_axes(c0_pos, axis_world)

    # ==================== 毛坯点云 ====================

    def load_workpiece_cloud(self, points: np.ndarray,
                             color: Tuple[int, int, int] = (0, 150, 255)) -> bool:
        """
        加载毛坯点云，并初始化点级透明度通道。

        参数:
            points: Nx3 点云
            color: RGB 颜色

        返回:
            bool: 加载是否成功
        """
        try:
            if points is None or len(points) == 0:
                print("[RenderEngine] 点云为空，跳过加载")
                return False

            self._original_points = points.copy()
            self._status = np.ones(len(points), dtype=np.int32)

            # Open3D 处理
            if self._pc_processor is not None:
                ok = self._pc_processor.load_source(points)
                if not ok:
                    print(f"[RenderEngine] Open3D 点云加载失败，尝试继续渲染")

            # PyVista 显示
            if self._plotter is None:
                print("[RenderEngine] Plotter 未初始化，跳过渲染")
                return False

            if self._workpiece_cloud:
                self._plotter.remove_actor("workpiece_cloud")

            self._workpiece_cloud = pv.PolyData(points)

            # 初始化 RGBA 颜色数组（全亮蓝色，与 stage_2_pyside6_app.py 一致）
            self._rgba_colors = np.full((len(points), 4), [0, 150, 255, 255], dtype=np.uint8)
            self._workpiece_cloud.point_data["colors"] = self._rgba_colors

            # 先移除旧的 actor
            if self._workpiece_cloud_actor is not None:
                self._plotter.remove_actor(self._workpiece_cloud_actor, reset_camera=False)

            self._workpiece_cloud_actor = self._plotter.add_mesh(
                self._workpiece_cloud,
                name="workpiece_cloud",
                scalars="colors",
                rgb=True,
                point_size=2.0,
                show_scalar_bar=False
            )

            # 构建 KDTree
            self._rebuild_kdtree()

            # 在点云坐标系原点渲染三色坐标轴（X=红, Y=绿, Z=蓝）
            self._update_pointcloud_axes()

            print(f"[RenderEngine] 点云加载成功: {len(points)} 点")
            return True

        except Exception as e:
            import traceback
            print(f"[RenderEngine] 加载点云失败: {e}")
            traceback.print_exc()
            return False

    def _refresh_point_colors(self):
        """刷新点云颜色 — 与 stage_2_pyside6_app.py 的 _refresh_point_colors 完全一致"""
        if self._workpiece_cloud is None or self._rgba_colors is None:
            return
        self._workpiece_cloud.point_data["colors"] = self._rgba_colors
        self._workpiece_cloud.Modified()
        self._plotter.render()

    def _update_pointcloud_axes(self):
        """
        在点云坐标系原点渲染/更新三色坐标轴。

        点云原点 = 文件原始坐标 (0, 0, 0)，与 mesh/point cloud bounds center 无关。
        坐标轴长度按点云包围盒对角线的 7.5% 自适应计算，最小 0.02m。
        随工件 actor 的 user_matrix 一起变换，确保跟随点云位姿移动。
        """
        if self._workpiece_cloud is None:
            return

        # 计算自适应轴长：包围盒对角线的 7.5%，最小 0.02m
        bounds = self._workpiece_cloud.bounds
        diagonal = np.sqrt(
            (bounds[1] - bounds[0]) ** 2 +
            (bounds[3] - bounds[2]) ** 2 +
            (bounds[5] - bounds[4]) ** 2
        )
        axis_length = max(diagonal * 0.075, 0.02)

        # 点云坐标系原点（文件原始坐标 (0, 0, 0)）
        origin = np.array([0.0, 0.0, 0.0])

        # X 轴（红色）, Y 轴（绿色）, Z 轴（蓝色）
        x_end = origin + np.array([axis_length, 0.0, 0.0])
        y_end = origin + np.array([0.0, axis_length, 0.0])
        z_end = origin + np.array([0.0, 0.0, axis_length])

        x_line = pv.Line(origin, x_end)
        y_line = pv.Line(origin, y_end)
        z_line = pv.Line(origin, z_end)

        if self._pc_axes_x_actor is None:
            # 首次创建
            self._pc_axes_x_actor = self._plotter.add_mesh(
                x_line, name="pc_axes_x", color='red',
                line_width=3, opacity=1.0, pickable=False
            )
            self._pc_axes_y_actor = self._plotter.add_mesh(
                y_line, name="pc_axes_y", color='green',
                line_width=3, opacity=1.0, pickable=False
            )
            self._pc_axes_z_actor = self._plotter.add_mesh(
                z_line, name="pc_axes_z", color='blue',
                line_width=3, opacity=1.0, pickable=False
            )
        else:
            # 复用已有 actor：更新几何数据
            self._pc_axes_x_actor.mapper.SetInputData(x_line)
            self._pc_axes_y_actor.mapper.SetInputData(y_line)
            self._pc_axes_z_actor.mapper.SetInputData(z_line)
            self._pc_axes_x_actor.mapper.Update()
            self._pc_axes_y_actor.mapper.Update()
            self._pc_axes_z_actor.mapper.Update()

        # 坐标轴跟随点云 actor 的 user_matrix（自动跟随工件位姿变换）
        workpiece_actor = self._plotter.actors.get("workpiece_cloud")
        if workpiece_actor is not None and hasattr(workpiece_actor, 'user_matrix'):
            T = workpiece_actor.user_matrix
            if self._pc_axes_x_actor is not None:
                self._pc_axes_x_actor.user_matrix = T
            if self._pc_axes_y_actor is not None:
                self._pc_axes_y_actor.user_matrix = T
            if self._pc_axes_z_actor is not None:
                self._pc_axes_z_actor.user_matrix = T

    def _clear_pointcloud_axes(self):
        """清除点云坐标系可视化 actor"""
        for actor in (self._pc_axes_x_actor, self._pc_axes_y_actor, self._pc_axes_z_actor):
            if actor is not None:
                self._plotter.remove_actor(actor, reset_camera=False)
        self._pc_axes_x_actor = None
        self._pc_axes_y_actor = None
        self._pc_axes_z_actor = None

    def _sync_pointcloud_axes_to_workpiece(self):
        """将点云坐标轴的 user_matrix 同步到 workpiece_cloud actor"""
        if self._pc_axes_x_actor is None:
            return
        workpiece_actor = self._plotter.actors.get("workpiece_cloud")
        if workpiece_actor is not None and hasattr(workpiece_actor, 'user_matrix'):
            T = workpiece_actor.user_matrix
            for actor in (self._pc_axes_x_actor, self._pc_axes_y_actor, self._pc_axes_z_actor):
                if actor is not None:
                    actor.user_matrix = T

    def _rebuild_kdtree(self):
        """重建 KDTree"""
        if self._original_points is None or self._status is None:
            return

        active_mask = self._status == 1
        active_points = self._original_points[active_mask]

        if len(active_points) > 0:
            self._kdtree = cKDTree(active_points)
            self._active_indices = np.where(active_mask)[0]
        else:
            self._kdtree = None
            self._active_indices = None

    def icp_refine_workpiece(self, target_points: np.ndarray,
                            max_distance: float = 0.05) -> bool:
        """
        ICP 配准毛坯点云

        参数:
            target_points: 目标点云
            max_distance: 最大对应距离

        返回:
            bool: 配准是否成功
        """
        self._pc_processor.load_target(target_points)
        success, T = self._pc_processor.icp_refine(max_correspondence_distance=max_distance)

        if success:
            transformed = self._pc_processor.get_transformed_source()
            if transformed is not None:
                self.load_workpiece_cloud(transformed)

        return success

    def filter_outliers(self, nb_neighbors: int = 20, std_ratio: float = 2.0) -> bool:
        """统计滤波去噪"""
        success = self._pc_processor.statistical_outlier_removal(nb_neighbors, std_ratio)
        if success:
            result = self._pc_processor.get_result_points()
            if result is not None:
                self.load_workpiece_cloud(result)
        return success

    def voxel_downsample_workpiece(self, voxel_size: float = 0.001) -> bool:
        """点云下采样"""
        success = self._pc_processor.voxel_downsample(voxel_size)
        if success:
            result = self._pc_processor.get_result_points()
            if result is not None:
                self.load_workpiece_cloud(result)
        return success

    def enable_crop_box(self):
        """
        Task 2 Step 1: 开启 Box 裁剪交互

        添加 BoxWidget，用户可在 3D 视图中拖动调整裁剪区域。
        回调仅保存 bounds 到 self._current_box_bounds，不执行实际裁剪。
        """
        bounds = self._workpiece_cloud.bounds if self._workpiece_cloud else [-1, 1, -1, 1, -1, 1]
        self._plotter.add_box_widget(
            callback=self._cache_box_bounds,
            bounds=bounds,
            color='white',
            opacity=0.3
        )

    def _cache_box_bounds(self, bounds: tuple):
        """
        Task 2: Box 裁剪回调 - 仅缓存边界

        参数:
            bounds: (x_min, x_max, y_min, y_max, z_min, z_max) 世界坐标系下的边界
        """
        self._current_box_bounds = bounds

    def confirm_crop_box(self, T_wcf_ocf: np.ndarray) -> bool:
        """
        Task 2 Step 2: 确认 Box 裁剪

        参数:
            T_wcf_ocf: 工件坐标系到世界坐标系的齐次变换矩阵

        返回:
            bool: 裁剪是否成功

        数学关键:
            - self._current_box_bounds 是世界坐标系
            - self._original_points 是局部坐标系 (工件坐标系)
            - 必须将 original_points 变换到世界坐标系后再裁剪
        """
        if self._current_box_bounds is None:
            print("[RenderEngine] 错误: 未设置裁剪框，请先拖动 Box")
            return False

        if self._original_points is None:
            print("[RenderEngine] 错误: 无点云数据")
            return False

        x_min, x_max, y_min, y_max, z_min, z_max = self._current_box_bounds

        # 将原始点从局部坐标系变换到世界坐标系
        # T_wcf_ocf: OCF -> WCF (工件坐标系 -> 世界坐标系)
        ones = np.ones((self._original_points.shape[0], 1))
        homogeneous_points = np.hstack([self._original_points, ones])  # (N, 4)
        world_points = (T_wcf_ocf @ homogeneous_points.T).T  # (N, 4)
        world_points = world_points[:, :3]  # 取前 3 列得到 (N, 3)

        # 根据世界坐标系下的 Box Bounds 进行过滤
        mask = (
            (world_points[:, 0] >= x_min) &
            (world_points[:, 0] <= x_max) &
            (world_points[:, 1] >= y_min) &
            (world_points[:, 1] <= y_max) &
            (world_points[:, 2] >= z_min) &
            (world_points[:, 2] <= z_max)
        )

        # 保留在 Box 内的点
        self._original_points = self._original_points[mask]
        self._status = np.ones(len(self._original_points), dtype=np.int32)

        # 重新加载点云
        self.load_workpiece_cloud(self._original_points)

        # 清除 Box Widget
        self._plotter.clear_box_widgets()

        # 重置缓存
        self._current_box_bounds = None

        print(f"[RenderEngine] Box 裁剪完成: 剩余 {len(self._original_points)} 点")
        return True

    # ==================== 刀具 ====================

    def create_tool(self, filepath: str):
        """
        创建刀具（仅支持自定义 STL 文件）。

        若已有刀具则先卸载，重置轴向后再加载新刀具，
        并立即同步到当前法兰盘位姿，避免残留旋转。
        """
        if self._tool_actor:
            self._plotter.remove_actor("cutting_tool")
            self._tool_actor = None
            # 重置轴向：防止旧机器人体位姿的旋转残留到新刀具上
            self._end_effector_axis = np.array([0.0, 0.0, 1.0])

        tool_mesh = pv.read(filepath)
        # CAD 默认 mm -> m 转换
        tool_mesh.points *= 0.001

        # 1. 将 CAD 刀具底部中心对齐到法兰原点
        bounds = tool_mesh.bounds
        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        z_bottom = bounds[4]
        tool_mesh.translate([-center_x, -center_y, -z_bottom], inplace=True)

        # 2. 计算刀具几何参数（对齐后）
        # 顶面高度（从底面到顶面的总高）
        self._tool_tip_height = float(bounds[5] - bounds[4])
        # 底面（刀尖处）XY 半径：测量 z = z_bottom 平面附近点的最大径向距离
        bottom_threshold = max(self._tool_tip_height * 0.02, 1e-5)
        bottom_mask = tool_mesh.points[:, 2] <= (0.0 + bottom_threshold)
        if np.any(bottom_mask):
            pts_bottom = tool_mesh.points[bottom_mask]
            rho = np.sqrt(pts_bottom[:, 0] ** 2 + pts_bottom[:, 1] ** 2)
            self._tool_tip_radius = float(rho.max()) if rho.max() > 0 else 0.005
        else:
            # fallback：测量整个 mesh 在底面四分之一高度处的最大半径
            z_quarter = self._tool_tip_height * 0.25
            quarter_mask = tool_mesh.points[:, 2] <= z_quarter
            if np.any(quarter_mask):
                pts_q = tool_mesh.points[quarter_mask]
                rho_q = np.sqrt(pts_q[:, 0] ** 2 + pts_q[:, 1] ** 2)
                self._tool_tip_radius = float(rho_q.max()) if rho_q.max() > 0 else 0.005
            else:
                self._tool_tip_radius = 0.005
        # 同时保留 _tool_radius（供兼容性使用）
        self._tool_radius = self._tool_tip_radius

        # 3. 旋转刀具使其局部 Z 轴与末端执行器轴对齐
        default_z = np.array([0.0, 0.0, 1.0])
        target_z = self._end_effector_axis

        v = np.cross(default_z, target_z)
        s = np.linalg.norm(v)
        c = np.dot(default_z, target_z)

        if s > 1e-6:
            v = v / s
            K = np.array([
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0]
            ])
            angle = np.arccos(np.clip(c, -1.0, 1.0))
            R = np.eye(3) + np.sin(angle) * K + (1 - c) * (K @ K)
            tool_mesh.points = (R @ tool_mesh.points.T).T

        self._tool_actor = self._plotter.add_mesh(
            tool_mesh,
            name="cutting_tool",
            # 正常刀具使用白色；红色只由碰撞高亮路径临时覆盖。
            color='#FFFFFF',
            opacity=1.0
        )
        self._tool_mesh = tool_mesh  # 缓存已单位转换+对齐后的 mesh（供 CollisionManager 复用）

        # 3. 立即同步到当前法兰盘位姿，避免新刀具叠加了旧旋转
        flange_world_pos = None
        if self._robot:
            self._sync_tool_to_flange()
            self._plotter.render()
            # 获取加载时刻的法兰盘世界坐标作为 C0
            model = self._robot._pinocchio_model
            data = self._robot._pinocchio_data
            if model and data:
                if model.existFrame("tool0"):
                    frame_id = model.getFrameId("tool0")
                    flange_se3 = data.oMf[frame_id]
                elif model.existFrame("flange"):
                    frame_id = model.getFrameId("flange")
                    flange_se3 = data.oMf[frame_id]
                else:
                    last_id = len(model.names) - 1
                    flange_se3 = data.oMi[last_id]
                flange_world_pos = (self._robot.base_transform @ flange_se3.homogeneous)[:3, 3]

        # 4. 输出刀具切削参数日志（加载时刻用 STL 测量值）
        c0_str = (f"({flange_world_pos[0]:.4f}, {flange_world_pos[1]:.4f}, "
                   f"{flange_world_pos[2]:.4f})") if flange_world_pos is not None else "(未知)"
        print(f"[RenderEngine] 刀具切削参数 → C0={c0_str}, "
              f"R={self._tool_tip_radius:.5f}m, H={self._tool_tip_height:.5f}m")

    def remove_tool(self) -> None:
        """移除刀具 actor（供切换机器人时清理残留）"""
        try:
            if self._tool_actor is not None:
                self._plotter.remove_actor(self._tool_actor, reset_camera=False)
        except Exception:
            try:
                self._plotter.remove_actor("cutting_tool", reset_camera=False)
            except Exception:
                pass
        self._tool_actor = None
        self._tool_mesh = None

    def update_tool_transform(self, flange_T: np.ndarray):
        """将刀具模型绑定到法兰盘的位姿上"""
        if self._tool_actor:
            self._tool_actor.user_matrix = flange_T

    def update_tool_position(self, position: np.ndarray):
        """更新刀具位置"""
        if self._tool_actor:
            self._tool_actor.SetPosition(position[0], position[1], position[2])

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """
        从 Pinocchio 模型返回关节位置上下限。

        Returns
        -------
        lower : np.ndarray
            各关节下限，shape (nq,)。
        upper : np.ndarray
            各关节上限，shape (nq,)。
        """
        import numpy as np
        model = getattr(self._robot, "_pinocchio_model", None)
        nq = int(getattr(self._robot, "_nq", 0))
        if model is None:
            return -np.pi * np.ones(nq), np.pi * np.ones(nq)
        lower = model.lowerPositionLimit.copy()
        upper = model.upperPositionLimit.copy()
        lower = np.nan_to_num(lower, nan=-np.pi, posinf=np.pi, neginf=-np.pi)
        upper = np.nan_to_num(upper, nan=np.pi, posinf=np.pi, neginf=np.pi)
        return lower, upper

    # ==================== 坐标系可视化 ====================

    def create_coordinate_axes(self, size: float = 0.1) -> pv.PolyData:
        """
        创建三色坐标系（用于可视化）。

        X=红, Y=绿, Z=蓝。通过分别渲染每根轴确保颜色正确。

        返回:
            pv.PolyData: 包含三根带颜色的线条
        """
        origin = np.array([0.0, 0.0, 0.0])

        # X 轴（红色）：两个端点
        x_pts = np.array([origin, np.array([size, 0.0, 0.0])])
        x_lines = np.hstack([[2, 0, 1]])
        x_poly = pv.PolyData(x_pts)
        x_poly.lines = x_lines

        # Y 轴（绿色）
        y_pts = np.array([origin, np.array([0.0, size, 0.0])])
        y_lines = np.hstack([[2, 0, 1]])
        y_poly = pv.PolyData(y_pts)
        y_poly.lines = y_lines

        # Z 轴（蓝色）
        z_pts = np.array([origin, np.array([0.0, 0.0, size])])
        z_lines = np.hstack([[2, 0, 1]])
        z_poly = pv.PolyData(z_pts)
        z_poly.lines = z_lines

        # 合并：使用 combine_filter 将三条线合并为一个 PolyData
        merged = x_poly.merge(y_poly).merge(z_poly)

        return merged

    def update_coordinate_frames(
        self,
        flange_pose: np.ndarray,
        tool_pose: Optional[np.ndarray] = None,
        flange_size: float = 0.1,
        tool_size: float = 0.08
    ):
        """
        更新法兰坐标系可视化（X=红, Y=绿, Z=蓝）。

        三根轴各自独立 actor，直接变换各自端点后更新，
        无需依赖 user_matrix，兼容 PyVista 所有版本。

        仅在 _show_coordinate_frames 为 True 时才渲染。
        """
        if not getattr(self, '_show_coordinate_frames', True):
            return

        try:
            R = flange_pose[:3, :3]
            t = flange_pose[:3, 3]

            # 局部坐标系三个轴的方向（归一化）
            local_x = R @ np.array([flange_size, 0.0, 0.0])
            local_y = R @ np.array([0.0, flange_size, 0.0])
            local_z = R @ np.array([0.0, 0.0, flange_size])

            origin = t

            if self._flange_axes_actor is None:
                # 首次创建：三根独立线段，分别上色
                x_line = pv.Line(origin, origin + local_x)
                y_line = pv.Line(origin, origin + local_y)
                z_line = pv.Line(origin, origin + local_z)

                self._flange_x_actor = self._plotter.add_mesh(x_line, name="flange_x", color='red',    line_width=4, opacity=1.0)
                self._flange_y_actor = self._plotter.add_mesh(y_line, name="flange_y", color='green',  line_width=4, opacity=1.0)
                self._flange_z_actor = self._plotter.add_mesh(z_line, name="flange_z", color='blue',   line_width=4, opacity=1.0)
                self._flange_axes_actor = True  # sentinel: mark as initialized
            else:
                # 更新三根轴的端点
                self._flange_x_actor.mapper.SetInputData(pv.Line(origin, origin + local_x))
                self._flange_y_actor.mapper.SetInputData(pv.Line(origin, origin + local_y))
                self._flange_z_actor.mapper.SetInputData(pv.Line(origin, origin + local_z))

                self._flange_x_actor.mapper.Update()
                self._flange_y_actor.mapper.Update()
                self._flange_z_actor.mapper.Update()

        except Exception as e:
            print(f"[RenderEngine] 坐标系可视化更新失败: {e}")

    def _clear_coordinate_frames(self):
        """清除坐标系可视化"""
        for actor_name in ("flange_x", "flange_y", "flange_z", "flange_axes"):
            if hasattr(self, '_plotter') and self._plotter is not None:
                self._plotter.remove_actor(actor_name)
        self._flange_axes_actor = None
        self._flange_x_actor = None
        self._flange_y_actor = None
        self._flange_z_actor = None
        if hasattr(self, '_tool_axes_actor') and self._tool_axes_actor is not None:
            self._plotter.remove_actor("tool_axes")
            self._tool_axes_actor = None

    def remove_coordinate_frames(self):
        """移除坐标系可视化"""
        self._clear_coordinate_frames()

    def _ensure_flange_axes_created(self):
        """
        用当前法兰盘位姿创建/更新法兰坐标系。

        若 _flange_axes_actor 未初始化（从未调用过 update_coordinate_frames），
        则传入当前法兰盘位姿触发首次创建；否则直接更新。
        """
        if not self._robot:
            return
        model = self._robot._pinocchio_model
        data = self._robot._pinocchio_data
        base_T = self._robot.base_transform

        if model.existFrame("tool0"):
            frame_id = model.getFrameId("tool0")
            flange_se3 = data.oMf[frame_id]
        elif model.existFrame("flange"):
            frame_id = model.getFrameId("flange")
            flange_se3 = data.oMf[frame_id]
        else:
            last_id = len(model.names) - 1
            flange_se3 = data.oMi[last_id]

        flange_T = base_T @ flange_se3.homogeneous
        self.update_coordinate_frames(flange_T)

    def toggle_coordinate_frames(self, show: Optional[bool] = None) -> bool:
        """
        切换坐标系显示

        参数:
            show: True 显示，False 隐藏，None 切换

        返回:
            bool: 当前显示状态
        """
        if not hasattr(self, '_show_coordinate_frames'):
            self._show_coordinate_frames = True

        if show is None:
            self._show_coordinate_frames = not self._show_coordinate_frames
        else:
            self._show_coordinate_frames = show

        if not self._show_coordinate_frames:
            self._clear_coordinate_frames()
        elif self._flange_axes_actor is None:
            self._ensure_flange_axes_created()

        return self._show_coordinate_frames

    def set_flange_axes_visible(self, visible: bool):
        """
        设置法兰坐标系（三色坐标轴）的可见性。

        参数:
            visible: True 显示，False 隐藏
        """
        if visible:
            self._show_coordinate_frames = True
            if self._flange_axes_actor is None:
                self._ensure_flange_axes_created()
        else:
            self._show_coordinate_frames = False
            self._clear_coordinate_frames()

    # ==================== 刀轨预览 ====================

    @staticmethod
    def _trajectory_color(kind) -> str:
        value = getattr(kind, "value", str(kind)).lower()
        return {
            "process": "#E53935",
            "approach": "#1E88E5",
            "retract": "#FB8C00",
            "rapid": "#8E24AA",
            "blend": "#FDD835",
        }.get(value, "#90A4AE")

    def _clear_named_actors(self, names: List[str]) -> None:
        for name in names:
            self._plotter.remove_actor(name, reset_camera=False)
        names.clear()

    def _add_semantic_polyline(
        self,
        points,
        segment_types,
        segment_ids,
        *,
        prefix,
        line_width,
        layer_ids=None,
    ):
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return []
        kinds = (
            np.full(len(points), "process", dtype=object)
            if segment_types is None else np.asarray(segment_types, dtype=object)
        )
        ids = (
            np.zeros(len(points), dtype=int)
            if segment_ids is None else np.asarray(segment_ids, dtype=int)
        )
        layers = (
            np.zeros(len(points), dtype=int)
            if layer_ids is None else np.asarray(layer_ids, dtype=int)
        )
        if (
            kinds.shape != (len(points),)
            or ids.shape != (len(points),)
            or layers.shape != (len(points),)
        ):
            raise ValueError("轨迹语义长度必须与轨迹点数一致")

        def color_at(index: int) -> str:
            return self._trajectory_color(kinds[index])

        names = []
        start = 0
        group = 0
        for index in range(1, len(points)):
            layer_boundary = layers[index] != layers[index - 1]
            boundary = (
                kinds[index] != kinds[index - 1]
                or ids[index] != ids[index - 1]
                or layer_boundary
            )
            if not boundary:
                continue
            # A real layer boundary is a pending TransitionRequest, not a
            # drawable edge. Other semantic changes on a continuous route keep
            # the shared endpoint for backward-compatible execution rendering.
            stop = index if layer_ids is not None and layer_boundary else index + 1
            if stop - start >= 2:
                name = f"{prefix}_{group}"
                polyline = pv.lines_from_points(points[start:stop], close=False)
                self._plotter.add_mesh(
                    polyline, name=name, color=color_at(index - 1),
                    line_width=line_width, opacity=0.95, pickable=False,
                )
                names.append(name)
                group += 1
            start = index
        if len(points) - start >= 2:
            name = f"{prefix}_{group}"
            polyline = pv.lines_from_points(points[start:], close=False)
            self._plotter.add_mesh(
                polyline, name=name, color=color_at(len(points) - 1),
                line_width=line_width, opacity=0.95, pickable=False,
            )
            names.append(name)
        return names

    def load_toolpath_preview(self, points: np.ndarray, normals: Optional[np.ndarray] = None,
                              T_wcf_ocf: Optional[np.ndarray] = None,
                              segment_types=None, segment_ids=None, layer_ids=None):
        """
        加载刀轨预览

        参数:
            points: Nx3 刀轨点（OCF 坐标）
            normals: Nx3 法向量（OCF 坐标）
            T_wcf_ocf: OCF → WCF 变换矩阵（使刀轨跟随零件移动）
        """
        self._clear_named_actors(self._toolpath_actor_names)

        # 应用 OCF → WCF 坐标变换（通过 CoordinateTransformer）
        if T_wcf_ocf is not None:
            self._transformer.T_wcf_ocf = T_wcf_ocf
        display_points, display_normals = self._transformer.transform_points_with_normals(
            points, normals, apply_ocf=(T_wcf_ocf is not None)
        )
        self._toolpath_display_points = np.asarray(display_points, dtype=float)
        self._toolpath_layer_ids = (
            np.zeros(len(display_points), dtype=int)
            if layer_ids is None else np.asarray(layer_ids, dtype=int)
        )
        self._toolpath_segment_ids = (
            np.zeros(len(display_points), dtype=int)
            if segment_ids is None else np.asarray(segment_ids, dtype=int)
        )
        self._toolpath_segment_types = (
            np.full(len(display_points), "process", dtype=object)
            if segment_types is None else np.asarray(segment_types, dtype=object)
        )

        # 预览只按轨迹语义上色；PROCESS 层不再按 layer_id 区分颜色。
        self._toolpath_points = pv.PolyData(display_points)
        self._toolpath_actor_names.extend(
            self._add_semantic_polyline(
                display_points, segment_types, segment_ids,
                prefix="toolpath_preview", line_width=4, layer_ids=layer_ids,
            )
        )
        point_name = "toolpath_semantic_points"
        self._plotter.add_mesh(
            self._toolpath_points,
            name=point_name,
            color="#FFFFFF",
            point_size=7,
            render_points_as_spheres=True,
            opacity=0.85,
            pickable=True,
        )
        self._toolpath_actor_names.append(point_name)
        if len(display_points):
            self.show_toolpath_point_metadata(0)
            self._toolpath_actor_names.append("toolpath_point_metadata")
            self._toolpath_actor_names.append("toolpath_selected_point")

            def on_pick(point):
                if point is None or not len(self._toolpath_display_points):
                    return
                index = int(np.argmin(
                    np.linalg.norm(self._toolpath_display_points - np.asarray(point), axis=1)
                ))
                self.show_toolpath_point_metadata(index)

            try:
                self._plotter.enable_point_picking(
                    callback=on_pick,
                    show_message=False,
                    show_point=False,
                    pickable_window=False,
                )
            except (TypeError, RuntimeError):
                try:
                    self._plotter.enable_point_picking(callback=on_pick)
                except RuntimeError:
                    pass

        # 法线箭头 (用球体代替 pv.arrows 以兼容所有 PyVista 版本)
        if display_normals is not None and len(display_normals) > 0:
            step = max(1, len(display_normals) // 50)
            arrow_start = display_points[::step]

            arrows = pv.PolyData(arrow_start)
            self._plotter.add_mesh(
                arrows,
                name="toolpath_normals",
                color='green',
                point_size=5,
                render_points_as_spheres=True,
                opacity=0.8,
                pickable=False,
            )
            self._toolpath_actor_names.append("toolpath_normals")

    def show_toolpath_point_metadata(self, index: int) -> str:
        """Display and return canonical semantics for one preview point."""
        if not 0 <= int(index) < len(self._toolpath_display_points):
            raise IndexError("toolpath point index out of range")
        index = int(index)
        kind = self._toolpath_segment_types[index]
        kind_value = getattr(kind, "value", str(kind))
        text = (
            f"Point {index} | layer_id={int(self._toolpath_layer_ids[index])} | "
            f"segment_id={int(self._toolpath_segment_ids[index])} | "
            f"segment_type={kind_value}"
        )
        self._plotter.add_text(
            text,
            position="upper_left",
            font_size=11,
            color="#FFFFFF",
            name="toolpath_point_metadata",
        )
        if hasattr(self._plotter, "add_mesh"):
            self._plotter.add_mesh(
                pv.PolyData(self._toolpath_display_points[[index]]),
                name="toolpath_selected_point",
                color="#00E5FF",
                point_size=16,
                render_points_as_spheres=True,
                pickable=False,
            )
        return text

    def load_execution_trajectory(self, tcp_poses, segment_types=None, segment_ids=None):
        """Show the full physical TCP route using persistent semantic colors."""
        poses = np.asarray(tcp_poses, dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("tcp_poses 必须为 (N,4,4)")
        self._clear_named_actors(self._execution_path_actor_names)
        self._execution_path_actor_names.extend(
            self._add_semantic_polyline(
                poses[:, :3, 3], segment_types, segment_ids,
                prefix="execution_path", line_width=5,
            )
        )
        self._plotter.add_text(
            "轨迹语义  红=加工  蓝=接近  橙=退刀  紫=快速  黄=过渡  青=法兰实走",
            position="lower_left", font_size=10, color="#ECEFF1",
            name="trajectory_semantic_legend",
        )
        self.reset_flange_trail()
        self._plotter.render()

    def reset_flange_trail(self):
        """Clear the real flange-origin trail before a new playback."""
        self._flange_trail_points.clear()
        self._plotter.remove_actor("flange_origin_trail", reset_camera=False)
        self._flange_trail_actor = None

    def append_flange_trail(self, point, sample_kind=None):
        """Append one actually executed flange-origin point to the live cyan trail."""
        del sample_kind  # Static route carries semantics; cyan is reserved for actual motion.
        point = np.asarray(point, dtype=float)
        if self._flange_trail_points:
            if np.linalg.norm(point - self._flange_trail_points[-1]) < 1e-5:
                return
        self._flange_trail_points.append(point.copy())
        if len(self._flange_trail_points) > 10_000:
            self._flange_trail_points = self._flange_trail_points[::2]
        if len(self._flange_trail_points) < 2:
            return
        trail = pv.lines_from_points(np.asarray(self._flange_trail_points), close=False)
        if self._flange_trail_actor is None:
            self._flange_trail_actor = self._plotter.add_mesh(
                trail, name="flange_origin_trail", color="#00E5FF",
                line_width=5, opacity=1.0, pickable=False,
            )
        else:
            self._flange_trail_actor.mapper.SetInputData(trail)
            self._flange_trail_actor.mapper.Update()

    def load_trajectory_axes(self, poses_ocf: List[np.ndarray], sample_step: int = 15,
                             axis_length: float = 0.015, point_size: float = 25.0,
                             T_wcf_ocf: Optional[np.ndarray] = None):
        """
        在每个轨迹点位置渲染 RGB 坐标系（X=红, Y=绿, Z=蓝）和紫色采样点。

        持久化模式：若 _trajectory_axes_actors 已缓存（Tab 切换场景），
        则跳过复建，仅恢复显示；否则构建新 actors。
        """
        if self._plotter is None:
            return

        # 新数据到达时，清除旧缓存强制重建（_trajectory_axes_actors 非空不代表数据最新）
        if self._trajectory_axes_actors:
            self.clear_trajectory_axes()

        if self._trajectory_axes_actors:
            # Actor 已缓存（Tab 切换后），仅恢复可见
            for name in self._trajectory_axes_actors:
                self._plotter.add_actor(name, reset_camera=False)
            self._plotter.render()
            return

        self._build_trajectory_axes_actors(
            poses_ocf, sample_step, axis_length, point_size, T_wcf_ocf)
        self._trajectory_axes_visible = True

    def _build_trajectory_axes_actors(self, poses_ocf: List[np.ndarray],
                                     sample_step: int, axis_length: float,
                                     point_size: float, T_wcf_ocf: Optional[np.ndarray]):
        """
        构建轨迹坐标系 actor 对象并存储到 _trajectory_axes_actors（不渲染）。
        与 _c0_axes_*_actor 相同的持久化模式。
        """
        self._trajectory_axes_actors.clear()

        if T_wcf_ocf is not None:
            self._transformer.T_wcf_ocf = T_wcf_ocf
        if not CoordinateTransformer.is_identity(self._transformer.T_wcf_ocf):
            poses_wcf = self._transformer.ocf_to_wcf_batch(poses_ocf)
        else:
            poses_wcf = poses_ocf

        sampled = poses_wcf[::sample_step]
        purple_pts = []

        RGB_COLORS = {
            'x': [1.0, 0.0, 0.0],
            'y': [0.0, 1.0, 0.0],
            'z': [0.0, 0.4, 1.0],
        }
        rgb_pts = {'x': [], 'y': [], 'z': []}
        rgb_lines = {'x': [], 'y': [], 'z': []}
        rgb_pt_idx = {'x': 0, 'y': 0, 'z': 0}

        Rx180 = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ])

        for T in sampled:
            pos = T[:3, 3]
            R = T[:3, :3]
            R_rotated = R @ Rx180
            purple_pts.append(pos)
            for axis_name, axis_vec in [('x', R_rotated[:, 0] * axis_length),
                                         ('y', R_rotated[:, 1] * axis_length),
                                         ('z', R_rotated[:, 2] * axis_length)]:
                idx = rgb_pt_idx[axis_name]
                rgb_pts[axis_name].append(pos)
                rgb_pts[axis_name].append(pos + axis_vec)
                rgb_lines[axis_name].append([2, idx, idx + 1])
                rgb_pt_idx[axis_name] += 2

        for axis_name in ('x', 'y', 'z'):
            if rgb_pts[axis_name]:
                pts_array = np.array(rgb_pts[axis_name])
                lines_array = (np.hstack(rgb_lines[axis_name])
                               if rgb_lines[axis_name] else np.array([], dtype=np.int64))
                poly = pv.PolyData(pts_array)
                poly.lines = lines_array
                actor = self._plotter.add_mesh(
                    poly,
                    name=f"trajectory_axis_{axis_name}",
                    color=RGB_COLORS[axis_name],
                    line_width=2,
                    opacity=1.0
                )
                # 存储 actor 对象（用于 SetVisibility 切换）
                self._trajectory_axes_actors.append(actor)

        if purple_pts:
            purple_poly = pv.PolyData(np.array(purple_pts))
            actor = self._plotter.add_mesh(
                purple_poly,
                name="trajectory_points",
                color=[0.5, 0.0, 0.5],
                point_size=max(2.0, point_size * 0.2),
                render_points_as_spheres=False
            )
            self._trajectory_axes_actors.append(actor)

    def set_trajectory_axes_visible(self, visible: bool,
                                    poses_ocf: Optional[List[np.ndarray]] = None,
                                    sample_step: int = 15,
                                    axis_length: float = 0.015,
                                    point_size: float = 25.0,
                                    T_wcf_ocf: Optional[np.ndarray] = None):
        """
        设置轨迹坐标系常态化显示（与刀具包围盒相同的持久化模式）。

        参数:
            visible: 是否显示
            poses_ocf: 轨迹位姿列表（仅在需要构建时传入）
            sample_step: 采样步长
            axis_length: 坐标轴长度
            point_size: 紫色点大小
            T_wcf_ocf: OCF→WCF 变换矩阵
        """
        if self._plotter is None:
            return

        self._trajectory_axes_visible = visible

        if visible:
            if not self._trajectory_axes_actors and poses_ocf is not None:
                self._build_trajectory_axes_actors(
                    poses_ocf, sample_step, axis_length, point_size, T_wcf_ocf)
            for actor in self._trajectory_axes_actors:
                actor.SetVisibility(True)
            self._plotter.render()
        else:
            for actor in self._trajectory_axes_actors:
                actor.SetVisibility(False)
            self._plotter.render()

    def clear_trajectory_axes(self):
        """彻底清除轨迹坐标系（销毁 actor，释放缓存）。"""
        if self._plotter:
            for axis_name in ('x', 'y', 'z'):
                self._plotter.remove_actor(f"trajectory_axis_{axis_name}")
            self._plotter.remove_actor("trajectory_points")
            self._trajectory_axes_actors.clear()
            self._trajectory_axes_visible = False
            self._plotter.render()

    def remove_toolpath_preview(self):
        """销毁全部刀轨预览 actor 与选点语义缓存。"""
        self._clear_named_actors(self._toolpath_actor_names)
        self._toolpath_points = None
        self._toolpath_display_points = np.empty((0, 3), dtype=float)
        self._toolpath_layer_ids = np.empty(0, dtype=int)
        self._toolpath_segment_ids = np.empty(0, dtype=int)
        self._toolpath_segment_types = np.empty(0, dtype=object)
        if self._plotter is not None:
            self._plotter.render()

    # ==================== 调试数据可视化 ====================

    def render_debug_items(self, debug_items: dict):
        """
        渲染调试数据（中间结果可视化）

        参数:
            debug_items: 调试数据字典，键为名称，值为 DebugItem 对象
        """
        # 清除旧的调试数据
        actor_names = list(self._plotter.actors.keys())
        for name in actor_names:
            if name.startswith("debug_"):
                self._plotter.remove_actor(name)

        # 遍历并渲染每种调试数据
        for label, item in debug_items.items():
            safe_name = f"debug_{label}"
            data = item.data

            if data is None or len(data) == 0:
                continue

            dtype_value = item.dtype.value if hasattr(item.dtype, 'value') else str(item.dtype)
            color = item.color if hasattr(item, 'color') else "#FFFFFF"

            try:
                if dtype_value in ("polyline", "curve"):
                    # 渲染多段线/曲线
                    if data.shape[1] == 3:
                        # 构建 PyVista PolyData (线条)
                        n_points = len(data)
                        lines = np.column_stack([
                            np.full(n_points - 1, 2, dtype=np.int64),
                            np.arange(n_points - 1, dtype=np.int64),
                            np.arange(1, n_points, dtype=np.int64)
                        ])
                        poly = pv.PolyData(data, lines)
                        self._plotter.add_mesh(
                            poly,
                            name=safe_name,
                            color=color,
                            line_width=3,
                            opacity=0.8
                        )

                elif dtype_value == "point_cloud":
                    # 渲染点云（默认点元，不画球 — 高性能档）
                    poly = pv.PolyData(data)
                    self._plotter.add_mesh(
                        poly,
                        name=safe_name,
                        color=color,
                        render_points_as_spheres=False,
                        point_size=2.0,
                        opacity=0.8
                    )

                elif dtype_value == "vector_field":
                    # 渲染向量场（箭头）
                    if data.shape[1] == 6:
                        points = data[:, :3]
                        vectors = data[:, 3:]
                        arrows = pv.arrows(points, vectors, mag=1.0)
                        self._plotter.add_mesh(
                            arrows,
                            name=safe_name,
                            color=color,
                            opacity=0.8
                        )

            except Exception as e:
                print(f"[RenderEngine] 渲染调试数据 '{label}' 失败: {e}")

    def clear_debug_items(self):
        """清除所有调试数据可视化"""
        actor_names = list(self._plotter.actors.keys())
        for name in actor_names:
            if name.startswith("debug_"):
                self._plotter.remove_actor(name)

    # ==================== 切削仿真 ====================

    def execute_cutting(self, world_position: np.ndarray) -> int:
        """
        执行切削

        参数:
            world_position: 刀具世界坐标

        返回:
            int: 被切削的点数
        """
        if self._kdtree is None or self._status is None or self._original_points is None:
            return 0

        indices = self._kdtree.query_ball_point(world_position, self._tool_radius)

        if len(indices) > 0:
            original_indices = self._active_indices[indices]
            self._status[original_indices] = 0

            # 更新显示
            self._rebuild_kdtree()
            self._update_point_cloud_colors()

            return len(indices)

        return 0

    def _update_point_cloud_colors(self):
        """更新点云颜色"""
        if self._workpiece_cloud is None or self._original_points is None:
            return

        # 只显示 active 的点
        active_mask = self._status == 1
        active_points = self._original_points[active_mask]

        if len(active_points) > 0:
            new_cloud = pv.PolyData(active_points)
            self._workpiece_cloud.overwrite(new_cloud)
            self._plotter.render()

    def perform_cutting_simulation(self, tool_tip_world: np.ndarray,
                                    tool_axis: np.ndarray,
                                    log_callback: Optional[Callable] = None,
                                    tolerance: float = 0.001,
                                    translucent_value: float = 0.0,
                                    render: bool = True) -> int:
        """
        基于世界坐标系解析几何公式执行实时切削模拟（与 stage_2_pyside6_app.py 一致的 RGBA 管线）。

        工件局部坐标系 (OCF) 下的点通过工件 Actor 的 user_matrix (T_wcf_ocf)
        一键投影到世界坐标系，再与刀具圆柱做布尔交集判定：

            轴向坐标：t = (p - C0) · u
            径向距离：rho = ||(p - C0) - t*u||

        点在圆柱内当且仅当：t ∈ [0, h] 且 rho <= r + tolerance

        参数:
            tool_tip_world: 刀具底面中心 C0 的世界坐标 (3,)
            tool_axis:     刀具轴向单位向量 u (3,)
            tolerance:     安全余量 r_eff = r + tolerance
            translucent_value: 已废弃，保留参数兼容性
            log_callback:  可选，日志回调函数

        返回:
            int: 本帧被标记为切削的点数
        """
        def _log(msg):
            if log_callback:
                log_callback(msg)

        if self._original_points is None or self._workpiece_cloud is None or self._rgba_colors is None:
            return 0

        # 1. 获取点云当前的世界变换矩阵 T_wcf_ocf
        workpiece_actor = self._plotter.actors.get("workpiece_cloud")
        T_wcf_ocf = workpiece_actor.user_matrix if workpiece_actor else np.eye(4)

        # 2. 将全量原始点一键转换到世界坐标系
        ones = np.ones((self._original_points.shape[0], 1))
        points_wcf = (T_wcf_ocf @ np.hstack([self._original_points, ones]).T).T[:, :3]

        # 3. 获取当前颜色通道（仅对尚未被切削的点计算）
        # 切削过的点 RGBA 为 [R=0, G=0, B=0, A=0]，据此判断 active 状态
        active_mask = (self._rgba_colors[:, 3] > 0)
        if not np.any(active_mask):
            return 0

        pts_to_check = points_wcf[active_mask]

        # 4. NumPy 向量化圆柱判定
        # 轴向坐标：t = (p - C0) · u
        diff = pts_to_check - tool_tip_world          # (M, 3)
        t = diff @ tool_axis                           # (M,)

        # 径向距离：rho = ||(p - C0) - t*u||
        proj_vector = t[:, np.newaxis] * tool_axis     # (M, 3)
        rho = np.linalg.norm(diff - proj_vector, axis=1)  # (M,)

        # 动态获取刀具总长度
        tool_length = self._get_tool_length()
        in_cylinder = (t >= 0) & (t <= tool_length) & (
            rho <= (self._tool_radius + tolerance)
        )

        if np.any(in_cylinder):
            # 5. 核心逻辑（与 stage_2_pyside6_app.py 一致）：
            #    切削掉的点设为 [R=0, G=0, B=0, A=0]，VTK PolyDataMapper + rgb=True 直接渲染
            global_indices = np.where(active_mask)[0][in_cylinder]
            self._rgba_colors[global_indices] = [0, 0, 0, 0]

            # 只更新数据并刷新画面（与 stage_2_pyside6_app.py 一致）
            self._workpiece_cloud.point_data["colors"] = self._rgba_colors
            self._workpiece_cloud.Modified()
            if render:
                self.render()

            n_cut = len(global_indices)
            _log(f"  [切削] 刀尖 ({tool_tip_world[0]:.3f}, {tool_tip_world[1]:.3f}, "
                 f"{tool_tip_world[2]:.3f}) → 切削 {n_cut} 点")
            return n_cut

        return 0

    def reset(self):
        """重置场景：恢复所有点的透明度，重新加载点云"""
        if self._original_points is not None:
            self._status = np.ones(len(self._original_points), dtype=np.int32)
            self.load_workpiece_cloud(self._original_points)
        self.reset_flange_trail()

    def delete_transparent_points(self):
        """
        仿真播放完成后统一调用：将所有已被透明化的点彻底从网格中删除。
        保留透明度仍接近 1.0 的点，重新构建干净的点云。
        """
        if self._original_points is None or self._workpiece_cloud is None:
            return

        # 根据 RGBA alpha 通道判断剩余点（与 perform_cutting_simulation 一致）
        remaining_mask = self._rgba_colors[:, 3] > 0 if self._rgba_colors is not None else np.ones(len(self._original_points), dtype=bool)

        # 保存当前工件 actor 的 user_matrix，重新加载后恢复
        saved_matrix = self._workpiece_cloud_actor.user_matrix if self._workpiece_cloud_actor else None

        if np.any(remaining_mask):
            self._original_points = self._original_points[remaining_mask]
            self._status = np.ones(len(self._original_points), dtype=np.int32)
            self.load_workpiece_cloud(self._original_points)
            # 恢复工件位姿
            if saved_matrix is not None and self._workpiece_cloud_actor is not None:
                self._workpiece_cloud_actor.user_matrix = saved_matrix
                self._sync_pointcloud_axes_to_workpiece()
        else:
            self._original_points = np.empty((0, 3))
            if self._workpiece_cloud_actor is not None:
                self._plotter.remove_actor(self._workpiece_cloud_actor, reset_camera=False)
                self._workpiece_cloud_actor = None
            self._workpiece_cloud = None

        self._plotter.render()
        print(f"[RenderEngine] 最终点云清洗完毕，剩余 {len(self._original_points)} 点")

    # ==================== 变换 ====================

    def transform_robot(self, T: np.ndarray):
        """变换机器人基座，同时同步到 CoordinateTransformer"""
        self._transformer.T_wcf_rcf = T
        if self._robot:
            self._robot.set_base_transform(T)

    def transform_workpiece(self, T: np.ndarray):
        """
        变换 CAD 模型的位姿（不影响点云，点云由 transform_pointcloud 单独控制）
        同时同步到 CoordinateTransformer

        参数:
            T: 4x4 齐次变换矩阵
        """
        self._transformer.T_wcf_ocf = T
        if self._cad_model:
            self._cad_model.user_matrix = T
            self._sync_cad_axes_to_model()
        self._plotter.render()

    def transform_pointcloud(self, base_T: np.ndarray, extra_offset: Optional[np.ndarray] = None):
        """
        变换毛坯点云的位姿，同时同步到 CoordinateTransformer

        参数:
            base_T: 基础变换矩阵（通常是工件位姿 T_wcf_ocf）
            extra_offset: 额外的偏移矩阵（点云相对于 base_T 的额外变换）
        """
        # 注意：不在此更新 CoordinateTransformer.T_wcf_ocf，因为 transform_pointcloud
        # 只负责 actor.user_matrix（点云视觉表现），CoordinateTransformer.T_wcf_ocf
        # 由 transform_workpiece 单独维护（CAD 的真实物理位姿）。
        workpiece_actor = self._plotter.actors.get("workpiece_cloud")
        if workpiece_actor is not None:
            if extra_offset is not None:
                final_T = base_T @ extra_offset
            else:
                final_T = base_T
            workpiece_actor.user_matrix = final_T
            # 同步点云坐标轴跟随工件变换
            self._sync_pointcloud_axes_to_workpiece()
            self._plotter.render()

    def set_pointcloud_extra_offset(self, offset: np.ndarray):
        """
        设置点云的额外偏移量

        参数:
            offset: 4x4 齐次变换矩阵，表示点云相对于工件的偏移
        """
        self._pointcloud_extra_offset = offset

    def get_pointcloud_extra_offset(self) -> Optional[np.ndarray]:
        """获取点云的额外偏移量"""
        return self._pointcloud_extra_offset

    def get_pointcloud_transform(self) -> np.ndarray:
        """
        获取毛坯点云的当前变换矩阵

        返回:
            4x4 齐次变换矩阵，如果不存在返回单位矩阵
        """
        workpiece_actor = self._plotter.actors.get("workpiece_cloud")
        if workpiece_actor is not None and hasattr(workpiece_actor, 'user_matrix'):
            return workpiece_actor.user_matrix
        return np.eye(4)

    # ==================== 工具方法 ====================

    def render(self):
        """触发渲染（带节流优化，避免频繁重复渲染）"""
        current_time = time.perf_counter()

        # 节流检查：距离上次渲染是否超过最小间隔
        if self._min_render_interval_ms > 0:
            elapsed_ms = (current_time - self._last_render_time) * 1000
            if elapsed_ms < self._min_render_interval_ms:
                # 标记需要渲染，但不立即执行
                self._render_pending = True
                return

        self._plotter.render()
        self._last_render_time = current_time
        self._render_pending = False

        # FPS 统计（每秒更新一次）
        self._frame_count += 1
        if current_time - self._last_fps_update >= 1.0:
            self._fps = self._frame_count / (current_time - self._last_fps_update)
            self._frame_count = 0
            self._last_fps_update = current_time

    def set_render_throttle(self, min_interval_ms: int):
        """
        设置最小渲染间隔（毫秒）
        - 设置为 0 表示不限制（每帧都渲染）
        - 设置为 16 相当于限制最大 60 FPS
        - 设置为 33 相当于限制最大 30 FPS
        """
        self._min_render_interval_ms = max(0, min_interval_ms)

    def force_render(self):
        """强制立即渲染，忽略节流限制"""
        self._plotter.render()
        self._last_render_time = time.perf_counter()
        self._render_pending = False

    def get_render_stats(self) -> dict:
        """获取渲染统计信息"""
        return {
            "fps": round(self._fps, 1),
            "render_pending": self._render_pending,
            "throttle_ms": self._min_render_interval_ms
        }

    def enable_actor_picking(self, callback: Callable):
        """启用 Actor 拾取"""
        self._plotter.enable_point_picking(callback=callback)

    def get_plotter(self) -> 'QtInteractor':
        """获取 plotter"""
        return self._plotter
