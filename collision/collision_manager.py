"""
collision/collision_manager.py

4 类网格 + 实时碰撞检测的集中管理器。

设计目标:
    - 把 CollisionWidget 中的"网格构建 + fcl 构建 + 检测 + 高亮分发"全部接管
    - 暴露最小化 API 给 main_app.py: register/unregister/update_robot_joints/refresh
    - 不直接依赖 Qt（manager 是纯逻辑层，UI 通过回调订阅）
    - 端到端的数据流: q → 机器人可视器 → 刀具可视器 → checker.check() → 红色高亮

颜色规范 (4 类统一灰色):
    机器人 灰色实线 #888888
    CAD    灰色实线 #888888
    环境  灰色实线 #888888
    刀具  灰色实线 #888888
    碰撞  红色     #FF3333
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Set

import numpy as np

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    pin = None

from collision.convex_hull import (
    CollisionMeshVisualizer,
    CADWireframeVisualizer,
    ToolWireframeVisualizer,
    EnvironmentMeshVisualizer,
    make_transform,
    parse_urdf_collision_meshes,
    auto_convert_mesh_to_meters,
)
from collision.robot_env_collision import (
    MultiEnvCollisionChecker,
    _local_aabb_from_pv,
    _world_aabb_from_local,
)
from collision.scene_snapshot import (
    AttachedBodySpec,
    MeshGeometry,
    RobotLinkSpec,
    SceneSnapshot,
    SceneSnapshotBuildError,
    StaticCollisionObjectSpec,
)


# ----- 颜色常量 -----
# 统一灰色（#888888）：所有碰撞网格线统一灰色，方便识别碰撞体轮廓
COLOR_ROBOT_WIRE = '#888888'   # 机器人线框
COLOR_CAD_WIRE   = '#888888'   # CAD 线框
COLOR_ENV_WIRE   = '#888888'   # 环境物体线框
COLOR_TOOL_WIRE  = '#FFFFFF'   # 刀具线框/正常刀具色
COLOR_COLLIDE    = '#FF3333'   # 碰撞高亮 红色


# 静态物体名约定
STATIC_NAME_CAD  = 'cad'
STATIC_NAME_TOOL = 'tool'


class CollisionManager:
    """
    4 类网格（机器人 / CAD / 环境 / 刀具）+ 实时碰撞检测的集中管理器。

    用法::

        mgr = CollisionManager(plotter)

        mgr.register_robot(urdf_path, fk_provider)
        mgr.register_cad(cad_stl_path, T_cad_world)
        mgr.register_env('workbench_1', workbench_stl_path, T_env_world)
        mgr.register_tool(tool_stl_path)

        # 每帧
        mgr.update_robot_joints(q)
        result = mgr.refresh()
        # result = {'robot': {link_2: ['env_workbench_1']},
        #           'static': {'env_workbench_1': ['link_2']}}
    """

    def __init__(self, plotter: 'pv.Plotter', render_engine=None):
        self._plotter = plotter
        self._render_engine = render_engine  # 用于查询工具/环境物体的真实 world transform

        # 4 类可视器（一对一 + 一对多）
        self._robot_viz: Optional[CollisionMeshVisualizer] = None
        self._cad_viz: Optional[CADWireframeVisualizer] = None
        self._env_viz: Dict[str, EnvironmentMeshVisualizer] = {}
        self._tool_viz: Optional[ToolWireframeVisualizer] = None
        self._tool_registration_error: Optional[str] = None

        # 关联数据
        self._checker: Optional[MultiEnvCollisionChecker] = None
        self._fk_provider: Optional[Callable] = None
        self._urdf_path: Optional[str] = None
        self._end_effector_link: Optional[str] = None

        # 机器人 local AABB 缓存（link -> (lmin, lmax)）
        self._robot_local_aabbs: Dict[str, tuple] = {}
        # SceneSnapshot uses copies of these meshes, never the visualizer/FCL objects.
        self._robot_collision_meshes: Dict[str, object] = {}

        # 当前 link 位姿缓存（每帧由 update_robot_joints 写入）
        self._robot_link_poses: Dict[str, np.ndarray] = {}
        self._current_joint_q: Optional[np.ndarray] = None

        # GUI feedback may reuse this main-thread-local service.  It is
        # invalidated whenever collision geometry/scene transforms change;
        # planning and validation workers never receive this object.
        self._live_snapshot: Optional[SceneSnapshot] = None
        self._live_snapshot_service = None
        self._live_snapshot_signature = None

        # CAD / Tool 的 local aabb 缓存（用于更新 world aabb）
        self._cad_local_aabb: Optional[tuple] = None
        self._tool_local_aabb: Optional[tuple] = None

        # 上次的高亮状态（用于重置）
        self._last_robot_highlight: Set[str] = set()
        self._last_static_highlight: Set[str] = set()

        # 状态订阅回调: fn(result_dict) -> None
        self._state_subscribers: List[Callable[[dict], None]] = []

        # 只创建一次的左上角碰撞状态文本；后续通过 set_text 原位更新。
        self._collision_overlay = None
        try:
            self._collision_overlay = self._plotter.add_text(
                "Collision objects: None",
                position="upper_left",
                font_size=11,
                color="#88FF88",
                shadow=True,
                name="collision_status_overlay",
                render=False,
            )
        except Exception as e:
            print(f"[CollisionManager] 左上角碰撞状态文本创建失败: {e}")

        # 视觉 mesh 的原始颜色缓存，用于碰撞结束后精确恢复。
        self._robot_visual_defaults: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # 订阅
    # ------------------------------------------------------------------

    def subscribe_state(self, callback: Callable[[dict], None]) -> None:
        """注册状态订阅回调（refresh 完会通知）"""
        self._state_subscribers.append(callback)

    def _notify_subscribers(self, result: dict) -> None:
        for cb in self._state_subscribers:
            try:
                cb(result)
            except Exception as e:
                print(f"[CollisionManager] 订阅者回调异常: {e}")

    # ------------------------------------------------------------------
    # 注册 / 反注册
    # ------------------------------------------------------------------

    def register_robot(
        self,
        urdf_path: str,
        fk_provider: Callable[[np.ndarray], Dict[str, 'pin.SE3']],
    ) -> None:
        """
        注册机器人（一次性）：构建精检测线框 + fcl 模型。
        """
        if not os.path.exists(urdf_path):
            print(f"[CollisionManager] URDF 不存在: {urdf_path}")
            return

        self._urdf_path = urdf_path
        self._fk_provider = fk_provider
        self._invalidate_live_snapshot()

        # ---- 构建可视器（灰色实线） ----
        self._robot_viz = CollisionMeshVisualizer(
            urdf_path=urdf_path,
            plotter=self._plotter,
            link_fk_provider=fk_provider,
            wireframe_color=COLOR_ROBOT_WIRE,
            render_collision_mesh=False,
        )
        self._cache_robot_visual_defaults()

        # ---- 推导 robot local AABB（从源 collision mesh 计算） ----
        self._robot_local_aabbs = {}
        self._robot_collision_meshes = {}
        collision_infos = parse_urdf_collision_meshes(urdf_path)
        for link_name, mesh_path, pose_info in collision_infos:
            if not mesh_path or not os.path.exists(mesh_path):
                continue
            if link_name not in self._robot_viz._fcl_objects:
                continue
            try:
                from collision.convex_hull import auto_convert_mesh_to_meters
                pv_mesh = auto_convert_mesh_to_meters(mesh_path, pose_info)
                lmin, lmax = _local_aabb_from_pv(pv_mesh)
                self._robot_local_aabbs[link_name] = (lmin, lmax)
                # Keep actor-free source geometry for worker SceneSnapshot copies.
                self._robot_collision_meshes[link_name] = pv_mesh.copy(deep=True)
            except Exception as e:
                print(f"[CollisionManager] 推导 {link_name} local AABB 失败: {e}")

        # ---- 末端 link 识别 ----
        self._end_effector_link = self._detect_end_effector(urdf_path)

        # ---- 构建 checker ----
        # 初始 poses 为单位矩阵
        initial_poses: Dict[str, np.ndarray] = {
            ln: np.eye(4, dtype=np.float64)
            for ln in self._robot_local_aabbs.keys()
        }
        self._robot_link_poses = initial_poses

        self._checker = MultiEnvCollisionChecker(
            robot_fcl_objects=self._robot_viz._fcl_objects,
            robot_local_aabbs=self._robot_local_aabbs,
            robot_link_poses=self._robot_link_poses,
        )

        # 注入 ignore pairs（相邻 + 末端 vs 刀具）
        self._register_ignore_pairs()

        # ---- 立即推一次初始关节位置（解决加载时所有 link 在原点的问题）----
        try:
            import pinocchio as _pin
            _model = _pin.buildModelFromUrdf(urdf_path)
            _nq = _model.nq
            q_zero = np.zeros(_nq, dtype=np.float64)
            self.update_robot_joints(q_zero)
        except Exception as _e:
            print(f"[CollisionManager] 初始关节位置推送失败（不影响后续）: {_e}")

        print(
            f"[CollisionManager] register_robot 完成: "
            f"links={len(self._robot_local_aabbs)}, "
            f"end_effector={self._end_effector_link}"
        )

    def unregister_robot(self) -> None:
        self._restore_robot_visuals()
        if self._robot_viz is not None:
            self._robot_viz.clear()
            self._robot_viz = None
        if self._checker is not None:
            self._checker.clear_static()
        self._checker = None
        self._robot_local_aabbs.clear()
        self._robot_collision_meshes.clear()
        self._robot_link_poses.clear()
        self._current_joint_q = None
        self._end_effector_link = None
        self._urdf_path = None
        self._fk_provider = None
        self._robot_visual_defaults.clear()
        self._invalidate_live_snapshot()

    def register_cad(
        self,
        mesh_path: str = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        注册 CAD 数模（灰色实线，静态）。

        重要：不创建新 mesh，复用 render_engine._cad_model 已加载的 mesh 和 actor。
        调用者必须先调用 re.load_cad_model(...)，再调用本方法。
        """
        re = self._render_engine
        if re is None or getattr(re, '_cad_model', None) is None:
            print("[CollisionManager] register_cad: render_engine 未加载 CAD，请先 re.load_cad_model()")
            return

        # 复用 RenderEngine 已有的 mesh（mm -> m 已转换）和 actor
        cad_mesh = re.get_cad_mesh()
        if cad_mesh is None:
            print("[CollisionManager] register_cad: get_cad_mesh() 为空")
            return

        if self._cad_viz is not None:
            self.unregister_cad()

        self._cad_viz = CADWireframeVisualizer(
            plotter=self._plotter,
            source_actor=re._cad_model,
            preloaded_mesh=cad_mesh,
            transform=transform if transform is not None else np.eye(4),
            wireframe_color=COLOR_CAD_WIRE,
            wireframe_width=2,
            name_prefix='cad',
            render_collision_mesh=False,
        )
        # 缓存 local AABB 用于后续 world AABB 更新
        self._cad_local_aabb = _local_aabb_from_pv(self._cad_viz._pv_mesh)
        # 推入 checker
        self._add_static_to_checker(STATIC_NAME_CAD, self._cad_viz)
        self._invalidate_live_snapshot()
        print(f"[CollisionManager] register_cad 完成（复用 RenderEngine._cad_model）")

    def unregister_cad(self) -> None:
        if self._cad_viz is not None:
            self._cad_viz.clear()
            self._cad_viz = None
        if self._checker is not None:
            self._checker.remove_static(STATIC_NAME_CAD)
        self._cad_local_aabb = None
        self._invalidate_live_snapshot()

    def register_env(
        self,
        name: str,
        mesh_path: str = None,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        注册一个环境物体（灰色实线，静态）。

        重要：不创建新 mesh，复用 render_engine._env_object_actors[name] 已加载的 mesh 和 actor。
        调用者必须先调用 re.load_env_object(...)，再调用本方法。
        """
        re = self._render_engine
        if re is None:
            print("[CollisionManager] register_env: render_engine 未提供")
            return
        env_entry = re._env_object_actors.get(name) if hasattr(re, '_env_object_actors') else None
        if env_entry is None:
            print(f"[CollisionManager] register_env: render_engine 未加载 {name}，请先 re.load_env_object()")
            return

        env_mesh = env_entry.get('mesh')
        env_actor = env_entry.get('actor')
        if env_mesh is None or env_actor is None:
            print(f"[CollisionManager] register_env: {name} 的 mesh/actor 缺失")
            return

        if name in self._env_viz:
            self.unregister_env(name)

        viz = EnvironmentMeshVisualizer(
            plotter=self._plotter,
            source_actor=env_actor,
            preloaded_mesh=env_mesh,
            transform=transform if transform is not None else np.eye(4),
            wireframe_color=COLOR_ENV_WIRE,
            wireframe_width=2,
            name_prefix=f'env_{name}',
            wireframe_only=True,
            render_collision_mesh=False,
        )
        self._env_viz[name] = viz
        self._add_static_to_checker(name, viz)
        self._invalidate_live_snapshot()
        print(f"[CollisionManager] register_env 完成（复用 RenderEngine._env_object_actors[{name}]）")

    def unregister_env(self, name: str) -> None:
        viz = self._env_viz.pop(name, None)
        if viz is not None:
            viz.clear()
        if self._checker is not None:
            self._checker.remove_static(name)
        self._invalidate_live_snapshot()

    def clear_env(self) -> None:
        for name in list(self._env_viz.keys()):
            self.unregister_env(name)

    def register_tool(self, mesh_path: str = None) -> bool:
        """Attach the rendered cutting tool to the GUI-side FCL scene."""
        re = self._render_engine
        source_actor = None if re is None else getattr(re, '_tool_actor', None)
        if source_actor is None:
            self._tool_registration_error = "RenderEngine has no cutting-tool actor"
            print("[CollisionManager] register_tool: RenderEngine has no cutting-tool actor")
            return False
        get_tool_mesh = getattr(re, 'get_tool_mesh', None)
        tool_mesh = get_tool_mesh() if callable(get_tool_mesh) else None
        if tool_mesh is None:
            self._tool_registration_error = "RenderEngine has no cutting-tool mesh"
            print("[CollisionManager] register_tool: RenderEngine has no cutting-tool mesh")
            return False

        if self._tool_viz is not None:
            self.unregister_tool()
        try:
            tool_viz = ToolWireframeVisualizer(
                plotter=self._plotter,
                source_actor=source_actor,
                preloaded_mesh=tool_mesh,
                wireframe_color=COLOR_TOOL_WIRE,
                wireframe_width=2,
                name_prefix='tool',
                render_collision_mesh=False,
            )
            tool_viz.sync_from_source()
            tool_local_aabb = _local_aabb_from_pv(tool_viz._pv_mesh)
            self._tool_viz = tool_viz
            self._tool_local_aabb = tool_local_aabb
            self._add_static_to_checker(STATIC_NAME_TOOL, tool_viz)
            self._register_ignore_pairs()
            self._invalidate_live_snapshot()
        except Exception as exc:
            if self._tool_viz is not None:
                try:
                    self._tool_viz.clear()
                except Exception:
                    pass
            if self._checker is not None:
                try:
                    self._checker.remove_static(STATIC_NAME_TOOL)
                except Exception:
                    pass
            self._tool_viz = None
            self._tool_local_aabb = None
            self._tool_registration_error = str(exc)
            print(f"[CollisionManager] register_tool failed: {exc}")
            return False

        self._tool_registration_error = None
        print("[CollisionManager] register_tool completed (RenderEngine tool source attached)")
        return True

    def ensure_attached_tool_registered(self) -> bool:
        """Synchronize an attached FCL tool from the render-owned tool actor."""
        re = self._render_engine
        source_actor = None if re is None else getattr(re, '_tool_actor', None)
        if source_actor is None:
            self._tool_registration_error = "RenderEngine has no cutting-tool actor"
            return False
        get_tool_mesh = getattr(re, 'get_tool_mesh', None)
        if not callable(get_tool_mesh) or get_tool_mesh() is None:
            self._tool_registration_error = "RenderEngine has no cutting-tool mesh"
            return False
        current = self._tool_viz
        if (
            current is not None
            and getattr(current, '_source_actor', None) is source_actor
            and getattr(current, '_pv_mesh', None) is not None
        ):
            self._tool_registration_error = None
            return True
        try:
            return bool(self.register_tool())
        except Exception as exc:
            self._tool_registration_error = str(exc)
            return False

    def unregister_tool(self) -> None:
        if self._tool_viz is not None:
            self._tool_viz.clear()
            self._tool_viz = None
        if self._checker is not None:
            self._checker.remove_static(STATIC_NAME_TOOL)
        self._tool_local_aabb = None
        self._invalidate_live_snapshot()

    def unregister_all(self) -> None:
        """一键清空所有（机器人/CAD/环境/刀具）"""
        self.unregister_cad()
        self.clear_env()
        self.unregister_tool()
        self.unregister_robot()
        self._last_robot_highlight.clear()
        self._last_static_highlight.clear()

    def set_robot_base_transform(self, T: np.ndarray) -> None:
        """
        设置机器人基座变换（世界坐标系）。

        在 RenderEngine.transform_robot() 被调用后，必须同步调用此方法，
        以确保碰撞网格线的 base_transform 与 visual mesh 一致。
        同时同步所有静态物体的 world AABB（因为机器人移动后 world 坐标变了）。
        """
        if self._robot_viz is not None:
            self._robot_viz.set_base_transform(T)
        self._invalidate_live_snapshot()
        self.sync_static_world_aabbs()

    # ------------------------------------------------------------------
    # Immutable planning / validation scene capture
    # ------------------------------------------------------------------

    def _invalidate_live_snapshot(self) -> None:
        self._live_snapshot = None
        self._live_snapshot_service = None
        self._live_snapshot_signature = None

    def invalidate_scene_snapshot_cache(self) -> None:
        """Drop the GUI-local snapshot after an external tool/TCP mutation.

        Tool TCP edits change the attached body's mount transform but need not
        change a CAD/environment transform.  Keeping this as a narrow public
        method prevents the GUI real-time checker from reusing a snapshot with
        a stale attached-tool transform, while worker snapshots are still
        always rebuilt per operation.
        """
        self._invalidate_live_snapshot()

    def _live_scene_signature(self):
        """Cheap transform-only cache key; meshes invalidate on registration."""
        if self._robot_viz is None:
            return None

        def frozen_bytes(matrix) -> bytes:
            return np.ascontiguousarray(np.asarray(matrix, dtype=np.float64)).tobytes()

        entries = [("robot_base", frozen_bytes(self._robot_viz._base_transform))]
        if self._cad_viz is not None:
            entries.append((STATIC_NAME_CAD, frozen_bytes(self._cad_viz.get_world_transform())))
        for name, viz in sorted(self._env_viz.items()):
            entries.append((name, frozen_bytes(viz.get_world_transform())))
        return tuple(entries)

    @staticmethod
    def is_result_unsafe(result: dict) -> bool:
        """Treat collision-service failures as unsafe, not as empty hits."""
        return (
            not bool(result.get("valid", True))
            or bool(result.get("robot"))
            or bool(result.get("static"))
        )

    def _state_validity_to_result(self, validity) -> dict:
        """Adapt structured D results to the legacy GUI highlight payload."""
        result = {
            "robot": {},
            "static": {},
            "pairs": [tuple(pair) for pair in validity.collision_pairs],
            "valid": bool(validity.valid),
            "error_code": validity.error_code,
            "detail": validity.detail,
            "scene_hash": validity.scene_hash,
        }
        if self._live_snapshot is None:
            return result
        robot_names = {item.name for item in self._live_snapshot.robot_links}
        static_names = {
            item.name for item in self._live_snapshot.static_objects
        } | {item.name for item in self._live_snapshot.attached_bodies}
        for first, second in validity.collision_pairs:
            if first in robot_names and second in robot_names:
                result["robot"].setdefault(first, []).append(f"Robot.{second}")
                result["robot"].setdefault(second, []).append(f"Robot.{first}")
            elif first in robot_names and second in static_names:
                result["robot"].setdefault(first, []).append(second)
                result["static"].setdefault(second, []).append(first)
            elif second in robot_names and first in static_names:
                result["robot"].setdefault(second, []).append(first)
                result["static"].setdefault(first, []).append(second)
            else:
                # tool-CAD and tool-environment pairs have no robot-link
                # member, but both visible objects must still highlight.
                result["static"].setdefault(first, []).append(second)
                result["static"].setdefault(second, []).append(first)
        return result

    def create_scene_snapshot(
        self,
        q: Optional[np.ndarray] = None,
        *,
        scene_version: int = 0,
        contact_rules=(),
        require_attached_tool: bool = True,
    ) -> SceneSnapshot:
        """Freeze the current GUI scene into actor-free collision data.

        This is deliberately a main-thread operation.  It may synchronize the
        live render once to capture the tool mount matrix, but the returned
        object contains only copied arrays and scalar metadata.  Worker
        threads must reconstruct their own FCL objects from this snapshot.
        """
        if (
            self._robot_viz is None
            or self._fk_provider is None
            or not self._urdf_path
            or not self._robot_collision_meshes
        ):
            raise SceneSnapshotBuildError(
                "scene_not_ready",
                "robot collision meshes and FK must be registered before capture",
            )

        if q is None:
            if self._current_joint_q is None:
                raise SceneSnapshotBuildError(
                    "missing_joint_state",
                    "capture requires the current robot joint configuration",
                )
            q_array = self._current_joint_q.copy()
        else:
            q_array = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_array.size == 0 or not np.all(np.isfinite(q_array)):
            raise SceneSnapshotBuildError(
                "invalid_joint_state",
                "capture joint configuration must be finite and non-empty",
            )

        try:
            # Capture the source tool actor at exactly the same q as the FK
            # used below.  This is the only actor interaction in the API.
            if self._render_engine is not None:
                update_render = getattr(self._render_engine, "update_robot_joints", None)
                if callable(update_render):
                    update_render(q_array, render=False)
            self.update_robot_joints(q_array)
            link_fk = self._fk_provider(q_array.copy())
        except Exception as exc:
            raise SceneSnapshotBuildError("scene_sync_failed", str(exc)) from exc

        robot_links: list[RobotLinkSpec] = []
        for link_name in sorted(self._robot_viz._fcl_objects):
            mesh = self._robot_collision_meshes.get(link_name)
            pose_info = self._robot_viz._local_poses.get(link_name)
            if mesh is None or pose_info is None:
                raise SceneSnapshotBuildError(
                    "robot_mesh_missing",
                    f"collision geometry or local pose is missing for {link_name}",
                )
            try:
                local_transform = make_transform(
                    pose_info.get("xyz", [0.0, 0.0, 0.0]),
                    pose_info.get("rpy", [0.0, 0.0, 0.0]),
                    pose_info.get("scale", [1.0, 1.0, 1.0]),
                )
                robot_links.append(
                    RobotLinkSpec(
                        name=link_name,
                        geometry=MeshGeometry.from_pyvista(mesh),
                        local_transform=local_transform,
                    )
                )
            except Exception as exc:
                raise SceneSnapshotBuildError(
                    "robot_mesh_invalid", f"{link_name}: {exc}"
                ) from exc

        static_specs: list[StaticCollisionObjectSpec] = []

        def capture_static(name: str, viz) -> None:
            try:
                static_specs.append(
                    StaticCollisionObjectSpec(
                        name=name,
                        geometry=MeshGeometry.from_pyvista(viz._pv_mesh),
                        world_transform=viz.get_world_transform(),
                    )
                )
            except Exception as exc:
                raise SceneSnapshotBuildError(
                    "static_mesh_invalid", f"{name}: {exc}"
                ) from exc

        if self._cad_viz is not None:
            capture_static(STATIC_NAME_CAD, self._cad_viz)
        for name, viz in sorted(self._env_viz.items()):
            capture_static(name, viz)

        attached_bodies: list[AttachedBodySpec] = []
        tool_registered = self.ensure_attached_tool_registered()
        if tool_registered:
            self._tool_viz.sync_from_source()
            parent_link = self._end_effector_link
            robot_by_name = {item.name: item for item in robot_links}
            if parent_link not in robot_by_name:
                raise SceneSnapshotBuildError(
                    "tool_parent_missing",
                    "the detected end-effector collision link is unavailable",
                )
            if parent_link not in link_fk:
                raise SceneSnapshotBuildError(
                    "tool_parent_fk_missing",
                    f"FK did not return the end-effector link {parent_link}",
                )
            try:
                parent_transform = (
                    self._robot_viz._base_transform
                    @ np.asarray(
                        getattr(link_fk[parent_link], "homogeneous", link_fk[parent_link]),
                        dtype=np.float64,
                    )
                    @ robot_by_name[parent_link].local_transform
                )
                tool_world = self._tool_viz.get_world_transform()
                mount_transform = np.linalg.inv(parent_transform) @ tool_world
                attached_bodies.append(
                    AttachedBodySpec(
                        name=STATIC_NAME_TOOL,
                        parent_link=parent_link,
                        geometry=MeshGeometry.from_pyvista(self._tool_viz._pv_mesh),
                        mount_transform=mount_transform,
                    )
                )
            except Exception as exc:
                raise SceneSnapshotBuildError("tool_attachment_invalid", str(exc)) from exc
        elif require_attached_tool:
            raise SceneSnapshotBuildError(
                "tool_not_registered",
                self._tool_registration_error
                or "an attached tool collision mesh is required for safe planning",
            )

        ignored_pairs = tuple(
            tuple(sorted(pair))
            for pair in getattr(self._robot_viz, "_adjacent_pairs", set())
            if len(pair) == 2
        )
        return SceneSnapshot(
            robot_urdf_path=self._urdf_path,
            robot_base_transform=self._robot_viz._base_transform,
            robot_links=tuple(robot_links),
            static_objects=tuple(static_specs),
            attached_bodies=tuple(attached_bodies),
            contact_rules=tuple(contact_rules),
            ignored_link_pairs=ignored_pairs,
            scene_version=int(scene_version),
        )

    def build_snapshot_collision_service(self, q: Optional[np.ndarray] = None, **kwargs):
        """Return a fresh local-FCL service for one caller, never GUI FCL data."""
        from collision.state_validity import SnapshotCollisionService

        snapshot = self.create_scene_snapshot(q, **kwargs)
        return snapshot, SnapshotCollisionService(snapshot, self._fk_provider)

    # ------------------------------------------------------------------
    # 每帧驱动
    # ------------------------------------------------------------------

    def update_robot_joints(self, q: np.ndarray) -> None:
        """
        每帧调用：
            1. 同步机器人线框 + fcl
            2. 末端 FK → 通过 render_engine._tool_actor 推 transform
            3. 从 source actor 拉取 CAD/env/tool 的 wireframe user_matrix
            4. 同步所有静态物体的 world AABB（如果位姿被外部修改）
            5. 写入 robot link world pose 给 checker
        """
        if self._robot_viz is None or self._fk_provider is None:
            return

        q_array = np.asarray(q, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(q_array)):
            raise ValueError("joint configuration contains non-finite values")
        self._current_joint_q = q_array.copy()

        # 1. 机器人
        self._robot_viz.update_joints(q_array)
        # 直接读取计算缓存；不依赖已取消创建的碰撞网格 actor。
        self._robot_link_poses.update(self._robot_viz.get_link_transforms())

        # 2. 刀具：从 render_engine._tool_actor.user_matrix 拉取（即末端 FK）
        #    wireframe 与 source actor 共享 user_matrix，自然跟随机器人
        if self._tool_viz is not None:
            self._tool_viz.sync_from_source()
            # 同步刀具 world AABB（用 source actor 的 user_matrix）
            if self._tool_local_aabb is not None and self._checker is not None:
                try:
                    T_tool = np.asarray(
                        self._tool_viz._source_actor.user_matrix, dtype=np.float64
                    )
                    lmin, lmax = self._tool_local_aabb
                    wmin, wmax = _world_aabb_from_local(lmin, lmax, T_tool)
                    self._checker.update_static_world_aabb(
                        STATIC_NAME_TOOL, wmin, wmax
                    )
                except Exception:
                    pass

        # 3. CAD/env wireframe：跟随 source actor 的 user_matrix
        if self._cad_viz is not None:
            self._cad_viz.sync_from_source()
        for viz in self._env_viz.values():
            viz.sync_from_source()

        # 4. 静态物体 world AABB 同步（CAD / env）
        self.sync_static_world_aabbs()

    def sync_static_world_aabbs(self) -> None:
        """
        手动同步所有静态物体（CAD + 环境）的 world AABB。

        在外部修改了以下内容后必须调用：
            - 机器人基座变换（影响所有 world 坐标）
            - CAD / 工件变换
            - 环境物体变换

        该方法在 update_robot_joints() 末尾自动调用，
        也可在独立修改 transform 后手动触发。
        """
        if self._checker is None:
            return
        # CAD
        if self._cad_viz is not None and self._cad_local_aabb is not None:
            T_cad = self._cad_viz.get_world_transform()
            wmin, wmax = _world_aabb_from_local(
                self._cad_local_aabb[0], self._cad_local_aabb[1], T_cad
            )
            self._checker.update_static_world_aabb(STATIC_NAME_CAD, wmin, wmax)
        # 环境物体
        for name, viz in self._env_viz.items():
            lmin, lmax = _local_aabb_from_pv(viz._pv_mesh)
            T_env = viz.get_world_transform()
            wmin, wmax = _world_aabb_from_local(lmin, lmax, T_env)
            self._checker.update_static_world_aabb(name, wmin, wmax)

    def update_env_object_transform(self, name: str, T: np.ndarray) -> None:
        """
        更新单个环境物体的世界变换（同时更新 actor user_matrix 和 world AABB）。

        在 RenderEngine.update_env_object_transform() 被调用后，同步调用此方法。
        """
        if name not in self._env_viz:
            return
        viz = self._env_viz[name]
        viz.set_world_transform(T)
        if self._checker is not None:
            lmin, lmax = _local_aabb_from_pv(viz._pv_mesh)
            wmin, wmax = _world_aabb_from_local(lmin, lmax, T)
            self._checker.update_static_world_aabb(name, wmin, wmax)
        self._invalidate_live_snapshot()

    def update_cad_transform(self, T: np.ndarray) -> None:
        """
        更新 CAD 数模的世界变换（同时更新 actor user_matrix 和 world AABB）。

        在 RenderEngine.transform_workpiece() 被调用后，同步调用此方法。
        """
        if self._cad_viz is None:
            return
        self._cad_viz.set_world_transform(T)
        if self._checker is not None and self._cad_local_aabb is not None:
            wmin, wmax = _world_aabb_from_local(
                self._cad_local_aabb[0], self._cad_local_aabb[1], T
            )
            self._checker.update_static_world_aabb(STATIC_NAME_CAD, wmin, wmax)
        self._invalidate_live_snapshot()

    def refresh(self) -> dict:
        """
        跑一次碰撞检测 + 应用红色高亮 + 通知订阅者。

        返回::
            {
              'robot': {link_name: [static_name, ...]},
              'static': {static_name: [link_name, ...]},
            }
        """
        if self._current_joint_q is None:
            result = {
                'robot': {},
                'static': {},
                'pairs': [],
                'valid': False,
                'error_code': 'missing_joint_state',
                'detail': 'no finite robot joint state has been supplied',
            }
        else:
            result = self.check_configuration(self._current_joint_q)
        self._apply_highlights(result)
        self._notify_subscribers(result)
        return result

    def check_configuration(self, q: np.ndarray) -> dict:
        """Exact, fail-closed GUI check built from the immutable D contract.

        This service is cache-local to the GUI thread.  Background workers use
        ``SceneSnapshot`` only and build a separate service in their own
        thread, so no actor or GUI FCL object leaks across the boundary.
        """
        q_array = np.asarray(q, dtype=np.float64).reshape(-1)
        if q_array.size == 0 or not np.all(np.isfinite(q_array)):
            return {
                'robot': {},
                'static': {},
                'pairs': [],
                'valid': False,
                'error_code': 'invalid_configuration',
                'detail': 'joint configuration must be finite and non-empty',
            }
        try:
            signature = self._live_scene_signature()
            if (
                self._live_snapshot_service is None
                or signature != self._live_snapshot_signature
            ):
                snapshot, service = self.build_snapshot_collision_service(q_array)
                self._live_snapshot = snapshot
                self._live_snapshot_service = service
                self._live_snapshot_signature = self._live_scene_signature()
            validity = self._live_snapshot_service.check_configuration(q_array)
            return self._state_validity_to_result(validity)
        except SceneSnapshotBuildError as exc:
            return {
                'robot': {},
                'static': {},
                'pairs': [],
                'valid': False,
                'error_code': exc.code,
                'detail': exc.detail,
            }
        except Exception as exc:
            return {
                'robot': {},
                'static': {},
                'pairs': [],
                'valid': False,
                'error_code': 'check_failed',
                'detail': str(exc),
            }

    def evaluate_configuration(self, q: np.ndarray) -> dict:
        """Atomically check one trajectory sample and publish its feedback."""
        result = self.check_configuration(q)
        self._apply_highlights(result)
        self._notify_subscribers(result)
        return result

    # ------------------------------------------------------------------
    # 高亮分发
    # ------------------------------------------------------------------

    def _apply_highlights(self, result: dict) -> None:
        """根据 result 应用/恢复红色高亮"""
        new_robot = set(result.get('robot', {}).keys())
        new_static = set(result.get('static', {}).keys())

        # ---- 机器人视觉 link 高亮（碰撞 mesh 不参与渲染） ----
        visual_actors = self._robot_visual_actors()
        if visual_actors:
            # 先把所有上轮的高亮恢复
            for link_name in self._last_robot_highlight - new_robot:
                try:
                    visual_actors[link_name].prop.color = (
                        self._robot_visual_defaults[link_name]
                    )
                except Exception:
                    pass
            # 红色高亮新增的
            for link_name in new_robot:
                try:
                    visual_actors[link_name].prop.color = COLOR_COLLIDE
                except Exception:
                    pass

        # ---- 静态物体高亮（CAD/env/tool） ----
        all_static_viz = self._collect_static_visualizers()
        # 先恢复
        for name in self._last_static_highlight - new_static:
            viz = all_static_viz.get(name)
            if viz is not None:
                viz.reset_color()
        # 变红
        for name in new_static:
            viz = all_static_viz.get(name)
            if viz is not None:
                viz.apply_wireframe_highlight(True)

        self._last_robot_highlight = set(new_robot)
        self._last_static_highlight = set(new_static)
        self._update_collision_overlay(result)

    @staticmethod
    def _merge_self_collisions(result: dict, pairs) -> None:
        """把机器人自碰撞对合并进统一 result，供 UI/高亮共同消费。"""
        robot = result.setdefault('robot', {})
        result.setdefault('static', {})
        result['self'] = list(pairs)
        for link_a, link_b in pairs:
            target_b = f"Robot.{link_b}"
            target_a = f"Robot.{link_a}"
            robot.setdefault(link_a, [])
            robot.setdefault(link_b, [])
            if target_b not in robot[link_a]:
                robot[link_a].append(target_b)
            if target_a not in robot[link_b]:
                robot[link_b].append(target_a)

    def _robot_visual_actors(self) -> Dict[str, object]:
        robot = getattr(self._render_engine, '_robot', None)
        actors = getattr(robot, '_actors', None)
        return actors if isinstance(actors, dict) else {}

    def _cache_robot_visual_defaults(self) -> None:
        self._robot_visual_defaults.clear()
        for link_name, actor in self._robot_visual_actors().items():
            try:
                self._robot_visual_defaults[link_name] = actor.prop.color
            except Exception:
                pass

    def _restore_robot_visuals(self) -> None:
        for link_name, actor in self._robot_visual_actors().items():
            if link_name not in self._robot_visual_defaults:
                continue
            try:
                actor.prop.color = self._robot_visual_defaults[link_name]
            except Exception:
                pass

    def _update_collision_overlay(self, result: dict) -> None:
        if self._collision_overlay is None:
            return

        pairs = []
        seen = set()
        for pair in result.get("pairs") or ():
            if len(pair) != 2:
                continue
            key = tuple(sorted((str(pair[0]), str(pair[1]))))
            if key not in seen:
                seen.add(key)
                pairs.append(" <-> ".join(key))
        for link_name, targets in (result.get('robot') or {}).items():
            for target in targets:
                if str(target).startswith("Robot."):
                    other = str(target).split(".", 1)[1]
                    key = tuple(sorted((f"Robot.{link_name}", f"Robot.{other}")))
                else:
                    key = (f"Robot.{link_name}", str(target))
                if key not in seen:
                    seen.add(key)
                    pairs.append(" <-> ".join(key))

        if not bool(result.get("valid", True)) and not pairs:
            error_code = result.get("error_code") or "collision_service_unavailable"
            pairs.append(f"CollisionService <-> {error_code}")

        text = (
            "Collision objects: None"
            if not pairs
            else "Collision objects:\n" + "\n".join(pairs)
        )
        try:
            self._collision_overlay.set_text("upper_left", text)
            self._collision_overlay.prop.color = (
                COLOR_COLLIDE if pairs else "#88FF88"
            )
        except Exception:
            pass

    def _collect_static_visualizers(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if self._cad_viz is not None:
            out[STATIC_NAME_CAD] = self._cad_viz
        if self._tool_viz is not None:
            out[STATIC_NAME_TOOL] = self._tool_viz
        for name, viz in self._env_viz.items():
            out[name] = viz
        return out

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _detect_end_effector(self, urdf_path: str) -> Optional[str]:
        """
        倒序扫描 URDF <link> 列表识别末端 link。

        跳过:
          - 空名
          - 保留字 'universe' / 'world'
          - 关键字 'base' / 'world' / 'mount' / 'root'（任一含此关键字即跳过）
          - 不在 Pinocchio model.names 中的伪 link

        失败返回 None（调用方不阻塞，log 提示）。
        """
        if not PINOCCHIO_AVAILABLE:
            print("[CollisionManager] pinocchio 未安装，无法识别末端 link")
            return None

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            link_names = [n.get('name', '') for n in root.findall('./link')]

            RESERVED = {'universe', 'world'}
            BASE_KEYWORDS = ('base', 'world', 'mount', 'root')

            model = pin.buildModelFromUrdf(urdf_path)
            # Pinocchio ``model.names`` contains joint names, whereas the
            # collision FK provider is keyed by URDF link names.  A robot can
            # therefore have perfectly valid collision links that are absent
            # from ``model.names`` (for example the ROKAE resources).
            model_links = set(model.names)
            collision_link_names = set(
                getattr(self._robot_viz, "_fcl_objects", {}).keys()
            )

            for link_name in reversed(link_names):
                if not link_name:
                    continue
                if link_name in RESERVED:
                    continue
                if any(kw in link_name.lower() for kw in BASE_KEYWORDS):
                    continue
                if (
                    link_name not in model_links
                    and link_name not in collision_link_names
                ):
                    continue
                return link_name

            print(f"[CollisionManager] 未能识别末端 link（urdf={urdf_path}）")
            return None
        except Exception as e:
            print(f"[CollisionManager] 末端 link 识别异常: {e}")
            return None

    def _register_ignore_pairs(self) -> None:
        """
        注入 ignore_pairs:
          - 相邻连杆（来自 robot_viz._adjacent_pairs）
          - 末端 link vs 刀具（无论末端 link 命名如何）

        由于末端 link 名在不同 URDF 中变化（link6 / link_6 / ee_link 等），
        这里额外支持"所有 link_6 / link6"类型自动忽略对刀具的碰撞。
        """
        if self._checker is None:
            return

        ignore: Set[frozenset] = set()

        # 相邻连杆
        if self._robot_viz is not None:
            for pair in self._robot_viz._adjacent_pairs:
                ignore.add(pair)

        # 末端 link vs tool
        # 1. 优先使用 _end_effector_link（明确识别的末端）
        if self._end_effector_link is not None:
            ignore.add(frozenset({self._end_effector_link, STATIC_NAME_TOOL}))

        # 2. 兜底：所有名为 link_6 / link6 / 末端的 link 都忽略对刀具的碰撞
        #    （覆盖识别失败的情况）
        if self._robot_viz is not None:
            EE_PATTERNS = ('link_6', 'link6', 'ee_link', 'tool0', 'flange', 'end_effector')
            for link_name in self._robot_viz._fcl_objects.keys():
                ln = link_name.lower()
                if any(pat in ln for pat in EE_PATTERNS):
                    ignore.add(frozenset({link_name, STATIC_NAME_TOOL}))

        self._checker.set_ignore_pairs(ignore)

    def _add_static_to_checker(
        self,
        name: str,
        viz: EnvironmentMeshVisualizer,
    ) -> None:
        """从 viz 的 fcl obj + world transform 注册到 checker"""
        if self._checker is None:
            return
        fcl_obj = viz.get_fcl_object()
        if fcl_obj is None:
            print(f"[CollisionManager] {name} fcl 模型为空，跳过")
            return
        lmin, lmax = _local_aabb_from_pv(viz._pv_mesh)
        T = viz.get_world_transform()
        wmin, wmax = _world_aabb_from_local(lmin, lmax, T)
        self._checker.add_static(name, fcl_obj, wmin, wmax)
