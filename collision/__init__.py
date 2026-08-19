"""
collision package - 碰撞检测与轨迹重规划

当前模块状态：
  - convex_hull.py: 可用（凸包构建与 PyVista 可视化）
  - robot_env_collision.py: 机器人↔环境碰撞检测（多静态物体）
  - collision_manager.py: 4 类网格集中管理 + 实时检测
  - 粗检测/精检测/轨迹重规划: 尚未实现

已导出公共 API::

    from collision import (
        parse_urdf_collision_meshes,
        build_convex_hull,
        cache_convex_hull,
        build_link_fk_provider,
        ConvexHullVisualizer,
        CollisionMeshVisualizer,
        EnvironmentMeshVisualizer,
        CADWireframeVisualizer,
        ToolWireframeVisualizer,
        WireframeVisualizerBase,
        RobotEnvCollisionChecker,
        MultiEnvCollisionChecker,
        CollisionManager,
        build_checker_from_visualizers,
        # 单位制处理
        detect_mesh_unit_system,
        detect_urdf_unit_system,
        auto_convert_mesh_to_meters,
        # 颜色常量
        COLOR_ROBOT_WIRE,
        COLOR_CAD_WIRE,
        COLOR_ENV_WIRE,
        COLOR_TOOL_WIRE,
        COLOR_COLLIDE,
    )
"""

from collision.convex_hull import (
    parse_urdf_collision_meshes,
    build_convex_hull,
    cache_convex_hull,
    build_link_fk_provider,
    CollisionMeshVisualizer,
    ConvexHullVisualizer,
    EnvironmentMeshVisualizer,
    CADWireframeVisualizer,
    ToolWireframeVisualizer,
    WireframeVisualizerBase,
    detect_mesh_unit_system,
    detect_urdf_unit_system,
    auto_convert_mesh_to_meters,
)
from collision.robot_env_collision import (
    RobotEnvCollisionChecker,
    MultiEnvCollisionChecker,
    build_checker_from_visualizers,
)
from collision.collision_manager import (
    CollisionManager,
    COLOR_ROBOT_WIRE,
    COLOR_CAD_WIRE,
    COLOR_ENV_WIRE,
    COLOR_TOOL_WIRE,
    COLOR_COLLIDE,
)
from collision.distance_field import (
    DistanceQueryResult,
    EnvironmentDistanceField,
)
from collision.link_proxy import (
    LinkProxy,
    LinkProxyModel,
)
from collision.collision_distance_service import (
    CollisionDistanceService,
    MinDistanceResult,
    TrajectoryCollisionReport,
)
from collision.scene_snapshot import (
    AttachedBodySpec,
    ContactRule,
    MeshGeometry,
    RobotLinkSpec,
    SceneSnapshot,
    SceneSnapshotBuildError,
    StaticCollisionObjectSpec,
)
from collision.state_validity import (
    EdgeValidity,
    SnapshotCollisionService,
    StateValidity,
)
from collision.sdf_path_planner import (
    plan_sdf_clearance_path,
    resample_polyline,
)

__all__ = [
    'parse_urdf_collision_meshes',
    'build_convex_hull',
    'cache_convex_hull',
    'build_link_fk_provider',
    'CollisionMeshVisualizer',
    'ConvexHullVisualizer',
    'EnvironmentMeshVisualizer',
    'CADWireframeVisualizer',
    'ToolWireframeVisualizer',
    'WireframeVisualizerBase',
    'RobotEnvCollisionChecker',
    'MultiEnvCollisionChecker',
    'CollisionManager',
    'build_checker_from_visualizers',
    # 单位制处理
    'detect_mesh_unit_system',
    'detect_urdf_unit_system',
    'auto_convert_mesh_to_meters',
    # 颜色常量
    'COLOR_ROBOT_WIRE',
    'COLOR_CAD_WIRE',
    'COLOR_ENV_WIRE',
    'COLOR_TOOL_WIRE',
    'COLOR_COLLIDE',
    'DistanceQueryResult',
    'EnvironmentDistanceField',
    'LinkProxy',
    'LinkProxyModel',
    'CollisionDistanceService',
    'MinDistanceResult',
    'TrajectoryCollisionReport',
    'AttachedBodySpec',
    'ContactRule',
    'MeshGeometry',
    'RobotLinkSpec',
    'SceneSnapshot',
    'SceneSnapshotBuildError',
    'StaticCollisionObjectSpec',
    'EdgeValidity',
    'SnapshotCollisionService',
    'StateValidity',
    'plan_sdf_clearance_path',
    'resample_polyline',
]
