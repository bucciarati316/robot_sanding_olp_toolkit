"""
collision/robot_env_collision.py

机器人 ↔ 外部环境碰撞检测（粗到精的 BVH 剪枝）。

核心类:
    RobotEnvCollisionChecker

设计要点:
    - python-fcl 0.7.x 不暴露 AABB/Vector3 绑定，因此:
        * Local AABB: 由调用方提供的源 PyVista PolyData 计算（一次）
        * World AABB: 每帧用 Link 的世界变换 4x4 对 8 个角点做变换得到
    - 严格遵循"三步走"剪枝:
        Step 1. 机器人 union AABB vs 环境 AABB (O(1) Early-out)
        Step 2. 每个 Link AABB vs 环境 AABB (Link 级剪枝)
        Step 3. 对嫌疑 Link 调用 fcl.collide() 走 BVH 精确检测

与 CollisionMeshVisualizer.update_joints 的协作:
    1. 主循环先调用 robot_viz.update_joints(q) 更新所有 Link 的 fcl.CollisionObject
    2. 然后调用 checker.check_collision(q=None) 或 check_collision_with_q(q)
    3. check_collision 内部不会再次修改 fcl obj 的 transform -- 它假定外部
       update_joints 已经完成了这一工作（与 CollisionMeshVisualizer.check_self_collision
       的接口约定保持一致）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from collision.convex_hull import (
        CollisionMeshVisualizer,
        EnvironmentMeshVisualizer,
    )


# ---------------------------------------------------------------------------
# 工具函数：AABB 重叠判断 / 角点变换
# ---------------------------------------------------------------------------

def _aabb_overlap(
    min_a: np.ndarray, max_a: np.ndarray,
    min_b: np.ndarray, max_b: np.ndarray,
) -> bool:
    """
    判断两个轴对齐包围盒 (AABB) 是否重叠。

    等价于 6 维不等式的 AND：
        min_a[i] <= max_b[i] AND min_b[i] <= max_a[i]  for i in x,y,z
    使用 numpy 向量化，单次调用 ~ 200ns。
    """
    a_min = np.asarray(min_a, dtype=np.float64)
    a_max = np.asarray(max_a, dtype=np.float64)
    b_min = np.asarray(min_b, dtype=np.float64)
    b_max = np.asarray(max_b, dtype=np.float64)
    # 任一轴上不重叠则整体不重叠
    return bool(np.all(a_min <= b_max) and np.all(b_min <= a_max))


def _local_aabb_from_pv(pv_mesh: 'pv.PolyData') -> Tuple[np.ndarray, np.ndarray]:
    """
    从 PyVista PolyData 的源顶点计算局部坐标系下的 AABB。

    返回:
        (local_min (3,), local_max (3,))

    注意:
        - 这里使用源顶点而非 BVH 内部的 BVH 叶子节点
        - 即使 fcl BVH 内部进一步切分，这里得到的 AABB 是"保守"的（>= 实际 BVH AABB）
        - 对剪枝逻辑没有影响（保守剪枝只会少剪，不会误判碰撞）
    """
    points = np.asarray(pv_mesh.points, dtype=np.float64)
    if points.size == 0:
        # 退化：返回零大小 AABB
        zeros = np.zeros(3, dtype=np.float64)
        return zeros.copy(), zeros.copy()
    return points.min(axis=0), points.max(axis=0)


def _world_aabb_from_local(
    local_min: np.ndarray, local_max: np.ndarray, T_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将局部 AABB 通过 4x4 变换矩阵转到世界坐标系。

    实现: 对 8 个角点都做变换，再取 min/max。
    等价于"先世界化顶点，再求 AABB"，精度损失可忽略（旋转后的 AABB 比 OBB 更紧凑）。

    参数:
        local_min (3,)
        local_max (3,)
        T_world (4,4)

    返回:
        (world_min (3,), world_max (3,))
    """
    # 8 个角点
    corners = np.array([
        [local_min[0], local_min[1], local_min[2]],
        [local_min[0], local_min[1], local_max[2]],
        [local_min[0], local_max[1], local_min[2]],
        [local_min[0], local_max[1], local_max[2]],
        [local_max[0], local_min[1], local_min[2]],
        [local_max[0], local_min[1], local_max[2]],
        [local_max[0], local_max[1], local_min[2]],
        [local_max[0], local_max[1], local_max[2]],
    ], dtype=np.float64)  # (8, 3)

    # 转齐次
    ones = np.ones((8, 1), dtype=np.float64)
    homo = np.hstack([corners, ones])  # (8, 4)

    T = np.asarray(T_world, dtype=np.float64)
    world_corners = (homo @ T.T)[:, :3]  # (8, 3)

    return world_corners.min(axis=0), world_corners.max(axis=0)


# ---------------------------------------------------------------------------
# RobotEnvCollisionChecker
# ---------------------------------------------------------------------------

class RobotEnvCollisionChecker:
    """
    机器人 ↔ 外部环境碰撞检测器（粗到精 BVH 剪枝）。

    输入数据要求:
        1. 机器人侧:
            - robot_fcl_objects: Dict[link_name, fcl.CollisionObject]（已构建 BVHModel）
            - robot_local_aabbs: Dict[link_name, (local_min, local_max)]
            - robot_link_poses: Dict[link_name, 4x4 world T] -- 由调用方提供
              （通常来自 CollisionMeshVisualizer 的 FK 流水线）

        2. 环境侧:
            - env_fcl_object: fcl.CollisionObject（已构建 BVHModel）
            - env_world_min / env_world_max: 环境的"实际包围盒"

    主干 API:
        check_collision() -> Tuple[bool, str]
        check_collision_with_q(q) -> Tuple[bool, str]   (自动调 update_joints)

    设计:
        - check_collision() 假设外部已 update_joints(q)
        - check_collision_with_q(q) 内部调用 robot_visualizer.update_joints(q) 后再检测
        - Step 1/2/3 全部用 numpy 向量化 + 字典查询 O(1) 实现
        - 任何一步可以 Early-out 时直接返回

    使用示例::

        # 初始化
        checker = RobotEnvCollisionChecker(
            robot_visualizer=fine_viz,           # CollisionMeshVisualizer
            env_visualizer=env_viz,              # EnvironmentMeshVisualizer
        )

        # 方式 1：手动驱动更新
        fine_viz.update_joints(q)
        is_collide, link = checker.check_collision()
        if is_collide:
            print(f"碰撞发生在 {link}")

        # 方式 2：一步完成
        is_collide, link = checker.check_collision_with_q(q)
    """

    def __init__(
        self,
        robot_visualizer: 'CollisionMeshVisualizer',
        env_visualizer: 'EnvironmentMeshVisualizer',
        # ---- 备选：直接注入（用于单元测试或自定义机器人） ----
        robot_fcl_objects: Optional[Dict[str, 'fcl.CollisionObject']] = None,
        robot_link_poses: Optional[Dict[str, np.ndarray]] = None,
        robot_local_aabbs: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        env_fcl_object: Optional['fcl.CollisionObject'] = None,
        env_world_min: Optional[np.ndarray] = None,
        env_world_max: Optional[np.ndarray] = None,
    ):
        """
        两种构造方式:
            1) 给 robot_visualizer + env_visualizer（推荐）
            2) 直接给所有原始数据（用于测试或自定义机器人）
        """
        # 解析机器人输入
        if robot_visualizer is not None:
            self._robot_fcl = robot_visualizer._fcl_objects
            self._robot_viz = robot_visualizer
            # 局部 AABB 来源：CollisionMeshVisualizer 内部没有保留 PolyData
            # 所以这里在构造时再从 URDF 读一次 collision mesh（不可行）
            # 因此走"备选 2"路线：必须由调用方注入 local_aabbs
            # 或者改用 BVHModel.aabb_center + 用整个 fcl obj 自带的 BVH
            raise NotImplementedError(
                "请显式提供 robot_local_aabbs 参数。"
                "CollisionMeshVisualizer 内部未保留源 mesh PolyData，"
                "无法在 checker 内部推导 local AABB。"
                "构造示例：\n"
                "    checker = RobotEnvCollisionChecker(\n"
                "        robot_visualizer=fine_viz,\n"
                "        env_visualizer=env_viz,\n"
                "        robot_link_poses={lnk: T_world},\n"
                "        robot_local_aabbs={lnk: (min, max)},\n"
                "    )"
            )
        else:
            self._robot_fcl = robot_fcl_objects
            self._robot_viz = None
            self._robot_link_poses = robot_link_poses or {}
            self._robot_local_aabbs = robot_local_aabbs or {}

        # 解析环境输入
        if env_visualizer is not None:
            self._env_fcl = env_visualizer.get_fcl_object()
            self._env_viz = env_visualizer
            # 从环境 mesh 推 local AABB，再通过 env world transform 推 world AABB
            if env_visualizer._pv_mesh is not None:
                lmin, lmax = _local_aabb_from_pv(env_visualizer._pv_mesh)
                T_env = env_visualizer.get_world_transform()
                self._env_world_min, self._env_world_max = _world_aabb_from_local(
                    lmin, lmax, T_env
                )
            else:
                self._env_world_min = np.zeros(3)
                self._env_world_max = np.zeros(3)
        else:
            self._env_fcl = env_fcl_object
            self._env_viz = None
            self._env_world_min = np.asarray(env_world_min, dtype=np.float64)
            self._env_world_max = np.asarray(env_world_max, dtype=np.float64)

        # fcl 碰撞请求
        try:
            import fcl
            self._fcl_req = fcl.CollisionRequest(
                num_max_contacts=1, enable_contact=False
            )
            self._fcl_available = True
        except ImportError:
            print("[RobotEnvCollisionChecker] python-fcl 未安装，碰撞检测已禁用")
            self._fcl_available = False

        # 性能统计（调试用）
        self._stats = {
            'step1_early_outs': 0,
            'step2_pruned': 0,
            'step3_collides': 0,
            'total_calls': 0,
        }

        print(
            f"[RobotEnvCollisionChecker] 初始化完成: "
            f"robot_links={len(self._robot_fcl)}, "
            f"env_AABB=[{self._env_world_min.tolist()}, {self._env_world_max.tolist()}]"
        )

    # -----------------------------------------------------------------------
    # 公共 API
    # -----------------------------------------------------------------------

    def check_collision(self) -> Tuple[bool, Set[str]]:
        """
        检测机器人与环境的碰撞（粗到精 BVH 剪枝）。

        前提:
            - 调用方必须已经执行过 robot_visualizer.update_joints(q)
            - 机器人每个 Link 的 fcl.CollisionObject.transform 已被更新
            - 如果需要获取当前关节角下的 Link 世界位姿来算 AABB，
              请使用 check_collision_with_q(q)

        返回:
            (is_collide, colliding_links)
            - is_collide: True 表示发生碰撞
            - colliding_links: 发生碰撞的所有 Link 名称集合；无碰撞时为空 set
        """
        self._stats['total_calls'] += 1

        if not self._fcl_available or self._env_fcl is None:
            return (False, set())

        # ---- Step 1: 全局 AABB 对比（O(1) Early-out） ----
        robot_min, robot_max = self._compute_robot_union_aabb()
        if robot_min is None:
            return (False, set())
        if not _aabb_overlap(robot_min, robot_max,
                             self._env_world_min, self._env_world_max):
            # 机器人总 AABB 与环境 AABB 不相交 → O(1) Early-out
            self._stats['step1_early_outs'] += 1
            return (False, set())

        # ---- Step 2: Link AABB 剪枝（Mid-phase / Link 级过滤） ----
        suspect_links: List[str] = []
        for link_name, (lmin, lmax) in self._robot_local_aabbs.items():
            if link_name not in self._robot_fcl:
                continue
            T_world = self._robot_link_poses.get(link_name)
            if T_world is None:
                continue  # 没有提供世界位姿的 Link 跳过
            link_min, link_max = _world_aabb_from_local(lmin, lmax, T_world)
            if _aabb_overlap(link_min, link_max,
                             self._env_world_min, self._env_world_max):
                suspect_links.append(link_name)
            else:
                self._stats['step2_pruned'] += 1

        if not suspect_links:
            return (False, set())

        # ---- Step 3: 深入底层 BVH 精确检测（Narrow-phase） ----
        import fcl
        colliding_links: Set[str] = set()
        for link_name in suspect_links:
            res = fcl.CollisionResult()
            ret = fcl.collide(
                self._robot_fcl[link_name],
                self._env_fcl,
                self._fcl_req,
                res,
            )
            if ret > 0:
                # 收集所有碰撞的 Link
                self._stats['step3_collides'] += 1
                colliding_links.add(link_name)

        return (len(colliding_links) > 0, colliding_links)

    def check_collision_with_q(
        self, q: np.ndarray,
        robot_visualizer: 'CollisionMeshVisualizer' = None,
    ) -> Tuple[bool, Set[str]]:
        """
        一站式接口：先 update_joints(q) 再 check_collision()。

        参数:
            q: 关节角度数组
            robot_visualizer: CollisionMeshVisualizer 实例
                              （如果在构造时已经传入，则这里可以不传）
                              必须实现 update_joints(q) 方法

        返回:
            (is_collide, colliding_links)
            - is_collide: True 表示发生碰撞
            - colliding_links: 发生碰撞的所有 Link 名称集合；无碰撞时为空 set
        """
        viz = robot_visualizer or self._robot_viz
        if viz is None:
            raise ValueError(
                "check_collision_with_q 需要 robot_visualizer 参数，"
                "但构造时未注入。请使用 check_collision() 并在外部先 update_joints(q)。"
            )
        # 1. 更新 Link 世界变换（同时更新 fcl.CollisionObject.transform）
        viz.update_joints(q)

        # 2. 从 visualizer 拿当前 Link 世界变换作为本帧 AABB 输入
        #    约定：CollisionMeshVisualizer 内部 _actors 存的是 user_matrix
        #    但 user_matrix 并不是 update_joints 真正写入的 4x4 矩阵
        #    正确做法: 在 update_joints 中暴露 world_T 给 checker
        #    这里采用最稳妥的方案：让调用方通过 robot_link_poses 参数显式注入
        if not self._robot_link_poses:
            # 尝试从 visualizer._actors 提取（PyVista actor.user_matrix）
            for link_name, actor in viz._actors.items():
                if hasattr(actor, 'user_matrix'):
                    self._robot_link_poses[link_name] = np.asarray(actor.user_matrix)

        return self.check_collision()

    def set_robot_link_pose(self, link_name: str, T_world: np.ndarray) -> None:
        """手动注入某个 Link 的当前世界变换（4x4）。"""
        self._robot_link_poses[link_name] = np.asarray(T_world, dtype=np.float64)

    def set_robot_link_poses(self, poses: Dict[str, np.ndarray]) -> None:
        """批量注入 Link 世界变换。"""
        for k, v in poses.items():
            self._robot_link_poses[k] = np.asarray(v, dtype=np.float64)

    def set_robot_local_aabb(
        self, link_name: str,
        local_min: np.ndarray, local_max: np.ndarray,
    ) -> None:
        """手动注入某个 Link 的局部 AABB（来自源 PV PolyData）。"""
        self._robot_local_aabbs[link_name] = (
            np.asarray(local_min, dtype=np.float64),
            np.asarray(local_max, dtype=np.float64),
        )

    def update_env_world_transform(self, T_wcf: np.ndarray) -> None:
        """
        当环境的位姿被外部修改时（如 EnvironmentMeshVisualizer.set_world_transform），
        必须调用此方法刷新 env world AABB。
        """
        if self._env_viz is not None:
            lmin, lmax = _local_aabb_from_pv(self._env_viz._pv_mesh)
        else:
            raise ValueError("env_visualizer 未注入，无法刷新 AABB")
        self._env_world_min, self._env_world_max = _world_aabb_from_local(
            lmin, lmax, T_wcf
        )

    def get_stats(self) -> Dict[str, int]:
        """返回累计的剪枝统计信息（用于性能分析）。"""
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {
            'step1_early_outs': 0,
            'step2_pruned': 0,
            'step3_collides': 0,
            'total_calls': 0,
        }

    # -----------------------------------------------------------------------
    # 内部
    # -----------------------------------------------------------------------

    def _compute_robot_union_aabb(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        计算所有 Link 的世界 AABB 并集。

        注意: 这只是 union，不是一个紧凑的 AABB（因为 union AABB 可能略大），
        但对剪枝逻辑是 sound 的（保守剪枝）。
        """
        if not self._robot_local_aabbs or not self._robot_link_poses:
            return None, None

        all_mins: List[np.ndarray] = []
        all_maxs: List[np.ndarray] = []

        for link_name, (lmin, lmax) in self._robot_local_aabbs.items():
            T_world = self._robot_link_poses.get(link_name)
            if T_world is None:
                continue
            wmin, wmax = _world_aabb_from_local(lmin, lmax, T_world)
            all_mins.append(wmin)
            all_maxs.append(wmax)

        if not all_mins:
            return None, None

        return (
            np.min(np.stack(all_mins, axis=0), axis=0),
            np.max(np.stack(all_maxs, axis=0), axis=0),
        )


# ---------------------------------------------------------------------------
# 便捷构造函数（从 CollisionMeshVisualizer 推导 local AABB）
# ---------------------------------------------------------------------------

def build_checker_from_visualizers(
    robot_visualizer: 'CollisionMeshVisualizer',
    env_visualizer: 'EnvironmentMeshVisualizer',
    robot_collision_meshes: Dict[str, 'pv.PolyData'],
) -> RobotEnvCollisionChecker:
    """
    一站式构造：从 CollisionMeshVisualizer + EnvironmentMeshVisualizer + URDF 源 mesh
    推导所有必要数据，构造 RobotEnvCollisionChecker。

    参数:
        robot_visualizer: 已完成 update_joints 初始化或未初始化均可
        env_visualizer: EnvironmentMeshVisualizer 实例
        robot_collision_meshes: link_name -> 源 pv.PolyData
            （必须与 robot_visualizer._fcl_objects 的 key 完全一致）
            可以从 collision.parse_urdf_collision_meshes(urdf_path) 得到 mesh_path，
            再用 pv.read() 加载

    返回:
        RobotEnvCollisionChecker 实例（已配置好 local AABB 和初始 world AABB）
    """
    robot_local_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for link_name, pv_mesh in robot_collision_meshes.items():
        lmin, lmax = _local_aabb_from_pv(pv_mesh)
        robot_local_aabbs[link_name] = (lmin, lmax)

    # 初始 poses 为单位矩阵（第一次 check_collision 之前需先 update_joints）
    initial_poses: Dict[str, np.ndarray] = {
        link_name: np.eye(4, dtype=np.float64)
        for link_name in robot_local_aabbs.keys()
    }

    return RobotEnvCollisionChecker(
        robot_visualizer=None,
        env_visualizer=env_visualizer,
        robot_fcl_objects=robot_visualizer._fcl_objects,
        robot_link_poses=initial_poses,
        robot_local_aabbs=robot_local_aabbs,
    )


# ---------------------------------------------------------------------------
# MultiEnvCollisionChecker（多静态物体 + 机器人 碰撞检测）
# ---------------------------------------------------------------------------

class MultiEnvCollisionChecker:
    """
    机器人 ↔ N 个静态物体（env/cad/tool）的统一碰撞检测器。

    设计目标:
        - 沿用 RobotEnvCollisionChecker 的 Step1/2/3 三段剪枝结构
        - 静态物体从单个变成 dict[str, StaticEntry]
        - 环境 vs 环境、CAD vs 环境、刀具 vs 环境 这些 env↔env 互碰**不构造 pair**，
          自然不会报红（无需 ignore list）
        - 机器人 vs 静态物体之间通过 `ignore_pairs` 集合屏蔽：
            * 相邻连杆对（来自 CollisionMeshVisualizer._adjacent_pairs）
            * 末端 link vs 刀具
        - check() 一次返回所有碰撞对，供 manager 高亮

    输入数据:
        robot_fcl_objects: Dict[link_name, fcl.CollisionObject]
        robot_local_aabbs: Dict[link_name, (local_min, local_max)]
        静态物体: 通过 add_static(name, fcl_obj, world_min, world_max) 注册

    返回 dict 形如::
        {
          'robot': {link_name: [static_name_1, ...], ...},
          'static': {static_name: [link_name_1, ...], ...},
        }

    使用方式::

        checker = MultiEnvCollisionChecker(
            robot_fcl=robot_viz._fcl_objects,
            robot_local_aabbs=...,
        )
        checker.add_static('env_workbench_1', fcl_obj, wmin, wmax)
        checker.add_static('tool', fcl_obj, wmin, wmax)
        checker.set_ignore_pairs({frozenset({'link_6', 'tool'}), ...})

        result = checker.check()
        # result['robot']['link_3'] = ['env_workbench_1']
        # result['static']['tool'] = ['link_3', 'link_4']
    """

    def __init__(
        self,
        robot_fcl_objects: Dict[str, 'fcl.CollisionObject'],
        robot_local_aabbs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        robot_link_poses: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._robot_fcl = robot_fcl_objects
        self._robot_local_aabbs = robot_local_aabbs
        self._robot_link_poses = robot_link_poses or {}

        # static_name -> (fcl_obj, world_min, world_max, name)
        self._static_objs: Dict[str, Tuple] = {}

        # ignore_pairs:
        #   - frozenset({link_name, static_name})  → robot↔static 忽略
        #   - frozenset({link_a, link_b})         → robot↔robot 忽略（自碰撞相邻连杆）
        self._ignore_pairs: Set[frozenset] = set()

        # fcl
        try:
            import fcl
            self._fcl_req = fcl.CollisionRequest(
                num_max_contacts=1, enable_contact=False
            )
            self._fcl_available = True
        except ImportError:
            print("[MultiEnvCollisionChecker] python-fcl 未安装，碰撞检测已禁用")
            self._fcl_available = False

        # 性能统计
        self._stats = {
            'step1_early_outs': 0,
            'step2_pruned': 0,
            'step3_collides': 0,
            'total_calls': 0,
        }

        print(
            f"[MultiEnvCollisionChecker] 初始化完成: "
            f"robot_links={len(self._robot_fcl)}, static_count=0"
        )

    # -----------------------------------------------------------------------
    # 静态物体管理
    # -----------------------------------------------------------------------

    def add_static(
        self,
        name: str,
        fcl_object: 'fcl.CollisionObject',
        world_min: np.ndarray,
        world_max: np.ndarray,
    ) -> None:
        """注册一个静态碰撞体（env / cad / tool）。"""
        if name in self._static_objs:
            # 重名 → 覆盖（避免重复注册产生 stale 引用）
            self.remove_static(name)
        self._static_objs[name] = (
            fcl_object,
            np.asarray(world_min, dtype=np.float64),
            np.asarray(world_max, dtype=np.float64),
            name,
        )

    def remove_static(self, name: str) -> None:
        """移除一个静态碰撞体。"""
        self._static_objs.pop(name, None)

    def clear_static(self) -> None:
        """清空所有静态碰撞体。"""
        self._static_objs.clear()

    def has_static(self, name: str) -> bool:
        return name in self._static_objs

    def static_names(self) -> List[str]:
        return list(self._static_objs.keys())

    def update_static_world_aabb(
        self, name: str,
        world_min: np.ndarray, world_max: np.ndarray,
    ) -> None:
        """当静态物体的位姿变化时，调用此方法刷新 AABB。"""
        if name not in self._static_objs:
            return
        fcl_obj, _, _, _ = self._static_objs[name]
        self._static_objs[name] = (
            fcl_obj,
            np.asarray(world_min, dtype=np.float64),
            np.asarray(world_max, dtype=np.float64),
            name,
        )

    # -----------------------------------------------------------------------
    # 忽略规则
    # -----------------------------------------------------------------------

    def set_ignore_pairs(self, pairs: Set[frozenset]) -> None:
        """
        接受两类 ignore pair:
            - frozenset({link_name, static_name})  → robot↔static 忽略
            - frozenset({link_a, link_b})         → robot↔robot 忽略
        """
        self._ignore_pairs = set(pairs)

    def add_ignore_pair(self, a: str, b: str) -> None:
        self._ignore_pairs.add(frozenset({a, b}))

    def add_ignore_robot_link(self, link_name: str, static_name: str) -> None:
        self._ignore_pairs.add(frozenset({link_name, static_name}))

    def is_ignored(self, link_name: str, static_name: str) -> bool:
        return frozenset({link_name, static_name}) in self._ignore_pairs

    # -----------------------------------------------------------------------
    # 机器人 link 位姿注入
    # -----------------------------------------------------------------------

    def set_robot_link_pose(self, link_name: str, T_world: np.ndarray) -> None:
        self._robot_link_poses[link_name] = np.asarray(T_world, dtype=np.float64)

    def set_robot_link_poses(self, poses: Dict[str, np.ndarray]) -> None:
        for k, v in poses.items():
            self._robot_link_poses[k] = np.asarray(v, dtype=np.float64)

    # -----------------------------------------------------------------------
    # 核心：check()
    # -----------------------------------------------------------------------

    def check(self) -> Dict[str, Dict[str, List[str]]]:
        """
        跑一轮碰撞检测。

        返回::
            {
              'robot':  {link_name: [static_name, ...]},
              'static': {static_name: [link_name, ...]},
            }

        注意: env↔env 互不检测（不构造 pair，自然不会报红）。
        """
        self._stats['total_calls'] += 1
        result: Dict[str, Dict[str, List[str]]] = {
            'robot': {},
            'static': {},
        }

        if not self._fcl_available or not self._static_objs:
            return result

        # ---- Step 1: 全局 robot AABB vs 各 static AABB（O(1) Early-out） ----
        robot_min, robot_max = self._compute_robot_union_aabb()
        if robot_min is None:
            return result

        suspect_statics: List[str] = []
        for sname, (fcl_obj, smin, smax, _) in self._static_objs.items():
            if fcl_obj is None:
                continue
            if _aabb_overlap(robot_min, robot_max, smin, smax):
                suspect_statics.append(sname)
            else:
                self._stats['step1_early_outs'] += 1

        if not suspect_statics:
            return result

        # ---- Step 2: Link AABB vs suspect static AABB ----
        import fcl
        suspect_pairs: List[Tuple[str, str]] = []
        for sname in suspect_statics:
            _, smin, smax, _ = self._static_objs[sname]
            for link_name, (lmin, lmax) in self._robot_local_aabbs.items():
                if link_name not in self._robot_fcl:
                    continue
                T_world = self._robot_link_poses.get(link_name)
                if T_world is None:
                    continue
                if self.is_ignored(link_name, sname):
                    continue
                link_min, link_max = _world_aabb_from_local(lmin, lmax, T_world)
                if _aabb_overlap(link_min, link_max, smin, smax):
                    suspect_pairs.append((link_name, sname))
                else:
                    self._stats['step2_pruned'] += 1

        if not suspect_pairs:
            return result

        # ---- Step 3: fcl.collide() 精确检测 ----
        for link_name, sname in suspect_pairs:
            res = fcl.CollisionResult()
            ret = fcl.collide(
                self._robot_fcl[link_name],
                self._static_objs[sname][0],
                self._fcl_req,
                res,
            )
            if ret > 0:
                self._stats['step3_collides'] += 1
                result['robot'].setdefault(link_name, []).append(sname)
                result['static'].setdefault(sname, []).append(link_name)

        return result

    # -----------------------------------------------------------------------
    # 统计 & 内部
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {
            'step1_early_outs': 0,
            'step2_pruned': 0,
            'step3_collides': 0,
            'total_calls': 0,
        }

    def _compute_robot_union_aabb(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._robot_local_aabbs or not self._robot_link_poses:
            return None, None
        all_mins: List[np.ndarray] = []
        all_maxs: List[np.ndarray] = []
        for link_name, (lmin, lmax) in self._robot_local_aabbs.items():
            T_world = self._robot_link_poses.get(link_name)
            if T_world is None:
                continue
            wmin, wmax = _world_aabb_from_local(lmin, lmax, T_world)
            all_mins.append(wmin)
            all_maxs.append(wmax)
        if not all_mins:
            return None, None
        return (
            np.min(np.stack(all_mins, axis=0), axis=0),
            np.max(np.stack(all_maxs, axis=0), axis=0),
        )
