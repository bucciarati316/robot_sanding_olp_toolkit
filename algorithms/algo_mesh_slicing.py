"""
algo_mesh_slicing - 基于论文的等弧长网格切片算法
================================================

一种基于研究论文的机器人磨削刀路生成算法：
"Region Growing Slicing and Constant Arc Length Resampling on Triangular Meshes"

算法核心思想：
1. 使用移动平面切片网格，获得原始交线
2. 对每条交线进行 B 样条拟合
3. 在弧长参数空间均匀重采样，获得等弧长分布的 CC 点
4. 通过 KDTree 查找最近顶点法向量
5. 构建刀具位姿矩阵 (TCP 姿态)

Author: AI Architect
Date: 2026-04-29
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
import logging

# ==================== 导入核心框架 ====================
import sys
from pathlib import Path

def _import_core():
    """动态导入核心框架，与插件架构完全解耦"""
    parent = Path(__file__).parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from core_algorithm import (
        BaseAlgorithm, ParamDef, ParamType, ToolpathResult, DataType, DebugItem
    )
    return BaseAlgorithm, ParamDef, ParamType, ToolpathResult, DataType, DebugItem

(BaseAlgorithm, ParamDef, ParamType, ToolpathResult, DataType, DebugItem) = _import_core()

# ==================== 导入第三方库 ====================
import trimesh
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev

logger = logging.getLogger(__name__)


class MeshSlicingAlgorithm(BaseAlgorithm):
    """
    基于论文的等弧长网格切片算法

    算法流程：
    1. 切片 - 使用移动平面切割网格
    2. 重采样 - B样条拟合 + 等弧长参数化
    3. 空间索引 - KDTree 加速最近邻查询
    4. 法向量计算 - 基于面积加权的顶点法向量
    5. 位姿生成 - 构建 TCP 变换矩阵
    """

    # 类属性 - 插件标识
    NAME = "Paper: Mesh Constant Arc Length Slicing"
    SUPPORTED_EXTS = ['.stl', '.ply', '.obj']

    def __init__(self):
        super().__init__()
        self.mesh: Optional[trimesh.Trimesh] = None
        self.kdtree: Optional[KDTree] = None
        self._cached_normals: Optional[np.ndarray] = None

    def get_parameters(self) -> List[ParamDef]:
        """
        返回参数定义列表，用于自动生成 GUI 控件

        Returns:
            List[ParamDef]: 参数定义列表
        """
        return [
            ParamDef(
                id="step_over",
                label="切片间距 (mm)",
                ptype=ParamType.FLOAT,
                default=2.0,
                min_val=0.1,
                max_val=20.0,
                step=0.5,
                desc="【mm】移动平面每步的切割距离。数值越小，层数越多，刀路越密集，但计算量越大"
            ),
            ParamDef(
                id="num_cc_points",
                label="CC点数量 (个/层)",
                ptype=ParamType.INT,
                default=100,
                min_val=10,
                max_val=1000,
                desc="【个/层】等弧长重采样点数，即每条轮廓曲线上均匀分布的CC（Cutter Contact）点数量。数值越大，轨迹越平滑，但数据量增加"
            ),
            ParamDef(
                id="cutting_axis",
                label="切片轴向",
                ptype=ParamType.CHOICE,
                default="Z",
                options=["X", "Y", "Z"],
                desc="【方向】切片移动平面的法向方向。Z轴表示水平逐层切割，X/Y轴用于侧向扫描"
            ),
        ]

    def load_geometry(self, filepath: str) -> bool:
        """
        加载网格几何体

        使用 trimesh 加载 STL/PLY/OBJ 文件，并初始化 KDTree 以便后续查询。

        Args:
            filepath: CAD 模型文件路径

        Returns:
            bool: 加载是否成功
        """
        try:
            self.log(f"正在加载网格: {filepath}")

            # 加载网格模型
            self.mesh = trimesh.load(force='mesh', file_obj=filepath)

            if not isinstance(self.mesh, trimesh.Trimesh):
                # 如果加载的是 Scene 或其他类型，尝试提取第一个网格
                if hasattr(self.mesh, 'dump'):
                    meshes = self.mesh.dump()
                    if len(meshes) > 0:
                        self.mesh = meshes[0]
                    else:
                        raise ValueError("Scene 中没有找到有效网格")
                else:
                    raise ValueError(f"加载的对象类型不支持: {type(self.mesh)}")

            # 确保网格有面和顶点数据
            if not self.mesh.is_watertight:
                self.log("[警告] 网格不是水密的，可能影响切片质量")

            # 网格加载完成
            self.log(f"网格加载完成: {len(self.mesh.vertices)} 顶点, {len(self.mesh.faces)} 面")

            # --- 单位修正 ---
            # CAD 软件导出默认使用毫米(mm)，这里统一缩小 1000 倍转化为米(m)
            # 这与 render_engine.py 中的处理保持一致
            self.mesh.vertices *= 0.001
            self.log("单位修正: mm → m (×0.001)")

            # 预计算面积加权顶点法向量
            self.mesh.vertex_normals

            # 构建 KDTree 用于快速最近邻查询
            self.kdtree = KDTree(self.mesh.vertices)

            self.is_loaded = True
            self.log("KDTree 构建完成，最近邻查询已就绪")
            return True

        except Exception as e:
            self.log(f"[错误] 网格加载失败: {e}")
            return False

    def generate(self, **kwargs) -> ToolpathResult:
        """
        执行网格切片与等弧长重采样，生成机器人磨削刀路

        参数从 kwargs 中提取，支持的参数：
            - step_over: 切片间距（默认 2.0 mm）
            - num_cc_points: 等弧长采样点数（默认 100）
            - cutting_axis: 切片轴向（默认 "Z"）

        Returns:
            ToolpathResult: 包含 points, normals, matrices 的刀路结果
        """
        if not self.is_loaded or self.mesh is None:
            raise RuntimeError("几何体未加载，请先调用 load_geometry()")

        # 提取参数
        step_over = kwargs.get('step_over', 2.0)
        num_cc_points = kwargs.get('num_cc_points', 100)
        cutting_axis = kwargs.get('cutting_axis', 'Z')

        self.log(f"开始生成刀路: axis={cutting_axis}, step_over={step_over}mm, CC点={num_cc_points}")

        # 创建结果容器
        result = ToolpathResult()

        # ==================== Step 1: 切片 ====================
        self.log("[Step 1] 执行网格切片...")
        raw_polylines = self._slice_mesh(step_over, cutting_axis)
        self.log(f"[Step 1] 切片完成，获得 {len(raw_polylines)} 条轮廓线")

        # 添加原始切片线段到调试数据
        for i, polyline in enumerate(raw_polylines[:10]):  # 只保存前10条用于可视化
            if len(polyline) >= 2:
                result.add_debug_data(
                    label=f"Step1_原始切片_{i}",
                    dtype=DataType.POLYLINE,
                    data=polyline,
                    color="#FF6600"  # 橙色
                )

        if len(raw_polylines) == 0:
            self.log("[警告] 未切出任何轮廓线")
            return result

        # ==================== Step 2: B样条拟合与等弧长重采样 ====================
        self.log("[Step 2] B样条拟合与等弧长重采样...")
        cc_points_list = []
        spline_params_list = []  # 每个曲线的样条参数，用于切向计算

        for i, polyline in enumerate(raw_polylines):
            if len(polyline) < 4:  # B样条至少需要4个点
                continue

            params = {}
            cc_points = self._resample_to_constant_arc_length(polyline, num_cc_points, params)
            if cc_points is not None and len(cc_points) > 0:
                cc_points_list.append(cc_points)
                spline_params_list.append(params)  # 包含 tck 和 u_equal
                self.log(f"  曲线 {i}: {len(polyline)} → {len(cc_points)} CC点")

        self.log(f"[Step 2] 重采样完成: {len(cc_points_list)}/{len(raw_polylines)} 条曲线成功处理")

        # 添加重采样曲线到调试数据
        for i, cc_points in enumerate(cc_points_list[:10]):
            if len(cc_points) > 1:
                result.add_debug_data(
                    label=f"Step2_CC曲线_{i}",
                    dtype=DataType.CURVE,
                    data=cc_points,
                    color="#00CC66"  # 绿色
                )

        if len(cc_points_list) == 0:
            self.log("[警告] 所有曲线重采样失败")
            return result

        # ==================== Step 3: 空间索引（KDTree已预构建）====================
        self.log("[Step 3] 空间索引就绪（KDTree已预构建）")

        # ==================== Step 4: 查找最近顶点法向量 ====================
        self.log("[Step 4] 查找最近顶点法向量...")
        all_normals = []

        for i, cc_points in enumerate(cc_points_list):
            normals_for_curve = []
            for point in cc_points:
                normal = self._get_nearest_normal(point)
                normals_for_curve.append(normal)
            all_normals.append(np.array(normals_for_curve))

        self.log(f"[Step 4] 法向量查询完成")

        # 添加法向量场到调试数据
        sample_idx = min(5, len(cc_points_list) - 1)
        if sample_idx >= 0:
            sample_points = cc_points_list[sample_idx]
            sample_normals = all_normals[sample_idx]
            if len(sample_points) > 0:
                # 将点和法向量组合用于可视化
                vectors = np.hstack([sample_points, sample_normals * 0.01])  # 缩短向量长度
                result.add_debug_data(
                    label="Step4_法向量场(采样)",
                    dtype=DataType.VECTOR_FIELD,
                    data=vectors,
                    color="#0066FF"  # 蓝色
                )

        # ==================== Step 5: 位姿矩阵生成 ====================
        self.log("[Step 5] 生成刀具位姿矩阵（解析切向）...")
        all_points = []
        all_output_normals = []
        all_matrices = []
        all_layer_indices = []

        for curve_idx, (cc_points, normals, spline_params) in enumerate(
            zip(cc_points_list, all_normals, spline_params_list)
        ):
            # 从保存的样条参数中获取 tck 和 u_equal
            tck = spline_params.get('tck')
            u_equal = spline_params.get('u_equal')

            if tck is None or u_equal is None:
                continue

            # 计算解析切向（B 样条 1 阶导数）
            # tangents 形状: [3, num_points]
            try:
                derivs = splev(u_equal, tck, der=1)
                tangents = np.array(derivs).T  # [num_points, 3]
            except Exception as e:
                logger.warning(f"切向计算失败，曲线 {curve_idx}: {e}")
                continue

            for i, (point, normal, tangent) in enumerate(zip(cc_points, normals, tangents)):
                # 归一化法向量 (Z 轴)
                z_axis = normal / (np.linalg.norm(normal) + 1e-10)

                # 归一化切向量 (曲线切向)
                t = tangent / (np.linalg.norm(tangent) + 1e-10)

                # 正确构建 TCP 位姿矩阵:
                # Z_axis = normal (刀具进给方向)
                # X_axis = normalized(cross(cross(normal, tangent), normal))
                # Y_axis = cross(Z_axis, X_axis)
                cross_n_t = np.cross(z_axis, t)
                x_axis = np.cross(cross_n_t, z_axis)
                x_norm = np.linalg.norm(x_axis)
                if x_norm < 1e-10:
                    # fallback: 找与法向量正交的任意向量
                    if abs(z_axis[0]) < 0.9:
                        arbitrary = np.array([1.0, 0.0, 0.0])
                    else:
                        arbitrary = np.array([0.0, 1.0, 0.0])
                    x_axis = np.cross(np.cross(z_axis, arbitrary), z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-10)

                # Y 轴由右手定则确定
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-10)

                # 构建 4x4 TCP 位姿矩阵 [R|t]
                T = np.eye(4)
                T[:3, 0] = x_axis   # X 轴
                T[:3, 1] = y_axis   # Y 轴
                T[:3, 2] = z_axis   # Z 轴
                T[:3, 3] = point    # 位置

                all_points.append(point)
                all_output_normals.append(normal)
                all_matrices.append(T)
                all_layer_indices.append(curve_idx)

        points_array = np.array(all_points) if all_points else np.array([]).reshape(0, 3)
        normals_array = (
            np.array(all_output_normals)
            if all_output_normals
            else np.array([]).reshape(0, 3)
        )
        matrices_array = np.array(all_matrices) if all_matrices else np.array([]).reshape(0, 4, 4)
        layer_indices_array = np.array(all_layer_indices, dtype=np.int32) if all_layer_indices else np.array([], dtype=np.int32)

        self.log(f"[Step 5] 位姿生成完成: {len(all_points)} 个刀位点")

        # 组装结果
        result.points = points_array
        result.normals = normals_array
        result.matrices = matrices_array
        result.layer_indices = layer_indices_array
        # 本插件矩阵使用切片解析切向生成，是可视化候选值。生产 TCP 仍由
        # PathSequencer 依据末端轴、刀具偏置和自由转角连续性统一重建。
        result.tcp_matrices_authoritative = False
        result.tcp_matrix_source = "mesh_slicing_candidate"
        result.validate()

        self.log(f"刀路生成完成: {result.num_waypoints} 个路径点")
        return result

    def _slice_mesh(self, step_over: float, axis: str) -> List[np.ndarray]:
        """
        使用移动平面切片网格，返回所有交线段

        Args:
            step_over: 切片间距（毫米）
            axis: 切片轴向 ('X', 'Y', 或 'Z')

        Returns:
            List[np.ndarray]: 每条轮廓线的顶点数组列表
        """
        # 获取包围盒
        bounds = self.mesh.bounds
        min_bound, max_bound = bounds[0], bounds[1]

        # 确定切片范围和方向
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
        z_min, z_max = min_bound[axis_index], max_bound[axis_index]

        # 将 step_over 从毫米转换为米（如果原始数据是米）
        step_m = step_over / 1000.0 if step_over > 1 else step_over

        # 计算切片层数
        total_layers = max(1, int((z_max - z_min) / step_m))

        # 收集所有切片平面与网格的交线段
        all_segments = []
        current_pos = z_min + step_m
        processed_layers = 0

        while current_pos < z_max:
            # 定义切片平面
            plane_origin = np.zeros(3)
            plane_normal = np.zeros(3)
            plane_origin[axis_index] = current_pos
            plane_normal[axis_index] = 1.0

            try:
                # 使用 trimesh 求交
                intersections = trimesh.intersections.mesh_plane(
                    self.mesh,
                    plane_normal,
                    plane_origin
                )

                if intersections is not None and len(intersections) > 0:
                    for segment in intersections:
                        if isinstance(segment, np.ndarray) and segment.shape == (2, 3):
                            all_segments.append(segment)

            except Exception as e:
                logger.debug(f"切片平面 {current_pos:.4f} 求交失败: {e}")

            current_pos += step_m
            processed_layers += 1

            # 每处理10层输出一次进度
            if processed_layers % 10 == 0:
                self.log(f"切片进度: {processed_layers}/{total_layers} 层, 累计线段: {len(all_segments)}")

        self.log(f"切片完成: 共 {processed_layers} 层, {len(all_segments)} 条线段")

        # 将线段连接成连续的轮廓线
        polylines = self._connect_segments(all_segments)

        return polylines

    def _connect_segments(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        将孤立的线段连接成连续的轮廓线

        使用 O(N) 哈希表/邻接表算法：
        1. 将所有端点坐标映射到字典（使用 round(pt, 5) 作为键）
        2. 构建端点到线段的邻接表
        3. 遍历图，提取最长的连续多段线

        Args:
            segments: 原始线段列表，每条线段是 [2, 3] 数组

        Returns:
            List[np.ndarray]: 连接后的连续多段线
        """
        if len(segments) == 0:
            return []

        if len(segments) == 1:
            return [segments[0]]

        self.log(f"开始连接 {len(segments)} 条线段（O(N) 哈希表算法）...")

        # 辅助函数：将点坐标转换为可哈希的元组
        def pt_to_key(pt: np.ndarray) -> tuple:
            return tuple(np.round(pt, 5).tolist())

        # 构建邻接表
        # point_to_segs[key] = [(seg_idx, is_start), ...]
        point_to_segs: dict = {}

        for seg_idx, seg in enumerate(segments):
            key_start = pt_to_key(seg[0])
            key_end = pt_to_key(seg[1])

            if key_start not in point_to_segs:
                point_to_segs[key_start] = []
            if key_end not in point_to_segs:
                point_to_segs[key_end] = []

            point_to_segs[key_start].append((seg_idx, True))   # 线段起点
            point_to_segs[key_end].append((seg_idx, False))   # 线段终点

        # 标记已使用的线段
        used = [False] * len(segments)
        polylines = []

        # 从任意未使用的端点开始，构建多段线
        for start_idx in range(len(segments)):
            if used[start_idx]:
                continue

            # 找到起始线段的两个端点
            seg = segments[start_idx]
            used[start_idx] = True

            key_a = pt_to_key(seg[0])
            key_b = pt_to_key(seg[1])

            # 构建当前多段线，从一个端点开始
            current_polyline = [seg[0], seg[1]]
            current_end_key = key_b

            # 贪婪扩展：优先连接度数小的端点
            changed = True
            while changed:
                changed = False

                if current_end_key not in point_to_segs:
                    break

                for seg_ref in point_to_segs[current_end_key]:
                    next_seg_idx, is_start = seg_ref
                    if used[next_seg_idx]:
                        continue

                    next_seg = segments[next_seg_idx]
                    used[next_seg_idx] = True

                    # 确定新端点
                    if is_start:
                        new_point = next_seg[1]
                    else:
                        new_point = next_seg[0]
                        next_seg = next_seg[::-1]  # 反转线段

                    current_polyline.append(new_point)
                    current_end_key = pt_to_key(new_point)
                    changed = True
                    break

            # 检查是否可以从另一端继续扩展（双向构建）
            reversed_polyline = current_polyline[::-1]
            reversed_end_key = pt_to_key(reversed_polyline[-1])

            while True:
                found = False
                if reversed_end_key not in point_to_segs:
                    break

                for seg_ref in point_to_segs[reversed_end_key]:
                    next_seg_idx, is_start = seg_ref
                    if used[next_seg_idx]:
                        continue

                    next_seg = segments[next_seg_idx]
                    used[next_seg_idx] = True

                    if is_start:
                        new_point = next_seg[1]
                    else:
                        new_point = next_seg[0]
                        next_seg = next_seg[::-1]

                    reversed_polyline.append(new_point)
                    reversed_end_key = pt_to_key(new_point)
                    found = True
                    break

                if not found:
                    break

            # 合并双向构建的结果
            if len(reversed_polyline) > len(current_polyline):
                polylines.append(np.array(reversed_polyline))
            else:
                polylines.append(np.array(current_polyline))

            # 进度日志
            used_count = sum(used)
            if used_count % 500 == 0:
                self.log(f"  连接进度: {used_count}/{len(segments)} 条线段已处理")

        self.log(f"轮廓线连接完成: {len(polylines)} 条连续轮廓")
        return polylines

    def _resample_to_constant_arc_length(
        self,
        polyline: np.ndarray,
        num_points: int,
        spline_params: dict = None
    ) -> Optional[np.ndarray]:
        """
        对多段线进行 B 样条拟合，然后使用真等弧长参数化重采样

        算法：
        1. B样条拟合原始多段线
        2. 密集采样（2000点）计算累积弦长
        3. 基于累积弦长进行真等弧长插值

        Args:
            polyline: 输入多段线顶点 [N, 3]
            num_points: 重采样后的点数
            spline_params: 输出参数，包含 (tck, u_equal) 用于计算切向

        Returns:
            np.ndarray: 均匀分布的 CC 点 [num_points, 3]，失败返回 None
        """
        if len(polyline) < 4:
            return None

        try:
            # 移除重复点
            polyline = self._remove_duplicate_points(polyline)

            if len(polyline) < 4:
                return None

            # B 样条拟合
            tck, u = splprep(
                polyline.T,  # splprep 需要 [3, N] 格式
                k=3,
                s=0.0,
                nest=-1
            )

            # 密集采样以精确计算弧长
            u_dense = np.linspace(0, 1, 2000)
            dense_points = np.array(splev(u_dense, tck)).T  # [2000, 3]

            # 计算累积弦长
            diffs = np.diff(dense_points, axis=0)  # [1999, 3]
            segment_lengths = np.linalg.norm(diffs, axis=1)  # [1999]
            cum_dist = np.zeros(len(u_dense))
            cum_dist[1:] = np.cumsum(segment_lengths)  # [2000]
            total_length = cum_dist[-1]

            if total_length < 1e-10:
                return None

            # 真等弧长参数化：均匀分布的目标距离
            target_dist = np.linspace(0, total_length, num_points)

            # 插值得到对应的参数值
            u_equal = np.interp(target_dist, cum_dist, u_dense)

            # 计算真等弧长分布的 CC 点
            cc_points = np.array(splev(u_equal, tck)).T  # [num_points, 3]

            # 如果需要，返回样条参数用于切向计算
            if spline_params is not None:
                spline_params['tck'] = tck
                spline_params['u_equal'] = u_equal

            return cc_points

        except Exception as e:
            logger.debug(f"B样条拟合失败: {e}")
            return None

    def _remove_duplicate_points(self, points: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        移除数组中的重复点

        Args:
            points: 输入点数组 [N, 3]
            tolerance: 距离容差

        Returns:
            np.ndarray: 去重后的点数组
        """
        if len(points) == 0:
            return points

        unique_mask = [True]
        for i in range(1, len(points)):
            is_unique = np.linalg.norm(points[i] - points[i - 1]) > tolerance
            unique_mask.append(is_unique)

        return points[unique_mask]

    def _get_nearest_normal(self, point: np.ndarray) -> np.ndarray:
        """
        查询最近顶点的面积加权法向量

        Args:
            point: 查询点 [3]

        Returns:
            np.ndarray: 法向量 [3]，已归一化
        """
        if self.kdtree is None or self.mesh is None:
            return np.array([0.0, 0.0, 1.0])

        # KDTree 查询最近邻
        dist, idx = self.kdtree.query(point)

        # 获取该顶点的法向量
        # trimesh 的 vertex_normals 是面积加权归一化的
        normal = self.mesh.vertex_normals[idx]

        # 确保法向量有效
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / norm

        return normal

    def __repr__(self) -> str:
        return f"MeshSlicingAlgorithm(NAME='{self.NAME}')"
