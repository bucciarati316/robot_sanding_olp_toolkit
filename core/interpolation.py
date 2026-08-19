import pandas as pd
import numpy as np
import os
from pathlib import Path
import pinocchio as pin
from .kinematics_engine import KinematicsEngine
from .schemas import SmoothingConfig, JointTrajectory

# =======================================================
# 全局配置区：声明输入文件路径及默认参数（仅用于独立脚本入口）
# =======================================================
# 在此处声明需要处理的文件的相对地址或绝对地址
# 例如: './joint_trajectorySLSQP.csv' (当前目录)
# 或 '../data/joint_trajectorySLSQP.csv' (上一级目录的data文件夹中)
INPUT_FILE_PATH = r'轨迹数据\5.18_0.1_0.1_0.9\joint_trajectory_SLSQP_mod.csv'

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 5.0
DEFAULT_N = 10
DEFAULT_METHOD = 'QuinticPolynomial'
DEFAULT_D = 0.1
DEFAULT_SIGN_BACK = -1
DEFAULT_URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / 'examples' / 'assets' / 'urdf' / 'demo_six_axis.urdf'
)
# =======================================================


class TrajectoryMethod:
    """轨迹插值方法基类"""
    def interpolate(self, q_start, q_end, p):
        raise NotImplementedError

class QuinticPolynomial(TrajectoryMethod):
    """五次多项式（S型曲线），保证位置、速度、加速度连续"""
    def interpolate(self, q_start, q_end, p):
        t = np.linspace(0, 1, p + 2)[1:-1]
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        return q_start + np.outer(s, (q_end - q_start))

class CubicPolynomial(TrajectoryMethod):
    """三次多项式曲线，保证位置、速度连续"""
    def interpolate(self, q_start, q_end, p):
        t = np.linspace(0, 1, p + 2)[1:-1]
        s = -2 * t**3 + 3 * t**2
        return q_start + np.outer(s, (q_end - q_start))

class TrajectorySmoother:
    """
    关节空间轨迹平滑处理器。

    检测关节轨迹中的跳变点，插入多项式插值使轨迹连续平滑。
    支持从传入的关节角度数组或 CSV 文件两种方式初始化。

    构造函数支持两种方式:
        新方式: TrajectorySmoother(config=SmoothingConfig(...), trajectory=positions_array)
        旧方式: TrajectorySmoother(filepath, method, alpha, beta, n)
    """

    def __init__(
        self,
        filepath_or_config=None,
        method=None,
        alpha=None,
        beta=None,
        n=None,
        config: SmoothingConfig = None,
        trajectory: np.ndarray = None,
    ):
        if config is not None:
            # 新方式：SmoothingConfig + trajectory 数组
            self.alpha = config.max_jump_threshold
            self.beta = config.velocity_threshold
            self.n = config.interpolation_degree
            self.method_name = 'QuinticPolynomial' if config.interpolation_degree == 5 else 'CubicPolynomial'
            self.data = trajectory
            self.columns = None
            self.filepath = None
        elif isinstance(filepath_or_config, str):
            # 旧方式：文件路径
            self.filepath = filepath_or_config
            self.df = pd.read_csv(filepath_or_config)
            self.data = self.df.values
            self.columns = self.df.columns
            self.alpha = alpha if alpha is not None else DEFAULT_ALPHA
            self.beta = beta if beta is not None else DEFAULT_BETA
            self.n = n if n is not None else DEFAULT_N
            self.method_name = method if method is not None else DEFAULT_METHOD
        else:
            raise TypeError("TrajectorySmoother 期望 filepath 字符串或 SmoothingConfig 配置对象")

        if self.method_name == 'QuinticPolynomial':
            self.method = QuinticPolynomial()
        elif self.method_name == 'CubicPolynomial':
            self.method = CubicPolynomial()
        else:
            raise ValueError(f"不支持的轨迹插值方法: {self.method_name}")

    def process(self):
        diffs = np.diff(self.data, axis=0)
        num_points, num_joints = self.data.shape
        jumps = []

        interval_count = self.n - 1
        for i in range(interval_count, len(diffs)):
            prev_diffs = diffs[i - interval_count : i]
            v_mean = np.mean(np.abs(prev_diffs), axis=0)

            curr_diff = diffs[i]
            prev_diff = diffs[i-1]

            is_jump = False
            max_p = 0

            for j in range(num_joints):
                cond1 = abs(curr_diff[j]) > self.beta * v_mean[j]
                cond2 = (np.sign(curr_diff[j]) * np.sign(prev_diff[j]) < 0)

                if cond1 and cond2:
                    is_jump = True
                    if v_mean[j] > 1e-6:
                        p_j = int(np.ceil(self.alpha * abs(curr_diff[j]) / v_mean[j]))
                    else:
                        p_j = 50

                    if p_j > max_p:
                        max_p = p_j

            if is_jump:
                jumps.append({
                    'start_idx': i,
                    'end_idx': i + 1,
                    'p': max_p
                })

        new_data = []
        jump_idx = 0

        for i in range(num_points):
            new_data.append(self.data[i])

            if jump_idx < len(jumps) and i == jumps[jump_idx]['start_idx']:
                p = jumps[jump_idx]['p']
                q_start = self.data[i]
                q_end = self.data[i+1]

                inserted_points = self.method.interpolate(q_start, q_end, p)
                for pt in inserted_points:
                    new_data.append(pt)

                jump_idx += 1

        new_df = pd.DataFrame(new_data, columns=self.columns)

        base_dir, file_name = os.path.split(self.filepath)
        name, ext = os.path.splitext(file_name)
        method_name = self.method.__class__.__name__

        # 将导出的文件存储在原文件相同的相对路径下
        output_name = os.path.join(base_dir, f"{name}_{method_name}{ext}")
        new_df.to_csv(output_name, index=False)
        return output_name, jumps


# =======================================================
# 三段式防碰撞轨迹平滑处理器
# =======================================================
class CartesianMovel:
    """笛卡尔空间直线 Movel 插值器: 位置 LERP + 姿态四元数 SLERP"""
    @staticmethod
    def interpolate(T_start, T_end, steps):
        if steps < 2:
            raise ValueError("插值步数必须 >= 2")
        path = []
        for k in range(steps):
            t = k / (steps - 1)
            p_k = (1 - t) * T_start[:3, 3] + t * T_end[:3, 3]
            q_k = pin.Quaternion(T_start[:3, :3]).slerp(t, pin.Quaternion(T_end[:3, :3]))
            T_k = np.eye(4)
            T_k[:3, :3] = q_k.matrix()
            T_k[:3, 3] = p_k
            path.append(T_k)
        return path


class ThreeSegmentSmoother:
    """
    三段式防碰撞轨迹平滑处理器

    处理流程（以单个跳变点为例，跳变发生在索引 i 和 i+1 之间）：

    1. FK求解：获取 q_i 和 q_{i+1} 对应的笛卡尔位姿 T1、T2
    2. 退刀：沿 T1 的 Z 轴方向移动 D，得到 T1_retract
    3. 进刀终点：沿 T2 的 Z 轴方向移动 D，得到 T2_retract
    4. 退刀段 Movel：T1 -> T1_retract，IK 得到关节轨迹
    5. 过渡段：退刀终点关节 -> 进刀起点关节，五项式插值
    6. 进刀段 Movel：T2_retract -> T2，IK 得到关节轨迹（反向插入）
    7. 拼接：new_data = [q_i] + q_retract + q_transition + q_approach
    """

    def __init__(
        self,
        filepath: str = INPUT_FILE_PATH,
        urdf_path: str = DEFAULT_URDF_PATH,
        method: str = DEFAULT_METHOD,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        n: int = DEFAULT_N,
        retract_distance: float = DEFAULT_D,
        sign_back: int = DEFAULT_SIGN_BACK,
        log_callback=None,
        kin_engine=None,
        ik_solver_plugin=None,
    ):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.data = self.df.values
        self.columns = self.df.columns

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.method_name = method
        self.retract_distance = retract_distance
        self.sign_back = sign_back
        self._log = log_callback if log_callback else (lambda msg: print(msg))

        if method == 'QuinticPolynomial':
            self.method = QuinticPolynomial()
        elif method == 'CubicPolynomial':
            self.method = CubicPolynomial()
        else:
            raise ValueError(f"不支持的轨迹插值方法: {method}")

        self._ik_solver = ik_solver_plugin
        if kin_engine is not None:
            self.kinematics = kin_engine
        else:
            self._init_kinematics_engine(urdf_path)

    def _init_kinematics_engine(self, urdf_path: str):
        """初始化 KinematicsEngine，基座为世界坐标系原点"""
        self.kinematics = KinematicsEngine(urdf_path, verbose=False)
        self.kinematics.set_transforms(np.eye(4), np.eye(4))
        self.kinematics.set_tcp_transform(np.eye(4))

    def _ik_solve(self, T_target: np.ndarray, q_seed: np.ndarray,
                  max_iterations: int = 500, tolerance: float = 1e-4) -> np.ndarray:
        """
        带三级回退的 IK 求解（通过插件式 IK 求解器）：
        1. 当前 seed
        2. 零位
        3. 中立姿态
        """
        if self._ik_solver is None:
            raise RuntimeError(
                "ThreeSegmentSmoother 未绑定 IK 求解器插件。"
                "请通过 kin_engine 和 ik_solver_plugin 参数传入。"
            )

        strategies = [
            ("seed", q_seed.copy()),
            ("zero", np.zeros(self.kinematics.nq)),
            ("neutral", self._get_neutral_pose()),
        ]
        last_error = float('inf')
        last_q = q_seed.copy()

        for _, q_init in strategies:
            success, q_sol = self._ik_solver.solve(T_target, q_init)
            final_error = self._compute_ik_error(T_target, q_sol)
            if success and final_error < tolerance:
                self.kinematics.current_q = q_sol.copy()
                return q_sol
            if final_error < last_error:
                last_error = final_error
                last_q = q_sol.copy()

        self.kinematics.current_q = last_q.copy()
        return last_q

    def _compute_ik_error(self, T_target: np.ndarray, q: np.ndarray) -> float:
        """计算 IK 求解误差（log6 向量范数）"""
        pin.framesForwardKinematics(self.kinematics.model, self.kinematics.data, q)
        current_SE3 = self.kinematics.data.oMf[self.kinematics.tool_frame_id]
        err = pin.log6(current_SE3.actInv(pin.SE3(T_target))).vector
        return float(np.linalg.norm(err))

    def _get_neutral_pose(self) -> np.ndarray:
        return np.zeros(self.kinematics.nq)

    def _detect_jumps(self) -> list:
        """
        检测关节轨迹中的跳变点。

        off-by-2 修复：判断 sign reversal 发生时，大幅跳变在 curr_diff 还是 prev_diff 中，
        将 start_idx 指向包含大幅跳变的 diff 对应的位置，使检测索引与实际跳变位置对齐。

        角度环绕误检修复：对于 sign reversal 时差值接近 ±pi 的关节（连续旋转经过 ±pi 边界），
        跳过误检。
        """
        diffs = np.diff(self.data, axis=0)
        num_points, num_joints = self.data.shape
        jumps = []

        interval_count = self.n - 1

        for i in range(interval_count, len(diffs)):
            prev_diffs = diffs[i - interval_count: i]
            v_mean = np.mean(np.abs(prev_diffs), axis=0)

            curr_diff = diffs[i]
            prev_diff = diffs[i - 1]

            is_jump = False
            max_p = 0
            jump_joint = -1
            jump_in_prev = False

            for j in range(num_joints):
                cond1_curr = abs(curr_diff[j]) > self.beta * v_mean[j]
                cond1_prev = abs(prev_diff[j]) > self.beta * v_mean[j]
                cond2 = (np.sign(curr_diff[j]) * np.sign(prev_diff[j]) < 0)

                if not cond2:
                    continue

                if abs(curr_diff[j]) > np.pi * 0.8:
                    continue

                if cond1_curr:
                    is_jump = True
                    jump_in_prev = False
                    if v_mean[j] > 1e-6:
                        p_j = int(np.ceil(self.alpha * abs(curr_diff[j]) / v_mean[j]))
                    else:
                        p_j = 50
                    if p_j > max_p:
                        max_p = p_j
                        jump_joint = j
                elif cond1_prev:
                    is_jump = True
                    jump_in_prev = True
                    if v_mean[j] > 1e-6:
                        p_j = int(np.ceil(self.alpha * abs(prev_diff[j]) / v_mean[j]))
                    else:
                        p_j = 50
                    if p_j > max_p:
                        max_p = p_j
                        jump_joint = j

            if is_jump:
                jumps.append({
                    'start_idx': i - 1 if jump_in_prev else i,
                    'end_idx': i if jump_in_prev else i + 1,
                    'p': max_p,
                    'joint': jump_joint,
                })

        return jumps

    def _three_segment(self, q_start: np.ndarray, q_end: np.ndarray,
                      T1: np.ndarray, T2: np.ndarray, p: int) -> tuple:
        """
        退刀 + 进刀两段：
        - 退刀【0→p】：沿 T1 的 TCP Z 轴退刀 D，IK 求解，方向正向。
        - 进刀【p→0】：计算 T2_engage = T2 沿 TCP Z 退刀 D，
                       从 T2 到 T2_engage 求 IK 后 reverse，方向正向【T2, ..., T2_engage】。

        参数:
            q_start: 跳变点起点关节角 (nq,)，即 q_i
            q_end: 跳变点终点关节角 (nq,)，即 q_{i+1}
            T1: q_start 对应的 TCP 位姿 (4x4)
            T2: q_end 对应的 TCP 位姿 (4x4)
            p: 总插值点数

        返回:
            tuple: (q_retract_list, p_retract, q_approach_list, p_approach)
        """
        D = self.retract_distance
        sign = self.sign_back
        p_retract = max(1, p)
        p_approach = max(1, p)

        # ---- 退刀段 [0 → p]：沿 T1_Z 退刀，方向正向 ----
        T1_Z = T1[:3, 2].copy()
        T1_retract = T1.copy()
        T1_retract[:3, 3] += D * sign * T1_Z

        cartesian_retract = CartesianMovel.interpolate(T1, T1_retract, p_retract + 2)
        q_retract_list = []
        current_q = q_start.copy()
        for T_step in cartesian_retract[1:-1]:
            current_q = self._ik_solve(T_step, current_q)
            q_retract_list.append(current_q.copy())
        q_retract_list.append(self._ik_solve(T1_retract, current_q))

        # ---- 进刀段 [p → 0]：T2_engage = T2 沿 TCP Z 退刀，反向求 IK 后 reverse ----
        T2_Z = T2[:3, 2].copy()
        T2_engage = T2.copy()
        T2_engage[:3, 3] += D * sign * T2_Z

        cartesian_to_engage = CartesianMovel.interpolate(T2, T2_engage, p_approach + 2)
        q_to_engage_list = []
        current_q = q_end.copy()
        for T_step in cartesian_to_engage[1:-1]:
            current_q = self._ik_solve(T_step, current_q)
            q_to_engage_list.append(current_q.copy())
        q_to_engage_list.append(self._ik_solve(T2_engage, current_q))

        q_to_engage_list.reverse()
        q_approach_list = q_to_engage_list

        return q_retract_list, p_retract, q_approach_list, p_approach

    def _three_segment_with_fallback(self, q_start: np.ndarray, q_end: np.ndarray,
                                     T1: np.ndarray, T2: np.ndarray,
                                     p: int) -> tuple:
        """退刀+进刀段健壮封装，失败时返回空列表"""
        try:
            result = self._three_segment(q_start, q_end, T1, T2, p)
            return result
        except Exception as e:
            self._log(f"  [严重警告] 退刀/进刀段异常: {e}，跳过该段")
            return [], max(1, p), [], max(1, p)

    def process(self) -> tuple:
        """
        执行三段式轨迹平滑处理

        返回:
            tuple: (output_file, jumps, segment_info)
        """
        jumps = self._detect_jumps()
        self._log(f"[检测] 共发现 {len(jumps)} 个跳变点")

        new_data = []
        segment_info = []
        num_points = len(self.data)
        processed_starts = set()

        for i in range(num_points):
            if i in processed_starts:
                continue

            matching_jumps = [j for j in jumps if j['start_idx'] == i]
            if matching_jumps:
                processed_starts.add(i)
                jump = matching_jumps[0]
                p = jump['p']
                q_i = self.data[i]
                q_i1 = self.data[i + 1]

                self._log(f"\n[处理跳变点] i={i}, p={p}")

                T1 = self.kinematics.forward_kinematics(q_i)
                T2 = self.kinematics.forward_kinematics(q_i1)

                self._log(f"  T1 位置: [{T1[0,3]:.4f}, {T1[1,3]:.4f}, {T1[2,3]:.4f}]")
                self._log(f"  T2 位置: [{T2[0,3]:.4f}, {T2[1,3]:.4f}, {T2[2,3]:.4f}]")

                (q_retract_list, p_retract,
                 q_approach_list, p_approach) = \
                    self._three_segment_with_fallback(q_i, q_i1, T1, T2, p)

                # ---- 过渡段：退刀终点 -> 进刀起点，五次多项式插值 ----
                q_transition_list = []
                p_transition = 0
                if q_retract_list and q_approach_list:
                    q_trans_start = q_retract_list[-1]
                    q_trans_end = q_approach_list[0]
                    joint_diffs = np.abs(q_trans_end - q_trans_start)
                    max_diff_joint = np.max(joint_diffs)
                    p_transition = max(2, int(np.ceil(max_diff_joint * 50)))
                    q_transition_list = self.method.interpolate(
                        q_trans_start, q_trans_end, p_transition
                    ).tolist()
                    q_transition_list = [np.array(q) for q in q_transition_list]
                    self._log(f"  过渡段: {len(q_transition_list)} 点已添加 (p={p_transition})")
                else:
                    self._log(f"  过渡段: 跳过（退刀或进刀段为空）")

                segment_info.append({
                    'jump_idx': i,
                    'p': p,
                    'p_retract': p_retract,
                    'p_approach': p_approach,
                    'p_transition': p_transition,
                    'T1': T1,
                    'T2': T2,
                    'q_retract_list': q_retract_list,
                    'q_approach_list': q_approach_list,
                    'q_transition_list': q_transition_list,
                })

                if q_retract_list:
                    new_data.append(q_i.copy())
                    for q in q_retract_list:
                        new_data.append(q)
                    self._log(f"  退刀段: {len(q_retract_list)} 点已添加（原始点已保留）")
                else:
                    new_data.append(q_i.copy())
                    self._log(f"  退刀段: 回退（跳过）")

                for q in q_transition_list:
                    new_data.append(q)

                if q_approach_list:
                    for q in q_approach_list:
                        new_data.append(q)
                    self._log(f"  进刀段: {len(q_approach_list)} 点已添加")
                else:
                    new_data.append(q_i1.copy())
                    self._log(f"  进刀段: 回退（跳过）")
            else:
                new_data.append(self.data[i].copy())

        new_df = pd.DataFrame(new_data, columns=self.columns)

        base_dir, file_name = os.path.split(self.filepath)
        name, ext = os.path.splitext(file_name)
        output_name = os.path.join(base_dir, f"{name}_ThreeSegment{ext}")
        new_df.to_csv(output_name, index=False)

        return output_name, jumps, segment_info

    def run_tests(self, segment_info: list):
        """
        验证退刀段 IK 求解质量：统计成功/失败数量，随机抽取误差最大的3个打印。
        """
        self._log("\n" + "=" * 60)
        self._log("退刀段 IK 求解质量验证")
        self._log("=" * 60)

        if not segment_info:
            self._log("[跳过] 未找到跳变段信息")
            return

        all_failures = []
        total_points = 0

        for seg in segment_info:
            i = seg['jump_idx']
            q_retract = seg['q_retract_list']
            if not q_retract:
                continue

            T1_Z = seg['T1'][:3, 2]
            T1_retract = seg['T1'].copy()
            T1_retract[:3, 3] += self.retract_distance * self.sign_back * T1_Z

            p_retract = seg['p_retract']
            cartesian_path = CartesianMovel.interpolate(seg['T1'], T1_retract, p_retract + 2)

            for pt_idx, q in enumerate(q_retract):
                total_points += 1
                T_fk = self.kinematics.forward_kinematics(q)
                T_target = cartesian_path[pt_idx + 1]
                dist = np.linalg.norm(T_fk[:3, 3] - T_target[:3, 3])
                if dist >= 1e-3:
                    all_failures.append({
                        'jump_idx': i,
                        'pt_idx': pt_idx,
                        'error': dist,
                        'q': q.copy(),
                    })

        success_count = total_points - len(all_failures)
        self._log(f"  求解成功: {success_count}/{total_points}")
        self._log(f"  求解失败: {len(all_failures)}/{total_points}")

        if all_failures:
            all_failures.sort(key=lambda x: x['error'], reverse=True)
            top_failures = all_failures[:3]
            self._log(f"\n  误差最大的 3 个失败点:")
            for f in top_failures:
                self._log(f"    跳变点 i={f['jump_idx']}, 点序={f['pt_idx']}, "
                      f"误差={f['error']:.6e} m")

        self._log("\n" + "=" * 60)


# -----------------
# 运行控制台
# -----------------
if __name__ == "__main__":
    print("[启动] 三段式防碰撞轨迹平滑测试")
    print(f"  输入文件: {INPUT_FILE_PATH}")
    print(f"  URDF路径: {DEFAULT_URDF_PATH}")
    print(f"  退刀距离: {DEFAULT_D} m, 方向符号: {DEFAULT_SIGN_BACK}")

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"[错误] 找不到指定的轨迹文件: '{INPUT_FILE_PATH}'")
        print("请检查代码顶部的 'INPUT_FILE_PATH' 相对路径配置是否正确。")
    elif not os.path.exists(DEFAULT_URDF_PATH):
        print(f"[错误] 找不到指定的 URDF 文件: '{DEFAULT_URDF_PATH}'")
        print("请检查 'DEFAULT_URDF_PATH' 配置是否正确。")
    else:
        from .auto_manager import get_manager
        manager = get_manager()

        kin_engine = KinematicsEngine(DEFAULT_URDF_PATH, verbose=False)
        kin_engine.set_transforms(np.eye(4), np.eye(4))
        kin_engine.set_tcp_transform(np.eye(4))

        SolverClass = manager.get_algorithm_class('SLSQP非线性优化', prefix='ik')
        ik_solver_plugin = SolverClass(kinematics_engine=kin_engine)

        smoother = ThreeSegmentSmoother(
            filepath=INPUT_FILE_PATH,
            kin_engine=kin_engine,
            ik_solver_plugin=ik_solver_plugin,
            method=DEFAULT_METHOD,
            alpha=DEFAULT_ALPHA,
            beta=DEFAULT_BETA,
            n=DEFAULT_N,
            retract_distance=DEFAULT_D,
            sign_back=DEFAULT_SIGN_BACK,
        )

        output_file, jumps, segment_info = smoother.process()

        print(f"\n[完成] 处理完成！")
        print(f"  检测跳变点数: {len(jumps)}")
        print(f"  输出文件: {output_file}")

        smoother.run_tests(segment_info)
