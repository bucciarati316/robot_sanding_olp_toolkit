"""
KinematicsEngine - 机器人运动学求解引擎
基于 Pinocchio 库实现正运动学(FK)和逆运动学(IK)
支持 URDF 模型加载、坐标系变换、伪逆雅可比 IK 求解器
"""
import numpy as np
import pinocchio as pin
import xml.etree.ElementTree as ET

from .schemas import FlangeToolParams, CoordinateFrames, KinematicsConfig
from .coordinate_transforms import ocf_to_rcf


class KinematicsEngine:
    """
    机器人运动学求解器

    功能：
    1. 加载 URDF 模型并初始化 Pinocchio model/data
    2. 设置世界坐标系到机器人基座、工件的变换矩阵
    3. 基于伪逆雅可比的迭代 IK 求解器，支持 seed 姿态
    """

    def __init__(self, config: KinematicsConfig, verbose: bool = False,
                 tool_library=None):
        """
        初始化运动学引擎。

        支持两种调用方式:
            新方式: KinematicsEngine(KinematicsConfig(...))
            旧方式: KinematicsEngine(urdf_path, verbose=False)  # 向后兼容

        参数:
            config: 运动学配置对象（KinematicsConfig 类型），或 URDF 路径字符串（向后兼容）
            verbose: 是否打印加载信息（仅在旧调用方式下有效）
            tool_library: 工具库引用（向后兼容参数，已废弃）
        """
        # ---- 向后兼容：检测旧调用方式 (urdf_path, verbose, tool_library) ----
        if isinstance(config, str):
            # 旧方式：传入的是 URDF 路径字符串
            urdf_path = config
            _coord_frames = CoordinateFrames(T_wcf_rcf=np.eye(4), T_wcf_ocf=np.eye(4))
            _ftp = FlangeToolParams()
            config = KinematicsConfig(
                urdf_path=urdf_path,
                flange_tool_params=_ftp,
                coordinate_frames=_coord_frames,
            )
        elif not isinstance(config, KinematicsConfig):
            raise TypeError(
                f"KinematicsEngine.__init__ 期望 KinematicsConfig 或 str，实际收到 {type(config).__name__}"
            )

        self._config = config
        self.urdf_path = config.urdf_path

        try:
            self.model = pin.buildModelFromUrdf(config.urdf_path)
            self.data = self.model.createData()
        except Exception as e:
            raise RuntimeError(f"URDF 加载失败: {e}")

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.joint_names = [name for name in self.model.names]

        self._end_effector_axis = self._detect_end_effector_axis(config.urdf_path)
        self._flange_axis = self._end_effector_axis
        self._tool0_axis = None
        self._tool_axis = None
        self._parse_urdf_axes(config.urdf_path)

        self._params = config.flange_tool_params
        self.T_flange_tcp = self._compute_tcp_from_params()

        # 从 CoordinateFrames 初始化，运行时可通过 set_transforms 更新
        if config.coordinate_frames is not None:
            self.T_wcf_rcf = config.coordinate_frames.T_wcf_rcf
            self.T_wcf_ocf = config.coordinate_frames.T_wcf_ocf
        else:
            self.T_wcf_rcf = np.eye(4)
            self.T_wcf_ocf = np.eye(4)

        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()
        self.tool_frame_id = self.model.getFrameId(self.model.names[-1])
        self.current_q = None

        if verbose:
            print("=" * 50)
            print(f"URDF 加载成功: {config.urdf_path}")
            print(f"关节数量: {self.nq}")
            print(f"关节名称: {self.joint_names}")
            print(f"末端执行器轴 (joint6): {self._end_effector_axis}")
            print(f"Flange 轴: {self._flange_axis}")
            print(f"Tool0 轴: {self._tool0_axis}")
            print(f"Tool 轴: {self._tool_axis}")
            print("=" * 50)

    def set_transforms(self, T_wcf_rcf: np.ndarray, T_wcf_ocf: np.ndarray):
        """
        设置坐标系变换矩阵。

        参数:
            T_wcf_rcf: 世界到机器人基座的 4x4 齐次变换矩阵 (SE3)
            T_wcf_ocf: 世界到工件坐标系的 4x4 齐次变换矩阵 (SE3)
        """
        assert T_wcf_rcf.shape == (4, 4), "T_wcf_rcf 必须是 4x4 矩阵"
        assert T_wcf_ocf.shape == (4, 4), "T_wcf_ocf 必须是 4x4 矩阵"
        self.T_wcf_rcf = T_wcf_rcf
        self.T_wcf_ocf = T_wcf_ocf
        print(f"[KinematicsEngine] 坐标系变换已更新")
        print(f"  T_wcf_rcf (世界->机器人基座):\n{self.T_wcf_rcf}")
        print(f"  T_wcf_ocf (世界->工件):\n{self.T_wcf_ocf}")

    def set_tcp_transform(self, T_flange_tcp: np.ndarray):
        """兼容旧代码：从 4x4 矩阵直接设置 T_flange_tcp"""
        self.T_flange_tcp = T_flange_tcp
        print(f"[KinematicsEngine] TCP 偏置已更新 (直接矩阵方式)")
        print(f"  T_flange_tcp =\n{self.T_flange_tcp}")
        print(f"  旋转矩阵 R =\n{self.T_flange_tcp[:3, :3]}")
        print(f"  平移向量 t = [{self.T_flange_tcp[0,3]}, {self.T_flange_tcp[1,3]}, {self.T_flange_tcp[2,3]}]")

    def set_tool_library(self, tool_library):
        """
        设置共享的 ToolLibrary 引用。

        设置后，IK 求解时通过 tool_library.T_flange_tcp 实时读取当前刀具偏置，
        保证 Step 5 修改刀具后立即在 Step 7 中生效。

        参数:
            tool_library: ToolLibrary 实例
        """
        self._tool_library = tool_library
        print(f"[KinematicsEngine] ToolLibrary 已绑定: 当前工具={tool_library.get_current_tool().name}")

    @property
    def effective_tcp(self) -> np.ndarray:
        """
        获取当前有效的 T_flange_tcp。

        优先级: _effective_tcp_override（solve_ik_indirect 临时覆盖）>
               ToolLibrary 引用（动态）> 本地缓存（静态）。
        """
        if hasattr(self, '_effective_tcp_override'):
            return self._effective_tcp_override
        if hasattr(self, '_tool_library') and self._tool_library is not None:
            return self._tool_library.T_flange_tcp
        return self.T_flange_tcp

    def _detect_end_effector_axis(self, urdf_path: str) -> str:
        """
        从 URDF 解析末端执行器的旋转轴

        通过查找最后一个 revolute 类型关节的 axis 属性来判断
        例如：
        - axis="1 0 0" -> 返回 'x'
        - axis="0 1 0" -> 返回 'y'
        - axis="0 0 1" -> 返回 'z'

        参数:
            urdf_path: URDF 文件路径

        返回:
            str: 'x', 'y' 或 'z'
        """
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # 找到最后一个 revolute 关节
            last_revolute_joint = None
            for joint in root.findall('joint'):
                joint_type = joint.get('type', '')
                if joint_type == 'revolute':
                    last_revolute_joint = joint

            if last_revolute_joint is None:
                print("[KinematicsEngine] 警告: 未找到 revolute 关节，默认使用 'z' 轴")
                return 'z'

            # 获取 axis 元素
            axis_elem = last_revolute_joint.find('axis')
            if axis_elem is None:
                print("[KinematicsEngine] 警告: 关节无 axis 属性，默认使用 'z' 轴")
                return 'z'

            axis_str = axis_elem.get('xyz', '0 0 1')
            axis_values = list(map(float, axis_str.split()))

            # 判断哪个轴为 1
            if len(axis_values) >= 1 and abs(axis_values[0] - 1.0) < 1e-6:
                return 'x'
            elif len(axis_values) >= 2 and abs(axis_values[1] - 1.0) < 1e-6:
                return 'y'
            elif len(axis_values) >= 3 and abs(axis_values[2] - 1.0) < 1e-6:
                return 'z'
            else:
                print("[KinematicsEngine] 警告: 无法解析 axis 值，默认使用 'z' 轴")
                return 'z'

        except Exception as e:
            print(f"[KinematicsEngine] 解析 URDF 末端轴失败: {e}，默认使用 'z' 轴")
            return 'z'

    def get_end_effector_axis(self) -> str:
        """
        获取末端执行器的旋转轴（兼容旧代码，等于 flange_axis）

        返回:
            str: 'x', 'y' 或 'z'
        """
        return self._flange_axis

    def _parse_urdf_axes(self, urdf_path: str):
        """
        从 URDF 解析 flange 的旋转轴。

        tool_axis 直接使用 flange_axis（joint6 的旋转轴），
        因为 TCP 偏移沿 flange 旋转轴方向。

        注意：此方法保留仅用于 UI 显示参考，不再用于 TCP 计算。
        """
        # 检测逻辑保持不变（用于 UI 显示 flange_axis）
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()

            # 找 flange 和 tool0 固定关节（仅用于显示，不影响 TCP 计算）
            flange_to_tool0_rpy = None
            for joint in root.findall('joint'):
                parent_link = joint.find('parent').get('link') if joint.find('parent') is not None else ''
                child_link = joint.find('child').get('link') if joint.find('child') is not None else ''
                joint_type = joint.get('type', '')

                if joint_type == 'fixed':
                    if 'flange' in parent_link.lower() and 'tool0' in child_link.lower():
                        origin = joint.find('origin')
                        if origin is not None:
                            rpy_str = origin.get('rpy', '0 0 0')
                            flange_to_tool0_rpy = [float(x) for x in rpy_str.split()]

            # tool_axis 直接使用 flange_axis
            self._tool_axis = self._flange_axis
            self._tool0_axis = self._tool_axis  # 保留兼容

            print(f"[KinematicsEngine] 旋转轴解析结果:")
            print(f"  joint6/flange 轴: {self._flange_axis}  (用于 Tool 偏移方向)")
            print(f"  tool 轴: {self._tool_axis}")

        except Exception as e:
            print(f"[KinematicsEngine] 解析 URDF 旋转轴失败: {e}，使用默认值")
            self._flange_axis = self._end_effector_axis
            self._tool_axis = self._end_effector_axis
            self._tool0_axis = self._end_effector_axis

    def _rotate_axis_by_rpy(self, axis_name: str, rpy: list) -> str:
        """
        将一个轴名经过给定的 rpy 旋转后，映射回带符号的主轴名。

        返回格式: 'x', 'y', 'z', 'x-', 'y-', 'z-'
        """
        axis_map = {'x': np.array([1., 0., 0.]), 'y': np.array([0., 1., 0.]), 'z': np.array([0., 0., 1.])}
        vec = axis_map.get(axis_name, axis_map['z'])

        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', rpy)
        new_vec = rot.apply(vec)

        abs_vals = np.abs(new_vec)
        max_idx = np.argmax(abs_vals)
        axes = ['x', 'y', 'z']
        sign = '-' if new_vec[max_idx] < 0 else ''
        return axes[max_idx] + sign

    def _build_se3(self, xyz, rpy) -> np.ndarray:
        """将 xyz (m) + rpy (rad) 转换为 4x4 SE3 矩阵"""
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', rpy).as_matrix()
        t = np.array(xyz).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:] = t
        return T

    def _compute_tcp_from_params(self) -> np.ndarray:
        """根据当前 _params 计算 T_flange_tcp = flange @ tool"""
        T = (self._build_se3(self._params.flange_xyz, self._params.flange_rpy) @
             self._build_se3(self._params.tool_xyz,   self._params.tool_rpy))
        return T

    def set_flange_tool_params(self, params: FlangeToolParams):
        """从 FlangeToolParams 计算并设置 T_flange_tcp"""
        self._params = params
        self.T_flange_tcp = self._compute_tcp_from_params()
        print(f"[KinematicsEngine] Flange/Tool 参数已更新:")
        print(f"  flange: xyz={params.flange_xyz}, rpy={params.flange_rpy}")
        print(f"  tool:   xyz={params.tool_xyz},   rpy={params.tool_rpy}")
        print(f"  T_flange_tcp =\n{self.T_flange_tcp}")
        print(f"  旋转矩阵 R =\n{self.T_flange_tcp[:3, :3]}")
        print(f"  平移向量 t = [{self.T_flange_tcp[0,3]}, {self.T_flange_tcp[1,3]}, {self.T_flange_tcp[2,3]}]")

    def get_flange_tool_params(self) -> FlangeToolParams:
        """返回当前的 FlangeToolParams"""
        return self._params

    def get_flange_axis(self) -> str:
        return self._flange_axis

    def get_tool0_axis(self) -> str:
        return self._tool0_axis

    def get_tool_axis(self) -> str:
        return self._tool_axis

    def set_joint_positions(self, q: np.ndarray):
        """
        设置当前机器人的关节角度状态
        """
        assert len(q) == self.nq, f"关节角度长度 {len(q)} 与模型不匹配 {self.nq}"
        # 自动应用限位钳制
        q_safe = self._apply_joint_limits(q)
        self.current_q = q_safe.copy()

    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """将关节角度钳制在模型定义的物理限位内"""
        lower, upper = self.get_joint_limits()
        return np.clip(q, lower, upper)

    def get_joint_limits(self):
        """获取模型定义的关节限位 (弧度)"""
        # Pinocchio 自动从 URDF 中解析这些值
        return self.model.lowerPositionLimit, self.model.upperPositionLimit

    def get_robot_metadata(self) -> dict:
        """
        从 URDF 中提取机器人元数据，供 UI 层动态生成控件使用。

        返回:
            dict: 包含以下键的字典
                - nq: 关节配置空间维度
                - nv: 速度空间维度
                - joint_names: 关节名称列表（排除 universe / fixed joints）
                - lower_limits: 位置下限数组（连续关节为 [-pi, pi]）
                - upper_limits: 位置上限数组
                - joint_types: 关节类型列表
                - raw_lower_limits: 原始下限数组（用于连续关节判断）
                - raw_upper_limits: 原始上限数组
        """
        joint_names = []
        joint_types = []
        lower_limits = []
        upper_limits = []
        raw_lower_limits = []
        raw_upper_limits = []

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]

            # 跳过固定关节（nq=0）：它们不在配置空间中
            if joint.nq == 0:
                continue

            name = self.model.names[i]
            joint_type = joint.shortname() if hasattr(joint, 'shortname') else 'unknown'

            # 连续关节（SO(2)，nq=2）使用 cos/sin 表示，无界，UI 工作范围设为 [-π, π]
            if joint.nq == 1:
                raw_lower = self.model.lowerPositionLimit[joint.idx_q]
                raw_upper = self.model.upperPositionLimit[joint.idx_q]
                # 只钳制 Pinocchio 的极端默认值（如无界时为 1e19），合法的工业限位（如 6.98 rad）必须透传
                safe_lower = raw_lower if raw_lower > -100 else -np.pi
                safe_upper = raw_upper if raw_upper < 100 else np.pi
                raw_lower_limits.append(raw_lower)
                raw_upper_limits.append(raw_upper)
            elif joint.nq == 2:
                safe_lower = -np.pi
                safe_upper = np.pi
                raw_lower_limits.append(np.pi)   # 标记为无界，供 UI 判断
                raw_upper_limits.append(-np.pi)  # (上<下 标志连续关节)
            else:
                # 标准 URDF 关节不会走到这里
                safe_lower = -np.pi
                safe_upper = np.pi
                raw_lower_limits.append(safe_lower)
                raw_upper_limits.append(safe_upper)

            joint_names.append(name)
            joint_types.append(joint_type)
            lower_limits.append(safe_lower)
            upper_limits.append(safe_upper)

        return {
            'nq': self.nq,
            'nv': self.nv,
            'joint_names': joint_names,
            'joint_types': joint_types,
            'lower_limits': np.array(lower_limits),
            'upper_limits': np.array(upper_limits),
            'raw_lower_limits': np.array(raw_lower_limits),
            'raw_upper_limits': np.array(raw_upper_limits),
        }

    def update_q_from_angles(self, angles: list):
        """
        将 UI 滑块返回的标量角度列表安全转换为 Pinocchio q 向量。

        这是 UI 与 Pinocchio 模型之间的防火墙，负责处理：
        - 连续关节（nq=2）的 cos/sin 映射
        - q 向量长度为 nq（而非活动关节数）
        - 限位钳制

        参数:
            angles: 标量关节角度列表（长度 = 活动关节数，与 UI 滑块一一对应）
        """
        q = pin.neutral(self.model)  # 自动创建大小为 nq 的向量，cos/sin 已初始化
        angle_idx = 0

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            if joint.nq == 0 or angle_idx >= len(angles):
                continue

            theta = angles[angle_idx]

            if joint.nq == 1:
                # 标准单自由度关节：直接存储标量角度
                q[joint.idx_q] = theta
            elif joint.nq == 2:
                # 连续关节（SO(2)）：存储 cos 和 sin 分量
                q[joint.idx_q] = np.cos(theta)
                q[joint.idx_q + 1] = np.sin(theta)

            angle_idx += 1

        # 应用限位钳制（连续关节的 [-pi, pi] 限位不影响 cos/sin 值）
        q_safe = self._apply_joint_limits(q)
        self.current_q = q_safe.copy()

        # 确保 FK 在 q 更新后被重新计算
        pin.forwardKinematics(self.model, self.data, self.current_q)

    def forward_kinematics(self, q: np.ndarray, joint_name: str = None) -> np.ndarray:
        """
        正运动学求解

        参数:
            q: 关节角度数组 (nq,)
            joint_name: 要查询的关节/连杆名称，默认返回末端执行器

        返回:
            4x4 齐次变换矩阵，表示该关节在在世界坐标系下的位姿
        """
        assert len(q) == self.nq, f"关节角度长度 {len(q)} 与模型不匹配 {self.nq}"

        # Pinocchio 正运动学计算
        # 会自动计算所有关节的位姿并存储在 data.oMi 中
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # 获取末端执行器（默认使用最后一个关节）
        if joint_name is None:
            # IK 求解器以 tool_frame_id 为末端约束。FK、验证和渲染必须读取
            # 同一个 frame，不能用最后一个 joint 的 oMi 代替；部分工业 URDF
            # 在 joint6 与 tool0 之间还有固定法兰变换，两者可能相差数十厘米。
            oMf = self.data.oMf[self.tool_frame_id]
            T_tcp_world = oMf.homogeneous.copy() @ self.effective_tcp
            return T_tcp_world
        else:
            # 根据名称查找关节
            frame_id = self.model.getFrameId(joint_name)
            oMf = self.data.oMf[frame_id]
            return oMf.homogeneous.copy() @ self.effective_tcp

    def compute_jacobian(self, q: np.ndarray, joint_name: str = None) -> np.ndarray:
        """
        计算几何雅可比矩阵

        参数:
            q: 关节角度数组
            joint_name: 末端关节名称

        返回:
            6 x nv 的雅可比矩阵
            前3行: 线速度 Jacobian (v)
            后3行: 角速度 Jacobian (omega)
        """
        assert len(q) == self.nq, f"关节角度长度 {len(q)} 与模型不匹配 {self.nq}"

        # 获取末端执行器 frame ID
        if joint_name is None:
            frame_id = self.model.getFrameId(self.model.names[-1])
        else:
            frame_id = self.model.getFrameId(joint_name)

        # 计算雅可比矩阵
        # Jacobian 是在世界坐标系下描述的
        J = pin.computeFrameJacobian(self.model, self.data, q, frame_id)

        return J.copy()

    # ==================== 坐标系变换辅助方法 ====================

    def _compute_robot_target_pose(self, target_pose_ocf: np.ndarray) -> np.ndarray:
        """
        将 OCF 坐标系下的目标位姿转换为机器人局部坐标系(RCF)。

        物理意义:
            - OCF 坐标 → WCF 坐标 → RCF 坐标
            - 完整变换链: T_rcf = T_wcf_rcf^-1 @ T_wcf_ocf @ T_ocf
        """
        return ocf_to_rcf(target_pose_ocf, self.T_wcf_rcf, self.T_wcf_ocf)

    # ==================== 构型辅助方法 ====================

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        """
        将单个角度归一化到 [-π, π] 区间。

        参数:
            angle: 输入角度（弧度）

        返回:
            float: 归一化后的角度
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def is_near_limit(self, q: np.ndarray, margin_deg: float = 5.0) -> bool:
        """
        检查当前关节角度是否距离任一限位边界小于安全裕度。

        参数:
            q: 关节角度向量 (nq,)
            margin_deg: 安全裕度（度），默认 5°

        返回:
            bool: True 表示至少有一个关节接近限位
        """
        margin_rad = np.deg2rad(margin_deg)

        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        if lower is None or len(lower) == 0:
            lower = -np.pi * np.ones(self.nq)
        if upper is None or len(upper) == 0:
            upper = np.pi * np.ones(self.nq)

        distance_to_lower = q - lower
        distance_to_upper = upper - q

        return bool(np.any(distance_to_lower < margin_rad) or
                    np.any(distance_to_upper < margin_rad))

    def generate_alternative_seeds(self, q_current: np.ndarray) -> list:
        """
        为 6 轴机器人生成备用构型种子点列表。

        包含标准的腕部翻转（Wrist Flip）逻辑：q4+π, -q5, q6+π。
        还包含中间关节偏移、零位偏移等启发式种子点以提高求解鲁棒性。

        参数:
            q_current: 当前关节角度向量 (nq,)

        返回:
            list: 备用种子点列表（每个元素为 np.ndarray）
        """
        seeds = []
        q_flip = q_current.copy()

        # 腕部翻转（Wrist Flip）：适用于 6 轴机器人
        # 轴 4: +π, 轴 5: 取反, 轴 6: +π
        if self.nq >= 6:
            q_flip[3] = self.wrap_to_pi(q_flip[3] + np.pi)
            q_flip[4] = -q_flip[4]
            q_flip[5] = self.wrap_to_pi(q_flip[5] + np.pi)
        elif self.nq >= 3:
            for idx in range(max(0, self.nq - 3), self.nq):
                if idx % 3 == 0:
                    q_flip[idx] = self.wrap_to_pi(q_flip[idx] + np.pi)
                elif idx % 3 == 1:
                    q_flip[idx] = -q_flip[idx]
        seeds.append(self._apply_joint_limits(q_flip))

        # 启发式种子：中间关节偏移 ±π/4
        for joint_idx in range(min(3, self.nq)):
            for sign in [1.0, -1.0]:
                q_offset = q_current.copy()
                q_offset[joint_idx] = self.wrap_to_pi(
                    q_offset[joint_idx] + sign * np.pi / 4
                )
                seeds.append(self._apply_joint_limits(q_offset))

        # 启发式种子：零位姿态（安全的起始构型）
        q_neutral = np.zeros(self.nq)
        seeds.append(self._apply_joint_limits(q_neutral))

        # 去重（保留顺序）
        unique_seeds = []
        for seed in seeds:
            if not any(np.allclose(seed, s, atol=1e-4) for s in unique_seeds):
                unique_seeds.append(seed)

        return unique_seeds

    # ==================== 关节限位 ====================

    @staticmethod
    def create_transform(position: np.ndarray, rotation: np.ndarray = None) -> np.ndarray:
        """
        创建 4x4 齐次变换矩阵

        参数:
            position: [x, y, z] 位置向量
            rotation: 3x3 旋转矩阵，如果为 None 则使用单位矩阵

        返回:
            4x4 齐次变换矩阵
        """
        T = np.eye(4)
        T[:3, 3] = position

        if rotation is not None:
            T[:3, :3] = rotation

        return T

    @staticmethod
    def rotation_matrix(axis: str, angle: float) -> np.ndarray:
        """
        创建基本旋转矩阵

        参数:
            axis: 'x', 'y', 或 'z'
            angle: 旋转角度（弧度）
        """
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ])
        elif axis == 'y':
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ])
        elif axis == 'z':
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"Unknown axis: {axis}")


# ==================== 测试模块 ====================

if __name__ == '__main__':
    import os
    import sys
    import numpy as np

    print("=" * 60)
    print("KinematicsEngine 通用化运动学引擎测试 (Data-Driven)")
    print("=" * 60)

    # ============ 1. 自动寻找 URDF 文件 ============
    # Use the authored public demo model, independent of the current directory.
    urdf_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'examples', 'assets', 'urdf', 'demo_six_axis.urdf'
    )
    if not os.path.exists(urdf_file):
        print(f"\n[错误] URDF 文件不存在: {urdf_file}")
        sys.exit(1)
    print(f"\n[1] 找到并使用模型: {urdf_file}")

    # ============ 2. 实例化引擎 ============
    engine = KinematicsEngine(urdf_file, verbose=True)

    # ============ 3. 动态生成安全初始姿态 (Dynamic Seed) ============
    lower_limits = engine.model.lowerPositionLimit
    upper_limits = engine.model.upperPositionLimit

    # 处理极限值：如果 (upper - lower) > 2*pi，说明是无极限的连续关节，设为 0
    q_init = np.zeros(engine.nq)
    for i in range(engine.nq):
        lower = lower_limits[i] if i < len(lower_limits) else -np.pi
        upper = upper_limits[i] if i < len(upper_limits) else np.pi

        if upper - lower > 2 * np.pi:
            q_init[i] = 0.0  # 连续关节设为 0
        else:
            q_init[i] = (upper + lower) / 2.0  # 取中间值

    print(f"\n[3] 生成的 q_init 数组:")
    print(f"    {q_init}")
    print(f"    (关节数量: {engine.nq})")

    # ============ 4. 动态生成 100% 可达的测试轨迹 ============
    T_safe = engine.forward_kinematics(q_init)
    position_safe = T_safe[:3, 3]
    rotation_safe = T_safe[:3, :3]

    print(f"\n[4] 基准安全位姿 T_safe 位置坐标:")
    print(f"    X: {position_safe[0]:.6f} m")
    print(f"    Y: {position_safe[1]:.6f} m")
    print(f"    Z: {position_safe[2]:.6f} m")

    # 沿 Y 轴生成 5 个等距目标点
    trajectory = []
    step_size = 0.05  # 每次移动 0.05m
    for i in range(5):
        target_pose = np.eye(4)
        target_pose[:3, 3] = position_safe + np.array([0, i * step_size, 0])
        target_pose[:3, :3] = rotation_safe
        trajectory.append(target_pose)

    print(f"\n    生成了 {len(trajectory)} 个测试目标点 (沿 Y 轴)")
    for i, pose in enumerate(trajectory):
        print(f"    点 {i+1}: ({pose[0,3]:.4f}, {pose[1,3]:.4f}, {pose[2,3]:.4f})")

    # ============ 5. 批量 IK 求解 ============
    engine.set_transforms(np.eye(4), np.eye(4))

    from .auto_manager import get_manager
    manager = get_manager()
    SolverClass = manager.get_algorithm_class('伪逆直接求解法', prefix='ik')
    solver = SolverClass(kinematics_engine=engine, alpha=0.5, max_iterations=500, tolerance=1e-5)

    results = []
    q_current = q_init.copy()
    for idx, T_ocf in enumerate(trajectory):
        success, q_sol, error = solver.solve(T_ocf, q_init=q_current)
        results.append({'index': idx, 'success': success, 'q': q_sol, 'error': error})
        if success:
            q_current = q_sol.copy()

    # ============ 6. 打印结果报表 ============
    print("\n" + "=" * 70)
    print("IK 求解结果汇总")
    print("=" * 70)

    # 表头
    header = f"{'点号':^6}|{'状态':^8}|{'误差(m)':^14}|{'关节角度 (rad)'}"
    print(header)
    print("-" * 70)

    success_count = 0
    for result in results:
        status = "V 成功" if result['success'] else "X 失败"
        q_str = np.array2string(result['q'], precision=3, suppress_small=True)
        print(f"{result['index']+1:^6}|{status:^8}|{result['error']:^14.6f}|{q_str}")
        if result['success']:
            success_count += 1

    print("-" * 70)
    print(f"收敛率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")

    # 计算相邻点之间的最大关节变化角度
    print("\n" + "-" * 70)
    print("关节连续性检查（相邻点之间最大关节变化）:")
    max_total_change = 0.0
    for i in range(1, len(results)):
        delta_q = np.abs(results[i]['q'] - results[i-1]['q'])
        max_delta = np.max(delta_q)
        max_total_change = max(max_total_change, max_delta)
        print(f"  点{i} -> 点{i+1}: 最大关节变化 = {max_delta:.4f} rad ({np.degrees(max_delta):.2f} deg)")

    print("-" * 70)
    print(f"轨迹最大关节跳变: {max_total_change:.4f} rad")
    if max_total_change < 0.5:
        print("  [OK] 关节变化平滑，连续性良好")
    else:
        print("  [警告] 关节跳变较大，请检查轨迹规划")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
