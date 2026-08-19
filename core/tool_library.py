"""
tool_library.py - 工业标准刀具库系统

设计原则（对标 ABB RAPID tooldata 架构）：
    1. 物理模型不可变：URDF/Kinematic Tree 以 tool0（法兰盘）为终点，只读。
    2. ToolData 数据解耦：刀具作为独立数据结构，在调用计算函数时作为参数传入。
    3. 运动学链动态扩展：用 Pinocchio 原生 Frame 机制动态挂载 TCP，
       IK 雅可比直接针对 TCP 计算，无需手动 T_target @ inv(T_tcp) 矩阵乘法。

YAML 工具库格式：
    tools:
      - name: "welding_torch"
        tcp_position: [0.0, 0.0, 0.35]  # 米
        tcp_orientation: [3.14159, 0.0, 0.0]  # Rx Ry Rz (rad), 对标工业惯例
        calibration_method: "6-point"  # 3-point / 6-point / CAD / manual
        description: "Mig welding torch, 350mm reach"
        mesh: "meshes/welding_torch.stl"
"""
import os
import yaml
import numpy as np
import pinocchio as pin
from typing import Optional, Dict, List
from dataclasses import dataclass, field, asdict


class ToolData:
    """
    工业标准刀具数据结构（对标 ABB RAPID tooldata）。

    与旧的 FlangeToolParams 的区别：
        - FlangeToolParams 用 flange/tool0/tool 三组独立的 xyz+rpy 参数，
          这是人为拆分，冗余且容易出错（法兰盘偏移属于 URDF 模型的一部分）。
        - ToolData 只维护一个 T_flange_tcp（法兰盘到 TCP 的 4x4 齐次变换矩阵），
          符合工业标准的数据结构。
    """

    def __init__(
        self,
        name: str = "default_tool",
        tcp_position: Optional[List[float]] = None,
        tcp_orientation: Optional[List[float]] = None,
        T_flange_tcp: Optional[np.ndarray] = None,
        calibration_method: str = "manual",
        description: str = "",
        mesh_path: str = "",
        mass: float = 0.0,
        com: Optional[List[float]] = None,
        inertia: Optional[List[List[float]]] = None,
    ):
        """
        参数:
            name: 刀具名称
            tcp_position: TCP 位置 [x, y, z]，单位米。如果与 T_flange_tcp 同时提供，
                         以 T_flange_tcp 为准。
            tcp_orientation: TCP 方向 [Rx, Ry, Rz]，单位弧度（工业惯例）。
                            配合 tcp_position 使用时为先平移后旋转。
            T_flange_tcp: 法兰盘到 TCP 的 4x4 齐次变换矩阵。如果提供，优先使用。
            calibration_method: 标定方法。"3-point" / "6-point" / "CAD" / "manual"
            description: 刀具描述
            mesh_path: 刀具 STL 几何文件路径
            mass: 刀具质量（kg），用于动力学仿真
            com: 刀具质心位置 [x, y, z]，单位米
            inertia: 刀具惯性张量 3x3，kg*m^2
        """
        self.name = name
        self.calibration_method = calibration_method
        self.description = description
        self.mesh_path = mesh_path
        self.mass = mass

        # 确定 T_flange_tcp
        if T_flange_tcp is not None:
            self.T_flange_tcp = T_flange_tcp.copy()
        elif tcp_position is not None or tcp_orientation is not None:
            self.T_flange_tcp = self._build_from_position_orientation(
                tcp_position or [0.0, 0.0, 0.0],
                tcp_orientation or [0.0, 0.0, 0.0]
            )
        else:
            self.T_flange_tcp = np.eye(4)

        # 动力学参数
        self.com = np.array(com if com is not None else [0.0, 0.0, 0.0])
        if inertia is not None:
            self.inertia = np.array(inertia)
        else:
            self.inertia = np.zeros((3, 3))

    @staticmethod
    def _build_from_position_orientation(
        position: List[float],
        orientation: List[float]
    ) -> np.ndarray:
        """从位置和欧拉角（Rx Ry Rz）构建 4x4 齐次变换矩阵"""
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', orientation).as_matrix()
        t = np.array(position).reshape(3, 1)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:] = t
        return T

    @classmethod
    def from_flange_tool_params(cls, params, name: str = "custom_tool") -> 'ToolData':
        """
        从 FlangeToolParams（旧接口）创建 ToolData（新接口）。

        变换链: flange -> tool0 -> tool(TCP)

        参数:
            params: FlangeToolParams 实例
            name: 刀具名称
        """
        from scipy.spatial.transform import Rotation as R_scipy

        def build_rpy(rpy_list):
            rot = R_scipy.from_euler('xyz', rpy_list).as_matrix()
            return rot

        R_flange = build_rpy(params.flange_rpy)
        R_tool = build_rpy(params.tool_rpy)

        R_total = R_flange @ R_tool

        t_flange = np.array(params.flange_xyz).reshape(3, 1)
        t_tool = np.array(params.tool_xyz).reshape(3, 1)

        # T_flange_tcp = T_flange @ T_tool
        # = [R_flange | t_flange] @ [R_tool | t_tool]
        # = [R_total | t_flange + R_flange @ t_tool]
        t03 = t_flange + R_flange @ t_tool

        T_flange_tcp = np.eye(4)
        T_flange_tcp[:3, :3] = R_total
        T_flange_tcp[:3, 3:] = t03.reshape(3, 1)

        return cls(name=name, T_flange_tcp=T_flange_tcp)

    @property
    def se3(self) -> pin.SE3:
        """返回 Pinocchio SE3 格式，用于底层 Frame 挂载"""
        return pin.SE3(self.T_flange_tcp.copy())

    @property
    def tcp_position(self) -> np.ndarray:
        """TCP 位置 [x, y, z]，单位米"""
        return self.T_flange_tcp[:3, 3].copy()

    @property
    def tcp_rotation(self) -> np.ndarray:
        """TCP 旋转矩阵 3x3"""
        return self.T_flange_tcp[:3, :3].copy()

    def to_dict(self) -> dict:
        """序列化为字典（用于 YAML 持久化）"""
        return {
            'name': self.name,
            'tcp_position': self.tcp_position.tolist(),
            'tcp_orientation': [0.0, 0.0, 0.0],  # placeholder, computed from rotation
            'calibration_method': self.calibration_method,
            'description': self.description,
            'mesh': self.mesh_path,
            'mass': self.mass,
            'com': self.com.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ToolData':
        """从字典反序列化"""
        pos = data.get('tcp_position', [0.0, 0.0, 0.0])
        ori = data.get('tcp_orientation', [0.0, 0.0, 0.0])
        return cls(
            name=data.get('name', 'unknown'),
            tcp_position=pos,
            tcp_orientation=ori,
            calibration_method=data.get('calibration_method', 'manual'),
            description=data.get('description', ''),
            mesh_path=data.get('mesh', ''),
            mass=data.get('mass', 0.0),
            com=data.get('com', [0.0, 0.0, 0.0]),
        )

    def __repr__(self) -> str:
        pos = self.tcp_position
        return (f"ToolData(name='{self.name}', "
                f"position=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "
                f"calibration='{self.calibration_method}')")


class ToolLibrary:
    """
    工具管理器 — T_flange_tcp 的唯一真实数据源（SSOT）。

    职责：
        1. 持有所有已注册的 ToolData 实例
        2. 管理当前激活的刀具
        3. 提供 YAML 持久化（导入/导出）
        4. 供 KinematicsEngine、DataPostProcessor、RenderEngine 等共享引用

    使用方式：
        tool_library = ToolLibrary()
        tool_library.add_tool(ToolData("welding_torch", tcp_position=[0, 0, 0.35]))
        tool_library.set_current_tool("welding_torch")

        # 任何需要刀具数据的模块共享同一个 tool_library 实例
        engine = KinematicsEngine(urdf_path, tool_library=tool_library)
        processor = DataPostProcessor(..., tool_library=tool_library)
    """

    def __init__(self):
        self._tools: Dict[str, ToolData] = {}
        self._current_tool: Optional[ToolData] = None
        self._default_tool = ToolData("empty_tool")

    def add_tool(self, tool: ToolData) -> None:
        """注册一个刀具到工具库"""
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        """从工具库移除一个刀具"""
        if name in self._tools:
            del self._tools[name]
        if self._current_tool is not None and self._current_tool.name == name:
            self._current_tool = self._default_tool

    def get_tool(self, name: str) -> Optional[ToolData]:
        """按名称获取刀具"""
        return self._tools.get(name)

    def get_current_tool(self) -> ToolData:
        """获取当前激活的刀具（永远不为 None）"""
        if self._current_tool is None:
            self._current_tool = self._default_tool
        return self._current_tool

    def set_current_tool(self, name: str) -> bool:
        """
        切换当前激活的刀具。

        返回:
            bool: 是否切换成功（刀具存在则返回 True）
        """
        tool = self._tools.get(name)
        if tool is not None:
            self._current_tool = tool
            return True
        return False

    def set_current_tool_data(self, tool: ToolData) -> None:
        """直接设置当前激活的刀具（无需先注册）"""
        self._current_tool = tool

    def list_tools(self) -> List[str]:
        """列出所有已注册的刀具名称"""
        return list(self._tools.keys())

    @property
    def T_flange_tcp(self) -> np.ndarray:
        """获取当前刀具的 T_flange_tcp 矩阵（便利属性）"""
        return self.get_current_tool().T_flange_tcp.copy()

    def save_to_yaml(self, path: str) -> None:
        """
        将工具库导出为 YAML 文件。

        参数:
            path: 输出文件路径
        """
        tools_list = []
        for tool in self._tools.values():
            d = tool.to_dict()
            # 如果有 rotation matrix 且非单位阵，额外保存方向
            if not np.allclose(tool.tcp_rotation, np.eye(3), atol=1e-9):
                from scipy.spatial.transform import Rotation as R_scipy
                ori = R_scipy.from_matrix(tool.tcp_rotation).as_euler('xyz')
                d['tcp_orientation'] = ori.tolist()
            tools_list.append(d)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump({'tools': tools_list}, f, allow_unicode=True, default_flow_style=False)

    def load_from_yaml(self, path: str) -> None:
        """
        从 YAML 文件加载工具库。

        参数:
            path: 输入文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"工具库文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if data is None or 'tools' not in data:
            return

        for tool_dict in data['tools']:
            tool = ToolData.from_dict(tool_dict)
            self.add_tool(tool)

        if self._tools and self._current_tool is None:
            self._current_tool = next(iter(self._tools.values()))
