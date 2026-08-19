"""
毛坯点云生成系统 (Point Cloud Generator)
作为 OLP 平台的独立辅助工具，用于生成标准几何体的毛坯点云。

功能:
- 支持生成: 长方体 (Box)、球体 (Sphere)、圆柱体 (Cylinder)、圆锥体 (Cone)
- 支持计算并显示基于 cm³ 的点云密度
- PyVista 独立预览
- 导出格式: PCD (支持多维属性), PLY (ASCII/Binary), XYZ

版本: v2.0
依赖: PySide6, PyVista, Open3D, NumPy
"""

import sys
import os
import warnings
import numpy as np

# PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QFileDialog, QMessageBox, QRadioButton,
    QButtonGroup, QFrame, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

# PyVista
import pyvista as pv
from pyvistaqt import QtInteractor

# Open3D (用于点云导出)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


# ==================== 逻辑层：点云数学引擎 ====================

class GeometryEngine:
    """
    提供点云生成的数学逻辑与体积计算
    所有输入参数单位: 米 (m)
    输出点云坐标单位: 米 (m)
    """

    # 默认青色 RGB (0, 150, 255) 归一化
    DEFAULT_COLOR = np.array([0.0, 150.0 / 255.0, 1.0])

    @staticmethod
    def generate_box(length: float, width: float, height: float, num_points: int) -> np.ndarray:
        """
        生成长方体体积点云 (均匀分布)
        底部贴齐 Z=0 平面

        参数:
            length: 长度 X (m)
            width:  宽度 Y (m)
            height: 高度 Z (m)
            num_points: 目标点数

        返回:
            (N, 3) numpy array
        """
        x = np.random.uniform(-length / 2, length / 2, num_points)
        y = np.random.uniform(-width / 2, width / 2, num_points)
        z = np.random.uniform(0, height, num_points)
        return np.column_stack([x, y, z])

    @staticmethod
    def generate_sphere(radius: float, num_points: int) -> np.ndarray:
        """
        生成球体体积点云 (拒绝采样保证均匀分布)
        底部贴齐 Z=0 平面

        参数:
            radius: 球半径 (m)
            num_points: 目标点数

        返回:
            (N, 3) numpy array
        """
        points = []
        batch_size = min(num_points * 3, 500000)  # 批量大小限制，避免内存溢出

        while len(points) < num_points:
            batch = np.random.uniform(-radius, radius, (batch_size, 3))
            distances = np.linalg.norm(batch, axis=1)
            valid_mask = distances <= radius
            valid = batch[valid_mask].copy()
            # Z 轴平移，使底部贴齐 Z=0
            valid[:, 2] += radius
            points.extend(valid)

        return np.array(points[:num_points])

    @staticmethod
    def generate_cylinder(radius: float, height: float, num_points: int) -> np.ndarray:
        """
        生成圆柱体体积点云 (拒绝采样)
        底部贴齐 Z=0 平面

        参数:
            radius: 圆柱半径 (m)
            height: 圆柱高度 (m)
            num_points: 目标点数

        返回:
            (N, 3) numpy array
        """
        points = []
        batch_size = min(num_points * 3, 500000)

        while len(points) < num_points:
            batch = np.random.uniform(-radius, radius, (batch_size, 2))
            distances = np.linalg.norm(batch, axis=1)
            valid_mask = distances <= radius
            valid_xy = batch[valid_mask]

            if len(valid_xy) > 0:
                z = np.random.uniform(0, height, len(valid_xy))
                valid_points = np.column_stack([valid_xy[:, 0], valid_xy[:, 1], z])
                points.extend(valid_points)

        return np.array(points[:num_points])

    @staticmethod
    def generate_cone(radius: float, height: float, num_points: int) -> np.ndarray:
        """
        生成圆锥体体积点云 (拒绝采样)
        顶点朝上，底部贴齐 Z=0 平面

        参数:
            radius: 底部半径 (m)
            height: 圆锥高度 (m)
            num_points: 目标点数

        返回:
            (N, 3) numpy array
        """
        points = []
        batch_size = min(num_points * 3, 500000)

        while len(points) < num_points:
            batch_xy = np.random.uniform(-radius, radius, (batch_size, 2))
            batch_z = np.random.uniform(0, height, batch_size)

            # 当前高度处允许的最大半径 (线性插值)
            allowed_radius = radius * (1.0 - batch_z / height)
            dist_xy = np.linalg.norm(batch_xy, axis=1)

            # 筛选在当前高度截面内的点
            valid_mask = dist_xy <= allowed_radius
            valid_points = np.column_stack([
                batch_xy[valid_mask, 0],
                batch_xy[valid_mask, 1],
                batch_z[valid_mask]
            ])
            points.extend(valid_points)

        return np.array(points[:num_points])

    @staticmethod
    def calculate_volume(shape_type: str, params: dict) -> float:
        """
        计算几何体体积

        参数:
            shape_type: 'box', 'sphere', 'cylinder', 'cone'
            params: 参数字典

        返回:
            体积 (m³)
        """
        if shape_type == "box":
            return params['length'] * params['width'] * params['height']
        elif shape_type == "sphere":
            return (4.0 / 3.0) * np.pi * (params['radius'] ** 3)
        elif shape_type == "cylinder":
            return np.pi * (params['radius'] ** 2) * params['height']
        elif shape_type == "cone":
            return (1.0 / 3.0) * np.pi * (params['radius'] ** 2) * params['height']
        return 1.0

    @staticmethod
    def calculate_density(volume_m3: float, num_points: int) -> float:
        """
        计算点云密度

        参数:
            volume_m3: 体积 (m³)
            num_points: 点数

        返回:
            密度 (points/cm³)
        """
        if volume_m3 <= 0:
            return 0.0
        # 1 m³ = 1,000,000 cm³
        volume_cm3 = volume_m3 * 1e6
        return num_points / volume_cm3


# ==================== 逻辑层：点云导出引擎 ====================

class ExportEngine:
    """
    提供多种格式的点云导出支持
    - PCD/PLY: 使用 Open3D (含法向估计和颜色)
    - XYZ: 使用 NumPy 纯文本
    """

    # 默认青色 RGB [0, 150, 255]
    DEFAULT_RGB = [0, 150, 255]

    @staticmethod
    def export_pcd(filepath: str, points: np.ndarray, write_ascii: bool = False) -> tuple[bool, str]:
        """
        导出 PCD 格式 (Point Cloud Data)

        PCD 格式原生支持:
        - VERSION
        - FIELDS x y z normal_x normal_y normal_z rgb
        - SIZE 4 4 4 4 4 4 4
        - TYPE F F F F F F F
        - COUNT 1 1 1 1 1 1 1
        - WIDTH/HEIGHT
        - POINTS
        - DATA ascii/binary

        参数:
            filepath: 保存路径
            points: (N, 3) 点坐标数组
            write_ascii: 是否使用 ASCII 格式

        返回:
            (success, message)
        """
        if not OPEN3D_AVAILABLE:
            return False, "Open3D 未安装，无法导出 PCD。请运行: pip install open3d"

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # 自动估计法向
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=15)

            # 设置默认青色
            colors = np.tile(ExportEngine.DEFAULT_RGB, (len(points), 1)).astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 写入 PCD
            success = o3d.io.write_point_cloud(
                filepath, pcd,
                print_progress=False,
                write_ascii=write_ascii
            )

            if success:
                mode = "ASCII" if write_ascii else "Binary"
                return True, f"PCD ({mode}) 导出成功\n文件: {os.path.basename(filepath)}\n点数: {len(points):,}"
            else:
                return False, "PCD 写入失败"

        except Exception as e:
            return False, f"PCD 导出异常: {str(e)}"

    @staticmethod
    def export_ply(filepath: str, points: np.ndarray, write_ascii: bool = False) -> tuple[bool, str]:
        """
        导出 PLY 格式 (Polygon File Format)

        参数:
            filepath: 保存路径
            points: (N, 3) 点坐标数组
            write_ascii: 是否使用 ASCII 格式

        返回:
            (success, message)
        """
        if not OPEN3D_AVAILABLE:
            return False, "Open3D 未安装，无法导出 PLY。请运行: pip install open3d"

        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # 自动估计法向
            pcd.estimate_normals()
            pcd.orient_normals_consistent_tangent_plane(k=15)

            # 设置默认青色
            colors = np.tile(ExportEngine.DEFAULT_RGB, (len(points), 1)).astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # 写入 PLY
            success = o3d.io.write_point_cloud(
                filepath, pcd,
                print_progress=False,
                write_ascii=write_ascii
            )

            if success:
                mode = "ASCII" if write_ascii else "Binary"
                return True, f"PLY ({mode}) 导出成功\n文件: {os.path.basename(filepath)}\n点数: {len(points):,}"
            else:
                return False, "PLY 写入失败"

        except Exception as e:
            return False, f"PLY 导出异常: {str(e)}"

    @staticmethod
    def export_xyz(filepath: str, points: np.ndarray) -> tuple[bool, str]:
        """
        导出 XYZ 格式 (纯文本坐标)

        参数:
            filepath: 保存路径
            points: (N, 3) 点坐标数组

        返回:
            (success, message)
        """
        try:
            np.savetxt(
                filepath,
                points,
                fmt="%.6f",
                delimiter=" ",
                newline="\n"
            )
            return True, f"XYZ 导出成功\n文件: {os.path.basename(filepath)}\n点数: {len(points):,}"

        except Exception as e:
            return False, f"XYZ 导出异常: {str(e)}"

    @classmethod
    def export(cls, filepath: str, points: np.ndarray, format_type: str, write_ascii: bool = False) -> tuple[bool, str]:
        """
        统一导出入口

        参数:
            filepath: 保存路径
            points: 点坐标数组
            format_type: 'pcd', 'ply', 'xyz'
            write_ascii: ASCII 模式 (仅 PCD/PLY)

        返回:
            (success, message)
        """
        if format_type == "xyz":
            return cls.export_xyz(filepath, points)
        elif format_type == "pcd":
            return cls.export_pcd(filepath, points, write_ascii)
        elif format_type == "ply":
            return cls.export_ply(filepath, points, write_ascii)
        else:
            return False, f"不支持的格式: {format_type}"


# ==================== 工作线程：后台生成 ====================

class PointCloudWorker(QThread):
    """
    后台点云生成线程，避免 UI 冻结
    """
    finished = Signal(object)  # np.ndarray
    error = Signal(str)
    progress = Signal(int, int)  # current, total

    def __init__(self, shape_type: str, params: dict, num_points: int, parent=None):
        super().__init__(parent)
        self._shape_type = shape_type
        self._params = params
        self._num_points = num_points

    def run(self):
        try:
            self.progress.emit(10, 100)
            self.finished.emit(None)  # placeholder
        except Exception as e:
            self.error.emit(str(e))


# ==================== UI 层：主窗口 ====================

class PointCloudGeneratorWindow(QMainWindow):
    """
    毛坯点云生成系统主窗口

    布局:
    ┌─────────────────────────────────────────────────────────┐
    │  标题栏: 毛坯点云生成系统 (OLP Point Cloud Tool)         │
    ├────────────────┬────────────────────────────────────────┤
    │                │                                        │
    │  1. 几何体设置 │                                        │
    │    - 形状选择   │                                        │
    │    - 动态参数   │                                        │
    │                │         PyVista 3D 预览                 │
    │  2. 密度设置    │         - 深色背景 #2b2b2b             │
    │    - 点数       │         - 坐标轴                       │
    │    - 密度显示   │         - 点云渲染                     │
    │                │                                        │
    │  [生成预览]     │                                        │
    │                │                                        │
    │  3. 导出设置    │                                        │
    │    - 格式选择   │                                        │
    │    - ASCII/BIN │                                        │
    │  [导出文件]     │                                        │
    │                │                                        │
    └────────────────┴────────────────────────────────────────┘
    """

    # 形状类型映射
    SHAPE_TYPES = ["box", "sphere", "cylinder", "cone"]
    SHAPE_NAMES = ["长方体 (Box)", "球体 (Sphere)", "圆柱体 (Cylinder)", "圆锥体 (Cone)"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("毛坯点云生成系统 (OLP Point Cloud Tool) v2.0")
        self.setGeometry(100, 100, 1280, 800)

        # 状态
        self.current_points: np.ndarray = None
        self.param_widgets: dict = {}

        # UI 设置
        self._setup_ui()
        self._init_pyvista()

    def _setup_ui(self):
        """初始化 UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 全局布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ========== 左侧控制面板 ==========
        left_panel = self._create_control_panel()
        left_panel.setMinimumWidth(360)
        left_panel.setMaximumWidth(420)
        main_layout.addWidget(left_panel)

        # ========== 右侧预览窗口 ==========
        right_panel = self._create_preview_panel()
        main_layout.addWidget(right_panel, 1)  # 右侧占据更多空间

        # 初始化动态参数 UI
        self._on_shape_changed()

    def _create_control_panel(self) -> QFrame:
        """创建左侧控制面板"""
        panel = QFrame()
        panel.setObjectName("ControlPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # ===== 1. 几何体设置 =====
        shape_group = QGroupBox("1. 几何体设置")
        shape_group.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        shape_layout = QVBoxLayout(shape_group)

        # 形状选择
        form_layout = QFormLayout()
        form_layout.setSpacing(12)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(self.SHAPE_NAMES)
        self.shape_combo.setMinimumHeight(36)
        self.shape_combo.setFont(QFont("Microsoft YaHei", 10))
        self.shape_combo.currentIndexChanged.connect(self._on_shape_changed)
        form_layout.addRow("几何形状:", self.shape_combo)

        shape_layout.addLayout(form_layout)

        # 动态参数区域
        self.dynamic_param_layout = QFormLayout()
        self.dynamic_param_layout.setSpacing(12)
        shape_layout.addLayout(self.dynamic_param_layout)

        layout.addWidget(shape_group)

        # ===== 2. 点云密度设置 =====
        density_group = QGroupBox("2. 点云密度设置")
        density_group.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        density_layout = QFormLayout(density_group)
        density_layout.setSpacing(12)

        self.points_spin = QSpinBox()
        self.points_spin.setRange(100, 10000000)
        self.points_spin.setValue(10000)
        self.points_spin.setSingleStep(1000)
        self.points_spin.setMinimumHeight(36)
        self.points_spin.setFont(QFont("Microsoft YaHei", 10))
        self.points_spin.valueChanged.connect(self._update_density_display)
        density_layout.addRow("目标点数:", self.points_spin)

        # 密度显示
        density_info_layout = QHBoxLayout()
        self.density_label = QLabel("0.00 points/cm³")
        self.density_label.setFont(QFont("Consolas", 11, QFont.Bold))
        self.density_label.setStyleSheet("""
            color: #00BFFF;
            background-color: #1E1E1E;
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #00BFFF;
        """)
        density_info_layout.addWidget(self.density_label)
        density_info_layout.addStretch()
        density_layout.addRow("预估密度:", density_info_layout)

        # 体积信息
        self.volume_label = QLabel("0.000000 m³")
        self.volume_label.setFont(QFont("Consolas", 9))
        self.volume_label.setStyleSheet("color: #888888;")
        density_layout.addRow("几何体积:", self.volume_label)

        layout.addWidget(density_group)

        # ===== 生成按钮 =====
        self.btn_generate = QPushButton("🔧 生成点云并预览")
        self.btn_generate.setMinimumHeight(48)
        self.btn_generate.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:pressed {
                background-color: #1E8449;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
        """)
        self.btn_generate.clicked.connect(self._on_generate)
        layout.addWidget(self.btn_generate)

        # ===== 3. 导出设置 =====
        export_group = QGroupBox("3. 点云导出")
        export_group.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)

        # 格式选择
        format_label = QLabel("导出格式:")
        format_label.setFont(QFont("Microsoft YaHei", 9))
        export_layout.addWidget(format_label)

        self.export_combo = QComboBox()
        self.export_combo.addItems([
            ".pcd - Point Cloud Data (含颜色/法向)",
            ".ply - Polygon File Format (ASCII/Binary)",
            ".xyz - 纯文本坐标"
        ])
        self.export_combo.setMinimumHeight(36)
        self.export_combo.setFont(QFont("Microsoft YaHei", 10))
        export_layout.addWidget(self.export_combo)

        # 模式选择 (PLY/PCD)
        mode_label = QLabel("编码模式:")
        mode_label.setFont(QFont("Microsoft YaHei", 9))
        export_layout.addWidget(mode_label)

        mode_layout = QHBoxLayout()
        self.ascii_radio = QRadioButton("ASCII (文本)")
        self.ascii_radio.setFont(QFont("Microsoft YaHei", 9))
        self.ascii_radio.setToolTip("文本格式，文件较大，但可用文本编辑器打开")
        self.binary_radio = QRadioButton("Binary (二进制)")
        self.binary_radio.setFont(QFont("Microsoft YaHei", 9))
        self.binary_radio.setChecked(True)
        self.binary_radio.setToolTip("二进制格式，推荐使用，文件较小")

        self.data_mode_group = QButtonGroup(self)
        self.data_mode_group.addButton(self.ascii_radio, 1)
        self.data_mode_group.addButton(self.binary_radio, 2)

        mode_layout.addWidget(self.ascii_radio)
        mode_layout.addWidget(self.binary_radio)
        mode_layout.addStretch()
        export_layout.addLayout(mode_layout)

        # 导出按钮
        self.btn_export = QPushButton("📥 导出点云文件")
        self.btn_export.setMinimumHeight(44)
        self.btn_export.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #1F618D;
            }
            QPushButton:disabled {
                background-color: #7F8C8D;
            }
        """)
        self.btn_export.clicked.connect(self._on_export)
        export_layout.addWidget(self.btn_export)

        # Open3D 状态提示
        if not OPEN3D_AVAILABLE:
            warning_label = QLabel("⚠ Open3D 未安装，PCD/PLY 导出不可用")
            warning_label.setStyleSheet("color: #E74C3C; font-size: 9pt;")
            export_layout.addWidget(warning_label)

        layout.addWidget(export_group)

        # 底部弹性空间
        layout.addStretch()

        return panel

    def _create_preview_panel(self) -> QFrame:
        """创建右侧 PyVista 预览面板"""
        panel = QFrame()
        panel.setObjectName("PreviewPanel")
        panel.setStyleSheet("""
            #PreviewPanel {
                background-color: #2b2b2b;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏
        header = QLabel("3D 点云预览")
        header.setStyleSheet("""
            color: #CCCCCC;
            background-color: #1E1E1E;
            padding: 8px 15px;
            border-radius: 4px 4px 0 0;
            font-size: 10pt;
        """)
        layout.addWidget(header)

        # PyVista QtInteractor
        self.plotter = QtInteractor(panel)
        self.plotter.set_background("#2b2b2b")
        self.plotter.add_axes(
            line_width=2,
            color="#AAAAAA",
            x_color="#FF5555",
            y_color="#55FF55",
            z_color="#5599FF",
            xlabel="X (m)",
            ylabel="Y (m)",
            zlabel="Z (m)"
        )
        layout.addWidget(self.plotter)

        return panel

    def _init_pyvista(self):
        """初始化 PyVista 场景"""
        # 禁用深度剥离，避免半透明问题 (使用正确的方法)
        self.plotter.renderer.disable_depth_peeling()

        # 添加参考网格
        grid = pv.Plane(
            i_size=1.0, j_size=1.0,
            i_resolution=10, j_resolution=10
        )
        self.plotter.add_mesh(
            grid,
            name="ground_grid",
            color="#3a3a3a",
            opacity=0.3,
            show_edges=False
        )

        # 初始相机位置
        self.plotter.camera_position = [(0.3, -0.5, 0.4), (0, 0.05, 0), (0, 0, 1)]
        self.plotter.render()

    def _clear_layout(self, layout):
        """
        [核心架构规则]
        安全清理布局中的动态控件，防止底层 C++ 控件堆叠重影

        步骤:
        1. 从布局中取出 item
        2. 对每个 widget: 先 setParent(None) 脱离视觉树，再 deleteLater()
        3. 对嵌套 layout: 递归清理，最后设置父对象为 None 并 deleteLater()
        """
        if layout is None:
            return

        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue

            widget = item.widget()
            if widget is not None:
                # 关键步骤 1: 先脱离 Qt 视觉树
                widget.setParent(None)
                # 关键步骤 2: 再排队删除
                widget.deleteLater()
            else:
                # 处理嵌套的子布局
                child_layout = item.layout()
                if child_layout is not None:
                    self._clear_layout(child_layout)
                    child_layout.setParent(None)
                    child_layout.deleteLater()

    def _create_double_spinbox(self, value: float, suffix: str = " m",
                               min_val: float = 0.001, max_val: float = 10.0,
                               decimals: int = 4) -> QDoubleSpinBox:
        """
        创建标准化的米(m)单位输入框

        参数:
            value: 默认值
            suffix: 后缀显示
            min_val: 最小值 (1mm = 0.001m)
            max_val: 最大值 (10m)
            decimals: 小数位数

        返回:
            QDoubleSpinBox 实例
        """
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.01)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumHeight(36)
        spin.setFont(QFont("Consolas", 10))
        spin.valueChanged.connect(self._update_density_display)
        return spin

    def _on_shape_changed(self):
        """
        形状切换时，动态更新参数输入表单

        遵循架构规则:
        - 切换前先完全清理旧布局
        - 重建参数控件
        - 更新密度显示
        """
        self._clear_layout(self.dynamic_param_layout)
        self.param_widgets.clear()

        shape_idx = self.shape_combo.currentIndex()
        shape_type = self.SHAPE_TYPES[shape_idx]

        # 根据形状创建对应参数控件
        if shape_type == "box":
            self.param_widgets['length'] = self._create_double_spinbox(0.1)   # 100mm
            self.param_widgets['width'] = self._create_double_spinbox(0.1)     # 100mm
            self.param_widgets['height'] = self._create_double_spinbox(0.05)   # 50mm

            self.dynamic_param_layout.addRow("长度 X:", self.param_widgets['length'])
            self.dynamic_param_layout.addRow("宽度 Y:", self.param_widgets['width'])
            self.dynamic_param_layout.addRow("高度 Z:", self.param_widgets['height'])

        elif shape_type == "sphere":
            self.param_widgets['radius'] = self._create_double_spinbox(0.05)   # 50mm
            self.dynamic_param_layout.addRow("半径 R:", self.param_widgets['radius'])

        elif shape_type == "cylinder":
            self.param_widgets['radius'] = self._create_double_spinbox(0.05)    # 50mm
            self.param_widgets['height'] = self._create_double_spinbox(0.1)     # 100mm

            self.dynamic_param_layout.addRow("半径 R:", self.param_widgets['radius'])
            self.dynamic_param_layout.addRow("高度 Z:", self.param_widgets['height'])

        elif shape_type == "cone":
            self.param_widgets['radius'] = self._create_double_spinbox(0.05)    # 50mm
            self.param_widgets['height'] = self._create_double_spinbox(0.1)     # 100mm

            self.dynamic_param_layout.addRow("底部半径 R:", self.param_widgets['radius'])
            self.dynamic_param_layout.addRow("高度 Z:", self.param_widgets['height'])

        self._update_density_display()

    def _get_current_params(self) -> dict:
        """提取用户输入的几何参数"""
        return {key: widget.value() for key, widget in self.param_widgets.items()}

    def _get_shape_type(self) -> str:
        """获取当前选中的形状类型"""
        return self.SHAPE_TYPES[self.shape_combo.currentIndex()]

    def _update_density_display(self):
        """
        实时计算并显示点云密度 (points/cm³)
        同时更新几何体积显示
        """
        shape_type = self._get_shape_type()
        params = self._get_current_params()
        num_points = self.points_spin.value()

        # 计算体积 (m³)
        vol_m3 = GeometryEngine.calculate_volume(shape_type, params)
        self.volume_label.setText(f"{vol_m3:.6f} m³")

        # 计算密度 (points/cm³)
        density = GeometryEngine.calculate_density(vol_m3, num_points)
        self.density_label.setText(f"{density:.2f} points/cm³")

    def _on_generate(self):
        """
        点击生成按钮
        1. 获取参数
        2. 生成点云
        3. 预览渲染
        4. 启用导出按钮
        """
        shape_type = self._get_shape_type()
        params = self._get_current_params()
        num_points = self.points_spin.value()

        # UI 反馈
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("⏳ 生成中...")
        self.btn_export.setEnabled(False)
        QApplication.processEvents()

        try:
            # 根据形状类型生成点云
            if shape_type == "box":
                self.current_points = GeometryEngine.generate_box(
                    params['length'], params['width'], params['height'], num_points
                )
            elif shape_type == "sphere":
                self.current_points = GeometryEngine.generate_sphere(
                    params['radius'], num_points
                )
            elif shape_type == "cylinder":
                self.current_points = GeometryEngine.generate_cylinder(
                    params['radius'], params['height'], num_points
                )
            elif shape_type == "cone":
                self.current_points = GeometryEngine.generate_cone(
                    params['radius'], params['height'], num_points
                )

            if self.current_points is not None and len(self.current_points) > 0:
                self._preview_point_cloud()
                self.btn_export.setEnabled(True)
                self.btn_generate.setText("✅ 生成成功!")
            else:
                QMessageBox.warning(self, "警告", "生成的点云为空，请检查参数设置")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成点云失败:\n{str(e)}")
            self.btn_generate.setText("🔧 生成点云并预览")

        finally:
            self.btn_generate.setEnabled(True)
            QApplication.processEvents()

    def _preview_point_cloud(self):
        """
        将生成的点云加载到 PyVista 中渲染

        渲染配置:
        - 颜色: 青色 (cyan)
        - 点大小: 3.0
        - 不渲染为球体 (性能优化)
        - 不透明度: 1.0
        """
        if self.current_points is None:
            return

        # 清理旧场景
        self.plotter.clear()

        # 重新添加坐标轴
        self.plotter.add_axes(
            line_width=2,
            color="#AAAAAA",
            x_color="#FF5555",
            y_color="#55FF55",
            z_color="#5599FF"
        )

        # 重新添加地面网格
        grid = pv.Plane(i_size=1.0, j_size=1.0, i_resolution=10, j_resolution=10)
        self.plotter.add_mesh(
            grid,
            name="ground_grid",
            color="#3a3a3a",
            opacity=0.3
        )

        # 创建 PyVista PolyData
        cloud = pv.PolyData(self.current_points)

        # 添加点云渲染
        self.plotter.add_mesh(
            cloud,
            name="preview_cloud",
            color="#00BFFF",       # 深天蓝色 (cyan-ish)
            point_size=3.0,
            render_points_as_spheres=False,
            opacity=1.0,
            reset_camera=False    # 保持当前相机位置
        )

        # 自动调整相机以显示所有点
        self.plotter.reset_camera()
        self.plotter.render()

    def _on_export(self):
        """
        点击导出按钮
        1. 选择文件保存路径
        2. 确定格式
        3. 调用 ExportEngine 导出
        4. 显示结果
        """
        if self.current_points is None or len(self.current_points) == 0:
            QMessageBox.warning(self, "警告", "没有可导出的点云数据")
            return

        export_idx = self.export_combo.currentIndex()

        # 格式映射
        format_config = {
            0: ("pcd", "Point Cloud Data (*.pcd)", ".pcd"),
            1: ("ply", "Polygon File Format (*.ply)", ".ply"),
            2: ("xyz", "XYZ Text Format (*.xyz)", ".xyz"),
        }

        format_type, file_filter, default_ext = format_config.get(
            export_idx, ("xyz", "XYZ Text Format (*.xyz)", ".xyz")
        )

        # 文件对话框
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存点云文件",
            f"workpiece_blank_{self._get_shape_type()}{default_ext}",
            file_filter
        )

        if not path:
            return

        # 确保扩展名正确
        if not path.lower().endswith(default_ext):
            path += default_ext

        # 获取 ASCII 模式
        write_ascii = self.ascii_radio.isChecked()

        # 导出
        success, msg = ExportEngine.export(path, self.current_points, format_type, write_ascii)

        if success:
            QMessageBox.information(self, "导出成功", msg)
        else:
            QMessageBox.critical(self, "导出失败", msg)

    def closeEvent(self, event):
        """窗口关闭时清理 PyVista 资源"""
        if hasattr(self, 'plotter'):
            self.plotter.close()
        event.accept()


# ==================== 程序入口 ====================

def main() -> int:
    """启动点云生成器 GUI，返回 Qt 事件循环退出码。"""
    # 启用高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # 设置 Fusion 风格
    app.setStyle("Fusion")

    # 设置全局字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    # 创建并显示主窗口
    window = PointCloudGeneratorWindow()
    window.show()

    # 运行事件循环
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
