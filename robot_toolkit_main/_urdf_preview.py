"""
robot_toolkit_main/_urdf_preview.py - URDF 预览器和 RobotPreview Tab
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import pyvista as pv
from pyvistaqt import QtInteractor


# =============================================================================
# LOGIC LAYER - UrdfPreviewer
# =============================================================================

class UrdfPreviewer:
    """
    URDF 网格可视化器逻辑 — 由 Pinocchio 提供正运动学支持。

    处理流程:
    - 通过向父级目录搜索包根目录来解析 package:// 路径
    - 构建 Pinocchio 模型并在中性位姿下计算各 frame 的变换(oMf)
    - 计算 visual 局部偏移,并组合为 4x4 全局矩阵
    - 加载网格文件(STL、OBJ、DAE)用于三维预览
    """

    _MAX_SEARCH_DEPTH = 15

    @staticmethod
    def resolve_package_path(urdf_path: str, mesh_path: str) -> str:
        if not mesh_path.startswith('package://'):
            if os.path.isabs(mesh_path):
                return mesh_path
            urdf_dir = str(Path(urdf_path).parent.resolve())
            resolved = os.path.normpath(os.path.join(urdf_dir, mesh_path))
            return resolved

        match = re.match(r'^package://([^/]+)/(.+)$', mesh_path)
        if not match:
            return mesh_path

        pkg_name = match.group(1)
        relative_path = match.group(2).lstrip('/')

        urdf_file = Path(urdf_path).resolve()
        urdf_dir = urdf_file.parent

        current = urdf_dir
        for _ in range(UrdfPreviewer._MAX_SEARCH_DEPTH):
            if current.name == pkg_name:
                candidate = current / relative_path
                if candidate.exists():
                    return str(candidate.resolve())

            try:
                for sibling in current.iterdir():
                    if sibling.is_dir() and sibling.name == pkg_name:
                        candidate = sibling / relative_path
                        if candidate.exists():
                            return str(candidate.resolve())
            except PermissionError:
                pass

            parent = current.parent
            if parent == current:
                break
            current = parent

        fallback = urdf_dir / pkg_name / relative_path
        if fallback.exists():
            return str(fallback.resolve())

        return mesh_path

    @staticmethod
    def compute_pinocchio_fk(urdf_path: str) -> tuple[dict, dict, list[str]]:
        logs = []
        link_fk = {}
        child_to_joint = {}

        try:
            logs.append(f"[INFO] Building Pinocchio model from: {urdf_path}")
            model = pin.buildModelFromUrdf(urdf_path)
            logs.append(f"[INFO] Pinocchio model: {model.nq} dof, {model.njoints} joints, "
                         f"{model.nframes} frames")

            data = model.createData()
            q = pin.neutral(model)
            pin.forwardKinematics(model, data, q)

            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for joint_elem in root.findall('./joint'):
                child_elem = joint_elem.find('child')
                if child_elem is None:
                    continue
                child_link = child_elem.get('link', '')
                joint_name = joint_elem.get('name', '')
                if child_link and joint_name:
                    child_to_joint[child_link] = joint_name

            logs.append(f"[INFO] Mapped {len(child_to_joint)} joints from XML")

            root_link = None
            for child_link in child_to_joint:
                pass
            for joint_elem in root.findall('./joint'):
                parent_elem = joint_elem.find('parent')
                child_elem = joint_elem.find('child')
                if parent_elem is None or child_elem is None:
                    continue
                parent_link = parent_elem.get('link', '')
                child_link = child_elem.get('link', '')
                if child_link not in child_to_joint:
                    root_link = parent_link
                    break
            if root_link is None:
                for joint_elem in root.findall('./joint'):
                    parent_elem = joint_elem.find('parent')
                    if parent_elem is not None:
                        root_link = parent_elem.get('link', '')
                        break

            logs.append(f"[INFO] Root link: {root_link}")

            joint_world_se3 = {}
            for jid, jname in enumerate(model.names):
                if jname == 'universe':
                    continue
                try:
                    joint_world_se3[jname] = data.oMi[jid]
                except Exception:
                    pass

            visited = set()

            def accumulate(link_name: str, world_se3: pin.SE3):
                if link_name in visited:
                    return
                visited.add(link_name)
                link_fk[link_name] = world_se3

                if link_name not in child_to_joint:
                    return
                joint_name = child_to_joint[link_name]
                joint_se3 = joint_world_se3.get(joint_name)
                if joint_se3 is None:
                    logs.append(f"[WARN] No FK for joint '{joint_name}'")
                    return
                accumulate(link_name, joint_se3)

            root_se3 = pin.SE3(np.eye(3), np.zeros(3))
            accumulate(root_link, root_se3)

            for child_link, joint_name in child_to_joint.items():
                if child_link in visited:
                    continue
                joint_se3 = joint_world_se3.get(joint_name)
                if joint_se3 is None:
                    continue
                accumulate(child_link, joint_se3)

            logs.append(f"[INFO] Link FK computed for {len(link_fk)} link(s)")
            for ln, se3 in link_fk.items():
                t = se3.translation
                logs.append(f"[DEBUG] {ln}: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")

        except RuntimeError as e:
            err_msg = str(e)
            logs.append(f"[ERROR] Pinocchio build failed: {err_msg}")
            if 'package://' in err_msg or 'file not found' in err_msg.lower():
                logs.append("[ERROR] URDF contains unresolved package:// paths.")
                logs.append("[ERROR] Please use Xacro Compiler to convert the xacro first.")
            return {}, {}, logs

        except Exception as e:
            import traceback
            logs.append(f"[ERROR] Pinocchio FK failed: {str(e)}")
            logs.append(f"[ERROR] {traceback.format_exc()}")
            return {}, {}, logs

        return link_fk, child_to_joint, logs

    @staticmethod
    def _visual_local_transform(origin_elem, scale: list = None) -> np.ndarray:
        if origin_elem is None:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        else:
            xyz_str = origin_elem.get('xyz', '0 0 0')
            rpy_str = origin_elem.get('rpy', '0 0 0')
            xyz = [float(v) for v in xyz_str.split()]
            rpy = [float(v) for v in rpy_str.split()]
            while len(xyz) < 3:
                xyz.append(0.0)
            while len(rpy) < 3:
                rpy.append(0.0)
            xyz = xyz[:3]
            rpy = rpy[:3]

        T = np.eye(4, dtype=np.float64)

        S = np.eye(4, dtype=np.float64)
        if scale is not None:
            S[0, 0] = scale[0]
            S[1, 1] = scale[1]
            S[2, 2] = scale[2]

        R = np.eye(4, dtype=np.float64)
        if any(rpy):
            R[:3, :3] = Rotation.from_euler('XYZ', rpy, degrees=False).as_matrix()

        Txyz = np.eye(4, dtype=np.float64)
        Txyz[:3, 3] = xyz

        T = Txyz @ R @ S

        return T

    @staticmethod
    def _se3_to_mat4x4(se3: pin.SE3) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = se3.rotation
        T[:3, 3] = se3.translation
        return T

    @staticmethod
    def parse_urdf_meshes(urdf_path: str) -> tuple[list, list[str]]:
        logs = []

        try:
            path = Path(urdf_path)
            if not path.exists():
                return [], [f"[ERROR] File not found: {urdf_path}"]

            abs_urdf_path = str(path.resolve())
            logs.append(f"[INFO] Parsing URDF: {path.name}")

            link_fk, child_to_joint, fk_logs = UrdfPreviewer.compute_pinocchio_fk(abs_urdf_path)
            logs.extend(fk_logs)

            if not link_fk:
                return [], logs

            tree = ET.parse(urdf_path)
            root = tree.getroot()

            mesh_data = []

            for link in root.findall('.//link'):
                link_name = link.get('name', 'unknown')

                if link_name not in link_fk:
                    continue
                world_se3 = link_fk[link_name]

                for visual in link.findall('visual'):
                    geometry = visual.find('geometry')
                    if geometry is None:
                        continue

                    mesh_elem = geometry.find('mesh')
                    if mesh_elem is None:
                        continue

                    filename = mesh_elem.get('filename', '')
                    if not filename:
                        continue

                    mesh_scale = [1.0, 1.0, 1.0]
                    scale_str = mesh_elem.get('scale', '1.0 1.0 1.0')
                    scale_vals = [float(v) for v in scale_str.split()]
                    while len(scale_vals) < 3:
                        scale_vals.append(1.0)
                    mesh_scale = scale_vals[:3]

                    resolved_path = UrdfPreviewer.resolve_package_path(abs_urdf_path, filename)

                    origin_elem = visual.find('origin')
                    T_visual_local = UrdfPreviewer._visual_local_transform(origin_elem, mesh_scale)

                    T_link_world = UrdfPreviewer._se3_to_mat4x4(world_se3)

                    total_T = T_link_world @ T_visual_local

                    pos = total_T[:3, 3]

                    mesh_data.append({
                        'link_name': link_name,
                        'mesh_path': resolved_path,
                        'total_T': total_T,
                        'T_link_world': T_link_world,
                        'T_visual_local': T_visual_local,
                        'mesh_scale': mesh_scale,
                        'position': pos.tolist(),
                        'filename': os.path.basename(filename),
                        'original_path': filename
                    })

                    logs.append(
                        f"[INFO] {link_name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                    )

            if not mesh_data:
                logs.append("[WARN] No visual meshes found in URDF")
            else:
                logs.append(f"[INFO] Total meshes: {len(mesh_data)}")

            return mesh_data, logs

        except ET.ParseError as e:
            return [], [f"[ERROR] Failed to parse URDF XML: {e}"]
        except Exception as e:
            import traceback
            return [], [f"[ERROR] Failed to parse URDF: {str(e)}\n{traceback.format_exc()}"]

    @staticmethod
    def load_mesh_for_preview(mesh_path: str) -> tuple[any, list[str]]:
        logs = []

        try:
            path = Path(mesh_path)
            if not path.exists():
                return None, [f"[ERROR] Mesh not found: {mesh_path}"]

            suffix = path.suffix.lower()

            if suffix == '.stl':
                pv_mesh = pv.read(mesh_path)

            elif suffix in ('.obj', '.ply', '.dae'):
                import trimesh as tm
                loaded = tm.load(mesh_path, process=False)

                if isinstance(loaded, tm.Scene):
                    meshes = []
                    if hasattr(loaded, 'geometry'):
                        for geom in loaded.geometry.values():
                            if isinstance(geom, tm.Trimesh):
                                meshes.append(geom)
                            elif hasattr(geom, 'geometry'):
                                for sub in geom.geometry.values():
                                    if isinstance(sub, tm.Trimesh):
                                        meshes.append(sub)
                    if not meshes:
                        return None, [f"[ERROR] Empty scene in {path.name}"]
                    combined = tm.util.concatenate(meshes)
                else:
                    combined = loaded

                pv_mesh = pv.PolyData(
                    combined.vertices,
                    np.column_stack([np.full(len(combined.faces), 3), combined.faces]).flatten()
                )

            else:
                pv_mesh = pv.read(mesh_path)

            logs.append(f"[INFO] Loaded: {path.name}")
            return pv_mesh, logs

        except Exception as e:
            return None, [f"[ERROR] Failed to load mesh: {str(e)}"]


# =============================================================================
# UI LAYER - RobotPreviewTab
# =============================================================================

class RobotPreviewTab(QWidget):
    """带 PyVista 三维查看器的 Robot Preview 选项卡。"""

    def __init__(self):
        super().__init__()
        self.loaded_meshes = []
        self._setup_ui()

    def _setup_ui(self):
        from ._xacro_compiler import LogWidget

        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        control_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load URDF")
        self.load_btn.setMinimumHeight(35)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.load_btn.clicked.connect(self._load_urdf)

        self.clear_btn = QPushButton("Clear Scene")
        self.clear_btn.setMinimumHeight(35)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #d13438;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #a82a2e;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_scene)

        self.reset_camera_btn = QPushButton("Reset Camera")
        self.reset_camera_btn.setMinimumHeight(35)

        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.reset_camera_btn)

        layout.addLayout(control_layout)

        self.plotter = QtInteractor(self)
        self.plotter.add_axes()
        self.reset_camera_btn.clicked.connect(self.plotter.reset_camera)

        self.plotter.set_background("#1e1e1e")
        self.plotter.show_axes()

        layout.addWidget(self.plotter, 1)

        self.status_label = QLabel("Ready - Load a URDF file to preview robot meshes")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #252526;
                color: #cccccc;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.status_label)

        log_label = QLabel("Loading Log")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_widget = LogWidget()
        self.log_widget.append_log("[SYSTEM] Robot Preview ready", "system")
        layout.addWidget(self.log_widget, 1)

    def _load_urdf(self):
        urdf_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select URDF File",
            "",
            "URDF Files (*.urdf);;All Files (*)"
        )

        if not urdf_path:
            return

        self.log_widget.append_log(f"[INFO] Loading URDF: {os.path.basename(urdf_path)}", "info")

        try:
            mesh_data, parse_logs = UrdfPreviewer.parse_urdf_meshes(urdf_path)

            for log in parse_logs:
                self._append_log(log)

            if not mesh_data:
                QMessageBox.warning(self, "Warning", "No visual meshes found in URDF")
                return

            self.log_widget.append_log(f"[INFO] Loading {len(mesh_data)} mesh(es)...", "info")

            self.plotter.clear()
            self.plotter.add_axes()
            self.loaded_meshes = []

            success_count = 0
            error_count = 0

            for mesh_info in mesh_data:
                mesh_path = mesh_info['mesh_path']
                mesh_file = Path(mesh_path)

                if not mesh_file.exists():
                    self.log_widget.append_log(f"[WARN] Mesh not found: {mesh_path}", "warning")
                    error_count += 1
                    continue

                try:
                    pv_mesh, load_logs = UrdfPreviewer.load_mesh_for_preview(mesh_path)

                    for log in load_logs:
                        self._append_log(log)

                    if pv_mesh is None:
                        error_count += 1
                        continue

                    link_hash = hash(mesh_info['link_name']) % 0xFFFFFF
                    color = [
                        ((link_hash >> 16) & 0xFF) / 255.0,
                        ((link_hash >> 8) & 0xFF) / 255.0,
                        (link_hash & 0xFF) / 255.0,
                        1.0
                    ]

                    total_T = mesh_info['total_T']

                    try:
                        transformed_mesh = pv_mesh.copy()
                        transformed_mesh.transform(total_T)
                    except Exception as e:
                        self.log_widget.append_log(
                            f"[WARN] Transform failed for {mesh_info['link_name']}: {e}", "warning"
                        )
                        transformed_mesh = pv_mesh

                    actor = self.plotter.add_mesh(
                        transformed_mesh,
                        color=color,
                        show_edges=True,
                        opacity=1.0,
                        name=mesh_info['link_name']
                    )

                    self.loaded_meshes.append({
                        'name': mesh_info['link_name'],
                        'mesh': transformed_mesh,
                        'total_T': total_T
                    })

                    pos = total_T[:3, 3]
                    self.log_widget.append_log(
                        f"[INFO] {mesh_info['link_name']} at "
                        f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})", "info"
                    )

                    success_count += 1

                except Exception as e:
                    self.log_widget.append_log(
                        f"[ERROR] {mesh_file.name}: {str(e)}", "error"
                    )
                    error_count += 1

            self.plotter.reset_camera()
            self.plotter.render()

            self.status_label.setText(
                f"Loaded {success_count} mesh(es) | {error_count} error(s)"
            )

            if success_count > 0:
                self.log_widget.append_log(
                    f"[SUCCESS] {success_count}/{len(mesh_data)} meshes placed via Pinocchio FK",
                    "success"
                )

        except Exception as e:
            import traceback
            self.log_widget.append_log(f"[ERROR] Failed: {str(e)}", "error")
            self.log_widget.append_log(traceback.format_exc(), "error")
            QMessageBox.critical(self, "Error", f"Failed to load URDF:\n{str(e)}")

    def _append_log(self, message: str):
        if '[ERROR]' in message:
            msg_type = 'error'
        elif '[SUCCESS]' in message:
            msg_type = 'success'
        elif '[WARN]' in message:
            msg_type = 'warning'
        elif '[INFO]' in message:
            msg_type = 'info'
        else:
            msg_type = 'info'

        self.log_widget.append_log(message, msg_type)

    def _clear_scene(self):
        self.plotter.clear()
        self.plotter.add_axes()
        self.loaded_meshes = []
        self.status_label.setText("Ready - Load a URDF file to preview robot meshes")
        self.log_widget.append_log("[INFO] Scene cleared", "info")
