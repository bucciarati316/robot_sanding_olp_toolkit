"""
robot_toolkit_main/_mesh_converter.py - Mesh 转换器和 Tab UI
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt

import numpy as np
import trimesh


# =============================================================================
# LOGIC LAYER - MeshConverter
# =============================================================================

class MeshConverter:
    """DAE 到 STL 的批量转换逻辑,带变换烘焙。"""

    @staticmethod
    def convert_dae_to_stl(dae_path: str) -> tuple[bool, str, list[str]]:
        logs = []

        try:
            path = Path(dae_path)
            if not path.exists():
                return False, f"File not found: {dae_path}", logs

            logs.append(f"[INFO] Loading: {path.name}")

            scene = trimesh.load(dae_path, process=False)

            if not hasattr(scene, 'geometry') or not scene.geometry:
                return False, "No geometry found in DAE file", logs

            meshes = []

            if hasattr(scene, 'graph') and hasattr(scene.graph, 'nodes_geometry'):
                for node_id in scene.graph.nodes_geometry:
                    try:
                        transform, geometry_key = scene.graph.get(node_id)

                        if geometry_key not in scene.geometry:
                            continue

                        geometry = scene.geometry[geometry_key]

                        if isinstance(geometry, trimesh.Scene):
                            for geom_key in geometry.geometry:
                                geom_mesh = geometry.geometry[geom_key]
                                full_transform = transform @ geometry.graph.get(
                                    geometry.nodes_geometry[0] if hasattr(geometry, 'nodes_geometry') else list(geometry.graph.nodes_geometry)[0]
                                )[0]
                                geom_mesh = geom_mesh.copy()
                                geom_mesh.apply_transform(full_transform)
                                meshes.append(geom_mesh)
                        else:
                            mesh = geometry.copy()
                            mesh.apply_transform(transform)
                            meshes.append(mesh)

                    except Exception as e:
                        logs.append(f"[WARN] Skipping node {node_id}: {e}")
                        continue
            else:
                for geom_key in scene.geometry:
                    meshes.append(scene.geometry[geom_key])

            if not meshes:
                return False, "No valid meshes extracted", logs

            logs.append(f"[INFO] Found {len(meshes)} mesh(es), concatenating...")
            final_mesh = trimesh.util.concatenate(meshes)

            output_stl = path.with_suffix('.stl')
            final_mesh.export(str(output_stl))

            logs.append(f"[SUCCESS] Exported: {output_stl.name}")
            logs.append(f"[INFO] Vertex count: {len(final_mesh.vertices)}")

            return True, str(output_stl), logs

        except Exception as e:
            logs.append(f"[ERROR] Conversion failed: {str(e)}")
            return False, str(e), logs

    @staticmethod
    def batch_convert(dae_paths: list[str], progress_callback=None) -> tuple[int, int, list[str]]:
        logs = []
        success_count = 0

        for i, dae_path in enumerate(dae_paths):
            success, _, path_logs = MeshConverter.convert_dae_to_stl(dae_path)
            logs.extend(path_logs)
            if success:
                success_count += 1

            if progress_callback:
                progress_callback(i + 1, len(dae_paths))

        logs.append(f"[INFO] Batch complete: {success_count}/{len(dae_paths)} succeeded")
        return success_count, len(dae_paths), logs


# =============================================================================
# UI LAYER - MeshConverterTab
# =============================================================================

class MeshConverterTab(QWidget):
    """Mesh Converter 选项卡界面。"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.file_paths = []
        self._setup_ui()

    def _setup_ui(self):
        from ._xacro_compiler import ConversionWorker, LogWidget

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        file_group = QGroupBox("DAE File Selection (Multi-select)")
        file_layout = QVBoxLayout(file_group)

        path_layout = QHBoxLayout()
        self.file_count_label = QLabel("No files selected")
        self.file_count_label.setStyleSheet("color: #808080;")

        self.select_btn = QPushButton("Select Multiple DAE")
        self.select_btn.clicked.connect(self._select_files)

        self.clear_btn = QPushButton("Clear Selection")
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
        self.clear_btn.clicked.connect(self._clear_selection)
        self.clear_btn.setEnabled(False)

        path_layout.addWidget(self.file_count_label)
        path_layout.addWidget(self.select_btn)
        path_layout.addWidget(self.clear_btn)
        file_layout.addLayout(path_layout)

        self.selected_files_label = QLabel("")
        self.selected_files_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.selected_files_label.setWordWrap(True)
        file_layout.addWidget(self.selected_files_label)

        layout.addWidget(file_group)

        self.convert_btn = QPushButton("Convert to STL")
        self.convert_btn.setEnabled(False)
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #107c10;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:enabled:hover {
                background-color: #0e6b0e;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #808080;
            }
        """)
        self.convert_btn.clicked.connect(self._run_conversion)
        layout.addWidget(self.convert_btn)

        info_label = QLabel("Transforms scene graph matrices will be baked into the meshes before export")
        info_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        log_label = QLabel("Conversion Log")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_widget = LogWidget()
        self.log_widget.append_log("[SYSTEM] Mesh Converter ready", "system")
        layout.addWidget(self.log_widget, 1)

    def _select_files(self):
        from ._xacro_compiler import ConversionWorker

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DAE Files",
            "",
            "Collada Files (*.dae);;All Files (*)"
        )

        if file_paths:
            self.file_paths.extend(file_paths)
            seen = set()
            unique_paths = []
            for path in self.file_paths:
                if path not in seen:
                    seen.add(path)
                    unique_paths.append(path)
            self.file_paths = unique_paths

            self._update_file_display()

    def _clear_selection(self):
        self.file_paths = []
        self._update_file_display()
        self.convert_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.log_widget.append_log("[INFO] File selection cleared", "info")

    def _update_file_display(self):
        from ._xacro_compiler import ConversionWorker

        total_files = len(self.file_paths)
        self.file_count_label.setText(f"{total_files} file(s) selected")
        self.clear_btn.setEnabled(total_files > 0)
        self.convert_btn.setEnabled(total_files > 0)

        if total_files > 0:
            display_text = "\n".join([os.path.basename(f) for f in self.file_paths[:5]])
            if total_files > 5:
                display_text += f"\n... and {total_files - 5} more"
            self.selected_files_label.setText(display_text)
            self.log_widget.append_log(f"[INFO] Total: {total_files} DAE file(s) selected", "info")
        else:
            self.selected_files_label.setText("")

    def _run_conversion(self):
        from ._xacro_compiler import ConversionWorker

        if not self.file_paths:
            return

        self.convert_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.log_widget.append_log("[INFO] Starting batch conversion...", "info")
        self.log_widget.append_log(f"[INFO] Total files to convert: {len(self.file_paths)}", "info")

        self.worker = ConversionWorker('mesh_batch', self.file_paths.copy())
        self.worker.log_message.connect(self._on_log)
        self.worker.task_finished.connect(self._on_finished)
        self.worker.start()

    def _on_log(self, message: str):
        if '[ERROR]' in message:
            msg_type = 'error'
        elif '[SUCCESS]' in message:
            msg_type = 'success'
        elif '[WARN]' in message:
            msg_type = 'warning'
        elif '[INFO]' in message or '[SYSTEM]' in message:
            msg_type = 'info'
        else:
            msg_type = 'info'

        self.log_widget.append_log(message, msg_type)

    def _on_finished(self, success: bool, message: str):
        self.convert_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.clear_btn.setEnabled(len(self.file_paths) > 0)

        if success:
            self.log_widget.append_log("[SUCCESS] All conversions completed!", "success")
            QMessageBox.information(self, "Success", message)
        else:
            self.log_widget.append_log(f"[WARNING] Some conversions failed: {message}", "warning")
            QMessageBox.warning(self, "Partial Success", message)
