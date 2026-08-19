"""
robot_toolkit_main/_xacro_compiler.py - Xacro 编译器和 Tab UI
"""

import os
import re
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLineEdit, QLabel, QMessageBox, QListView,
    QTreeView, QAbstractItemView, QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

import xacro


# =============================================================================
# LOGIC LAYER - XacroCompiler
# =============================================================================

class XacroCompiler:
    """
    XACRO 到 URDF 的编译器逻辑。

    处理流程:
    - 预处理:在无 ROS 环境下解析 $(find pkg) 与 package:// 引用
    - Xacro 处理:通过临时文件进行
    - 后处理:移除 package:// 协议,替换 DAE->STL
    """

    _MAX_SEARCH_DEPTH = 15

    @staticmethod
    def _find_package_root(xacro_dir: Path, package_name: str) -> tuple[Path | None, int]:
        current = xacro_dir.resolve()
        for depth in range(XacroCompiler._MAX_SEARCH_DEPTH):
            if current.name == package_name:
                return current, depth
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None, 0

    @staticmethod
    def _resolve_xacro_content(
        xacro_path: Path,
        content: str,
        ros_base_dir: Path
    ) -> tuple[str, list[str]]:
        logs = []

        def replace_find(match):
            pkg_name = match.group(1)
            pkg_root, depth = XacroCompiler._find_package_root(xacro_path.parent, pkg_name)
            if pkg_root is not None:
                logs.append(f"[INFO] Resolved $(find {pkg_name}) -> {pkg_root} (up {depth} dirs)")
                return str(pkg_root).replace('\\', '/')
            else:
                candidate = ros_base_dir / pkg_name
                if candidate.exists():
                    logs.append(f"[INFO] Resolved $(find {pkg_name}) via ros_base -> {candidate}")
                    return str(candidate).replace('\\', '/')
                logs.append(f"[WARN] Could not resolve $(find {pkg_name})")
                return match.group(0)

        content = re.sub(r'\$\(find\s+(\w+)\)', replace_find, content)

        def replace_package_url(match):
            pkg_name = match.group(1)
            relative_path = match.group(2)
            pkg_root, depth = XacroCompiler._find_package_root(xacro_path.parent, pkg_name)
            if pkg_root is not None:
                resolved = str(pkg_root / relative_path).replace('\\', '/')
                logs.append(f"[INFO] Resolved package://{pkg_name}/ -> {resolved}")
                return resolved
            else:
                candidate = ros_base_dir / pkg_name / relative_path
                if candidate.exists():
                    resolved = str(candidate).replace('\\', '/')
                    logs.append(f"[INFO] Resolved package://{pkg_name}/ via ros_base -> {resolved}")
                    return resolved
                logs.append(f"[WARN] Could not resolve package://{pkg_name}/{relative_path}")
                return match.group(0)

        content = re.sub(r'package://([^/]+)/([^"\'<>|\s]+)', replace_package_url, content)

        return content, logs

    @staticmethod
    def _recursive_inline_includes(
        xacro_path: Path,
        ros_base_dir: Path,
        visited: set | None = None,
        logs: list | None = None
    ) -> tuple[str, list[str]]:
        if visited is None:
            visited = set()
        if logs is None:
            logs = []

        abs_path = str(xacro_path.resolve())
        if abs_path in visited:
            logs.append(f"[WARN] Circular include detected: {xacro_path.name}, skipping")
            return "", logs
        visited.add(abs_path)

        try:
            raw_content = xacro_path.read_text(encoding='utf-8')
        except Exception as e:
            logs.append(f"[ERROR] Cannot read {xacro_path}: {e}")
            return "", logs

        logs.append(f"[INFO] Processing: {xacro_path.name}")

        resolved_content, pre_logs = XacroCompiler._resolve_xacro_content(
            xacro_path, raw_content, ros_base_dir
        )
        logs.extend(pre_logs)

        result_lines = []
        include_pattern = re.compile(r'<xacro:include\s+filename=["\']([^"\']+)["\']\s*/>')

        for line in resolved_content.splitlines(keepends=True):
            match = include_pattern.search(line)
            if match:
                included_path_str = match.group(1).strip()

                if os.path.isabs(included_path_str):
                    inc_path = Path(included_path_str)
                else:
                    inc_path = (xacro_path.parent / included_path_str).resolve()

                if inc_path.exists():
                    logs.append(f"[INFO] Inlining: {included_path_str}")
                    sub_content, sub_logs = XacroCompiler._recursive_inline_includes(
                        inc_path, ros_base_dir, visited, logs
                    )
                    inner = XacroCompiler._strip_robot_tags(sub_content)
                    result_lines.append(f"<!-- === inlined from: {included_path_str} === -->\n")
                    result_lines.append(inner + "\n")
                else:
                    logs.append(f"[WARN] Include not found: {included_path_str}, keeping as-is")
                    result_lines.append(line)
            else:
                result_lines.append(line)

        return ''.join(result_lines), logs

    @staticmethod
    def _strip_robot_tags(content: str) -> str:
        lines = content.splitlines()
        stripped_lines = [l for l in lines if not l.strip().startswith('<?xml')]
        content = '\n'.join(stripped_lines)
        content = re.sub(r'<robot\b[^>]*>', '', content)
        content = re.sub(r'</robot>', '', content)
        return content

    @staticmethod
    def compile(xacro_path: str) -> tuple[bool, str, list[str]]:
        logs = []

        try:
            xacro_file = Path(xacro_path)
            if not xacro_file.exists():
                return False, f"File not found: {xacro_path}", logs

            logs.append(f"[INFO] Processing: {xacro_path}")

            xacro_dir = xacro_file.parent.resolve()
            parent_dir = xacro_dir.parent
            grandparent_dir = parent_dir.parent if parent_dir != parent_dir.parent else None

            ros_base_dir = None
            if grandparent_dir and grandparent_dir.exists():
                try:
                    siblings = [d.name for d in grandparent_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if len(siblings) > 1 or any('support' in s or 'description' in s for s in siblings):
                        ros_base_dir = grandparent_dir
                except PermissionError:
                    pass

            if ros_base_dir is None:
                ros_base_dir = xacro_dir.parent.parent if xacro_dir.parent.parent != xacro_dir.parent else xacro_dir.parent

            logs.append(f"[INFO] Workspace root: {ros_base_dir}")

            logs.append("[INFO] Inlining xacro includes...")
            inlined_content, inline_logs = XacroCompiler._recursive_inline_includes(
                xacro_file, ros_base_dir, logs=logs
            )
            logs.extend(inline_logs)

            if not inlined_content.strip():
                return False, "Failed to inline xacro content", logs

            if '<robot' not in inlined_content:
                inlined_content = f'<robot name="robot">\n{inlined_content}\n</robot>'

            import tempfile
            tmp_xacro = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.xacro', delete=False, encoding='utf-8'
                ) as f:
                    f.write(inlined_content)
                    tmp_xacro = f.name
            except Exception as e:
                return False, f"Cannot create temp file: {e}", logs

            logs.append(f"[INFO] Temp xacro: {tmp_xacro}")

            try:
                doc = xacro.process_file(tmp_xacro)
                urdf_str = doc.toprettyxml(indent='  ')
            except Exception as e:
                logs.append(f"[ERROR] xacro.process_file failed: {e}")
                return False, f"xacro processing failed: {e}", logs
            finally:
                try:
                    os.unlink(tmp_xacro)
                except Exception:
                    pass

            urdf_dir = xacro_file.parent.resolve()

            def resolve_package_ref(match):
                pkg_name = match.group(1)
                rel_path = match.group(2)
                pkg_root, depth = XacroCompiler._find_package_root(urdf_dir, pkg_name)
                if pkg_root is not None:
                    resolved = str(pkg_root / rel_path).replace('\\', '/')
                    logs.append(f"[INFO] URDF package://{pkg_name}/ -> {resolved}")
                    return resolved
                else:
                    candidate = ros_base_dir / pkg_name / rel_path
                    if candidate.exists():
                        resolved = str(candidate).replace('\\', '/')
                        logs.append(f"[INFO] URDF package://{pkg_name}/ -> {resolved}")
                        return resolved
                    logs.append(f"[WARN] Could not resolve package://{pkg_name}/{rel_path}")
                    return match.group(0)

            urdf_str = re.sub(r'package://([^/]+)/([^"\'<>|\s]+)', resolve_package_ref, urdf_str)

            dae_replaced = re.sub(r'\.dae\b', '.stl', urdf_str, flags=re.IGNORECASE)
            dae_count = len(re.findall(r'\.stl', dae_replaced))
            logs.append(f"[INFO] Replaced mesh refs to .stl: {dae_count}")

            def make_relative_path(match):
                full_path = match.group(0)
                full_path_normalized = full_path.replace('\\', '/')
                try:
                    abs_path = Path(full_path)
                    if abs_path.is_absolute():
                        rel_path = os.path.relpath(abs_path, urdf_dir)
                        rel_path_normalized = rel_path.replace('\\', '/')
                        logs.append(f"[INFO] Converted to relative: {rel_path_normalized}")
                        return rel_path_normalized
                except Exception:
                    pass
                return full_path

            dae_replaced = re.sub(
                r'(?:[A-Za-z]:[/\\][^\s"\'<>|]+|/[^/\s"\'<>|]+)',
                make_relative_path,
                dae_replaced
            )

            unresolved_pkg = re.findall(r'package://[^/]+/', dae_replaced)
            if unresolved_pkg:
                logs.append(f"[WARN] {len(unresolved_pkg)} package:// refs could not be resolved")

            output_path = xacro_file.with_suffix('.urdf')
            output_path.write_text(dae_replaced, encoding='utf-8')

            logs.append(f"[SUCCESS] URDF saved: {output_path}")
            return True, str(output_path), logs

        except Exception as e:
            import traceback
            logs.append(f"[ERROR] Compilation failed: {str(e)}")
            logs.append(f"[ERROR] Traceback: {traceback.format_exc()}")
            return False, str(e), logs


# =============================================================================
# UI LAYER - XacroCompilerTab
# =============================================================================

class LogWidget(QTextEdit):
    """带彩色输出的自定义日志显示控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px;
            }
        """)

    def append_log(self, message: str, msg_type: str = "info"):
        color_map = {
            "info": "#4fc3f7",
            "success": "#66bb6a",
            "warning": "#ffb74d",
            "error": "#ef5350",
            "system": "#9e9e9e"
        }

        color = color_map.get(msg_type, "#d4d4d4")

        char_format = QTextCharFormat()
        char_format.setForeground(QColor(color))

        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End)
        cursor.setCharFormat(char_format)
        cursor.insertText(message + "\n")

        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class ConversionWorker(QThread):
    """转换任务的后台工作线程。"""

    task_finished = Signal(bool, str)
    log_message = Signal(str)

    def __init__(self, task_type: str, file_paths: list):
        super().__init__()
        self.task_type = task_type
        self.file_paths = file_paths

    def run(self):
        if self.task_type == 'xacro':
            self._run_xacro()
        elif self.task_type == 'mesh_batch':
            self._run_mesh_batch()

    def _run_xacro(self):
        if not self.file_paths:
            self.task_finished.emit(False, "No file selected")
            return

        self.log_message.emit(f"[INFO] Starting Xacro compilation...")
        success, message, logs = XacroCompiler.compile(self.file_paths[0])

        for log in logs:
            self.log_message.emit(log)

        self.task_finished.emit(success, message)

    def _run_mesh_batch(self):
        from ._mesh_converter import MeshConverter
        if not self.file_paths:
            self.task_finished.emit(False, "No files selected")
            return

        self.log_message.emit(f"[INFO] Starting batch DAE->STL conversion...")
        self.log_message.emit(f"[INFO] Files selected: {len(self.file_paths)}")

        success_count, total_count, logs = MeshConverter.batch_convert(
            self.file_paths,
            progress_callback=lambda c, t: self.log_message.emit(f"[INFO] Progress: {c}/{t}")
        )

        for log in logs:
            self.log_message.emit(log)

        self.task_finished.emit(
            success_count == total_count,
            f"Converted {success_count}/{total_count} files"
        )


class BatchDirectoryWorker(QThread):
    """批量目录转换(DAE->STL 与 Xacro->URDF)的后台工作线程。"""

    task_finished = Signal(bool, str)
    log_message = Signal(str)
    progress = Signal(int, int)

    def __init__(self, directories: list[str]):
        super().__init__()
        self.directories = directories

    def run(self):
        from ._mesh_converter import MeshConverter

        self.log_message.emit(f"[INFO] Starting batch directory conversion...")
        self.log_message.emit(f"[INFO] Target directories: {len(self.directories)}")

        all_dae_files = []
        all_xacro_files = []

        for directory in self.directories:
            self.log_message.emit(f"[INFO] Scanning: {directory}")
            dae_files = self._find_dae_files(directory)
            all_dae_files.extend(dae_files)
            xacro_files = self._find_xacro_files(directory)
            all_xacro_files.extend(xacro_files)

        self.log_message.emit(f"[INFO] Found {len(all_dae_files)} DAE file(s) total")
        self.log_message.emit(f"[INFO] Found {len(all_xacro_files)} Xacro file(s) total")

        dae_success = 0
        dae_total = len(all_dae_files)

        for i, dae_path in enumerate(all_dae_files):
            self.log_message.emit(f"[INFO] Converting DAE: {dae_path}")
            success, _, logs = MeshConverter.convert_dae_to_stl(dae_path)
            for log in logs:
                self.log_message.emit(log)
            if success:
                dae_success += 1
            self.progress.emit(i + 1, dae_total + dae_total)

        self.log_message.emit(f"[INFO] DAE conversion complete: {dae_success}/{dae_total}")

        xacro_success = 0
        xacro_total = len(all_xacro_files)

        for i, xacro_path in enumerate(all_xacro_files):
            self.log_message.emit(f"[INFO] Compiling Xacro: {xacro_path}")
            success, _, logs = XacroCompiler.compile(xacro_path)
            for log in logs:
                self.log_message.emit(log)
            if success:
                xacro_success += 1
            self.progress.emit(dae_total + i + 1, dae_total + xacro_total)

        self.log_message.emit(f"[INFO] Xacro compilation complete: {xacro_success}/{xacro_total}")

        total_files = dae_total + xacro_total
        total_success = dae_success + xacro_success
        self.log_message.emit(f"[INFO] ========================================")
        self.log_message.emit(f"[INFO] Batch conversion summary:")
        self.log_message.emit(f"[INFO]   DAE -> STL:  {dae_success}/{dae_total}")
        self.log_message.emit(f"[INFO]   Xacro -> URDF: {xacro_success}/{xacro_total}")
        self.log_message.emit(f"[INFO]   Total: {total_success}/{total_files}")
        self.log_message.emit(f"[INFO] ========================================")

        self.task_finished.emit(
            total_success == total_files,
            f"Completed: {total_success}/{total_files} files processed"
        )

    def _find_dae_files(self, directory: str) -> list[str]:
        dae_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.dae'):
                        dae_files.append(os.path.join(root, file))
        except Exception as e:
            self.log_message.emit(f"[ERROR] Error searching for DAE files: {e}")
        return dae_files

    def _find_xacro_files(self, directory: str) -> list[str]:
        xacro_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.xacro') \
                       and not file.lower().endswith('.urdf.xacro') \
                       and not file.lower().endswith('.macro.xacro') \
                       and not file.lower().endswith('_macro.xacro'):
                        xacro_path = os.path.join(root, file)
                        if self._is_robot_root_xacro(xacro_path):
                            xacro_files.append(xacro_path)
        except Exception as e:
            self.log_message.emit(f"[ERROR] Error searching for Xacro files: {e}")
        return xacro_files

    def _is_robot_root_xacro(self, xacro_path: str) -> bool:
        try:
            with open(xacro_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<robot' in content:
                    if '<xacro:include' in content or 'xacro:macro' in content:
                        return True
                    return True
        except Exception:
            pass
        return False


class XacroCompilerTab(QWidget):
    """Xacro Compiler 选项卡界面。"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.batch_worker = None
        self.batch_directories = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        file_group = QGroupBox("Xacro File Selection")
        file_layout = QVBoxLayout(file_group)

        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("Select a .xacro file...")

        self.select_btn = QPushButton("Select Xacro")
        self.select_btn.clicked.connect(self._select_file)

        path_layout.addWidget(self.file_path_edit)
        path_layout.addWidget(self.select_btn)
        file_layout.addLayout(path_layout)

        layout.addWidget(file_group)

        self.compile_btn = QPushButton("Compile to URDF")
        self.compile_btn.setEnabled(False)
        self.compile_btn.setMinimumHeight(40)
        self.compile_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:enabled:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #808080;
            }
        """)
        self.compile_btn.clicked.connect(self._run_compilation)
        layout.addWidget(self.compile_btn)

        info_label = QLabel("This will compile the xacro file and replace all .dae references with .stl")
        info_label.setStyleSheet("color: #808080; font-size: 11px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #3c3c3c; margin: 10px 0;")
        layout.addWidget(separator)

        batch_group = QGroupBox("Batch Directory Conversion")
        batch_layout = QVBoxLayout(batch_group)

        batch_info_label = QLabel(
            "Recursively process robot description packages.\n"
            "All .dae files will be converted to .stl, and all .xacro files will be compiled to .urdf."
        )
        batch_info_label.setStyleSheet("color: #808080; font-size: 11px;")
        batch_info_label.setWordWrap(True)
        batch_layout.addWidget(batch_info_label)

        folder_row = QHBoxLayout()
        self.batch_count_label = QLabel("0 folders selected")
        self.batch_count_label.setStyleSheet("color: #808080;")

        self.add_folder_btn = QPushButton("Add Folder")
        self.add_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.add_folder_btn.clicked.connect(self._add_batch_folder)

        self.clear_folders_btn = QPushButton("Clear Folders")
        self.clear_folders_btn.setEnabled(False)
        self.clear_folders_btn.setStyleSheet("""
            QPushButton {
                background-color: #d13438;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #a82a2e;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #808080;
            }
        """)
        self.clear_folders_btn.clicked.connect(self._clear_batch_folders)

        folder_row.addWidget(self.batch_count_label)
        folder_row.addStretch()
        folder_row.addWidget(self.add_folder_btn)
        folder_row.addWidget(self.clear_folders_btn)
        batch_layout.addLayout(folder_row)

        self.batch_dirs_label = QLabel("")
        self.batch_dirs_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        self.batch_dirs_label.setWordWrap(True)
        batch_layout.addWidget(self.batch_dirs_label)

        self.batch_convert_btn = QPushButton("Start Batch Conversion")
        self.batch_convert_btn.setEnabled(False)
        self.batch_convert_btn.setMinimumHeight(40)
        self.batch_convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #6a0dad;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:enabled:hover {
                background-color: #7b2dbf;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
                color: #808080;
            }
        """)
        self.batch_convert_btn.clicked.connect(self._run_batch_conversion)
        batch_layout.addWidget(self.batch_convert_btn)

        layout.addWidget(batch_group)

        log_label = QLabel("Conversion Log")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_widget = LogWidget()
        self.log_widget.append_log("[SYSTEM] Xacro Compiler ready", "system")
        layout.addWidget(self.log_widget, 1)

    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Xacro File",
            "",
            "Xacro Files (*.xacro *.urdf.xacro);;All Files (*)"
        )

        if file_path:
            self.file_path_edit.setText(file_path)
            self.compile_btn.setEnabled(True)
            self.log_widget.append_log(f"[INFO] Selected: {os.path.basename(file_path)}", "info")

    def _run_compilation(self):
        if not self.file_path_edit.text():
            return

        self.compile_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.log_widget.append_log("[INFO] Starting compilation...", "info")

        self.worker = ConversionWorker('xacro', [self.file_path_edit.text()])
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
        self.compile_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        if success:
            self.log_widget.append_log("[SUCCESS] Compilation completed successfully!", "success")
            QMessageBox.information(self, "Success", f"URDF saved to:\n{message}")
        else:
            self.log_widget.append_log(f"[ERROR] Compilation failed: {message}", "error")
            QMessageBox.critical(self, "Error", f"Compilation failed:\n{message}")

    def _run_batch_conversion(self):
        if not self.batch_directories:
            return

        self.batch_convert_btn.setEnabled(False)
        self.add_folder_btn.setEnabled(False)
        self.clear_folders_btn.setEnabled(False)
        self.compile_btn.setEnabled(False)
        self.select_btn.setEnabled(False)

        self.log_widget.append_log("[INFO] Starting batch directory conversion...", "info")
        self.log_widget.append_log(f"[INFO] Directories: {len(self.batch_directories)}", "info")

        self.batch_worker = BatchDirectoryWorker(self.batch_directories.copy())
        self.batch_worker.log_message.connect(self._on_batch_log)
        self.batch_worker.task_finished.connect(self._on_batch_finished)
        self.batch_worker.start()

    def _add_batch_folder(self):
        dialog = QFileDialog(self, "Select Multiple Directories")
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        for view in dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for view in dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        if dialog.exec():
            paths = dialog.selectedFiles()
            added_count = 0
            for path in paths:
                if path and path not in self.batch_directories:
                    self.batch_directories.append(path)
                    added_count += 1
            if added_count > 0:
                self._update_batch_display()
                self.log_widget.append_log(f"[INFO] Added {added_count} folder(s)", "info")

    def _clear_batch_folders(self):
        self.batch_directories = []
        self._update_batch_display()
        self.log_widget.append_log("[INFO] Folder selection cleared", "info")

    def _update_batch_display(self):
        total_folders = len(self.batch_directories)
        self.batch_count_label.setText(f"{total_folders} folder(s) selected")
        self.clear_folders_btn.setEnabled(total_folders > 0)
        self.batch_convert_btn.setEnabled(total_folders > 0)

        if total_folders > 0:
            display_text = "\n".join([os.path.basename(f) for f in self.batch_directories[:3]])
            if total_folders > 3:
                display_text += f"\n... and {total_folders - 3} more"
            self.batch_dirs_label.setText(display_text)
        else:
            self.batch_dirs_label.setText("")

    def _on_batch_log(self, message: str):
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

    def _on_batch_finished(self, success: bool, message: str):
        self.batch_convert_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.clear_folders_btn.setEnabled(len(self.batch_directories) > 0)
        self.compile_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        if success:
            self.log_widget.append_log("[SUCCESS] Batch conversion completed successfully!", "success")
            QMessageBox.information(self, "Success", message)
        else:
            self.log_widget.append_log(f"[WARNING] Batch conversion completed with some failures: {message}", "warning")
            QMessageBox.warning(self, "Partial Success", message)
