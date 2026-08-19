"""
robot_toolkit_main/_registry.py - Robot Registry Tab
"""

import os
import re
import xml.etree.ElementTree as ET

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QFileDialog, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QListView, QTreeView, QAbstractItemView,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from core.robot_registry import create_default_registry, RobotConfig


class RobotRegistryTab(QWidget):
    """Robot Registry 选项卡 - 多文件夹发现与机器人管理。"""

    def __init__(self):
        super().__init__()
        self._scan_folders = []
        self._setup_ui()
        self._refresh_registered_list()

    def _setup_ui(self):
        from ._xacro_compiler import LogWidget

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        discover_group = QGroupBox("Discover & Register")
        discover_layout = QVBoxLayout(discover_group)

        folder_btn_row = QHBoxLayout()

        self.add_folders_btn = QPushButton("Add Folders")
        self.add_folders_btn.setMinimumHeight(32)
        self.add_folders_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.add_folders_btn.clicked.connect(self._add_folders)

        self.remove_folders_btn = QPushButton("Remove Selected Folders")
        self.remove_folders_btn.setMinimumHeight(32)
        self.remove_folders_btn.setStyleSheet("""
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
        self.remove_folders_btn.clicked.connect(self._remove_selected_folders)

        folder_btn_row.addWidget(self.add_folders_btn)
        folder_btn_row.addWidget(self.remove_folders_btn)
        folder_btn_row.addStretch()
        discover_layout.addLayout(folder_btn_row)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.folder_list.setMaximumHeight(120)
        self.folder_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px 8px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
        """)
        discover_layout.addWidget(self.folder_list)

        self.scan_register_btn = QPushButton("Scan & Auto-Register")
        self.scan_register_btn.setMinimumHeight(36)
        self.scan_register_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.scan_register_btn.clicked.connect(self._scan_and_register)
        discover_layout.addWidget(self.scan_register_btn)

        layout.addWidget(discover_group)

        manage_group = QGroupBox("Manage Registered Robots")
        manage_layout = QVBoxLayout(manage_group)

        delete_btn_row = QHBoxLayout()

        self.delete_robots_btn = QPushButton("Delete Selected Robots")
        self.delete_robots_btn.setMinimumHeight(32)
        self.delete_robots_btn.setStyleSheet("""
            QPushButton {
                background-color: #d13438;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #a82a2e;
            }
        """)
        self.delete_robots_btn.clicked.connect(self._delete_selected_robots)

        delete_btn_row.addWidget(self.delete_robots_btn)
        delete_btn_row.addStretch()
        manage_layout.addLayout(delete_btn_row)

        self.registered_list = QListWidget()
        self.registered_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.registered_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px 10px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:checked {
                color: #66bb6a;
            }
        """)
        manage_layout.addWidget(self.registered_list, 1)

        layout.addWidget(manage_group)

        log_label = QLabel("Activity Log")
        log_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(log_label)

        self.log_widget = LogWidget()
        self.log_widget.append_log("[SYSTEM] Robot Registry ready", "system")
        layout.addWidget(self.log_widget, 1)

    def _add_folders(self):
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
                if path and path not in self._scan_folders:
                    existing_items = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
                    if path not in existing_items:
                        self.folder_list.addItem(path)
                        self._scan_folders.append(path)
                        added_count += 1

            if added_count > 0:
                self.log_widget.append_log(f"[INFO] Added {added_count} folder(s)", "info")
            else:
                self.log_widget.append_log("[INFO] No new folders to add", "info")

    def _remove_selected_folders(self):
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            self.log_widget.append_log("[INFO] No folders selected for removal", "info")
            return

        removed_count = 0
        for item in selected_items:
            path = item.text()
            row = self.folder_list.row(item)
            self.folder_list.takeItem(row)
            if path in self._scan_folders:
                self._scan_folders.remove(path)
            removed_count += 1

        self.log_widget.append_log(f"[INFO] Removed {removed_count} folder(s)", "info")

    def _scan_and_register(self):
        if self.folder_list.count() == 0:
            self.log_widget.append_log("[WARN] No folders to scan", "warning")
            return

        registry = create_default_registry()
        total_found = 0
        total_registered = 0
        total_skipped = 0

        for folder_idx in range(self.folder_list.count()):
            folder_path = self.folder_list.item(folder_idx).text()

            self.log_widget.append_log(f"[INFO] Scanning: {folder_path}", "info")

            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.urdf'):
                        urdf_path = os.path.join(root, file)
                        robot_name = self._extract_robot_name(urdf_path)
                        rel_path = os.path.relpath(urdf_path).replace('\\', '/')
                        total_found += 1

                        if registry.get(robot_name):
                            self.log_widget.append_log(f"[SKIP] {robot_name} (already registered)", "info")
                            total_skipped += 1
                            continue

                        config = RobotConfig(
                            name=robot_name,
                            urdf_path=rel_path
                        )
                        registry.register(config)
                        total_registered += 1
                        self.log_widget.append_log(f"[OK] Registered: {robot_name}", "success")

        try:
            registry.save_custom_robots()
            self.log_widget.append_log(
                f"[INFO] Saved custom robots to custom_robots.json", "info"
            )
        except Exception as e:
            self.log_widget.append_log(f"[ERROR] Failed to save: {e}", "error")

        self.log_widget.append_log(
            f"[SUMMARY] Found: {total_found} | Registered: {total_registered} | Skipped: {total_skipped}",
            "success"
        )

        self.folder_list.clear()
        self._scan_folders = []

        self._refresh_registered_list()

    def _refresh_registered_list(self):
        self.registered_list.clear()

        registry = create_default_registry()
        robot_names = registry.list_robots()

        for name in robot_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)

            if name in registry._builtin_names:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setText(f"{name} (Built-in)")
                item.setForeground(QColor("#666666"))

            self.registered_list.addItem(item)

        self.log_widget.append_log(f"[INFO] Loaded {len(robot_names)} robot(s)", "info")

    def _delete_selected_robots(self):
        registry = create_default_registry()
        deleted_count = 0
        checked_names = []

        for i in range(self.registered_list.count()):
            item = self.registered_list.item(i)
            if item.checkState() == Qt.Checked:
                name = item.text()
                if name.endswith(" (Built-in)"):
                    name = name[:-11]
                checked_names.append(name)

        if not checked_names:
            self.log_widget.append_log("[INFO] No robots selected for deletion", "info")
            return

        for name in checked_names:
            if name in registry._builtin_names:
                self.log_widget.append_log(f"[SKIP] Cannot delete built-in: {name}", "warning")
                continue

            if registry.remove(name):
                deleted_count += 1
                self.log_widget.append_log(f"[OK] Deleted: {name}", "success")
            else:
                self.log_widget.append_log(f"[WARN] Robot not found: {name}", "warning")

        try:
            registry.save_custom_robots()
            self.log_widget.append_log(
                f"[INFO] Saved custom robots to custom_robots.json", "info"
            )
        except Exception as e:
            self.log_widget.append_log(f"[ERROR] Failed to save: {e}", "error")

        self.log_widget.append_log(
            f"[SUMMARY] Deleted {deleted_count} robot(s)", "success"
        )

        self._refresh_registered_list()

    def _extract_robot_name(self, urdf_path: str) -> str:
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            robot_name = root.get('name')
            if robot_name:
                return robot_name
        except Exception:
            pass

        return os.path.splitext(os.path.basename(urdf_path))[0]
