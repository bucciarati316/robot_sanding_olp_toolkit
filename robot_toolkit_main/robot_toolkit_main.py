"""
robot_toolkit_main/robot_toolkit_main.py - Robot Toolkit Main (Facade)

重构自 robot_toolkit_main.py (Phase 10):
- XacroCompiler + XacroCompilerTab → _xacro_compiler.py
- MeshConverter + MeshConverterTab → _mesh_converter.py
- UrdfPreviewer + RobotPreviewTab → _urdf_preview.py
- RobotRegistryTab → _registry.py
- RobotToolkitMainWindow → 本文件
"""

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from ._xacro_compiler import XacroCompilerTab
from ._mesh_converter import MeshConverterTab
from ._urdf_preview import RobotPreviewTab
from ._registry import RobotRegistryTab


# =============================================================================
# MAIN WINDOW
# =============================================================================

class RobotToolkitMainWindow(QMainWindow):
    """主应用程序窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Development Toolkit")
        self.setMinimumSize(1200, 800)
        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        header_label = QLabel("Robot Development Toolkit")
        header_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: #0078d4;
                padding: 10px;
            }
        """)
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 10px 20px;
                border: 1px solid #3c3c3c;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3c3c3c;
            }
        """)

        self.xacro_tab = XacroCompilerTab()
        self.mesh_tab = MeshConverterTab()
        self.preview_tab = RobotPreviewTab()
        self.registry_tab = RobotRegistryTab()

        self.tabs.addTab(self.xacro_tab, "Xacro Compiler")
        self.tabs.addTab(self.mesh_tab, "Mesh Converter")
        self.tabs.addTab(self.preview_tab, "Robot Preview")
        self.tabs.addTab(self.registry_tab, "Robot Registry")

        main_layout.addWidget(self.tabs)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """应用入口点。"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            background-color: #252526;
            color: #cccccc;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #0078d4;
        }
        QLineEdit {
            background-color: #3c3c3c;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
        }
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
        QLabel {
            color: #cccccc;
        }
    """)

    window = RobotToolkitMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
