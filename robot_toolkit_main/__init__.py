"""
robot_toolkit_main/ - Robot Toolkit Main Package

Split from robot_toolkit_main.py (Phase 10):
- _xacro_compiler.py: XacroCompiler, XacroCompilerTab
- _mesh_converter.py: MeshConverter, MeshConverterTab
- _urdf_preview.py: UrdfPreviewer, ConversionWorker, BatchDirectoryWorker,
                    LogWidget, RobotPreviewTab
- _registry.py: RobotRegistryTab
- robot_toolkit_main.py: RobotToolkitMainWindow, main (facade)
"""

from .robot_toolkit_main import RobotToolkitMainWindow, main

__all__ = ["RobotToolkitMainWindow", "main"]
