"""
generate_stl.py
================

生成用于 Robot OLP Toolkit 测试 / 调试 / 碰撞包围盒的 STL 文件。

注意单位约定:
    - 用户参数: **毫米 (mm)** — 与 SolidWorks 导出 STL 的默认单位一致
    - STL 文件: **毫米 (mm)**
    - 项目导入: render_engine.py 在 load_cad_model / load_external_stl / load_tool_stl 时
      会自动执行 `points *= 0.001`，把 mm 转成 m
    - 因此: 你在下面 GEOMETRY 区填多少 mm，导入工具包后就是多少 mm

输出:
    ./generated_stl/
        cube.stl         - 正方体（轴对齐，中心在原点）
        sphere.stl       - 球体
        cylinder.stl     - 圆柱体（轴沿 +Z，底面在 z=0，顶面在 z=height）
"""

import io
import os
import sys
import numpy as np
import pyvista as pv

# Windows 终端默认 GBK，强制 UTF-8 以避免中文/特殊符号报错
if sys.platform.startswith("win"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# PyVista 在 save / read 时会向 stdout 打 INFO 级消息，
# 在 Windows GBK 控制台下可能引发 UnicodeEncodeError，把全局 verbosity 关掉。
try:
    pv.set_verbosity("error")
except Exception:
    pass

# ==================== GEOMETRY（在这里调整所有尺寸，单位 mm）====================

# 通用：输出目录（相对当前工作目录）
OUTPUT_DIR = "generated_stl"

# 正方体：边长 (mm)
CUBE_SIDE_LENGTH = 200.0  # 200 mm = 20 cm

# 球体：半径 (mm)
SPHERE_RADIUS = 100.0  # 100 mm = 10 cm

# 圆柱体：底面半径 (mm) + 总高 (mm)
CYLINDER_RADIUS = 50.0
CYLINDER_HEIGHT = 150.0

# 球体网格密度（经线/纬线细分数，越大越圆，但 STL 越大）
SPHERE_THETA_RESOLUTION = 30
SPHERE_PHI_RESOLUTION = 30

# 圆柱体侧面细分数（越大越光滑）
CYLINDER_RESOLUTION = 256

# ==================== 单位换算 ====================
# 用户填 mm，内部用 pv 在 m 单位下建 mesh，导出时再 *1000 写成 mm 数值。
# 这样与 render_engine.py 第 1716/1864/2398 行的 `points *= 0.001` 严格互逆。
_MM_TO_M = 0.001


def mm_to_m(value_mm: float) -> float:
    return value_mm * _MM_TO_M


# ==================== 工具函数 ====================


def make_cube(side_length_mm: float) -> pv.PolyData:
    """生成边长 side_length_mm 的正方体 mesh，中心位于 (0,0,0)。"""
    return pv.Cube(
        center=(0.0, 0.0, 0.0),
        x_length=mm_to_m(side_length_mm),
        y_length=mm_to_m(side_length_mm),
        z_length=mm_to_m(side_length_mm),
    )


def make_sphere(radius_mm: float,
                theta_resolution: int = 30,
                phi_resolution: int = 30) -> pv.PolyData:
    """生成半径 radius_mm 的球体 mesh，中心位于 (0,0,0)。"""
    return pv.Sphere(
        radius=mm_to_m(radius_mm),
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )


def make_cylinder(radius_mm: float,
                   height_mm: float,
                   resolution: int = 64) -> pv.PolyData:
    """
    生成圆柱体 mesh:
        - 中心轴沿 +Z 方向
        - 底面圆心位于 (0, 0, 0)
        - 顶面圆心位于 (0, 0, height_mm)
    """
    cyl = pv.Cylinder(
        direction=(0.0, 0.0, 1.0),
        radius=mm_to_m(radius_mm),
        height=mm_to_m(height_mm),
        resolution=resolution,
        center=(0.0, 0.0, mm_to_m(height_mm) / 2.0),
    )
    return cyl


def save_stl_mm(mesh_m: pv.PolyData, filepath: str, name: str,
                expected_size_mm: tuple) -> None:
    """
    把建在米单位下的 mesh 顶点数乘 1000 转成毫米，再以二进制 STL 输出。

    参数:
        mesh_m: 在 m 单位下构建的 pv.PolyData
        filepath: STL 文件路径
        name: 名字（用于日志）
        expected_size_mm: 期望的 (sx, sy, sz)，单位 mm
    """
    mesh_mm = mesh_m.copy(deep=False)
    mesh_mm.points *= 1000.0  # m -> mm
    mesh_mm.save(filepath, binary=True)
    n_cells = mesh_mm.n_cells if hasattr(mesh_mm, "n_cells") else 0
    n_points = mesh_mm.n_points
    size_kb = os.path.getsize(filepath) / 1024
    print(
        f"  [{name}] -> {filepath}  "
        f"(triangles={n_cells}, vertices={n_points}, {size_kb:.1f} KB, "
        f"expected mm={expected_size_mm})"
    )


# ==================== 主流程 ====================


def main():
    out_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[stl_generator] output dir: {out_dir}")

    print("[stl_generator] building meshes (units: mm) ...")
    cube = make_cube(CUBE_SIDE_LENGTH)
    sphere = make_sphere(SPHERE_RADIUS, SPHERE_THETA_RESOLUTION, SPHERE_PHI_RESOLUTION)
    cylinder = make_cylinder(CYLINDER_RADIUS, CYLINDER_HEIGHT, CYLINDER_RESOLUTION)

    print("[stl_generator] writing STL files (binary, mm) ...")
    save_stl_mm(
        cube,
        os.path.join(out_dir, "cube.stl"),
        "cube",
        expected_size_mm=(CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH, CUBE_SIDE_LENGTH),
    )
    save_stl_mm(
        sphere,
        os.path.join(out_dir, "sphere.stl"),
        "sphere",
        expected_size_mm=(2 * SPHERE_RADIUS, 2 * SPHERE_RADIUS, 2 * SPHERE_RADIUS),
    )
    save_stl_mm(
        cylinder,
        os.path.join(out_dir, "cylinder.stl"),
        "cylinder",
        expected_size_mm=(2 * CYLINDER_RADIUS, 2 * CYLINDER_RADIUS, CYLINDER_HEIGHT),
    )

    print("[stl_generator] done.")
    print("[stl_generator] hint: import into Robot OLP Toolkit via "
          "load_cad_model() / load_external_stl() — units will be auto-converted "
          "from mm to m by render_engine.py.")


if __name__ == "__main__":
    main()
