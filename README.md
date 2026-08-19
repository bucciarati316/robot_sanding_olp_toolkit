# Robot OLP Toolkit

> A public, portfolio-ready offline-programming (OLP) toolkit for **robotic
> sanding, polishing, deburring, and other surface-finishing tasks**, built
> with **Pinocchio**, **PyVista**, and **PySide6**.

![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D4?logo=windows&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PySide6%20%2B%20PyVista-41CD52)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Public repository: [github.com/bucciarati316/robot_sanding_olp_toolkit](https://github.com/bucciarati316/robot_sanding_olp_toolkit)

## 目录 / Contents

- [中文说明](#中文说明)
- [English documentation](#english-documentation)

## 中文说明

### 项目定位

Robot OLP Toolkit 是一个面向**机器人打磨、抛光、去毛刺和其他表面精加工任务**的离线编程（OLP）工具包，基于 **Pinocchio、PyVista 和 PySide6** 构建。

项目从机器人 URDF、工件 CAD/点云、打磨工具和 TCP 配置出发，帮助用户完成表面跟随刀路、连续逆运动学（IK）、关节约束、碰撞诊断、轨迹验证和三维回放。它用于仿真和工程验证，不是实际机器人控制器，也不是经过安全认证的运动规划器。

### 运行环境

- **操作系统**：Windows + PowerShell
- **Python**：3.10
- **环境管理**：Conda，环境文件为 [`environment.yml`](environment.yml)
- **原生依赖**：Pinocchio、碰撞检测库及其相关二进制组件

当前公开版本以 Windows/Conda 为支持环境，Linux 和 macOS 尚未作为目标环境进行验证。

### 面向打磨任务的典型流程

1. 加载机器人 URDF 和工件 CAD/点云模型。
2. 配置砂轮/打磨工具、法兰、TCP 偏置和接触姿态。
3. 生成或导入沿工件表面的打磨刀路，并设置工艺参数。
4. 对连续位姿序列执行 IK，检查关节限位和构型分支跳变。
5. 对插值后的轨迹进行碰撞、时间参数化和 TCP/FK 误差检查，再进行回放或导出。

同一套流程也可用于抛光、去毛刺等需要同时关注工具姿态和接触路径的任务。

### 主要内容

- **URDF 工具箱**：URDF 加载、xacro 编译、网格引用检查、网格转换和预览。
- **打磨 GUI**：机器人/工件场景、工具与 TCP、刀路工艺、IK、轨迹回放和碰撞诊断。
- **可复用模块**：Pinocchio 运动学、位姿求解、轨迹插值、FCL 碰撞服务、PyVista 渲染和规划适配器。
- **公开示例机器人**：自有的无网格六轴 URDF，不携带厂商机器人模型。
- **演示素材**：机器人场景、碰撞高亮、打磨轨迹曲线和 GIF 回放。

### GUI 使用指引

| 界面区域 | 作用 | 首次操作 |
| --- | --- | --- |
| 机器人与场景 | 加载机器人、坐标系、CAD 和环境物体 | 加载 `Demo Six Axis` URDF |
| 工件/点云 | 导入、滤波、裁剪和对齐打磨表面 | 仅测试 URDF 时可跳过 |
| 工具与 TCP | 配置砂轮/工具、法兰和 TCP 偏置 | 在三维视图中检查接触坐标轴 |
| 刀路与工艺 | 设置打磨参数并生成表面跟随刀路 | 先使用小范围工件测试 |
| IK 与关节约束 | 求解连续关节轨迹并检查限位 | 导入位姿/刀路 CSV |
| 轨迹与回放 | 时间参数化、验证、回放和导出 | 通过验证后再导出 |
| 碰撞与诊断 | 构建碰撞体并报告碰撞对象 | 重建场景后运行碰撞检查 |

### Windows 启动方式

```powershell
conda env create -f environment.yml
conda activate robot-olp-toolkit

# 启动主 GUI
python -m main_app.main_app

# 启动 URDF 工具箱
python -m robot_toolkit_main.robot_toolkit_main
```

已有环境也可以使用：

```powershell
.\scripts\activate_env.ps1
python -m main_app.main_app
```

### URDF 工具箱流程

1. 运行 `python -m robot_toolkit_main.robot_toolkit_main`。
2. 选择 URDF；公开示例位于 `examples/assets/urdf/demo_six_axis.urdf`。
3. 如果项目使用 xacro，使用本机可用的 xacro/ROS 工具链进行编译。
4. 检查网格引用，并对拥有合法授权的网格机器人进行转换和预览。
5. 在主 GUI 中加载同一个 URDF，继续检查运动学和场景行为。

### 公开范围与安全边界

公开仓库只包含可复用源码、GUI、文档、自有示例 URDF 和演示素材；不包含私有日志、实验数据、验证产物、编辑器设置或厂商机器人 URDF/网格。截图和 GIF 是 GUI 视觉证据，不代表真实机器人已经获得执行许可。正式打磨前仍需针对目标工件完成连续轨迹、碰撞、动力学、TCP/FK 和现场安全验证。

更多英文说明见下方 **English documentation**。第三方资产使用规则见 [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)。

## English documentation

## Runtime environment

The public workflow is maintained and demonstrated on **Windows with
PowerShell**, using **Python 3.10** and a Conda environment created from
[`environment.yml`](environment.yml). Pinocchio and the collision bindings
contain native components, so the Windows/Conda setup is the supported path for
the GUI and URDF toolbox. Linux and macOS are not the tested target environments
for this public subset.

Robot OLP Toolkit packages the reusable parts of a research-oriented robot
simulation workflow for robotic sanding and surface finishing. Starting from a
robot URDF and a workpiece represented by CAD or point-cloud data, it helps
organize tool/TCP setup, surface-following toolpaths, continuous inverse
kinematics (IK), collision diagnostics, and trajectory playback. It is a
**simulation and engineering-validation toolkit**, not a robot controller or a
safety-certified motion planner.

## Target task: robotic sanding and surface finishing

The public subset is organized around this practical sequence:

1. Load a robot URDF and a CAD/point-cloud workpiece into the scene.
2. Configure the sanding tool, flange, TCP offset, and contact orientation.
3. Generate or import a surface-following toolpath with process parameters.
4. Solve a continuous IK path while monitoring joint limits and branch changes.
5. Re-check interpolated collision edges, time parameterization, and TCP/FK
   error before playback or export.

This structure also supports polishing, deburring, and similar tool-surface
tasks where end-effector orientation and contact path matter as much as
position.

## What is included

- **URDF toolbox** — load a URDF, compile xacro when the local ROS/xacro tools
  are available, inspect mesh references, and convert mesh assets for preview.
- **Sanding-oriented GUI workflow** — robot/workpiece setup, abrasive tool and
  TCP configuration, toolpath and IK panels, trajectory playback, and
  collision diagnostics.
- **Reusable Python modules** — Pinocchio kinematics, pose solving, trajectory
  interpolation, FCL-based collision services, rendering helpers, and planner
  adapters.
- **A public demo robot** — an authored, mesh-free six-axis URDF that keeps the
  default registry runnable without shipping a vendor robot model.
- **README evidence** — screenshots and a GIF showing the GUI, collision
  highlighting, sanding trajectory plots, and playback.

## Visual demo

The following assets are GUI evidence from the source project. They show a
robot scene, collision diagnostics, a sanding toolpath with joint curves, and
animated playback. They are not claims of a certified robot trajectory and do
not include the vendor model packages used to produce the screenshots.

<p align="center">
  <img src="docs/evidence/robot_scene_irb4600.png" alt="Robot scene and FK controls" width="92%">
</p>

<p align="center">
  <img src="docs/evidence/collision_highlight_sphere.png" alt="Collision objects highlighted in red" width="92%">
</p>

<p align="center">
  <img src="docs/evidence/sanding_trajectory.png" alt="Robotic sanding toolpath and joint trajectory plot" width="92%">
</p>

<p align="center">
  <img src="docs/evidence/sanding_playback.gif" alt="Animated robotic sanding trajectory playback" width="92%">
</p>

## Quick start on Windows

The most reproducible route is Conda, because Pinocchio and its collision
bindings contain native components:

```powershell
conda env create -f environment.yml
conda activate robot-olp-toolkit

# Main seven-step GUI
python -m main_app.main_app

# URDF toolbox window
python -m robot_toolkit_main.robot_toolkit_main
```

If the environment already exists, the local helper performs the same checks:

```powershell
.\scripts\activate_env.ps1
python -m main_app.main_app
```

Optional utilities:

```powershell
# Generate small local STL primitives under generated_stl/ (ignored by Git)
python .\scripts\generate_stl.py

# Launch the point-cloud helper
python -m apps.point_cloud_generator
```

The `requirements-public.txt` file lists the Python-side packages. On a
different platform, install the Pinocchio/HPP-FCL stack using the equivalent
native packages for that platform before launching the GUI.

## GUI tour

| Area | Purpose | First action |
| --- | --- | --- |
| Robot and scene | Select the robot, coordinate frames, CAD, and environment objects | Load the bundled `Demo Six Axis` URDF |
| Workpiece / point cloud | Import the sanding surface, filter, crop, and align data | Skip this page for a URDF-only smoke test |
| Tool and TCP | Configure the abrasive tool, flange, and TCP offsets | Check the contact/tool axes in the 3-D view |
| Toolpath and process | Choose sanding/process parameters and generate a surface path | Use a small path before a full workpiece |
| IK and joint constraints | Solve a continuous joint path and inspect limits | Provide a pose/path CSV or use the previous page |
| Trajectory and playback | Time-parameterize, validate, play back, and export | Validate before treating a path as usable |
| Collision and diagnostics | Build collision geometry and report contacts | Rebuild the scene, then run the collision test |

The GUI is intentionally fail-closed around engineering checks: optimizer
convergence, a few collision-free samples, or a visually smooth playback are
not sufficient evidence for real hardware execution. Validate FK/TCP error,
joint margins, branch continuity, dynamics/time parameterization, and
interpolated collision edges for the target scene.

## Typical sanding workflow

For a first surface-finishing experiment, use a small CAD patch or point-cloud
region rather than a complete workpiece. Confirm the following in order:

- the tool frame points along the intended sanding direction and the TCP is at
  the abrasive contact point;
- adjacent surface poses produce a continuous joint branch instead of sudden
  flips;
- the tool, robot links, workpiece, and environment remain collision-free at
  the chosen interpolation resolution;
- the final playback shows the intended contact motion, not merely a reachable
  sequence of isolated poses.

## URDF toolbox workflow

1. Start `python -m robot_toolkit_main.robot_toolkit_main`.
2. Select a URDF (the bundled smoke-test file is
   `examples/assets/urdf/demo_six_axis.urdf`).
3. If your project uses xacro, compile it with the local xacro/ROS toolchain.
4. Inspect mesh references and use the converter/preview actions for licensed
   mesh-based robots.
5. Load the same URDF in the main GUI to check kinematics and scene behavior.

The preview code understands mesh references; primitive-only URDFs are still
valid Pinocchio inputs and use the renderer's simplified fallback model.

## Public project boundary

This is a curated public subset of a larger private workspace. The repository
contains source modules, the GUI, public documentation, an authored demo URDF,
and visual evidence. It intentionally excludes private logs, experimental
datasets, generated validation artifacts, editor settings, and vendor robot
URDF/mesh packages. See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) before
adding external assets.

## Project structure

```text
core/                 # Reusable kinematics, IK, toolpath, and pipeline modules
main_app/             # PySide6 seven-step GUI
robot_toolkit_main/   # URDF/xacro/mesh toolbox GUI
render_engine/        # PyVista robot and scene rendering
collision/            # FCL collision services and state validity helpers
trajectory/           # Planning, validation, and export adapters
algorithms/           # IK, slicing, and optimization algorithms
plotting/             # Trajectory plots and limit helpers
apps/                 # Optional point-cloud utility
scripts/              # Environment and STL helper scripts
examples/assets/      # Authored, mesh-free public URDF
docs/evidence/        # README screenshots and GIF
tests/                # Public layout/import smoke tests
```

## Verification

After installing the environment, run the lightweight public checks:

```powershell
python -m pytest tests -q
```

The checks validate the public URDF, package boundaries, and imports. Full
trajectory and collision acceptance still depends on the selected robot,
licensed CAD, and the native Pinocchio/FCL versions installed on the machine.

## License

The original source code, documentation, and authored demo URDF are released
under the [MIT License](LICENSE). The files in `docs/evidence/` are GUI visual
evidence; if a screenshot depicts a third-party robot or CAD asset, that asset
is not relicensed by this repository. External URDFs, meshes, point clouds,
and other assets loaded by a user keep their own licenses and attribution
requirements.
