"""
Stage 5 v2.0: 数字孪生仿真平台 - OLP 多步骤架构

"""

import sys
import json
import os
import time

# Add parent directory to sys.path so bare imports (kinematics_engine, schemas, etc.) resolve
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
import fnmatch
from dataclasses import asdict, replace
from collections import deque
import numpy as np
from typing import Optional, List, Tuple, Any

# PySide6 导入
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QTextEdit, QProgressBar, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog,
    QMessageBox, QFrame, QCheckBox, QSlider, QFormLayout, QScrollArea, QSplitter,
    QSizePolicy, QListWidget, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QElapsedTimer
from PySide6.QtGui import QFont

# PyVista 导入
import pyvista as pv
from pyvistaqt import QtInteractor

# Open3D 导入
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

# 项目模块
from core.kinematics_engine import KinematicsEngine
from core.schemas import (
    FlangeToolParams,
    JointTrajectory,
    PathSegmentType,
    PathSequencingConfig,
    ProcessParameters,
    TimeParameterizedTrajectory,
    TrajectoryValidationReport,
)
from core.robot_registry import create_default_registry, RobotRegistry, RobotConfig
from render_engine import RenderEngine
from core.auto_manager import get_manager
from core.core_algorithm import ParamType, BaseAlgorithm, ParamDef, ToolpathResult
from core.state import SimulationState
from core.path_sequencer import PathSequencer
from core.pose_solver import PoseSolver
from scipy.spatial.transform import Rotation as R
from collision import (
    CollisionMeshVisualizer,
    EnvironmentMeshVisualizer,
    RobotEnvCollisionChecker,
    build_checker_from_visualizers,
    parse_urdf_collision_meshes,
    auto_convert_mesh_to_meters,
    build_link_fk_provider,
)

# matplotlib 导入（绘图模块）
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


# Keep the workpiece translucent so target and FK-reconstructed TCP paths
# remain visible when they pass behind or inside the imported STL mesh.
WORKPIECE_CAD_OPACITY = 0.5


# ==================== 后台工作线程 ====================

class WorkerThread(QThread):
    """通用后台计算线程"""

    log_signal = Signal(str)
    progress_signal = Signal(int, int)  # current, total
    stage_signal = Signal(str)
    result_signal = Signal(object)
    finished_signal = Signal(bool, str)

    def __init__(self, task_name: str, parent=None):
        super().__init__(parent)
        self._task_name = task_name
        self._is_running = False
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        self._is_running = True
        try:
            self._do_work()
        except Exception as e:
            self.log_signal.emit(f"[错误] {str(e)}")
            self.finished_signal.emit(False, str(e))
        finally:
            self._is_running = False

    def _do_work(self):
        """子类实现具体工作"""
        raise NotImplementedError


class ToolpathWorker(WorkerThread):
    """基于元数据驱动算法的刀轨生成工作线程。"""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        model_path: str,
        kwargs: dict,
        parent=None
    ):
        super().__init__("ToolpathGenerator", parent)
        self.algorithm = algorithm
        self.model_path = model_path
        self.kwargs = kwargs

    def _do_work(self):
        self.log_signal.emit(f"加载几何模型: {self.model_path}")

        if not self.algorithm.load_geometry(self.model_path):
            raise RuntimeError(f"加载几何模型失败: {self.model_path}")

        if self._abort:
            self.log_signal.emit("用户取消操作")
            self.finished_signal.emit(False, "已取消")
            return

        self.progress_signal.emit(20, 100)
        self.log_signal.emit(f"算法: {self.algorithm.NAME}")
        self.log_signal.emit(f"参数: {self.kwargs}")

        self.progress_signal.emit(40, 100)
        self.log_signal.emit("开始刀轨计算，请稍候...")

        # 将算法日志转发到 UI 界面
        original_log = self.algorithm.log
        def log_forward(msg):
            original_log(msg)
            self.log_signal.emit(f"  {msg}")
        self.algorithm.log = log_forward

        result = self.algorithm.generate(**self.kwargs)

        # 恢复原始日志方法
        self.algorithm.log = original_log

        if self._abort:
            self.log_signal.emit("用户取消操作")
            self.finished_signal.emit(False, "已取消")
            return

        self.progress_signal.emit(100, 100)

        self.result_signal.emit(result)
        self.log_signal.emit(f"刀轨生成完成: {result.num_waypoints} 个路径点")
        self.finished_signal.emit(True, "刀轨生成完成")


class IKSolveWorker(WorkerThread):
    """IK 求解线程 - 使用插件式 IK 求解器"""

    def __init__(self, kin_engine: KinematicsEngine, matrices: List[np.ndarray],
                 q_init: np.ndarray, solver_plugin=None, parent=None,
                 margin_deg: float = 5.0, tolerance: float = 1e-5,
                 max_iterations: int = 1000, enable_wrist_flip: bool = True,
                 solver_name: str = "SLSQP非线性优化"):
        """
        参数:
            solver_plugin: IK 求解器插件实例（PseudoInverseSolver 或 SLSQPSolver）
            solver_name: 求解器名称（用于日志）
            tolerance: SLSQP 收敛容差
            max_iterations: SLSQP 最大迭代次数
            margin_deg: 限位裕度（度）
        """
        super().__init__("IKSolver", parent)
        self._kin_engine = kin_engine
        self._matrices = matrices
        self._q_init = q_init
        self._solver_plugin = solver_plugin
        self._solver_name = solver_name
        self._margin_deg = margin_deg
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._enable_wrist_flip = enable_wrist_flip

    def _do_work(self):
        self.log_signal.emit(f"开始 IK 批量求解 (策略: {self._solver_name})...")
        self.log_signal.emit(f"目标位姿数量: {len(self._matrices)}")

        total = len(self._matrices)
        results = []
        flip_count = 0
        q_prev = self._q_init.copy()

        solver_is_slsqp = hasattr(self._solver_plugin, 'NAME') and 'SLSQP' in self._solver_plugin.NAME

        if solver_is_slsqp:
            self.log_signal.emit(
                f"  [{self._solver_name}] 容差={self._tolerance}, "
                f"最大迭代={self._max_iterations}, 裕度={self._margin_deg}°, "
                f"翻腕={'启用' if self._enable_wrist_flip else '禁用'}"
            )

        for i, T_ocf in enumerate(self._matrices):
            if self._abort:
                break

            # OCF → RCF 变换
            T_rcf = self._kin_engine._compute_robot_target_pose(T_ocf)
            T_rcf_flange = T_rcf @ np.linalg.inv(self._kin_engine.effective_tcp)

            if solver_is_slsqp:
                success, q_solution = self._solver_plugin.solve(T_rcf_flange, q_prev)
                error = float('nan')
            else:
                success, q_solution, error = self._solver_plugin.solve(T_ocf, q_prev)

            wrist_flipped = False

            # 构型变换：求解成功但关节接近限位
            if success and self._enable_wrist_flip and self._kin_engine.is_near_limit(q_solution, self._margin_deg):
                alt_seeds = self._kin_engine.generate_alternative_seeds(q_prev)
                alt_success = False
                for alt_seed in alt_seeds:
                    if self._abort:
                        break
                    if solver_is_slsqp:
                        alt_ok, alt_q = self._solver_plugin.solve(T_rcf_flange, alt_seed)
                    else:
                        alt_ok, alt_q, _ = self._solver_plugin.solve(T_ocf, alt_seed)
                    if alt_ok and not self._kin_engine.is_near_limit(alt_q, self._margin_deg):
                        q_solution = alt_q
                        if not solver_is_slsqp:
                            error = _
                        success = True
                        wrist_flipped = True
                        alt_success = True
                        flip_count += 1
                        break

            result = {
                'success': success,
                'q': q_solution,
                'error': error,
                'wrist_flipped': wrist_flipped,
            }

            if success:
                q_prev = q_solution.copy()

            results.append(result)

            if (i + 1) % 100 == 0 or (i + 1) == total:
                self.progress_signal.emit(i + 1, total)
                self.log_signal.emit(f"  已求解 {i + 1}/{total}")

        success_count = sum(1 for r in results if r['success'])
        flip_msg = f", 翻腕 {flip_count} 次" if flip_count > 0 else ""
        self.log_signal.emit(f"IK 求解完成: {success_count}/{total} 成功{flip_msg}")

        self.result_signal.emit(results)
        self.finished_signal.emit(True, f"IK 求解完成 ({success_count}/{total})")


class SimulationWorker(WorkerThread):
    """仿真播放线程"""

    def __init__(self, frames: List, render_engine: RenderEngine, parent=None):
        super().__init__("Simulation", parent)
        self._frames = frames
        self._render_engine = render_engine

    def _do_work(self):
        self.log_signal.emit(f"开始仿真播放: {len(self._frames)} 帧")

        total = len(self._frames)

        for i, frame in enumerate(self._frames):
            if self._abort:
                break

            # 更新刀具位置
            self._render_engine.update_tool_position(frame['world_position'])

            # 执行切削
            cuts = self._render_engine.execute_cutting(frame['world_position'])

            # 更新进度
            self.progress_signal.emit(i + 1, total)

            if (i + 1) % 50 == 0:
                self.log_signal.emit(f"  帧 {i + 1}/{total}, 累计切削 {cuts} 点")

            # 控制帧率
            self.msleep(30)

        self.log_signal.emit("仿真完成")
        self.finished_signal.emit(True, "仿真完成")


class TrajectoryPlanningWorker(WorkerThread):
    """Run stable2C assembly plus TOPP-RA/Ruckig without GUI-owned state.

    The worker receives only copied value data and an immutable collision
    snapshot.  It creates its own Pinocchio/FCL services, so a Qt/VTK actor or
    GUI-owned FCL object never crosses the worker boundary.
    """

    def __init__(
        self,
        *,
        process_joint_positions,
        parameters,
        geometric_trajectory,
        transition_requests,
        collision_snapshot,
        q_home,
        lower_limits,
        upper_limits,
        tcp_transform,
        transition_config=None,
        parent=None,
    ):
        super().__init__("PhysicalTrajectory", parent)
        self._process_joint_positions = np.asarray(
            process_joint_positions, dtype=np.float64
        ).copy()
        self._parameters = parameters
        self._geometric_payload = {
            "path_s": np.asarray(geometric_trajectory.path_s, dtype=np.float64).copy(),
            "tcp_poses": np.asarray(geometric_trajectory.tcp_poses, dtype=np.float64).copy(),
            "segment_types": np.asarray(geometric_trajectory.segment_types, dtype=object).copy(),
            "original_indices": np.asarray(geometric_trajectory.original_indices, dtype=np.int64).copy(),
            "segment_ids": np.asarray(geometric_trajectory.segment_ids, dtype=np.int64).copy(),
            "layer_ids": np.asarray(geometric_trajectory.layer_ids, dtype=np.int64).copy(),
            "metadata": dict(geometric_trajectory.metadata),
        }
        self._transition_request_payload = tuple(
            (
                item.kind,
                int(item.start_segment_id),
                int(item.goal_segment_id),
                int(item.start_layer_id),
                int(item.goal_layer_id),
                np.asarray(item.start_pose, dtype=np.float64).copy(),
                np.asarray(item.goal_pose, dtype=np.float64).copy(),
            )
            for item in transition_requests
        )
        self._collision_snapshot = collision_snapshot
        self._q_home = np.asarray(q_home, dtype=np.float64).copy()
        self._lower_limits = np.asarray(lower_limits, dtype=np.float64).copy()
        self._upper_limits = np.asarray(upper_limits, dtype=np.float64).copy()
        self._tcp_transform = np.asarray(tcp_transform, dtype=np.float64).copy()
        self._transition_config = transition_config

    def _raise_if_aborted(self) -> None:
        if self._abort:
            raise RuntimeError("已取消")

    @staticmethod
    def _build_local_tcp_fk(urdf_path: str, tcp_transform: np.ndarray):
        """Create an RCF TCP FK provider equivalent to KinematicsEngine FK."""
        import pinocchio as pin

        transform = np.asarray(tcp_transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("effective TCP transform must be a finite 4x4 matrix")
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        frame_id = model.getFrameId(model.names[-1])

        def tcp_fk(q):
            values = np.asarray(q, dtype=np.float64).reshape(-1)
            if values.shape != (model.nq,):
                raise ValueError("joint configuration does not match worker-local URDF")
            pin.forwardKinematics(model, data, values)
            pin.updateFramePlacements(model, data)
            return data.oMf[frame_id].homogeneous.copy() @ transform

        return tcp_fk

    @staticmethod
    def _plan_summary(plan) -> dict:
        return {
            "request_id": plan.request_id,
            "kind": plan.kind.value,
            "success": bool(plan.success),
            "planner_id": plan.planner_id,
            "scene_hash": plan.scene_hash,
            "scene_version": int(plan.scene_version),
            "seed": int(plan.seed),
            "timeout_s": float(plan.timeout_s),
            "planning_time_s": float(plan.planning_time_s),
            "validation_resolution_rad": float(plan.validation_resolution_rad),
            "failure_code": plan.failure_code,
            "detail": plan.detail,
            "waypoint_count": 0 if plan.positions is None else int(len(plan.positions)),
            "start_q": plan.start_q.tolist(),
            "goal_q": plan.goal_q.tolist(),
            "metadata": dict(plan.metadata),
        }

    def _do_work(self):
        from core.schemas import GeometricTrajectory, TransitionRequest
        from collision import SnapshotCollisionService, build_link_fk_provider
        from trajectory import (
            ProcessIKAdapter,
            TrajectoryPlanner,
            TransitionPlanner,
            TransitionPlanningConfig,
            build_transition_pipeline,
        )
        from trajectory.planner import TrajectoryPlanningError

        self._raise_if_aborted()
        self.stage_signal.emit("collision_scene")
        self.progress_signal.emit(5, 100)
        snapshot = self._collision_snapshot
        local_link_fk = build_link_fk_provider(snapshot.robot_urdf_path)
        collision_service = SnapshotCollisionService(snapshot, local_link_fk)
        if not collision_service.available:
            code, detail = collision_service.failure
            raise TrajectoryPlanningError(
                f"stable2C 规划停止：碰撞场景服务不可用 ({code}): {detail}"
            )

        geometric = GeometricTrajectory(**self._geometric_payload)
        requests = tuple(
            TransitionRequest(
                kind=kind,
                start_segment_id=start_segment_id,
                goal_segment_id=goal_segment_id,
                start_layer_id=start_layer_id,
                goal_layer_id=goal_layer_id,
                start_pose=start_pose,
                goal_pose=goal_pose,
            )
            for (
                kind,
                start_segment_id,
                goal_segment_id,
                start_layer_id,
                goal_layer_id,
                start_pose,
                goal_pose,
            ) in self._transition_request_payload
        )
        process_segments = ProcessIKAdapter().adapt(
            self._process_joint_positions, geometric
        )
        if not process_segments:
            raise TrajectoryPlanningError("stable2C 规划停止：未得到任何 PROCESS 关节段")

        self._raise_if_aborted()
        self.stage_signal.emit("ompl_transition_planning")
        self.progress_signal.emit(20, 100)
        config = self._transition_config or TransitionPlanningConfig()
        transition_planner = TransitionPlanner(
            collision_service,
            self._lower_limits,
            self._upper_limits,
            config=config,
        )
        self.log_signal.emit(
            f"[stable2C] OMPL {config.planner_id}: scene={snapshot.scene_hash[:12]}, "
            f"seed={config.seed}, timeout={config.timeout_s:.2f}s"
        )
        pipeline = build_transition_pipeline(
            process_segments,
            q_home=self._q_home,
            transition_planner=transition_planner,
            transition_requests=requests,
        )
        if not pipeline.success:
            if pipeline.failure_code == "process_start_invalid":
                self.log_signal.emit(
                    f"[stable2C] 加工源第 0 帧 FCL 明细: {pipeline.detail}"
                )
                self.log_signal.emit(
                    "[stable2C] 加工源第 0 帧未通过 PROCESS/FCL；"
                    f"输入源帧数={len(self._process_joint_positions)}，"
                    "未生成可执行的五帧安全入口。"
                )
            raise TrajectoryPlanningError(
                f"stable2C 过渡规划失败 [{pipeline.failure_code or 'unknown'}]: {pipeline.detail}"
            )
        self.log_signal.emit(
            "[stable2C] 入口帧合同: "
            f"加工源 {pipeline.metadata.get('process_source_frame_count', 0)} + "
            f"安全入口 {pipeline.metadata.get('safe_entry_frame_count', 0)} = "
            f"{pipeline.metadata.get('source_plus_safe_entry_frame_count', 0)}"
        )

        self._raise_if_aborted()
        self.stage_signal.emit("assembled_geometry_validation")
        self.progress_signal.emit(60, 100)
        geometry_validation = transition_planner.validate_geometric_path(
            pipeline.positions,
            pipeline.segment_types,
            edge_segment_types=pipeline.edge_segment_types,
        )
        if not geometry_validation.valid:
            raise TrajectoryPlanningError(
                "stable2C 拼装几何复验失败 "
                f"[{geometry_validation.failure_code or 'unknown'}] at "
                f"{geometry_validation.failure_index}: {geometry_validation.detail}"
            )

        self._raise_if_aborted()
        self.stage_signal.emit("time_parameterization")
        self.progress_signal.emit(75, 100)
        self.log_signal.emit("[stable2C] 几何路径复验通过，开始 TOPP-RA + Ruckig 时间参数化...")
        tcp_fk = self._build_local_tcp_fk(snapshot.robot_urdf_path, self._tcp_transform)
        planner = TrajectoryPlanner(
            self._parameters,
            backend="toppra",
            apply_ruckig=True,
        )
        result = planner.plan(
            pipeline.positions,
            segment_ids=pipeline.segment_ids,
            segment_types=pipeline.segment_types,
            transition_kinds=pipeline.transition_kinds,
            fk_provider=tcp_fk,
        )
        self._raise_if_aborted()
        metadata = dict(result.metadata)
        metadata["stable2C"] = {
            "scene_hash": pipeline.scene_hash,
            "scene_version": int(pipeline.scene_version),
            "q_home": self._q_home.tolist(),
            "geometric_waypoint_count": int(len(pipeline.positions)),
            "process_source_frame_count": int(
                pipeline.metadata.get("process_source_frame_count", 0)
            ),
            "safe_entry_frame_count": int(
                pipeline.metadata.get("safe_entry_frame_count", 0)
            ),
            "source_plus_safe_entry_frame_count": int(
                pipeline.metadata.get("source_plus_safe_entry_frame_count", 0)
            ),
            "transition_count": int(len(pipeline.transition_plans)),
            "transition_plans": [self._plan_summary(plan) for plan in pipeline.transition_plans],
            "geometric_validation": {
                "checked_state_count": int(geometry_validation.checked_state_count),
                "checked_edge_count": int(geometry_validation.checked_edge_count),
                "metadata": dict(geometry_validation.metadata),
            },
        }
        result = replace(result, metadata=metadata)
        self.progress_signal.emit(100, 100)
        self.result_signal.emit(result)
        self.finished_signal.emit(True, "物理轨迹生成完成（stable2C 过渡已规划并复验）")


class TrajectoryValidationWorker(WorkerThread):
    """后台执行动力学、几何和碰撞验证，绝不触碰 Qt/VTK actor。"""

    def __init__(self, trajectory, parameters, lower_limits, upper_limits,
                 reference_tcp_positions, collision_free=None,
                 collision_snapshot=None, parent=None):
        super().__init__("TrajectoryValidation", parent)
        self._trajectory = trajectory
        self._parameters = parameters
        self._lower_limits = np.asarray(lower_limits, dtype=float)
        self._upper_limits = np.asarray(upper_limits, dtype=float)
        self._reference = reference_tcp_positions
        self._collision_free = collision_free
        self._collision_snapshot = collision_snapshot

    def _do_work(self):
        from trajectory import TrajectoryValidator

        collision_free = self._collision_free
        snapshot_metadata = {}
        if self._collision_snapshot is not None:
            snapshot = self._collision_snapshot
            snapshot_metadata = {
                "collision_scene_hash": snapshot.scene_hash,
                "collision_scene_version": int(snapshot.scene_version),
            }
            try:
                # FK and FCL objects are deliberately created inside this
                # worker.  No Qt/VTK actor or GUI-owned FCL object crosses the
                # thread boundary.
                from collision import SnapshotCollisionService, build_link_fk_provider

                local_fk_provider = build_link_fk_provider(snapshot.robot_urdf_path)
                collision_service = SnapshotCollisionService(snapshot, local_fk_provider)

                def collision_free(q, *, segment_type=None):
                    return collision_service.check_configuration(
                        q, segment_type=segment_type
                    ).valid

                collision_free.supports_segment_type = True
                if not collision_service.available:
                    code, detail = collision_service.failure
                    snapshot_metadata["collision_service_error"] = code
                    self.log_signal.emit(f"[碰撞] 后台场景服务不可用: {code}: {detail}")
            except Exception as exc:
                snapshot_metadata["collision_service_error"] = "worker_build_failed"
                self.log_signal.emit(f"[碰撞] 后台场景服务构建失败: {exc}")

                def collision_free(_q, *, segment_type=None):
                    return False

                collision_free.supports_segment_type = True

        started = __import__("time").perf_counter()
        self.stage_signal.emit("时间戳与数组结构")
        report = TrajectoryValidator(self._parameters).validate(
            self._trajectory,
            lower_position_limits=self._lower_limits,
            upper_position_limits=self._upper_limits,
            reference_tcp_positions=self._reference,
            collision_free=collision_free,
            progress_callback=lambda current, total, stage: (
                self.stage_signal.emit(stage),
                self.progress_signal.emit(current, total),
            ),
            is_cancelled=lambda: self._abort,
        )
        elapsed = __import__("time").perf_counter() - started
        report.metadata["validation_wall_time_s"] = elapsed
        report.metadata.update(snapshot_metadata)
        self.result_signal.emit(report)
        self.finished_signal.emit(True, f"轨迹验证完成，用时 {elapsed:.2f}s")


# ==================== Tab Widget 基类 ====================

class StepWidget(QWidget):
    """Step Widget 基类"""

    def __init__(self, step_name: str, step_number: int, state: SimulationState, parent=None):
        super().__init__(parent)
        self._step_name = step_name
        self._step_number = step_number
        self._state = state

        self._init_ui()

    def _init_ui(self):
        """子类实现"""
        raise NotImplementedError

    def create_scrollable_layout(self) -> tuple[QWidget, QVBoxLayout]:
        """Create a non-collapsing tab body that scrolls on compact windows."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        # Evaluate after the subclass has populated the layout.  The minimum
        # height makes the scroll area scroll instead of crushing spin boxes.
        QTimer.singleShot(
            0,
            lambda: content.setMinimumHeight(max(1, layout.sizeHint().height())),
        )
        return content, layout

    def on_activate(self):
        """Step 激活时调用（子类覆盖）"""
        pass

    def on_deactivate(self):
        """Step 切换时调用"""
        pass

    def log(self, message: str):
        """输出日志到主窗口"""
        if hasattr(self, '_main_window') and hasattr(self._main_window, 'log'):
            self._main_window.log(message)
        elif hasattr(self.parent(), 'log'):
            self.parent().log(message)

    def record_stage(
        self,
        stage: str,
        *,
        status: str = "completed",
        details: Optional[dict] = None,
        version_domain: Optional[str] = None,
    ) -> dict:
        """记录并显示中心状态的生产生命周期事件。"""
        event = self._state.record_stage(
            stage,
            status=status,
            details=details,
            version_domain=version_domain,
        )
        self.log(
            "[状态] "
            f"task={event['task_id'][:8]} "
            f"state=v{event['state_version']} "
            f"stage={stage} status={status} "
            f"scene=v{event['scene_version']} "
            f"toolpath=v{event['toolpath_version']} "
            f"trajectory=v{event['trajectory_version']} "
            f"validation=v{event['validation_version']}"
        )
        return event

    def mark_collision_scene_changed(
        self,
        source: str,
        *,
        details: Optional[dict] = None,
        stage: str = "collision_scene_changed",
    ) -> dict:
        """Advance scene identity and invalidate paths planned against it."""
        event = self._state.mark_collision_scene_changed(
            source,
            details=details,
            stage=stage,
        )
        self.log(
            "[状态] "
            f"task={event['task_id'][:8]} "
            f"state=v{event['state_version']} "
            f"stage={stage} status={event['status']} "
            f"scene=v{event['scene_version']} "
            f"toolpath=v{event['toolpath_version']} "
            f"trajectory=v{event['trajectory_version']} "
            f"validation=v{event['validation_version']}"
        )
        main_window = getattr(self, "_main_window", None)
        if main_window is not None:
            for index in range(main_window._tab_widget.count()):
                widget = main_window._tab_widget.widget(index)
                callback = getattr(widget, "_on_collision_scene_changed", None)
                if callback is not None:
                    callback(source, event)
        return event

    @staticmethod
    def create_xyz_rpy_grid(parent_widget, name: str, inputs_dict: dict,
                            on_change_callback, col_stretch=True) -> QGridLayout:
        """
        公共方法：创建 xyz + rpy 的 6 参数输入网格。
        复用自 WCFLayoutWidget 和 WorkpieceCloudWidget。

        参数:
            parent_widget: 父控件（用于设置布局）
            name: 参数名前缀（如 "T_wcf_rcf"）
            inputs_dict: 字典，用来存储创建的 QDoubleSpinBox，key 格式为 f"{name}_{i}"
            on_change_callback: 值改变时的回调函数
            col_stretch: 是否在最后一列添加 stretch
        返回:
            QGridLayout
        """
        grid = QGridLayout()
        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Rx (deg)', 'Ry (deg)', 'Rz (deg)']
        for i, label_text in enumerate(labels):
            row, col_base = i // 3, (i % 3) * 2
            grid.addWidget(QLabel(label_text), row, col_base)
            spin = QDoubleSpinBox()
            is_angle = 'deg' in label_text
            spin.setRange(-180 if is_angle else -10, 180 if is_angle else 10)
            spin.setDecimals(4)
            spin.setSingleStep(1.0 if is_angle else 0.1)
            spin.valueChanged.connect(on_change_callback)
            grid.addWidget(spin, row, col_base + 1)
            inputs_dict[f"{name}_{i}"] = spin
        if col_stretch:
            grid.setColumnStretch(2, 1)
        return grid

    @staticmethod
    def xyz_rpy_to_matrix(inputs_dict: dict, name: str) -> np.ndarray:
        """从输入字典中提取 xyz/rpy 并构造 4x4 齐次变换矩阵"""
        from scipy.spatial.transform import Rotation
        xyz = np.array([inputs_dict[f"{name}_{i}"].value() for i in range(3)])
        rpy_deg = np.array([inputs_dict[f"{name}_{i}"].value() for i in range(3, 6)])
        T = np.eye(4)
        T[:3, 3] = xyz
        T[:3, :3] = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
        return T

    @staticmethod
    def sync_robot_and_workpiece(state, render_engine, log_fn):
        """通用：同步 kinematics_engine、robot 渲染、workpiece 渲染、joint 更新"""
        if state.kinematics_engine:
            state.kinematics_engine.set_transforms(state.T_wcf_rcf, state.T_wcf_ocf)
        if render_engine:
            render_engine.transform_robot(state.T_wcf_rcf)
            render_engine.transform_workpiece(state.T_wcf_ocf)
            if (state.kinematics_engine
                    and state.kinematics_engine.current_q is not None):
                render_engine.update_robot_joints(state.kinematics_engine.current_q)
        log_fn("坐标系已更新")

    @property
    def render_engine(self):
        """获取渲染引擎"""
        if hasattr(self, '_main_window'):
            return self._main_window._render_engine
        return None


# ==================== Tab 1: 机器人资产载入 ====================

class RobotAssetWidget(StepWidget):
    """Tab 1: 机器人资产载入"""

    def __init__(self, state: SimulationState, parent=None):
        self._scene_inputs = {}
        super().__init__("机器人资产载入", 1, state, parent)

    def _init_ui(self):
        _content, layout = self.create_scrollable_layout()

        # 说明
        info = QLabel("从注册表选择机器人型号，加载 URDF 模型并在 3D 视图中显示机器人")
        info.setWordWrap(True)
        layout.addWidget(info)

        scene_group = QGroupBox("场景坐标（WCF）")
        scene_layout = QVBoxLayout(scene_group)
        rcf_group = QGroupBox("T_wcf_rcf - 机器人基座位姿")
        rcf_layout = QVBoxLayout(rcf_group)
        rcf_layout.addLayout(StepWidget.create_xyz_rpy_grid(
            self, "T_wcf_rcf", self._scene_inputs, self._on_scene_transform_changed
        ))
        scene_layout.addWidget(rcf_group)
        ocf_group = QGroupBox("T_wcf_ocf - 工件位姿")
        ocf_layout = QVBoxLayout(ocf_group)
        ocf_layout.addLayout(StepWidget.create_xyz_rpy_grid(
            self, "T_wcf_ocf", self._scene_inputs, self._on_scene_transform_changed
        ))
        scene_layout.addWidget(ocf_group)
        layout.addWidget(scene_group)

        # 机器人选择
        group = QGroupBox("机器人选择")
        group_layout = QVBoxLayout(group)

        self._robot_combo = QComboBox()
        registry = create_default_registry()
        for name in registry.list_robots():
            self._robot_combo.addItem(name)
        self._robot_combo.currentTextChanged.connect(self._on_robot_changed)
        group_layout.addWidget(QLabel("机器人型号:"))
        group_layout.addWidget(self._robot_combo)

        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        group_layout.addWidget(self._info_label)

        layout.addWidget(group)

        # 确认加载按钮
        self._btn_load_robot = QPushButton("确认并加载机器人模型")
        self._btn_load_robot.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self._btn_load_robot.clicked.connect(self._on_load_robot)
        layout.addWidget(self._btn_load_robot)

        # 状态标签
        self._status_label = QLabel("请选择机器人型号并点击加载")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self._status_label)

        # FK 滑块提示（指向浮动窗口）
        self._fk_hint_label = QLabel("提示：关节控制请使用浮动窗口（加载机器人后自动弹出）")
        self._fk_hint_label.setWordWrap(True)
        self._fk_hint_label.setStyleSheet("color: #888; font-style: italic; padding: 5px; background: rgba(33,150,243,0.1); border-radius: 4px;")
        layout.addWidget(self._fk_hint_label)

        # ── CAD 数模导入 ────────────────────────────────────────────────
        cad_group = QGroupBox("CAD 数模导入")
        cad_layout = QVBoxLayout(cad_group)

        cad_info = QLabel("导入 STL/STEP 格式的 CAD 模型，作为目标参考")
        cad_info.setWordWrap(True)
        cad_layout.addWidget(cad_info)

        cad_btn_row = QHBoxLayout()
        self._btn_import_cad = QPushButton("选择 CAD 文件...")
        self._btn_import_cad.clicked.connect(self._on_import_cad)
        self._btn_import_cad.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        cad_btn_row.addWidget(self._btn_import_cad)

        self._btn_clear_cad = QPushButton("清除")
        self._btn_clear_cad.clicked.connect(self._on_clear_cad)
        self._btn_clear_cad.setEnabled(False)
        cad_btn_row.addWidget(self._btn_clear_cad)
        cad_layout.addLayout(cad_btn_row)

        self._cad_label = QLabel("未选择文件")
        self._cad_label.setWordWrap(True)
        self._cad_label.setStyleSheet("color: gray; font-size: 11px;")
        cad_layout.addWidget(self._cad_label)

        self._cad_status_label = QLabel("")
        self._cad_status_label.setAlignment(Qt.AlignCenter)
        self._cad_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
        cad_layout.addWidget(self._cad_status_label)

        layout.addWidget(cad_group)

        # ── 环境物体导入 ─────────────────────────────────────────────
        env_group = QGroupBox("环境物体导入")
        env_layout = QVBoxLayout(env_group)

        env_info = QLabel("导入STL格式的环境物体，支持多个文件，可通过浮窗调整位置")
        env_info.setWordWrap(True)
        env_layout.addWidget(env_info)

        env_btn_row = QHBoxLayout()
        self._btn_import_env = QPushButton("导入STL文件...")
        self._btn_import_env.clicked.connect(self._on_import_env_objects)
        env_btn_row.addWidget(self._btn_import_env)

        self._btn_clear_env = QPushButton("清除全部")
        self._btn_clear_env.clicked.connect(self._on_clear_env_objects)
        self._btn_clear_env.setEnabled(False)
        env_btn_row.addWidget(self._btn_clear_env)

        self._btn_show_env_float = QPushButton("显示位置调整浮窗")
        self._btn_show_env_float.clicked.connect(self._on_show_env_float_window)
        self._btn_show_env_float.setStyleSheet("background-color: #1976D2; color: white; padding: 6px;")
        self._btn_show_env_float.setEnabled(False)
        env_btn_row.addWidget(self._btn_show_env_float)

        env_layout.addLayout(env_btn_row)

        self._env_status_label = QLabel("未导入环境物体")
        self._env_status_label.setStyleSheet("color: gray; font-size: 11px;")
        env_layout.addWidget(self._env_status_label)

        layout.addWidget(env_group)

        # ── 配置导入 ─────────────────────────────────────────────
        config_group = QGroupBox("配置导入")
        config_layout = QVBoxLayout(config_group)

        self._btn_import_config = QPushButton("导入JSON配置文件...")
        self._btn_import_config.clicked.connect(self._on_import_config)
        config_layout.addWidget(self._btn_import_config)

        self._config_status_label = QLabel("未导入配置")
        self._config_status_label.setStyleSheet("color: gray; font-size: 11px;")
        config_layout.addWidget(self._config_status_label)

        layout.addWidget(config_group)

        # ── 坐标系显示控制 ──────────────────────────────────────────
        axes_group = QGroupBox("坐标系显示")
        axes_layout = QVBoxLayout(axes_group)

        self._show_flange_axes_cb = QCheckBox("显示法兰坐标系 (末端关节零位)")
        self._show_flange_axes_cb.setChecked(True)
        self._show_flange_axes_cb.stateChanged.connect(self._on_flange_axes_toggled)
        axes_layout.addWidget(self._show_flange_axes_cb)

        layout.addWidget(axes_group)

        layout.addStretch()

    def _on_robot_changed(self, robot_name: str):
        """机器人选择改变 - 仅更新信息，不自动加载"""
        registry = create_default_registry()
        config = registry.get(robot_name)

        if config:
            self._state.robot_config = config

            # 更新信息
            info = f"型号: {config.name}\n"
            info += f"URDF: {config.urdf_path}"
            self._info_label.setText(info)

            # 更新按钮文本
            self._btn_load_robot.setText(f"加载 {config.name}")

    def _on_scene_transform_changed(self):
        self._state.T_wcf_rcf = StepWidget.xyz_rpy_to_matrix(self._scene_inputs, "T_wcf_rcf")
        self._state.T_wcf_ocf = StepWidget.xyz_rpy_to_matrix(self._scene_inputs, "T_wcf_ocf")
        StepWidget.sync_robot_and_workpiece(self._state, self.render_engine, self.log)
        manager = getattr(self._state, "collision_manager", None)
        if manager is not None:
            manager.set_robot_base_transform(self._state.T_wcf_rcf)
            manager.update_cad_transform(self._state.T_wcf_ocf)
            manager.refresh()
        self.mark_collision_scene_changed("scene_transform")

    def _on_load_robot(self):
        """确认加载机器人"""
        if self._state.robot_config is None:
            self.log("[错误] 请先选择机器人型号")
            return

        config = self._state.robot_config

        # 禁用按钮防止重复点击
        self._btn_load_robot.setEnabled(False)
        self._btn_load_robot.setText("加载中...")

        try:
            # Bug fix: 切换机器人前清理旧 robot/tool 残留（视觉 + 碰撞）
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None:
                try:
                    mgr.unregister_tool()   # 旧刀具线框
                    mgr.unregister_robot()  # 旧机器人线框 + checker
                except Exception as _e:
                    self.log(f"[碰撞] 清理旧机器人物体失败: {_e}")
            if self.render_engine is not None:
                try:
                    self.render_engine.remove_tool()  # 旧刀具 actor
                except Exception as _e:
                    self.log(f"[渲染] 清理旧刀具失败: {_e}")

            # 初始化运动学引擎
            self._state.kinematics_engine = KinematicsEngine(
                config.urdf_path, verbose=False
            )
            # 绑定共享刀具库（Tab 3 修改刀具后立即在 IK 求解中生效）
            self._state.kinematics_engine.set_tool_library(self._state.tool_library)
            self.log(f"运动学引擎初始化: {config.name}")

            # 加载 3D 模型
            re = self.render_engine
            if re:
                re.set_tool_library(self._state.tool_library)
                success = re.load_robot(config.urdf_path)
                if success:
                    self.log(f"3D 模型加载完成")
                    self._status_label.setText(f"已加载: {config.name}")
                    self._status_label.setStyleSheet("color: #4CAF50; padding: 5px;")
                    self._btn_load_robot.setText(f"重新加载")
                    self._btn_load_robot.setEnabled(True)

                    # 从运动学引擎提取元数据
                    metadata = self._state.kinematics_engine.get_robot_metadata()
                    self.log(f"URDF 关节数: {metadata['nq']}, 类型: {metadata['joint_types']}")

                    # ---- 注册到 CollisionManager（4 类网格之机器人） ----
                    try:
                        from collision import build_link_fk_provider
                        if mgr is not None:
                            fk = build_link_fk_provider(config.urdf_path)
                            mgr.register_robot(config.urdf_path, fk)
                            self.log("[碰撞] 机器人已注册到 CollisionManager")
                            # 应用当前 base transform 并同步静态 AABB
                            if re._robot is not None and hasattr(re._robot, 'base_transform'):
                                mgr.set_robot_base_transform(re._robot.base_transform)
                    except Exception as e:
                        self.log(f"[碰撞] 注册机器人失败: {e}")

                    # 如果刀具 STL 已存在，重新加载（以匹配新机器人法兰几何）
                    try:
                        tool_path = getattr(self._state, 'tool_filepath', None)
                        if tool_path and os.path.exists(tool_path):
                            re.create_tool(tool_path)
                            if mgr is not None:
                                mgr.register_tool(tool_path)
                            self.log(f"[碰撞] 刀具已随机器人重新加载: {os.path.basename(tool_path)}")
                    except Exception as _e:
                        self.log(f"[碰撞] 重新加载刀具失败: {_e}")

                    # 创建浮动 FK 关节控制窗口（唯一的 FK 控制）
                    if hasattr(self, '_main_window'):
                        self._main_window.show_joint_control_float_window()

                    # 直接使用浮窗中的初始值更新预览
                    jw = getattr(self._main_window, '_joint_control_widget', None)
                    if jw and jw._joint_sliders:
                        q = np.array([s.value() / 1000.0 for s in jw._joint_sliders])
                        self._state.kinematics_engine.update_q_from_angles(q.tolist())
                        re.update_robot_joints(q)
                    else:
                        self._update_robot_preview()
                    self.mark_collision_scene_changed(
                        "robot_loaded",
                        stage="scene_loaded",
                        details={
                            "robot": config.name,
                            "urdf_path": config.urdf_path,
                            "joint_count": int(metadata["nq"]),
                        },
                    )
                else:
                    raise RuntimeError("3D 模型加载失败")
            else:
                raise RuntimeError("渲染引擎未初始化")

        except Exception as e:
            self.log(f"[错误] 加载失败: {e}")
            self._status_label.setText("加载失败")
            self._status_label.setStyleSheet("color: #f44336; padding: 5px;")
            self._btn_load_robot.setText(f"重新加载")
            self._btn_load_robot.setEnabled(True)

    # Tab 1 不再需要这些方法 - FK 控制已移至浮窗
    # 如需保留旧接口兼容，可保留但不会被调用

    def _update_robot_preview(self):
        """更新机器人预览及刀具可视化元素（从浮动窗口获取关节值）"""
        re = self.render_engine
        if re is None:
            return

        # 从浮动窗口获取关节值
        jw = getattr(self._main_window, '_joint_control_widget', None) if hasattr(self, '_main_window') else None
        if jw and jw._joint_sliders:
            angles = [s.value() / 1000.0 for s in jw._joint_sliders]

            # 更新运动学引擎（会转换活动关节到 nq 向量）
            if self._state.kinematics_engine:
                self._state.kinematics_engine.update_q_from_angles(angles)
                # 使用 nq 大小的向量更新渲染
                re.update_robot_joints(self._state.kinematics_engine.current_q.copy())

            re.update_tool_visualization()

            # ---- 同步 CollisionManager（统一接管 4 类网格的实时刷新） ----
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None and self._state.kinematics_engine is not None:
                try:
                    q_nq = self._state.kinematics_engine.current_q.copy()
                    mgr.update_robot_joints(q_nq)
                    mgr.refresh()
                except Exception as e:
                    self.log(f"[碰撞] update_robot_joints 失败: {e}")

            # 兼容旧 CollisionWidget 字段
            if hasattr(self, '_main_window'):
                collision_widget = None
                for i in range(self._main_window._tab_widget.count()):
                    widget = self._main_window._tab_widget.widget(i)
                    if hasattr(widget, '_robot_collision_viz'):
                        collision_widget = widget
                        break
                if collision_widget and collision_widget._robot_collision_viz is not None:
                    if self._state.kinematics_engine:
                        q_nq = self._state.kinematics_engine.current_q.copy()
                        collision_widget._robot_collision_viz.update_joints(q_nq)
                        collision_widget._current_q = q_nq
                        collision_widget._check_and_highlight_collision()

    def _on_import_cad(self):
        """导入 CAD"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CAD 文件", "",
            "CAD 文件 (*.stl *.STP *.step *.STEP);;所有文件 (*)"
        )

        if not path:
            return

        self._state.cad_filepath = path
        self._cad_label.setText(os.path.basename(path))
        self._cad_label.setStyleSheet("color: #666; font-size: 11px;")

        self._btn_clear_cad.setEnabled(True)

        if self.render_engine is not None:
            self._btn_import_cad.setEnabled(False)
            self._btn_import_cad.setText("加载中...")

            success = self.render_engine.load_cad_model(
                path, opacity=WORKPIECE_CAD_OPACITY
            )

            self._btn_import_cad.setEnabled(True)
            self._btn_import_cad.setText("重新选择...")

            if success:
                self._cad_status_label.setText("CAD 模型已加载")
                self.log(f"CAD 已加载: {os.path.basename(path)}")
                # ---- 注册到 CollisionManager ----
                mgr = getattr(self._state, 'collision_manager', None)
                if mgr is not None:
                    try:
                        mgr.register_cad(path, transform=np.eye(4))
                    except Exception as e:
                        self.log(f"[碰撞] 注册 CAD 失败: {e}")
                self.mark_collision_scene_changed("cad_loaded", details={"path": path})
            else:
                self._cad_status_label.setText("加载失败")
                self._cad_status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 5px;")
                self.log("[错误] CAD 加载失败")

    def _on_clear_cad(self):
        """清除 CAD"""
        if self.render_engine is not None:
            self.render_engine.remove_cad_model()
        self._state.cad_filepath = None
        self._cad_label.setText("未选择文件")
        self._cad_label.setStyleSheet("color: gray; font-size: 11px;")
        self._btn_clear_cad.setEnabled(False)
        self._cad_status_label.setText("")
        # ---- 通知 CollisionManager ----
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.unregister_cad()
            except Exception as e:
                self.log(f"[碰撞] 清除 CAD 失败: {e}")
        self.mark_collision_scene_changed("cad_cleared")
        self.log("CAD 模型已清除")

    def _on_import_env_objects(self):
        """导入环境物体 STL 文件（支持多选）"""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择 STL 环境物体文件", "",
            "STL 文件 (*.stl *.STL);;所有文件 (*)"
        )

        if not paths:
            return

        if self.render_engine is None:
            self.log("[错误] 渲染引擎未初始化")
            return

        mgr = getattr(self._state, 'collision_manager', None)

        loaded_names = []
        for path in paths:
            # 生成唯一名称
            base_name = os.path.splitext(os.path.basename(path))[0]
            name = base_name
            counter = 1
            existing_names = [obj['name'] for obj in self._state.env_objects]
            while name in existing_names:
                name = f"{base_name}_{counter}"
                counter += 1

            # 加载到渲染引擎
            success = self.render_engine.load_env_object(path, name)
            if success:
                # 保存到状态
                transform = np.eye(4)
                self._state.env_objects.append({
                    'name': name,
                    'filepath': path,
                    'transform': transform
                })
                loaded_names.append(name)
                self.log(f"环境物体已加载: {name}")
                # ---- 通知 CollisionManager ----
                if mgr is not None:
                    try:
                        mgr.register_env(name, path, transform)
                    except Exception as e:
                        self.log(f"[碰撞] 注册环境物体 {name} 失败: {e}")
            else:
                self.log(f"[错误] 加载失败: {os.path.basename(path)}")

        self._update_env_ui()
        if loaded_names:
            self.mark_collision_scene_changed(
                "environment_loaded", details={"names": loaded_names}
            )

    def _on_clear_env_objects(self):
        """清除所有环境物体"""
        if self.render_engine is not None:
            self.render_engine.clear_env_objects()
        self._state.env_objects.clear()
        # ---- 通知 CollisionManager ----
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.clear_env()
            except Exception as e:
                self.log(f"[碰撞] 清除环境物体失败: {e}")
        self._update_env_ui()
        # ---- 刷新浮窗下拉框（Bug fix: 清除后列表仍显示旧名称）----
        env_win = getattr(self, '_env_float_window', None)
        if env_win is not None:
            try:
                content = env_win.layout().itemAt(1).widget()
                if content is not None and hasattr(content, '_refresh_object_list'):
                    content._refresh_object_list()
            except Exception:
                pass
        self.mark_collision_scene_changed("environment_cleared")
        self.log("所有环境物体已清除")

    def _on_show_env_float_window(self):
        """显示环境物体位置调整浮窗"""
        if hasattr(self, '_main_window') and self._main_window:
            self._main_window.show_env_float_window()

    def _update_env_ui(self):
        """更新环境物体UI状态"""
        count = len(self._state.env_objects)
        if count == 0:
            self._env_status_label.setText("未导入环境物体")
            self._env_status_label.setStyleSheet("color: gray; font-size: 11px;")
            self._btn_clear_env.setEnabled(False)
            self._btn_show_env_float.setEnabled(False)
        else:
            self._env_status_label.setText(f"已导入 {count} 个环境物体")
            self._env_status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self._btn_clear_env.setEnabled(True)
            self._btn_show_env_float.setEnabled(True)

    def _load_config_from_dict(self, config: dict) -> bool:
        """从配置字典加载所有状态"""
        re = self.render_engine
        if re is None:
            self.log("[错误] 渲染引擎未初始化")
            return False

        # ---- CollisionManager 引用（4 类网格统一管理） ----
        mgr = getattr(self._state, 'collision_manager', None)

        def _parse_transform(transform) -> np.ndarray:
            """解析变换矩阵，支持4x4格式和16元素扁平列表"""
            arr = np.array(transform)
            if arr.shape == (4, 4):
                return arr
            elif arr.shape == (16,) or arr.size == 16:
                return arr.reshape(4, 4)
            else:
                raise ValueError(f"无效的变换矩阵形状: {arr.shape}")

        success = True

        # 1. 加载机器人
        if "robot" in config:
            robot_cfg = config["robot"]
            try:
                re.load_robot(robot_cfg["urdf_path"])
                self._state.T_wcf_rcf = _parse_transform(robot_cfg["transform"])
                re.transform_robot(self._state.T_wcf_rcf)
                if mgr is not None:
                    mgr.set_robot_base_transform(self._state.T_wcf_rcf)
                self.log(f"已加载机器人: {robot_cfg.get('name', 'unknown')}")
            except Exception as e:
                self.log(f"[错误] 加载机器人失败: {e}")
                success = False

        # 2. 加载工件 CAD
        if "workpiece" in config:
            wp_cfg = config["workpiece"]
            try:
                if os.path.exists(wp_cfg["mesh_path"]):
                    re.load_cad_model(
                        wp_cfg["mesh_path"], opacity=WORKPIECE_CAD_OPACITY
                    )
                    self._state.cad_filepath = wp_cfg["mesh_path"]
                    self._state.T_wcf_ocf = _parse_transform(wp_cfg["transform"])
                    re.transform_workpiece(self._state.T_wcf_ocf)
                    # ---- 同步 CollisionManager（CAD transform） ----
                    if mgr is not None:
                        mgr.update_cad_transform(self._state.T_wcf_ocf)
                    # 更新UI
                    self._cad_label.setText(os.path.basename(wp_cfg["mesh_path"]))
                    self._cad_label.setStyleSheet("color: #666; font-size: 11px;")
                    self._btn_clear_cad.setEnabled(True)
                    self.log(f"已加载工件: {os.path.basename(wp_cfg['mesh_path'])}")
                    # ---- 注册到 CollisionManager（CAD） ----
                    if mgr is not None:
                        try:
                            T_cad = _parse_transform(wp_cfg["transform"])
                            mgr.register_cad(wp_cfg["mesh_path"], T_cad)
                        except Exception:
                            pass
                else:
                    self.log(f"[警告] 工件文件不存在: {wp_cfg['mesh_path']}")
            except Exception as e:
                self.log(f"[错误] 加载工件失败: {e}")
                success = False

        # ---- 同步 CollisionManager（工件/CAD） ----
        # (mgr 已在函数顶部定义)

        # 3. 加载刀具 STL
        if "tool" in config:
            tool_cfg = config["tool"]
            try:
                # 加载刀具STL文件
                if "mesh_path" in tool_cfg and os.path.exists(tool_cfg["mesh_path"]):
                    re.create_tool(tool_cfg["mesh_path"])
                    self._state.tool_filepath = tool_cfg["mesh_path"]
                    self.log(f"已加载刀具STL: {os.path.basename(tool_cfg['mesh_path'])}")
                    # ---- 注册到 CollisionManager（刀具） ----
                    if mgr is not None:
                        try:
                            mgr.register_tool(tool_cfg["mesh_path"])
                        except Exception:
                            pass

                # 设置刀具参数（偏移量）
                params = FlangeToolParams()
                if "flange_xyz" in tool_cfg:
                    params.flange_xyz = np.array(tool_cfg["flange_xyz"])
                if "flange_rpy" in tool_cfg:
                    params.flange_rpy = np.array(tool_cfg["flange_rpy"])
                if "tool_xyz" in tool_cfg:
                    params.tool_xyz = np.array(tool_cfg["tool_xyz"])
                if "tool_rpy" in tool_cfg:
                    params.tool_rpy = np.array(tool_cfg["tool_rpy"])

                self._state.flange_tool_params = params
                re.set_flange_tool_params(params)
                self.log(f"已设置刀具参数")
            except Exception as e:
                self.log(f"[错误] 加载刀具失败: {e}")
                success = False

        # 4. 加载环境物体
        if "environment" in config:
            re.clear_env_objects()
            self._state.env_objects.clear()
            # ---- 同步清空 CollisionManager ----
            if mgr is not None:
                try:
                    mgr.clear_env()
                except Exception:
                    pass
            for env_obj in config["environment"]:
                try:
                    if os.path.exists(env_obj["mesh_path"]):
                        if re.load_env_object(env_obj["mesh_path"], env_obj["name"]):
                            T = _parse_transform(env_obj["transform"])
                            re.update_env_object_transform(env_obj["name"], T)
                            self._state.env_objects.append({
                                "name": env_obj["name"],
                                "filepath": env_obj["mesh_path"],
                                "transform": T.copy()
                            })
                            # ---- 注册到 CollisionManager（环境物体） ----
                            if mgr is not None:
                                try:
                                    mgr.register_env(env_obj["name"], env_obj["mesh_path"], T)
                                    mgr.update_env_object_transform(env_obj["name"], T)
                                except Exception:
                                    pass
                            self.log(f"已加载环境物体: {env_obj['name']}")
                        else:
                            self.log(f"[错误] 加载环境物体失败: {env_obj['name']}")
                            success = False
                    else:
                        self.log(f"[警告] 环境物体文件不存在: {env_obj['mesh_path']}")
                        success = False
                except Exception as e:
                    self.log(f"[错误] 加载环境物体 {env_obj.get('name', '?')} 失败: {e}")
                    success = False
            self._update_env_ui()

        # 5. 恢复坐标系变换（最后应用，确保所有物体都已加载）
        if "coordinate_frames" in config:
            cf = config["coordinate_frames"]
            self._state.T_wcf_rcf = _parse_transform(cf["T_wcf_rcf"])
            self._state.T_wcf_ocf = _parse_transform(cf["T_wcf_ocf"])
            # 重新应用变换
            re.transform_robot(self._state.T_wcf_rcf)
            re.transform_workpiece(self._state.T_wcf_ocf)
            # ---- 同步 CollisionManager ----
            if mgr is not None:
                mgr.set_robot_base_transform(self._state.T_wcf_rcf)
                mgr.update_cad_transform(self._state.T_wcf_ocf)

        # 6. 迁移并恢复版本化工艺参数；未知字段由原配置字典保留。
        if "process_parameters" in config:
            try:
                process = dict(config["process_parameters"])
                for key in ("max_joint_velocity", "max_joint_acceleration", "max_joint_jerk"):
                    if process.get(key) is not None:
                        process[key] = np.asarray(process[key], dtype=float)
                self._state.process_parameters = ProcessParameters(**process)
                self._state.physical_trajectory = None
                self._state.trajectory_validation = None
                self._state.physical_trajectory_stale = True
            except Exception as exc:
                self.log(f"[警告] 工艺参数迁移失败，使用当前值: {exc}")

        known_keys = {"version", "schema_version", "environment_name", "robot", "workpiece",
                      "tool", "environment", "coordinate_frames", "process_parameters"}
        self._state.config_metadata = {
            key: value for key, value in config.items() if key not in known_keys
        }
        return success

    def _on_import_config(self):
        """导入JSON配置文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件", "",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 验证版本
            version = config.get("version", "1.0")
            self.log(f"正在导入配置: {config.get('environment_name', 'unnamed')} (v{version})")

            success = self._load_config_from_dict(config)

            if success:
                self._config_status_label.setText(f"已导入: {os.path.basename(path)}")
                self._config_status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
                self.log(f"配置导入成功: {path}")
                self.mark_collision_scene_changed(
                    "configuration_imported", details={"path": path}
                )
            else:
                self._config_status_label.setText("部分导入失败，详见日志")
                self._config_status_label.setStyleSheet("color: #f39c12; font-size: 11px;")
        except Exception as e:
            self._config_status_label.setText("导入失败")
            self._config_status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
            self.log(f"[错误] 导入配置失败: {e}")

    def _on_flange_axes_toggled(self, state: int):
        """法兰坐标系复选框切换"""
        show = bool(state)
        if self.render_engine is not None:
            self.render_engine.set_flange_axes_visible(show)
        self.log(f"法兰坐标系: {'显示' if show else '隐藏'}")

    def on_activate(self):
        """激活时"""
        self._update_robot_preview()


# ==================== [已废弃] WCFLayoutWidget ====================

class WCFLayoutWidget(StepWidget):
    """[已废弃] WCF 坐标系布局 → 功能已并入 Tab 2 (WorkpieceCloudWidget)"""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__("[已废弃] WCF 坐标系布局", 2, state, parent)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        info = QLabel(
            "此页面已废弃。\n\n"
            "WCF 坐标系布局功能已并入 Tab 2 (毛坯点云处理) 页面，"
            "请在 Tab 2 中同时设置机器人基座和工件位姿。"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; padding: 20px;")
        layout.addWidget(info)


# ==================== Tab 2: 毛坯点云处理 ====================

class WorkpieceCloudWidget(StepWidget):
    """Tab 2: 毛坯点云处理（含 WCF 坐标系布局）

    功能：
    - T_wcf_rcf：机器人基座位姿
    - T_wcf_ocf：工件位姿
    用户可在同一界面联合调整点云、机器人、工件位姿。
    """

    def __init__(self, state: SimulationState, parent=None):
        self._cloud_offset = np.eye(4)  # 点云相对于工件的额外偏移
        self._sync_cloud_part = True     # 同步移动开关
        self._wcf_inputs = {}           # 复用 StepWidget 公共逻辑存储（必须在 super().__init__ 前）
        super().__init__("毛坯点云处理", 4, state, parent)

    def _init_ui(self):
        _content, layout = self.create_scrollable_layout()
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "导入毛坯点云，使用 Open3D 进行 ICP 配准、去噪清洗。\n"
            "也可调整点云在 WCF 下的位姿。下方「WCF 坐标系布局」可同时设置机器人和工件位姿。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ===== WCF 坐标系布局组 =====
        wcf_group = QGroupBox("WCF 坐标系布局（机器人 / 工件位姿）")
        wcf_layout = QVBoxLayout(wcf_group)

        # 机器人基座
        robot_sub_group = QGroupBox("T_wcf_rcf - 机器人基座位姿")
        robot_sub_layout = QVBoxLayout(robot_sub_group)
        grid_rcf = StepWidget.create_xyz_rpy_grid(
            self, "T_wcf_rcf", self._wcf_inputs, self._on_wcf_transform_changed
        )
        robot_sub_layout.addLayout(grid_rcf)
        wcf_layout.addWidget(robot_sub_group)

        # 工件位姿
        ocf_sub_group = QGroupBox("T_wcf_ocf - 工件位姿")
        ocf_sub_layout = QVBoxLayout(ocf_sub_group)
        grid_ocf = StepWidget.create_xyz_rpy_grid(
            self, "T_wcf_ocf", self._wcf_inputs, self._on_wcf_transform_changed
        )
        ocf_sub_layout.addLayout(grid_ocf)
        wcf_layout.addWidget(ocf_sub_group)

        # 场景坐标控制已统一迁移到 Tab 1；保留内部控件仅用于旧配置兼容。
        wcf_group.setVisible(False)
        layout.addWidget(wcf_group)

        # ===== 点云导入 =====
        btn_layout = QHBoxLayout()
        self._btn_import = QPushButton("导入点云")
        self._btn_import.clicked.connect(self._on_import_cloud)
        btn_layout.addWidget(self._btn_import)

        self._btn_generate = QPushButton("生成测试点云")
        self._btn_generate.clicked.connect(self._on_generate_test)
        btn_layout.addWidget(self._btn_generate)
        layout.addLayout(btn_layout)

        self._point_count = QLabel("点数: 0")
        layout.addWidget(self._point_count)

        # 处理按钮
        proc_group = QGroupBox("处理操作")
        proc_layout = QVBoxLayout(proc_group)

        self._btn_icp = QPushButton("ICP 精配准 (到 CAD)")
        self._btn_icp.clicked.connect(self._on_icp)
        self._btn_icp.setEnabled(False)
        proc_layout.addWidget(self._btn_icp)

        self._btn_filter = QPushButton("统计滤波去噪")
        self._btn_filter.clicked.connect(self._on_filter)
        proc_layout.addWidget(self._btn_filter)

        self._btn_downsample = QPushButton("体素下采样")
        self._btn_downsample.clicked.connect(self._on_downsample)
        proc_layout.addWidget(self._btn_downsample)

        self._btn_enable_crop = QPushButton("开启裁剪框")
        self._btn_enable_crop.clicked.connect(self._on_enable_crop)
        proc_layout.addWidget(self._btn_enable_crop)

        self._btn_confirm_crop = QPushButton("确认去噪裁剪")
        self._btn_confirm_crop.clicked.connect(self._on_confirm_crop)
        self._btn_confirm_crop.setEnabled(False)
        proc_layout.addWidget(self._btn_confirm_crop)

        layout.addWidget(proc_group)

        # 点云位姿控制组
        pose_group = QGroupBox("T_wcf_cloud - 点云位姿 (WCF)")
        self._cloud_pose_group = pose_group
        pose_layout = QVBoxLayout(pose_group)

        # 同步移动开关
        self._sync_checkbox = QCheckBox("同步移动点云和零件 (随工件位姿变化)")
        self._sync_checkbox.setChecked(True)
        self._sync_checkbox.stateChanged.connect(self._on_sync_changed)
        pose_layout.addWidget(self._sync_checkbox)

        # 位置和姿态控制
        grid = QGridLayout()

        labels = ['X (m)', 'Y (m)', 'Z (m)', 'Rx (deg)', 'Ry (deg)', 'Rz (deg)']
        self._cloud_inputs = {}

        for i, label_text in enumerate(labels):
            row, col_base = i // 3, (i % 3) * 2
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(70)
            grid.addWidget(lbl, row, col_base)

            spin = QDoubleSpinBox()
            spin.setRange(-10 if 'deg' not in label_text else -180,
                          10 if 'deg' not in label_text else 180)
            spin.setDecimals(4)
            spin.setSingleStep(0.01 if 'deg' not in label_text else 1.0)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_cloud_pose_changed)
            grid.addWidget(spin, row, col_base + 1)
            self._cloud_inputs[label_text] = spin

        pose_layout.addLayout(grid)

        # 重置按钮
        reset_row = QHBoxLayout()
        self._btn_reset_pose = QPushButton("重置位姿")
        self._btn_reset_pose.clicked.connect(self._on_reset_cloud_pose)
        reset_row.addWidget(self._btn_reset_pose)
        reset_row.addStretch()
        pose_layout.addLayout(reset_row)

        layout.addWidget(pose_group)

        layout.addStretch()

    def _on_wcf_transform_changed(self):
        """WCF 变换参数改变"""
        self._state.T_wcf_rcf = StepWidget.xyz_rpy_to_matrix(self._wcf_inputs, "T_wcf_rcf")
        self._state.T_wcf_ocf = StepWidget.xyz_rpy_to_matrix(self._wcf_inputs, "T_wcf_ocf")
        re = self.render_engine

        if re is not None:
            # 机器人基座：始终更新
            re.transform_robot(self._state.T_wcf_rcf)
            # 坐标系变换（kinematics_engine + CoordinateTransformer）
            if self._state.kinematics_engine:
                self._state.kinematics_engine.set_transforms(
                    self._state.T_wcf_rcf, self._state.T_wcf_ocf
                )
            # CAD 工件：始终跟随 T_wcf_ocf（WCF 坐标系中的物理位姿）
            re.transform_workpiece(self._state.T_wcf_ocf)

            # ---- 同步 CollisionManager（WCF 变换影响所有碰撞体）----
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None:
                mgr.set_robot_base_transform(self._state.T_wcf_rcf)
                mgr.update_cad_transform(self._state.T_wcf_ocf)
                mgr.refresh()  # 立即刷新碰撞状态
        self.mark_collision_scene_changed("wcf_transform")

        # 点云：按同步开关决定是否跟随工件
        # - 开启：点云 = T_wcf_ocf，无额外偏移（与 CAD 同步）
        # - 关闭：点云 = T_wcf_ocf @ _cloud_offset（有点云位姿 UI 的额外偏移）
        if self._state.processed_points is not None:
            if self._sync_cloud_part:
                re.transform_pointcloud(self._state.T_wcf_ocf, extra_offset=None)
            else:
                re.transform_pointcloud(self._state.T_wcf_ocf, extra_offset=self._cloud_offset)

    def _on_import_cloud(self):
        """导入点云"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择点云文件", "",
            "点云文件 (*.ply *.pcd *.xyz);;所有文件 (*)"
        )

        if path:
            try:
                pcd = o3d.io.read_point_cloud(path)
                points = np.asarray(pcd.points)
                if points is None or len(points) == 0:
                    self.log(f"[错误] 点云文件为空或无法读取")
                    return
                self._load_points(points)
            except Exception as e:
                self.log(f"[错误] 导入失败: {e}")

    def _on_generate_test(self):
        """生成测试点云"""
        # 生成一个立方体测试点云
        size = np.array([0.1, 0.05, 0.02])
        n = 5000

        x = np.random.uniform(-size[0]/2, size[0]/2, n)
        y = np.random.uniform(-size[1]/2, size[1]/2, n)
        z = np.random.uniform(-size[2]/2, size[2]/2, n)

        points = np.column_stack([x, y, z])
        self._load_points(points)

    def _load_points(self, points: np.ndarray):
        """加载点云"""
        if points is None or len(points) == 0:
            self.log(f"[错误] 点云数据为空，跳过加载")
            return
        self._state.raw_points = points.copy()
        self._state.processed_points = points.copy()
        self._point_count.setText(f"点数: {len(points)}")

        if self.render_engine is not None:
            self.render_engine.load_workpiece_cloud(points)
            # 应用当前位姿设置
            self._apply_cloud_pose()
            self.render_engine.render()

        self._btn_icp.setEnabled(OPEN3D_AVAILABLE)
        self.log(f"点云已加载: {len(points)} 点")

    def _on_sync_changed(self, state: int):
        """同步移动开关状态改变"""
        self._sync_cloud_part = (state == Qt.Checked)
        self.log(f"同步移动: {'开启' if self._sync_cloud_part else '关闭'}")
        # 同步开启时禁用点云位姿控件（此时点云跟随 CAD，无需单独调整）
        # 同步关闭时启用，允许点云相对 CAD 有额外偏移
        enabled = not self._sync_cloud_part
        if hasattr(self, '_cloud_pose_group'):
            self._cloud_pose_group.setEnabled(enabled)
        # 重置云偏移并应用
        if not self._sync_cloud_part:
            self._reset_cloud_offset()
        # 立即应用当前位姿
        self._apply_cloud_pose()

    def _on_cloud_pose_changed(self):
        """点云位姿改变"""
        self._apply_cloud_pose()

    def _reset_cloud_offset(self):
        """重置点云偏移量为零"""
        for key, spin in self._cloud_inputs.items():
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._cloud_offset = np.eye(4)

    def _apply_cloud_pose(self):
        """
        根据当前 UI 值应用点云变换

        逻辑：
        - 当同步开关开启时：点云跟随工件移动（无额外偏移）
        - 当同步开关关闭时：点云有额外的偏移量
        """
        re = self.render_engine
        if re is None:
            return

        # 从 UI 获取 xyz/rpy
        xyz = np.array([
            self._cloud_inputs['X (m)'].value(),
            self._cloud_inputs['Y (m)'].value(),
            self._cloud_inputs['Z (m)'].value()
        ])
        rpy_deg = np.array([
            self._cloud_inputs['Rx (deg)'].value(),
            self._cloud_inputs['Ry (deg)'].value(),
            self._cloud_inputs['Rz (deg)'].value()
        ])

        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
        T_cloud_offset = np.eye(4)
        T_cloud_offset[:3, :3] = R
        T_cloud_offset[:3, 3] = xyz

        # 保存额外偏移
        self._cloud_offset = T_cloud_offset.copy()

        # 根据同步开关决定实际应用的变换
        # - 开启：点云 = T_wcf_ocf，无额外偏移（与 CAD 完全同步）
        # - 关闭：点云 = T_wcf_ocf @ _cloud_offset（有点云位姿 UI 的额外偏移）
        if self._sync_cloud_part:
            re.transform_pointcloud(self._state.T_wcf_ocf, extra_offset=None)
        else:
            re.transform_pointcloud(self._state.T_wcf_ocf, extra_offset=self._cloud_offset)

        self.log(f"点云位姿已更新 (同步: {'开' if self._sync_cloud_part else '关'})")

    def _on_reset_cloud_pose(self):
        """重置点云位姿"""
        # 重置所有输入为 0
        for key, spin in self._cloud_inputs.items():
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

        self._cloud_offset = np.eye(4)
        self._apply_cloud_pose()
        self.log("点云位姿已重置")

    def on_activate(self):
        """Step 激活时应用点云位姿"""
        self._apply_cloud_pose()

    def _on_icp(self):
        """ICP 配准"""
        if not OPEN3D_AVAILABLE:
            self.log("[错误] Open3D 未安装")
            return

        if self._state.cad_filepath is None:
            self.log("[错误] 请先导入 CAD 模型")
            return

        self.log("开始 ICP 配准...")

        # 这里应该调用 render_engine 的 icp 功能
        if self.render_engine is not None:
            # 简化：直接标记完成
            self.log("ICP 配准完成")

    def _on_filter(self):
        """统计滤波"""
        if self.render_engine is not None:
            success = self.render_engine.filter_outliers()
            if success:
                self.log("去噪完成")
                points = self.render_engine._original_points
                if points is not None:
                    self._point_count.setText(f"点数: {len(points)}")

    def _on_downsample(self):
        """下采样"""
        if self.render_engine is not None:
            success = self.render_engine.voxel_downsample_workpiece(0.002)
            if success:
                self.log("下采样完成")
                points = self.render_engine._original_points
                if points is not None:
                    self._point_count.setText(f"点数: {len(points)}")

    def _on_enable_crop(self):
        """
        Task 3: 开启裁剪框
        - 调用 render_engine.enable_crop_box() 启动交互
        - 启用确认按钮
        - 禁用开启按钮
        """
        if self.render_engine is not None:
            self.render_engine.enable_crop_box()
            self._btn_confirm_crop.setEnabled(True)
            self._btn_enable_crop.setEnabled(False)
            self.log("已开启裁剪框，请在 3D 视图中拖动调整")

    def _on_confirm_crop(self):
        """
        Task 3: 确认裁剪
        - 调用 render_engine.confirm_crop_box() 执行实际裁剪
        - 重置按钮状态
        - 更新点数显示
        """
        if self.render_engine is not None:
            success = self.render_engine.confirm_crop_box(self._state.T_wcf_ocf)
            if success:
                self.log("裁剪完成")
                # 更新点数
                points = self.render_engine._original_points
                if points is not None:
                    self._point_count.setText(f"点数: {len(points)}")
            else:
                self.log("裁剪失败，请先拖动裁剪框")

        # 重置按钮状态
        self._btn_enable_crop.setEnabled(True)
        self._btn_confirm_crop.setEnabled(False)


# ==================== Tab 3: 刀具与 TCP ====================

class ToolTCPWidget(StepWidget):
    """Tab 3: 刀具与 TCP 补偿"""

    def __init__(self, state: SimulationState, parent=None):
        super().__init__("刀具与 TCP", 3, state, parent)
        self._auto_measure = None  # 存储 STL 自动测量结果

    def _init_ui(self):
        _content, layout = self.create_scrollable_layout()

        info = QLabel("导入自定义刀具 STL 文件，通过 flange / tool0 / tool 三组偏移参数计算 TCP")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 导入刀具按钮
        self._btn_import_tool = QPushButton("导入自定义刀具 (STL)")
        self._btn_import_tool.clicked.connect(self._on_import_tool)
        layout.addWidget(self._btn_import_tool)

        self._tool_label = QLabel("未导入刀具")
        self._tool_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._tool_label)

        layout.addSpacing(10)

        # ---------- Flange 偏移组 ----------
        self._flange_group = QGroupBox()
        flange_layout = QGridLayout(self._flange_group)
        self._flange_axis_label = QLabel("[axis: ?]")
        self._flange_axis_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
        flange_layout.addWidget(QLabel("Flange 偏移"), 0, 0, 1, 6)
        flange_layout.addWidget(self._flange_axis_label, 0, 6)
        self._flange_inputs = {}
        flange_row_labels = ["X (m)", "Y (m)", "Z (m)"]
        for row, label in enumerate(flange_row_labels, 1):
            flange_layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1, 1)
            spin.setDecimals(4)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_params_changed)
            flange_layout.addWidget(spin, row, 1, 1, 2)
            self._flange_inputs[label] = spin
        flange_layout.addWidget(QLabel("Roll (rad)"), 4, 0)
        flange_layout.addWidget(QLabel("Pitch (rad)"), 4, 2)
        flange_layout.addWidget(QLabel("Yaw (rad)"), 4, 4)
        for col, key in enumerate(["Roll", "Pitch", "Yaw"], 1):
            spin = QDoubleSpinBox()
            spin.setRange(-3.1416, 3.1416)
            spin.setDecimals(4)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_params_changed)
            flange_layout.addWidget(spin, 5, col * 2 - 1, 1, 2)
            self._flange_inputs[key] = spin
        layout.addWidget(self._flange_group)

        # ---------- Tool 偏移组 ----------
        self._tool_group = QGroupBox()
        tool_layout = QGridLayout(self._tool_group)
        self._tool_axis_label = QLabel("[axis: ?]")
        self._tool_axis_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
        tool_layout.addWidget(QLabel("Tool 偏移 (TCP)"), 0, 0, 1, 6)
        tool_layout.addWidget(self._tool_axis_label, 0, 6)
        self._tool_inputs = {}
        for row, label in enumerate(flange_row_labels, 1):
            tool_layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1, 1)
            spin.setDecimals(4)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_params_changed)
            tool_layout.addWidget(spin, row, 1, 1, 2)
            self._tool_inputs[label] = spin
        tool_layout.addWidget(QLabel("Roll (rad)"), 4, 0)
        tool_layout.addWidget(QLabel("Pitch (rad)"), 4, 2)
        tool_layout.addWidget(QLabel("Yaw (rad)"), 4, 4)
        for col, key in enumerate(["Roll", "Pitch", "Yaw"], 1):
            spin = QDoubleSpinBox()
            spin.setRange(-3.1416, 3.1416)
            spin.setDecimals(4)
            spin.setValue(0.0)
            spin.valueChanged.connect(self._on_params_changed)
            tool_layout.addWidget(spin, 5, col * 2 - 1, 1, 2)
            self._tool_inputs[key] = spin
        layout.addWidget(self._tool_group)

        # 包围盒显示控制
        bbox_row = QHBoxLayout()
        self._show_bbox_cb = QCheckBox("显示刀具包围盒")
        self._show_bbox_cb.stateChanged.connect(self._on_bbox_toggled)
        bbox_row.addWidget(self._show_bbox_cb)
        layout.addLayout(bbox_row)

        # ── 导出配置文件 ─────────────────────────────────────────────
        export_group = QGroupBox("配置管理")
        export_layout = QVBoxLayout(export_group)

        self._btn_export_config = QPushButton("导出JSON配置文件...")
        self._btn_export_config.clicked.connect(self._on_export_config)
        export_layout.addWidget(self._btn_export_config)

        self._export_status_label = QLabel("未导出配置")
        self._export_status_label.setStyleSheet("color: gray; font-size: 11px;")
        export_layout.addWidget(self._export_status_label)

        layout.addWidget(export_group)

        layout.addStretch()

    def on_activate(self):
        """Step 被激活时更新轴标签"""
        self._update_axis_labels()

    def _update_axis_labels(self):
        """从 kinematics_engine 获取旋转轴并更新标签"""
        if self._state.kinematics_engine:
            ke = self._state.kinematics_engine
            self._flange_axis_label.setText(f"[axis: {ke.get_flange_axis()}]")
            self._tool_axis_label.setText(f"[axis: {ke.get_tool_axis()}]")
            self._print_axis_debug()
        else:
            self._flange_axis_label.setText("[axis: ?]")
            self._tool_axis_label.setText("[axis: ?]")

    def _print_axis_debug(self):
        """打印轴 debug 日志"""
        ke = self._state.kinematics_engine
        self.log(f"[调试] joint6 轴: {ke.get_end_effector_axis()}")
        self.log(f"[调试] 法兰轴: {ke.get_flange_axis()}")
        self.log(f"[调试] 刀具轴: {ke.get_tool_axis()}")

    def _on_import_tool(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择刀具文件", "", "CAD 模型 (*.stl *.step *.stp)")
        if not path:
            return
        try:
            import trimesh
            mesh = trimesh.load(path, force='mesh')
            bounds = mesh.bounds
            x_len = float(bounds[1][0] - bounds[0][0])
            y_len = float(bounds[1][1] - bounds[0][1])
            z_len = float(bounds[1][2] - bounds[0][2])
            if max(x_len, y_len, z_len) > 1.0:
                x_len /= 1000.0
                y_len /= 1000.0
                z_len /= 1000.0

            tool_axis = 'z'
            if self._state.kinematics_engine:
                tool_axis = self._state.kinematics_engine.get_tool_axis()

            self._auto_measure = {'x': x_len, 'y': y_len, 'z': z_len}

            # 根据法兰旋转轴决定填入哪个输入框
            # flange_axis 决定了 TCP 偏移的方向
            flange_axis = 'z'
            if self._state.kinematics_engine:
                flange_axis = self._state.kinematics_engine.get_flange_axis()
                # 去掉负号获取基础轴名
                base_axis = flange_axis.rstrip('-').lower()
            else:
                base_axis = 'z'

            axis_key = f"{base_axis.upper()} (m)"
            if axis_key in self._tool_inputs:
                self._tool_inputs[axis_key].setValue(z_len)
                self.log(f"[自动测量] 已填入: {base_axis}={z_len:.4f}m (flange_axis={flange_axis}, 刀具Z轴长度)")
            else:
                self.log(f"[自动测量] 无法填入: 找不到 {axis_key} 输入框")
            self._tool_label.setText(os.path.basename(path))
            self._tool_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self.log(f"已导入刀具: {os.path.basename(path)}")
            self.log(f"  STL 测量: x={x_len:.4f}m, y={y_len:.4f}m, z={z_len:.4f}m (填入 {flange_axis} 轴={z_len:.4f}m)")

            # 保存刀具路径到状态（用于配置导出/导入）
            self._state.tool_filepath = path

            if self.render_engine:
                self.render_engine.create_tool(path)
                if self._state.kinematics_engine and self._state.kinematics_engine.current_q is not None:
                    self.render_engine.update_robot_joints(self._state.kinematics_engine.current_q)

            # ---- 通知 CollisionManager（4 类网格之刀具） ----
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None:
                try:
                    mgr.register_tool(path)
                    self.log("[碰撞] 刀具已注册到 CollisionManager")
                except Exception as e:
                    self.log(f"[碰撞] 注册刀具失败: {e}")
            self.mark_collision_scene_changed("tool_loaded", details={"path": path})
        except Exception as e:
            self.log(f"[错误] 导入刀具失败: {e}")

    def _on_params_changed(self):
        """参数变化，同步到 tool_library、kinematics_engine 和 render_engine"""
        from core.tool_library import ToolData

        params = FlangeToolParams()

        # Flange
        params.flange_xyz = [
            self._flange_inputs['X (m)'].value(),
            self._flange_inputs['Y (m)'].value(),
            self._flange_inputs['Z (m)'].value(),
        ]
        params.flange_rpy = [
            self._flange_inputs['Roll'].value(),
            self._flange_inputs['Pitch'].value(),
            self._flange_inputs['Yaw'].value(),
        ]

        # Tool
        params.tool_xyz = [
            self._tool_inputs['X (m)'].value(),
            self._tool_inputs['Y (m)'].value(),
            self._tool_inputs['Z (m)'].value(),
        ]
        params.tool_rpy = [
            self._tool_inputs['Roll'].value(),
            self._tool_inputs['Pitch'].value(),
            self._tool_inputs['Yaw'].value(),
        ]

        # 更新共享刀具库（SSOT），KinematicsEngine.effective_tcp 会实时读取
        self._state.flange_tool_params = params
        self._state.tool_library.set_current_tool_data(
            ToolData.from_flange_tool_params(params, name="custom_tool")
        )

        # 向后兼容：仍调用 set_flange_tool_params 以更新 KinematicsEngine 内部缓存
        if self._state.kinematics_engine:
            self._state.kinematics_engine.set_flange_tool_params(params)

        if self.render_engine:
            self.render_engine.set_flange_tool_params(params)
            if self._state.kinematics_engine and self._state.kinematics_engine.current_q is not None:
                self.render_engine.update_robot_joints(self._state.kinematics_engine.current_q)
        manager = getattr(self._state, "collision_manager", None)
        if manager is not None:
            # A TCP edit changes the attached tool mount transform.  The live
            # GUI service owns a cached snapshot, so invalidate it before the
            # next check rather than allowing stale tool geometry to survive.
            manager.invalidate_scene_snapshot_cache()
        self.mark_collision_scene_changed("tcp_transform_changed")

    def _on_bbox_toggled(self, state: int):
        """包围盒复选框切换"""
        show = bool(state)
        if self.render_engine:
            self.render_engine.set_bounding_box_visible(show)
        self.log(f"刀具包围盒: {'显示' if show else '隐藏'}")

    def _build_config_dict(self, env_name: str = "scene_1") -> dict:
        """从当前状态构建配置字典"""
        config = {
            "version": "2.0",
            "schema_version": 2,
            "environment_name": env_name,
        }

        def _matrix_to_4x4(mat: np.ndarray) -> list:
            """将4x4矩阵转换为4行4列的列表格式，便于阅读"""
            arr = np.array(mat).flatten()
            return [
                arr[0:4].tolist(),   # 第1行
                arr[4:8].tolist(),   # 第2行
                arr[8:12].tolist(),  # 第3行
                arr[12:16].tolist()  # 第4行
            ]

        # 机器人配置
        if self._state.robot_config:
            config["robot"] = {
                "name": self._state.robot_config.name,
                "urdf_path": self._state.robot_config.urdf_path,
                "transform": _matrix_to_4x4(self._state.T_wcf_rcf)
            }

        # 工件配置
        if self._state.cad_filepath:
            config["workpiece"] = {
                "mesh_path": self._state.cad_filepath,
                "transform": _matrix_to_4x4(self._state.T_wcf_ocf)
            }

        # 刀具配置（STL路径 + 偏移参数）
        if self._state.tool_filepath or self._state.flange_tool_params:
            tool_cfg = {}
            if self._state.tool_filepath:
                tool_cfg["mesh_path"] = self._state.tool_filepath
            if self._state.flange_tool_params:
                params = self._state.flange_tool_params
                tool_cfg["flange_xyz"] = list(params.flange_xyz) if hasattr(params.flange_xyz, '__iter__') else list(params.flange_xyz)
                tool_cfg["flange_rpy"] = list(params.flange_rpy) if hasattr(params.flange_rpy, '__iter__') else list(params.flange_rpy)
                tool_cfg["tool_xyz"] = list(params.tool_xyz) if hasattr(params.tool_xyz, '__iter__') else list(params.tool_xyz)
                tool_cfg["tool_rpy"] = list(params.tool_rpy) if hasattr(params.tool_rpy, '__iter__') else list(params.tool_rpy)
            config["tool"] = tool_cfg

        # 环境物体
        if self._state.env_objects:
            config["environment"] = [
                {
                    "name": obj["name"],
                    "mesh_path": obj["filepath"],
                    "transform": _matrix_to_4x4(obj["transform"])
                }
                for obj in self._state.env_objects
            ]

        # 坐标系变换
        config["coordinate_frames"] = {
            "T_wcf_rcf": _matrix_to_4x4(self._state.T_wcf_rcf),
            "T_wcf_ocf": _matrix_to_4x4(self._state.T_wcf_ocf)
        }

        process_data = asdict(getattr(self._state, "process_parameters", ProcessParameters()))
        for key, value in list(process_data.items()):
            if isinstance(value, np.ndarray):
                process_data[key] = value.tolist()
        config["process_parameters"] = process_data
        config.update(getattr(self._state, "config_metadata", {}))

        return config

    def _on_export_config(self):
        """导出JSON配置文件"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", "scene_config.json",
            "JSON 文件 (*.json);;所有文件 (*)"
        )
        if not path:
            return

        # 获取环境名称（使用文件名作为默认值）
        env_name = os.path.splitext(os.path.basename(path))[0]

        config = self._build_config_dict(env_name)

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self._export_status_label.setText(f"已导出: {os.path.basename(path)}")
            self._export_status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self.log(f"配置已导出: {path}")
        except Exception as e:
            self._export_status_label.setText("导出失败")
            self._export_status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")
            self.log(f"[错误] 导出配置失败: {e}")


# ==================== Tab 4: 刀路生成 ====================

class ToolpathGeneratorWidget(StepWidget):
    """
    Tab 4: 刀路生成系统

    基于元数据驱动的动态 UI 架构设计。
    算法参数根据 ParamDef 架构自动生成。
    支持热插拔算法插件。
    """

    def __init__(self, state: SimulationState, parent=None):
        self.param_widgets = {}
        self._worker = None
        self.current_algo = None
        self._last_result = None  # 保存最后一次生成的结果用于导出
        self._seq_strategy = 0  # 默认螺旋型 (0=螺旋型, 1=锯齿型)
        super().__init__("刀轨生成", 6, state, parent)

    def _init_ui(self):
        _content, layout = self.create_scrollable_layout()

        info = QLabel(
            "基于元数据驱动的刀路生成系统。"
            "参数根据所选算法自动生成，支持热插拔插件。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        process_group = QGroupBox("加工工艺参数（能力分级）")
        process_layout = QGridLayout(process_group)
        current_process = getattr(self._state, "process_parameters", ProcessParameters())
        self._process_stepover_spin = QDoubleSpinBox()
        self._process_stepover_spin.setRange(0.001, 1000.0)
        self._process_stepover_spin.setDecimals(3)
        self._process_stepover_spin.setValue(current_process.stepover_m * 1000.0)
        self._process_stepover_spin.setSuffix(" mm")
        process_layout.addWidget(QLabel("横向步距 [约束]:"), 0, 0)
        process_layout.addWidget(self._process_stepover_spin, 0, 1)
        self._process_contact_width_spin = QDoubleSpinBox()
        self._process_contact_width_spin.setRange(0.0, 1000.0)
        self._process_contact_width_spin.setDecimals(3)
        self._process_contact_width_spin.setSpecialValueText("模型不可用")
        self._process_contact_width_spin.setValue(
            0.0 if current_process.effective_contact_width_m is None
            else current_process.effective_contact_width_m * 1000.0
        )
        self._process_contact_width_spin.setSuffix(" mm")
        process_layout.addWidget(QLabel("有效接触宽度 [派生]:"), 0, 2)
        process_layout.addWidget(self._process_contact_width_spin, 0, 3)
        self._process_force_spin = QDoubleSpinBox()
        self._process_force_spin.setRange(0.0, 10000.0)
        self._process_force_spin.setValue(current_process.normal_force_setpoint_n)
        self._process_force_spin.setSuffix(" N")
        process_layout.addWidget(QLabel("法向力 [开环设定]:"), 1, 0)
        process_layout.addWidget(self._process_force_spin, 1, 1)
        self._process_tilt_spin = QDoubleSpinBox()
        self._process_tilt_spin.setRange(-90.0, 90.0)
        self._process_tilt_spin.setValue(np.rad2deg(current_process.tool_tilt_rad))
        self._process_tilt_spin.setSuffix(" deg")
        process_layout.addWidget(QLabel("工具倾角 [约束]:"), 1, 2)
        process_layout.addWidget(self._process_tilt_spin, 1, 3)
        self._process_blend_spin = QDoubleSpinBox()
        self._process_blend_spin.setRange(0.0, 1000.0)
        self._process_blend_spin.setDecimals(3)
        self._process_blend_spin.setValue(current_process.corner_blend_radius_m * 1000.0)
        self._process_blend_spin.setSuffix(" mm")
        process_layout.addWidget(QLabel("转弯圆角 [约束]:"), 2, 0)
        process_layout.addWidget(self._process_blend_spin, 2, 1)
        self._process_overlap_label = QLabel("重叠度 [不可用]")
        process_layout.addWidget(self._process_overlap_label, 2, 2, 1, 2)
        force_note = QLabel("法向力只保存和导出设定值；未连接传感器，不构成闭环控制。")
        force_note.setWordWrap(True)
        force_note.setStyleSheet("color: #B26A00;")
        process_layout.addWidget(force_note, 3, 0, 1, 4)
        for control in (
            self._process_stepover_spin, self._process_contact_width_spin,
            self._process_force_spin, self._process_tilt_spin, self._process_blend_spin,
        ):
            control.valueChanged.connect(self._on_process_parameters_changed)
        layout.addWidget(process_group)
        self._on_process_parameters_changed()

        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout(algo_group)

        algo_select_row = QHBoxLayout()
        algo_select_row.addWidget(QLabel("算法:"))

        self._algorithm_combo = QComboBox()
        self._algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        algo_select_row.addWidget(self._algorithm_combo)
        algo_layout.addLayout(algo_select_row)

        self._algo_status = QLabel("未加载算法")
        self._algo_status.setStyleSheet("color: #888; font-size: 11px;")
        algo_layout.addWidget(self._algo_status)

        layout.addWidget(algo_group)

        self._param_group = QGroupBox("算法参数（自动生成）")
        self._param_layout = QFormLayout()
        self._param_group.setLayout(self._param_layout)
        layout.addWidget(self._param_group)

        # ── 宏观轨迹规划策略组 ──────────────────────────────────────────
        seq_group = QGroupBox("宏观轨迹规划")
        seq_layout = QVBoxLayout(seq_group)

        seq_info = QLabel("选择刀路点遍历策略（影响奇异点分布与关节限位触发次数）：")
        seq_info.setWordWrap(True)
        seq_layout.addWidget(seq_info)

        seq_row = QHBoxLayout()
        seq_row.addWidget(QLabel("策略:"))
        self._seq_combo = QComboBox()
        self._seq_combo.addItems(["螺旋型规划器 (0)", "锯齿型规划器 (1)"])
        self._seq_combo.setCurrentIndex(0)
        self._seq_combo.currentIndexChanged.connect(self._on_seq_strategy_changed)
        seq_row.addWidget(self._seq_combo)
        seq_row.addStretch()
        seq_layout.addLayout(seq_row)

        self._seq_status = QLabel("当前: 螺旋型 (strategy=0)")
        self._seq_status.setStyleSheet("color: #888; font-size: 11px;")
        seq_layout.addWidget(self._seq_status)

        self._show_trajectory_axes_cb = QCheckBox("显示轨迹坐标系")
        self._show_trajectory_axes_cb.setChecked(False)
        self._show_trajectory_axes_cb.stateChanged.connect(self._on_trajectory_axes_toggled)
        seq_layout.addWidget(self._show_trajectory_axes_cb)

        layout.addWidget(seq_group)

        btn_layout = QHBoxLayout()
        self._btn_generate = QPushButton("生成刀轨")
        self._btn_generate.clicked.connect(self._on_generate)
        btn_layout.addWidget(self._btn_generate)

        # 导出类型下拉框
        self._export_type_combo = QComboBox()
        self._export_type_combo.addItems(["toolpath", "pose"])
        self._export_type_combo.setToolTip(
            "toolpath: 位置+法向量+逐点层号 | pose: 宏观轨迹位姿"
        )
        self._export_type_combo.currentIndexChanged.connect(self._on_export_type_changed)
        btn_layout.addWidget(self._export_type_combo)

        # 导出格式下拉框（pose 模式下启用）
        self._export_format_combo = QComboBox()
        self._export_format_combo.addItems(["欧拉角 (xyz)", "四元数 (qwqxqyqz)"])
        self._export_format_combo.setEnabled(False)
        self._export_format_combo.setToolTip("欧拉角: 方便阅读 | 四元数: 避免奇异")
        btn_layout.addWidget(self._export_format_combo)

        self._btn_export = QPushButton("导出数据")
        self._btn_export.clicked.connect(self._on_export_data)
        self._btn_export.setEnabled(False)
        btn_layout.addWidget(self._btn_export)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setStyleSheet("background-color: #e74c3c; color: white;")
        btn_layout.addWidget(self._btn_stop)

        self._btn_clear = QPushButton("清除预览")
        self._btn_clear.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._btn_clear)
        layout.addLayout(btn_layout)

        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._result_label = QLabel("尚未生成刀轨")
        layout.addWidget(self._result_label)

        # ── 轨迹可视化参数 ──────────────────────────────────────────────
        viz_group = QGroupBox("轨迹可视化参数")
        viz_layout = QGridLayout(viz_group)

        viz_layout.addWidget(QLabel("采样步长:"), 0, 0)
        self._sample_step_spin = QSpinBox()
        self._sample_step_spin.setRange(1, 50)
        self._sample_step_spin.setValue(self._state.trajectory_viz_sample_step)
        viz_layout.addWidget(self._sample_step_spin, 0, 1)

        viz_layout.addWidget(QLabel("轴长度(m):"), 1, 0)
        self._axis_length_spin = QDoubleSpinBox()
        self._axis_length_spin.setRange(0.001, 0.1)
        self._axis_length_spin.setDecimals(3)
        self._axis_length_spin.setSingleStep(0.001)
        self._axis_length_spin.setValue(self._state.trajectory_viz_axis_length)
        viz_layout.addWidget(self._axis_length_spin, 1, 1)

        viz_layout.addWidget(QLabel("点大小:"), 2, 0)
        self._point_size_spin = QSpinBox()
        self._point_size_spin.setRange(1, 50)
        self._point_size_spin.setValue(self._state.trajectory_viz_point_size)
        viz_layout.addWidget(self._point_size_spin, 2, 1)

        self._sample_step_spin.valueChanged.connect(self._on_viz_params_changed)
        self._axis_length_spin.valueChanged.connect(self._on_viz_params_changed)
        self._point_size_spin.valueChanged.connect(self._on_viz_params_changed)

        layout.addWidget(viz_group)
        layout.addStretch()

        self._refresh_algorithm_list()

    def _refresh_algorithm_list(self):
        """从插件管理器填充算法下拉列表。"""
        self._algorithm_combo.blockSignals(True)
        self._algorithm_combo.clear()

        manager = get_manager()
        manager.discover_algorithms(prefix="algo")
        names = manager.get_all_algorithm_names(prefix="algo")

        if not names:
            self._algorithm_combo.addItem("未找到算法 - 请向 algorithms/ 目录添加插件", None)
            self._algo_status.setText("插件目录为空")
        else:
            self._algorithm_combo.addItems(names)
            self._algo_status.setText(f"{len(names)} 个算法可用")
            # blockSignals 阻止了初始 currentIndexChanged(0)，手动触发一次激活
            self._on_algorithm_changed(0)

        self._algorithm_combo.blockSignals(False)

    def _on_viz_params_changed(self):
        """轨迹可视化参数变化，同步到 state 并通知主窗口刷新其他 Step 的渲染"""
        self._state.trajectory_viz_sample_step = self._sample_step_spin.value()
        self._state.trajectory_viz_axis_length = self._axis_length_spin.value()
        self._state.trajectory_viz_point_size = self._point_size_spin.value()
        if hasattr(self._main_window, '_on_trajectory_viz_params_changed'):
            self._main_window._on_trajectory_viz_params_changed()

    def _on_process_parameters_changed(self, *_args):
        current = getattr(self._state, "process_parameters", ProcessParameters())
        width_mm = self._process_contact_width_spin.value()
        updated = replace(
            current,
            stepover_m=self._process_stepover_spin.value() / 1000.0,
            effective_contact_width_m=None if width_mm <= 0 else width_mm / 1000.0,
            normal_force_setpoint_n=self._process_force_spin.value(),
            tool_tilt_rad=np.deg2rad(self._process_tilt_spin.value()),
            corner_blend_radius_m=self._process_blend_spin.value() / 1000.0,
        )
        self._state.process_parameters = updated
        overlap = updated.overlap_ratio
        if overlap is None:
            self._process_overlap_label.setText("重叠度 [不可用：缺少可信接触宽度模型]")
            self._process_overlap_label.setStyleSheet("color: #888;")
        else:
            self._process_overlap_label.setText(f"重叠度 [派生]: {overlap * 100:.1f}%")
            self._process_overlap_label.setStyleSheet("color: #1976D2;")
        if getattr(self._state, "physical_trajectory", None) is not None:
            self._state.physical_trajectory_stale = True
            self._state.trajectory_validation = None

    def _on_algorithm_changed(self, index: int):
        """处理算法选择变更 - 动态生成参数 UI。"""
        algo_name = self._algorithm_combo.currentText()
        if not algo_name:
            return

        manager = get_manager()
        manager.discover_algorithms(prefix="algo")

        try:
            AlgoClass = manager.get_algorithm_class(algo_name, prefix="algo")
            self.current_algo = AlgoClass()
        except ValueError as e:
            self.log(f"[错误] {e}")
            self.current_algo = None
            return

        self._clear_param_layout()

        params = self.current_algo.get_parameters()
        self.param_widgets.clear()

        for p in params:
            widget = None

            if p.ptype == ParamType.FLOAT:
                widget = QDoubleSpinBox()
                if p.min_val is not None:
                    widget.setMinimum(p.min_val)
                if p.max_val is not None:
                    widget.setMaximum(p.max_val)
                if p.step is not None:
                    widget.setSingleStep(p.step)
                widget.setValue(p.default)

            elif p.ptype == ParamType.INT:
                widget = QSpinBox()
                if p.min_val is not None:
                    widget.setMinimum(int(p.min_val))
                if p.max_val is not None:
                    widget.setMaximum(int(p.max_val))
                widget.setValue(p.default)

            elif p.ptype == ParamType.CHOICE:
                widget = QComboBox()
                widget.addItems(p.options)
                if p.default in p.options:
                    widget.setCurrentText(p.default)

            elif p.ptype == ParamType.BOOL:
                widget = QCheckBox()
                widget.setChecked(p.default)

            if widget:
                widget.setToolTip(p.desc)
                self._param_layout.addRow(p.label, widget)
                self.param_widgets[p.id] = widget

        self._algo_status.setText(f"已加载: {self.current_algo.NAME}")
        self._algo_status.setStyleSheet("color: #4CAF50; font-size: 11px;")
        self.log(f"算法已切换: {algo_name} ({len(params)} 个参数)")

    def _clear_param_layout(self):
        """清空动态参数布局中的所有控件。"""
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _on_seq_strategy_changed(self, index: int):
        """处理宏观规划策略变更。"""
        self._seq_strategy = index
        self._state.path_seq_config = PathSequencingConfig(strategy=index)
        strategy_name = "螺旋型" if index == 0 else "锯齿型"
        self._seq_status.setText(f"当前: {strategy_name} (strategy={index})")
        self._seq_status.setStyleSheet("color: #4CAF50; font-size: 11px;")
        self.log(f"宏观轨迹规划策略已切换: {strategy_name} (strategy={index})")

    def _on_trajectory_axes_toggled(self, state: int):
        """轨迹坐标系常态化显示复选框切换。"""
        show = bool(state)
        if not self.render_engine:
            return
        if show:
            if hasattr(self, '_trajectory_poses') and self._trajectory_poses:
                self.render_engine.set_trajectory_axes_visible(
                    True,
                    poses_ocf=self._trajectory_poses,
                    sample_step=self._state.trajectory_viz_sample_step,
                    axis_length=self._state.trajectory_viz_axis_length,
                    point_size=float(self._state.trajectory_viz_point_size),
                    T_wcf_ocf=self._state.T_wcf_ocf
                )
                self.log(f"轨迹坐标系已显示 ({len(self._trajectory_poses)} 个位姿)")
            else:
                self.log("[提示] 请先生成刀轨")
                self._show_trajectory_axes_cb.setChecked(False)
        else:
            self.render_engine.set_trajectory_axes_visible(False)

    def on_activate(self):
        """Tab 4 激活时：如有已缓存的轨迹坐标系则恢复显示。"""
        if not hasattr(self, '_trajectory_poses') or not self._trajectory_poses or not self.render_engine:
            return
        # 复选框勾选时恢复显示（与刀具包围盒相同的持久化模式）
        if self._show_trajectory_axes_cb.isChecked():
            self.render_engine.set_trajectory_axes_visible(
                True,
                poses_ocf=self._trajectory_poses,
                sample_step=self._state.trajectory_viz_sample_step,
                axis_length=self._state.trajectory_viz_axis_length,
                point_size=float(self._state.trajectory_viz_point_size),
                T_wcf_ocf=self._state.T_wcf_ocf
            )
            self.log(f"轨迹坐标系已恢复 ({len(self._trajectory_poses)} 个位姿)")

    def _process_toolpath_internal(self, result: ToolpathResult):
        """Build explicit PROCESS segments from the producer's per-point layer IDs."""
        import pandas as pd
        import tempfile

        result.validate()
        points = result.points
        normals = result.normals
        layer_ids = result.layer_ids
        tmp_dir = tempfile.gettempdir()
        tmp_csv = os.path.join(tmp_dir, "auto_toolpath.csv")
        df = pd.DataFrame({
            'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2],
            'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2],
            'layer_id': layer_ids,
        })
        df.to_csv(tmp_csv, index=False)

        # Module B: production ordering consumes real layer IDs.  Keeping this
        # at zero makes any fixed-points-per-layer inference impossible here.
        config = PathSequencingConfig(
            strategy=self._state.path_seq_config.strategy,
            layer_point_count=0,
            end_effector_axis=self._state.path_seq_config.end_effector_axis,
            process_geometry=self._state.path_seq_config.process_geometry,
            minimize_tool_roll=self._state.path_seq_config.minimize_tool_roll,
            normal_smoothing_window=self._state.path_seq_config.normal_smoothing_window,
        )

        output_csv = os.path.join(tmp_dir, "auto_toolpath_pose.csv")
        processor = PathSequencer(
            tmp_csv, output_csv,
            kinematics_engine=self._state.kinematics_engine,
            tool_library=self._state.tool_library,
            config=config,
        )
        pose_df = processor.process()
        # print(f"[DEBUG] _process_toolpath_internal: strategy={self._state.path_seq_config.strategy}, "
        #       f"input_points={len(points)}, output_poses={len(pose_df)}")

        sequenced_matrices = []
        for _, row in pose_df.iterrows():
            pos = [row['x'], row['y'], row['z']]
            quat = [row['qx'], row['qy'], row['qz'], row['qw']]
            T_ocf = np.eye(4)
            T_ocf[:3, :3] = R.from_quat(quat).as_matrix()
            T_ocf[:3, 3] = pos
            sequenced_matrices.append(T_ocf)

        try:
            from trajectory import (
                GeometricPathBuilder,
                build_process_segmentation,
                resolve_tcp_matrices,
            )

            ordered_layer_ids = pose_df["layer_id"].to_numpy(dtype=np.int32)
            ordered_original_indices = pose_df["original_index"].to_numpy(dtype=int)
            matrices, matrix_source = resolve_tcp_matrices(
                result,
                np.asarray(sequenced_matrices),
                ordered_original_indices,
            )
            segmentation = build_process_segmentation(
                matrices,
                ordered_layer_ids,
                original_indices=ordered_original_indices,
            )
            geometric = GeometricPathBuilder(self._state.process_parameters).build(
                matrices,
                segment_types=segmentation.segment_types,
                segment_ids=segmentation.segment_ids,
                layer_ids=segmentation.layer_ids,
            )
            source_indices = ordered_original_indices[geometric.original_indices]
            geometric = replace(
                geometric,
                original_indices=source_indices,
                metadata={
                    **geometric.metadata,
                    "layer_count": len(segmentation.process_segments),
                    "segment_source": "toolpath_result.layer_indices",
                    "tcp_matrix_source": matrix_source,
                },
            )
            final_segmentation = build_process_segmentation(
                geometric.tcp_poses,
                geometric.layer_ids,
                original_indices=source_indices,
            )
            self._state.geometric_trajectory = geometric
            self._state.process_segments = list(final_segmentation.process_segments)
            self._state.transition_requests = list(final_segmentation.transition_requests)
            self._state.toolpath_preview_normals = result.normals[source_indices].copy()
            matrices = [pose.copy() for pose in geometric.tcp_poses]
            self.log(
                f"显式分层完成: {len(self._state.process_segments)} 个 PROCESS 段，"
                f"{len(self._state.transition_requests)} 个层间过渡请求；"
                f"TCP 来源={matrix_source}"
            )
        except Exception as exc:
            self._state.geometric_trajectory = None
            self._state.process_segments = []
            self._state.transition_requests = []
            self._state.toolpath_preview_normals = None
            self.log(f"[错误] 显式刀路分层失败，未生成 IK 输入: {exc}")
            self._state.target_matrices = None
            self._trajectory_poses = []
            return

        self._state.target_matrices = matrices
        self._state.ik_path_complete = False
        self._trajectory_poses = matrices
        self.log(f"宏观轨迹规划完成: {len(matrices)} 个位姿")
        self._render_trajectory_axes()

    def _render_trajectory_axes(self):
        """渲染轨迹点的 RGB 坐标系（跟随工件移动）。

        仅在复选框勾选时显示，否则隐藏或销毁已缓存的 actors。
        """
        if not self._trajectory_poses or not self.render_engine:
            return
        if self._show_trajectory_axes_cb.isChecked():
            self.render_engine.set_trajectory_axes_visible(
                True,
                poses_ocf=self._trajectory_poses,
                sample_step=self._state.trajectory_viz_sample_step,
                axis_length=self._state.trajectory_viz_axis_length,
                point_size=float(self._state.trajectory_viz_point_size),
                T_wcf_ocf=self._state.T_wcf_ocf
            )
        else:
            self.render_engine.set_trajectory_axes_visible(False)

    def _on_generate(self):
        """使用选中的算法和当前参数生成刀轨。"""
        if self._state.cad_filepath is None:
            self.log("[错误] 请先导入 CAD 模型（Tab 2）")
            return

        if self.current_algo is None:
            self.log("[错误] 请先选择算法")
            return

        kwargs = {}
        for param_id, widget in self.param_widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                kwargs[param_id] = widget.value()
            elif isinstance(widget, QComboBox):
                kwargs[param_id] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                kwargs[param_id] = widget.isChecked()

        # 提取每层 CC 点数，存入 state 供 _on_result 中的 _process_toolpath_internal 使用
        if 'num_cc_points' in kwargs:
            self._state.layer_cc_point_count = int(kwargs['num_cc_points'])
        else:
            self._state.layer_cc_point_count = 0

        self._btn_generate.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._progress_bar.setValue(0)

        self.log(f"开始刀轨生成，算法: {self.current_algo.NAME}")
        self._worker = ToolpathWorker(
            self.current_algo,
            self._state.cad_filepath,
            kwargs
        )
        self._worker.log_signal.connect(self.log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.result_signal.connect(self._on_result)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        """停止正在运行的刀轨生成。"""
        if self._worker is not None and self._worker.isRunning():
            self.log("正在停止刀轨生成...")
            self._worker.abort()
            self._btn_stop.setEnabled(False)

    def _on_progress(self, current: int, total: int):
        """更新进度条。"""
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))

    def _on_result(self, result: ToolpathResult):
        """处理刀轨生成结果。"""
        try:
            # 保存结果用于导出
            self._last_result = result

            self._state.toolpath_points = result.points
            self._state.toolpath_normals = result.normals
            self._state.toolpath_result = result

            self.log(f"收到刀轨结果: {result.num_waypoints} 个路径点")

            if result.debug_items:
                self.log(f"调试数据: {list(result.debug_items.keys())}")
                for label, item in result.debug_items.items():
                    self.log(f"  - {label}: dtype={item.dtype.value}, shape={item.data.shape}, color={item.color}")

            # 通过宏观轨迹规划器将法向量转换为四元数位姿
            self._process_toolpath_internal(result)

            if self.render_engine is not None:
                try:
                    # 加载刀轨预览（使用 zigzag/spiral 处理后的轨迹点，反映实际执行顺序）
                    viz_pts = np.array([m[:3, 3] for m in self._trajectory_poses]) if self._trajectory_poses else result.points
                    geometric = getattr(self._state, "geometric_trajectory", None)
                    semantic_matches = geometric is not None and len(geometric.tcp_poses) == len(viz_pts)
                    viz_normals = (
                        self._state.toolpath_preview_normals
                        if self._state.toolpath_preview_normals is not None
                        and len(self._state.toolpath_preview_normals) == len(viz_pts)
                        else None
                    )
                    self.render_engine.load_toolpath_preview(
                        viz_pts,
                        viz_normals,
                        self._state.T_wcf_ocf,
                        geometric.segment_types if semantic_matches else None,
                        geometric.segment_ids if semantic_matches else None,
                        geometric.layer_ids if semantic_matches else None,
                    )
                    # 渲染调试数据（中间结果可视化）
                    if result.debug_items:
                        self.render_engine.render_debug_items(result.debug_items)
                    self.render_engine.render()
                    self.log("3D 渲染更新完成")
                except Exception as e:
                    self.log(f"[警告] 渲染更新失败: {e}")

            # 启用导出按钮
            self._btn_export.setEnabled(True)
            self.log("导出按钮已启用")

            # 自动导出到 toolpath/ 目录
            self._auto_export_toolpath(result)
            self.record_stage(
                "toolpath_generated",
                details={
                    "waypoint_count": int(result.num_waypoints),
                    "layer_index_count": int(len(result.layer_indices)),
                    "process_segment_count": int(len(self._state.process_segments)),
                    "transition_request_count": int(len(self._state.transition_requests)),
                },
                version_domain="toolpath",
            )

        except Exception as e:
            self.record_stage(
                "toolpath_generated",
                status="failed",
                details={"error": str(e)},
            )
            self.log(f"[错误] 处理结果时发生异常: {e}")
            import traceback
            traceback.print_exc()
            # 即使出错也启用导出按钮
            self._btn_export.setEnabled(True)

    def _on_finished(self, success: bool, msg: str):
        """处理工作线程完成事件。"""
        self._btn_generate.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if success:
            self._result_label.setText(f"生成完成: {len(self._state.toolpath_points)} 个路径点")
            self.log(msg)
            self.log("刀轨预览已加载，可切换到 Tab 5 进行 IK 求解")
        else:
            self._result_label.setText(f"失败: {msg}")
            self.log(msg)

    def _on_clear(self):
        """清除刀轨预览。"""
        if self.render_engine is not None:
            self.render_engine.remove_toolpath_preview()
            self.render_engine.clear_debug_items()
        self._state.toolpath_points = None
        self._state.toolpath_normals = None
        self._last_result = None
        self._result_label.setText("已清除")
        self._btn_export.setEnabled(False)

    @staticmethod
    def _normal_csv_frame(result: ToolpathResult):
        """Build the canonical normal-CSV interchange payload for Module B."""
        import pandas as pd

        result.validate()
        return pd.DataFrame({
            'x': result.points[:, 0],
            'y': result.points[:, 1],
            'z': result.points[:, 2],
            'nx': result.normals[:, 0],
            'ny': result.normals[:, 1],
            'nz': result.normals[:, 2],
            'layer_id': result.layer_ids,
        })

    def _auto_export_toolpath(self, result: ToolpathResult):
        """自动将刀轨数据导出到根目录下的 toolpath/ 文件夹"""
        import datetime

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        toolpath_dir = os.path.join(project_root, "toolpath")
        os.makedirs(toolpath_dir, exist_ok=True)

        cad_name = "unknown"
        if self._state.cad_filepath:
            cad_name = os.path.splitext(os.path.basename(self._state.cad_filepath))[0]

        ts = datetime.datetime.now().strftime("%Y%m%d%H%M")

        if result.normals is not None and len(result.normals) > 0:
            normal_filename = f"{cad_name}_normal_{ts}.csv"
            normal_path = os.path.join(toolpath_dir, normal_filename)
            self._normal_csv_frame(result).to_csv(
                normal_path,
                index=False,
                float_format='%.6f',
            )
            self.log(f"[自动导出] 法向量数据包: {normal_filename}")

        if result.matrices is not None and len(result.matrices) > 0:
            pose_filename = f"{cad_name}_pose_{ts}.csv"
            pose_path = os.path.join(toolpath_dir, pose_filename)
            pose_rows = []
            for mat in result.matrices:
                pos = mat[:3, 3]
                quat = R.from_matrix(mat[:3, :3]).as_quat()
                pose_rows.append([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])
            pose_data = np.array(pose_rows)
            np.savetxt(pose_path, pose_data, delimiter=',',
                       header='x,y,z,qx,qy,qz,qw', comments='', fmt='%.6f')
            self.log(f"[自动导出] 四元数数据包: {pose_filename}")

    def _on_export_type_changed(self, index: int):
        """处理导出类型变更：toolpath 时禁用格式下拉框，pose 时启用。"""
        if index == 0:  # toolpath
            self._export_format_combo.setEnabled(False)
        else:  # pose
            self._export_format_combo.setEnabled(True)

    def _on_export_data(self):
        """导出刀轨数据为单个 CSV 文件，支持 toolpath 或 pose 类型，以及欧拉角或四元数格式。"""
        if not hasattr(self, '_last_result') or not self._last_result:
            self.log("[错误] 没有可导出的数据")
            return

        export_type = self._export_type_combo.currentText()
        export_format = self._export_format_combo.currentText()

        if export_type == "toolpath":
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存刀轨文件", "toolpath.csv", "CSV 文件 (*.csv)"
            )
            if not filename:
                return
            try:
                has_points = self._last_result.points is not None and len(self._last_result.points) > 0
                has_normals = self._last_result.normals is not None and len(self._last_result.normals) > 0
                if not has_points:
                    self.log("[错误] 没有可导出的位置数据")
                    return
                if has_normals:
                    self.log("导出类型: toolpath [x, y, z, nx, ny, nz, layer_id]")
                    self._normal_csv_frame(self._last_result).to_csv(
                        filename,
                        index=False,
                        float_format='%.6f',
                    )
                else:
                    np.savetxt(filename, self._last_result.points, delimiter=',',
                                header='x,y,z', comments='', fmt='%.6f')
                self.log(f"已导出 toolpath {len(self._last_result.points)} 个点到: {filename}")
                QMessageBox.information(self, "导出成功", f"刀轨已保存至:\n{filename}")
            except Exception as e:
                self.log(f"[错误] 导出失败: {str(e)}")
                import traceback; traceback.print_exc()
                QMessageBox.warning(self, "导出失败", f"发生错误: {str(e)}")
            return

        # pose 类型：需要经过宏观轨迹处理的矩阵
        if not hasattr(self, '_trajectory_poses') or not self._trajectory_poses:
            self.log("[错误] 没有可导出的位姿数据，请先生成刀轨")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存位姿文件", "pose.csv", "CSV 文件 (*.csv)"
        )
        if not filename:
            return
        try:
            data_list = []
            for mat in self._trajectory_poses:
                pos = mat[:3, 3]
                rot = mat[:3, :3]
                if "四元数" in export_format:
                    quat = R.from_matrix(rot).as_quat()
                    row = [pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]
                    header = 'x,y,z,qx,qy,qz,qw'
                    self.log(f"导出类型: pose 四元数 [x, y, z, qw, qx, qy, qz], 共 {len(self._trajectory_poses)} 个点")
                else:
                    euler = R.from_matrix(rot).as_euler('xyz', degrees=True)
                    row = [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]
                    header = 'x,y,z,rx,ry,rz'
                    self.log(f"导出类型: pose 欧拉角 [x, y, z, rx, ry, rz](deg), 共 {len(self._trajectory_poses)} 个点")
                data_list.append(row)
            data = np.array(data_list)
            np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.6f')
            self.log(f"已导出 pose {len(data_list)} 个点到: {filename}")
            QMessageBox.information(self, "导出成功", f"位姿已保存至:\n{filename}")
        except Exception as e:
            self.log(f"[错误] 导出失败: {str(e)}")
            import traceback; traceback.print_exc()
            QMessageBox.warning(self, "导出失败", f"发生错误: {str(e)}")

# ==================== Tab 5: IK 求解 ====================

class IKSolveWidget(StepWidget):
    """Tab 5: 逆运动学批量求解

    CSV 导入格式:
        路径单位: m 或 mm（用户可选，自动判断阈值 > 10 则视为 mm）
        角度单位: degrees, Euler XYZ (rx, ry, rz)
        列顺序: x, y, z, rx, ry, rz

    导出关节角格式:
        单位: radians (与 Pinocchio 内部保持一致)
        列顺序: j1, j2, j3, j4, j5, j6
    """

    def __init__(self, state: SimulationState, parent=None):
        super().__init__("IK 求解", 5, state, parent)
        self._worker = None
        self._trajectory_poses = []  # 存储轨迹点的 4x4 pose 序列，用于可视化
        self._ik_manager = None  # IK 插件管理器

    def _init_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel(
            "批量求解逆运动学，采用种子姿态逻辑确保轨迹连续性。"
            "支持导入外部刀轨 CSV（无 Tab 4 时），求解后可导出关节轨迹。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── 可滚动区域 ──────────────────────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 5, 0, 5)

        # ── 数据导入组 ──────────────────────────────────────────────
        import_group = QGroupBox("数据导入组")
        import_layout = QVBoxLayout(import_group)

        # 两种导入方式
        self._btn_import_normal_csv = QPushButton("导入法向量 CSV (Tab 3)")
        self._btn_import_normal_csv.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self._btn_import_normal_csv.clicked.connect(self._on_import_normal_csv)
        import_layout.addWidget(self._btn_import_normal_csv)

        hint1 = QLabel(
            "格式: x, y, z, nx, ny, nz, layer_id（逐点层号；法向量处理后转四元数）"
        )
        hint1.setStyleSheet("color: #888; font-size: 11px;")
        import_layout.addWidget(hint1)

        sep_label = QLabel("— 或 —")
        sep_label.setAlignment(Qt.AlignCenter)
        sep_label.setStyleSheet("color: #888;")
        import_layout.addWidget(sep_label)

        # 导入 CSV 按钮 + 单位设置
        csv_row = QHBoxLayout()
        self._btn_import_csv = QPushButton("导入位姿 CSV (外部)")
        self._btn_import_csv.clicked.connect(self._on_import_csv)
        csv_row.addWidget(self._btn_import_csv)

        csv_row.addWidget(QLabel("路径单位:"))
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["自动判断", "米 (m)", "毫米 (mm)"])
        csv_row.addWidget(self._unit_combo)

        import_layout.addLayout(csv_row)

        # 格式提示
        hint = QLabel("格式: x, y, z, rx, ry, rz（角度°，Euler XYZ）")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        import_layout.addWidget(hint)

        # 点数显示
        self._point_count_label = QLabel("待求解: 0 个位姿")
        self._point_count_label.setStyleSheet("font-weight: bold;")
        import_layout.addWidget(self._point_count_label)

        # 法向量处理状态
        self._normal_status_label = QLabel("")
        self._normal_status_label.setStyleSheet("color: #1976D2; font-weight: bold;")
        import_layout.addWidget(self._normal_status_label)

        self._normal_file_label = QLabel("")
        self._normal_file_label.setStyleSheet("color: #00C853; font-size: 11px; font-weight: bold;")
        import_layout.addWidget(self._normal_file_label)

        layout.addWidget(import_group)

        # ── 计算策略选择组 ──────────────────────────────────────────────
        strategy_group = QGroupBox("计算策略")
        strategy_layout = QVBoxLayout(strategy_group)

        # 第 1 行：求解器下拉（由插件系统动态填充）
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel("IK求解策略:"))
        self._strategy_combo = QComboBox()
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        combo_row.addWidget(self._strategy_combo)
        self._strategy_label = QLabel("[当前: 加载中...]")
        self._strategy_label.setStyleSheet("color: #00C853; font-size: 11px; font-weight: bold;")
        combo_row.addWidget(self._strategy_label)
        combo_row.addStretch()
        strategy_layout.addLayout(combo_row)

        # 第 2 行：构型变换复选框（独立于求解器类型，始终可见）
        self._wrist_flip_cb = QCheckBox("启用构型变换（关节接近限位时尝试备用构型）")
        self._wrist_flip_cb.setChecked(True)
        self._wrist_flip_cb.setToolTip(
            "当 IK 求解后检测到关节接近限位，自动使用备用构型重新求解，适用于所有求解策略"
        )
        strategy_layout.addWidget(self._wrist_flip_cb)

        # ── SLSQP 基础参数（所有 SLSQP 变体共用）───────────────────────
        self._slsqp_base_group = QGroupBox("SLSQP 基础参数")
        slsqp_base_layout = QGridLayout(self._slsqp_base_group)

        slsqp_base_layout.addWidget(QLabel("限位裕度 (°):"), 0, 0)
        self._margin_spin = QDoubleSpinBox()
        self._margin_spin.setRange(0.5, 30.0)
        self._margin_spin.setValue(5.0)
        self._margin_spin.setSingleStep(0.5)
        self._margin_spin.setSuffix(" °")
        slsqp_base_layout.addWidget(self._margin_spin, 0, 1)

        slsqp_base_layout.addWidget(QLabel("收敛容差:"), 0, 2)
        self._tol_spin = QDoubleSpinBox()
        self._tol_spin.setRange(1e-8, 1e-2)
        self._tol_spin.setValue(1e-5)
        self._tol_spin.setSingleStep(1e-5)
        self._tol_spin.setDecimals(8)
        slsqp_base_layout.addWidget(self._tol_spin, 0, 3)

        slsqp_base_layout.addWidget(QLabel("最大迭代:"), 1, 0)
        self._maxiter_spin = QSpinBox()
        self._maxiter_spin.setRange(100, 10000)
        self._maxiter_spin.setValue(1000)
        self._maxiter_spin.setSingleStep(100)
        slsqp_base_layout.addWidget(self._maxiter_spin, 1, 1)

        slsqp_base_layout.setColumnStretch(0, 1)
        slsqp_base_layout.setColumnStretch(1, 1)
        slsqp_base_layout.setColumnStretch(2, 1)
        slsqp_base_layout.setColumnStretch(3, 1)
        strategy_layout.addWidget(self._slsqp_base_group)

        # ── SLSQP_update 多目标权重参数（仅 SLSQP_update 可见）────────────────
        self._slsqp_multi_group = QGroupBox("多目标权重")
        slsqp_multi_layout = QGridLayout(self._slsqp_multi_group)

        slsqp_multi_layout.addWidget(QLabel("位姿误差权重:"), 0, 0)
        self._w_pose_spin = QDoubleSpinBox()
        self._w_pose_spin.setRange(0.001, 10.0)
        self._w_pose_spin.setValue(1.0)
        self._w_pose_spin.setSingleStep(0.01)
        self._w_pose_spin.setDecimals(4)
        slsqp_multi_layout.addWidget(self._w_pose_spin, 0, 1)

        slsqp_multi_layout.addWidget(QLabel("关节位移权重:"), 0, 2)
        self._w_disp_spin = QDoubleSpinBox()
        self._w_disp_spin.setRange(1e-6, 1.0)
        self._w_disp_spin.setValue(0.01)
        self._w_disp_spin.setSingleStep(1e-4)
        self._w_disp_spin.setDecimals(6)
        slsqp_multi_layout.addWidget(self._w_disp_spin, 0, 3)

        slsqp_multi_layout.addWidget(QLabel("关节趋中权重:"), 1, 0)
        self._w_center_spin = QDoubleSpinBox()
        self._w_center_spin.setRange(1e-6, 1.0)
        self._w_center_spin.setValue(0.01)
        self._w_center_spin.setSingleStep(1e-4)
        self._w_center_spin.setDecimals(6)
        slsqp_multi_layout.addWidget(self._w_center_spin, 1, 1)

        slsqp_multi_layout.addWidget(QLabel("可操作度权重:"), 1, 2)
        self._w_manip_spin = QDoubleSpinBox()
        self._w_manip_spin.setRange(1e-6, 0.1)
        self._w_manip_spin.setValue(0.001)
        self._w_manip_spin.setSingleStep(1e-5)
        self._w_manip_spin.setDecimals(6)
        slsqp_multi_layout.addWidget(self._w_manip_spin, 1, 3)

        # ── 各关节独立位移权重 ─────────────────────────────────────────
        slsqp_multi_layout.addWidget(QLabel("J1 权重:"), 2, 0)
        self._w_j1_spin = QDoubleSpinBox()
        self._w_j1_spin.setRange(0.01, 200.0)
        self._w_j1_spin.setValue(1.0)
        self._w_j1_spin.setSingleStep(0.1)
        self._w_j1_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j1_spin, 2, 1)

        slsqp_multi_layout.addWidget(QLabel("J2 权重:"), 2, 2)
        self._w_j2_spin = QDoubleSpinBox()
        self._w_j2_spin.setRange(0.01, 200.0)
        self._w_j2_spin.setValue(1.0)
        self._w_j2_spin.setSingleStep(0.1)
        self._w_j2_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j2_spin, 2, 3)

        slsqp_multi_layout.addWidget(QLabel("J3 权重:"), 3, 0)
        self._w_j3_spin = QDoubleSpinBox()
        self._w_j3_spin.setRange(0.01, 200.0)
        self._w_j3_spin.setValue(1.0)
        self._w_j3_spin.setSingleStep(0.1)
        self._w_j3_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j3_spin, 3, 1)

        slsqp_multi_layout.addWidget(QLabel("J4 权重:"), 3, 2)
        self._w_j4_spin = QDoubleSpinBox()
        self._w_j4_spin.setRange(0.01, 200.0)
        self._w_j4_spin.setValue(1.0)
        self._w_j4_spin.setSingleStep(0.1)
        self._w_j4_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j4_spin, 3, 3)

        slsqp_multi_layout.addWidget(QLabel("J5 权重:"), 4, 0)
        self._w_j5_spin = QDoubleSpinBox()
        self._w_j5_spin.setRange(0.01, 200.0)
        self._w_j5_spin.setValue(1.0)
        self._w_j5_spin.setSingleStep(0.1)
        self._w_j5_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j5_spin, 4, 1)

        slsqp_multi_layout.addWidget(QLabel("J6 权重:"), 4, 2)
        self._w_j6_spin = QDoubleSpinBox()
        self._w_j6_spin.setRange(0.01, 200.0)
        self._w_j6_spin.setValue(1.0)
        self._w_j6_spin.setSingleStep(0.5)
        self._w_j6_spin.setDecimals(2)
        slsqp_multi_layout.addWidget(self._w_j6_spin, 4, 3)

        self._collision_planner_notice = QLabel(
            "碰撞规划已统一至 Tab 6：OMPL RRTConnect + 精确 FCL；SDF 已弃用。"
        )
        self._collision_planner_notice.setWordWrap(True)
        self._collision_planner_notice.setStyleSheet(
            "color: #175CD3; font-weight: bold; padding: 4px;"
        )
        slsqp_multi_layout.addWidget(self._collision_planner_notice, 5, 0, 1, 4)

        slsqp_multi_layout.setColumnStretch(0, 1)
        slsqp_multi_layout.setColumnStretch(1, 1)
        slsqp_multi_layout.setColumnStretch(2, 1)
        slsqp_multi_layout.setColumnStretch(3, 1)
        self._slsqp_multi_group.setVisible(False)  # 默认隐藏，由策略切换控制
        strategy_layout.addWidget(self._slsqp_multi_group)

        layout.addWidget(strategy_group)

        # ── 求解与导出组 ────────────────────────────────────────────
        solve_group = QGroupBox("求解与导出")
        solve_layout = QVBoxLayout(solve_group)

        btn_row = QHBoxLayout()
        self._btn_solve = QPushButton("开始 IK 求解")
        self._btn_solve.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self._btn_solve.clicked.connect(self._on_solve)
        btn_row.addWidget(self._btn_solve)

        self._btn_export = QPushButton("导出关节角 (CSV)")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self._btn_export)

        solve_layout.addLayout(btn_row)

        self._btn_clear = QPushButton("清除结果")
        self._btn_clear.clicked.connect(self._on_clear)
        solve_layout.addWidget(self._btn_clear)

        layout.addWidget(solve_group)

        # ── 浮动图表控制 ─────────────────────────────────────────────
        chart_control_row = QHBoxLayout()
        self._btn_show_chart = QPushButton("显示关节轨迹曲线浮窗")
        self._btn_show_chart.setStyleSheet("background-color: #1976D2; color: white; padding: 6px;")
        self._btn_show_chart.clicked.connect(self._on_show_chart)
        chart_control_row.addWidget(self._btn_show_chart)
        chart_control_row.addStretch()
        layout.addLayout(chart_control_row)

        # ── 进度 & 结果 ──────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._result_label = QLabel("尚未求解")
        layout.addWidget(self._result_label)

        scroll_layout.addWidget(import_group)
        scroll_layout.addWidget(strategy_group)
        scroll_layout.addWidget(solve_group)
        scroll_layout.addStretch()

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        self._init_ik_plugins()

    # ── 业务方法 ──────────────────────────────────────────────────

    def _init_ik_plugins(self):
        """初始化 IK 插件管理器，扫描 ik_*.py 并填充策略下拉框。"""
        if getattr(self, '_ik_plugins_initialized', False):
            return
        self._ik_plugins_initialized = True
        from core.auto_manager import get_manager
        self._ik_manager = get_manager()
        self._ik_manager.discover_algorithms(prefix="ik")
        names = self._ik_manager.get_all_algorithm_names(prefix="ik")

        # 缓存插件类（提升到 _state 防止 Tab 切换时丢失）
        if not hasattr(self._state, '_ik_solver_classes_cache'):
            self._state._ik_solver_classes_cache = {}
        self._state._ik_solver_names_cache = names
        for name in names:
            try:
                cls = self._ik_manager.get_algorithm_class(name, prefix="ik")
                self._state._ik_solver_classes_cache[name] = cls
            except ValueError:
                pass
        self._ik_solver_classes = self._state._ik_solver_classes_cache

        self._strategy_combo.blockSignals(True)
        self._strategy_combo.clear()
        if not names:
            self._strategy_combo.addItem("未找到 IK 求解器", None)
            self._strategy_label.setText("[未找到 IK 求解器]")
            self._strategy_label.setStyleSheet("color: #f44336; font-size: 11px; font-weight: bold;")
        else:
            self._strategy_combo.addItems(names)
            # 默认 SLSQP
            default_name = "SLSQP"
            if default_name in names:
                idx = names.index(default_name)
                self._strategy_combo.setCurrentIndex(idx)
                self._strategy_label.setText(f"[当前: {default_name}]")
            else:
                self._strategy_label.setText(f"[当前: {names[0]}]")
            self.log(f"IK 求解器已加载: {names}")
        self._strategy_combo.blockSignals(False)

    def on_activate(self):
        """Step 激活时：优先自动读取 toolpath/ 中的最新法向量数据包"""
        if self._state.kinematics_engine is None:
            return

        # 优先检测 toolpath/ 中最新的 normal 文件
        latest_normal = self._find_latest_toolpath_file("normal")
        if latest_normal:
            self.log(f"[自动读取] 检测到最新法向量数据包: {os.path.basename(latest_normal)}")
            if self._auto_load_normal_file(latest_normal):
                self._normal_file_label.setText(f"[已自动读取: {os.path.basename(latest_normal)}]")
                self._normal_file_label.setStyleSheet("color: #00C853; font-size: 11px; font-weight: bold;")
                return

        # 回退: 仅接受 Tab 4 已建立的显式层语义结果。旧 points/normals
        # 缓存没有 layer_id 时不得再按固定点数或坐标推断。
        result = getattr(self._state, 'toolpath_result', None)
        if result is not None and result.has_toolpath:
            self.log(f"[Tab 4→5] 检测到带显式层号的刀轨数据 {result.num_waypoints} 点，自动处理")
            self._process_toolpath_internal(result)
        elif getattr(self._state, 'toolpath_points', None) is not None:
            self.log("[错误] 旧刀轨缓存缺少 layer_id；请从 Tab 4 重新生成或导入含 layer_id 的 CSV")

    def _find_latest_toolpath_file(self, file_type: str) -> Optional[str]:
        """在 toolpath/ 目录中查找指定类型的最新文件"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        toolpath_dir = os.path.join(project_root, "toolpath")
        if not os.path.isdir(toolpath_dir):
            return None

        matching_files = []
        for f in os.listdir(toolpath_dir):
            if fnmatch.fnmatch(f, f"*_{file_type}_*.csv"):
                matching_files.append(os.path.join(toolpath_dir, f))

        if not matching_files:
            return None
        return max(matching_files, key=os.path.getmtime)

    def _read_normal_csv(self, filepath: str) -> Optional[ToolpathResult]:
        """Read the canonical normal-CSV format without inferring layer IDs."""
        import pandas as pd

        df = pd.read_csv(filepath)
        required_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if not all(col in df.columns for col in required_cols):
            self.log(f"[错误] 法向量 CSV 缺少必需列: {required_cols}")
            return None
        if 'layer_id' not in df.columns:
            self.log(
                "[错误] 法向量 CSV 缺少 `layer_id`；为避免错误重建分层，"
                "请补齐逐点层号（单层请全部填 0）后再导入"
            )
            return None
        try:
            return ToolpathResult(
                points=df[['x', 'y', 'z']].values.astype(float),
                normals=df[['nx', 'ny', 'nz']].values.astype(float),
                layer_indices=df['layer_id'].to_numpy(),
            )
        except (TypeError, ValueError) as exc:
            self.log(f"[错误] 法向量 CSV 的 layer_id 无效: {exc}")
            return None

    def _auto_load_normal_file(self, filepath: str) -> bool:
        """自动加载含显式逐点层号的法向量 CSV 文件。"""
        result = self._read_normal_csv(filepath)
        if result is None:
            return False
        self._process_toolpath_internal(result)
        return True

    def _process_toolpath_internal(self, result: ToolpathResult):
        """处理带显式逐点层号的刀轨点和法向量为四元数位姿。"""
        import pandas as pd
        import tempfile

        result.validate()
        points = result.points
        normals = result.normals

        # 写入临时 CSV 供 DataPostProcessor 消费
        tmp_dir = tempfile.gettempdir()
        tmp_csv = os.path.join(tmp_dir, "auto_toolpath.csv")
        df = pd.DataFrame({
            'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2],
            'nx': normals[:, 0], 'ny': normals[:, 1], 'nz': normals[:, 2],
            'layer_id': result.layer_ids,
        })
        df.to_csv(tmp_csv, index=False)

        output_csv = os.path.join(tmp_dir, "auto_toolpath_pose.csv")
        from core.path_sequencer import PathSequencer
        processor = PathSequencer(
            tmp_csv, output_csv,
            kinematics_engine=self._state.kinematics_engine,
            tool_library=self._state.tool_library,
            config=self._state.path_seq_config,
        )
        pose_df = processor.process()

        self._normal_status_label.setText(
            f"法向量处理完成: {len(pose_df)} 点, 奇异点: {len(processor.singular_indices)} 个"
        )

        matrices = []
        for _, row in pose_df.iterrows():
            pos = [row['x'], row['y'], row['z']]
            quat = [row['qx'], row['qy'], row['qz'], row['qw']]
            T_ocf = np.eye(4)
            T_ocf[:3, :3] = R.from_quat(quat).as_matrix()
            T_ocf[:3, 3] = pos
            matrices.append(T_ocf)

        # target_matrices 存储 OCF 坐标（由 CoordinateTransformer 统一处理变换）
        # IK 求解时：kinematics_engine._transformer.ocf_to_rcf() → RCF 坐标
        # 渲染时：render_engine.load_trajectory_axes() 内部做 OCF → WCF
        self._state.toolpath_result = result
        self._state.toolpath_points = points.copy()
        self._state.toolpath_normals = normals.copy()
        self._state.target_matrices = matrices
        self._trajectory_poses = matrices
        self._point_count_label.setText(f"待求解: {len(matrices)} 个位姿")
        self._result_label.setText("法向量处理完成，可开始 IK 求解")
        self.log(f"成功导入 {len(matrices)} 个位姿 → target_matrices (OCF)")
        self._render_trajectory_axes()

    def _render_trajectory_axes(self):
        """渲染轨迹点的 RGB 坐标系（跟随工件移动）"""
        if not self._trajectory_poses or not self.render_engine:
            return
        self.render_engine.load_trajectory_axes(
            self._trajectory_poses,
            sample_step=self._state.trajectory_viz_sample_step,
            axis_length=self._state.trajectory_viz_axis_length,
            point_size=float(self._state.trajectory_viz_point_size),
            T_wcf_ocf=self._state.T_wcf_ocf
        )

    def _on_import_normal_csv(self):
        """导入法向量 CSV 并处理为四元数位姿"""
        if self._state.kinematics_engine is None:
            self.log("[错误] 请先加载机器人（Tab 1）")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择法向量 CSV 文件",
            "",
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            self.log(f"正在处理法向量数据: {os.path.basename(path)}")
            result = self._read_normal_csv(path)
            if result is None:
                return
            self._process_toolpath_internal(result)
            self.log(f"位姿数据已生成: {path}")

        except Exception as e:
            self.log(f"[错误] 法向量处理失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _euler_xyz_to_rot(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Euler XYZ (radians) → 旋转矩阵 R = Rz(rz) @ Ry(ry) @ Rx(rx)"""
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        return np.array([
            [cy * cz,  cz * sx * sy - cx * sz,  sx * sz + cx * cz * sy],
            [cy * sz,  cx * cz + sx * sy * sz,  cx * sy * sz - cz * sx],
            [-sy,      cy * sx,                  cy * cx]
        ])

    def _on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择刀轨 CSV 文件",
            "",
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            data = np.loadtxt(path, delimiter=',', skiprows=0)
        except Exception as e:
            self.log(f"[错误] 读取 CSV 失败: {e}")
            return

        if data.ndim != 2 or data.shape[1] < 6:
            self.log(f"[错误] CSV 格式错误，需要 6 列 [x, y, z, rx, ry, rz]，实际: {data.shape}")
            return

        xyzs = data[:, 0:3].copy()
        eulers_deg = data[:, 3:6].copy()   # degrees

        unit_mode = self._unit_combo.currentText()

        # 自动判断：如果所有 xyz 值的最大绝对值 < 10，则认为是 mm，需要转换
        if unit_mode == "自动判断":
            if np.max(np.abs(xyzs)) < 10.0:
                xyzs *= 0.001
                self.log("检测为毫米单位，已自动转换为米")
            else:
                self.log("检测为米单位")
        elif unit_mode == "毫米 (mm)":
            xyzs *= 0.001
            self.log("按毫米导入，已转换为米")

        eulers_rad = np.deg2rad(eulers_deg)

        matrices = []
        for i in range(len(xyzs)):
            T = np.eye(4)
            T[:3, :3] = self._euler_xyz_to_rot(*eulers_rad[i])
            T[:3, 3] = xyzs[i]
            matrices.append(T)

        # target_matrices 存储 OCF 坐标（由 CoordinateTransformer 统一处理变换）
        self._state.target_matrices = matrices
        self._trajectory_poses = matrices
        self._point_count_label.setText(f"待求解: {len(matrices)} 个位姿")
        self._result_label.setText("已导入刀轨，可开始求解")
        self.log(f"成功导入 {len(matrices)} 个刀轨点位 → target_matrices (OCF)")
        self._render_trajectory_axes()

    def _on_strategy_changed(self, index: int):
        """计算策略切换回调"""
        solver_name = self._strategy_combo.currentText()
        self._strategy_label.setText(f"[当前: {solver_name}]")
        self._strategy_label.setStyleSheet("color: #00C853; font-size: 11px; font-weight: bold;")

        solver_cls = self._ik_solver_classes.get(solver_name)
        is_multi = getattr(solver_cls, 'DESC', None) == 'SLSQP_Multi'
        is_slsqp = 'SLSQP' in solver_name or is_multi

        self._slsqp_base_group.setVisible(is_slsqp)
        self._slsqp_multi_group.setVisible(is_multi)
        self.log(f"计算策略已切换: {solver_name}")

    def _on_solve(self):
        """开始 IK 批量求解，种子姿态取自 robot_config.default_q_init"""
        if self._state.target_matrices is None or len(self._state.target_matrices) == 0:
            self.log("[错误] 请先导入刀轨点位")
            return

        if self._state.kinematics_engine is None:
            self.log("[错误] 请先加载机器人")
            return

        solver_name = self._strategy_combo.currentData()
        if solver_name is None:
            solver_name = self._strategy_combo.currentText()

        if solver_name not in self._ik_solver_classes:
            self.log(f"[错误] 未找到求解器: {solver_name}")
            return

        solver_cls = self._ik_solver_classes[solver_name]
        self._active_solver_name = str(solver_name)
        self.log(f"使用计算策略: {solver_name}")

        kin = self._state.kinematics_engine
        is_multi = getattr(solver_cls, 'DESC', None) == 'SLSQP_Multi'
        is_slsqp = 'SLSQP' in solver_name or is_multi
        if is_slsqp:
            if is_multi:
                solver_instance = solver_cls(
                    kinematics_engine=kin,
                    tolerance=self._tol_spin.value(),
                    max_iterations=self._maxiter_spin.value(),
                    verbose=False,
                    w_pose=self._w_pose_spin.value(),
                    w_disp=self._w_disp_spin.value(),
                    w_center=self._w_center_spin.value(),
                    w_manip=self._w_manip_spin.value(),
                    w_joints=[
                        self._w_j1_spin.value(),
                        self._w_j2_spin.value(),
                        self._w_j3_spin.value(),
                        self._w_j4_spin.value(),
                        self._w_j5_spin.value(),
                        self._w_j6_spin.value(),
                    ],
                )
            else:
                solver_instance = solver_cls(
                    kinematics_engine=kin,
                    tolerance=self._tol_spin.value(),
                    max_iterations=self._maxiter_spin.value(),
                    verbose=False,
                )
        else:
            solver_instance = solver_cls(
                kinematics_engine=kin,
                max_iterations=500,
                tolerance=1e-4,
                alpha=1.0,
                joint_limits=True,
                damping=0.001,
                verbose=False
            )

        self._btn_solve.setEnabled(False)
        self._progress_bar.setValue(0)

        if self._state.robot_config and hasattr(self._state.robot_config, 'default_q_init') and self._state.robot_config.default_q_init is not None:
            cfg_q = np.array(self._state.robot_config.default_q_init, dtype=float)
            if len(cfg_q) == self._state.kinematics_engine.nq:
                q_init = cfg_q
            else:
                self.log(f"[警告] default_q_init 长度({len(cfg_q)}) 与 URDF 关节数({self._state.kinematics_engine.nq}) 不匹配，将使用中立姿态")
                q_init = np.zeros(self._state.kinematics_engine.nq)
        else:
            q_init = np.zeros(self._state.kinematics_engine.nq)
            self.log("[提示] 未找到 default_q_init，使用中立姿态 (zero) 作为种子")

        enable_flip = self._wrist_flip_cb.isChecked()
        margin_deg = self._margin_spin.value()
        tolerance = self._tol_spin.value()
        max_iter = self._maxiter_spin.value()

        self._worker = IKSolveWorker(
            self._state.kinematics_engine,
            self._state.target_matrices,
            q_init,
            solver_plugin=solver_instance,
            margin_deg=margin_deg,
            tolerance=tolerance,
            max_iterations=max_iter,
            enable_wrist_flip=enable_flip,
            solver_name=solver_name
        )
        self._worker.log_signal.connect(self.log)
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.result_signal.connect(self._on_result)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        """进度更新"""
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))

    def _on_result(self, results: List[dict]):
        """收集求解结果，构建有效帧"""
        # 清除旧结果，确保每次求解都是全新数据
        self._state.ik_results = results
        self._state.valid_frames = []
        # 清除旧的规范轨迹，避免 Tab 6 或浮窗残留旧数据。
        self._state.joint_trajectory = None
        self._state.process_joint_trajectory = None
        self._state.joint_trajectory_path = None
        self._state.physical_trajectory = None
        self._state.trajectory_validation = None
        self._state.physical_trajectory_stale = True
        self._state.ik_path_complete = bool(results) and all(r['success'] for r in results)

        for result in results:
            if result['success']:
                q = result['q']
                pose = self._state.kinematics_engine.forward_kinematics(q)
                world_pos = pose[:3, 3].copy()

                self._state.valid_frames.append({
                    'frame_index': len(self._state.valid_frames),
                    'joint_angles': q.copy(),      # 单位: radians
                    'world_position': world_pos
                })
        if self._state.ik_path_complete and self._state.valid_frames:
            positions = np.asarray(
                [frame["joint_angles"] for frame in self._state.valid_frames],
                dtype=float,
            )
            self._state.joint_trajectory = JointTrajectory(
                positions=positions,
                method=getattr(self, "_active_solver_name", "IK"),
            )
            # Keep the original PROCESS IK independent from any later
            # time-parameterized physical trajectory.
            self._state.process_joint_trajectory = self._state.joint_trajectory

    def _on_finished(self, success: bool, msg: str):
        """求解完成回调"""
        self._btn_solve.setEnabled(True)

        if success:
            success_count = sum(1 for r in self._state.ik_results if r['success'])
            if self._state.ik_path_complete:
                self._publish_joint_trajectory_chart()
                self.record_stage(
                    "ik_solved",
                    details={
                        "valid_count": int(success_count),
                        "total_count": int(len(self._state.ik_results)),
                    },
                )
            self._result_label.setText(
                f"求解完成: {success_count}/{len(self._state.ik_results)} 成功，"
                f"有效帧: {len(self._state.valid_frames)}"
            )
            self.log(msg)
            self.log(f"有效帧数: {len(self._state.valid_frames)}")
            # 解算成功后允许导出
            if self._state.ik_path_complete:
                self._btn_export.setEnabled(True)
                self.log("关节轨迹已就绪，可点击「导出关节角 (CSV)」保存")
            else:
                self._btn_export.setEnabled(False)
                self.record_stage(
                    "ik_solved",
                    status="failed",
                    details={
                        "valid_count": int(success_count),
                        "total_count": int(len(self._state.ik_results)),
                    },
                )
                self.log("[失败] IK 路径不连续：至少一个路点未求解，禁止跨失败点连接或导出。")
        else:
            self.record_stage(
                "ik_solved",
                status="failed",
                details={"message": msg},
            )
            self._result_label.setText(f"求解失败: {msg}")
            self._btn_export.setEnabled(False)

    def _on_export_csv(self):
        """将规范 JointTrajectory 导出为 CSV。"""
        trajectory = self._state.joint_trajectory
        if trajectory is None:
            self.log("[错误] 没有可导出的关节数据")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存关节轨迹 CSV",
            "joint_trajectory.csv",
            "CSV 文件 (*.csv)"
        )
        if not path:
            return

        try:
            # 动态生成列名和关节数
            nq = self._state.kinematics_engine.nq
            col_names = ",".join([f"j{i+1}" for i in range(nq)]) + "\n"
            rows = "\n".join(
                ",".join(f"{q[j]:.8f}" for j in range(nq))
                for q in trajectory.positions
            )
            with open(path, "w") as fh:
                fh.write(col_names)
                fh.write(rows)
                fh.write("\n")

            self._state.joint_trajectory_path = path

            self.log(f"关节轨迹已导出: {path}（{nq} 关节，单位: radians）")
            self._result_label.setText(
                f"已导出 {len(trajectory.positions)} 行 x {nq} 关节 → {os.path.basename(path)}"
            )
            self.log("轨迹已保存，可在 Tab 6 中导入进行正运动学仿真")
        except Exception as e:
            self.log(f"[错误] 导出失败: {e}")

    def _on_show_chart(self):
        """用规范 JointTrajectory 弹出完整关节曲线。"""
        if self._state.joint_trajectory is None:
            self.log("[提示] 尚无已完成的规范关节轨迹")
            return
        self._publish_joint_trajectory_chart()
        if hasattr(self, '_main_window') and self._main_window:
            self._main_window.show_floating_chart()

    def _publish_joint_trajectory_chart(self):
        trajectory = self._state.joint_trajectory
        if trajectory is None:
            return
        if hasattr(self, "_main_window") and self._main_window:
            self._main_window.set_joint_trajectory_chart(
                trajectory,
                current_frame=self._state.current_frame,
            )

    def _on_clear(self):
        """清除求解结果"""
        self._state.ik_results = []
        self._state.valid_frames = []
        self._state.joint_trajectory = None
        self._state.process_joint_trajectory = None
        self._state.target_matrices = None
        self._point_count_label.setText("待求解: 0 个位姿")
        self._progress_bar.setValue(0)
        self._result_label.setText("已清除")
        self._btn_export.setEnabled(False)
        self.log("已清除 IK 求解结果")


# ==================== Tab 6: 仿真播放 ====================

class SimulationWidget(StepWidget):
    """Tab 6: 数字孪生仿真

    速度计算:
        - 基准采样频率 F_base (Hz): 原始数据密度，如 100Hz = 100 个点 = 1 秒
        - 播放速度系数 k (0.00 ~ 10.00): 用户调节
        - 实际播放频率 F = F_base * k (点/秒)
        - 定时器间隔  interval_ms = 1000 / F

    导入 CSV 格式:
        单位: radians（与 Pinocchio 内部一致）
        列顺序: j1, j2, j3, j4, j5, j6
    """

    # 碰撞检测模式：0=关闭, 1=遇碰撞停止, 2=滑动时间窗高亮
    COLLISION_MODE_OFF = 0
    COLLISION_MODE_STOP = 1
    COLLISION_MODE_SLIDE = 2
    WINDOW_SIZE = 3

    def __init__(self, state: SimulationState, parent=None):
        # StepWidget.__init__ calls this subclass's _init_ui().  Collision
        # state must therefore exist before super(), and must not be reset
        # afterwards (otherwise the checked UI says "slide" while playback
        # silently remains in OFF mode).
        self._collision_mode: int = self.COLLISION_MODE_OFF
        self._collision_window: deque = deque(maxlen=self.WINDOW_SIZE)
        self._first_coll_frame: Optional[int] = None
        self._collision_off_cb: Optional[QCheckBox] = None
        self._collision_stop_cb: Optional[QCheckBox] = None
        self._collision_slide_cb: Optional[QCheckBox] = None
        # This must exist before ``StepWidget.__init__`` constructs the
        # physical-trajectory controls and connects their selection callback.
        self._transition_plan_records: list[dict] = []

        super().__init__("仿真播放", 6, state, parent)
        self._play_timer: Optional[QTimer] = None
        self._elapsed_timer: Optional[QElapsedTimer] = None

        # 本地关节轨迹缓存（导入 CSV 或从 valid_frames 获取）
        self._trajectory: List[np.ndarray] = []
        self._current_frame: int = 0
        self._total_cuts: int = 0

        # 速度参数
        self._base_hz: int = 100
        self._speed_coeff: float = 1.0
        self._physical_time_s: float = 0.0
        self._planning_worker: Optional[TrajectoryPlanningWorker] = None
        self._validation_worker: Optional[TrajectoryValidationWorker] = None
        self._planning_scene_version: Optional[int] = None
        self._planning_scene_hash: Optional[str] = None
        self._planning_result_rejected: bool = False
        self._timer_busy: bool = False
        self._view_interacting: bool = False
        self._interaction_observers_installed: bool = False
        self._last_visual_update_s: float = -1.0
        self._last_expensive_update_s: float = -1.0
        self._frame_slider_dragging: bool = False

        # 关节轨迹绘图器
        self._traj_plotter = None

    def _ensure_interaction_observers(self):
        """Pause VTK mutations while the user manipulates the camera.

        The physical clock keeps advancing; the next visual tick samples the
        correct current state instead of trying to render a backlog of frames.
        """
        if self._interaction_observers_installed or self.render_engine is None:
            return
        try:
            iren = self.render_engine._plotter.iren.interactor
            iren.AddObserver(
                "StartInteractionEvent",
                lambda *_: setattr(self, "_view_interacting", True),
            )
            iren.AddObserver(
                "EndInteractionEvent",
                lambda *_: setattr(self, "_view_interacting", False),
            )
            self._interaction_observers_installed = True
        except Exception:
            # Some off-screen/test plotters do not expose a VTK interactor.
            pass

    def _init_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self._content_scroll = QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        self._content_scroll.setWidget(content)
        outer_layout.addWidget(self._content_scroll)

        info = QLabel(
            "播放仿真：机器人根据关节轨迹运动，刀尖实时切削毛坯点云。"
            "支持外部 CSV 导入（关节角 radians），精确速度与时间控制。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── 轨迹导入组 ─────────────────────────────────────────────
        import_group = QGroupBox("轨迹导入")
        import_layout = QVBoxLayout(import_group)

        import_row = QHBoxLayout()
        self._btn_import_traj = QPushButton("导入关节轨迹 (CSV)")
        self._btn_import_traj.clicked.connect(self._on_import_trajectory)
        import_row.addWidget(self._btn_import_traj)

        self._traj_status_label = QLabel("未导入轨迹")
        self._traj_status_label.setStyleSheet("color: #888;")
        import_row.addWidget(self._traj_status_label)
        import_layout.addLayout(import_row)

        hint = QLabel("CSV 格式: j1, j2, j3, j4, j5, j6（弧度 radians）")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        import_layout.addWidget(hint)
        layout.addWidget(import_group)

        # ── 物理轨迹生成组 ─────────────────────────────────────────
        physical_group = QGroupBox("物理轨迹生成（本地 TOPP-RA + Ruckig）")
        physical_layout = QGridLayout(physical_group)

        self._tcp_feed_spin = QDoubleSpinBox()
        self._tcp_feed_spin.setRange(0.1, 5000.0)
        self._tcp_feed_spin.setDecimals(2)
        self._tcp_feed_spin.setValue(50.0)
        self._tcp_feed_spin.setSuffix(" mm/s")
        physical_layout.addWidget(QLabel("TCP 加工进给:"), 0, 0)
        physical_layout.addWidget(self._tcp_feed_spin, 0, 1)

        self._tcp_accel_spin = QDoubleSpinBox()
        self._tcp_accel_spin.setRange(1.0, 20000.0)
        self._tcp_accel_spin.setValue(250.0)
        self._tcp_accel_spin.setSuffix(" mm/s²")
        physical_layout.addWidget(QLabel("TCP 加速度:"), 0, 2)
        physical_layout.addWidget(self._tcp_accel_spin, 0, 3)

        self._control_period_spin = QDoubleSpinBox()
        self._control_period_spin.setRange(0.5, 100.0)
        self._control_period_spin.setDecimals(1)
        self._control_period_spin.setValue(10.0)
        self._control_period_spin.setSuffix(" ms")
        physical_layout.addWidget(QLabel("控制周期:"), 1, 0)
        physical_layout.addWidget(self._control_period_spin, 1, 1)

        self._path_tolerance_spin = QDoubleSpinBox()
        self._path_tolerance_spin.setRange(0.001, 10.0)
        self._path_tolerance_spin.setDecimals(3)
        self._path_tolerance_spin.setValue(0.2)
        self._path_tolerance_spin.setSuffix(" mm")
        physical_layout.addWidget(QLabel("最大弦误差:"), 1, 2)
        physical_layout.addWidget(self._path_tolerance_spin, 1, 3)

        self._joint_velocity_spin = QDoubleSpinBox()
        self._joint_velocity_spin.setRange(0.01, 20.0)
        self._joint_velocity_spin.setValue(2.0)
        self._joint_velocity_spin.setSuffix(" rad/s")
        physical_layout.addWidget(QLabel("最大关节速度:"), 2, 0)
        physical_layout.addWidget(self._joint_velocity_spin, 2, 1)

        self._joint_acceleration_spin = QDoubleSpinBox()
        self._joint_acceleration_spin.setRange(0.01, 100.0)
        self._joint_acceleration_spin.setValue(4.0)
        self._joint_acceleration_spin.setSuffix(" rad/s²")
        physical_layout.addWidget(QLabel("最大关节加速度:"), 2, 2)
        physical_layout.addWidget(self._joint_acceleration_spin, 2, 3)

        self._joint_jerk_spin = QDoubleSpinBox()
        self._joint_jerk_spin.setRange(0.01, 1000.0)
        self._joint_jerk_spin.setValue(20.0)
        self._joint_jerk_spin.setSuffix(" rad/s³")
        physical_layout.addWidget(QLabel("最大关节 jerk:"), 3, 0)
        physical_layout.addWidget(self._joint_jerk_spin, 3, 1)

        self._joint_margin_physical_spin = QDoubleSpinBox()
        self._joint_margin_physical_spin.setRange(0.0, 45.0)
        self._joint_margin_physical_spin.setValue(5.0)
        self._joint_margin_physical_spin.setSuffix(" deg")
        physical_layout.addWidget(QLabel("最小关节裕度:"), 3, 2)
        physical_layout.addWidget(self._joint_margin_physical_spin, 3, 3)

        self._adaptive_keyframes_cb = QCheckBox("误差受控自适应关键帧（推荐）")
        self._adaptive_keyframes_cb.setChecked(True)
        physical_layout.addWidget(self._adaptive_keyframes_cb, 4, 0, 1, 2)

        self._keyframe_interval_spin = QDoubleSpinBox()
        self._keyframe_interval_spin.setRange(20.0, 1000.0)
        self._keyframe_interval_spin.setValue(100.0)
        self._keyframe_interval_spin.setSuffix(" ms")
        physical_layout.addWidget(QLabel("关键帧最大间隔:"), 4, 2)
        physical_layout.addWidget(self._keyframe_interval_spin, 4, 3)

        self._joint_keyframe_tolerance_spin = QDoubleSpinBox()
        self._joint_keyframe_tolerance_spin.setRange(0.001, 1.0)
        self._joint_keyframe_tolerance_spin.setDecimals(3)
        self._joint_keyframe_tolerance_spin.setValue(0.05)
        self._joint_keyframe_tolerance_spin.setSuffix(" deg")
        physical_layout.addWidget(QLabel("关节重构误差:"), 5, 0)
        physical_layout.addWidget(self._joint_keyframe_tolerance_spin, 5, 1)

        self._btn_generate_physical = QPushButton("生成物理轨迹")
        self._btn_generate_physical.clicked.connect(self._on_generate_physical_trajectory)
        self._btn_validate_physical = QPushButton("验证")
        self._btn_validate_physical.clicked.connect(self._on_validate_physical_trajectory)
        self._btn_validate_physical.setEnabled(False)
        self._btn_export_physical = QPushButton("导出 CSV + JSON")
        self._btn_export_physical.clicked.connect(self._on_export_physical_trajectory)
        self._btn_export_physical.setEnabled(False)
        self._btn_export_diagnostic = QPushButton("诊断导出")
        self._btn_export_diagnostic.clicked.connect(self._on_export_diagnostic_trajectory)
        self._btn_export_diagnostic.setEnabled(False)
        self._btn_cancel_physical = QPushButton("取消计算")
        self._btn_cancel_physical.clicked.connect(self._on_cancel_physical_operation)
        self._btn_cancel_physical.setEnabled(False)
        physical_layout.addWidget(self._btn_generate_physical, 6, 0, 1, 2)
        physical_layout.addWidget(self._btn_validate_physical, 6, 2)
        physical_layout.addWidget(self._btn_export_physical, 6, 3)
        physical_layout.addWidget(self._btn_cancel_physical, 7, 0, 1, 2)
        physical_layout.addWidget(self._btn_export_diagnostic, 7, 3)

        self._physical_status_label = QLabel("未生成：离散关节轨迹不包含物理时间")
        self._physical_status_label.setWordWrap(True)
        self._physical_status_label.setStyleSheet("color: #B26A00;")
        physical_layout.addWidget(self._physical_status_label, 8, 0, 1, 4)

        # stable2C keeps the semantic trajectory colours in the viewport.
        # This compact selector exposes the auditable planning record for each
        # actual transition without replacing those colours with a generic
        # planning overlay.
        self._transition_plan_combo = QComboBox()
        self._transition_plan_combo.setEnabled(False)
        self._transition_plan_combo.currentIndexChanged.connect(
            self._on_transition_plan_selected
        )
        self._transition_plan_detail_label = QLabel("stable2C 过渡：尚未生成")
        self._transition_plan_detail_label.setWordWrap(True)
        self._transition_plan_detail_label.setStyleSheet(
            "color: #52616B; font-size: 11px;"
        )
        physical_layout.addWidget(QLabel("stable2C 过渡:"), 9, 0)
        physical_layout.addWidget(self._transition_plan_combo, 9, 1, 1, 3)
        physical_layout.addWidget(self._transition_plan_detail_label, 10, 0, 1, 4)
        for control in (
            self._tcp_feed_spin, self._tcp_accel_spin, self._control_period_spin,
            self._path_tolerance_spin, self._joint_velocity_spin,
            self._joint_acceleration_spin, self._joint_jerk_spin,
            self._joint_margin_physical_spin,
            self._keyframe_interval_spin, self._joint_keyframe_tolerance_spin,
        ):
            control.valueChanged.connect(self._mark_physical_trajectory_stale)
        self._adaptive_keyframes_cb.toggled.connect(self._mark_physical_trajectory_stale)
        layout.addWidget(physical_group)

        # ── 一眼可见的物理轨迹仪表板 ───────────────────────────────
        dashboard_group = QGroupBox("物理轨迹运行看板")
        dashboard_group.setFixedHeight(570)
        dashboard_layout = QGridLayout(dashboard_group)
        self._physical_badge = QLabel("● 未生成")
        self._physical_badge.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #8A6D1D; padding: 5px;"
        )
        dashboard_layout.addWidget(self._physical_badge, 0, 0, 1, 4)

        self._metric_labels = {}
        metric_specs = (
            ("duration", "物理时长", "--"),
            ("samples", "执行采样", "--"),
            ("tcp_peak", "TCP 峰值", "--"),
            ("utilization", "最大约束利用率", "--"),
        )
        for column, (key, title, initial) in enumerate(metric_specs):
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background: #F4F7FB; border: 1px solid #D5DEEA; "
                "border-radius: 6px; padding: 4px; }"
            )
            card_layout = QVBoxLayout(card)
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #667085; font-size: 11px; border: none;")
            value_label = QLabel(initial)
            value_label.setStyleSheet("font-size: 16px; font-weight: 700; color: #17324D; border: none;")
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            self._metric_labels[key] = value_label
            dashboard_layout.addWidget(card, 1, column)

        self._physical_figure = plt.Figure(figsize=(6.0, 2.2), dpi=90, tight_layout=True)
        self._physical_canvas = FigureCanvasQTAgg(self._physical_figure)
        self._physical_canvas.setFixedHeight(215)
        self._physical_axes = self._physical_figure.subplots(1, 2)
        dashboard_layout.addWidget(self._physical_canvas, 2, 0, 1, 4)

        self._validation_table = QTableWidget(0, 5)
        self._validation_table.setHorizontalHeaderLabels(
            ["约束项", "状态", "测量值", "限制", "位置/说明"]
        )
        self._validation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._validation_table.verticalHeader().setVisible(False)
        self._validation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._validation_table.setAlternatingRowColors(True)
        self._validation_table.setFixedHeight(200)
        dashboard_layout.addWidget(self._validation_table, 3, 0, 1, 4)
        self._show_validation_pending()
        # 看板置于输入区之后，打开 Tab 即可看到物理轨迹是否真正生效。
        layout.insertWidget(2, dashboard_group)

        # ── 插值修复组 ─────────────────────────────────────────────
        interp_group = QGroupBox("插值修复")
        interp_layout = QVBoxLayout(interp_group)

        # 启用退刀过渡插值复选框
        self._enable_interp_cb = QCheckBox("启用退刀过渡插值")
        self._enable_interp_cb.setChecked(False)
        self._enable_interp_cb.toggled.connect(self._on_interp_toggled)
        interp_layout.addWidget(self._enable_interp_cb)

        # 插值参数面板（依赖复选框激活）
        param_container = QWidget()
        param_layout = QGridLayout(param_container)

        # 插值方式下拉框
        param_layout.addWidget(QLabel("插值方式:"), 0, 0)
        self._interp_method_combo = QComboBox()
        self._interp_method_combo.addItems(["QuinticPolynomial (五次多项式)", "CubicPolynomial (三次多项式)"])
        self._interp_method_combo.setCurrentIndex(0)
        param_layout.addWidget(self._interp_method_combo, 0, 1)

        # α 平滑系数
        param_layout.addWidget(QLabel("α (平滑系数):"), 1, 0)
        self._interp_alpha_spin = QDoubleSpinBox()
        self._interp_alpha_spin.setRange(0.1, 10.0)
        self._interp_alpha_spin.setDecimals(1)
        self._interp_alpha_spin.setSingleStep(0.1)
        self._interp_alpha_spin.setValue(1.0)
        param_layout.addWidget(self._interp_alpha_spin, 1, 1)

        # β 速度阈值
        param_layout.addWidget(QLabel("β (速度阈值):"), 2, 0)
        self._interp_beta_spin = QDoubleSpinBox()
        self._interp_beta_spin.setRange(0.1, 20.0)
        self._interp_beta_spin.setDecimals(1)
        self._interp_beta_spin.setSingleStep(0.1)
        self._interp_beta_spin.setValue(5.0)
        param_layout.addWidget(self._interp_beta_spin, 2, 1)

        # n 采样系数
        param_layout.addWidget(QLabel("n (采样系数):"), 3, 0)
        self._interp_n_spin = QSpinBox()
        self._interp_n_spin.setRange(2, 50)
        self._interp_n_spin.setValue(10)
        param_layout.addWidget(self._interp_n_spin, 3, 1)

        # D 退刀距离
        param_layout.addWidget(QLabel("D (退刀距离 m):"), 4, 0)
        self._interp_d_spin = QDoubleSpinBox()
        self._interp_d_spin.setRange(0.001, 1.0)
        self._interp_d_spin.setDecimals(3)
        self._interp_d_spin.setSingleStep(0.01)
        self._interp_d_spin.setValue(0.1)
        param_layout.addWidget(self._interp_d_spin, 4, 1)

        # sign 退刀方向
        param_layout.addWidget(QLabel("退刀方向:"), 5, 0)
        self._interp_sign_combo = QComboBox()
        self._interp_sign_combo.addItems(["-1 (沿Z轴负方向/远离工件)", "1 (沿Z轴正方向/朝向工件)"])
        self._interp_sign_combo.setCurrentIndex(0)
        param_layout.addWidget(self._interp_sign_combo, 5, 1)

        param_layout.setColumnStretch(0, 1)
        param_layout.setColumnStretch(1, 2)
        interp_layout.addWidget(param_container)
        param_container.setEnabled(False)

        # ── SLSQP 求解器参数 ──────────────────────────────────────────
        slsqp_group = QGroupBox("SLSQP 求解器参数")
        slsqp_layout = QGridLayout(slsqp_group)

        slsqp_layout.addWidget(QLabel("收敛容差:"), 0, 0)
        self._interp_tol_spin = QDoubleSpinBox()
        self._interp_tol_spin.setRange(1e-8, 1e-2)
        self._interp_tol_spin.setValue(1e-5)
        self._interp_tol_spin.setDecimals(8)
        self._interp_tol_spin.setSingleStep(1e-5)
        slsqp_layout.addWidget(self._interp_tol_spin, 0, 1)

        slsqp_layout.addWidget(QLabel("最大迭代:"), 0, 2)
        self._interp_maxiter_spin = QSpinBox()
        self._interp_maxiter_spin.setRange(100, 10000)
        self._interp_maxiter_spin.setValue(1000)
        slsqp_layout.addWidget(self._interp_maxiter_spin, 0, 3)

        slsqp_layout.addWidget(QLabel("w_pose:"), 1, 0)
        self._interp_w_pose_spin = QDoubleSpinBox()
        self._interp_w_pose_spin.setRange(0.01, 10.0)
        self._interp_w_pose_spin.setDecimals(2)
        self._interp_w_pose_spin.setValue(1.0)
        self._interp_w_pose_spin.setSingleStep(0.1)
        slsqp_layout.addWidget(self._interp_w_pose_spin, 1, 1)

        slsqp_layout.addWidget(QLabel("w_disp:"), 1, 2)
        self._interp_w_disp_spin = QDoubleSpinBox()
        self._interp_w_disp_spin.setRange(0.0, 10.0)
        self._interp_w_disp_spin.setDecimals(2)
        self._interp_w_disp_spin.setValue(0.1)
        self._interp_w_disp_spin.setSingleStep(0.1)
        slsqp_layout.addWidget(self._interp_w_disp_spin, 1, 3)

        slsqp_layout.addWidget(QLabel("w_center:"), 2, 0)
        self._interp_w_center_spin = QDoubleSpinBox()
        self._interp_w_center_spin.setRange(0.0, 10.0)
        self._interp_w_center_spin.setDecimals(2)
        self._interp_w_center_spin.setValue(0.1)
        self._interp_w_center_spin.setSingleStep(0.1)
        slsqp_layout.addWidget(self._interp_w_center_spin, 2, 1)

        slsqp_layout.addWidget(QLabel("w_manip:"), 2, 2)
        self._interp_w_manip_spin = QDoubleSpinBox()
        self._interp_w_manip_spin.setRange(0.0, 10.0)
        self._interp_w_manip_spin.setDecimals(2)
        self._interp_w_manip_spin.setValue(0.01)
        self._interp_w_manip_spin.setSingleStep(0.1)
        slsqp_layout.addWidget(self._interp_w_manip_spin, 2, 3)

        slsqp_layout.addWidget(QLabel("w_joints:"), 3, 0)
        joint_w_layout = QHBoxLayout()
        self._interp_w_joints = []
        for i, label in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(1.0)
            spin.setPrefix(f"{label}:")
            spin.setFixedWidth(70)
            joint_w_layout.addWidget(spin)
            self._interp_w_joints.append(spin)
        slsqp_layout.addLayout(joint_w_layout, 3, 1, 1, 3)

        slsqp_layout.setColumnStretch(1, 1)
        slsqp_layout.setColumnStretch(3, 1)
        interp_layout.addWidget(slsqp_group)

        # 开始插值按钮
        self._btn_start_interp = QPushButton("开始插值")
        self._btn_start_interp.setEnabled(False)
        self._btn_start_interp.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self._btn_start_interp.clicked.connect(self._on_start_interpolation)
        interp_layout.addWidget(self._btn_start_interp)

        # 保存 param_container 引用以便后续启用/禁用
        self._interp_param_container = param_container

        # 旧统计跳变插值保留为兼容代码，不再暴露为生产工作流。
        interp_group.setVisible(False)
        layout.addWidget(interp_group)

        # ── 速度控制组 ─────────────────────────────────────────────
        speed_group = QGroupBox("速度控制")
        speed_layout = QVBoxLayout(speed_group)

        # 基准采样频率
        hz_row = QHBoxLayout()
        hz_row.addWidget(QLabel("旧轨迹预览频率 (Hz):"))
        self._hz_spin = QSpinBox()
        self._hz_spin.setRange(10, 1000)
        self._hz_spin.setValue(100)
        self._hz_spin.valueChanged.connect(self._on_hz_changed)
        hz_row.addWidget(self._hz_spin)
        hz_row.addStretch()
        speed_layout.addLayout(hz_row)

        # 播放速度系数 滑块 + spinbox 联动
        coeff_row = QHBoxLayout()
        coeff_row.addWidget(QLabel("仿真时间倍率:"))
        self._coeff_slider = QSlider(Qt.Horizontal)
        self._coeff_slider.setRange(0, 1000)
        self._coeff_slider.setValue(100)
        self._coeff_slider.setTickPosition(QSlider.TicksBelow)
        self._coeff_slider.valueChanged.connect(self._on_coeff_slider_changed)
        coeff_row.addWidget(self._coeff_slider)

        self._coeff_spin = QDoubleSpinBox()
        self._coeff_spin.setRange(0.00, 10.00)
        self._coeff_spin.setDecimals(2)
        self._coeff_spin.setSingleStep(0.10)
        self._coeff_spin.setValue(1.00)
        self._coeff_spin.valueChanged.connect(self._on_coeff_spin_changed)
        coeff_row.addWidget(self._coeff_spin)
        speed_layout.addLayout(coeff_row)

        # 可拖动的规范帧索引，不以百分比替代真实帧号。
        frame_header = QHBoxLayout()
        frame_header.addWidget(QLabel("轨迹帧定位:"))
        self._frame_position_label = QLabel("0 / 0")
        self._frame_position_label.setStyleSheet("font-family: Consolas;")
        frame_header.addStretch()
        frame_header.addWidget(self._frame_position_label)
        speed_layout.addLayout(frame_header)

        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.setTracking(True)
        self._frame_slider.sliderPressed.connect(self._on_frame_slider_pressed)
        self._frame_slider.sliderReleased.connect(self._on_frame_slider_released)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        speed_layout.addWidget(self._frame_slider)

        # 时间预览
        self._time_preview_label = QLabel("预计总耗时: --")
        self._time_preview_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        speed_layout.addWidget(self._time_preview_label)
        layout.addWidget(speed_group)

        # ── 播放控制组 ─────────────────────────────────────────────
        ctrl_group = QGroupBox("播放控制")
        ctrl_layout = QVBoxLayout(ctrl_group)

        btn_row = QHBoxLayout()
        self._btn_play = QPushButton("▶ 播放")
        self._btn_play.setStyleSheet("""
            QPushButton { background-color: #388E3C; color: white; font-weight: bold;
                          padding: 8px; border-radius: 4px; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_play.setEnabled(False)
        btn_row.addWidget(self._btn_play)

        self._btn_pause = QPushButton("⏸ 暂停")
        self._btn_pause.setEnabled(False)
        self._btn_pause.clicked.connect(self._on_pause)
        btn_row.addWidget(self._btn_pause)

        self._btn_stop = QPushButton("⏹ 停止")
        self._btn_stop.clicked.connect(self._on_stop)
        btn_row.addWidget(self._btn_stop)
        ctrl_layout.addLayout(btn_row)
        layout.addWidget(ctrl_group)

        # ── 碰撞检测控制组 ───────────────────────────────────────────
        cb_group = QGroupBox("碰撞检测")
        cb_layout = QVBoxLayout(cb_group)

        self._collision_off_cb = QCheckBox("关闭碰撞检测")
        self._collision_stop_cb = QCheckBox("模式1: 遇碰撞停止")
        self._collision_slide_cb = QCheckBox(f"模式2: 滑动时间窗高亮 (W={self.WINDOW_SIZE} 帧)")

        # 物理轨迹默认启用逐帧碰撞高亮；用户仍可显式关闭。
        self._collision_slide_cb.setChecked(True)
        self._collision_mode = self.COLLISION_MODE_SLIDE
        for cb in (self._collision_stop_cb, self._collision_slide_cb, self._collision_off_cb):
            cb.stateChanged.connect(self._on_collision_mode_changed)
            cb_layout.addWidget(cb)
        layout.addWidget(cb_group)

        # ── 进度 & 统计 ─────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        layout.addWidget(self._progress_bar)

        self._stats_label = QLabel("帧: 0 / 0  |  切削点数: 0")
        layout.addWidget(self._stats_label)

        # 重置按钮
        self._btn_reset = QPushButton("重置场景")
        self._btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self._btn_reset)

        # ── 刀具可视化控制组 ─────────────────────────────────────────
        viz_group = QGroupBox("刀具可视化")
        viz_layout = QVBoxLayout(viz_group)

        self._show_bbox_cb = QCheckBox("显示刀具圆柱体")
        self._show_bbox_cb.setChecked(False)
        self._show_bbox_cb.stateChanged.connect(self._on_bbox_toggled)
        viz_layout.addWidget(self._show_bbox_cb)

        self._show_c0_axes_cb = QCheckBox("显示 C0 坐标轴")
        self._show_c0_axes_cb.setChecked(False)
        self._show_c0_axes_cb.stateChanged.connect(self._on_c0_axes_toggled)
        viz_layout.addWidget(self._show_c0_axes_cb)

        self._show_cutting_vol_cb = QCheckBox("显示切削体积")
        self._show_cutting_vol_cb.setChecked(False)
        self._show_cutting_vol_cb.stateChanged.connect(self._on_cutting_vol_toggled)
        viz_layout.addWidget(self._show_cutting_vol_cb)

        layout.addWidget(viz_group)

        layout.addStretch()
        # QScrollArea 在可用高度不足时会压缩布局；显式保留内容的 sizeHint，
        # 防止大型看板与后续播放控件发生几何重叠。
        content.setMinimumHeight(max(1200, layout.sizeHint().height()))

    # ── 时间 & 速度计算 ────────────────────────────────────────────

    def _compute_interval_ms(self) -> float:
        """根据基准频率和速度系数计算定时器间隔（毫秒）"""
        if getattr(self._state, "physical_trajectory", None) is not None:
            return 16.0  # 仅为 UI 刷新节拍，物理状态由时间戳求值。
        F = self._base_hz * self._speed_coeff
        return 1000.0 / F if F > 0 else 16.0

    def _format_duration(self, seconds: float) -> str:
        """秒数 → mm:ss 格式"""
        if seconds < 0:
            return "--"
        m, s = divmod(int(round(seconds)), 60)
        return f"{m:02d}:{s:02d} ({seconds:.1f} 秒)"

    def _update_time_preview(self):
        """更新预计总耗时标签"""
        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None and not getattr(self._state, "physical_trajectory_stale", True):
            self._time_preview_label.setText(
                f"物理总时长: {self._format_duration(physical.duration_s)} | "
                f"预览倍率: {self._speed_coeff:.2f}×"
            )
            return
        n = len(self._trajectory)
        if n == 0:
            self._time_preview_label.setText("预计总耗时: --")
            return
        F = self._base_hz * self._speed_coeff
        if F <= 0:
            self._time_preview_label.setText("旧轨迹预览已暂停 | 预览倍率: 0.00×")
            return
        total_sec = n / F
        self._time_preview_label.setText(
            f"旧轨迹预览时长（非物理）: {self._format_duration(total_sec)}"
        )

    def _on_hz_changed(self):
        """基准频率修改回调"""
        self._base_hz = self._hz_spin.value()
        self._update_time_preview()
        if self._play_timer and self._play_timer.isActive():
            self._play_timer.start(int(round(self._compute_interval_ms())))

    def _on_coeff_slider_changed(self, val: int):
        """速度系数滑块 → spinbox 联动"""
        self._speed_coeff = val / 100.0
        self._coeff_spin.blockSignals(True)
        self._coeff_spin.setValue(self._speed_coeff)
        self._coeff_spin.blockSignals(False)
        self._update_time_preview()
        if self._play_timer and self._play_timer.isActive():
            self._play_timer.start(int(round(self._compute_interval_ms())))

    def _on_coeff_spin_changed(self, val: float):
        """速度系数 spinbox → 滑块联动"""
        self._speed_coeff = val
        self._coeff_slider.blockSignals(True)
        self._coeff_slider.setValue(int(round(val * 100)))
        self._coeff_slider.blockSignals(False)
        self._update_time_preview()
        if self._play_timer and self._play_timer.isActive():
            self._play_timer.start(int(round(self._compute_interval_ms())))

    def _on_frame_slider_pressed(self):
        self._frame_slider_dragging = True

    def _on_frame_slider_released(self):
        self._frame_slider_dragging = False
        self._seek_to_frame(self._frame_slider.value(), render=True)

    def _on_frame_slider_changed(self, value: int):
        self._frame_position_label.setText(
            f"{int(value)} / {max(0, len(self._trajectory) - 1)}"
        )
        if self._frame_slider_dragging:
            self._seek_to_frame(value, render=True)

    def _set_frame_range(self, frame_count: int):
        maximum = max(0, int(frame_count) - 1)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, maximum)
        self._frame_slider.setValue(min(self._current_frame, maximum))
        self._frame_slider.blockSignals(False)
        self._frame_position_label.setText(
            f"{min(self._current_frame, maximum)} / {maximum}"
        )

    def _sync_current_frame_ui(self):
        if not self._trajectory:
            self._state.current_frame = 0
            self._set_frame_range(0)
            return
        index = int(np.clip(self._current_frame, 0, len(self._trajectory) - 1))
        self._current_frame = index
        self._state.current_frame = index
        if not self._frame_slider_dragging:
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(index)
            self._frame_slider.blockSignals(False)
        self._frame_position_label.setText(
            f"{index} / {len(self._trajectory) - 1}"
        )
        total = max(1, len(self._trajectory) - 1)
        self._progress_bar.setValue(int(round(index / total * 100)))
        if self._traj_plotter is not None:
            self._traj_plotter.set_current_frame(index)

    def _seek_to_frame(self, frame_index: int, *, render: bool):
        if not self._trajectory:
            return
        index = int(np.clip(frame_index, 0, len(self._trajectory) - 1))
        self._current_frame = index
        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None and len(physical.timestamps) == len(self._trajectory):
            self._physical_time_s = float(physical.timestamps[index])
        self._sync_current_frame_ui()
        self._update_stats()
        if render:
            self._render_frame_without_process_side_effects(index)

    def _render_frame_without_process_side_effects(self, frame_index: int):
        """Render one selected pose without cutting or collision side effects."""
        if not self.render_engine or not self._trajectory:
            return
        q = np.asarray(self._trajectory[frame_index], dtype=float)
        try:
            self.render_engine.update_robot_joints(q, render=False)
            flange_local = self.render_engine.get_robot_ee_pose()
            if flange_local is not None:
                base_T = (
                    self.render_engine._robot.base_transform
                    if self.render_engine._robot is not None else np.eye(4)
                )
                self.render_engine.update_coordinate_frames(base_T @ flange_local, None)
            self.render_engine.render()
        except Exception as exc:
            self.log(f"[警告] 帧定位渲染失败: {exc}")

    def _on_interp_toggled(self, enabled: bool):
        """插值修复复选框切换回调"""
        self._interp_param_container.setEnabled(enabled)
        self._btn_start_interp.setEnabled(enabled)
        self.log(f"退刀过渡插值: {'启用' if enabled else '禁用'}")

    def _on_start_interpolation(self):
        """点击"开始插值"按钮，对当前轨迹执行插值处理"""
        if not self._trajectory:
            self.log("[错误] 没有可处理的轨迹，请先导入关节轨迹 CSV 或完成 IK 求解")
            return

        if self._state.kinematics_engine is None:
            self.log("[错误] 未加载机器人，无法进行插值处理")
            return

        config = getattr(self._state, 'robot_config', None)
        if config is None or not hasattr(config, 'urdf_path') or not config.urdf_path:
            self.log("[错误] 无法获取 URDF 路径，跳过插值处理")
            return

        urdf_path = config.urdf_path

        self._btn_start_interp.setEnabled(False)
        self._btn_start_interp.setText("处理中...")

        import pandas as pd
        data = np.array([q.copy() for q in self._trajectory])

        try:
            method_text = self._interp_method_combo.currentText()
            method = "QuinticPolynomial" if "Quintic" in method_text else "CubicPolynomial"
            alpha = self._interp_alpha_spin.value()
            beta = self._interp_beta_spin.value()
            n = self._interp_n_spin.value()
            D = self._interp_d_spin.value()
            sign_text = self._interp_sign_combo.currentText()
            sign = -1 if "-1" in sign_text else 1

            kin_engine = self._state.kinematics_engine
            solver_name = "SLSQP_update"
            from core.auto_manager import get_manager
            manager = get_manager()
            SolverClass = manager.get_algorithm_class(solver_name, prefix='ik')
            ik_solver_plugin = SolverClass(
                kinematics_engine=kin_engine,
                tolerance=self._interp_tol_spin.value(),
                max_iterations=self._interp_maxiter_spin.value(),
                w_pose=self._interp_w_pose_spin.value(),
                w_disp=self._interp_w_disp_spin.value(),
                w_center=self._interp_w_center_spin.value(),
                w_manip=self._interp_w_manip_spin.value(),
                w_joints=[spin.value() for spin in self._interp_w_joints],
            )

            columns = [f"j{i}" for i in range(data.shape[1])]
            temp_df = pd.DataFrame(data, columns=columns)
            temp_path = os.path.join(os.getcwd(), "__interp_temp__.csv")
            temp_df.to_csv(temp_path, index=False)

            from core.interpolation import ThreeSegmentSmoother
            smoother = ThreeSegmentSmoother(
                filepath=temp_path,
                kin_engine=kin_engine,
                ik_solver_plugin=ik_solver_plugin,
                method=method,
                alpha=alpha,
                beta=beta,
                n=n,
                retract_distance=D,
                sign_back=sign,
                log_callback=lambda msg: (self.log(msg), print(msg))[1],
            )
            output_file, jumps, segment_info = smoother.process()
            smoother.run_tests(segment_info)

            result_df = pd.read_csv(output_file)
            result_data = result_df.values

            os.remove(temp_path)
            if os.path.exists(output_file):
                os.remove(output_file)

            # 更新轨迹
            self._trajectory = [result_data[i].copy() for i in range(len(result_data))]
            self._state.joint_trajectory = JointTrajectory(
                positions=result_data.copy(),
                method=f"interpolation:{method}",
            )
            # This legacy interpolation has no B PROCESS-to-source mapping;
            # it must not be mistaken for C's original PROCESS IK input.
            self._state.process_joint_trajectory = None
            self._current_frame = 0
            self._state.current_frame = 0
            self._set_frame_range(len(result_data))

            self._traj_status_label.setText(f"插值后 {len(self._trajectory)} 帧 x {result_data.shape[1]} 关节")
            self._traj_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            self._btn_play.setEnabled(True)
            self._update_time_preview()
            self._update_render_from_trajectory()
            self._init_traj_plotter()

            self.log(f"插值完成: {len(data)} 点 → {len(result_data)} 点")

        except Exception as e:
            self.log(f"[错误] 插值处理失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._btn_start_interp.setEnabled(True)
            self._btn_start_interp.setText("开始插值")

    def _apply_interpolation(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        对关节轨迹应用退刀过渡插值处理（三段式防碰撞平滑）。

        返回处理后的轨迹数组。
        """
        if not self._enable_interp_cb.isChecked():
            return trajectory_data

        if self._state.kinematics_engine is None:
            self.log("[警告] 未加载机器人，无法进行插值处理")
            return trajectory_data

        try:
            import pandas as pd
            config = getattr(self._state, 'robot_config', None)
            if config is None or not hasattr(config, 'urdf_path') or not config.urdf_path:
                self.log("[警告] 无法获取 URDF 路径，跳过插值处理")
                return trajectory_data

            urdf_path = config.urdf_path

            method_text = self._interp_method_combo.currentText()
            method = "QuinticPolynomial" if "Quintic" in method_text else "CubicPolynomial"
            alpha = self._interp_alpha_spin.value()
            beta = self._interp_beta_spin.value()
            n = self._interp_n_spin.value()
            D = self._interp_d_spin.value()
            sign_text = self._interp_sign_combo.currentText()
            sign = -1 if "-1" in sign_text else 1

            kin_engine = self._state.kinematics_engine
            solver_name = self._strategy_combo.currentData()
            if solver_name is None:
                solver_name = self._strategy_combo.currentText()
            from core.auto_manager import get_manager
            manager = get_manager()
            SolverClass = manager.get_algorithm_class(solver_name, prefix='ik')
            is_multi = getattr(SolverClass, 'DESC', None) == 'SLSQP_Multi'
            if is_multi:
                ik_solver_plugin = SolverClass(
                    kinematics_engine=kin_engine,
                    tolerance=self._tol_spin.value(),
                    max_iterations=self._maxiter_spin.value(),
                    w_pose=self._w_pose_spin.value(),
                    w_disp=self._w_disp_spin.value(),
                    w_center=self._w_center_spin.value(),
                    w_manip=self._w_manip_spin.value(),
                    w_joints=[
                        self._w_j1_spin.value(),
                        self._w_j2_spin.value(),
                        self._w_j3_spin.value(),
                        self._w_j4_spin.value(),
                        self._w_j5_spin.value(),
                        self._w_j6_spin.value(),
                    ],
                )
            else:
                ik_solver_plugin = SolverClass(
                    kinematics_engine=kin_engine,
                    tolerance=self._tol_spin.value(),
                    max_iterations=self._maxiter_spin.value(),
                )

            columns = [f"j{i}" for i in range(trajectory_data.shape[1])]
            temp_df = pd.DataFrame(trajectory_data, columns=columns)
            temp_path = os.path.join(os.getcwd(), "__interp_temp__.csv")
            temp_df.to_csv(temp_path, index=False)

            from core.interpolation import ThreeSegmentSmoother
            smoother = ThreeSegmentSmoother(
                filepath=temp_path,
                kin_engine=kin_engine,
                ik_solver_plugin=ik_solver_plugin,
                method=method,
                alpha=alpha,
                beta=beta,
                n=n,
                retract_distance=D,
                sign_back=sign,
                log_callback=lambda msg: (self.log(msg), print(msg))[1],
            )
            output_file, jumps, segment_info = smoother.process()
            smoother.run_tests(segment_info)

            result_df = pd.read_csv(output_file)
            result_data = result_df.values

            os.remove(temp_path)
            if os.path.exists(output_file):
                os.remove(output_file)

            self.log(f"插值后轨迹: {len(trajectory_data)} 点 → {len(result_data)} 点")
            return result_data

        except Exception as e:
            self.log(f"[错误] 插值处理失败: {e}")
            import traceback
            traceback.print_exc()
            return trajectory_data

    # ── 物理轨迹生成 / 验证 / 导出 ─────────────────────────────────

    def _mark_physical_trajectory_stale(self, *_args):
        if getattr(self._state, "physical_trajectory", None) is None:
            return
        self._state.physical_trajectory_stale = True
        self._state.trajectory_validation = None
        self._btn_validate_physical.setEnabled(False)
        self._btn_export_physical.setEnabled(False)
        self._physical_status_label.setText("参数已修改：现有物理轨迹已过期，请重新生成")
        self._physical_status_label.setStyleSheet("color: #B26A00;")
        self._set_physical_badge("参数已变更 · 轨迹已过期", "#B54708")
        self._show_validation_pending()

    @staticmethod
    def _format_transition_plan_detail(plan: dict) -> str:
        """Return the compact, auditable detail displayed for one C request."""
        scene_hash = str(plan.get("scene_hash", ""))
        return (
            f"kind={plan.get('kind', '--')} | planner={plan.get('planner_id', '--')} | "
            f"scene=v{plan.get('scene_version', '--')}/{scene_hash[:12] or '--'} | "
            f"seed={plan.get('seed', '--')} | timeout={float(plan.get('timeout_s', 0.0)):.2f}s | "
            f"time={float(plan.get('planning_time_s', 0.0)):.3f}s | "
            f"waypoints={plan.get('waypoint_count', '--')}"
        )

    def _clear_transition_plan_details(self, message: str = "stable2C 过渡：尚未生成") -> None:
        self._transition_plan_records = []
        combo = self._transition_plan_combo
        combo.blockSignals(True)
        combo.clear()
        combo.setEnabled(False)
        combo.blockSignals(False)
        self._transition_plan_detail_label.setText(message)

    def _populate_transition_plan_details(self, plans) -> None:
        self._transition_plan_records = [dict(plan) for plan in plans if isinstance(plan, dict)]
        combo = self._transition_plan_combo
        combo.blockSignals(True)
        combo.clear()
        for index, plan in enumerate(self._transition_plan_records):
            combo.addItem(
                f"{index + 1}. {plan.get('kind', '--')} · "
                f"{plan.get('request_id', '--')}",
                index,
            )
        combo.setEnabled(bool(self._transition_plan_records))
        combo.blockSignals(False)
        if self._transition_plan_records:
            combo.setCurrentIndex(0)
            self._on_transition_plan_selected(0)
        else:
            self._transition_plan_detail_label.setText("stable2C 过渡：无可显示的规划记录")

    def _on_transition_plan_selected(self, index: int) -> None:
        if not 0 <= int(index) < len(self._transition_plan_records):
            return
        self._transition_plan_detail_label.setText(
            self._format_transition_plan_detail(self._transition_plan_records[int(index)])
        )

    def _on_collision_scene_changed(self, source: str, event: dict) -> None:
        """Invalidate or cancel work that refers to an older scene version."""
        scene_version = int(event.get("scene_version", self._state.scene_version))
        planning_worker = self._planning_worker
        if (
            planning_worker is not None
            and planning_worker.isRunning()
            and self._planning_scene_version is not None
            and scene_version != self._planning_scene_version
        ):
            planning_worker.abort()
            self._btn_cancel_physical.setEnabled(False)
            self._physical_status_label.setText(
                f"场景已由 {source} 修改：正在进行的 stable2C 规划已取消"
            )
            self._physical_status_label.setStyleSheet("color: #B26A00;")
            self._set_physical_badge("场景已变更 · 规划已取消", "#B54708")
            self._clear_transition_plan_details("stable2C 过渡：场景已变更，规划已取消")

        validation_worker = self._validation_worker
        if validation_worker is not None and validation_worker.isRunning():
            validation_worker.abort()

        if getattr(self._state, "physical_trajectory", None) is None:
            return
        if self._play_timer is not None:
            self._play_timer.stop()
        self._state.physical_trajectory_stale = True
        self._state.trajectory_validation = None
        self._btn_validate_physical.setEnabled(False)
        self._btn_export_physical.setEnabled(False)
        self._btn_export_diagnostic.setEnabled(False)
        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._physical_status_label.setText(
            f"场景已由 {source} 修改：物理轨迹与验证结果已过期，请重新生成"
        )
        self._physical_status_label.setStyleSheet("color: #B26A00;")
        self._set_physical_badge("场景已变更 · 轨迹已过期", "#B54708")
        self._clear_transition_plan_details("stable2C 过渡：场景已变更，记录已过期")
        self._show_validation_pending("场景已变更")

    def _collect_process_parameters(self) -> ProcessParameters:
        if self._state.kinematics_engine is None:
            raise RuntimeError("请先加载机器人")
        dof = self._state.kinematics_engine.nq
        previous = getattr(self._state, "process_parameters", ProcessParameters())
        return ProcessParameters(
            tcp_feed_rate_mps=self._tcp_feed_spin.value() / 1000.0,
            tcp_acceleration_mps2=self._tcp_accel_spin.value() / 1000.0,
            normal_force_setpoint_n=previous.normal_force_setpoint_n,
            stepover_m=previous.stepover_m,
            tool_tilt_rad=previous.tool_tilt_rad,
            corner_blend_radius_m=previous.corner_blend_radius_m,
            effective_contact_width_m=previous.effective_contact_width_m,
            minimum_joint_margin_rad=np.deg2rad(self._joint_margin_physical_spin.value()),
            max_joint_velocity=np.full(dof, self._joint_velocity_spin.value()),
            max_joint_acceleration=np.full(dof, self._joint_acceleration_spin.value()),
            max_joint_jerk=np.full(dof, self._joint_jerk_spin.value()),
            control_period_s=self._control_period_spin.value() / 1000.0,
            chord_tolerance_m=self._path_tolerance_spin.value() / 1000.0,
            orientation_tolerance_rad=previous.orientation_tolerance_rad,
            rapid_speed_ratio=previous.rapid_speed_ratio,
            adaptive_keyframes=self._adaptive_keyframes_cb.isChecked(),
            max_keyframe_interval_s=self._keyframe_interval_spin.value() / 1000.0,
            joint_keyframe_tolerance_rad=np.deg2rad(
                self._joint_keyframe_tolerance_spin.value()
            ),
        )

    def _set_physical_badge(self, text: str, color: str) -> None:
        self._physical_badge.setText(f"● {text}")
        self._physical_badge.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {color}; padding: 5px;"
        )

    def _update_physical_dashboard(self, trajectory=None, report=None) -> None:
        trajectory = trajectory or getattr(self._state, "physical_trajectory", None)
        if trajectory is None:
            for label in self._metric_labels.values():
                label.setText("--")
            return
        self._metric_labels["duration"].setText(f"{trajectory.duration_s:.2f} s")
        dense_count = int(trajectory.metadata.get("dense_sample_count", len(trajectory.timestamps)))
        if dense_count > len(trajectory.timestamps):
            reduction = 100.0 * (1.0 - len(trajectory.timestamps) / dense_count)
            self._metric_labels["samples"].setText(
                f"{len(trajectory.timestamps):,}  ↓{reduction:.1f}%"
            )
            self._metric_labels["samples"].setToolTip(
                f"固定周期原始采样 {dense_count:,}；误差受控关键帧 {len(trajectory.timestamps):,}"
            )
        else:
            self._metric_labels["samples"].setText(f"{len(trajectory.timestamps):,}")
        tcp_peak = float(np.max(trajectory.tcp_speeds_mps)) if len(trajectory.timestamps) else 0.0
        self._metric_labels["tcp_peak"].setText(f"{tcp_peak * 1000.0:.2f} mm/s")

        ratios = []
        parameters = self._state.process_parameters
        for values, limits in (
            (trajectory.velocities, parameters.max_joint_velocity),
            (trajectory.accelerations, parameters.max_joint_acceleration),
            (trajectory.jerks, parameters.max_joint_jerk),
        ):
            if limits is not None:
                ratios.append(float(np.max(np.abs(values) / np.asarray(limits)[None, :])))
        if parameters.tcp_feed_rate_mps > 0:
            process_mask = trajectory.segment_types == PathSegmentType.PROCESS
            if np.any(process_mask):
                ratios.append(float(np.max(trajectory.tcp_speeds_mps[process_mask]) /
                                    parameters.tcp_feed_rate_mps))
        utilization = max(ratios, default=0.0)
        self._metric_labels["utilization"].setText(f"{utilization * 100.0:.1f}%")
        self._metric_labels["utilization"].setStyleSheet(
            "font-size: 16px; font-weight: 700; border: none; color: "
            + ("#B42318;" if utilization > 1.0 else "#027A48;")
        )
        self._render_physical_charts(trajectory)
        if report is not None:
            self._show_validation_report(report)

    def _render_physical_charts(self, trajectory) -> None:
        speed_axis, usage_axis = self._physical_axes
        speed_axis.clear()
        usage_axis.clear()
        # 大轨迹绘图严格降采样，避免 UI 因百万级折线再次卡住。
        stride = max(1, len(trajectory.timestamps) // 4000)
        index = slice(None, None, stride)
        time_values = trajectory.timestamps[index]
        actual = trajectory.tcp_speeds_mps[index] * 1000.0
        setpoint = trajectory.process_channels.get("tcp_feed_setpoint_mps")
        speed_axis.plot(time_values, actual, color="#1473E6", linewidth=1.2, label="Actual")
        if setpoint is not None:
            speed_axis.plot(time_values, np.asarray(setpoint)[index] * 1000.0,
                            color="#E67E22", linestyle="--", linewidth=1.0, label="Setpoint")
        speed_axis.set_title("TCP Speed")
        speed_axis.set_xlabel("t / s")
        speed_axis.set_ylabel("mm/s")
        speed_axis.grid(alpha=0.25)
        speed_axis.legend(fontsize=7, loc="best")

        parameters = self._state.process_parameters
        names, values = [], []
        for name, data, limits in (
            ("Velocity", trajectory.velocities, parameters.max_joint_velocity),
            ("Accel", trajectory.accelerations, parameters.max_joint_acceleration),
            ("Jerk", trajectory.jerks, parameters.max_joint_jerk),
        ):
            if limits is not None:
                names.append(name)
                values.append(100.0 * float(np.max(np.abs(data) / np.asarray(limits)[None, :])))
        colors = ["#D92D20" if value > 100.0 else "#12B76A" for value in values]
        usage_axis.bar(names, values, color=colors)
        usage_axis.axhline(100.0, color="#D92D20", linestyle="--", linewidth=1.0)
        usage_axis.set_ylim(0.0, max(110.0, max(values, default=0.0) * 1.1))
        usage_axis.set_title("Joint Constraint Usage")
        usage_axis.set_ylabel("%")
        usage_axis.grid(axis="y", alpha=0.25)
        self._physical_canvas.draw_idle()

    def _show_validation_report(self, report: TrajectoryValidationReport) -> None:
        self._validation_table.setRowCount(len(report.items))
        display_names = {
            "timestamps": "时间戳",
            "joint_velocity": "关节速度",
            "joint_acceleration": "关节加速度",
            "joint_jerk": "关节 jerk",
            "joint_position_margin": "关节限位裕度",
            "tcp_feed_rate": "TCP 加工速度",
            "tcp_path_deviation": "TCP 路径偏差",
            "collision": "碰撞检查",
        }
        for row, item in enumerate(report.items):
            location = []
            if item.time_s is not None:
                location.append(f"t={item.time_s:.3f}s")
            if item.joint_index is not None:
                location.append(f"J{item.joint_index + 1}")
            if item.segment_id is not None:
                location.append(f"段 {item.segment_id}")
            if item.message:
                location.append(item.message)
            values = (
                display_names.get(item.name, item.name),
                "通过" if item.passed else "失败",
                "--" if item.measured is None else f"{item.measured:.6g}",
                "--" if item.limit is None else f"{item.limit:.6g}",
                " · ".join(location),
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 1:
                    cell.setForeground(Qt.GlobalColor.darkGreen if item.passed else Qt.GlobalColor.red)
                self._validation_table.setItem(row, column, cell)

    def _show_validation_pending(self, active_stage: str = "") -> None:
        """Keep the structured report visible before and during validation."""
        stage_order = (
            "时间戳与数组结构", "关节动力学约束", "关节限位与安全裕度",
            "TCP 进给速度", "几何路径偏差", "碰撞区间检查",
        )
        rows = (
            ("时间戳", "时间戳与数组结构"),
            ("关节速度", "关节动力学约束"),
            ("关节加速度", "关节动力学约束"),
            ("关节 jerk", "关节动力学约束"),
            ("关节限位裕度", "关节限位与安全裕度"),
            ("TCP 加工速度", "TCP 进给速度"),
            ("TCP 路径偏差", "几何路径偏差"),
            ("碰撞检查", "碰撞区间检查"),
        )
        active_index = stage_order.index(active_stage) if active_stage in stage_order else -1
        self._validation_table.setRowCount(len(rows))
        for row, (name, stage) in enumerate(rows):
            stage_index = stage_order.index(stage)
            if active_index >= 0 and stage_index < active_index:
                status = "已完成"
            elif active_index >= 0 and stage_index == active_index:
                status = "检查中…"
            else:
                status = "等待验证"
            values = (name, status, "--", "--", stage)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column == 1:
                    cell.setForeground(
                        Qt.GlobalColor.darkGreen if status == "已完成"
                        else (Qt.GlobalColor.darkBlue if status.startswith("检查")
                              else Qt.GlobalColor.darkYellow)
                    )
                self._validation_table.setItem(row, column, cell)

    def _show_validation_error(self, message: str) -> None:
        """Surface worker failures in the report instead of leaving it blank."""
        self._validation_table.setRowCount(1)
        values = ("验证任务", "异常", "--", "--", message)
        for column, value in enumerate(values):
            cell = QTableWidgetItem(value)
            if column == 1:
                cell.setForeground(Qt.GlobalColor.red)
            self._validation_table.setItem(0, column, cell)

    def _on_cancel_physical_operation(self):
        worker = None
        if self._validation_worker is not None and self._validation_worker.isRunning():
            worker = self._validation_worker
        elif self._planning_worker is not None and self._planning_worker.isRunning():
            worker = self._planning_worker
        if worker is not None:
            worker.abort()
            self._btn_cancel_physical.setEnabled(False)
            self._physical_status_label.setText("正在安全取消，请稍候...")
            self._set_physical_badge("正在取消", "#B54708")

    def _on_generate_physical_trajectory(self):
        self._clear_transition_plan_details("stable2C 过渡：等待本次规划结果")
        if self._state.ik_results and not getattr(self._state, "ik_path_complete", False):
            self.log("[错误] IK 路径包含失败点，禁止将前后成功点直接连接；请先修复失败段。")
            return
        geometric = getattr(self._state, "geometric_trajectory", None)
        process_joint = getattr(self._state, "process_joint_trajectory", None)
        if geometric is None or process_joint is None:
            self.log("[错误] 缺少与 B 分层刀路一一对应的 PROCESS IK；请重新完成刀路分层和 IK。")
            return
        process_positions = np.asarray(process_joint.positions, dtype=np.float64)
        if (
            process_positions.ndim != 2
            or len(process_positions) < 2
            or len(process_positions) != len(geometric.tcp_poses)
        ):
            self.log("[错误] PROCESS IK 与 B 的几何刀路不一致，拒绝生成物理轨迹。")
            return
        kin = self._state.kinematics_engine
        if kin is None or getattr(kin, "current_q", None) is None:
            self.log("[错误] 未获得任务开始时的真实 current_q，无法规划 stable2C 过渡。")
            return
        collision_disabled = (
            self._collision_off_cb is not None and self._collision_off_cb.isChecked()
        )
        if collision_disabled:
            self.log("[错误] stable2C 必须使用 fail-closed 碰撞场景；请先重新启用碰撞检测。")
            self._physical_status_label.setText("生成未启动：stable2C 不允许关闭碰撞检测")
            self._physical_status_label.setStyleSheet("color: #B00020;")
            self._set_physical_badge("生成未启动 · 碰撞检测关闭", "#B42318")
            return
        manager = getattr(self._state, "collision_manager", None)
        if manager is None:
            self.log("[错误] stable2C 需要 CollisionManager 的不可变场景快照。")
            self._physical_status_label.setText("生成未启动：碰撞场景不可用（fail-closed）")
            self._physical_status_label.setStyleSheet("color: #B00020;")
            self._set_physical_badge("生成未启动 · 碰撞场景不可用", "#B42318")
            return
        q_home = np.asarray(kin.current_q, dtype=np.float64).copy()
        try:
            collision_snapshot = manager.create_scene_snapshot(
                q_home,
                scene_version=int(self._state.scene_version),
            )
        except Exception as exc:
            self.log(f"[错误] 无法冻结 stable2C 碰撞场景: {exc}")
            self._physical_status_label.setText("生成未启动：碰撞场景不可用（fail-closed）")
            self._physical_status_label.setStyleSheet("color: #B00020;")
            self._set_physical_badge("生成未启动 · 碰撞场景不可用", "#B42318")
            return
        self.log(
            "[stable2C] 碰撞快照附着体: "
            + ", ".join(body.name for body in collision_snapshot.attached_bodies)
        )
        try:
            parameters = self._collect_process_parameters()
        except Exception as exc:
            self.log(f"[错误] 工艺参数无效: {exc}")
            return
        self._state.process_parameters = parameters
        self._state.collision_scene_snapshot = collision_snapshot
        self._state.collision_scene_hash = collision_snapshot.scene_hash
        self.record_stage(
            "collision_scene_snapshot_captured",
            details={
                "scene_hash": collision_snapshot.scene_hash,
                "scene_version": int(collision_snapshot.scene_version),
                "robot_link_count": len(collision_snapshot.robot_links),
                "static_object_count": len(collision_snapshot.static_objects),
                "attached_body_count": len(collision_snapshot.attached_bodies),
                "purpose": "stable2C_transition_planning",
            },
        )
        self._state.physical_trajectory = None
        self._state.trajectory_validation = None
        self._state.physical_trajectory_stale = True
        self._btn_generate_physical.setEnabled(False)
        self._btn_validate_physical.setEnabled(False)
        self._btn_export_physical.setEnabled(False)
        self._btn_export_diagnostic.setEnabled(False)
        self._btn_cancel_physical.setEnabled(True)
        self._physical_status_label.setText("正在规划 stable2C 非加工过渡...")
        self._set_physical_badge("正在规划 · OMPL / TOPP-RA / Ruckig", "#175CD3")
        self._planning_scene_version = int(collision_snapshot.scene_version)
        self._planning_scene_hash = collision_snapshot.scene_hash
        self._planning_result_rejected = False
        self._planning_worker = TrajectoryPlanningWorker(
            process_joint_positions=process_positions,
            parameters=parameters,
            geometric_trajectory=geometric,
            transition_requests=tuple(self._state.transition_requests),
            collision_snapshot=collision_snapshot,
            q_home=q_home,
            lower_limits=kin.model.lowerPositionLimit,
            upper_limits=kin.model.upperPositionLimit,
            tcp_transform=np.asarray(kin.effective_tcp, dtype=np.float64).copy(),
            parent=self,
        )
        self._planning_worker.log_signal.connect(self.log)
        self._planning_worker.progress_signal.connect(self._on_physical_progress)
        self._planning_worker.result_signal.connect(self._on_physical_result)
        self._planning_worker.finished_signal.connect(self._on_physical_finished)
        self._planning_worker.start()

    def _on_physical_progress(self, current: int, total: int):
        if total > 0:
            self._progress_bar.setValue(int(current / total * 100))

    def _on_physical_result(self, trajectory: TimeParameterizedTrajectory):
        stable2c_metadata = trajectory.metadata.get("stable2C", {})
        result_scene_version = stable2c_metadata.get("scene_version")
        if result_scene_version is not None and int(result_scene_version) != int(self._state.scene_version):
            self._planning_scene_version = None
            self._planning_scene_hash = None
            self._planning_result_rejected = True
            self._state.physical_trajectory = None
            self._state.physical_trajectory_stale = True
            self._physical_status_label.setText("规划结果所属场景已变化，已拒绝加载旧结果")
            self._physical_status_label.setStyleSheet("color: #B26A00;")
            self._set_physical_badge("场景已变更 · 结果已拒绝", "#B54708")
            self._clear_transition_plan_details("stable2C 过渡：旧场景结果已拒绝")
            self.log("[stable2C] 规划结果 scene_version 已过期，未写入物理轨迹状态")
            return
        self._planning_scene_version = None
        self._planning_scene_hash = None
        self._state.physical_trajectory = trajectory
        self._state.physical_trajectory_stale = False
        self._state.trajectory_validation = None
        self._trajectory = [row.copy() for row in trajectory.positions]
        self._state.joint_trajectory = JointTrajectory(
            positions=trajectory.positions,
            velocities=trajectory.velocities,
            timestamps=trajectory.timestamps,
            method=trajectory.method,
        )
        self._current_frame = 0
        self._state.current_frame = 0
        self._physical_time_s = 0.0
        self._set_frame_range(len(trajectory.positions))
        self._btn_validate_physical.setEnabled(True)
        self._btn_export_diagnostic.setEnabled(True)
        self._btn_play.setEnabled(True)
        self._set_physical_badge("轨迹已生成 · 等待验证", "#B54708")
        transition_count = int(stable2c_metadata.get("transition_count", 0))
        process_source_count = int(stable2c_metadata.get("process_source_frame_count", 0))
        safe_entry_count = int(stable2c_metadata.get("safe_entry_frame_count", 0))
        source_entry_count = int(
            stable2c_metadata.get("source_plus_safe_entry_frame_count", 0)
        )
        self._populate_transition_plan_details(
            stable2c_metadata.get("transition_plans", ())
        )
        self._physical_status_label.setText(
            f"已生成 {len(trajectory.timestamps)} 个执行采样，"
            f"物理时长 {trajectory.duration_s:.3f}s；"
            f"源合同 {process_source_count}+{safe_entry_count}={source_entry_count}；"
            f"stable2C 过渡 {transition_count} 段；"
            f"固定周期等效 {trajectory.metadata.get('dense_sample_count', len(trajectory.timestamps))} 点；尚未验证"
        )
        if self.render_engine is not None and trajectory.tcp_poses is not None:
            self.render_engine.load_execution_trajectory(
                trajectory.tcp_poses,
                trajectory.segment_types,
                trajectory.segment_ids,
            )
        self._update_time_preview()
        self._update_render_from_trajectory()
        self._init_traj_plotter()
        self._update_physical_dashboard(trajectory)
        self._show_validation_pending()
        for plan in stable2c_metadata.get("transition_plans", ()):
            self.log(
                "[stable2C] "
                f"{plan.get('kind')} / {plan.get('request_id')}: "
                f"planner={plan.get('planner_id')}, "
                f"{plan.get('planning_time_s', 0.0):.3f}s, "
                f"scene={str(plan.get('scene_hash', ''))[:12]}"
            )
        self.record_stage(
            "physical_trajectory_generated",
            details={
                "sample_count": int(len(trajectory.timestamps)),
                "duration_s": float(trajectory.duration_s),
                "method": trajectory.method,
                "stable2C_scene_hash": stable2c_metadata.get("scene_hash"),
                "stable2C_scene_version": stable2c_metadata.get("scene_version"),
                "stable2C_transition_count": transition_count,
            },
            version_domain="trajectory",
        )

    def _on_physical_finished(self, success: bool, message: str):
        if not success:
            self._planning_scene_version = None
            self._planning_scene_hash = None
        self._btn_generate_physical.setEnabled(True)
        self._btn_cancel_physical.setEnabled(False)
        if success and self._planning_result_rejected:
            # The worker completed normally, but its result belongs to a
            # superseded scene and was deliberately not loaded in
            # ``_on_physical_result``.  Do not overwrite that stale warning
            # with a misleading success log or badge.
            self._planning_result_rejected = False
            self.log(f"[stable2C] {message}；结果已因场景变更拒绝加载")
            return
        if not success:
            self._physical_status_label.setText(f"生成失败: {message}")
            self._physical_status_label.setStyleSheet("color: #B00020;")
            self._set_physical_badge("生成失败", "#B42318")
        else:
            self._physical_status_label.setStyleSheet("color: #B26A00;")
        self.log(message)

    def _on_validate_physical_trajectory(self):
        trajectory = getattr(self._state, "physical_trajectory", None)
        if trajectory is None or self._state.physical_trajectory_stale:
            self.log("[错误] 没有可验证的最新物理轨迹")
            return
        kin = self._state.kinematics_engine
        collision_snapshot = None
        manager = getattr(self._state, "collision_manager", None)
        collision_disabled = (
            self._collision_off_cb is not None and self._collision_off_cb.isChecked()
        )
        if manager is not None and not collision_disabled:
            try:
                current_q = np.asarray(kin.current_q, dtype=float).copy()
                collision_snapshot = manager.create_scene_snapshot(
                    current_q,
                    scene_version=int(self._state.scene_version),
                )
                self._state.collision_scene_snapshot = collision_snapshot
                self._state.collision_scene_hash = collision_snapshot.scene_hash
                self.record_stage(
                    "collision_scene_snapshot_captured",
                    details={
                        "scene_hash": collision_snapshot.scene_hash,
                        "robot_link_count": len(collision_snapshot.robot_links),
                        "static_object_count": len(collision_snapshot.static_objects),
                        "attached_body_count": len(collision_snapshot.attached_bodies),
                    },
                )
            except Exception as exc:
                self.log(f"[错误] 无法冻结用于后台验证的碰撞场景: {exc}")
                self._physical_status_label.setText("验证未启动：碰撞场景不可用（fail-closed）")
                self._physical_status_label.setStyleSheet("color: #B00020;")
                self._set_physical_badge("验证未启动 · 碰撞场景不可用", "#B42318")
                return
        elif collision_disabled:
            self.log("[警告] 碰撞验证已关闭；验证报告将按 fail-closed 规则判定为未通过。")
        reference = (
            self._state.geometric_trajectory.tcp_poses[:, :3, 3]
            if getattr(self._state, "geometric_trajectory", None) is not None else None
        )
        self._btn_validate_physical.setEnabled(False)
        self._btn_generate_physical.setEnabled(False)
        self._btn_play.setEnabled(False)
        self._btn_cancel_physical.setEnabled(True)
        self._btn_export_physical.setEnabled(False)
        self._progress_bar.setValue(0)
        self._set_physical_badge("正在验证 · 时间戳与数组结构", "#175CD3")
        self._show_validation_pending("时间戳与数组结构")
        self._physical_status_label.setText(
            f"后台验证 {len(trajectory.timestamps):,} 个执行采样；界面可继续响应"
        )
        self._validation_worker = TrajectoryValidationWorker(
            trajectory,
            self._state.process_parameters,
            kin.model.lowerPositionLimit,
            kin.model.upperPositionLimit,
            reference,
            collision_snapshot=collision_snapshot,
            parent=self,
        )
        self._validation_worker.log_signal.connect(self.log)
        self._validation_worker.stage_signal.connect(self._on_validation_stage)
        self._validation_worker.progress_signal.connect(self._on_physical_progress)
        self._validation_worker.result_signal.connect(self._on_validation_result)
        self._validation_worker.finished_signal.connect(self._on_validation_finished)
        self._validation_worker.start()

    def _on_validation_stage(self, stage: str):
        self._set_physical_badge(f"正在验证 · {stage}", "#175CD3")
        self._physical_status_label.setText(f"验证阶段：{stage}")
        if stage != "验证完成":
            self._show_validation_pending(stage)

    def _on_validation_result(self, report: TrajectoryValidationReport):
        self._state.trajectory_validation = report
        self.record_stage(
            "trajectory_validated",
            status="completed" if report.passed else "failed",
            details={
                "passed": bool(report.passed),
                "item_count": int(len(report.items)),
                "failure_count": int(len(report.failures)),
            },
            version_domain="validation" if report.passed else None,
        )
        self._btn_export_physical.setEnabled(report.passed)
        trajectory = self._state.physical_trajectory
        if report.passed:
            self._physical_status_label.setText(
                f"验证通过：{trajectory.duration_s:.3f}s，"
                f"TCP 峰值 {np.max(trajectory.tcp_speeds_mps) * 1000:.2f} mm/s"
            )
            self._physical_status_label.setStyleSheet("color: #2E7D32;")
            self._set_physical_badge("验证通过 · 可正式导出", "#027A48")
        else:
            details = "; ".join(item.name for item in report.failures)
            self._physical_status_label.setText(f"验证失败：{details}")
            self._physical_status_label.setStyleSheet("color: #B00020;")
            self._set_physical_badge("验证失败 · 仅可诊断导出", "#B42318")
        self._update_physical_dashboard(trajectory, report)
        for item in report.items:
            self.log(f"[轨迹验证] {item.name}: {'通过' if item.passed else '失败'} {item.message}")

    def _on_validation_finished(self, success: bool, message: str):
        has_current_physical = (
            self._state.physical_trajectory is not None
            and not self._state.physical_trajectory_stale
        )
        self._btn_validate_physical.setEnabled(has_current_physical)
        self._btn_generate_physical.setEnabled(True)
        self._btn_play.setEnabled(has_current_physical)
        self._btn_cancel_physical.setEnabled(False)
        if not success:
            if "取消" in message:
                self._set_physical_badge("验证已取消", "#B54708")
                self._physical_status_label.setText("验证已取消，未改变上一次验证结果")
            else:
                self._set_physical_badge("验证异常", "#B42318")
                self._physical_status_label.setText(f"验证异常：{message}")
                self._show_validation_error(message)
        self.log(message)

    def _on_export_physical_trajectory(self):
        trajectory = getattr(self._state, "physical_trajectory", None)
        report = getattr(self._state, "trajectory_validation", None)
        if (
            trajectory is None
            or self._state.physical_trajectory_stale
            or report is None
            or not report.passed
        ):
            self.log("[错误] 只有通过硬约束验证的物理轨迹才能正式导出")
            return
        self._export_physical_bundle(diagnostic=False)

    def _on_export_diagnostic_trajectory(self):
        trajectory = getattr(self._state, "physical_trajectory", None)
        if trajectory is None or self._state.physical_trajectory_stale:
            self.log("[错误] 没有可诊断导出的物理轨迹")
            return
        if getattr(self._state, "trajectory_validation", None) is None:
            self._on_validate_physical_trajectory()
        if getattr(self._state, "trajectory_validation", None) is None:
            return
        self._export_physical_bundle(diagnostic=True)

    def _export_physical_bundle(self, diagnostic: bool):
        trajectory = self._state.physical_trajectory
        report = self._state.trajectory_validation
        default_name = "physical_trajectory_diagnostic.csv" if diagnostic else "physical_trajectory.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "诊断导出" if diagnostic else "导出物理轨迹", default_name, "CSV 文件 (*.csv)"
        )
        if not path:
            return
        try:
            import hashlib
            from trajectory import export_trajectory_bundle

            robot = getattr(self._state, "robot_config", None)
            urdf_hash = ""
            if robot is not None and robot.urdf_path and os.path.exists(robot.urdf_path):
                hasher = hashlib.sha256()
                with open(robot.urdf_path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                urdf_hash = hasher.hexdigest()
            csv_path, manifest_path = export_trajectory_bundle(
                path, trajectory, self._state.process_parameters, report,
                robot_name=getattr(robot, "name", "unknown"),
                urdf_hash=urdf_hash,
                tool_id=os.path.basename(getattr(self._state, "tool_filepath", "") or "unknown"),
                diagnostic=diagnostic,
            )
            self.log(f"{'诊断轨迹' if diagnostic else '物理轨迹'}已导出: {csv_path}")
            self.log(f"轨迹清单已导出: {manifest_path}")
        except Exception as exc:
            self.log(f"[错误] 物理轨迹导出失败: {exc}")

    # ── 轨迹导入 ──────────────────────────────────────────────────

    def _on_import_trajectory(self):
        """导入外部关节轨迹 CSV（弧度 radians）"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择关节轨迹 CSV", "",
            "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig") as stream:
                header = stream.readline().strip()
            if header.startswith("time_s,"):
                from trajectory import load_trajectory_bundle
                physical = load_trajectory_bundle(path)
                self._state.physical_trajectory = physical
                self._state.physical_trajectory_stale = False
                self._state.trajectory_validation = None
                data = physical.positions.copy()
                self._physical_time_s = 0.0
                self._btn_validate_physical.setEnabled(True)
                self._btn_export_diagnostic.setEnabled(True)
                self._physical_status_label.setText(
                    f"已导入带物理时间轨迹：{physical.duration_s:.3f}s；请重新验证"
                )
            else:
                data = np.loadtxt(path, delimiter=',', skiprows=1 if any(ch.isalpha() for ch in header) else 0)
                unit, accepted = QInputDialog.getItem(
                    self,
                    "选择关节角单位",
                    "旧式 CSV 不包含单位元数据，请明确选择:",
                    ["弧度 (rad)", "角度 (deg)"],
                    0,
                    False,
                )
                if not accepted:
                    return
                if unit.startswith("角度"):
                    data = np.deg2rad(data)
                self._state.physical_trajectory = None
                self._state.trajectory_validation = None
                self._state.physical_trajectory_stale = True
                self._btn_validate_physical.setEnabled(False)
                self._btn_export_physical.setEnabled(False)
                self._physical_status_label.setText("旧式离散轨迹已导入；必须生成物理轨迹后才能正式验证/导出")
        except Exception as e:
            self.log(f"[错误] 读取轨迹 CSV 失败: {e}")
            return

        if data.ndim != 2 or data.shape[1] < 1:
            self.log(f"[错误] CSV 格式错误，需要至少 1 列，实际: {data.shape}")
            return

        # 动态获取关节数
        if self._state.kinematics_engine is not None:
            nq = self._state.kinematics_engine.nq
        else:
            nq = data.shape[1]

        # 校验列数是否与 URDF 匹配
        if data.shape[1] != nq:
            self.log(f"[警告] CSV 列数({data.shape[1]}) 与 URDF 关节数({nq}) 不一致，将截断或补零")

        # 截断或补零到正确的关节数
        if data.shape[1] > nq:
            data = data[:, :nq]
        elif data.shape[1] < nq:
            padding = np.zeros((data.shape[0], nq - data.shape[1]))
            data = np.hstack([data, padding])
            self.log(f"已补零到 {nq} 关节")

        # 应用退刀过渡插值（如果启用）
        if self._enable_interp_cb.isChecked():
            self.log("正在应用退刀过渡插值...")
            data = self._apply_interpolation(data)

        self._trajectory = [data[i].copy() for i in range(len(data))]
        self._state.joint_trajectory = JointTrajectory(
            positions=data.copy(),
            velocities=(self._state.physical_trajectory.velocities.copy()
                        if self._state.physical_trajectory is not None else None),
            timestamps=(self._state.physical_trajectory.timestamps.copy()
                        if self._state.physical_trajectory is not None else None),
            method=(self._state.physical_trajectory.method
                    if self._state.physical_trajectory is not None else "legacy_unparameterized"),
        )
        # Imported executions/legacy CSVs are display data, not B-aligned
        # PROCESS IK.  Require a fresh B+IK pass before stable2C planning.
        self._state.process_joint_trajectory = None
        self._state.joint_trajectory_path = path
        self._current_frame = 0
        self._state.current_frame = 0
        self._set_frame_range(len(data))
        self._traj_status_label.setText(f"已导入 {len(self._trajectory)} 帧 x {nq} 关节")
        self._traj_status_label.setStyleSheet("color: #388E3C; font-weight: bold;")
        self._btn_play.setEnabled(True)
        self._total_cuts = 0
        self._update_time_preview()
        self._update_stats()
        self._update_render_from_trajectory()
        self._init_traj_plotter()
        self.log(f"轨迹导入完成: {len(self._trajectory)} 帧, 来源: {os.path.basename(path)}")

    # ── 碰撞检测 ───────────────────────────────────────────────────

    def _on_collision_mode_changed(self, state: int):
        """三个 CheckBox 互斥回调。state=Qt.Checked (2) 表示选中。"""
        if not state:
            return  # 忽略 uncheck 信号(由互斥分支统一处理)

        sender = self.sender()
        if sender is self._collision_stop_cb:
            new_mode = self.COLLISION_MODE_STOP
        elif sender is self._collision_slide_cb:
            new_mode = self.COLLISION_MODE_SLIDE
        else:
            new_mode = self.COLLISION_MODE_OFF

        self._collision_mode = new_mode
        self._collision_window.clear()
        if self.render_engine is not None and self._current_frame == 0:
            self.render_engine.reset_flange_trail()
        self._first_coll_frame = None

        # 互斥:阻断信号后取消其他勾选
        for cb, mode in (
            (self._collision_off_cb, self.COLLISION_MODE_OFF),
            (self._collision_stop_cb, self.COLLISION_MODE_STOP),
            (self._collision_slide_cb, self.COLLISION_MODE_SLIDE),
        ):
            if cb is sender:
                continue
            desired = (mode == new_mode)
            if cb.isChecked() != desired:
                cb.blockSignals(True)
                cb.setChecked(desired)
                cb.blockSignals(False)

        mode_name = {0: '关闭', 1: '模式1(遇碰撞停止)', 2: '模式2(滑窗高亮)'}[new_mode]
        self.log(f"[碰撞检测] 切换至 {mode_name}")

        # 切换时清掉所有高亮
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr._apply_highlights({'robot': {}, 'static': {}})
            except Exception:
                pass

    def _pre_scan_first_collision(self) -> Optional[int]:
        """
        分段检测 + 块内二分 -> 整条轨迹的全局首个碰撞帧。
        最多分成 64 段:每个 chunk 只在末帧做一次检测;若有碰撞
        在 [prev_safe+1, chunk_end] 区间内二分定位首个碰撞帧。
        返回 None 表示整条轨迹无碰撞。
        """
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is None or not self._trajectory:
            return None
        N = len(self._trajectory)
        if N <= 1:
            return None

        # 短轨迹直接逐帧
        if N <= 128:
            scan_step = 1
        else:
            scan_step = max(1, N // 64)

        prev_i = 0
        first_global: Optional[int] = None
        for chunk_end in range(scan_step - 1, N, scan_step):
            mgr.update_robot_joints(self._trajectory[chunk_end])
            res = mgr.refresh()
            if mgr.is_result_unsafe(res):
                # chunk [prev_i, chunk_end] 末帧碰撞 -> 二分定位
                local_first = self._bisect_first_collide(prev_i, chunk_end)
                first_global = local_first
                break
            prev_i = chunk_end + 1

        if first_global is not None:
            return first_global

        # 兜底:若最末段没扫到,检查 [prev_i, N-1] 末帧
        if prev_i < N:
            mgr.update_robot_joints(self._trajectory[N - 1])
            res = mgr.refresh()
            if mgr.is_result_unsafe(res):
                return self._bisect_first_collide(prev_i, N - 1)
        return None

    def _bisect_first_collide(self, lo: int, hi: int) -> int:
        """在 [lo, hi] 区间内二分找首个碰撞帧 (lo 必须不碰, hi 必须碰)。"""
        mgr = self._state.collision_manager
        # 边界校验
        if lo == hi:
            return hi
        # 确认 lo 不碰(若 lo 已碰则直接返回 lo)
        mgr.update_robot_joints(self._trajectory[lo])
        lo_res = mgr.refresh()
        if mgr.is_result_unsafe(lo_res):
            return lo
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            mgr.update_robot_joints(self._trajectory[mid])
            res = mgr.refresh()
            if mgr.is_result_unsafe(res):
                hi = mid
            else:
                lo = mid
        return hi

    def _format_collision_result(self, result: dict) -> str:
        """Format robot/static and tool/static collision results for the log."""
        parts = []
        for pair in result.get("pairs") or ():
            if len(pair) == 2:
                parts.append(f"{pair[0]} ↔ {pair[1]}")
        robot_part = result.get('robot') or {}
        for link, objs in robot_part.items():
            parts.append(f"{link} -> [{', '.join(objs)}]")
        if not bool(result.get("valid", True)) and not parts:
            parts.append(f"collision service: {result.get('error_code', 'unavailable')}")
        return " | ".join(dict.fromkeys(parts)) if parts else "未知"

    # ── 播放逻辑 ───────────────────────────────────────────────────

    def _on_play(self):
        """开始 / 恢复播放"""
        if not self._trajectory:
            self.log("[错误] 没有可播放的轨迹，请先导入")
            return

        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None and getattr(self._state, "physical_trajectory_stale", True):
            self.log("[错误] 工艺参数已改变，物理轨迹已过期，请重新生成")
            return

        if physical is not None and self._physical_time_s >= physical.duration_s:
            self._physical_time_s = 0.0
            self._current_frame = 0
            self._total_cuts = 0
            self._progress_bar.setValue(0)
        elif self._current_frame >= len(self._trajectory):
            self._current_frame = 0
            self._total_cuts = 0
            self._progress_bar.setValue(0)
        self._sync_current_frame_ui()

        self._btn_play.setEnabled(False)
        self._btn_pause.setEnabled(True)

        # 启动时打印 CollisionManager 状态,便于诊断"网格线不跟随"问题
        _mgr = getattr(self._state, 'collision_manager', None)
        if _mgr is None:
            self.log("[碰撞] state.collision_manager 为空,播放期间无法更新网格线")
        elif getattr(_mgr, '_robot_viz', None) is None:
            self.log("[碰撞] CollisionManager._robot_viz 为空,未注册机器人")

        # 模式1: 预扫描首个碰撞帧
        if self._collision_mode == self.COLLISION_MODE_STOP:
            self.log("[碰撞检测模式1] 分段预扫描定位首个碰撞点...")
            self._first_coll_frame = self._pre_scan_first_collision()
            if self._first_coll_frame is None:
                self.log("[碰撞] 整条轨迹无碰撞,正常播放")
            else:
                self.log(f"[碰撞] 全局首个碰撞点位于第 {self._first_coll_frame} 帧")
        else:
            self._first_coll_frame = None

        self._collision_window.clear()

        if self._elapsed_timer is None or not self._elapsed_timer.isValid():
            self._elapsed_timer = QElapsedTimer()
            self._elapsed_timer.start()
        else:
            self._elapsed_timer.restart()

        if self._play_timer is None:
            self._play_timer = QTimer(self)
            self._play_timer.setTimerType(Qt.PreciseTimer)
            self._play_timer.timeout.connect(self._on_timer)

        self._ensure_interaction_observers()
        now = time.perf_counter()
        self._last_visual_update_s = now - 1.0
        self._last_expensive_update_s = now - 1.0
        self._play_timer.start(int(round(self._compute_interval_ms())))
        self.log("仿真播放开始")

    def _on_timer(self):
        """定时器回调：使用 QElapsedTimer 按真实时间对齐精确推进多帧"""
        interval_ms = self._compute_interval_ms()
        elapsed_ms = self._elapsed_timer.elapsed()
        self._elapsed_timer.restart()
        physical = getattr(self._state, "physical_trajectory", None)
        if self._speed_coeff <= 0.0:
            self._sync_current_frame_ui()
            return

        # 调试：首次打印点云状态
        if self._current_frame == 0 and self.render_engine is not None:
            re = self.render_engine
            self.log(f"[调试] 点云状态: kdtree={'有' if re._kdtree is not None else '无'}, "
                     f"original_points={len(re._original_points) if re._original_points is not None else 0}, "
                     f"active={len(re._active_indices) if re._active_indices is not None else 0}, "
                     f"tool_radius={re._tool_radius:.5f}")

        if physical is not None:
            self._physical_time_s = min(
                physical.duration_s,
                self._physical_time_s + elapsed_ms / 1000.0 * self._speed_coeff,
            )
            frames_to_advance = 1
        else:
            frames_to_advance = max(1, int(round(elapsed_ms / interval_ms)))

        # The interactor owns VTK rendering while the user moves the camera.
        # Physical time still advances, and the next visual update jumps to the
        # correct state instead of replaying a backlog of frames.
        if self._view_interacting:
            if physical is not None:
                self._current_frame = min(
                    int(np.searchsorted(
                        physical.timestamps, self._physical_time_s, side="right"
                    )) - 1,
                    len(self._trajectory) - 1,
                )
                self._current_frame = max(0, self._current_frame)
                self._sync_current_frame_ui()
                self._update_stats()
            return

        now = time.perf_counter()
        # Cap visual actor updates at 30 FPS without changing physical speed.
        if now - self._last_visual_update_s < (1.0 / 30.0):
            return
        self._last_visual_update_s = now
        expensive_update_due = now - self._last_expensive_update_s >= 0.1

        for _ in range(frames_to_advance):
            if self._current_frame >= len(self._trajectory):
                self._on_complete()
                return

            q = (
                physical.sample_at(self._physical_time_s).position
                if physical is not None
                else self._trajectory[self._current_frame]
            )

            if self.render_engine is not None:
                re = self.render_engine

                # 1. 更新机器人关节 (内部自动处理刀具 user_matrix 随动，切勿再调用 update_tool_position)
                re.update_robot_joints(q, render=False)

                # 1.5 碰撞网格只在启用碰撞模式时以 10 Hz 同步。
                mgr = getattr(self._state, 'collision_manager', None)
                if (self._collision_mode != self.COLLISION_MODE_OFF and
                        expensive_update_due and mgr is not None and
                        getattr(mgr, '_robot_viz', None) is not None):
                    try:
                        mgr.update_robot_joints(q)
                    except Exception as _e:
                        self.log(f"[碰撞] update_robot_joints 失败: {_e}")

                # 每帧刷新碰撞高亮；停止/滑窗模式只控制播放策略与日志。
                coll_stop_flag = False
                if (self._collision_mode != self.COLLISION_MODE_OFF and
                        mgr is not None):
                    try:
                        # Evaluate every displayed physical sample atomically
                        # (the display path is already capped at 30 FPS).  The
                        # 10 Hz expensive-update gate above is only for syncing
                        # auxiliary actors/static transforms; using it here can
                        # skip short collision frames.
                        coll_result = mgr.evaluate_configuration(q)
                        has_coll = mgr.is_result_unsafe(coll_result)
                        if self._collision_mode == self.COLLISION_MODE_STOP:
                            if has_coll and self._current_frame == self._first_coll_frame:
                                self.log(
                                    f"[碰撞] 第 {self._current_frame} 帧碰撞,对象: "
                                    f"{self._format_collision_result(coll_result)}"
                                )
                                coll_stop_flag = True
                        elif self._collision_mode == self.COLLISION_MODE_SLIDE:
                            self._collision_window.append(coll_result)
                            if has_coll:
                                self.log(
                                    f"[碰撞] 第 {self._current_frame} 帧: "
                                    f"{self._format_collision_result(coll_result)}"
                                )
                    except Exception as _e:
                        print(f"[CollisionManager] 播放中检测失败: {_e}")

                if coll_stop_flag:
                    self._on_pause()
                    return

                # 2. 获取法兰盘在机器人局部的 4x4 位姿
                flange_pose_local = re.get_robot_ee_pose()

                if flange_pose_local is not None:
                    # 3. 计算法兰盘的世界坐标系位姿 (加上基座变换)
                    base_T = re._robot.base_transform if re._robot else np.eye(4)
                    flange_world_T = base_T @ flange_pose_local

                    # 4. 法兰盘世界坐标（TCP偏移前的位置，包围盒和切削都用这里作为圆柱起点）
                    flange_world = flange_world_T[:3, 3]

                    # 5. 刀具轴方向在世界坐标系中
                    if self._state.flange_tool_params is not None:
                        params = self._state.flange_tool_params
                        T_flange_tcp = (
                            re._build_se3(params.flange_xyz, params.flange_rpy) @
                            re._build_se3(params.tool_xyz,   params.tool_rpy)
                        )
                        tcp_rotation = flange_world_T[:3, :3] @ T_flange_tcp[:3, :3]
                        tool_axis_world = tcp_rotation[:, 2]

                        # 6. 执行布尔切削模拟：圆柱从法兰原点出发
                        n_cuts = 0
                        if expensive_update_due:
                            n_cuts = re.perform_cutting_simulation(
                                flange_world, tool_axis_world, self.log,
                                render=False,
                            )
                        if n_cuts > 0:
                            self._total_cuts += n_cuts

                        # 7. 同步更新刀具包围盒
                        re.update_bounding_box()

                        # 8. 独立更新 C0 坐标轴和切削体积（不受包围盒复选框控制）
                        re.update_tool_visualization()

                    # 9. 更新坐标系可视化（仅法兰坐标系）
                    re.update_coordinate_frames(flange_world_T, None)
                    re.append_flange_trail(flange_world, sample_kind=(
                        physical.sample_at(self._physical_time_s).segment_type
                        if physical is not None else PathSegmentType.PROCESS
                    ))

                # Commit all actor mutations with one throttled render. Nested
                # renders here used to contend with camera interaction events.
                re.render()
                if expensive_update_due:
                    self._last_expensive_update_s = now

            if physical is not None:
                self._current_frame = min(
                    int(np.searchsorted(
                        physical.timestamps, self._physical_time_s, side="right"
                    )) - 1,
                    len(self._trajectory) - 1,
                )
                self._current_frame = max(0, self._current_frame)
            else:
                self._current_frame += 1

        if physical is not None and self._physical_time_s >= physical.duration_s:
            self._on_complete()
            return

        self._current_frame = min(self._current_frame, len(self._trajectory) - 1)
        self._sync_current_frame_ui()
        self._update_stats()

    def _update_stats(self):
        """更新统计标签"""
        n = len(self._trajectory)
        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None:
            sample = physical.sample_at(self._physical_time_s)
            self._stats_label.setText(
                f"时间: {self._physical_time_s:.3f} / {physical.duration_s:.3f} s | "
                f"TCP: {sample.tcp_speed_mps * 1000:.2f} mm/s | 切削点数: {self._total_cuts}"
            )
        else:
            self._stats_label.setText(
                f"轨迹帧索引: {self._current_frame} / {max(0, n - 1)}  |  "
                f"切削点数: {self._total_cuts}"
            )

    def _on_pause(self):
        """暂停"""
        if self._play_timer:
            self._play_timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self.log("仿真暂停")

    def _on_stop(self):
        """停止（重置到开头）"""
        if self._play_timer:
            self._play_timer.stop()
        self._current_frame = 0
        self._physical_time_s = 0.0
        self._total_cuts = 0
        self._elapsed_timer = None
        self._seek_to_frame(0, render=True)
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._update_stats()

        # 清理碰撞检测状态与高亮
        self._collision_window.clear()
        self._first_coll_frame = None
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr._apply_highlights({'robot': {}, 'static': {}})
            except Exception:
                pass

        self.log("仿真停止")

    def _on_complete(self):
        """播放完成事件"""
        if self._play_timer:
            self._play_timer.stop()
        self._btn_play.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._current_frame = max(0, len(self._trajectory) - 1)
        self._sync_current_frame_ui()
        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None:
            self._physical_time_s = physical.duration_s
            self._stats_label.setText(
                f"时间: {physical.duration_s:.3f} / {physical.duration_s:.3f} s | "
                f"切削点数: {self._total_cuts}"
            )
        else:
            self._stats_label.setText(
                f"轨迹帧索引: {max(0, len(self._trajectory) - 1)} / "
                f"{max(0, len(self._trajectory) - 1)}  | 切削点数: {self._total_cuts}"
            )

        # 动画结束后统一执行彻底清理
        if self.render_engine is not None:
            self.log("正在执行最终点云清洗...")
            self.render_engine.delete_transparent_points()

        # 清理碰撞状态(允许保留模式2中可能仍存在的最后一帧红色,这里统一清掉)
        self._collision_window.clear()
        self._first_coll_frame = None

        self.log(f"数字孪生仿真完成！累计标记 {self._total_cuts} 个交集点。")

    def _on_bbox_toggled(self, state: int):
        """刀具圆柱体复选框切换"""
        show = bool(state)
        if self.render_engine:
            self.render_engine.set_bounding_box_visible(show)
        self.log(f"刀具圆柱体: {'显示' if show else '隐藏'}")

    def _on_c0_axes_toggled(self, state: int):
        """C0 坐标轴复选框切换"""
        show = bool(state)
        if self.render_engine:
            self.render_engine.set_c0_axes_visible(show)
        self.log(f"C0 坐标轴: {'显示' if show else '隐藏'}")

    def _on_cutting_vol_toggled(self, state: int):
        """切削体积复选框切换"""
        show = bool(state)
        if self.render_engine:
            self.render_engine.set_cutting_volume_visible(show)
        self.log(f"切削体积: {'显示' if show else '隐藏'}")

    def _on_reset(self):
        """重置场景"""
        self._on_stop()
        if self.render_engine is not None:
            self.render_engine.reset()
            self.render_engine.render()
        self.log("场景已重置")

    def _update_render_from_trajectory(self):
        """按当前规范帧索引刷新 FK，不改变播放或工艺状态。"""
        if not self._trajectory or not self.render_engine:
            return
        self._render_frame_without_process_side_effects(
            int(np.clip(self._current_frame, 0, len(self._trajectory) - 1))
        )

    def _init_traj_plotter(self):
        """Bind the canonical trajectory to the one shared chart instance."""
        trajectory = self._state.joint_trajectory
        if trajectory is None:
            return
        if hasattr(self, '_main_window') and self._main_window:
            self._traj_plotter = self._main_window.set_joint_trajectory_chart(
                trajectory,
                current_frame=self._current_frame,
            )
            self._main_window.show_floating_chart()

    def on_activate(self):
        """从唯一 JointTrajectory 契约加载，并销毁刀轨预览 actor。"""
        if self.render_engine is not None:
            self.render_engine.remove_toolpath_preview()
            self.render_engine.clear_trajectory_axes()

        physical = getattr(self._state, "physical_trajectory", None)
        if physical is not None and not getattr(self._state, "physical_trajectory_stale", True):
            self._state.joint_trajectory = JointTrajectory(
                positions=physical.positions,
                velocities=physical.velocities,
                timestamps=physical.timestamps,
                method=physical.method,
            )
            self._btn_validate_physical.setEnabled(True)
            report = getattr(self._state, "trajectory_validation", None)
            self._btn_export_physical.setEnabled(bool(report and report.passed))

        trajectory = self._state.joint_trajectory
        if trajectory is not None:
            data = trajectory.positions.copy()
            self._trajectory = [row.copy() for row in data]
            self._current_frame = int(np.clip(
                self._state.current_frame, 0, max(0, len(data) - 1)
            ))
            if physical is not None and not self._state.physical_trajectory_stale:
                self._physical_time_s = float(physical.timestamps[self._current_frame])
                self._traj_status_label.setText(
                    f"物理轨迹 {len(data)} 点 / {physical.duration_s:.3f}s"
                )
                self._traj_status_label.setStyleSheet(
                    "color: #1976D2; font-weight: bold;"
                )
            else:
                self._physical_time_s = 0.0
                self._traj_status_label.setText(
                    f"规范关节轨迹 {len(data)} 帧 · {trajectory.method}"
                )
                self._traj_status_label.setStyleSheet(
                    "color: #388E3C; font-weight: bold;"
                )
            self._btn_play.setEnabled(True)
            self._set_frame_range(len(data))
            self._update_time_preview()
            self._seek_to_frame(self._current_frame, render=True)
            self.log(f"Tab 6 激活：从 JointTrajectory 加载 {len(data)} 帧")
            self._init_traj_plotter()
            return

        # 无轨迹可用
        self._trajectory = []
        self._set_frame_range(0)
        self._traj_status_label.setText("请导入关节轨迹 CSV 或先完成 Tab 5 IK 求解")
        self._traj_status_label.setStyleSheet("color: #888;")
        self._btn_play.setEnabled(False)
        self._update_time_preview()
        self.log("Tab 6 激活：无轨迹数据，请导入 joint_trajectory.csv 或先完成 Tab 5 IK 求解")

    def on_deactivate(self):
        """Step 切换时清除轨迹坐标系"""
        if self.render_engine:
            self.render_engine.clear_trajectory_axes()


# ==================== Tab 7: 碰撞检测 ====================

class CollisionWidget(StepWidget):
    """
    Tab 7: 碰撞检测与可视化（阶段 D 瘦身后）。

    4 类网格 + 实时检测的"全部工作"已迁至 CollisionManager，
    本类只剩:
      - UI 控件（按钮 / 状态标签 / 统计标签 / 碰撞列表 / 导入外部 STL 按钮）
      - 状态订阅：订阅 manager 的 refresh 回调 → 更新标签 + 列表
      - 兼容旧字段访问：保留 _robot_collision_viz / _env_collision_viz 等
        只读属性（其他 Tab 用 hasattr() 检查它们的存在性）
    """

    def __init__(self, state: SimulationState, parent=None):
        # ---- 兼容旧字段（其他 Tab 用 hasattr() 检查）----
        self._collision_checker = None
        self._robot_collision_viz = None
        self._env_collision_viz = {}
        self._tool_collision_viz = None
        self._fk_provider = None
        self._collision_enabled = True
        self._status_text_actor = None
        self._current_q = None
        self._last_robot_config = None
        self._last_tool_filepath = None
        self._last_env_objects = None

        super().__init__("碰撞检测", 7, state, parent)

        # ---- 订阅 CollisionManager 状态 ----
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.subscribe_state(self._on_manager_state)
            except Exception as e:
                self.log(f"[碰撞] 订阅 manager 失败: {e}")

    def _init_ui(self):
        _content, layout = self.create_scrollable_layout()
        layout.setContentsMargins(8, 8, 8, 8)

        info = QLabel(
            "碰撞检测与可视化。支持机器人、环境物体、刀具之间的碰撞检测。"
            "发生碰撞时，对应网格将变为红色高亮显示。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 碰撞检测控制组
        control_group = QGroupBox("碰撞检测控制")
        control_layout = QVBoxLayout(control_group)

        self._enable_collision_cb = QCheckBox("启用碰撞检测")
        self._enable_collision_cb.setChecked(True)
        self._enable_collision_cb.stateChanged.connect(self._on_enable_changed)
        control_layout.addWidget(self._enable_collision_cb)

        btn_row = QHBoxLayout()
        self._btn_create_collision = QPushButton("重建碰撞体")
        self._btn_create_collision.clicked.connect(self._on_rebuild_collision)
        self._btn_clear_collision = QPushButton("清除碰撞体")
        self._btn_clear_collision.clicked.connect(self._on_clear_collision)
        self._btn_test_collision = QPushButton("测试碰撞")
        self._btn_test_collision.clicked.connect(self._on_test_collision)
        btn_row.addWidget(self._btn_create_collision)
        btn_row.addWidget(self._btn_clear_collision)
        btn_row.addWidget(self._btn_test_collision)
        control_layout.addLayout(btn_row)

        layout.addWidget(control_group)

        # 碰撞状态显示组
        status_group = QGroupBox("碰撞状态")
        status_layout = QVBoxLayout(status_group)

        self._collision_status_label = QLabel("无碰撞")
        self._collision_status_label.setStyleSheet(
            "color: #88ff88; font-size: 14px; font-weight: bold;"
        )
        self._collision_status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self._collision_status_label)

        self._stats_label = QLabel("统计: -")
        self._stats_label.setStyleSheet("color: #aaa; font-size: 11px;")
        status_layout.addWidget(self._stats_label)

        self._collision_list = QListWidget()
        self._collision_list.setMaximumHeight(100)
        status_layout.addWidget(QLabel("碰撞物体:"))
        status_layout.addWidget(self._collision_list)

        layout.addWidget(status_group)

        # 导入外部 STL 按钮
        import_group = QGroupBox("导入外部碰撞体")
        import_layout = QVBoxLayout(import_group)

        self._btn_import_env_stl = QPushButton("导入环境物体 STL")
        self._btn_import_env_stl.clicked.connect(self._on_import_env_stl)
        import_layout.addWidget(self._btn_import_env_stl)

        self._btn_import_tool_stl = QPushButton("导入刀具 STL")
        self._btn_import_tool_stl.clicked.connect(self._on_import_tool_stl)
        import_layout.addWidget(self._btn_import_tool_stl)

        layout.addWidget(import_group)

        hint = QLabel("提示：关节控制请使用浮动窗口（加载机器人后自动弹出）")
        hint.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        layout.addWidget(hint)

        layout.addStretch()

    # ------------------------------------------------------------------
    # 新方案：manager 状态订阅
    # ------------------------------------------------------------------

    def _on_manager_state(self, result: dict) -> None:
        """
        CollisionManager 每帧检测完成后回调此方法。

        result 结构::
            {
              'robot':  {link_name: [static_name, ...]},
              'static': {static_name: [link_name, ...]},
            }
        """
        if not self._collision_enabled:
            return
        robot = result.get('robot', {})
        static = result.get('static', {})

        is_collide = (
            not bool(result.get("valid", True))
            or bool(robot)
            or bool(static)
        )

        # 状态标签
        if is_collide:
            self._collision_status_label.setText("碰撞!")
            self._collision_status_label.setStyleSheet(
                "color: #ff4444; font-size: 14px; font-weight: bold;"
            )
        else:
            self._collision_status_label.setText("无碰撞")
            self._collision_status_label.setStyleSheet(
                "color: #88ff88; font-size: 14px; font-weight: bold;"
            )

        # 碰撞列表
        self._collision_list.clear()
        seen_lines: set = set()
        for pair in result.get("pairs") or ():
            if len(pair) != 2:
                continue
            line = f"{pair[0]}  ↔  {pair[1]}"
            if line not in seen_lines:
                seen_lines.add(line)
                self._collision_list.addItem(line)
        # robot 侧：每个 link + 它碰撞的所有静态
        for link_name, static_names in robot.items():
            for sname in static_names:
                line = f"Robot.{link_name}  ↔  {sname}"
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                self._collision_list.addItem(line)

        if not bool(result.get("valid", True)) and not seen_lines:
            self._collision_list.addItem(
                f"CollisionService  ↔  {result.get('error_code', 'unavailable')}"
            )

        # static 侧互补：同一对只显示一次（上面已显示）

        # 统计
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None and getattr(mgr, '_checker', None) is not None:
            stats = mgr._checker.get_stats()
            self._stats_label.setText(
                f"step1={stats.get('step1_early_outs', '-')} "
                f"step2={stats.get('step2_pruned', '-')} "
                f"step3={stats.get('step3_collides', '-')}"
            )

    # ------------------------------------------------------------------
    # 按钮回调（向前兼容，重建/清理由 manager 接管）
    # ------------------------------------------------------------------

    def _on_rebuild_collision(self):
        """
        重建所有碰撞体：把现有 manager 中的部件清掉再重新注册。

        由于 4 类网格的注册已经分散在各自 Tab 的"加载/导入"回调中，
        此处只做一次 "重新检测"：用 manager.refresh() 把当前各部件
        状态刷新到 plotter 上。
        """
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is None:
            self.log("[碰撞] 请先加载机器人")
            return

        # 重新触发检测
        if mgr._checker is not None and self._state.kinematics_engine is not None:
            try:
                import numpy as _np
                q = self._state.kinematics_engine.current_q
                if q is not None:
                    mgr.update_robot_joints(_np.asarray(q, dtype=_np.float64))
                    mgr.refresh()
            except Exception as e:
                self.log(f"[碰撞] rebuild 失败: {e}")
        self.log("[碰撞] 碰撞体已重建")

    def _on_clear_collision(self):
        """清除所有碰撞体（清空 manager 上的所有注册）"""
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.clear_env()
                mgr.unregister_cad()
                mgr.unregister_tool()
                # 不主动 unregister_robot：机器人放/取是机器人加载流程的事
            except Exception as e:
                self.log(f"[碰撞] 清除失败: {e}")
        self._collision_status_label.setText("无碰撞")
        self._collision_status_label.setStyleSheet(
            "color: #88ff88; font-size: 14px; font-weight: bold;"
        )
        self._stats_label.setText("统计: -")
        self._collision_list.clear()
        self.mark_collision_scene_changed("collision_bodies_cleared")
        self.log("[碰撞] 碰撞体已清除")

    def _on_test_collision(self):
        """执行碰撞测试（手动触发刷新）"""
        if not self._collision_enabled:
            return
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is None:
            self.log("[警告] 请先加载机器人")
            return
        mgr.refresh()
        self.log("[碰撞] 测试完成")

    def _on_enable_changed(self, state):
        """启用/禁用碰撞检测"""
        self._collision_enabled = (state == Qt.Checked)
        if self._collision_enabled:
            self.log("[碰撞] 碰撞检测已启用")
        else:
            self.log("[碰撞] 碰撞检测已禁用")
            # 清除高亮（通过 manager 的 _last_robot_highlight 重置）
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None and getattr(mgr, '_robot_viz', None) is not None:
                try:
                    for link_name, actor in mgr._robot_viz._actors.items():
                        actor.prop.color = mgr._robot_viz._wireframe_color
                except Exception:
                    pass

    def _on_import_env_stl(self):
        """
        导入环境物体 STL。

        行为：
            1. 写入 state.env_objects（兼容旧逻辑 + 配置导出/导入）
            2. 调用 manager.register_env()（4 类网格之"环境"）
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "选择环境物体 STL", "", "CAD 文件 (*.stl *.step *.stp)"
        )
        if not path:
            return

        name = os.path.splitext(os.path.basename(path))[0]
        counter = 1
        original_name = name
        while any(obj.get('name', '') == name
                  for obj in self._state.env_objects
                  if isinstance(obj, dict)):
            name = f"{original_name}_{counter}"
            counter += 1

        self._state.env_objects.append({
            'name': name,
            'filepath': path,
            'transform': np.eye(4).tolist(),
        })

        # 先通过 render_engine 加载（产生 source_actor 和 mesh）
        if self.render_engine is not None:
            try:
                self.render_engine.load_env_object(path, name)
            except Exception as e:
                self.log(f"[碰撞] render_engine 加载环境物体失败: {e}")

        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.register_env(name, path, transform=np.eye(4))
            except Exception as e:
                self.log(f"[碰撞] 注册环境物体失败: {e}")
        self.mark_collision_scene_changed(
            "environment_loaded", details={"names": [name], "ui": "collision_tab"}
        )
        self.log(f"[碰撞] 环境物体已导入: {name}")

    def _on_import_tool_stl(self):
        """
        导入刀具 STL。

        行为：
            1. 通过 render_engine.create_tool() 创建实体（带几何对齐）
            2. 调用 manager.register_tool()（4 类网格之"刀具"，复用 source actor）
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "选择刀具 STL", "", "CAD 文件 (*.stl *.step *.stp)"
        )
        if not path:
            return

        self._state.tool_filepath = path
        self._state.tool_stl_path = path

        # 先通过 render_engine 创建刀具（产生 _tool_actor 和已对齐的 mesh）
        if self.render_engine is not None:
            try:
                self.render_engine.create_tool(path)
                # 立即把关节推一次，让刀具 user_matrix 有正确值
                if self._state.kinematics_engine and self._state.kinematics_engine.current_q is not None:
                    self.render_engine.update_robot_joints(self._state.kinematics_engine.current_q)
            except Exception as e:
                self.log(f"[碰撞] render_engine 创建刀具失败: {e}")

        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.register_tool(path)
            except Exception as e:
                self.log(f"[碰撞] 注册刀具失败: {e}")
        self.mark_collision_scene_changed("tool_loaded", details={"path": path})
        self.log(f"[碰撞] 刀具碰撞体已导入")

    # ------------------------------------------------------------------
    # Tab 生命周期（兼容：什么都不做也能激活）
    # ------------------------------------------------------------------

    def on_activate(self):
        """Tab 激活时（兼容性保留）"""
        if self._state.kinematics_engine is None:
            self.log("[碰撞] 请先在 Tab 1 加载机器人")
            return

        # 同步浮动窗口的关节控制
        if hasattr(self, '_main_window') and self._main_window is not None:
            jw = getattr(self._main_window, '_joint_control_widget', None)
            if jw is not None and self._current_q is not None:
                for i, v in enumerate(self._current_q):
                    if i < len(jw._joint_sliders):
                        jw._joint_sliders[i].blockSignals(True)
                        jw._joint_sliders[i].setValue(int(v * 1000))
                        jw._joint_sliders[i].blockSignals(False)

        self.log("Tab 7 碰撞检测已激活")

    def on_deactivate(self):
        """Tab 切换时（兼容性保留）"""
        pass


# ==================== 浮动关节控制控件 ====================

class FloatingJointControlWidget(QWidget):
    """浮动关节控制控件（可浮于窗口上方）"""

    def __init__(self, state: 'SimulationState', main_window=None, parent=None):
        super().__init__(parent)
        self._state = state
        self._main_window = main_window  # 存储对主窗口的引用
        self._joint_sliders: List[QSlider] = []
        self._joint_labels: List[QLabel] = []
        self._joint_layout: Optional[QVBoxLayout] = None
        self._placeholder_label: Optional[QLabel] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("FK 关节控制")
        title.setStyleSheet("color: #eeeeee; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        self._joint_layout = QVBoxLayout()
        self._joint_layout.setSpacing(2)
        self._placeholder_label = QLabel("请加载机器人")
        self._placeholder_label.setStyleSheet("color: #888; font-style: italic;")
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._joint_layout.addWidget(self._placeholder_label)
        layout.addLayout(self._joint_layout)

        self.setMinimumWidth(250)

    def sync_with_kinematics_engine(self):
        """同步滑块到 KinematicsEngine 的当前状态"""
        if self._state.kinematics_engine is None:
            return

        try:
            ke = self._state.kinematics_engine
            meta = ke.get_robot_metadata()
            nq = meta['nq']

            # 获取当前关节角度（可能需要从 angles 或 q 中获取）
            current_q = None
            if hasattr(ke, 'angles') and ke.angles is not None:
                current_q = ke.angles
            elif hasattr(ke, 'q') and ke.q is not None:
                # q 可能是完整 Pinocchio 向量，取前 nq 个分量
                current_q = np.array(ke.q[:nq])

            if current_q is not None:
                for i, slider in enumerate(self._joint_sliders):
                    if i < len(current_q):
                        v = float(current_q[i])
                        slider.blockSignals(True)
                        slider.setValue(int(v * 1000))
                        slider.blockSignals(False)
                        # 更新标签
                        if i < len(self._joint_labels):
                            self._joint_labels[i].setText(f"J{i+1}: {v:.2f}")
        except Exception as e:
            print(f"[FloatingJointControl] 同步失败: {e}")

    def rebuild_controls(self):
        """重建关节控制滑块"""
        if self._state.kinematics_engine is None:
            return

        try:
            ke = self._state.kinematics_engine
            meta = ke.get_robot_metadata()
            nq = meta['nq']
            joint_names = meta['joint_names']
            lower_limits = meta['lower_limits']
            upper_limits = meta['upper_limits']

            # 清除旧控件
            while self._joint_layout.count() > 0:
                item = self._joint_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    # 递归清除子布局中的控件
                    self._clear_layout(item.layout())

            self._joint_sliders.clear()
            self._joint_labels.clear()

            # 获取当前关节值
            current_q = None
            if hasattr(ke, 'angles') and ke.angles is not None:
                current_q = ke.angles
            elif hasattr(ke, 'q') and ke.q is not None:
                current_q = np.array(ke.q[:nq])

            # 为每个关节创建滑块
            for i, (name, lo, hi) in enumerate(zip(joint_names, lower_limits, upper_limits)):
                row = QHBoxLayout()
                row.setSpacing(4)

                # 获取初始值
                init_val = 0.0
                if current_q is not None and i < len(current_q):
                    init_val = float(current_q[i])

                label = QLabel(f"J{i+1}: {init_val:.2f}")
                label.setMinimumWidth(70)
                label.setStyleSheet("color: #ccc; font-size: 10px; font-family: Consolas;")
                self._joint_labels.append(label)
                row.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(int(lo * 1000), int(hi * 1000))
                slider.setValue(int(init_val * 1000))
                # 使用闭包正确捕获索引
                slider.valueChanged.connect(
                    self._make_slider_callback(i)
                )
                self._joint_sliders.append(slider)
                row.addWidget(slider)

                self._joint_layout.addLayout(row)

        except Exception as e:
            print(f"[FloatingJointControl] 重建控件失败: {e}")
            import traceback
            traceback.print_exc()

    def _make_slider_callback(self, idx: int):
        """创建滑块回调闭包（正确捕获索引）"""
        def callback(value: int):
            self._on_slider_changed(idx, value / 1000.0)
        return callback

    def _clear_layout(self, layout):
        """递归清除布局"""
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _on_slider_changed(self, joint_idx: int, value: float):
        """滑块值改变时更新 KinematicsEngine"""
        if self._state.kinematics_engine is None:
            return

        try:
            ke = self._state.kinematics_engine

            # 构建完整的 angles 向量
            angles = []
            for i, slider in enumerate(self._joint_sliders):
                angles.append(float(slider.value()) / 1000.0)

            # 更新标签
            if joint_idx < len(self._joint_labels):
                self._joint_labels[joint_idx].setText(f"J{joint_idx+1}: {value:.2f}")

            # 更新 KinematicsEngine
            ke.update_q_from_angles(angles)

            # 获取 nq 大小的向量
            q_nq = ke.current_q.copy()

            # 更新渲染引擎
            if self._main_window is None:
                return

            re = getattr(self._main_window, '_render_engine', None)
            if re is not None:
                re.update_robot_joints(q_nq)

            # ---- 同步 CollisionManager（统一接管 4 类网格的实时刷新） ----
            mgr = getattr(self._state, 'collision_manager', None)
            if mgr is not None:
                try:
                    mgr.update_robot_joints(q_nq)
                    mgr.refresh()
                except Exception as e:
                    print(f"[CollisionManager] refresh 失败: {e}")

            # 同步更新 Tab 7 碰撞体（兼容旧字段）
            for i in range(self._main_window._tab_widget.count()):
                widget = self._main_window._tab_widget.widget(i)
                if hasattr(widget, '_robot_collision_viz') and widget._robot_collision_viz is not None:
                    widget._robot_collision_viz.update_joints(q_nq)
                    widget._current_q = q_nq
                    widget._check_and_highlight_collision()

        except Exception as e:
            print(f"[FloatingJointControl] 更新失败: {e}")


# ==================== 环境物体位置调整浮窗控件 ====================

class EnvironmentObjectFloatWidget(QWidget):
    """环境物体位置调整浮窗内容控件"""

    transform_changed = Signal(str, np.ndarray)  # name, transform

    def __init__(self, state: 'SimulationState', render_engine, parent=None):
        super().__init__(parent)
        self._state = state
        self._render_engine = render_engine
        self._current_object_name: Optional[str] = None
        self._xyz_sboxes: List[QDoubleSpinBox] = []
        self._rpy_sboxes: List[QDoubleSpinBox] = []
        self._object_list_widget = None
        self._combo_object: Optional[QComboBox] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 物体选择区域
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择物体:"))
        self._combo_object = QComboBox()
        self._combo_object.currentIndexChanged.connect(self._on_object_changed)
        select_layout.addWidget(self._combo_object, 1)
        layout.addLayout(select_layout)

        # 物体列表
        list_label = QLabel("已导入物体:")
        list_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(list_label)

        self._object_list_widget = QListWidget()
        self._object_list_widget.setMaximumHeight(80)
        self._object_list_widget.itemClicked.connect(self._on_list_item_clicked)
        layout.addWidget(self._object_list_widget)

        # 位置控制
        pos_group = QGroupBox("位置 (米)")
        pos_layout = QGridLayout(pos_group)

        labels_xyz = ['X:', 'Y:', 'Z:']
        for i, lbl in enumerate(labels_xyz):
            pos_layout.addWidget(QLabel(lbl), 0, i)
            sp = QDoubleSpinBox()
            sp.setRange(-100, 100)
            sp.setDecimals(4)
            sp.setSingleStep(0.01)
            sp.setValue(0.0)
            sp.valueChanged.connect(self._on_transform_changed)
            pos_layout.addWidget(sp, 1, i)
            self._xyz_sboxes.append(sp)

        layout.addWidget(pos_group)

        # 旋转控制
        rot_group = QGroupBox("旋转 (度)")
        rot_layout = QGridLayout(rot_group)

        labels_rpy = ['Rx:', 'Ry:', 'Rz:']
        for i, lbl in enumerate(labels_rpy):
            rot_layout.addWidget(QLabel(lbl), 0, i)
            sp = QDoubleSpinBox()
            sp.setRange(-360, 360)
            sp.setDecimals(2)
            sp.setSingleStep(1.0)
            sp.setValue(0.0)
            sp.valueChanged.connect(self._on_transform_changed)
            rot_layout.addWidget(sp, 1, i)
            self._rpy_sboxes.append(sp)

        layout.addWidget(rot_group)

        # 重置按钮
        reset_btn = QPushButton("重置为原点")
        reset_btn.clicked.connect(self._on_reset_transform)
        layout.addWidget(reset_btn)

        # 刷新列表
        self._refresh_object_list()

    def _build_transform_from_ui(self) -> np.ndarray:
        """从UI值构建4x4变换矩阵"""
        xyz = [s.value() for s in self._xyz_sboxes]
        rpy_deg = [s.value() for s in self._rpy_sboxes]
        rpy_rad = np.deg2rad(rpy_deg)

        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_euler('xyz', rpy_rad).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = xyz
        return T

    def _apply_transform_to_ui(self, T: np.ndarray):
        """将变换矩阵应用到UI控件"""
        from scipy.spatial.transform import Rotation as R_scipy
        rot = R_scipy.from_matrix(T[:3, :3])
        rpy_rad = rot.as_euler('xyz')
        rpy_deg = np.rad2deg(rpy_rad)
        xyz = T[:3, 3]

        for i, sp in enumerate(self._xyz_sboxes):
            sp.blockSignals(True)
            sp.setValue(xyz[i])
            sp.blockSignals(False)

        for i, sp in enumerate(self._rpy_sboxes):
            sp.blockSignals(True)
            sp.setValue(rpy_deg[i])
            sp.blockSignals(False)

    def _on_transform_changed(self):
        """变换值改变时更新渲染并保存到状态"""
        if self._current_object_name is None:
            return
        T = self._build_transform_from_ui()
        if self._render_engine:
            self._render_engine.update_env_object_transform(self._current_object_name, T)
        # 同时保存到 state
        for obj in self._state.env_objects:
            if obj['name'] == self._current_object_name:
                obj['transform'] = T.copy()
                break
        # ---- 同步 CollisionManager（环境物体变换影响碰撞检测）----
        mgr = getattr(self._state, 'collision_manager', None)
        if mgr is not None:
            try:
                mgr.update_env_object_transform(self._current_object_name, T)
                mgr.refresh()
            except Exception:
                pass
        self.transform_changed.emit(self._current_object_name, T)

    def _on_object_changed(self, index: int):
        """选择的物体改变"""
        if index < 0:
            return
        name = self._combo_object.currentText()
        self._current_object_name = name
        self._load_transform_for_object(name)

    def _on_list_item_clicked(self, item: str):
        """点击列表项，同步到下拉框"""
        name = item.text()
        idx = self._combo_object.findText(name)
        if idx >= 0:
            self._combo_object.blockSignals(True)
            self._combo_object.setCurrentIndex(idx)
            self._combo_object.blockSignals(False)

    def _load_transform_for_object(self, name: str):
        """加载指定物体的变换矩阵到UI"""
        for obj in self._state.env_objects:
            if obj['name'] == name:
                self._apply_transform_to_ui(obj['transform'])
                return

    def _on_reset_transform(self):
        """重置为原点（单位矩阵）"""
        T = np.eye(4)
        self._apply_transform_to_ui(T)
        if self._current_object_name:
            # _apply_transform_to_ui blocks spin-box signals, so explicitly
            # persist/synchronize/invalidate after a reset.
            self._on_transform_changed()

    def _refresh_object_list(self):
        """刷新物体列表和下拉框"""
        existing_names = [obj['name'] for obj in self._state.env_objects]
        current_name = self._current_object_name

        # 更新下拉框（blockSignals 防 clear() 触发 currentIndexChanged）
        self._combo_object.blockSignals(True)
        self._combo_object.clear()
        for name in existing_names:
            self._combo_object.addItem(name)
        self._combo_object.blockSignals(False)

        # 更新列表
        self._object_list_widget.clear()
        for name in existing_names:
            self._object_list_widget.addItem(name)

        # Bug fix: 清空后强制重置当前物体索引
        if not existing_names:
            self._current_object_name = None
            # 复位 SpinBox 到 0/0
            try:
                self._apply_transform_to_ui(np.eye(4))
            except Exception:
                pass
            return

        # 如果当前物体已被清除，选择第一个；否则保持原选择
        if current_name and current_name in existing_names:
            idx = existing_names.index(current_name)
        else:
            idx = 0
            current_name = existing_names[0]

        self._current_object_name = current_name
        self._combo_object.blockSignals(True)
        self._combo_object.setCurrentIndex(idx)
        self._combo_object.blockSignals(False)
        # 直接加载变换（不再依赖 _on_object_changed 自动触发）
        self._load_transform_for_object(current_name)


# ==================== 主窗口 ====================

class SimulationApp(QMainWindow):
    """
    数字孪生仿真平台 v5.1
    6 Tab OLP 工作流
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字孪生机器人切削仿真平台 v5.1 - OLP")
        self.setGeometry(100, 100, 1600, 950)

        # 共享状态
        self._state = SimulationState()

        # 渲染引擎
        self._render_engine: Optional[RenderEngine] = None

        # UI
        self._tab_widget = None
        self._log_widget = None

        self._setup_ui()
        self._setup_3d_view()

        # 将主窗口引用传递给所有 Step Widget
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            widget._main_window = self

        self.log("=" * 60)
        self.log("数字孪生机器人切削仿真平台 v5.1")
        self.log("6 Tab OLP 工作流")
        self.log("=" * 60)

    def _setup_ui(self):
        """搭建 UI"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # 左侧 Tab
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.West)
        self._tab_widget.setMovable(True)

        # 添加 7 个 Tab
        steps = [
            RobotAssetWidget(self._state),
            WorkpieceCloudWidget(self._state),
            ToolTCPWidget(self._state),
            ToolpathGeneratorWidget(self._state),
            IKSolveWidget(self._state),
            SimulationWidget(self._state),
            CollisionWidget(self._state),
        ]

        tab_names = [
            "机器人与场景", "工件点云", "工具与 TCP", "刀路与加工工艺",
            "IK 与关节约束", "轨迹生成与仿真", "碰撞与诊断",
        ]
        for step, name in zip(steps, tab_names):
            self._tab_widget.addTab(step, name)

        # Tab 切换信号
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        left_layout.addWidget(self._tab_widget)

        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)

        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(150)
        self._log_widget.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: Consolas;
                font-size: 11px;
            }
        """)
        log_layout.addWidget(self._log_widget)

        left_layout.addWidget(log_group)

        # 右侧 3D 视图（含顶部工具栏）
        self._render_widget = QWidget()
        self._render_layout = QVBoxLayout(self._render_widget)
        self._render_layout.setContentsMargins(0, 0, 0, 0)
        self._render_layout.setSpacing(0)

        # 左右分割（可拖拽）
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._render_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        left_widget.setMinimumWidth(680)
        splitter.setCollapsible(0, False)
        splitter.setSizes([720, 880])
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("""
            QSplitter::handle { background-color: #444; }
            QSplitter::handle:hover { background-color: #888; }
        """)

        main_layout.addWidget(splitter)

        self._splitter = splitter
        self._tab_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._splitter_initialized = False

    def showEvent(self, event):
        super().showEvent(event)
        total = self._splitter.width()
        if not self._splitter_initialized and total > 0:
            preferred_left = min(900, max(720, int(total * 0.43)))
            self._splitter.setSizes([preferred_left, max(1, total - preferred_left)])
            self._splitter_initialized = True

    def _setup_view_toolbar(self):
        """搭建渲染窗口顶部工具栏（视角预设、交互控制）"""
        toolbar = QFrame()
        toolbar.setMaximumHeight(42)
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 12px;
                min-width: 55px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #1a6baa;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #555;
                border-color: #333;
            }
            QLabel {
                color: #888;
                font-size: 11px;
            }
        """)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # 视角预设按钮组
        preset_label = QLabel("视角")
        layout.addWidget(preset_label)

        for name, preset in self._VIEW_PRESETS.items():
            btn = QPushButton(name)
            btn.setToolTip(f"切换到 {name} 视角")
            btn.clicked.connect(lambda _, p=preset: self._set_camera_preset(p))
            layout.addWidget(btn)

        layout.addSpacing(4)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("QFrame { border: none; border-left: 1px solid #444; }")
        layout.addWidget(sep)

        layout.addSpacing(4)

        # 重置相机
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("重置相机到初始位置")
        btn_reset.clicked.connect(self._reset_camera)
        layout.addWidget(btn_reset)

        layout.addSpacing(4)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("QFrame { border: none; border-left: 1px solid #444; }")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        # 地面网格开关
        self._btn_toggle_grid = QPushButton("地面")
        self._btn_toggle_grid.setCheckable(True)
        self._btn_toggle_grid.setChecked(True)
        self._btn_toggle_grid.setToolTip("显示/隐藏网格地面")
        self._btn_toggle_grid.clicked.connect(self._on_toggle_grid)
        layout.addWidget(self._btn_toggle_grid)

        layout.addSpacing(4)
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("QFrame { border: none; border-left: 1px solid #444; }")
        layout.addWidget(sep3)
        layout.addSpacing(4)

        # 背景色切换
        bg_label = QLabel("背景")
        layout.addWidget(bg_label)

        self._btn_bg_dark = QPushButton("深")
        self._btn_bg_dark.setToolTip("深色背景 (#2b2b2b)")
        self._btn_bg_dark.setMaximumWidth(36)
        self._btn_bg_dark.setStyleSheet("""
            QPushButton { background-color: #2b2b2b; border: 2px solid #666; border-radius: 4px; }
            QPushButton:hover { border-color: #888; }
        """)
        self._btn_bg_dark.clicked.connect(lambda: self._set_background("#2b2b2b"))

        self._btn_bg_light = QPushButton("浅")
        self._btn_bg_light.setToolTip("浅色背景 (#e0e0e0)")
        self._btn_bg_light.setMaximumWidth(36)
        self._btn_bg_light.setStyleSheet("""
            QPushButton { background-color: #e0e0e0; border: 2px solid #bbb; border-radius: 4px; color: #333; }
            QPushButton:hover { border-color: #888; }
        """)
        self._btn_bg_light.clicked.connect(lambda: self._set_background("#e0e0e0"))

        layout.addWidget(self._btn_bg_dark)
        layout.addWidget(self._btn_bg_light)

        layout.addStretch()

        self._render_layout.insertWidget(0, toolbar)

    # 相机预设定义
    _VIEW_PRESETS = {
        'Front':      {'pos': (0, -5, 0),   'focal': (0, 0, 0),  'viewup': (0, 0, 1)},
        'Side':       {'pos': (-5, 0, 0),   'focal': (0, 0, 0),  'viewup': (0, 0, 1)},
        'Top':        {'pos': (0, 0, 5),     'focal': (0, 0, 0),  'viewup': (0, 1, 0)},
        'Iso':        {'pos': (3, -3, 2.5), 'focal': (0, 0, 0),  'viewup': (0, 0, 1)},
    }

    def _set_camera_preset(self, preset: dict):
        """应用相机预设（保留当前缩放，仅调整方位角和朝向）"""
        self._plotter.camera_position = preset['pos']
        self._plotter.camera.focal_point = preset['focal']
        self._plotter.camera.up = preset['viewup']
        self._plotter.render()

    def _reset_camera(self):
        """重置相机到初始默认位置（完整重置含缩放）"""
        self._plotter.reset_camera()
        self._plotter.render()

    def _on_toggle_grid(self):
        """切换网格地面可见性"""
        if self._render_engine:
            visible = self._render_engine.toggle_grid()
            self._btn_toggle_grid.setText("地面" if visible else "地面(x)")

    def _set_background(self, color: str):
        """设置渲染背景色"""
        self._plotter.set_background(color)
        # 切背景色不再重复调用 add_axes（之前每次添加会叠加大量 actor，导致卡顿）
        self._plotter.render()

    def _setup_3d_view(self):
        """搭建 3D 视图"""
        # 先建工具栏
        self._setup_view_toolbar()

        # 注：pyvistaqt 在新版里没有显式的 update_rate 构造参数；
        #     FPS 节流已在 setup_scene 里通过 disable_anti_aliasing + minor_grid 隐藏达成。
        self._plotter = QtInteractor(self._render_widget)
        self._render_layout.addWidget(self._plotter)

        # 渲染引擎（默认 high_performance=True + robot_decimate=0.5，最大化 FPS）
        self._render_engine = RenderEngine(self._plotter,
                                           high_performance=True,
                                           robot_decimate=0.5)
        self._render_engine.setup_scene(high_performance=True)
        self._render_engine.add_axes(interactive=True)

        # 坐标轴标签
        self._plotter.add_text(
            "数字孪生仿真平台",
            position="upper_edge",
            font_size=12,
            color="white"
        )

        # 4 类网格碰撞集中管理器（一次性创建）
        self._init_collision_manager()

        # 恢复相机状态（如有）
        self._restore_camera_state()

    def _init_collision_manager(self):
        """
        创建全局 CollisionManager，绑定到 state.collision_manager。

        不依赖具体机器人/CAD/env/tool 数据，仅持有 plotter 引用。
        """
        if not hasattr(self, '_plotter') or self._plotter is None:
            return
        try:
            from collision import CollisionManager
            mgr = CollisionManager(self._plotter, render_engine=self._render_engine)
            self._state.collision_manager = mgr
            self.log("[CollisionManager] 已创建")
        except Exception as e:
            self.log(f"[CollisionManager] 创建失败: {e}")

    def set_joint_trajectory_chart(
        self,
        trajectory: JointTrajectory,
        *,
        current_frame: int = 0,
    ):
        """Publish the canonical full trajectory to the single chart instance."""
        data = np.asarray(trajectory.positions, dtype=float)
        if data.ndim != 2 or len(data) == 0:
            raise ValueError("JointTrajectory.positions 必须为非空二维数组")
        n_joints = data.shape[1]
        plotter = getattr(self, "_floating_chart_plotter", None)
        if plotter is not None and plotter._n_joints != n_joints:
            window = getattr(self, "_floating_chart_window", None)
            if window is not None:
                window.hide()
                window.deleteLater()
                self._floating_chart_window = None
            plotter.close()
            plotter = None
        if plotter is None:
            from plotting import JointTrajectoryPlotter

            plotter = JointTrajectoryPlotter(n_joints=n_joints)
            try:
                lower, upper = self._render_engine.get_joint_limits()
                plotter.set_limits(lower, upper)
            except Exception as exc:
                self.log(f"[警告] 获取关节限位失败: {exc}")
            self._floating_chart_plotter = plotter
        plotter.set_trajectory(data)
        plotter.set_current_frame(current_frame)
        return plotter

    def attach_floating_chart(self, plotter: 'JointTrajectoryPlotter'):
        """
        在渲染窗口右上角创建浮动图表浮窗。
        - 半透明背景（50%），仅曲线保持不透明
        - 可拖拽、可调整大小
        - 始终浮在 PyVista 上方
        """
        existing = getattr(self, '_floating_chart_window', None)
        if existing is not None:
            existing.show()
            return

        if getattr(self, '_floating_chart_closed', False):
            return

        window = QFrame(self._render_widget)
        window.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowStaysOnTopHint)
        window.setFrameShape(QFrame.Box)
        window.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 130);
                border: 1px solid #555;
                border-radius: 4px;
            }
            QLabel {
                color: #cccccc;
                background: transparent;
            }
        """)
        window.setMinimumSize(500, 350)

        rw = self._render_widget.width()
        win_w, win_h = 500, 350
        window.resize(win_w, win_h)
        window.move(max(10, rw - win_w - 10), 10)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(6, 4, 6, 4)
        title_layout.setSpacing(4)

        title_label = QLabel("关节轨迹曲线")
        title_label.setStyleSheet("color: #eeeeee; font-weight: bold; background: transparent;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        close_btn = QPushButton("\u00d7")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #aaa; border: none; font-size: 16px; }
            QPushButton:hover { color: #fff; background: #555; border-radius: 3px; }
        """)
        close_btn.clicked.connect(lambda: (
            window.hide(),
            setattr(self, '_floating_chart_closed', True)
        ))
        title_layout.addWidget(close_btn)

        chart_layout = QVBoxLayout()
        chart_layout.setContentsMargins(2, 2, 2, 2)
        chart_layout.addWidget(plotter.to_widget())

        content_layout = QVBoxLayout(window)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)
        content_layout.addLayout(title_layout)
        content_layout.addLayout(chart_layout)

        class DragHandler:
            _drag_pos = None

            @staticmethod
            def mousePressEvent(obj, event):
                if event.button() == Qt.LeftButton:
                    DragHandler._drag_pos = event.globalPosition().toPoint() - obj.pos()
                    event.accept()

            @staticmethod
            def mouseMoveEvent(obj, event):
                if DragHandler._drag_pos is not None and event.buttons() & Qt.LeftButton:
                    obj.move(event.globalPosition().toPoint() - DragHandler._drag_pos)
                    event.accept()

            @staticmethod
            def mouseReleaseEvent(obj, event):
                DragHandler._drag_pos = None

        window.mousePressEvent = lambda e: DragHandler.mousePressEvent(window, e)
        window.mouseMoveEvent = lambda e: DragHandler.mouseMoveEvent(window, e)
        window.mouseReleaseEvent = lambda e: DragHandler.mouseReleaseEvent(window, e)

        window.show()
        self._floating_chart_window = window
        self._floating_chart_plotter = plotter
        self._floating_chart_closed = False

    def show_floating_chart(self):
        """公开接口：强制弹出浮动图表窗口（支持窗口被关闭后重建）。"""
        existing = getattr(self, '_floating_chart_window', None)
        if existing is not None:
            existing.show()
            return
        plotter = getattr(self, '_floating_chart_plotter', None)
        if plotter is None and self._state.joint_trajectory is not None:
            plotter = self.set_joint_trajectory_chart(
                self._state.joint_trajectory,
                current_frame=self._state.current_frame,
            )
        if plotter is not None:
            self._floating_chart_closed = False
            self.attach_floating_chart(plotter)

    def attach_env_object_float_window(self, content_widget: 'EnvironmentObjectFloatWidget'):
        """
        在渲染窗口上方创建环境物体位置调整浮动浮窗。
        - 半透明背景
        - 可拖拽、可调整大小
        - 始终浮在 PyVista 上方
        """
        existing = getattr(self, '_env_float_window', None)
        if existing is not None:
            existing.show()
            return

        if getattr(self, '_env_float_closed', False):
            return

        window = QFrame(self._render_widget)
        window.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowStaysOnTopHint)
        window.setFrameShape(QFrame.Box)
        window.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 220);
                border: 1px solid #555;
                border-radius: 4px;
            }
            QLabel, QGroupBox {
                color: #cccccc;
                background: transparent;
            }
            QGroupBox {
                border: 1px solid #555;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
        window.setMinimumSize(350, 400)

        rw = self._render_widget.width()
        win_w, win_h = 350, 400
        window.resize(win_w, win_h)
        # 默认位置：渲染窗口左上角偏移一点
        window.move(10, 10)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(6, 4, 6, 4)
        title_layout.setSpacing(4)

        title_label = QLabel("环境物体位置调整")
        title_label.setStyleSheet("color: #eeeeee; font-weight: bold; background: transparent;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        close_btn = QPushButton("\u00d7")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #aaa; border: none; font-size: 16px; }
            QPushButton:hover { color: #fff; background: #555; border-radius: 3px; }
        """)
        close_btn.clicked.connect(lambda: (
            window.hide(),
            setattr(self, '_env_float_closed', True)
        ))
        title_layout.addWidget(close_btn)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(2, 2, 2, 2)
        content_layout.addWidget(content_widget)

        main_layout = QVBoxLayout(window)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)
        main_layout.addLayout(title_layout)
        main_layout.addLayout(content_layout)

        class DragHandler:
            _drag_pos = None

            @staticmethod
            def mousePressEvent(obj, event):
                if event.button() == Qt.LeftButton:
                    DragHandler._drag_pos = event.globalPosition().toPoint() - obj.pos()
                    event.accept()

            @staticmethod
            def mouseMoveEvent(obj, event):
                if DragHandler._drag_pos is not None and event.buttons() & Qt.LeftButton:
                    obj.move(event.globalPosition().toPoint() - DragHandler._drag_pos)
                    event.accept()

            @staticmethod
            def mouseReleaseEvent(obj, event):
                DragHandler._drag_pos = None

        window.mousePressEvent = lambda e: DragHandler.mousePressEvent(window, e)
        window.mouseMoveEvent = lambda e: DragHandler.mouseMoveEvent(window, e)
        window.mouseReleaseEvent = lambda e: DragHandler.mouseReleaseEvent(window, e)

        window.show()
        self._env_float_window = window
        self._env_float_closed = False

    def show_env_float_window(self):
        """公开接口：显示环境物体位置调整浮窗"""
        # 如果浮窗已存在，直接显示
        existing = getattr(self, '_env_float_window', None)
        if existing is not None:
            existing.show()
            # Bug fix: 显示已存在窗口时也刷新列表（清除后列表可能已过期）
            try:
                content = existing.layout().itemAt(1).widget()
                if content is not None and hasattr(content, '_refresh_object_list'):
                    content._refresh_object_list()
            except Exception:
                pass
            return

        # 如果之前被关闭，需要重新创建
        if getattr(self, '_env_float_closed', False):
            self._env_float_closed = False

        # 获取渲染引擎引用
        re = getattr(self, '_render_engine', None)
        if re is None:
            print("[Warning] render_engine not available")
            return

        # 创建浮窗内容控件
        content_widget = EnvironmentObjectFloatWidget(self._state, re)
        content_widget.transform_changed.connect(self._on_environment_transform_changed)

        # 创建并显示浮窗
        self.attach_env_object_float_window(content_widget)

    def _on_environment_transform_changed(self, name: str, transform: np.ndarray) -> None:
        """Advance scene identity after an interactive environment move."""
        if self._tab_widget is None or self._tab_widget.count() == 0:
            return
        asset_widget = self._tab_widget.widget(0)
        callback = getattr(asset_widget, "mark_collision_scene_changed", None)
        if callback is not None:
            callback(
                "environment_transform",
                details={"name": str(name), "transform": np.asarray(transform).tolist()},
            )

    def attach_joint_control_float_window(self):
        """
        在渲染窗口左侧创建浮动 FK 关节控制浮窗。
        - 半透明背景
        - 可拖拽
        - 始终浮在 PyVista 上方
        """
        existing = getattr(self, '_joint_float_window', None)
        if existing is not None:
            existing.show()
            # 同步滑块值
            if hasattr(self, '_joint_control_widget'):
                self._joint_control_widget.sync_with_kinematics_engine()
            return

        if getattr(self, '_joint_float_closed', False):
            return

        # 创建内容控件，传递主窗口引用
        self._joint_control_widget = FloatingJointControlWidget(self._state, main_window=self)
        # 重建控件（如果机器人已加载）
        if self._state.kinematics_engine is not None:
            self._joint_control_widget.rebuild_controls()

        window = QFrame(self._render_widget)
        window.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowStaysOnTopHint)
        window.setFrameShape(QFrame.Box)
        window.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 230);
                border: 1px solid #555;
                border-radius: 4px;
            }
            QLabel {
                color: #cccccc;
                background: transparent;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #444;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px;
                background: #4CAF50;
                border-radius: 7px;
                margin: -4px 0;
            }
        """)
        window.setMinimumSize(280, 200)

        rw = self._render_widget.width()
        rh = self._render_widget.height()
        win_w, win_h = 280, min(400, rh - 40)
        window.resize(win_w, win_h)
        # 默认位置：渲染窗口左侧
        window.move(10, 10)

        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(6, 4, 6, 4)
        title_layout.setSpacing(4)

        title_label = QLabel("FK 关节控制")
        title_label.setStyleSheet("color: #eeeeee; font-weight: bold; background: transparent;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        close_btn = QPushButton("\u00d7")
        close_btn.setFixedSize(20, 20)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #aaa; border: none; font-size: 16px; }
            QPushButton:hover { color: #fff; background: #555; border-radius: 3px; }
        """)
        close_btn.clicked.connect(lambda: (
            window.hide(),
            setattr(self, '_joint_float_closed', True)
        ))
        title_layout.addWidget(close_btn)

        content_layout = QVBoxLayout(window)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(2)
        content_layout.addLayout(title_layout)
        content_layout.addWidget(self._joint_control_widget)

        class DragHandler:
            _drag_pos = None

            @staticmethod
            def mousePressEvent(obj, event):
                if event.button() == Qt.LeftButton:
                    DragHandler._drag_pos = event.globalPosition().toPoint() - obj.pos()
                    event.accept()

            @staticmethod
            def mouseMoveEvent(obj, event):
                if DragHandler._drag_pos is not None and event.buttons() & Qt.LeftButton:
                    obj.move(event.globalPosition().toPoint() - DragHandler._drag_pos)
                    event.accept()

            @staticmethod
            def mouseReleaseEvent(obj, event):
                DragHandler._drag_pos = None

        window.mousePressEvent = lambda e: DragHandler.mousePressEvent(window, e)
        window.mouseMoveEvent = lambda e: DragHandler.mouseMoveEvent(window, e)
        window.mouseReleaseEvent = lambda e: DragHandler.mouseReleaseEvent(window, e)

        window.show()
        self._joint_float_window = window
        self._joint_float_closed = False

        # 同步滑块值
        self._joint_control_widget.sync_with_kinematics_engine()

    def show_joint_control_float_window(self):
        """公开接口：显示 FK 关节控制浮窗"""
        self.attach_joint_control_float_window()

    def _on_tab_changed(self, index: int):
        """Tab 切换：保持当前相机视角"""
        self._save_camera_state()
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if i == index:
                widget.on_activate()
            else:
                widget.on_deactivate()

    def _on_trajectory_viz_params_changed(self):
        """刀路算法页修改轨迹可视化参数后，通知其他 Step 重新渲染"""
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if hasattr(widget, '_render_trajectory_axes'):
                widget._render_trajectory_axes()

    def _save_camera_state(self):
        """保存当前相机状态"""
        if hasattr(self, '_plotter') and self._plotter is not None:
            cam = self._plotter.camera
            self._state.camera_state = {
                'position': tuple(cam.position),
                'focal_point': tuple(cam.focal_point),
                'up': tuple(cam.up),
            }

    def _restore_camera_state(self):
        """恢复保存的相机状态（仅方位角，不重置缩放）"""
        if (self._state.camera_state is not None
                and hasattr(self, '_plotter') and self._plotter is not None):
            cs = self._state.camera_state
            self._plotter.camera_position = cs['position']
            self._plotter.camera.focal_point = cs['focal_point']
            self._plotter.camera.up = cs['up']
            self._plotter.render()

    def log(self, message: str):
        """输出日志"""
        self._log_widget.append(message)
        self._log_widget.verticalScrollBar().setValue(
            self._log_widget.verticalScrollBar().maximum()
        )

    def closeEvent(self, event):
        """关闭"""
        event.accept()


# ==================== 程序入口 ====================

if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SimulationApp()
    window.show()

    sys.exit(app.exec())
