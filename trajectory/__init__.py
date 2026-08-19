"""生产级轨迹连续化、时间参数化、验证与导出。"""

from .geometry import GeometricPathBuilder, unwrap_revolute_trajectory
from .segmentation import (
    ProcessSegmentation,
    build_process_segmentation,
    contiguous_layer_runs,
    normalize_layer_ids,
    resolve_tcp_matrices,
    zigzag_order_indices,
)
from .compression import KeyframeSelection, select_adaptive_keyframes
from .planner import TrajectoryPlanner, TrajectoryPlanningError
from .validation import TrajectoryValidator, TrajectoryValidationCancelled
from .exporter import export_trajectory_bundle, load_trajectory_bundle
from .ompl_adapter import OMPLAdapter, OMPLPlanResult
from .transition_planner import (
    GeometricPathValidation,
    TransitionPlanner,
    TransitionPlanningConfig,
    TransitionPlanResult,
)
from .process_ik_adapter import ProcessIKAdapter, ProcessJointSegment
from .assembly import TransitionPipelineResult, build_transition_pipeline

__all__ = [
    "GeometricPathBuilder",
    "unwrap_revolute_trajectory",
    "ProcessSegmentation",
    "build_process_segmentation",
    "contiguous_layer_runs",
    "normalize_layer_ids",
    "resolve_tcp_matrices",
    "zigzag_order_indices",
    "KeyframeSelection",
    "select_adaptive_keyframes",
    "TrajectoryPlanner",
    "TrajectoryPlanningError",
    "TrajectoryValidator",
    "TrajectoryValidationCancelled",
    "export_trajectory_bundle",
    "load_trajectory_bundle",
    "OMPLAdapter",
    "OMPLPlanResult",
    "TransitionPlanner",
    "TransitionPlanningConfig",
    "TransitionPlanResult",
    "GeometricPathValidation",
    "ProcessIKAdapter",
    "ProcessJointSegment",
    "TransitionPipelineResult",
    "build_transition_pipeline",
]
