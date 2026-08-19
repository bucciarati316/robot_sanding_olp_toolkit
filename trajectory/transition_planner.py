"""Collision-checked global planning for non-process joint-space transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.schemas import PathSegmentType, TransitionKind
from trajectory.ompl_adapter import OMPLAdapter, OMPLPlanResult


def _frozen_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    result = vector.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TransitionPlanningConfig:
    """Recorded settings for one reproducible OMPL transition query."""

    planner_id: str = "ompl.RRTConnect"
    seed: int = 17
    timeout_s: float = 5.0
    range_rad: float = 0.25
    state_check_resolution_fraction: float = 0.005
    edge_max_joint_step_rad: float = 0.05
    simplify: bool = True

    def __post_init__(self) -> None:
        if self.planner_id != "ompl.RRTConnect":
            raise ValueError("stable2C only permits the auditable OMPL RRTConnect backend")
        if self.timeout_s <= 0 or self.range_rad <= 0 or self.edge_max_joint_step_rad <= 0:
            raise ValueError("timeout, range, and edge resolution must be positive")
        if not 0 < self.state_check_resolution_fraction <= 1:
            raise ValueError("state_check_resolution_fraction must lie in (0, 1]")


@dataclass(frozen=True)
class TransitionPlanResult:
    """A transition succeeds only with a revalidated, endpoint-exact path."""

    request_id: str
    kind: TransitionKind
    start_q: np.ndarray
    goal_q: np.ndarray
    success: bool
    positions: Optional[np.ndarray]
    planner_id: str
    scene_hash: str
    scene_version: int
    seed: int
    timeout_s: float
    planning_time_s: float
    validation_resolution_rad: float
    failure_code: Optional[str] = None
    detail: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        start = _frozen_vector(self.start_q, "start_q")
        goal = _frozen_vector(self.goal_q, "goal_q")
        if start.shape != goal.shape:
            raise ValueError("start_q and goal_q must have the same shape")
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.success:
            if self.positions is None:
                raise ValueError("successful transition requires positions")
            positions = np.asarray(self.positions, dtype=np.float64)
            minimum_count = 1 if np.allclose(start, goal, atol=1e-12, rtol=0.0) else 2
            if (
                positions.ndim != 2
                or positions.shape[1:] != start.shape
                or len(positions) < minimum_count
                or not np.all(np.isfinite(positions))
            ):
                raise ValueError("transition positions have an invalid shape")
            if not np.allclose(positions[0], start, atol=1e-8, rtol=0.0):
                raise ValueError("transition path does not preserve its start endpoint")
            if not np.allclose(positions[-1], goal, atol=1e-8, rtol=0.0):
                raise ValueError("transition path does not preserve its goal endpoint")
            if self.failure_code is not None:
                raise ValueError("successful transition cannot carry a failure code")
            positions = positions.copy()
            positions.setflags(write=False)
            object.__setattr__(self, "positions", positions)
        elif self.positions is not None:
            raise ValueError("failed transition must not carry a pseudo path")
        object.__setattr__(self, "start_q", start)
        object.__setattr__(self, "goal_q", goal)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class GeometricPathValidation:
    """Independent state/edge validation for an assembled C geometry path."""

    valid: bool
    scene_hash: str
    checked_state_count: int
    checked_edge_count: int
    failure_code: Optional[str] = None
    detail: str = ""
    failure_index: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid and self.failure_code is not None:
            raise ValueError("a valid geometric path cannot carry a failure code")
        object.__setattr__(self, "metadata", dict(self.metadata))


class TransitionPlanner:
    """Plans one transition with OMPL and application-side edge verification."""

    def __init__(
        self,
        collision_service,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        config: Optional[TransitionPlanningConfig] = None,
        backend=None,
    ) -> None:
        self._collision_service = collision_service
        self._lower = _frozen_vector(lower_limits, "lower_limits")
        self._upper = _frozen_vector(upper_limits, "upper_limits")
        if self._lower.shape != self._upper.shape or np.any(self._lower >= self._upper):
            raise ValueError("joint bounds must be finite and strictly ordered")
        self.config = config or TransitionPlanningConfig()
        self._backend = backend or OMPLAdapter()

    @property
    def scene_hash(self) -> str:
        snapshot = getattr(self._collision_service, "snapshot", None)
        return str(getattr(snapshot, "scene_hash", ""))

    @property
    def scene_version(self) -> int:
        snapshot = getattr(self._collision_service, "snapshot", None)
        return int(getattr(snapshot, "scene_version", -1))

    def check_configuration(self, q: np.ndarray, *, segment_type: PathSegmentType):
        return self._collision_service.check_configuration(q, segment_type=segment_type)

    def check_edge(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        *,
        segment_type: PathSegmentType,
    ):
        return self._collision_service.check_edge(
            start_q,
            goal_q,
            max_joint_step=self.config.edge_max_joint_step_rad,
            segment_type=segment_type,
        )

    def _failure(
        self,
        request_id: str,
        kind: TransitionKind,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        code: str,
        detail: str,
        *,
        planning_time_s: float = 0.0,
        metadata=None,
    ) -> TransitionPlanResult:
        return TransitionPlanResult(
            request_id=request_id,
            kind=kind,
            start_q=start_q,
            goal_q=goal_q,
            success=False,
            positions=None,
            planner_id=self.config.planner_id,
            scene_hash=self.scene_hash,
            scene_version=self.scene_version,
            seed=self.config.seed,
            timeout_s=self.config.timeout_s,
            planning_time_s=float(planning_time_s),
            validation_resolution_rad=self.config.edge_max_joint_step_rad,
            failure_code=code,
            detail=detail,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _transition_segment_type(kind: TransitionKind) -> PathSegmentType:
        return (
            PathSegmentType.APPROACH
            if kind == TransitionKind.CURRENT_TO_PROCESS
            else PathSegmentType.RAPID
        )

    def _effective_ompl_state_resolution_fraction(self) -> float:
        """Keep OMPL's motion checks conservatively denser than final FCL edges.

        OMPL expresses validity-check resolution as a fraction of the maximum
        joint-space extent, whereas the exact post-plan gate uses an absolute
        joint increment.  A quarter-step safety margin keeps the planning
        discretization from being weaker than the final FCL edge check.
        """
        joint_space_extent = float(np.linalg.norm(self._upper - self._lower))
        if not np.isfinite(joint_space_extent) or joint_space_extent <= 0.0:
            return self.config.state_check_resolution_fraction
        fcl_safe_fraction = self.config.edge_max_joint_step_rad / (4.0 * joint_space_extent)
        return min(self.config.state_check_resolution_fraction, fcl_safe_fraction)

    @staticmethod
    def _edge_segment_type(left, right) -> PathSegmentType:
        """Use the non-process side of a semantic boundary conservatively."""
        left_kind = left if isinstance(left, PathSegmentType) else PathSegmentType(str(left))
        right_kind = right if isinstance(right, PathSegmentType) else PathSegmentType(str(right))
        if left_kind != PathSegmentType.PROCESS:
            return left_kind
        return right_kind

    def validate_geometric_path(
        self,
        positions: np.ndarray,
        segment_types: np.ndarray,
        *,
        edge_segment_types: Optional[np.ndarray] = None,
    ) -> GeometricPathValidation:
        """Recheck all assembled geometry before time parameterization.

        This independently includes PROCESS edges and semantic boundaries; it
        is not a substitute for the per-transition OMPL edge checks.
        """
        values = np.asarray(positions, dtype=np.float64)
        kinds = np.asarray(segment_types, dtype=object)
        edge_kinds = (
            None
            if edge_segment_types is None
            else np.asarray(edge_segment_types, dtype=object)
        )
        if (
            values.ndim != 2
            or len(values) < 2
            or not np.all(np.isfinite(values))
            or kinds.shape != (len(values),)
            or (edge_kinds is not None and edge_kinds.shape != (len(values) - 1,))
        ):
            return GeometricPathValidation(
                valid=False,
                scene_hash=self.scene_hash,
                checked_state_count=0,
                checked_edge_count=0,
                failure_code="invalid_assembled_geometry",
                detail="positions and segment_types must be aligned finite arrays",
            )

        checked_states = 0
        checked_edges = 0
        for index, q in enumerate(values):
            try:
                item = kinds[index]
                segment_type = item if isinstance(item, PathSegmentType) else PathSegmentType(str(item))
                result = self._collision_service.check_configuration(q, segment_type=segment_type)
            except Exception as exc:
                return GeometricPathValidation(
                    valid=False,
                    scene_hash=self.scene_hash,
                    checked_state_count=checked_states,
                    checked_edge_count=checked_edges,
                    failure_code="assembled_state_check_error",
                    detail=str(exc),
                    failure_index=index,
                )
            checked_states += 1
            if not result.valid:
                return GeometricPathValidation(
                    valid=False,
                    scene_hash=self.scene_hash,
                    checked_state_count=checked_states,
                    checked_edge_count=checked_edges,
                    failure_code="assembled_state_invalid",
                    detail=getattr(result, "error_code", None) or getattr(result, "detail", "") or "collision",
                    failure_index=index,
                    metadata={"collision": getattr(result, "as_dict", lambda: {})()},
                )

        for index, (left, right) in enumerate(zip(values[:-1], values[1:])):
            try:
                edge_segment_type = (
                    self._edge_segment_type(kinds[index], kinds[index + 1])
                    if edge_kinds is None
                    else (
                        edge_kinds[index]
                        if isinstance(edge_kinds[index], PathSegmentType)
                        else PathSegmentType(str(edge_kinds[index]))
                    )
                )
                edge = self._collision_service.check_edge(
                    left,
                    right,
                    max_joint_step=self.config.edge_max_joint_step_rad,
                    segment_type=edge_segment_type,
                )
            except Exception as exc:
                return GeometricPathValidation(
                    valid=False,
                    scene_hash=self.scene_hash,
                    checked_state_count=checked_states,
                    checked_edge_count=checked_edges,
                    failure_code="assembled_edge_check_error",
                    detail=str(exc),
                    failure_index=index,
                )
            checked_edges += 1
            if not edge.valid:
                invalid = getattr(edge, "first_invalid", None)
                return GeometricPathValidation(
                    valid=False,
                    scene_hash=self.scene_hash,
                    checked_state_count=checked_states,
                    checked_edge_count=checked_edges,
                    failure_code="assembled_edge_invalid",
                    detail=(
                        getattr(invalid, "error_code", None)
                        or getattr(invalid, "detail", "")
                        or "controlled edge check failed"
                    ),
                    failure_index=index,
                    metadata={"edge_sample_count": getattr(edge, "sample_count", None)},
                )
        return GeometricPathValidation(
            valid=True,
            scene_hash=self.scene_hash,
            checked_state_count=checked_states,
            checked_edge_count=checked_edges,
            metadata={"edge_max_joint_step_rad": self.config.edge_max_joint_step_rad},
        )

    def plan(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        *,
        kind: TransitionKind,
        request_id: str,
    ) -> TransitionPlanResult:
        """Return an endpoint-exact OMPL path or an explicit non-path failure."""
        try:
            start = _frozen_vector(start_q, "start_q")
            goal = _frozen_vector(goal_q, "goal_q")
        except Exception as exc:
            return self._failure(
                request_id,
                kind,
                np.asarray(start_q, dtype=float),
                np.asarray(goal_q, dtype=float),
                "invalid_endpoint",
                str(exc),
            )
        if start.shape != self._lower.shape or goal.shape != self._lower.shape:
            return self._failure(
                request_id, kind, start, goal, "invalid_endpoint_dimension",
                "transition endpoints do not match the configured robot joint dimension",
            )
        if np.any(start < self._lower) or np.any(start > self._upper):
            return self._failure(
                request_id, kind, start, goal, "start_out_of_bounds",
                "start joint configuration violates configured joint bounds",
            )
        if np.any(goal < self._lower) or np.any(goal > self._upper):
            return self._failure(
                request_id, kind, start, goal, "goal_out_of_bounds",
                "goal joint configuration violates configured joint bounds",
            )

        segment_type = self._transition_segment_type(kind)
        try:
            start_validity = self._collision_service.check_configuration(
                start, segment_type=segment_type
            )
            if not start_validity.valid:
                return self._failure(
                    request_id, kind, start, goal, "start_invalid",
                    start_validity.error_code or start_validity.detail or "collision",
                    metadata={"collision_pairs": start_validity.as_dict().get("collision_pairs", [])},
                )
            goal_validity = self._collision_service.check_configuration(
                goal, segment_type=segment_type
            )
            if not goal_validity.valid:
                return self._failure(
                    request_id, kind, start, goal, "goal_invalid",
                    goal_validity.error_code or goal_validity.detail or "collision",
                    metadata={"collision_pairs": goal_validity.as_dict().get("collision_pairs", [])},
                )
        except Exception as exc:
            return self._failure(
                request_id, kind, start, goal, "collision_service_error", str(exc)
            )

        if np.allclose(start, goal, atol=1e-12, rtol=0.0):
            return TransitionPlanResult(
                request_id=request_id,
                kind=kind,
                start_q=start,
                goal_q=goal,
                success=True,
                positions=start[None, :],
                planner_id="no_op_validated",
                scene_hash=self.scene_hash,
                scene_version=self.scene_version,
                seed=self.config.seed,
                timeout_s=self.config.timeout_s,
                planning_time_s=0.0,
                validation_resolution_rad=self.config.edge_max_joint_step_rad,
                metadata={"no_op": True},
            )

        requested_resolution = self.config.state_check_resolution_fraction
        effective_resolution = self._effective_ompl_state_resolution_fraction()
        backend_result: OMPLPlanResult = self._backend.plan(
            start,
            goal,
            lower_limits=self._lower,
            upper_limits=self._upper,
            state_validity=lambda q: self._collision_service.check_configuration(
                q, segment_type=segment_type
            ).valid,
            timeout_s=self.config.timeout_s,
            seed=self.config.seed,
            range_rad=self.config.range_rad,
            state_check_resolution_fraction=effective_resolution,
            simplify=self.config.simplify,
        )
        backend_metadata = {
            **backend_result.metadata,
            "requested_state_check_resolution_fraction": requested_resolution,
            "effective_state_check_resolution_fraction": effective_resolution,
            "effective_state_check_resolution_source": "fcl_edge_max_joint_step_rad/(4*joint_space_extent)",
        }
        if not backend_result.success:
            return self._failure(
                request_id,
                kind,
                start,
                goal,
                backend_result.failure_code or "planner_failed",
                backend_result.detail,
                planning_time_s=backend_result.planning_time_s,
                metadata=backend_metadata,
            )

        positions = backend_result.positions
        if not np.allclose(positions[0], start, atol=1e-8, rtol=0.0) or not np.allclose(
            positions[-1], goal, atol=1e-8, rtol=0.0
        ):
            return self._failure(
                request_id,
                kind,
                start,
                goal,
                "planner_endpoint_mismatch",
                "OMPL returned a path that does not preserve the requested endpoints",
                planning_time_s=backend_result.planning_time_s,
                metadata=backend_metadata,
            )

        for edge_index, (left, right) in enumerate(zip(positions[:-1], positions[1:])):
            try:
                edge = self._collision_service.check_edge(
                    left,
                    right,
                    max_joint_step=self.config.edge_max_joint_step_rad,
                    segment_type=segment_type,
                )
            except Exception as exc:
                return self._failure(
                    request_id,
                    kind,
                    start,
                    goal,
                    "edge_validation_error",
                    str(exc),
                    planning_time_s=backend_result.planning_time_s,
                    metadata={**backend_metadata, "edge_index": edge_index},
                )
            if not edge.valid:
                first_invalid = getattr(edge, "first_invalid", None)
                return self._failure(
                    request_id,
                    kind,
                    start,
                    goal,
                    "edge_validation_failed",
                    getattr(first_invalid, "error_code", None) or "controlled edge check failed",
                    planning_time_s=backend_result.planning_time_s,
                    metadata={
                        **backend_metadata,
                        "edge_index": edge_index,
                        "edge_sample_count": getattr(edge, "sample_count", None),
                    },
                )

        return TransitionPlanResult(
            request_id=request_id,
            kind=kind,
            start_q=start,
            goal_q=goal,
            success=True,
            positions=positions,
            planner_id=backend_result.planner_id,
            scene_hash=self.scene_hash,
            scene_version=self.scene_version,
            seed=self.config.seed,
            timeout_s=self.config.timeout_s,
            planning_time_s=backend_result.planning_time_s,
            validation_resolution_rad=self.config.edge_max_joint_step_rad,
            metadata=backend_metadata,
        )
