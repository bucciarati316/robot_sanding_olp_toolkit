from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from core.schemas import PathSegmentType, TransitionKind, TransitionRequest
from trajectory.process_ik_adapter import ProcessJointSegment
from trajectory.transition_planner import TransitionPlanResult


SAFE_ENTRY_FRAME_COUNT = 5


def _frozen_array(value, *, dtype, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has an invalid shape or non-finite value")
    result = array.copy()
    result.setflags(write=False)
    return result


def _segment_type(value) -> PathSegmentType:
    return value if isinstance(value, PathSegmentType) else PathSegmentType(str(value))


def _edge_type(left, right) -> PathSegmentType:
    left_type = _segment_type(left)
    right_type = _segment_type(right)
    return left_type if left_type != PathSegmentType.PROCESS else right_type


def _detail(value, fallback: str) -> str:
    error_code = getattr(value, "error_code", None)
    detail = getattr(value, "detail", None)
    pairs = getattr(value, "collision_pairs", None)
    if pairs is None:
        as_dict = getattr(value, "as_dict", None)
        if callable(as_dict):
            try:
                pairs = as_dict().get("collision_pairs")
            except Exception:
                pairs = None
    if pairs:
        rendered_pairs = ", ".join(
            f"{first} <-> {second}"
            for first, second in sorted(
                tuple(sorted((str(item[0]), str(item[1]))))
                for item in pairs
                if len(item) == 2
            )
        )
        if rendered_pairs:
            return f"{error_code or detail or fallback}; pairs={rendered_pairs}"
    return str(error_code or detail or fallback)


@dataclass(frozen=True)
class TransitionPipelineResult:
    success: bool
    positions: Optional[np.ndarray]
    segment_ids: Optional[np.ndarray]
    segment_types: Optional[np.ndarray]
    transition_kinds: Optional[np.ndarray]
    process_source_indices: Optional[np.ndarray]
    edge_segment_types: Optional[np.ndarray]
    transition_plans: tuple[TransitionPlanResult, ...]
    scene_hash: str
    scene_version: int
    failure_code: Optional[str] = None
    detail: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (
            self.positions,
            self.segment_ids,
            self.segment_types,
            self.transition_kinds,
            self.process_source_indices,
            self.edge_segment_types,
        )
        if self.success:
            if any(value is None for value in values):
                raise ValueError("successful pipeline requires aligned geometric arrays")
            positions = _frozen_array(
                self.positions,
                dtype=np.float64,
                name="positions",
                ndim=2,
            )
            segment_ids = np.asarray(self.segment_ids, dtype=np.int64)
            segment_types = np.asarray(self.segment_types, dtype=object)
            transition_kinds = np.asarray(self.transition_kinds, dtype=object)
            source_indices = np.asarray(self.process_source_indices, dtype=np.int64)
            edge_types = np.asarray(self.edge_segment_types, dtype=object)
            if (
                segment_ids.shape != (len(positions),)
                or segment_types.shape != (len(positions),)
                or transition_kinds.shape != (len(positions),)
                or source_indices.shape != (len(positions),)
                or edge_types.shape != (len(positions) - 1,)
            ):
                raise ValueError("pipeline semantic arrays must align with positions")
            for name, array in (
                ("segment_ids", segment_ids),
                ("segment_types", segment_types),
                ("transition_kinds", transition_kinds),
                ("process_source_indices", source_indices),
                ("edge_segment_types", edge_types),
            ):
                copied = array.copy()
                copied.setflags(write=False)
                object.__setattr__(self, name, copied)
            object.__setattr__(self, "positions", positions)
            if self.failure_code is not None:
                raise ValueError("successful pipeline cannot carry a failure code")
        elif any(value is not None for value in values):
            raise ValueError("failed pipeline must not carry a pseudo geometry path")
        object.__setattr__(self, "transition_plans", tuple(self.transition_plans))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _failure(
    plans: list[TransitionPlanResult],
    scene_hash: str,
    scene_version: int,
    code: str,
    detail: str,
) -> TransitionPipelineResult:
    return TransitionPipelineResult(
        success=False,
        positions=None,
        segment_ids=None,
        segment_types=None,
        transition_kinds=None,
        process_source_indices=None,
        edge_segment_types=None,
        transition_plans=tuple(plans),
        scene_hash=scene_hash,
        scene_version=scene_version,
        failure_code=code,
        detail=detail,
    )


def _resample_path(positions: np.ndarray, count: int) -> Optional[np.ndarray]:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or count < 2:
        return None
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 1e-12:
        return None
    result = np.empty((count, values.shape[1]), dtype=np.float64)
    for output_index, distance in enumerate(np.linspace(0.0, cumulative[-1], count)):
        right = int(np.searchsorted(cumulative, distance, side="right"))
        if right == 0:
            result[output_index] = values[0]
        elif right >= len(values):
            result[output_index] = values[-1]
        else:
            left = right - 1
            width = cumulative[right] - cumulative[left]
            ratio = 0.0 if width <= 1e-12 else (distance - cumulative[left]) / width
            result[output_index] = values[left] + ratio * (values[right] - values[left])
    return result


def _safe_entry_goal(
    home: np.ndarray,
    process_start: np.ndarray,
    transition_planner,
) -> tuple[Optional[np.ndarray], Optional[float], Optional[str], str]:
    try:
        process_validity = transition_planner.check_configuration(
            process_start,
            segment_type=PathSegmentType.PROCESS,
        )
    except Exception as exc:
        return None, None, "process_start_check_error", str(exc)
    if not getattr(process_validity, "valid", False):
        return None, None, "process_start_invalid", _detail(process_validity, "collision")
    delta = process_start - home
    if float(np.linalg.norm(delta)) <= 1e-10:
        return None, None, "safe_entry_degenerate", "q_home equals the first PROCESS configuration"
    last_safe_alpha: Optional[float] = None
    for alpha in np.linspace(1.0 / 64.0, 63.0 / 64.0, 63):
        candidate = home + alpha * delta
        try:
            state = transition_planner.check_configuration(
                candidate,
                segment_type=PathSegmentType.APPROACH,
            )
            edge = transition_planner.check_edge(
                home,
                candidate,
                segment_type=PathSegmentType.APPROACH,
            )
        except Exception as exc:
            return None, None, "safe_entry_scan_error", str(exc)
        if not getattr(state, "valid", False) or not getattr(edge, "valid", False):
            break
        last_safe_alpha = float(alpha)
    if last_safe_alpha is None:
        return None, None, "safe_entry_unavailable", "no collision-free entry configuration exists on the first-source approach ray"
    return home + last_safe_alpha * delta, last_safe_alpha, None, ""


def build_transition_pipeline(
    process_segments: Sequence[ProcessJointSegment],
    *,
    q_home: np.ndarray,
    transition_planner,
    transition_requests: Sequence[TransitionRequest] = (),
) -> TransitionPipelineResult:
    segments = tuple(process_segments)
    scene_hash = str(getattr(transition_planner, "scene_hash", ""))
    scene_version = int(getattr(transition_planner, "scene_version", -1))
    if not segments:
        return _failure(
            [],
            scene_hash,
            scene_version,
            "missing_process_segments",
            "no PROCESS joint segments were supplied",
        )
    home = np.asarray(q_home, dtype=np.float64).reshape(-1)
    if home.shape != segments[0].start_q.shape or not np.all(np.isfinite(home)):
        return _failure(
            [],
            scene_hash,
            scene_version,
            "invalid_home_state",
            "q_home must match the PROCESS joint dimension",
        )
    if not hasattr(transition_planner, "check_configuration") or not hasattr(
        transition_planner,
        "check_edge",
    ):
        return _failure(
            [],
            scene_hash,
            scene_version,
            "planner_contract_error",
            "transition planner must expose exact configuration and edge checks",
        )

    entry_goal, entry_alpha, entry_code, entry_detail = _safe_entry_goal(
        home,
        segments[0].start_q,
        transition_planner,
    )
    if entry_code is not None:
        return _failure([], scene_hash, scene_version, entry_code, entry_detail)

    expected_layer_requests = {
        (int(item.start_segment_id), int(item.goal_segment_id))
        for item in transition_requests
    }
    plans: list[TransitionPlanResult] = []

    def plan(start, goal, kind, ordinal):
        result = transition_planner.plan(
            start,
            goal,
            kind=kind,
            request_id=f"{kind.value}:{ordinal}",
        )
        plans.append(result)
        return result

    raw_entry_plan = plan(
        home,
        entry_goal,
        TransitionKind.CURRENT_TO_PROCESS,
        0,
    )
    if not raw_entry_plan.success:
        return _failure(
            plans,
            scene_hash,
            scene_version,
            raw_entry_plan.failure_code or "transition_failed",
            raw_entry_plan.detail,
        )
    entry_positions = _resample_path(raw_entry_plan.positions, SAFE_ENTRY_FRAME_COUNT)
    if entry_positions is None:
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "safe_entry_resample_failed",
            "OMPL entry path cannot be represented by five non-degenerate frames",
        )
    entry_validation = transition_planner.validate_geometric_path(
        entry_positions,
        np.full(SAFE_ENTRY_FRAME_COUNT, PathSegmentType.APPROACH, dtype=object),
        edge_segment_types=np.full(
            SAFE_ENTRY_FRAME_COUNT - 1,
            PathSegmentType.APPROACH,
            dtype=object,
        ),
    )
    if not entry_validation.valid:
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "safe_entry_resample_invalid",
            entry_validation.detail or entry_validation.failure_code or "FCL rejected the five-frame entry",
        )
    try:
        process_boundary = transition_planner.check_edge(
            entry_positions[-1],
            segments[0].start_q,
            segment_type=PathSegmentType.PROCESS,
        )
    except Exception as exc:
        return _failure(plans, scene_hash, scene_version, "process_entry_edge_error", str(exc))
    if not getattr(process_boundary, "valid", False):
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "process_entry_edge_invalid",
            _detail(getattr(process_boundary, "first_invalid", process_boundary), "collision"),
        )
    entry_plan = replace(
        raw_entry_plan,
        positions=entry_positions,
        metadata={
            **raw_entry_plan.metadata,
            "raw_waypoint_count": int(len(raw_entry_plan.positions)),
            "safe_entry_frame_count": SAFE_ENTRY_FRAME_COUNT,
            "safe_entry_goal_alpha": float(entry_alpha),
        },
    )
    plans[-1] = entry_plan

    q_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    type_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    next_transition_id = max(int(segment.segment_id) for segment in segments) + 1

    def append(
        values,
        segment_id,
        segment_type,
        transition_kind,
        *,
        source_indices=None,
        preserve_first=False,
    ):
        selected = np.asarray(values, dtype=np.float64)
        if selected.ndim != 2 or len(selected) == 0:
            raise ValueError("pipeline pieces must be non-empty joint matrices")
        sources = (
            np.full(len(selected), -1, dtype=np.int64)
            if source_indices is None
            else np.asarray(source_indices, dtype=np.int64)
        )
        if sources.shape != (len(selected),):
            raise ValueError("process source indices must align with their PROCESS frames")
        if (
            q_parts
            and not preserve_first
            and np.allclose(q_parts[-1][-1], selected[0], atol=1e-8, rtol=0.0)
        ):
            selected = selected[1:]
            sources = sources[1:]
        if len(selected) == 0:
            return
        q_parts.append(selected)
        id_parts.append(np.full(len(selected), int(segment_id), dtype=np.int64))
        type_parts.append(np.full(len(selected), segment_type, dtype=object))
        kind_parts.append(np.full(len(selected), transition_kind, dtype=object))
        source_parts.append(sources.copy())

    append(
        entry_plan.positions,
        next_transition_id,
        PathSegmentType.APPROACH,
        entry_plan.kind.value,
    )
    entry_boundary_index = len(entry_plan.positions) - 1
    next_transition_id += 1
    source_offset = 0

    for index, segment in enumerate(segments):
        source_indices = np.arange(
            source_offset,
            source_offset + len(segment.positions),
            dtype=np.int64,
        )
        append(
            segment.positions,
            segment.segment_id,
            PathSegmentType.PROCESS,
            None,
            source_indices=source_indices,
            preserve_first=True,
        )
        source_offset += len(segment.positions)
        if index + 1 >= len(segments):
            continue
        following = segments[index + 1]
        if segment.source_segment_id == following.source_segment_id:
            kind = TransitionKind.CONFIGURATION_SWITCH
        else:
            kind = TransitionKind.LAYER_TRANSITION
            expected = (segment.source_segment_id, following.source_segment_id)
            if expected not in expected_layer_requests:
                return _failure(
                    plans,
                    scene_hash,
                    scene_version,
                    "missing_transition_request",
                    f"no B TransitionRequest links PROCESS segments {expected[0]} -> {expected[1]}",
                )
        planned = plan(segment.end_q, following.start_q, kind, index + 1)
        if not planned.success:
            return _failure(
                plans,
                scene_hash,
                scene_version,
                planned.failure_code or "transition_failed",
                planned.detail,
            )
        append(
            planned.positions,
            next_transition_id,
            PathSegmentType.RAPID,
            planned.kind.value,
        )
        next_transition_id += 1

    last = plan(segments[-1].end_q, home, TransitionKind.RETURN_HOME, len(segments))
    if not last.success:
        return _failure(
            plans,
            scene_hash,
            scene_version,
            last.failure_code or "transition_failed",
            last.detail,
        )
    append(
        last.positions,
        next_transition_id,
        PathSegmentType.RAPID,
        last.kind.value,
    )

    positions = np.vstack(q_parts)
    segment_ids = np.concatenate(id_parts)
    segment_types = np.concatenate(type_parts)
    transition_kinds = np.concatenate(kind_parts)
    process_source_indices = np.concatenate(source_parts)
    if len(positions) < 2:
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "assembled_path_too_short",
            "assembled path has fewer than two points",
        )
    if not np.allclose(positions[0], home, atol=1e-8, rtol=0.0):
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "home_endpoint_mismatch",
            "assembled path does not start at q_home",
        )
    if not np.allclose(positions[-1], home, atol=1e-8, rtol=0.0):
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "return_home_mismatch",
            "assembled path does not end at q_home",
        )
    retained_sources = process_source_indices[process_source_indices >= 0]
    if not np.array_equal(retained_sources, np.arange(source_offset, dtype=np.int64)):
        return _failure(
            plans,
            scene_hash,
            scene_version,
            "process_source_mapping_invalid",
            "the assembled path did not retain every PROCESS source frame exactly once",
        )
    edge_segment_types = np.asarray(
        [_edge_type(left, right) for left, right in zip(segment_types[:-1], segment_types[1:])],
        dtype=object,
    )
    edge_segment_types[entry_boundary_index] = PathSegmentType.PROCESS
    return TransitionPipelineResult(
        success=True,
        positions=positions,
        segment_ids=segment_ids,
        segment_types=segment_types,
        transition_kinds=transition_kinds,
        process_source_indices=process_source_indices,
        edge_segment_types=edge_segment_types,
        transition_plans=tuple(plans),
        scene_hash=scene_hash,
        scene_version=scene_version,
        metadata={
            "planner_ids": [plan.planner_id for plan in plans],
            "request_ids": [plan.request_id for plan in plans],
            "transition_count": len(plans),
            "process_segment_count": len(segments),
            "process_source_frame_count": int(source_offset),
            "safe_entry_frame_count": SAFE_ENTRY_FRAME_COUNT,
            "source_plus_safe_entry_frame_count": int(source_offset + SAFE_ENTRY_FRAME_COUNT),
            "safe_entry_goal_alpha": float(entry_alpha),
            "process_source_index_range": [0, max(0, int(source_offset) - 1)],
        },
    )
