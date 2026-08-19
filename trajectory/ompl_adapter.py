"""Auditable OMPL bridge for joint-space transition planning.

This module intentionally has no PRM, interpolation, or straight-line
fallback.  It accepts either OMPL's public Python bindings or the project's
native ``_ompl_native`` RRTConnect bridge.  If neither binding is available
or either fails at runtime, callers receive a structured failure and must not
manufacture a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import time
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class OMPLPlanResult:
    """The low-level, geometry-only result returned by an OMPL backend."""

    success: bool
    positions: Optional[np.ndarray]
    planner_id: str
    planning_time_s: float
    failure_code: Optional[str] = None
    detail: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success:
            if self.positions is None:
                raise ValueError("a successful OMPL result requires positions")
            values = np.asarray(self.positions, dtype=np.float64)
            if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
                raise ValueError("OMPL positions must be a finite (K,nq) array")
            object.__setattr__(self, "positions", values.copy())
            if self.failure_code is not None:
                raise ValueError("a successful OMPL result cannot have a failure code")
        elif self.positions is not None:
            raise ValueError("a failed OMPL result must not expose a fallback path")
        object.__setattr__(self, "metadata", dict(self.metadata))


class OMPLAdapter:
    """RRTConnect adapter with explicit bounds and validity callbacks."""

    planner_id = "ompl.RRTConnect"

    def __init__(self, *, bindings=None, native_bindings=None) -> None:
        self._ob = None
        self._og = None
        self._ou = None
        self._native = None
        self._backend_kind: Optional[str] = None
        self._import_error: Optional[str] = None
        if bindings is not None:
            self._ob, self._og, self._ou = bindings
            self._backend_kind = "official_python"
            return
        if native_bindings is not None:
            self._native = native_bindings
            self._backend_kind = "native_bridge"
            return

        import_errors = []
        try:
            self._ob = importlib.import_module("ompl.base")
            self._og = importlib.import_module("ompl.geometric")
            try:
                self._ou = importlib.import_module("ompl.util")
            except Exception:
                # RNG control is helpful for reproducibility but not required
                # for a binding that otherwise supports planning.
                self._ou = None
            self._backend_kind = "official_python"
            return
        except Exception as exc:
            import_errors.append(f"official Python bindings: {exc}")

        try:
            native = importlib.import_module("_ompl_native")
            if not callable(getattr(native, "plan_transition", None)):
                raise RuntimeError("_ompl_native.plan_transition is unavailable")
            if getattr(native, "PlanningConfig", None) is None:
                raise RuntimeError("_ompl_native.PlanningConfig is unavailable")
            self._native = native
            self._backend_kind = "native_bridge"
        except Exception as exc:
            import_errors.append(f"native OMPL bridge: {exc}")
            self._import_error = "; ".join(import_errors)

    @property
    def available(self) -> bool:
        if self._backend_kind == "native_bridge":
            return self._import_error is None and self._native is not None
        return (
            self._backend_kind == "official_python"
            and self._import_error is None
            and self._ob is not None
            and self._og is not None
        )

    @property
    def failure_detail(self) -> Optional[str]:
        return self._import_error

    @staticmethod
    def _state_to_vector(state, dof: int) -> np.ndarray:
        return np.asarray([float(state[index]) for index in range(dof)], dtype=np.float64)

    @staticmethod
    def _assign_state(state, values: np.ndarray) -> None:
        for index, value in enumerate(values):
            state[index] = float(value)

    def _plan_with_native_bridge(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        state_validity: Callable[[np.ndarray], bool],
        timeout_s: float,
        seed: int,
        range_rad: float,
        state_check_resolution_fraction: float,
        simplify: bool,
        started: float,
    ) -> OMPLPlanResult:
        """Call the compiled OMPL bridge and reject all non-exact results."""
        try:
            config = self._native.PlanningConfig()
            config.timeout_s = float(timeout_s)
            # The native bridge names its joint-space scale
            # ``interpolation_step_rad``; record this explicit mapping in the
            # result metadata so C's public configuration stays auditable.
            config.interpolation_step_rad = float(range_rad)
            config.state_validity_resolution = float(state_check_resolution_fraction)
            config.random_seed = int(seed)
            config.simplify_path = bool(simplify)

            dof = len(start)

            def native_state_validity(value) -> bool:
                try:
                    q = np.asarray(value, dtype=np.float64).reshape(-1)
                    if q.shape != (dof,) or not np.all(np.isfinite(q)):
                        return False
                    return bool(state_validity(q.copy()))
                except Exception:
                    # A callback failure is a collision failure by contract.
                    return False

            raw = self._native.plan_transition(
                start.copy(),
                goal.copy(),
                lower.copy(),
                upper.copy(),
                native_state_validity,
                config,
            )
            if not isinstance(raw, dict):
                raise TypeError("_ompl_native.plan_transition must return a dict")

            metadata = {
                "binding": "_ompl_native",
                "native_planner_id": str(raw.get("planner_id", "RRTConnect")),
                "native_status": str(raw.get("status", "")),
                "native_ompl_version": str(raw.get("ompl_version", "")),
                "native_exact": bool(raw.get("exact", False)),
                "native_approximate": bool(raw.get("approximate", False)),
                "native_cancelled": bool(raw.get("cancelled", False)),
                "seed": int(seed),
                "timeout_s": float(timeout_s),
                "range_rad": float(range_rad),
                "state_check_resolution_fraction": float(state_check_resolution_fraction),
                "simplify": bool(simplify),
                "native_interpolation_step_rad": float(config.interpolation_step_rad),
            }
            if "state_validity_calls" in raw:
                metadata["native_state_validity_calls"] = int(raw["state_validity_calls"])

            native_elapsed = raw.get("planning_time_s")
            try:
                elapsed = float(native_elapsed)
                if elapsed < 0.0 or not np.isfinite(elapsed):
                    raise ValueError
            except (TypeError, ValueError):
                elapsed = time.perf_counter() - started

            if not bool(raw.get("success", False)):
                code = "ompl_cancelled" if bool(raw.get("cancelled", False)) else "ompl_no_solution"
                return OMPLPlanResult(
                    success=False,
                    positions=None,
                    planner_id=self.planner_id,
                    planning_time_s=elapsed,
                    failure_code=code,
                    detail=f"native RRTConnect status: {metadata['native_status'] or 'no solution'}",
                    metadata=metadata,
                )
            if not bool(raw.get("exact", False)) or bool(raw.get("approximate", False)):
                return OMPLPlanResult(
                    success=False,
                    positions=None,
                    planner_id=self.planner_id,
                    planning_time_s=elapsed,
                    failure_code="ompl_approximate_solution",
                    detail="native RRTConnect did not return an exact solution",
                    metadata=metadata,
                )

            positions = np.asarray(raw.get("positions"), dtype=np.float64)
            if (
                positions.ndim != 2
                or positions.shape[1:] != start.shape
                or len(positions) < 2
                or not np.all(np.isfinite(positions))
                or not np.allclose(positions[0], start, atol=1e-8, rtol=0.0)
                or not np.allclose(positions[-1], goal, atol=1e-8, rtol=0.0)
            ):
                return OMPLPlanResult(
                    success=False,
                    positions=None,
                    planner_id=self.planner_id,
                    planning_time_s=elapsed,
                    failure_code="ompl_invalid_result",
                    detail="native RRTConnect returned a malformed or endpoint-inexact path",
                    metadata=metadata,
                )
            metadata["raw_state_count"] = int(len(positions))
            return OMPLPlanResult(
                success=True,
                positions=positions,
                planner_id=self.planner_id,
                planning_time_s=elapsed,
                metadata=metadata,
            )
        except Exception as exc:
            return OMPLPlanResult(
                success=False,
                positions=None,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                failure_code="ompl_runtime_error",
                detail=str(exc),
                metadata={"binding": "_ompl_native", "seed": int(seed), "timeout_s": float(timeout_s)},
            )

    def plan(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        *,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        state_validity: Callable[[np.ndarray], bool],
        timeout_s: float,
        seed: int,
        range_rad: float,
        state_check_resolution_fraction: float,
        simplify: bool,
    ) -> OMPLPlanResult:
        """Plan with OMPL only; all failures remain explicit failures."""
        started = time.perf_counter()
        if not self.available:
            return OMPLPlanResult(
                success=False,
                positions=None,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                failure_code="ompl_unavailable",
                detail=self._import_error or "OMPL Python bindings are unavailable",
            )

        start = np.asarray(start_q, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal_q, dtype=np.float64).reshape(-1)
        lower = np.asarray(lower_limits, dtype=np.float64).reshape(-1)
        upper = np.asarray(upper_limits, dtype=np.float64).reshape(-1)
        if (
            start.shape != goal.shape
            or start.shape != lower.shape
            or start.shape != upper.shape
            or start.size == 0
            or not np.all(np.isfinite(np.r_[start, goal, lower, upper]))
            or np.any(lower >= upper)
        ):
            return OMPLPlanResult(
                success=False,
                positions=None,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                failure_code="invalid_joint_bounds",
                detail="start, goal, and finite lower/upper joint bounds must agree",
            )
        if timeout_s <= 0 or range_rad <= 0 or not (0 < state_check_resolution_fraction <= 1):
            return OMPLPlanResult(
                success=False,
                positions=None,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                failure_code="invalid_ompl_configuration",
                detail="timeout, range, and state-check resolution are invalid",
            )

        if self._backend_kind == "native_bridge":
            return self._plan_with_native_bridge(
                start,
                goal,
                lower,
                upper,
                state_validity=state_validity,
                timeout_s=timeout_s,
                seed=seed,
                range_rad=range_rad,
                state_check_resolution_fraction=state_check_resolution_fraction,
                simplify=simplify,
                started=started,
            )

        try:
            if self._ou is not None:
                rng = getattr(self._ou, "RNG", None)
                if rng is not None and hasattr(rng, "setSeed"):
                    rng.setSeed(int(seed))

            dof = len(start)
            space = self._ob.RealVectorStateSpace(dof)
            bounds = self._ob.RealVectorBounds(dof)
            for index in range(dof):
                bounds.setLow(index, float(lower[index]))
                bounds.setHigh(index, float(upper[index]))
            space.setBounds(bounds)
            si = self._ob.SpaceInformation(space)

            def is_state_valid(state) -> bool:
                try:
                    q = self._state_to_vector(state, dof)
                    return bool(state_validity(q))
                except Exception:
                    # A validity callback exception is unsafe by definition.
                    return False

            si.setStateValidityChecker(is_state_valid)
            si.setStateValidityCheckingResolution(float(state_check_resolution_fraction))
            planner = self._og.RRTConnect(si)
            planner.setRange(float(range_rad))

            setup = self._og.SimpleSetup(si)
            setup.setPlanner(planner)
            # ``ob.State(space)`` is the public Python binding wrapper used
            # by OMPL's own examples.  Avoid ``SpaceInformation.allocState``
            # here: some Windows bindings expose its raw state pointer without
            # the Python sequence interface needed by ``_assign_state``.
            start_state = self._ob.State(space)
            goal_state = self._ob.State(space)
            self._assign_state(start_state, start)
            self._assign_state(goal_state, goal)
            setup.setStartAndGoalStates(start_state, goal_state)
            setup.setup()
            solved = bool(setup.solve(float(timeout_s)))
            elapsed = time.perf_counter() - started
            if not solved:
                return OMPLPlanResult(
                    success=False,
                    positions=None,
                    planner_id=self.planner_id,
                    planning_time_s=elapsed,
                    failure_code="ompl_no_solution",
                    detail="RRTConnect did not find a path before the timeout",
                    metadata={"seed": int(seed), "timeout_s": float(timeout_s)},
                )
            if simplify:
                setup.simplifySolution()
            path = setup.getSolutionPath()
            count = int(path.getStateCount())
            positions = np.asarray(
                [self._state_to_vector(path.getState(index), dof) for index in range(count)],
                dtype=np.float64,
            )
            return OMPLPlanResult(
                success=True,
                positions=positions,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                metadata={
                    "seed": int(seed),
                    "timeout_s": float(timeout_s),
                    "range_rad": float(range_rad),
                    "state_check_resolution_fraction": float(state_check_resolution_fraction),
                    "simplify": bool(simplify),
                    "raw_state_count": count,
                },
            )
        except Exception as exc:
            return OMPLPlanResult(
                success=False,
                positions=None,
                planner_id=self.planner_id,
                planning_time_s=time.perf_counter() - started,
                failure_code="ompl_runtime_error",
                detail=str(exc),
                metadata={"seed": int(seed), "timeout_s": float(timeout_s)},
            )
