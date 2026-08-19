"""Error-bounded adaptive keyframes for time-parameterized robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class KeyframeSelection:
    indices: np.ndarray
    max_joint_error_rad: float
    max_tcp_chord_error_m: float
    max_orientation_error_rad: float
    max_interval_s: float


def _quintic_states(times, q0, q1, v0, v1, a0, a1, t0, t1):
    h = float(t1 - t0)
    u = (np.asarray(times, dtype=float) - t0) / h
    displacement = q1 - q0
    c0 = q0
    c1 = v0 * h
    c2 = 0.5 * a0 * h * h
    c3 = 10.0 * displacement - (6.0 * v0 + 4.0 * v1) * h - (1.5 * a0 - 0.5 * a1) * h * h
    c4 = -15.0 * displacement + (8.0 * v0 + 7.0 * v1) * h + (1.5 * a0 - a1) * h * h
    c5 = 6.0 * displacement - 3.0 * (v0 + v1) * h - 0.5 * (a0 - a1) * h * h
    uu = u[:, None]
    q = c0 + c1 * uu + c2 * uu**2 + c3 * uu**3 + c4 * uu**4 + c5 * uu**5
    qd = (c1 + 2.0 * c2 * uu + 3.0 * c3 * uu**2 + 4.0 * c4 * uu**3 + 5.0 * c5 * uu**4) / h
    qdd = (2.0 * c2 + 6.0 * c3 * uu + 12.0 * c4 * uu**2 + 20.0 * c5 * uu**3) / (h * h)
    qddd = (6.0 * c3 + 24.0 * c4 * uu + 60.0 * c5 * uu**2) / (h**3)
    return q, qd, qdd, qddd


def select_adaptive_keyframes(
    timestamps: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    jerks: np.ndarray,
    *,
    joint_tolerance_rad: float,
    max_interval_s: float,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    jerk_limits: np.ndarray,
    tcp_poses: Optional[np.ndarray] = None,
    chord_tolerance_m: float = 0.0002,
    orientation_tolerance_rad: float = np.deg2rad(0.5),
    segment_ids: Optional[np.ndarray] = None,
    segment_types: Optional[np.ndarray] = None,
) -> KeyframeSelection:
    """Select a compact, reconstructable set of quintic Hermite keyframes.

    Every accepted interval is bounded by joint reconstruction error, TCP chord
    error, TCP orientation error, dynamic limits, and a maximum time gap.
    Semantic boundaries and global dynamic extrema are always retained.
    """
    t = np.asarray(timestamps, dtype=float)
    q = np.asarray(positions, dtype=float)
    qd = np.asarray(velocities, dtype=float)
    qdd = np.asarray(accelerations, dtype=float)
    qddd = np.asarray(jerks, dtype=float)
    count = len(t)
    if count <= 2:
        return KeyframeSelection(np.arange(count), 0.0, 0.0, 0.0, float(t[-1] - t[0]))

    mandatory = {0, count - 1}
    for values in (qd, qdd, qddd):
        mandatory.update(int(np.argmax(np.abs(values[:, joint]))) for joint in range(values.shape[1]))
    if segment_ids is not None:
        ids = np.asarray(segment_ids)
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        for index in changes:
            mandatory.update((int(index - 1), int(index)))
    if segment_types is not None:
        kinds = np.asarray(segment_types, dtype=object)
        changes = np.flatnonzero(kinds[1:] != kinds[:-1]) + 1
        for index in changes:
            mandatory.update((int(index - 1), int(index)))

    keep = set(mandatory)
    max_joint_error = 0.0
    max_tcp_error = 0.0
    max_orientation_error = 0.0

    def inspect_interval(left: int, right: int):
        nonlocal max_joint_error, max_tcp_error, max_orientation_error
        if right - left <= 1:
            return None
        interior = np.arange(left + 1, right)
        predicted_q, predicted_qd, predicted_qdd, predicted_qddd = _quintic_states(
            t[interior], q[left], q[right], qd[left], qd[right],
            qdd[left], qdd[right], t[left], t[right],
        )
        joint_error_by_row = np.max(np.abs(predicted_q - q[interior]), axis=1)
        interval_joint_error = float(np.max(joint_error_by_row, initial=0.0))
        scores = joint_error_by_row / joint_tolerance_rad
        interval_tcp_error = 0.0
        interval_orientation_error = 0.0

        for predicted, limits in (
            (predicted_qd, velocity_limits),
            (predicted_qdd, acceleration_limits),
            (predicted_qddd, jerk_limits),
        ):
            dynamic_score = np.max(np.abs(predicted) / np.asarray(limits)[None, :], axis=1)
            scores = np.maximum(scores, dynamic_score)

        if tcp_poses is not None:
            tcp = np.asarray(tcp_poses, dtype=float)
            fraction = ((t[interior] - t[left]) / (t[right] - t[left]))[:, None]
            predicted_xyz = (
                (1.0 - fraction) * tcp[left, :3, 3]
                + fraction * tcp[right, :3, 3]
            )
            tcp_error = np.linalg.norm(predicted_xyz - tcp[interior, :3, 3], axis=1)
            interval_tcp_error = float(np.max(tcp_error, initial=0.0))
            scores = np.maximum(scores, tcp_error / chord_tolerance_m)

            endpoint_rotations = Rotation.from_matrix(tcp[[left, right], :3, :3])
            predicted_rotations = Slerp([t[left], t[right]], endpoint_rotations)(t[interior])
            actual_rotations = Rotation.from_matrix(tcp[interior, :3, :3])
            orientation_error = (predicted_rotations.inv() * actual_rotations).magnitude()
            interval_orientation_error = float(np.max(orientation_error, initial=0.0))
            scores = np.maximum(scores, orientation_error / orientation_tolerance_rad)

        worst = int(np.argmax(scores))
        if float(scores[worst]) > 1.0 + 1e-8:
            return int(interior[worst])
        max_joint_error = max(max_joint_error, interval_joint_error)
        max_tcp_error = max(max_tcp_error, interval_tcp_error)
        max_orientation_error = max(max_orientation_error, interval_orientation_error)
        return None

    sorted_mandatory = sorted(mandatory)
    stack = [(left, right) for left, right in zip(sorted_mandatory[:-1], sorted_mandatory[1:])]
    while stack:
        left, right = stack.pop()
        if right - left <= 1:
            continue
        if t[right] - t[left] > max_interval_s * (1.0 + 1e-12):
            midpoint_time = 0.5 * (t[left] + t[right])
            split = int(np.searchsorted(t, midpoint_time))
            split = min(max(split, left + 1), right - 1)
        else:
            split = inspect_interval(left, right)
        if split is not None:
            keep.add(split)
            stack.append((left, split))
            stack.append((split, right))

    indices = np.asarray(sorted(keep), dtype=int)
    actual_max_interval = float(np.max(np.diff(t[indices]))) if len(indices) > 1 else 0.0
    return KeyframeSelection(
        indices=indices,
        max_joint_error_rad=max_joint_error,
        max_tcp_chord_error_m=max_tcp_error,
        max_orientation_error_rad=max_orientation_error,
        max_interval_s=actual_max_interval,
    )
