"""Adapt already-solved PROCESS IK into segments without altering the solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.schemas import GeometricTrajectory, PathSegmentType


@dataclass(frozen=True)
class ProcessJointSegment:
    """One collision-free-unknown PROCESS joint segment awaiting C assembly."""

    segment_id: int
    layer_id: int
    positions: np.ndarray
    tcp_poses: np.ndarray
    source_indices: np.ndarray
    source_segment_id: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        poses = np.asarray(self.tcp_poses, dtype=np.float64)
        indices = np.asarray(self.source_indices, dtype=np.int64)
        if positions.ndim != 2 or len(positions) == 0 or not np.all(np.isfinite(positions)):
            raise ValueError("ProcessJointSegment.positions must be a non-empty finite (N,nq) array")
        if poses.shape != (len(positions), 4, 4) or not np.all(np.isfinite(poses)):
            raise ValueError("ProcessJointSegment.tcp_poses must be (N,4,4)")
        if indices.shape != (len(positions),):
            raise ValueError("ProcessJointSegment.source_indices length must match positions")
        positions = positions.copy()
        poses = poses.copy()
        indices = indices.copy()
        positions.setflags(write=False)
        poses.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "tcp_poses", poses)
        object.__setattr__(self, "source_indices", indices)

    @property
    def start_q(self) -> np.ndarray:
        return self.positions[0].copy()

    @property
    def end_q(self) -> np.ndarray:
        return self.positions[-1].copy()


class ProcessIKAdapter:
    """Separates implicit IK branch jumps from true PROCESS geometry.

    It consumes the existing frozen SLSQP/IK output as data.  It never invokes
    or mutates ``algorithms/ik_SLSQP_update.py``; large branch changes become
    explicit C transition requests instead of being relabelled in-place.
    """

    def __init__(self, *, branch_jump_threshold_rad: float = 0.75) -> None:
        if branch_jump_threshold_rad <= 0:
            raise ValueError("branch_jump_threshold_rad must be positive")
        self.branch_jump_threshold_rad = float(branch_jump_threshold_rad)

    def adapt(
        self,
        joint_positions: np.ndarray,
        geometric_trajectory: GeometricTrajectory,
    ) -> tuple[ProcessJointSegment, ...]:
        q = np.asarray(joint_positions, dtype=np.float64)
        geometric = geometric_trajectory
        if q.ndim != 2 or len(q) != len(geometric.tcp_poses) or not np.all(np.isfinite(q)):
            raise ValueError("joint positions must align exactly with geometric TCP samples")

        kinds = np.asarray(geometric.segment_types, dtype=object)
        ids = np.asarray(geometric.segment_ids, dtype=int)
        layers = np.asarray(geometric.layer_ids, dtype=int)
        process_mask = np.asarray(
            [
                item == PathSegmentType.PROCESS
                if isinstance(item, PathSegmentType)
                else PathSegmentType(str(item)) == PathSegmentType.PROCESS
                for item in kinds
            ],
            dtype=bool,
        )
        if not np.any(process_mask):
            raise ValueError("geometric trajectory has no PROCESS samples")

        output: list[ProcessJointSegment] = []
        index = 0
        next_split_id = int(np.max(ids)) + 1
        while index < len(q):
            if not process_mask[index]:
                index += 1
                continue
            start = index
            source_id = int(ids[index])
            layer_id = int(layers[index])
            while (
                index + 1 < len(q)
                and process_mask[index + 1]
                and int(ids[index + 1]) == source_id
                and int(layers[index + 1]) == layer_id
            ):
                index += 1
            stop = index + 1
            split_edges = np.flatnonzero(
                np.max(np.abs(np.diff(q[start:stop], axis=0)), axis=1)
                > self.branch_jump_threshold_rad
            )
            boundaries = [start] + [start + int(edge) + 1 for edge in split_edges] + [stop]
            for part_index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                segment_id = source_id if len(boundaries) == 2 else next_split_id
                if len(boundaries) != 2:
                    next_split_id += 1
                output.append(
                    ProcessJointSegment(
                        segment_id=segment_id,
                        layer_id=layer_id,
                        positions=q[left:right],
                        tcp_poses=geometric.tcp_poses[left:right],
                        source_indices=np.arange(left, right, dtype=np.int64),
                        source_segment_id=source_id,
                    )
                )
            index = stop
        return tuple(output)
