"""时间轨迹 CSV + JSON manifest 原子导出与回读。"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from core.schemas import (
    PathSegmentType,
    ProcessParameters,
    TimeParameterizedTrajectory,
    TrajectoryValidationReport,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (PathSegmentType,)):
        return value.value
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def export_trajectory_bundle(
    csv_path: str | os.PathLike,
    trajectory: TimeParameterizedTrajectory,
    parameters: ProcessParameters,
    report: TrajectoryValidationReport,
    *,
    robot_name: str = "unknown",
    urdf_hash: str = "",
    tool_id: str = "unknown",
    diagnostic: bool = False,
) -> tuple[Path, Path]:
    """导出轨迹；硬约束失败时只有 diagnostic=True 才允许保存。"""
    if not report.passed and not diagnostic:
        raise ValueError("轨迹未通过硬约束验证，只能以 diagnostic=True 诊断导出")
    csv_target = Path(csv_path)
    if csv_target.suffix.lower() != ".csv":
        csv_target = csv_target.with_suffix(".csv")
    manifest_target = csv_target.with_suffix(".json")
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = csv_target.with_suffix(".csv.tmp")
    manifest_temp = manifest_target.with_suffix(".json.tmp")

    dof = trajectory.dof
    headers = ["time_s", "segment_id", "segment_type", "transition_kind"]
    for prefix, unit in (("q", "rad"), ("qd", "rad_s"), ("qdd", "rad_s2"), ("qddd", "rad_s3")):
        headers.extend(f"{prefix}_{index + 1}_{unit}" for index in range(dof))
    headers.extend([
        "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw",
        "tcp_speed_mps", "tcp_feed_setpoint_mps", "normal_force_setpoint_n",
    ])
    with csv_temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for index, time_s in enumerate(trajectory.timestamps):
            kind = trajectory.segment_types[index]
            kind_value = kind.value if isinstance(kind, PathSegmentType) else str(kind)
            transition_kind = trajectory.transition_kinds[index]
            row = [
                float(time_s),
                int(trajectory.segment_ids[index]),
                kind_value,
                "" if transition_kind is None else str(transition_kind),
            ]
            row.extend(trajectory.positions[index].tolist())
            row.extend(trajectory.velocities[index].tolist())
            row.extend(trajectory.accelerations[index].tolist())
            row.extend(trajectory.jerks[index].tolist())
            if trajectory.tcp_poses is None:
                row.extend([float("nan")] * 7)
            else:
                pose = trajectory.tcp_poses[index]
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
                row.extend(pose[:3, 3].tolist())
                row.extend(quaternion.tolist())
            row.extend([
                float(trajectory.tcp_speeds_mps[index]),
                float(trajectory.process_channels.get("tcp_feed_setpoint_mps", np.zeros(len(trajectory.timestamps)))[index]),
                float(trajectory.process_channels.get("normal_force_setpoint_n", np.zeros(len(trajectory.timestamps)))[index]),
            ])
            writer.writerow(row)

    manifest = {
        "schema_version": parameters.schema_version,
        "validated": report.passed,
        "diagnostic_export": bool(diagnostic),
        "robot": {"name": robot_name, "urdf_sha256": urdf_hash},
        "tool": {"id": tool_id},
        "units": {
            "position": "m", "time": "s", "joint_position": "rad",
            "joint_velocity": "rad/s", "joint_acceleration": "rad/s^2",
            "joint_jerk": "rad/s^3", "force": "N",
        },
        "trajectory": {
            "method": trajectory.method,
            "duration_s": trajectory.duration_s,
            "sample_count": len(trajectory.timestamps),
            "control_period_s": parameters.control_period_s,
            "metadata": trajectory.metadata,
        },
        "process_parameters": asdict(parameters),
        "capabilities": parameters.capabilities,
        "validation": asdict(report),
    }
    with manifest_temp.open("w", encoding="utf-8") as stream:
        json.dump(_json_value(manifest), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(csv_temp, csv_target)
    os.replace(manifest_temp, manifest_target)
    return csv_target, manifest_target


def load_trajectory_bundle(csv_path: str | os.PathLike) -> TimeParameterizedTrajectory:
    """回读由 export_trajectory_bundle 生成的通用轨迹 CSV。"""
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("轨迹 CSV 至少需要两行数据")
    q_columns = sorted(
        [name for name in rows[0] if name.startswith("q_") and name.endswith("_rad")],
        key=lambda name: int(name.split("_")[1]),
    )
    dof = len(q_columns)
    if dof == 0:
        raise ValueError("轨迹 CSV 缺少 q_*_rad 列")
    prefix_columns = {
        "positions": q_columns,
        "velocities": [f"qd_{i + 1}_rad_s" for i in range(dof)],
        "accelerations": [f"qdd_{i + 1}_rad_s2" for i in range(dof)],
        "jerks": [f"qddd_{i + 1}_rad_s3" for i in range(dof)],
    }
    arrays = {
        name: np.asarray([[float(row[column]) for column in columns] for row in rows])
        for name, columns in prefix_columns.items()
    }
    tcp_values = np.asarray([[float(row[column]) for column in (
        "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"
    )] for row in rows])
    tcp_poses = None
    if np.all(np.isfinite(tcp_values)):
        tcp_poses = np.repeat(np.eye(4)[None, :, :], len(rows), axis=0)
        tcp_poses[:, :3, 3] = tcp_values[:, :3]
        tcp_poses[:, :3, :3] = Rotation.from_quat(tcp_values[:, 3:]).as_matrix()
    return TimeParameterizedTrajectory(
        timestamps=np.asarray([float(row["time_s"]) for row in rows]),
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        accelerations=arrays["accelerations"],
        jerks=arrays["jerks"],
        tcp_poses=tcp_poses,
        tcp_speeds_mps=np.asarray([float(row["tcp_speed_mps"]) for row in rows]),
        segment_ids=np.asarray([int(row["segment_id"]) for row in rows]),
        segment_types=np.asarray([PathSegmentType(row["segment_type"]) for row in rows], dtype=object),
        transition_kinds=np.asarray([
            row.get("transition_kind") or None for row in rows
        ], dtype=object),
        process_channels={
            "tcp_feed_setpoint_mps": np.asarray([float(row["tcp_feed_setpoint_mps"]) for row in rows]),
            "normal_force_setpoint_n": np.asarray([float(row["normal_force_setpoint_n"]) for row in rows]),
        },
        method="imported_bundle",
        metadata={"source": str(path)},
    )
