"""Reusable global Cartesian free-space planning against an environment SDF."""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np

from collision.distance_field import EnvironmentDistanceField


def _segment_clear(
    start: np.ndarray,
    goal: np.ndarray,
    field: EnvironmentDistanceField,
    clearance: float,
    step: float,
) -> bool:
    count = max(2, int(np.ceil(np.linalg.norm(goal - start) / max(step, 1e-4))) + 1)
    return bool(np.all(field.distance(np.linspace(start, goal, count)) >= clearance - 1e-6))


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Arc-length resample a collision-certified polyline."""
    points = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(lengths.sum())
    if total < 1e-12:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, total, count)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    out = np.empty((count, 3), dtype=np.float64)
    cursor = 0
    for i, target in enumerate(targets):
        while cursor < len(lengths) - 1 and target > cumulative[cursor + 1]:
            cursor += 1
        ratio = (target - cumulative[cursor]) / max(lengths[cursor], 1e-12)
        out[i] = (1.0 - ratio) * points[cursor] + ratio * points[cursor + 1]
    return out


def plan_sdf_clearance_path(
    start: np.ndarray,
    goal: np.ndarray,
    distance_field: EnvironmentDistanceField,
    *,
    clearance: float,
    grid_resolution: float = 0.015,
    max_nodes: int = 1_200_000,
) -> np.ndarray:
    """Return a globally searched, surface-clear Cartesian polyline.

    The planner intentionally searches three dimensions rather than pushing
    points along one local normal; it therefore works with concave and
    disconnected valid STL meshes.  It raises if an endpoint is infeasible or
    no route exists in its progressively enlarged search volume.
    """
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    clearance = max(0.0, float(clearance))
    resolution = max(0.002, float(grid_resolution))
    if not _segment_clear(start, start, distance_field, clearance, resolution):
        raise RuntimeError("SDF path start violates the requested clearance")
    if not _segment_clear(goal, goal, distance_field, clearance, resolution):
        raise RuntimeError("SDF path goal violates the requested clearance")
    if _segment_clear(start, goal, distance_field, clearance, resolution * 0.5):
        return np.vstack([start, goal])

    bounds = np.asarray(distance_field.mesh.bounds, dtype=np.float64).reshape(3, 2)
    mesh_min, mesh_max = bounds[:, 0], bounds[:, 1]
    direct_length = float(np.linalg.norm(goal - start))
    mesh_extent = float(np.linalg.norm(mesh_max - mesh_min))
    offsets = np.asarray(
        [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
         if (x, y, z) != (0, 0, 0)], dtype=np.int32,
    )
    base_padding = max(0.08, 2.0 * clearance + 2.0 * resolution, 0.15 * direct_length)

    for search_attempt in range(4):
        padding = base_padding + search_attempt * max(0.10, 0.5 * mesh_extent, 0.25 * direct_length)
        lower = np.minimum(np.minimum(start, goal), mesh_min) - padding
        upper = np.maximum(np.maximum(start, goal), mesh_max) + padding
        shape = np.ceil((upper - lower) / resolution).astype(np.int32) + 1
        effective_resolution = resolution
        if int(np.prod(shape)) > max_nodes:
            effective_resolution *= (int(np.prod(shape)) / max_nodes) ** (1.0 / 3.0)
            shape = np.ceil((upper - lower) / effective_resolution).astype(np.int32) + 1
        source = tuple(np.rint((start - lower) / effective_resolution).astype(int))
        target = tuple(np.rint((goal - lower) / effective_resolution).astype(int))

        def world(index: tuple[int, int, int]) -> np.ndarray:
            return lower + np.asarray(index, dtype=np.float64) * effective_resolution

        free_cache: dict[tuple[int, int, int], bool] = {}

        def free(index: tuple[int, int, int]) -> bool:
            if any(index[a] < 0 or index[a] >= shape[a] for a in range(3)):
                return False
            if index not in free_cache:
                free_cache[index] = bool(distance_field.distance(world(index))[0] >= clearance - 1e-6)
            return free_cache[index]

        if not free(source) or not free(target):
            continue
        queue = [(float(np.linalg.norm(world(source) - world(target))), 0.0, source)]
        previous: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        cost = {source: 0.0}
        while queue:
            _, current_cost, current = heappop(queue)
            if current_cost != cost.get(current):
                continue
            if current == target:
                indices = [target]
                while indices[-1] != source:
                    indices.append(previous[indices[-1]])
                path = np.vstack([start, *(world(i) for i in reversed(indices)), goal])
                return _shortcut(path, distance_field, clearance, effective_resolution)
            current_world = world(current)
            for delta in offsets:
                neighbor = tuple((np.asarray(current) + delta).tolist())
                if not free(neighbor):
                    continue
                neighbor_world = world(neighbor)
                if not _segment_clear(current_world, neighbor_world, distance_field, clearance, effective_resolution * 0.4):
                    continue
                next_cost = current_cost + float(np.linalg.norm(neighbor_world - current_world))
                if next_cost >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = next_cost
                previous[neighbor] = current
                heappush(queue, (next_cost + float(np.linalg.norm(neighbor_world - world(target))), next_cost, neighbor))
    raise RuntimeError("no SDF-clear Cartesian transfer path exists")


def _shortcut(path: np.ndarray, field: EnvironmentDistanceField, clearance: float, resolution: float) -> np.ndarray:
    result = [path[0]]
    current = 0
    while current < len(path) - 1:
        candidate = len(path) - 1
        while candidate > current + 1 and not _segment_clear(path[current], path[candidate], field, clearance, resolution * 0.35):
            candidate -= 1
        result.append(path[candidate])
        current = candidate
    return np.asarray(result, dtype=np.float64)


__all__ = ["plan_sdf_clearance_path", "resample_polyline"]
