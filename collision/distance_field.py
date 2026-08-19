"""
Environment distance field utilities.

The primary backend uses Open3D RaycastingScene signed distances.  VTK exact
triangle distances provide the normal fallback; a lightweight KDTree remains
only for minimal environments without either backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}")
    return arr


def _mesh_to_triangles(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangle indices from a PyVista-like mesh."""
    tri_mesh = mesh.triangulate()
    ug = tri_mesh.cast_to_unstructured_grid()
    cells = np.asarray(ug.cells)
    triangles = []
    idx = 0
    for _ in range(ug.n_cells):
        n_verts = int(cells[idx])
        idx += 1
        if n_verts == 3:
            triangles.append([int(cells[idx]), int(cells[idx + 1]), int(cells[idx + 2])])
        idx += n_verts
    if not triangles:
        raise ValueError("mesh contains no triangles")
    return (
        np.ascontiguousarray(ug.points.astype(np.float64)),
        np.ascontiguousarray(np.asarray(triangles, dtype=np.int32)),
    )


def _load_pyvista_mesh(mesh_or_path):
    import pyvista as pv

    if isinstance(mesh_or_path, str):
        mesh = pv.read(mesh_or_path)
    else:
        mesh = mesh_or_path.copy(deep=True) if hasattr(mesh_or_path, "copy") else mesh_or_path
    if mesh is None or getattr(mesh, "n_points", 0) == 0:
        raise ValueError("environment mesh is empty")

    bounds = np.ptp(np.asarray(mesh.points), axis=0)
    if np.max(bounds) > 1.0:
        mesh.points = np.asarray(mesh.points, dtype=np.float64) * 0.001
    return mesh


@dataclass
class DistanceQueryResult:
    distances: np.ndarray
    backend: str


class EnvironmentDistanceField:
    """
    Query signed or conservative unsigned distances to a static environment mesh.

    Distances are in meters.  Negative values mean "inside" only when the active
    backend can determine inside/outside reliably.
    """

    def __init__(self, mesh: Union[str, object], *, prefer_open3d: bool = True):
        self._mesh = _load_pyvista_mesh(mesh)
        self._vertices, self._triangles = _mesh_to_triangles(self._mesh)
        self._backend = "kdtree"
        self._scene = None
        self._tree = None
        self._implicit_distance = None
        self._vtk_signed = False
        self._signed_distance_valid = bool(
            getattr(self._mesh, "n_open_edges", 1) == 0
            and getattr(self._mesh, "n_non_manifold_edges", 0) == 0
        )
        self._closed_surface = None

        if prefer_open3d and self._try_init_open3d():
            self._backend = "open3d"
        else:
            self._init_kdtree()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def mesh(self):
        return self._mesh

    def distance(self, points: np.ndarray) -> np.ndarray:
        points_arr = _as_points(points)
        if self._backend == "open3d":
            distances = self._distance_open3d(points_arr)
        else:
            distances = self._distance_kdtree(points_arr)
        # Open/non-manifold STL files do not define an inside/outside region.
        # Use their unsigned triangle distance so clearance is enforced on
        # both sides instead of trusting arbitrary facet winding.
        return distances if self._signed_distance_valid else np.abs(distances)

    def query(self, points: np.ndarray) -> DistanceQueryResult:
        return DistanceQueryResult(self.distance(points), self._backend)

    def min_distance(self, points: np.ndarray) -> float:
        distances = self.distance(points)
        if distances.size == 0:
            return float("inf")
        return float(np.min(distances))

    def gradient(self, points: np.ndarray, *, epsilon: float = 1e-4) -> np.ndarray:
        """Return normalized outward signed-distance gradients for query points."""
        points_arr = _as_points(points)
        epsilon = max(float(epsilon), 1e-7)
        gradients = np.empty_like(points_arr)
        for axis in range(3):
            delta = np.zeros(3, dtype=np.float64)
            delta[axis] = epsilon
            gradients[:, axis] = (
                self.distance(points_arr + delta) - self.distance(points_arr - delta)
            ) / (2.0 * epsilon)
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-9
        gradients[valid] /= norms[valid]
        gradients[~valid] = 0.0
        return gradients

    def project_to_clearance(
        self,
        points: np.ndarray,
        clearance: float,
        *,
        tangent: Optional[np.ndarray] = None,
        direction_hint: Optional[np.ndarray] = None,
        tolerance: float = 1e-5,
        max_iterations: int = 12,
    ) -> np.ndarray:
        """Push points along local SDF normals until surface clearance is met."""
        projected = _as_points(points).copy()
        clearance = max(0.0, float(clearance))
        tolerance = max(0.0, float(tolerance))
        tangent_vec = None
        if tangent is not None:
            tangent_vec = np.asarray(tangent, dtype=np.float64).reshape(3)
            tangent_norm = float(np.linalg.norm(tangent_vec))
            if tangent_norm > 1e-9:
                tangent_vec /= tangent_norm
            else:
                tangent_vec = None
        hint_vec = None
        if direction_hint is not None:
            hint_vec = np.asarray(direction_hint, dtype=np.float64).reshape(3)
            if tangent_vec is not None:
                hint_vec -= np.dot(hint_vec, tangent_vec) * tangent_vec
            hint_norm = float(np.linalg.norm(hint_vec))
            if hint_norm > 1e-9:
                hint_vec /= hint_norm
            else:
                hint_vec = None
        for _ in range(max(1, int(max_iterations))):
            distances = self.distance(projected)
            violating = distances < clearance - tolerance
            if not np.any(violating):
                break
            gradients = self.gradient(projected[violating])
            if hint_vec is not None:
                directions = np.repeat(hint_vec.reshape(1, 3), len(gradients), axis=0)
            else:
                directions = gradients.copy()
                if tangent_vec is not None:
                    directions -= np.outer(directions @ tangent_vec, tangent_vec)
            direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
            valid = direction_norms[:, 0] > 1e-9
            if not np.any(valid):
                break
            directions[valid] /= direction_norms[valid]
            indices = np.flatnonzero(violating)[valid]
            mesh_scale = float(np.linalg.norm(np.ptp(self._vertices, axis=0)))
            search_limit = max(mesh_scale + clearance, clearance * 2.0, 1e-3)
            for local_index, point_index in enumerate(indices):
                origin = projected[point_index].copy()
                direction = directions[valid][local_index]
                low = 0.0
                high = max(clearance - distances[point_index], tolerance, 1e-5)
                high_distance = float(self.distance(origin + high * direction)[0])
                while high_distance < clearance and high < search_limit:
                    high *= 2.0
                    high_distance = float(self.distance(origin + high * direction)[0])
                high = min(high, search_limit)
                if high_distance < clearance:
                    continue
                for _ in range(24):
                    middle = 0.5 * (low + high)
                    middle_distance = float(self.distance(origin + middle * direction)[0])
                    if middle_distance < clearance:
                        low = middle
                    else:
                        high = middle
                projected[point_index] = origin + (high + tolerance) * direction
        return projected

    def _try_init_open3d(self) -> bool:
        try:
            import open3d as o3d

            mesh = o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(self._vertices, dtype=o3d.core.Dtype.Float32),
                o3d.core.Tensor(self._triangles, dtype=o3d.core.Dtype.UInt32),
            )
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh)
            self._scene = scene
            self._o3d = o3d
            return True
        except Exception:
            self._scene = None
            return False

    def _distance_open3d(self, points: np.ndarray) -> np.ndarray:
        tensor = self._o3d.core.Tensor(points.astype(np.float32), dtype=self._o3d.core.Dtype.Float32)
        distances = self._scene.compute_signed_distance(tensor).numpy()
        return np.asarray(distances, dtype=np.float64).reshape(-1)

    def _init_kdtree(self) -> None:
        # VTK evaluates distance to the actual triangles, unlike a KD-tree of
        # sampled surface points which can overestimate clearance on large or
        # irregular STL triangles.  Keep the sample tree only as a last-resort
        # fallback for minimal environments without VTK.
        try:
            from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

            implicit_distance = vtkImplicitPolyDataDistance()
            implicit_distance.SetInput(self._mesh)
            self._implicit_distance = implicit_distance
            # An open STL has no well-defined inside/outside.  Its absolute
            # triangle distance is still a valid clearance field, whereas the
            # orientation-dependent VTK sign would incorrectly block one side.
            self._vtk_signed = self._signed_distance_valid
            self._backend = "vtk"
            return
        except Exception:
            self._implicit_distance = None
        from scipy.spatial import cKDTree

        surface = self._sample_surface_points()
        self._tree = cKDTree(surface)
        self._surface_points = surface
        self._closed_surface = None
        try:
            self._closed_surface = self._mesh.extract_surface(algorithm="dataset_surface").triangulate()
        except Exception:
            self._closed_surface = None

    def _sample_surface_points(self) -> np.ndarray:
        points = [self._vertices]
        tri = self._vertices[self._triangles]
        centroids = tri.mean(axis=1)
        points.append(centroids)
        mids = np.concatenate(
            [
                (tri[:, 0, :] + tri[:, 1, :]) * 0.5,
                (tri[:, 1, :] + tri[:, 2, :]) * 0.5,
                (tri[:, 2, :] + tri[:, 0, :]) * 0.5,
            ],
            axis=0,
        )
        points.append(mids)
        return np.ascontiguousarray(np.vstack(points), dtype=np.float64)

    def _distance_kdtree(self, points: np.ndarray) -> np.ndarray:
        if self._implicit_distance is not None:
            distances = np.asarray(
                [self._implicit_distance.EvaluateFunction(point) for point in points],
                dtype=np.float64,
            )
            return distances if self._vtk_signed else np.abs(distances)
        distances, _ = self._tree.query(points)
        distances = np.asarray(distances, dtype=np.float64)
        signs = self._inside_sign(points)
        return distances * signs

    def _inside_sign(self, points: np.ndarray) -> np.ndarray:
        if self._closed_surface is None:
            return np.ones(points.shape[0], dtype=np.float64)
        try:
            import pyvista as pv

            cloud = pv.PolyData(points)
            if hasattr(cloud, "select_interior_points"):
                selected = cloud.select_interior_points(
                    self._closed_surface,
                    tolerance=1e-8,
                    check_surface=False,
                )
            else:
                selected = cloud.select_enclosed_points(
                    self._closed_surface,
                    tolerance=1e-8,
                    check_surface=False,
                )
            inside = np.asarray(selected.point_data["SelectedPoints"], dtype=bool)
            signs = np.ones(points.shape[0], dtype=np.float64)
            signs[inside] = -1.0
            return signs
        except Exception:
            return np.ones(points.shape[0], dtype=np.float64)


__all__ = ["DistanceQueryResult", "EnvironmentDistanceField"]
