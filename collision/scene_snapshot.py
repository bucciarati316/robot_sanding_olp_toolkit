"""Immutable, actor-free collision scene input for planning and validation.

The GUI owns PyVista actors and its live FCL objects.  A planner must never
reuse those mutable objects from a worker thread, so this module captures only
plain mesh arrays, transforms, attachment data, and a deterministic hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import numpy as np


class SceneSnapshotBuildError(RuntimeError):
    """A scene cannot be safely frozen for collision planning."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = str(code)
        self.detail = str(detail)


def _frozen_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是有限的 (4,4) 齐次变换矩阵")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _frozen_array(value: np.ndarray, *, dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _frozen_vector3(value: np.ndarray, name: str) -> np.ndarray:
    vector = _frozen_array(value, dtype=np.float64, name=name).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    vector.setflags(write=False)
    return vector


def _update_digest_with_array(digest, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())


@dataclass(frozen=True)
class MeshGeometry:
    """Triangulated mesh data expressed in its local coordinate frame."""

    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        vertices = _frozen_array(self.vertices, dtype=np.float64, name="vertices")
        triangles = _frozen_array(self.triangles, dtype=np.int32, name="triangles")
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
            raise ValueError("MeshGeometry.vertices 必须是非空 (N,3) 数组")
        if triangles.ndim != 2 or triangles.shape[1:] != (3,) or len(triangles) == 0:
            raise ValueError("MeshGeometry.triangles 必须是非空 (M,3) 数组")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("MeshGeometry.vertices 包含非有限值")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("MeshGeometry.triangles 包含越界顶点索引")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)

    @classmethod
    def from_pyvista(cls, mesh) -> "MeshGeometry":
        """Copy a PyVista mesh into plain triangulated arrays."""
        triangulated = mesh.triangulate()
        cells = np.asarray(triangulated.faces, dtype=np.int64)
        triangles: list[list[int]] = []
        index = 0
        while index < len(cells):
            vertex_count = int(cells[index])
            index += 1
            if vertex_count != 3 or index + vertex_count > len(cells):
                raise ValueError("PyVista 网格三角化结果无效")
            triangles.append(cells[index:index + vertex_count].tolist())
            index += vertex_count
        return cls(
            vertices=np.asarray(triangulated.points, dtype=np.float64),
            triangles=np.asarray(triangles, dtype=np.int32),
        )


@dataclass(frozen=True)
class RobotLinkSpec:
    """One robot collision mesh and its fixed transform inside a link frame."""

    name: str
    geometry: MeshGeometry
    local_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RobotLinkSpec.name 不能为空")
        object.__setattr__(
            self, "local_transform", _frozen_matrix(self.local_transform, "local_transform")
        )


@dataclass(frozen=True)
class StaticCollisionObjectSpec:
    """A non-moving CAD or environment collision object in world coordinates."""

    name: str
    geometry: MeshGeometry
    world_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("StaticCollisionObjectSpec.name 不能为空")
        object.__setattr__(
            self, "world_transform", _frozen_matrix(self.world_transform, "world_transform")
        )


@dataclass(frozen=True)
class AttachedBodySpec:
    """A collision mesh rigidly attached to one robot collision link."""

    name: str
    parent_link: str
    geometry: MeshGeometry
    mount_transform: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        if not self.name or not self.parent_link:
            raise ValueError("AttachedBodySpec.name 和 parent_link 均不能为空")
        object.__setattr__(
            self, "mount_transform", _frozen_matrix(self.mount_transform, "mount_transform")
        )


@dataclass(frozen=True)
class ContactRule:
    """A narrowly scoped exception for an intentional tool-workpiece contact.

    The region is an axis-aligned box in the *static object's local frame*.
    A rule is never inferred from names: the caller must specify the attached
    body, target object, permitted segment types, and the permitted region.
    With no rules (the production default), every FCL collision is unsafe.
    """

    attached_body: str
    static_object: str
    allowed_segment_types: tuple[str, ...]
    static_local_region_min: np.ndarray
    static_local_region_max: np.ndarray

    def __post_init__(self) -> None:
        if not self.attached_body or not self.static_object:
            raise ValueError("ContactRule body and static object names are required")
        segment_types = tuple(
            sorted({str(value).strip().lower() for value in self.allowed_segment_types if str(value).strip()})
        )
        if not segment_types:
            raise ValueError("ContactRule requires at least one allowed segment type")
        minimum = _frozen_vector3(self.static_local_region_min, "static_local_region_min")
        maximum = _frozen_vector3(self.static_local_region_max, "static_local_region_max")
        if np.any(minimum > maximum):
            raise ValueError("ContactRule region minimum must not exceed maximum")
        object.__setattr__(self, "allowed_segment_types", segment_types)
        object.__setattr__(self, "static_local_region_min", minimum)
        object.__setattr__(self, "static_local_region_max", maximum)


@dataclass(frozen=True)
class SceneSnapshot:
    """Pure immutable collision input that can be reconstructed per worker."""

    robot_urdf_path: str
    robot_base_transform: np.ndarray
    robot_links: tuple[RobotLinkSpec, ...]
    static_objects: tuple[StaticCollisionObjectSpec, ...]
    attached_bodies: tuple[AttachedBodySpec, ...]
    contact_rules: tuple[ContactRule, ...] = ()
    ignored_link_pairs: tuple[tuple[str, str], ...] = ()
    scene_version: int = 0
    schema_version: str = "stable2d-v1"
    scene_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.robot_links:
            raise ValueError("SceneSnapshot 至少需要一个机器人碰撞 link")
        object.__setattr__(
            self,
            "robot_base_transform",
            _frozen_matrix(self.robot_base_transform, "robot_base_transform"),
        )
        links = tuple(self.robot_links)
        statics = tuple(self.static_objects)
        attached = tuple(self.attached_bodies)
        contact_rules = tuple(self.contact_rules)
        if len({item.name for item in links}) != len(links):
            raise ValueError("SceneSnapshot.robot_links 名称必须唯一")
        if len({item.name for item in statics}) != len(statics):
            raise ValueError("SceneSnapshot.static_objects 名称必须唯一")
        if len({item.name for item in attached}) != len(attached):
            raise ValueError("SceneSnapshot.attached_bodies 名称必须唯一")
        link_names = {item.name for item in links}
        missing_parents = {item.parent_link for item in attached} - link_names
        if missing_parents:
            raise ValueError(f"附着体父 link 不存在: {sorted(missing_parents)}")
        attached_names = {item.name for item in attached}
        static_names = {item.name for item in statics}
        invalid_rules = [
            (rule.attached_body, rule.static_object)
            for rule in contact_rules
            if rule.attached_body not in attached_names
            or rule.static_object not in static_names
        ]
        if invalid_rules:
            raise ValueError(f"ContactRule references unknown bodies: {invalid_rules}")
        rule_keys = [(rule.attached_body, rule.static_object) for rule in contact_rules]
        if len(set(rule_keys)) != len(rule_keys):
            raise ValueError("only one ContactRule is allowed per attached/static pair")
        pairs = tuple(
            sorted(
                tuple(sorted((str(a), str(b))))
                for a, b in self.ignored_link_pairs
                if str(a) and str(b) and str(a) != str(b)
            )
        )
        object.__setattr__(self, "robot_links", links)
        object.__setattr__(self, "static_objects", statics)
        object.__setattr__(self, "attached_bodies", attached)
        object.__setattr__(self, "contact_rules", contact_rules)
        object.__setattr__(self, "ignored_link_pairs", pairs)
        object.__setattr__(self, "scene_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        digest = sha256()
        digest.update(self.schema_version.encode("utf-8"))
        digest.update(str(int(self.scene_version)).encode("ascii"))
        digest.update(self.robot_urdf_path.encode("utf-8"))
        _update_digest_with_array(digest, self.robot_base_transform)
        for item in sorted(self.robot_links, key=lambda value: value.name):
            digest.update(b"robot-link")
            digest.update(item.name.encode("utf-8"))
            _update_digest_with_array(digest, item.geometry.vertices)
            _update_digest_with_array(digest, item.geometry.triangles)
            _update_digest_with_array(digest, item.local_transform)
        for item in sorted(self.static_objects, key=lambda value: value.name):
            digest.update(b"static")
            digest.update(item.name.encode("utf-8"))
            _update_digest_with_array(digest, item.geometry.vertices)
            _update_digest_with_array(digest, item.geometry.triangles)
            _update_digest_with_array(digest, item.world_transform)
        for item in sorted(self.attached_bodies, key=lambda value: value.name):
            digest.update(b"attached")
            digest.update(item.name.encode("utf-8"))
            digest.update(item.parent_link.encode("utf-8"))
            _update_digest_with_array(digest, item.geometry.vertices)
            _update_digest_with_array(digest, item.geometry.triangles)
            _update_digest_with_array(digest, item.mount_transform)
        for rule in sorted(
            self.contact_rules,
            key=lambda value: (value.attached_body, value.static_object),
        ):
            digest.update(b"contact-rule")
            digest.update(rule.attached_body.encode("utf-8"))
            digest.update(rule.static_object.encode("utf-8"))
            digest.update(repr(rule.allowed_segment_types).encode("utf-8"))
            _update_digest_with_array(digest, rule.static_local_region_min)
            _update_digest_with_array(digest, rule.static_local_region_max)
        for pair in self.ignored_link_pairs:
            digest.update(repr(pair).encode("utf-8"))
        return digest.hexdigest()

    def metadata(self) -> dict:
        """Serializable audit information without duplicating mesh payloads."""
        return {
            "schema_version": self.schema_version,
            "scene_version": int(self.scene_version),
            "scene_hash": self.scene_hash,
            "robot_urdf_path": self.robot_urdf_path,
            "robot_links": [item.name for item in self.robot_links],
            "static_objects": [item.name for item in self.static_objects],
            "attached_bodies": [item.name for item in self.attached_bodies],
            "contact_rules": [
                {
                    "attached_body": rule.attached_body,
                    "static_object": rule.static_object,
                    "allowed_segment_types": list(rule.allowed_segment_types),
                    "static_local_region_min": rule.static_local_region_min.tolist(),
                    "static_local_region_max": rule.static_local_region_max.tolist(),
                }
                for rule in self.contact_rules
            ],
            "ignored_link_pairs": [list(pair) for pair in self.ignored_link_pairs],
        }
