"""Fail-closed configuration and edge validation built from ``SceneSnapshot``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from collision.scene_snapshot import MeshGeometry, SceneSnapshot


_AUTO_FCL = object()


def _frozen_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    result = np.array(vector, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _frozen_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("body transform 必须是有限 (4,4) 矩阵")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _as_fk_matrix(value) -> np.ndarray:
    matrix = getattr(value, "homogeneous", value)
    return _frozen_matrix(matrix)


@dataclass(frozen=True)
class StateValidity:
    """Structured, fail-closed result for one configuration."""

    valid: bool
    scene_hash: str
    checked_q: np.ndarray
    collision_pairs: tuple[tuple[str, str], ...] = ()
    allowed_contact_pairs: tuple[tuple[str, str], ...] = ()
    error_code: Optional[str] = None
    detail: str = ""
    body_transforms: tuple[tuple[str, np.ndarray], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checked_q", _frozen_vector(self.checked_q))
        pairs = tuple(tuple(sorted((str(a), str(b)))) for a, b in self.collision_pairs)
        object.__setattr__(self, "collision_pairs", tuple(sorted(set(pairs))))
        allowed_pairs = tuple(
            tuple(sorted((str(a), str(b)))) for a, b in self.allowed_contact_pairs
        )
        object.__setattr__(self, "allowed_contact_pairs", tuple(sorted(set(allowed_pairs))))
        transforms = tuple(
            (str(name), _frozen_matrix(transform))
            for name, transform in self.body_transforms
        )
        object.__setattr__(self, "body_transforms", transforms)

    def transform_for(self, name: str) -> np.ndarray:
        for body_name, transform in self.body_transforms:
            if body_name == name:
                return transform.copy()
        raise KeyError(name)

    def as_dict(self) -> dict:
        return {
            "valid": bool(self.valid),
            "scene_hash": self.scene_hash,
            "collision_pairs": [list(pair) for pair in self.collision_pairs],
            "allowed_contact_pairs": [list(pair) for pair in self.allowed_contact_pairs],
            "error_code": self.error_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EdgeValidity:
    """Result of controlled-resolution interpolation over one joint-space edge."""

    valid: bool
    scene_hash: str
    sample_count: int
    first_invalid_sample: Optional[int] = None
    first_invalid: Optional[StateValidity] = None


class SnapshotCollisionService:
    """A worker-local exact FCL service constructed only from a snapshot.

    It intentionally owns new FCL objects rather than mutating GUI collision
    objects.  Missing FCL, mesh construction, FK, transform, or collision
    errors all return ``valid=False``; callers therefore cannot mistake an
    unavailable service for a collision-free configuration.
    """

    def __init__(
        self,
        snapshot: SceneSnapshot,
        fk_provider: Callable[[np.ndarray], dict],
        *,
        fcl_module=_AUTO_FCL,
    ) -> None:
        self.snapshot = snapshot
        self._fk_provider = fk_provider
        self._fcl = None
        self._request = None
        self._robot_objects: dict[str, object] = {}
        self._static_objects: dict[str, object] = {}
        self._attached_objects: dict[str, object] = {}
        self._contact_rules = {
            (rule.attached_body, rule.static_object): rule
            for rule in snapshot.contact_rules
        }
        self._build_error: Optional[tuple[str, str]] = None

        if fcl_module is _AUTO_FCL:
            try:
                import fcl as imported_fcl
            except Exception as exc:
                self._build_error = ("fcl_unavailable", str(exc))
                return
            self._fcl = imported_fcl
        elif fcl_module is None:
            self._build_error = ("fcl_unavailable", "FCL backend was not supplied")
            return
        else:
            self._fcl = fcl_module

        try:
            self._request = self._fcl.CollisionRequest(
                num_max_contacts=1,
                enable_contact=False,
            )
            self._robot_objects = {
                item.name: self._build_fcl_object(item.geometry)
                for item in snapshot.robot_links
            }
            self._static_objects = {
                item.name: self._build_fcl_object(item.geometry)
                for item in snapshot.static_objects
            }
            self._attached_objects = {
                item.name: self._build_fcl_object(item.geometry)
                for item in snapshot.attached_bodies
            }
            for item in snapshot.static_objects:
                self._set_transform(self._static_objects[item.name], item.world_transform)
        except Exception as exc:
            self._build_error = ("build_failed", str(exc))
            self._robot_objects.clear()
            self._static_objects.clear()
            self._attached_objects.clear()

    @property
    def available(self) -> bool:
        return self._build_error is None

    @property
    def failure(self) -> Optional[tuple[str, str]]:
        return self._build_error

    def _build_fcl_object(self, geometry: MeshGeometry):
        vertices = np.ascontiguousarray(geometry.vertices, dtype=np.float64)
        triangles = np.ascontiguousarray(geometry.triangles, dtype=np.int32)
        bvh = self._fcl.BVHModel()
        bvh.beginModel(len(vertices), len(triangles))
        bvh.addSubModel(vertices, triangles)
        bvh.endModel()
        return self._fcl.CollisionObject(
            bvh,
            self._fcl.Transform(np.eye(3), np.zeros(3)),
        )

    def _set_transform(self, obj, transform: np.ndarray) -> None:
        matrix = _frozen_matrix(transform)
        obj.setTransform(
            self._fcl.Transform(
                matrix[:3, :3].astype(np.float64),
                matrix[:3, 3].astype(np.float64),
            )
        )

    def _invalid(self, q, code: str, detail: str, *, transforms=()) -> StateValidity:
        return StateValidity(
            valid=False,
            scene_hash=self.snapshot.scene_hash,
            checked_q=np.asarray(q, dtype=np.float64),
            error_code=code,
            detail=detail,
            body_transforms=tuple(transforms),
        )

    def _collides(self, first, second) -> bool:
        result = self._fcl.CollisionResult()
        return bool(self._fcl.collide(first, second, self._request, result) > 0)

    @staticmethod
    def _segment_type_value(segment_type) -> str:
        value = getattr(segment_type, "value", segment_type)
        return str(value).strip().lower()

    def _allows_contact(
        self,
        body_name: str,
        static_name: str,
        *,
        segment_type,
        body_transform: np.ndarray,
        static_transform: np.ndarray,
    ) -> bool:
        """Allow only an explicitly declared PROCESS-region contact."""
        rule = self._contact_rules.get((body_name, static_name))
        if rule is None or self._segment_type_value(segment_type) not in rule.allowed_segment_types:
            return False
        body_origin = np.ones(4, dtype=np.float64)
        body_origin[:3] = body_transform[:3, 3]
        static_local = np.linalg.inv(static_transform) @ body_origin
        point = static_local[:3]
        tolerance = 1e-9
        return bool(
            np.all(point >= rule.static_local_region_min - tolerance)
            and np.all(point <= rule.static_local_region_max + tolerance)
        )

    def check_configuration(self, q: np.ndarray, *, segment_type=None) -> StateValidity:
        """Check robot, attached bodies, and statics without GUI actor state."""
        checked_q = np.asarray(q, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(checked_q)):
            return self._invalid(checked_q, "invalid_configuration", "关节向量包含非有限值")
        if self._build_error is not None:
            code, detail = self._build_error
            return self._invalid(checked_q, code, detail)

        try:
            fk_result = self._fk_provider(checked_q.copy())
            link_transforms: dict[str, np.ndarray] = {}
            for link in self.snapshot.robot_links:
                if link.name not in fk_result:
                    return self._invalid(
                        checked_q,
                        "fk_missing_link",
                        f"FK 未返回碰撞 link: {link.name}",
                    )
                transform = (
                    self.snapshot.robot_base_transform
                    @ _as_fk_matrix(fk_result[link.name])
                    @ link.local_transform
                )
                self._set_transform(self._robot_objects[link.name], transform)
                link_transforms[link.name] = transform

            body_transforms: dict[str, np.ndarray] = dict(link_transforms)
            for body in self.snapshot.attached_bodies:
                transform = link_transforms[body.parent_link] @ body.mount_transform
                self._set_transform(self._attached_objects[body.name], transform)
                body_transforms[body.name] = transform

            collisions: set[tuple[str, str]] = set()
            allowed_contacts: set[tuple[str, str]] = set()
            link_names = [item.name for item in self.snapshot.robot_links]
            ignored = {frozenset(pair) for pair in self.snapshot.ignored_link_pairs}
            for index, first_name in enumerate(link_names):
                for second_name in link_names[index + 1:]:
                    if frozenset({first_name, second_name}) in ignored:
                        continue
                    if self._collides(
                        self._robot_objects[first_name], self._robot_objects[second_name]
                    ):
                        collisions.add(tuple(sorted((first_name, second_name))))

            for link_name in link_names:
                for static in self.snapshot.static_objects:
                    if self._collides(self._robot_objects[link_name], self._static_objects[static.name]):
                        collisions.add(tuple(sorted((link_name, static.name))))

            for body in self.snapshot.attached_bodies:
                body_object = self._attached_objects[body.name]
                for static in self.snapshot.static_objects:
                    if self._collides(body_object, self._static_objects[static.name]):
                        pair = tuple(sorted((body.name, static.name)))
                        if self._allows_contact(
                            body.name,
                            static.name,
                            segment_type=segment_type,
                            body_transform=body_transforms[body.name],
                            static_transform=static.world_transform,
                        ):
                            allowed_contacts.add(pair)
                        else:
                            collisions.add(pair)
                for link_name in link_names:
                    if link_name == body.parent_link:
                        continue
                    if self._collides(body_object, self._robot_objects[link_name]):
                        collisions.add(tuple(sorted((body.name, link_name))))

            ordered_transforms = tuple(sorted(body_transforms.items(), key=lambda item: item[0]))
            return StateValidity(
                valid=not collisions,
                scene_hash=self.snapshot.scene_hash,
                checked_q=checked_q,
                collision_pairs=tuple(sorted(collisions)),
                allowed_contact_pairs=tuple(sorted(allowed_contacts)),
                error_code="collision_detected" if collisions else None,
                detail="碰撞体重叠" if collisions else "",
                body_transforms=ordered_transforms,
            )
        except Exception as exc:
            return self._invalid(checked_q, "check_failed", str(exc))

    def check_edge(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        *,
        max_joint_step: float = 0.05,
        segment_type=None,
    ) -> EdgeValidity:
        """Validate all controlled joint-space samples on a candidate edge."""
        start = np.asarray(start_q, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal_q, dtype=np.float64).reshape(-1)
        if start.shape != goal.shape or start.size == 0:
            invalid = self._invalid(start, "invalid_edge", "起终点维度不一致或为空")
            return EdgeValidity(False, self.snapshot.scene_hash, 0, 0, invalid)
        if not np.isfinite(max_joint_step) or max_joint_step <= 0:
            invalid = self._invalid(start, "invalid_edge_resolution", "max_joint_step 必须为正")
            return EdgeValidity(False, self.snapshot.scene_hash, 0, 0, invalid)
        sample_count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step))) + 1
        for index, ratio in enumerate(np.linspace(0.0, 1.0, sample_count)):
            result = self.check_configuration(
                start + ratio * (goal - start),
                segment_type=segment_type,
            )
            if not result.valid:
                return EdgeValidity(
                    False,
                    self.snapshot.scene_hash,
                    sample_count,
                    index,
                    result,
                )
        return EdgeValidity(True, self.snapshot.scene_hash, sample_count)
