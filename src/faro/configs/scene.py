"""Scene geometry and placement authoring; tasks consume resolved numeric contacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pinocchio as pin

from faro.collision.collision_rules import CollisionRules
from faro.configs.objects import FixedBox, MovableBox, Plane
from faro.configs.robot import RobotConfig
from faro.configs.identifiers import InterfaceKey
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL, CONTACT_TYPE_SLIDING
from faro.utils.geom_utils import BOX_FACES

Face = tuple[str, str]


@dataclass(frozen=True)
class Pose:
    """World position and XYZW quaternion."""
    q: tuple[float, ...] = (0, 0, 0, 0, 0, 0, 1)

    @classmethod
    def identity(cls):
        return cls()


@dataclass(frozen=True)
class OnFace:
    """Place a box face against another face, in the supporting face's frame."""
    on: Face
    xy_yaw: tuple[float, float, float] = (0, 0, 0)
    self_face: str = "bottom"
    contact_type: int = 1


@dataclass(frozen=True)
class Standing:
    """Initialize above a face with the configured standing contacts."""
    on: Face
    xy_yaw: tuple[float, float, float] = (0, 0, 0)
    base_height: float | None = None
    qj: tuple[float, ...] | None = None


@dataclass(frozen=True)
class RobotConfiguration:
    q: tuple[float, ...]
    v: tuple[float, ...] | None = None


@dataclass(frozen=True)
class StateSpec:
    robots: Mapping[str, Standing | RobotConfiguration]
    objects: Mapping[str, Pose | OnFace] = field(default_factory=dict)
    # Explicit numerical contact tuples override contacts inferred from placement.
    contacts: Mapping[InterfaceKey, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class ResolvedRobotState:
    q: np.ndarray
    v: np.ndarray
    com: np.ndarray
    x: np.ndarray

    @property
    def qb(self):
        return self.q[:7]

    @property
    def qj(self):
        return self.q[7:]

    @property
    def mj_q(self):
        return np.concatenate((self.q[:3], self.q[[6, 3, 4, 5]], self.q[7:]))


@dataclass
class ResolvedObjectState:
    q: np.ndarray
    v: np.ndarray
    x: np.ndarray

    @property
    def mj_q(self):
        return np.concatenate((self.q[:3], self.q[[6, 3, 4, 5]]))


@dataclass
class ResolvedSceneState:
    robots: dict[str, ResolvedRobotState]
    fixed_env_objects: dict[str, ResolvedObjectState] = field(default_factory=dict)
    env_objects: dict[str, ResolvedObjectState] = field(default_factory=dict)
    contact_mode: dict[InterfaceKey, tuple[int, ...]] = field(default_factory=dict)


class FaceIndex:
    """The single source of global IDs, ownership, slices and permissions.

    Global IDs are one-based, fixed objects precede movable objects, and box
    local IDs follow BOX_FACES. Slices address zero-based solver arrays.
    """
    def __init__(self, objects):
        by_name, by_id, slices, ids, admitted = {}, {}, {}, {}, {}
        cursor = 1
        for name, cfg in objects.items():
            faces = ('surface',) if isinstance(cfg, Plane) else BOX_FACES
            slices[name] = slice(cursor - 1, cursor - 1 + len(faces))
            ids[name] = tuple(range(cursor, cursor + len(faces)))
            for local_id, face in enumerate(faces, 1):
                by_name[name, face] = cursor
                by_id[cursor] = (name, local_id)
                admitted[cursor] = tuple(dict.fromkeys(
                    (*cfg.contact_types, *cfg.face_contact_types.get(face, ()))
                ))
                cursor += 1
        self.by_name = MappingProxyType(by_name)
        self.by_id = MappingProxyType(by_id)
        self.slices = MappingProxyType(slices)
        self.ids_by_object = MappingProxyType(ids)
        self.admitted = MappingProxyType(admitted)

    def id(self, name: str, face: str) -> int:
        try:
            return self.by_name[name, face]
        except KeyError as exc:
            raise ValueError(f"Unknown scene face: {name}.{face}") from exc

    def owner(self, feature_id: int) -> str:
        try:
            return self.by_id[feature_id][0]
        except KeyError as exc:
            raise ValueError(f"Unknown global feature ID: {feature_id}") from exc


class FeatureIndex:
    """Global contact targets with typed owners; object face IDs stay stable."""

    def __init__(self, scene):
        by_id = {i: (("obj", name), local_id)
                 for i, (name, local_id) in scene.faces.by_id.items()}
        admitted = dict(scene.faces.admitted)
        ee_ids = {}
        if scene.include_robot_ees:
            for key in scene.robot_interfaces:
                feature_id = len(by_id) + 1
                by_id[feature_id] = (key, 1)
                admitted[feature_id] = (CONTACT_TYPE_UNILATERAL, CONTACT_TYPE_SLIDING)
                ee_ids[key] = feature_id
        self.by_id = MappingProxyType(by_id)
        self.admitted = MappingProxyType(admitted)
        self.ee_ids = MappingProxyType(ee_ids)

    def owner(self, feature_id):
        try:
            return self.by_id[feature_id][0]
        except KeyError as exc:
            raise ValueError(f"Unknown global feature ID: {feature_id}") from exc


@dataclass(frozen=True)
class Scene:
    name: str
    robots: Mapping[str, RobotConfig]
    objects: Mapping[str, Plane | FixedBox | MovableBox]
    include_robot_ees: bool = False
    collision_rules: CollisionRules = field(default_factory=CollisionRules)

    def __post_init__(self):
        if not isinstance(self.include_robot_ees, bool):
            raise TypeError("include_robot_ees must be a bool.")
        if not isinstance(self.collision_rules, CollisionRules):
            raise TypeError("collision_rules must be CollisionRules.")
        robots = dict(self.robots)
        if not robots:
            raise ValueError("A scene requires at least one robot.")
        for name, cfg in robots.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid robot instance ID: {name!r}")
            if not isinstance(cfg, RobotConfig):
                raise TypeError(f"Expected RobotConfig for robot {name!r}.")
        object.__setattr__(self, 'robots', MappingProxyType(robots))
        objects = dict(self.objects)
        for name, cfg in objects.items():
            if type(cfg) not in (Plane, FixedBox, MovableBox):
                raise TypeError(f"Unsupported scene object '{name}': {type(cfg).__name__}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid or conflicting scene object name: {name!r}")
        # Fixed objects come first; preserve insertion order within each group.
        # This determines feature IDs and movable-object qpos order.
        fixed = {n: c for n, c in objects.items() if not isinstance(c, MovableBox)}
        movable = {n: c for n, c in objects.items() if isinstance(c, MovableBox)}
        object.__setattr__(self, 'objects', MappingProxyType({**fixed, **movable}))

    @cached_property
    def fixed_env_objects(self):
        return MappingProxyType({n: c for n, c in self.objects.items() if not isinstance(c, MovableBox)})

    @cached_property
    def env_objects(self):
        return MappingProxyType({n: c for n, c in self.objects.items() if isinstance(c, MovableBox)})

    @cached_property
    def faces(self):
        return FaceIndex(self.objects)

    @cached_property
    def features(self):
        return FeatureIndex(self)

    @property
    def env_obj_order(self):
        return tuple(self.objects)

    @property
    def total_env_features(self):
        return len(self.features.by_id)

    @cached_property
    def robot_interfaces(self):
        return MappingProxyType({("ee", rid, patch): bounds
                                 for rid, cfg in self.robots.items()
                                 for patch, bounds in cfg.patch_ees.items()})

    @property
    def active_patches(self):
        return (*self.robot_interfaces, *(("obj", name) for name in self.env_objects))

    def face_id(self, name: str, face: str) -> int:
        """Authoring convenience; returns the global integer used by tasks."""
        return self.faces.id(name, face)

    def ee_feature_id(self, robot_id: str, patch: str) -> int:
        """Return an EE target ID; requires include_robot_ees=True."""
        try:
            return self.features.ee_ids["ee", robot_id, patch]
        except KeyError as exc:
            raise ValueError(f"EE feature unavailable: {(robot_id, patch)!r}; "
                             "enable include_robot_ees and use a configured patch.") from exc

    def contact_mode(self, spec: StateSpec) -> dict[InterfaceKey, tuple[int, ...]]:
        mode = dict.fromkeys(self.active_patches, ())
        for rid, placement in spec.robots.items():
            if isinstance(placement, Standing):
                contact = (self.face_id(*placement.on), 1)
                mode.update({("ee", rid, name): contact
                             for name in self.robots[rid].standing_contacts})
        for name in self.env_objects:
            placement = spec.objects[name]
            if isinstance(placement, OnFace):
                mode["obj", name] = (BOX_FACES.index(placement.self_face) + 1,
                              self.face_id(*placement.on), placement.contact_type)
        if unknown := spec.contacts.keys() - mode.keys():
            raise ValueError(f"Unknown contact interfaces: {sorted(unknown)}")
        mode.update(spec.contacts)
        for name, contact in mode.items():
            cnt = tuple(int(x) for x in contact)
            expected = 3 if name[0] == "obj" else 2
            if len(cnt) not in (0, expected):
                raise ValueError(f"Contact for '{name}' must be empty or have {expected} entries.")
            if cnt:
                self.features.owner(cnt[-2])
                if expected == 3 and not 1 <= cnt[0] <= 6:
                    raise ValueError(f"Invalid box self face: {cnt[0]}")
            mode[name] = cnt
        from faro.utils.contact_validation import validate_contact_mode
        validate_contact_mode(self, mode)
        return mode

    def resolve_objects(self, placements: Mapping[str, Pose | OnFace]):
        """Resolve object placements using the same geometry for states and goals."""
        missing = self.objects.keys() - placements.keys()
        extra = placements.keys() - self.objects.keys()
        if missing or extra:
            raise ValueError(f"Object placements differ from scene: missing={sorted(missing)}, unknown={sorted(extra)}")
        resolved, visiting = {}, set()

        def resolve_object(name):
            if name in resolved:
                return resolved[name]
            if name in visiting:
                raise ValueError(f"Cyclic placement dependency at '{name}'.")
            visiting.add(name)
            placement = placements[name]
            if isinstance(placement, Pose):
                q = _configuration(placement.q, 7)
            elif isinstance(placement, OnFace):
                target, _ = placement.on
                self.face_id(*placement.on)  # Validate before traversing the dependency.
                if name in self.fixed_env_objects and target in self.env_objects:
                    raise ValueError(f"Fixed object '{name}' cannot be placed on movable object '{target}'.")
                if isinstance(self.objects[name], Plane):
                    raise ValueError("A plane requires a world Pose.")
                support = pin.XYZQUATToSE3(resolve_object(target).q) * self._local_face(*placement.on)
                flip = pin.SE3(np.diag([1., -1., -1.]), np.zeros(3))
                pose = support * _relative_pose(placement.xy_yaw) * flip * self._local_face(name, placement.self_face).inverse()
                q = pin.SE3ToXYZQUAT(pose)
            else:
                raise TypeError(f"Expected Pose or OnFace for '{name}'.")
            resolved[name] = ResolvedObjectState(q, np.zeros(6), np.r_[q, np.zeros(6)])
            visiting.remove(name)
            return resolved[name]

        for name in self.objects:
            resolve_object(name)
        return resolved

    def resolve(self, spec: StateSpec) -> ResolvedSceneState:
        resolved = self.resolve_objects(spec.objects)
        if spec.robots.keys() != self.robots.keys():
            missing = self.robots.keys() - spec.robots.keys()
            extra = spec.robots.keys() - self.robots.keys()
            raise ValueError(f"Robot placements differ from scene: missing={sorted(missing)}, unknown={sorted(extra)}")
        robots = {}
        for rid, cfg in self.robots.items():
            placement = spec.robots[rid]
            if isinstance(placement, Standing):
                self.face_id(*placement.on)
                support = pin.XYZQUATToSE3(resolved[placement.on[0]].q) * self._local_face(*placement.on)
                height = cfg.default_standing_height if placement.base_height is None else placement.base_height
                if height is None:
                    raise ValueError("Standing requires base_height or a robot default_standing_height.")
                base = support * _relative_pose(placement.xy_yaw, height)
                joints = cfg.default_qj if placement.qj is None else placement.qj
                q = _configuration(np.r_[pin.SE3ToXYZQUAT(base), joints], cfg.nq)
                v = np.zeros(cfg.nv)
            elif isinstance(placement, RobotConfiguration):
                q = _configuration(placement.q, cfg.nq)
                v = np.zeros(cfg.nv) if placement.v is None else np.asarray(placement.v, dtype=float).copy()
                if v.shape != (cfg.nv,) or not np.all(np.isfinite(v)):
                    raise ValueError(f"Robot {rid!r} requires a finite velocity of length {cfg.nv}.")
            else:
                raise TypeError(f"Expected Standing or RobotConfiguration for robot {rid!r}.")
            data = cfg.pin_model.createData()
            com = np.asarray(pin.centerOfMass(cfg.pin_model, data, q)).copy()
            momentum = np.asarray(pin.computeCentroidalMomentum(cfg.pin_model, data, q, v).vector).copy()
            robots[rid] = ResolvedRobotState(q, v, com, np.r_[q, v, com, momentum])
        return ResolvedSceneState(
            robots=robots,
            fixed_env_objects={n: resolved[n] for n in self.fixed_env_objects},
            env_objects={n: resolved[n] for n in self.env_objects},
            contact_mode=self.contact_mode(spec),
        )

    def _local_face(self, name, face):
        feature_id = self.face_id(name, face)
        _, local_id = self.faces.by_id[feature_id]
        feature = self.objects[name].features[:, local_id - 1]
        return pin.SE3(feature[3:12].reshape(3, 3, order='F'), feature[:3])

    def get_scene_definition(self):
        objects, patches = {}, {}
        for i, (name, cfg) in enumerate(self.objects.items(), 1):
            ids = self.faces.ids_by_object[name]
            shape = 'plane' if isinstance(cfg, Plane) else 'box'
            kind = 'moving' if isinstance(cfg, MovableBox) else 'fixed'
            objects[name] = dict(id=i, type=f'{kind}_{shape}',
                features=f'{len(ids)} ({", ".join(map(str, ids))})', feature_indices=list(ids))
        for i, name in enumerate(self.active_patches, 1):
            movable = name[0] == "obj"
            patches[name] = dict(id=i, type='object_patch' if movable else 'robot_ee',
                tuple_def='(self, env, contact_type)' if movable else '(env, contact_type)')
        result = dict(environment_objects=objects, active_patches=patches)
        if self.include_robot_ees:
            result['ee_features'] = dict(self.features.ee_ids)
        return result

    def print_scene_info(self):
        print(f'Scene: {self.name}')
        for (name, face), feature_id in self.faces.by_name.items():
            print(f'  {feature_id:3}: {name}.{face}')
        for key, feature_id in self.features.ee_ids.items():
            print(f'  {feature_id:3}: {key!r}')
        print('Contact interfaces:', ', '.join(map(str, self.active_patches)))


def _configuration(q, size):
    q = np.asarray(q, dtype=float)
    if q.shape != (size,) or not np.all(np.isfinite(q)):
        raise ValueError(f"Expected a finite configuration of length {size}.")
    if not np.isclose(np.linalg.norm(q[3:7]), 1, atol=1e-5):
        raise ValueError("Configuration quaternion must have unit length (XYZW).")
    return q.copy()


def _relative_pose(xy_yaw, height=0):
    x, y, yaw = xy_yaw
    c, s = np.cos(yaw), np.sin(yaw)
    return pin.SE3(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), np.array([x, y, height], dtype=float))
