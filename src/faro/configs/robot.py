"""Robot properties consumed by scenes, tasks, and formulations."""
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import mujoco
import numpy as np
import pinocchio as pin

from faro.collision.robot_collision_model import RobotCollisionModel


@dataclass(frozen=True)
class ActuatorLimits:
    """Symmetric joint effort and no-load speed (Nm, rad/s; N, m/s for sliders)."""
    torque: float
    velocity: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.torque, self.velocity)):
            raise ValueError("Actuator torque and velocity limits must be positive and finite.")


@dataclass
class RobotConfig:
    """A free-flyer followed by scalar joints, in Pinocchio configuration order.

    Configure before constructing tasks; rebuild formulations after changes.
    Actuator limits are resolved by joint name, never dictionary insertion order.
    """
    name: str
    pin_model: pin.Model
    mj_spec: mujoco.MjSpec
    actuator_limits: Mapping[str, ActuatorLimits]
    collision_model: RobotCollisionModel
    patch_ees: Mapping[str, np.ndarray] = field(default_factory=dict)
    default_qj: Sequence[float] | None = None
    standing_contacts: tuple[str, ...] = ()
    default_standing_height: float | None = None
    switch_contacts: tuple[str, ...] = ()
    plane_contact_exemptions: Mapping[str, str] = field(default_factory=dict)
    contact_capsules: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    joint_names: tuple[str, ...] = field(init=False)
    tau_max: np.ndarray = field(init=False, repr=False)
    vel_max: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        model = self.pin_model
        if (model.njoints < 2 or model.joints[1].shortname() != "JointModelFreeFlyer"
                or model.parents[1] != 0
                or (model.joints[1].idx_q, model.joints[1].idx_v) != (0, 0)):
            raise ValueError("Robot model must start with a free-flyer joint.")
        for i, joint in enumerate(list(model.joints)[2:]):
            if (model.parents[i + 2] == 0
                    or (joint.nq, joint.nv, joint.idx_q, joint.idx_v) != (1, 1, 7 + i, 6 + i)):
                raise ValueError("Robot model must have scalar joints after the free-flyer.")
        self.joint_names = tuple(model.names)[2:]
        missing = set(self.joint_names) - self.actuator_limits.keys()
        extra = self.actuator_limits.keys() - set(self.joint_names)
        if missing or extra:
            raise ValueError(f"Actuator limits differ from robot joints: missing={sorted(missing)}, unknown={sorted(extra)}")
        if not all(isinstance(v, ActuatorLimits) for v in self.actuator_limits.values()):
            raise TypeError("actuator_limits values must be ActuatorLimits.")
        self.actuator_limits = dict(self.actuator_limits)
        self.tau_max = np.array([self.actuator_limits[n].torque for n in self.joint_names], dtype=float)
        self.vel_max = np.array([self.actuator_limits[n].velocity for n in self.joint_names], dtype=float)
        self.tau_max.setflags(write=False)
        self.vel_max.setflags(write=False)

        qj = pin.neutral(model)[7:] if self.default_qj is None else self.default_qj
        qj = np.asarray(qj, dtype=float)
        if qj.shape != (self.nj,) or not np.all(np.isfinite(qj)):
            raise ValueError(f"Robot '{self.name}' default_qj must be finite with length {self.nj}.")
        self.default_qj = qj.tolist()
        if (np.any(np.isnan(self.q_min)) or np.any(np.isnan(self.q_max))
                or np.any(self.q_min > self.q_max)):
            raise ValueError("Robot joint position bounds must be ordered and not NaN.")

        self.patch_ees = {n: np.asarray(v, dtype=float).copy() for n, v in self.patch_ees.items()}
        for name, bounds in self.patch_ees.items():
            if not model.existFrame(name):
                raise ValueError(f"Unknown contact frame: {name!r}")
            if bounds.shape != (4,) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
                raise ValueError(f"Contact patch '{name}' requires four finite nonnegative bounds.")
        for label in ("standing_contacts", "switch_contacts"):
            names = tuple(getattr(self, label))
            if len(set(names)) != len(names) or set(names) - self.patch_ees.keys():
                raise ValueError(f"{label} must contain unique configured contact patches.")
            setattr(self, label, names)
        if self.default_standing_height is not None and not np.isfinite(self.default_standing_height):
            raise ValueError("default_standing_height must be finite.")

        capsules = self.collision_model.capsules
        for name, capsule in capsules.items():
            for frame in (capsule.frame, capsule.parent_frame):
                if not model.existFrame(frame):
                    raise ValueError(f"Unknown collision frame {frame!r} for capsule {name!r}")
        for a, b in self.collision_model.self_collision_pairs:
            if a not in capsules or b not in capsules:
                raise ValueError(f"Unknown capsule in self-collision pair: {(a, b)}")
        self.plane_contact_exemptions = dict(self.plane_contact_exemptions)
        for capsule, patch in self.plane_contact_exemptions.items():
            if capsule not in capsules or patch not in self.patch_ees:
                raise ValueError(f"Unknown capsule or patch in plane contact exemption: {capsule!r}, {patch!r}")
        self.contact_capsules = {patch: tuple(names) for patch, names in self.contact_capsules.items()}
        for patch, names in self.contact_capsules.items():
            if patch not in self.patch_ees or set(names) - capsules.keys():
                raise ValueError(f"Unknown patch or capsule in contact_capsules: {patch!r}, {names!r}")
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate contact capsule for {patch!r}.")

    @property
    def q_min(self) -> np.ndarray:
        return self.pin_model.lowerPositionLimit[7:].copy()

    @property
    def q_max(self) -> np.ndarray:
        return self.pin_model.upperPositionLimit[7:].copy()

    @property
    def nj(self) -> int:
        return self.pin_model.nv - 6

    @property
    def nq(self) -> int:
        return self.pin_model.nq

    @property
    def nv(self) -> int:
        return self.pin_model.nv

    @property
    def mass(self) -> float:
        return sum(I.mass for I in self.pin_model.inertias[1:])


def make_g1_robot() -> RobotConfig:
    """Compatibility entrypoint for the G1 preset."""
    from faro.configs.g1 import make_g1_robot as make_robot
    return make_robot()
