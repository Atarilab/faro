from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from faro.graph.ct_types import GoalAlternatives

if TYPE_CHECKING:
    from faro.configs.scene import OnFace, Pose, Scene, StateSpec


DiscreteGoalAssignment = Mapping[tuple, tuple]
DiscreteGoalSpec = DiscreteGoalAssignment | tuple[DiscreteGoalAssignment, ...]


@dataclass(frozen=True, slots=True)
class DiscreteGoal:
    discrete: DiscreteGoalSpec = field(default_factory=dict)

    def normalized_alternatives(self) -> tuple[DiscreteGoalAssignment, ...]:
        if isinstance(self.discrete, Mapping):
            return () if not self.discrete else (self.discrete,)

        if isinstance(self.discrete, tuple):
            if not self.discrete:
                raise ValueError("Use DiscreteGoal() for no discrete goal.")
            for i, alt in enumerate(self.discrete):
                if not isinstance(alt, Mapping):
                    raise TypeError(f"Alternative {i} must be a mapping.")
                if not alt:
                    raise ValueError(f"Alternative {i} cannot be empty.")
            return self.discrete

        raise TypeError("DiscreteGoal.discrete must be a mapping or tuple of mappings.")

    def is_empty(self) -> bool:
        return len(self.normalized_alternatives()) == 0

    def compile(self, ms) -> CompiledDiscreteGoal:
        compiled: list[tuple[tuple[int, int], ...]] = []
        for alt_i, alt in enumerate(self.normalized_alternatives()):
            cur: list[tuple[int, int]] = []
            for key, value in alt.items():
                if key not in ms.key_to_i:
                    raise KeyError(f"Unknown goal key {key!r} in alternative {alt_i}.")
                i = ms.key_to_i[key]
                if value not in ms.value_to_id[i]:
                    raise ValueError(f"Invalid goal value {value!r} for key {key!r}.")
                cur.append((i, ms.value_to_id[i][value]))
            compiled.append(tuple(cur))
        return CompiledDiscreteGoal(alternatives=tuple(compiled))


@dataclass(frozen=True, slots=True)
class CompiledDiscreteGoal:
    alternatives: GoalAlternatives

    def is_empty(self) -> bool:
        return len(self.alternatives) == 0

    def matches(self, node: tuple[int, ...]) -> bool:
        return any(all(node[i] == vid for i, vid in alt) for alt in self.alternatives)


def _nonnegative_block(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim > 1 or not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar or vector.")
    return value


@dataclass(frozen=True, slots=True)
class GoalWeights:
    """Diagonal soft-cost weights, in position / rotation / joint order."""

    position: Any = 1.0
    rotation: Any = 1.0
    joints: Any = 1.0

    def __post_init__(self):
        for name in ("position", "rotation", "joints"):
            _nonnegative_block(getattr(self, name), name)

    def vector(self, blocks):
        values = []
        for name, size in blocks:
            value = _nonnegative_block(getattr(self, name), name)
            if value.ndim == 0:
                value = np.full(size, float(value))
            elif value.shape != (size,):
                raise ValueError(f"{name} must be a scalar or have shape {(size,)}.")
            values.append(value)
        return np.concatenate(values)


@dataclass(frozen=True, slots=True)
class GoalTolerances(GoalWeights):
    """Componentwise hard bounds; zero means an exact equality.

    Position is in world metres, rotation in target-frame radians, and joints
    in their native units. These are independent of solver convergence tolerances.
    """

    position: Any = 0.0
    rotation: Any = 0.0
    joints: Any = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class _TargetTuning:
    kso_weights: GoalWeights = field(default_factory=lambda: GoalWeights(.01, .01, .01))
    to_weights: GoalWeights = field(default_factory=GoalWeights)
    kso_tolerance: GoalTolerances = field(default_factory=GoalTolerances)
    to_tolerance: GoalTolerances = field(default_factory=GoalTolerances)

    def _validate_tuning(self):
        for solver in ("kso", "to"):
            weights = getattr(self, solver + "_weights")
            tolerance = getattr(self, solver + "_tolerance")
            if not isinstance(weights, GoalWeights):
                raise TypeError(f"{solver}_weights must be GoalWeights.")
            if not isinstance(tolerance, GoalTolerances):
                raise TypeError(f"{solver}_tolerance must be GoalTolerances.")
            self.weights(solver)
            self.tolerance(solver)

    def weights(self, solver):
        if solver not in ("kso", "to"):
            raise ValueError("Solver must be 'kso' or 'to'.")
        return getattr(self, solver + "_weights").vector(self._blocks)

    def tolerance(self, solver):
        if solver not in ("kso", "to"):
            raise ValueError("Solver must be 'kso' or 'to'.")
        return getattr(self, solver + "_tolerance").vector(self._blocks)

    @property
    def residual_size(self):
        return sum(size for _, size in self._blocks)


def _configuration_value(value, name, *, size=None):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 1 or value.size < 7 or
            (size is not None and value.size != size) or
            not np.all(np.isfinite(value))):
        expected = f"length {size}" if size is not None else "at least 7 entries"
        raise ValueError(f"{name} must be a finite configuration vector of {expected}.")
    # Validation only: preserve the supplied representation and quaternion sign.
    if not np.isclose(np.linalg.norm(value[3:7]), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} requires a unit XYZW quaternion.")
    return value


@dataclass(frozen=True, slots=True)
class PositionTarget(_TargetTuning):
    """World position of an object centre or robot base."""

    xyz: np.ndarray

    def __post_init__(self):
        xyz = np.asarray(self.xyz, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("PositionTarget.xyz must be a finite vector of shape (3,).")
        object.__setattr__(self, "xyz", xyz)
        self._validate_tuning()

    @property
    def value(self):
        return self.xyz

    @property
    def _blocks(self):
        return (("position", 3),)


@dataclass(frozen=True, slots=True)
class PoseTarget(_TargetTuning):
    """World XYZ + XYZW pose of an object or robot base."""

    q: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "q", _configuration_value(self.q, "PoseTarget.q", size=7))
        self._validate_tuning()

    @property
    def value(self):
        return self.q

    @property
    def _blocks(self):
        return (("position", 3), ("rotation", 3))


@dataclass(frozen=True, slots=True)
class ConfigurationTarget(_TargetTuning):
    """Robot XYZ + XYZW base pose followed by joints in model order."""

    q: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "q", _configuration_value(self.q, "ConfigurationTarget.q"))
        self._validate_tuning()

    @property
    def value(self):
        return self.q

    @property
    def _blocks(self):
        return (("position", 3), ("rotation", 3), ("joints", len(self.q) - 7))


@lru_cache(maxsize=1)
def _numeric_pose_error():
    import casadi as cs
    from faro.ocp.utils import pose_error

    value, reference = cs.SX.sym("value", 7), cs.SX.sym("reference", 7)
    return cs.Function("goal_pose_error", [value, reference], [pose_error(value, reference)])


def _target_error(target, state):
    if isinstance(target, PositionTarget):
        return state[:3] - target.xyz
    pose = np.asarray(_numeric_pose_error()(state[:7], target.q[:7])).reshape(-1)
    if isinstance(target, ConfigurationTarget):
        return np.r_[pose, state[7:len(target.q)] - target.q[7:]]
    return pose


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalGoal:
    """Owned geometric targets and explicit treatment by each solver.

    Modes are 'soft', 'hard', or 'off'. Targets and tuning are independent of
    state regularization; score_weight affects ranking only.
    """

    kso: str
    to: str
    robots: Mapping[str, PositionTarget | PoseTarget | ConfigurationTarget] = field(default_factory=dict)
    objects: Mapping[str, PositionTarget | PoseTarget] = field(default_factory=dict)
    score_weight: float = 1.0

    def __post_init__(self):
        for solver in ("kso", "to"):
            if getattr(self, solver) not in ("soft", "hard", "off"):
                raise ValueError(f"{solver} must be 'soft', 'hard', or 'off'.")
        if not np.isfinite(self.score_weight) or self.score_weight < 0:
            raise ValueError("score_weight must be finite and nonnegative.")
        if not isinstance(self.robots, Mapping) or not isinstance(self.objects, Mapping):
            raise TypeError("TerminalGoal robots and objects must be mappings.")
        for target in self.robots.values():
            if not isinstance(target, (PositionTarget, PoseTarget, ConfigurationTarget)):
                raise TypeError("Robot goals require a PositionTarget, PoseTarget, or ConfigurationTarget.")
        for target in self.objects.values():
            if not isinstance(target, (PositionTarget, PoseTarget)):
                raise TypeError("Object goals require a PositionTarget or PoseTarget.")

    @classmethod
    def from_state(cls, scene: Scene, state_spec: StateSpec, *,
                   robots: Mapping[str, str] | None = None,
                   objects: Mapping[str, str] | None = None,
                   kso: str, to: str, score_weight: float = 1.0) -> TerminalGoal:
        """Resolve a complete state and copy explicitly selected geometric targets.

        Selections map owner names to 'position' or 'pose'; robots also accept
        'configuration'. Omitted selections target nothing. Supports come from
        state_spec, and contacts, velocities, and momentum do not become goals.
        Targets keep their default weights and tolerances.
        """
        state = scene.resolve(state_spec)

        def make_targets(selection, states, *, robot):
            if selection is None:
                return {}
            if not isinstance(selection, Mapping):
                raise TypeError("State goal selections must be mappings of names to target kinds.")
            owner = "robot" if robot else "movable object"
            targets = {}
            for name, kind in selection.items():
                if name not in states:
                    raise KeyError(f"Unknown terminal-goal {owner} {name!r}.")
                q = states[name].q
                if kind == "position":
                    target = PositionTarget(q[:3].copy())
                elif kind == "pose":
                    target = PoseTarget(q[:7].copy())
                elif kind == "configuration" and robot:
                    target = ConfigurationTarget(q.copy())
                else:
                    raise ValueError(f"Unsupported {owner} target kind {kind!r} for {name!r}.")
                targets[name] = target
            return targets

        goal = cls(
            robots=make_targets(robots, state.robots, robot=True),
            objects=make_targets(objects, state.env_objects, robot=False),
            kso=kso, to=to, score_weight=score_weight,
        )
        goal.validate(scene)
        return goal

    @classmethod
    def from_placements(cls, scene, resolved_state, placements: Mapping[str, Pose | OnFace],
                        *, kind="position", **options):
        """Resolve selected movable-object placements against the current layout."""
        from faro.configs.scene import Pose

        if kind not in ("position", "pose"):
            raise ValueError("Placement goal kind must be 'position' or 'pose'.")
        if unknown := placements.keys() - scene.env_objects.keys():
            raise ValueError(f"Goal placements must name movable boxes: {sorted(unknown)}")
        current = {**resolved_state.fixed_env_objects, **resolved_state.env_objects}
        objects = scene.resolve_objects({
            **{name: Pose(state.q) for name, state in current.items()},
            **placements,
        })
        target = (lambda q: PositionTarget(q[:3])) if kind == "position" else PoseTarget
        return cls(objects={name: target(objects[name].q) for name in placements}, **options)

    def score(self, outcome: SolverOutcome) -> float:
        """Sum owner residual norms, independent of solver modes and cost weights."""
        error = 0.0
        for targets, key in ((self.robots, "robot_xs"), (self.objects, "obj_xs")):
            for name, target in targets.items():
                state = np.asarray(outcome.results[key][name])[-1]
                error += np.linalg.norm(_target_error(target, state))
        return float(self.score_weight * error)

    def is_empty(self) -> bool:
        return not self.robots and not self.objects

    def validate(self, scene) -> None:
        self.__post_init__()
        for robot_id, target in self.robots.items():
            if robot_id not in scene.robots:
                raise KeyError(f"Unknown terminal-goal robot {robot_id!r}.")
            if isinstance(target, ConfigurationTarget):
                cfg = scene.robots[robot_id]
                _configuration_value(target.q, f"Robot {robot_id!r} goal", size=cfg.nq)
                if np.any(target.q[7:] < cfg.q_min) or np.any(target.q[7:] > cfg.q_max):
                    raise ValueError(f"Robot {robot_id!r} goal joints exceed position limits.")
        for name in self.objects:
            if name not in scene.env_objects:
                raise KeyError(f"Unknown terminal-goal object {name!r}.")
        for target in (*self.robots.values(), *self.objects.values()):
            # Arrays need not be immutable; recheck their values before a solve.
            target.__post_init__()


@dataclass(frozen=True, slots=True)
class SolverOutcome:
    converged: bool
    results: dict[str, Any]
    perf_metrics: dict[str, Any] | None = None
    scene: Any | None = None


class OutcomeScorer(Protocol):
    def __call__(self, outcome: SolverOutcome) -> float: ...
