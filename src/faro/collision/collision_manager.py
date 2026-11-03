from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import coal

from faro.collision.collision_index import CollisionIndex, PairDef
from faro.collision.robot_collision_model import RobotCollisionModel

from faro.configs.scene import ResolvedSceneState
from faro.configs.identifiers import capsule_name


def _coal_T(T: np.ndarray) -> coal.Transform3s:
    """Convert 4x4 homogeneous transform to coal.Transform3s."""
    tf = coal.Transform3s()
    tf.setTranslation(T[:3, 3])
    tf.setRotation(T[:3, :3])
    return tf


@dataclass
class CollisionBuffers:
    """
    Runtime buffers allocated for a horizon.
    - coll_pair_vals[pair_name]: (9, N+1) where 9 = [p1_loc(3), p2_loc(3), n(3)]
    - coll_pair_dists[base_pair_name]:
        - dist: (N+1,)
        - p1_W: (N+1,3)
        - p2_W: (N+1,3)
    """
    coll_pair_vals: Dict[str, np.ndarray]
    coll_pair_dists: Dict[str, Dict[str, np.ndarray]]


class CollisionManager:
    def __init__(self, scene, collision_index: CollisionIndex):
        self.scene = scene
        self.pin_models = {rid: cfg.pin_model for rid, cfg in scene.robots.items()}
        self.pin_datas = {rid: model.createData() for rid, model in self.pin_models.items()}
        self.capsules = {capsule_name(rid, name): capsule
                         for rid, cfg in scene.robots.items()
                         for name, capsule in cfg.collision_model.capsules.items()}
        self._capsule_shapes = {name: capsule.coal_shape() for name, capsule in self.capsules.items()}
        self._robot_poses_ready = False

        self.index: CollisionIndex = collision_index

        self.N: Optional[int] = None
        self.buffers: Optional[CollisionBuffers] = None

        # Capsule transforms over horizon:
        self.cap_T_W: Dict[str, np.ndarray] = {}
        self.cap_T_W_inv: Dict[str, np.ndarray] = {}

        # Movable object transforms over horizon (set by set_movable_poses):
        self.obj_T_W: Dict[str, np.ndarray] = {}
        self.obj_T_W_inv: Dict[str, np.ndarray] = {}

        # Fixed environment geometry is static in time but numerically bound later.
        self._fixed_env_ready: bool = False
        self._fixed_env_T_W: Dict[str, np.ndarray] = {}
        self._fixed_env_T_W_inv: Dict[str, np.ndarray] = {}
        self._fixed_env_shapes: Dict[str, coal.ShapeBase] = {
            name: cfg.coal_shape for name, cfg in self.scene.fixed_env_objects.items()
        }

        self._movable_shapes = {name: cfg.coal_shape for name, cfg in self.scene.env_objects.items()}

        # coal distance request/response
        self.dist_req = coal.DistanceRequest()
        self.dist_res = coal.DistanceResult()

    def set_fixed_env_qs(self, resolved_state: ResolvedSceneState) -> None:
        """
        Bind the numerical poses of the fixed part of the environment from a resolved scene state.

        This is intentionally separate from __init__ so task construction does not
        depend on a scene-owned initial state.
        """
        fixed_env_qs = {name: obj.q for name, obj in resolved_state.fixed_env_objects.items()}

        missing = set(self.scene.fixed_env_objects.keys()) - set(fixed_env_qs.keys())
        extra = set(fixed_env_qs.keys()) - set(self.scene.fixed_env_objects.keys())

        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing fixed environment poses for: {missing_str}")
        if extra:
            extra_str = ", ".join(sorted(extra))
            raise ValueError(f"Received unknown fixed environment poses for: {extra_str}")

        self._fixed_env_T_W.clear()
        self._fixed_env_T_W_inv.clear()

        for name in self.scene.fixed_env_objects.keys():
            q = np.asarray(fixed_env_qs[name], dtype=float)
            if q.shape[0] < 7:
                raise ValueError(
                    f"Fixed environment pose for '{name}' must have at least 7 entries, got shape {q.shape}."
                )
            T = pin.XYZQUATToSE3(q[:7])
            self._fixed_env_T_W[name] = T.homogeneous
            self._fixed_env_T_W_inv[name] = T.inverse().homogeneous

        self._fixed_env_ready = True

    def allocate(self, N: int) -> CollisionBuffers:
        """
        Allocate all runtime buffers for horizon length N (=> N+1 nodes).
        """
        self._robot_poses_ready = False
        self.N = int(N)
        if self.N < 0:
            raise ValueError("N must be >= 0")

        # Allocate pair buffers
        coll_pair_vals: Dict[str, np.ndarray] = {}
        coll_pair_dists: Dict[str, Dict[str, np.ndarray]] = {}

        for pair_name in self.index.pair_names:
            coll_pair_vals[pair_name] = np.zeros((9, self.N + 1), dtype=float)
            coll_pair_dists[pair_name] = {
                "dist": np.zeros(self.N + 1, dtype=float),
                "p1_W": np.zeros((self.N + 1, 3), dtype=float),
                "p2_W": np.zeros((self.N + 1, 3), dtype=float),
            }

        self.buffers = CollisionBuffers(coll_pair_vals=coll_pair_vals, coll_pair_dists=coll_pair_dists)

        # Allocate capsule transform buffers
        self.cap_T_W.clear()
        self.cap_T_W_inv.clear()

        for cap in self.capsules.keys():
            self.cap_T_W[cap] = np.zeros((self.N + 1, 4, 4), dtype=float)
            self.cap_T_W_inv[cap] = np.zeros((self.N + 1, 4, 4), dtype=float)

        return self.buffers

    def set_robot_trajectories(self, q_trajectories: Dict[str, np.ndarray]) -> None:
        """Update every robot's local capsule transforms using its own model."""
        if self.N is None or self.buffers is None:
            raise RuntimeError("Call allocate(N) before set_robot_trajectories().")
        if q_trajectories.keys() != self.scene.robots.keys():
            raise ValueError("Robot trajectory keys must match scene robot IDs.")
        trajectories = {}
        for rid, model in self.pin_models.items():
            values = np.asarray(q_trajectories[rid], dtype=float)
            if values.shape != (self.N + 1, model.nq) or not np.all(np.isfinite(values)):
                raise ValueError(f"Robot {rid!r} trajectory must be finite with shape {(self.N + 1, model.nq)}.")
            trajectories[rid] = values
        for rid, q_traj in trajectories.items():
            model, data = self.pin_models[rid], self.pin_datas[rid]
            for k, q in enumerate(q_traj):
                pin.framesForwardKinematics(model, data, q)
                for local, spec in self.scene.robots[rid].collision_model.capsules.items():
                    name = capsule_name(rid, local)
                    transform = data.oMf[model.getFrameId(spec.frame)]
                    self.cap_T_W[name][k] = transform.homogeneous
                    self.cap_T_W_inv[name][k] = transform.inverse().homogeneous
        self._robot_poses_ready = True

    def set_movable_poses(self, obj_xyzquat_traj: Dict[str, np.ndarray]) -> None:
        if self.N is None or self.buffers is None:
            raise RuntimeError("Call allocate(N) before set_movable_poses().")

        for name, traj in obj_xyzquat_traj.items():
            if name not in self.scene.env_objects:
                raise ValueError(f"Unknown movable object '{name}' (not in scene.env_objects)")
            traj = np.asarray(traj, dtype=float)
            if traj.shape[0] != self.N + 1 or traj.shape[1] < 7:
                raise ValueError(f"{name} traj must be (N+1, >=7), got {traj.shape}")

            T_W = np.zeros((self.N + 1, 4, 4), dtype=float)
            T_W_inv = np.zeros((self.N + 1, 4, 4), dtype=float)
            for k in range(self.N + 1):
                T = pin.XYZQUATToSE3(traj[k, :7]).homogeneous
                T_W[k] = T
                T_W_inv[k] = pin.SE3(T).inverse().homogeneous

            self.obj_T_W[name] = T_W
            self.obj_T_W_inv[name] = T_W_inv

    def resolve_all(self) -> None:
        """
        Compute all collisions for all pairs at all nodes.
        Requires:
          - allocate(N)
          - set_fixed_env_qs(...)
          - set_robot_trajectories(...)
          - set_movable_poses(...) for any movable objects involved in pairs

        """
        if self.N is None or self.buffers is None:
            raise RuntimeError("Call allocate(N) before resolve_all().")
        if not self._robot_poses_ready:
            raise RuntimeError("Robot trajectories are not set.")
        if not self._fixed_env_ready and self.scene.fixed_env_objects:
            raise RuntimeError(
                "Fixed environment poses are not set. "
                "Call set_fixed_env_qs({name: q, ...}) before resolve_all()."
            )

        def cap_T(cap: str, k: int) -> np.ndarray:
            return self.cap_T_W[cap][k]

        def cap_T_inv(cap: str, k: int) -> np.ndarray:
            return self.cap_T_W_inv[cap][k]

        def obj_T(name: str, k: int) -> np.ndarray:
            if name not in self.obj_T_W:
                raise RuntimeError(
                    f"Movable pose for '{name}' not set. "
                    "Call set_movable_poses({name: traj, ...}) before resolve_all()."
                )
            return self.obj_T_W[name][k]

        def obj_T_inv(name: str, k: int) -> np.ndarray:
            if name not in self.obj_T_W_inv:
                raise RuntimeError(
                    f"Movable pose for '{name}' not set. "
                    "Call set_movable_poses({name: traj, ...}) before resolve_all()."
                )
            return self.obj_T_W_inv[name][k]

        def eval_pair_at_k(pair: PairDef, k: int) -> None:
            vals = self.buffers.coll_pair_vals
            dists = self.buffers.coll_pair_dists

            i, j = pair.name_i, pair.name_j

            # --- i side ---
            if i in self.capsules:
                shape_i = self._capsule_shapes[i]
                T_i = cap_T(i, k)
                T_i_inv = cap_T_inv(i, k)
            elif i in self._fixed_env_shapes:
                shape_i = self._fixed_env_shapes[i]
                T_i = self._fixed_env_T_W[i]
                T_i_inv = self._fixed_env_T_W_inv[i]
            else:
                # movable env object
                shape_i = self._movable_shapes[i]
                T_i = obj_T(i, k)
                T_i_inv = obj_T_inv(i, k)

            # --- j side ---
            if j in self.capsules:
                shape_j = self._capsule_shapes[j]
                T_j = cap_T(j, k)
                T_j_inv = cap_T_inv(j, k)
            elif j in self._fixed_env_shapes:
                shape_j = self._fixed_env_shapes[j]
                T_j = self._fixed_env_T_W[j]
                T_j_inv = self._fixed_env_T_W_inv[j]
            else:
                shape_j = self._movable_shapes[j]
                T_j = obj_T(j, k)
                T_j_inv = obj_T_inv(j, k)

            # --- distance query ---
            d = coal.distance(
                shape_i, _coal_T(T_i),
                shape_j, _coal_T(T_j),
                self.dist_req, self.dist_res
            )

            n = np.array(self.dist_res.normal, dtype=float).reshape(3)
            p1_W = np.array(self.dist_res.getNearestPoint1(), dtype=float).reshape(3)
            p2_W = np.array(self.dist_res.getNearestPoint2(), dtype=float).reshape(3)

            p1_loc = (T_i_inv @ np.array([p1_W[0], p1_W[1], p1_W[2], 1.0]))[:3]
            p2_loc = (T_j_inv @ np.array([p2_W[0], p2_W[1], p2_W[2], 1.0]))[:3]

            vals[pair.name][:, k] = np.hstack((p1_loc, p2_loc, n))

            dists[pair.name]["dist"][k] = float(d)
            dists[pair.name]["p1_W"][k, :] = p1_W
            dists[pair.name]["p2_W"][k, :] = p2_W

            self.dist_res.clear()

        # Evaluate all pairs at all trajectory nodes.
        for k in range(self.N + 1):
            for grp in self.index.groups.values():
                for p in grp.values():
                    eval_pair_at_k(p, k)
