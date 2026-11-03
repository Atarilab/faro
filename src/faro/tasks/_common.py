"""Shared scene setup and parameter/visualization helpers for tasks."""

from collections.abc import Mapping

import numpy as np

from faro.collision.collision_manager import CollisionManager
from faro.collision.collision_index import CollisionIndex
from kinevis.vis_tools import KinematicRobotVisualizer
from faro.configs.identifiers import robot_name


def initialize_scene(task, scene, resolved_state):
    """Populate the scene and collision attributes exposed by every task."""
    task.scene = scene
    task.resolved = resolved_state
    if resolved_state.robots.keys() != scene.robots.keys():
        raise ValueError("Resolved robot IDs must match the scene.")
    task.robot_cfgs = scene.robots
    task.fixed_objs = scene.fixed_env_objects
    task.env_objs = scene.env_objects
    task.robot_collision_models = {rid: cfg.collision_model for rid, cfg in scene.robots.items()}
    task.collision_rules = scene.collision_rules
    task.collision_index = CollisionIndex.from_scene(scene, task.collision_rules)
    task.collision_mgr = CollisionManager(scene, task.collision_index)


def ik_constant_params(scene, resolved, n_nodes=None):
    const_params = {robot_name(rid, "q0"): resolved.robots[rid].q for rid in scene.robots}
    if n_nodes is not None:
        const_params["n_nodes"] = n_nodes

    for name, config in scene.fixed_env_objects.items():
        const_params[name + "_q0"] = resolved.fixed_env_objects[name].q
        const_params[name + "_size"] = config.half_extents

    for name, config in scene.env_objects.items():
        const_params[name + "_q0"] = resolved.env_objects[name].q
        const_params[name + "_size"] = config.half_extents
    return const_params


_ROBOT_COST_FIELDS = frozenset((
    "rob_state_cost_W", "rob_ctrl_cost_W", "rob_state_cost_We",
    "state_cost_W", "state_cost_We", "acc_cost_W", "wrench",
))
_SHARED_COST_FIELDS = frozenset((
    "object_state_cost_W", "object_state_cost_We", "object_ctrl_cost_W",
    "time_scale_cost_W",
))


def flatten_cost_weights(scene, weights, parameter_symbols):
    """Validate owned cost overrides before producing flat solver parameters.

    Public weights keep robot fields under ``weights['robots'][robot_id]``.
    Partial overrides are allowed; every supplied field must belong to this
    formulation and match its parameter vector. Returned arrays are copies.
    """
    if weights is None:
        return {}
    if not isinstance(weights, Mapping):
        raise ValueError("Cost weights must be a mapping")
    unknown = set(weights) - _SHARED_COST_FIELDS - {"robots"}
    if unknown:
        raise ValueError(f"Unknown shared cost fields: {unknown}")
    robots = weights.get("robots", {})
    if not isinstance(robots, Mapping):
        raise ValueError("Cost weights['robots'] must map robot IDs to field mappings")
    unknown = set(robots) - set(scene.robots)
    if unknown:
        raise ValueError(f"Unknown cost robot IDs: {unknown}")

    params = {}
    def add(name, value, label):
        if name not in parameter_symbols:
            raise ValueError(f"Unsupported cost field {label!r} for this formulation")
        symbol = parameter_symbols[name]
        size = int((symbol.sx if hasattr(symbol, "sx") else symbol).numel())
        array = np.asarray(value, dtype=float)
        if array.ndim == 0 and size == 1:
            array = array.reshape(1)
        if array.shape != (size,):
            raise ValueError(f"{label}: expected shape {(size,)}, got {array.shape}")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(f"{label}: cost weights must be finite and nonnegative")
        params[name] = array.copy()

    for rid, fields in robots.items():
        if not isinstance(fields, Mapping):
            raise ValueError(f"Cost weights for robot {rid!r} must be a field mapping")
        unknown = set(fields) - _ROBOT_COST_FIELDS
        if unknown:
            raise ValueError(f"Unknown cost fields for robot {rid!r}: {unknown}")
        for field, value in fields.items():
            add(robot_name(rid, field), value, f"robots.{rid}.{field}")
    for field, value in weights.items():
        if field != "robots":
            add(field, value, field)
    return params


def visualize_ik_trajectory(scene, mj_model, robot_qs, obj_qs, num_steps,
                            transparent_robot=False, robot_alpha=0.1):
    from faro.configs.mujoco import configurations_to_qpos
    viz = KinematicRobotVisualizer(mj_model, transparent_robot=transparent_robot,
                                   robot_alpha=robot_alpha)
    mujoco_q_traj = configurations_to_qpos(scene, mj_model, robot_qs, obj_qs)
    viz.visualize_trajectory(mujoco_q_traj, np.full(num_steps, 0.3))
