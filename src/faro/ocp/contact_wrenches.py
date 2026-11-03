"""Robot wrenches are world-aligned [force, moment] about each EE origin.

Object wrenches are body-local [moment, force] about the object origin.
In TO both decision representations refer to the interval's endpoint state.
Reactions shift between body/EE origins at that same endpoint.
"""
from numbers import Real

import casadi as cs
import pinocchio.casadi as cpin

from faro.configs.identifiers import interface_name


def reaction_at(wrench, source_position, target_position):
    """Equal-and-opposite world-aligned wrench about the receiving origin."""
    return cs.vertcat(-wrench[:3],
                      -wrench[3:] - cs.cross(source_position - target_position, wrench[:3]))


def robot_contact_wrenches(scene, ee_poses, object_qs, robot_wrenches,
                           object_wrenches, params):
    """Return total loads without modifying source/contact decision variables.

    All poses must be the poses where the input wrenches are represented.
    object_wrenches is keyed by object name, in body [moment, force] order.
    A receiving EE can have zero outgoing wrench and any number of incoming loads.
    """
    total = {rid: dict(patches) for rid, patches in robot_wrenches.items()}
    object_world_wrenches = {}
    for target, feature_id in scene.features.ee_ids.items():
        _, rid, patch = target
        position = ee_poses[target][1]
        for source in scene.robot_interfaces:
            if source == target:
                continue
            selector = params[interface_name(source) + '_env_feat_oh'][feature_id - 1]
            if isinstance(selector, Real) and selector == 0:
                continue
            wrench = robot_wrenches[source[1]][source[2]]
            total[rid][patch] = total[rid][patch] + selector * reaction_at(wrench, ee_poses[source][1], position)
        for name in scene.env_objects:
            selector = params[name + '_env_feat_oh'][feature_id - 1]
            if isinstance(selector, Real) and selector == 0:
                continue
            if name not in object_world_wrenches:
                pose = cpin.XYZQUATToSE3(cs.SX(object_qs[name][:7]))
                body = object_wrenches[name]
                world = cs.vertcat(pose.rotation @ body[3:], pose.rotation @ body[:3])
                object_world_wrenches[name] = world, pose.translation
            world, source_position = object_world_wrenches[name]
            total[rid][patch] = total[rid][patch] + selector * reaction_at(world, source_position, position)
    return total


def robot_ee_poses(robots, qs):
    return {('ee', rid, patch): pose
            for rid, block in robots.items()
            for (kind, patch), pose in block.kinematics.poses(qs[rid]).items()
            if kind == 'ee'}


def robot_wrench_trajectory(scene, robots, robot_xs, object_xs, robot_wrenches,
                            object_wrenches, interface_params):
    """Route interval k wrenches at state k+1; input states have N+1 rows.

    Contact selectors still belong to interval k. Source controls stay intact.
    """
    import numpy as np

    total = {rid: {patch: np.array(w, copy=True) for patch, w in patches.items()}
             for rid, patches in robot_wrenches.items()}
    if not scene.include_robot_ees:
        return total
    count = min((len(w) for patches in total.values() for w in patches.values()), default=0)
    ee_indices = np.array(list(scene.features.ee_ids.values()), dtype=int) - 1
    for k in range(count):
        params = {key: np.asarray(value)[:, k] for key, value in interface_params.items()
                  if key.endswith('_env_feat_oh')}
        if not any(np.any(value[ee_indices]) for value in params.values()):
            continue
        qs = {rid: xs[k + 1, :robots[rid].nq] for rid, xs in robot_xs.items()}
        loads = robot_contact_wrenches(
            scene, robot_ee_poses(robots, qs),
            {name: xs[k + 1, :7] for name, xs in object_xs.items()},
            {rid: {patch: w[k] for patch, w in patches.items()}
             for rid, patches in robot_wrenches.items()},
            {name: w[k] for name, w in object_wrenches.items()}, params)
        for rid, patches in loads.items():
            for patch, wrench in patches.items():
                total[rid][patch][k] = np.asarray(cs.DM(wrench)).reshape(6)
    return total
