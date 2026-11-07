"""KSO tangent-error weights: XYZ/rotation (six), then robot joints.

Quaternion storage has four components, but its orientation error has three.
The control weights regularize the next configuration against q0, just as the
state weights regularize the current configuration; they are not velocities.
"""
import numpy as np


def make_ocp_weights(robots):
    weights = {"robots": {}}
    for rid, robot in robots.items():
        robot_weight = np.hstack((1e-3 * np.ones(6), 1e-4 * np.ones(robot.nj)))
        weights["robots"][rid] = {
            field: robot_weight.copy()
            for field in ("rob_state_cost_W", "rob_ctrl_cost_W", "rob_state_cost_We")
        }
    for field in ("object_state_cost_W", "object_ctrl_cost_W", "object_state_cost_We"):
        weights[field] = 1e-3 * np.ones(6)
    return weights
