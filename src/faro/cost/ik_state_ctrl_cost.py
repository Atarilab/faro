"""Regularization of the single IK configuration against its reference."""
import numpy as np


def make_ocp_weights(robots):
    weights = {"robots": {}}
    for rid, robot in robots.items():
        robot_weight = np.hstack((1e-4 * np.ones(6), 1e-1 * np.ones(robot.nj)))
        weights["robots"][rid] = {"rob_state_cost_We": robot_weight}
    weights["object_state_cost_We"] = 1e-3 * np.ones(6)
    return weights
