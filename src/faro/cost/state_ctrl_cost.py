"""Default cost weights sized for the configured robot."""
import numpy as np


def _robot_and_shared_weights(robot):
    # Robot Cost
    W_base_pos = np.array([1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-3])
    W_joint_pos = 1e-1 * np.ones(robot.nj)

    W_base_vel = 1e-2 * np.ones(6)
    W_joint_vel = 1e-2 * np.ones(robot.nj)

    W_com   = 1e-3 * np.ones(3)                  # Linear
    W_mom   = 1e-2 * np.ones(6)         # Linear, Angular
    W_base_acc  = 1e-3 * np.ones(6)
    W_joint_acc = 1e-3 * np.ones(robot.nj)

    W_robot_wrench = np.array([1e-4]*6)
    Time_scale = np.array([1e1])

    # Object Cost
    W_obj_base_pos   = 1e-2 * np.ones(6)
    W_obj_base_vel   = 1e-2 * np.ones(6)

    # Control Costs
    W_obj_base_acc    = 1e-2 * np.ones(6) # linear, angular
    W_obj_body_wrench = 1e-4 * np.ones(6) # tau, f

    W_robot_state = np.hstack((W_base_pos, W_joint_pos, W_base_vel,
                               W_joint_vel, W_com, W_mom))

    W_robot_acc = np.hstack((W_base_acc, W_joint_acc))

    W_object_state = np.hstack((W_obj_base_pos, W_obj_base_vel))
    W_object_ctrl = np.hstack((W_obj_base_acc, W_obj_body_wrench))

    return {
        "rob_state_cost_W":     W_robot_state,
        "rob_acc_cost_W":       W_robot_acc,
        "rob_wrench":           W_robot_wrench,
        "time_scale_cost_W":    Time_scale,
        "object_state_cost_W":  W_object_state,
        "object_ctrl_cost_W":   W_object_ctrl,
        "rob_state_cost_We":    1e2 * np.hstack((W_base_pos*0.01, W_joint_pos, W_base_vel,
                                           W_joint_vel, W_com, W_mom)),
        "object_state_cost_We": 1e2 * np.hstack((W_obj_base_pos*0.01, W_obj_base_vel))
    }


def make_ocp_weights(robots):
    """Robot-owned costs plus one shared set of object and time weights."""
    result = {"robots": {}}
    fields = {"rob_state_cost_W": "state_cost_W", "rob_state_cost_We": "state_cost_We",
              "rob_acc_cost_W": "acc_cost_W", "rob_wrench": "wrench"}
    for rid, robot in robots.items():
        result["robots"][rid] = {}
        for name, value in _robot_and_shared_weights(robot).items():
            if name in fields:
                result["robots"][rid][fields[name]] = value
            elif name not in result:
                result[name] = value
    return result
