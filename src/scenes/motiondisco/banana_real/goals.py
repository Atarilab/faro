from __future__ import annotations

import numpy as np

from faro.search.goal import DiscreteGoal, OutcomeScorer, PositionTarget, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL


def grasp_banana(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("ee", "robot", "left_ee"): (scene.face_id('banana', 'left'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "right_ee"): (scene.face_id('banana', 'right'), CONTACT_TYPE_UNILATERAL),
        },
    )


def robot_base_while_grasping(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(
        kso="hard", to="off",
        robots={"robot": PositionTarget(
            np.array([0.988, 0.0, 1.82]),
        )},
    )


def robot_base_while_grasping_score(scene, resolved_state) -> OutcomeScorer:
    return robot_base_while_grasping(scene, resolved_state).score


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {
    "grasp_banana": grasp_banana,
}

TERMINAL_GOALS = {
    "robot_base_while_grasping": robot_base_while_grasping,
    "none": none,
}

OUTCOME_SCORERS = {
    "robot_base_while_grasping": robot_base_while_grasping_score,
}
