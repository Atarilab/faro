from __future__ import annotations

import numpy as np

from faro.search.goal import GoalTolerances, DiscreteGoal, OutcomeScorer, PositionTarget, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL


def stand_on_floor(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("ee", "robot", "left_sole"): (scene.face_id('floor', 'surface'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "right_sole"): (scene.face_id('floor', 'surface'), CONTACT_TYPE_UNILATERAL),
        },
    )


def rob_base_goal(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(
        kso="hard", to="hard",
        robots={"robot": PositionTarget(
            np.array([2.0, 0.0, 0.775]),
            to_tolerance=GoalTolerances(position=0.1),
        )},
    )


def rob_base_goal_score(scene, resolved_state) -> OutcomeScorer:
    return rob_base_goal(scene, resolved_state).score


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {
    "stand_on_floor": stand_on_floor,
}

TERMINAL_GOALS = {
    "rob_base_goal": rob_base_goal,
    "none": none,
}

OUTCOME_SCORERS = {
    "rob_base_goal": rob_base_goal_score,
}
