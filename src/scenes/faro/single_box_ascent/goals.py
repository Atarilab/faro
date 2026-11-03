from __future__ import annotations

import numpy as np

from faro.search.goal import GoalTolerances, DiscreteGoal, OutcomeScorer, PositionTarget, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL

def on_table_holding_box(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("ee", "robot", "left_sole"):  (scene.face_id('table_short', 'top'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "right_sole"): (scene.face_id('table_short', 'top'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "left_ee"):    (scene.face_id('white_box', 'left'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "right_ee"):   (scene.face_id('white_box', 'right'), CONTACT_TYPE_UNILATERAL),
            ("obj", "white_box"): (),
        },
    )

def on_table_holding_box_term(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(
        kso="hard", to="hard",
        robots={"robot": PositionTarget(
            np.array([0.85, 0.0, 1.504679]),
            to_tolerance=GoalTolerances(position=0.1),
        )},
        objects={
            "white_box": PositionTarget(
                np.array([1.11007743, -.00356788320, 1.60690200]),
                to_tolerance=GoalTolerances(position=0.1),
            ),
        },
    )

def on_table_holding_box_score(scene, resolved_state) -> OutcomeScorer:
    return on_table_holding_box_term(scene, resolved_state).score


DISCRETE_GOALS = {
    "on_table_holding_box": on_table_holding_box,
}

TERMINAL_GOALS = {
    "on_table_holding_box_term": on_table_holding_box_term,
}

OUTCOME_SCORERS = {
    "on_table_holding_box_score": on_table_holding_box_score,
}
