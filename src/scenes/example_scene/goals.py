from __future__ import annotations

import numpy as np

from faro.search.goal import DiscreteGoal, OutcomeScorer, TerminalGoal, PositionTarget
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL


def place_box_on_table(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("obj", "box1"): (6, scene.face_id('table1', 'top'), CONTACT_TYPE_UNILATERAL),
        },
    )


def box_center_near_table(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(
        kso="soft", to="off",
        objects={
            "box1": PositionTarget(
                np.array([1.0, 0.0, 0.95]),
            ),
        },
    )


def box_center_near_table_score(scene, resolved_state) -> OutcomeScorer:
    return box_center_near_table(scene, resolved_state).score


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {
    "place_box_on_table": place_box_on_table,
}

TERMINAL_GOALS = {
    "box_center_near_table": box_center_near_table,
    "none": none,
}

OUTCOME_SCORERS = {
    "box_center_near_table": box_center_near_table_score,
}
