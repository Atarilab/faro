from __future__ import annotations

import numpy as np

from faro.search.goal import DiscreteGoal, OutcomeScorer, TerminalGoal, PositionTarget
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL


def place_box_on_table(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("obj", "white_box"): (6, scene.face_id('table_short', 'top'), CONTACT_TYPE_UNILATERAL),
        },
    )


def box_on_table(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(
        kso="soft", to="off",
        objects={
            "white_box": PositionTarget(
                np.array([1.0, 0.0, 0.865]),
            ),
        },
    )


def box_on_table_score(scene, resolved_state) -> OutcomeScorer:
    return box_on_table(scene, resolved_state).score


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {
    "place_box_on_table": place_box_on_table,
}

TERMINAL_GOALS = {
    "box_on_table": box_on_table,
    "none": none,
}

OUTCOME_SCORERS = {
    "box_on_table": box_on_table_score,
}
