from __future__ import annotations

from faro.search.goal import DiscreteGoal, OutcomeScorer, SolverOutcome, TerminalGoal
from faro.utils.contact_type import CONTACT_TYPE_UNILATERAL

def box_and_robot_on_floor(scene) -> DiscreteGoal:
    return DiscreteGoal(
        discrete={
            ("ee", "robot", "left_sole"):  (scene.face_id('floor', 'surface'), CONTACT_TYPE_UNILATERAL),
            ("ee", "robot", "right_sole"): (scene.face_id('floor', 'surface'), CONTACT_TYPE_UNILATERAL),
            ("obj", "white_box"): (6, scene.face_id('floor', 'surface'), CONTACT_TYPE_UNILATERAL),
        },
    )

def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")

class EmptyGoal:
    def __call__(self, outcome: SolverOutcome) -> float:
        
        return 0


def empty_goal(scene, resolved_state) -> OutcomeScorer:
    return EmptyGoal()


DISCRETE_GOALS = {
    "box_and_robot_on_floor": box_and_robot_on_floor,
}

TERMINAL_GOALS = {
    "none": none,
}

OUTCOME_SCORERS = {
    "empty_goal": empty_goal,
}
