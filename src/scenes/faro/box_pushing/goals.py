from __future__ import annotations

from faro.search.goal import OutcomeScorer, TerminalGoal

from .initial_conditions import goal as goal_state

from dataclasses import replace
from faro.search.goal import GoalTolerances

def box_at_goal(scene, resolved_state) -> TerminalGoal:
    goal = TerminalGoal.from_state(
        scene,
        goal_state(scene),
        objects={"white_box": "pose"},
        kso="hard",
        to="hard",
    )
    target = replace(
        goal.objects["white_box"],
        to_tolerance=GoalTolerances(position=0),
    )
    return replace(goal, objects={"white_box": target})


def box_at_goal_score(scene, resolved_state) -> OutcomeScorer:
    return box_at_goal(scene, resolved_state).score


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {
}

TERMINAL_GOALS = {
    "box_at_goal": box_at_goal,
    "none": none,
}

OUTCOME_SCORERS = {
    "box_at_goal": box_at_goal_score,
}
