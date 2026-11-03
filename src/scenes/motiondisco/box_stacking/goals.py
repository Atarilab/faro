from __future__ import annotations

import numpy as np
import pinocchio as pin

from faro.search.goal import OutcomeScorer, SolverOutcome, TerminalGoal


class BoxStackHeight:
    def __call__(self, outcome: SolverOutcome) -> float:
        box_heights = []

        for k, v in outcome.results["obj_xs"].items():
            R = pin.XYZQUATToSE3(v[-1, :7]).rotation
            box_half_size = np.array(outcome.scene.env_objects[k].half_extents)

            box_heights.append(v[-1, 2] + (R @ box_half_size)[2])
        
        return max(box_heights)


def box_stack_height(scene, resolved_state) -> OutcomeScorer:
    return BoxStackHeight()


def none(scene, resolved_state) -> TerminalGoal:
    return TerminalGoal(kso="off", to="off")


DISCRETE_GOALS = {}

TERMINAL_GOALS = {
    "none": none,
}

OUTCOME_SCORERS = {
    "box_stack_height": box_stack_height,
}
