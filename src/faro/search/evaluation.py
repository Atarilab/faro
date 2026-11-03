from __future__ import annotations

import math
from typing import Callable

ContactModeSeq = dict[tuple, list[tuple]]

PathEvaluator = Callable[[ContactModeSeq], float | bool]
ToEvaluator = Callable[[ContactModeSeq], tuple[bool, float]]


def num_modes_cms(contact_mode_seq: ContactModeSeq) -> int:
    if not contact_mode_seq:
        return 0
    return len(next(iter(contact_mode_seq.values())))


def coerce_cost_or_false(result: float | bool, *, source: str) -> float | None:
    if isinstance(result, bool):
        if result is False:
            return None
        raise TypeError(
            f"{source} returned True, which is ambiguous in a float|bool API. "
            f"Return a numeric cost for success or False for infeasibility."
        )
    cost = float(result)
    if not math.isfinite(cost):
        raise ValueError(f"{source} returned a nonfinite cost.")
    return cost


def coerce_to_result(result: tuple[bool, float]) -> tuple[bool, float]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "ToEvaluator must return a 2-tuple: (converged: bool, to_score: float)."
        )

    converged, to_score = result

    if not isinstance(converged, bool):
        raise TypeError(
            "ToEvaluator result[0] must be a bool indicating solver convergence."
        )

    if isinstance(to_score, bool):
        raise TypeError(
            "ToEvaluator result[1] must be a numeric score, not a bool."
        )

    score = float(to_score)
    if not math.isfinite(score):
        raise ValueError("ToEvaluator returned a nonfinite score.")
    return bool(converged), score
