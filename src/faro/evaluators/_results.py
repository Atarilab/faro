"""Optional detailed filter results; ordinary planner calls keep their APIs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleEvaluationDetails:
    feasible: bool
    solve_time: float


@dataclass(frozen=True)
class KsoEvaluationDetails:
    cost: float | bool
    solve_time: float

    @property
    def feasible(self) -> bool:
        return isinstance(self.cost, float)
