"""Task solve timings, in seconds, independent of evaluator construction."""

from dataclasses import dataclass
from functools import wraps
from time import perf_counter


@dataclass(frozen=True)
class SolveTiming:
    wall_time: float
    initialization_time: float = 0.0

    @property
    def solve_time(self) -> float:
        return max(0.0, self.wall_time - self.initialization_time)


class TimedTask:
    """Expose timings for the latest completed ``solve()`` call.

    Construction and ``initialize_solver()`` are outside the measurement.
    Moto's profiled native initialization is also excluded; collision updates
    and checks performed inside ``solve()`` remain included. A new call replaces
    the previous timing, and an exception leaves no completed timing.

    This excludes the normal first-use kernel preparation, not arbitrary lazy
    work inside iterations (for example, Moto's restoration overlay setup).
    """

    _solve_timing: SolveTiming | None = None

    def get_solve_timing(self) -> SolveTiming:
        if self._solve_timing is None:
            raise RuntimeError("No completed solve timing; call solve() first.")
        return self._solve_timing

    def get_solve_time(self) -> float:
        """Seconds spent solving, excluding setup and native initialization."""
        return self.get_solve_timing().solve_time

    def _moto_update(self, solver, iterations, *, verbose=False):
        result = solver.update(iterations, verbose=verbose, profile=True)
        # Read immediately: the next update replaces the native profile.
        self._solve_initialization_time += solver.get_profile_report().initialize_ms / 1000.0
        return result


def timed_solve(solve):
    """Measure a task solve without changing its return value or exceptions."""
    @wraps(solve)
    def measured(self, *args, **kwargs):
        self._solve_timing = None
        self._solve_initialization_time = 0.0
        start = perf_counter()
        result = solve(self, *args, **kwargs)
        self._solve_timing = SolveTiming(
            wall_time=perf_counter() - start,
            initialization_time=self._solve_initialization_time,
        )
        return result

    return measured
