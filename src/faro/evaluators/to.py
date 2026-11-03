from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from faro.tasks.to_task import ToTask
from faro.solvers.moto import MotoBuildConfig
from faro.search.evaluation import ContactModeSeq, num_modes_cms
from faro.cost.state_ctrl_cost import make_ocp_weights
from faro.configs.scene import ResolvedSceneState
from faro.search.goal import TerminalGoal


@dataclass(frozen=True)
class ToEvaluationDetails:
    feasible: bool
    cost: float
    max_constraint_violation: float
    solver_status: int
    results: dict[str, Any]
    time: np.ndarray
    num_modes: int
    nodes_per_mode: int
    solve_time: float | None = None


ToEvaluationReturn = tuple[bool, float] | ToEvaluationDetails


def create_to_func(
    scene: Any,
    resolved_state: ResolvedSceneState,
    terminal_goal: TerminalGoal | None = None,
    return_details: bool = False,
    *,
    n_job: int = 1,
    presolve_max_iters: int = 10,
) -> Callable[[ContactModeSeq], ToEvaluationReturn]:
    """Create a TO evaluator with configurable workers and optional presolve.

    ``n_job`` controls both Moto solvers and defaults to one worker for
    consistent TO throughput with and without preceding acados calls.
    Set ``presolve_max_iters=0`` to
    start directly with the collision-aware solve.
    """
    if terminal_goal is not None:
        terminal_goal.validate(scene)

    def evaluate(cms: ContactModeSeq) -> ToEvaluationReturn:
        num_modes = num_modes_cms(cms)

        task = ToTask(scene=scene, resolved_state=resolved_state,
                      terminal_goal=terminal_goal)

        nodes_per_mode = 10
        time_per_mode = 1.0
        num_nodes = num_modes * nodes_per_mode

        task.build_solver(
            cfg=MotoBuildConfig(
                num_modes=num_modes,
                nodes_per_mode=nodes_per_mode,
                time_per_mode=time_per_mode,
                tol=1e-2,
                n_job=n_job,
                presolve_max_iters=presolve_max_iters,
            ),
        )

        task.initialize_solver(cms)

        task.update_mode_oh_via_tup(cms)

        task.set_fixed_params(
            n_nodes=num_modes * nodes_per_mode,
            weights=make_ocp_weights(scene.robots),
        )

        status = task.solve(max_nlp_iters=100)
        results = task.get_results(include_torque=True, coll=True)
        constr_vio = task.sum_constraint_violations()
        max_constr_vio = task.max_constraint_violation()
        ctrl_margin = task.get_finite_diff_jerk_cost(results["vdot_rob"])

        cost = 1e2 * constr_vio / num_nodes + 1e-2 * ctrl_margin / num_modes

        if status == 2 and cost == 0:
            cost = 1e4

        feasible = status == 0
        if status not in (0, 1, 2):
            raise ValueError(f"TO solver returned unsupported status {status}.")

        if return_details:
            return ToEvaluationDetails(
                feasible=feasible,
                cost=float(cost),
                max_constraint_violation=float(max_constr_vio),
                solver_status=status,
                results=results,
                time=task.get_ts(results),
                num_modes=num_modes,
                nodes_per_mode=nodes_per_mode,
                solve_time=task.get_solve_time(),
            )
        return feasible, cost

    return evaluate
