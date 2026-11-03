from __future__ import annotations

import numpy as np
from typing import Any, Callable

from faro.tasks.kso_task import KsoTask
from faro.solvers.acados import AcadosBuildConfig
from faro.search.evaluation import ContactModeSeq, num_modes_cms
from faro.cost.kso_state_ctrl_cost import make_ocp_weights
from faro.evaluators._results import KsoEvaluationDetails

from faro.configs.scene import ResolvedSceneState

from faro.search.goal import OutcomeScorer, SolverOutcome, TerminalGoal

def create_kso_func(
    max_num_nodes: int,
    scene: Any,
    resolved_state: ResolvedSceneState,
    terminal_goal: TerminalGoal | None = None,
    outcome_scorer: OutcomeScorer | None = None,
    *,
    free_init_state: bool = False,
    return_details: bool = False,
) -> Callable[[ContactModeSeq], float | bool | KsoEvaluationDetails]:
    """Evaluate schedules of up to max_num_nodes modes at their transitions."""
    if max_num_nodes < 2:
        raise ValueError("max_num_nodes must be >= 2.")

    if terminal_goal is not None:
        terminal_goal.validate(scene)

    code_gen_path = "solvers/c_generated_code"
    solver_config_path = "solvers"

    tasks_by_num_modes: dict[int, KsoTask] = {}

    for num_modes in range(2, max_num_nodes + 1):
        task = KsoTask(
            scene=scene,
            resolved_state=resolved_state,
            terminal_goal=terminal_goal,
        )

        task.build_solver(
            num_modes=num_modes,
            build_cfg=AcadosBuildConfig(
                name=f"{scene.name.replace('.', '_')}_{num_modes}",
                solver_config_path=solver_config_path,
                code_gen_path=code_gen_path,
                force_recompile=False,
                qp_max_iter=100,
                max_nlp_iters=500,
                tol=1e-2,
                free_init_state=free_init_state,
            ),
        )

        tasks_by_num_modes[num_modes] = task

    def evaluate(cms: ContactModeSeq,
                 sqp_iters_per_collision: int = 5) -> float | bool | KsoEvaluationDetails:
        num_modes = num_modes_cms(cms)

        if num_modes not in tasks_by_num_modes:
            raise ValueError(f"Num modes '{num_modes}' not in range 2 - {max_num_nodes}.")

        task = tasks_by_num_modes[num_modes]

        try:
            task.update_mode_oh_via_tup(cms)
        except ValueError:
            return KsoEvaluationDetails(False, 0.0) if return_details else False

        task.set_fixed_params(weights=make_ocp_weights(scene.robots))

        task.initialize_solver()

        status = task.solve(nlp_iters=100, print_stats=False,
                            sqp_iters_per_collision=sqp_iters_per_collision,
                            res_ineq_tol=1e-2)
        perf_metrics = task.get_performance_metrics()

        if status == 1 and perf_metrics['max_violation'] > 1e-2:
            result = False
        elif status == 0 or perf_metrics['max_violation'] < 1e-2:
            results = task.get_results()

            outcome = SolverOutcome(
                converged=(status == 0),
                results=results,
                perf_metrics=perf_metrics,
                scene=task.scene,
            )

            if outcome_scorer is None:
                result = 0.0
            else:
                result = float(outcome_scorer(outcome))
        else:
            raise ValueError(f"Status '{status}' not supported.")

        return KsoEvaluationDetails(result, task.get_solve_time()) if return_details else result

    return evaluate
