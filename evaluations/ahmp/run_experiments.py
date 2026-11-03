"""Configure and run the AHMP CEM comparison using FARO evaluators."""
from dataclasses import asdict
import math
from pathlib import Path
import sys
import time

# Edit these settings. Selection strings accept spaces or commas.
SCENES_TO_RUN = "tree_easy tree_hard" # Short or dotted scene IDs
METHODS_TO_RUN = "CEM_TO CEM_MEKSO"   # Also accepts TREE_SEARCH_KSO, TREE_SEARCH_MEKSO, TREE_SEARCH_MEKSOTO, TREE_SEARCH_TO
RUNS_TO_RUN = "1 2 3 4 5"             # Any subset, e.g. "1 3"; seed = BASE_SEED + run - 1
TIME_PER_RUN = "2h"                   # Search budget per run, e.g. "10m", "2h"
OUTPUT_DIR = "results"                # Created automatically, relative to this script
PROGRESS_INTERVAL = "30s"             # Console summaries between CEM generations

# leave these unchanged for the main comparison.
BASE_SEED = 0
POPULATION_SIZE = 8
ELITE_SIZE = 4
MIN_PROB = 0.05
TO_WORKERS = 1                     # Same TO worker limit for every CEM and tree search variant

# Support the IDE's "Run Python File" as well as python -m.
ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from evaluations.search_batch import BatchConfig, run_batch
from evaluations.runner_utils import batch_settings, duration_seconds, selection


def settings_from_file():
    settings = batch_settings(Path(__file__).resolve().parent, scenes=SCENES_TO_RUN,
        runs=RUNS_TO_RUN, output_dir=OUTPUT_DIR)
    return BatchConfig(**asdict(settings), methods=selection(METHODS_TO_RUN, "METHODS_TO_RUN"),
        timeout_s=duration_seconds(TIME_PER_RUN),
        base_seed=BASE_SEED, population_size=POPULATION_SIZE,
        elite_size=ELITE_SIZE, min_prob=MIN_PROB, to_workers=TO_WORKERS,
        progress_interval_s=duration_seconds(PROGRESS_INTERVAL))


def build_cem_problem(scene_name, method, spec, *, to_workers=1):
    from evaluations.ahmp.planner import ContactSequenceSpace, SequenceEvaluator
    from faro.evaluators.to import create_to_func
    from faro.graph.oracle_view import LazyOracleGraphView
    from faro.search.problem import build_search_problem
    from faro.search.tree_search import TreeSearchEvaluationMode

    if method not in ("CEM_TO", "CEM_MEKSO"):
        raise ValueError(f"Unknown CEM method: {method}")
    pruning = method == "CEM_MEKSO"
    expected_mode = TreeSearchEvaluationMode.KSO_TO_GOAL if pruning else TreeSearchEvaluationMode.TO_ONLY
    if (spec.evaluation_mode is not expected_mode or
            spec.use_node_oracle_cache != pruning or spec.use_edge_oracle_cache != pruning):
        raise ValueError(f"{method} requires the corresponding registered tree search evaluator/filter settings")

    problem = build_search_problem(scene_name, spec)
    scene, resolved = problem.scene, problem.resolved_state
    ms, graph, goal = problem.ms, problem.graph, problem.discrete_goal
    space = ContactSequenceSpace(graph, problem.initial_node, goal, spec.max_depth)
    to = create_to_func(scene, resolved, terminal_goal=problem.terminal_goal,
                        return_details=True, n_job=to_workers)
    first_trajectory = []

    def evaluate_to(schedule):
        details = to(schedule)
        if (details.feasible and math.isfinite(details.cost) and not first_trajectory
                and goal.matches(ms.encode_contact_mode_seq(schedule)[-1])):
            first_trajectory.append((schedule, details))
        return details.feasible, details.cost

    view, kso = None, None
    oracle_calls = {"mode_calls": 0, "edge_calls": 0}
    if method == "CEM_MEKSO":
        from faro.evaluators.kin_node import create_node_oracle
        from faro.evaluators.kin_edge import create_edge_oracle
        from faro.evaluators.kso import create_kso_func

        def counted(name, func):
            def evaluate(*args):
                oracle_calls[name] += 1
                return func(*args)
            return evaluate

        view = LazyOracleGraphView(graph,
            node_oracle=counted("mode_calls", create_node_oracle(ms, scene, resolved)),
            edge_oracle=counted("edge_calls", create_edge_oracle(ms, scene, resolved)))
        kso = create_kso_func(spec.max_depth + 1, scene, resolved,
                              terminal_goal=problem.terminal_goal, outcome_scorer=problem.outcome_scorer)
    evaluator = SequenceEvaluator(space, evaluate_to, graph_view=view, kso_evaluator=kso,
                                  eval_cost_weight=spec.eval_cost_weight, to_cost_weight=spec.to_cost_weight)
    return space, evaluator, scene, first_trajectory, oracle_calls


def run_cem(scene_name, method, spec, cem_config, search_config, output, *,
            to_workers=1, progress_interval_s=30.):
    from evaluations.ahmp.planner import search

    label = f"[{scene_name} {method} {output.name} seed={cem_config.seed}]"
    print(f"{label} SETUP: constructing evaluators; TO workers={to_workers}", flush=True)
    setup_start = time.perf_counter()
    space, evaluator, scene, first_trajectory, oracle_calls = build_cem_problem(
        scene_name, method, spec, to_workers=to_workers)
    setup_s = time.perf_counter() - setup_start
    print(f"{label} SEARCH started; setup took {setup_s:.1f}s; "
          f"budget={search_config.timeout_s / 60:g}m", flush=True)
    last_elapsed, last_solutions = -math.inf, 0

    def report_generation(event):
        nonlocal last_elapsed, last_solutions
        elapsed = event["elapsed_s"]
        solutions = event["converged_goal_solutions"]
        if elapsed - last_elapsed < progress_interval_s and solutions == last_solutions:
            return
        last_elapsed, last_solutions = elapsed, solutions
        counts = evaluator.counts
        print(f"{label} SEARCH {elapsed / 60:.1f}/{search_config.timeout_s / 60:g}m "
              f"generation={event['generation']} samples={event['samples_evaluated']} "
              f"M={oracle_calls['mode_calls']} E={oracle_calls['edge_calls']} "
              f"KSO={counts['kso_calls']} TO={counts['to_calls']} solutions={solutions}", flush=True)

    result = search(space, evaluator, cem_config, search_config, on_generation=report_generation)
    stats = result["stats"]
    print(f"{label} SEARCH finished: {stats['stop_reason']}; "
          f"{stats['samples']} samples; {stats['converged_goal_solutions']} solutions", flush=True)
    result["stats"].update(setup_time_s=setup_s, **oracle_calls)
    result["mode_encoding"] = {"keys": space.graph.ms.keys, "domains": space.graph.ms.domains,
                               "categories": space.categories, "initial": space.initial,
                               "num_switches": space.max_depth,
                               "num_modes": space.max_depth + 1}
    if first_trajectory and result["solutions"]:
        import numpy as np
        from faro.utils.results import pin_to_mj
        schedule, details = first_trajectory[0]
        trajectory = pin_to_mj(scene, details.results, details.time, schedule,
                               details.num_modes, details.nodes_per_mode)
        trajectory.update(scene_name=scene_name, method=method, seed=cem_config.seed,
                          num_modes=details.num_modes, nodes_per_mode=details.nodes_per_mode,
                          solver_status=details.solver_status)
        np.savez_compressed(output / "first_solution.npz", **trajectory)
    return result


def main():
    run_batch(settings_from_file(), cem_runner=run_cem)


if __name__ == "__main__":
    main()
