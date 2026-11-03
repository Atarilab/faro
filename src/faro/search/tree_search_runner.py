from __future__ import annotations

import argparse
from dataclasses import replace

from scenes.common import registry
from faro.evaluators.kso import create_kso_func
from faro.evaluators.to import create_to_func
from faro.evaluators.kin_node import create_node_oracle
from faro.evaluators.kin_edge import create_edge_oracle
from faro.graph.oracle_view import LazyOracleGraphView
from faro.search.tree_search import TreeSearchEvaluationMode, TreeSearchResult, TreeSearchPlanner
from faro.search.problem import build_search_problem


def default_output_path(scene_name: str, experiment_name: str, spec) -> str:
    return (
        f"{scene_name}"
        f"__exp_{experiment_name}"
        f"__state_{spec.initial_condition_name}"
        f"__domain_{spec.domain_name}"
        f"__goal_{spec.discrete_goal_name or 'none'}.txt"
    )


def run_experiment(
    *,
    scene_name: str,
    experiment_name: str,
    output_path: str | None = None,
    spec_override=None,
    show_progress_bar: bool = True,
    to_workers: int = 1,
) -> TreeSearchResult:
    spec = spec_override or registry.get_experiment(scene_name, experiment_name)

    problem = build_search_problem(scene_name, spec)
    scene, resolved_state = problem.scene, problem.resolved_state
    ms, graph = problem.ms, problem.graph
    compiled_discrete_goal = problem.discrete_goal
    terminal_goal, outcome_scorer = problem.terminal_goal, problem.outcome_scorer

    graph_node_oracle = (
        create_node_oracle(ms, scene, resolved_state)
        if spec.use_node_oracle_cache else lambda node: True
    )
    graph_edge_oracle = (
        create_edge_oracle(ms, scene, resolved_state)
        if spec.use_edge_oracle_cache else lambda a, b: True
    )

    graph_view = LazyOracleGraphView(
        base=graph,
        node_oracle=graph_node_oracle,
        edge_oracle=graph_edge_oracle,
    )

    initial_node = problem.initial_node

    path_evaluator = None
    if spec.evaluation_mode is not TreeSearchEvaluationMode.TO_ONLY:
        path_evaluator = create_kso_func(
            max_num_nodes=spec.max_depth + 1,
            scene=scene,
            resolved_state=resolved_state,
            terminal_goal=terminal_goal,
            outcome_scorer=outcome_scorer,
        )
    to_evaluator = create_to_func(
        scene=scene,
        resolved_state=resolved_state,
        terminal_goal=terminal_goal,
        n_job=to_workers,
    )

    planner = TreeSearchPlanner(
        ms=ms,
        graph_view=graph_view,
        evaluator=path_evaluator,
        to_evaluator=to_evaluator,
        config=spec.to_planner_config(show_progress_bar=show_progress_bar),
    )

    return planner.plan(
        initial_node=initial_node,
        goal=compiled_discrete_goal,
        solutions_txt_path=(
            output_path or default_output_path(scene_name, experiment_name, spec)
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a named tree search experiment for a scene.",
    )
    parser.add_argument("experiment", nargs="?", help="Experiment name to run.")
    parser.add_argument(
        "--scene",
        default="example_scene",
        help="Scene module ID to load experiments from, e.g. faro.tree_easy.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments for the selected scene and exit.",
    )
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--output")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list:
        for name in registry.list_experiments(args.scene):
            print(name)
        return

    if args.experiment is None:
        parser.error("Please provide an experiment name, or use --list.")

    spec = registry.get_experiment(args.scene, args.experiment)

    overrides = {}
    if args.max_depth is not None:
        overrides["max_depth"] = args.max_depth
    if args.max_iterations is not None:
        overrides["max_iterations"] = args.max_iterations
    if args.seed is not None:
        overrides["random_seed"] = args.seed
    if args.timeout_s is not None:
        overrides["timeout_s"] = args.timeout_s

    if overrides:
        spec = replace(spec, **overrides)

    result = run_experiment(
        scene_name=args.scene,
        experiment_name=args.experiment,
        output_path=args.output,
        spec_override=spec,
    )

    print("best_path_cost:", result.best_path_cost)
    print("goal_hits:", result.stats.goal_reaching_sequences)
    print("converged_solutions_found:", result.stats.converged_goal_solutions)


if __name__ == "__main__":
    main()
