from dataclasses import dataclass, replace

from faro.search.tree_search import TreeSearchEvaluationMode, PlannerConfig


@dataclass(frozen=True, slots=True)
class TreeSearchExperimentSpec:
    # Scene Dependent Modules
    initial_condition_name: str = "default"
    domain_name: str = "default"
    discrete_goal_name: str | None = "place_box_on_table"
    terminal_goal_name: str | None = None
    outcome_scorer_name: str | None = None

    # Tree search configuration
    max_depth: int = 3
    max_iterations: int = 100
    evaluation_mode: TreeSearchEvaluationMode = TreeSearchEvaluationMode.KSO_TO_GOAL
    uct_c: float = 1.4142135623730951
    progressive_widening_k: float | None = 1.0
    progressive_widening_alpha: float = 0.5
    stop_on_first_solution: bool = False
    eval_cost_weight: float = 1.0
    to_cost_weight: float = 1.0
    random_seed: int | None = None
    timeout_s: float | None = None
    shuffle_unexpanded_children: bool = True
    use_node_oracle_cache: bool = False
    use_edge_oracle_cache: bool = False

    require_pass_through_free: bool = True
    max_interfaces_changed_per_step: int = 5

    def to_planner_config(self, *, show_progress_bar: bool = True) -> PlannerConfig:
        return PlannerConfig(
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            evaluation_mode=self.evaluation_mode,
            uct_c=self.uct_c,
            progressive_widening_k=self.progressive_widening_k,
            progressive_widening_alpha=self.progressive_widening_alpha,
            stop_on_first_solution=self.stop_on_first_solution,
            eval_cost_weight=self.eval_cost_weight,
            to_cost_weight=self.to_cost_weight,
            random_seed=self.random_seed,
            timeout_s=self.timeout_s,
            shuffle_unexpanded_children=self.shuffle_unexpanded_children,
            show_progress_bar=show_progress_bar,
        )


def box_placement_experiments(scene_name: str, *, max_depth: int) -> dict[str, TreeSearchExperimentSpec]:
    """The four filter variants shared by the easy and hard box-placement tasks."""
    base = TreeSearchExperimentSpec(
        terminal_goal_name="box_on_table",
        outcome_scorer_name="box_on_table",
        max_depth=max_depth,
        max_iterations=10000,
        uct_c=4.0,
        timeout_s=120 * 60,
        max_interfaces_changed_per_step=4,
    )
    return {
        f"{scene_name}_{name}": replace(base, evaluation_mode=mode,
            use_node_oracle_cache=screen_modes, use_edge_oracle_cache=screen_modes)
        for name, mode, screen_modes in (
            ("KSO", TreeSearchEvaluationMode.KSO_TO_GOAL, False),
            ("MEKSO", TreeSearchEvaluationMode.KSO_TO_GOAL, True),
            ("MEKSOTO", TreeSearchEvaluationMode.KSO_THEN_TO, True),
            ("TO", TreeSearchEvaluationMode.TO_ONLY, False),
        )
    }
