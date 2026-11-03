from __future__ import annotations

from scenes.common.tree_search_experiment_spec import TreeSearchExperimentSpec

EXPERIMENTS: dict[str, TreeSearchExperimentSpec] = {
    "example_scene_default": TreeSearchExperimentSpec(
        initial_condition_name="default",
        domain_name="default",
        discrete_goal_name="place_box_on_table",
        terminal_goal_name="box_center_near_table",
        outcome_scorer_name="box_center_near_table",
        max_depth=3,
        max_iterations=5,
        stop_on_first_solution=False,
    ),
}
