"""Shared scene, graph, and goal construction for search algorithms."""
from dataclasses import dataclass
from typing import Any

from scenes.common import registry
from faro.graph.contact_mode_graph import ContactModeGraph
from faro.graph.domains import build_modespace_from_allowed
from faro.graph.modespace import ModeSpace
from faro.graph.rules import RuleSet
from faro.graph.rule_def.node_invariant import (
    inv_no_reciprocal_object_contacts, inv_robot_is_grounded,
)
from faro.graph.rule_def.transition import (
    tr_grounded_object_cannot_release_without_robot_contact,
    tr_no_reciprocal_object_contacts_across_transition,
)
from faro.search.goal import CompiledDiscreteGoal, DiscreteGoal, OutcomeScorer, TerminalGoal


@dataclass(frozen=True)
class SearchProblem:
    scene: Any
    resolved_state: Any
    ms: ModeSpace
    graph: ContactModeGraph
    initial_node: tuple[int, ...]
    discrete_goal: CompiledDiscreteGoal
    terminal_goal: TerminalGoal | None
    outcome_scorer: OutcomeScorer | None


def build_search_problem(scene_name, spec):
    scene, resolved = registry.make_scene_with_state(scene_name, spec.initial_condition_name)
    ms = build_modespace_from_allowed(
        scene, registry.get_domains(scene_name, spec.domain_name, scene=scene))
    goal = (DiscreteGoal() if spec.discrete_goal_name is None else
            registry.get_discrete_goal(scene_name, spec.discrete_goal_name, scene=scene))
    terminal = (None if spec.terminal_goal_name is None else registry.get_terminal_goal(
        scene_name, spec.terminal_goal_name, scene=scene, resolved_state=resolved))
    scorer = (None if spec.outcome_scorer_name is None else registry.get_outcome_scorer(
        scene_name, spec.outcome_scorer_name, scene=scene, resolved_state=resolved))
    rules = RuleSet(
        node_invariants=[inv_no_reciprocal_object_contacts(scene, ms), inv_robot_is_grounded(scene, ms)],
        transition_rules=[tr_no_reciprocal_object_contacts_across_transition(scene, ms),
                          tr_grounded_object_cannot_release_without_robot_contact(scene, ms)],
    )
    graph = ContactModeGraph(ms, rules,
        require_pass_through_free=spec.require_pass_through_free,
        max_interfaces_changed_per_step=spec.max_interfaces_changed_per_step)
    return SearchProblem(scene, resolved, ms, graph, ms.encode_contact_mode(resolved.contact_mode),
                         goal.compile(ms), terminal, scorer)
