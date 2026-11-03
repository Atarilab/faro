from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence

from faro.graph.ct_types import EncodedNode, EncodedPath, GoalAlternatives
from faro.graph.modespace import ModeSpace
from faro.search.evaluation import PathEvaluator, ToEvaluator, coerce_cost_or_false, coerce_to_result
from faro.search.goal import CompiledDiscreteGoal
from faro.search.reporting import IterationProgressBar, PlannerStats, write_result


class GraphView(Protocol):
    node_evaluation_time_s: float
    edge_evaluation_time_s: float

    def symbolic_children(self, node: EncodedNode) -> list[EncodedNode]: ...

    def is_feasible_child(self, parent: EncodedNode, child: EncodedNode) -> bool: ...

    def min_steps_to_discrete_goal(self, node: EncodedNode,
                                   alternatives: GoalAlternatives) -> int | None: ...


class TreeSearchEvaluationMode(str, Enum):
    KSO_TO_GOAL = "kso_to_goal"
    KSO_THEN_TO = "kso_then_to"
    TO_ONLY = "to_only"


@dataclass(slots=True)
class SearchNode:
    state: EncodedNode
    parent: SearchNode | None = None
    children: list[SearchNode] = field(default_factory=list)
    visits: int = 0
    cost_sum: float = 0.0
    depth: int = 0
    evaluation_cost: float | None = None
    symbolic_candidates: list[EncodedNode] | None = None
    next_candidate_idx: int = 0
    subtree_exhausted: bool = False

    @property
    def mean_cost(self) -> float:
        return self.cost_sum / self.visits if self.visits else 0.0

    @property
    def candidate_scan_complete(self) -> bool:
        return (self.symbolic_candidates is not None
                and self.next_candidate_idx == len(self.symbolic_candidates))


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    max_depth: int
    max_iterations: int
    evaluation_mode: TreeSearchEvaluationMode = TreeSearchEvaluationMode.KSO_TO_GOAL
    uct_c: float = 1.4142135623730951
    progressive_widening_k: float | None = 1.0
    progressive_widening_alpha: float = 0.5
    stop_on_first_solution: bool = True
    eval_cost_weight: float = 1.0
    to_cost_weight: float = 1.0
    random_seed: int | None = None
    timeout_s: float | None = None
    shuffle_unexpanded_children: bool = True

    show_progress_bar: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("PlannerConfig.max_depth must be >= 1.")
        if self.max_iterations < 1:
            raise ValueError("PlannerConfig.max_iterations must be >= 1.")
        if not isinstance(self.evaluation_mode, TreeSearchEvaluationMode):
            raise TypeError(
                "PlannerConfig.evaluation_mode must be a TreeSearchEvaluationMode."
            )
        if self.uct_c < 0.0:
            raise ValueError("PlannerConfig.uct_c must be >= 0.")
        if self.progressive_widening_k is not None:
            if isinstance(self.progressive_widening_k, bool) or not isinstance(
                self.progressive_widening_k, (int, float)
            ):
                raise TypeError(
                    "PlannerConfig.progressive_widening_k must be a number or None."
                )
            if not math.isfinite(self.progressive_widening_k):
                raise ValueError(
                    "PlannerConfig.progressive_widening_k must be finite."
                )
            if self.progressive_widening_k <= 0.0:
                raise ValueError(
                    "PlannerConfig.progressive_widening_k must be > 0."
                )
        if isinstance(self.progressive_widening_alpha, bool) or not isinstance(
            self.progressive_widening_alpha, (int, float)
        ):
            raise TypeError(
                "PlannerConfig.progressive_widening_alpha must be a number."
            )
        if not math.isfinite(self.progressive_widening_alpha):
            raise ValueError(
                "PlannerConfig.progressive_widening_alpha must be finite."
            )
        if not 0.0 <= self.progressive_widening_alpha <= 1.0:
            raise ValueError(
                "PlannerConfig.progressive_widening_alpha must be between 0 and 1."
            )
        if self.eval_cost_weight < 0.0:
            raise ValueError("PlannerConfig.eval_cost_weight must be >= 0.")
        if self.to_cost_weight < 0.0:
            raise ValueError("PlannerConfig.to_cost_weight must be >= 0.")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError("PlannerConfig.random_seed must be an int or None.")
        if self.timeout_s is not None:
            if isinstance(self.timeout_s, bool) or not isinstance(
                self.timeout_s, (int, float)
            ):
                raise TypeError("PlannerConfig.timeout_s must be a number or None.")
            if self.timeout_s <= 0.0:
                raise ValueError("PlannerConfig.timeout_s must be > 0.")
        if not isinstance(self.shuffle_unexpanded_children, bool):
            raise TypeError("PlannerConfig.shuffle_unexpanded_children must be a bool.")
        if not isinstance(self.show_progress_bar, bool):
            raise TypeError("PlannerConfig.show_progress_bar must be a bool.")


@dataclass(slots=True)
class PathSolutionRecord:
    encoded_path: EncodedPath
    evaluation_cost: float
    to_score: float
    combined_cost: float
    converged: bool


@dataclass(slots=True)
class TreeSearchResult:
    root: SearchNode
    best_path: EncodedPath | None = None
    best_path_cost: float | None = None
    path_solutions: list[PathSolutionRecord] = field(default_factory=list)
    stats: PlannerStats = field(default_factory=PlannerStats)

    @property
    def converged_solutions(self) -> list[PathSolutionRecord]:
        return [solution for solution in self.path_solutions if solution.converged]

    @property
    def best_path_solution(self) -> PathSolutionRecord | None:
        return min(self.path_solutions, key=lambda s: s.combined_cost, default=None)

    @property
    def best_solution(self) -> PathSolutionRecord | None:
        return min(self.converged_solutions, key=lambda s: s.combined_cost, default=None)


@dataclass
class _RunState:
    goal: CompiledDiscreteGoal
    result: TreeSearchResult
    start_time: float
    deadline: float | None
    to_results: dict[EncodedPath, tuple[bool, float]] = field(default_factory=dict)
    solutions: dict[EncodedPath, PathSolutionRecord] = field(default_factory=dict)

    @property
    def stats(self) -> PlannerStats:
        return self.result.stats

    def timed_out(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def consider_best_path(self, path: EncodedPath, cost: float) -> None:
        if self.result.best_path_cost is None or cost < self.result.best_path_cost:
            self.result.best_path, self.result.best_path_cost = path, cost


class TreeSearchPlanner:
    """Path-based tree search with lazy graph screening and cached TO evaluations.

    States can recur along different paths: graph feasibility is shared, while
    costs and goal solutions are keyed by the entire path. Each iteration grows
    at most one tree node, then backs up its cost without a separate rollout.
    """

    def __init__(self, *, ms: ModeSpace, graph_view: GraphView,
                 evaluator: PathEvaluator | None, to_evaluator: ToEvaluator | None,
                 config: PlannerConfig) -> None:
        if to_evaluator is None:
            raise ValueError("All tree search evaluation modes require a TO evaluator.")
        if config.evaluation_mode is not TreeSearchEvaluationMode.TO_ONLY and evaluator is None:
            raise ValueError(f"{config.evaluation_mode.name} requires a KSO evaluator.")
        self.ms = ms
        self.graph_view = graph_view
        self.evaluator = evaluator
        self.to_evaluator = to_evaluator
        self.config = config
        self._rng = random.Random(config.random_seed)

    def _evaluate_kso(self, path: EncodedPath, run: _RunState) -> float | None:
        schedule = self.ms.decode_contact_mode_seq(path)
        start = time.perf_counter()
        if self.evaluator is None:
            raise RuntimeError("KSO evaluation requested without an evaluator.")
        cost = coerce_cost_or_false(self.evaluator(schedule), source='PathEvaluator')
        run.stats.kso_evaluation_time_s += time.perf_counter() - start
        run.stats.kso_calls += 1
        return cost

    def _evaluate_to(self, path: EncodedPath, run: _RunState) -> tuple[bool, float]:
        if path not in run.to_results:
            schedule = self.ms.decode_contact_mode_seq(path)
            start = time.perf_counter()
            run.to_results[path] = coerce_to_result(self.to_evaluator(schedule))
            run.stats.to_evaluation_time_s += time.perf_counter() - start
            run.stats.to_calls += 1
        return run.to_results[path]

    def _screen_candidate(self, parent: SearchNode, child: EncodedNode,
                          run: _RunState) -> float | None:
        if not self.graph_view.is_feasible_child(parent.state, child):
            return None
        if not run.goal.is_empty():
            remaining = self.config.max_depth - parent.depth - 1
            bound = self.graph_view.min_steps_to_discrete_goal(child, run.goal.alternatives)
            if bound is not None:
                if isinstance(bound, bool):
                    raise TypeError('min_steps_to_discrete_goal must return int | None, not bool.')
                if int(bound) < 0:
                    raise ValueError('min_steps_to_discrete_goal must return a nonnegative integer.')
                if int(bound) > remaining:
                    return None
            if remaining == 0 and not run.goal.matches(child):
                return None

        path = extract_path(parent) + (child,)
        mode = self.config.evaluation_mode
        cost = 0.0
        if mode is not TreeSearchEvaluationMode.TO_ONLY:
            cost = self._evaluate_kso(path, run)
            if cost is None:
                return None
        if mode is not TreeSearchEvaluationMode.KSO_TO_GOAL:
            if run.timed_out():
                return None
            converged, score = self._evaluate_to(path, run)
            if not converged:
                return None
            if mode is TreeSearchEvaluationMode.TO_ONLY:
                cost = score
        return cost

    def _expand(self, parent: SearchNode, run: _RunState) -> SearchNode | None:
        if parent.symbolic_candidates is None:
            parent.symbolic_candidates = list(self.graph_view.symbolic_children(parent.state))
            if self.config.shuffle_unexpanded_children:
                self._rng.shuffle(parent.symbolic_candidates)
        while not parent.candidate_scan_complete:
            if run.timed_out():
                break
            state = parent.symbolic_candidates[parent.next_candidate_idx]
            parent.next_candidate_idx += 1
            cost = self._screen_candidate(parent, state, run)
            # A solver may finish after the deadline; don't add its candidate.
            if run.timed_out():
                break
            if cost is not None:
                child = SearchNode(state, parent=parent, depth=parent.depth + 1,
                                   evaluation_cost=cost)
                parent.children.append(child)
                parent.subtree_exhausted = False
                run.stats.tree_nodes_created += 1
                return child
        return None

    def _select(self, root: SearchNode) -> list[SearchNode]:
        visited = [root]
        current = root
        while current.depth < self.config.max_depth:
            children = [child for child in current.children if not child.subtree_exhausted]
            if self._should_widen(current, has_selectable_children=bool(children)) or not children:
                break
            current = self._select_child_uct(current, children)
            visited.append(current)
        return visited

    def _record_goal(self, path: EncodedPath, cost: float,
                     run: _RunState) -> PathSolutionRecord | None:
        run.stats.goal_reaching_sequences += 1
        if run.timed_out():
            return None
        converged, score = self._evaluate_to(path, run)
        if self.config.evaluation_mode is TreeSearchEvaluationMode.TO_ONLY:
            cost = combined_cost = score
        else:
            combined_cost = self.config.eval_cost_weight * cost + self.config.to_cost_weight * score
        if not math.isfinite(combined_cost):
            raise ValueError("Goal evaluation produced a nonfinite combined cost.")
        solution = PathSolutionRecord(path, cost, score, combined_cost, converged)
        run.solutions[path] = solution
        run.consider_best_path(path, combined_cost)
        if converged and run.stats.time_to_first_converged_goal_solution_s is None:
            run.stats.time_to_first_converged_goal_solution_s = time.perf_counter() - run.start_time
        return solution

    def _backup(self, visited: Sequence[SearchNode], cost: float) -> None:
        for node in reversed(visited):
            node.visits += 1
            node.cost_sum += cost
            node.subtree_exhausted = (
                all(child.subtree_exhausted for child in node.children)
                and (node.depth >= self.config.max_depth or node.candidate_scan_complete))

    def plan(self, *, initial_node: EncodedNode, goal: CompiledDiscreteGoal,
             solutions_txt_path: str | None = None) -> TreeSearchResult:
        self._validate_initial_node(initial_node)
        self._rng = random.Random(self.config.random_seed)
        start = time.perf_counter()
        run = _RunState(goal, TreeSearchResult(SearchNode(initial_node)), start,
                        None if self.config.timeout_s is None else start + self.config.timeout_s)
        progress = IterationProgressBar(total=self.config.max_iterations,
                                        enabled=self.config.show_progress_bar)
        progress.update(0)
        try:
            for _ in range(self.config.max_iterations):
                if run.timed_out():
                    run.stats.stop_reason = 'timeout'
                    break
                visited = self._select(run.result.root)
                current = visited[-1]
                path = extract_path(current)
                # Evaluate a newly reached goal before expanding through it on later visits.
                should_expand = not goal.matches(current.state) or path in run.solutions
                if (current.depth < self.config.max_depth and should_expand
                        and not current.candidate_scan_complete):
                    child = self._expand(current, run)
                    if run.timed_out():
                        run.stats.stop_reason = 'timeout'
                        break
                    if child is not None:
                        current = child
                        visited.append(child)
                path = extract_path(current)
                cost = current.evaluation_cost if current.evaluation_cost is not None else 0.0
                should_stop = False
                if current.evaluation_cost is not None and goal.is_empty():
                    run.consider_best_path(path, cost)
                if goal.matches(current.state):
                    solution = run.solutions.get(path)
                    if solution is None:
                        solution = self._record_goal(path, cost, run)
                        if solution is None:
                            run.stats.stop_reason = 'timeout'
                            break
                        should_stop = solution.converged and self.config.stop_on_first_solution
                    cost = solution.combined_cost
                self._backup(visited, cost)
                run.stats.iterations_completed += 1
                progress.update(run.stats.iterations_completed)
                if run.result.root.subtree_exhausted:
                    run.stats.stop_reason = 'search_exhausted'
                    break
                if should_stop:
                    run.stats.stop_reason = 'first_solution'
                    break
        finally:
            progress.close()
        result = self._finalize(run)
        write_result(self.ms, result, solutions_txt_path)
        return result

    def _finalize(self, run: _RunState) -> TreeSearchResult:
        result, stats = run.result, run.stats
        stats.planner_time_s = time.perf_counter() - run.start_time
        stats.evaluation_mode = self.config.evaluation_mode.value
        for name in ('max_depth', 'max_iterations', 'timeout_s',
                     'progressive_widening_k', 'progressive_widening_alpha'):
            setattr(stats, name, getattr(self.config, name))
        stats.stop_reason = stats.stop_reason or 'max_iterations'
        stats.graph_node_evaluation_time_s = self.graph_view.node_evaluation_time_s
        stats.graph_edge_evaluation_time_s = self.graph_view.edge_evaluation_time_s
        result.path_solutions = list(run.solutions.values())
        stats.converged_goal_solutions = len(result.converged_solutions)
        if result.best_path_cost is not None:
            stats.best_path_cost_found = result.best_path_cost
        if (best := result.best_solution) is not None:
            stats.best_goal_solution_cost_found = best.combined_cost
        return result

    def _validate_initial_node(self, initial_node: EncodedNode) -> None:
        if len(initial_node) != self.ms.n_interfaces:
            raise ValueError(
                "initial_node has the wrong length for this ModeSpace. "
                f"Expected {self.ms.n_interfaces}, got {len(initial_node)}."
            )

        for i, vid in enumerate(initial_node):
            if vid < 0 or vid >= len(self.ms.domains[i]):
                raise ValueError(
                    f"initial_node contains invalid value id {vid} "
                    f"at interface index {i}."
                )

    def _select_child_uct(
        self,
        node: SearchNode,
        selectable_children: Sequence[SearchNode],
    ) -> SearchNode:
        if not selectable_children:
            raise ValueError("Cannot select a UCT child from an empty child set.")

        parent_visits = max(1, node.visits)

        def uct_score(child: SearchNode) -> float:
            if child.visits == 0:
                return float("inf")

            exploit = -child.mean_cost
            explore = self.config.uct_c * math.sqrt(math.log(parent_visits) / child.visits)
            return exploit + explore

        scored_children = [(child, uct_score(child)) for child in selectable_children]
        best_score = max(score for _, score in scored_children)

        tied_children = [
            child
            for child, score in scored_children
            if math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
        ]

        return self._rng.choice(tied_children)

    def _progressive_widening_limit(self, node: SearchNode) -> int | None:
        k = self.config.progressive_widening_k
        if k is None:
            return None

        visits = max(1, node.visits)
        return max(
            1,
            math.ceil(k * visits ** self.config.progressive_widening_alpha),
        )

    def _should_widen(
        self,
        node: SearchNode,
        *,
        has_selectable_children: bool,
    ) -> bool:
        if node.candidate_scan_complete:
            return False

        limit = self._progressive_widening_limit(node)
        if limit is None:
            return True
        if not has_selectable_children:
            return True

        known_feasible_children = len(node.children)
        return known_feasible_children < limit


def extract_path(node: SearchNode) -> EncodedPath:
    path_rev: list[EncodedNode] = []
    cur: SearchNode | None = node
    while cur is not None:
        path_rev.append(cur.state)
        cur = cur.parent
    path_rev.reverse()
    return tuple(path_rev)
