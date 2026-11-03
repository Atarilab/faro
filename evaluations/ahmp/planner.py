"""Fixed-length categorical CEM using FARO tree-search contact constraints."""
from collections import Counter
from dataclasses import dataclass
from itertools import product
import math
from numbers import Integral
import time

import numpy as np

from evaluations.ahmp.cem import CEMConfig, DiscreteCEM
from faro.graph.contact_mode_graph import ContactModeGraph
from faro.graph.oracle_view import LazyOracleGraphView
from faro.search.evaluation import coerce_cost_or_false, coerce_to_result


@dataclass(frozen=True)
class SearchConfig:
    max_depth: int
    timeout_s: float = 7200.0

    def __post_init__(self):
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, Integral):
            raise ValueError("max_depth must be an integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")


class ContactSequenceSpace:
    """One whole-mode category per switch, with a fixed initial mode.

    Every sample contains exactly max_depth switches and ends in a goal mode.
    Prefix screening uses the tree's transition and remaining-depth goal rules.
    """

    def __init__(self, graph: ContactModeGraph, initial, goal, max_depth: int):
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        self.graph, self.initial, self.goal = graph, tuple(initial), goal
        self.max_depth = max_depth
        modes = tuple(product(*(range(len(d)) for d in graph.ms.domains)))
        terminal_modes = modes if goal.is_empty() else tuple(m for m in modes if goal.matches(m))
        if not terminal_modes:
            raise ValueError("No contact mode satisfies the discrete goal")
        self.categories = (modes,) * (max_depth - 1) + (terminal_modes,)
        self._children = {}

    @property
    def n_values(self):
        return tuple(len(c) for c in self.categories)

    def decode(self, sample):
        if len(sample) != self.max_depth:
            raise ValueError("Expected exactly max_depth mode categories")
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or not 0 <= value < len(categories)
               for value, categories in zip(sample, self.categories)):
            raise ValueError("Invalid mode category")
        return (self.initial,) + tuple(categories[int(value)]
                                       for value, categories in zip(sample, self.categories))

    def valid_shape(self, path):
        return (2 <= len(path) <= self.max_depth + 1 and path[0] == self.initial
                and all(len(node) == self.graph.ms.n_interfaces and
                        all(0 <= vid < len(domain) for vid, domain in zip(node, self.graph.ms.domains))
                        for node in path))

    def symbolic_transition(self, parent, child):
        if not self.graph.rules.node_ok(self.graph.ms, parent):
            return False
        if parent not in self._children:
            self._children[parent] = set(self.graph.neighbors(parent))
        if child not in self._children[parent] or not self.graph.rules.node_ok(self.graph.ms, child):
            return False
        return True

    def within_goal_bound(self, child, depth):
        if not 1 <= depth <= self.max_depth:
            return False
        if not self.goal.is_empty():
            remaining = self.max_depth - depth
            bound = self.graph.min_steps_to_discrete_goal(child, self.goal.alternatives)
            if bound > remaining or (remaining == 0 and not self.goal.matches(child)):
                return False
        return True


class SequenceEvaluator:
    """Evaluate complete CEM plans, with optional FARO prefix pruning.

    Fitness is negative cost because AHMP maximizes. KSO prefix costs are not
    summed: a prefix already represents a complete sequence optimization.
    Both methods call TO only on the full, goal-ending sampled plan.
    A finite TO cost participates in CEM even when the solve does not converge;
    feasibility controls solution counting, not eligibility for elite selection.
    """

    def __init__(self, space, to_evaluator, *, graph_view=None, kso_evaluator=None,
                 eval_cost_weight=1.0, to_cost_weight=1.0):
        if (graph_view is None) != (kso_evaluator is None):
            raise ValueError("The augmented method requires both M/E and KSO")
        for weight in (eval_cost_weight, to_cost_weight):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Cost weights must be finite and nonnegative")
        self.space, self.to = space, to_evaluator
        self.view: LazyOracleGraphView | None = graph_view
        self.kso = kso_evaluator
        self.eval_cost_weight, self.to_cost_weight = eval_cost_weight, to_cost_weight
        self.cache, self.kso_costs, self.to_results = {}, {}, {}
        self.counts = Counter()
        self.times = Counter()

    def _to(self, path):
        if path not in self.to_results:
            start = time.perf_counter()
            self.to_results[path] = coerce_to_result(self.to(self.space.graph.ms.decode_contact_mode_seq(path)))
            self.times["to_evaluation_time_s"] += time.perf_counter() - start
            self.counts["to_calls"] += 1
            if not self.to_results[path][0]:
                self.counts["to_failures"] += 1
        return self.to_results[path]

    def _kso(self, path):
        if path not in self.kso_costs:
            start = time.perf_counter()
            self.kso_costs[path] = coerce_cost_or_false(
                self.kso(self.space.graph.ms.decode_contact_mode_seq(path)), source="KSO")
            self.times["kso_evaluation_time_s"] += time.perf_counter() - start
            self.counts["kso_calls"] += 1
        return self.kso_costs[path]

    def evaluate(self, path, expired, on_goal=None):
        if not self.space.valid_shape(path) or len(path) != self.space.max_depth + 1:
            raise ValueError("Path must start at the initial mode and contain exactly max_depth switches")
        if path in self.cache:
            self.counts["sequence_cache_hits"] += 1
        for depth in range(1, len(path)):
            if expired():
                return None
            prefix = path[:depth + 1]
            if prefix in self.cache:
                result = self.cache[prefix]
            else:
                result = self._evaluate_prefix(prefix, expired)
                if result is None:
                    return None
                # Symbolic and M/E failures are already cheap/cached by state.
                if result["outcome"] not in ("symbolic", "mode", "edge"):
                    self.cache[prefix] = result
            if not result["accepted"]:
                return result
        if result["goal_reached"] and on_goal is not None:
            on_goal(path, result)
        return result

    def _evaluate_prefix(self, path, expired):
        depth = len(path) - 1
        parent, child = path[-2:]
        if not self.space.symbolic_transition(parent, child):
            return self._reject("symbolic", depth)
        complete = depth == self.space.max_depth
        goal = complete and self.space.goal.matches(child)
        kso_cost = None
        to_cost, converged = None, False
        if self.view is not None:
            for node in (parent, child):
                if expired():
                    return None
                if not self.view.ensure_node(node):
                    return self._reject("mode", depth)
            if expired():
                return None
            if not self.view.ensure_edge(parent, child):
                return self._reject("edge", depth)
        # Tree search screens M/E before checking the remaining-depth goal bound.
        if not self.space.within_goal_bound(child, depth):
            return self._reject("symbolic", depth)
        if self.view is not None:
            if expired():
                return None
            kso_cost = self._kso(path)
            if expired():  # Tree search does not admit an expansion after its deadline.
                return None
            if kso_cost is None:
                return self._reject("kso", depth)
            cost = kso_cost
        else:
            cost = None
        # The same full-plan TO path and failed-solve policy apply to both
        # methods. Only symbolic/M/E/KSO pruning removes a sample from fitting.
        if complete:
            if expired():
                return None
            converged, to_cost = self._to(path)
            cost = (to_cost if kso_cost is None else
                    self.eval_cost_weight * kso_cost + self.to_cost_weight * to_cost)
        # A full-plan TO started within budget may finish after the deadline.
        # search records such solutions with within_budget=False for both methods.
        if cost is not None and not math.isfinite(cost):
            raise ValueError("Goal evaluation produced a nonfinite combined cost.")
        return {"outcome": "goal" if goal else "expansion", "depth": depth,
                "accepted": True, "goal_reached": goal, "feasible": goal and converged,
                "cost": cost, "fitness": -cost if cost is not None else None, "kso_cost": kso_cost,
                "to_cost": to_cost, "to_converged": converged if to_cost is not None else None}

    def _reject(self, stage, depth):
        self.counts[f"{stage}_rejections"] += 1
        return {"outcome": stage, "depth": depth, "accepted": False, "goal_reached": False,
                "cost": None, "fitness": None, "feasible": False}


def search(space, evaluator, cem_config: CEMConfig, config: SearchConfig, *, on_generation=None):
    if config.max_depth != space.max_depth:
        raise ValueError("Search and contact-space depth limits must match")
    cem = DiscreteCEM(space.n_values, cem_config)
    start = time.perf_counter()
    expired = lambda: time.perf_counter() - start >= config.timeout_s
    generations, samples = 0, 0
    pruned_by_stage = Counter()
    seen_samples, seen_pruned = set(), set()
    solutions, seen_solutions, seen_goals = [], set(), set()
    first_solution = None
    best = None
    best_fitness = -math.inf

    def record_goal(path, outcome):
        nonlocal first_solution
        seen_goals.add(path)
        if outcome["feasible"] and path not in seen_solutions:
            seen_solutions.add(path)
            elapsed = time.perf_counter() - start
            if first_solution is None:
                first_solution = elapsed
            solutions.append({"path": path, "elapsed_s": elapsed,
                              "within_budget": elapsed <= config.timeout_s, **outcome})

    while not expired():
        population = cem.sample()
        fitness = np.full(cem_config.population_size, -math.inf)
        samples_before = samples
        for i in range(cem_config.population_size):
            if expired():
                break
            sample = population[:, i]
            path = space.decode(sample)
            outcome = evaluator.evaluate(path, expired, record_goal)
            if outcome is None:
                break
            samples += 1
            seen_samples.add(path)
            if not outcome["accepted"]:
                pruned_by_stage[outcome["outcome"]] += 1
                seen_pruned.add(path)
                # Pruning carries no objective value and contributes no elite
                # frequencies, including those of an already evaluated prefix.
                continue
            fitness[i] = outcome["fitness"]
            if fitness[i] > best_fitness:
                best_fitness = float(fitness[i])
                best = {"path": path, **outcome}
        complete = samples - samples_before == cem_config.population_size
        if complete:
            if np.isfinite(fitness).any():
                cem.update(population, fitness)
            generations += 1
        if on_generation is not None:
            on_generation({"generation": generations if complete else generations + 1,
                           "elapsed_s": time.perf_counter() - start,
                           "samples_evaluated": samples,
                           "converged_goal_solutions": len(solutions)})
        if not complete:
            break
    elapsed = time.perf_counter() - start
    stats = {**dict(evaluator.counts), **dict(evaluator.times),
             "generations": generations, "samples": samples, "elapsed_s": elapsed,
             "unique_sampled_plans": len(seen_samples),
             "duplicate_samples": samples - len(seen_samples),
             "pruned_samples": sum(pruned_by_stage.values()),
             "pruned_samples_by_stage": {stage: pruned_by_stage[stage]
                                         for stage in ("symbolic", "mode", "edge", "kso")},
             "unique_pruned_plans": len(seen_pruned),
             "accepted_samples": samples - sum(pruned_by_stage.values()),
             "stop_reason": "timeout", "converged_goal_solutions": len(solutions),
             "goal_reaching_sequences": len(seen_goals),
             "solutions_within_budget": sum(s["within_budget"] for s in solutions),
             "best_goal_solution_cost_found": min((s["cost"] for s in solutions), default=None),
             "time_to_first_converged_goal_solution_s": first_solution}
    for name in ("to_calls", "to_failures", "kso_calls", "mode_rejections", "edge_rejections", "kso_rejections",
                 "to_rejections", "symbolic_rejections", "sequence_cache_hits"):
        stats.setdefault(name, 0)
    for name in ("to_evaluation_time_s", "kso_evaluation_time_s"):
        stats.setdefault(name, 0.0)
    if evaluator.view is not None:
        stats.update(node_evaluation_time_s=evaluator.view.node_evaluation_time_s,
                     edge_evaluation_time_s=evaluator.view.edge_evaluation_time_s)
    return {"stats": stats, "solutions": solutions, "best_candidate": best}
