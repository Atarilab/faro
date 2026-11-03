from __future__ import annotations

import time
from collections.abc import Callable

from faro.graph.contact_mode_graph import ContactModeGraph
from faro.graph.ct_types import EncodedNode, GoalAlternatives

NodeOracle = Callable[[EncodedNode], bool]
EdgeOracle = Callable[[EncodedNode, EncodedNode], bool]


class LazyOracleGraphView:
    """Cache state/edge feasibility; path-dependent costs belong to search.

    Symbolic expansion checks the parent and transition rules. Child node and
    edge oracles run only when search actually screens a candidate. Caches and
    evaluation timings live as long as this view, including across plan calls.
    """

    def __init__(self, base: ContactModeGraph, *, node_oracle: NodeOracle,
                 edge_oracle: EdgeOracle) -> None:
        self.base = base
        self.node_oracle = node_oracle
        self.edge_oracle = edge_oracle
        self._nodes: dict[EncodedNode, bool] = {}
        self._edges: dict[tuple[EncodedNode, EncodedNode], bool] = {}
        self.node_evaluation_time_s = 0.0
        self.edge_evaluation_time_s = 0.0

    def ensure_node(self, node: EncodedNode) -> bool:
        if node not in self._nodes:
            start = time.perf_counter()
            allowed = self.base.rules.node_ok(self.base.ms, node)
            ok = bool(self.node_oracle(node)) if allowed else False
            self._nodes[node] = ok
            self.node_evaluation_time_s += time.perf_counter() - start
        return self._nodes[node]

    def ensure_edge(self, parent: EncodedNode, child: EncodedNode) -> bool:
        key = (parent, child)
        if key not in self._edges:
            start = time.perf_counter()
            allowed = self.base.rules.edge_ok(self.base.ms, parent, child)
            ok = bool(self.edge_oracle(parent, child)) if allowed else False
            self._edges[key] = ok
            self.edge_evaluation_time_s += time.perf_counter() - start
        return self._edges[key]

    def symbolic_children(self, node: EncodedNode) -> list[EncodedNode]:
        if not self.ensure_node(node):
            return []
        children = []
        for child in self.base.raw_neighbors(node):
            start = time.perf_counter()
            if not self.base.rules.edge_ok(self.base.ms, node, child):
                if (node, child) not in self._edges:
                    self._edges[node, child] = False
                    self.edge_evaluation_time_s += time.perf_counter() - start
            else:
                children.append(child)
        return children

    def is_feasible_child(self, parent: EncodedNode, child: EncodedNode) -> bool:
        return (self.ensure_node(parent) and self.ensure_node(child)
                and self.ensure_edge(parent, child))

    def min_steps_to_discrete_goal(self, node: EncodedNode, alternatives: GoalAlternatives) -> int:
        return self.base.min_steps_to_discrete_goal(node, alternatives)
