from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from faro.graph.modespace import ModeSpace

# Node invariants depend only on a node
NodeInvariant = Callable[[ModeSpace, tuple[int, ...]], bool]

# Transition rules depend on parent->child
TransitionRule = Callable[[ModeSpace, tuple[int, ...], tuple[int, ...]], bool]


@dataclass
class RuleSet:
    """
    Pure symbolic pruning hooks.
    These must stay CHEAP and deterministic.

    Later stages:
      - Node IK feasibility is cached by the oracle view.
      - Edge/path IK feasibility goes into oracles (not here).
    """
    node_invariants: Sequence[NodeInvariant] = ()
    transition_rules: Sequence[TransitionRule] = ()

    def node_ok(self, ms: ModeSpace, node: tuple[int, ...]) -> bool:
        return all(inv(ms, node) for inv in self.node_invariants)

    def edge_ok(self, ms: ModeSpace, a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        return all(rule(ms, a, b) for rule in self.transition_rules)