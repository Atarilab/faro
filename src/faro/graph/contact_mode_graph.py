from __future__ import annotations

from itertools import combinations, product

from faro.graph.ct_types import EncodedNode, GoalAlternatives, InterfaceValue
from faro.graph.modespace import ModeSpace
from faro.graph.rules import RuleSet


def contact_target(value: InterfaceValue) -> tuple[int, ...] | None:
    """Ignore contact type; object targets include the object's own face."""
    if not value:
        return None
    if len(value) not in (2, 3):
        raise ValueError(f"Unexpected contact mode tuple shape: {value!r}")
    return tuple(int(v) for v in value[:-1])


class ContactModeGraph:
    """Pure symbolic transitions over integer-encoded contact modes.

    Changing targets passes through FREE unless explicitly disabled. Contact
    type changes on the same target are direct. Neighbor order is deterministic.
    """

    def __init__(self, ms: ModeSpace, rules: RuleSet | None = None, *,
                 require_pass_through_free: bool = True,
                 max_interfaces_changed_per_step: int = 1):
        self.ms = ms
        self.rules = rules or RuleSet()
        self.require_pass_through_free = bool(require_pass_through_free)
        self.max_interfaces_changed_per_step = int(max_interfaces_changed_per_step)
        if self.max_interfaces_changed_per_step < 1:
            raise ValueError("max_interfaces_changed_per_step must be >= 1.")
        self._targets = [[contact_target(value) for value in domain] for domain in ms.domains]
        self._same_target_ids = []
        for targets in self._targets:
            groups: dict[tuple[int, ...], list[int]] = {}
            for vid, target in enumerate(targets):
                if target is not None:
                    groups.setdefault(target, []).append(vid)
            self._same_target_ids.append(groups)

    def _allowed_next_value_ids(self, i: int, current: int) -> list[int]:
        size = len(self.ms.domains[i])
        if not self.require_pass_through_free:
            return [vid for vid in range(size) if vid != current]
        if current == 0:
            return list(range(1, size))
        return [0] + [vid for vid in self._same_target_ids[i].get(self._targets[i][current], [])
                      if vid != current]

    def raw_neighbors(self, node: EncodedNode) -> list[EncodedNode]:
        choices = [self._allowed_next_value_ids(i, vid) for i, vid in enumerate(node)]
        neighbors = []
        for count in range(1, self.max_interfaces_changed_per_step + 1):
            for changed in combinations(range(self.ms.n_interfaces), count):
                for values in product(*(choices[i] for i in changed)):
                    child = list(node)
                    for i, value in zip(changed, values):
                        child[i] = value
                    neighbors.append(tuple(child))
        return neighbors

    def neighbors(self, node: EncodedNode) -> list[EncodedNode]:
        """Filter transitions; child invariants are checked by the lazy view."""
        return [child for child in self.raw_neighbors(node)
                if self.rules.edge_ok(self.ms, node, child)]

    def min_steps_to_discrete_goal(self, node: EncodedNode, alternatives: GoalAlternatives) -> int:
        """Conservative bound: maximum per-interface distance, best alternative.

        Ignoring simultaneous-change limits and rules can only lower the bound.
        """
        def distance(i: int, goal_vid: int) -> int:
            current = node[i]
            if self.ms.domains[i][current] == self.ms.domains[i][goal_vid]:
                return 0
            current_target, goal_target = self._targets[i][current], self._targets[i][goal_vid]
            target_change = (current_target is not None and goal_target is not None
                             and current_target != goal_target)
            return 2 if self.require_pass_through_free and target_change else 1

        return min((max((distance(i, vid) for i, vid in alternative), default=0)
                    for alternative in alternatives), default=0)
