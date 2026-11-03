from __future__ import annotations

import math
import sys
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, TextIO

from faro.graph.ct_types import EncodedPath
from faro.graph.modespace import ModeSpace

if TYPE_CHECKING:
    from faro.search.tree_search import TreeSearchResult


@dataclass(slots=True)
class PlannerStats:
    iterations_completed: int = 0
    stop_reason: str = ""
    evaluation_mode: str = "kso_to_goal"
    max_depth: int = 0
    max_iterations: int = 0
    timeout_s: float | None = None
    progressive_widening_k: float | None = 1.0
    progressive_widening_alpha: float = 0.5
    tree_nodes_created: int = 1
    kso_calls: int = 0
    to_calls: int = 0
    goal_reaching_sequences: int = 0
    converged_goal_solutions: int = 0
    best_path_cost_found: float = float("inf")
    best_goal_solution_cost_found: float = float("inf")
    time_to_first_converged_goal_solution_s: float | None = None
    planner_time_s: float = 0.0
    graph_node_evaluation_time_s: float = 0.0
    graph_edge_evaluation_time_s: float = 0.0
    kso_evaluation_time_s: float = 0.0
    to_evaluation_time_s: float = 0.0


class IterationProgressBar:
    def __init__(self, *, total: int, enabled: bool, stream: TextIO | None = None):
        self.total = total
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream

    def update(self, current: int) -> None:
        if self.enabled:
            self.stream.write(f"\rTree search iterations {current}/{self.total}")
            self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _format_value(value: object) -> str:
    return f"{value:.12g}" if isinstance(value, float) and math.isfinite(value) else str(value)


def _write_sequence(ms: ModeSpace, path: EncodedPath, stream: TextIO) -> None:
    stream.write("{\n")
    for name, sequence in ms.decode_contact_mode_seq(path).items():
        stream.write(f'  {name!r}: {sequence},\n')
    stream.write("}\n")


def write_result(ms: ModeSpace, result: TreeSearchResult, output_path: str | None) -> None:
    """Write aggregate statistics and ranked goal candidates after search."""
    if output_path is None:
        return
    ranked = sorted(result.path_solutions, key=lambda solution: solution.combined_cost)
    with open(output_path, 'w', encoding='utf-8') as stream:
        stream.write('Planner stats\n\n')
        for stat in fields(result.stats):
            stream.write(f'{stat.name}: {_format_value(getattr(result.stats, stat.name))}\n')
        stream.write(f'\nRanked goal-reaching solutions: {len(ranked)}\n\n')
        if not ranked:
            stream.write('No goal-reaching solutions found.\n')
        for rank, solution in enumerate(ranked, start=1):
            stream.write(f'Rank: {rank}\n')
            for label, value in (('Combined cost', solution.combined_cost),
                                 ('Evaluation cost', solution.evaluation_cost),
                                 ('TO score', solution.to_score),
                                 ('Converged', solution.converged)):
                stream.write(f'{label}: {_format_value(value)}\n')
            stream.write('Contact mode sequence:\n')
            _write_sequence(ms, solution.encoded_path, stream)
