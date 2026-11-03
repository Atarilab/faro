"""Mode/edge measurements stored in plan results and their pruning replay."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import TYPE_CHECKING

from faro.evaluators._results import OracleEvaluationDetails

if TYPE_CHECKING:
    from evaluations.llm.run_experiments import ContactPlan

ModeKey = tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]
EdgeKey = tuple[ModeKey, ModeKey]
ModePath = tuple[ModeKey, ...]


def canonical_mode(mode) -> ModeKey:
    return tuple(sorted((tuple(key), tuple(value)) for key, value in mode.items()))


def plan_nodes(plan: ContactPlan, interfaces) -> ModePath:
    interfaces = tuple(interfaces)
    schedule = plan.contact_mode_seq
    if (not schedule or plan.sequence_length < 1 or
            {len(values) for values in schedule.values()} != {plan.sequence_length}):
        raise ValueError(f"Plan {plan.plan}: contact schedules must have equal, nonzero lengths")
    if set(schedule) - set(interfaces):
        raise ValueError(f"Plan {plan.plan}: unknown contact interfaces")
    return tuple(canonical_mode({key: schedule[key][i] if key in schedule else ()
                                 for key in interfaces})
                 for i in range(plan.sequence_length))


def required_checks(paths: list[ModePath]) -> tuple[list[ModeKey], list[EdgeKey]]:
    """Deduplicate in first-appearance order, preserving edge direction."""
    nodes = list(dict.fromkeys(node for path in paths for node in path))
    edges = list(dict.fromkeys(edge for path in paths for edge in zip(path, path[1:])))
    return nodes, edges


@dataclass
class Measurements:
    nodes: dict[ModeKey, OracleEvaluationDetails] = field(default_factory=dict)
    edges: dict[EdgeKey, OracleEvaluationDetails] = field(default_factory=dict)

    def missing(self, paths: list[ModePath]) -> tuple[list[ModeKey], list[EdgeKey]]:
        nodes, edges = required_checks(paths)
        return ([node for node in nodes if node not in self.nodes],
                [edge for edge in edges if edge not in self.edges])


def _details(feasible, solve_time) -> OracleEvaluationDetails:
    if (not isinstance(feasible, bool) or isinstance(solve_time, bool) or
            not isinstance(solve_time, (int, float)) or
            not math.isfinite(solve_time) or solve_time < 0):
        raise ValueError("M/E measurements require boolean feasibility and finite, nonnegative solve time")
    return OracleEvaluationDetails(feasible, float(solve_time))


def build_modespace(nodes: list[ModeKey]):
    from faro.graph.modespace import ModeSpace

    keys = [key for key, _ in nodes[0]]
    values = {key: set() for key in keys}
    for node in nodes:
        for key, value in node:
            if value:
                values[key].add(value)
    return ModeSpace(keys, {key: sorted(domain) for key, domain in values.items()})


def measure_plan(path: ModePath, measurements: Measurements, ms, mode, edge) -> dict:
    """Measure every mode/edge, reusing completed checks within this run."""
    nodes, edges = measurements.missing([path])
    for key in nodes:
        result = mode(ms.encode(dict(key)))
        measurements.nodes[key] = _details(result.feasible, result.solve_time)
    for key in edges:
        result = edge(*(ms.encode(dict(node)) for node in key))
        measurements.edges[key] = _details(result.feasible, result.solve_time)
    return {
        "mode_checks": [asdict(measurements.nodes[node]) for node in path],
        "edge_checks": [asdict(measurements.edges[key]) for key in zip(path, path[1:])],
    }


def measurements_from_rows(rows: list[dict], paths: list[ModePath]) -> Measurements:
    """Restore only the measurements saved in this run's completed plan rows."""
    if len(rows) != len(paths):
        raise ValueError("Results and contact plans must have the same length")
    measurements = Measurements()
    for row, path in zip(rows, paths):
        for field, keys, entries in (
            ("mode_checks", path, measurements.nodes),
            ("edge_checks", tuple(zip(path, path[1:])), measurements.edges),
        ):
            checks = row[field]
            if len(checks) != len(keys):
                raise ValueError(f"Plan {row['plan']}: {field} length differs from contact plan")
            for key, check in zip(keys, checks):
                result = _details(check["feasible"], check["solve_time"])
                if key in entries and entries[key] != result:
                    raise ValueError(f"Plan {row['plan']}: conflicting repeated {field} measurement")
                entries[key] = result
    return measurements


@dataclass
class MeReplay:
    rejection: str | None = None
    m_calls: int = 0
    e_calls: int = 0
    m_solve_time: float = 0.0
    e_solve_time: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.rejection is None


def replay_me(paths: list[ModePath], cache: Measurements) -> list[MeReplay]:
    """Use measured labels only as checks are reached; never learn from future plans."""
    missing_nodes, missing_edges = cache.missing(paths)
    if missing_nodes or missing_edges:
        raise ValueError(f"Missing M/E measurements: {len(missing_nodes)} modes, {len(missing_edges)} edges")
    seen_nodes, seen_edges = set(), set()
    results = []
    for path in paths:
        result = MeReplay()
        edges = tuple(zip(path, path[1:]))
        if any(node in seen_nodes and not cache.nodes[node].feasible for node in path):
            result.rejection = "cached_m"
        elif any(edge in seen_edges and not cache.edges[edge].feasible for edge in edges):
            result.rejection = "cached_e"
        else:
            for keys, seen, entries, stage in (
                (path, seen_nodes, cache.nodes, "m"), (edges, seen_edges, cache.edges, "e"),
            ):
                for key in keys:
                    if key in seen:
                        continue
                    seen.add(key)
                    check = entries[key]
                    setattr(result, f"{stage}_calls", getattr(result, f"{stage}_calls") + 1)
                    setattr(result, f"{stage}_solve_time", getattr(result, f"{stage}_solve_time") + check.solve_time)
                    if not check.feasible:
                        result.rejection = stage
                        break
                if not result.accepted:
                    break
        results.append(result)
    return results
