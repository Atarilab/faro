"""Evaluate LLM plans with mode, edge, KSO, and TO checks in one run."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from faro.search.evaluation import ContactModeSeq


# Edit these settings, then run this file.
# Selections accept spaces or commas and run sequentially.
SCENES_TO_RUN = "llm_1a llm_1b llm_2a llm_2b"  # Short or dotted scene IDs
RUNS_TO_RUN = "1 2 3 4 5"       # Any subset, e.g. "1 3"
OUTPUT_DIR = "results"         # Relative to this script, or an absolute path
TO_WORKERS = 1
SQP_ITERS_PER_COL_UPDATE = 5

LLM_DIR = Path(__file__).resolve().parent
ROOT = LLM_DIR.parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from evaluations.runner_utils import (
    BatchSettings, batch_settings, resolve_directory, working_directory,
)
from evaluations.llm.util.mode_edge import (
    build_modespace, measure_plan, measurements_from_rows, plan_nodes, required_checks,
)

CONTACT_PLAN_DIR = LLM_DIR / "contact_plans"


def settings_from_file() -> BatchSettings:
    return batch_settings(LLM_DIR, scenes=SCENES_TO_RUN, runs=RUNS_TO_RUN,
                          output_dir=OUTPUT_DIR)


@dataclass(frozen=True)
class ContactPlan:
    plan: int
    sequence_length: int
    contact_mode_seq: ContactModeSeq


def plan_path(scene: str, run: int) -> Path:
    return CONTACT_PLAN_DIR / scene / f"{scene}_{run}.txt"


def result_path(scene: str, run: int) -> Path:
    return resolve_directory(OUTPUT_DIR, LLM_DIR) / scene / f"run_{run}" / "results.jsonl"


def parse_contact_plans(path: Path, scene) -> list[ContactPlan]:
    from faro.configs.identifiers import import_legacy_contacts
    blocks = re.split(r"^Plan\s+(\d+)\s+\(\d+\s+modes?\):", path.read_text(), flags=re.MULTILINE)
    plans = []
    for number, body in zip(blocks[1::2], blocks[2::2]):
        literal = body.split("contact_mode_seq", 1)[1].split("=", 1)[1].strip()
        schedule = import_legacy_contacts(scene, ast.literal_eval(literal))
        plans.append(ContactPlan(int(number), len(next(iter(schedule.values()))), schedule))
    return plans


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_evaluators(scene_name: str, max_num_nodes: int, ms):
    from faro.evaluators.kin_node import create_node_oracle
    from faro.evaluators.kin_edge import create_edge_oracle
    from faro.evaluators.kso import create_kso_func
    from faro.evaluators.to import create_to_func
    from scenes.common.registry import make_scene_with_state

    scene, resolved = make_scene_with_state(scene_name, "default")
    mode = create_node_oracle(ms, scene, resolved, return_details=True)
    edge = create_edge_oracle(ms, scene, resolved, return_details=True)
    kso = create_kso_func(max_num_nodes=max_num_nodes, scene=scene,
                         resolved_state=resolved, return_details=True)
    to = create_to_func(scene, resolved, return_details=True, n_job=TO_WORKERS)
    return scene, mode, edge, kso, to


def evaluate_plans(scene, mode, edge, kso, to, ms, measurements,
                   run: int, label: str, plans: list[ContactPlan]) -> None:
    for plan in plans:
        print(f"{scene.name}: run {run}, plan {plan.plan} (length={plan.sequence_length})", flush=True)
        checks = measure_plan(plan_nodes(plan, scene.active_patches), measurements, ms, mode, edge)
        print(f"  M feasible={all(check['feasible'] for check in checks['mode_checks'])}; "
              f"E feasible={all(check['feasible'] for check in checks['edge_checks'])}")
        result = kso(plan.contact_mode_seq, sqp_iters_per_collision=SQP_ITERS_PER_COL_UPDATE)
        print(f"  KSO feasible={result.feasible} ({result.solve_time:.6f} sec)")

        details = to(plan.contact_mode_seq)
        to_time = details.solve_time
        print(f"  TO feasible={details.feasible} ({to_time:.6f} sec, "
              f"cost={details.cost}, max_constraint_violation={details.max_constraint_violation})")

        row = {
            "plan": plan.plan,
            "plan_set": scene.name,
            "plan_file": label,
            "scene": scene.name,
            "sequence_length": plan.sequence_length,
            **checks,
            "kso_feasible": result.feasible,
            "to_feasible": bool(details.feasible),
            "kso_solve_time": result.solve_time,
            "to_solve_time": to_time,
            "to_max_constraint_violation": details.max_constraint_violation,
        }
        output = result_path(scene.name, run)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def run_scene(scene_name: str, settings: BatchSettings) -> None:
    from scenes.common.registry import make_scene

    plan_scene = make_scene(scene_name)
    # Read all selected input files before constructing native evaluators. Their
    # maximum length allows the same evaluators to serve every selected run.
    inputs = []
    for run in settings.runs:
        path = plan_path(scene_name, run)
        plans = parse_contact_plans(path, plan_scene)
        if not plans or len({plan.plan for plan in plans}) != len(plans):
            raise ValueError(f"Expected nonempty, uniquely numbered contact plans in {path}")
        inputs.append((run, str(path.relative_to(CONTACT_PLAN_DIR)), plans))

    with ExitStack() as stack:
        evaluators = None
        ms = None
        for run, label, plans in inputs:
            rows = load_result_rows(result_path(scene_name, run))
            by_plan = {(label, plan.plan): plan for plan in plans}
            completed = {(row["plan_file"], row["plan"]) for row in rows}
            if len(completed) != len(rows) or completed - by_plan.keys():
                raise ValueError(f"Duplicate or unexpected results for {scene_name} run={run}")
            saved_plans = [by_plan[row["plan_file"], row["plan"]] for row in rows]
            if any(row["sequence_length"] != plan.sequence_length for row, plan in zip(rows, saved_plans)):
                raise ValueError(f"Saved sequence lengths differ from contact plans for {scene_name} run={run}")
            measurements = measurements_from_rows(rows, [
                plan_nodes(plan, plan_scene.active_patches) for plan in saved_plans])
            pending = [p for p in plans if (label, p.plan) not in completed]
            state = "RUN" if pending else "DONE"
            print(f"{state} {scene_name} run={run}: {len(pending)}/{len(plans)} plans pending"
                  f" -> {result_path(scene_name, run)}", flush=True)
            if not pending:
                continue
            if evaluators is None:
                workspace = settings.solver_cache_dir / scene_name
                print(f"Solver compile directory: {workspace}; TO workers={TO_WORKERS}", flush=True)
                stack.enter_context(working_directory(workspace))
                max_num_nodes = max(p.sequence_length for _, _, ps in inputs for p in ps)
                nodes, _ = required_checks([
                    plan_nodes(plan, plan_scene.active_patches) for _, _, ps in inputs for plan in ps])
                ms = build_modespace(nodes)
                evaluators = build_evaluators(scene_name, max_num_nodes, ms)
            evaluate_plans(*evaluators, ms, measurements, run, label, pending)


def main() -> None:
    settings = settings_from_file()
    print(f"M, E, acados KSO, TO; TO workers={TO_WORKERS}; "
          f"solver cache={settings.solver_cache_dir}", flush=True)
    for scene_name in settings.scenes:
        run_scene(scene_name, settings)


if __name__ == "__main__":
    main()
