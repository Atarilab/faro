"""Shared batch execution, completion records and provenance for FARO search evaluations."""
import ast
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata as packages
import json
import math
import os
from pathlib import Path
import platform
import subprocess

from evaluations.runner_utils import BatchSettings, working_directory

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("faro.tree_easy", "faro.tree_hard")
CEM_METHODS = ("CEM_TO", "CEM_MEKSO")
TREE_METHODS = ("TREE_SEARCH_KSO", "TREE_SEARCH_MEKSO", "TREE_SEARCH_MEKSOTO", "TREE_SEARCH_TO")


@dataclass(frozen=True)
class BatchConfig(BatchSettings):
    methods: tuple[str, ...]
    timeout_s: float = 7200.
    base_seed: int = 0
    population_size: int = 8
    elite_size: int = 4
    min_prob: float = 0.05
    to_workers: int = 1
    progress_interval_s: float = 30.


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprint():
    digest = hashlib.sha256()
    cem_entrypoint = ROOT / "evaluations/ahmp/run_experiments.py"
    paths = [Path(__file__), ROOT / "evaluations/runner_utils.py",
             cem_entrypoint, *(ROOT / "evaluations/ahmp" / name for name in ("cem.py", "planner.py"))]
    for directory in (ROOT / "src/faro", ROOT / "src/scenes"):
        paths.extend(directory.rglob("*.py"))
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        content = path.read_bytes()
        if path == cem_entrypoint:
            # Fingerprint execution code; effective preset settings are in metadata.
            module = ast.parse(content)
            module.body = [node for node in module.body if
                isinstance(node, (ast.Import, ast.ImportFrom)) or
                isinstance(node, ast.FunctionDef) and node.name in ("build_cem_problem", "run_cem")]
            content = ast.dump(module, include_attributes=False).encode()
        digest.update(content)
    return digest.hexdigest()


def runtime_versions():
    versions = {}
    for package in ("numpy", "casadi", "pin", "mujoco", "moto", "acados_template"):
        try:
            versions[package] = packages.version(package)
        except packages.PackageNotFoundError:
            versions[package] = None
    for name in ("moto", "acados"):
        revision = subprocess.run(["git", "-C", str(ROOT / "extern" / name), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True)
        versions[f"{name}_source_commit"] = revision.stdout.strip()
    return versions


def completed(path, metadata):
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get("status") != "complete":
        return False
    if result.get("metadata") != metadata:
        raise ValueError(f"Settings/source changed for {path}; use a separate OUTPUT_DIR")
    completion_summary(result, metadata)
    return True


def completion_summary(result, metadata):
    """Validate a finished run; elapsed search time, not file existence, proves budget use."""
    stats = result["stats"]
    elapsed = stats.get("elapsed_s", stats.get("planner_time_s"))
    budget = metadata["scene_spec"]["timeout_s"]
    reason = stats.get("stop_reason")
    if elapsed is None or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Finished run is missing valid elapsed search time")
    full_budget = reason == "timeout" and elapsed >= budget
    valid = full_budget or (metadata["method"] in TREE_METHODS and (
        reason == "search_exhausted" or
        (reason == "max_iterations" and stats.get("iterations_completed", 0) >=
         metadata["scene_spec"]["max_iterations"]) or
        (reason == "first_solution" and metadata["scene_spec"]["stop_on_first_solution"]
         and stats.get("converged_goal_solutions", 0) > 0)))
    if not valid:
        raise ValueError(f"Run did not finish its configured search: stop={reason!r}, elapsed={elapsed}, budget={budget}")
    return {"full_search_budget_completed": full_budget, "stop_reason": reason,
            "search_budget_s": budget, "elapsed_search_s": elapsed}


def run_tree(scene_name, method, spec, output, *, to_workers=1):
    from faro.search.tree_search_runner import run_experiment
    result = run_experiment(scene_name=scene_name,
        experiment_name=f"{scene_name}_{method.removeprefix('TREE_SEARCH_')}",
        output_path=str(output / "solutions.txt"), spec_override=spec,
        show_progress_bar=False, to_workers=to_workers)
    return {"stats": asdict(result.stats), "solutions": [asdict(s) for s in result.path_solutions]}


def run_batch(args: BatchConfig, *, cem_runner=None):
    """Run the selected experiments sequentially in one process."""
    from scenes.common import registry
    for name, selected, allowed in (("SCENES_TO_RUN", args.scenes, SCENES),
                                     ("METHODS_TO_RUN", args.methods, CEM_METHODS + TREE_METHODS)):
        if not selected or len(set(selected)) != len(selected) or any(s not in allowed for s in selected):
            raise ValueError(f"{name} must contain distinct choices from {allowed}")
    if not args.runs or any(run < 1 for run in args.runs) or len(set(args.runs)) != len(args.runs):
        raise ValueError("RUNS_TO_RUN must contain distinct positive run IDs")
    if args.base_seed < 0 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("Require a nonnegative base seed and a finite, positive search budget")
    if any(m in CEM_METHODS for m in args.methods) and cem_runner is None:
        raise ValueError("CEM methods require a CEM runner")
    fingerprint = source_fingerprint()
    versions = runtime_versions()
    output_root = args.output_dir.resolve()
    selected = []
    skipped = 0
    for scene in args.scenes:
        for method in args.methods:
            variant = (method.removeprefix("TREE_SEARCH_") if method in TREE_METHODS
                       else "TO" if method == "CEM_TO" else "MEKSO")
            original = registry.get_experiment(scene, f"{scene}_{variant}")
            for run in args.runs:
                seed = args.base_seed + run - 1
                timeout = args.timeout_s
                spec = replace(original, random_seed=seed, timeout_s=timeout)
                cem = config = upstream_commit = None
                if method in CEM_METHODS:
                    from evaluations.ahmp.cem import CEMConfig, UPSTREAM_COMMIT as upstream_commit
                    from evaluations.ahmp.planner import SearchConfig

                    cem = CEMConfig(args.population_size, args.elite_size, args.min_prob, seed)
                    config = SearchConfig(spec.max_depth, timeout)
                metadata = json_safe({"schema_version": 9, "scene": scene, "method": method,
                    "run": run, "seed": seed,
                    "cem": asdict(cem) if method in CEM_METHODS else None,
                    "search": asdict(config) if method in CEM_METHODS else None,
                    "scene_spec": asdict(spec),
                    "to_workers": args.to_workers,
                    "cem_to_failure_policy": "returned_cost" if method in CEM_METHODS else None,
                    "source_sha256": fingerprint, "ahmp_commit": upstream_commit,
                    "python": platform.python_version(),
                    "dependencies": versions,
                    "threads": {k: os.environ.get(k) for k in (
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}})
                # JSON normalizes enum values, lists, etc. before resume checks.
                metadata = json.loads(json.dumps(metadata))
                output = output_root / f"{scene}_{method}" / f"run_{run}"
                done = completed(output / "result.json", metadata)
                if done:
                    skipped += 1
                    details = completion_summary(json.loads((output / "result.json").read_text()), metadata)
                    label = "DONE (full search budget)" if details["full_search_budget_completed"] else f"DONE ({details['stop_reason']}; ended early)"
                else:
                    label = "RUN"
                    selected.append((scene, method, spec, cem, config, output, metadata))
                print(f"{label} {scene} {method} run={run} seed={seed} "
                      f"depth={spec.max_depth} budget={timeout:g}s -> {output}", flush=True)
    print(f"{skipped} completed; {len(selected)} scheduled. "
          f"Planned new search time: {len(selected) * args.timeout_s / 3600:g} h "
          "(setup and solver overruns are additional).", flush=True)
    for scene, method, spec, cem, config, output, metadata in selected:
        # Reuse compiled artifacts across runs and restarts of this task.
        solver_directory = output_root / "solver_cache" / f"{scene}_{method}"
        print(f"Solver compile directory: {solver_directory}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "first_solution.npz").unlink(missing_ok=True)
        with working_directory(solver_directory):
            if method in CEM_METHODS:
                result = cem_runner(scene, method, spec, cem, config, output,
                                    to_workers=args.to_workers, progress_interval_s=args.progress_interval_s)
            else:
                result = run_tree(scene, method, spec, output, to_workers=args.to_workers)
        completion = completion_summary(result, metadata)
        write_json(output / "result.json", {"status": "complete", "metadata": metadata,
                                            "completion": completion,
                                            "solver_directory": str(solver_directory), **result})
        print(f"Finished {scene} {method} seed={metadata['seed']}: "
              f"{result['stats']['converged_goal_solutions']} solutions", flush=True)

