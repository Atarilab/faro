"""Summarize completed selected runs, including failures in success rates."""
from collections import Counter
import json
import math
import statistics

from evaluations.search_batch import CEM_METHODS, completion_summary

TASK_FIELDS = (
    "initial_condition_name", "domain_name", "discrete_goal_name", "terminal_goal_name",
    "outcome_scorer_name", "max_depth", "require_pass_through_free",
    "max_interfaces_changed_per_step", "eval_cost_weight", "to_cost_weight",
    "timeout_s", "stop_on_first_solution",
)


def mean_std(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    if not values:
        return "n/a"
    deviation = f"{statistics.stdev(values):.2f}" if len(values) > 1 else "n/a"
    return f"{statistics.fmean(values):.2f} +/- {deviation}"


def summarize(results):
    stats = [r["stats"] for r in results]
    within_budget = [
        s["time_to_first_converged_goal_solution_s"] is not None
        and s["time_to_first_converged_goal_solution_s"] <= r["metadata"]["scene_spec"]["timeout_s"]
        for r, s in zip(results, stats)]
    return {
        "completed": len(results), "successful_within_budget": sum(within_budget),
        "solutions": mean_std([s["converged_goal_solutions"] for s in stats]),
        "first_solution_s_successes_only": mean_std([
            s["time_to_first_converged_goal_solution_s"] for s, ok in zip(stats, within_budget) if ok]),
        "to_calls": mean_std([s["to_calls"] for s in stats]),
        "to_failures": mean_std([s.get("to_failures") for s in stats]),
        "pruned_samples": mean_std([s.get("pruned_samples") for s in stats]),
        "unique_pruned_plans": mean_std([s.get("unique_pruned_plans") for s in stats]),
        "kso_calls": mean_std([s["kso_calls"] for s in stats]),
        "search_time_s": mean_std([s.get("elapsed_s", s.get("planner_time_s")) for s in stats]),
        "best_successful_goal_cost": mean_std([s.get("best_goal_solution_cost_found") for s in stats]),
        "stop_reasons": dict(Counter(s.get("stop_reason", "unknown") for s in stats)),
    }


def validate_comparison(groups):
    """Reject hidden task/runtime differences between selected method groups."""
    runtime, tasks, seeds, cem_settings, tree_caps = set(), {}, {}, {}, {}
    for scene, method, results in groups:
        for result in results:
            m = result["metadata"]
            runtime.add(json.dumps({**{k: m[k] for k in
                ("schema_version", "source_sha256", "python", "dependencies", "threads")},
                "to_workers": m.get("to_workers")}, sort_keys=True))
            task = {f: m["scene_spec"][f] for f in TASK_FIELDS}
            tasks.setdefault(scene, set()).add(json.dumps(task, sort_keys=True))
            seeds.setdefault((scene, m["run"]), set()).add(m["seed"])
            if method in CEM_METHODS:
                settings = {**m["cem"], "search": m["search"],
                            "to_failure_policy": m.get("cem_to_failure_policy")}
                settings.pop("seed")
                cem_settings.setdefault(scene, set()).add(json.dumps(settings, sort_keys=True))
            else:
                tree_caps.setdefault(scene, set()).add(m["scene_spec"]["max_iterations"])
    if len(runtime) > 1:
        raise ValueError("Selected runs have different source/runtime/thread settings")
    for scene, settings in tasks.items():
        if len(settings) > 1:
            raise ValueError(f"Selected methods have different task/budget settings for {scene}")
    for (scene, run), values in seeds.items():
        if len(values) > 1:
            raise ValueError(f"Selected methods have different seeds for {scene} run {run}")
    for scene, settings in cem_settings.items():
        if len(settings) > 1:
            raise ValueError(f"Selected CEM methods have different sampling/stopping settings for {scene}")
    for scene, caps in tree_caps.items():
        if len(caps) > 1:
            raise ValueError(f"Selected tree search methods have different iteration caps for {scene}")


def load_runs(root, scene, method, runs, timeout_s=7200.):
    results = []
    for run in runs:
        path = root / f"{scene}_{method}" / f"run_{run}" / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("status") == "complete":
                metadata = result["metadata"]
                if (metadata["scene"], metadata["method"], metadata["run"]) != (scene, method, run):
                    raise ValueError(f"Misidentified result: {path}")
                if metadata["scene_spec"]["timeout_s"] != timeout_s:
                    raise ValueError(f"Search budget mismatch for {path}; expected {timeout_s:g}s")
                completion_summary(result, metadata)
                results.append(result)
    # Do not average runs with different settings/code (seed and run ID may differ).
    signatures = set()
    for result in results:
        metadata = json.loads(json.dumps(result["metadata"]))
        metadata.pop("run")
        metadata.pop("seed")
        metadata["scene_spec"].pop("random_seed")
        if metadata["cem"] is not None:
            metadata["cem"].pop("seed")
        signatures.add(json.dumps(metadata, sort_keys=True))
    if len(signatures) > 1:
        raise ValueError(f"Mixed settings/source in {scene} {method}; summarize separate output directories")
    return results


def main(config):
    groups = []
    for scene in config.scenes:
        for method in config.methods:
            results = load_runs(config.output_dir, scene, method, config.runs, config.timeout_s)
            groups.append((scene, method, results))
    validate_comparison(groups)
    for scene, method, results in groups:
        values = summarize(results)
        print(f"\n{scene} {method}: {values.pop('completed')}/{len(config.runs)} completed")
        print(f"  completed_runs: {[r['metadata']['run'] for r in results]}")
        for name, value in values.items():
            print(f"  {name}: {value}")
    print("\nMeans +/- sample SD. First-solution times condition on success within budget; "
          "failed runs remain in completed/success counts. Solutions include final solves "
          "that overran the soft deadline; CEM results flag those individually. "
          "Check stop_reasons: registered tree search runs also have a 10,000-iteration cap.")
