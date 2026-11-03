"""Compare KSO and MEKSO results using the settings in run_experiments.py."""

import statistics
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

from evaluations.llm.run_experiments import (
    load_result_rows, result_path, settings_from_file,
)
from evaluations.llm import run_experiments as llm
from evaluations.llm.util.mode_edge import MeReplay, measurements_from_rows, plan_nodes, replay_me


METRIC_GROUPS = (
    (
        "Counts And Rates",
        (
            ("total_plans", "n", "count"),
            ("avg_sequence_length", "avg_len", "length"),
            ("kso_feasible", "filter_ok", "count"),
            ("to_feasible", "to_ok", "count"),
            ("tp", "tp", "count"),
            ("fp", "fp", "count"),
            ("tn", "tn", "count"),
            ("fn", "fn", "count"),
            ("fnr", "fnr", "rate"),
            ("fpr", "fpr", "rate"),
            ("precision", "prec", "rate"),
            ("recall", "recall", "rate"),
        ),
    ),
    (
        "Timing",
        (
            ("kso_time_feasible", "kso_ok_s", "time"),
            ("kso_time_infeasible", "kso_bad_s", "time"),
            ("kso_time_all", "kso_all_s", "time"),
            ("to_time_feasible", "to_ok_s", "time"),
            ("to_time_infeasible", "to_bad_s", "time"),
            ("to_time_all", "to_all_s", "time"),
        ),
    ),
    (
        "Savings",
        (
            ("baseline_to_time", "base_to_s", "time"),
            ("filtered_time", "filter_s", "time"),
            ("seconds_saved", "saved_s", "time"),
            ("percent_saved", "saved_%", "pct"),
            ("speedup", "speedup", "speedup"),
            ("to_calls_skipped", "to_skip", "count"),
            ("feasible_plans_missed", "missed", "count"),
        ),
    ),
)

SCREENING_METRICS = (
    ("m_calls", "M_calls", "count"),
    ("e_calls", "E_calls", "count"),
    ("cached_m_rejections", "cached_M_reject", "count"),
    ("cached_e_rejections", "cached_E_reject", "count"),
    ("m_rejections", "M_reject", "count"),
    ("e_rejections", "E_reject", "count"),
    ("kso_rejections", "KSO_reject", "count"),
    ("to_rejections", "TO_reject", "count"),
    ("kso_calls", "KSO_calls", "count"),
    ("kso_calls_skipped", "KSO_skip", "count"),
    ("m_time_total", "M_paid_s", "time"),
    ("e_time_total", "E_paid_s", "time"),
    ("kso_time_total", "KSO_paid_s", "time"),
    ("to_time_total", "TO_paid_s", "time"),
    ("speedup_vs_kso", "vs_KSO", "speedup"),
)


def divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def average(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    kso_ok = sum(row["kso_feasible"] for row in rows)
    to_ok = sum(row["to_feasible"] for row in rows)
    tp = sum(row["kso_feasible"] and row["to_feasible"] for row in rows)
    fp = sum(row["kso_feasible"] and not row["to_feasible"] for row in rows)
    tn = sum(not row["kso_feasible"] and not row["to_feasible"] for row in rows)
    fn = sum(not row["kso_feasible"] and row["to_feasible"] for row in rows)

    kso_times = [row["kso_solve_time"] for row in rows]
    to_times = [row["to_solve_time"] for row in rows]
    baseline_time = sum(to_times)
    filtered_time = sum(kso_times) + sum(
        row["to_solve_time"] for row in rows if row["kso_feasible"]
    )
    saved_time = baseline_time - filtered_time

    summary = {
        "total_plans": total,
        "avg_sequence_length": average([row["sequence_length"] for row in rows]),
        "kso_feasible": kso_ok,
        "to_feasible": to_ok,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fnr": divide(fn, tp + fn),
        "fpr": divide(fp, fp + tn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "baseline_to_time": baseline_time,
        "filtered_time": filtered_time,
        "seconds_saved": saved_time,
        "percent_saved": divide(saved_time, baseline_time),
        "speedup": divide(baseline_time, filtered_time),
        "to_calls_skipped": total - kso_ok,
        "feasible_plans_missed": fn,
    }
    for evaluator in ("kso", "to"):
        for suffix, feasible in (("feasible", True), ("infeasible", False), ("all", None)):
            times = [row[f"{evaluator}_solve_time"] for row in rows
                     if feasible is None or row[f"{evaluator}_feasible"] == feasible]
            summary[f"{evaluator}_time_{suffix}"] = average(times)
    return summary


def format_value(value: Any, kind: str, aggregate: bool = False) -> str:
    if value is None:
        return "n/a"
    if kind == "count":
        return f"{value:.1f}" if aggregate else str(int(value))
    formats = {"length": ".2f", "rate": ".3f", "time": ".2f", "pct": ".1%"}
    if kind == "speedup":
        return f"{value:.2f}x"
    return format(value, formats[kind])


def aggregate_value(summaries: list[dict[str, Any]], key: str, kind: str) -> str:
    values = [summary[key] for summary in summaries if summary[key] is not None]
    if not values:
        return "n/a"
    mean = format_value(statistics.fmean(values), kind, aggregate=True)
    std = format_value(statistics.stdev(values), kind, aggregate=True) if len(values) > 1 else "n/a"
    return f"{mean} +/- {std}"


def load_run_results(scene: str, runs: tuple[int, ...]) -> list[tuple[str, list[dict[str, Any]]]]:
    results = []
    for run in runs:
        path = result_path(scene, run)
        if path.is_file():
            results.append((f"{scene}_{run}", load_result_rows(path)))
    return results


def summarize_strategies(rows: list[dict[str, Any]],
                         replay: list[MeReplay]) -> dict[str, dict[str, Any]]:
    """Compare strategies on identical, ordered plans using task solve timings."""
    if len(rows) != len(replay):
        raise ValueError("M/E replay and KSO/TO rows must describe the same plans")
    summaries = {}
    baseline = summarize_results(rows)
    for strategy, checks in (("kso", [MeReplay() for _ in rows]), ("mekso", replay)):
        # Summarize the combined filter decision and total filtering cost.
        effective = [dict(row,
                          kso_feasible=check.accepted and row["kso_feasible"],
                          kso_solve_time=check.m_solve_time + check.e_solve_time +
                          (row["kso_solve_time"] if check.accepted else 0.0))
                     for row, check in zip(rows, checks)]
        summary = summarize_results(effective)
        reached = [row for row, check in zip(rows, checks) if check.accepted]
        # KSO means describe actual reached KSO calls, not combined M/E cost.
        for suffix, feasible in (("feasible", True), ("infeasible", False), ("all", None)):
            summary[f"kso_time_{suffix}"] = average([
                row["kso_solve_time"] for row in reached
                if feasible is None or row["kso_feasible"] == feasible])
        for stage in ("m", "e"):
            summary[f"{stage}_calls"] = sum(getattr(check, f"{stage}_calls") for check in checks)
            summary[f"{stage}_time_total"] = sum(getattr(check, f"{stage}_solve_time") for check in checks)
        for stage in ("cached_m", "cached_e", "m", "e"):
            summary[f"{stage}_rejections"] = sum(check.rejection == stage for check in checks)
        summary.update(
            kso_calls=len(reached), kso_calls_skipped=len(rows) - len(reached),
            kso_rejections=sum(not row["kso_feasible"] for row in reached),
            to_rejections=sum(row["kso_feasible"] and not row["to_feasible"] for row in reached),
            kso_time_total=sum(row["kso_solve_time"] for row in reached),
            to_time_total=sum(row["to_solve_time"] for row in reached if row["kso_feasible"]),
            speedup_vs_kso=divide(baseline["filtered_time"], summary["filtered_time"]),
        )
        summaries[strategy] = summary
    return summaries


def complete_run(scene: str, run: int, rows: list[dict[str, Any]], plan_scene):
    """Match saved rows to the full input and restore proposal order for replay."""
    source = llm.plan_path(scene, run)
    if not source.is_file():
        print(f"Comparison unavailable: missing contact plans {source}")
        return None
    plans = llm.parse_contact_plans(source, plan_scene)
    if not plans or len({plan.plan for plan in plans}) != len(plans):
        raise ValueError(f"Expected nonempty, uniquely numbered contact plans in {source}")
    label = str(source.relative_to(llm.CONTACT_PLAN_DIR))
    by_plan = {(row["plan_file"], row["plan"]): row for row in rows}
    expected = {(label, plan.plan) for plan in plans}
    if len(by_plan) != len(rows) or set(by_plan) - expected:
        raise ValueError(f"Duplicate or unexpected KSO/TO results for {scene} run={run}")
    if set(by_plan) != expected:
        print(f"Incomplete KSO/TO coverage: {scene} run={run}: {len(by_plan)}/{len(plans)} plans; "
              "excluded from comparisons (no solves launched)")
        return None
    ordered = [by_plan[label, plan.plan] for plan in plans]
    if any(row["sequence_length"] != plan.sequence_length for row, plan in zip(ordered, plans)):
        raise ValueError(f"Saved sequence lengths differ from contact plans for {scene} run={run}")
    return ordered, [plan_nodes(plan, plan_scene.active_patches) for plan in plans]


def print_comparison(label: str, summaries: dict[str, list[dict[str, Any]]], *, aggregate: bool = False) -> None:
    title = f"Mean/Std Comparison: {label}" if aggregate else f"Comparison: {label}"
    print(f"\n{title}\n{'-' * len(title)}")
    strategies = ("kso", "mekso")
    groups = METRIC_GROUPS + (("Screening And Reached Solves", SCREENING_METRICS),)
    headers = ("metric", "KSO", "MEKSO")
    for group, metrics in groups:
        rows = []
        for key, heading, kind in metrics:
            values = [aggregate_value(summaries[strategy], key, kind) if aggregate
                      else format_value(summaries[strategy][0][key], kind)
                      for strategy in strategies]
            rows.append((heading, *values))
        widths = [max(len(header), *(len(row[i]) for row in rows))
                  for i, header in enumerate(headers)]
        print(f"\n{group}")
        for row in [headers, *rows]:
            print(f"  {row[0]:<{widths[0]}}  " + "  ".join(
                f"{value:>{widths[i]}}" for i, value in enumerate(row[1:], 1)))


def main() -> None:
    settings = settings_from_file()
    for scene in settings.scenes:
        print(f"LLM Result Summaries: {scene}")
        missing = [str(result_path(scene, run).relative_to(settings.output_dir))
                   for run in settings.runs if not result_path(scene, run).is_file()]
        if missing:
            print(f"Missing result files: {', '.join(missing)}")

        run_results = load_run_results(scene, settings.runs)
        if not run_results:
            continue
        from scenes.common.registry import make_scene
        plan_scene = make_scene(scene)
        runs_by_label = {f"{scene}_{run}": run for run in settings.runs}
        comparisons_by_run = {}
        complete_count = 0
        for label, rows in run_results:
            matched = complete_run(scene, runs_by_label[label], rows, plan_scene)
            if matched is None:
                continue
            complete_count += 1
            rows, paths = matched
            measurements = measurements_from_rows(rows, paths)
            comparisons = summarize_strategies(rows, replay_me(paths, measurements))
            for name, summary in comparisons.items():
                comparisons_by_run.setdefault(name, []).append(summary)
            print_comparison(label, {name: [summary] for name, summary in comparisons.items()})
        if comparisons_by_run:
            print(f"\nComparison coverage: {complete_count}/{len(settings.runs)} complete runs; "
                  "all aggregate columns use these same runs.")
            print_comparison(scene, comparisons_by_run, aggregate=True)

    print("\nRates use TO feasibility as the oracle.")
    print("KSO-only time = KSO on every plan + TO on KSO-feasible plans.")
    print("MEKSO uses an empty M/E history for each run: cached failures, then M, then E, then KSO, then TO.")
    print("MEKSO time charges only reached solves; cached checks cost zero. No solvers are run by statistics.")
    print("filter_ok is the combined filter decision; KSO timing means use reached KSO calls. "
          "TO timing means describe the shared unfiltered measurements.")


if __name__ == "__main__":
    main()
