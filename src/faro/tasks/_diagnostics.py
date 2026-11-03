"""Shared task diagnostics, independent of solver backends."""

import numpy as np


def _to_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    try:
        return list(v)
    except Exception:
        return [v]


def merge_stats_into_full(full_stats_dict, stats_dict):
    """
    Merge per-solve stats_dict into full_stats_dict with consistent row alignment.

    - Builds a GLOBAL iteration counter in full_stats_dict["#it"] (monotonic).
    - Uses n_new = max length of list values in stats_dict (excluding '#it', but ok if included).
    - For each key:
        * if stats_dict has >= n_new values -> append first n_new
        * if stats_dict has < n_new values -> append all, then hold last
        * if missing -> extend by holding previous full value (or 0 if none)
    - Special: "time_tot_ms_cpu" is accumulated across solves (added to previous total).
    """

    sd = {k: _to_list(v) for k, v in (stats_dict or {}).items()}

    # determine how many new rows this solve adds
    n_new = max((len(v) for k, v in sd.items() if k != "#it"), default=0)
    if n_new == 0:
        n_new = len(sd.get("#it", []))
    if n_new == 0:
        return

    full_stats_dict.setdefault("#it", [])
    start_it = len(full_stats_dict["#it"])
    full_stats_dict["#it"].extend(range(start_it, start_it + n_new))

    def extend_hold_last(dst, src):
        """Extend dst by n_new values using src, holding last if src is short; hold dst last if src empty."""
        if len(src) >= n_new:
            dst.extend(src[:n_new])
            return
        if not src:
            fill = dst[-1] if dst else 0
            dst.extend([fill] * n_new)
            return
        dst.extend(src)
        dst.extend([src[-1]] * (n_new - len(src)))

    # ensure keys exist for incoming keys
    for k in sd.keys():
        full_stats_dict.setdefault(k, [])

    # merge incoming keys (except '#it')
    for k, v in sd.items():
        if k == "#it":
            continue

        if k == "time_tot_ms_cpu":
            # accumulate across solves:
            # base = last full value (or 0) + per-solve series (held if needed)
            dst = full_stats_dict.setdefault(k, [])
            base = float(dst[-1]) if dst else 0.0

            # build an aligned increment series (length n_new)
            inc = v[:n_new] if len(v) >= n_new else (v + [v[-1]] * (n_new - len(v)) if v else [0.0] * n_new)

            running = base
            for x in inc:
                try:
                    running += float(x)
                except Exception:
                    # if non-numeric sneaks in, treat as 0 increment
                    pass
                dst.append(running)
        else:
            extend_hold_last(full_stats_dict[k], v)

    # align any keys already in full but missing from this stats_dict
    for k, lst in full_stats_dict.items():
        if k in ("#it",) or k in sd:
            continue
        fill = lst[-1] if lst else 0
        lst.extend([fill] * n_new)


def print_iterate_stats(stats_dict, rows_printed=0, first_iteration: bool = False, debug_qp: bool = False):
    """
    Prints accumulating iterate table.

    Columns:
    #it, time_tot_ms_cpu, res_stat, res_eq, res_ineq, res_comp,
    qp1_stat, qp1_it, qp2_stat, qp2_it, qp3_stat, qp3_it,
    alpha, ||pi||, ||lam||

    Returns the updated printed-row cursor; the caller retains it between solves.
    """

    d = {k: _to_list(v) for k, v in (stats_dict or {}).items()}

    it_list = d.get("#it", [])
    if not it_list:
        if first_iteration:
            print("(stats_dict has no '#it' entries yet)")
        return rows_printed

    def pick_hold(vlist, i):
        if not vlist:
            return None
        return vlist[i] if i < len(vlist) else vlist[-1]

    def pick_blank(vlist, i):
        return vlist[i] if i < len(vlist) else None

    def fmt_float(x, width=10, sci=True):
        if x is None:
            return " " * width
        try:
            xf = float(x)
            s = f"{xf:.2e}" if sci else f"{xf:.2f}"
            return s.rjust(width)
        except Exception:
            return str(x)[:width].rjust(width)

    def fmt_int(x, width=8):
        if x is None:
            return " " * width
        try:
            return f"{int(round(float(x))):d}".rjust(width)
        except Exception:
            return str(x)[:width].rjust(width)

    w_it = 5
    w_time = 10
    w_res = 10
    w_qpstat = 10
    w_qpit = 8
    w_alpha = 10
    w_norm = 8

    if first_iteration:
        rows_printed = 0
        print(
            "  "
            + "#it".rjust(w_it)
            + " " + "time_ms".rjust(w_time)
            + " " + "res_stat".rjust(w_res)
            + " " + "res_eq".rjust(w_res)
            + " " + "res_ineq".rjust(w_res)
            + " " + "res_comp".rjust(w_res)
            + " " + "qp1_stat".rjust(w_qpstat)
            + " " + "qp1_it".rjust(w_qpit)
            + " " + "qp2_stat".rjust(w_qpstat)
            + " " + "qp2_it".rjust(w_qpit)
            + " " + "qp3_stat".rjust(w_qpstat)
            + " " + "qp3_it".rjust(w_qpit)
            + " " + "alpha".rjust(w_alpha)
            + " " + "||pi||".rjust(w_norm)
            + " " + "||lam||".rjust(w_norm)
        )

        if debug_qp:
            for k in ["qp1_it", "qp2_it", "qp3_it", "qp1_stat", "qp2_stat", "qp3_stat"]:
                arr = d.get(k, [])
                nz = []
                for idx, val in enumerate(arr):
                    try:
                        if int(round(float(val))) != 0:
                            nz.append(idx)
                    except Exception:
                        pass
                print(f"[debug] {k}: len={len(arr)}, nonzero_indices={nz[:20]}{'...' if len(nz) > 20 else ''}")

    start = rows_printed
    n_it = len(it_list)
    if start >= n_it:
        return rows_printed

    for i in range(start, n_it):
        parts = [
            fmt_int(pick_hold(d.get("#it", []), i), width=w_it),

            fmt_float(pick_hold(d.get("time_tot_ms_cpu", []), i), width=w_time, sci=False),

            fmt_float(pick_hold(d.get("res_stat", []), i), width=w_res, sci=True),
            fmt_float(pick_hold(d.get("res_eq", []), i), width=w_res, sci=True),
            fmt_float(pick_hold(d.get("res_ineq", []), i), width=w_res, sci=True),
            fmt_float(pick_hold(d.get("res_comp", []), i), width=w_res, sci=True),

            fmt_int(pick_blank(d.get("qp1_stat", []), i), width=w_qpstat),
            fmt_int(pick_blank(d.get("qp1_it", []), i), width=w_qpit),
            fmt_int(pick_blank(d.get("qp2_stat", []), i), width=w_qpstat),
            fmt_int(pick_blank(d.get("qp2_it", []), i), width=w_qpit),
            fmt_int(pick_blank(d.get("qp3_stat", []), i), width=w_qpstat),
            fmt_int(pick_blank(d.get("qp3_it", []), i), width=w_qpit),

            fmt_float(pick_hold(d.get("alpha", []), i), width=w_alpha, sci=True),

            fmt_float(pick_hold(d.get("||pi||", []), i), width=w_norm, sci=False),
            fmt_float(pick_hold(d.get("||lam||", []), i), width=w_norm, sci=False),
        ]
        print("  " + " ".join(parts))

    return n_it


def top_collisions(dist_dict, n=10, do_print=False):
    """Rank finite signed distances across pairs and trajectory nodes."""
    items = []
    for pair_name, rec in dist_dict.items():
        if rec is None or "dist" not in rec:
            continue
        dists = np.asarray(rec["dist"]).reshape(-1)

        for k, d in enumerate(dists):
            if not np.isfinite(d):
                continue
            items.append({"pair": pair_name, "k": int(k), "dist": float(d)})

    if not items:
        if do_print:
            print("(no finite collision distances found)")
        return []

    items.sort(key=lambda x: x["dist"])
    top = items[: int(n)]

    if do_print:
        print(f"{'#':>3} {'k':>6}  {'pair':<40}  {'dist':>12}")
        for i, e in enumerate(top):
            print(f"{i:>3} {e['k']:>6}  {e['pair']:<40}  {e['dist']:>12.6e}")

    return top


def collision_metrics(dist_dict, T, penetration_threshold=1e-3):
    """
    Collision metrics from distances.
    Only counts penetrations with depth > penetration_threshold.
    """

    if T < 1:
        return {
            "total_collisions": 0,
            "max_in_node_collisions": 0,
            "total_penetration": 0.0,
            "max_in_node_penetration": 0.0,
        }

    num_collisions_k = np.zeros(T, dtype=int)
    penetration_sum_k = np.zeros(T, dtype=float)

    thr = float(penetration_threshold)

    for pair_name, rec in dist_dict.items():
        if rec is None or "dist" not in rec:
            continue

        dists = np.asarray(rec["dist"]).reshape(-1)
        if dists.size == 0:
            continue

        K = min(T, dists.size)

        finite = np.isfinite(dists[:K])
        pen = -dists[:K]                  # penetration depth
        deep = finite & (pen > thr)       # only count deeper penetrations

        num_collisions_k[:K] += deep.astype(int)
        penetration_sum_k[:K] += pen * deep

    return {
        "total_collisions": int(np.sum(num_collisions_k)),
        "max_in_node_collisions": int(np.max(num_collisions_k)) if T else 0,
        "total_penetration": float(np.sum(penetration_sum_k)),
        "max_in_node_penetration": float(np.max(penetration_sum_k)) if T else 0.0,
    }


def performance_metrics(solver, include_details=False):
    cost, max_vio, sum_vio, sorted_vios, sorted_vio_names = solver.compute_detailed_metrics()
    metrics = {
        "cost": cost,
        "max_violation": max_vio,
        "sum_violation": sum_vio,
        "sorted_vios": sorted_vios,
        "sorted_vio_names": sorted_vio_names,
    }
    if include_details:
        metrics["sorted_vio_details"] = getattr(solver, "last_violation_details", [])
    return metrics


def print_max_k_constr_violations(perf_metrics, k=20, include_details=False):
    k = min(k, len(perf_metrics["sorted_vio_names"]))
    vios = perf_metrics["sorted_vios"][:k]
    names = perf_metrics["sorted_vio_names"][:k]
    details = perf_metrics.get("sorted_vio_details", [])[:k] if include_details else []

    print(f"\nTop {k} violations")
    print("-" * 80)
    if details:
        print(f"{'#':>3}  {'violation':>14}  {'side':>4}  {'value':>14}  {'lb':>10}  {'ub':>10}  name")
    else:
        print(f"{'#':>3}  {'violation':>14}  name")
    print("-" * 80)
    for i, (v, n) in enumerate(zip(vios, names), 1):
        if details:
            d = details[i - 1]
            print(
                f"{i:>3}  {v:>14.6e}  {d['side']:>4}  "
                f"{d['value']:>14.6e}  {d['lb']:>10.3e}  {d['ub']:>10.3e}  {n}"
            )
        else:
            print(f"{i:>3}  {v:>14.6e}  {n}")
    print("-" * 80)
