#!/usr/bin/env python3
"""Headline table from summary.csv: one row per (machine, model, platform,
profile, condition, scenario) with single-user latency, capacity, peak
throughput, energy and data-quality flags. Rules are in docs/METHODOLOGY.md §6.

Usage: python harness/report.py results/summary.csv -o results/headline.csv [--md]
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

# Constraint settings (power source, offload, power cap) are part of the key so a
# constrained run never merges with the unconstrained baseline.
KEY = ["machine_id", "tier", "machine_class", "model", "quant", "platform", "mode", "engine_profile",
       "power_source", "cpu_offload_gb", "gpu_power_limit_w", "target_condition", "scenario"]
MACHINE_COLS = ["spec_machine_name", "spec_machine_class", "spec_form_factor", "spec_suggested_tier",
                "spec_memory_kind", "spec_model_memory_gb", "spec_gpu_count", "spec_gpu_power_max", "spec_ecc_memory", "spec_cpu_model", "spec_cpu_cores", "spec_cpu_isa", "spec_ram_gb",
                "spec_ram_desc", "spec_gpu", "spec_gpu_driver", "spec_gpu_power_limit", "spec_gpu_pcie",
                "spec_os", "spec_kernel", "spec_power", "spec_power_profile", "spec_summary"]
CAPACITY_GOODPUT = 0.9
UNSTABLE_COV = 0.10


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(rows, col):
    v = [f(r[col]) for r in rows if f(r.get(col)) is not None]
    return round(statistics.median(v), 2) if v else None


def cov(rows, col):
    v = [f(r[col]) for r in rows if f(r.get(col)) is not None]
    if len(v) < 2 or statistics.mean(v) == 0:
        return None
    return round(statistics.stdev(v) / statistics.mean(v), 3)


def tokenization_table(out, path):
    """Phase 2: each flores-<lang> row relative to flores-eng for the same machine,
    model, platform, profile and condition (identical meaning, different language)."""
    base_key = lambda r: tuple(r[k] for k in KEY if k != "scenario")
    eng = {base_key(r): r for r in out if r["scenario"] == "flores-eng"}
    rows = []
    for r in out:
        e = eng.get(base_key(r))
        if not r["scenario"].startswith("flores-") or not e:
            continue

        def ratio(col):
            return round(r[col] / e[col], 3) if r.get(col) and e.get(col) else None

        rows.append({**{k: r[k] for k in KEY if k != "scenario"}, "machine_name": r["machine_name"],
                     "lang": r["scenario"].removeprefix("flores-"),
                     "mean_input_tokens": r["mean_input_tokens"], "input_tokens_vs_eng": ratio("mean_input_tokens"),
                     "c1_ttft_p50_ms": r["c1_ttft_p50_ms"], "ttft_vs_eng": ratio("c1_ttft_p50_ms"),
                     "capacity_users": r["capacity_users"], "capacity_vs_eng": ratio("capacity_users"),
                     "peak_output_tok_s": r["peak_output_tok_s"], "peak_vs_eng": ratio("peak_output_tok_s")})
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} tokenization rows -> {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("-o", "--out", default="results/headline.csv")
    ap.add_argument("--machines-out", help="machine spec table (default: machines.csv next to --out)")
    ap.add_argument("--md", action="store_true", help="also print markdown tables")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.summary)) if r["point_status"] == "ok"]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[tuple(r[k] for k in KEY)][int(r["concurrency"])].append(r)

    out = []
    for key, by_c in sorted(groups.items()):
        c1 = by_c.get(1, [])
        capacity, cap_rows = 0, []
        for c in sorted(by_c):
            pts = by_c[c]
            if (med(pts, "goodput_ratio") or 0) >= CAPACITY_GOODPUT and all(
                    (f(p["failed"]) or 0) == 0 for p in pts):
                capacity, cap_rows = c, pts
        peak_c = max(by_c, key=lambda c: med(by_c[c], "output_throughput") or 0)
        all_pts = [p for pts in by_c.values() for p in pts]
        match = [p["condition_match"] == "True" for p in all_pts if p.get("condition_match") in ("True", "False")]
        c1_cov = cov(c1, "output_throughput")
        out.append({
            **dict(zip(KEY, key)),
            "machine_name": all_pts[0].get("spec_machine_name", ""),
            "machine_spec": all_pts[0].get("spec_summary", ""),
            "repeats_c1": len(c1),
            "c1_ttft_p50_ms": med(c1, "median_ttft_ms"),
            "c1_ttft_p90_ms": med(c1, "p90_ttft_ms"),
            "c1_tpot_p50_ms": med(c1, "median_tpot_ms"),
            "c1_output_tok_s": med(c1, "output_throughput"),
            "mean_input_tokens": med(all_pts, "mean_input_tokens"),
            "c1_spec_acceptance_rate": med(c1, "spec_acceptance_rate"),
            "capacity_users": capacity,
            "capacity_output_tok_s": med(cap_rows, "output_throughput"),
            "capacity_tpot_p90_ms": med(cap_rows, "p90_tpot_ms"),
            "peak_output_tok_s": med(by_c[peak_c], "output_throughput"),
            "peak_at_concurrency": peak_c,
            "peak_mem_used_gb": max((f(p["peak_mem_used_gb"]) or 0 for p in all_pts), default=None),
            "peak_gpu_mem_used_gb": max((f(p.get("peak_gpu_mem_used_gb")) or 0 for p in all_pts), default=None),
            "peak_power_w": max((f(p.get("peak_power_w")) or 0 for p in all_pts), default=None) or None,
            "wh_per_1k_tok_at_capacity": med(cap_rows, "wh_per_1k_output_tokens"),
            "power_source": next((p["power_source"] for p in all_pts if p.get("power_source")), ""),
            "c1_throughput_cov": c1_cov,
            "flag_unstable": c1_cov is not None and c1_cov > UNSTABLE_COV,
            "condition_match_rate": round(sum(match) / len(match), 2) if match else None,
            "flag_throttled": any(p.get("gpu_throttled") == "True" for p in all_pts),
            "flag_swapped": any((f(p.get("peak_swap_used_gb")) or 0) > 0.1 for p in all_pts),
            # KV cache ran out and vLLM paused requests: a memory limit reached without a crash.
            "flag_preempted": any((f(p.get("preemptions")) or 0) > 0 for p in all_pts),
            "max_kv_cache_usage": max((f(p.get("max_kv_cache_usage")) or 0 for p in all_pts), default=None),
        })

    if not out:
        print("no ok points in summary", file=sys.stderr)
        return
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {args.out}", file=sys.stderr)

    tokenization_table(out, os.path.join(os.path.dirname(os.path.abspath(args.out)), "tokenization.csv"))

    # One row per machine; a machine_id whose spec changed between runs is
    # reported so the operator can give the changed hardware a new id.
    machines = {}
    for r in rows:
        spec = {c: r.get(c, "") for c in MACHINE_COLS}
        prev = machines.setdefault(r["machine_id"], {"machine_id": r["machine_id"], "tier": r["tier"], **spec})
        if prev["spec_summary"] != spec["spec_summary"]:
            print(f"WARNING: {r['machine_id']} has differing specs across runs:\n  {prev['spec_summary']}\n"
                  f"  {spec['spec_summary']}\n  use a new BENCH_MACHINE_ID after hardware changes")
    mpath = args.machines_out or os.path.join(os.path.dirname(os.path.abspath(args.out)), "machines.csv")
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["machine_id", "tier"] + MACHINE_COLS)
        w.writeheader()
        w.writerows(machines.values())
    print(f"wrote {len(machines)} machines -> {mpath}", file=sys.stderr)

    if args.md:
        mcols = ["machine_id", "tier", "spec_machine_class", "spec_machine_name", "spec_cpu_model", "spec_cpu_cores", "spec_ram_gb",
                 "spec_ram_desc", "spec_gpu", "spec_gpu_driver", "spec_os", "spec_power"]
        print("\n## Machines\n")
        print("| " + " | ".join(c.removeprefix("spec_") for c in mcols) + " |")
        print("|" + "---|" * len(mcols))
        for m in machines.values():
            print("| " + " | ".join(str(m.get(c, "")) for c in mcols) + " |")
        fpath = os.path.join(os.path.dirname(os.path.abspath(args.summary)), "failures.csv")
        if os.path.exists(fpath):
            fcols = ["machine_id", "model", "platform", "phase", "scenario", "concurrency", "category",
                     "recovered", "recovery_s", "evidence"]
            print("\n## Server failures\n")
            print("| " + " | ".join(fcols) + " |")
            print("|" + "---|" * len(fcols))
            for r in csv.DictReader(open(fpath)):
                print("| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in fcols) + " |")
        print("\n## Results\n")
        cols = ["machine_id", "machine_name", "model", "platform", "target_condition", "scenario", "c1_ttft_p50_ms",
                "c1_tpot_p50_ms", "c1_output_tok_s", "capacity_users", "peak_output_tok_s",
                "wh_per_1k_tok_at_capacity", "flag_unstable", "condition_match_rate"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in out:
            print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
