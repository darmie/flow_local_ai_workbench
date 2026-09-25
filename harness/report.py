#!/usr/bin/env python3
"""Headline table from summary.csv: one row per (machine, model, platform,
profile, condition, scenario) with single-user latency, capacity, peak
throughput, energy and data-quality flags. Rules are in docs/METHODOLOGY.md §6.

Usage: python harness/report.py results/summary.csv -o results/headline.csv [--md]
"""
import argparse
import csv
import statistics
from collections import defaultdict

KEY = ["machine_id", "tier", "model", "quant", "platform", "mode", "engine_profile",
       "target_condition", "scenario"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("-o", "--out", default="results/headline.csv")
    ap.add_argument("--md", action="store_true", help="also print a markdown table")
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
            "repeats_c1": len(c1),
            "c1_ttft_p50_ms": med(c1, "median_ttft_ms"),
            "c1_ttft_p90_ms": med(c1, "p90_ttft_ms"),
            "c1_tpot_p50_ms": med(c1, "median_tpot_ms"),
            "c1_output_tok_s": med(c1, "output_throughput"),
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
        })

    if not out:
        print("no ok points in summary")
        return
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows -> {args.out}")
    if args.md:
        cols = ["machine_id", "model", "platform", "target_condition", "scenario", "c1_ttft_p50_ms",
                "c1_tpot_p50_ms", "c1_output_tok_s", "capacity_users", "peak_output_tok_s",
                "wh_per_1k_tok_at_capacity", "flag_unstable", "condition_match_rate"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in out:
            print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
