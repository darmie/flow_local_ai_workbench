#!/usr/bin/env python3
"""Aggregate run directories into one tidy CSV (one row per benchmark point).

Each point joins: run manifest (machine/tier/model/mode/condition), the
`vllm bench serve` result JSON, and telemetry reduced over the point's time
window (peak memory, mean/peak power, energy, max temperature, throttling).

Usage: python harness/summarize.py results/ -o results/summary.csv
"""
import argparse
import csv
import glob
import json
import os
import statistics

import yaml

from hostload import classify

HERE = os.path.dirname(os.path.abspath(__file__))
CONDITIONS = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "conditions.yaml")))

MANIFEST_KEYS = ["run_id", "machine_id", "tier", "model", "model_label", "quant", "params_b",
                 "platform", "mode", "engine_profile", "max_model_len", "vllm_version", "vllm_image",
                 "harness_commit", "harness_dirty", "startup_s", "stress_ng",
                 "machine_class", "power_source", "cpu_offload_gb", "gpu_power_limit_w"]
FAILURE_KEYS = ["run_id", "machine_id", "tier", "machine_class", "model", "platform", "engine_profile",
                "power_source", "cpu_offload_gb", "gpu_power_limit_w"]
# Keys pulled from vLLM's result JSON if present (other numeric keys are ignored).
BENCH_KEYS = [
    "completed", "failed", "duration", "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput", "request_goodput",
    "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p90_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
]


def load_spec(run_dir):
    """Machine specification from the run's fingerprint, as spec_* columns."""
    try:
        fp = json.load(open(os.path.join(run_dir, "fingerprint.json")))
    except (OSError, ValueError):
        return {}
    return {f"spec_{k}": v for k, v in fp.get("spec", {}).items()}


def load_telemetry(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def window_stats(rows, t0, t1):
    rows = [r for r in rows if t0 <= float(r["ts"]) <= t1]

    def col(name):
        out = []
        for r in rows:
            try:
                out.append(float(r[name]))
            except (ValueError, TypeError, KeyError):
                pass
        return out

    def agg(name, fn):
        v = col(name)
        return round(fn(v), 3) if v else ""

    # Energy: prefer an external wall meter, else GPU + CPU package (lower bound).
    energy_src, watts = "", []
    if col("ext_power_w"):
        energy_src, watts = "wall", col("ext_power_w")
    elif col("soc_power_w"):
        # Apple Silicon: CPU + GPU + ANE package power from powermetrics.
        energy_src, watts = "soc", col("soc_power_w")
    else:
        g, c = col("gpu_power_w"), col("cpu_pkg_power_w")
        if g or c:
            energy_src = "+".join(s for s, v in (("gpu", g), ("cpu_pkg", c)) if v)
            n = max(len(g), len(c))
            g += [0.0] * (n - len(g))
            c += [0.0] * (n - len(c))
            watts = [a + b for a, b in zip(g, c)]
    throttled = [r.get("gpu_throttle_reasons", "") for r in rows]

    def delta(name):
        v = col(name)
        return v[-1] - v[0] if len(v) >= 2 else None

    drafts, draft_tok, accepted = (delta("vllm_spec_drafts_total"), delta("vllm_spec_draft_tokens_total"),
                                   delta("vllm_spec_accepted_tokens_total"))
    bg_cpu, swap_in = col("bg_cpu_pct"), col("swap_in_mb_s")
    measured = classify(statistics.median(bg_cpu), None, statistics.median(swap_in) if swap_in else 0,
                        CONDITIONS["classes"]) if bg_cpu else ""
    return {
        "median_bg_cpu_pct": round(statistics.median(bg_cpu), 2) if bg_cpu else "",
        "median_bg_mem_used_gb": agg("bg_mem_used_gb", statistics.median),
        "measured_condition": measured,
        "tel_samples": len(rows),
        "peak_mem_used_gb": agg("mem_used_gb", max),
        "peak_gpu_mem_used_gb": agg("gpu_mem_used_gb", max),
        "peak_swap_used_gb": agg("swap_used_gb", max),
        "mean_cpu_util_pct": agg("cpu_util_pct", statistics.mean),
        "mean_gpu_util_pct": agg("gpu_util_pct", statistics.mean),
        "max_cpu_temp_c": agg("cpu_temp_c", max),
        "max_gpu_temp_c": agg("gpu_temp_c", max),
        "min_cpu_freq_mhz": agg("cpu_freq_mhz", min),
        "max_kv_cache_usage": agg("vllm_kv_cache_usage", max),
        "max_waiting": agg("vllm_waiting", max),
        "power_source": energy_src,
        "mean_power_w": round(statistics.mean(watts), 2) if watts else "",
        "peak_power_w": round(max(watts), 2) if watts else "",
        "energy_wh": round(sum(watts) / 3600, 4) if watts else "",  # 1 Hz samples
        "gpu_throttled": any(_is_throttled(t) for t in throttled),
        "preemptions": int(delta("vllm_preemptions_total")) if delta("vllm_preemptions_total") is not None else "",
        "spec_acceptance_rate": round(accepted / draft_tok, 3) if draft_tok and accepted is not None else "",
        "spec_mean_accepted_len": round(1 + accepted / drafts, 2) if drafts and accepted is not None else "",
    }


# nvidia-smi clocks_throttle_reasons bits: SW power cap, HW slowdown, SW/HW thermal, power brake.
_THROTTLE_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


def _is_throttled(field):
    for part in (field or "").split("|"):
        try:
            if int(part, 16) & _THROTTLE_MASK:
                return True
        except ValueError:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("-o", "--out", default="results/summary.csv")
    args = ap.parse_args()

    out_rows, failures = [], []
    for root in args.roots:
        for manifest_path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
            run_dir = os.path.dirname(manifest_path)
            manifest = json.load(open(manifest_path))
            tel = load_telemetry(os.path.join(run_dir, "telemetry.csv"))
            base = {k: manifest.get(k, "") for k in FAILURE_KEYS}
            if manifest.get("startup_failure"):
                failures.append({**base, "phase": "startup", **manifest["startup_failure"], "recovered": False})
            for c in manifest.get("crashes", []):
                failures.append({**base, **{k: c.get(k, "") for k in ("phase", "condition", "scenario", "concurrency",
                                 "repeat", "category", "evidence", "recovered", "recovery_s")}})
            spec = load_spec(run_dir)
            for point in manifest.get("points", []):
                res_path = os.path.join(run_dir, point["result_file"])
                if not os.path.exists(res_path):
                    continue
                res = json.load(open(res_path))
                row = {k: manifest.get(k, "") for k in MANIFEST_KEYS}
                row.update(spec)
                row.update({"target_condition": point["condition"], "scenario": point["scenario"],
                            "concurrency": point["concurrency"], "repeat": point["repeat"],
                            "num_prompts": point.get("num_prompts"),
                            "point_status": point.get("status", "")})
                row.update({k: res.get(k, "") for k in BENCH_KEYS})
                row.update(window_stats(tel, point["t_start"], point["t_end"]))
                order = list(CONDITIONS["classes"])
                target_cls = CONDITIONS["targets"][point["condition"]]["class"]
                row["condition_match"] = (row["measured_condition"] == target_cls
                                          if row["measured_condition"] else "")
                row["condition_heavier"] = (bool(row["measured_condition"]) and
                                            order.index(row["measured_condition"]) > order.index(target_cls))
                if res.get("completed"):
                    row["mean_input_tokens"] = round(res.get("total_input_tokens", 0) / res["completed"], 1)
                if row["request_throughput"]:
                    row["goodput_ratio"] = round((row["request_goodput"] or 0) / row["request_throughput"], 3)
                if row["energy_wh"] and row["total_output_tokens"]:
                    row["wh_per_1k_output_tokens"] = round(
                        row["energy_wh"] / row["total_output_tokens"] * 1000, 4)
                out_rows.append(row)

    fpath = os.path.join(os.path.dirname(os.path.abspath(args.out)), "failures.csv")
    if failures:
        with open(fpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in failures for k in r)))
            w.writeheader()
            w.writerows(failures)
        print(f"wrote {len(failures)} server failures -> {fpath}")
    elif os.path.exists(fpath):
        os.remove(fpath)
    if not out_rows:
        print("no results found")
        return
    fields = list(dict.fromkeys(k for r in out_rows for k in r))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
