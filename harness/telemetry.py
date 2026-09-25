#!/usr/bin/env python3
"""1 Hz host + accelerator + vLLM telemetry sampler.

CPU and memory are split into harness (vLLM server, bench client, harness
scripts) and background (everything else) so each point can be labelled with
the host condition it actually ran under.

Writes one CSV row per second until SIGINT/SIGTERM. Every source is optional:
a missing tool (nvidia-smi, rocm-smi, RAPL) leaves its columns empty instead
of failing the run.

Usage:
  python harness/telemetry.py --out run_dir/telemetry.csv \
      --metrics-url http://localhost:8000/metrics [--interval 1.0]
"""
import argparse
import csv
import glob
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request

import psutil

from hostload import BackgroundSampler

FIELDS = [
    "ts", "cpu_util_pct", "bg_cpu_pct", "harness_cpu_pct", "cpu_freq_mhz", "load1",
    "mem_used_gb", "mem_avail_gb", "bg_mem_used_gb", "harness_rss_gb",
    "swap_used_gb", "swap_in_mb_s", "cpu_temp_c", "cpu_pkg_power_w",
    "gpu_util_pct", "gpu_mem_used_gb", "gpu_mem_total_gb", "gpu_power_w",
    "gpu_temp_c", "gpu_sm_clock_mhz", "gpu_throttle_reasons",
    "vllm_running", "vllm_waiting", "vllm_kv_cache_usage", "vllm_preemptions_total",
    "vllm_prefix_cache_hits_total", "vllm_prefix_cache_queries_total",
    "vllm_prompt_tokens_total", "vllm_generation_tokens_total",
    "ext_power_w",
]

# vLLM metric names drift between releases; match on any of these aliases.
VLLM_METRICS = {
    "vllm_running": ["vllm:num_requests_running"],
    "vllm_waiting": ["vllm:num_requests_waiting"],
    "vllm_kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "vllm_preemptions_total": ["vllm:num_preemptions_total"],
    "vllm_prefix_cache_hits_total": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "vllm_prefix_cache_queries_total": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
    "vllm_prompt_tokens_total": ["vllm:prompt_tokens_total"],
    "vllm_generation_tokens_total": ["vllm:generation_tokens_total"],
}


class Rapl:
    """Intel/AMD package energy via powercap sysfs (Linux). Needs read access."""

    def __init__(self):
        self.paths = [p for p in glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
                      if p.count(":") == 1 and os.access(p, os.R_OK)]
        self.last = None

    def watts(self):
        if not self.paths:
            return ""
        now = time.monotonic()
        total = sum(int(open(p).read()) for p in self.paths)
        prev, self.last = self.last, (now, total)
        if prev is None or total < prev[1]:  # first sample or counter wrap
            return ""
        return round((total - prev[1]) / 1e6 / (now - prev[0]), 2)


def cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        return ""
    for key in ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz"):
        if temps.get(key):
            return max(t.current for t in temps[key])
    return ""


def nvidia():
    if not shutil.which("nvidia-smi"):
        return {}
    q = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks_throttle_reasons.active"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return {}
    rows = [[c.strip() for c in line.split(",")] for line in out.splitlines() if line]
    if not rows:
        return {}

    def num(i, scale=1.0):
        vals = [float(r[i]) for r in rows if r[i] not in ("", "[N/A]", "N/A")]
        return round(sum(vals) * scale, 2) if vals else ""

    # Multi-GPU: sum memory/power, max util/temp.
    return {
        "gpu_util_pct": max((float(r[0]) for r in rows if r[0].replace(".", "").isdigit()), default=""),
        "gpu_mem_used_gb": num(1, 1 / 1024),
        "gpu_mem_total_gb": num(2, 1 / 1024),
        "gpu_power_w": num(3),
        "gpu_temp_c": max((float(r[4]) for r in rows if r[4].replace(".", "").isdigit()), default=""),
        "gpu_sm_clock_mhz": rows[0][5],
        "gpu_throttle_reasons": "|".join(r[6] for r in rows),
    }


def rocm():
    if not shutil.which("rocm-smi"):
        return {}
    try:
        out = subprocess.run(["rocm-smi", "--showuse", "--showpower", "--showtemp",
                              "--showmemuse", "--csv"], capture_output=True, text=True,
                             timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    lines = [l for l in out.splitlines() if l and not l.startswith("=")]
    if len(lines) < 2:
        return {}
    row = dict(zip(lines[0].split(","), lines[1].split(",")))

    def pick(pattern):
        for k, v in row.items():
            if re.search(pattern, k, re.I):
                return v
        return ""

    return {
        "gpu_util_pct": pick(r"GPU use"),
        "gpu_power_w": pick(r"Power"),
        "gpu_temp_c": pick(r"Temperature.*edge|Temperature"),
    }


def scrape_raw(url):
    """All Prometheus samples at `url`, summed over label sets, keyed by metric name."""
    if not url:
        return {}
    try:
        text = urllib.request.urlopen(url, timeout=2).read().decode()
    except OSError:
        return {}
    sums = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+naN]+)", line)
        if m:
            sums[m.group(1)] = sums.get(m.group(1), 0.0) + float(m.group(3))
    return sums


def scrape_vllm(url):
    sums = scrape_raw(url)
    out = {}
    for col, names in VLLM_METRICS.items():
        for n in names:
            if n in sums:
                out[col] = round(sums[n], 4)
                break
    return out


def ext_power(path):
    """Optional external wall-meter reading: a file holding the latest watts value,
    written by whatever smart plug / meter logger the site has."""
    if not path:
        return ""
    try:
        return float(open(path).read().strip())
    except (OSError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics-url", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--ext-power-file", default=os.environ.get("BENCH_EXT_POWER_FILE", ""))
    args = ap.parse_args()

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    rapl = Rapl()
    rapl.watts()
    bg = BackgroundSampler()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        while not stop:
            t0 = time.monotonic()
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            freq = psutil.cpu_freq()
            b = bg.sample()
            row = {
                "ts": round(time.time(), 3),
                "cpu_util_pct": b["total_cpu_pct"],
                "bg_cpu_pct": b["bg_cpu_pct"],
                "harness_cpu_pct": b["harness_cpu_pct"],
                "bg_mem_used_gb": b["bg_mem_used_gb"],
                "harness_rss_gb": b["harness_rss_gb"],
                "swap_in_mb_s": b["swap_in_mb_s"],
                "cpu_freq_mhz": round(freq.current) if freq else "",
                "load1": round(os.getloadavg()[0], 2),
                "mem_used_gb": round((vm.total - vm.available) / 2**30, 3),
                "mem_avail_gb": round(vm.available / 2**30, 3),
                "swap_used_gb": round(sw.used / 2**30, 3),
                "cpu_temp_c": cpu_temp(),
                "cpu_pkg_power_w": rapl.watts(),
                "ext_power_w": ext_power(args.ext_power_file),
            }
            row.update(nvidia() or rocm())
            row.update(scrape_vllm(args.metrics_url))
            w.writerow(row)
            f.flush()
            time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
