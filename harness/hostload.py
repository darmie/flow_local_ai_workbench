#!/usr/bin/env python3
"""Measure the machine's existing background load and decide how to reach a
target host condition (configs/conditions.yaml).

Background = whole-machine usage minus the vLLM server, the benchmark client
and this harness, so a personal computer is measured as its owner left it.

Usage:
  python harness/hostload.py --target quiet  --out run_dir/hostload_quiet.json
  python harness/hostload.py --target office --out run_dir/hostload_office.json [--no-topup]
Exit status: 0 = proceed (JSON says whether a top-up is needed), 1 = cannot reach target.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import psutil
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# BENCH_EXTRA_HARNESS_PROCS: extra regex of processes that are part of the system
# under test (e.g. Garden's workerd/postgres) and so are not background load.
HARNESS_PAT = re.compile("|".join(filter(None, [
    r"vllm|run_suite\.py|telemetry\.py|hostload\.py|garden_agentic\.py",
    os.environ.get("BENCH_EXTRA_HARNESS_PROCS", "")])), re.I)


def is_harness(proc):
    try:
        if proc.pid == os.getpid():
            return True
        text = proc.name() + " " + " ".join(proc.cmdline())
        return bool(HARNESS_PAT.search(text))
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def parse_typeperf_line(line):
    """One typeperf CSV data row -> (cpu %, available MB), or None for headers/blanks."""
    import csv as _csv
    cells = next(_csv.reader([line.strip()]), [])
    if len(cells) < 3 or cells[0].startswith("(PDH"):
        return None
    try:
        return float(cells[1]), float(cells[2])
    except ValueError:
        return None


class WindowsHost:
    """Whole-Windows CPU and memory when running inside WSL2, where psutil only
    sees the Linux VM. Streams typeperf.exe once per second."""

    PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    TYPEPERF = "/mnt/c/Windows/System32/typeperf.exe"

    def __init__(self):
        self.latest, self.proc = None, None
        if "microsoft" not in os.uname().release.lower() or not os.path.exists(self.TYPEPERF):
            return
        try:
            info = subprocess.run([self.PS, "-NoProfile", "-Command",
                                   "[Environment]::ProcessorCount; "
                                   "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
                                  capture_output=True, text=True, timeout=30).stdout.split()
            self.ncpu, self.total_gb = int(info[0]), int(info[1]) / 2**30
            self.proc = subprocess.Popen([self.TYPEPERF, r"\Processor(_Total)\% Processor Time",
                                          r"\Memory\Available MBytes", "-si", "1"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self.proc = None
            return
        import threading
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parsed = parse_typeperf_line(line)
            if parsed:
                self.latest = parsed

    @property
    def active(self):
        return self.proc is not None and self.latest is not None


class BackgroundSampler:
    """Per-interval split of CPU and memory into harness vs background.

    Keeps psutil.Process objects between calls because cpu_percent() is
    measured since the previous call on the same object.
    """

    def __init__(self):
        self.procs = {}
        self.ncpu = psutil.cpu_count() or 1
        psutil.cpu_percent(None)
        self._refresh()
        self.swap_in = psutil.swap_memory().sin
        self.t = time.monotonic()
        self.win = WindowsHost()

    def _refresh(self):
        live = {}
        for p in psutil.process_iter():
            cached = self.procs.get(p.pid)
            if cached is None:
                try:
                    p.cpu_percent(None)
                    p._bench_harness = is_harness(p)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                cached = p
            live[p.pid] = cached
        self.procs = live

    def sample(self):
        total_cpu = psutil.cpu_percent(None)
        harness_cpu, harness_rss = 0.0, 0
        for p in list(self.procs.values()):
            if not getattr(p, "_bench_harness", False):
                continue
            try:
                harness_cpu += p.cpu_percent(None)
                harness_rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._refresh()
        harness_cpu_pct = min(total_cpu, harness_cpu / self.ncpu)
        vm = psutil.virtual_memory()
        used, mem_total, mem_avail = vm.total - vm.available, vm.total, vm.available
        scope = "local"
        if self.win.active:
            # Measure against the whole Windows host so apps outside WSL2 count as background.
            host_cpu, avail_mb = self.win.latest
            harness_cpu_pct = min(host_cpu, harness_cpu / self.win.ncpu)
            total_cpu = host_cpu
            mem_total = self.win.total_gb * 2**30
            mem_avail = avail_mb * 2**20
            used = mem_total - mem_avail
            scope = "windows-host"
        now, sin = time.monotonic(), psutil.swap_memory().sin
        swap_in_mb_s = (sin - self.swap_in) / 2**20 / max(1e-3, now - self.t)
        self.swap_in, self.t = sin, now
        return {
            "total_cpu_pct": total_cpu,
            "harness_cpu_pct": round(harness_cpu_pct, 2),
            "bg_cpu_pct": round(max(0.0, total_cpu - harness_cpu_pct), 2),
            "harness_rss_gb": round(harness_rss / 2**30, 3),
            "bg_mem_used_gb": round(max(0, used - harness_rss) / 2**30, 3),
            "mem_total_gb": round(mem_total / 2**30, 3),
            "mem_avail_gb": round(mem_avail / 2**30, 3),
            "local_mem_avail_gb": round(vm.available / 2**30, 3),
            "host_scope": scope,
            "swap_in_mb_s": round(swap_in_mb_s, 3),
        }


def foreign_gpu():
    """GPU compute processes that are not ours, plus whole-GPU utilisation when
    none of ours is on the GPU (otherwise utilisation cannot be attributed)."""
    if not shutil.which("nvidia-smi"):
        return {"foreign_gpu_procs": [], "bg_gpu_util_pct": None}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    foreign, ours = [], False
    for line in filter(None, (l.strip() for l in apps.splitlines())):
        pid = int(line.split(",")[0])
        try:
            harness = is_harness(psutil.Process(pid))
        except psutil.NoSuchProcess:
            continue
        ours |= harness
        if not harness:
            foreign.append(line)
    util = None
    if not ours:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split()
        util = max((float(u) for u in out), default=None)
    return {"foreign_gpu_procs": foreign, "bg_gpu_util_pct": util}


def load_conditions():
    return yaml.safe_load(open(os.path.join(HERE, "..", "configs", "conditions.yaml")))


def classify(bg_cpu, bg_gpu, swap_in, classes):
    for name, lim in classes.items():
        if (bg_cpu <= lim["bg_cpu_pct_max"] and (bg_gpu is None or bg_gpu <= lim["bg_gpu_util_pct_max"])
                and swap_in <= lim["swap_in_mb_s_max"]):
            return name
    return list(classes)[-1]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def measure(window):
    s = BackgroundSampler()
    samples = []
    for _ in range(window):
        time.sleep(1)
        samples.append(s.sample())
    gpu = foreign_gpu()
    return {
        "bg_cpu_pct": median([x["bg_cpu_pct"] for x in samples]),
        "bg_cpu_pct_max": max(x["bg_cpu_pct"] for x in samples),
        "bg_mem_used_gb": median([x["bg_mem_used_gb"] for x in samples]),
        "harness_rss_gb": median([x["harness_rss_gb"] for x in samples]),
        "mem_total_gb": samples[-1]["mem_total_gb"],
        "mem_avail_gb": samples[-1]["mem_avail_gb"],
        "local_mem_avail_gb": samples[-1]["local_mem_avail_gb"],
        "host_scope": samples[-1]["host_scope"],
        "swap_in_mb_s": median([x["swap_in_mb_s"] for x in samples]),
        **gpu,
    }


def plan(target_name, m, cfg, allow_topup):
    classes, target = cfg["classes"], cfg["targets"][target_name]
    order = list(classes)
    measured = classify(m["bg_cpu_pct"], m["bg_gpu_util_pct"], m["swap_in_mb_s"], classes)
    want = target["class"]
    out = {"target": target_name, "measured_class": measured, "action": "run", "topup": None}

    if target_name == "quiet":
        if measured != "quiet" or m["foreign_gpu_procs"]:
            out["action"] = "blocked"
            out["reason"] = "machine is not quiet; close the apps listed in top_procs and retry"
        return out

    if order.index(measured) > order.index(want):
        out["action"] = "run"
        out["note"] = f"background already heavier than '{want}'; results will be labelled '{measured}'"
        return out

    cpu_gap = max(0.0, target.get("bg_cpu_pct_min", 0) - m["bg_cpu_pct"])
    mem_goal = target.get("bg_mem_frac_min", 0) * m["mem_total_gb"]
    # Never push free memory below max(2 GB, 10% of RAM): the point is contention, not OOM.
    # Top-up memory is allocated where the harness runs (a WSL2 VM may be smaller than the host).
    mem_room = min(m["mem_avail_gb"], m.get("local_mem_avail_gb", m["mem_avail_gb"])) - max(2.0, 0.1 * m["mem_total_gb"])
    mem_gap = max(0.0, min(mem_goal - m["bg_mem_used_gb"], mem_room))
    if cpu_gap < 2 and mem_gap < 0.25 and not target.get("mem_bandwidth_hog"):
        out["note"] = "organic background load already meets the target"
        return out
    if not allow_topup:
        out["action"] = "run"
        out["note"] = "below target but top-up disabled; results labelled with the measured class"
        return out
    out["action"] = "topup"
    out["topup"] = {"cpu_load_pct": round(cpu_gap, 1), "vm_gb": round(mem_gap, 2),
                    "stream": 1 if target.get("mem_bandwidth_hog") else 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["quiet", "office", "heavy"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int)
    ap.add_argument("--no-topup", action="store_true")
    ap.add_argument("--force", action="store_true", help="proceed even if the target cannot be reached")
    ap.add_argument("--verify", action="store_true", help="only report the measured class (after a top-up)")
    args = ap.parse_args()

    cfg = load_conditions()
    m = measure(args.window or cfg.get("sample_window_s", 30))
    p = plan(args.target, m, cfg, not args.no_topup and not args.verify)
    if args.verify:
        p = {"target": args.target, "measured_class": p["measured_class"], "action": "verify", "topup": None}
    top = sorted(psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]),
                 key=lambda x: x.info["memory_percent"] or 0, reverse=True)[:10]
    batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    report = {**p, "measurement": m, "on_ac_power": None if batt is None else batt.power_plugged,
              "top_procs": [x.info for x in top], "forced": args.force}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[hostload] target={args.target} measured={p['measured_class']} "
          f"bg_cpu={m['bg_cpu_pct']}% bg_mem={m['bg_mem_used_gb']}GB action={p['action']}"
          + (f" topup={p['topup']}" if p["topup"] else "") + (f" ({p.get('note')})" if p.get("note") else ""))
    if batt is not None and not batt.power_plugged:
        print("[hostload] WARNING: on battery; plug in (power state changes clocks)", file=sys.stderr)
    if p["action"] == "blocked" and not args.force:
        print(f"[hostload] {p['reason']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
