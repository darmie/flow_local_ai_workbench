#!/usr/bin/env python3
"""Background host contention for LOADED runs.

Simulates the machine doing other work while it serves the model: an office
laptop with a browser, spreadsheet and sync client open, or a shared server
running other services. Uses stress-ng when installed (preferred, reproducible),
otherwise a pure-Python fallback.

Profiles (fractions are of logical cores):
  office  - 25% of cores at 40% duty + 1 GB memory churn + light disk IO
  heavy   - 50% of cores at 80% duty + memory-bandwidth stream + 2 GB churn
  memband - memory-bandwidth hog only (stresses the decode bottleneck)

Usage: python harness/contention.py --profile office   (runs until SIGTERM)
"""
import argparse
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time

PROFILES = {
    "office":  {"cpu_frac": 0.25, "load": 40, "vm_gb": 1, "stream": 0, "io": 1},
    "heavy":   {"cpu_frac": 0.50, "load": 80, "vm_gb": 2, "stream": 1, "io": 1},
    "memband": {"cpu_frac": 0.0,  "load": 0,  "vm_gb": 0, "stream": 2, "io": 0},
}


def stress_ng_cmd(p):
    n = max(1, int(os.cpu_count() * p["cpu_frac"])) if p["cpu_frac"] else 0
    cmd = ["stress-ng", "--quiet", "--timeout", "0"]
    if n:
        cmd += ["--cpu", str(n), "--cpu-load", str(p["load"])]
    if p["vm_gb"]:
        cmd += ["--vm", "1", "--vm-bytes", f"{p['vm_gb']}G", "--vm-keep"]
    if p["stream"]:
        cmd += ["--stream", str(p["stream"])]
    if p["io"]:
        cmd += ["--hdd", "1", "--hdd-bytes", "256M"]
    return cmd


def _duty_worker(load_pct):
    period = 0.1
    busy = period * load_pct / 100
    while True:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < busy:
            pass
        time.sleep(max(0.0, period - busy))


def _mem_worker(gb):
    buf = bytearray(int(gb * 2**30))
    step = 4096
    while True:
        for i in range(0, len(buf), step):
            buf[i] = (buf[i] + 1) & 0xFF


def python_fallback(p):
    procs = []
    n = max(1, int(os.cpu_count() * p["cpu_frac"])) if p["cpu_frac"] else 0
    for _ in range(n):
        procs.append(mp.Process(target=_duty_worker, args=(p["load"],), daemon=True))
    for _ in range(max(p["vm_gb"] and 1, p["stream"])):
        procs.append(mp.Process(target=_mem_worker, args=(max(p["vm_gb"], 1),), daemon=True))
    for pr in procs:
        pr.start()
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, required=True)
    args = ap.parse_args()
    p = PROFILES[args.profile]

    if shutil.which("stress-ng"):
        cmd = stress_ng_cmd(p)
        print("[contention] " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        signal.signal(signal.SIGTERM, lambda *_: (proc.terminate(), sys.exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (proc.terminate(), sys.exit(0)))
        proc.wait()
    else:
        print("[contention] stress-ng not found; using Python fallback (less precise, "
              "record this in the run notes)", flush=True)
        procs = python_fallback(p)

        def _stop(*_):
            for pr in procs:
                pr.terminate()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
