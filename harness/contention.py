#!/usr/bin/env python3
"""Synthetic background load that tops a machine up to a target condition.

Only the gap between the machine's organic load and the target is added
(hostload.py computes it). Uses stress-ng when installed, else a pure-Python
fallback, which is less precise; the run manifest records which was used.

Usage:
  python harness/contention.py --plan run_dir/hostload_office.json   # uses plan["topup"]
  python harness/contention.py --cpu-load-pct 20 --vm-gb 1.5 --stream 0
Runs until SIGTERM/SIGINT.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time


def stress_ng_cmd(cpu_pct, vm_gb, stream):
    cmd = ["stress-ng", "--quiet", "--timeout", "0"]
    if cpu_pct > 0:
        # --cpu 0 = one worker per logical CPU, each at cpu_pct duty: cpu_pct of total capacity.
        cmd += ["--cpu", "0", "--cpu-load", str(int(round(cpu_pct)))]
    if vm_gb > 0:
        cmd += ["--vm", "1", "--vm-bytes", f"{int(vm_gb * 1024)}M", "--vm-keep"]
    if stream:
        cmd += ["--stream", str(stream)]
    return cmd


def _duty_worker(load_pct):
    period = 0.1
    busy = period * load_pct / 100
    while True:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < busy:
            pass
        time.sleep(max(0.0, period - busy))


def _mem_worker(gb, stream, ready):
    buf = bytearray(int(gb * 2**30))
    ready.set()
    while True:
        if stream:
            # Bandwidth hog: continuous full-buffer copies.
            buf[:] = bytes(len(buf))
        else:
            # Resident-set hog: keep pages touched without burning a core.
            buf[::4096] = bytes([int(time.time()) & 0xFF]) * len(range(0, len(buf), 4096))
            time.sleep(2)


def python_fallback(cpu_pct, vm_gb, stream):
    procs, events = [], []
    if cpu_pct > 0:
        procs += [mp.Process(target=_duty_worker, args=(cpu_pct,), daemon=True)
                  for _ in range(os.cpu_count() or 1)]
    for gb, is_stream in ([(vm_gb, False)] if vm_gb > 0 else []) + [(0.5, True)] * stream:
        events.append(mp.Event())
        procs.append(mp.Process(target=_mem_worker, args=(gb, is_stream, events[-1]), daemon=True))
    for p in procs:
        p.start()
    for e in events:
        e.wait()
    return procs


def mark_ready(path):
    """Signal that allocation transients are over and the load is steady."""
    if path:
        open(path, "w").write(str(time.time()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan")
    ap.add_argument("--cpu-load-pct", type=float, default=0)
    ap.add_argument("--vm-gb", type=float, default=0)
    ap.add_argument("--stream", type=int, default=0)
    ap.add_argument("--ready-file", help="written once the load has reached steady state")
    args = ap.parse_args()
    if args.plan:
        topup = json.load(open(args.plan)).get("topup") or {}
        args.cpu_load_pct = topup.get("cpu_load_pct", 0)
        args.vm_gb = topup.get("vm_gb", 0)
        args.stream = topup.get("stream", 0)

    if shutil.which("stress-ng"):
        cmd = stress_ng_cmd(args.cpu_load_pct, args.vm_gb, args.stream)
        print("[contention] " + " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        time.sleep(10)  # stress-ng vm workers allocate and fill their buffers first
        mark_ready(args.ready_file)
        stop = lambda *_: (proc.terminate(), sys.exit(0))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        proc.wait()
        return

    print("[contention] stress-ng not found; using Python fallback", flush=True)
    procs = python_fallback(args.cpu_load_pct, args.vm_gb, args.stream)
    time.sleep(3)
    mark_ready(args.ready_file)

    def _stop(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
