#!/usr/bin/env bash
# Server crash detection and recovery: the stub server is killed mid-point and a
# supervisor restarts it after logging a CUDA OOM line; the suite must classify
# the crash, fail that point, wait for recovery and continue.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
export PATH="$ROOT/tests/stub:$PATH"
LOG="$OUT/server.log"
( while [ ! -f "$OUT/stop" ]; do
    python3 "$ROOT/tests/stub/vllm" serve --port 8017
    echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB" >> "$LOG"
    sleep 3
  done ) & SUP=$!
trap 'touch "$OUT/stop"; pkill -f "vllm serve --port 8017"; kill $SUP 2>/dev/null' EXIT
sleep 1
STUB_CRASH_CONC=2 python3 "$ROOT/harness/run_suite.py" --model llama3.2-3b --platform cpu --tier t1-minimal \
  --base-url http://127.0.0.1:8017 --server-log "$LOG" --external-recovery-timeout 30 --client local \
  --repeats 1 --scenarios chat,generate --concurrency 1,2,4 --cooldown 0 --force --results "$OUT/results"
RC=$?
python3 "$ROOT/harness/summarize.py" "$OUT/results" -o "$OUT/summary.csv" >/dev/null
# Unreachable server: startup failure, exit 3, recorded in the manifest.
python3 "$ROOT/harness/run_suite.py" --model llama3.2-3b --platform cpu --tier t1-minimal \
  --base-url http://127.0.0.1:8099 --client local --repeats 1 --scenarios chat --force --results "$OUT/down" >/dev/null 2>&1
RC_DOWN=$?
python3 - "$OUT" "$RC" "$RC_DOWN" <<'PY'
import csv, glob, json, sys
out, rc, rc_down = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = json.load(open(glob.glob(out + "/results/*/manifest.json")[0]))
assert rc == 2, rc
crash = m["crashes"][0]
assert crash["category"] == "gpu_oom" and crash["recovered"] and crash["phase"] == "during_point", crash
pts = {(p["scenario"], p["concurrency"]): p for p in m["points"]}
assert pts[("chat", 2)]["reason"] == "server crashed: gpu_oom", pts[("chat", 2)]
assert ("chat", 4) not in pts, "sweep must stop at the crash"
assert pts[("generate", 4)]["status"] == "ok", "suite must continue after recovery"
fail = list(csv.DictReader(open(out + "/failures.csv")))
assert fail and fail[0]["category"] == "gpu_oom", fail
down = json.load(open(glob.glob(out + "/down/*/manifest.json")[0]))
assert rc_down == 3 and down["startup_failure"]["category"] == "unreachable", (rc_down, down.get("startup_failure"))
print("crash smoke OK:", out)
PY
