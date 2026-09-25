#!/usr/bin/env bash
# LAN-client smoke test: probe.py stands in for the machine under test (same host here).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
export PATH="$ROOT/tests/stub:$PATH" BENCH_PROBE_TOKEN=smoke-token
python3 "$ROOT/tests/stub/vllm" serve --port 8014 & V=$!
python3 "$ROOT/harness/probe.py" --port 9119 --bind 127.0.0.1 > "$OUT/probe.log" 2>&1 & P=$!
trap 'kill $V $P 2>/dev/null' EXIT
sleep 1
python3 "$ROOT/harness/run_suite.py" --model llama3.2-3b --platform cpu --tier t1-minimal \
  --base-url http://127.0.0.1:8014 --probe http://127.0.0.1:9119 --client local --repeats 1 \
  --scenarios chat --conditions quiet,office --concurrency 1 --cooldown 0 --force --results "$OUT"
RUN=$(ls -d "$OUT"/*/ | head -1)
python3 - "$RUN" <<'PY'
import csv, json, os, sys
run = sys.argv[1]
m = json.load(open(os.path.join(run, "manifest.json")))
assert m["client"] == "lan", m["client"]
assert os.path.exists(os.path.join(run, "client_fingerprint.json"))
rows = list(csv.DictReader(open(os.path.join(run, "telemetry.csv"))))
assert rows and rows[-1]["vllm_running"] != "", "probe telemetry missing vLLM columns"
assert m["conditions"]["office"][0]["verified"], m["conditions"]
assert len(m["points"]) == 2 and all(p["status"] == "ok" for p in m["points"])
print("lan smoke OK:", run)
PY
