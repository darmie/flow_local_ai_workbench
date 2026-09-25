#!/usr/bin/env bash
# garden_agentic.py smoke test against stub Garden, stub psql and stub vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
export PATH="$ROOT/tests/stub:$PATH" GARDEN_STUB_STATE="$OUT/state.json"
python3 "$ROOT/tests/stub/garden_stub.py" 8012 & G=$!
python3 "$ROOT/tests/stub/vllm" serve --port 8013 & V=$!
trap 'kill $G $V 2>/dev/null' EXIT
sleep 1
GARDEN_URL=http://127.0.0.1:8012 GARDEN_EMAIL=bench@example.local GARDEN_PASSWORD=pw \
GARDEN_AGENT_ID=00000000-0000-0000-0000-000000000001 \
python3 "$ROOT/harness/garden_agentic.py" --run-dir "$OUT/run" --repeats 1 --only compare-quotes,extract-invoice \
  --metrics-url http://127.0.0.1:8013/metrics --force
CSV="$OUT/run/agentic_quiet_p1.csv"
python3 - "$CSV" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {r["task"]: r for r in rows}
assert by["compare-quotes"]["success"] == "True", by["compare-quotes"]
assert by["compare-quotes"]["attachments"] == "3"
assert by["compare-quotes"]["model_time_source"] == "vllm_metrics"
assert by["compare-quotes"]["measured_condition"], "missing measured condition"
assert by["compare-quotes"]["machine_spec"]
print("garden smoke OK:", sys.argv[1])
PY
