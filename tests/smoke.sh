#!/usr/bin/env bash
# End-to-end harness smoke test with a stubbed vLLM (no model, no GPU, no network).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
export PATH="$ROOT/tests/stub:$PATH"
python3 "$ROOT/tests/stub/vllm" serve --port 8011 & SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
sleep 1
python3 "$ROOT/harness/run_suite.py" --model llama3.2-3b --platform cpu --tier t1-minimal \
  --base-url http://127.0.0.1:8011 --client local --repeats 2 --scenarios chat,generate \
  --conditions quiet,office --concurrency 1,2 --cooldown 0 --force --results "$OUT"
python3 "$ROOT/harness/summarize.py" "$OUT" -o "$OUT/summary.csv"
python3 "$ROOT/harness/report.py" "$OUT/summary.csv" -o "$OUT/headline.csv" --md
echo "smoke OK: $OUT"
