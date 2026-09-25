#!/usr/bin/env bash
# Phase 2 pipeline smoke test on synthetic parallel text (tests/fixtures/flores_plus),
# a locally trained tokenizer and the stub vLLM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
export PATH="$ROOT/tests/stub:$PATH" BENCH_DATASETS_ROOT="$OUT"
SRC="$ROOT/tests/fixtures/flores_plus"
# Stand-in tokenizer at the pinned snapshot path fertility() reads.
REV=$(python3 -c "import yaml;print(yaml.safe_load(open('$ROOT/configs/models.yaml'))['models']['inkubalm-0.4b']['revision'])")
SNAP="$OUT/hf/hub/models--lelapa--InkubaLM-0.4B/snapshots/$REV"
mkdir -p "$SNAP"
python3 - "$SRC" "$SNAP" <<'PY'
import glob, json, sys
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
texts = [json.loads(l)["text"] for f in glob.glob(sys.argv[1] + "/devtest/eng_Latn.jsonl") for l in open(f)]
tok = Tokenizer(models.BPE(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(texts * 20, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"]))
tok.save(sys.argv[2] + "/tokenizer.json")
PY
python3 "$ROOT/harness/flores.py" build --src "$SRC" --sentences 4 --out "$OUT/datasets/flores"
python3 "$ROOT/harness/flores.py" fertility --src "$SRC" --models inkubalm-0.4b --hf-cache "$OUT/hf" --csv "$OUT/fertility.csv"
python3 "$ROOT/tests/stub/vllm" serve --port 8016 & V=$!
trap 'kill $V 2>/dev/null' EXIT
sleep 1
python3 "$ROOT/harness/run_suite.py" --model inkubalm-0.4b --platform cpu --tier t1-minimal \
  --base-url http://127.0.0.1:8016 --client local --repeats 1 --scenarios flores --concurrency 1 \
  --cooldown 0 --force --results "$OUT/results" >/dev/null
python3 "$ROOT/harness/summarize.py" "$OUT/results" -o "$OUT/summary.csv"
python3 "$ROOT/harness/report.py" "$OUT/summary.csv" -o "$OUT/headline.csv"
python3 - "$OUT" <<'PY'
import csv, sys
out = sys.argv[1]
fert = {r["lang"]: float(r["fertility_vs_eng"]) for r in csv.DictReader(open(out + "/fertility.csv"))}
assert fert["eng_Latn"] == 1.0 and fert["yor_Latn"] > 1.5, fert
tok = {r["lang"]: r for r in csv.DictReader(open(out + "/tokenization.csv"))}
assert set(tok) == {"eng", "swh", "yor", "hau", "ibo", "zul", "xho"}, tok.keys()
assert float(tok["yor"]["input_tokens_vs_eng"]) > 1.5 and float(tok["yor"]["ttft_vs_eng"]) > 1.5
print("flores smoke OK:", out)
PY
