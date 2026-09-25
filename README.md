# Flow Local AI Workbench

A reproducible harness for benchmarking local LLMs served with
[vLLM](https://github.com/vllm-project/vllm) on constrained machines, from
office laptops (CPU / Iris Xe) to RTX workstations, Apple Silicon and Strix
Halo. Agentic workloads are evaluated through
[Garden](https://github.com/Flow-Research/garden).

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: variables, host conditions (quiet vs under load),
  GPU vs CPU arms, procedure, metrics, SLOs, pitfalls. Start here.
- **[docs/GARDEN_AGENTIC.md](docs/GARDEN_AGENTIC.md)**: agent task success, latency and token cost via Garden.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
tests/smoke.sh                                   # validate the harness (stub vLLM, ~4 min)

export BENCH_TIER=t2-entry-dgpu BENCH_MACHINE_ID=<unique-name> BENCH_OPERATOR=<you>
hf download Qwen/Qwen3-8B-AWQ                    # once, online
python harness/run_suite.py --model qwen3-8b-awq --platform cuda                 # quiet, GPU
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --conditions office
python harness/run_suite.py --model qwen3-8b-awq --platform cpu                  # without GPU
python harness/summarize.py results/ -o results/summary.csv
python harness/report.py results/summary.csv -o results/headline.csv --md
```

## Layout

| Path | Purpose |
|---|---|
| `configs/` | Tiers, platforms (images and launch), models, scenarios + SLOs, engine profiles, host conditions, Garden tasks |
| `harness/run_suite.py` | Orchestrates one machine × model × platform × profile suite |
| `harness/hostload.py` | Measures existing background load; plans top-up to a target condition |
| `harness/contention.py` | Adds only the missing background load (stress-ng or Python fallback) |
| `harness/telemetry.py` | 1 Hz host/GPU/vLLM telemetry, split into harness vs background |
| `harness/fingerprint.py` | Machine and software snapshot stored with every run |
| `harness/summarize.py`, `harness/report.py` | Point-level CSV and headline table (capacity, energy, flags) |
| `harness/garden_agentic.py` | Garden issue-run driver and scorer |
| `tests/smoke.sh` | End-to-end harness test against a stub vLLM |

## Issue tracking

Issues live in [git-bug](https://github.com/git-bug/git-bug) (`refs/bugs/*`); see `CLAUDE.md`.
`git-bug pull` fetches them after cloning.
