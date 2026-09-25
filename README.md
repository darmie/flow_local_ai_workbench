# Flow Local AI Workbench

A reproducible harness for benchmarking local LLMs served with
[vLLM](https://github.com/vllm-project/vllm) on constrained machines: office
laptops (CPU / Iris Xe), gaming laptops and desktops, pro workstations (RTX Ada /
Blackwell, Radeon PRO, Arc Pro), unified-memory machines (Apple Silicon, Strix
Halo, DGX Spark) and multi-GPU servers. Agentic workloads are evaluated through
[Garden](https://github.com/Flow-Research/garden).

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: variables, host conditions (quiet vs under load),
  GPU vs CPU arms, procedure, metrics, SLOs, pitfalls. Start here.
- **[docs/GARDEN_AGENTIC.md](docs/GARDEN_AGENTIC.md)**: agent task success, latency and token cost via Garden.
- **[AGENTS.md](AGENTS.md)**: operating procedure for AI coding agents asked to run the benchmark on a
  machine and write its report.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
tests/smoke.sh                                   # validate the harness (stub vLLM, ~4 min)

python harness/fingerprint.py                    # machine, tier and class are detected; check the "spec" block
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --print-download | sh   # once, online
python harness/run_suite.py --model qwen3-8b-awq --platform cuda                 # quiet, GPU
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --conditions office
python harness/run_suite.py --model qwen3-8b-awq --platform cpu                  # without GPU
python harness/summarize.py results/ -o results/summary.csv
python harness/report.py results/summary.csv -o results/headline.csv --md   # + results/machines.csv
```

Reports name each machine and its specification (make/model, CPU, RAM type and
speed, GPU/VRAM, OS); see METHODOLOGY §6.5.

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
| `harness/probe.py` | Host-side agent on the machine under test for LAN-client runs |
| `harness/flores.py` | Phase 2: FLORES+ parallel-text datasets and tokenizer fertility |
| `tests/*_smoke.sh`, `tests/test_parsers.py` | End-to-end tests (serving suite, Garden runner, LAN mode, Phase 2 pipeline) against stubs, and parser unit tests |

## Issue tracking

Issues live in [git-bug](https://github.com/git-bug/git-bug) (`refs/bugs/*`); see `CLAUDE.md`.
`git-bug pull` fetches them after cloning.
