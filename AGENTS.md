# Agent guide: running and reporting benchmarks

This file is for AI coding agents (Claude Code, Codex, Cursor and similar)
asked to benchmark a team member's machine with this harness. The methodology
itself is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md) and
[docs/GARDEN_AGENTIC.md](docs/GARDEN_AGENTIC.md). This guide is the operating
procedure an agent follows so every machine gets the same treatment.

Commit and issue rules are in [CLAUDE.md](CLAUDE.md).

## Ground rules

- **The machine belongs to someone.** Ask the user before you:
  - close or kill any process;
  - install system packages (Docker, drivers, stress-ng);
  - download models larger than a few GB;
  - add synthetic background load.

  Never kill a process yourself. The quiet check lists the heaviest processes;
  show that list to the user and let them close things.
- **Do not change controlled variables to make a run succeed.** Leave these as
  they are:
  - `configs/scenarios.yaml` (lengths, SLOs);
  - `configs/tiers.yaml` (sweeps, `max_model_len`);
  - `configs/engine_profiles.yaml`;
  - the pinned image tags.

  If a model does not fit, that is a result: record it and move to the next
  model. Do not lower `--max-model-len` or `--gpu-memory-utilization`. If a
  deviation is truly needed, get the user's approval, pass it through
  `--extra-serve-args`, and state it in `--notes` and in the report.
- **Never edit anything under `results/`.** Summaries are regenerated from raw
  run directories. If a run is bad, run it again; the old directory stays.
- **Report what happened, including failures.**
  - A failed point, a flag or a condition mismatch goes in the report.
  - Never smooth, round away, extrapolate or omit numbers.
  - Never present stub or smoke-test output as a benchmark result.
- **Use `--force` only with the user's approval.** It records a non-quiet run
  as-is, and the report must say so.
- **Keep the harness checkout clean** (`git status` empty) while measuring. A
  dirty checkout is flagged in every result.

## 1. Establish the machine

Run each step and confirm with the user where noted.

1. **Fingerprint.** Run `python harness/fingerprint.py` and read the `spec`
   block.
2. **Choose the tier** from the fingerprint using this table, then confirm it
   with the user:

   | Condition | Tier |
   |---|---|
   | 2+ discrete GPUs | `t4-multi-gpu` |
   | One discrete GPU with ≥ 20 GB VRAM, Apple Silicon with ≥ 48 GB, or AMD Strix Halo | `t3-pro` |
   | One discrete GPU with 8–16 GB VRAM, or Apple Silicon with < 48 GB | `t2-entry-dgpu` |
   | No discrete GPU (CPU and/or integrated graphics) | `t1-minimal` |

3. **Name the machine.**
   - If `spec.machine_name` is missing or generic (a hostname, "vm", "System
     Product Name"), ask the user for the make and model. Set it as
     `BENCH_MACHINE_NAME`.
   - If `spec.ram_desc` is `(type unknown)`, ask the user to run
     `sudo dmidecode -t memory | grep -E "Type:|Speed:"`, or run it yourself if
     they approve sudo. Set `BENCH_RAM_DESC`, for example
     `DDR5-4800 dual-channel`.
   - Choose a stable `BENCH_MACHINE_ID` (lowercase, no spaces, e.g.
     `ade-thinkpad-x1`) and set `BENCH_OPERATOR` to the user's name.
4. **Save the settings.** Export them for the session, and suggest the user adds
   them to their shell profile:

   ```bash
   export BENCH_TIER=... BENCH_MACHINE_ID=... BENCH_OPERATOR=...
   export BENCH_MACHINE_NAME="..." BENCH_RAM_DESC="..."
   ```

## 2. Prepare the environment

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
tests/smoke.sh          # must end with "smoke OK"; do not continue if it fails
```

Then check the platform. Install only what is missing, and only with approval.

| Platform | Check |
|---|---|
| `cuda` | `docker run --rm --gpus all ubuntu nvidia-smi` prints the GPU (needs NVIDIA Container Toolkit) |
| `cpu` | `docker info` works; Linux x86-64 only (use WSL2 on Windows) |
| `rocm` | `docker run --rm --device /dev/kfd --device /dev/dri rocm/rocm-terminal rocm-smi` |
| `openvino`, `metal` | Server is launched by hand, see METHODOLOGY §4.3. On macOS ask the user to run `sudo -v` first so telemetry can read `powermetrics` |

Other checks:

- **Docker images.** Pull the pinned image for the platform from
  `configs/platforms.yaml`.
- **Disk space.** Check `df -h ~/.cache/huggingface` before downloading models.

## 3. Choose models and download them

- **Pick the models.** Use the entries in `configs/models.yaml` whose `tiers`
  include this machine's tier. For a first session, use the smallest
  tier-appropriate model plus one larger one. Confirm the list with the user.
- **Download** each model online once, with `hf download <hf id>`. Gated Llama
  models need `hf auth login` and an accepted licence; ask the user to do that
  step.
- **Runs are offline** (`HF_HUB_OFFLINE=1`). If a run fails with a Hugging Face
  offline error, the download was incomplete. Download again; do not pass
  `--online`.

## 4. Run the suite

Run these steps in order, one model at a time. Each command creates its own
directory under `results/`, and each step is one command.

```bash
M=<model key>
P=<cuda|rocm|cpu>

# 1. Quiet baseline on the primary backend
python harness/run_suite.py --model $M --platform $P

# 2. Under host load. Ask the user first: synthetic top-up adds CPU/RAM load.
python harness/run_suite.py --model $M --platform $P --conditions office,heavy
#    ...or, if they decline synthetic load, measure the load they already have:
python harness/run_suite.py --model $M --platform $P --conditions office --no-topup

# 3. Without GPU (GPU machines only; skip if the model does not fit in RAM)
python harness/run_suite.py --model $M --platform cpu

# 4. Engine profiles, only if the user asks for Phase 4 work
python harness/run_suite.py --model $M --platform $P --profile prefix-cache
```

Runs take a long time: on tier 1, a single suite can take hours. Run the
command in the background, check its log periodically, and tell the user what
is running and roughly where it is. Do not start a second suite while one is
running; they would measure each other.

### Handling problems

| Symptom | What to do |
|---|---|
| `machine is not quiet` | Show the user `top_procs` from `results/<run>/hostload_quiet.json` and ask them to close those apps. Retry. Use `--force` only if they approve, and say so in the report. |
| Server exits during start-up | Read `results/<run>/server.log`. **Out of memory:** record "does not fit" for this model × platform and move on. **Unsupported quantisation or backend:** record it and file a git-bug issue (`area:configs`). |
| Points marked `failed`, or "saturated at c=N" | Expected at high concurrency. It is data, so no action is needed. |
| `harness_dirty: true` in the manifest | Commit or stash changes, then re-run. |
| On battery warning | Ask the user to plug in, then re-run. |
| Harness bug (traceback, wrong columns) | Stop. File a git-bug issue (`bug area:harness`) with the command and traceback. Do not patch the harness mid-measurement. |

## 5. Aggregate

```bash
python harness/summarize.py results/ -o results/summary.csv
python harness/report.py results/summary.csv -o results/headline.csv --md > results/report_tables.md
```

Before writing the report, check `headline.csv` for:

- `flag_unstable`: re-run that model × platform once. If it is still unstable,
  report it as unstable.
- `condition_match_rate` below 1.0: some points ran under a different host
  condition than targeted. Report them under the measured condition.
- `flag_throttled` and `flag_swapped`: report them; they explain slow results.
- Any `harness_dirty` runs: exclude them, or re-run.

## 6. Write the report

- **Where:** write the report to `results/REPORT_<machine_id>_<YYYYmmdd>.md`.
  `results/` is git-ignored; never commit reports.
- **Delivery:** give the user the file path and the key findings in chat.
- **Structure:** use this skeleton. Every number comes from `headline.csv`,
  `machines.csv` or the run manifests; copy them, don't retype them from memory.

```markdown
# Local LLM benchmark: <machine_name>

**Operator:** <BENCH_OPERATOR>   **Date:** <date>   **Harness commit:** <sha>   **vLLM:** 0.30.0

## Machine
<the Machines table from report_tables.md>
Tier: <tier>. Power: <AC/battery, profile>. Room temperature: <if known>.

## What was run
| Model | Quant | Platform | Profile | Conditions | Repeats | Run directory |
|---|---|---|---|---|---|---|

Deviations from the standard procedure: <none | --force, --no-topup, extra serve args, skipped steps, and why>.

## Results
<the Results table from report_tables.md>

### Single user (concurrency 1, quiet)
For each model: TTFT p50, TPOT p50, output tokens/s, for chat / rag / generate.

### Capacity (max users within SLO)
For each model × platform × condition: capacity_users per scenario.

### GPU vs CPU
Same model, same scenario: GPU vs CPU single-user tokens/s and capacity; ratio.

### Quiet vs under load
Same model: quiet vs office/heavy (measured class), change in TTFT p90 / TPOT p50 / capacity.

### Energy
Wh per 1k output tokens at capacity, peak power, and power_source (wall, or GPU+CPU lower bound).

## Failures and flags
Models that did not fit; failed or saturated points; unstable, throttled or swapped results; condition mismatches.

## Raw data
Paths of every run directory used.
```

Follow these rules when writing it:

- **Keep facts and interpretation apart.** A sentence such as "decode is
  memory-bandwidth-bound here" is fine when it is labelled as interpretation.
- **Always put the machine specification next to the numbers.**
- **Say plainly what was not run** and why.
- **Compare tokens/s across models with care.** Different models use different
  tokenizers, so compare them only in task terms (see METHODOLOGY §7).

## 7. Agentic runs (Garden)

Run these only if the user asks. Follow
[docs/GARDEN_AGENTIC.md](docs/GARDEN_AGENTIC.md):

1. Serve the model with the `agentic` profile and `--serve-only`.
2. The user sets up Garden and the benchmark agent; you need their credentials
   as environment variables, never in files.
3. Run `harness/garden_agentic.py` at `--parallel 1`, then 2.

Add a "Agentic tasks" section to the report with, per task: success rate,
median wall time, model vs tool time, steps and tokens. Also include the
Garden commit.

## 8. After the session

- Tell the user which run directories to upload to the team's shared results
  store.
- File git-bug issues for anything that blocked the procedure (missing platform
  support, harness bugs, wrong docs), following CLAUDE.md. Do not push them
  (`git-bug push`) unless asked.
