# Local LLM Benchmarking Methodology

How the team measures local LLM serving performance with vLLM on constrained
machines, so that any team member can reproduce a run on their own machine and
the results are comparable across machines.

It covers Phases 1–5 of the research roadmap: the baseline on minimal hardware,
tokenization, UMA vs discrete GPU, engine optimisation, and stress testing.
Agentic workloads run through Garden and are covered in
[GARDEN_AGENTIC.md](GARDEN_AGENTIC.md).

---

## 1. Principles

1. **Change one variable at a time.** Every comparison holds all but one
   variable fixed (§2). A comparison that changes two variables at once is not
   reported.
2. **Pin everything.**
   - vLLM image tag, model checkpoint (and revision), engine flags and the
     harness commit are all recorded in each run's `manifest.json` and
     `fingerprint.json`.
   - A run from a dirty harness checkout is flagged (`harness_dirty`).
3. **Run offline.**
   - Models are downloaded once, then every run uses `HF_HUB_OFFLINE=1`, as a
     deployment in the field would.
   - No network traffic happens during a measurement.
4. **Measure the host condition; don't assume it.**
   - Personal machines are rarely idle.
   - Background load is measured before and during every run, and each result
     is labelled with the condition it actually ran under (§3.2).
5. **Repeat and report the spread.**
   - Every point runs at least 3 times.
   - We report the median across repeats and flag unstable results (§6.4).
6. **Keep raw data.**
   - Per-request results, 1 Hz telemetry and server logs are kept alongside the
     summaries, so any number can be recomputed.

---

## 2. Variables

### 2.1 Independent variables (what a comparison changes)

| Variable | Values | Where it is set |
|---|---|---|
| Hardware tier | `t1-minimal`, `t2-entry`, `t3-pro`, `t4-workstation`, `t5-multi-gpu` | `configs/tiers.yaml`, `--tier` (§2.5) |
| Machine class | `office-laptop`, `office-desktop`, `gaming-laptop`, `gaming-desktop`, `pro-laptop`, `pro-workstation`, `uma-workstation`, `apple`, `server` | detected (§2.5); `BENCH_MACHINE_CLASS` overrides |
| Constraint | power source (AC / battery), GPU power cap, weights offloaded to system RAM | `--gpu-power-limit`, `--cpu-offload-gb`, unplugging (§2.6) |
| Compute mode | `gpu` (CUDA / ROCm / Metal), `cpu`, `igpu` (OpenVINO on Iris Xe) | `--platform` (`configs/platforms.yaml`) |
| Model + quantisation | e.g. `qwen3-8b-awq`, `llama3.2-3b` | `--model` (`configs/models.yaml`) |
| Engine profile | `baseline`, `prefix-cache`, `fp8-kv`, `interactive`, `spec-ngram`, `spec-eagle3`, `agentic` | `--profile` (`configs/engine_profiles.yaml`) |
| Workload scenario | `chat`, `rag`, `generate`, `agent-prefix`, `longctx` | `--scenarios` (`configs/scenarios.yaml`) |
| Request load | concurrent users: 1, 2, 4 … (sweep set by the tier) | `--concurrency` or the tier default |
| Host condition | `quiet`, `office`, `heavy` | `--conditions` (`configs/conditions.yaml`) |

### 2.2 Controlled variables (held fixed)

- **vLLM version.** `0.30.0` everywhere, using the pinned image tag for each
  backend. The OpenVINO and Metal plugins track their own vLLM version, so their
  plugin commit is recorded in `--notes`.
- **Token counts.**
  - Synthetic prompts have fixed input and output lengths (`--random-range-ratio 0`,
    `--ignore-eos`).
  - Sampling uses `--temperature 0`.
  - The seed for each repeat is `1000 + repeat`.
- **Prefix caching.** Off in `baseline`, so repeated prompts cannot reuse KV cache
  from earlier runs.
- **`--max-model-len`.** Fixed per tier (8k / 16k / 32k) so KV-cache sizing is
  comparable within a tier.
- **Power.**
  - The machine is on AC power.
  - The OS power profile is set to *performance* (Linux `powerprofilesctl set performance`,
    Windows "Best performance", macOS "High Power" where available).
  - The harness records power state in the fingerprint and warns when on battery.
- **Physical setup.** The laptop lid is open and the machine sits on a hard
  surface. Note the room temperature in `--notes`.

### 2.3 Two kinds of "load"

The word "load" covers two separate axes. They are varied independently.

| Axis | Meaning | How it's varied |
|---|---|---|
| **Request load** | How many users hit the model at once | Concurrency sweep (§5.3) |
| **Host condition** | What else the machine is doing (browser, spreadsheet, sync client, other services) | `quiet` / `office` / `heavy` (§3.2) |

"Quiet" means the host is idle apart from the benchmark. "Under load" means the
host is carrying the background load of normal use while it serves the model.

### 2.4 With and without GPU

- **GPU machines** run each model twice, keeping the model, scenarios and
  concurrency identical:
  - the GPU arm (`--platform cuda|rocm|xpu|metal`);
  - the CPU arm (`--platform cpu`), which runs the vLLM CPU image with
    `CUDA_VISIBLE_DEVICES=` so the GPU is invisible.
- **Checkpoint choice.** Pick one that runs on both backends: bf16, AWQ, GPTQ or
  INT8 W8A8. FP8 does not run on x86 CPUs.
- **Fit on the CPU arm.** The model also has to fit in system RAM with the tier's
  `cpu_kv_cache_gib` on top.
- **iGPU (Iris Xe).** It is reached only through the OpenVINO plugin, so on tier 1
  the comparison is `openvino` (`VLLM_OPENVINO_DEVICE=GPU`) vs `openvino`
  (`VLLM_OPENVINO_DEVICE=CPU`) vs `cpu` (stock vLLM).

### 2.5 Tiers and machine classes

Two labels describe a machine; both are in every result.

- **Tier** = how much memory the model can live in. It decides which models a
  machine runs and how far the concurrency sweep goes.
- **Machine class** = what kind of machine it is, which decides its other
  constraints: power and cooling on laptops, VRAM vs system RAM on gaming
  desktops, ECC and power budget on workstations, shared bandwidth on
  unified-memory machines.

| Tier | Model memory | Examples |
|---|---|---|
| `t1-minimal` | No discrete GPU; up to 16 GB shared RAM | Iris Xe / Ryzen APU office laptops, 8 GB Apple M-series |
| `t2-entry` | 8–16 GB VRAM, or 16–36 GB Apple unified | RTX 4060–4070 Ti / 5060 Ti (desktop or laptop), Arc B580, Apple M1–M5 / Pro up to 36 GB |
| `t3-pro` | 20–32 GB VRAM, or 48–64 GB Apple unified | RTX 4090 / 5090, RTX 4500 / 5000 Ada, Radeon AI PRO R9700, Arc Pro B60, Apple Pro / Max 48–64 GB |
| `t4-workstation` | One accelerator with 48 GB+ VRAM, or 96 GB+ unified | RTX 6000 Ada, RTX PRO 6000 Blackwell, Radeon PRO W7900, Strix Halo 128 GB, DGX Spark, Mac Studio Max / Ultra |
| `t5-multi-gpu` | Two or more discrete GPUs | 2× RTX 4090 / 5090, 2× RTX 6000 Ada |

`python harness/fingerprint.py` prints `spec.suggested_tier` and
`spec.machine_class`, detected from GPU memory, unified-memory platforms
(Apple Silicon, Ryzen AI Max, GB10), the SMBIOS chassis type, and pro vs
consumer GPU lines, workstation CPUs and ECC memory. The operator confirms
both. A VM is labelled `virtual` and should not be used for published numbers.

Compare within a class first (gaming laptop vs gaming laptop), then across
classes at the same tier: a gaming laptop and a gaming desktop with the same
GPU name differ in power limit, clocks and cooling, which the fingerprint
records (`gpu_power_limit`, `gpu_power_max`, PCIe link).

### 2.6 Constraint tests

Optional runs that each change one constraint on the same machine, compared
with that machine's baseline. Each is part of the result key, so it never
merges with the baseline.

| Test | How | What it answers | Typical class |
|---|---|---|---|
| Model larger than VRAM | `--cpu-offload-gb N` (weights kept in system RAM, streamed over PCIe) | How far a 12–16 GB gaming GPU with 32–64 GB RAM can stretch, and how steep the PCIe cliff is | gaming desktop / laptop |
| GPU power cap | `--gpu-power-limit W` (NVIDIA, needs `sudo -v` and the owner's approval; restored after the run) | Throughput per watt under a UPS or solar budget | gaming desktop, pro workstation |
| On battery | Unplug and run with `--notes battery`; the fingerprint records the power source | How much a laptop slows down off mains, e.g. during load-shedding | any laptop |

---

## 3. Host conditions

### 3.1 Classes

Background load is whole-machine usage minus the vLLM server, the benchmark
client and the harness. It is measured every second by `harness/telemetry.py`
(`bg_cpu_pct`, `bg_mem_used_gb`, `swap_in_mb_s`). Each point is classified by
the median of those samples:

| Class | Background CPU (% of all cores) | Background GPU util | Swap-in |
|---|---|---|---|
| quiet | ≤ 5 | ≤ 5 | ≤ 0.1 MB/s |
| light | ≤ 15 | ≤ 20 | ≤ 1 MB/s |
| office | ≤ 40 | ≤ 40 | ≤ 5 MB/s |
| heavy | > 40 | — | — |

### 3.2 Reaching a target condition on a personal machine

`harness/hostload.py` measures the machine as its owner left it (30 s window),
then decides what to do:

- **`quiet` target.**
  - If the machine is not quiet, or another process is using the GPU, the run
    stops and prints the heaviest processes. Close them and retry.
  - `--force` records a non-quiet run anyway. It is labelled with its measured
    class and does not count as a quiet result.
- **`office` and `heavy` targets:**
  - **Already in band** (organic load meets the floor): the run goes ahead as-is
    and is labelled *organic*.
  - **Below the floor:** only the shortfall is topped up with synthetic load
    (`harness/contention.py`, using `stress-ng` if installed). The run is
    labelled *topped-up*.
    - Floors: `office` is 15% background CPU and 35% of RAM in use; `heavy` is
      40% CPU, 50% of RAM, plus a memory-bandwidth hog.
    - The memory top-up never pushes free memory below max(2 GB, 10% of RAM).
    - The load is re-measured after the top-up has settled, and the verified
      class is stored in the manifest.
  - **Above the band:** the run goes ahead and is labelled with the heavier
    class it actually ran under.
  - **`--no-topup`:** never adds synthetic load; points are labelled with what
    was measured. Use it when a team member does not want stress processes on
    their machine.
- **Load drift.** A user may open a browser halfway through a run. Every point
  therefore carries `measured_condition` and `condition_match` in
  `summary.csv`. Points where these don't match are reported under their
  measured class, not their target.
  - For **quiet** points: re-run.
  - For **loaded** points: report them under the class they were measured in.

---

## 4. Setup

### 4.1 Common

```bash
git clone <this repo> && cd flow_local_ai_workbench
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
sudo apt install stress-ng        # optional; improves top-up precision (brew install stress-ng on macOS)
export BENCH_TIER=t1-minimal      # your tier, see configs/tiers.yaml
export BENCH_MACHINE_ID=ade-thinkpad-x1   # stable, unique id for this machine
export BENCH_MACHINE_NAME="Lenovo ThinkPad X1 Carbon Gen 10"   # make and model, as shown in reports
export BENCH_RAM_DESC="LPDDR5-5200 dual-channel"               # memory type, speed, channels
export BENCH_OPERATOR=ade
```

Every report names the machine and its specification (§6.5). The harness
detects the CPU, core count, RAM size, GPU, VRAM, driver, OS and power state
itself. Two things need help:

- **Make and model** come from DMI on Linux and `system_profiler` on macOS.
  Virtual machines, WSL2 and some desktops report nothing useful, so set
  `BENCH_MACHINE_NAME`.
- **RAM type and speed** need root (`dmidecode`). Run
  `sudo dmidecode -t memory | grep -E "Type:|Speed:"` once and put the answer in
  `BENCH_RAM_DESC`. DDR4 vs DDR5, and single vs dual channel, largely explain
  decode speed on tier 1, so don't skip this.

Check the result with `python harness/fingerprint.py | grep -A16 '"spec"'`.
If the hardware changes (RAM upgrade, GPU swap), give the machine a new
`BENCH_MACHINE_ID`. The report warns when one id has shown two different specs.

### 4.2 Download models once (online), then run offline

Every model in `configs/models.yaml` is pinned to a commit (`revision`). Download
exactly that commit; the harness prints the commands:

```bash
pip install -U "huggingface_hub[cli]"
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --print-download
# hf download Qwen/Qwen3-8B-AWQ --revision 4da05a8e...
# (plus the EAGLE3 draft head when you add --profile spec-eagle3)
```

Models marked `gated: true` (Llama 3.2 3B, InkubaLM) need `hf auth login` and
an accepted licence on the model page first. The benchmark client tokenizes
from the same pinned snapshot, so runs stay offline.

### 4.3 Per platform

| Platform | Setup | Launch |
|---|---|---|
| `cuda` (GeForce, RTX Ada / Blackwell pro cards, DGX Spark) | NVIDIA driver + NVIDIA Container Toolkit; `docker pull vllm/vllm-openai:v0.30.0` (amd64 and arm64) | managed by harness |
| `cpu` (any x86-64, Linux) | Docker; `docker pull vllm/vllm-openai-cpu:v0.30.0-x86_64`. Prefer the image: the native CPU wheel needs glibc ≥ 2.39, `libnuma1`, tcmalloc and a CPU build of torch | managed by harness |
| `rocm` (Radeon RX 7900 / 9070, Radeon PRO W7900, AI PRO R9700, Strix Halo) | ROCm ≥ 7.0.2 on the host; `docker pull vllm/vllm-openai-rocm:v0.30.0` | managed by harness |
| `xpu` (Intel Arc B-series, Arc Pro B60) | Intel GPU driver; `docker pull vllm/vllm-openai-xpu:v0.30.0` | managed by harness |
| `openvino` (Iris Xe / Intel CPU) | Build [vllm-openvino](https://github.com/vllm-project/vllm-openvino) in a venv (`VLLM_TARGET_DEVICE=empty pip install .`) | external: `VLLM_OPENVINO_DEVICE=GPU vllm serve <model> --port 8000 …` |
| `metal` (Apple Silicon, macOS 15+) | Install [vllm-metal](https://github.com/vllm-project/vllm-metal) v0.30.0 (Homebrew tap). Serves each model's `mlx` 4-bit checkpoint from `models.yaml` | external: `vllm serve $(python harness/run_suite.py --model M --platform metal --print-serve-args)` |

- **Windows machines:** run the harness inside WSL2 (the NVIDIA Container
  Toolkit and `nvidia-smi` work there; the CPU arm is Linux-only, so it also
  runs in WSL2). Inside WSL2, `psutil` sees only the Linux VM, so the harness
  reads whole-Windows CPU and memory through `typeperf.exe` and measures host
  conditions against the Windows host (`host_scope: windows-host` in
  telemetry). Windows has no RAPL, so energy needs a wall meter.
- **Apple Silicon:** telemetry reads CPU, GPU, ANE and package power plus GPU
  residency from `powermetrics`, which needs root. Run `sudo -v` in the same
  terminal before starting a suite so the harness can use `sudo -n`.
- **External launches:** start the server with the same flags the harness
  would use. Print them with

  ```bash
  python harness/run_suite.py --model phi4-mini-ov --platform openvino --print-serve-args
  ```

  then prefix the output with `vllm serve`. Record the exact command, and the
  plugin commit, in `--notes`.

### 4.4 Validate the harness

```bash
tests/smoke.sh     # stubbed vLLM; no GPU, model or network needed; ~4 minutes
```

---

## 5. Procedure

### 5.1 Pre-run checklist

- [ ] On AC power, OS power profile set to performance, lid open.
- [ ] Other GPU users closed (games, video calls, other model servers).
- [ ] For `quiet`: close browsers, IDEs, sync clients and chat apps.
- [ ] Models downloaded; no pending OS updates.
- [ ] `git status` clean on the harness checkout.
- [ ] Room temperature noted.

### 5.2 Standard sequence per machine

Run these in order. A step's output feeds the next one.

```bash
M=qwen3-8b-awq   # pick from configs/models.yaml for your tier

# 1. Quiet baseline, GPU (or your primary backend)
python harness/run_suite.py --model $M --platform cuda

# 2. Same model under host load (organic first; top-up only if the owner agrees)
python harness/run_suite.py --model $M --platform cuda --conditions office,heavy

# 3. Without GPU: same model on the CPU arm, quiet
python harness/run_suite.py --model $M --platform cpu

# 4. Engine profiles (Phase 4), quiet, on the machine chosen in Phase 3
for P in prefix-cache fp8-kv interactive spec-ngram spec-eagle3; do
  python harness/run_suite.py --model $M --platform cuda --profile $P
done

# 5. Aggregate
python harness/summarize.py results/ -o results/summary.csv
python harness/report.py results/summary.csv -o results/headline.csv --md
```

A step can be interrupted with Ctrl-C. Points completed so far are saved, the
server is stopped and the top-up is removed.

### 5.3 What a suite run does

1. Writes `fingerprint.json`: CPU/ISA, RAM and DIMMs, GPUs and drivers, power
   state, OS, image digest, harness commit.
2. Runs the quiet gate (`hostload_quiet.json`) **before** the server starts, so
   loading the model is not mistaken for background noise.
3. Starts vLLM (docker) and records the start-up time from launch to a healthy
   `/health`, plus a `/metrics` snapshot after start-up (`metrics_ready.txt`,
   which includes KV-cache block counts).
4. Starts 1 Hz telemetry: host CPU and memory split into harness vs background;
   GPU utilisation, memory, power, temperature and throttle reasons; CPU package
   power (RAPL); vLLM queue depth, KV-cache usage, preemptions and prefix-cache
   hits; plus an optional wall meter (§6.3).
5. Loops `repeat → condition → scenario → concurrency`, running
   `vllm bench serve` for each point:
   - `num_prompts = max(min_prompts, concurrency × prompts_per_slot)`.
   - Warm-up requests are equal to the concurrency.
   - Repeats form the outermost loop, so slow drift (thermals, background
     changes) shows up as spread rather than as bias against one scenario.
6. Gives each point a status from its result file: `ok`, `partial` (some
   requests failed) or `failed` (bench error, or no request completed;
   `vllm bench serve` itself exits 0 in that case). Only `ok` points feed the
   headline numbers. A suite with any failed point exits with status 2.
7. Skips a scenario whose prompt + output does not fit the server's context
   length (read from `/v1/models` for external servers), and stops sweeping a
   scenario once it saturates: the point is not `ok`, or fewer than 25% of its
   requests meet the SLO. Higher concurrency would only fail slower.
8. Writes `manifest.json`: every point with timestamps, the exact
   server/bench commands, condition plans and their verification.

### 5.4 Tokenization premium (Phase 2)

FLORES+ provides the same sentences in every language. The Phase 2 workload
serves identical passages (8 aligned sentences each) in English, Swahili,
Yoruba, Hausa, Igbo, isiZulu and isiXhosa, so any difference in input tokens,
TTFT, KV-cache use and capacity comes from the tokenizer alone.

```bash
# once: accept the FLORES+ terms on Hugging Face, then
hf download openlanguagedata/flores_plus --repo-type dataset \
  --revision 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 --include "devtest/*_Latn.jsonl"
SRC=~/.cache/huggingface/hub/datasets--openlanguagedata--flores_plus/snapshots/5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06
python harness/flores.py build --src $SRC                  # -> datasets/flores/*.jsonl (git-ignored)
python harness/flores.py fertility --src $SRC \
  --models inkubalm-0.4b,llama3.1-8b-w4a16,qwen3-8b-awq     # -> results/fertility.csv

# serving cost per language, per model (repeat for each model)
python harness/run_suite.py --model inkubalm-0.4b --platform cpu --scenarios flores
```

- `fertility.csv` gives tokens per language relative to English for each
  model's tokenizer (`fertility_vs_eng`), plus characters per token.
- `report.py` writes `tokenization.csv`: for every non-English `flores-*` row,
  input tokens, TTFT, capacity and peak throughput relative to `flores-eng`
  on the same machine, model and condition.
- The FLORES+ terms forbid re-hosting the text where crawlers can reach it.
  `datasets/` is git-ignored; never commit or upload the built prompts.

### 5.5 Running the benchmark client elsewhere

On tier-1 machines the benchmark client competes with the model for CPU. The
client's CPU is counted as harness load (§3.1), not background. For final
tier-1 numbers, run the client from a second machine on the LAN. All host-side
work (fingerprint, host-condition checks and top-up, telemetry) then runs on
the machine under test through `harness/probe.py`:

```bash
# machine under test
export BENCH_PROBE_TOKEN=<shared secret>
python harness/probe.py --port 9109 &
python harness/run_suite.py --model $M --platform $P --serve-only

# client machine (same checkout, same BENCH_* settings)
export BENCH_PROBE_TOKEN=<shared secret>
python harness/run_suite.py --model $M --platform $P \
  --base-url http://<target-ip>:8000 --probe http://<target-ip>:9109
```

- `fingerprint.json` describes the machine under test; the client's own is
  kept as `client_fingerprint.json`.
- The manifest records `client: lan`.
- The probe only accepts requests carrying the token. Keep port 9109 on the
  local network.

---

## 6. Metrics

### 6.1 Per request (from `vllm bench serve`)

| Metric | Meaning | Mostly bound by |
|---|---|---|
| TTFT | Time from submission to first token | Prefill compute (FLOPs), plus queueing |
| TPOT | Mean time per output token after the first, per request | Decode memory bandwidth |
| ITL | Gap between consecutive tokens (every gap) | Decode bandwidth, scheduling stalls |
| E2EL | Total request latency | Both |

Each is reported as mean, median, p90, p95 and p99. The per-request lists are
kept in the result JSON.

### 6.2 Per point (system level)

- `output_throughput`: output tokens per second across all users.
- `total_token_throughput`: input plus output tokens per second.
- `request_goodput`: requests per second that meet the scenario SLO.
  `goodput_ratio = goodput / throughput`.
- Peak memory: system RAM, GPU memory and swap. Any swap use raises
  `flag_swapped`; on a 16 GB machine this is often the real limit.
- Background CPU and memory, `measured_condition`, and `condition_match`.

### 6.3 Energy and power

| Source | Coverage | Label |
|---|---|---|
| External wall meter (smart plug or inline meter writing the latest watts to a file; set `BENCH_EXT_POWER_FILE`) | Whole system | `power_source=wall` |
| `powermetrics` combined power (Apple Silicon) | CPU + GPU + ANE package; excludes display, SSD, PSU losses | `power_source=soc` (a lower bound) |
| `nvidia-smi` board power + RAPL CPU package power | GPU and CPU package only; excludes RAM, fans, PSU losses | `power_source=gpu+cpu_pkg` (a **lower bound**) |

- Reported values are `mean_power_w`, `peak_power_w` (sampled at 1 Hz; true
  transients are shorter, so size inverters from wall-meter peaks with the
  roadmap's 20–25% margin), `energy_wh`, and `wh_per_1k_output_tokens`.
- Phase 5 inverter and battery sizing must use wall-meter data.
- RAPL readings need read access: `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj`
  for the session.

### 6.4 Derived headline numbers (`harness/report.py`)

One row per machine × model × platform × profile × condition × scenario. Each
row carries `machine_name` and `machine_spec`, a one-line summary of make and
model, CPU and cores, RAM size and type, GPU and VRAM, and OS:

- **Single-user experience:** median across repeats of TTFT p50/p90, TPOT p50
  and tokens/s at concurrency 1.
- **Capacity (max users):** the highest concurrency where the median
  `goodput_ratio` across repeats is ≥ 0.9 and no request failed. This is how
  many simultaneous employees the machine serves within the SLO.
- **Peak throughput:** the highest median output tokens/s, and the concurrency
  it occurred at.
- **Energy per 1k tokens at capacity.**
- **Quality flags:**
  - `flag_unstable`: coefficient of variation of concurrency-1 throughput
    across repeats > 10%. Investigate or re-run; don't publish.
  - `flag_throttled`: the GPU reported a power or thermal slowdown.
  - `flag_swapped`: swap was used.
  - `condition_match_rate`.

### 6.5 Machine specification in reports

`report.py` also writes `machines.csv`, one row per machine with the full
specification from its runs' `fingerprint.json`:

- make and model
- CPU model, cores/threads and ISA extensions
- RAM size, type and speed
- GPU(s) with VRAM, driver/CUDA version, power limit and PCIe link
- OS and kernel
- power source and power profile

With `--md` it prints a **Machines** table ahead of the **Results** table.
Share both together: a result without the machine's specification cannot be
interpreted. Garden results (`agentic_<condition>_p<N>.csv`) carry the same `machine_name`
and `machine_spec` columns.

### 6.6 SLOs per scenario

TPOT 150 ms is about 6–7 tokens/s, just above reading speed.

| Scenario | Input / output tokens | TTFT SLO | TPOT SLO | Use case |
|---|---|---|---|---|
| chat | 512 / 256 | 2 s | 150 ms | Assistant turn |
| rag | 3072 / 256 | 5 s | 150 ms | Q&A over retrieved documents |
| generate | 128 / 1024 | 2 s | 150 ms | Drafting reports |
| agent-prefix | 2048 shared + 256 / 128 | 3 s | 150 ms | Agent steps with shared tool schemas |
| longctx | 12288 / 256 | 15 s | 200 ms | Long-document analysis (tiers with ≥ 16k context) |

The SLOs live in `configs/scenarios.yaml`. Changing one changes capacity
numbers, so they are versioned with the harness and changes are discussed in an
issue first.

---

## 7. Results and data management

```
results/<YYYYmmdd-HHMMSS>_<machine>_<model>_<platform>_<profile>/
  fingerprint.json          machine + software snapshot
  hostload_<cond>.json      condition measurement, plan, verification
  manifest.json             run config, commands, every point with timestamps
  telemetry.csv             1 Hz host/GPU/vLLM samples
  <cond>_<scenario>_c<N>_r<R>.json   vllm bench serve results (incl. per-request lists)
  bench.log, server.log     client and server output
  metrics_ready.txt         /metrics after start-up
```

- `results/` is git-ignored. Share a run by compressing its directory into the
  team's shared results store; never edit a run directory by hand.
- `summary.csv` (one row per point, with every `spec_*` column), `headline.csv`
  and `machines.csv` are always regenerated from the raw directories.
- **Tokens are model-specific.** Throughput in tokens/s is comparable across
  machines for the *same model*. Across models with different tokenizers,
  compare task-level numbers (requests/s at SLO, Garden task time) instead. This
  is also the core of the Phase 2 tokenization-premium study.

---

## 8. Validity checklist and known pitfalls

| Pitfall | Mitigation in this methodology |
|---|---|
| Thermal throttling on laptops over a long sweep | Repeats in the outer loop; cooldown between points; temperature, clock and throttle telemetry; `flag_unstable` |
| Prefix-cache reuse across repeats inflating results | Prefix caching off in `baseline`; different seed per repeat |
| Benchmark client stealing CPU on small machines | Client CPU counted as harness load; LAN-client option (§5.5) |
| KV cache silently too small on CPU (`VLLM_CPU_KVCACHE_SPACE`) | Set explicitly per tier; `metrics_ready.txt` records block counts; preemptions tracked |
| Swapping on 16 GB machines | Swap telemetry; `flag_swapped`; memory top-up floor |
| GPU→CPU offload cliff (model > VRAM) | Measured deliberately in Phase 3 with `--extra-serve-args "--cpu-offload-gb N"`; never mixed into baseline |
| Server counters include warm-up requests | Telemetry deltas per point (prefix-cache hits, spec-decode acceptance) include the point's warm-up requests; per-request latency and throughput from the bench result do not |
| Plugin version skew (OpenVINO, Metal) | Compared only within their own platform; plugin commit in `--notes` |
| Background load changing mid-run | Per-point measured condition; mismatches reported under the measured class |
| Driver or power-limit differences between "identical" GPUs | Driver, power limit and PCIe link recorded in fingerprint |

---

## 9. Mapping to the roadmap

| Phase | Runs |
|---|---|
| 1. Infrastructure baseline (tier 1) | `llama3.2-3b`, `phi4-mini-w4a16`, `mistral7b-v0.3-w4a16`, `llama3.1-8b-w8a8`, `phi4-mini-ov` on `cpu` and `openvino` (CPU and GPU), quiet + office; Garden `plan-rollout` task |
| 2. Tokenization | `inkubalm-0.4b` vs `llama3.1-8b-w4a16` / `qwen3-8b-awq` on the FLORES+ scenarios; Garden `swahili-reply`, `yoruba-classify` |
| 3. UMA vs discrete GPU | `gemma3-12b-w4a16`, `qwen2.5-14b-awq`, `qwen3-14b-awq`, the 32B set (`qwen2.5-32b-awq`, `qwen3-32b-awq`, `deepseek-r1-distill-32b-w4a16`) and 70B+ (`llama3.3-70b-w4a16`, `qwen2.5-72b-awq`) on `cuda` / `rocm` / `metal`, all scenarios incl. `longctx`, plus deliberate offload runs |
| 4. Engine optimisation | Every profile in `engine_profiles.yaml` vs `baseline` on the Phase 3 winner; `spec-eagle3` reports acceptance rate and mean accepted length |
| 5. Environmental stress | `heavy` condition, Garden runs at parallel 2–4, wall-meter power |
