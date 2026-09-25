# Agentic Workload Evaluation with Garden

Synthetic serving benchmarks ([METHODOLOGY.md](METHODOLOGY.md)) show how fast
a machine serves tokens. This guide measures whether a local model on that
machine completes real agent tasks, and how long they take. It drives
[Garden](https://github.com/Flow-Research/garden) issue runs against the local
vLLM server and scores each run with deterministic checks.

The same principles apply: pinned versions, offline operation, measured host
condition, at least 3 repeats, and raw data kept.

---

## 1. What is measured

| Metric | Source | Meaning |
|---|---|---|
| Task success rate | Run status `succeeded` and every `expect` check passes | Can this model on this machine do the job? |
| Wall time per task | Harness clock, from submit to terminal status | What the employee waits |
| Model time vs tool time | `issue_run_event` `tool_finished.duration_ms` (tool); wall time minus tool time (model) | Where the time goes |
| Steps, tool calls, tool errors | `issue_run.usage_json.step_count`, `issue_run_event` | Agent efficiency and robustness |
| Input / output / cached tokens | `issue_run.usage_json` | Token cost of an agentic task (Phase 2 and 6 inputs) |
| Server-side behaviour | `agentic_p<N>_telemetry.csv` (vLLM `/metrics` + host) | KV-cache pressure, prefix-cache hit rate, queueing, power |
| Concurrency effect | `--parallel 1, 2, 4` | How the task experience degrades as more staff use the box |

## 2. Setup (once per machine)

### 2.1 Serve the model with tool calling

Garden's agents need tool calling. Use the `agentic` engine profile, which adds
`--enable-auto-tool-choice` and the model's tool-call and reasoning parsers from
`configs/models.yaml`:

```bash
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --profile agentic --serve-only
# for external platforms: --print-serve-args, then `vllm serve <args>`
```

`--serve-only` runs the quiet gate, starts the server and keeps it up until you
press Ctrl-C.

### 2.2 Run Garden offline, pointed at vLLM

Follow Garden's README for Node ≥ 22.12 and pnpm, then:

```bash
cd garden
cp .env.example .env
# in .env:
#   BETTER_AUTH_SECRET=$(openssl rand -base64 32)
#   EXECUTOR_SECRET_KEY=$(openssl rand -base64 32)
#   GARDEN_MODEL_PROVIDER=openai-compatible
#   GARDEN_MODEL_BASE_URL=http://localhost:8000/v1
#   GARDEN_MODEL_ID=qwen3-8b-awq              # the --served-model-name (the models.yaml key)
#   GARDEN_MODEL_CONTEXT_WINDOW_TOKENS=8192   # = --max-model-len for your tier
pnpm install --frozen-lockfile
pnpm offline:up && pnpm --filter @garden/db db:migrate
pnpm --filter @garden/web skills:seed-builtin
pnpm dev:offline                           # http://localhost:3000
```

Record the Garden commit (`git -C garden rev-parse HEAD`) in the run notes.
Agent behaviour depends on Garden's prompts and tools as much as on the model.

### 2.3 Benchmark workspace and agent

In the Garden UI:

1. Sign up a dedicated benchmark user and create a workspace named `bench`.
2. Create one agent (for example `bench-agent`) with the default skills.
3. Set its tool permissions to **auto**. Tools set to *ask* park the run in
   `waiting_for_approval`, which the harness scores as a failure.
4. Do not connect external connectors. The task set is designed for offline mode.

Then export:

```bash
export GARDEN_URL=http://localhost:3000
export GARDEN_EMAIL=bench@example.local GARDEN_PASSWORD=...
export GARDEN_WORKSPACE_ID=<workspace uuid>    # from the URL or GET /api/workspaces
export GARDEN_AGENT_ID=<agent uuid>            # from GET /api/agents
export GARDEN_DATABASE_URL=postgresql://garden:garden@localhost:55432/garden
```

The harness exports run data straight from Garden's Postgres with `psql`, so
install the PostgreSQL client (`apt install postgresql-client`).

## 3. Procedure

```bash
RUN=results/$(date +%Y%m%d-%H%M%S)_${BENCH_MACHINE_ID}_qwen3-8b-awq_cuda_agentic
python harness/garden_agentic.py --run-dir $RUN --parallel 1 --repeats 3
python harness/garden_agentic.py --run-dir $RUN --parallel 2 --repeats 3
python harness/garden_agentic.py --run-dir $RUN --parallel 4 --repeats 3   # tiers 2+
# with Garden in containers mode (pnpm dev:containers), also:
python harness/garden_agentic.py --run-dir $RUN --with sandbox --only sandbox-sales-xlsx
```

For each task × repeat, the harness:

1. Creates an issue (status `todo`, no auto-start) assigned to the benchmark agent.
2. Starts the run explicitly with `POST /api/issues/:id/runs`, so the start time
   is under the harness's control.
3. Polls until the run reaches a terminal status: `succeeded`, `failed`,
   `cancelled` or `blocked`. Runs that stall (`waiting_for_input`,
   `waiting_for_approval`) or exceed the timeout are cancelled and scored as
   failures.
4. Exports the `issue_run` and `issue_run_event` rows and the issue's work
   products, then scores the run.

Outputs in `$RUN`:

- `garden_runs.jsonl`: one record per run.
- `agentic_p<N>.csv`: the summary.
- `agentic_p<N>_telemetry.csv`: vLLM and host telemetry for the whole batch.
- `agentic_p<N>_hostload.json`: the host condition at the start of the batch.

The host condition rules from METHODOLOGY.md §3 apply unchanged. Run the quiet
batch first. For loaded batches, run `harness/contention.py --plan <hostload json>`
alongside, or rely on organic load, and label the batch with the measured
class.

## 4. Task set

The tasks are in `configs/garden_tasks.yaml`. Each maps to an SMB use case and a
roadmap phase:

| Task | Category | Exercises |
|---|---|---|
| `plan-rollout` | planning | Phase 1 "step-by-step agentic planning" prompt; plan work product |
| `extract-invoice` | extraction | Structured output from a document; exact values checked |
| `summarize-policy` | long-context | Faithful summarisation; every number must survive |
| `decompose-project` | multi-step | Several tool calls (child issues + checklist) |
| `swahili-reply`, `yoruba-classify` | multilingual | Phase 2 language handling on real agent turns |
| `sandbox-sales-xlsx` | tool-use | Sandbox code execution + XLSX document skill (needs containers mode) |

Scoring is deterministic: regexes over the run result and work products,
required tools used, and a step budget. Every task instructs the agent not to
ask questions.

Adding a task means adding an entry with `expect` checks that an unambiguous
correct answer always passes. Run it against the strongest available model
first, to confirm the checks are not the thing failing.

## 5. Reading the results

- **Success rate** is the headline. A machine and model pair that is fast but
  succeeds on fewer than 80% of tasks is not a viable deployment.
- Compare **wall time at parallel 1** with the serving benchmark's `chat` and
  `agent-prefix` TTFT/TPOT. Agent runs send long, repeated prefixes (system
  prompt and tool schemas), so the `prefix-cache` / `agentic` profile usually
  matters more here than in synthetic chat.
- **Model time share** (model_time_s / wall_s) near 1 means the hardware is the
  bottleneck. A large tool-time share means the bottleneck is elsewhere.
- **Cached input tokens** show how much the prefix cache is being used.

## 6. Known limits of the current Garden integration

These are tracked as issues in this repo (`git-bug bug --label area:garden`):

- Token usage is recorded per run, not per LLM call. Per-call latency goes only
  to PostHog, so model time is inferred as wall time minus tool time.
- The API accepts session cookies only, with no API keys, so the harness signs
  in as the benchmark user.
- Sandbox execution needs `pnpm dev:containers`, which is not fully offline.
- Brain (knowledge-graph) embeddings always call Workers AI, so avoid tasks that
  need Brain search in offline mode.
