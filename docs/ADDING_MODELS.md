# Adding a model

A model is one entry in [`configs/models.yaml`](../configs/models.yaml). Every
machine benchmarks exactly the checkpoint that entry pins, so an entry must
name a specific repository **and** commit. `harness/check_models.py` enforces
the rules below; run it before opening a pull request.

## 1. Choose the checkpoint

- **vLLM must support the architecture.** Check the model's `config.json`
  `architectures` against the
  [vLLM supported models list](https://docs.vllm.ai/en/v0.30.0/models/supported_models.html).
  A repo that declares a standard architecture (e.g. `LlamaForCausalLM`) is
  served natively even if it ships custom code.
- **Pick a quantisation the target platforms can load:**

  | Format | `quant` value | NVIDIA | AMD ROCm | Intel XPU | x86 / arm64 CPU | Apple |
  |---|---|---|---|---|---|---|
  | Unquantised bf16 | `bf16` | yes | yes | yes | yes | via `mlx` entry |
  | AWQ 4-bit | `awq` | yes | no | yes | x86 | via `mlx` entry |
  | GPTQ 4-bit | `gptq-w4` | yes | no | yes | x86 | via `mlx` entry |
  | compressed-tensors W4A16 | `w4a16` | yes | UNVERIFIED | UNVERIFIED | UNVERIFIED | via `mlx` entry |
  | INT8 W8A8 | `int8-w8a8` | yes | UNVERIFIED | UNVERIFIED | x86 and arm64 | via `mlx` entry |
  | FP8 | `fp8` | Ada / Hopper | yes | UNVERIFIED | **no** | via `mlx` entry |
  | OpenVINO IR int4 | `int4-ov` | no | no | no | OpenVINO plugin only | no |

  "yes" and "no" follow the vLLM 0.30 quantisation support table; UNVERIFIED
  cells need a short real run (step 4) before the entry is relied on.
  For the with/without-GPU comparison, the checkpoint must load on both the
  GPU and the CPU backend: bf16, AWQ, GPTQ or INT8 W8A8.
- **Prefer first-party or well-maintained quantisations**: the model's own
  organisation (e.g. `Qwen/*-AWQ`), `RedHatAI/*`, then widely used community
  repos. Note gated repos (licence acceptance needed).
- **Apple Silicon** runs a separate MLX conversion (usually
  `mlx-community/<model>-4bit`), added as the entry's `mlx` field.

## 2. Pin the commit

```bash
python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("Qwen/Qwen3-8B-AWQ")
print(info.sha, "gated:", info.gated)
PY
```

Use that full 40-character sha as `revision`. Never a branch or tag: `main`
moves, and results from different weights must not share a model key.

## 3. Write the entry

```yaml
  qwen3-8b-awq:                      # key: <family><size>-<quant>, lowercase
    hf: Qwen/Qwen3-8B-AWQ
    revision: 4da05a8edb55c6046cce958586c33b61da07bb79
    quant: awq
    params_b: 8.2                    # billions of parameters
    tiers: [t1-minimal, t2-integrated, t3-entry]
    tool_call_parser: hermes         # only if the model supports tool calling (Garden)
    reasoning_parser: qwen3          # only for thinking models
    mlx: {hf: mlx-community/Qwen3-8B-4bit, revision: 545dc4251c05440727734bcd94334791f6ab0192}
    speculator: {hf: RedHatAI/Qwen3-8B-speculator.eagle3, revision: 08610ffa01dd9f16731fe8f627b85905b6aa51c4}
```

| Field | Required | Meaning |
|---|---|---|
| `hf`, `revision` | yes | Repository and pinned commit |
| `quant` | yes | Format from the table above |
| `params_b` | yes | Parameter count in billions |
| `tiers` | yes | Tiers it is meant for: its 4-bit weights (about 0.6 GB per billion parameters) plus a working KV cache must fit that tier's model memory |
| `gated` | if gated | `true` when `hf auth login` and a licence are needed |
| `max_model_len` | if smaller than the tiers' | The model's own context limit |
| `chat_template: false` | base models | Model has no chat template |
| `serve_args` | if needed | Extra vLLM flags the model always needs (e.g. disabling image inputs) |
| `tool_call_parser`, `reasoning_parser` | for Garden / thinking models | vLLM parser names (the validator lists the valid ones) |
| `platforms` | if restricted | Only these platforms can run it (e.g. `[openvino]`) |
| `mlx` | for Apple | MLX checkpoint + pinned revision |
| `speculator` | for `spec-eagle3` | EAGLE3 draft head + pinned revision |
| `status: experimental` | optional | Not yet verified on real hardware |

**Changing an existing entry's checkpoint or revision invalidates its past
results.** Add a new key instead (e.g. `qwen3-8b-awq-v2`) so old and new runs
stay separate.

## 4. Validate

```bash
python harness/check_models.py --model qwen3-8b-awq
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --print-serve-args
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --print-download | sh
python harness/run_suite.py --model qwen3-8b-awq --platform cuda --repeats 1 --scenarios chat --concurrency 1,2
```

The last command is a short real run: the server must start (not `did not
start`), and both points must be `ok`. Do it on at least one machine of each
tier listed.

## 5. Submit

Open a pull request with the entry. Per [CLAUDE.md](../CLAUDE.md), file a
git-bug issue with the evidence (the checkpoint choice, the short run's
result) and cite it in the commit trailer.
