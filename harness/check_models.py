#!/usr/bin/env python3
"""Validate configs/models.yaml: required fields, pinned revisions, known tiers,
platforms, quantisation formats and parser names. Exit 1 on any problem.

Usage: python harness/check_models.py [--model KEY]
"""
import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA = re.compile(r"^[0-9a-f]{40}$")
KEY = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")
QUANTS = {"bf16", "fp16", "awq", "gptq-w4", "w4a16", "int8-w8a8", "fp8", "int4-ov"}
# Parser names registered in vLLM 0.30.0 (vllm/tool_parsers, vllm/reasoning).
TOOL_PARSERS = {
    "apertus", "cohere_command3", "cohere_command4", "deepseek_v3", "deepseek_v31", "deepseek_v32",
    "deepseek_v4", "deepseek_v41", "dots", "ernie45", "functiongemma", "gemma4", "gigachat3", "glm45",
    "glm47", "granite", "granite4", "hermes", "hunyuan_a13b", "hy_v3", "hy_v4", "inkling", "internlm",
    "jamba", "k2_horizon", "kimi_k2", "kimi_k3", "lfm2", "ling3", "llama3_json", "llama4_json",
    "llama4_pythonic", "longcat", "mimo", "minicpm5", "minimax_m2", "minimax_m3", "mistral", "muse_glimmer",
    "olmo3", "openai", "phi4_mini_json", "poolside_v1", "pythonic", "qwen3_coder", "qwen3_xml", "seed_oss",
    "step3", "step3p5", "xlam",
}
REASONING_PARSERS = {
    "cohere_command3", "cohere_command4", "deepseek_r1", "deepseek_v3", "deepseek_v4", "deepseek_v41",
    "ernie45", "gemma4", "glm45", "glm47", "granite", "holo2", "hunyuan_a13b", "hy_v3", "hy_v4", "inkling",
    "k2_horizon", "kimi_k2", "kimi_k3", "ling3", "mimo", "minimax_m2", "minimax_m2_append_think",
    "minimax_m3", "mistral", "muse_glimmer", "nemotron_v3", "olmo3", "openai_gptoss", "poolside_v1", "qwen3",
    "seed_oss", "step3", "step3p5",
}
KNOWN = {"hf", "revision", "gated", "quant", "params_b", "tiers", "platforms", "max_model_len", "serve_args",
         "chat_template", "tool_call_parser", "reasoning_parser", "speculator", "mlx", "status", "tokenizer"}


def load(name):
    return yaml.safe_load(open(os.path.join(ROOT, "configs", f"{name}.yaml")))


def check(key, m, tiers, platforms):
    errs = []
    if not KEY.match(key):
        errs.append("key must be lowercase letters, digits, '.' and '-'")
    for field in ("hf", "revision", "quant", "params_b", "tiers"):
        if field not in m:
            errs.append(f"missing `{field}`")
    if "/" not in str(m.get("hf", "")):
        errs.append("`hf` must be an org/name repo id")
    if m.get("revision") and not SHA.match(str(m["revision"])):
        errs.append("`revision` must be the full 40-character commit sha, not a branch or tag")
    if m.get("quant") and m["quant"] not in QUANTS:
        errs.append(f"`quant` {m['quant']!r} not in {sorted(QUANTS)} (add it here if vLLM supports it)")
    if not isinstance(m.get("params_b"), (int, float)):
        errs.append("`params_b` must be a number (billions of parameters)")
    for t in m.get("tiers", []):
        if t not in tiers:
            errs.append(f"unknown tier {t!r}")
    for p in m.get("platforms", []):
        if p not in platforms:
            errs.append(f"unknown platform {p!r}")
    if m.get("tool_call_parser") and m["tool_call_parser"] not in TOOL_PARSERS:
        errs.append(f"unknown tool_call_parser {m['tool_call_parser']!r}")
    if m.get("reasoning_parser") and m["reasoning_parser"] not in REASONING_PARSERS:
        errs.append(f"unknown reasoning_parser {m['reasoning_parser']!r}")
    for sub in ("speculator", "mlx"):
        if sub in m:
            s = m[sub] or {}
            if "/" not in str(s.get("hf", "")) or not SHA.match(str(s.get("revision", ""))):
                errs.append(f"`{sub}` needs `hf` (org/name) and a 40-character `revision`")
    if m.get("quant") == "fp8" and "cpu" in m.get("platforms", []):
        errs.append("fp8 checkpoints do not run on the CPU backend")
    unknown = set(m) - KNOWN
    if unknown:
        errs.append(f"unknown fields {sorted(unknown)}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="check only this key")
    args = ap.parse_args()
    models = load("models")["models"]
    tiers, platforms = load("tiers")["tiers"], load("platforms")["platforms"]
    keys = [args.model] if args.model else list(models)
    bad = 0
    for key in keys:
        if key not in models:
            print(f"{key}: not in configs/models.yaml")
            bad += 1
            continue
        for e in check(key, models[key], tiers, platforms):
            print(f"{key}: {e}")
            bad += 1
    print(f"{len(keys)} model(s) checked, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
