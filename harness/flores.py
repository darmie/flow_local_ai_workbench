#!/usr/bin/env python3
"""Phase 2 tokenization-premium workload from FLORES+ parallel text.

FLORES+ gives the same sentences in every language (aligned by `id`), so
prompts built from the same ids carry identical meaning and differ only in
language. Two subcommands:

  build      write one `vllm bench serve --dataset-name custom` JSONL per
             language into datasets/flores/ (git-ignored: the FLORES+ terms
             forbid re-hosting it where crawlers can reach)
  fertility  count tokens per language with each model's tokenizer and write
             results/fertility.csv (tokens relative to English)

The source is a local copy of openlanguagedata/flores_plus (gated: accept the
terms on Hugging Face, then):
  hf download openlanguagedata/flores_plus --repo-type dataset \
      --revision 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 --include "devtest/*_Latn.jsonl"

Usage:
  python harness/flores.py build --src <flores_plus snapshot dir>
  python harness/flores.py fertility --src <snapshot dir> --models llama3.1-8b-w4a16,inkubalm-0.4b
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LANGS = ["eng_Latn", "swh_Latn", "yor_Latn", "hau_Latn", "ibo_Latn", "zul_Latn", "xho_Latn"]
OUT_DIR = os.path.join(ROOT, "datasets", "flores")


def load_split(src, split, langs):
    """{lang: {id: text}} for the languages present under <src>/<split>/."""
    data = {}
    for lang in langs:
        path = os.path.join(src, split, f"{lang}.jsonl")
        if not os.path.exists(path):
            sys.exit(f"missing {path}; download it (see --help)")
        data[lang] = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                data[lang][rec["id"]] = rec["text"]
    common = set.intersection(*(set(d) for d in data.values()))
    return {lang: {i: d[i] for i in common} for lang, d in data.items()}, sorted(common)


def build(args):
    data, ids = load_split(args.src, args.split, args.langs)
    os.makedirs(args.out, exist_ok=True)
    groups = [ids[i:i + args.sentences] for i in range(0, len(ids) - args.sentences + 1, args.sentences)]
    for lang in args.langs:
        path = os.path.join(args.out, f"{lang}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for g in groups:
                # Passage only, no instruction: the prompt is 100% target-language text.
                f.write(json.dumps({"prompt": " ".join(data[lang][i] for i in g), "ids": g},
                                   ensure_ascii=False) + "\n")
        print(f"[flores] {lang}: {len(groups)} prompts x {args.sentences} sentences -> {path}")


def load_tokenizer(model, hf_cache):
    """Tokenizer from the model's pinned local snapshot (offline)."""
    from tokenizers import Tokenizer
    if model.get("tokenizer") and os.path.isdir(model["tokenizer"]):
        snap = model["tokenizer"]
    else:
        snap = os.path.join(hf_cache, "hub", "models--" + model["hf"].replace("/", "--"),
                            "snapshots", model.get("revision", ""))
    path = os.path.join(snap, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; download the model at its pinned revision first")
    return Tokenizer.from_file(path)


def fertility(args):
    data, ids = load_split(args.src, args.split, args.langs)
    models = yaml.safe_load(open(os.path.join(ROOT, "configs", "models.yaml")))["models"]
    rows = []
    for key in args.models.split(","):
        try:
            tok = load_tokenizer(models[key], args.hf_cache)
        except (FileNotFoundError, KeyError) as e:
            print(f"[flores] skip {key}: {e}", file=sys.stderr)
            continue
        counts = {lang: [len(tok.encode(data[lang][i], add_special_tokens=False).ids) for i in ids]
                  for lang in args.langs}
        eng = sum(counts["eng_Latn"])
        for lang in args.langs:
            total = sum(counts[lang])
            chars = sum(len(data[lang][i]) for i in ids)
            rows.append({
                "model": key, "tokenizer_of": models[key]["hf"], "lang": lang, "sentences": len(ids),
                "tokens": total, "tokens_per_sentence": round(total / len(ids), 2),
                "chars_per_token": round(chars / total, 3),
                "fertility_vs_eng": round(total / eng, 3),
                "median_sentence_ratio_vs_eng": round(statistics.median(
                    c / e for c, e in zip(counts[lang], counts["eng_Latn"]) if e), 3),
            })
        print(f"[flores] {key}: " + ", ".join(
            f"{r['lang'][:3]} x{r['fertility_vs_eng']}" for r in rows if r["model"] == key))
    if not rows:
        sys.exit("no tokenizers available")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[flores] wrote {len(rows)} rows -> {args.csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "fertility"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True, help="local flores_plus snapshot (contains devtest/)")
        p.add_argument("--split", default="devtest")
        p.add_argument("--langs", default=",".join(LANGS), type=lambda s: s.split(","))
    b = sub.choices["build"]
    b.add_argument("--sentences", type=int, default=8, help="sentences per prompt")
    b.add_argument("--out", default=OUT_DIR)
    f = sub.choices["fertility"]
    f.add_argument("--models", required=True, help="comma-separated keys from configs/models.yaml")
    f.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    f.add_argument("--csv", default=os.path.join(ROOT, "results", "fertility.csv"))
    args = ap.parse_args()
    if "eng_Latn" not in args.langs:
        sys.exit("eng_Latn is the reference language and must be included")
    build(args) if args.cmd == "build" else fertility(args)


if __name__ == "__main__":
    main()
