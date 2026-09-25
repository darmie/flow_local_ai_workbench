#!/usr/bin/env python3
"""Run the serving benchmark suite for one (machine, model, platform, profile).

Steps: fingerprint -> quiet check -> start vLLM -> telemetry on -> for each
repeat x host condition x scenario x concurrency: `vllm bench serve` -> stop
everything -> manifest.json. Aggregate afterwards with summarize.py.

Examples:
  # GPU arm, then the CPU arm of the same model on the same machine
  python harness/run_suite.py --model qwen3-8b-awq --platform cuda
  python harness/run_suite.py --model qwen3-8b-awq --platform cpu
  # Loaded conditions, organic load only (no synthetic top-up)
  python harness/run_suite.py --model llama3.2-3b --platform cpu --conditions quiet,office --no-topup
  # Server started by hand (OpenVINO, vllm-metal)
  python harness/run_suite.py --model phi4-mini-ov --platform openvino --base-url http://localhost:8000
"""
import argparse
import datetime as dt
import json
import os
import platform as pyplatform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 8000


def cfg(name):
    return yaml.safe_load(open(os.path.join(ROOT, "configs", f"{name}.yaml")))


def log(msg):
    print(f"[suite {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def http_ok(url, timeout=3):
    try:
        return urllib.request.urlopen(url, timeout=timeout).status == 200
    except OSError:
        return False


def py(script, *args, **kw):
    return subprocess.run([sys.executable, os.path.join(HERE, script), *args], **kw)


class Server:
    """vLLM server lifecycle: docker-managed or external."""

    def __init__(self, args, plat, model, profile, tier, run_dir):
        self.args, self.plat, self.model = args, plat, model
        self.profile, self.tier, self.run_dir = profile, tier, run_dir
        self.name = f"bench-vllm-{os.getpid()}"
        self.base_url = args.base_url or f"http://127.0.0.1:{PORT}"
        self.managed = not args.base_url and plat.get("launch") == "docker"
        self.cmd = None

    def serve_args(self):
        m, p = self.model, self.profile
        a = [m["hf"], "--served-model-name", self.args.model, "--port", str(PORT),
             "--max-model-len", str(self.args.max_model_len)]
        a += self.plat.get("serve_args", []) + p.get("serve_args", [])
        if p.get("needs_tool_parser") and m.get("tool_call_parser"):
            a += ["--tool-call-parser", m["tool_call_parser"]]
        if p.get("needs_tool_parser") and m.get("reasoning_parser"):
            a += ["--reasoning-parser", m["reasoning_parser"]]
        if m.get("revision"):
            a += ["--revision", m["revision"]]
        if self.args.tp > 1:
            a += ["--tensor-parallel-size", str(self.args.tp)]
        if self.args.pp > 1:
            a += ["--pipeline-parallel-size", str(self.args.pp)]
        return a + shlex.split(self.args.extra_serve_args)

    def start(self):
        if not self.managed:
            if not self.args.base_url:
                sys.exit(f"platform '{self.args.platform}' is launch: external; start vLLM yourself "
                         f"and pass --base-url")
            log(f"using external server at {self.base_url}")
            if not http_ok(self.base_url + "/health"):
                sys.exit(f"{self.base_url}/health is not responding")
            return 0.0
        env = []
        for k, v in self.plat.get("env", {}).items():
            env += ["-e", f"{k}={str(v).format(cpu_kv_cache_gib=self.tier['cpu_kv_cache_gib'])}"]
        for k in ("HF_TOKEN",):
            if os.environ.get(k):
                env += ["-e", k]
        if not self.args.online:
            env += ["-e", "HF_HUB_OFFLINE=1"]
        self.cmd = (["docker", "run", "-d", "--name", self.name, "--network", "host",
                     "-v", f"{self.args.hf_cache}:/root/.cache/huggingface"]
                    + env + self.plat.get("docker_args", []) + [self.plat["image"]] + self.serve_args())
        log("starting server: " + shlex.join(self.cmd))
        t0 = time.time()
        subprocess.run(self.cmd, check=True, stdout=subprocess.DEVNULL)
        while time.time() - t0 < self.args.startup_timeout:
            if http_ok(self.base_url + "/health"):
                startup = time.time() - t0
                log(f"server ready in {startup:.0f}s")
                return startup
            if not self.alive():
                self.dump_logs()
                sys.exit(f"server exited during startup; see {self.run_dir}/server.log")
            time.sleep(3)
        self.dump_logs()
        self.stop()
        sys.exit("server did not become ready before --startup-timeout")

    def alive(self):
        if not self.managed:
            return http_ok(self.base_url + "/health")
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name],
                             capture_output=True, text=True).stdout.strip()
        return out == "true"

    def dump_logs(self):
        if self.managed:
            with open(os.path.join(self.run_dir, "server.log"), "w") as f:
                subprocess.run(["docker", "logs", self.name], stdout=f, stderr=subprocess.STDOUT)

    def stop(self):
        if self.managed:
            self.dump_logs()
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def bench_cmd(args, server, model, scen, conc, n_prompts, seed, out_dir, fname, meta):
    a = ["bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
         "--base-url", server.base_url, "--model", model["hf"], "--served-model-name", args.model,
         "--num-prompts", str(n_prompts), "--max-concurrency", str(conc), "--request-rate", "inf",
         "--num-warmups", str(conc), "--seed", str(seed), "--ignore-eos", "--temperature", "0",
         "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99",
         "--goodput"] + [f"{k}:{v}" for k, v in scen["slo"].items()] + [
         "--save-result", "--result-dir", out_dir, "--result-filename", fname,
         "--disable-tqdm", "--metadata"] + [f"{k}={v}" for k, v in meta.items()]
    if scen["dataset"] == "random":
        a += ["--dataset-name", "random", "--random-input-len", str(scen["input_len"]),
              "--random-output-len", str(scen["output_len"]), "--random-range-ratio", "0.0"]
    elif scen["dataset"] == "prefix_repetition":
        a += ["--dataset-name", "prefix_repetition",
              "--prefix-repetition-prefix-len", str(scen["prefix_len"]),
              "--prefix-repetition-suffix-len", str(scen["suffix_len"]),
              "--prefix-repetition-num-prefixes", str(scen["num_prefixes"]),
              "--prefix-repetition-output-len", str(scen["output_len"])]
    elif scen["dataset"] == "custom":
        a += ["--dataset-name", "custom", "--dataset-path", scen["dataset_path"],
              "--custom-output-len", str(scen["output_len"])]
    if args.client == "local":
        return ["vllm"] + a
    # Containerised client: same image as the server, results written via a bind mount.
    a[a.index("--result-dir") + 1] = "/out"
    return (["docker", "run", "--rm", "--network", "host", "--entrypoint", "vllm",
             "-v", f"{args.hf_cache}:/root/.cache/huggingface", "-v", f"{out_dir}:/out",
             "-e", "HF_HUB_OFFLINE=1" if not args.online else "HF_HUB_OFFLINE=0",
             "-e", "CUDA_VISIBLE_DEVICES=", args.client_image] + a)


class Condition:
    """Brings the host to a target condition; tops up only the shortfall."""

    def __init__(self, args, run_dir):
        self.args, self.run_dir, self.proc = args, run_dir, None

    def enter(self, target):
        plan_path = os.path.join(self.run_dir, f"hostload_{target}.json")
        flags = ["--target", target, "--out", plan_path]
        flags += ["--no-topup"] if self.args.no_topup else []
        flags += ["--force"] if self.args.force else []
        if py("hostload.py", *flags).returncode != 0:
            return None
        plan = json.load(open(plan_path))
        if plan["action"] == "topup":
            ready = os.path.join(self.run_dir, f".contention_{target}_ready")
            if os.path.exists(ready):
                os.remove(ready)
            self.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "contention.py"),
                                          "--plan", plan_path, "--ready-file", ready])
            t0 = time.time()
            while not os.path.exists(ready) and time.time() - t0 < 300:
                time.sleep(1)
            verify = os.path.join(self.run_dir, f"hostload_{target}_verify.json")
            py("hostload.py", "--target", target, "--out", verify, "--verify", "--window", "15")
            plan["verified"] = json.load(open(verify))["measured_class"]
        return plan

    def exit(self):
        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=30)
            self.proc = None


def goodput_ratio(path):
    try:
        r = json.load(open(path))
        return (r.get("request_goodput") or 0) / r["request_throughput"] if r.get("request_throughput") else 0
    except (OSError, ValueError, KeyError):
        return 0


def main():
    models, plats, tiers = cfg("models")["models"], cfg("platforms"), cfg("tiers")["tiers"]
    scenarios_cfg, profiles = cfg("scenarios"), cfg("engine_profiles")["profiles"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=models)
    ap.add_argument("--platform", required=True, choices=plats["platforms"])
    ap.add_argument("--tier", default=os.environ.get("BENCH_TIER"), choices=tiers)
    ap.add_argument("--profile", default="baseline", choices=profiles)
    ap.add_argument("--scenarios", default=",".join(scenarios_cfg["default_scenarios"]))
    ap.add_argument("--conditions", default="quiet", help="comma list of quiet,office,heavy")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", default="", help="override the tier's sweep, e.g. 1,4,8")
    ap.add_argument("--max-model-len", type=int)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--extra-serve-args", default="")
    ap.add_argument("--base-url", default="", help="benchmark an already-running server")
    ap.add_argument("--client", choices=["local", "docker"], default="local" if shutil.which("vllm") else "docker")
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    ap.add_argument("--online", action="store_true", help="allow HF downloads during the run (default: offline)")
    ap.add_argument("--no-topup", action="store_true", help="never add synthetic load; label runs by measured load")
    ap.add_argument("--force", action="store_true", help="proceed even if the quiet check fails")
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--print-serve-args", action="store_true",
                    help="print the vLLM serve arguments for this config and exit (for external launches)")
    ap.add_argument("--serve-only", action="store_true",
                    help="start the server with this config and keep it up until Ctrl-C (for Garden runs)")
    args = ap.parse_args()
    if not args.tier:
        sys.exit("set --tier or BENCH_TIER (see configs/tiers.yaml)")

    tier, plat, model = tiers[args.tier], plats["platforms"][args.platform], models[args.model]
    profile = profiles[args.profile]
    if model.get("platforms") and args.platform not in model["platforms"]:
        sys.exit(f"{args.model} only runs on {model['platforms']}")
    args.max_model_len = args.max_model_len or tier["max_model_len"]
    args.client_image = plat.get("image", "vllm/vllm-openai-cpu:v0.30.0-x86_64")
    if plat.get("launch") == "external" and args.client == "docker":
        args.client_image = plats["platforms"]["cpu"]["image"]

    if args.print_serve_args:
        print(shlex.join(Server(args, plat, model, profile, tier, None).serve_args()))
        return

    machine = os.environ.get("BENCH_MACHINE_ID", pyplatform.node())
    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}_{machine}_{args.model}_{args.platform}_{args.profile}"
    run_dir = os.path.abspath(os.path.join(args.results, run_id))
    os.makedirs(run_dir)
    log(f"run dir: {run_dir}")

    env = dict(os.environ, BENCH_TIER=args.tier, VLLM_IMAGE=plat.get("image", ""))
    with open(os.path.join(run_dir, "fingerprint.json"), "w") as f:
        py("fingerprint.py", stdout=f, env=env, check=True)
    fp = json.load(open(os.path.join(run_dir, "fingerprint.json")))

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cond = Condition(args, run_dir)
    # The quiet gate runs before the server exists so it cannot count the model load as noise.
    if "quiet" in conditions and cond.enter("quiet") is None:
        sys.exit("machine is not quiet (see hostload_quiet.json); close apps or pass --force")

    manifest = {
        "run_id": run_id, "machine_id": machine, "tier": args.tier, "model": args.model,
        "model_label": model["hf"], "quant": model["quant"], "params_b": model.get("params_b"),
        "platform": args.platform, "mode": plat["mode"], "engine_profile": args.profile,
        "vllm_version": plats["vllm_version"], "vllm_image": plat.get("image", "external"),
        "harness_commit": fp["software"].get("harness_commit"),
        "harness_dirty": fp["software"].get("harness_dirty"),
        "max_model_len": args.max_model_len, "repeats": args.repeats, "client": args.client,
        "stress_ng": bool(shutil.which("stress-ng")), "operator": os.environ.get("BENCH_OPERATOR", ""),
        "notes": args.notes, "conditions": {}, "points": [],
    }
    mpath = os.path.join(run_dir, "manifest.json")

    def save():
        json.dump(manifest, open(mpath, "w"), indent=2, default=str)

    server = Server(args, plat, model, profile, tier, run_dir)
    manifest["serve_cmd"] = server.cmd
    telemetry = None
    try:
        manifest["startup_s"] = round(server.start(), 1)
        manifest["serve_cmd"] = server.cmd
        if args.serve_only:
            save()
            log(f"serving {args.model} at {server.base_url}/v1 (Ctrl-C to stop)")
            while server.alive():
                time.sleep(5)
            raise RuntimeError("server exited")
        try:
            open(os.path.join(run_dir, "metrics_ready.txt"), "w").write(
                urllib.request.urlopen(server.base_url + "/metrics", timeout=5).read().decode())
        except OSError:
            pass
        telemetry = subprocess.Popen([sys.executable, os.path.join(HERE, "telemetry.py"),
                                      "--out", os.path.join(run_dir, "telemetry.csv"),
                                      "--metrics-url", server.base_url + "/metrics"])
        save()

        scen_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        saturated = set()  # (condition, scenario, concurrency) whose goodput collapsed
        for rep in range(args.repeats):
            for target in conditions:
                if target != "quiet" or rep > 0:
                    plan = cond.enter(target)
                    if plan is None:
                        log(f"skipping condition {target}: target not reachable")
                        continue
                    manifest["conditions"].setdefault(target, []).append(
                        {k: plan.get(k) for k in ("measured_class", "action", "topup", "verified", "note")})
                sweep = [int(c) for c in args.concurrency.split(",")] if args.concurrency else (
                    tier["concurrency"] if target == "quiet" else tier["loaded_concurrency"])
                try:
                    for sname in scen_names:
                        scen = scenarios_cfg["scenarios"][sname]
                        if scen.get("min_model_len", 0) > args.max_model_len:
                            continue
                        for conc in sweep:
                            if any(k[:2] == (target, sname) and k[2] <= conc for k in saturated):
                                continue
                            if not server.alive():
                                raise RuntimeError("server died")
                            n = max(tier["min_prompts"], conc * tier["prompts_per_slot"])
                            fname = f"{target}_{sname}_c{conc}_r{rep}.json"
                            meta = {"run_id": run_id, "condition": target, "scenario": sname, "repeat": rep}
                            cmd = bench_cmd(args, server, model, scen, conc, n, 1000 + rep, run_dir, fname, meta)
                            log(f"rep {rep} [{target}] {sname} c={conc} n={n}")
                            t0 = time.time()
                            with open(os.path.join(run_dir, "bench.log"), "a") as blog:
                                blog.write(f"\n$ {shlex.join(cmd)}\n")
                                blog.flush()
                                rc = subprocess.run(cmd, stdout=blog, stderr=subprocess.STDOUT).returncode
                            t1 = time.time()
                            ok = rc == 0 and os.path.exists(os.path.join(run_dir, fname))
                            manifest["points"].append({
                                "condition": target, "scenario": sname, "concurrency": conc, "repeat": rep,
                                "num_prompts": n, "t_start": t0, "t_end": t1, "exit_code": rc,
                                "status": "ok" if ok else "failed", "result_file": fname})
                            save()
                            if not ok or goodput_ratio(os.path.join(run_dir, fname)) < 0.25:
                                saturated.add((target, sname, conc))
                                log(f"  {sname} saturated at c={conc}; skipping higher concurrency")
                            time.sleep(args.cooldown)
                finally:
                    cond.exit()
    except KeyboardInterrupt:
        log("interrupted; saving partial results")
    except RuntimeError as e:
        log(f"aborting: {e}")
        manifest["aborted"] = str(e)
    finally:
        if telemetry:
            telemetry.send_signal(signal.SIGINT)
            telemetry.wait(timeout=10)
        cond.exit()
        server.stop()
        save()
    log(f"done: {len(manifest['points'])} points -> {mpath}")
    log(f"aggregate with: python harness/summarize.py {args.results}")


if __name__ == "__main__":
    main()
