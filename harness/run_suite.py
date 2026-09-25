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
import urllib.parse
import urllib.request

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oom  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 8000
# BENCH_DATASETS_ROOT: directory that scenario dataset_path values are relative to (default: repo root).
DATASETS_ROOT = os.environ.get("BENCH_DATASETS_ROOT", ROOT)


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


class SuiteStop(Exception):
    """Ends the suite early with a process exit code."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class StartupFailure(Exception):
    """The server did not become healthy; `info` holds the oom.classify() result."""

    def __init__(self, info):
        super().__init__(f"{info['category']}: {info['evidence']}")
        self.info = info


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
        a += self.plat.get("serve_args", []) + m.get("serve_args", []) + p.get("serve_args", [])
        if p.get("needs_speculator"):
            spec = m["speculator"]
            a += ["--speculative-config", json.dumps({
                "method": "eagle3", "model": spec["hf"], "revision": spec.get("revision"),
                "num_speculative_tokens": p.get("num_speculative_tokens", 3)})]
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
        if self.args.cpu_offload_gb:
            a += ["--cpu-offload-gb", str(self.args.cpu_offload_gb)]
        return a + shlex.split(self.args.extra_serve_args)

    def start(self):
        if not self.managed:
            if not self.args.base_url:
                sys.exit(f"platform '{self.args.platform}' is launch: external; start vLLM yourself "
                         f"and pass --base-url")
            log(f"using external server at {self.base_url}")
            if not http_ok(self.base_url + "/health"):
                raise StartupFailure({"category": "unreachable", "evidence": f"{self.base_url}/health not responding"})
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
        self.t_start = t0 = time.time()
        self._dumped = None
        subprocess.run(self.cmd, check=True, stdout=subprocess.DEVNULL)
        while time.time() - t0 < self.args.startup_timeout:
            if http_ok(self.base_url + "/health"):
                startup = time.time() - t0
                log(f"server ready in {startup:.0f}s")
                return startup
            if not self._running():
                info = self.diagnose()
                self.stop()
                raise StartupFailure(info)
            time.sleep(3)
        info = self.diagnose()
        self.stop()
        raise StartupFailure({"category": "startup_timeout", "evidence": info["evidence"]})

    def _running(self):
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", self.name],
                             capture_output=True, text=True).stdout.strip()
        return out == "true"

    def alive(self):
        """Process up and /health answering. A running container whose engine
        died still fails /health, so both are checked (with retries: /health
        can be slow while the server is saturated)."""
        if self.managed and not self._running():
            return False
        for _ in range(3):
            if http_ok(self.base_url + "/health", timeout=10):
                return True
            time.sleep(5)
        return False

    def identity(self):
        """Server process start time from /metrics; changes when the server restarts."""
        try:
            text = urllib.request.urlopen(self.base_url + "/metrics", timeout=5).read().decode()
        except OSError:
            return None
        m = re.search(r"^process_start_time_seconds\s+([\d.eE+]+)", text, re.M)
        return float(m.group(1)) if m else None

    def remember_identity(self):
        self.ident = self.identity()

    def restarted(self):
        """True if the server was replaced since remember_identity() (a crash that a
        supervisor or container runtime already recovered from)."""
        now = self.identity()
        return getattr(self, "ident", None) is not None and now is not None and now != self.ident

    def diagnose(self):
        """Why the server is down: log patterns, container OOM kill, kernel OOM killer."""
        text, oom_killed, exit_code = "", False, None
        if self.managed:
            text = self.dump_logs()
            state = subprocess.run(["docker", "inspect", "-f", "{{.State.OOMKilled}} {{.State.ExitCode}}", self.name],
                                   capture_output=True, text=True).stdout.split()
            if len(state) == 2:
                oom_killed, exit_code = state[0] == "true", int(state[1])
        elif self.args.server_log and os.path.exists(self.args.server_log):
            text = open(self.args.server_log, errors="replace").read()
        since = time.time() - getattr(self, "t_start", time.time() - 3600) + 60
        return oom.classify(text, oom_killed, exit_code, oom.kernel_oom_lines(since))

    def restart(self):
        """Replace a dead managed server; returns start-up seconds."""
        self.stop()
        return self.start()

    def dump_logs(self):
        """Append this container's log to server.log (once per incarnation); returns the log text."""
        if not self.managed or getattr(self, "_dumped", None) is not None:
            return getattr(self, "_dumped", "") or ""
        text = subprocess.run(["docker", "logs", self.name], capture_output=True, text=True, errors="replace")
        text = text.stdout + text.stderr
        with open(os.path.join(self.run_dir, "server.log"), "a") as f:
            f.write(f"\n===== {self.name} until {dt.datetime.now():%H:%M:%S} =====\n{text}")
        self._dumped = text
        return text

    def stop(self):
        if self.managed and self._exists():
            self.dump_logs()
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)

    def _exists(self):
        return subprocess.run(["docker", "inspect", self.name], capture_output=True).returncode == 0


def snapshot_dir(hf_cache, repo, revision):
    """HF cache path of a pinned snapshot (hub/models--org--name/snapshots/<sha>)."""
    return os.path.join(hf_cache, "hub", "models--" + repo.replace("/", "--"), "snapshots", revision)


def tokenizer_ref(args, model):
    """Pinned local snapshot when downloaded, so the offline client tokenizes with
    exactly the served revision; otherwise the repo id."""
    if model.get("tokenizer"):
        return model["tokenizer"]
    if model.get("revision") and os.path.isdir(snapshot_dir(args.hf_cache, model["hf"], model["revision"])):
        root = "/root/.cache/huggingface" if args.client == "docker" else args.hf_cache
        return snapshot_dir(root, model["hf"], model["revision"])
    return model["hf"]


def bench_cmd(args, server, model, scen, conc, n_prompts, seed, out_dir, fname, meta):
    a = ["bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
         # --model is the served name so /tokenize and requests resolve; tokenizer comes from the pinned snapshot.
         "--base-url", server.base_url, "--model", args.model, "--tokenizer", tokenizer_ref(args, model),
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
        path = os.path.join(DATASETS_ROOT, scen["dataset_path"])
        if args.client == "docker":
            path = "/workbench/" + scen["dataset_path"]
        a += ["--dataset-name", "custom", "--dataset-path", path,
              "--custom-output-len", str(scen["output_len"])]
        if model.get("chat_template") is False:
            a += ["--skip-chat-template"]
    if args.client == "local":
        return ["vllm"] + a
    # Containerised client: same image as the server, results written via a bind mount.
    a[a.index("--result-dir") + 1] = "/out"
    return (["docker", "run", "--rm", "--network", "host", "--entrypoint", "vllm",
             "-v", f"{args.hf_cache}:/root/.cache/huggingface", "-v", f"{out_dir}:/out",
             "-v", f"{os.path.join(DATASETS_ROOT, 'datasets')}:/workbench/datasets:ro",
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


class GpuPowerCap:
    """Caps every NVIDIA GPU at `watts` (needs sudo) and restores the default limit."""

    def __init__(self, watts):
        out = subprocess.run(["nvidia-smi", "--query-gpu=power.default_limit", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        self.defaults = [float(x) for x in out.stdout.split()] if out.returncode == 0 else []
        if not self.defaults:
            sys.exit("--gpu-power-limit needs an NVIDIA GPU and nvidia-smi")
        if subprocess.run(["sudo", "-n", "nvidia-smi", "-pl", str(watts)]).returncode != 0:
            sys.exit("could not set the power limit; run `sudo -v` first (needs the owner's approval)")
        log(f"GPU power limit set to {watts} W")

    def restore(self):
        for i, w in enumerate(self.defaults):
            subprocess.run(["sudo", "-n", "nvidia-smi", "-i", str(i), "-pl", f"{w:g}"])
        log("GPU power limit restored")


class Probe:
    """Client for harness/probe.py running on the machine under test (LAN-client runs)."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.token = os.environ.get("BENCH_PROBE_TOKEN") or sys.exit("set BENCH_PROBE_TOKEN for --probe")

    def call(self, method, path, body=None, raw=False, timeout=600):
        req = urllib.request.Request(self.url + path, data=body, method=method,
                                     headers={"X-Probe-Token": self.token})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return data if raw else json.loads(data or b"{}")


class RemoteCondition:
    """Condition, but measured and topped up on the machine under test via the probe."""

    def __init__(self, args, run_dir, probe):
        self.args, self.run_dir, self.probe, self.active = args, run_dir, probe, False

    def _hostload(self, target, **flags):
        q = urllib.parse.urlencode({"target": target, **{k: "1" for k, v in flags.items() if v is True},
                                    **{k: v for k, v in flags.items() if not isinstance(v, bool)}})
        res = self.probe.call("GET", f"/hostload?{q}")
        print(res["log"], end="", flush=True)
        return res

    def enter(self, target):
        res = self._hostload(target, **{"no-topup": self.args.no_topup, "force": self.args.force})
        plan = res["report"]
        json.dump(plan, open(os.path.join(self.run_dir, f"hostload_{target}.json"), "w"), indent=2, default=str)
        if res["exit_code"] != 0:
            return None
        if plan["action"] == "topup":
            self.probe.call("POST", "/contention/start", json.dumps(plan).encode())
            self.active = True
            t0 = time.time()
            while not self.probe.call("GET", "/contention/ready")["ready"] and time.time() - t0 < 300:
                time.sleep(2)
            verify = self._hostload(target, verify=True, window="15")["report"]
            json.dump(verify, open(os.path.join(self.run_dir, f"hostload_{target}_verify.json"), "w"),
                      indent=2, default=str)
            plan["verified"] = verify["measured_class"]
        return plan

    def exit(self):
        if self.active:
            self.probe.call("POST", "/contention/stop")
            self.active = False


def point_status(rc, path):
    """ok | partial (some requests failed) | failed (bench error or no request completed).
    `vllm bench serve` exits 0 even when every request fails, so the result file decides."""
    if rc != 0 or not os.path.exists(path):
        return "failed", f"exit code {rc}" if rc else "no result file"
    try:
        r = json.load(open(path))
    except ValueError:
        return "failed", "unreadable result file"
    if not r.get("completed"):
        return "failed", "no request completed"
    if r.get("failed"):
        return "partial", f"{r['failed']} of {r['completed'] + r['failed']} requests failed"
    return "ok", ""


def scenario_tokens(scen):
    """Prompt + output tokens one request of this scenario needs."""
    if scen["dataset"] == "prefix_repetition":
        return scen["prefix_len"] + scen["suffix_len"] + scen["output_len"]
    return scen.get("input_len", scen.get("input_len_hint", 0)) + scen["output_len"]


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
    ap.add_argument("--cpu-offload-gb", type=float, default=0,
                    help="constraint test: keep this many GB of weights in system RAM (model larger than VRAM)")
    ap.add_argument("--gpu-power-limit", type=int, default=0,
                    help="constraint test: cap NVIDIA GPU power (W) for the run via sudo nvidia-smi -pl; restored after")
    ap.add_argument("--base-url", default="", help="benchmark an already-running server")
    ap.add_argument("--probe", default="", help="harness/probe.py URL on the machine under test (LAN-client runs)")
    ap.add_argument("--client", choices=["local", "docker"], default="local" if shutil.which("vllm") else "docker")
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    ap.add_argument("--online", action="store_true", help="allow HF downloads during the run (default: offline)")
    ap.add_argument("--no-topup", action="store_true", help="never add synthetic load; label runs by measured load")
    ap.add_argument("--force", action="store_true", help="proceed even if the quiet check fails")
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--startup-timeout", type=int, default=1800)
    ap.add_argument("--max-restarts", type=int, default=2,
                    help="restart a crashed server this many times per run before aborting")
    ap.add_argument("--server-log", default="",
                    help="log file of an external server, read to classify crashes (OOM etc.)")
    ap.add_argument("--external-recovery-timeout", type=int, default=180,
                    help="seconds to wait for an external server to come back after a crash")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--print-serve-args", action="store_true",
                    help="print the vLLM serve arguments for this config and exit (for external launches)")
    ap.add_argument("--print-download", action="store_true",
                    help="print the `hf download` commands for this model (and its speculator) and exit")
    ap.add_argument("--serve-only", action="store_true",
                    help="start the server with this config and keep it up until Ctrl-C (for Garden runs)")
    args = ap.parse_args()
    if not args.tier:
        sys.exit("set --tier or BENCH_TIER (see configs/tiers.yaml)")

    tier, plat, model = tiers[args.tier], plats["platforms"][args.platform], dict(models[args.model])
    if plat.get("model_field"):
        # Platforms with their own checkpoint format (Apple MLX) swap in that repo.
        alt = model.get(plat["model_field"])
        if not alt:
            sys.exit(f"{args.model} has no `{plat['model_field']}` checkpoint for platform {args.platform}")
        model.update(hf=alt["hf"], revision=alt.get("revision"), quant=f"{plat['model_field']}-4bit")
    profile = profiles[args.profile]
    if model.get("platforms") and args.platform not in model["platforms"]:
        sys.exit(f"{args.model} only runs on {model['platforms']}")
    args.max_model_len = args.max_model_len or min(tier["max_model_len"], model.get("max_model_len", 1 << 30))
    if profile.get("needs_tool_parser") and not model.get("tool_call_parser"):
        sys.exit(f"{args.model} has no tool_call_parser; it cannot serve the '{args.profile}' profile")
    if profile.get("gpu_only") and plat["mode"] != "gpu":
        sys.exit(f"profile '{args.profile}' needs a GPU platform")
    if profile.get("needs_speculator") and not model.get("speculator"):
        sys.exit(f"{args.model} has no speculator in models.yaml; '{args.profile}' needs one")
    if args.print_download:
        for repo in filter(None, [model, model.get("speculator") if profile.get("needs_speculator") else None]):
            print(shlex.join(["hf", "download", repo["hf"]] + (["--revision", repo["revision"]] if repo.get("revision") else [])))
        return
    args.client_image = plat.get("image", "vllm/vllm-openai-cpu:v0.30.0-x86_64")
    if plat.get("launch") == "external" and args.client == "docker":
        args.client_image = plats["platforms"]["cpu"]["image"]

    if args.print_serve_args:
        print(shlex.join(Server(args, plat, model, profile, tier, None).serve_args()))
        return

    machine = os.environ.get("BENCH_MACHINE_ID", pyplatform.node())
    constraint = "".join([f"_off{args.cpu_offload_gb:g}" if args.cpu_offload_gb else "",
                          f"_pl{args.gpu_power_limit}" if args.gpu_power_limit else ""])
    run_id = f"{dt.datetime.now():%Y%m%d-%H%M%S}_{machine}_{args.model}_{args.platform}_{args.profile}{constraint}"
    if (args.cpu_offload_gb or args.gpu_power_limit) and plat["mode"] != "gpu":
        sys.exit("--cpu-offload-gb and --gpu-power-limit apply to GPU platforms")
    run_dir = os.path.abspath(os.path.join(args.results, run_id))
    os.makedirs(run_dir)
    log(f"run dir: {run_dir}")

    env = dict(os.environ, BENCH_TIER=args.tier, VLLM_IMAGE=plat.get("image", ""))
    probe = Probe(args.probe) if args.probe else None
    if probe and not args.base_url:
        sys.exit("--probe needs --base-url: start the server on the machine under test with --serve-only")
    if probe:
        # fingerprint.json always describes the machine under test.
        open(os.path.join(run_dir, "fingerprint.json"), "wb").write(probe.call("GET", "/fingerprint", raw=True))
        with open(os.path.join(run_dir, "client_fingerprint.json"), "w") as f:
            py("fingerprint.py", stdout=f, env=env, check=True)
    else:
        with open(os.path.join(run_dir, "fingerprint.json"), "w") as f:
            py("fingerprint.py", stdout=f, env=env, check=True)
    fp = json.load(open(os.path.join(run_dir, "fingerprint.json")))

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cond = RemoteCondition(args, run_dir, probe) if probe else Condition(args, run_dir)
    # The quiet gate runs before the server exists so it cannot count the model load as noise.
    if "quiet" in conditions and cond.enter("quiet") is None:
        sys.exit("machine is not quiet (see hostload_quiet.json); close apps or pass --force")

    manifest = {
        "run_id": run_id, "machine_id": machine, "tier": args.tier, "model": args.model,
        "model_label": model["hf"], "quant": model["quant"], "params_b": model.get("params_b"),
        "platform": args.platform, "mode": plat["mode"], "engine_profile": args.profile,
        "vllm_version": plats["vllm_version"], "vllm_image": "external" if args.base_url else plat.get("image", "external"),
        "harness_commit": fp["software"].get("harness_commit"),
        "harness_dirty": fp["software"].get("harness_dirty"),
        "cpu_offload_gb": args.cpu_offload_gb or 0, "gpu_power_limit_w": args.gpu_power_limit or 0,
        "power_source": fp["spec"].get("power", ""), "machine_class": fp["spec"].get("machine_class", ""),
        "max_model_len": args.max_model_len, "repeats": args.repeats,
        "client": "lan" if probe else args.client, "probe": args.probe or None,
        "stress_ng": bool(shutil.which("stress-ng")), "operator": os.environ.get("BENCH_OPERATOR", ""),
        "notes": args.notes, "conditions": {}, "points": [], "crashes": [],
    }
    mpath = os.path.join(run_dir, "manifest.json")

    def save():
        json.dump(manifest, open(mpath, "w"), indent=2, default=str)

    power_cap = GpuPowerCap(args.gpu_power_limit) if args.gpu_power_limit else None
    server = Server(args, plat, model, profile, tier, run_dir)
    manifest["serve_cmd"] = server.cmd
    telemetry = None
    exit_code = 0

    def recover(where):
        """Classify a dead server, restart it (managed) or wait for it (external),
        and record the crash with its recovery time. Raises SuiteStop when it cannot recover."""
        was_restarted = server.alive() and server.restarted()
        info = server.diagnose()
        crash = {"t_detected": time.time(), **where, **info, "recovered": False}
        if was_restarted:
            crash["restarted_externally_at"] = server.identity()
        manifest["crashes"].append(crash)
        log(f"  server down: {info['category']}: {info['evidence']}")
        if len(manifest["crashes"]) > args.max_restarts:
            save()
            raise SuiteStop(2, f"server crashed {len(manifest['crashes'])} times; last: {info['category']}")
        t0 = time.time()
        if server.managed:
            try:
                crash["restart_startup_s"] = round(server.restart(), 1)
            except StartupFailure as f:
                crash["restart_failure"] = f.info
                save()
                raise SuiteStop(2, f"restart failed: {f.info['category']}")
        else:
            while time.time() - t0 < args.external_recovery_timeout and not http_ok(server.base_url + "/health"):
                time.sleep(5)
            if not http_ok(server.base_url + "/health"):
                save()
                raise SuiteStop(2, f"external server did not come back ({info['category']})")
        crash["recovered"], crash["recovery_s"] = True, round(time.time() - t0, 1)
        server.remember_identity()
        log(f"  server recovered in {crash['recovery_s']}s")
        save()
        return crash

    try:
        try:
            manifest["startup_s"] = round(server.start(), 1)
            server.remember_identity()
        except StartupFailure as f:
            manifest["startup_failure"] = f.info
            log(f"server did not start: {f.info['category']}: {f.info['evidence']}")
            raise SuiteStop(3, f"server did not start ({f.info['category']})")
        manifest["serve_cmd"] = server.cmd
        if not server.managed:
            # An external server's real context limit decides which scenarios fit.
            try:
                served = json.load(urllib.request.urlopen(server.base_url + "/v1/models", timeout=5))["data"][0]
                if served.get("max_model_len"):
                    args.max_model_len = manifest["max_model_len"] = served["max_model_len"]
            except (OSError, ValueError, KeyError, IndexError):
                pass
        if args.serve_only:
            save()
            log(f"serving {args.model} at {server.base_url}/v1 (Ctrl-C to stop)")
            while server.alive():
                time.sleep(5)
            info = server.diagnose()
            manifest["crashes"].append({"t_detected": time.time(), "phase": "serve_only", **info, "recovered": False})
            raise SuiteStop(2, f"server exited: {info['category']}: {info['evidence']}")
        try:
            open(os.path.join(run_dir, "metrics_ready.txt"), "w").write(
                urllib.request.urlopen(server.base_url + "/metrics", timeout=5).read().decode())
        except OSError:
            pass
        if probe:
            # The probe scrapes vLLM from the machine under test, so it uses the server's local port.
            port = urllib.parse.urlparse(server.base_url).port or PORT
            probe.call("POST", "/telemetry/start?" + urllib.parse.urlencode(
                {"metrics_url": f"http://127.0.0.1:{port}/metrics"}))
            telemetry = "probe"
        else:
            telemetry = subprocess.Popen([sys.executable, os.path.join(HERE, "telemetry.py"),
                                          "--out", os.path.join(run_dir, "telemetry.csv"),
                                          "--metrics-url", server.base_url + "/metrics"])
        save()

        groups = scenarios_cfg.get("groups", {})
        scen_names = [n for s in args.scenarios.split(",") if s.strip()
                      for n in groups.get(s.strip(), [s.strip()])]
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
                        if scen["dataset"] == "custom" and not os.path.exists(os.path.join(DATASETS_ROOT, scen["dataset_path"])):
                            if rep == 0:
                                log(f"skipping {sname}: {scen['dataset_path']} missing (build it with harness/flores.py)")
                            continue
                        need = max(scen.get("min_model_len", 0), scenario_tokens(scen))
                        if need > args.max_model_len:
                            if rep == 0:
                                log(f"skipping {sname}: needs {need} tokens, server max_model_len is {args.max_model_len}")
                            continue
                        for conc in sweep:
                            if any(k[:2] == (target, sname) and k[2] <= conc for k in saturated):
                                continue
                            where = {"condition": target, "scenario": sname, "concurrency": conc, "repeat": rep}
                            if not server.alive() or server.restarted():
                                recover({**where, "phase": "before_point"})
                            n = max(tier["min_prompts"], conc * tier["prompts_per_slot"])
                            fname = f"{target}_{sname}_c{conc}_r{rep}.json"
                            meta = {"run_id": run_id, "condition": target, "scenario": sname, "repeat": rep}
                            cmd = bench_cmd(args, server, model, scen, conc, n, 1000 + rep, run_dir, fname, meta)
                            log(f"rep {rep} [{target}] {sname} c={conc} n={n}")
                            t0 = time.time()
                            with open(os.path.join(run_dir, "bench.log"), "a") as blog:
                                blog.write(f"\n$ {shlex.join(cmd)}\n")
                                blog.flush()
                                rc = subprocess.run(cmd, stdout=blog, stderr=subprocess.STDOUT,
                                                    env=dict(os.environ, HF_HUB_OFFLINE="0" if args.online else "1")
                                                    ).returncode
                            t1 = time.time()
                            status, reason = point_status(rc, os.path.join(run_dir, fname))
                            if not server.alive() or server.restarted():
                                # The server died during this point: the point failed because of the crash.
                                crash = recover({**where, "phase": "during_point"})
                                status, reason = "failed", f"server crashed: {crash['category']}"
                            manifest["points"].append({
                                "condition": target, "scenario": sname, "concurrency": conc, "repeat": rep,
                                "num_prompts": n, "t_start": t0, "t_end": t1, "exit_code": rc,
                                "status": status, "reason": reason, "result_file": fname})
                            save()
                            if status != "ok":
                                saturated.add((target, sname, conc))
                                log(f"  {sname} c={conc} {status.upper()}: {reason} (see bench.log); "
                                    f"skipping higher concurrency")
                            elif goodput_ratio(os.path.join(run_dir, fname)) < 0.25:
                                saturated.add((target, sname, conc))
                                log(f"  {sname} saturated at c={conc}; skipping higher concurrency")
                            time.sleep(args.cooldown)
                finally:
                    cond.exit()
    except KeyboardInterrupt:
        log("interrupted; saving partial results")
    except SuiteStop as e:
        log(f"stopping: {e}")
        manifest["aborted"] = str(e)
        exit_code = e.code
    finally:
        if telemetry == "probe":
            probe.call("POST", "/telemetry/stop")
            open(os.path.join(run_dir, "telemetry.csv"), "wb").write(probe.call("GET", "/telemetry.csv", raw=True))
        elif telemetry:
            telemetry.send_signal(signal.SIGINT)
            telemetry.wait(timeout=10)
        cond.exit()
        server.stop()
        if power_cap:
            power_cap.restore()
        save()
    log(f"done: {len(manifest['points'])} points -> {mpath}")
    log(f"aggregate with: python harness/summarize.py {args.results}")
    failed = [p for p in manifest["points"] if p["status"] == "failed"]
    if manifest["crashes"]:
        log(f"{len(manifest['crashes'])} server crash(es): "
            + ", ".join(f"{c['category']} ({'recovered in %ss' % c['recovery_s'] if c['recovered'] else 'not recovered'})"
                        for c in manifest["crashes"]))
    if exit_code or failed:
        log(f"{len(failed)} point(s) failed; see bench.log and server.log")
        sys.exit(exit_code or 2)


if __name__ == "__main__":
    main()
