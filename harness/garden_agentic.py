#!/usr/bin/env python3
"""Agentic workload benchmark: drive Garden issue runs against a local vLLM
endpoint and collect per-run outcome, latency, token and tool-call metrics.

Garden must already be running with GARDEN_MODEL_BASE_URL pointed at the vLLM
server under test (see docs/GARDEN_AGENTIC.md). This script:
  1. signs in to Garden (email/password) and uses the benchmark workspace,
  2. brings the host to the requested condition (as run_suite.py does),
  3. for each task x repeat: uploads the task's attachments, creates an issue
     assigned to the benchmark agent, starts a run and polls until it is terminal,
  4. exports issue_run + issue_run_event rows from Garden's Postgres and the
     vLLM /metrics deltas for the run,
  5. scores each run against the task's `expect` checks,
  6. writes garden_runs.jsonl + agentic_<condition>_p<N>.csv into the run directory.

Usage:
  python harness/garden_agentic.py --run-dir results/<run_id> --parallel 1 --repeats 3
  python harness/garden_agentic.py --run-dir results/<run_id> --condition office --no-topup
Env:
  GARDEN_URL (default http://localhost:3000), GARDEN_EMAIL, GARDEN_PASSWORD,
  GARDEN_WORKSPACE_ID, GARDEN_AGENT_ID,
  GARDEN_DATABASE_URL (default postgresql://garden:garden@localhost:55432/garden)
"""
import argparse
import concurrent.futures as cf
import csv
import json
import mimetypes
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.cookiejar import CookieJar

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TERMINAL = {"succeeded", "failed", "cancelled", "blocked"}
STALLED = {"waiting_for_input", "waiting_for_approval"}
# Garden's own processes serve the agent, so they count as the system under test.
GARDEN_PROCS = r"workerd|miniflare|wrangler|postgres|helix|vite|pnpm"

# vLLM counters whose per-run deltas describe the model calls an agent run made.
# Each entry lists accepted metric names, newest spelling first.
LLM_COUNTERS = {
    "llm_calls": ["vllm:request_success_total", "vllm:request_success"],
    "llm_time_s": ["vllm:e2e_request_latency_seconds_sum"],
    "llm_ttft_sum_s": ["vllm:time_to_first_token_seconds_sum"],
    "llm_ttft_count": ["vllm:time_to_first_token_seconds_count"],
    "llm_queue_s": ["vllm:request_queue_time_seconds_sum"],
    "llm_prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "llm_generation_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "llm_prefix_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
    "llm_prefix_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
}

sys.path.insert(0, HERE)
from telemetry import scrape_raw  # noqa: E402


class Garden:
    """Session-cookie client. Garden has no API keys, so the harness signs in
    as the benchmark user and signs in again if the session expires mid-batch."""

    def __init__(self, base, workspace_id, email, password):
        self.base = base.rstrip("/")
        self.ws = workspace_id
        self.creds = (email, password)
        self.lock = threading.Lock()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def _request(self, method, path, data=None, content_type="application/json"):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        req.add_header("Origin", self.base)
        if self.ws:
            req.add_header("X-Workspace-ID", self.ws)
        with self.opener.open(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None

    def call(self, method, path, body=None, data=None, content_type="application/json", _retry=True):
        if body is not None:
            data = json.dumps(body).encode()
        try:
            return self._request(method, path, data, content_type)
        except urllib.error.HTTPError as e:
            if e.code == 401 and _retry and not path.startswith("/api/auth/"):
                with self.lock:
                    self.sign_in()
                return self.call(method, path, data=data, content_type=content_type, _retry=False)
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:300]!r}") from None

    def sign_in(self):
        email, password = self.creds
        self.call("POST", "/api/auth/sign-in/email", {"email": email, "password": password}, _retry=False)
        me = self.call("GET", "/api/me", _retry=False)
        if not me:
            raise RuntimeError("signed in but /api/me returned nothing; check GARDEN_EMAIL/GARDEN_PASSWORD")
        return me

    def upload(self, path):
        """Upload a file as an issue attachment; returns the attachment record."""
        boundary = uuid.uuid4().hex
        name = os.path.basename(path)
        ctype = mimetypes.guess_type(name)[0] or "text/plain"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
        return self.call("POST", "/api/upload-file", data=body,
                         content_type=f"multipart/form-data; boundary={boundary}")


def psql(db_url, sql):
    out = subprocess.run(["psql", db_url, "-At", "-c", sql], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


def export_run(db_url, run_id):
    run = psql(db_url, f"""select row_to_json(r) from (
        select id, status, error, usage_json, result_json, started_at, finished_at, created_at
        from issue_run where id = '{run_id}') r""")
    events = psql(db_url, f"""select coalesce(json_agg(e order by e.seq), '[]') from (
        select seq, event_type, level, message, payload, created_at
        from issue_run_event where run_id = '{run_id}') e""")
    return json.loads(run) if run else {}, json.loads(events) if events else []


def work_product_text(db_url, issue_id):
    # Dump work products whole so checks can match on whatever the agent produced.
    try:
        return psql(db_url, f"select coalesce(json_agg(w)::text, '') from issue_work_product w "
                            f"where issue_id = '{issue_id}'")
    except RuntimeError:
        return ""


def llm_snapshot(url):
    raw = scrape_raw(url)
    out = {}
    for key, names in LLM_COUNTERS.items():
        out[key] = next((raw[n] for n in names if n in raw), None)
    return out


def llm_delta(before, after):
    d = {k: (after[k] - before[k]) if after.get(k) is not None and before.get(k) is not None else None
         for k in LLM_COUNTERS}
    calls, ttft_n = d["llm_calls"], d["llm_ttft_count"]
    return {
        "llm_calls": int(calls) if calls is not None else None,
        "llm_time_s": round(d["llm_time_s"], 2) if d["llm_time_s"] is not None else None,
        "llm_mean_call_s": round(d["llm_time_s"] / calls, 2) if calls and d["llm_time_s"] is not None else None,
        "llm_mean_ttft_s": round(d["llm_ttft_sum_s"] / ttft_n, 3) if ttft_n and d["llm_ttft_sum_s"] is not None else None,
        "llm_queue_s": round(d["llm_queue_s"], 2) if d["llm_queue_s"] is not None else None,
        "llm_prompt_tokens": int(d["llm_prompt_tokens"]) if d["llm_prompt_tokens"] is not None else None,
        "llm_generation_tokens": int(d["llm_generation_tokens"]) if d["llm_generation_tokens"] is not None else None,
        "llm_prefix_hit_rate": (round(d["llm_prefix_hits"] / d["llm_prefix_queries"], 3)
                                if d["llm_prefix_queries"] and d["llm_prefix_hits"] is not None else None),
    }


def score(task, text, tool_events, status):
    checks = {"run_succeeded": status == "succeeded"}
    for i, pattern in enumerate(task.get("expect", {}).get("regex", [])):
        checks[f"regex_{i}"] = re.search(pattern, text, re.I | re.S) is not None
    used = {(e.get("payload") or {}).get("tool") for e in tool_events}
    for tool in task.get("expect", {}).get("tools_used", []):
        checks[f"tool_{tool}"] = tool in used
    return checks


def describe(task, preamble, attachments):
    text = (preamble + "\n\n" + task["prompt"].strip()).strip()
    if attachments:
        text += "\n\nAttachments (read each with read_attachment):\n" + "\n".join(
            f"- {a['filename']}: attachment_id {a['id']}" for a in attachments)
    return text


def run_one(g, args, task, repeat):
    attachments = [g.upload(os.path.join(ROOT, "configs", "garden_fixtures", p))
                   for p in task.get("attachments", [])]
    per_run_llm = args.parallel == 1 and args.metrics_url
    before = llm_snapshot(args.metrics_url) if per_run_llm else None
    t_submit = time.time()
    issue = g.call("POST", "/api/issues", {
        "title": f"[bench] {task['id']} r{repeat}",
        "description": describe(task, args.preamble, attachments),
        "status": "todo",
        "assignee_type": "agent",
        "assignee_id": args.agent_id,
        "auto_start": False,
        **({"attachment_ids": [a["id"] for a in attachments]} if attachments else {}),
    })
    issue_id = issue["id"]
    started = g.call("POST", f"/api/issues/{issue_id}/runs") or {}
    run_id = started.get("id") or (started.get("run") or {}).get("id")

    status, deadline = "queued", t_submit + task.get("timeout_s", args.timeout)
    while time.time() < deadline:
        runs = g.call("GET", f"/api/issues/{issue_id}/runs")
        runs = runs.get("runs", runs) if isinstance(runs, dict) else runs
        if runs:
            cur = next((r for r in runs if r.get("id") == run_id), runs[0])
            run_id, status = cur.get("id"), cur.get("status")
            if status in TERMINAL or status in STALLED:
                break
        time.sleep(2)
    t_done = time.time()
    if status not in TERMINAL:
        try:
            g.call("POST", f"/api/issues/{issue_id}/cancel")
        except RuntimeError:
            pass
        status = f"timeout({status})"
    llm = llm_delta(before, llm_snapshot(args.metrics_url)) if per_run_llm else {}

    run_row, events = export_run(args.db_url, run_id) if run_id else ({}, [])
    tools = [e for e in events if e["event_type"] == "issue_run:tool_finished"]
    usage = run_row.get("usage_json") or {}
    text = json.dumps(run_row.get("result_json") or "") + work_product_text(args.db_url, issue_id)
    checks = score(task, text, tools, status)
    steps, max_steps = usage.get("step_count"), task.get("expect", {}).get("max_steps")
    if max_steps and steps is not None:
        checks["within_step_budget"] = steps <= max_steps
    tool_ms = sum((e.get("payload") or {}).get("duration_ms") or 0 for e in tools)
    wall = t_done - t_submit
    if llm.get("llm_time_s") is not None:
        model_time, source = llm["llm_time_s"], "vllm_metrics"
    else:
        # Concurrent runs share the counters, so fall back to wall minus tool time.
        model_time, source = max(0.0, wall - tool_ms / 1000), "wall_minus_tools"
    return {
        "task": task["id"], "category": task.get("category", ""), "repeat": repeat,
        "issue_id": issue_id, "run_id": run_id, "status": status,
        "success": all(checks.values()), "checks": checks,
        "t_submit": t_submit, "t_done": t_done, "wall_s": round(wall, 2),
        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"), "steps": steps,
        "tool_calls": len(tools),
        "tool_errors": sum(1 for e in tools if (e.get("payload") or {}).get("ok") is False),
        "tool_time_s": round(tool_ms / 1000, 2),
        "model_time_s": round(model_time, 2), "model_time_source": source,
        **llm,
        "attachments": len(attachments),
        "error": run_row.get("error"),
        "model": usage.get("model"),
    }


def label_conditions(results, telemetry_csv):
    """Tag each run with the host condition measured over its own time window."""
    from hostload import classify, load_conditions
    classes = load_conditions()["classes"]
    try:
        rows = list(csv.DictReader(open(telemetry_csv)))
    except OSError:
        return
    for r in results:
        if "t_submit" not in r:
            continue
        win = [x for x in rows if r["t_submit"] <= float(x["ts"]) <= r["t_done"]]
        bg = [float(x["bg_cpu_pct"]) for x in win if x.get("bg_cpu_pct")]
        sw = [float(x["swap_in_mb_s"]) for x in win if x.get("swap_in_mb_s")]
        if bg:
            r["median_bg_cpu_pct"] = round(statistics.median(bg), 2)
            r["measured_condition"] = classify(statistics.median(bg), None,
                                               statistics.median(sw) if sw else 0, classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=os.path.join(ROOT, "configs", "garden_tasks.yaml"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--parallel", type=int, default=1, help="concurrent agent runs (simulated users)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--with", dest="with_", default="", help="enable tasks needing these features, e.g. sandbox")
    ap.add_argument("--condition", default="quiet", choices=["quiet", "office", "heavy"])
    ap.add_argument("--no-topup", action="store_true", help="never add synthetic load; label runs by measured load")
    ap.add_argument("--force", action="store_true", help="proceed even if the quiet check fails")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--garden-url", default=os.environ.get("GARDEN_URL", "http://localhost:3000"))
    ap.add_argument("--db-url", default=os.environ.get(
        "GARDEN_DATABASE_URL", "postgresql://garden:garden@localhost:55432/garden"))
    ap.add_argument("--agent-id", default=os.environ.get("GARDEN_AGENT_ID"))
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics",
                    help="vLLM /metrics, scraped per run and by telemetry; empty disables both")
    args = ap.parse_args()
    if not args.agent_id:
        sys.exit("set GARDEN_AGENT_ID (the benchmark agent's id in the workspace)")
    os.environ.setdefault("BENCH_EXTRA_HARNESS_PROCS", GARDEN_PROCS)

    spec = yaml.safe_load(open(args.tasks))
    args.preamble = spec.get("preamble", "")
    enabled = set(filter(None, args.with_.split(",")))
    tasks = [t for t in spec["tasks"] if set(t.get("requires", [])) <= enabled]
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]

    g = Garden(args.garden_url, os.environ.get("GARDEN_WORKSPACE_ID", ""),
               os.environ["GARDEN_EMAIL"], os.environ["GARDEN_PASSWORD"])
    g.sign_in()

    os.makedirs(args.run_dir, exist_ok=True)
    fp_path = os.path.join(args.run_dir, "fingerprint.json")
    if not os.path.exists(fp_path):
        with open(fp_path, "w") as f:
            subprocess.run([sys.executable, os.path.join(HERE, "fingerprint.py")], stdout=f, check=True)
    fp = json.load(open(fp_path))
    machine = {"machine_id": fp["machine_id"], "machine_name": fp["spec"]["machine_name"],
               "machine_spec": fp["spec"]["summary"]}

    from run_suite import Condition
    cond = Condition(args, args.run_dir)
    plan = cond.enter(args.condition)
    if plan is None:
        sys.exit(f"cannot reach host condition '{args.condition}' (see hostload_{args.condition}.json)")

    tag = f"{args.condition}_p{args.parallel}"
    tel_path = os.path.join(args.run_dir, f"agentic_{tag}_telemetry.csv")
    telemetry = None
    if args.metrics_url:
        telemetry = subprocess.Popen([sys.executable, os.path.join(HERE, "telemetry.py"), "--out", tel_path,
                                      "--metrics-url", args.metrics_url])
    jobs = [(t, r) for r in range(args.repeats) for t in tasks]
    results = []
    try:
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool, \
                open(os.path.join(args.run_dir, "garden_runs.jsonl"), "a") as log:
            futs = {pool.submit(run_one, g, args, t, r): (t["id"], r) for t, r in jobs}
            for fut in cf.as_completed(futs):
                tid, r = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # keep going; record the harness failure
                    res = {"task": tid, "repeat": r, "status": "harness_error", "success": False,
                           "error": str(e)}
                res.update(parallel=args.parallel, target_condition=args.condition, **machine)
                results.append(res)
                log.write(json.dumps(res, default=str) + "\n")
                log.flush()
                print(f"[garden] {tid} r{r}: {res['status']} success={res['success']} "
                      f"wall={res.get('wall_s')}s llm_calls={res.get('llm_calls')}", flush=True)
    finally:
        if telemetry:
            telemetry.send_signal(signal.SIGINT)
            telemetry.wait(timeout=10)
        cond.exit()

    label_conditions(results, tel_path)
    fields = ["machine_id", "machine_name", "machine_spec", "task", "category", "repeat", "parallel",
              "target_condition", "measured_condition", "median_bg_cpu_pct", "status", "success",
              "wall_s", "model_time_s", "model_time_source", "tool_time_s",
              "llm_calls", "llm_time_s", "llm_mean_call_s", "llm_mean_ttft_s", "llm_queue_s",
              "llm_prompt_tokens", "llm_generation_tokens", "llm_prefix_hit_rate",
              "steps", "tool_calls", "tool_errors", "attachments",
              "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens",
              "t_submit", "t_done", "model", "run_id", "error"]
    path = os.path.join(args.run_dir, f"agentic_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(results, key=lambda x: (x["task"], x["repeat"])))
    json.dump({"condition_plan": plan, "tasks": [t["id"] for t in tasks]},
              open(os.path.join(args.run_dir, f"agentic_{tag}_meta.json"), "w"), indent=2, default=str)
    ok = sum(1 for x in results if x["success"])
    print(f"[garden] {ok}/{len(results)} succeeded -> {path}")


if __name__ == "__main__":
    main()
