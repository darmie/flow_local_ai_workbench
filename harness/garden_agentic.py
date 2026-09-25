#!/usr/bin/env python3
"""Agentic workload benchmark: drive Garden issue runs against a local vLLM
endpoint and collect per-run outcome, latency, token and tool-call metrics.

Garden must already be running with GARDEN_MODEL_BASE_URL pointed at the vLLM
server under test (see docs/GARDEN_AGENTIC.md). This script:
  1. signs in to Garden (email/password) and uses the benchmark workspace,
  2. for each task x repeat: creates an issue assigned to the benchmark agent,
     starts a run, polls until the run is terminal (or times out),
  3. exports issue_run + issue_run_event rows from Garden's Postgres,
  4. scores each run against the task's `expect` checks,
  5. writes runs.jsonl + agentic_summary.csv into the run directory.

Usage:
  python harness/garden_agentic.py --tasks configs/garden_tasks.yaml \
      --run-dir results/<run_id> --parallel 1 --repeats 3
Env:
  GARDEN_URL (default http://localhost:3000), GARDEN_EMAIL, GARDEN_PASSWORD,
  GARDEN_WORKSPACE_ID, GARDEN_AGENT_ID,
  GARDEN_DATABASE_URL (default postgresql://garden:garden@localhost:55432/garden)
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

import yaml

TERMINAL = {"succeeded", "failed", "cancelled", "blocked"}
STALLED = {"waiting_for_input", "waiting_for_approval"}


class Garden:
    def __init__(self, base, workspace_id):
        self.base = base.rstrip("/")
        self.ws = workspace_id
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Origin", self.base)
        if self.ws:
            req.add_header("X-Workspace-ID", self.ws)
        try:
            with self.opener.open(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:300]!r}") from None

    def sign_in(self, email, password):
        self.call("POST", "/api/auth/sign-in/email", {"email": email, "password": password})


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
    # Work products are stored as rows on issue_work_product; dump them whole so
    # checks can match on whatever the agent produced.
    try:
        return psql(db_url, f"select coalesce(json_agg(w)::text, '') from issue_work_product w "
                            f"where issue_id = '{issue_id}'")
    except RuntimeError:
        return ""


def score(task, text, tool_events, status):
    checks = {"run_succeeded": status == "succeeded"}
    for i, pattern in enumerate(task.get("expect", {}).get("regex", [])):
        checks[f"regex_{i}"] = re.search(pattern, text, re.I | re.S) is not None
    used = {e.get("payload", {}).get("tool") for e in tool_events}
    for tool in task.get("expect", {}).get("tools_used", []):
        checks[f"tool_{tool}"] = tool in used
    max_steps = task.get("expect", {}).get("max_steps")
    return checks, max_steps


def run_one(g, args, task, repeat):
    t_submit = time.time()
    issue = g.call("POST", "/api/issues", {
        "title": f"[bench] {task['id']} r{repeat}",
        "description": (args.preamble + "\n\n" + task["prompt"].strip()).strip(),
        "status": "todo",
        "assignee_type": "agent",
        "assignee_id": args.agent_id,
        "auto_start": False,
    })
    issue_id = issue["id"]
    started = g.call("POST", f"/api/issues/{issue_id}/runs")
    run_id = (started or {}).get("id") or (started or {}).get("run", {}).get("id")

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

    run_row, events = export_run(args.db_url, run_id) if run_id else ({}, [])
    tools = [e for e in events if e["event_type"] == "issue_run:tool_finished"]
    usage = run_row.get("usage_json") or {}
    text = json.dumps(run_row.get("result_json") or "") + work_product_text(args.db_url, issue_id)
    checks, max_steps = score(task, text, tools, status)
    steps = usage.get("step_count")
    if max_steps and steps is not None:
        checks["within_step_budget"] = steps <= max_steps
    tool_ms = sum((e.get("payload") or {}).get("duration_ms") or 0 for e in tools)
    wall = t_done - t_submit
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
        # Everything that is not tool execution is (to first order) model time.
        "model_time_s": round(max(0.0, wall - tool_ms / 1000), 2),
        "error": run_row.get("error"),
        "model": usage.get("model"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="configs/garden_tasks.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--parallel", type=int, default=1, help="concurrent agent runs (simulated users)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--with", dest="with_", default="", help="enable tasks needing these features, e.g. sandbox")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--garden-url", default=os.environ.get("GARDEN_URL", "http://localhost:3000"))
    ap.add_argument("--db-url", default=os.environ.get(
        "GARDEN_DATABASE_URL", "postgresql://garden:garden@localhost:55432/garden"))
    ap.add_argument("--agent-id", default=os.environ.get("GARDEN_AGENT_ID"))
    ap.add_argument("--metrics-url", default="http://localhost:8000/metrics",
                    help="vLLM /metrics scraped by telemetry during the runs; empty disables telemetry")
    args = ap.parse_args()
    if not args.agent_id:
        sys.exit("set GARDEN_AGENT_ID (the benchmark agent's id in the workspace)")

    spec = yaml.safe_load(open(args.tasks))
    args.preamble = spec.get("preamble", "")
    enabled = set(filter(None, args.with_.split(",")))
    tasks = [t for t in spec["tasks"] if set(t.get("requires", [])) <= enabled]
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in keep]

    g = Garden(args.garden_url, os.environ.get("GARDEN_WORKSPACE_ID", ""))
    g.sign_in(os.environ["GARDEN_EMAIL"], os.environ["GARDEN_PASSWORD"])

    os.makedirs(args.run_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "hostload.py"), "--target", "quiet", "--verify",
                    "--window", "10", "--out", os.path.join(args.run_dir, f"agentic_p{args.parallel}_hostload.json")])
    telemetry = None
    if args.metrics_url:
        telemetry = subprocess.Popen([sys.executable, os.path.join(here, "telemetry.py"), "--out",
                                      os.path.join(args.run_dir, f"agentic_p{args.parallel}_telemetry.csv"),
                                      "--metrics-url", args.metrics_url])
    jobs = [(t, r) for r in range(args.repeats) for t in tasks]
    results = []
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
            res["parallel"] = args.parallel
            results.append(res)
            log.write(json.dumps(res, default=str) + "\n")
            log.flush()
            print(f"[garden] {tid} r{r}: {res['status']} success={res['success']} "
                  f"wall={res.get('wall_s')}s tools={res.get('tool_calls')}", flush=True)

    if telemetry:
        telemetry.send_signal(signal.SIGINT)
        telemetry.wait(timeout=10)

    fields = ["task", "category", "repeat", "parallel", "status", "success", "wall_s",
              "model_time_s", "tool_time_s", "steps", "tool_calls", "tool_errors",
              "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens",
              "t_submit", "t_done", "model", "run_id", "error"]
    path = os.path.join(args.run_dir, f"agentic_p{args.parallel}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(results, key=lambda x: (x["task"], x["repeat"])))
    ok = sum(1 for x in results if x["success"])
    print(f"[garden] {ok}/{len(results)} succeeded -> {path}")


if __name__ == "__main__":
    main()
