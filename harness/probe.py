#!/usr/bin/env python3
"""Probe service for LAN-client runs: runs on the machine under test and does
the host-side work (fingerprint, host-condition checks and top-up, telemetry)
for a run_suite.py driving the benchmark from another machine.

On the machine under test:
  BENCH_PROBE_TOKEN=<secret> python harness/probe.py --port 9109
  python harness/run_suite.py --model M --platform P --serve-only
On the client machine:
  BENCH_PROBE_TOKEN=<secret> python harness/run_suite.py --model M --platform P \
      --base-url http://<target>:8000 --probe http://<target>:9109

Every request must carry the token in X-Probe-Token; do not expose the port
beyond the local network.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = tempfile.mkdtemp(prefix="bench-probe-")
procs = {}
lock = threading.Lock()


def run_py(script, *args):
    return subprocess.run([sys.executable, os.path.join(HERE, script), *args], capture_output=True, text=True)


def stop(name):
    p = procs.pop(name, None)
    if p and p.poll() is None:
        p.send_signal(signal.SIGINT if name == "telemetry" else signal.SIGTERM)
        p.wait(timeout=30)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(raw)

    def _route(self, method):
        if self.headers.get("X-Probe-Token") != self.server.token:
            return self._send(403, {"error": "bad token"})
        url = urllib.parse.urlparse(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with lock:
            return self._dispatch(method, url.path, q, body)

    def _dispatch(self, method, path, q, body):
        if (method, path) == ("GET", "/fingerprint"):
            r = run_py("fingerprint.py")
            return self._send(200 if r.returncode == 0 else 500, r.stdout.encode() or r.stderr.encode())
        if (method, path) == ("GET", "/hostload"):
            out = os.path.join(WORK, f"hostload_{q['target']}.json")
            flags = ["--target", q["target"], "--out", out]
            flags += [f"--{k}" for k in ("no-topup", "force", "verify") if q.get(k) == "1"]
            flags += ["--window", q["window"]] if q.get("window") else []
            r = run_py("hostload.py", *flags)
            report = json.load(open(out)) if os.path.exists(out) else {}
            return self._send(200, {"exit_code": r.returncode, "log": r.stdout + r.stderr, "report": report})
        if (method, path) == ("POST", "/contention/start"):
            stop("contention")
            plan = os.path.join(WORK, "contention_plan.json")
            ready = os.path.join(WORK, "contention_ready")
            open(plan, "wb").write(body)
            if os.path.exists(ready):
                os.remove(ready)
            procs["contention"] = subprocess.Popen([sys.executable, os.path.join(HERE, "contention.py"),
                                                    "--plan", plan, "--ready-file", ready])
            return self._send(200, {"ready_file": ready})
        if (method, path) == ("GET", "/contention/ready"):
            return self._send(200, {"ready": os.path.exists(os.path.join(WORK, "contention_ready"))})
        if (method, path) == ("POST", "/contention/stop"):
            stop("contention")
            return self._send(200, {})
        if (method, path) == ("POST", "/telemetry/start"):
            stop("telemetry")
            out = os.path.join(WORK, "telemetry.csv")
            procs["telemetry"] = subprocess.Popen([sys.executable, os.path.join(HERE, "telemetry.py"),
                                                   "--out", out, "--metrics-url", q.get("metrics_url", "")])
            return self._send(200, {})
        if (method, path) == ("POST", "/telemetry/stop"):
            stop("telemetry")
            return self._send(200, {})
        if (method, path) == ("GET", "/telemetry.csv"):
            path = os.path.join(WORK, "telemetry.csv")
            return self._send(200, open(path, "rb").read() if os.path.exists(path) else b"", "text/csv")
        return self._send(404, {"error": f"no route {method} {path}"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def log_message(self, fmt, *a):
        print("[probe] " + fmt % a, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9109)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    token = os.environ.get("BENCH_PROBE_TOKEN")
    if not token:
        sys.exit("set BENCH_PROBE_TOKEN (shared secret; the client must send the same value)")
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.token = token

    def shutdown(*_):
        for name in list(procs):
            stop(name)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"[probe] listening on {args.bind}:{args.port}, work dir {WORK}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
