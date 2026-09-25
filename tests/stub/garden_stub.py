#!/usr/bin/env python3
"""Stub of the Garden HTTP API for garden_agentic.py smoke tests.

Expires the session once (after the first issue is created) to exercise
re-authentication. Run state lives in a JSON file that stub/psql reads.
"""
import http.server
import json
import os
import re
import sys
import time
import uuid

STATE = os.environ.get("GARDEN_STUB_STATE", "/tmp/garden_stub_state.json")
state = {"issues": {}, "runs": {}, "attachments": {}, "expired_once": False, "token": None}


def save():
    json.dump(state, open(STATE, "w"))


class H(http.server.BaseHTTPRequestHandler):
    def _json(self, code, body, headers=()):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _authed(self):
        return state["token"] and f"session={state['token']}" in (self.headers.get("Cookie") or "")

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_POST(self):
        body = self._body()
        if self.path == "/api/auth/sign-in/email":
            creds = json.loads(body)
            if creds.get("password") != "pw":
                return self._json(401, {"error": "bad credentials"})
            state["token"] = uuid.uuid4().hex
            return self._json(200, {"ok": True}, [("Set-Cookie", f"session={state['token']}; Path=/")])
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/api/upload-file":
            name = re.search(rb'filename="([^"]+)"', body).group(1).decode()
            aid = str(uuid.uuid4())
            state["attachments"][aid] = name
            save()
            return self._json(201, {"id": aid, "filename": name})
        if self.path == "/api/issues":
            req = json.loads(body)
            iid = str(uuid.uuid4())
            state["issues"][iid] = req
            if not state["expired_once"]:
                state["expired_once"] = True
                state["token"] = None  # next call gets 401
            save()
            return self._json(201, {"id": iid, "status": "todo"})
        m = re.match(r"^/api/issues/([^/]+)/runs$", self.path)
        if m:
            rid = str(uuid.uuid4())
            state["runs"][rid] = {"issue_id": m.group(1), "started": time.time()}
            save()
            return self._json(202, {"id": rid, "status": "queued"})
        if self.path.endswith("/cancel"):
            return self._json(200, {})
        self._json(404, {})

    def do_GET(self):
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/api/me":
            return self._json(200, {"user": {"email": "bench@example.local"}})
        m = re.match(r"^/api/issues/([^/]+)/runs$", self.path)
        if m:
            runs = [{"id": rid, "status": "succeeded" if time.time() - r["started"] > 1 else "running"}
                    for rid, r in state["runs"].items() if r["issue_id"] == m.group(1)]
            return self._json(200, runs)
        self._json(404, {})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    save()
    http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
