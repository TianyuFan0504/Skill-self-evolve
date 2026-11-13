#!/usr/bin/env python3
"""CPU-only idempotent request and HTTP transport contracts."""

import contextlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load("alfworld_request_client")
BASE = "http://127.0.0.1:45678/session/" + "a" * 32
RID = "b" * 32
HEALTH = {"ready": True, "action_request_protocol": client.PROTOCOL, "server_instance_id": "instance1"}
STATE = {"observation": "fixture state", "steps": 1, "won": False}


def completed(status=200, result=STATE):
    return 200, {"request_id": RID, "status": "completed", "response_status": status, "result": result}


class Scripted:
    def __init__(self, results):
        self.results, self.calls, self.now = list(results), [], 0.0

    def __call__(self, method, url, payload, timeout):
        self.now += 0.01
        self.calls.append((method, url, payload))
        if not self.results:
            raise AssertionError("Unexpected HTTP request")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def run(self, **kwargs):
        return client.request_action(BASE, " look ", transport=self, request_id=RID,
                                     clock=lambda: self.now, sleep=self.sleep, **kwargs)

    def sleep(self, seconds):
        self.now += seconds

    @property
    def posts(self):
        return [row for row in self.calls if row[0] == "POST"]


class RetryTests(unittest.TestCase):
    def test_timeout_then_cached_result_never_reexecutes(self):
        script = Scripted([(200, HEALTH), TimeoutError(), completed()])
        self.assertEqual(script.run(), STATE)
        self.assertEqual(len(script.posts), 1)
        self.assertEqual(script.posts[0][2], {"action": "look", "request_id": RID})

    def test_unknown_reuses_exact_id_after_instance_check(self):
        script = Scripted([(200, HEALTH), TimeoutError(), (404, {}), (200, HEALTH), (200, STATE)])
        self.assertEqual(script.run(), STATE)
        self.assertEqual(len(script.posts), 2)
        self.assertEqual(script.posts[0][2], script.posts[1][2])

    def test_pending_and_lookup_failures_only_query(self):
        pending = {"request_id": RID, "status": "pending", "timing": 0.1}
        script = Scripted([(200, HEALTH), OSError(), (202, pending), OSError(), (503, {}), completed()])
        self.assertEqual(script.run(), STATE)
        self.assertEqual(len(script.posts), 1)

    def test_cached_error_is_final(self):
        script = Scripted([(200, HEALTH), TimeoutError(), completed(500, {"error": "engine failure"})])
        with self.assertRaises(client.ActionResponseError) as error:
            script.run()
        self.assertEqual(error.exception.status, 500)
        self.assertEqual(len(script.posts), 1)

    def test_direct_conflict_is_final(self):
        script = Scripted([(200, HEALTH), (409, {"error": "conflict"})])
        with self.assertRaises(client.ActionResponseError):
            script.run()
        self.assertEqual(len(script.calls), 2)

    def test_legacy_server_cannot_receive_action(self):
        script = Scripted([(200, {"ready": True})])
        with self.assertRaises(client.RequestProtocolError):
            script.run()
        self.assertEqual(script.posts, [])

    def test_server_restart_cannot_receive_replay(self):
        script = Scripted([(200, HEALTH), TimeoutError(), (404, {}),
                           (200, {**HEALTH, "server_instance_id": "instance2"})])
        with self.assertRaises(client.RequestProtocolError):
            script.run()
        self.assertEqual(len(script.posts), 1)

    def test_bad_completed_id_is_unknown(self):
        status, body = completed()
        script = Scripted([(200, HEALTH), TimeoutError(), (status, {**body, "request_id": "c" * 32})])
        with self.assertRaises(client.RequestProtocolError):
            script.run()

    def test_deadline_does_not_create_new_id(self):
        script = Scripted([(200, HEALTH), TimeoutError(),
                           (202, {"request_id": RID, "status": "pending"})] * 5)
        with self.assertRaises(TimeoutError):
            script.run(total_timeout=0.15, poll_interval=0.2)
        self.assertEqual(len(script.posts), 1)

    def test_bounds_and_loopback_only(self):
        with self.assertRaises(ValueError):
            client.request_action(BASE, "look", total_timeout=float("nan"))
        for url in ("https://127.0.0.1:5/session/" + "a" * 32, "http://example.com/session/" + "a" * 32):
            with self.assertRaises(client.RequestProtocolError):
                client.checked_base(url)

    def test_cli_state_only_by_default_and_opt_in_debug_trace(self):
        def action(base, text, *, trace):
            trace.update(request_id=RID, posts=1)
            return STATE

        for debug in (False, True):
            stdout, stderr = io.StringIO(), io.StringIO()
            environ = {"ALFWORLD_SESSION_URL": BASE, "ALFWORLD_REQUEST_TRACE": "1" if debug else "0"}
            with patch.object(client, "request_action", side_effect=action), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(client.main(["look"], environ), 0)
            self.assertEqual(json.loads(stdout.getvalue()), STATE)
            if debug:
                self.assertTrue(stderr.getvalue().startswith("ALFWORLD_REQUEST_TIMING "))
            else:
                self.assertEqual(stderr.getvalue(), "")


class HttpClientTests(unittest.TestCase):
    def test_actual_public_tool_recovers_dropped_post_over_http_without_duplicate_step(self):
        records = []

        class Handler(BaseHTTPRequestHandler):
            def response(self, value):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())

            def do_GET(self):
                if self.path == "/health":
                    self.response(HEALTH)
                else:
                    record = records[0]
                    if self.path != "/session/" + "a" * 32 + "/requests/" + record["request_id"]:
                        raise AssertionError("Unexpected lookup path")
                    self.response({"request_id": record["request_id"], "status": "completed",
                                   "response_status": 200, "result": STATE})

            def do_POST(self):
                record = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.headers.get("X-Request-ID") != record["request_id"]:
                    raise AssertionError("Header/body request IDs differ")
                records.append(record)
                # Simulate completed execution whose HTTP response was lost.
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()

            def log_message(self, *args):
                pass

        service = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            env = {**os.environ, "ALFWORLD_SESSION_URL":
                   f"http://127.0.0.1:{service.server_port}/session/" + "a" * 32}
            process = subprocess.run([sys.executable, str(HERE / "alfworld_request_client.py"), "look"],
                                     env=env, capture_output=True, text=True, timeout=5, check=True)
            self.assertEqual(json.loads(process.stdout), STATE)
            self.assertEqual(process.stderr, "")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["action"], "look")
            self.assertEqual(len(records[0]["request_id"]), 32)
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
