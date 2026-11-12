#!/usr/bin/env python3
"""CPU-only request replay and immutable public wrapper tests."""

import contextlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client, adapter = load("alfworld_request_client"), load("train_parallel_client")
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


class AdapterTests(unittest.TestCase):
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
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, _, instructions = self.prepare(root)
                result = adapter.prepare_adapter(
                    root / "run", root / "profile.json", instructions, public_root=root / "public",
                    client=(HERE / "alfworld_request_client.py").read_text(), shell="exit 0",
                    environment_python=sys.executable, server_script=root / "parallel_environment_server.py",
                )
                tool = Path(result["tool_path"])
                env = {**os.environ, "ALFWORLD_SESSION_URL":
                       f"http://127.0.0.1:{service.server_port}/session/" + "a" * 32}
                process = subprocess.run([str(tool), "look"], env=env, capture_output=True,
                                         text=True, timeout=5, check=True)
                self.assertEqual(json.loads(process.stdout), STATE)
                self.assertEqual(process.stderr, "")
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["action"], "look")
                self.assertEqual(len(records[0]["request_id"]), 32)
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=5)

    def test_health_requires_exact_observed_source_and_four_io_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "parallel_environment_server.py"
            source.write_text("CPU source fixture\n")
            extension = {"source_path": str(source), "source_sha256": adapter.digest(source.read_bytes()),
                         "policy": "sticky-process-environments-durable-io-v1"}
            health = {**HEALTH, "environment_processes": 4, "io_concurrency": 4,
                      "environment_policy": extension["policy"], "environment_extension": extension}
            settings = {"url": "http://127.0.0.1:45678", "environment_extension": extension}
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(health).encode())):
                self.assertEqual(adapter.environment_observation(settings), health)
            bad = {**health, "io_concurrency": 1}
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(bad).encode())):
                with self.assertRaises(ValueError):
                    adapter.environment_observation(settings)
            source.write_text("Changed source\n")
            with patch.object(adapter.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(health).encode())):
                with self.assertRaises(ValueError):
                    adapter.environment_observation(settings)

    def prepare(self, root):
        run = root / "run"
        run.mkdir()
        profile = root / "profile.json"
        rows = [{"name": "alfworld_skill_training", "agent": {"codex_bin": "/old/codex-entrypoint",
                 "run_timeout": 900, "model_catalog_path": "/unchanged/catalog.json"}}]
        profile.write_text(json.dumps(rows))
        instructions = root / "instructions.md"
        instructions.write_text("Unchanged instructions\n")
        tool = "def main():\n import json,sys\n print(json.dumps({'tool':True,'args':sys.argv[1:]}))\n return 0\n"
        shell = ('/usr/bin/python3 -c \'import json,os,sys;'
                 'print(json.dumps({"args":sys.argv[1:],"path":os.environ["PATH"]}))\' "$@"')
        server = root / "parallel_environment_server.py"
        server.write_text("import json,os,sys;print(json.dumps({'args':sys.argv[1:],'pid':os.getpid()}))\n")
        result = adapter.prepare_adapter(run, profile, instructions, public_root=root / "public",
                                         client=tool, shell=shell, environment_python=sys.executable,
                                         server_script=server)
        return result, rows, instructions

    def test_profile_only_changes_native_wrapper_and_repeats_immutably(self):
        with tempfile.TemporaryDirectory() as folder:
            result, rows, instructions = self.prepare(Path(folder))
            changed = json.loads(Path(result["task_profile"]["path"]).read_text())
            changed[0]["agent"]["codex_bin"] = rows[0]["agent"]["codex_bin"]
            self.assertEqual(changed, rows)
            self.assertEqual(instructions.read_bytes(), Path(result["instructions"]["path"]).read_bytes())
            self.assertEqual(Path(result["tool_path"]).stat().st_ino, Path(result["wrapper"]["path"]).stat().st_ino)

    @unittest.skipUnless(Path("/usr/bin/bash").is_file(), "Production wrapper uses the Linux bash path")
    def test_actual_wrapper_argument_forwarding_and_shell_tool_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _ = self.prepare(Path(folder))
            wrapper = result["wrapper"]["path"]
            process = subprocess.run([wrapper, "argument with spaces", "--model"], capture_output=True,
                                     text=True, check=True)
            data = json.loads(process.stdout)
            self.assertEqual(data["args"], ["argument with spaces", "--model"])
            public = str(Path(wrapper).parent)
            self.assertEqual(data["path"].split(os.pathsep)[0], public)
            env = {**os.environ, "PATH": public + os.pathsep + os.environ["PATH"]}
            process = subprocess.run(["/usr/bin/bash", "-c", "command -v alfworld-step; alfworld-step look"],
                                     env=env, capture_output=True, text=True, check=True)
            lines = process.stdout.splitlines()
            self.assertEqual(lines[0], result["tool_path"])
            self.assertEqual(json.loads(lines[1]), {"tool": True, "args": ["look"]})

    def test_environment_python_exact_translation_pid_and_other_script_forwarding(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _ = self.prepare(Path(folder))
            route = result["environment_python_route"]
            executable = result["environment_python_adapter"]["path"]
            args = ["--allowed-splits", "test_seen", "test_unseen", "--max-steps", "50"]
            process = subprocess.Popen([executable, route["translated_server"], *args],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual((process.returncode, stderr), (0, ""))
            data = json.loads(stdout)
            self.assertEqual(data["pid"], process.pid)
            self.assertEqual(data["args"], [*args, "--environment-processes", "4", "--io-workers", "4"])
            stdout = subprocess.check_output([executable, route["parallel_server"]["path"], "unchanged"], text=True)
            self.assertEqual(json.loads(stdout)["args"], ["unchanged"])
            stdout = subprocess.check_output([executable, "-c", "print('original interpreter')"], text=True)
            self.assertEqual(stdout, "original interpreter\n")

    def test_environment_adapter_starts_owned_tiny_test_only_service(self):
        source = r'''
import argparse,json,os
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser();p.add_argument('--config-path');p.add_argument('--allowed-splits',nargs='+')
p.add_argument('--environment-processes',type=int);p.add_argument('--io-workers',type=int);a=p.parse_args()
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(settings).encode())
 def log_message(self,*args): pass
server=HTTPServer(('127.0.0.1',0),Handler)
settings={'ready':True,'pid':os.getpid(),'url':'http://127.0.0.1:'+str(server.server_port),
          'allowed_splits':a.allowed_splits,'environment_processes':a.environment_processes,
          'io_workers':a.io_workers}
with open(a.config_path,'w') as f:json.dump(settings,f)
server.handle_request();server.server_close()
'''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture = root / "tiny_service.py"
            fixture.write_text(source)
            script, metadata = adapter.environment_python_wrapper(sys.executable, fixture)
            executable = root / "environment-python"
            executable.write_bytes(script)
            executable.chmod(0o755)
            settings_path = root / "test-only.json"
            process = subprocess.Popen([str(executable), metadata["translated_server"], "--config-path",
                                        str(settings_path), "--allowed-splits", "test_seen", "test_unseen"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not settings_path.exists() and time.monotonic() < deadline and process.poll() is None:
                    time.sleep(0.01)
                settings = json.loads(settings_path.read_bytes())
                with urllib.request.urlopen(settings["url"] + "/health", timeout=5) as response:
                    health = json.load(response)
                self.assertEqual(health["pid"], process.pid)
                self.assertEqual(health["allowed_splits"], ["test_seen", "test_unseen"])
                self.assertEqual((health["environment_processes"], health["io_workers"]), (4, 4))
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual((process.returncode, stdout, stderr), (0, "", ""))
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=5)

    def test_outer_entry_observes_then_delegates_without_model(self):
        with tempfile.TemporaryDirectory() as folder:
            result, _, _ = self.prepare(Path(folder))
            settings = Path(folder) / "environment.json"
            settings.write_text(json.dumps({"url": "http://127.0.0.1:45678", "admin_token": "do-not-persist"}))
            ns = types.SimpleNamespace
            config = ns(skill_harness=ns(artifact_root=folder), actor_rollout_ref=ns(rollout=ns(custom=ns(
                agent_framework=ns(agent_runners=ns(task=ns(runner_kwargs=ns(task_config_path="old"))))))))
            captured = []
            base = ns(_publish=lambda path, payload: captured.append(payload))
            spawn = ns(extension_identity=lambda: {"policy": "fixture"}, _base=lambda: base,
                       entry=lambda value: "delegated")
            omega = ns(OmegaConf=ns(to_container=lambda value, resolve: {"fixture": True}),
                       open_dict=lambda value: contextlib.nullcontext())
            package = types.ModuleType("skill_harness_examples")
            package.train_spawn_safe = spawn
            train = ns(_redact=lambda value: value)
            environment = {"ALFWORLD_SERVER_CONFIG": str(settings), "ALFWORLD_TASK_CONFIG": "old",
                           "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": "old"}
            with (
                patch.dict(sys.modules, {"omegaconf": omega, "skill_harness_examples": package,
                                         "verl.experimental.skill_harness.train": train}),
                patch.dict(os.environ, environment),
                patch.object(adapter, "environment_observation", return_value={**HEALTH, "environment_processes": 4}),
                patch.object(adapter, "prepare_adapter", return_value=result),
            ):
                self.assertEqual(adapter.entry(config), "delegated")
                self.assertEqual(config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task
                                 .runner_kwargs.task_config_path, result["task_profile"]["path"])
            self.assertTrue(captured[0]["not_a_post_init_or_served_model_receipt"])
            self.assertNotIn("do-not-persist", json.dumps(captured))


if __name__ == "__main__":
    unittest.main()
