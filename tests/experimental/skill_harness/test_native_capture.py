# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise the actual Task capture class with real CPU subprocess groups."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def load_capture_class():
    """Load its source without importing the optional Uni-Agent/model stack."""
    path = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/task.py"
    tree = ast.parse(path.read_text())
    spec = importlib.util.spec_from_file_location("_native_capture_guard", path.with_name("terminal_guard.py"))
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "AlfworldUserSandbox")
    account_spec = importlib.util.spec_from_file_location("_capture_accounts_cpu", path.with_name("accounts.py"))
    accounts = importlib.util.module_from_spec(account_spec)
    account_spec.loader.exec_module(accounts)
    namespace = {
        "AccountIsolationError": accounts.AccountIsolationError,
        "decode_launcher_proof": accounts.decode_launcher_proof,
        "record_isolation_incident": Mock(),
        "asyncio": asyncio,
        "json": json,
        "os": os,
        "signal": signal,
        "time": time,
        "Path": Path,
        "LocalSandbox": object,
        "ExecResult": ExecResult,
        "monitor_terminal_process": guard.monitor_terminal_process,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["AlfworldUserSandbox"]


Sandbox = load_capture_class()


def running(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False)
    return result.returncode == 0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


class NativeCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sandbox = object.__new__(Sandbox)
        self.sandbox.output = self.root
        self.sandbox.session_url = "http://127.0.0.1/unused"
        self.sandbox.account = types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())
        self.sandbox.account_pool = types.SimpleNamespace(register_resource=Mock(), mark_launcher=Mock())
        self.sandbox.account_lease = {"lease_id": "cpu-only", "owner": {"run_root": str(self.root)}}
        self.sandbox.attempt_id = "cpu-attempt"
        self.sandbox.native_model_catalogs = []
        self.sandbox.native_started = False
        self.sandbox.launcher_proofs = []
        self.sandbox.terminal_stop_evidence = []
        self.sandbox.host_stop_evidence = []
        self.requested_host_stops = []
        self.sandbox._request_backend_stop = self.requested_host_stops.append

        async def register_budget(timeout):
            started_ns = time.monotonic_ns()
            return {
                "started_ns": started_ns,
                "deadline_ns": started_ns + int(timeout * 1_000_000_000),
                "budget_seconds": timeout,
                "registration_sha256": "c" * 64,
            }

        async def budget_deadline_stop():
            self.requested_host_stops.append("budget_timeout")

        self.sandbox.register_native_budget = register_budget
        self.sandbox._budget_deadline_stop = budget_deadline_stop
        self.sandbox.private_tmp_identity = None
        self.sandbox.CAPTURE_DRAIN_TIMEOUT_SECONDS = 0.08
        self.sandbox.TERMINATION_TIMEOUT_SECONDS = 0.08
        self.sandbox.TERMINAL_GRACE_SECONDS = 0.04
        self.sandbox.TERMINAL_POLL_SECONDS = 0.005
        self.sandbox.TERMINAL_OBSERVATION_TIMEOUT_SECONDS = 0.03

        async def nonterminal():
            return {"done": False, "won": False, "steps": 0, "max_steps": 50}

        self.sandbox._observe_terminal_state = nonterminal
        self.created = []
        self.loop_errors = []
        self.loop = asyncio.get_running_loop()
        self.old_exception_handler = self.loop.get_exception_handler()
        self.loop.set_exception_handler(lambda _loop, context: self.loop_errors.append(context))
        original = asyncio.create_subprocess_exec

        def fake_launcher(account, argv, *, home, cwd, ready_fd, **kwargs):
            # This fixture exercises capture/drain only. The real kernel launcher
            # has separate OS acceptance; no setpriv/Landlock runs on the Mac.
            self.assertEqual(Path(cwd), self.root)
            self.assertTrue(Path(home).is_dir())
            proof = {
                "policy_version": "alfworld-uid-landlock-v1",
                "uid": account.pw_uid,
                "euid": account.pw_uid,
                "gid": account.pw_gid,
                "groups": [],
                "resuid": [account.pw_uid] * 3,
                "resgid": [account.pw_gid] * 3,
                "no_new_privs": 1,
                "capabilities_zero": True,
                "landlock_abi": 1,
                "landlock_applied": True,
                "seccomp_guard": "abi1-truncate-x86_64-v1",
            }
            os.write(ready_fd, json.dumps(proof).encode())
            return argv, {}, None

        async def direct_exec(*argv, **kwargs):
            process = await original(*argv, **kwargs)
            self.created.append(process)
            return process

        self.launch_patcher = patch.dict(Sandbox._exec.__globals__, {"launch_arguments": fake_launcher})
        self.launch_patcher.start()
        self.patcher = patch.object(asyncio, "create_subprocess_exec", side_effect=direct_exec)
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()
        self.launch_patcher.stop()
        for home in getattr(self.sandbox, "native_codex_homes", ()):
            shutil.rmtree(home, ignore_errors=True)
        for process in self.created:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        await asyncio.sleep(0.02)
        self.temporary.cleanup()
        self.loop.set_exception_handler(self.old_exception_handler)
        self.assertEqual(self.loop_errors, [], "Capture must not leave unobserved asyncio future exceptions")

    def diagnostic(self):
        return json.loads((self.root / "native-capture-diagnostics.jsonl").read_text().splitlines()[-1])

    async def execute(self, script, *, timeout=5):
        home = "/tmp/codex-home-" + uuid.uuid4().hex
        command = "env CODEX_HOME=" + home + " " + shlex.join([sys.executable, "-c", script])
        return await self.sandbox._exec(["bash", "-c", command], timeout=timeout, workdir=str(self.root))

    async def wait_file(self, path):
        async def available():
            while not path.is_file():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(available(), timeout=3)

    async def test_completed_event_survives_descendant_pipe_drain_timeout(self):
        child_file = self.root / "child.pid"
        script = (
            "import json,pathlib,subprocess,sys\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(child_file)!r}).write_text(str(child.pid))\n"
            "print(json.dumps({'type':'turn.completed','usage':{'output_tokens':1}}),flush=True)\n"
            "print('native stderr stays native',file=sys.stderr,flush=True)\n"
        )
        result = await self.execute(script)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.stdout)["type"], "turn.completed")
        self.assertEqual(result.stderr, "native stderr stays native\n")
        self.assertEqual((self.root / "codex-events.jsonl").read_text(), result.stdout)
        self.assertEqual((self.root / "codex-stderr.log").read_text(), result.stderr)
        diagnostic = self.diagnostic()
        self.assertTrue(diagnostic["drain_timed_out"])
        self.assertFalse(diagnostic["timed_out"])
        self.assertEqual(diagnostic["native_exit_code"], 0)
        self.assertEqual(diagnostic["stdout_bytes"], len(result.stdout.encode()))
        self.assertIn("SIGTERM", diagnostic["termination_signals"])
        self.assertFalse(running(int(child_file.read_text())))

    async def test_partial_output_and_actual_nonzero_exit_are_not_replaced(self):
        result = await self.execute("import sys; print('partial native output',flush=True); sys.exit(7)")
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, "partial native output\n")
        self.assertEqual(result.stderr, "")
        self.assertFalse(self.diagnostic()["drain_timed_out"])
        self.assertEqual(self.diagnostic()["returned_exit_code"], 7)

    def stubborn_script(self, ready):
        return (
            "import pathlib,signal,subprocess,sys,time\n"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])\n"
            f"pathlib.Path({str(ready)!r}).write_text(str(child.pid))\n"
            "print('actual partial output',flush=True)\n"
            "time.sleep(60)\n"
        )

    async def test_actual_timeout_returns_minus_one_and_kills_owned_group(self):
        ready = self.root / "timeout-child.pid"
        original_signal = os.killpg

        def fenced_signal(pid, number):
            self.assertEqual(self.requested_host_stops, ["budget_timeout"])
            return original_signal(pid, number)

        with patch.object(os, "killpg", side_effect=fenced_signal):
            result = await self.execute(self.stubborn_script(ready), timeout=0.3)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("actual partial output", result.stdout)
        self.assertEqual(result.stderr, "")
        diagnostic = self.diagnostic()
        self.assertTrue(diagnostic["timed_out"])
        self.assertFalse(diagnostic["cancelled"])
        self.assertEqual(self.requested_host_stops, ["budget_timeout"])
        self.assertIn("SIGKILL", diagnostic["termination_signals"])
        self.assertFalse(running(diagnostic["pid"]))
        self.assertFalse(running(int(ready.read_text())))

    async def test_cancellation_still_kills_owned_group_and_preserves_logs(self):
        ready = self.root / "cancel-child.pid"
        task = asyncio.create_task(self.execute(self.stubborn_script(ready)))
        await self.wait_file(ready)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        diagnostic = self.diagnostic()
        self.assertTrue(diagnostic["cancelled"])
        self.assertFalse(diagnostic["timed_out"])
        self.assertEqual(self.requested_host_stops, ["task_cancel"])
        self.assertIn("actual partial output", (self.root / "codex-events.jsonl").read_text())
        self.assertEqual((self.root / "codex-stderr.log").read_text(), "")
        self.assertFalse(running(diagnostic["pid"]))
        self.assertFalse(running(int(ready.read_text())))

    async def test_cancellation_during_inherited_pipe_drain_also_terminates_group(self):
        ready = self.root / "draining-child.pid"
        self.sandbox.CAPTURE_DRAIN_TIMEOUT_SECONDS = 5
        script = (
            "import pathlib,subprocess,sys\n"
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
            f"pathlib.Path({str(ready)!r}).write_text(str(child.pid))\n"
            'print(\'{"type":"turn.completed"}\',flush=True)\n'
        )
        task = asyncio.create_task(self.execute(script))
        await self.wait_file(ready)
        while self.created[0].returncode is None:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.03)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.diagnostic()["cancelled"])
        self.assertIsNone(self.diagnostic()["returned_exit_code"])
        self.assertIn("turn.completed", (self.root / "codex-events.jsonl").read_text())
        self.assertFalse(running(int(ready.read_text())))

    async def test_proven_terminal_world_bounds_cli_but_preserves_actual_exit_and_missing_completion(self):
        async def terminal():
            return {"done": True, "won": False, "steps": 50, "max_steps": 50}

        self.sandbox._observe_terminal_state = terminal
        script = (
            "import sys,time\n"
            "print('actual-native-prefix',flush=True)\n"
            "print('native-warning',file=sys.stderr,flush=True)\n"
            "time.sleep(60)\n"
        )
        result = await self.execute(script)
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "actual-native-prefix\n")
        self.assertEqual(result.stderr, "native-warning\n")
        self.assertNotIn("turn.completed", result.stdout)
        proof = self.sandbox.terminal_stop_evidence[-1]
        self.assertEqual(proof["status"], "terminal_stop")
        self.assertTrue(proof["termination_requested"])
        self.assertTrue(proof["termination_completed"])
        self.assertIsNone(proof["backend_stop_confirmed"])
        self.assertEqual(self.diagnostic()["native_exit_code"], result.exit_code)
        self.assertFalse(self.diagnostic()["timed_out"])
        self.assertFalse(running(self.diagnostic()["pid"]))

    async def test_unknown_observer_does_not_authorize_terminal_stop(self):
        async def unknown():
            raise OSError("transient public observer failure")

        self.sandbox._observe_terminal_state = unknown
        result = await self.execute("import time; print('real output',flush=True); time.sleep(.03)")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "real output\n")
        proof = self.sandbox.terminal_stop_evidence[-1]
        self.assertFalse(proof["termination_requested"])
        self.assertGreater(proof["observation_error_count"], 0)

    async def test_actual_backend_binder_api_and_task_relative_import_order(self):
        task_path = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/task.py"
        tree = ast.parse(task_path.read_text())
        imports = [
            node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "backend_terminal"
        ]
        names = {alias.name for node in imports for alias in node.names}
        self.assertTrue({"bind_episode", "wait_for_terminal_drain", "fence_host_stop"}.issubset(names))
        self.assertTrue(all(node.level == 1 for node in imports))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        bind_line = next(
            node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "bind_episode"
        )
        drain_line = next(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "wait_for_terminal_drain"
        )
        agent_line = next(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "agent"
            and node.func.attr == "run"
        )
        close_line = next(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "post"
            and any(isinstance(value, ast.Constant) and value.value == "/admin/close/" for value in ast.walk(node))
        )
        self.assertLess(bind_line, agent_line)
        self.assertLess(drain_line, close_line)
        module_stub = types.ModuleType("verl.workers.rollout.llm_server")
        module_stub.LLMServerClient = object
        path = task_path.with_name("backend_terminal.py")
        spec = importlib.util.spec_from_file_location("_task_actual_backend_api", path)
        backend = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: backend, "verl.workers.rollout.llm_server": module_stub}):
            spec.loader.exec_module(backend)
            binding = backend.bind_episode(
                self.root / "bindings",
                gateway_base_url="http://127.0.0.1:9/sessions/gateway-cpu/v1",
                environment_session_url="http://127.0.0.1:9/session/environment-cpu",
                attempt_id="attempt-cpu",
            )
            self.assertEqual(binding.gateway_session_id, "gateway-cpu")
            self.assertEqual(binding.environment_session_id, "environment-cpu")
            self.assertEqual(len(binding.binding_sha256), 64)
            backend.fence_episode(binding, {"done": True, "won": False, "steps": 50, "max_steps": 50})
            receipt = await backend.wait_for_terminal_drain(binding, timeout_seconds=0.05)
        self.assertTrue(receipt["backend_stop_confirmed"])
        self.assertTrue(receipt["fenced"])
        self.assertEqual(receipt["binding_sha256"], binding.binding_sha256)
        self.assertFalse(receipt["expected_backend_terminal_stop"])


if __name__ == "__main__":
    unittest.main()
