# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise the actual Task capture class with real CPU subprocess groups."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def load_capture_class():
    """Load its source without importing the optional Uni-Agent/model stack."""
    path = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/task.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "AlfworldUserSandbox")
    namespace = {
        "asyncio": asyncio, "json": json, "os": os, "signal": signal, "time": time, "Path": Path,
        "LocalSandbox": object, "ExecResult": ExecResult,
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
        self.sandbox.CAPTURE_DRAIN_TIMEOUT_SECONDS = 0.08
        self.sandbox.TERMINATION_TIMEOUT_SECONDS = 0.08
        self.created = []
        self.loop_errors = []
        self.loop = asyncio.get_running_loop()
        self.old_exception_handler = self.loop.get_exception_handler()
        self.loop.set_exception_handler(lambda _loop, context: self.loop_errors.append(context))
        original = asyncio.create_subprocess_exec

        async def direct_exec(*argv, **kwargs):
            self.assertEqual(argv[:4], ("/usr/sbin/runuser", "-u", "alfworld_eval", "--"))
            # The capture implementation is unchanged; only the test-user switch
            # is removed, allowing the same real child test on macOS and Linux.
            process = await original(*argv[4:], **kwargs)
            self.created.append(process)
            return process

        self.patcher = patch.object(asyncio, "create_subprocess_exec", side_effect=direct_exec)
        self.patcher.start()

    async def asyncTearDown(self):
        self.patcher.stop()
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
        return await self.sandbox._exec([sys.executable, "-c", script], timeout=timeout, workdir=str(self.root))

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
        result = await self.execute(self.stubborn_script(ready), timeout=0.3)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("actual partial output", result.stdout)
        self.assertEqual(result.stderr, "")
        diagnostic = self.diagnostic()
        self.assertTrue(diagnostic["timed_out"])
        self.assertFalse(diagnostic["cancelled"])
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
            "print('{\"type\":\"turn.completed\"}',flush=True)\n"
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


if __name__ == "__main__":
    unittest.main()
