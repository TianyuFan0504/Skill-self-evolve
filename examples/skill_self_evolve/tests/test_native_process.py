"""Native spawn ownership, pipe flow control, and cancellation contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_native_process_test_module", ROOT / "verl/experimental/skill_harness/alfworld/native_process.py"
)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class MockLaunchTests(unittest.IsolatedAsyncioTestCase):
    """Exercise failure ownership on macOS too, without creating any process."""

    async def asyncSetUp(self):
        self.pid, self.exited, self.status = 987654, False, 0
        self.acknowledgement = b"\x01"
        self.owned_fds, self.child_fds, self.processes = [], [], []
        self.spawn_arguments = None

        def pipe():
            descriptors = os.pipe()
            self.owned_fds.extend(descriptors)
            return descriptors

        def spawn(executable, arguments, environment, **kwargs):
            self.spawn_arguments = executable, arguments, environment, kwargs
            for action in kwargs["file_actions"]:
                if action[0] == os.POSIX_SPAWN_DUP2 and action[2] in (1, 2):
                    self.child_fds.append(os.dup(action[1]))
                if action[0] == os.POSIX_SPAWN_DUP2 and action[2] == 0:
                    self.owned_fds.append(action[1])
            if self.acknowledgement:
                os.write(int(arguments[7]), self.acknowledgement)
            return self.pid

        def killpg(pid, number):
            self.assertEqual((pid, number), (self.pid, signal.SIGKILL))
            self.exited, self.status = True, signal.SIGKILL

        patches = [
            patch.object(native, "_require_linux"),
            patch.object(native, "_pipe", side_effect=pipe),
            patch.object(native.os, "posix_spawn", side_effect=spawn),
            patch.object(
                native.os, "waitpid", side_effect=lambda pid, flags: (pid, self.status) if self.exited else (0, 0)
            ),
            patch.object(native.os, "killpg", side_effect=killpg),
            patch.object(native.os, "kill", side_effect=killpg),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    async def asyncTearDown(self):
        self.exited = True
        for process in self.processes:
            await process.wait()
            for transport in process._transports:
                transport.close()
        for descriptor in self.child_fds:
            os.close(descriptor)

    async def launch(self):
        process = await native.spawn_native_process(["/trusted/launcher", "argument"], env={"TEST": "value"}, cwd=ROOT)
        self.processes.append(process)
        return process

    def assert_owned_closed(self):
        for descriptor in self.owned_fds:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    async def test_wait_timeout_preserves_exit_proof_without_waiting_for_pipe_eof(self):
        process = await self.launch()
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(process.wait(), 0.01)
        self.exited, self.status = True, 23 << 8
        self.assertEqual(await asyncio.wait_for(process.wait(), 1), 23)
        self.assertEqual(process.returncode, 23)
        self.assertFalse(process.stdout.at_eof())
        executable, arguments, environment, options = self.spawn_arguments
        self.assertTrue(Path(executable).is_absolute())
        self.assertEqual(arguments[1:3], ["-I", "-S"])
        self.assertTrue(Path(arguments[3]).is_absolute())
        self.assertEqual(environment, {"TEST": "value"})
        self.assertNotIn("setsid", options)
        self.assertEqual(options["setsigmask"], ())
        for transport in process._transports:
            transport.close()
        self.assert_owned_closed()

    async def test_attachment_failure_kills_and_reaps_without_leaking_owned_fds(self):
        original, calls = native._attach_reader, 0

        async def attach(descriptor):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("attachment failure")
            return await original(descriptor)

        with patch.object(native, "_attach_reader", side_effect=attach):
            with self.assertRaisesRegex(RuntimeError, "attachment failure"):
                await native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT)
        self.assertTrue(self.exited)
        self.assert_owned_closed()

    async def test_cancellation_during_blocked_attachment_kills_and_reaps(self):
        entered = asyncio.Event()
        original, calls = native._attach_reader, 0

        async def blocked(descriptor):
            nonlocal calls
            calls += 1
            if calls == 1:
                return await original(descriptor)
            entered.set()
            await asyncio.Future()

        with patch.object(native, "_attach_reader", side_effect=blocked):
            task = asyncio.create_task(native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        self.assertTrue(self.exited)
        self.assert_owned_closed()

    async def test_cancel_all_tasks_cannot_cancel_owned_exit_polling(self):
        process = await self.launch()
        asyncio.create_task(process.wait())
        await asyncio.sleep(0)
        cancelled = asyncio.all_tasks() - {asyncio.current_task()}
        self.assertTrue(cancelled)
        for task in cancelled:
            task.cancel()
        await asyncio.gather(*cancelled, return_exceptions=True)
        self.assertFalse(process._exit.cancelled())
        self.exited, self.status = True, 0
        self.assertEqual(await asyncio.wait_for(process.wait(), 1), 0)

    async def test_repeated_cancellation_still_waits_for_owned_reap(self):
        entered, killed = asyncio.Event(), asyncio.Event()

        async def blocked(descriptor):
            entered.set()
            await asyncio.Future()

        with (
            patch.object(native, "_attach_reader", side_effect=blocked),
            patch.object(native.os, "kill", side_effect=lambda pid, number: killed.set()),
            patch.object(native.os, "getpgid", return_value=12345),
            patch.object(native.os, "killpg") as kill_group,
        ):
            task = asyncio.create_task(native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT))
            await entered.wait()
            task.cancel()
            await killed.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.exited, self.status = True, signal.SIGKILL
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            kill_group.assert_not_called()
        self.assert_owned_closed()

    async def test_bootstrap_eof_before_acknowledgement_is_reaped(self):
        self.acknowledgement = b""
        with self.assertRaisesRegex(RuntimeError, "before session acknowledgement"):
            await native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT)
        self.assertTrue(self.exited)
        self.assert_owned_closed()

    async def test_ack_completion_cannot_swallow_owner_cancellation(self):
        original, calls = native._attach_reader, 0

        class RacingAcknowledgement:
            async def readexactly(self, count):
                asyncio.get_running_loop().call_soon(owner.cancel, "ACK owner cancellation")
                return b"\x01"

        async def attach(descriptor):
            nonlocal calls
            calls += 1
            reader, transport = await original(descriptor)
            return (RacingAcknowledgement() if calls == 3 else reader), transport

        with patch.object(native, "_attach_reader", side_effect=attach):
            owner = asyncio.create_task(native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT))
            with self.assertRaises(asyncio.CancelledError) as cancelled:
                await owner
        self.assertEqual(str(cancelled.exception), "ACK owner cancellation")
        self.assertTrue(self.exited)
        self.assert_owned_closed()

    async def test_failure_constructing_process_wrapper_still_reaps_exact_child(self):
        with patch.object(native, "NativeProcess", side_effect=RuntimeError("wrapper failure")):
            with self.assertRaisesRegex(RuntimeError, "wrapper failure"):
                await native.spawn_native_process(["/trusted/launcher"], env={}, cwd=ROOT)
        self.assertTrue(self.exited)
        self.assert_owned_closed()


@unittest.skipUnless(sys.platform == "linux", "Real native execution requires Linux")
class LinuxLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, code, *, environment=None, pass_fds=(), cwd=None):
        return await native.spawn_native_process(
            [sys.executable, "-I", "-S", "-c", code],
            env=dict(os.environ) if environment is None else environment,
            cwd=ROOT if cwd is None else cwd,
            pass_fds=pass_fds,
        )

    async def test_bootstrap_closes_late_inheritable_fd_and_preserves_pass_fd(self):
        ready_read, ready_write = os.pipe()
        self.addCleanup(os.close, ready_read)
        self.addCleanup(os.close, ready_write)
        self.assertFalse(os.get_inheritable(ready_write))
        original_spawn, late = os.posix_spawn, []

        def spawn(executable, arguments, environment, **kwargs):
            descriptor = os.open(os.devnull, os.O_RDONLY)
            os.set_inheritable(descriptor, True)
            late.append(descriptor)
            environment["LEAK_FD"] = str(descriptor)
            return original_spawn(executable, arguments, environment, **kwargs)

        code = """
import os,json
try:
    os.fstat(int(os.environ['LEAK_FD']))
    leaked = True
except OSError:
    leaked = False
os.write(int(os.environ['PASS_FD']), b'ready')
print(json.dumps({'leaked':leaked,'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),'cwd':os.getcwd(),'stdin':os.read(0,1).hex()}))
"""
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(native.os, "posix_spawn", side_effect=spawn):
                process = await self.execute(
                    code, environment={"PASS_FD": str(ready_write)}, pass_fds=(ready_write,), cwd=directory
                )
                output, errors, status = await asyncio.gather(
                    process.stdout.read(), process.stderr.read(), process.wait()
                )
                self.assertEqual((errors, status), (b"", 0))
                data = json.loads(output)
                self.assertFalse(data["leaked"])
                self.assertEqual(data["pid"], data["sid"])
                self.assertEqual(data["pid"], data["pgid"])
                self.assertEqual(data["cwd"], directory)
                self.assertEqual(data["stdin"], "")
                self.assertEqual(os.read(ready_read, 5), b"ready")
                self.assertFalse(os.get_inheritable(ready_write))
        finally:
            for descriptor in late:
                os.close(descriptor)

    async def test_large_output_resumes_after_stream_reader_backpressure(self):
        process = await self.execute("import os; os.write(1, b'x' * 4000000); os.write(2, b'error')")
        output, errors, status = await asyncio.wait_for(
            asyncio.gather(process.stdout.read(), process.stderr.read(), process.wait()), 10
        )
        self.assertEqual((len(output), errors, status), (4000000, b"error", 0))

    async def test_exec_failure_is_reported_and_direct_child_is_reaped(self):
        process = await native.spawn_native_process(["/no/such/native/executable"], env={}, cwd=ROOT)
        output, errors, status = await asyncio.gather(process.stdout.read(), process.stderr.read(), process.wait())
        self.assertEqual(output, b"")
        self.assertNotEqual(status, 0)
        self.assertIn(b"FileNotFoundError", errors)
        with self.assertRaises(ChildProcessError):
            os.waitpid(process.pid, os.WNOHANG)

    async def test_direct_child_exit_does_not_wait_for_descendant_pipe_eof(self):
        code = """
import os,sys
os.posix_spawn(sys.executable, [sys.executable,'-I','-S','-c','import time; time.sleep(1)'], {})
print('direct child exited')
"""
        process = await self.execute(code)
        self.assertEqual(await asyncio.wait_for(process.wait(), 0.5), 0)
        self.assertFalse(process.stdout.at_eof())
        self.assertEqual(await asyncio.wait_for(process.stdout.read(), 3), b"direct child exited\n")
        self.assertEqual(await process.stderr.read(), b"")

    async def test_native_signal_defaults_survive_python_bootstrap(self):
        process = await native.spawn_native_process(["/bin/sh", "-c", "kill -PIPE $$; exit 99"], env={}, cwd=ROOT)
        output, errors, status = await asyncio.gather(process.stdout.read(), process.stderr.read(), process.wait())
        self.assertEqual((output, errors, status), (b"", b"", -signal.SIGPIPE))

    async def test_many_launches_with_196_live_background_threads(self):
        stop = threading.Event()
        threads = [threading.Thread(target=stop.wait, daemon=True) for _ in range(196)]
        for thread in threads:
            thread.start()
        try:
            async def one():
                process = await self.execute("print('spawned')")
                output, errors, status = await asyncio.gather(
                    process.stdout.read(), process.stderr.read(), process.wait()
                )
                self.assertEqual((output, errors, status), (b"spawned\n", b"", 0))

            await asyncio.wait_for(asyncio.gather(*(one() for _ in range(32))), 20)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
