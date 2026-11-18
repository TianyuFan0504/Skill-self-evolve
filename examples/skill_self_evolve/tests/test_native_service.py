"""Owned synchronous service launch and exact-child lifecycle contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_native_service_test_module", ROOT / "verl/experimental/skill_harness/alfworld/native_process.py"
)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class OwnedServiceHandleTests(unittest.TestCase):
    def test_poll_reaps_only_owned_pid_and_never_signals_a_reaped_pid(self):
        process = native.OwnedServiceProcess(987654)
        with (
            patch.object(native.os, "waitpid", return_value=(process.pid, 17 << 8)) as wait,
            patch.object(native.os, "kill") as kill,
            patch.object(native.os, "killpg") as kill_group,
        ):
            self.assertEqual(process.poll(), 17)
            self.assertEqual(process.wait(timeout=0), 17)
            process.terminate()
            process.kill()
        wait.assert_called_once_with(process.pid, os.WNOHANG)
        kill.assert_not_called()
        kill_group.assert_not_called()

    def test_timeout_keeps_ownership_for_direct_term_and_group_kill_fallback(self):
        process = native.OwnedServiceProcess(987654)
        process._session_ready = True
        with (
            patch.object(native.os, "waitpid", return_value=(0, 0)),
            patch.object(native.os, "kill") as kill,
            patch.object(native.os, "killpg") as kill_group,
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                process.wait(timeout=0)
            self.assertIsNone(process.returncode)
            process.terminate()
            kill.assert_called_once_with(process.pid, signal.SIGTERM)
            kill_group.assert_not_called()
            process.kill()
            self.assertEqual(kill.call_args.args, (process.pid, signal.SIGKILL))
            kill_group.assert_called_once_with(process.pid, signal.SIGKILL)

    def test_ack_failure_closes_ready_fd_and_kills_and_reaps_exact_child(self):
        ready_read, ready_write = os.pipe()
        os.close(ready_write)
        with (
            tempfile.TemporaryFile() as output,
            patch.object(native, "_spawn_bootstrap", return_value=(987654, ready_read)),
            patch.object(native.os, "waitpid", return_value=(987654, signal.SIGKILL)) as wait,
            patch.object(native.os, "kill") as kill,
            patch.object(native.os, "getpgid", return_value=987654),
            patch.object(native.os, "killpg") as kill_group,
        ):
            with self.assertRaisesRegex(RuntimeError, "before session acknowledgement"):
                native.spawn_owned_service(["/trusted/service"], stdout=output)
        kill.assert_called_once_with(987654, signal.SIGKILL)
        kill_group.assert_called_once_with(987654, signal.SIGKILL)
        wait.assert_called_once_with(987654, os.WNOHANG)
        with self.assertRaises(OSError):
            os.fstat(ready_read)

    def test_handle_construction_failure_still_kills_and_reaps(self):
        ready_read, ready_write = os.pipe()
        os.close(ready_write)
        with (
            tempfile.TemporaryFile() as output,
            patch.object(native, "_spawn_bootstrap", return_value=(987654, ready_read)),
            patch.object(native, "OwnedServiceProcess", side_effect=RuntimeError("handle failure")),
            patch.object(native.os, "waitpid", return_value=(987654, signal.SIGKILL)) as wait,
            patch.object(native.os, "kill") as kill,
            patch.object(native.os, "getpgid", return_value=12345),
            patch.object(native.os, "killpg") as kill_group,
        ):
            with self.assertRaisesRegex(RuntimeError, "handle failure"):
                native.spawn_owned_service(["/trusted/service"], stdout=output)
        kill.assert_called_once_with(987654, signal.SIGKILL)
        kill_group.assert_not_called()
        wait.assert_called_once_with(987654, os.WNOHANG)
        with self.assertRaises(OSError):
            os.fstat(ready_read)


@unittest.skipUnless(sys.platform == "linux", "Real owned service execution requires Linux")
class LinuxOwnedServiceTests(unittest.TestCase):
    def test_explicit_stderr_file_keeps_version_stdout_exact(self):
        code = "import os; os.write(1,b'codex-cli 0.167.0\\n'); os.write(2,b'PATH warning\\n')"
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = native.spawn_owned_service(
                [sys.executable, "-I", "-S", "-c", code], stdout=output, stderr=errors
            )
            self.assertEqual(process.wait(timeout=5), 0)
            output.seek(0)
            errors.seek(0)
            self.assertEqual(output.read(), b"codex-cli 0.167.0\n")
            self.assertEqual(errors.read(), b"PATH warning\n")

    def test_logs_cwd_session_stdin_inherited_env_and_strict_fd_closure(self):
        descriptor = os.open(os.devnull, os.O_RDONLY)
        os.set_inheritable(descriptor, True)
        self.addCleanup(os.close, descriptor)
        code = """
import os,json
try:
    os.fstat(int(os.environ['LEAK_FD']))
    leaked = True
except OSError:
    leaked = False
data = {'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),'cwd':os.getcwd(),
        'stdin':os.read(0,1).hex(),'marker':os.environ['SERVICE_MARKER'],'leaked':leaked}
os.write(1, json.dumps(data).encode() + b'\\n')
os.write(2, b'service stderr\\n')
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            with path.open("ab") as output, patch.dict(
                os.environ, {"SERVICE_MARKER": "inherited", "LEAK_FD": str(descriptor)}
            ):
                process = native.spawn_owned_service(
                    [sys.executable, "-I", "-S", "-c", code], stdout=output, cwd=directory
                )
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(process.poll(), 0)
            lines = path.read_bytes().splitlines()
            data = json.loads(lines[0])
            self.assertEqual(lines[1], b"service stderr")
            self.assertEqual(data["pid"], process.pid)
            self.assertEqual(data["sid"], process.pid)
            self.assertEqual(data["pgid"], process.pid)
            self.assertEqual(data["cwd"], directory)
            self.assertEqual(data["stdin"], "")
            self.assertEqual(data["marker"], "inherited")
            self.assertFalse(data["leaked"])

    def test_real_bootstrap_failure_is_reaped_without_descriptor_leaks(self):
        original, children = os.posix_spawn, []

        def spawn(*args, **kwargs):
            pid = original(*args, **kwargs)
            children.append(pid)
            return pid

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as output:
            before = set(os.listdir("/proc/self/fd"))
            with patch.object(native.os, "posix_spawn", side_effect=spawn):
                with self.assertRaises(RuntimeError):
                    native.spawn_owned_service(
                        [sys.executable, "-I", "-S", "-c", "raise AssertionError('must not execute')"],
                        stdout=output, cwd=Path(directory) / "missing", env={},
                    )
            self.assertEqual(set(os.listdir("/proc/self/fd")), before)
            self.assertEqual(len(children), 1)
            with self.assertRaises(ChildProcessError):
                os.waitpid(children[0], os.WNOHANG)

    def test_term_timeout_falls_back_to_owned_group_kill(self):
        code = """
import os,signal,sys,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.posix_spawn(sys.executable, [sys.executable,'-I','-S','-c',
    'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'], {})
print(child, flush=True)
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            with path.open("ab") as output:
                process = native.spawn_owned_service(
                    [sys.executable, "-I", "-S", "-c", code], stdout=output, env={}
                )
            try:
                deadline = time.monotonic() + 5
                while not path.read_bytes().strip() and time.monotonic() < deadline:
                    time.sleep(0.01)
                descendant = int(path.read_bytes().strip())
                process.terminate()
                with self.assertRaises(subprocess.TimeoutExpired):
                    process.wait(timeout=0.05)
                process.kill()
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                stat_path = Path("/proc") / str(descendant) / "stat"

                def descendant_running():
                    try:
                        return stat_path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
                    except FileNotFoundError:
                        return False

                deadline = time.monotonic() + 3
                while descendant_running() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(descendant_running())
            finally:
                if process.returncode is None:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
