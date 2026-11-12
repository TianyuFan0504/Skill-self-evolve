"""CPU-only launcher planning/BPF evaluation; no identity or kernel changes."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_account_launcher_cpu", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py"
)
accounts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accounts)


def evaluate(program, number, *, flags=0, arch=0xC000003E):
    """Interpret the emitted classic-BPF subset against seccomp_data words."""
    words, accumulator, pc = {0: number, 4: arch, 24: flags, 32: flags}, 0, 0
    for _ in range(100):
        code, yes, no, value = program[pc]
        if code == 0x20:
            accumulator = words[value]
        elif code == 0x15:
            pc += yes if accumulator == value else no
        elif code == 0x35:
            pc += yes if accumulator >= value else no
        elif code == 0x54:
            accumulator &= value
        elif code == 0x05:
            pc += value
        elif code == 0x06:
            return value
        else:
            raise AssertionError("Unexpected classic-BPF instruction")
        pc += 1
    raise AssertionError("BPF did not terminate")


class AccountLauncherTests(unittest.TestCase):
    def test_truncate_holes_blocked_and_sqlite_write_modes_preserved(self):
        program = accounts._seccomp_program()
        denied, allowed = 0x50000 | errno.EPERM, 0x7FFF0000
        for number in (2, 257):
            for flags in (os.O_TRUNC, os.O_TRUNC | os.O_CLOEXEC, os.O_TRUNC | 3 | os.O_CLOEXEC):
                with self.subTest(number=number, flags=flags):
                    self.assertEqual(evaluate(program, number, flags=flags), denied)
            for flags in (os.O_RDONLY, os.O_WRONLY | os.O_TRUNC, os.O_RDWR | os.O_TRUNC | os.O_CLOEXEC):
                self.assertEqual(evaluate(program, number, flags=flags), allowed)
        self.assertEqual(evaluate(program, 76), denied)
        self.assertEqual(evaluate(program, 77), allowed)  # SQLite ftruncate writable own FD.

    def test_pointer_flags_uring_and_x32_have_no_unchecked_path(self):
        program = accounts._seccomp_program()
        for number in (437, 425, 426, 427, 0x40000002, 0x40000101):
            self.assertEqual(evaluate(program, number), 0x50000 | errno.ENOSYS)
        self.assertEqual(evaluate(program, 2, arch=0x40000003), 0x50000 | errno.EPERM)

    def test_private_tmp_is_reused_by_inode_and_pair_visible_identity_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            home.mkdir(mode=0o700)
            account = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name="private_cpu")
            with mock.patch.dict(os.environ, {}, clear=True):
                argv, env, identity = accounts.launch_arguments(
                    account, ["/bin/echo", "ok"], home=home, cwd=root, ready_fd=17
                )
                _, again, second = accounts.launch_arguments(
                    account, ["/bin/echo", "ok"], home=home, cwd=root, ready_fd=18, expected_tmp=identity
                )
            self.assertEqual(identity, second)
            self.assertEqual(env, again)
            self.assertEqual(env["HOME"], "/nonexistent")
            self.assertEqual(env["TMPDIR"], str(root / ".native-tmp"))
            self.assertIn("--clear-groups", argv)
            self.assertIn("--no-new-privs", argv)
            self.assertIn("--bounding-set=-all", argv)
            read = json.loads(argv[argv.index("--read-roots") + 1])
            write = json.loads(argv[argv.index("--write-roots") + 1])
            self.assertNotIn("/tmp", read + write)
            self.assertNotIn("/proc", read + write)
            self.assertIn(str(home), write)
            temporary = root / ".native-tmp"
            temporary.rmdir()
            temporary.symlink_to(home, target_is_directory=True)
            with self.assertRaises(accounts.AccountIsolationError):
                accounts.launch_arguments(
                    account, ["/bin/echo"], home=home, cwd=root, ready_fd=17, expected_tmp=identity
                )

    def test_readiness_requires_saved_ids_and_actual_seccomp(self):
        proof = {
            "policy_version": accounts.POLICY_VERSION,
            "uid": 19000,
            "euid": 19000,
            "gid": 19000,
            "groups": [],
            "resuid": [19000] * 3,
            "resgid": [19000] * 3,
            "no_new_privs": 1,
            "capabilities_zero": True,
            "landlock_applied": True,
            "landlock_abi": 1,
            "seccomp_guard": accounts.SECCOMP_GUARD,
        }
        self.assertEqual(accounts.decode_launcher_proof(json.dumps(proof), uid=19000, gid=19000), proof)
        for changes in ({"resuid": [19000, 19000, 0]}, {"seccomp_guard": None}, {"groups": [0]}):
            with self.assertRaises(accounts.AccountIsolationError):
                accounts.decode_launcher_proof(json.dumps({**proof, **changes}), uid=19000, gid=19000)

    def test_pidfd_signal_rechecks_uid_without_killing_numeric_pid(self):
        with (
            mock.patch.object(os, "pidfd_open", create=True, return_value=91) as opened,
            mock.patch.object(accounts.signal, "pidfd_send_signal", create=True) as sent,
            mock.patch.object(os, "close") as closed,
            mock.patch.object(Path, "read_text", return_value="Name:\ttest\nUid:\t19000 19000 19000 19000\n"),
        ):
            accounts._signal_uid_process(12345, 19000, accounts.signal.SIGTERM)
            opened.assert_called_once_with(12345, 0)
            sent.assert_called_once_with(91, accounts.signal.SIGTERM, None, 0)
            closed.assert_called_once_with(91)
        with (
            mock.patch.object(os, "pidfd_open", create=True, return_value=92),
            mock.patch.object(accounts.signal, "pidfd_send_signal", create=True) as sent,
            mock.patch.object(os, "close"),
            mock.patch.object(Path, "read_text", return_value="Uid:\t20000 20000 20000 20000\n"),
        ):
            with self.assertRaises(accounts.AccountIsolationError):
                accounts._signal_uid_process(12345, 19000, accounts.signal.SIGTERM)
            sent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
