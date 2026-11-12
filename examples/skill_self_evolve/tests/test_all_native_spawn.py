"""Strict all-role spawn routing checks; no process or account is created."""

from __future__ import annotations

import asyncio
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from skill_harness_examples import all_native_spawn as native  # noqa: E402


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir(mode=0o711)
        self.cwd = self.work / ("a" * 64)
        self.cwd.mkdir(mode=0o700)
        (self.cwd / ".home").mkdir(mode=0o700)
        self.package = self.root / "package"
        self.package.mkdir()
        self.account = types.SimpleNamespace(pw_name="custom_curator", pw_uid=32001, pw_gid=32002)
        self.profile = native.CuratorProfile("custom_curator", 32001, 32002, str(self.work), str(self.package))
        self.args = ("-u", "custom_curator", "--", "/bin/bash", "-c", "printf '%s' 'unchanged host script'")
        self.options = {
            "stdin": asyncio.subprocess.DEVNULL, "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE, "start_new_session": True, "cwd": self.cwd,
            "env": {**native._ENVIRONMENT, "HOME": str(self.cwd / ".home"),
                    "CURATOR_CODEX_HOME": str(self.cwd / ".home"),
                    "CODEX_MANAGED_PACKAGE_ROOT": str(self.package)},
        }

    def match(self, args=None, options=None, profiles=None):
        with patch.object(native.pwd, "getpwnam", return_value=self.account), \
                patch.object(Path, "lstat", return_value=types.SimpleNamespace(
                    st_mode=stat.S_IFDIR | 0o700, st_uid=32001, st_gid=32002)):
            return native.curator_arguments("/usr/sbin/runuser", args or self.args,
                                            self.options if options is None else options,
                                            {self.profile.user: self.profile} if profiles is None else profiles)

    def test_custom_registered_nonroot_account_preserves_exact_argv_and_script(self):
        self.assertEqual(self.match(), ["/usr/sbin/runuser", *self.args])

    def test_foreign_or_unregistered_account_is_not_silently_forked(self):
        with self.assertRaises(ValueError):
            self.match(profiles={})
        with self.assertRaises(ValueError):
            self.match(args=("-u", "other", *self.args[2:]))

    def test_changed_uid_and_root_are_refused(self):
        for uid in (0, 32003):
            self.account.pw_uid = uid
            with self.assertRaises(ValueError):
                self.match()

    def test_unexpected_env_capability_or_home_is_refused(self):
        for change in ({"ALFWORLD_SESSION_URL": "http://127.0.0.1/private"}, {"HOME": "/tmp/foreign"}):
            options = {**self.options, "env": {**self.options["env"], **change}}
            with self.assertRaises(ValueError):
                self.match(options=options)

    def test_nonstdio_fds_session_and_unknown_options_are_refused(self):
        for change in ({"pass_fds": (99,)}, {"close_fds": False}, {"start_new_session": False}, {"shell": True}):
            with self.assertRaises(ValueError):
                self.match(options={**self.options, **change})

    def test_default_and_explicit_close_fds_shapes_both_match(self):
        self.assertEqual(self.match(), self.match(options={**self.options, "close_fds": True, "pass_fds": ()}))

    def test_foreign_cwd_collection_and_home_permissions_are_refused(self):
        for cwd in ("relative", self.root / ("a" * 64), self.work / "not-a-collection"):
            with self.assertRaises(ValueError):
                self.match(options={**self.options, "cwd": cwd})
        with patch.object(native.pwd, "getpwnam", return_value=self.account), \
                patch.object(Path, "lstat", return_value=types.SimpleNamespace(
                    st_mode=stat.S_IFDIR | 0o755, st_uid=32001, st_gid=32002)):
            with self.assertRaises(ValueError):
                native.curator_arguments(
                    "/usr/sbin/runuser", self.args, self.options, {self.profile.user: self.profile}
                )

    def test_unknown_commands_keep_standard_routing(self):
        self.assertIsNone(native.curator_arguments("/bin/true", (), {}, {}))
        self.assertIsNone(native.curator_arguments("/usr/sbin/runuser", ("-u", "someone"), {"env": {}}, {}))

    def test_runtime_registration_supports_custom_identity_and_rejects_solver_alias(self):
        runtime = types.SimpleNamespace(unix_user="custom_curator", uid=32001, gid=32002,
                                        work_root=str(self.work), package_root=str(self.package))

        def account(name):
            if name == "custom_curator":
                return self.account
            raise KeyError(name)

        with patch.object(native.pwd, "getpwnam", side_effect=account):
            self.assertEqual(native.CuratorProfile.from_runtime(runtime), self.profile)
        with patch.object(native.pwd, "getpwnam", return_value=self.account):
            with self.assertRaisesRegex(ValueError, "solver"):
                native.CuratorProfile.from_runtime(runtime)

    def test_curator_loop_dispatches_three_stdio_fds_without_rewriting_script(self):
        loop = native.AllNativeSelectorLoop(object(), {self.profile.user: self.profile})
        self.addCleanup(loop.close)
        spawned = AsyncMock(return_value=("transport", "protocol"))
        loop._spawn_native = spawned
        with patch.object(native, "curator_arguments", return_value=["/usr/sbin/runuser", *self.args]):
            value = loop.run_until_complete(loop.subprocess_exec(object, "/usr/sbin/runuser", *self.args,
                                                                  **self.options))
        self.assertEqual(value, ("transport", "protocol"))
        self.assertEqual(spawned.call_args.kwargs["ready_fd"], -1)
        self.assertEqual(spawned.call_args.args[1], ["/usr/sbin/runuser", *self.args])

    def test_solver_loop_still_uses_original_ready_fd_and_matcher(self):
        loop = native.AllNativeSelectorLoop(object(), {})
        self.addCleanup(loop.close)
        spawned = AsyncMock(return_value=("transport", "protocol"))
        loop._spawn_native = spawned
        argv = ["/usr/bin/setpriv", "--reuid=20001", "--regid=20001", "--clear-groups",
                "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs",
                "/usr/bin/python3", native.solver.ACCOUNTS, "isolate-exec", "--uid", "20001", "--gid", "20001",
                "--ready-fd", "91", "--read-roots", "[]", "--write-roots", "[]", "--", "/bin/true"]
        options = {"stdin": asyncio.subprocess.DEVNULL, "stdout": asyncio.subprocess.PIPE,
                   "stderr": asyncio.subprocess.PIPE, "cwd": "/tmp", "env": {}, "start_new_session": True,
                   "close_fds": True, "pass_fds": (91,)}
        loop.run_until_complete(loop.subprocess_exec(object, argv[0], *argv[1:], **options))
        self.assertEqual(spawned.call_args.kwargs["ready_fd"], 91)
        self.assertEqual(spawned.call_args.args[1][16], "3")


if __name__ == "__main__":
    unittest.main()
