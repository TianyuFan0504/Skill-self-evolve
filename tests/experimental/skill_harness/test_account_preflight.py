"""Production preflight orchestration with only identity/process boundary mocked."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AccountPreflightTests(unittest.TestCase):
    def test_version_closes_only_after_valid_archive_and_failure_quarantines(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                self._version_case(exit_code)

    def _version_case(self, exit_code):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, work = root / "run", root / "work"
            run.mkdir()
            work.mkdir()
            accounts = source_module(
                "_preflight_accounts", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py"
            )
            files = source_module("_preflight_files", ROOT / "verl/experimental/skill_harness/alfworld/native_files.py")
            preflight = source_module("_preflight_recipe", ROOT / "examples/skill_self_evolve/scripts/preflight.py")
            resources, transitions = [], []
            uid, gid = os.getuid(), os.getgid()
            lease = {
                "lease_id": "test-only",
                "uid": uid,
                "gid": gid,
                "name": "cpu",
                "owner": {"run_root": str(run)},
            }
            proof = {
                "policy_version": accounts.POLICY_VERSION,
                "uid": uid,
                "euid": uid,
                "gid": gid,
                "resuid": [uid] * 3,
                "resgid": [gid] * 3,
                "groups": [],
                "no_new_privs": 1,
                "capabilities_zero": True,
                "landlock_applied": True,
                "landlock_abi": 1,
                "seccomp_guard": accounts.SECCOMP_GUARD,
            }

            def close(ref, attempt, *, receipt_path, **kwargs):
                receipt = json.loads(Path(receipt_path).read_text())
                self.assertEqual(receipt["status"], "archived_and_removed")
                self.assertEqual({row["path"] for row in receipt["sources"]}, set(map(str, resources)))
                self.assertTrue(all(not path.exists() for path in resources))
                transitions.append("close")
                return {"status": "attempt_closed"}

            pool = types.SimpleNamespace(
                roster=[None] * 128,
                available=lambda: 64,
                acquire=lambda **kwargs: lease,
                begin_attempt=mock.Mock(),
                account=lambda ref: types.SimpleNamespace(pw_uid=uid, pw_gid=gid, pw_name="cpu"),
                register_resource=lambda ref, attempt, path, kind: resources.append(path),
                mark_launcher=mock.Mock(),
                drain=lambda ref: transitions.append("drain"),
                close_attempt=close,
                quarantine=lambda *args, **kwargs: transitions.append("quarantine"),
            )

            def launch(account, argv, *, ready_fd, **kwargs):
                os.write(ready_fd, json.dumps(proof).encode())
                return argv, {}, None

            original_archive = files.archive_native_files

            def archive(*args, **kwargs):
                transitions.append("archive")
                return original_archive(*args, workspace_root=work, **kwargs)

            modules = {
                name: types.ModuleType(name)
                for name in (
                    "verl",
                    "verl.experimental",
                    "verl.experimental.skill_harness",
                    "verl.experimental.skill_harness.alfworld",
                )
            }
            modules.update(
                {
                    "verl.experimental.skill_harness.alfworld.accounts": accounts,
                    "verl.experimental.skill_harness.alfworld.native_files": files,
                }
            )
            child = types.SimpleNamespace(
                returncode=exit_code, communicate=lambda **kwargs: ("codex fixture\n", "failure")
            )
            try:
                with (
                    mock.patch.dict(sys.modules, modules),
                    mock.patch.object(accounts, "AccountPool", return_value=pool),
                    mock.patch.object(accounts, "WORK_ROOT", work),
                    mock.patch.object(accounts, "landlock_abi", return_value=1),
                    mock.patch.object(accounts, "launch_arguments", side_effect=launch),
                    mock.patch.object(files, "archive_native_files", side_effect=archive),
                    mock.patch.object(preflight.subprocess, "Popen", return_value=child),
                    mock.patch.dict(
                        os.environ,
                        {"CODEX_NATIVE_BIN": "/usr/bin/true", "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": "/unused"},
                    ),
                ):
                    if exit_code:
                        with self.assertRaises(accounts.AccountIsolationError):
                            preflight.restricted_native_preflight(run, "/trusted/mock-native")
                        self.assertEqual(transitions, ["quarantine"])
                        self.assertTrue(all(path.exists() for path in resources))
                    else:
                        result = preflight.restricted_native_preflight(run, "/trusted/mock-native")
                        self.assertEqual(transitions, ["drain", "archive", "close"])
                        self.assertEqual(result["closure"]["status"], "attempt_closed")
            finally:
                for path in resources:
                    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
