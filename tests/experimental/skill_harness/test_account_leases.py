"""CPU checks of explicit provider configuration and locally fenced UID leases."""

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from _native_runtime_test_support import accounts, linux


class AccountProviderTest(unittest.TestCase):
    def test_missing_configuration_has_no_implicit_host_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(accounts.TaskSetupError):
                accounts.AccountPool()

    def test_explicit_factory_and_environment_config(self):
        provider = object.__new__(linux.LinuxAccountPool)
        # The facade validates the complete lifecycle, without host account mutation.
        accounts.register_runtime_provider("unit_backend", lambda **kwargs: provider)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "runtime.json"
            config.write_text(json.dumps({"account_pool": {"factory": "unit_backend"}}))
            with patch.dict(os.environ, {"ALFWORLD_RUNTIME_CONFIG": str(config)}, clear=True):
                pool = accounts.AccountPool()
            self.assertIs(pool.provider, provider)

    def test_invalid_provider_is_setup_error(self):
        accounts.register_runtime_provider("incomplete_unit_backend", lambda: object())
        with self.assertRaises(accounts.TaskSetupError):
            accounts.AccountPool({"factory": "incomplete_unit_backend"})


class LinuxLeaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.pool = linux.LinuxAccountPool(
            root=self.root / "registry",
            work_root=self.root / "work",
            roster=[{"name": "unit_runtime_account", "uid": 12345, "gid": 12345}],
            validate=lambda rows: rows,
            processes=lambda uid: [],
            signal_process=lambda *args: None,
        )
        self.pool.initialize()

    def test_empty_attempt_releases_exclusive_lease(self):
        ref = self.pool.acquire(run_root=self.run, scope_id="one")
        with self.assertRaises(linux.AccountIsolationError):
            self.pool.acquire(run_root=self.run, scope_id="two")
        attempt = uuid.uuid4().hex
        self.pool.begin_attempt(ref, attempt_id=attempt, output=self.run)
        self.pool.close_empty_attempt(ref, attempt, backend_drained=True)
        self.assertEqual(self.pool.available(), 1)
        replacement = self.pool.acquire(run_root=self.run, scope_id="two")
        self.assertNotEqual(ref["lease_id"], replacement["lease_id"])

    def test_cleanup_closure_requires_no_archive(self):
        ref = self.pool.acquire(run_root=self.run, scope_id="one")
        attempt = uuid.uuid4().hex
        self.pool.begin_attempt(ref, attempt_id=attempt, output=self.run)
        receipt = self.run / "native-cleanup.json"
        receipt.write_text(
            json.dumps(
                {"status": "cleaned", "agent_uid": ref["uid"], "account_lease_id": ref["lease_id"], "sources": []}
            )
        )
        closed = self.pool.close_attempt(ref, attempt, receipt_path=receipt, backend_drained=True, native_started=False)
        self.assertEqual(closed["cleanup_receipt"], str(receipt))
        self.assertEqual(self.pool.available(), 1)

    def test_unconfirmed_drain_withholds_only_owned_uid(self):
        ref = self.pool.acquire(run_root=self.run, scope_id="one")
        attempt = uuid.uuid4().hex
        self.pool.begin_attempt(ref, attempt_id=attempt, output=self.run)
        with self.assertRaises(linux.AccountIsolationError):
            self.pool.close_attempt(
                ref, attempt, receipt_path=self.run / "missing", backend_drained=False, native_started=False
            )
        self.assertEqual(self.pool.available(), 0)
        self.assertEqual(list(self.run.iterdir()), [])

    def test_explicit_roster_rejects_root_identity(self):
        with self.assertRaises(linux.AccountIsolationError):
            linux.LinuxAccountPool(
                root=self.root / "bad", work_root=self.root / "work", roster=[{"name": "root", "uid": 0, "gid": 0}]
            )


if __name__ == "__main__":
    unittest.main()
