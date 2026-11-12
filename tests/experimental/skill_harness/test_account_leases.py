"""Private lease state-machine checks; no OS accounts, signals, or models."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
MODULE = Path(
    os.environ.get("SKILL_ACCOUNT_MODULE_PATH", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py")
)


def load_accounts():
    spec = importlib.util.spec_from_file_location("_private_account_lease_cpu", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AccountLeaseTests(unittest.TestCase):
    def setUp(self):
        self.module = load_accounts()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir(mode=0o700)
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)
        self.patch = mock.patch.object(self.module, "WORK_ROOT", self.work)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.roster = [{"name": f"fake_cpu_{i}", "uid": 45000 + i, "gid": 45000 + i} for i in range(2)]
        self.live = {}
        self.signals = mock.Mock(side_effect=AssertionError("No real process may be signalled by these fixtures"))
        self.pool = self.module.AccountPool(
            self.root / "pool",
            roster=self.roster,
            processes=lambda uid: list(self.live.get(uid, [])),
            validate=lambda rows: rows,
            signal_process=self.signals,
        )
        self.pool.initialize()

    def lease(self, *, pair=False, scope=None):
        return self.pool.acquire(
            run_root=self.run,
            scope_id=scope or (uuid.uuid4().hex * 2 if pair else uuid.uuid4().hex),
            kind="pair" if pair else "ordinary",
        )

    def state(self, ref):
        return json.loads((self.pool.root / "leases" / (ref["lease_id"] + ".json")).read_text())

    def begin(self, ref, variant="ordinary", *, resources=True):
        attempt = uuid.uuid4().hex
        output = self.run / "episodes" / attempt
        output.mkdir(parents=True, mode=0o700)
        self.pool.begin_attempt(ref, attempt_id=attempt, output=output, variant=variant)
        if not resources:
            return attempt, output, None
        path = self.work / ("credit_" + ref["owner"]["scope_id"] if variant != "ordinary" else uuid.uuid4().hex)
        path.mkdir(mode=0o700)
        (path / "native.txt").write_text("Actual temporary retained bytes; no agent or model executed.\n")
        # The fake roster does not create/chown real OS users. Preserve the
        # real dev/inode while injecting its private UID/GID at this boundary.
        original_lstat = Path.lstat
        identity = path.lstat()
        assigned = SimpleNamespace(
            st_mode=identity.st_mode,
            st_dev=identity.st_dev,
            st_ino=identity.st_ino,
            st_uid=ref["uid"],
            st_gid=ref["gid"],
        )

        def assigned_lstat(candidate, *args, **kwargs):
            return assigned if candidate == path else original_lstat(candidate, *args, **kwargs)

        with mock.patch.object(Path, "lstat", assigned_lstat):
            self.pool.register_resource(ref, attempt, path, "workspace")
        proof = {
            "policy_version": self.module.POLICY_VERSION,
            "uid": ref["uid"],
            "euid": ref["uid"],
            "gid": ref["gid"],
            "resuid": [ref["uid"]] * 3,
            "resgid": [ref["gid"]] * 3,
            "groups": [],
            "no_new_privs": 1,
            "capabilities_zero": True,
            "landlock_abi": 1,
            "landlock_applied": True,
            "seccomp_guard": "abi1-truncate-x86_64-v1",
        }
        self.pool.mark_launcher(ref, attempt, proof)
        return attempt, output, path

    def receipt(self, ref, attempt, output, workspace, *, remove=True):
        info = workspace.stat()
        manifest = output / "source-manifest.json"
        manifest.write_text(json.dumps({"native.txt": {"sha256": sha(workspace / "native.txt")}}))
        archive = output / "native-files.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(workspace, arcname="workspace")
        with tarfile.open(archive, "r:gz") as stream:
            self.assertEqual(stream.extractfile("workspace/native.txt").read(), (workspace / "native.txt").read_bytes())
        data = {
            "schema_version": 1,
            "status": "archived_and_removed",
            "attempt_id": attempt,
            "account_lease_id": ref["lease_id"],
            "agent_uid": ref["uid"],
            "archive_path": str(archive),
            "archive_sha256": sha(archive),
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": sha(manifest),
            "sources": [
                {
                    "path": str(workspace),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "original_uid": ref["uid"],
                    "original_gid": ref["gid"],
                    "removed": True,
                }
            ],
        }
        if remove:
            shutil.rmtree(workspace)
        path = output / "native-files.json"
        path.write_text(json.dumps(data))
        return path

    def close(self, ref, variant="ordinary"):
        attempt, output, workspace = self.begin(ref, variant)
        receipt = self.receipt(ref, attempt, output, workspace)
        self.pool.close_attempt(ref, attempt, receipt_path=receipt, backend_drained=True, native_started=True)
        return attempt

    def test_replace_then_fsync_release_failure_cannot_make_uid_reusable(self):
        ref = self.lease(pair=True)
        self.close(ref, "with_skill")
        original = self.module._atomic

        def fail_after_publish(path, value):
            original(path, value)
            if Path(path) == self.pool.root / "state.json" and value["slots"][str(ref["uid"])] is None:
                raise OSError("simulated directory fsync failure after free-state replacement")

        with mock.patch.object(self.module, "_atomic", side_effect=fail_after_publish):
            with self.assertRaises(self.module.AccountIsolationError):
                self.pool.finish(ref, abandoned=True)
        self.assertTrue(self.pool._blocked(ref["uid"]))
        with self.assertRaises(self.module.AccountIsolationError):
            self.module.assert_no_isolation_incidents(self.run)
        next_ref = self.lease()
        self.assertNotEqual(next_ref["uid"], ref["uid"])

    def test_paused_old_drain_cannot_signal_a_new_lease_on_same_uid(self):
        pool = self.module.AccountPool(
            self.root / "single-pool",
            roster=self.roster[:1],
            processes=lambda uid: [],
            validate=lambda rows: rows,
            signal_process=self.signals,
        )
        pool.initialize()
        old = pool.acquire(run_root=self.run, scope_id="a" * 64, kind="pair")
        switched = []

        def replace_before_signal(uid):
            if switched:
                return []
            switched.append(True)
            pool.finish(old, abandoned=True)
            new = pool.acquire(run_root=self.run, scope_id=uuid.uuid4().hex)
            self.assertEqual(new["uid"], old["uid"])
            return [123456]  # Only an injected PID; never an OS process.

        pool.processes = replace_before_signal
        with self.assertRaises(self.module.AccountIsolationError):
            pool.drain(old)
        self.signals.assert_not_called()

    def test_exclusive_rotation_release_and_new_fencing(self):
        first, second = self.lease(), self.lease()
        self.assertNotEqual(first["uid"], second["uid"])
        with self.assertRaises(self.module.AccountIsolationError):
            self.lease()
        self.close(first)
        replacement = self.lease()
        self.assertEqual(replacement["uid"], first["uid"])
        self.assertNotEqual(replacement["lease_id"], first["lease_id"])
        with self.assertRaises(self.module.AccountIsolationError):
            self.pool.begin_attempt(first, attempt_id=uuid.uuid4().hex, output=self.run)
        self.signals.assert_not_called()

    def test_pair_keeps_uid_and_scope_exclusive_until_both_variants_close(self):
        group = uuid.uuid4().hex * 2
        pair = self.lease(pair=True, scope=group)
        self.close(pair, "with_skill")
        self.assertEqual(self.state(pair)["status"], "leased")
        self.assertEqual(self.pool.available(), 1)
        with self.assertRaises(self.module.AccountIsolationError):
            self.lease(pair=True, scope=group)
        other = self.lease()
        self.assertNotEqual(other["uid"], pair["uid"])
        self.close(pair, "without_skill")
        self.assertEqual(self.state(pair)["status"], "leased")
        self.pool.finish(pair)
        history = self.pool.historical_pair(self.pool.public(pair), run_root=self.run, group=group)
        self.assertEqual(history["uid"], pair["uid"])
        self.assertEqual({x["variant"] for x in history["attempts"].values()}, {"with_skill", "without_skill"})

    def test_safely_closed_first_variant_can_be_abandoned_without_faking_second(self):
        pair = self.lease(pair=True)
        self.close(pair, "with_skill")
        self.pool.finish(pair, abandoned=True)
        value = self.state(pair)
        self.assertEqual(value["status"], "released")
        self.assertTrue(value["abandoned"])
        self.assertEqual(len(value["attempts"]), 1)
        with self.assertRaises(self.module.AccountIsolationError):
            self.pool.historical_pair(self.pool.public(pair), run_root=self.run, group=pair["owner"]["scope_id"])

    def test_unstarted_first_variant_requires_explicit_empty_closure_before_abandon(self):
        pair = self.lease(pair=True)
        attempt, _, _ = self.begin(pair, "with_skill", resources=False)
        self.pool.close_empty_attempt(pair, attempt, backend_drained=True)
        self.pool.finish(pair, abandoned=True)
        self.assertEqual(self.state(pair)["status"], "released")
        self.assertTrue(self.state(pair)["attempts"][attempt]["no_native_resources_created"])

    def test_unfinished_attempt_quarantines_instead_of_releasing(self):
        pair = self.lease(pair=True)
        self.begin(pair, "with_skill")
        with self.assertRaises(self.module.AccountIsolationError):
            self.pool.finish(pair, abandoned=True)
        self.assertEqual(self.state(pair)["status"], "quarantined")
        self.assertEqual(self.pool.available(), 1)
        with self.assertRaises(self.module.AccountIsolationError):
            self.module.assert_no_isolation_incidents(self.run)

    def test_remaining_processes_quarantine_even_after_they_disappear(self):
        ref = self.lease()
        attempt, output, workspace = self.begin(ref)
        receipt = self.receipt(ref, attempt, output, workspace)
        self.live[ref["uid"]] = [987654321]  # Inventory fixture only, never a signal target.
        with self.assertRaises(self.module.AccountIsolationError):
            self.pool.close_attempt(ref, attempt, receipt_path=receipt, backend_drained=True, native_started=True)
        self.live.clear()
        self.assertEqual(self.state(ref)["status"], "quarantined")
        self.assertEqual(self.pool.available(), 1)
        self.signals.assert_not_called()

    def test_archive_hash_failure_and_unremoved_sources_quarantine(self):
        for remove, corrupt in ((True, True), (False, False)):
            with self.subTest(remove=remove, corrupt=corrupt):
                ref = self.lease()
                attempt, output, workspace = self.begin(ref)
                receipt = self.receipt(ref, attempt, output, workspace, remove=remove)
                if corrupt:
                    (output / "native-files.tar.gz").write_bytes(b"Changed after readback")
                with self.assertRaises(self.module.AccountIsolationError):
                    self.pool.close_attempt(
                        ref, attempt, receipt_path=receipt, backend_drained=True, native_started=True
                    )
                self.assertEqual(self.state(ref)["status"], "quarantined")
        self.assertEqual(self.pool.available(), 0)

    def test_pid_disappearance_and_pool_reopen_do_not_auto_free_a_retained_lease(self):
        ref = self.lease(pair=True)
        self.live[ref["uid"]] = [987654322]
        self.live.clear()
        reopened = self.module.AccountPool(
            self.pool.root,
            roster=self.roster,
            processes=lambda _uid: [],
            validate=lambda rows: rows,
            signal_process=self.signals,
        )
        reopened.initialize()
        self.assertEqual(reopened.available(), 1)
        self.assertEqual(self.state(ref)["status"], "leased")
        self.assertNotEqual(self.lease()["uid"], ref["uid"])
        self.signals.assert_not_called()

    def test_failed_incident_write_still_leaves_a_sticky_update_barrier(self):
        with mock.patch.object(self.module, "_atomic", side_effect=OSError("CPU interrupted incident write")):
            with self.assertRaises(OSError):
                self.module.record_isolation_incident(self.run, "CPU fixture")
        directory = self.run / self.module.INCIDENT_DIRECTORY
        self.assertTrue(directory.is_dir())
        self.assertFalse(list(directory.glob("*.json")))
        with self.assertRaises(self.module.AccountIsolationError):
            self.module.assert_no_isolation_incidents(self.run)


if __name__ == "__main__":
    unittest.main()
