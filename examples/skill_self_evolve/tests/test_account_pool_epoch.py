"""Real frozen lease registry fixtures; no host accounts, model or environment."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


epoch = load("_account_pool_epoch_cpu", HERE / "scripts/account_pool_epoch.py")
accounts = load("_epoch_frozen_accounts_cpu", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py")
native = load("_epoch_native_spawn_cpu", HERE / "scripts/native_posix_spawn.py")


def file_hashes(root):
    return {str(path.relative_to(root)): (path.stat().st_mode, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*") if path.is_file()}


class EpochTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.new, self.old, self.run = [self.root / name for name in ("epoch3", "old_v2", "run")]
        self.old.mkdir(mode=0o700)
        (self.old / "state.json").write_text('{"slots":{"20071":"protected"},"do_not_reset":true}')
        self.run.mkdir(mode=0o700)
        self.manifest = self.root / "epoch3.manifest.json"
        patches = [patch.object(epoch, "EPOCH_POOL_ROOT", self.new), patch.object(epoch, "OLD_POOL_ROOT", self.old),
                   patch.object(epoch, "_accounts", return_value=accounts),
                   patch.object(epoch, "process_identity", return_value={"pid": 123, "start_ticks": 456}),
                   patch.object(epoch, "_active", None), patch.object(epoch, "_original_archive", None),
                   patch.object(accounts, "validate_system_roster", return_value=epoch.roster()),
                   patch.object(accounts, "uid_processes", return_value=[]),
                   patch.object(native, "_account_pool_epoch", None),
                   patch.object(native, "_account_uid_gid_pairs", None)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        ref = epoch.initialize_epoch(self.manifest)
        self.env = {epoch.MANIFEST_ENV: ref["path"], epoch.MANIFEST_SHA_ENV: ref["sha256"],
                    "SKILL_RUN_DIR": str(self.run)}
        environment = patch.dict(os.environ, self.env)
        environment.start()
        self.addCleanup(environment.stop)
        self.task = types.ModuleType("verl.experimental.skill_harness.alfworld.task")
        self.probes = types.ModuleType("verl.experimental.skill_harness.probes")
        self.sampler = types.ModuleType("verl.experimental.skill_harness.group_sampler")
        self.task.AccountPool = self.probes.AccountPool = accounts.AccountPool
        self.sampler.local_archive_failed_groups = accounts.local_archive_failed_groups
        namespace = types.ModuleType("skill_harness_examples")
        namespace.__path__ = []
        namespace.account_pool_epoch = epoch
        namespace.native_posix_spawn = native
        self.modules = {self.task.__name__: self.task, self.probes.__name__: self.probes,
                        self.sampler.__name__: self.sampler, "skill_harness_examples": namespace,
                        "skill_harness_examples.account_pool_epoch": epoch,
                        "skill_harness_examples.native_posix_spawn": native}

    def test_import_is_inert_without_manifest_or_accounts(self):
        with patch.object(importlib, "import_module", side_effect=AssertionError("No runtime imports")), \
                patch.object(Path, "mkdir", side_effect=AssertionError("No creation")):
            imported = load("_inert_epoch_cpu", HERE / "scripts/account_pool_epoch.py")
        self.assertIsNone(imported._active)

    def test_readonly_preflight_never_changes_either_registry(self):
        before = file_hashes(self.root)
        observation = epoch.readonly_preflight()
        self.assertEqual(observation["available"], 128)
        self.assertTrue(observation["read_only"])
        self.assertEqual(file_hashes(self.root), before)

    def test_hash_roster_and_permissions_fail_closed(self):
        original = self.manifest.read_bytes()
        self.manifest.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            epoch.read_manifest()
        self.manifest.write_bytes(original)
        self.manifest.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private"):
            epoch.read_manifest()
        self.manifest.chmod(0o600)
        value = json.loads(original)
        value["roster"][0]["uid"] = 20071
        self.manifest.write_text(json.dumps(value))
        os.environ[epoch.MANIFEST_SHA_ENV] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            epoch.read_manifest()

    def test_initialize_refuses_existing_even_partial_registry(self):
        before = file_hashes(self.root)
        with self.assertRaisesRegex(ValueError, "cannot reset"):
            epoch.initialize_epoch(self.root / "another.json")
        self.assertEqual(file_hashes(self.root), before)

    def test_manifest_symlink_rejected(self):
        linked = self.root / "linked.json"
        linked.symlink_to(self.manifest)
        env = {**self.env, epoch.MANIFEST_ENV: str(linked)}
        with self.assertRaisesRegex(ValueError, "symlinks"):
            epoch.read_manifest(environ=env)

    def test_clean_capacity_excludes_quarantine_leases_and_live_uids(self):
        state_path = self.new / "state.json"
        state = json.loads(state_path.read_text())
        for uid in range(30000, 30089):
            target = self.new / "quarantined-uids" / str(uid)
            target.mkdir(parents=True)
        for uid in range(30089, 30120):
            state["slots"][str(uid)] = "retained_lease"
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "8 available"):
            epoch.readonly_preflight()
        with patch.object(accounts, "uid_processes", side_effect=lambda uid: [777] if uid == 30120 else []):
            report = epoch.readonly_preflight(minimum_available=0)
        self.assertEqual(report["available"], 7)
        self.assertEqual(report["live_without_lease"], [30120])

    def test_process_aliases_reuse_real_pair_lease_and_never_touch_old_registry(self):
        old = file_hashes(self.old)
        with patch.dict(sys.modules, self.modules):
            ref = epoch.install_process(role="ray_worker")
            probe_pool, task_pool = self.probes.AccountPool(), self.task.AccountPool()
            lease = probe_pool.acquire(run_root=self.run, scope_id="c" * 64, kind="pair")
            self.assertEqual(task_pool.account(lease).pw_uid, 30000)
            self.assertEqual(probe_pool.root, task_pool.root)
            self.assertIsNot(self.task.AccountPool, accounts.AccountPool)
            self.assertTrue(native._account_uid_gid_allowed(30127, 30127))
            self.assertFalse(native._account_uid_gid_allowed(20071, 20071))
            markers = [json.loads(Path(item["path"]).read_text()) for item in epoch.activation_refs()]
        self.assertEqual({item["activation_role"] for item in markers},
                         {"ray_worker", "task_account_pool", "probe_account_pool"})
        self.assertEqual(json.loads(Path(ref["path"]).read_text())["native_spawn_account_pool"], epoch.identity())
        self.assertEqual(file_hashes(self.old), old)

    def test_conflicting_alias_rejects_before_any_alias_changes(self):
        previous = self.task.AccountPool
        self.probes.AccountPool = lambda: None
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(ValueError, "Another provider"):
            epoch.install_process(role="driver")
        self.assertIs(self.task.AccountPool, previous)
        self.assertIsNone(epoch._active)

    def test_hook_runs_original_bootstrap_before_alias_installation(self):
        events = []
        bootstrap = types.ModuleType("verl.experimental.skill_harness.bootstrap")
        bootstrap.configure_runtime = lambda **kw: events.append(("bootstrap", kw))
        with patch.dict(sys.modules, {**self.modules, bootstrap.__name__: bootstrap}), \
                patch.object(epoch, "install_process", side_effect=lambda **kw: events.append(("select", kw))):
            epoch.setup_worker()
        self.assertEqual(events, [("bootstrap", {"register_tasks": True}), ("select", {"role": "ray_worker"})])

    def test_model_free_acceptance_requires_hook_and_constructs_pools_without_leases(self):
        before = (self.new / "state.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "did not execute"):
            epoch.inspect_worker_bindings()
        with patch.dict(sys.modules, self.modules):
            epoch.install_process(role="ray_worker")
            report = epoch.inspect_worker_bindings()
        self.assertFalse(report["lease_acquired"])
        self.assertEqual(report["task_pool_root"], str(self.new))
        self.assertEqual(report["probe_pool_root"], str(self.new))
        self.assertEqual((self.new / "state.json").read_bytes(), before)

    def test_learner_rejects_overwritten_hook_even_after_successful_worker_installation(self):
        runtime = {"worker_process_setup_hook": epoch.WORKER_SETUP_HOOK, "env_vars": dict(self.env)}
        omega = types.ModuleType("omegaconf")
        omega.OmegaConf = types.SimpleNamespace(select=lambda *args, **kwargs: runtime)
        config = types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(self.run)))
        with patch.dict(sys.modules, {**self.modules, "omegaconf": omega}):
            epoch.install_process(role="ray_worker")
            self.assertEqual(epoch.require_worker_activation(config), epoch.identity())
            runtime["worker_process_setup_hook"] = "verl.experimental.skill_harness.bootstrap.configure_worker"
            with self.assertRaisesRegex(ValueError, "missing the selected"):
                epoch.require_worker_activation(config)

    def test_native_default_is_historical_and_epoch_env_without_hook_rejects(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(native._account_uid_gid_allowed(20000, 20000))
            self.assertFalse(native._account_uid_gid_allowed(30000, 30000))
        with self.assertRaisesRegex(ValueError, "missing its worker setup"):
            native._account_uid_gid_allowed(20000, 20000)

    def test_actual_native_matcher_uses_registered_roster_and_retains_fd_argv_guards(self):
        argv = ["/usr/bin/setpriv", "--reuid=30001", "--regid=30001", "--clear-groups",
                "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs",
                "/usr/bin/python3", native.ACCOUNTS, "isolate-exec", "--uid", "30001", "--gid", "30001",
                "--ready-fd", "91", "--read-roots", "[]", "--write-roots", "[]", "--", "/bin/true"]
        options = dict(stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                       stderr=asyncio.subprocess.PIPE, cwd="/tmp", env={"ONLY": "exact"},
                       start_new_session=True, close_fds=True, pass_fds=(91,))
        with patch.dict(sys.modules, self.modules):
            epoch.install_process(role="ray_worker")
            rewritten, ready = native._native_arguments(argv[0], argv[1:], options)
            self.assertEqual((ready, rewritten[16], argv[16]), (91, "3", "91"))
            for changed in ({**options, "pass_fds": (91, 92)}, {**options, "close_fds": False}):
                with self.assertRaises(ValueError):
                    native._native_arguments(argv[0], argv[1:], changed)
            argv[1], argv[2], argv[12], argv[14] = "--reuid=20071", "--regid=20071", "20071", "20071"
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                native._native_arguments(argv[0], argv[1:], options)

    def test_archive_reader_unions_old_and_new_training_dispositions_only(self):
        for root, uid, label, role in ((self.old, 20000, "old", "solver"),
                                      (self.new, 30000, "new", "solver"),
                                      (self.new, 30001, "probe", "credit")):
            target = root / "quarantined-uids" / str(uid) / "local-archive.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"kind": "local_archive_data_invalid", "training_data_valid": False,
                                         "uid_drained": True, "backend_drained": True,
                                         "account": {"owner": {"run_root": str(self.run)}},
                                         "identity": {"dataset_phase": "train", "role": role,
                                                      "rollout_uid": label}}))
        with patch.dict(sys.modules, self.modules):
            epoch.install_process(role="driver")
            self.assertEqual(self.sampler.local_archive_failed_groups(self.run), {"old", "new"})

    def test_plan_checks_conflicts_and_never_executes_or_modifies_users(self):
        with patch.object(epoch.pwd, "getpwall", return_value=[]), \
                patch.object(epoch.grp, "getgrall", return_value=[]):
            result = epoch.provisioning_plan()
        self.assertEqual(len(result["commands"]), 256)
        self.assertEqual(result["commands"][0], ["/usr/sbin/groupadd", "--gid", "30000", "alfworld_ep3_000"])
        self.assertEqual(result["fenced_pool"]["protected_uid"], 20071)
        user = types.SimpleNamespace(pw_name="other", pw_uid=30000, pw_gid=999)
        with patch.object(epoch.pwd, "getpwall", return_value=[user]), \
                patch.object(epoch.grp, "getgrall", return_value=[]), \
                patch.object(accounts, "validate_system_roster", side_effect=ValueError("identity conflict")), \
                self.assertRaisesRegex(ValueError, "identity conflict"):
            epoch.provisioning_plan()

    def test_dangling_primary_gid_conflict_rejects_before_provisioning(self):
        user = types.SimpleNamespace(pw_name="unrelated", pw_uid=25000, pw_gid=30000)
        with patch.object(epoch.pwd, "getpwall", return_value=[user]), \
                self.assertRaisesRegex(ValueError, "primary GID"):
            epoch.provisioning_plan()


if __name__ == "__main__":
    unittest.main()
