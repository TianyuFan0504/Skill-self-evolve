"""Complete synthetic Direct-entry stages; no model, service or live checkpoint."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

import test_compare_parallel_profile as parallel_fixture

fixture = parallel_fixture.fixture
ROOT = parallel_fixture.ROOT
read, write, sha = parallel_fixture.read, parallel_fixture.write, parallel_fixture.sha


class DirectCheckpointComparisonTests(unittest.TestCase):
    compare_pair = fixture.CompareOfficialRunsTests.compare_pair
    change = fixture.CompareOfficialRunsTests.change
    assert_incomplete = fixture.CompareOfficialRunsTests.assert_incomplete

    @classmethod
    def setUpClass(cls):
        parallel_fixture.ParallelProfileTests.setUpClass.__func__(cls)
        compare = cls.compare
        cls.source = Path(cls.spawn["driver_path"]).parents[3]
        for relative in compare.DIRECT_FILES.values():
            target = cls.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
            subprocess.check_call(["git", "-C", str(cls.source), "add", relative])
        if subprocess.run(["git", "-C", str(cls.source), "diff", "--cached", "--quiet"]).returncode:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(cls.source),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-q",
                    "-m",
                    "Direct fixture",
                ],
                check=True,
                capture_output=True,
            )
        commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()
        for identity in (cls.spawn, cls.loaded_spawn):
            identity["driver_source_commit"] = commit
            identity["composed_extensions"]["resilient_setup"]["driver_source_commit"] = commit
        cls.identity = {
            "policy": compare.DIRECT_POLICY,
            "driver_source_commit": commit,
            "actual_entry": str(cls.source / compare.DIRECT_FILES["entry"]),
            "files": {
                name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                for name, relative in compare.DIRECT_FILES.items()
            },
            "core_fingerprint_scope": "Unchanged legacy core; direct-restore implementation is identified separately",
        }
        base = cls.pair.root / "base"
        manifest = read(base / "training-manifest.json")
        manifest["source_commit"] = commit
        write(base / "training-manifest.json", manifest)
        observation = read(cls.observations["base"])
        observation.update(source_commit=commit, spawn_extension=cls.spawn)
        write(cls.observations["base"], observation)
        cls.legacy_observation = read(cls.observations["trained"])
        cls.observations["trained"].unlink()
        cls.direct_prelaunch = cls.pair.root / "trained/driver_runs/101.direct_checkpoint.json"
        cls.direct_initialized = cls.pair.root / "trained/driver_runs/102.direct_checkpoint.initialized.json"
        cls.source4 = cls.source_checkpoint_path
        cls.install_source(5)

    @classmethod
    def install_source(cls, step):
        compare, run = cls.compare, cls.pair.root / "trained"
        checkpoint = run / f"checkpoints/global_step_{step}"
        if step != 64:
            shutil.copytree(cls.source4, checkpoint)
            controller_path = checkpoint / "joint_state/controller.json"
            controller = read(controller_path)
            controller["optimizer_step"] = step
            write(controller_path, controller)
            envelope = json.loads((checkpoint / "joint_manifest.json").read_bytes())
            payload = envelope["payload"]
            payload["spec"].update(global_step=step, optimizer_step=step)
            for row in payload["ema_metadata"]:
                row["last_optimizer_step"] = step
            payload["files"]["joint_state/controller.json"] = {
                "size": controller_path.stat().st_size,
                "sha256": sha(controller_path),
            }
            envelope["manifest_id"] = "sha256:" + fixture.sha(payload)
            write(checkpoint / "joint_manifest.json", envelope)
        envelope = json.loads((checkpoint / "joint_manifest.json").read_bytes())
        provider = fixture.module(
            "_comparison_direct_fixture_provider", ROOT / compare.DIRECT_FILES["restore_provider"]
        )
        core = compare.pure_module("joint_checkpoint")
        coordinator = provider.coordinator_class(core)()
        mutable = run / "restored_state" / (f"step_{step}_" + "c" * 32)
        coordinator.restore(
            checkpoint,
            expected=core.JointCheckpointSpec(**envelope["payload"]["spec"]).compatibility(),
            destination_root=mutable,
            load_online=lambda *args: None,
            load_ema=lambda *args: None,
            load_training_state=lambda *args: None,
            load_controller=lambda *args, **kwargs: None,
            fence=nullcontext,
        )
        cls.restore_path = mutable / "DIRECT_READ_RESTORE.json"
        cls.current_source = checkpoint
        initialized = read(cls.initialized_path)
        initialized.update(cls.loaded_spawn)
        effective = initialized["effective_config_after_trainer_init"]
        effective["trainer"].update(resume_mode="resume_path", resume_from_path=str(checkpoint))
        for name, directory in (("store_root", "skill_store"), ("replay_root", "replay"), ("ledger_root", "ledger")):
            effective["skill_harness"][name] = str(mutable / directory)
        restored = {
            "checkpoint_dir": str(checkpoint),
            "manifest_id": envelope["manifest_id"],
            "spec": envelope["payload"]["spec"],
        }
        initialized["restored_checkpoint"] = restored
        write(cls.initialized_path, initialized)
        receipt = read(cls.receipt_path)
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            value = read(path)
            value.update(cls.loaded_spawn)
            value.update(
                store_root=str(mutable / "skill_store"),
                lifecycle_extension=cls.loaded_spawn,
                seal_origin="restored_complete_final_checkpoint"
                if step == 64
                else "final_checkpoint_saved_after_quiescence",
                initialized_receipt_ref={"path": str(cls.initialized_path), "sha256": sha(cls.initialized_path)},
            )
            write(path, value)
            receipt[key]["sha256"] = sha(path)
        receipt.update(cls.loaded_spawn)
        receipt.update(store_root=str(mutable / "skill_store"), seal_origin=value["seal_origin"])
        write(cls.receipt_path, receipt)
        # The prelaunch config precedes only the already enumerated constructor
        # normalizations and the new writable-state bindings.
        prelaunch = copy.deepcopy(read(run / "training-manifest.json")["config"])
        prelaunch["trainer"].update(resume_mode="resume_path", resume_from_path=str(checkpoint))
        task = prelaunch["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]["agent_runners"]["task"]
        task.update(runner_fqn=compare.SPAWN_RUNNER, max_concurrent_sessions=32)
        task["runner_kwargs"]["task_config_path"] = cls.adapters["trained"]["task_profile"]["path"]
        prelaunch["trainer"]["v1"]["sampler"]["custom_sampler"].update(
            path=str(cls.source / compare.SETUP_DRIVER), name="ResilientReplayBuffer"
        )
        prelaunch["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"].update(
            cls.adapters["trained"]["environment_variable_changes"]
        )
        observation = cls.legacy_observation
        write(
            cls.direct_prelaunch,
            {
                **cls.identity,
                "schema_version": 1,
                "kind": "prelaunch_direct_checkpoint_parallel_components",
                "observed_at_ns": 101,
                "imports": read(run / "training-manifest.json")["import_provenance"],
                "parallel_component_ref": cls.identity["files"]["parallel_component"],
                "spawn_component_identity": cls.spawn,
                "client_adapter": observation["client_adapter"],
                "environment_health": observation["environment_health"],
                "environment_config_ref": observation["environment_config_ref"],
                "config": prelaunch,
                "original_training_manifest_preserved": True,
                "not_a_post_init_or_served_model_receipt": True,
            },
        )
        write(
            cls.direct_initialized,
            {
                **cls.identity,
                "schema_version": 1,
                "kind": "initialized_direct_checkpoint_entry",
                "effective_config_after_trainer_init": effective,
                "original_training_manifest_sha256": sha(run / "training-manifest.json"),
                "restore_receipt_ref": {"path": str(cls.restore_path), "sha256": sha(cls.restore_path)},
                "spawn_component_initialized_ref": {
                    "path": str(cls.initialized_path),
                    "sha256": sha(cls.initialized_path),
                },
                "restored_checkpoint": restored,
                "restore_policy": compare.DIRECT_RESTORE_POLICY,
                "core_version_binding": restored["spec"]["versions"],
            },
        )

    @contextmanager
    def changed_restore(self, callback):
        paths = (self.restore_path, self.direct_initialized)
        before = {path: path.read_bytes() for path in paths}
        try:
            value = read(self.restore_path)
            callback(value)
            write(self.restore_path, value)
            initialized = read(self.direct_initialized)
            initialized["restore_receipt_ref"]["sha256"] = sha(self.restore_path)
            write(self.direct_initialized, initialized)
            yield
        finally:
            for path, content in before.items():
                path.write_bytes(content)

    def test_full_direct_source5_stage_matches_zero_epoch_base_inference(self):
        old = self.observations["trained"]
        stale = copy.deepcopy(self.legacy_observation)
        stale["environment_config_ref"]["sha256"] = "f" * 64
        write(old, stale)
        self.addCleanup(old.unlink)
        stale_direct = self.direct_prelaunch.with_name("90.direct_checkpoint.json")
        stale_value = read(self.direct_prelaunch)
        stale_value["environment_config_ref"]["sha256"] = "e" * 64
        stale_value["config"]["trainer"]["resume_from_path"] = str(self.source4)
        write(stale_direct, stale_value)
        self.addCleanup(stale_direct.unlink)
        stale_initialized = self.direct_initialized.with_name("91.direct_checkpoint.initialized.json")
        stale_value = read(self.direct_initialized)
        stale_value["spawn_component_initialized_ref"]["sha256"] = "e" * 64
        stale_value["restored_checkpoint"]["checkpoint_dir"] = str(self.source4)
        write(stale_initialized, stale_value)
        self.addCleanup(stale_initialized.unlink)
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        evidence = report["direct_checkpoint_restore_evidence"]
        self.assertEqual(evidence["source_checkpoint"]["spec"]["optimizer_step"], 5)
        self.assertEqual((evidence["full_validation_passes"], evidence["ema_files_copied"]), (1, 0))
        self.assertEqual(report["pooled"]["games"], 274)
        self.assertNotIn("private-fixture-token", str(report))

    def test_direct_operational_source_change_cannot_hide_behind_matching_core(self):
        path = self.source / self.compare.DIRECT_FILES["restore_provider"]
        data = path.read_bytes()
        self.addCleanup(path.write_bytes, data)
        path.write_bytes(data + b"\n# changed operational source\n")
        self.assert_incomplete(self.compare_pair())

    def test_missing_direct_initialized_cannot_substitute_old_parallel_observation(self):
        data = self.direct_initialized.read_bytes()
        self.addCleanup(self.direct_initialized.write_bytes, data)
        self.direct_initialized.unlink()
        old = self.observations["trained"]
        write(old, self.legacy_observation)
        self.addCleanup(old.unlink)
        self.assert_incomplete(self.compare_pair())

    def test_source_checkpoint_bytes_and_initialized_component_link_are_checked(self):
        path = self.current_source / "actor/model_world_size_4_rank_0.pt"
        data = path.read_bytes()
        self.addCleanup(path.write_bytes, data)
        path.write_bytes(b"altered source5")
        self.assert_incomplete(self.compare_pair())

    def test_wrong_spawn_component_reference_is_rejected(self):
        self.change(
            self.direct_initialized, lambda value: value["spawn_component_initialized_ref"].update(sha256="f" * 64)
        )
        self.assert_incomplete(self.compare_pair())

    def test_rehashed_restore_path_count_or_mutable_binding_mismatch_is_rejected(self):
        for callback in (
            lambda value: value.update(ema_source=str(Path(value["mutable_state_root"]) / "ema")),
            lambda value: value.update(full_validation_passes=2),
            lambda value: value.update(ema_files_copied=4),
            lambda value: value.update(mutable_copy_bytes=0),
            lambda value: value.update(mutable_state_root=str(self.pair.root / "trained/skill_store")),
        ):
            with self.subTest(callback=callback), self.changed_restore(callback):
                self.assert_incomplete(self.compare_pair())

    def test_direct_prelaunch_inference_drift_is_rejected(self):
        original = self.direct_prelaunch.read_bytes()
        for change in (
            lambda value: value["config"]["actor_rollout_ref"]["rollout"].update(seed=43),
            lambda value: value["environment_health"].update(environment_processes=1),
            lambda value: value["environment_config_ref"].update(sha256="f" * 64),
        ):
            try:
                value = read(self.direct_prelaunch)
                change(value)
                write(self.direct_prelaunch, value)
                self.assert_incomplete(self.compare_pair())
            finally:
                self.direct_prelaunch.write_bytes(original)

    def test_final_checkpoint_source64_requires_final_only_seal(self):
        receipt = read(self.receipt_path)
        paths = [
            self.receipt_path,
            self.initialized_path,
            self.direct_prelaunch,
            self.direct_initialized,
            *[Path(receipt[key]["path"]) for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete")],
        ]
        before = {path: path.read_bytes() for path in paths}
        old_restore, old_source = self.restore_path, self.current_source
        try:
            type(self).install_source(64)
            report = self.compare_pair()
            self.assertTrue(report["comparison_supported"], report["reasons"])
            self.assertEqual(
                report["direct_checkpoint_restore_evidence"]["source_checkpoint"]["spec"]["optimizer_step"], 64
            )
            wrong = read(self.receipt_path)
            wrong["seal_origin"] = "final_checkpoint_saved_after_quiescence"
            for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
                path = Path(wrong[key]["path"])
                value = read(path)
                value["seal_origin"] = wrong["seal_origin"]
                write(path, value)
                wrong[key]["sha256"] = sha(path)
            write(self.receipt_path, wrong)
            self.assert_incomplete(self.compare_pair())
        finally:
            for path, content in before.items():
                path.write_bytes(content)
            type(self).restore_path, type(self).current_source = old_restore, old_source


if __name__ == "__main__":
    unittest.main()
