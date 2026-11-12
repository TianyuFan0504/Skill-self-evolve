"""Full synthetic AllNative stages; no helper, account, model or service runs.

The generated curator helper digest is the explicit binary-boundary injection:
inert fixture bytes replace the pinned production binary. Published Python/C,
all receipt links, profiles, checkpoints and274 case joins are checked normally.
"""

from __future__ import annotations

import copy
import shutil
import subprocess
import unittest
from contextlib import contextmanager
from pathlib import Path

import test_compare_direct_checkpoint as direct_fixture

fixture = direct_fixture.fixture
ROOT = direct_fixture.ROOT
read, write, sha = direct_fixture.read, direct_fixture.write, direct_fixture.sha


class AllNativeComparisonTests(unittest.TestCase):
    compare_pair = fixture.CompareOfficialRunsTests.compare_pair
    change = fixture.CompareOfficialRunsTests.change
    assert_incomplete = fixture.CompareOfficialRunsTests.assert_incomplete
    install_source = classmethod(direct_fixture.DirectCheckpointComparisonTests.install_source.__func__)

    @classmethod
    def setUpClass(cls):
        direct_fixture.DirectCheckpointComparisonTests.setUpClass.__func__(cls)
        compare = cls.compare
        source_files = [
            *compare.ALL_NATIVE_FILES.values(),
            *[path for name, path in compare.ALL_NATIVE_TRANSPORT_FILES.items() if name != "helper"],
        ]
        for relative in source_files:
            path = cls.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
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
                    "AllNative fixture",
                ],
                check=True,
                capture_output=True,
            )
        commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()
        for identity in (cls.spawn, cls.loaded_spawn):
            identity["driver_source_commit"] = commit
            identity["composed_extensions"]["resilient_setup"]["driver_source_commit"] = commit
        package = (Path(cls.temporary.name) / "curator-package").resolve()
        package.mkdir()
        work = Path("/tmp").resolve() / "offline-all-native-curator-fixture"
        for role in ("base", "trained"):
            run = cls.pair.root / role
            manifest = read(run / "training-manifest.json")
            manifest["config"]["skill_harness"]["joint"]["curator_work_root"] = str(work)
            if role == "base":
                manifest["source_commit"] = commit
            write(run / "training-manifest.json", manifest)
            preflight = read(run / "preflight.json")
            preflight["curator_runtime"]["unix_user"] = "fixture_curator"
            write(run / "preflight.json", preflight)
            launch = read(run / "launch_manifest.json")
            launch["environment_paths"]["ALFWORLD_CODEX_PACKAGE_ROOT"] = str(package)
            write(run / "launch_manifest.json", launch)
        base_observation = read(cls.observations["base"])
        base_observation.update(source_commit=commit, spawn_extension=cls.spawn)
        base_observation["effective_config_before_entry"]["skill_harness"]["joint"]["curator_work_root"] = str(work)
        write(cls.observations["base"], base_observation)
        # Create a real bounded direct-read receipt for source10, with no tensor load.
        direct_fixture.DirectCheckpointComparisonTests.install_source.__func__(cls, 10)
        helper = cls.source / compare.ALL_NATIVE_TRANSPORT_FILES["helper"]
        helper.write_bytes(b"Inert reviewed curator-helper fixture; never loaded or executed")
        compare.ALL_NATIVE_HELPER_SHA256 = sha(helper)
        compare.ALL_NATIVE_SOLVER_HELPER_SHA256 = cls.loaded_spawn["composed_extensions"]["native_spawn"]["helper"][
            "sha256"
        ]
        native = {
            "policy": compare.ALL_NATIVE_TRANSPORT,
            "files": {
                name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                for name, relative in compare.ALL_NATIVE_TRANSPORT_FILES.items()
            },
            "loaded_curator_helper_sha256": sha(helper),
            "solver_component": cls.loaded_spawn["composed_extensions"]["native_spawn"],
            "loop_class": compare.ALL_NATIVE_MODULE + ".AllNativeSelectorLoop",
            "parent_fd_fork_guard": "Python os.register_at_fork exclusion; parent stdio writers close before await",
            "curator_profiles": [
                {
                    "user": "fixture_curator",
                    "uid": 999,
                    "gid": 999,
                    "work_root": str(work),
                    "package_root": str(package),
                }
            ],
        }
        cls.all_identity = {
            "policy": compare.ALL_NATIVE_POLICY,
            "actual_entry": str(cls.source / compare.ALL_NATIVE_FILES["entry"]),
            "driver_source_commit": commit,
            "files": {
                name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                for name, relative in compare.ALL_NATIVE_FILES.items()
            },
            "all_role_native_spawn": native,
            "solver_spawn_component_identity": cls.loaded_spawn,
            "core_fingerprint_scope": (
                "Unchanged core; actual all-role spawn and direct-read providers recorded separately"
            ),
        }
        cls.all_prelaunch = cls.direct_prelaunch.with_name("201.all_native_spawn.json")
        cls.all_initialized = cls.direct_initialized.with_name("202.all_native_spawn.initialized.json")
        old = read(cls.direct_initialized)
        effective = old["effective_config_after_trainer_init"]
        effective["trainer"]["save_freq"] = 10
        effective["skill_harness"]["joint"]["curator_work_root"] = str(work)
        value = {
            "schema_version": 1,
            "kind": "initialized_all_native_spawn_entry",
            **cls.all_identity,
            "imports": read(cls.initialized_path)["imports"],
            "effective_config_after_trainer_init": effective,
            "original_training_manifest_sha256": old["original_training_manifest_sha256"],
            "counterfactual_loop_class": compare.ALL_NATIVE_MODULE + ".AllNativeSelectorLoop",
            "event_loop_policy_class": compare.ALL_NATIVE_MODULE + ".AllNativePolicy",
            "registered_curator_scope": "Actual NativeCuratorRuntime account, work root, package and clean environment",
            "primary_worker_scope": "Unchanged SpawnSafeRunner installs the solver-only component in episode workers",
            **{
                key: old[key]
                for key in ("restore_receipt_ref", "restored_checkpoint", "restore_policy", "core_version_binding")
            },
            "component_receipt_scope": (
                "Frozen final-fit body identity is a component; lifecycle extension uses this receipt"
            ),
        }
        write(cls.all_initialized, value)
        prelaunch = read(cls.direct_prelaunch)
        config = prelaunch["config"]
        config["trainer"]["save_freq"] = 10
        before_identity = copy.deepcopy(cls.all_identity)
        before_identity["all_role_native_spawn"]["curator_profiles"] = []
        write(
            cls.all_prelaunch,
            {
                "schema_version": 1,
                "kind": "prelaunch_all_native_spawn_parallel_components",
                **before_identity,
                "observed_at_ns": 201,
                "imports": value["imports"],
                "parallel_component_ref": cls.all_identity["files"]["parallel_component"],
                **{key: prelaunch[key] for key in ("client_adapter", "environment_health", "environment_config_ref")},
                "config": config,
                "original_training_manifest_preserved": True,
                "not_a_post_init_or_served_model_receipt": True,
            },
        )
        receipt = read(cls.receipt_path)
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            row = read(path)
            row.update(
                lifecycle_extension=cls.all_identity,
                initialized_receipt_ref={"path": str(cls.all_initialized), "sha256": sha(cls.all_initialized)},
            )
            write(path, row)
            receipt[key]["sha256"] = sha(path)
        write(cls.receipt_path, receipt)
        # The actual new mode neither invokes nor invents these old initializers.
        cls.initialized_path.unlink()
        cls.direct_initialized.unlink()
        cls.direct_prelaunch.unlink()

    @contextmanager
    def mutated_actual(self, callback):
        receipt = read(self.receipt_path)
        paths = [
            self.receipt_path,
            self.all_initialized,
            *[Path(receipt[key]["path"]) for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete")],
        ]
        before = {path: path.read_bytes() for path in paths}
        try:
            value = read(self.all_initialized)
            callback(value)
            write(self.all_initialized, value)
            extension = {key: value[key] for key in self.compare.ALL_NATIVE_IDENTITY_FIELDS}
            for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
                path = Path(receipt[key]["path"])
                row = read(path)
                row.update(
                    lifecycle_extension=extension,
                    initialized_receipt_ref={"path": str(self.all_initialized), "sha256": sha(self.all_initialized)},
                )
                write(path, row)
                receipt[key]["sha256"] = sha(path)
            write(self.receipt_path, receipt)
            yield
        finally:
            for path, content in before.items():
                path.write_bytes(content)

    def test_source10_actual_all_native_chain_matches_unchanged_baseline_inference(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        evidence = report["all_native_spawn_evidence"]
        self.assertEqual(evidence["initial_source_step"], 10)
        self.assertEqual(evidence["actual_learner_loop"], self.compare.ALL_NATIVE_MODULE + ".AllNativeSelectorLoop")
        self.assertEqual(evidence["checkpoint_save_frequency"]["initialized"], 10)
        self.assertEqual(report["direct_checkpoint_restore_evidence"]["full_validation_passes"], 1)
        self.assertEqual(report["direct_checkpoint_restore_evidence"]["ema_files_copied"], 0)
        self.assertEqual(report["pooled"]["games"], 274)
        self.assertFalse(self.initialized_path.exists())
        self.assertNotIn("private-fixture-token", str(report))

    def test_wrong_curator_loaded_helper_or_solver_component_is_rejected(self):
        changes = [
            lambda value: value["all_role_native_spawn"].update(loaded_curator_helper_sha256="f" * 64),
            lambda value: value["all_role_native_spawn"]["solver_component"]["helper"].update(loaded_sha256="f" * 64),
        ]
        for change in changes:
            with self.subTest(change=change), self.mutated_actual(change):
                self.assert_incomplete(self.compare_pair())

    def test_old_learner_loop_or_wrong_new_policy_cannot_be_relabelled_as_all_native(self):
        for change in (
            lambda value: value.update(
                counterfactual_loop_class=self.compare.SPAWN_MODULE + ".NativeSpawnSelectorLoop"
            ),
            lambda value: value.update(event_loop_policy_class=self.compare.SPAWN_MODULE + ".NativeSpawnPolicy"),
        ):
            with self.subTest(change=change), self.mutated_actual(change):
                self.assert_incomplete(self.compare_pair())

    def test_wrong_registered_curator_profile_is_rejected_with_consistent_links(self):
        for update in (
            {"user": "other_curator"},
            {"uid": 0},
            {"work_root": "/tmp/unregistered-curator"},
            {"package_root": str(self.source)},
        ):
            with (
                self.subTest(update=update),
                self.mutated_actual(
                    lambda value, update=update: value["all_role_native_spawn"]["curator_profiles"][0].update(update)
                ),
            ):
                self.assert_incomplete(self.compare_pair())

    def test_published_transport_source_and_source10_checkpoint_bytes_are_bound(self):
        for path in (
            self.source / self.compare.ALL_NATIVE_TRANSPORT_FILES["c"],
            self.current_source / "actor/model_world_size_4_rank_0.pt",
        ):
            data = path.read_bytes()
            try:
                path.write_bytes(data + b"changed source bytes")
                self.assert_incomplete(self.compare_pair())
            finally:
                path.write_bytes(data)

    def test_lifecycle_cannot_link_a_foreign_initialized_hash(self):
        self.change(self.receipt_path, lambda value: value["checkpoint_seal"].update(sha256="f" * 64))
        self.assert_incomplete(self.compare_pair())


class FreshAllNativeComparisonTests(unittest.TestCase):
    compare_pair = AllNativeComparisonTests.compare_pair
    assert_incomplete = AllNativeComparisonTests.assert_incomplete
    mutated_actual = AllNativeComparisonTests.mutated_actual
    install_source = classmethod(direct_fixture.DirectCheckpointComparisonTests.install_source.__func__)

    @classmethod
    def setUpClass(cls):
        AllNativeComparisonTests.setUpClass.__func__(cls)
        run = cls.pair.root / "trained"
        value = read(cls.all_initialized)
        old_mutable = Path(value["restore_receipt_ref"]["path"]).parent
        for source, target in (
            ("skill_store", "skill_store"),
            ("replay", "curate_replay"),
            ("ledger", "credit_ledger"),
        ):
            shutil.copytree(old_mutable / source, run / target)
        manifest = read(run / "training-manifest.json")
        manifest["config"]["trainer"].update(resume_mode="auto", resume_from_path=None)
        write(run / "training-manifest.json", manifest)
        effective = value["effective_config_after_trainer_init"]
        effective["trainer"].update(resume_mode="auto", resume_from_path=None)
        for name, relative in (
            ("store_root", "skill_store"),
            ("replay_root", "curate_replay"),
            ("ledger_root", "credit_ledger"),
        ):
            effective["skill_harness"][name] = str(run / relative)
        value.update(
            restored_checkpoint=None,
            restore_receipt_ref=None,
            restore_policy=None,
            core_version_binding=None,
            original_training_manifest_sha256=sha(run / "training-manifest.json"),
        )
        write(cls.all_initialized, value)
        observation = read(cls.all_prelaunch)
        observation["config"]["trainer"].update(resume_mode="auto", resume_from_path=None)
        write(cls.all_prelaunch, observation)
        receipt = read(cls.receipt_path)
        receipt.update(
            store_root=str(run / "skill_store"), original_training_manifest_sha256=sha(run / "training-manifest.json")
        )
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            row = read(path)
            row.update(
                store_root=receipt["store_root"],
                original_training_manifest_sha256=receipt["original_training_manifest_sha256"],
                initialized_receipt_ref={"path": str(cls.all_initialized), "sha256": sha(cls.all_initialized)},
            )
            write(path, row)
            receipt[key]["sha256"] = sha(path)
        write(cls.receipt_path, receipt)

    def test_fresh_auto_source0_has_no_restore_claim_and_seals_original_store(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertEqual(report["all_native_spawn_evidence"]["initial_source_step"], 0)
        self.assertEqual(report["all_native_spawn_evidence"]["initial_hf_fingerprint"], self.pair.fingerprint)
        self.assertIsNone(report["all_native_spawn_evidence"]["restore_receipt_ref"])
        self.assertIsNone(report["direct_checkpoint_restore_evidence"])
        self.assertEqual(
            report["lifecycle_evidence"]["trained"]["store_root"], str(self.pair.root / "trained/skill_store")
        )

    def test_fresh_source0_rejects_forged_or_contradictory_restore_claims(self):
        for change in (
            lambda value: value.update(restored_checkpoint={"checkpoint_dir": str(self.current_source)}),
            lambda value: value.update(restore_policy=self.compare.DIRECT_RESTORE_POLICY),
            lambda value: value.update(
                restore_receipt_ref={"path": str(self.restore_path), "sha256": sha(self.restore_path)}
            ),
            lambda value: value.update(core_version_binding={"code": "forged"}),
            lambda value: value["effective_config_after_trainer_init"]["trainer"].update(
                resume_from_path=str(self.current_source)
            ),
        ):
            with self.subTest(change=change), self.mutated_actual(change):
                self.assert_incomplete(self.compare_pair())


if __name__ == "__main__":
    unittest.main()
