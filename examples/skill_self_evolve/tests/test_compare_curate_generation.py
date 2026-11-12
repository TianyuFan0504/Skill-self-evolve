"""Synthetic source10-to64 generation namespaces; no models or services run.

As in the existing AllNative fixture, helper bytes are inert binary-boundary
injections. Published provider sources, activation/state links, restored
checkpoint metadata and all274 original-case joins are verified normally.
"""

from __future__ import annotations

import copy
import json
import subprocess
import unittest
from contextlib import contextmanager
from pathlib import Path

import test_compare_all_native_spawn as native_fixture

ROOT = native_fixture.ROOT
read, write, sha = native_fixture.read, native_fixture.write, native_fixture.sha


class CurateGenerationComparisonTests(unittest.TestCase):
    compare_pair = native_fixture.AllNativeComparisonTests.compare_pair
    assert_incomplete = native_fixture.AllNativeComparisonTests.assert_incomplete
    install_source = classmethod(native_fixture.AllNativeComparisonTests.install_source.__func__)

    @classmethod
    def setUpClass(cls):
        native_fixture.AllNativeComparisonTests.setUpClass.__func__(cls)
        compare = cls.compare
        for relative in {*compare.CURATE_GENERATION_FILES.values(), *compare.CURATE_GENERATION_PROVIDER_FILES.values()}:
            path = cls.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
            subprocess.check_call(["git", "-C", str(cls.source), "add", relative])
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
                "Curate generation fixture",
            ],
            check=True,
            capture_output=True,
        )
        commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()
        for component in (cls.spawn, cls.loaded_spawn):
            component["driver_source_commit"] = commit
            component["composed_extensions"]["resilient_setup"]["driver_source_commit"] = commit
        cls.all_identity.update(
            policy=compare.CURATE_GENERATION_POLICY,
            actual_entry=str(cls.source / compare.CURATE_GENERATION_FILES["entry"]),
            driver_source_commit=commit,
            files={
                name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                for name, relative in compare.CURATE_GENERATION_FILES.items()
            },
            curate_generation_extension={
                "policy": compare.CURATE_GENERATION_NAMESPACE_POLICY,
                "controller_class": compare.CURATE_GENERATION_CONTROLLER,
                "files": {
                    name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                    for name, relative in compare.CURATE_GENERATION_PROVIDER_FILES.items()
                },
            },
        )
        base = cls.pair.root / "base"
        manifest = read(base / "training-manifest.json")
        manifest["source_commit"] = commit
        write(base / "training-manifest.json", manifest)
        observation = read(cls.observations["base"])
        observation.update(source_commit=commit, spawn_extension=cls.spawn)
        write(cls.observations["base"], observation)
        run = cls.pair.root / "trained"
        cls.namespace = "a" * 64
        directory = run / "controller/generation_epochs"
        cls.namespace_path = directory / (cls.namespace + ".json")
        cls.activation_path = directory / (cls.namespace + ".activated.json")
        value = read(cls.all_initialized)
        restored = value["restored_checkpoint"]
        write(
            cls.namespace_path,
            {
                "schema_version": 1,
                "kind": "prepared_curate_generation_namespace",
                "policy": compare.CURATE_GENERATION_NAMESPACE_POLICY,
                "namespace": cls.namespace,
                "created_at_ns": 1000,
                "owner": {"pid": 1234, "start_ticks": 5678},
                "source_checkpoint": {
                    "checkpoint_dir": restored["checkpoint_dir"],
                    "manifest_id": restored["manifest_id"],
                    "optimizer_step": 10,
                },
                "provider_ref": cls.all_identity["curate_generation_extension"]["files"]["provider"],
                "scope": compare.CURATE_GENERATION_SCOPE,
            },
        )
        namespace_ref = {"path": str(cls.namespace_path), "sha256": sha(cls.namespace_path)}
        write(
            cls.activation_path,
            {
                "schema_version": 1,
                "kind": "activated_curate_generation_controller",
                "namespace": cls.namespace,
                "namespace_ref": namespace_ref,
                "controller_class": compare.CURATE_GENERATION_CONTROLLER,
                "optimizer_step": 10,
            },
        )
        value.update(cls.all_identity, kind="initialized_curate_generation_entry", curate_generation_ref=namespace_ref)
        new_initialized = cls.all_initialized.with_name("302.curate_generation.initialized.json")
        write(new_initialized, value)
        cls.all_initialized.unlink()
        cls.all_initialized = new_initialized
        before = read(cls.all_prelaunch)
        identity = copy.deepcopy(cls.all_identity)
        identity["all_role_native_spawn"]["curator_profiles"] = []
        before.update(identity, kind="prelaunch_curate_generation_parallel_components")
        new_prelaunch = cls.all_prelaunch.with_name("301.curate_generation.json")
        write(new_prelaunch, before)
        cls.all_prelaunch.unlink()
        cls.all_prelaunch = new_prelaunch
        cls.live_controller = run / "controller/state.json"
        cls.final_controller = Path(cls.checkpoint.checkpoint_dir) / "joint_state/controller.json"
        cls.final_manifest = cls.final_controller.parent.parent / "joint_manifest.json"
        for path in (cls.live_controller, cls.final_controller):
            state = read(path)
            state["curate_generation_activation_ref"] = {
                "path": str(cls.activation_path),
                "sha256": sha(cls.activation_path),
            }
            episode = str(run / "episodes/fresh_source/attempt_fresh")
            group = "cohort_" + native_fixture.fixture.sha(["update11", episode, cls.namespace])
            state["source_queue"] = {
                "jobs": {
                    "fixture_source": {
                        "source_step": 11,
                        "update_id": "update11",
                        "episode_output_dir": episode,
                        "curate_generation_namespace": cls.namespace,
                        "curator_jobs": [{"candidate_group_id": group}],
                    }
                }
            }
            write(path, state, wrapped=True)
        cls.relink()

    @classmethod
    def relink(cls):
        envelope = json.loads(cls.final_manifest.read_text())
        envelope["payload"]["files"]["joint_state/controller.json"] = {
            "sha256": sha(cls.final_controller),
            "size": cls.final_controller.stat().st_size,
        }
        envelope["manifest_id"] = "sha256:" + native_fixture.fixture.sha(envelope["payload"])
        write(cls.final_manifest, envelope)
        receipt = read(cls.receipt_path)
        receipt.update(cls.loaded_spawn)
        receipt["manifest_id"] = envelope["manifest_id"]
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            row = read(path)
            row.update(cls.loaded_spawn)
            row.update(
                manifest_id=envelope["manifest_id"],
                lifecycle_extension=cls.all_identity,
                initialized_receipt_ref={"path": str(cls.all_initialized), "sha256": sha(cls.all_initialized)},
            )
            write(path, row)
            receipt[key]["sha256"] = sha(path)
        write(cls.receipt_path, receipt)

    @contextmanager
    def mutated(self):
        receipt = read(self.receipt_path)
        paths = [
            self.namespace_path,
            self.activation_path,
            self.all_initialized,
            self.all_prelaunch,
            self.live_controller,
            self.final_controller,
            self.final_manifest,
            self.receipt_path,
            *[Path(receipt[key]["path"]) for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete")],
        ]
        original = {path: path.read_bytes() for path in paths}
        identity = copy.deepcopy(self.all_identity)
        try:
            yield
        finally:
            for path, content in original.items():
                path.write_bytes(content)
            self.all_identity.clear()
            self.all_identity.update(identity)

    def relink_activation(self):
        ref = {"path": str(self.activation_path), "sha256": sha(self.activation_path)}
        for path in (self.live_controller, self.final_controller):
            state = read(path)
            state["curate_generation_activation_ref"] = ref
            write(path, state, wrapped=True)
        self.relink()

    def assert_rejected_for(self, reason):
        report = self.compare_pair()
        self.assert_incomplete(report)
        self.assertTrue(any(reason in item for item in report["reasons"]), report["reasons"])

    def test_source10_generation_entry_matches_original_base_and_full274(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        evidence = report["all_native_spawn_evidence"]
        self.assertEqual(evidence["initial_source_step"], 10)
        self.assertEqual(evidence["curate_generation"]["namespace"], self.namespace)
        self.assertEqual(evidence["curate_generation"]["controller_class"], self.compare.CURATE_GENERATION_CONTROLLER)
        self.assertEqual(report["pooled"]["games"], 274)
        self.assertFalse((self.pair.root / "base/controller/generation_epochs").exists())

    def test_prepared_namespace_is_not_activation_proof(self):
        with self.mutated():
            self.activation_path.unlink()
            self.assert_rejected_for(".activated.json")

    def test_final_only_source64_is_rejected_before_activation(self):
        with self.mutated():
            # This mode refuses source64 before it can publish an activation.
            # The retained complete checkpoint is real fixture metadata; no
            # activation for that unsupported source is manufactured here.
            checkpoint = json.loads(self.final_manifest.read_text())
            restored = {
                "checkpoint_dir": str(self.final_manifest.parent),
                "manifest_id": checkpoint["manifest_id"],
                "spec": checkpoint["payload"]["spec"],
            }
            prepared = read(self.namespace_path)
            prepared["source_checkpoint"] = {
                "checkpoint_dir": restored["checkpoint_dir"],
                "manifest_id": restored["manifest_id"],
                "optimizer_step": 64,
            }
            write(self.namespace_path, prepared)
            initialized = read(self.all_initialized)
            initialized.update(
                restored_checkpoint=restored,
                curate_generation_ref={"path": str(self.namespace_path), "sha256": sha(self.namespace_path)},
            )
            write(self.all_initialized, initialized)
            self.activation_path.unlink()
            self.relink()
            self.assert_rejected_for("source optimizer step below the final target")

    def test_legacy_checkpoint_actions_keep_their_original_namespace(self):
        with self.mutated():
            for path in (self.live_controller, self.final_controller):
                value = read(path)
                value["source_queue"]["jobs"]["legacy_source"] = {
                    "source_step": 10,
                    "update_id": "old_update10",
                    "episode_output_dir": "/preserved/old/source",
                    "curate_generation_namespace": "b" * 64,
                    "curator_jobs": [{"candidate_group_id": "original_preserved_group"}],
                }
                write(path, value, wrapped=True)
            self.relink()
            report = self.compare_pair()
            self.assertTrue(report["comparison_supported"], report["reasons"])

    def test_rehashed_wrong_activation_namespace_controller_or_source_clock_rejects(self):
        for updates in ({"namespace": "b" * 64}, {"controller_class": "forged.Controller"}, {"optimizer_step": 11}):
            with self.subTest(updates=updates), self.mutated():
                value = read(self.activation_path)
                value.update(updates)
                write(self.activation_path, value)
                self.relink_activation()
                self.assert_rejected_for("activation namespace/controller/source clock differs")

    def test_rehashed_wrong_prepared_source_or_namespace_rejects(self):
        for updates in ({"namespace": "b" * 64}, {"source_checkpoint": None}):
            with self.subTest(updates=updates), self.mutated():
                value = read(self.namespace_path)
                value.update(updates)
                write(self.namespace_path, value)
                ref = {"path": str(self.namespace_path), "sha256": sha(self.namespace_path)}
                value = read(self.all_initialized)
                value["curate_generation_ref"] = ref
                write(self.all_initialized, value)
                value = read(self.activation_path)
                value["namespace_ref"] = ref
                write(self.activation_path, value)
                self.relink_activation()
                self.assert_rejected_for(
                    "namespace/provider/scope differs" if "namespace" in updates else "source checkpoint differs"
                )

    def test_live_and_captured_controller_must_both_bind_activation(self):
        for path in (self.live_controller, self.final_controller):
            with self.subTest(path=path), self.mutated():
                value = read(path)
                value["curate_generation_activation_ref"]["sha256"] = "f" * 64
                write(path, value, wrapped=True)
                self.relink()
                self.assert_rejected_for("controller/checkpoint does not bind")

    def test_rehashed_wrong_source_group_or_missing_post_restore_namespace_rejects(self):
        for field, replacement in (
            ("curate_generation_namespace", "b" * 64),
            ("curator_jobs", [{"candidate_group_id": "old"}]),
        ):
            with self.subTest(field=field), self.mutated():
                for path in (self.live_controller, self.final_controller):
                    value = read(path)
                    value["source_queue"]["jobs"]["fixture_source"][field] = replacement
                    write(path, value, wrapped=True)
                self.relink()
                self.assert_rejected_for(
                    "active curate generation namespace" if field == "curate_generation_namespace" else "cohort differs"
                )

    def test_changed_published_generation_provider_rejects(self):
        path = self.source / self.compare.CURATE_GENERATION_PROVIDER_FILES["provider"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# changed provider\n")
            self.assert_rejected_for("Extension bytes differ from published source")
        finally:
            path.write_bytes(original)


class GuardedCurateGenerationComparisonTests(unittest.TestCase):
    """v2 separates process-default, loader-transport and controller evidence."""

    compare_pair = CurateGenerationComparisonTests.compare_pair
    assert_incomplete = CurateGenerationComparisonTests.assert_incomplete
    assert_rejected_for = CurateGenerationComparisonTests.assert_rejected_for
    install_source = classmethod(CurateGenerationComparisonTests.install_source.__func__)
    relink_activation = CurateGenerationComparisonTests.relink_activation

    @classmethod
    def setUpClass(cls):
        CurateGenerationComparisonTests.setUpClass.__func__(cls)
        compare = cls.compare
        for relative in compare.CURATE_RUNTIME_GUARDS_FILES.values():
            path = cls.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
            subprocess.check_call(["git", "-C", str(cls.source), "add", relative])
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
                "Guarded curate generation fixture",
            ],
            check=True,
            capture_output=True,
        )
        commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()
        for component in (cls.spawn, cls.loaded_spawn):
            component["driver_source_commit"] = commit
            component["composed_extensions"]["resilient_setup"]["driver_source_commit"] = commit
        cls.all_identity.update(
            policy=compare.CURATE_GENERATION_V2_POLICY,
            driver_source_commit=commit,
            runtime_guards_extension={
                "policy": compare.CURATE_RUNTIME_GUARDS_POLICY,
                "files": {
                    name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                    for name, relative in compare.CURATE_RUNTIME_GUARDS_FILES.items()
                },
                "learner_start_method": "spawn",
                "credit_probe_concurrency": 2,
            },
        )
        for role in ("base", "trained"):
            path = cls.pair.root / role / "training-manifest.json"
            value = read(path)
            value["config"]["data"]["dataloader_num_workers"] = 4
            value["config"]["skill_harness"]["joint"]["credit_concurrency"] = 8
            if role == "base":
                value["source_commit"] = commit
            write(path, value)
        value = read(cls.observations["base"])
        value.update(source_commit=commit, spawn_extension=cls.spawn)
        value["effective_config_before_entry"]["data"]["dataloader_num_workers"] = 4
        value["effective_config_before_entry"]["skill_harness"]["joint"]["credit_concurrency"] = 8
        write(cls.observations["base"], value)
        cls.learner_path = cls.pair.root / "trained/driver_runs/400.learner_runtime.json"
        owner = read(cls.namespace_path)["owner"]
        write(
            cls.learner_path,
            {
                "schema_version": 1,
                "kind": "observed_spawn_learner_default",
                "policy": compare.CURATE_RUNTIME_GUARDS_POLICY,
                "owner": owner,
                "observed_before": "trainer_construct",
                "previous_default_start_method": "fork",
                "observed_default_start_method": "spawn",
                "configured_dataloader_workers": 4,
                "configured_credit_concurrency": 8,
                "physical_credit_probe_concurrency": 2,
                "provider_ref": cls.all_identity["runtime_guards_extension"]["files"]["learner"],
                "scope": compare.CURATE_LEARNER_DEFAULT_SCOPE,
            },
        )
        ref = {"path": str(cls.learner_path), "sha256": sha(cls.learner_path)}
        value = read(cls.all_initialized)
        value.update(cls.all_identity, learner_runtime_ref=ref)
        value["effective_config_after_trainer_init"]["data"]["dataloader_num_workers"] = 4
        value["effective_config_after_trainer_init"]["skill_harness"]["joint"]["credit_concurrency"] = 8
        value["original_training_manifest_sha256"] = sha(cls.pair.root / "trained/training-manifest.json")
        value["learner_runtime_observation"] = {
            "default_start_method": "spawn",
            "scope": compare.CURATE_LOADER_TRANSPORT_SCOPE,
            **{
                role: {
                    "dataset_class": "verl.utils.dataset.rl_dataset.RLHFDataset",
                    "num_workers": 4,
                    "multiprocessing_context": None,
                    "iterator_started": False,
                    "serialize_dataset_before": False,
                    "serialize_dataset_after": True,
                    "dataframe_fingerprint": "synthetic_" + role,
                    "dataset_length": length,
                    "filtered_dataframe_unchanged": True,
                    "processor_before_class": "transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor",
                    "processor_after_class": None,
                    "trainer_processor_unchanged": True,
                    "text_only_transport": True,
                }
                for role, length in (("train", 3), ("val", 2))
            },
        }
        write(cls.all_initialized, value)
        value = read(cls.all_prelaunch)
        identity = copy.deepcopy(cls.all_identity)
        identity["all_role_native_spawn"]["curator_profiles"] = []
        value.update(identity)
        value["config"]["data"]["dataloader_num_workers"] = 4
        value["config"]["skill_harness"]["joint"]["credit_concurrency"] = 8
        write(cls.all_prelaunch, value)
        value = read(cls.activation_path)
        value["runtime_guards"] = {
            "learner_runtime_ref": ref,
            "observed_default_start_method": "spawn",
            "probe_semaphore_class": "asyncio.locks.Semaphore",
            "probe_semaphore_initial_value": 2,
            "configured_credit_concurrency": 8,
        }
        write(cls.activation_path, value)
        for path in (cls.live_controller, cls.final_controller):
            state = read(path)
            state["curate_generation_activation_ref"] = {
                "path": str(cls.activation_path),
                "sha256": sha(cls.activation_path),
            }
            write(path, state, wrapped=True)
        cls.relink()

    @classmethod
    def relink(cls):
        CurateGenerationComparisonTests.relink.__func__(cls)
        receipt = read(cls.receipt_path)
        original = sha(cls.pair.root / "trained/training-manifest.json")
        receipt["original_training_manifest_sha256"] = original
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            value = read(path)
            value["original_training_manifest_sha256"] = original
            write(path, value)
            receipt[key]["sha256"] = sha(path)
        write(cls.receipt_path, receipt)

    @contextmanager
    def mutated(self):
        original = self.learner_path.read_bytes()
        try:
            with CurateGenerationComparisonTests.mutated(self):
                yield
        finally:
            self.learner_path.write_bytes(original)

    def relink_learner(self):
        ref = {"path": str(self.learner_path), "sha256": sha(self.learner_path)}
        value = read(self.all_initialized)
        value["learner_runtime_ref"] = ref
        write(self.all_initialized, value)
        value = read(self.activation_path)
        value["runtime_guards"]["learner_runtime_ref"] = ref
        write(self.activation_path, value)
        self.relink_activation()

    def test_v2_source10_full274_matches_original_base_without_runtime_guard_execution(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        guards = report["all_native_spawn_evidence"]["curate_generation"]["runtime_guards"]
        self.assertEqual(guards["observed_default_start_method"], "spawn")
        self.assertEqual(guards["delegated_probe_limit"], 2)
        self.assertIn("source-linked inference", guards["scope"])
        self.assertEqual(report["pooled"]["games"], 274)
        self.assertFalse((self.pair.root / "base/driver_runs/400.learner_runtime.json").exists())

    def test_prelaunch_runtime_intent_cannot_replace_actual_constructor_observation(self):
        with self.mutated():
            self.learner_path.unlink()
            self.assert_rejected_for("learner_runtime.json")

    def update_identity_links(self):
        initialized = read(self.all_initialized)
        initialized.update(self.all_identity)
        write(self.all_initialized, initialized)
        before = read(self.all_prelaunch)
        identity = copy.deepcopy(self.all_identity)
        identity["all_role_native_spawn"]["curator_profiles"] = []
        before.update(identity)
        write(self.all_prelaunch, before)
        self.relink()

    def test_rehashed_runtime_intent_requires_spawn_and_two_delegated_probes(self):
        for field, replacement in (("learner_start_method", "fork"), ("credit_probe_concurrency", 3)):
            with self.subTest(field=field), self.mutated():
                self.all_identity["runtime_guards_extension"][field] = replacement
                self.update_identity_links()
                self.assert_rejected_for("runtime guard intent differs")

    def test_v1_policy_cannot_silently_carry_v2_observations(self):
        with self.mutated():
            self.all_identity["policy"] = self.compare.CURATE_GENERATION_POLICY
            self.update_identity_links()
            self.assert_rejected_for("AllNative lifecycle extension: unsupported or missing schema fields")

    def test_rehashed_wrong_observed_default_owner_or_constructor_stage_rejects(self):
        for updates in (
            {"observed_default_start_method": "fork"},
            {"observed_before": "trainer_init"},
            {"owner": {"pid": 9999, "start_ticks": 5678}},
            {"configured_dataloader_workers": 0},
            {"configured_credit_concurrency": 2},
        ):
            with self.subTest(updates=updates), self.mutated():
                value = read(self.learner_path)
                value.update(updates)
                write(self.learner_path, value)
                self.relink_learner()
                self.assert_rejected_for("Observed learner spawn default, owner, timing")

    def test_rehashed_loader_data_change_or_started_iterator_rejects(self):
        for role in ("train", "val"):
            for updates in (
                {"serialize_dataset_after": False},
                {"iterator_started": True},
                {"filtered_dataframe_unchanged": False},
                {"num_workers": 0},
                {"multiprocessing_context": "fork"},
                {"dataframe_fingerprint": ""},
                {"text_only_transport": False},
                {"trainer_processor_unchanged": False},
                {"processor_after_class": "fixture.Processor"},
            ):
                with self.subTest(role=role, updates=updates), self.mutated():
                    value = read(self.all_initialized)
                    value["learner_runtime_observation"][role].update(updates)
                    write(self.all_initialized, value)
                    self.relink()
                    self.assert_rejected_for("loader transport changed data, workers or iteration boundary")

    def test_rehashed_wrong_probe_semaphore_or_activation_default_rejects(self):
        for updates in (
            {"probe_semaphore_initial_value": 3},
            {"probe_semaphore_class": "asyncio.Lock"},
            {"observed_default_start_method": "fork"},
            {"configured_credit_concurrency": 2},
        ):
            with self.subTest(updates=updates), self.mutated():
                value = read(self.activation_path)
                value["runtime_guards"].update(updates)
                write(self.activation_path, value)
                self.relink_activation()
                self.assert_rejected_for("Activated controller spawn default or delegated probe semaphore differs")

    def test_source_linked_inference_cannot_claim_per_worker_attestation(self):
        with self.mutated():
            value = read(self.all_initialized)
            value["learner_runtime_observation"]["scope"] = "Every worker and every orphan was attested clean"
            write(self.all_initialized, value)
            self.relink()
            self.assert_rejected_for("source-linked inference scope differs")

    def test_changed_published_learner_provider_rejects(self):
        path = self.source / self.compare.CURATE_RUNTIME_GUARDS_FILES["learner"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# changed learner guard\n")
            self.assert_rejected_for("Extension bytes differ from published source")
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
