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


if __name__ == "__main__":
    unittest.main()
