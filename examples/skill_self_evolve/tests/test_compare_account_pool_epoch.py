"""Full synthetic epoch3 comparisons with real source/manifest/marker checks.

Only the existing fixture's inert registration account is moved to epoch3 at
creation. No Unix identities, registry, helper, model or server is executed.
"""

from __future__ import annotations

import copy
import subprocess
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import test_compare_curate_generation as generation_fixture

ROOT = generation_fixture.ROOT
read, write, sha = generation_fixture.read, generation_fixture.write, generation_fixture.sha
native_fixture = generation_fixture.native_fixture
fixture = native_fixture.fixture


class AccountEpochComparisonTests(unittest.TestCase):
    compare_pair = generation_fixture.GuardedCurateGenerationComparisonTests.compare_pair
    assert_incomplete = generation_fixture.GuardedCurateGenerationComparisonTests.assert_incomplete
    assert_rejected_for = generation_fixture.GuardedCurateGenerationComparisonTests.assert_rejected_for
    install_source = classmethod(generation_fixture.GuardedCurateGenerationComparisonTests.install_source.__func__)
    relink = classmethod(generation_fixture.GuardedCurateGenerationComparisonTests.relink.__func__)
    relink_activation = generation_fixture.GuardedCurateGenerationComparisonTests.relink_activation
    update_identity_links = generation_fixture.GuardedCurateGenerationComparisonTests.update_identity_links

    @classmethod
    def setUpClass(cls):
        original = fixture.budget.make_registration

        def epoch_registration(payload, **kwargs):
            payload["account"].update(uid=30000, gid=30000, name="alfworld_ep3_000")
            return original(payload, **kwargs)

        with patch.object(fixture.budget, "make_registration", epoch_registration):
            generation_fixture.GuardedCurateGenerationComparisonTests.setUpClass.__func__(cls)
        compare = cls.compare
        relatives = {*compare.CURATE_EPOCH_FILES.values(), compare.ACCOUNT_EPOCH_PROVIDER,
                     compare.ACCOUNT_EPOCH_BASELINE_DRIVER, "verl/experimental/skill_harness/train.py"}
        for relative in relatives:
            path = cls.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
            subprocess.check_call(["git", "-C", str(cls.source), "add", relative])
        if subprocess.run(["git", "-C", str(cls.source), "diff", "--cached", "--quiet"]).returncode:
            subprocess.run(["git", "-C", str(cls.source), "-c", "user.name=Fixture", "-c",
                            "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "-c",
                            "core.hooksPath=/dev/null", "commit", "-q", "-m", "Epoch fixture"],
                           check=True, capture_output=True)
        commit = subprocess.check_output(["git", "-C", str(cls.source), "rev-parse", "HEAD"], text=True).strip()
        for identity in (cls.spawn, cls.loaded_spawn):
            identity["driver_source_commit"] = commit
            identity["composed_extensions"]["resilient_setup"]["driver_source_commit"] = commit
        cls.operator_manifest = Path(cls.temporary.name).resolve() / "operator-epoch3.json"
        roster = [{"name": f"alfworld_ep3_{index:03d}", "uid": 30000 + index, "gid": 30000 + index}
                  for index in range(128)]
        write(cls.operator_manifest, {
            "schema_version": 1, "kind": "operator_provisioned_account_pool_epoch",
            "policy": compare.ACCOUNT_EPOCH_POLICY,
            "epoch": 3, "pool_root": compare.ACCOUNT_EPOCH_ROOT, "roster": roster,
            "roster_digest": fixture.sha(roster), "isolation_policy": "alfworld-uid-landlock-v1",
            "fenced_pool": {"root": "/var/lib/alfworld-skill-harness/account_pool_v2", "uid_min": 20000,
                            "uid_max": 20127, "protected_uid": 20071,
                            "operation": "preserve-without-reset-release-remap"}, "created_at_ns": 1,
        })
        cls.epoch_extension = {
            "policy": compare.ACCOUNT_EPOCH_POLICY,
            "provider_ref": {"path": str(cls.source / compare.ACCOUNT_EPOCH_PROVIDER),
                             "sha256": sha(cls.source / compare.ACCOUNT_EPOCH_PROVIDER)},
            "manifest_ref": {"path": str(cls.operator_manifest), "sha256": sha(cls.operator_manifest)},
            "pool_root": compare.ACCOUNT_EPOCH_ROOT, "roster_digest": fixture.sha(roster),
            "worker_setup_hook": compare.ACCOUNT_EPOCH_HOOK, "expected_capacity": 128,
        }
        cls.all_identity.update(
            policy=compare.CURATE_EPOCH_POLICY, driver_source_commit=commit,
            actual_entry=str(cls.source / compare.CURATE_EPOCH_FILES["entry"]),
            files={name: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
                   for name, relative in compare.CURATE_EPOCH_FILES.items()},
            account_pool_epoch_extension=cls.epoch_extension,
        )
        cls.all_identity["curate_generation_extension"]["controller_class"] = (
            "skill_harness_examples.train_curate_epoch.EpochGenerationController")
        cls.all_identity["runtime_guards_extension"]["credit_probe_concurrency"] = 8
        owner = read(cls.namespace_path)["owner"]
        cls.epoch_markers = {}
        for role in ("base", "trained"):
            run = cls.pair.root / role
            directory = run / "runtime/account_pool_epoch/activations"
            roles = [("driver", {"pid": 1, "start_ticks": 1}),
                     ("ray_worker", {"pid": 2, "start_ticks": 2}),
                     ("task_account_pool", {"pid": 2, "start_ticks": 2})]
            if role == "trained":
                roles += [("ray_worker", owner), ("learner", owner), ("probe_account_pool", owner)]
            for activation_role, process in roles:
                path = directory / f"{process['pid']}-{process['start_ticks']}.{activation_role}.json"
                write(path, {
                    "schema_version": 1, "kind": "activated_account_pool_epoch", "policy": compare.ACCOUNT_EPOCH_POLICY,
                    "owner": process, "activation_role": activation_role,
                    **{key: cls.epoch_extension[key] for key in ("provider_ref", "manifest_ref", "pool_root",
                                                                "roster_digest", "worker_setup_hook")},
                    "binding_modules": compare.ACCOUNT_EPOCH_BINDINGS,
                    "native_spawn_account_pool": cls.epoch_extension,
                    "preserved_core_account_policy": "alfworld-uid-landlock-v1",
                })
                cls.epoch_markers[(role, activation_role)] = path
        baseline_ref = lambda relative: {"path": str(cls.source / relative), "sha256": sha(cls.source / relative)}
        cls.baseline_expected = cls.pair.root / "base" / compare.ACCOUNT_EPOCH_EXPECTED_RECEIPT
        write(cls.baseline_expected, {
            "schema_version": 1, "kind": "account_pool_epoch_inference_expected",
            "account_pool_epoch_extension": cls.epoch_extension,
            "activation_directory": str(cls.pair.root / "base/runtime/account_pool_epoch/activations"),
            "baseline_extension": {
                "policy": compare.ACCOUNT_EPOCH_BASELINE_POLICY,
                "actual_entry": str(cls.source / compare.ACCOUNT_EPOCH_BASELINE_DRIVER),
                "entry_ref": baseline_ref(compare.ACCOUNT_EPOCH_BASELINE_DRIVER),
                "original_entry_ref": baseline_ref("verl/experimental/skill_harness/train.py"),
                "task_runner_class": "verl.experimental.skill_harness.train.SkillTaskRunner",
            },
        })
        base_manifest = cls.pair.root / "base/training-manifest.json"
        value = read(base_manifest)
        value["source_commit"] = commit
        cls.epoch_config(value["config"])
        write(base_manifest, value)
        value = read(cls.observations["base"])
        value.update(source_commit=commit, spawn_extension=cls.spawn,
                     entrypoint=compare.ACCOUNT_EPOCH_BASELINE_MODULE)
        value["original_entry_command"][2] = compare.ACCOUNT_EPOCH_BASELINE_MODULE
        cls.epoch_config(value["effective_config_before_entry"])
        write(cls.observations["base"], value)
        value = read(cls.all_initialized)
        value.update(cls.all_identity, kind="initialized_curate_epoch_entry")
        value["primary_worker_scope"] = (
            "Official Ray setup_worker selects the fresh pool in actual episode workers; unchanged SpawnSafeRunner")
        cls.epoch_config(value["effective_config_after_trainer_init"])
        value["account_pool_epoch_activation_directory"] = str(
            cls.pair.root / "trained/runtime/account_pool_epoch/activations")
        learner_marker = cls.epoch_markers[("trained", "learner")]
        value["account_pool_epoch_activation_refs"] = [{"path": str(learner_marker), "sha256": sha(learner_marker)}]
        new = cls.all_initialized.with_name("402.curate_epoch.initialized.json")
        write(new, value)
        cls.all_initialized.unlink()
        cls.all_initialized = new
        value = read(cls.all_prelaunch)
        identity = copy.deepcopy(cls.all_identity)
        identity["all_role_native_spawn"]["curator_profiles"] = []
        value.update(identity, kind="prelaunch_curate_epoch_parallel_components")
        cls.epoch_config(value["config"])
        new = cls.all_prelaunch.with_name("401.curate_epoch.json")
        write(new, value)
        cls.all_prelaunch.unlink()
        cls.all_prelaunch = new
        value = read(cls.learner_path)
        value["physical_credit_probe_concurrency"] = 8
        write(cls.learner_path, value)
        learner_ref = {"path": str(cls.learner_path), "sha256": sha(cls.learner_path)}
        value = read(cls.all_initialized)
        value["learner_runtime_ref"] = learner_ref
        write(cls.all_initialized, value)
        value = read(cls.activation_path)
        value["controller_class"] = cls.all_identity["curate_generation_extension"]["controller_class"]
        value["runtime_guards"]["probe_semaphore_initial_value"] = 8
        value["runtime_guards"]["learner_runtime_ref"] = learner_ref
        write(cls.activation_path, value)
        cls.relink_activation(cls)

    @classmethod
    def epoch_config(cls, config):
        runtime = config.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {})
        runtime["worker_process_setup_hook"] = cls.compare.ACCOUNT_EPOCH_HOOK
        runtime.setdefault("env_vars", {}).update({
            "ALFWORLD_ACCOUNT_POOL_MANIFEST": cls.epoch_extension["manifest_ref"]["path"],
            "ALFWORLD_ACCOUNT_POOL_MANIFEST_SHA256": cls.epoch_extension["manifest_ref"]["sha256"],
        })

    @contextmanager
    def mutated(self):
        extra = [self.operator_manifest, self.baseline_expected, *self.epoch_markers.values()]
        saved = {path: path.read_bytes() for path in extra}
        try:
            with generation_fixture.GuardedCurateGenerationComparisonTests.mutated(self):
                yield
        finally:
            for path, data in saved.items():
                path.write_bytes(data)

    def test_full274_matches_same_pool_original_trainer_baseline_and_actual_workers(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertEqual(report["pooled"]["games"], 274)
        guards = report["all_native_spawn_evidence"]["curate_generation"]["runtime_guards"]
        self.assertEqual(guards["delegated_probe_limit"], 8)
        evidence = report["account_pool_epoch_evidence"]
        self.assertEqual(evidence["shared_identity"], self.epoch_extension)
        self.assertEqual(evidence["base"]["actual_roles"], ["driver", "ray_worker", "task_account_pool"])
        self.assertIn("probe_account_pool", evidence["trained"]["actual_roles"])
        self.assertFalse((self.pair.root / "base/controller/generation_epochs").exists())

    def test_intent_alone_cannot_replace_actual_primary_or_probe_activation(self):
        for key in (("base", "task_account_pool"), ("trained", "task_account_pool"), ("trained", "probe_account_pool")):
            with self.subTest(key=key), self.mutated():
                self.epoch_markers[key].unlink()
                self.assert_rejected_for("actual")

    def test_initialization_cannot_predict_future_worker_activation(self):
        with self.mutated():
            value = read(self.all_initialized)
            value["account_pool_epoch_activation_refs"] = [{"path": "/missing/worker.json", "sha256": "a" * 64}]
            write(self.all_initialized, value)
            self.relink()
            self.assert_rejected_for("initialized activation reference differs")

    def test_rehashed_manifest_cannot_reuse_old_uids_or_weaken_protected_fence(self):
        for target in ("roster", "fence"):
            with self.subTest(target=target), self.mutated():
                value = read(self.operator_manifest)
                if target == "roster":
                    value["roster"][0]["uid"] = 20071
                else:
                    value["fenced_pool"]["operation"] = "reset"
                write(self.operator_manifest, value)
                ref = {"path": str(self.operator_manifest), "sha256": sha(self.operator_manifest)}
                self.all_identity["account_pool_epoch_extension"]["manifest_ref"] = ref
                baseline = read(self.baseline_expected)
                baseline["account_pool_epoch_extension"]["manifest_ref"] = ref
                write(self.baseline_expected, baseline)
                self.update_identity_links()
                self.assert_rejected_for("manifest roster or old-pool fence differs")

    def test_wrong_hook_and_worker_binding_are_rejected(self):
        with self.mutated():
            path = self.epoch_markers[("trained", "task_account_pool")]
            value = read(path)
            value["binding_modules"] = value["binding_modules"][:-1]
            write(path, value)
            self.assert_rejected_for("worker activation owner, bindings or provider differs")

    def test_native_spawn_registration_must_match_the_selected_epoch(self):
        with self.mutated():
            path = self.epoch_markers[("trained", "task_account_pool")]
            value = read(path)
            value["native_spawn_account_pool"]["pool_root"] = "/var/lib/alfworld-skill-harness/account_pool_v2"
            write(path, value)
            self.assert_rejected_for("worker activation owner, bindings or provider differs")

    def test_v3_requires_eight_physical_probes(self):
        with self.mutated():
            self.all_identity["runtime_guards_extension"]["credit_probe_concurrency"] = 2
            self.update_identity_links()
            self.assert_rejected_for("runtime guard intent differs")

    def test_valid_old_uid_case_cannot_hide_behind_fresh_worker_markers(self):
        run = self.pair.root / "base"
        output = next((run / "episodes").rglob("budget-registration.json")).parent
        original = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
        try:
            # Recreate internally valid old-roster evidence with the existing
            # inert fixture producer; no account, lease or process is touched.
            self.pair._registration(run, output, read(output / "result.json"))
            self.assert_rejected_for("did not use the selected account pool epoch")
        finally:
            for path, data in original.items():
                path.write_bytes(data)

    def test_unrelated_diagnostic_files_are_not_epoch_activation_evidence(self):
        path = self.pair.root / "trained/runtime/account_pool_epoch/activations/unrelated.json"
        try:
            path.write_text('{"kind":"other_diagnostic"}')
            report = self.compare_pair()
            self.assertTrue(report["comparison_supported"], report["reasons"])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
