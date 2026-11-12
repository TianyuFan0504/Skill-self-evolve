"""Owned v3 entry and original baseline ordering without importing a model."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_train_curate_generation as generation_contract

HERE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load("_train_curate_epoch_cpu", "train_curate_epoch.py")
baseline = load("_train_pool_baseline_cpu", "train_account_pool_baseline.py")


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = generation_contract.EntryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.modules, self.namespace = self.fixture.modules, self.fixture.namespace
        self.namespace.train_curate_generation = generation_contract.entry
        self.namespace.account_pool_epoch = types.SimpleNamespace(
            identity=lambda: {"selected": "private-manifest-epoch3"})

    def test_v3_identity_links_v2_component_and_records_cap8_without_rewriting_v2(self):
        with patch.dict(sys.modules, self.modules):
            identity = entry.extension_identity(self.fixture.fixture.policy)
            v2 = generation_contract.entry.extension_identity(self.fixture.fixture.policy)
        self.assertEqual(identity["policy"], entry.POLICY)
        self.assertEqual(identity["files"]["generation_component"]["path"],
                         str(HERE / "scripts/train_curate_generation.py"))
        self.assertEqual(identity["account_pool_epoch_extension"], {"selected": "private-manifest-epoch3"})
        self.assertEqual(identity["runtime_guards_extension"]["credit_probe_concurrency"], 8)
        self.assertEqual(v2["runtime_guards_extension"]["credit_probe_concurrency"], 2)
        self.assertEqual(identity["curate_generation_extension"]["controller_class"],
                         "skill_harness_examples.train_curate_epoch.EpochGenerationController")

    def test_learner_requires_actual_hook_then_observes_spawn_and_cap8(self):
        events = []
        self.namespace.account_pool_epoch = types.SimpleNamespace(
            require_worker_activation=lambda config: events.append("verify-hook"),
            install_process=lambda **kw: events.append(("install", kw)))
        self.namespace.learner_runtime = types.SimpleNamespace(
            prepare_learner=lambda config, **kw: events.append(("spawn", kw)) or "observed-ref")
        with patch.dict(sys.modules, self.modules):
            self.assertEqual(entry.prepare_learner("config"), "observed-ref")
        self.assertEqual(events, ["verify-hook", ("install", {"role": "learner"}),
                                  ("spawn", {"physical_probe_concurrency": 8})])

    def test_all_native_composition_places_driver_before_health_and_hook_after_bootstrap_merge(self):
        path = HERE / "scripts/train_all_native_spawn.py"
        tree = ast.parse(path.read_text())
        body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "entry")
        calls = sorted((node.lineno, ast.unparse(node.func)) for node in ast.walk(body) if isinstance(node, ast.Call))
        lines = {name: line for line, name in calls}
        self.assertLess(lines["provider.prepare_driver"], lines["parallel.environment_observation"])
        self.assertLess(lines["apply_runtime_environment"], lines["provider.prepare_runtime_environment"])
        self.assertLess(lines["provider.prepare_runtime_environment"], lines["run_ppo"])

    def test_old_all_native_entry_rejects_selected_epoch_before_health_or_training(self):
        self.namespace.train_parallel_client = types.SimpleNamespace(
            environment_observation=lambda settings: self.fail("No health or model work before epoch rejection"))
        self.modules["omegaconf"].open_dict = lambda config: None
        self.modules["verl.experimental.skill_harness.bootstrap"].apply_runtime_environment = lambda config: None
        with patch.dict(sys.modules, self.modules), \
                patch.dict(os.environ, {"ALFWORLD_ACCOUNT_POOL_MANIFEST": "/selected/epoch3.json"}), \
                self.assertRaisesRegex(ValueError, "explicit train_curate_epoch"):
            self.namespace.train_all_native_spawn.entry(self.fixture.fixture.config)

    def test_v3_delegates_the_real_shared_entry_with_explicit_provider(self):
        self.namespace.train_all_native_spawn = types.SimpleNamespace(entry=lambda config, **kw: (config, kw))
        with patch.dict(sys.modules, self.modules):
            self.assertEqual(entry.entry("config"),
                             ("config", {"extension_module": "skill_harness_examples.train_curate_epoch"}))

    def test_actual_v3_controller_gate_has_eight_slots_and_rejects_a_cap_two_learner(self):
        contract = generation_contract.contract
        fixture = contract.GenerationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        namespace = sys.modules["skill_harness_examples"]
        namespace.curate_generation = contract.generation
        namespace.account_pool_epoch = types.SimpleNamespace(_active="selected", identity=lambda: "selected")
        selected = type("EpochGenerationController", (entry.EpochGenerationControllerMixin, contract.Controller),
                        {"__module__": entry.__name__})
        original_read = contract.generation.read_ref

        def read(ref):
            if ref["path"] == "runtime-reference":
                return {"physical_credit_probe_concurrency": 8}
            return original_read(ref)

        with patch.object(contract, "Controller", selected), \
                patch.object(contract.generation, "read_ref", side_effect=read):
            controller, _, _ = fixture.controller()
        self.assertEqual(controller._physical_probe_gate._value, 8)
        with patch.object(contract.generation, "read_ref", return_value={"physical_credit_probe_concurrency": 2}), \
                self.assertRaisesRegex(ValueError, "physical probe limits"):
            controller._credit_probe_concurrency()


class BaselineTests(unittest.TestCase):
    def config(self):
        return types.SimpleNamespace(
            skill_harness=types.SimpleNamespace(mode="skill_joint", artifact_root="/trusted/run"),
            trainer=types.SimpleNamespace(total_epochs=0, resume_mode="disable", resume_from_path=None,
                                          val_before_train=False, val_only=False))

    def test_original_public_body_and_runner_are_used_with_common_pool_hook(self):
        events = []
        original = types.ModuleType("verl.experimental.skill_harness.train")
        original.__file__ = str(HERE.parents[1] / "verl/experimental/skill_harness/train.py")
        original.SkillTaskRunner = object()
        for name in ("auto_set_device", "apply_runtime_environment", "save_run_manifest"):
            setattr(original, name, lambda config, name=name: events.append(name))
        original.need_reference_policy = original.need_critic = lambda config: False
        original.validate_config = lambda config, **kw: events.append("validate_config")
        original.run_ppo = lambda config, **kw: events.append(("run_ppo", kw["task_runner_class"]))
        original.ray = types.SimpleNamespace(is_initialized=lambda: True, shutdown=lambda: events.append("shutdown"))
        package = types.ModuleType("verl.experimental.skill_harness")
        package.__path__ = []
        package.train = original
        namespace = types.ModuleType("skill_harness_examples")
        namespace.__path__ = []
        namespace.account_pool_epoch = types.SimpleNamespace(
            prepare_driver=lambda config: events.append("pool-driver"),
            prepare_runtime_environment=lambda config: events.append("pool-hook"),
            publish_inference_expected=lambda root, **kw: events.append(("expected", kw["baseline_extension"])))
        modules = {"skill_harness_examples": namespace, package.__name__: package, original.__name__: original}
        with patch.dict(sys.modules, modules):
            baseline.entry(self.config())
        names = [event[0] if isinstance(event, tuple) else event for event in events]
        self.assertEqual(names, ["pool-driver", "auto_set_device", "validate_config", "apply_runtime_environment",
                                 "pool-hook", "expected", "save_run_manifest", "run_ppo", "shutdown"])
        self.assertEqual(events[-2][1], original.SkillTaskRunner)
        self.assertEqual(events[5][1]["task_runner_class"],
                         "verl.experimental.skill_harness.train.SkillTaskRunner")

    def test_baseline_rejects_training_and_checkpoint_restore(self):
        for name, value in (("total_epochs", 1), ("resume_mode", "auto"),
                            ("resume_from_path", "/checkpoint10"), ("val_only", True)):
            config = self.config()
            setattr(config.trainer, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                baseline.validate_baseline(config)

    def test_baseline_never_references_all_native_or_curator_or_generation_provider(self):
        tree = ast.parse((HERE / "scripts/train_account_pool_baseline.py").read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any("curate" in name or "all_native" in name for name in imports))
        functions = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertIn("original.run_ppo", functions)
        self.assertNotIn("trainer_class", functions)


if __name__ == "__main__":
    unittest.main()
