"""The opt-in provider shares the real entry body and keeps truthful receipts."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_curate_generation as contract
import test_learner_runtime as runtime_contract
import test_train_all_native_spawn as legacy

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("_generation_entry_cpu", HERE / "scripts/train_curate_generation.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = legacy.EntryTests()
        self.fixture.setUp()
        self.modules = self.fixture.modules
        self.namespace = self.modules["skill_harness_examples"]
        self.namespace.train_all_native_spawn = legacy.entry
        self.namespace.curate_generation = contract.generation
        self.namespace.learner_runtime = runtime_contract.runtime
        self.modules["skill_harness_examples.curate_generation"] = contract.generation
        self.modules["skill_harness_examples.learner_runtime"] = runtime_contract.runtime
        self.namespace.train_resilient_setup = types.SimpleNamespace(
            __file__=str(HERE / "scripts/train_resilient_setup.py"))
        self.modules["skill_harness_examples.train_curate_generation"] = entry
        self.fixture.config.data = types.SimpleNamespace(dataloader_num_workers=4)
        self.fixture.config.skill_harness.joint = types.SimpleNamespace(credit_concurrency=8)
        self.method = "fork"

        def set_method(method, force):
            self.method = method
            self.fixture.events.append("spawn-default")

        mp = patch.object(runtime_contract.runtime, "multiprocessing", types.SimpleNamespace(
            set_start_method=set_method, get_start_method=lambda **kwargs: self.method))
        mp.start()
        self.addCleanup(mp.stop)

    def test_explicit_identity_names_new_entry_provider_and_composed_all_native_body(self):
        with patch.dict(sys.modules, self.modules):
            identity = entry.extension_identity(self.fixture.policy)
        self.assertEqual(identity["policy"], "parallel-direct-all-native-spawn-curate-generation-v2")
        self.assertEqual(identity["actual_entry"], str(HERE / "scripts/train_curate_generation.py"))
        self.assertEqual(identity["files"]["all_native_component"]["path"],
                         str(HERE / "scripts/train_all_native_spawn.py"))
        self.assertEqual(identity["curate_generation_extension"]["policy"], contract.generation.POLICY)
        self.assertEqual(identity["solver_spawn_component_identity"], {"component": "frozen-solver-spawn"})
        self.assertEqual(identity["runtime_guards_extension"]["credit_probe_concurrency"], 2)

    def test_namespace_is_prepared_after_restore_before_initialized_receipt_and_fit(self):
        events = self.fixture.events

        class Trainer:
            def __init__(self, config):
                events.append("trainer-construct")
                self.config = config
                self.logger = types.SimpleNamespace(finish=lambda **kwargs: None)

            def init(self):
                events.append("restored")
                self._joint_restored = None
                self.train_dataloader_it = None
                self.train_dataloader = types.SimpleNamespace(
                    dataset=runtime_contract.Dataset(), sampler=object(), num_workers=4,
                    multiprocessing_context=None, _iterator=None)
                self.val_dataloader = types.SimpleNamespace(
                    dataset=runtime_contract.Dataset(), sampler=object(), num_workers=4,
                    multiprocessing_context=None, _iterator=None)

            def _identity(self):
                return types.SimpleNamespace(runtime="actual-runtime")

            def get_llm_client(self):
                return "solver"

            def get_teacher_client(self):
                return None

            def get_reward_handles(self):
                return None

            def fit(self, manager):
                events.append("fit")
                assert Path(self._curate_generation_ref["path"]).is_file()

        self.fixture.direct.trainer_class = lambda: Trainer
        with tempfile.TemporaryDirectory() as temporary:
            self.fixture.config.skill_harness.artifact_root = temporary
            original_prepare = entry.prepare_trainer

            def prepare(trainer):
                events.append("prepare")
                original_prepare(trainer)

            def receipt(trainer, *args):
                assert contract.generation.read_ref(trainer._curate_generation_ref)["source_checkpoint"] is None
                events.append("new-initialized")

            with patch.dict(sys.modules, self.modules), patch.object(entry, "prepare_trainer", prepare), \
                    patch.object(entry, "initialized_receipt", receipt), \
                    patch.object(contract.generation, "process_identity", return_value=contract.OWNER):
                legacy.entry.runner_class("skill_harness_examples.train_curate_generation")().run(self.fixture.config)
        self.assertLess(events.index("restored"), events.index("prepare"))
        self.assertLess(events.index("spawn-default"), events.index("trainer-construct"))
        self.assertLess(events.index("prepare"), events.index("new-initialized"))
        self.assertLess(events.index("new-initialized"), events.index("fit"))

    def test_initialized_receipt_separates_prepared_namespace_from_live_controller_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training-manifest.json").write_text("{}")
            policy = self.fixture.policy
            policy.curator_profiles["custom"] = "profile"
            loop = self.fixture.Loop()
            loop.native_spawner, loop.curator_profiles = policy.spawner, policy.curator_profiles
            trainer = types.SimpleNamespace(
                _joint_loop=loop, _joint_restored=None,
                _joint_coordinator=types.SimpleNamespace(last_restore_receipt=None),
                _curate_generation_ref={"path": "prepared-reference", "sha256": "a" * 64},
                _learner_runtime_ref={"path": "runtime-reference", "sha256": "b" * 64},
                _learner_runtime_observation={"fixture": "observed loader transport"},
            )
            captured = []
            self.fixture.spawn._base = lambda: types.SimpleNamespace(
                _publish=lambda path, value: captured.append((path, value)))
            config = types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(root)))
            with patch.dict(sys.modules, self.modules), \
                    patch.object(asyncio, "get_event_loop_policy", return_value=policy):
                entry.initialized_receipt(trainer, config, {}, policy)
        path, value = captured[0]
        self.assertTrue(path.name.endswith(".curate_generation.initialized.json"))
        self.assertEqual(value["kind"], "initialized_curate_generation_entry")
        self.assertEqual(value["curate_generation_ref"], trainer._curate_generation_ref)
        self.assertEqual(value["learner_runtime_ref"], trainer._learner_runtime_ref)
        self.assertNotIn("actual_controller_class", value)

    def test_opt_in_entry_uses_provider_without_replacing_default_module_globals(self):
        with patch.dict(sys.modules, self.modules), patch.object(legacy.entry, "entry", return_value="called") as call:
            self.assertEqual(entry.entry("config"), "called")
        call.assert_called_once_with("config", extension_module="skill_harness_examples.train_curate_generation")

    def test_actual_lifecycle_factory_selects_opt_in_controller_and_preserves_default(self):
        spec = importlib.util.spec_from_file_location(
            "_generation_lifecycle_hook", HERE / "scripts/train_resilient_setup.py")
        setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup)

        class CoreController:
            @classmethod
            async def create(cls, *args, **kwargs):
                controller = cls()
                controller.protocol_fingerprint, controller.frozen_version = "protocol", "base"
                return controller

        controller_module = types.ModuleType("verl.experimental.skill_harness.controller")
        controller_module.SkillJointController = CoreController
        trainer_module = types.ModuleType("verl.experimental.skill_harness.trainer")
        trainer_module.SkillSyncTrainer = types.SimpleNamespace(on_train_begin=lambda trainer: "solver-on-begin")
        for enabled in (False, True):
            bases = ((contract.generation.GenerationTrainerMixin,) if enabled else ()) + (
                setup.ResilientLifecycleMixin,)
            trainer = type("Trainer", bases, {})()
            trainer._joint_lock, trainer._joint_controller, trainer._joint_restored = threading.RLock(), None, None
            trainer._identity = lambda: types.SimpleNamespace(
                runtime="actual", protocol_fingerprint="protocol", frozen_version="base")
            trainer._await = asyncio.run
            with patch.dict(sys.modules, {
                "verl.experimental.skill_harness.controller": controller_module,
                "verl.experimental.skill_harness.trainer": trainer_module,
                "skill_harness_examples.train_resilient_setup": setup,
            }), patch.object(contract.generation, "_controller_class", None):
                self.assertEqual(trainer.on_train_begin(), "solver-on-begin")
            self.assertEqual(type(trainer._joint_controller).__name__,
                             "GenerationController" if enabled else "SafeSetupController")


if __name__ == "__main__":
    unittest.main()
