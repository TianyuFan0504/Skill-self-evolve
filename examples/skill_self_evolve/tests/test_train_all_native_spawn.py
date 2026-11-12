"""All-role entry ordering/provenance checks with inert runtime providers."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_all_native_entry_cpu", HERE / "scripts/train_all_native_spawn.py")
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


def module(name, **values):
    value = types.ModuleType(name)
    value.__dict__.update(values)
    return value


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class Loop:
            pass

        self.Loop = Loop
        self.policy = types.SimpleNamespace(spawner=object(), curator_profiles={})

        def register(runtime):
            events.append(("register", runtime))
            self.policy.curator_profiles["actual_curator"] = "registered-profile"

        self.policy.register_curator = register
        self.native = module(
            "skill_harness_examples.all_native_spawn",
            install_policy=lambda: events.append("all-policy") or self.policy,
            identity=lambda policy: {"policy": "actual-all-native", "curator_profiles": list(policy.curator_profiles)},
            AllNativeSelectorLoop=Loop,
        )
        self.spawn = module(
            "skill_harness_examples.train_spawn_safe", __file__=str(HERE / "scripts/train_spawn_safe.py"),
            configure_extensions=lambda config: events.append("extensions"),
            _source_commit=lambda root: "a" * 40, extension_identity=lambda: {"component": "frozen-solver-spawn"},
            initialized_receipt=lambda *args: (_ for _ in ()).throw(AssertionError("old loop receipt must not run")),
        )
        self.direct = module(
            "skill_harness_examples.train_direct_checkpoint", __file__=str(HERE / "scripts/train_direct_checkpoint.py"),
            passive_diagnostics=lambda run: types.SimpleNamespace(
                attach=lambda trainer: events.append("observer-attach"), stage=lambda stage: None,
                close=lambda: events.append("observer-close"),
            ),
        )
        namespace = module("skill_harness_examples", all_native_spawn=self.native,
                           train_spawn_safe=self.spawn, train_direct_checkpoint=self.direct)
        namespace.__path__ = []
        self.modules = {
            "ray": module("ray", remote=lambda cls: cls),
            "transfer_queue": module("transfer_queue", init=lambda config: events.append("tq-init"),
                                     close=lambda: events.append("tq-close")),
            "omegaconf": module("omegaconf", OmegaConf=types.SimpleNamespace(
                resolve=lambda config: events.append("resolve"), to_container=lambda *args, **kwargs: {})),
            "skill_harness_examples": namespace, self.native.__name__: self.native,
            self.spawn.__name__: self.spawn, self.direct.__name__: self.direct,
            "verl.experimental.skill_harness.bootstrap": module(
                "verl.experimental.skill_harness.bootstrap", configure_runtime=lambda: {}),
            "verl.experimental.skill_harness.train": module(
                "verl.experimental.skill_harness.train", _redact=lambda value: value,
                preserve_failed_queue=lambda *args: events.append("preserve-failure")),
            "verl.utils.import_utils": module("verl.utils.import_utils", load_class_from_fqn=lambda *args:
                types.SimpleNamespace(create=lambda **kwargs: events.append("manager") or "manager")),
            "verl.utils.logging_utils": module("verl.utils.logging_utils", configure_verl_logging=lambda: None),
        }
        self.config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(v1=types.SimpleNamespace(trainer_mode="sync")),
            skill_harness=types.SimpleNamespace(mode="skill_joint", artifact_root="/unused-root"),
            transfer_queue=types.SimpleNamespace(enable=False),
            actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(
                agent=types.SimpleNamespace(agent_loop_manager_class="factory"))),
        )

    def execute(self, failure=None):
        events = self.events

        class Trainer:
            def __init__(self, config):
                events.append("trainer-construct")
                self.logger = types.SimpleNamespace(finish=lambda **kwargs: events.append(("finish", kwargs)))

            def init(self):
                events.append("trainer-init")
                if failure == "init":
                    raise RuntimeError("initialization failed")

            def _identity(self):
                return types.SimpleNamespace(runtime="actual-runtime")

            def get_llm_client(self):
                return "solver"

            def get_teacher_client(self):
                return None

            def get_reward_handles(self):
                return None

            def fit(self, manager):
                events.append(("fit", manager))
                if failure == "fit":
                    raise RuntimeError("fit failed")

        self.direct.trainer_class = lambda: Trainer
        with patch.dict(sys.modules, self.modules), \
                patch.object(entry, "initialized_receipt", side_effect=lambda *args: events.append("all-initialized")):
            entry.runner_class()().run(self.config)

    def test_policy_precedes_construction_and_actual_profile_precedes_manager_and_fit(self):
        self.execute()
        self.assertLess(self.events.index("all-policy"), self.events.index("trainer-construct"))
        self.assertLess(self.events.index("trainer-init"), self.events.index(("register", "actual-runtime")))
        self.assertLess(self.events.index(("register", "actual-runtime")), self.events.index("all-initialized"))
        self.assertLess(self.events.index("all-initialized"), self.events.index("manager"))
        self.assertIn(("fit", "manager"), self.events)
        self.assertEqual(self.events[-3:], [("finish", {"exit_code": 0}), "tq-close", "observer-close"])

    def test_init_failure_does_not_register_or_create_model_manager(self):
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.execute("init")
        self.assertNotIn("manager", self.events)
        self.assertNotIn(("register", "actual-runtime"), self.events)
        self.assertIn("preserve-failure", self.events)
        self.assertEqual(self.events[-2:], ["tq-close", "observer-close"])

    def test_failed_profile_registration_cannot_silently_use_fork_fallback(self):
        self.policy.register_curator = lambda runtime: (_ for _ in ()).throw(ValueError("untrusted profile"))
        with self.assertRaisesRegex(ValueError, "untrusted profile"):
            self.execute()
        self.assertNotIn("manager", self.events)
        self.assertNotIn("all-initialized", self.events)
        self.assertEqual(self.events[-2:], ["tq-close", "observer-close"])

    def test_fit_failure_keeps_original_failure_and_closes_observer(self):
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.execute("fit")
        self.assertIn("preserve-failure", self.events)
        self.assertEqual(self.events[-3:], [("finish", {"exit_code": 1}), "tq-close", "observer-close"])

    def test_identity_names_actual_entry_and_explicit_frozen_components(self):
        with patch.dict(sys.modules, self.modules):
            value = entry.extension_identity(self.policy)
        self.assertEqual(value["actual_entry"], str(HERE / "scripts/train_all_native_spawn.py"))
        self.assertEqual(value["files"]["direct_component"]["path"], self.direct.__file__)
        self.assertEqual(value["solver_spawn_component_identity"], {"component": "frozen-solver-spawn"})
        self.assertEqual(value["all_role_native_spawn"]["policy"], "actual-all-native")

    def test_initialized_receipt_reports_actual_loop_profile_restore_without_old_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training-manifest.json").write_text('{"original":true}')
            mutable = root / "mutable"
            mutable.mkdir()
            restore = mutable / "DIRECT_READ_RESTORE.json"
            restore.write_text('{"full_validation_passes":1,"ema_files_copied":0}')
            loop = self.Loop()
            loop.native_spawner, loop.curator_profiles = self.policy.spawner, self.policy.curator_profiles
            self.policy.curator_profiles["custom"] = "profile"
            checkpoint = types.SimpleNamespace(checkpoint_dir="original-checkpoint", manifest_id="manifest",
                                                payload={"spec": {"optimizer_step": 8, "versions": {"code": "same"}}})
            trainer = types.SimpleNamespace(
                _joint_loop=loop, _joint_restored=types.SimpleNamespace(checkpoint=checkpoint),
                _joint_coordinator=types.SimpleNamespace(last_restore_receipt={
                    "mutable_state_root": str(mutable), "policy": "direct-read"}),
            )
            captured = []
            self.spawn._base = lambda: types.SimpleNamespace(
                _publish=lambda path, value: captured.append((path, value))
            )
            config = types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(root)))
            with patch.dict(sys.modules, self.modules), patch.object(asyncio, "get_event_loop_policy",
                                                                     return_value=self.policy):
                entry.initialized_receipt(trainer, config, {"actual": "imports"}, self.policy)
            value = captured[0][1]
            self.assertEqual(value["kind"], "initialized_all_native_spawn_entry")
            self.assertNotIn("spawn_component_initialized_ref", value)
            self.assertEqual(value["restored_checkpoint"]["spec"]["optimizer_step"], 8)
            self.assertEqual(value["restore_receipt_ref"]["sha256"], hashlib.sha256(restore.read_bytes()).hexdigest())
            self.assertTrue(captured[0][0].name.endswith(".all_native_spawn.initialized.json"))
            self.assertEqual(value["counterfactual_loop_class"], type(loop).__module__ + "." + type(loop).__qualname__)

    def test_unregistered_or_replaced_learner_loop_fails_before_initialized_publication(self):
        trainer = types.SimpleNamespace(_joint_loop=object())
        with patch.dict(sys.modules, self.modules):
            with self.assertRaisesRegex(RuntimeError, "registered all-role"):
                entry.initialized_receipt(trainer, self.config, {}, self.policy)


if __name__ == "__main__":
    unittest.main()
