"""New entry composition checks; all Ray/model/process operations are inert stubs."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


entry = load("_direct_entry_cpu", HERE / "scripts/train_direct_checkpoint.py")
restore = load("_direct_restore_cpu_entry", HERE / "scripts/direct_checkpoint_restore.py")


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events
        self.spawn = module(
            "skill_harness_examples.train_spawn_safe",
            configure_extensions=lambda config: events.append("extensions"),
            extension_identity=lambda: {"component": "frozen-spawn"},
            initialized_receipt=lambda *args: events.append("spawn-initialized") or {"path": "spawn", "sha256": "hash"},
            _source_commit=lambda root: "a" * 40,
        )
        namespace = module("skill_harness_examples", train_spawn_safe=self.spawn)
        namespace.__path__ = []
        self.modules = {
            "ray": module("ray", remote=lambda cls: cls),
            "transfer_queue": module(
                "transfer_queue", init=lambda config: events.append("tq-init"), close=lambda: events.append("tq-close")
            ),
            "omegaconf": module(
                "omegaconf",
                OmegaConf=types.SimpleNamespace(
                    resolve=lambda config: events.append("resolve"), to_container=lambda *args, **kwargs: {}
                ),
            ),
            "skill_harness_examples": namespace,
            self.spawn.__name__: self.spawn,
            "skill_harness_examples.direct_checkpoint_restore": restore,
            "skill_harness_examples.native_posix_spawn": module(
                "skill_harness_examples.native_posix_spawn", install_spawn_policy=lambda: events.append("spawn-policy")
            ),
            "verl.experimental.skill_harness.bootstrap": module(
                "verl.experimental.skill_harness.bootstrap", configure_runtime=lambda: events.append("runtime") or {}
            ),
            "verl.experimental.skill_harness.train": module(
                "verl.experimental.skill_harness.train",
                _redact=lambda value: value,
                preserve_failed_queue=lambda *args: events.append("preserve-failure"),
            ),
            "verl.utils.import_utils": module(
                "verl.utils.import_utils",
                load_class_from_fqn=lambda *args: types.SimpleNamespace(
                    create=lambda **kwargs: events.append("manager") or "manager"
                ),
            ),
            "verl.utils.logging_utils": module(
                "verl.utils.logging_utils", configure_verl_logging=lambda: events.append("logging")
            ),
        }
        self.config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(v1=types.SimpleNamespace(trainer_mode="sync")),
            skill_harness=types.SimpleNamespace(mode="skill_joint"),
            transfer_queue=types.SimpleNamespace(enable=False),
            actor_rollout_ref=types.SimpleNamespace(
                rollout=types.SimpleNamespace(agent=types.SimpleNamespace(agent_loop_manager_class="factory"))
            ),
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

        with (
            patch.dict(sys.modules, self.modules),
            patch.object(entry, "trainer_class", return_value=Trainer),
            patch.object(entry, "initialized_receipt", side_effect=lambda *args: events.append("direct-initialized")),
        ):
            entry.runner_class()().run(self.config)

    def test_spawn_policy_precedes_trainer_and_initialized_receipts_precede_fit(self):
        self.execute()
        order = self.events
        self.assertLess(order.index("spawn-policy"), order.index("trainer-construct"))
        self.assertLess(order.index("trainer-init"), order.index("spawn-initialized"))
        self.assertLess(order.index("spawn-initialized"), order.index("direct-initialized"))
        self.assertLess(order.index("direct-initialized"), order.index("manager"))
        self.assertIn(("fit", "manager"), order)
        self.assertEqual(order[-2:], [("finish", {"exit_code": 0}), "tq-close"])
        self.assertTrue(self.config.transfer_queue.enable)

    def test_init_failure_preserves_evidence_and_closes_tracking_and_queue(self):
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.execute("init")
        self.assertIn("preserve-failure", self.events)
        self.assertNotIn("direct-initialized", self.events)
        self.assertEqual(self.events[-2:], [("finish", {"exit_code": 1}), "tq-close"])

    def test_fit_failure_preserves_evidence_and_closes_tracking_and_queue(self):
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self.execute("fit")
        self.assertIn("preserve-failure", self.events)
        self.assertEqual(self.events[-2:], [("finish", {"exit_code": 1}), "tq-close"])

    def test_trainer_inherits_direct_restore_and_unchanged_frozen_fit(self):
        class Frozen:
            def fit(self, manager):
                return manager

        self.spawn.trainer_class = lambda: Frozen
        with patch.dict(sys.modules, self.modules):
            trainer = entry.trainer_class()
        self.assertTrue(issubclass(trainer, restore.DirectReadRestoreMixin))
        self.assertIs(trainer.fit, Frozen.fit)
        self.assertIs(trainer._load_checkpoint, restore.DirectReadRestoreMixin._load_checkpoint)

    def test_initialized_receipt_names_actual_entry_restore_and_spawn_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "training-manifest.json").write_text('{"original":true}')
            state = root / "mutable"
            state.mkdir()
            receipt_file = state / "DIRECT_READ_RESTORE.json"
            receipt_file.write_text('{"ema_files_copied":0,"full_validation_passes":1}')
            checkpoint = types.SimpleNamespace(
                checkpoint_dir="original-checkpoint",
                manifest_id="manifest",
                payload={"spec": {"optimizer_step": 4, "versions": {"code": "frozen"}}},
            )
            component = {"path": "component-initialized", "sha256": "component-hash"}
            trainer = types.SimpleNamespace(
                _joint_restored=types.SimpleNamespace(checkpoint=checkpoint),
                _joint_coordinator=types.SimpleNamespace(
                    last_restore_receipt={"mutable_state_root": str(state), "policy": restore.POLICY}
                ),
                _resilient_initialized_receipt=component,
            )
            captured = []
            self.spawn._base = lambda: types.SimpleNamespace(
                _publish=lambda path, value: captured.append((path, value))
            )
            config = types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(root)))
            with patch.dict(sys.modules, self.modules):
                entry.initialized_receipt(trainer, config)
            value = captured[0][1]
            self.assertEqual(value["actual_entry"], str(HERE / "scripts/train_direct_checkpoint.py"))
            self.assertEqual(
                value["files"]["restore_provider"]["path"], str(HERE / "scripts/direct_checkpoint_restore.py")
            )
            self.assertEqual(value["spawn_component_initialized_ref"], component)
            self.assertEqual(
                value["restore_receipt_ref"]["sha256"], hashlib.sha256(receipt_file.read_bytes()).hexdigest()
            )
            self.assertEqual(value["core_version_binding"], {"code": "frozen"})
            self.assertEqual(value["restored_checkpoint"]["checkpoint_dir"], "original-checkpoint")
            self.assertEqual(value["kind"], "initialized_direct_checkpoint_entry")
            self.assertTrue(captured[0][0].name.endswith(".direct_checkpoint.initialized.json"))


if __name__ == "__main__":
    unittest.main()
