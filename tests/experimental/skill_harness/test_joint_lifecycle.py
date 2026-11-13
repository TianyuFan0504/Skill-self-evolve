"""CPU lifecycle checks over the actual trainer methods, with service boundaries stubbed."""

from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = ROOT / "verl/experimental/skill_harness/joint_trainer.py"


class Options(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class FinalLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.events = Path(temporary.name), []
        self.clock, self.head = 2, "snapshot:final"
        test = self

        class Parent:
            def fit(self, manager):
                test.events.append("fit")
                self.agent_loop_manager = manager
                if getattr(self, "fail_fit", False):
                    raise RuntimeError("solver rollout failed")
                if not getattr(self, "skip_training", False):
                    self._save_checkpoint()
                    self.global_steps += 1
                return "fit-result"

            def _save_checkpoint(self):
                test.events.append("native_save")
                self._last_joint_checkpoint = SimpleNamespace(
                    checkpoint_dir=str(test.root / "checkpoints/global_step_2"), manifest_id="manifest:complete",
                    payload={"spec": {"global_step": self.global_steps, "optimizer_step": test.clock},
                             "snapshots": {"skill_heads": {"main": test.head}}},
                )

        tree = ast.parse(TRAINER_PATH.read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "SkillJointSyncTrainer")
        names = {"_stop_background", "_checkpoint_seal", "_save_checkpoint", "_assert_final_seal",
                 "_pinned_validation_manager", "_evaluate_final", "_shutdown_joint_services", "fit"}
        selected = copy.deepcopy(original)
        selected.bases = [ast.Name(id="Parent", ctx=ast.Load())]
        selected.body = [method for method in selected.body if isinstance(method, ast.FunctionDef)
                         and method.name in names]
        module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
        self.factory = Mock()

        def create(**kwargs):
            self.events.append("pinned_manager")
            return SimpleNamespace(config=kwargs["config"])

        self.factory.create.side_effect = create
        package = "_joint_lifecycle_cpu"
        namespace = {"__name__": package + ".joint_trainer", "__package__": package,
                     "Parent": Parent, "Path": Path, "json": json, "copy": copy, "sys": sys,
                     "open_dict": nullcontext, "load_class_from_fqn": Mock(return_value=self.factory)}
        exec(compile(module, str(TRAINER_PATH), "exec"), namespace)
        self.driver = object.__new__(namespace["SkillJointSyncTrainer"])
        driver = self.driver
        driver.config = Options(
            skill_harness=Options(evaluation=Options(final=True),
                                  loading=Options(snapshot_id=None), store_root="restored/skill_store"),
            actor_rollout_ref=Options(rollout=Options(agent=Options(agent_loop_manager_class="Manager"))),
        )
        driver.artifact_root = self.root
        driver._joint_lock = threading.RLock()
        driver._joint_store = SimpleNamespace(head=lambda: self.head)
        driver._joint_restored = None
        driver._await = lambda value: value
        driver._ema_clock = lambda: self.clock
        driver.global_steps = driver.total_training_steps = 2
        driver._joint_loop = SimpleNamespace(is_running=lambda: False, close=Mock())
        driver._background_stopped = False
        driver.get_llm_client = lambda: "solver-client"
        driver.get_teacher_client = lambda: None
        driver.get_reward_handles = lambda: None
        driver._init_dump_executor = lambda: self.events.append("writer_open")
        driver._shutdown_dump_executor = lambda: self.events.append("writer_close")
        driver._joint_controller = SimpleNamespace(
            stop_background=lambda: self.events.append("background_stop"),
            pause_background_writes=lambda: self.events.append("pause_writes"),
            resume_background_writes=lambda _state: self.events.append("resume_writes"),
            ema_gateway=SimpleNamespace(shutdown=lambda: self.events.append("ema_shutdown")),
            frozen_gateway=SimpleNamespace(shutdown=lambda: self.events.append("frozen_shutdown")),
        )
        evaluation = types.ModuleType(package + ".evaluation")
        evaluation.evaluate_official_splits = Mock(side_effect=self.official_evaluation)
        self.evaluation = evaluation.evaluate_official_splits
        self.modules = patch.dict(sys.modules, {package + ".evaluation": evaluation})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def official_evaluation(self, trainer):
        self.events.append("official_final")
        self.assertEqual(trainer.agent_loop_manager.config.skill_harness.loading.snapshot_id, self.head)

    def test_sole_trainer_directly_extends_supported_sync_controller(self):
        tree = ast.parse(TRAINER_PATH.read_text())
        trainers = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("Trainer")]
        self.assertEqual([node.name for node in trainers], ["SkillJointSyncTrainer"])
        self.assertEqual([ast.unparse(base) for base in trainers[0].bases], ["PPOTrainerSync"])
        entry = ast.parse((TRAINER_PATH.parent / "train.py").read_text())
        runners = [node for node in entry.body if isinstance(node, ast.ClassDef)]
        self.assertEqual([node.name for node in runners], ["SkillTaskRunner"])
        self.assertEqual([ast.unparse(item) for item in runners[0].decorator_list], ["ray.remote"])
        self.assertFalse((TRAINER_PATH.parent / "trainer.py").exists())

    def test_final_checkpoint_stops_collection_before_save_and_pins_eval_snapshot(self):
        self.assertEqual(self.driver.fit("training-manager"), "fit-result")
        order = [self.events.index(name) for name in (
            "background_stop", "native_save", "ema_shutdown", "pinned_manager", "official_final",
        )]
        self.assertEqual(order, sorted(order))
        self.assertEqual(self.events.count("background_stop"), 1)
        self.assertNotIn("resume_writes", self.events)
        self.assertEqual(self.driver.config.skill_harness.loading.snapshot_id, None)
        self.assertEqual(self.factory.create.call_args.kwargs["config"].skill_harness.store_root,
                         "restored/skill_store")
        self.assertIsNot(self.factory.create.call_args.kwargs["config"], self.driver.config)
        lifecycle = json.loads((self.root / "evaluation/final/complete.json").read_text())
        self.assertEqual(lifecycle["optimizer_updates_during_test"], 0)
        self.assertEqual(lifecycle["snapshot_id"], self.head)
        self.driver._joint_loop.close.assert_called_once()

    def test_failed_training_never_runs_official_evaluation(self):
        self.driver.fail_fit = True
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            self.driver.fit("training-manager")
        self.evaluation.assert_not_called()
        self.assertEqual(self.events.count("background_stop"), 1)
        self.driver._joint_loop.close.assert_called_once()

    def test_eval_policy_or_library_mutation_prevents_completion_record(self):
        def mutating_evaluation(*_args, **_kwargs):
            self.head = "snapshot:unexpected"

        self.evaluation.side_effect = mutating_evaluation
        with self.assertRaisesRegex(RuntimeError, "policy or library changed"):
            self.driver.fit("training-manager")
        self.assertFalse((self.root / "evaluation/final/complete.json").exists())
        self.assertEqual(self.events[-1], "writer_close")

    def test_final_eval_rejects_a_fresh_untrained_checkpoint(self):
        self.clock = 0
        with self.assertRaisesRegex(RuntimeError, "completed trained"):
            self.driver.fit("training-manager")
        self.evaluation.assert_not_called()
        self.assertNotIn("pinned_manager", self.events)

    def test_restored_checkpoint_is_evaluated_without_another_model_save(self):
        self.driver.skip_training = True
        self.driver._joint_restored = SimpleNamespace(checkpoint=SimpleNamespace(
            checkpoint_dir="immutable/checkpoint", manifest_id="restored:complete",
            payload={"spec": {"global_step": 2, "optimizer_step": 2},
                     "snapshots": {"skill_heads": {"main": self.head}}},
        ))
        self.driver._joint_controller = None
        self.driver.fit("training-manager")
        self.assertNotIn("native_save", self.events)
        self.evaluation.assert_called_once()
        lifecycle = json.loads((self.root / "evaluation/final/complete.json").read_text())
        self.assertEqual(lifecycle["manifest_id"], "restored:complete")

    def test_existing_final_outputs_are_preserved(self):
        output = self.root / "evaluation/final"
        output.mkdir(parents=True)
        (output / "existing.json").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.driver.fit("training-manager")
        self.assertNotIn("fit", self.events)
        self.evaluation.assert_not_called()
        self.assertEqual((output / "existing.json").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
