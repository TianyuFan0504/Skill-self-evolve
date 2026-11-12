"""CPU lifecycle contracts; no Ray cluster, model, GPU or evaluation is run."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_quiescent_eval.py"
spec = importlib.util.spec_from_file_location("_quiescent_eval_cpu", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ParentTrainer:
    def __init__(self, root, *, final=True):
        self.events, self.clock, self.global_steps, self.total_training_steps = [], 63, 63, 64
        self.loop_closed = False
        self.fail_training = self.fail_eval = self.change_eval_clock = self.change_checkpoint_head = False
        self.head = "f" * 64
        self._joint_lock = threading.RLock()
        self.config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(val_before_train=False),
            skill_harness=types.SimpleNamespace(
                artifact_root=str(root),
                evaluation={"initial": False, "final": final},
                loading=types.SimpleNamespace(snapshot_id=None),
                store_root=str(root / "restored_state/step2/skill_store"),
            ),
            actor_rollout_ref=types.SimpleNamespace(
                rollout=types.SimpleNamespace(agent=types.SimpleNamespace(agent_loop_manager_class="fixture.Factory"))
            ),
        )
        store = types.SimpleNamespace(head=lambda: self.head, root=Path(self.config.skill_harness.store_root))

        async def stop():
            self.events.append("stop_background")
            self.background_stopped = True

        self._joint_controller = types.SimpleNamespace(store=store, stop_background=stop)
        self.background_stopped = False

    def _await(self, value):
        if self.loop_closed:
            raise AssertionError("Closed joint loop must not be used during final evaluation")
        return asyncio.run(value)

    def _ema_clock(self):
        if self.loop_closed:
            raise AssertionError("Old joint-loop clock helper is unavailable after original fit")
        return self.clock

    def _closed_loop_clock(self):
        self.events.append("metadata_clock_read")
        return self.clock

    def _save_checkpoint(self):
        self.events.append("complete_checkpoint")
        self._last_joint_checkpoint = types.SimpleNamespace(
            checkpoint_dir="fixture/global_step_64",
            manifest_id="sha256:" + "a" * 64,
            payload={
                "spec": {"optimizer_step": self.clock, "versions": {"code": "unchanged-core"}},
                "snapshots": {"skill_heads": {"main": self.head}},
            },
        )
        if self.change_checkpoint_head:
            self.head = "e" * 64
        return "checkpoint-result"

    def fit(self, manager):
        self.events.append("original_fit")
        self.agent_loop_manager = manager
        try:
            if self.fail_training:
                raise RuntimeError("original training failed")
            if self.global_steps == self.total_training_steps:
                self.events.append("no_new_optimizer_steps")
                self.global_steps += 1
                return "original-result"
            self.clock = self.global_steps = 64
            self._save_checkpoint()
            self.events.append("solver_weights64_published")
            self.global_steps = 65
            return "original-result"
        finally:
            self._await(self._joint_controller.stop_background())
            self.events.append("original_ema_cf_cleanup")
            self.loop_closed = True

    def get_llm_client(self):
        return "still-live-solver"

    def get_teacher_client(self):
        return None

    def get_reward_handles(self):
        return ["same-reward-handles"]

    def _init_dump_executor(self):
        self.events.append("open_dump")

    def _shutdown_dump_executor(self):
        self.events.append("close_dump")

    def _official_final(self):
        self.events.append("test_service_swap")
        if not self.background_stopped or not self.loop_closed:
            raise AssertionError("Test service must not replace train config while background work is alive")
        self.seen_heads = []
        for partition in ("seen", "unseen"):
            self.events.append(partition)
            self.seen_heads.append(self.agent_loop_manager.config.skill_harness.loading.snapshot_id)
        if self.change_eval_clock:
            self.clock += 1
        if self.fail_eval:
            raise RuntimeError("fixture final evaluation failed")


class Trainer(driver.QuiescentFinalMixin, ParentTrainer):
    # Explicit CPU metadata boundary; production uses real public worker RPCs.
    _closed_loop_clock = ParentTrainer._closed_loop_clock
    _official_final = ParentTrainer._official_final


class QuiescentEvalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        factory = types.SimpleNamespace(create=lambda **kwargs: types.SimpleNamespace(**kwargs))
        omega = types.ModuleType("omegaconf")
        omega.open_dict = lambda _: contextlib.nullcontext()
        imports = types.ModuleType("verl.utils.import_utils")
        imports.load_class_from_fqn = lambda *args: factory
        self.patched = patch.dict(sys.modules, {"omegaconf": omega, "verl.utils.import_utils": imports})
        self.patched.start()
        self.addCleanup(self.patched.stop)

    def test_quiesce_seals_checkpoint_before_service_and_both_splits_pin_same_head(self):
        trainer = Trainer(self.root)
        self.assertEqual(trainer.fit("old-manager"), "original-result")
        self.assertLess(trainer.events.index("stop_background"), trainer.events.index("complete_checkpoint"))
        self.assertLess(trainer.events.index("original_ema_cf_cleanup"), trainer.events.index("test_service_swap"))
        self.assertEqual(trainer.seen_heads, ["f" * 64, "f" * 64])
        self.assertIsNone(trainer.config.skill_harness.loading.snapshot_id)
        self.assertEqual(
            trainer.agent_loop_manager.config.skill_harness.store_root, trainer.config.skill_harness.store_root
        )
        self.assertEqual(trainer.agent_loop_manager.llm_client, "still-live-solver")
        self.assertEqual(trainer.clock, 64)
        self.assertEqual(trainer.global_steps, 65)
        records = list((self.root / "evaluation_lifecycle").glob("*/*.json"))
        self.assertEqual(len(records), 3)
        completed = json.loads(next(p for p in records if p.name == "evaluation_complete.json").read_text())
        self.assertEqual(completed["optimizer_step"], 64)
        self.assertEqual(completed["evaluation_cursor"], 65)
        self.assertTrue(completed["driver_sha256"])
        pointer = json.loads((self.root / "evaluation/final/lifecycle.json").read_text())
        self.assertEqual(pointer["kind"], "quiescent_final_evaluation")
        self.assertEqual(pointer["snapshot_id"], "f" * 64)
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            ref = pointer[key]
            self.assertEqual(ref["sha256"], driver.hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest())

    def test_final_false_is_original_fit_without_extra_evaluation_or_lifecycle_files(self):
        trainer = Trainer(self.root, final=False)
        self.assertEqual(trainer.fit("old-manager"), "original-result")
        self.assertEqual(trainer.events.count("stop_background"), 1)  # Original cleanup only.
        self.assertNotIn("test_service_swap", trainer.events)
        self.assertFalse((self.root / "evaluation_lifecycle").exists())

    def test_training_exception_restores_flag_and_never_starts_test(self):
        trainer = Trainer(self.root)
        trainer.fail_training = True
        with self.assertRaisesRegex(RuntimeError, "original training failed"):
            trainer.fit("old-manager")
        self.assertTrue(trainer.config.skill_harness.evaluation["final"])
        self.assertTrue(trainer.background_stopped)
        self.assertNotIn("test_service_swap", trainer.events)

    def test_evaluation_exception_still_closes_dump_executor(self):
        trainer = Trainer(self.root)
        trainer.fail_eval = True
        with self.assertRaisesRegex(RuntimeError, "fixture final evaluation failed"):
            trainer.fit("old-manager")
        self.assertEqual(trainer.events[-1], "close_dump")
        self.assertTrue(trainer.config.skill_harness.evaluation["final"])

    def test_parameter_clock_change_during_evaluation_is_not_reported_complete(self):
        trainer = Trainer(self.root)
        trainer.change_eval_clock = True
        with self.assertRaisesRegex(RuntimeError, "changed the optimizer clock"):
            trainer.fit("old-manager")
        self.assertFalse(list((self.root / "evaluation_lifecycle").glob("*/evaluation_complete.json")))

    def test_checkpoint_head_change_is_rejected_before_evaluation(self):
        trainer = Trainer(self.root)
        trainer.change_checkpoint_head = True
        with self.assertRaisesRegex(RuntimeError, "COMPLETE checkpoint"):
            trainer.fit("old-manager")
        self.assertNotIn("test_service_swap", trainer.events)

    def test_existing_failed_final_artifacts_are_never_overwritten(self):
        final = self.root / "evaluation/final"
        final.mkdir(parents=True)
        old = final / "failure.json"
        old.write_text('{"original":"failure"}\n')
        trainer = Trainer(self.root)
        with self.assertRaises(FileExistsError):
            trainer.fit("old-manager")
        self.assertEqual(old.read_text(), '{"original":"failure"}\n')
        self.assertNotIn("original_fit", trainer.events)

    def test_pretraining_evaluation_is_never_enabled(self):
        for initial, before in ((True, False), (False, True)):
            trainer = Trainer(self.root)
            trainer.config.skill_harness.evaluation["initial"] = initial
            trainer.config.trainer.val_before_train = before
            with self.assertRaisesRegex(ValueError, "before training"):
                trainer.fit("old-manager")
            self.assertNotIn("original_fit", trainer.events)

    def test_initialized_receipt_preserves_original_and_records_actual_restore_bindings(self):
        original = self.root / "training-manifest.json"
        original.write_text('{"original":"training evidence"}\n')
        before = original.read_bytes()
        checkpoint = types.SimpleNamespace(
            checkpoint_dir="original/checkpoints/global_step_2",
            manifest_id="sha256:fixture",
            payload={"spec": {"optimizer_step": 2}},
        )
        restored = str(self.root / "restored_state/step2/skill_store")
        driver.record_initialized_driver(
            self.root, {"skill_harness": {"store_root": restored}}, {"verl_version": "0.9.1"}, checkpoint
        )
        receipt = json.loads(next((self.root / "driver_runs").glob("*.initialized.json")).read_text())
        self.assertEqual(receipt["effective_config_after_trainer_init"]["skill_harness"]["store_root"], restored)
        self.assertEqual(receipt["restored_checkpoint"]["spec"]["optimizer_step"], 2)
        self.assertEqual(original.read_bytes(), before)

    def test_restored_complete_final_checkpoint_evaluates_with_no_new_update_or_save(self):
        trainer = Trainer(self.root)
        trainer.clock = trainer.global_steps = 64
        trainer._joint_restored = types.SimpleNamespace(
            checkpoint=types.SimpleNamespace(
                checkpoint_dir="existing/global_step_64",
                manifest_id="sha256:complete64",
                payload={
                    "spec": {"global_step": 64, "optimizer_step": 64, "versions": {"code": "unchanged-core"}},
                    "snapshots": {"skill_heads": {"main": trainer.head}},
                },
            )
        )
        self.assertEqual(trainer.fit("old-manager"), "original-result")
        self.assertIn("no_new_optimizer_steps", trainer.events)
        self.assertNotIn("complete_checkpoint", trainer.events)
        self.assertEqual(trainer.clock, 64)
        self.assertEqual(trainer.seen_heads, ["f" * 64, "f" * 64])
        pointer = json.loads((self.root / "evaluation/final/lifecycle.json").read_text())
        self.assertEqual(pointer["seal_origin"], "restored_complete_final_checkpoint")
        self.assertEqual(pointer["checkpoint_cursor"], 64)


class ActualOmegaConfFactoryTest(unittest.TestCase):
    def test_production_pinned_manager_uses_actual_config_copy_after_restore(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("Run this production-config contract in the existing server OmegaConf environment")
        config = OmegaConf.create(
            {
                "skill_harness": {"loading": {"enabled": True}, "store_root": "/actual/restored/skill_store"},
                "actor_rollout_ref": {"rollout": {"agent": {"agent_loop_manager_class": "fixture.Factory"}}},
            }
        )
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return "new-pinned-validation-manager"

        imports = types.ModuleType("verl.utils.import_utils")
        imports.load_class_from_fqn = lambda *args: types.SimpleNamespace(create=create)
        trainer = types.SimpleNamespace(
            config=config,
            get_llm_client=lambda: "existing-solver",
            get_teacher_client=lambda: None,
            get_reward_handles=lambda: ["reward"],
        )
        with patch.dict(sys.modules, {"verl.utils.import_utils": imports}):
            result = driver.QuiescentFinalMixin._pinned_manager(trainer, "f" * 64)
        self.assertEqual(result, "new-pinned-validation-manager")
        self.assertEqual(captured["config"].skill_harness.loading.snapshot_id, "f" * 64)
        self.assertEqual(captured["config"].skill_harness.store_root, "/actual/restored/skill_store")
        self.assertIsNone(OmegaConf.select(config, "skill_harness.loading.snapshot_id"))
        self.assertEqual(captured["llm_client"], "existing-solver")


if __name__ == "__main__":
    unittest.main()
