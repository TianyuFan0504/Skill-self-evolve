"""A learner validates its own batch without consulting run-global incidents."""

from __future__ import annotations

import ast
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]


class LearnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        path = ROOT / "verl/experimental/skill_harness/joint_trainer.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "SkillJointSyncTrainer")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_actor")
        scope = {"normalize_actor_metrics": lambda _output: {}, "Mapping": Mapping}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        self.update = scope["_update_actor"]
        self.prepared = SimpleNamespace(data="prepared native batch", expected_optimizer_step=1, acknowledgments=[])
        self.controller = SimpleNamespace(
            prepare_update=Mock(return_value=self.prepared), after_update_and_sync=Mock(return_value={}),
            settings=SimpleNamespace(curator_gpus=2, credit_gpus=2),
        )
        self.holder = SimpleNamespace(
            _joint_lock=nullcontext(),
            config=SimpleNamespace(skill_harness=SimpleNamespace(artifact_root=str(self.root))),
            _joint_controller=self.controller,
            _materialize_solver=Mock(return_value={"input_ids": [101, 102]}),
            _await=lambda value: value,
            _ema_clock=lambda: 1,
            global_steps=1,
            actor_rollout_wg=SimpleNamespace(update_actor=Mock(), world_size=4),
        )

    def test_unrelated_incident_directories_do_not_gate_a_valid_batch(self):
        for relative in ("isolation_incidents", "protocol_incidents/unsupported_media"):
            directory = self.root / relative
            directory.mkdir(parents=True)
            (directory / "old.json").write_text("unrelated old failed probe")
        batch, metrics = SimpleNamespace(), {}
        self.assertIs(self.update(self.holder, batch, metrics), batch)
        self.holder.actor_rollout_wg.update_actor.assert_called_once_with(self.prepared.data)
        self.assertEqual(metrics["skill/optimizer_step"], 1)
        self.controller.after_update_and_sync.assert_called_once()

    def test_invalid_current_batch_fails_before_preparation_or_optimizer(self):
        self.holder._materialize_solver.side_effect = ValueError("nonempty current media")
        with self.assertRaisesRegex(ValueError, "current media"):
            self.update(self.holder, SimpleNamespace(), {})
        self.controller.prepare_update.assert_not_called()
        self.holder.actor_rollout_wg.update_actor.assert_not_called()

    def test_failed_current_preparation_never_issues_an_optimizer_call(self):
        self.controller.prepare_update.side_effect = ValueError("invalid replay alignment")
        with self.assertRaisesRegex(ValueError, "replay alignment"):
            self.update(self.holder, SimpleNamespace(), {})
        self.holder.actor_rollout_wg.update_actor.assert_not_called()
        self.controller.after_update_and_sync.assert_not_called()

    def test_unobserved_optimizer_clock_does_not_acknowledge_credit(self):
        self.holder._ema_clock = lambda: 0
        with self.assertRaisesRegex(RuntimeError, "expected observed optimizer step"):
            self.update(self.holder, SimpleNamespace(), {})
        self.holder.actor_rollout_wg.update_actor.assert_called_once()
        self.controller.after_update_and_sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
