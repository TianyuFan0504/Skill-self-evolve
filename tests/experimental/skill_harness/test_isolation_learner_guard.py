"""Persisted host failures block the actual learner method before optimizer RPC."""

from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_actual_account_incidents", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py"
)
accounts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accounts)


class IsolationLearnerGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        path = ROOT / "verl/experimental/skill_harness/joint_trainer.py"
        tree = ast.parse(path.read_text())
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkillJointSyncTrainer"
        )
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_actor")
        scope = {
            "assert_no_isolation_incidents": accounts.assert_no_isolation_incidents,
            "assert_no_nontext_incidents": Mock(),
            "normalize_actor_metrics": lambda _output: {},
            "Mapping": Mapping,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        self.update = scope["_update_actor"]
        self.prepared = SimpleNamespace(data="actual prepared boundary", expected_optimizer_step=1, acknowledgments=[])
        controller = SimpleNamespace(
            prepare_update=Mock(return_value=self.prepared),
            after_update_and_sync=Mock(return_value={}),
            settings=SimpleNamespace(curator_gpus=2, credit_gpus=2),
        )
        self.holder = SimpleNamespace(
            _joint_lock=nullcontext(),
            config=SimpleNamespace(skill_harness=SimpleNamespace(artifact_root=str(self.root))),
            _joint_controller=controller,
            _materialize_solver=Mock(return_value={"input_ids": [101, 102]}),
            _await=lambda value: value,
            _ema_clock=lambda: 1,
            global_steps=1,
            actor_rollout_wg=SimpleNamespace(update_actor=Mock(), world_size=4),
        )

    def test_clean_batch_preserves_one_optimizer_rpc(self):
        batch, metrics = SimpleNamespace(), {}
        self.assertIs(self.update(self.holder, batch, metrics), batch)
        self.holder.actor_rollout_wg.update_actor.assert_called_once_with(self.prepared.data)
        self.assertEqual(metrics["skill/optimizer_step"], 1)

    def test_filtered_failed_task_cannot_hide_a_persisted_host_incident(self):
        accounts.record_isolation_incident(self.root, "CPU failed launcher fixture")
        with self.assertRaises(accounts.AccountIsolationError):
            self.update(self.holder, SimpleNamespace(), {})
        self.holder._materialize_solver.assert_not_called()
        self.holder._joint_controller.prepare_update.assert_not_called()
        self.holder.actor_rollout_wg.update_actor.assert_not_called()

    def test_isolation_failure_during_probe_prepare_blocks_update(self):
        def prepare(*_):
            accounts.record_isolation_incident(self.root, "CPU failed paired cleanup fixture")
            return self.prepared

        self.holder._joint_controller.prepare_update.side_effect = prepare
        with self.assertRaises(accounts.AccountIsolationError):
            self.update(self.holder, SimpleNamespace(), {})
        self.holder.actor_rollout_wg.update_actor.assert_not_called()

    def test_interrupted_incident_detail_write_still_blocks_update(self):
        with patch.object(accounts, "_atomic", side_effect=OSError("CPU interrupted detail write")):
            with self.assertRaises(OSError):
                accounts.record_isolation_incident(self.root, "CPU interrupted quarantine fixture")
        with self.assertRaises(accounts.AccountIsolationError):
            self.update(self.holder, SimpleNamespace(), {})
        self.holder.actor_rollout_wg.update_actor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
