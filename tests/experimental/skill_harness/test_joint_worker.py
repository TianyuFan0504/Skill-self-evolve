"""CPU tests of the real stock minibatch iterator, joint hook and EMA observer.

No Worker constructor, Ray actor, process group or GPU engine is started. Only
the stock GPU train_batch adapter is replaced by a CPU Engine double; the real
train_mini_batch implementation, loss callback, optimizer and EMA are exercised.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict

from verl.experimental.skill_harness import ema_worker
from verl.experimental.skill_harness.losses import ROLE_NORMALIZATION_KEY, skill_harness_loss
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import ppo_loss

MODULE_PATH = Path(os.environ.get(
    "SKILL_JOINT_WORKER_PATH",
    Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/joint_worker.py",
))
SPEC = importlib.util.spec_from_file_location("verl.experimental.skill_harness.joint_worker", MODULE_PATH)
joint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = joint
SPEC.loader.exec_module(joint)


def actor_config(**changes):
    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1, **changes)


def mixed_data():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    prompts = torch.tensor([[10, 11], [12, 13], [14, 15], [16, 17]])
    responses = torch.tensor([[20, 21, 0], [22, 0, 0], [23, 24, 25], [26, 0, 0]])
    data = TensorDict({
        "role_id": torch.tensor([0, 0, 1, 1]), "response_mask": mask,
        "old_log_probs": torch.full((4, 3), -1.0),
        "advantages": torch.tensor([[1., 1., 0.], [-1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
        "curate_behavior_log_probs": torch.full((4, 3), -1.0),
        "curate_anchor_log_probs": torch.full((4, 3), -1.0),
        "curate_advantages": torch.tensor([0., 0., 2., -1.]),
        "curate_trajectory_count": torch.ones(4, dtype=torch.long),
        "curate_candidate_id": torch.tensor([-1, -1, 101, 102]),
        "prompts": prompts, "responses": responses,
        "input_ids": torch.nested.as_nested_tensor([
            torch.cat((prompts[i], responses[i][mask[i]])) for i in range(4)
        ], layout=torch.jagged),
    }, batch_size=[4])
    tu.assign_non_tensor(data, mini_batch_size=4, epochs=1, seed=12, dataloader_kwargs={"shuffle": True})
    tu.assign_non_tensor_stack(data, "uid", ["solver-a", "solver-b", "curate-a", "curate-b"])
    tu.assign_non_tensor_stack(data, "session_id", ["s-a", "s-b", "c-a", "c-b"])
    tu.assign_non_tensor_stack(data, "tools_kwargs", [
        {"task": {"metadata": {"game_sha256": str(i) * 64, "skill_snapshot_id": "snapshot-7"}}}
        for i in range(4)
    ])
    tu.assign_non_tensor_stack(data, "extra_fields", [{"trace": f"trace-{i}"} for i in range(4)])
    return data


def native_snapshot(data):
    result = {}
    uids, sessions = tu.get(data, "uid"), tu.get(data, "session_id")
    tools, extras = tu.get(data, "tools_kwargs"), tu.get(data, "extra_fields")
    for index, uid in enumerate(uids):
        result[uid] = {
            "session_id": sessions[index], "tools_kwargs": tools[index], "extra_fields": extras[index],
            "input_ids": data["input_ids"][index].tolist(), "prompts": data["prompts"][index].tolist(),
            "responses": data["responses"][index].tolist(), "role_id": int(data["role_id"][index]),
        }
    return result


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_probability = torch.nn.Parameter(torch.tensor(-1.0))


class CPUEngine:
    """Exercises real joint losses/microbatch backward and real optimizer hooks."""

    def __init__(self):
        self.module = TinyPolicy()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.05)
        self.calls, self.optimizer_calls, self.microbatch_calls = 0, 0, 0
        self.step_behavior = "normal"
        self.optimizer.register_step_post_hook(self._count_step)

    def _count_step(self, *_args):
        self.optimizer_calls += 1

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    def is_mp_src_rank_with_outputs(self):
        return True

    def train_mode(self, **_kwargs):
        return nullcontext()

    def train_batch(self, data, loss_function):
        self.calls += 1
        self.last_data = data
        self.normalization = tu.get(data, ROLE_NORMALIZATION_KEY)
        self.native_fields = native_snapshot(data)
        self.optimizer.zero_grad()
        total = 0.0
        for indices in ([0], [1, 2], [3]):
            micro = tu.index_select_tensor_dict(data, torch.tensor(indices))
            log_probs = self.module.log_probability.expand(len(indices), data["response_mask"].shape[-1])
            value, _metrics = loss_function(model_output={"log_probs": log_probs}, data=micro, dp_group=None)
            value.backward()
            total += float(value.detach())
            self.microbatch_calls += 1
        self.last_gradient = self.module.log_probability.grad.detach().clone()
        if self.step_behavior != "skip":
            self.optimizer.step()
            if self.step_behavior == "double":
                self.optimizer.step()
        return tu.get_tensordict({}, {"metrics": {"skill/joint_loss": total}})


def cpu_training_adapter(worker, data):
    return worker.engine.train_batch(data, loss_function=worker.loss_fn)


class JointWorkerTests(unittest.TestCase):
    def setUp(self):
        # Bypass the real constructor, which intentionally creates distributed
        # engine infrastructure. All extension methods themselves are real.
        self.worker = object.__new__(joint.JointEMATrainingWorker)
        self.worker._joint_control_lock = threading.RLock()
        self.worker._joint_update_active = False
        self.worker._rank = 0
        self.worker.profiler = SimpleNamespace(step=Mock())
        self.worker.engine = CPUEngine()
        self.worker.set_loss_fn(partial(ppo_loss, config=actor_config()))
        self.worker.initialize_ema(decay=0.5)

    def run_update(self, data):
        # Stock's native input_ids accounting gathers token lengths. Mock only
        # this one-rank accounting collective; no process group is initialized.
        def gather(output, value, *_args, **_kwargs):
            output[0] = value

        with (
            patch.object(TrainingWorker, "train_batch", cpu_training_adapter),
            patch.object(torch.distributed, "get_world_size", return_value=1),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            return self.worker.train_mini_batch(data)

    def test_installs_role_loss_through_stock_public_setter(self):
        self.assertIs(self.worker.loss_fn.func, skill_harness_loss)
        metadata = self.worker.joint_training_metadata()
        self.assertTrue(metadata["single_optimizer_step_per_rpc"])
        self.assertEqual(metadata["solver_role_id"], 0)
        self.assertEqual(metadata["curator_role_id"], 1)
        self.assertEqual(metadata["curate"]["coefficient"], 1)

    def test_real_stock_minibatch_boundary_one_optimizer_step_and_ema(self):
        data = mixed_data()
        original = native_snapshot(data)
        self.assertNotIn(ROLE_NORMALIZATION_KEY, data)
        with patch.object(joint, "prepare_role_batch_metadata", wraps=joint.prepare_role_batch_metadata) as prepare:
            output = self.run_update(data)
        self.assertEqual(prepare.call_count, 1)
        selected_data = prepare.call_args.args[0]
        self.assertIsNot(selected_data, data)  # stock iterator selected/shuffled it first
        self.assertIs(self.worker.engine.last_data, selected_data)
        self.assertEqual(self.worker.engine.normalization, {
            "dp_size": 1, "solver_tokens": 3, "solver_sequences": 2, "curate_sequences": 2,
        })
        self.assertEqual(self.worker.engine.native_fields, original)
        self.assertEqual(self.worker.engine.calls, 1)
        self.assertEqual(self.worker.engine.optimizer_calls, 1)
        self.assertEqual(self.worker.engine.microbatch_calls, 3)
        self.assertEqual(self.worker._observed_optimizer_step, 1)
        self.assertEqual(self.worker.parameter_ema.num_updates, 1)
        online = self.worker.engine.module.log_probability.detach()
        shadow = self.worker.parameter_ema.parameters["log_probability"]
        torch.testing.assert_close(shadow, (-1 + online) / 2)
        self.assertFalse(shadow.requires_grad)
        self.worker.profiler.step.assert_called_once()
        self.assertIn("skill/joint_loss", tu.get(output, "metrics"))

    def test_rejects_multiple_optimizer_updates_before_any_training(self):
        for settings in ({"epochs": 2}, {"mini_batch_size": 2}, {"num_mini_batch": 2}):
            with self.subTest(settings=settings):
                data = mixed_data()
                tu.assign_non_tensor(data, **settings)
                with self.assertRaises(ValueError):
                    self.run_update(data)
        self.assertEqual(self.worker.engine.calls, 0)
        self.assertEqual(self.worker.engine.optimizer_calls, 0)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)

    def test_skipped_optimizer_does_not_advance_ema_or_fake_a_clock(self):
        self.worker.engine.step_behavior = "skip"
        with self.assertRaisesRegex(RuntimeError, "Optimizer skipped"):
            self.run_update(mixed_data())
        self.assertEqual(self.worker._observed_optimizer_step, 0)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)
        self.worker._ready()

    def test_double_optimizer_step_is_detected_without_ema_update(self):
        self.worker.engine.step_behavior = "double"
        with self.assertRaisesRegex(RuntimeError, "multiple updates"):
            self.run_update(mixed_data())
        self.assertEqual(self.worker._observed_optimizer_step, 2)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)
        with self.assertRaisesRegex(RuntimeError, "not been incorporated"):
            self.worker._ready()

    def test_loss_failure_before_step_is_recoverable_and_not_marked_applied(self):
        data = mixed_data()
        del data["curate_behavior_log_probs"]
        with self.assertRaisesRegex(ValueError, "curate_behavior_log_probs"):
            self.run_update(data)
        self.assertEqual(self.worker.engine.optimizer_calls, 0)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)
        self.assertFalse(self.worker._joint_update_active)
        self.worker._ready()
        self.run_update(mixed_data())
        self.assertEqual(self.worker.engine.optimizer_calls, 1)
        self.assertEqual(self.worker.parameter_ema.num_updates, 1)

    def test_configuration_is_validated_before_replacing_loss(self):
        result = self.worker.configure_joint_loss(coefficient=0.4, historical_is_cap=3)
        self.assertEqual(result["curate"]["coefficient"], 0.4)
        self.assertEqual(result["curate"]["historical_is_cap"], 3)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)
        previous = self.worker.loss_fn
        with self.assertRaises(ValueError):
            self.worker.configure_joint_loss(coefficient=-1)
        self.assertIs(self.worker.loss_fn, previous)
        with self.assertRaises(TypeError):
            self.worker.set_loss_fn(lambda **_kwargs: None)
        with self.assertRaises(ValueError):
            self.worker.set_loss_fn(partial(lambda **_kwargs: None, config=actor_config()))
        self.assertIs(self.worker.loss_fn, previous)

    def test_reconfiguration_during_update_or_export_fails(self):
        self.worker._joint_update_active = True
        try:
            with self.assertRaisesRegex(RuntimeError, "cannot overlap"):
                self.worker.configure_joint_loss(coefficient=0.3)
        finally:
            self.worker._joint_update_active = False
        self.worker.parameter_ema._swapped = True
        try:
            with self.assertRaisesRegex(RuntimeError, "export is active"):
                self.worker.configure_joint_loss(coefficient=0.3)
        finally:
            self.worker.parameter_ema._swapped = False

    def test_another_thread_cannot_start_overlapping_optimizer_update(self):
        errors = []

        def attempt():
            try:
                self.worker.configure_joint_loss(coefficient=0.2)
            except RuntimeError as error:
                errors.append(str(error))

        with self.worker._joint_control_lock:
            thread = threading.Thread(target=attempt)
            thread.start()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn("in progress", errors[0])

    def test_actor_subclass_preserves_native_update_and_delegates_configuration(self):
        self.assertTrue(issubclass(joint.JointEMAActorRolloutRefWorker, ema_worker.EMAActorRolloutRefWorker))
        self.assertIs(joint.JointEMAActorRolloutRefWorker.actor_worker_cls, joint.JointEMATrainingWorker)
        self.assertIs(joint.JointEMAActorRolloutRefWorker.update_actor,
                      ema_worker.EMAActorRolloutRefWorker.update_actor)
        actor = object.__new__(joint.JointEMAActorRolloutRefWorker)
        actor._is_actor, actor.actor = True, self.worker
        self.assertEqual(actor.configure_joint_loss(coefficient=0.25)["curate"]["coefficient"], 0.25)
        self.assertEqual(actor.get_joint_training_metadata()["curate"]["coefficient"], 0.25)
        actor._is_actor = False
        with self.assertRaisesRegex(RuntimeError, "initialized actor"):
            actor.configure_joint_loss(coefficient=0.25)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
