"""CPU contracts for immutable token replay and native joint TensorDict assembly."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from verl.experimental.skill_harness.alfworld.data import stable_game_id
from verl.experimental.skill_harness.credit import CreditLedger, CreditProtocol, EvaluationOutcome, UsageEvent
from verl.experimental.skill_harness.replay import CurateReplay, CurateReplayRecord
from verl.utils import tensordict_utils as tu

MODULE_PATH = Path(os.environ.get(
    "SKILL_REPLAY_BATCH_PATH",
    Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/replay_batch.py",
))
SPEC = importlib.util.spec_from_file_location("verl.experimental.skill_harness.replay_batch", MODULE_PATH)
batcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = batcher
SPEC.loader.exec_module(batcher)


def metadata(*, split="train", phase="train"):
    path = f"json_2.1.1/{split}/pick-task/trial-1/game.tw-pddl"
    digest = "a" * 64
    return {"game_id": stable_game_id(path, digest), "relative_game_path": path, "game_sha256": digest,
            "catalog_digest": "b" * 64, "split": split, "dataset_phase": phase,
            "task_type": "pick_and_place_simple", "scene_id": "room-1"}


def solver_batch(planes=4):
    prompts, responses = [torch.tensor([10, 11]), torch.tensor([12])], [torch.tensor([20, 21]), torch.tensor([22])]
    ids = [torch.cat((p, r)) for p, r in zip(prompts, responses, strict=False)]
    data = tu.get_tensordict({
        "prompts": batcher._nested(prompts), "responses": batcher._nested(responses),
        "input_ids": batcher._nested(ids),
        "position_ids": batcher._nested([batcher._position_ids(x.numel(), planes) for x in ids]),
        "response_mask": batcher._nested([torch.tensor([1, 1]), torch.tensor([1])]),
        "loss_mask": batcher._nested([torch.tensor([1, 1]), torch.tensor([1])]),
        "old_log_probs": batcher._nested([torch.tensor([-0.5, -0.6]), torch.tensor([-0.7])]),
        "advantages": batcher._nested([torch.tensor([1., 1.]), torch.tensor([-1.])]),
        "ref_log_prob": batcher._nested([torch.tensor([-0.8, -0.9]), torch.tensor([-1.])]),
        "rollout_is_weights": batcher._nested([torch.tensor([0.9, 1.1]), torch.tensor([1.])]),
        "uid": ["solver-a", "solver-b"], "session_id": ["sample-a", "sample-b"],
        "tools_kwargs": [{"task": {"metadata": metadata()}}] * 2,
        "extra_fields": [{"trace": "trace-a"}, {"trace": "trace-b"}],
        "multi_modal_inputs": [{}, {}],
    }, {"temperature": 0.7, "calculate_entropy": True, "global_batch_size": 2, "mini_batch_size": 2,
        "epochs": 3, "pad_mode": "no_padding"})
    return data


class ReplayBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.replay = CurateReplay(Path(temporary.name) / "replay")
        self.ledger = CreditLedger(Path(temporary.name) / "ledger", self.replay)

    def settled(self, identity="record-1", *, delta=1, task_metadata=None):
        digest = hashlib.sha256(identity.encode()).hexdigest()
        skill = {"skill_id": "skill-" + identity, "version": "sha256:" + digest, "content_hash": digest}
        record = CurateReplayRecord(
            record_id=identity, producer_uid="producer-" + identity, skill_ref=skill,
            prompt_ids=(100, 101, 102), action_ids=(200, 201, 202), curate_mask=(1, 0, 1),
            behavior_logprobs=(-0.2, 0, -0.8), behavior_policy_version="ema:old",
            sampling_config={"temperature": 1., "top_p": 1., "top_k": -1, "logprobs_mode": "processed_logprobs"},
            tokenizer_version="tok:v1", template_version="template:v1",
            context_snapshot={"task_metadata": task_metadata or metadata(), "context_hash": "context:v1"},
            skill_snapshot_id="skills:v1", candidate_group_id="group-" + identity,
            candidate_index=0, candidate_group_size=1, created_at=10,
        )
        self.replay.append(record)
        proto = CreditProtocol("evaluator:v1", "probe:v1", "train:v1", "env:v1", "skills:base", 1)
        window = "window-" + identity
        self.ledger.register_window(window, identity, proto, opened_at=11, deadline_at=100)
        common = {"task_id": "probe-" + identity, "model_version": proto.evaluator_version,
                  "environment_version": proto.environment_version, "environment_state_hash": "c" * 64,
                  "baseline_snapshot_id": proto.baseline_snapshot_id}
        event = UsageEvent("event-" + identity, window, identity, skill, "probe-1", proto.protocol_id,
                           EvaluationOutcome(**common, score=max(delta, 0), trajectory_ref="with.json",
                                             loaded_skill_ref=skill),
                           EvaluationOutcome(**common, score=max(-delta, 0), trajectory_ref="without.json"), 20)
        self.ledger.add_usage(event, now=30)
        label = self.ledger.settle(window, now=30)
        self.ledger.lease(label.settlement_id, "update-1", owner="learner", lease_seconds=100, now=40)
        return record, label

    def prepare(self, labels, **changes):
        arguments = {"update_id": "update-1", "owner": "learner", "anchor_policy_version": "online:7",
                     "tokenizer_version": "tok:v1", "template_version": "template:v1", "dp_size": 2,
                     "pad_token_id": 0, "position_id_planes": batcher.infer_position_id_planes(solver_batch()),
                     "now": 50}
        arguments.update(changes)
        return batcher.prepare_curate_anchor_inputs(self.replay, self.ledger,
                                                    [item.settlement_id for item in labels], **arguments)

    def anchor_output(self, plan):
        return TensorDict({"log_probs": batcher._nested([
            torch.full((ids.numel(),), -0.3) for ids in batcher._rows(plan.data["input_ids"])
        ])}, batch_size=[len(plan.data)])

    def finalize(self, plan, **changes):
        arguments = {"ledger": self.ledger, "anchor_policy_version": plan.anchor_policy_version,
                     "anchor_input_fingerprint": plan.fingerprint, "solver_temperature": 0.7, "now": 51}
        arguments.update(changes)
        return batcher.finalize_joint_batch(solver_batch(), plan, self.anchor_output(plan), **arguments)

    def test_exact_archived_tokens_masks_mu_and_native_four_plane_positions(self):
        record, label = self.settled()
        plan = self.prepare([label])
        self.assertEqual(len(plan.data), 2)  # one actual row, one DP computational pad
        self.assertEqual(plan.data["prompts"][0].tolist(), list(record.prompt_ids))
        self.assertEqual(plan.data["responses"][0].tolist(), list(record.action_ids))
        self.assertEqual(plan.data["input_ids"][0].tolist(), list(record.prompt_ids + record.action_ids))
        self.assertEqual(plan.data["response_mask"][0].tolist(), [1, 0, 1])
        torch.testing.assert_close(plan.data["curate_behavior_log_probs"][0], torch.tensor([-0.2, 0, -0.8]))
        self.assertEqual(plan.data["position_ids"][0].shape, (4, 6))
        self.assertEqual(plan.lineages[0].record_payload_hash, self.replay.get_ref(record.record_id).payload_hash)
        self.assertFalse(tu.get(plan.data, "compute_loss"))
        self.assertEqual(tu.get(plan.data, "temperature"), 1)
        self.assertEqual(plan.leased_ids, (label.settlement_id,))

    def test_solver_only_startup_preserves_grpo_and_has_no_synthetic_curate_credit(self):
        original = solver_batch()
        result = batcher.prepare_solver_only_batch(original, dp_size=4, pad_token_id=0)
        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.lineages, ())
        self.assertEqual(result.acknowledgments, ())
        self.assertEqual(result.anchor_policy_version, "")
        self.assertEqual(result.data["role_id"].tolist(), [0, 0, 0, 0])
        self.assertEqual(result.data["curate_candidate_id"].tolist(), [-1, -1, -1, -1])
        for index in range(2):
            torch.testing.assert_close(result.data["advantages"][index], original["advantages"][index])
            torch.testing.assert_close(result.data["input_ids"][index], original["input_ids"][index])
            self.assertFalse(result.data["curate_token_mask"][index].any())
        for index in (2, 3):
            self.assertFalse(result.data["response_mask"][index].any())
        self.assertEqual(tu.get(result.data, "mini_batch_size"), 4)
        self.assertEqual(tu.get(result.data, "epochs"), 1)

    def test_joint_batch_preserves_solver_grpo_and_zeros_curator_solver_losses(self):
        record, label = self.settled()
        plan = self.prepare([label])
        solver = solver_batch()
        result = batcher.finalize_joint_batch(solver, plan, self.anchor_output(plan), ledger=self.ledger,
                                             anchor_policy_version=plan.anchor_policy_version,
                                             anchor_input_fingerprint=plan.fingerprint, solver_temperature=0.7, now=51)
        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.data["role_id"].tolist(), [0, 0, 1, 0])
        for field in ("input_ids", "position_ids", "old_log_probs", "advantages", "ref_log_prob"):
            for index in (0, 1):
                torch.testing.assert_close(result.data[field][index], solver[field][index])
        self.assertEqual(result.data["advantages"][2].tolist(), [0, 0, 0])
        self.assertEqual(result.data["solver_token_mask"][2].tolist(), [False, False, False])
        self.assertEqual(result.data["curate_advantages"].tolist(), [0, 0, 1, 0])
        torch.testing.assert_close(result.data["curate_anchor_log_probs"][2], torch.tensor([-0.3, 0, -0.3]))
        self.assertEqual(result.data["responses"][2].tolist(), list(record.action_ids))
        self.assertFalse(result.data["response_mask"][3].any())
        self.assertEqual(result.lineages[0].row_index, 2)
        self.assertEqual(result.acknowledgments[0]["settlement_id"], label.settlement_id)
        for key in ("global_batch_size", "mini_batch_size"):
            self.assertEqual(tu.get(result.data, key), 4)
        self.assertEqual(tu.get(result.data, "epochs"), 1)
        torch.testing.assert_close(result.data["temperature"], torch.tensor([0.7, 0.7, 1, 1]))
        self.assertEqual(tu.get(solver, "epochs"), 3)  # source never mutated
        self.assertEqual(self.ledger.training_status(label.settlement_id, now=51)["state"], "leased")

    def test_positive_zero_and_negative_labels_stay_uncentered_and_ids_stable(self):
        labels = [self.settled(f"record-{i}", delta=delta)[1] for i, delta in enumerate((1, 0, -1))]
        plan = self.prepare(labels)
        self.assertEqual(plan.data["curate_advantages"].tolist(), [1, 0, -1, 0])
        again = self.prepare(labels)
        self.assertEqual(plan.fingerprint, again.fingerprint)
        self.assertEqual(plan.data["curate_candidate_id"].tolist(), again.data["curate_candidate_id"].tolist())
        self.assertEqual(len({item.candidate_id for item in plan.lineages}), 3)

    def test_complete_native_joint_batch_runs_public_loss_and_excludes_dp_padding(self):
        from verl.experimental.skill_harness.losses import prepare_role_batch_metadata, skill_harness_loss
        from verl.workers.config import ActorConfig

        _, label = self.settled()
        result = self.finalize(self.prepare([label]))
        norm = prepare_role_batch_metadata(result.data)
        self.assertEqual((norm.solver_tokens, norm.solver_sequences, norm.curate_sequences), (3, 2, 1))
        value = torch.nn.Parameter(torch.tensor(-0.3))
        output = {"log_probs": batcher._nested([
            value.expand(ids.numel()) for ids in batcher._rows(result.data["input_ids"])
        ])}
        config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
                             use_kl_loss=True, kl_loss_coef=0.01)
        loss, _ = skill_harness_loss(config, output, result.data)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(value.grad))
        self.assertNotEqual(float(value.grad), 0)

    def test_anchor_version_fingerprint_or_changed_mask_is_rejected(self):
        _, label = self.settled()
        plan = self.prepare([label])
        with self.assertRaisesRegex(ValueError, "attestation"):
            self.finalize(plan, anchor_policy_version="online:8")
        with self.assertRaisesRegex(ValueError, "attestation"):
            self.finalize(plan, anchor_input_fingerprint="wrong")
        plan.data["curate_token_mask"].values()[0] = False
        with self.assertRaisesRegex(ValueError, "attestation"):
            self.finalize(plan)

    def test_active_lease_runtime_versions_and_complete_anchor_are_required(self):
        _, label = self.settled()
        with self.assertRaisesRegex(ValueError, "lease"):
            self.prepare([label], owner="other")
        with self.assertRaisesRegex(ValueError, "tokenizer/template"):
            self.prepare([label], tokenizer_version="tok:v2")
        plan = self.prepare([label])
        with self.assertRaisesRegex(ValueError, "lease"):
            self.finalize(plan, now=141)
        wrong = TensorDict({"log_probs": batcher._nested([torch.zeros(5), torch.zeros(2)])}, batch_size=[2])
        with self.assertRaisesRegex(ValueError, "cover every"):
            batcher.finalize_joint_batch(solver_batch(), plan, wrong, ledger=self.ledger,
                                         anchor_policy_version=plan.anchor_policy_version,
                                         anchor_input_fingerprint=plan.fingerprint, solver_temperature=1, now=51)

    def test_dev_test_producers_and_solver_rows_cannot_enter_training(self):
        for index, source in enumerate((metadata(phase="dev"), metadata(split="valid_unseen", phase="test_unseen"))):
            _, label = self.settled(f"bad-{index}", task_metadata=source)
            with self.assertRaisesRegex(ValueError, "train"):
                self.prepare([label])
        _, label = self.settled()
        plan = self.prepare([label])
        solver = solver_batch()
        tu.assign_non_tensor_stack(solver, "tools_kwargs", [{"task": {"metadata": metadata(phase="dev")}}] * 2)
        with self.assertRaisesRegex(ValueError, "train"):
            batcher.finalize_joint_batch(solver, plan, self.anchor_output(plan), ledger=self.ledger,
                                         anchor_policy_version=plan.anchor_policy_version,
                                         anchor_input_fingerprint=plan.fingerprint, solver_temperature=1, now=51)

    def test_missing_grpo_and_unknown_engine_tensor_fail_closed(self):
        _, label = self.settled()
        plan = self.prepare([label])
        solver = solver_batch()
        del solver["advantages"]
        with self.assertRaisesRegex(ValueError, "AFTER solver GRPO"):
            batcher.finalize_joint_batch(solver, plan, self.anchor_output(plan), ledger=self.ledger,
                                         anchor_policy_version=plan.anchor_policy_version,
                                         anchor_input_fingerprint=plan.fingerprint, solver_temperature=1, now=51)
        solver = solver_batch()
        solver["routed_experts"] = torch.zeros((2, 4), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "unaudited"):
            batcher.finalize_joint_batch(solver, plan, self.anchor_output(plan), ledger=self.ledger,
                                         anchor_policy_version=plan.anchor_policy_version,
                                         anchor_input_fingerprint=plan.fingerprint, solver_temperature=1, now=51)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
