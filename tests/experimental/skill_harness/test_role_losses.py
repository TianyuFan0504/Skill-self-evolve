"""CPU numerical tests; run directly with unittest, no Ray/GPU required.

SKILL_HARNESS_LOSSES_PATH can point at the module in an isolated test transfer.
The two optional adapter tests use the installed checkout's verl dependencies.
"""

import importlib.util
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(os.environ.get(
    "SKILL_HARNESS_LOSSES_PATH",
    Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/losses.py",
))
SPEC = importlib.util.spec_from_file_location("skill_role_losses_under_test", MODULE_PATH)
losses = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = losses
SPEC.loader.exec_module(losses)


def mixed_batch():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    return {
        "role_id": torch.tensor([0, 0, 1, 1]), "response_mask": mask,
        "old_log_probs": torch.full((4, 3), -1.0),
        "advantages": torch.tensor([[1., 1., 0.], [-1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
        "curate_behavior_log_probs": torch.full((4, 3), -1.0),
        "curate_anchor_log_probs": torch.full((4, 3), -1.0),
        "curate_advantages": torch.tensor([0., 0., 2., -1.]),
        "curate_trajectory_count": torch.ones(4, dtype=torch.long),
        "curate_candidate_id": torch.tensor([-1, -1, 101, 102]),
    }


def select(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def one_curator(advantage=1.0, length=2):
    batch = select(mixed_batch(), [2])
    for key in ("response_mask", "old_log_probs", "advantages", "curate_behavior_log_probs", "curate_anchor_log_probs"):
        batch[key] = batch[key][:, :length]
    batch["curate_advantages"] = torch.tensor([advantage])
    return batch


def normalization(batch, dp_size=1):
    roles, solver, curator = losses._masks(batch["role_id"], batch["response_mask"],
                                         batch.get("solver_token_mask"), batch.get("curate_token_mask"))
    return losses.RoleNormalization(dp_size, int(solver.sum()), int(solver.any(-1).sum()),
                                    int(curator.any(-1).sum()))


def compute(current, batch, norm=None, **kwargs):
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl.workers.config.actor import ActorConfig

    norm = norm or normalization(batch)
    settings = kwargs.get("solver_config", losses.SolverLossConfig())
    config = ActorConfig(
        strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=settings.clip_low, clip_ratio_high=settings.clip_high,
        clip_ratio_c=settings.dual_clip,
        global_batch_info={"dp_size": norm.dp_size, "batch_num_tokens": norm.solver_tokens,
                           "global_batch_size": norm.solver_sequences,
                           "loss_scale_factor": settings.loss_scale_factor},
    )
    def solver(old, current, advantage, mask, weights):
        return get_policy_loss_fn("vanilla")(
            old_log_prob=old, log_prob=current, advantages=advantage, response_mask=mask,
            loss_agg_mode=settings.loss_agg_mode, config=config, rollout_is_weights=weights,
        )
    kwargs.setdefault("solver_policy_loss", solver)
    return losses.role_aware_ppo_loss(current, normalization=norm, **batch, **kwargs)


def _distributed_counts_worker(rank, init_file, result_dir):
    """Two Gloo CPU ranks, one solver-only and one curator-only."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        batch = select(mixed_batch(), [0, 1] if rank == 0 else [2, 3])
        norm = losses.prepare_role_batch_metadata(batch)
        del batch[losses.ROLE_NORMALIZATION_KEY]
        del batch["skill_curate_candidates_verified"]
        current = torch.full((2, 3), -1.0, requires_grad=True)
        value, _ = compute(current, batch, norm)
        value.backward()
        torch.save({"norm": vars(norm), "gradient": current.grad, "loss": value.detach()},
                   Path(result_dir) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


class RoleLossTests(unittest.TestCase):
    def test_solver_requires_the_verl_policy_callback(self):
        batch = mixed_batch()
        with self.assertRaisesRegex(ValueError, "policy-loss callback"):
            losses.role_aware_ppo_loss(torch.full((4, 3), -1.0), normalization=normalization(batch), **batch)

    def test_shared_parameter_updates_once_from_the_sum_of_both_losses(self):
        batch = mixed_batch()
        theta = torch.nn.Parameter(torch.tensor(-1.0))
        optimizer = torch.optim.SGD([theta], lr=0.1)
        current = theta.expand(4, 3)
        value, _ = compute(current, batch)
        # A loss function must not step any model itself.
        self.assertEqual(float(theta.detach()), -1.0)
        value.backward()
        expected_gradient = -1 / 3 - (2 * 3 - 1) / 2
        self.assertAlmostEqual(float(theta.grad), expected_gradient, places=5)
        optimizer.step()
        self.assertAlmostEqual(float(theta.detach()), -1 - 0.1 * expected_gradient, places=5)

    def test_shared_online_tensor_receives_both_role_gradients(self):
        batch = mixed_batch()
        current = torch.full((4, 3), -1.0, requires_grad=True)
        value, _ = compute(current, batch, curate_config=losses.CurateLossConfig(coefficient=0.5))
        value.backward()
        self.assertLess(current.grad[0, 0], 0)  # reinforce successful solver
        self.assertGreater(current.grad[1, 0], 0)  # discourage failed solver
        self.assertLess(current.grad[2, 0], 0)  # positive delayed curate delta
        self.assertGreater(current.grad[3, 0], 0)  # negative delayed curate delta
        self.assertEqual(current.grad[0, 2], 0)
        self.assertAlmostEqual(float(current.grad[2, 0]), -0.5)

    def test_behavior_anchor_advantage_and_solver_old_are_detached(self):
        batch = mixed_batch()
        frozen_fields = ("old_log_probs", "advantages", "curate_behavior_log_probs",
                         "curate_anchor_log_probs", "curate_advantages")
        for field in frozen_fields:
            batch[field].requires_grad_()
        current = torch.full((4, 3), -1.0, requires_grad=True)
        compute(current, batch)[0].backward()
        for field in frozen_fields:
            self.assertIsNone(batch[field].grad, field)
        self.assertIsNotNone(current.grad)

    def test_positive_negative_zero_labels_are_not_recentered(self):
        for label, sign in ((2.0, -1), (-2.0, 1), (0.0, 0)):
            with self.subTest(label=label):
                batch = one_curator(label)
                current = torch.full((1, 2), -1.0, requires_grad=True)
                value, _ = compute(current, batch)
                value.backward()
                self.assertAlmostEqual(float(value), -label)
                self.assertEqual(int(current.grad[0, 0].sign()), sign)
        batch = select(mixed_batch(), [2, 3])
        batch["curate_advantages"][:] = 2
        value, _ = compute(torch.full((2, 3), -1.0), batch)
        self.assertAlmostEqual(float(value), -2)  # group centering would incorrectly make this zero

    def test_whole_macro_ratio_clips_when_each_token_ratio_would_not(self):
        batch = one_curator(1.0)
        current = torch.full((1, 2), -0.89, requires_grad=True)
        value, metrics = compute(current, batch)
        value.backward()
        self.assertGreater(math.exp(2 * 0.11), 1.2)
        self.assertLess(math.exp(0.11), 1.2)
        self.assertAlmostEqual(float(value), -1.2, places=5)
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))
        self.assertAlmostEqual(float(metrics["skill/curate_clip_fraction"]), 1)

    def test_negative_advantage_uses_lower_sequence_clip(self):
        batch = one_curator(-2.0)
        current = torch.full((1, 2), -1.2, requires_grad=True)
        value, _ = compute(current, batch)
        value.backward()
        self.assertAlmostEqual(float(value), 1.6, places=5)
        self.assertTrue(torch.equal(current.grad, torch.zeros_like(current)))

    def test_historical_is_uses_product_and_detached_cap(self):
        batch = one_curator()
        batch["curate_behavior_log_probs"][:] = -2
        batch["curate_behavior_log_probs"].requires_grad_()
        current = torch.full((1, 2), -1.0, requires_grad=True)
        value, metrics = compute(current, batch, curate_config=losses.CurateLossConfig(historical_is_cap=3.0))
        value.backward()
        self.assertAlmostEqual(float(value), -3)
        self.assertAlmostEqual(float(current.grad[0, 0]), -3)
        self.assertIsNone(batch["curate_behavior_log_probs"].grad)
        self.assertAlmostEqual(float(metrics["skill/curate_is_cap_fraction"]), 1)

    def test_candidate_mean_is_independent_of_macro_length(self):
        batch = select(mixed_batch(), [2, 3])
        batch["curate_advantages"][:] = 1
        current = torch.full((2, 3), -1.0, requires_grad=True)
        value, _ = compute(current, batch)
        value.backward()
        self.assertAlmostEqual(float(value), -1)
        self.assertAlmostEqual(float(current.grad[0, 0]), -0.5)
        self.assertAlmostEqual(float(current.grad[1, 0]), -0.5)
        self.assertEqual(float(current.grad[1, 1]), 0)

    def test_microbatch_sum_equals_full_optimizer_batch(self):
        batch = mixed_batch()
        current = torch.full((4, 3), -1.0, requires_grad=True)
        reference, _ = compute(current, batch)
        expected = torch.autograd.grad(reference, current)[0]
        current2 = current.detach().clone().requires_grad_()
        accumulated = sum(compute(current2[index], select(batch, index), normalization(batch))[0]
                          for index in ([0], [1, 2], [3]))
        accumulated.backward()
        self.assertTrue(torch.allclose(reference, accumulated))
        self.assertTrue(torch.allclose(expected, current2.grad))

    def test_dp_role_absence_scales_to_the_global_gradient(self):
        batch = mixed_batch()
        full = torch.full((4, 3), -1.0, requires_grad=True)
        reference = compute(full, batch)[0]
        expected = torch.autograd.grad(reference, full)[0]
        shards = full.detach().clone().requires_grad_()
        norm = normalization(batch, dp_size=2)
        # DDP/FSDP averages gradients across DP ranks after partial sums.
        actual = sum(compute(shards[index], select(batch, index), norm)[0]
                     for index in ([0, 1], [2, 3])) / 2
        actual.backward()
        self.assertTrue(torch.allclose(reference, actual))
        self.assertTrue(torch.allclose(expected, shards.grad))

    def test_metric_keys_and_global_values_survive_role_absence(self):
        batch = mixed_batch()
        current = torch.full((4, 3), -1.0)
        _, full_metrics = compute(current, batch)
        parts = [compute(current[ix], select(batch, ix), normalization(batch, dp_size=2))[1]
                 for ix in ([0, 1], [2, 3])]
        self.assertEqual(set(parts[0]), set(parts[1]))
        self.assertEqual(set(full_metrics), set(parts[0]))
        for key in full_metrics:
            self.assertTrue(torch.allclose(full_metrics[key], (parts[0][key] + parts[1][key]) / 2), key)

    def test_global_solver_normalization_modes_survive_microbatching(self):
        batch = mixed_batch()
        for mode in ("token-mean", "token-sum", "seq-mean-token-mean", "seq-mean-token-sum",
                     "seq-mean-token-sum-norm"):
            with self.subTest(mode=mode):
                cfg = losses.SolverLossConfig(loss_agg_mode=mode, loss_scale_factor=10)
                current = torch.full((4, 3), -1.0, requires_grad=True)
                full = compute(current, batch, solver_config=cfg)[0]
                micro = sum(compute(current[ix], select(batch, ix), normalization(batch), solver_config=cfg)[0]
                            for ix in ([0], [1], [2, 3]))
                self.assertTrue(torch.allclose(full, micro))

    def test_only_authored_action_tokens_are_trained(self):
        batch = one_curator()
        batch["curate_token_mask"] = torch.tensor([[1, 0]], dtype=torch.bool)
        current = torch.tensor([[-1., float("nan")]], requires_grad=True)
        value, _ = compute(current, batch)
        value.backward()
        self.assertAlmostEqual(float(value), -1)
        self.assertEqual(float(current.grad[0, 1]), 0)
        invalid = mixed_batch()
        invalid["curate_token_mask"] = invalid["response_mask"].clone()
        with self.assertRaisesRegex(ValueError, "subsets"):
            compute(torch.full((4, 3), -1.), invalid)

    def test_solver_entropy_and_kl_exclude_curator_tokens(self):
        batch = mixed_batch()
        current = torch.full((4, 3), -1.0, requires_grad=True)
        entropy = torch.ones((4, 3), requires_grad=True)
        kl = (current + 2).square() / 2
        cfg = losses.SolverLossConfig(entropy_coefficient=0.3, kl_coefficient=0.4)
        value, metrics = compute(current, batch, entropy=entropy, solver_kl_values=kl, solver_config=cfg)
        value.backward()
        self.assertAlmostEqual(float(metrics["skill/solver_entropy"]), 1)
        self.assertAlmostEqual(float(metrics["skill/solver_kl"]), 0.5)
        self.assertTrue(torch.equal(entropy.grad[2:], torch.zeros_like(entropy.grad[2:])))
        self.assertAlmostEqual(float(entropy.grad[0, 0]), -0.1, places=6)

    def test_validation_fails_for_segments_duplicate_ids_and_token_labels(self):
        batch = one_curator()
        batch["curate_trajectory_count"][:] = 2
        with self.assertRaisesRegex(ValueError, "multi-TITO"):
            compute(torch.full((1, 2), -1.), batch)
        batch = select(mixed_batch(), [2, 3])
        batch["curate_candidate_id"][:] = 101
        with self.assertRaisesRegex(ValueError, "distinct"):
            compute(torch.full((2, 3), -1.), batch)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            losses.prepare_role_batch_metadata(batch)
        batch = one_curator()
        batch["curate_advantages"] = torch.ones((1, 2))
        with self.assertRaisesRegex(ValueError, "one uncentered"):
            compute(torch.full((1, 2), -1.), batch)

    def test_nonfinite_labels_probabilities_and_missing_counts_fail(self):
        batch = one_curator(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute(torch.full((1, 2), -1.), batch)
        batch = one_curator()
        batch["curate_behavior_log_probs"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compute(torch.full((1, 2), -1.), batch)
        with self.assertRaisesRegex(ValueError, "normalization"):
            compute(torch.full((1, 2), -1.), one_curator(), losses.RoleNormalization(1, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "fixed positive"):
            losses.SolverLossConfig(loss_agg_mode="seq-mean-token-sum-norm")

    def test_prepare_counts_and_candidate_verification(self):
        batch = mixed_batch()
        norm = losses.prepare_role_batch_metadata(batch)
        self.assertEqual(norm, losses.RoleNormalization(1, 3, 2, 2))
        self.assertEqual(batch[losses.ROLE_NORMALIZATION_KEY], vars(norm))
        self.assertTrue(batch["skill_curate_candidates_verified"])

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "needs Gloo")
    def test_real_two_rank_cpu_collective_with_missing_roles(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_distributed_counts_worker, args=(str(Path(directory) / "init"), directory), nprocs=2, join=True)
            rank0 = torch.load(Path(directory) / "rank0.pt", weights_only=True)
            rank1 = torch.load(Path(directory) / "rank1.pt", weights_only=True)
        self.assertEqual(rank0["norm"], vars(losses.RoleNormalization(2, 3, 2, 2)))
        self.assertEqual(rank0["norm"], rank1["norm"])
        full = torch.full((4, 3), -1.0, requires_grad=True)
        expected = torch.autograd.grad(compute(full, mixed_batch())[0], full)[0]
        actual = torch.cat((rank0["gradient"], rank1["gradient"])) / 2
        self.assertTrue(torch.allclose(actual, expected))


class EngineAdapterTests(unittest.TestCase):
    """Dependencies are optional for isolated Torch-only installations."""

    @classmethod
    def setUpClass(cls):
        try:
            from tensordict import TensorDict

            from verl.workers.config import ActorConfig
        except ImportError as error:
            raise unittest.SkipTest(f"full verl dependencies unavailable: {error}") from error
        cls.TensorDict, cls.ActorConfig = TensorDict, ActorConfig

    def test_native_nested_output_uses_official_causal_alignment(self):
        data = {
            "prompts": torch.nested.as_nested_tensor([torch.tensor([10, 11]), torch.tensor([12])], layout=torch.jagged),
            "responses": torch.nested.as_nested_tensor(
                [torch.tensor([20, 21]), torch.tensor([22])], layout=torch.jagged,
            ),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
        }
        values = torch.arange(6, dtype=torch.float32, requires_grad=True)
        nested = torch.nested.nested_tensor_from_jagged(values, torch.tensor([0, 4, 6]))
        actual = losses.response_model_output(nested, data)
        self.assertTrue(torch.equal(actual, torch.tensor([[1., 2.], [4., 0.]])))
        actual.sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.tensor([0., 1., 1., 0., 1., 0.])))

    def test_public_callback_matches_stock_solver_with_entropy_and_kl(self):
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.losses import ppo_loss

        batch = select(mixed_batch(), [0, 1])
        batch["prompts"] = torch.tensor([[1, 2], [3, 4]])
        batch["responses"] = torch.tensor([[5, 6, 0], [7, 0, 0]])
        batch["attention_mask"] = torch.cat((torch.ones((2, 2), dtype=torch.bool), batch["response_mask"]), -1)
        batch["ref_log_prob"] = torch.full((2, 3), -1.2)
        batch["ref_log_prob"].requires_grad_()
        data = self.TensorDict(batch, batch_size=[2])
        losses.prepare_role_batch_metadata(data)
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=3, global_batch_size=2)
        config = self.ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
                                  use_kl_loss=True, kl_loss_coef=0.04,
                                  entropy_coeff=0.02, kl_loss_type="k3+")
        before = dict(config.global_batch_info)
        # Padded response callback accepts exact response-aligned Engine output.
        current = torch.full((2, 3), -0.95, requires_grad=True)
        entropy = (current + 2).square()
        actual, _ = losses.skill_harness_loss(config, {"log_probs": current, "entropy": entropy}, data)
        reference_gradient = torch.autograd.grad(actual, data["ref_log_prob"], allow_unused=True,
                                                 retain_graph=True)[0]
        self.assertIsNone(reference_gradient)
        actual_gradient = torch.autograd.grad(actual, current)[0]
        self.assertEqual(config.global_batch_info, before)
        # Stock expects flattened full-sequence outputs. Reconstruct the exact
        # causal alignment: indices 1,2 and 5 correspond to response logprobs.
        flat = torch.stack((current[0, 0] * 0, current[0, 0], current[0, 1], current[0, 2] * 0,
                            current[0, 2] * 0, current[1, 0], current[1, 1] * 0))
        flat_entropy = (flat + 2).square()
        expected, _ = ppo_loss(config, {"log_probs": flat, "entropy": flat_entropy}, data)
        expected_gradient = torch.autograd.grad(expected, current)[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(actual_gradient, expected_gradient, atol=1e-6))

    def test_public_callback_mixed_batch_matches_joint_torch_objective(self):
        batch = mixed_batch()
        data = self.TensorDict(batch, batch_size=[4])
        norm = losses.prepare_role_batch_metadata(data)
        config = self.ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
        current = torch.full((4, 3), -1.0, requires_grad=True)
        curate_config = losses.CurateLossConfig(coefficient=0.3)
        actual, metrics = losses.skill_harness_loss(config, {"log_probs": current}, data,
                                                   curate_config=curate_config)
        expected, _ = compute(current, batch, norm, curate_config=curate_config)
        actual_gradient = torch.autograd.grad(actual, current, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, current)[0]
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(actual_gradient, expected_gradient))
        self.assertIn("skill/curate_loss", metrics)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
