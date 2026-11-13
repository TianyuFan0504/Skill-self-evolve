"""Run the stock sampler loop over queue metadata with a failed sibling group."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from verl.experimental.skill_harness.group_sampler import CompleteGroupReplayBuffer
from verl.trainer.ppo.v1 import replay_buffer


class CompleteGroupSamplerTests(unittest.TestCase):
    def test_failed_uid_with_materialized_sibling_is_evicted_and_refilled(self):
        data = {"train": {
            "bad": {"is_prompt": True, "status": "failure", "global_steps": 1},
            "bad_0_0": {"status": "success", "uid": "bad"},
        }}
        refills = []

        def clear(*, partition_id, keys):
            for key in keys:
                data[partition_id].pop(key, None)

        def refill(count):
            refills.append(count)
            data["train"].update({
                "good": {"is_prompt": True, "status": "finished", "global_steps": 1},
                "good_0_0": {"status": "success", "uid": "good"},
                "good_1_0": {"status": "success", "uid": "good"},
            })
            return count

        sampler = CompleteGroupReplayBuffer(
            trainer_mode="sync", trainer_config={}, max_off_policy_threshold=None,
            max_off_policy_strategy="drop", sampler_kwargs={}, refill_fn=refill,
        )
        self.assertIsInstance(sampler, replay_buffer.ReplayBuffer)
        with (
            patch.object(replay_buffer.tq, "kv_list", side_effect=lambda: copy.deepcopy(data)),
            patch.object(replay_buffer.tq, "kv_clear", side_effect=clear),
        ):
            batch, metrics = sampler.sample(global_steps=1, partition_id="train", batch_size=1)
        self.assertEqual(set(batch.keys), {"good_0_0", "good_1_0"})
        self.assertEqual(refills, [1])
        self.assertEqual(metrics["training/rollout_failure/evicted_samples"], 1)
        self.assertNotIn("bad_0_0", data["train"])

    def test_validation_failure_remains_visible_to_case_accounting(self):
        sampler = CompleteGroupReplayBuffer(
            trainer_mode="sync", trainer_config={}, max_off_policy_threshold=None,
            max_off_policy_strategy="drop", sampler_kwargs={}, refill_fn=lambda _count: None,
        )
        sampler.failure_keys["val"].add("case")
        sampler.partitions["val"]["case_0_0"] = {"status": "success", "uid": "case"}
        reasons = sampler._terminal_eviction_reasons(global_steps=1, partition_id="val")
        self.assertEqual(reasons[2], set())
        self.assertEqual(sampler._sampleable_terminal_keys("val", reasons), {"case"})


if __name__ == "__main__":
    unittest.main()
