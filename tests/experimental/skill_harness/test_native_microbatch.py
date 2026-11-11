"""CPU packing regression: raw-token budgets do not bound dense padded work."""

import unittest

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches


class NativeMicrobatchTest(unittest.TestCase):
    # Actual lengths present in the failed v2 native step-1 archives. This is a
    # packing fixture, not a reconstruction claim about the failing rank's packet.
    lengths = (19120, 8463, 8453)

    def data(self, dynamic):
        rows = [torch.arange(length, dtype=torch.int64) + 50000 * index for index, length in enumerate(self.lengths)]
        data = TensorDict({"input_ids": tu.nested_tensor_from_tensor_list(rows)}, batch_size=[len(rows)])
        tu.assign_non_tensor(
            data,
            use_dynamic_bsz=dynamic,
            sp_size=1,
            force_group_size=1,
            max_token_len_per_gpu=36864,
            micro_batch_size_per_gpu=1,
            global_batch_size=len(rows),
            mini_batch_size=len(rows),
            epochs=1,
        )
        return data, rows

    def test_dynamic_budget_ignores_micro_one_and_permits_dense_padding_expansion(self):
        data, original = self.data(True)
        batches, order = prepare_micro_batches(data, same_micro_num_in_dp=False)
        self.assertEqual(len(batches), 1)
        batch = batches[0]
        self.assertEqual(len(batch), 3)
        actual_lengths = batch["input_ids"].offsets().diff()
        raw_tokens = int(actual_lengths.sum())
        padded = torch.nested.to_padded_tensor(
            batch["input_ids"], padding=0, output_size=(len(batch), int(actual_lengths.max()))
        )
        self.assertEqual(raw_tokens, 36036)
        self.assertLessEqual(raw_tokens, 36864)
        self.assertEqual(tuple(padded.shape), (3, 19120))
        self.assertEqual(padded.numel(), 57360)
        self.assertGreater(padded.numel(), 36864)
        for position, source_index in enumerate(order[0]):
            torch.testing.assert_close(batch["input_ids"][position], original[source_index])
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(torch.distributed.is_initialized())

    def test_static_micro_one_retains_every_id_and_global_optimizer_batch(self):
        data, original = self.data(False)
        batches, order = prepare_micro_batches(data, same_micro_num_in_dp=False)
        self.assertIsNone(order)
        self.assertEqual(len(batches), len(original))
        for batch, expected in zip(batches, original, strict=True):
            self.assertEqual(len(batch), 1)
            torch.testing.assert_close(batch["input_ids"][0], expected)
            self.assertEqual(tu.get(batch, "global_batch_size"), len(original))
            self.assertEqual(tu.get(batch, "mini_batch_size"), len(original))
            self.assertEqual(tu.get(batch, "epochs"), 1)
        self.assertEqual(sum(batch["input_ids"].values().numel() for batch in batches), sum(self.lengths))
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(torch.distributed.is_initialized())


if __name__ == "__main__":
    unittest.main()
