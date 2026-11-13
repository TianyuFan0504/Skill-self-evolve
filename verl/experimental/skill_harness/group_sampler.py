"""Refill incomplete ALFWorld groups through the official sampler extension."""

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class CompleteGroupReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        kwargs["sync_refill_failed_groups"] = True
        kwargs["gen_batch_size"] = 1
        super().__init__(*args, **kwargs)

    def _terminal_eviction_reasons(self, global_steps, partition_id):
        stale, filtered, failed, counts = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id == "train":
            failed = failed | self.failure_keys[partition_id]
        return stale, filtered, failed, counts
