# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Refill complete GRPO groups after a host-proven local archive failure.

Use the supported V1 custom-sampler constructor and core eviction/refill path.
Raw private artifacts remain retained; no missing sibling becomes reward zero.
"""

from __future__ import annotations

import os
from pathlib import Path

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

from .alfworld.accounts import local_archive_failed_groups


class ArchiveAwareReplayBuffer(ReplayBuffer):
    """Reject a terminal failed group even when some siblings reached TQ."""

    def __init__(self, trainer_config, **kwargs):
        if kwargs.get("trainer_mode") != "sync":
            raise ValueError("ArchiveAwareReplayBuffer requires the synchronous trainer")
        batch_size = trainer_config.get("train_batch_size")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("ArchiveAwareReplayBuffer requires trainer.v1.sync.train_batch_size")
        run_root = os.environ.get("SKILL_RUN_DIR")
        if not run_root or not Path(run_root).is_absolute():
            raise ValueError("Archive-aware sampling requires the trusted absolute SKILL_RUN_DIR")
        self.archive_run_root = str(Path(run_root).resolve())
        fixed = {"train_batch_size": batch_size, "gen_batch_size": 1, "sync_refill_failed_groups": True}
        for name, expected in fixed.items():
            if name in kwargs and kwargs.pop(name) != expected:
                raise ValueError(f"Archive-aware sampling requires {name}={expected}")
        super().__init__(trainer_config=trainer_config, **kwargs, **fixed)

    def _terminal_eviction_reasons(self, global_steps, partition_id):
        stale, filtered, failed, counts = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id == "val":
            return stale, filtered, failed, counts
        terminal = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        # A local marker precedes the Task exception and any parent terminal
        # status. Intersect terminal UIDs so siblings still running can finish
        # before the core clears every row and requests one replacement prompt.
        archive_failed = local_archive_failed_groups(self.archive_run_root) & terminal
        return stale, filtered, failed | self.failure_keys[partition_id] | archive_failed, counts
