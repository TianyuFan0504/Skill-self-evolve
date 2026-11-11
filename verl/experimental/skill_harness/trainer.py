"""Small trainer extensions over the supported V1 synchronous controller."""

from __future__ import annotations

import math
from pathlib import Path

from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.transferqueue_utils import tq

from .telemetry import append_metrics, episode_metrics


class SkillSyncTrainer(PPOTrainerSync):
    """Preserve solver GRPO semantics while saving auditable episode metrics.

    Joint skill learning is enabled only when its worker/loss implementation is
    installed. A requested joint run must never silently become a solver-only run.
    """

    def __init__(self, config):
        mode = config.skill_harness.mode
        if mode != "solver_baseline":
            raise NotImplementedError(f"Training mode {mode!r} is not wired yet; refusing a solver-only substitute")
        self.artifact_root = Path(config.skill_harness.artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        super().__init__(config)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "session_id", "rm_scores", "extra_fields"],
        )
        uids = fields["uid"].tolist()
        sessions = fields["session_id"].tolist()
        scores = fields["rm_scores"].to_padded_tensor(0.0).sum(dim=-1).tolist()
        extras = fields["extra_fields"].tolist()
        rows = []
        for i, (uid, session, score, extra) in enumerate(zip(uids, sessions, scores, extras, strict=True)):
            info = extra.get("reward_extra_info", {})
            if "acc" not in info:
                raise RuntimeError("Native ALFWorld runner did not provide authoritative TaskResult.accuracy")
            rows.append(
                {
                    "uid": uid,
                    "session_id": session,
                    "reward": score,
                    "accuracy": info["acc"],
                    "error": info.get("error"),
                    "padding": batch.tags[i].get("is_padding", False),
                }
            )
        metrics.update(episode_metrics(rows))
        for key in ("actor/grad_norm", "actor/pg_loss"):
            value = metrics.get(key)
            if value is not None and not math.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite optimizer metric: {key}={value}")
        append_metrics(self.artifact_root, step=global_steps, phase="train", metrics=metrics)

    def _validate(self):
        metrics = super()._validate()
        append_metrics(self.artifact_root, step=self.global_steps, phase="dev", metrics=metrics)
        return metrics
