#!/usr/bin/env python3
"""Original zero-epoch trainer with the common fresh-account inference hook."""

from __future__ import annotations

import hashlib
from pathlib import Path

POLICY = "zero-epoch-original-trainer-account-pool-epoch-v1"


def identity(original):
    entry, component = Path(__file__).resolve(), Path(original.__file__).resolve()
    return {"policy": POLICY, "actual_entry": str(entry),
            "entry_ref": {"path": str(entry), "sha256": hashlib.sha256(entry.read_bytes()).hexdigest()},
            "original_entry_ref": {"path": str(component),
                                   "sha256": hashlib.sha256(component.read_bytes()).hexdigest()},
            "task_runner_class": "verl.experimental.skill_harness.train.SkillTaskRunner"}


def validate_baseline(config):
    if (config.skill_harness.mode != "skill_joint" or config.trainer.total_epochs != 0
            or config.trainer.resume_mode != "disable"
            or getattr(config.trainer, "resume_from_path", None) is not None):
        raise ValueError("The common-pool baseline requires fresh skill_joint, epochs0 and disabled restore")
    if config.trainer.val_before_train or config.trainer.val_only:
        raise ValueError("The common-pool baseline retains the original official evaluation entry")


def entry(config):
    from skill_harness_examples import account_pool_epoch
    from verl.experimental.skill_harness import train as original

    validate_baseline(config)
    account_pool_epoch.prepare_driver(config)
    # Preserve the original entry's trainer, manifest and shutdown behavior.
    # The only extra runtime operation is selection/evidence of the common pool.
    original.auto_set_device(config)
    original.validate_config(config, use_reference_policy=original.need_reference_policy(config),
                             use_critic=original.need_critic(config))
    original.apply_runtime_environment(config)
    account_pool_epoch.prepare_runtime_environment(config)
    account_pool_epoch.publish_inference_expected(config.skill_harness.artifact_root,
                                                 baseline_extension=identity(original))
    original.save_run_manifest(config)
    try:
        original.run_ppo(config, task_runner_class=original.SkillTaskRunner)
    finally:
        if original.ray.is_initialized():
            original.ray.shutdown()


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
