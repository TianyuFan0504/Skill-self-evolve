"""Offline ALFWorld training entrypoint using native Uni-Agent and Codex."""

from __future__ import annotations

# ruff: noqa: E402 - runtime provenance must be pinned before trainer imports.
from .bootstrap import apply_runtime_environment, configure_runtime

# Run before imports from the external Uni-Agent stack or distributed trainer.
IMPORT_PROVENANCE = configure_runtime()

import hashlib
import json
import subprocess
from pathlib import Path

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

from .telemetry import json_value
from .trainer import SkillSyncTrainer


def _redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = key.lower()
            secret = any(
                part in lowered
                for part in (
                    "api_key",
                    "admin_token",
                    "access_token",
                    "bearer_token",
                    "secret_key",
                    "password",
                )
            )
            result[key] = "[REDACTED]" if secret else _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def save_run_manifest(config):
    root = Path(config.skill_harness.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    source = Path(IMPORT_PROVENANCE["verl_source"])
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    diff = subprocess.run(["git", "-C", str(source), "diff", "--binary"], capture_output=True, check=True).stdout
    resolved = _redact(OmegaConf.to_container(config, resolve=True))
    data = {
        "schema_version": 1,
        "source_commit": commit,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "import_provenance": IMPORT_PROVENANCE,
        "config": resolved,
    }
    (root / "training-manifest.json").write_text(json.dumps(json_value(data), indent=2), encoding="utf-8")
    (root / "resolved-config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(resolved)), encoding="utf-8")


@ray.remote
class SkillTaskRunner:
    """Use the documented custom TaskRunner contract without replacing factories."""

    def run(self, config):
        import transfer_queue as tq

        configure_runtime()
        configure_verl_logging()
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("This initial adapter preserves the literal sync trainer semantics")
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        trainer = None
        succeeded = False
        tq.init(config.transfer_queue)
        try:
            trainer = SkillSyncTrainer(config)
            trainer.init()
            manager_class = load_class_from_fqn(
                config.actor_rollout_ref.rollout.agent.agent_loop_manager_class,
                "AgentLoopManager",
            )
            manager = manager_class.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    save_run_manifest(config)
    try:
        run_ppo(config, task_runner_class=SkillTaskRunner)
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
