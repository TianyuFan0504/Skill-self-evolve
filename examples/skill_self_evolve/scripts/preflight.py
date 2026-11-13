#!/usr/bin/env python3
"""Check installed dependencies and resolve the joint configuration on CPU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, as accepted by run.sh")
    args = parser.parse_args()
    modules = ("hydra", "omegaconf", "ray", "torch", "transfer_queue", "uni_agent", "verl")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Install the runtime dependencies in this interpreter: " + ", ".join(missing))

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[1] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="alfworld_joint", overrides=args.overrides)
    OmegaConf.resolve(config)
    task_runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task
    required_files = {
        "training data": config.data.train_files,
        "development data": config.data.val_files,
        "split manifest": config.skill_harness.split_manifest,
        "agent task configuration": task_runner.runner_kwargs.task_config_path,
        "environment backend": config.skill_harness.evaluation.server_script,
        "environment service configuration": os.environ.get("ALFWORLD_SERVER_CONFIG", ""),
    }
    for label, path in required_files.items():
        paths = path if OmegaConf.is_list(path) or isinstance(path, list) else [path]
        for item in paths:
            if not Path(item).is_file():
                parser.error(f"Missing {label}: {item}")
    runtime_path = os.environ.get("ALFWORLD_RUNTIME_CONFIG")
    if not runtime_path:
        parser.error("ALFWORLD_RUNTIME_CONFIG must identify the provisioned runtime JSON")
    runtime = json.loads(Path(runtime_path).read_text())
    provider = runtime.get("account_pool", {})
    if not provider.get("factory") or not isinstance(provider.get("kwargs"), dict):
        parser.error("Runtime JSON requires account_pool.factory and account_pool.kwargs")
    print("Joint configuration and installed dependencies are available; no training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
