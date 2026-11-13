"""Use installed packages and Ray configuration without modifying imports."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def configure_runtime(*, register_tasks=True):
    verl = importlib.import_module("verl")
    agent = importlib.import_module("uni_agent")
    if register_tasks:
        importlib.import_module("verl.experimental.skill_harness.alfworld.task")
    return {
        "python": sys.executable,
        "verl_source": str(Path(verl.__file__).resolve().parent.parent),
        "verl_version": getattr(verl, "__version__", None),
        "uni_agent_origin": str(Path(agent.__file__).resolve()),
        "tasks_registered": register_tasks,
    }


def configure_worker():
    configure_runtime()


def runtime_env_config():
    names = (
        "ALFWORLD_RUNTIME_CONFIG", "ALFWORLD_SERVER_CONFIG", "ALFWORLD_TASK_CONFIG",
        "ALFWORLD_ENV_PYTHON", "ALFWORLD_ENV_SERVER", "ALFWORLD_BENCHMARK_ROOT",
        "ALFWORLD_CODEX_PACKAGE_ROOT", "CODEX_NATIVE_BIN", "ALFWORLD_SOLVER_INSTRUCTIONS_FILE",
        "SKILL_RUN_DIR", "SKILL_DATA_DIR", "ALFWORLD_DATA_DIR", "MODEL_PATH", "MODEL_NAME",
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "UV_OFFLINE",
        "PIP_NO_INDEX", "WANDB_MODE", "VLLM_NO_USAGE_STATS", "DO_NOT_TRACK",
        "TOKENIZERS_PARALLELISM", "VLLM_WORKER_MULTIPROC_METHOD",
    )
    return {
        "env_vars": {name: os.environ[name] for name in names if name in os.environ},
        "worker_process_setup_hook": "verl.experimental.skill_harness.bootstrap.configure_worker",
    }


def apply_runtime_environment(config, runtime=None):
    from omegaconf import OmegaConf

    current = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env", default={}) or {}
    plain = OmegaConf.to_container(current, resolve=True) if OmegaConf.is_config(current) else dict(current)
    if plain.get("pip") or plain.get("conda"):
        raise ValueError("Install dependencies before launching; Ray workers must not download packages")
    merged = OmegaConf.merge(plain, runtime if runtime is not None else runtime_env_config())
    OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env", merged, merge=False, force_add=True)
