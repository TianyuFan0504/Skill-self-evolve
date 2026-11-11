"""Pin the training checkout before importing the external Uni-Agent runtime.

The benchmark's inference venv contains another editable verl checkout.  Its
site-packages may supply missing *dependencies*, but its ``.pth`` files must
never execute in the training process.  This module is also a Ray worker setup
hook, so driver and workers resolve the same source tree.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ROOT = "/workspace/verl-0.9.1"
DEFAULT_BENCHMARK_ROOT = "/root/nas_chaohuang/benchmarks/alfworld"
EXPECTED_VERL_VERSION = "0.9.1"


def _resolve_paths() -> tuple[Path, Path, Path | None]:
    benchmark = Path(os.environ.get("ALFWORLD_BENCHMARK_ROOT", DEFAULT_BENCHMARK_ROOT)).resolve()
    source = Path(os.environ.get("VERL_SOURCE_ROOT", DEFAULT_SOURCE_ROOT)).resolve()
    uni_agent = Path(os.environ.get("UNI_AGENT_SOURCE_ROOT", str(benchmark / "uni-agent"))).resolve()
    raw_site = os.environ.get("UNI_AGENT_DEPENDENCY_SITE")
    if raw_site == "":
        dependency_site = None
    elif raw_site:
        dependency_site = Path(raw_site).resolve()
    else:
        dependency_site = (
            benchmark / "uni-env" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        )
    return source, uni_agent, dependency_site


def _check_dependency_abi(site_packages: Path) -> None:
    """Reject a dependency environment from another Python minor version."""
    config = site_packages.parent.parent.parent / "pyvenv.cfg"
    if not site_packages.is_dir():
        raise RuntimeError(f"Uni-Agent dependency site does not exist: {site_packages}")
    if not config.is_file():
        raise RuntimeError(f"Dependency venv has no ABI manifest: {config}")
    text = config.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version(?:_info)?\s*=\s*(\d+)\.(\d+)", text)
    if match is None:
        raise RuntimeError(f"Cannot determine the Python ABI from {config}")
    found = tuple(int(value) for value in match.groups())
    current = (sys.version_info.major, sys.version_info.minor)
    if found != current:
        raise RuntimeError(f"Dependency venv ABI {found} does not match training Python {current}")
    implementation = re.search(r"(?m)^implementation\s*=\s*(\S+)", text)
    if implementation and implementation.group(1).lower() != sys.implementation.name:
        raise RuntimeError(f"Dependency implementation does not match {sys.implementation.name}: {config}")


def configure_runtime(*, register_tasks: bool = True) -> dict[str, Any]:
    """Configure offline imports and fail closed on the wrong verl source.

    No package manager, network request, ``site.addsitedir`` or executable
    ``.pth`` file is used.  The optional inference dependency directory is
    appended after the base environment, not ahead of it.
    """
    source, uni_agent, dependency_site = _resolve_paths()
    if not (source / "verl" / "version" / "version").is_file():
        raise RuntimeError(f"The requested verl checkout is missing: {source}")
    if not (uni_agent / "uni_agent" / "__init__.py").is_file():
        raise RuntimeError(f"The Uni-Agent source tree is missing: {uni_agent}")
    if dependency_site is not None:
        _check_dependency_abi(dependency_site)

    leading = [str(source), str(uni_agent)]
    forbidden = {str(uni_agent / "verl")}
    remainder = [path for path in sys.path if path not in leading and path not in forbidden]
    if dependency_site is not None:
        remainder = [path for path in remainder if path != str(dependency_site)]
    sys.path[:] = leading + remainder
    if dependency_site is not None:
        sys.path.append(str(dependency_site))

    os.environ["VERL_SOURCE_ROOT"] = str(source)
    os.environ["UNI_AGENT_SOURCE_ROOT"] = str(uni_agent)
    if dependency_site is not None:
        os.environ["UNI_AGENT_DEPENDENCY_SITE"] = str(dependency_site)
    os.environ["PYTHONPATH"] = os.pathsep.join(leading)
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "UV_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
        "WANDB_MODE": "offline",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }.items():
        os.environ[key] = value

    verl = importlib.import_module("verl")
    actual = Path(verl.__file__).resolve()
    expected = source / "verl" / "__init__.py"
    if actual != expected or getattr(verl, "__version__", None) != EXPECTED_VERL_VERSION:
        raise RuntimeError(
            "Wrong verl import. Launch with the base training interpreter and place "
            f"{source} first in PYTHONPATH. Expected {expected} @ {EXPECTED_VERL_VERSION}; "
            f"got {actual} @ {getattr(verl, '__version__', None)}"
        )

    module = importlib.import_module("uni_agent")
    uni_origin = Path(module.__file__).resolve()
    if uni_origin != uni_agent / "uni_agent" / "__init__.py":
        raise RuntimeError(f"Wrong Uni-Agent import: {uni_origin}")
    if register_tasks:
        importlib.import_module("verl.experimental.skill_harness.alfworld.task")
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "verl_source": str(source),
        "verl_origin": str(actual),
        "verl_version": verl.__version__,
        "uni_agent_source": str(uni_agent),
        "uni_agent_origin": str(uni_origin),
        "dependency_site": None if dependency_site is None else str(dependency_site),
        "tasks_registered": register_tasks,
    }


def configure_worker() -> None:
    """Importable Ray ``worker_process_setup_hook``; accepts no arguments."""
    configure_runtime(register_tasks=True)


def runtime_env_config() -> dict[str, Any]:
    """Return a secret-free Ray runtime environment for this checkout."""
    source, uni_agent, dependency_site = _resolve_paths()
    env_vars = {
        "VERL_SOURCE_ROOT": str(source),
        "UNI_AGENT_SOURCE_ROOT": str(uni_agent),
        "PYTHONPATH": os.pathsep.join((str(source), str(uni_agent))),
        "UNI_AGENT_DEPENDENCY_SITE": "" if dependency_site is None else str(dependency_site),
    }
    passthrough = (
        "ALFWORLD_BENCHMARK_ROOT",
        "ALFWORLD_SERVER_CONFIG",
        "ALFWORLD_CODEX_PACKAGE_ROOT",
        "CODEX_NATIVE_BIN",
        "ALFWORLD_SOLVER_INSTRUCTIONS_FILE",
        "SKILL_RUN_DIR",
        "SKILL_DATA_DIR",
        "ALFWORLD_TASK_CONFIG",
        "ALFWORLD_DATA_DIR",
        "MODEL_PATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "UV_OFFLINE",
        "PIP_NO_INDEX",
        "WANDB_MODE",
        "VLLM_NO_USAGE_STATS",
        "DO_NOT_TRACK",
        "TOKENIZERS_PARALLELISM",
        "VLLM_WORKER_MULTIPROC_METHOD",
    )
    env_vars.update({key: os.environ[key] for key in passthrough if key in os.environ})
    return {
        "env_vars": env_vars,
        "worker_process_setup_hook": "verl.experimental.skill_harness.bootstrap.configure_worker",
    }


def apply_runtime_environment(config, runtime: dict[str, Any] | None = None) -> None:
    """Replace an inherited struct runtime-env node instead of merging into it."""
    from omegaconf import OmegaConf

    current = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env", default={}) or {}
    plain = OmegaConf.to_container(current, resolve=True) if OmegaConf.is_config(current) else dict(current)
    if plain.get("pip") or plain.get("conda"):
        raise ValueError("Remote dependency downloads are disabled; relay packages through the workstation")
    merged = OmegaConf.merge(plain, runtime if runtime is not None else runtime_env_config())
    OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env", merged, merge=False, force_add=True)
