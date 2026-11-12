#!/usr/bin/env python3
"""Observe the new environment/client profile, then use the frozen spawn entry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import time
import urllib.request
from pathlib import Path

PROTOCOL = "alfworld-step-request-id-v1"
POLICY = "parallel-environment-idempotent-client-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def immutable(path, data, mode):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("An immutable adapter file cannot be a symlink")
    try:
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != data or path.stat().st_uid != os.geteuid():
            raise ValueError("Conflicting immutable adapter file") from None
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise ValueError("Immutable adapter file permissions differ") from None
    return {"path": str(path), "sha256": digest(data)}


def public_wrapper(client, shell):
    """Both names resolve to one allowed inode; no extra Landlock read root."""
    return ("#!/usr/bin/python3\nimport os,sys\nfrom pathlib import Path\n"
            "CLIENT=" + repr(client) + "\nSHELL=" + repr(shell) + "\n"
            "if Path(sys.argv[0]).name == 'alfworld-step':\n"
            "    scope={'__name__':'alfworld_public_request_tool'}\n"
            "    exec(compile(CLIENT,'<alfworld-request-client>','exec'),scope)\n"
            "    raise SystemExit(scope['main']())\n"
            "public=str(Path(__file__).resolve().parent)\n"
            "os.environ['PATH']=public+os.pathsep+os.environ.get('PATH','/usr/bin:/bin')\n"
            "os.execve('/usr/bin/bash',['/usr/bin/bash','-c',SHELL,sys.argv[0],*sys.argv[1:]],os.environ)\n").encode()


def environment_python_wrapper(environment_python, server_script):
    # Preserve the invocation path: resolving a venv's Python symlink loses its
    # environment even when the executable bytes are identical.
    python, server = Path(environment_python).absolute(), Path(server_script).resolve()
    core = Path(os.environ.get("VERL_SOURCE_ROOT", "/workspace/verl-0.9.1")) / (
        "verl/experimental/skill_harness/alfworld/server.py"
    )
    if not python.is_file() or not os.access(python, os.X_OK) or not server.is_file():
        raise ValueError("The original environment interpreter and parallel server are required")
    metadata = {
        "original_python": {"path": str(python), "sha256": digest(python.read_bytes())},
        "parallel_server": {"path": str(server), "sha256": digest(server.read_bytes())},
        "translated_server": str(core), "environment_processes": 4, "io_workers": 4,
    }
    data = ("#!/usr/bin/python3\n# ALFWORLD_ENV_ADAPTER " + json.dumps(metadata, sort_keys=True) + "\n"
            "import os,sys\nfrom pathlib import Path\n"
            "python=" + repr(str(python)) + "\ncore=" + repr(str(core)) + "\nserver=" + repr(str(server)) + "\n"
            "args=sys.argv[1:]\n"
            "if args and str(Path(args[0]).resolve()) == core:\n"
            "    args=[server,*args[1:],'--environment-processes','4','--io-workers','4']\n"
            "os.execve(python,[python,*args],os.environ)\n").encode()
    return data, metadata


def prepare_adapter(run, task_profile, instructions, *, public_root=None, client=None, shell=None,
                    environment_python=None, server_script=None):
    run, task_profile, instructions = Path(run).resolve(), Path(task_profile), Path(instructions)
    here = Path(__file__).resolve().parent
    client = (here / "alfworld_request_client.py").read_text() if client is None else client
    shell = (here / "codex_entrypoint.sh").read_text() if shell is None else shell
    environment_python = environment_python or os.environ.get("ALFWORLD_ENV_PYTHON") or str(
        Path(os.environ["ALFWORLD_BENCHMARK_ROOT"]) / "env/bin/python"
    )
    # A prepared launch may already have installed this adapter in the env.
    python_path = Path(environment_python)
    if python_path.name == "environment-python" and python_path.is_file():
        marker = python_path.read_text().splitlines()[1]
        if not marker.startswith("# ALFWORLD_ENV_ADAPTER "):
            raise ValueError("Unrecognized existing environment Python adapter")
        environment_python = json.loads(marker[len("# ALFWORLD_ENV_ADAPTER "):])["original_python"]["path"]
    env_wrapper, env_metadata = environment_python_wrapper(
        environment_python, server_script or here / "parallel_environment_server.py",
    )
    instruction_bytes, profile_bytes = instructions.read_bytes(), task_profile.read_bytes()
    wrapper = public_wrapper(client, shell)
    asset_id = digest(str(run).encode() + wrapper + instruction_bytes + env_wrapper)[:24]
    public_root = Path(public_root or "/opt/alfworld-tools/skill_self_evolve")
    public_root.mkdir(parents=True, mode=0o755, exist_ok=True)
    public = public_root / ("idempotent-" + asset_id)
    public.mkdir(mode=0o755, exist_ok=True)
    for path in (public_root, public):
        if path.is_symlink() or path.stat().st_uid != os.geteuid():
            raise ValueError("Public adapter directory must be host-owned")
        path.chmod(0o755)
    instruction_ref = immutable(public / "solver_system.md", instruction_bytes, 0o644)
    wrapper_ref = immutable(public / "codex-entrypoint", wrapper, 0o755)
    environment_ref = immutable(public / "environment-python", env_wrapper, 0o755)
    tool = public / "alfworld-step"
    if tool.is_symlink():
        if os.readlink(tool) != "codex-entrypoint":
            raise ValueError("Unexpected public tool symlink")
    elif tool.exists():
        raise ValueError("Public tool path is not the trusted wrapper symlink")
    else:
        tool.symlink_to("codex-entrypoint")
    assert tool.stat().st_ino == (public / "codex-entrypoint").stat().st_ino
    rows = json.loads(profile_bytes)
    selected = [row for row in rows if row.get("name") == "alfworld_skill_training"]
    if len(selected) != 1:
        raise ValueError("Exactly one original ALFWorld Task profile is required")
    updated = copy.deepcopy(rows)
    next(row for row in updated if row.get("name") == "alfworld_skill_training")["agent"]["codex_bin"] = (
        wrapper_ref["path"]
    )
    data = (json.dumps(updated, indent=2, sort_keys=True) + "\n").encode()
    private = run / "client_adapter"
    private.mkdir(mode=0o700, exist_ok=True)
    if private.is_symlink() or private.stat().st_uid != os.geteuid() or stat.S_IMODE(private.stat().st_mode) != 0o700:
        raise ValueError("Client adapter records require a private host-owned directory")
    task_ref = immutable(private / ("task_profile_" + digest(data) + ".json"), data, 0o600)
    return {
        "policy": POLICY, "action_request_protocol": PROTOCOL,
        "original_task_profile": {"path": str(task_profile), "sha256": digest(profile_bytes)},
        "task_profile": task_ref, "instructions": instruction_ref, "wrapper": wrapper_ref,
        "environment_python_adapter": environment_ref, "environment_python_route": env_metadata,
        "tool_path": str(tool), "tool_and_wrapper_same_inode": True,
        "client_source_sha256": digest(client.encode()), "native_shell_source_sha256": digest(shell.encode()),
        "instruction_bytes_unchanged": instruction_bytes == Path(instruction_ref["path"]).read_bytes(),
        "only_task_profile_change": "agent.codex_bin", "environment_variable_changes": {
            "ALFWORLD_TASK_CONFIG": task_ref["path"],
            "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": instruction_ref["path"],
            "ALFWORLD_ENV_PYTHON": environment_ref["path"],
        },
    }


def environment_observation(settings):
    with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    if (health.get("ready") is not True or health.get("environment_processes") != 4
            or health.get("io_concurrency") != 4 or health.get("action_request_protocol") != PROTOCOL
            or not health.get("server_instance_id")
            or health.get("environment_policy") != "sticky-process-environments-durable-io-v1"):
        raise ValueError("The new entry requires the ready four-process request-ID environment")
    extension = health.get("environment_extension", {})
    if (extension.get("policy") != health["environment_policy"]
            or not isinstance(extension.get("source_path"), str)
            or digest(Path(extension["source_path"]).read_bytes()) != extension.get("source_sha256")):
        raise ValueError("Environment extension source does not match its health identity")
    if settings.get("environment_extension") != extension:
        raise ValueError("Environment settings and live service extension identities differ")
    return health


def entry(config):
    from omegaconf import OmegaConf, open_dict

    from skill_harness_examples import train_spawn_safe

    run = Path(config.skill_harness.artifact_root)
    settings_path = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
    settings = json.loads(settings_path.read_bytes())
    health = environment_observation(settings)
    adapter = prepare_adapter(run, os.environ["ALFWORLD_TASK_CONFIG"],
                              os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"])
    os.environ.update(adapter["environment_variable_changes"])
    with open_dict(config):
        config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path = (
            adapter["task_profile"]["path"]
        )
    source = Path(__file__).resolve()
    receipt = {
        "schema_version": 1, "kind": "prelaunch_parallel_environment_client_observation",
        "observed_at_ns": time.time_ns(), "driver_path": str(source),
        "driver_sha256": digest(source.read_bytes()), "client_adapter": adapter,
        "environment_health": health,
        "environment_config_ref": {"path": str(settings_path), "sha256": digest(settings_path.read_bytes())},
        "core_fingerprint_scope": "Legacy core component; new environment/client behavior is supplemented here",
        "hash_link_scope": "Immutable Task profile path/hash binds the actual initialized runner configuration",
        "spawn_extension": train_spawn_safe.extension_identity(),
        "not_a_post_init_or_served_model_receipt": True,
    }
    # Only configuration metadata is retained; redact secret-bearing keys first.
    from verl.experimental.skill_harness.train import _redact

    receipt["effective_config_before_spawn_entry"] = _redact(OmegaConf.to_container(config, resolve=True))
    reference = train_spawn_safe._base()._publish(
        run / "driver_runs" / (str(time.time_ns()) + ".parallel_client.json"), receipt,
    )
    print("PARALLEL_CLIENT_OBSERVER_REF " + json.dumps(reference, sort_keys=True), flush=True)
    return train_spawn_safe.entry(config)


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
