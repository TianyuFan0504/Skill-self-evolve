#!/usr/bin/env python3
"""Prepare and supervise one ALFWorld joint run from an explicit deployment JSON."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
ROOT = RECIPE.parents[1]
LINUX_FACTORY = "verl.experimental.skill_harness.backends.linux:LinuxAccountPool"
SMOKE_OVERRIDES = [
    "trainer.total_training_steps=6",
    "trainer.save_freq=6",
    "skill_harness.evaluation.final=false",
    "data.train_batch_size=16",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.rollout.n=4",
    "skill_harness.joint.probes_per_step=16",
    "skill_harness.joint.replay_batch_size=8",
    "skill_harness.joint.min_delay_steps=2",
    "skill_harness.joint.required_exposures=2",
]


def absolute_path(value, label):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path: {value}")
    return path


def substitute_paths(value, run_dir):
    """Expand only two literal path placeholders, without shell evaluation."""
    if isinstance(value, str):
        return value.replace("{run_dir}", str(run_dir)).replace("{run_id}", run_dir.name)
    if isinstance(value, list):
        return [substitute_paths(item, run_dir) for item in value]
    if isinstance(value, dict):
        return {key: substitute_paths(item, run_dir) for key, item in value.items()}
    return value


def write_private_json(path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def process_start_ticks(pid):
    """Linux PID identity, independent of wall-clock changes and PID reuse."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Supervisor:
    """Own only children started here; never discover or stop older runs."""

    def __init__(self, run_dir, environment):
        self.run_dir, self.environment = run_dir, environment
        self.children, self.previous_handlers = [], {}
        self.stop_signal = None

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous_handlers[number] = signal.signal(number, self.request_stop)
        return self

    def request_stop(self, number, _frame):
        self.stop_signal = number

    def check_stop(self):
        if self.stop_signal is not None:
            raise InterruptedError(f"Launcher received signal {self.stop_signal}")

    def spawn(self, name, command):
        self.check_stop()
        print(f"[{name}] starting", flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=self.environment, start_new_session=True)
        self.children.append(
            (
                process,
                {
                    "name": name,
                    "pid": process.pid,
                    "start_ticks": process_start_ticks(process.pid),
                },
            )
        )
        self.record()
        return process

    def record(self):
        write_private_json(
            self.run_dir / "launch-processes.json",
            {
                "launcher_pid": os.getpid(),
                "launcher_start_ticks": process_start_ticks(os.getpid()),
                "children": [{**identity, "returncode": process.poll()} for process, identity in self.children],
            },
        )

    def wait(self, process, *, backend=None):
        while process.poll() is None:
            self.check_stop()
            if backend is not None and backend.poll() is not None:
                raise RuntimeError(f"Environment backend exited during training ({backend.returncode})")
            time.sleep(0.2)
        self.check_stop()
        self.record()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, process.args)

    def run(self, name, command):
        self.wait(self.spawn(name, command))

    def __exit__(self, _kind, _value, _traceback):
        errors = []
        try:
            for process, identity in reversed(self.children):
                if process.poll() is not None:
                    continue
                ticks = identity["start_ticks"]
                try:
                    # Let the leader clean its workers before group fallback.
                    # Python's SIGINT handler enters the trainer's finally
                    # block, allowing ray.shutdown() before forced cleanup.
                    graceful = signal.SIGINT if identity["name"] == "training" else signal.SIGTERM
                    stops = [(False, graceful, 20), (True, signal.SIGTERM, 15), (True, signal.SIGKILL, 5)]
                    for index, (whole_group, number, timeout) in enumerate(stops):
                        if index and process.poll() is not None:
                            break
                        if ticks is not None and process_start_ticks(process.pid) != ticks:
                            break
                        try:
                            (os.killpg if whole_group else os.kill)(process.pid, number)
                            process.wait(timeout=timeout)
                            break
                        except subprocess.TimeoutExpired:
                            if number == signal.SIGKILL:
                                raise
                except ProcessLookupError:
                    pass
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{identity['name']}: {error}")
            self.record()
        finally:
            for number, previous in self.previous_handlers.items():
                signal.signal(number, previous)
        if errors:
            message = "Owned process cleanup failed: " + "; ".join(errors)
            if _kind is None:
                raise RuntimeError(message)
            print(message, file=sys.stderr, flush=True)


def wait_for_backend(supervisor, process, config_path, *, processes, io_workers, digest, timeout):
    deadline = time.monotonic() + timeout
    last_error = "service configuration has not been published"
    while time.monotonic() < deadline:
        supervisor.check_stop()
        if process.poll() is not None:
            raise RuntimeError(f"Environment backend exited before readiness ({process.returncode})")
        try:
            settings = json.loads(config_path.read_text())
            if settings.get("pid") != process.pid:
                raise ValueError("Service configuration does not belong to the owned backend PID")
            parts = urllib.parse.urlsplit(settings["url"])
            if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("Environment service must publish a loopback HTTP endpoint")
            with urllib.request.urlopen(settings["url"] + "/health", timeout=2) as response:
                health = json.load(response)
            expected = {
                "ready": True,
                "catalog_digest": digest,
                "environment_processes": processes,
                "io_concurrency": io_workers,
                "action_request_ids": True,
            }
            if any(health.get(key) != value for key, value in expected.items()):
                raise ValueError("Environment health does not match the prepared catalog and parallel configuration")
            if len(set(health.get("environment_worker_pids", []))) != processes:
                raise ValueError("Environment worker process identities are incomplete")
            return settings
        except (OSError, ValueError, KeyError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(0.2)
    raise TimeoutError(f"Environment readiness timed out after {timeout}s: {last_error}")


def deployment_environment(profile, run_dir):
    declared = profile.get("environment", {})
    if not isinstance(declared, dict) or any(not isinstance(value, str) for value in declared.values()):
        raise ValueError("deployment.environment must map environment names to strings")
    required = (
        "TRAINING_PYTHON",
        "ALFWORLD_ENV_PYTHON",
        "MODEL_PATH",
        "MODEL_NAME",
        "ALFWORLD_BENCHMARK_ROOT",
        "CODEX_NATIVE_BIN",
        "ALFWORLD_CODEX_PACKAGE_ROOT",
        "ALFWORLD_SOLVER_INSTRUCTIONS_FILE",
        "CURATOR_USER",
        "CURATOR_PUBLIC_ROOT",
        "CURATOR_WORK_ROOT",
    )
    missing = [name for name in required if not declared.get(name)]
    if missing:
        raise ValueError("Deployment is missing environment values: " + ", ".join(missing))
    for name in required:
        if name not in ("MODEL_NAME", "CURATOR_USER"):
            absolute_path(declared[name], name)
    # Driver and Ray workers use installed packages in the selected environment.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
        )
    }
    environment.update(declared)
    if environment.get("PYTHONPATH") or environment.get("PYTHONHOME"):
        raise ValueError("Install dependencies in TRAINING_PYTHON rather than configuring Python import overlays")
    environment["PATH"] = str(Path(declared["TRAINING_PYTHON"]).parent) + os.pathsep + declared.get("PATH", os.defpath)
    environment.update(
        {
            "SKILL_RUN_DIR": str(run_dir),
            "SKILL_DATA_DIR": str(run_dir / "data"),
            "ALFWORLD_DATA_DIR": str(run_dir / "data"),
            "ALFWORLD_RUNTIME_CONFIG": str(run_dir / "config/runtime.json"),
            "ALFWORLD_SERVER_CONFIG": str(run_dir / "config/environment.json"),
            "ALFWORLD_ENV_SERVER": declared.get(
                "ALFWORLD_ENV_SERVER", str(RECIPE / "scripts/parallel_environment_server.py")
            ),
        }
    )
    return environment


def publish_assets(profile):
    for item in profile.get("public_assets", []):
        source = absolute_path(item["source"], "public asset source")
        destination = absolute_path(item["destination"], "public asset destination")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        if source != destination:
            shutil.copyfile(source, destination)
        mode = item.get("mode", "0644")
        destination.chmod(int(mode, 8) if isinstance(mode, str) else mode)
    tool_dir = absolute_path(profile["tool_dir"], "tool_dir")
    tool_python = absolute_path(profile["tool_python"], "tool_python")
    tool_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    client = tool_dir / "alfworld_request_client.py"
    shutil.copyfile(RECIPE / "scripts/alfworld_request_client.py", client)
    client.chmod(0o644)
    command = tool_dir / "alfworld-step"
    command.write_text(f'#!/bin/sh\nexec {shlex.quote(str(tool_python))} {shlex.quote(str(client))} "$@"\n')
    command.chmod(0o755)


def launch(profile, run_dir, *, smoke=False, overrides=()):
    profile = substitute_paths(profile, run_dir)
    environment = deployment_environment(profile, run_dir)
    runtime = dict(profile["runtime"])
    initialize = runtime.pop("initialize_if_missing", False)
    provider = runtime.get("account_pool", {})
    if not isinstance(provider.get("factory"), str) or not isinstance(provider.get("kwargs"), dict):
        raise ValueError("runtime.account_pool requires factory and kwargs")
    if initialize and provider["factory"] != LINUX_FACTORY:
        raise ValueError("initialize_if_missing is supported only for the explicit optional Linux provider")
    if not isinstance(profile.get("hydra_overrides", []), list):
        raise ValueError("hydra_overrides must be a list")
    processes, io_workers = int(profile.get("environment_processes", 4)), int(profile.get("io_workers", 4))
    if min(processes, io_workers) < 1:
        raise ValueError("Environment process and I/O counts must be positive")
    # A fresh run never overwrites previous data, capabilities or account state.
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    (run_dir / "config").mkdir(mode=0o700)
    write_private_json(Path(environment["ALFWORLD_RUNTIME_CONFIG"]), runtime)
    if "task_config" in profile:
        task_path = run_dir / "config/task.json"
        write_private_json(task_path, profile["task_config"])
        environment["ALFWORLD_TASK_CONFIG"] = str(task_path)
    elif not environment.get("ALFWORLD_TASK_CONFIG"):
        raise ValueError("Supply task_config or environment.ALFWORLD_TASK_CONFIG")
    publish_assets(profile)
    python = environment["TRAINING_PYTHON"]
    with Supervisor(run_dir, environment) as supervisor:
        if initialize:
            registry = absolute_path(provider["kwargs"]["root"], "account pool root")
            if not registry.exists():
                supervisor.run(
                    "initialize-native-runtime",
                    [
                        python,
                        "-c",
                        (
                            "from verl.experimental.skill_harness.alfworld.accounts import AccountPool; "
                            "AccountPool().initialize()"
                        ),
                    ],
                )
        supervisor.run(
            "prepare-data",
            [
                python,
                str(ROOT / "verl/experimental/skill_harness/alfworld/prepare_data.py"),
                "--benchmark-root",
                environment["ALFWORLD_BENCHMARK_ROOT"],
                "--output-dir",
                environment["SKILL_DATA_DIR"],
                "--run-dir",
                str(run_dir),
                "--prompt-file",
                str(RECIPE / "prompts/solver_task.txt"),
            ],
        )
        manifest = json.loads((run_dir / "data/split_manifest.json").read_text())
        backend = supervisor.spawn(
            "environment",
            [
                environment["ALFWORLD_ENV_PYTHON"],
                environment["ALFWORLD_ENV_SERVER"],
                "--benchmark-root",
                environment["ALFWORLD_BENCHMARK_ROOT"],
                "--run-root",
                str(run_dir),
                "--catalog-path",
                str(run_dir / "data/catalog.json"),
                "--config-path",
                environment["ALFWORLD_SERVER_CONFIG"],
                "--allowed-splits",
                "train",
                "--environment-processes",
                str(processes),
                "--io-workers",
                str(io_workers),
                "--max-steps",
                str(profile.get("max_steps", 50)),
                "--budget-seconds",
                str(profile.get("budget_seconds", 900)),
            ],
        )
        settings = wait_for_backend(
            supervisor,
            backend,
            Path(environment["ALFWORLD_SERVER_CONFIG"]),
            processes=processes,
            io_workers=io_workers,
            digest=manifest["catalog_digest"],
            timeout=float(profile.get("readiness_timeout_seconds", 180)),
        )
        settings["account_pool"] = runtime["account_pool"]
        settings["native_policy"] = runtime.get(
            "native_policy",
            {
                "instructions_file": environment["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"],
            },
        )
        settings["process_start_ticks"] = process_start_ticks(backend.pid)
        write_private_json(Path(environment["ALFWORLD_SERVER_CONFIG"]), settings)
        arguments = [*profile.get("hydra_overrides", []), *(SMOKE_OVERRIDES if smoke else []), *overrides]
        supervisor.run("preflight", [python, str(RECIPE / "scripts/preflight.py"), *arguments])
        trainer = supervisor.spawn(
            "training",
            [
                python,
                "-m",
                "verl.experimental.skill_harness.train",
                f"--config-path={RECIPE / 'config'}",
                "--config-name=alfworld_joint",
                *arguments,
            ],
        )
        supervisor.wait(trainer, backend=backend)
    print(f"Training completed. Outputs: {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment", type=Path, required=True, help="Explicit host deployment JSON (keep outside Git)"
    )
    parser.add_argument(
        "--run-dir", type=Path, help="Fresh absolute output directory; otherwise deployment.environment.SKILL_RUN_DIR"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Six real joint updates with delayed credit and replay enabled"
    )
    args, overrides = parser.parse_known_args()
    try:
        profile = json.loads(args.deployment.read_text())
        output = args.run_dir or profile.get("environment", {}).get("SKILL_RUN_DIR")
        if output is None:
            raise ValueError("Supply --run-dir or environment.SKILL_RUN_DIR in the deployment")
        launch(profile, absolute_path(output, "run directory"), smoke=args.smoke, overrides=overrides)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"Launch failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
