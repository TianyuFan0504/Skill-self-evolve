#!/usr/bin/env python3
"""Prepare an offline ALFWorld run and supervise the real Codex RL entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def cpu_quota() -> int:
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    p = Path("/sys/fs/cgroup/cpu.max")
    if p.is_file():
        quota, period = p.read_text().split()
        if quota != "max":
            count = min(count, max(1, int(quota) // int(period)))
    return max(1, count - min(4, count // 8))


def gpu_count() -> int:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = len(output.splitlines())
    return min(count, len([value for value in visible.split(",") if value])) if visible is not None else count


def archive_previous_attempt(run: Path) -> None:
    """Preserve terminal attempt metadata while the full training log stays append-only."""
    if not (run / "status.json").is_file() or not (run / "training.log").exists():
        return
    previous = json.loads((run / "status.json").read_text())
    if previous.get("state") == "running":
        raise RuntimeError("A running attempt must be reconciled before replacing its metadata")
    folder = run / "attempt_history" / str(time.time_ns())
    folder.mkdir(parents=True)
    for name in (
        "status.json",
        "training-manifest.json",
        "launch_manifest.json",
        "resolved-config.yaml",
        "hydra_overrides.json",
        "preflight.json",
        "supervisor-input.json",
    ):
        source = run / name
        if source.is_file():
            shutil.copy2(source, folder / name)
    write_json(
        folder / "log-reference.json",
        {
            "path": str(run / "training.log"),
            "bytes_at_attempt_end": (run / "training.log").stat().st_size,
        },
    )


def install_codex_runtime(run: Path, benchmark: Path, capacity: int) -> tuple[Path, Path]:
    """Install only public instructions/launcher outside the private NAS tree."""
    public = Path("/opt/alfworld-tools/skill_self_evolve") / hashlib.sha256(str(run).encode()).hexdigest()[:16]
    public.mkdir(parents=True, exist_ok=True, mode=0o755)
    public.chmod(0o755)
    instructions = public / "solver_system.md"
    shutil.copyfile(RECIPE / "prompts/solver_system.md", instructions)
    instructions.chmod(0o644)
    wrapper = public / "codex-entrypoint"
    shutil.copyfile(RECIPE / "scripts/codex_entrypoint.sh", wrapper)
    wrapper.chmod(0o755)
    helper = Path("/opt/alfworld-tools/alfworld-step")
    if not helper.is_file():
        shutil.copyfile(benchmark / "adapter/alfworld_step.py", helper)
        helper.chmod(0o755)
    source = benchmark / "adapter/model_catalog_qwen35_4b.json"
    catalog = json.loads(source.read_text())
    if not catalog.get("models"):
        raise RuntimeError(f"The native Codex model catalog is empty: {source}")
    for model in catalog["models"]:
        model["slug"] = "Qwen3.5-4B-nothink"
        model["display_name"] = "Qwen3.5-4B-nothink"
        model["context_window"] = capacity
        model["supports_parallel_tool_calls"] = True
        model["base_instructions"] = instructions.read_text()
    catalog_path = run / "model_catalog.json"
    write_json(catalog_path, catalog)
    os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"] = str(instructions)
    task_config = run / "task_config.yaml"
    # JSON is a YAML subset, avoiding another dependency in this bootstrap CLI.
    write_json(
        task_config,
        [
            {
                "name": "alfworld_skill_training",
                "sandbox": {"provider": "local", "runtime_timeout": int(os.environ.get("AGENT_TIMEOUT", "900"))},
                "agent": {
                    "name": "codex",
                    "run_timeout": int(os.environ.get("AGENT_TIMEOUT", "900")),
                    "codex_bin": str(wrapper),
                    "model_catalog_path": str(catalog_path),
                },
            }
        ],
    )
    return task_config, instructions


def start_environment(run: Path, benchmark: Path, source: Path) -> tuple[int, bool]:
    config = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
    if config.is_file():
        pid_file = run / "environment-server.pid"
        pid = int(pid_file.read_text()) if pid_file.is_file() else 0
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        settings = json.loads(config.read_text())
        if set(settings.get("allowed_splits", [])) != {"train"}:
            raise RuntimeError("A training run cannot attach to a service that allows official evaluation games")
        try:
            with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=5) as response:
                health = json.load(response)
            expected = json.loads((run / "data/split_manifest.json").read_text())["catalog_digest"]
            if health.get("ready") is True and health.get("catalog_digest") == expected:
                return pid, False
            raise RuntimeError("The existing environment service has incompatible state")
        except (OSError, ValueError) as error:
            if alive:
                raise RuntimeError(
                    "The existing environment process is alive but its health could not be confirmed"
                ) from error
            config.rename(config.with_name("environment.stale-" + str(time.time_ns()) + ".json"))
    python = os.environ.get("ALFWORLD_ENV_PYTHON", str(benchmark / "env/bin/python"))
    script = source / "verl/experimental/skill_harness/alfworld/server.py"
    argv = [
        python,
        str(script),
        "--benchmark-root",
        str(benchmark),
        "--run-root",
        str(run),
        "--config-path",
        str(config),
        "--catalog-path",
        str(run / "data/catalog.json"),
        "--allowed-splits",
        "train",
        "--port",
        "0",
    ]
    with (run / "environment-server.log").open("a") as output:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    (run / "environment-server.pid").write_text(str(process.pid) + "\n")
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(
                f"ALFWorld service exited with {process.returncode}; see {run / 'environment-server.log'}"
            )
        if config.is_file():
            try:
                settings = json.loads(config.read_text())
                with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=1) as response:
                    health = json.load(response)
                if health.get("ready") is True:
                    config.chmod(0o600)
                    return process.pid, True
            except (OSError, ValueError, KeyError):
                pass
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError(f"ALFWorld service did not publish {config} within 30 seconds")


def supervise(run: Path, argv: list[str], service_pid: int | None) -> int:
    started = time.time()
    status = run / "status.json"
    with (run / "training.log").open("a", buffering=1) as output:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=os.environ["VERL_SOURCE_ROOT"],
            start_new_session=True,
            close_fds=True,
        )
        write_json(
            status,
            {
                "state": "running",
                "started_at_epoch": started,
                "training_pid": process.pid,
                "supervisor_pid": os.getpid(),
                "environment_pid": service_pid,
                "command": argv,
            },
        )

        def forward_signal(number: int, _frame: object) -> None:
            if process.poll() is None:
                os.killpg(process.pid, number)

        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)
        code = process.wait()
    (run / "exit-code.txt").write_text(str(code) + "\n")
    write_json(
        status,
        {
            "state": "completed" if code == 0 else "failed",
            "exit_code": code,
            "started_at_epoch": started,
            "finished_at_epoch": time.time(),
            "duration_seconds": round(time.time() - started, 2),
            "training_pid": process.pid,
            "environment_pid": service_pid,
            "command": argv,
        },
    )
    if (run / "metrics.jsonl").is_file():
        with (run / "plot-metrics.log").open("a") as output:
            subprocess.run(
                [sys.executable, str(RECIPE / "scripts/plot_metrics.py"), str(run)],
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--foreground", action="store_true", help="Wait for training instead of detaching the supervisor"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Write data/configuration without starting services or GPUs"
    )
    parser.add_argument(
        "--preflight-only", action="store_true", help="Prepare and run CPU import/service checks, then stop"
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing prepared/checkpointed run directory")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    args, overrides = parser.parse_known_args()
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    os.umask(0o077)
    if args.supervise:
        payload = json.loads(args.supervise.read_text())
        return supervise(Path(payload["run_dir"]), payload["command"], payload["environment_pid"])

    source = Path(os.environ["VERL_SOURCE_ROOT"]).resolve()
    sys.path.insert(0, str(source))
    from verl.experimental.skill_harness.bootstrap import configure_runtime

    provenance = configure_runtime(register_tasks=False)
    benchmark = Path(os.environ["ALFWORLD_BENCHMARK_ROOT"]).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = (args.run_dir or benchmark / "training_runs" / f"codex_qwen35_4b_{stamp}").resolve()
    if run.exists() and (run / "training.log").exists() and not args.resume:
        raise RuntimeError(f"An existing training run requires --resume: {run}")
    if args.resume and not run.is_dir():
        raise RuntimeError(f"Cannot resume a missing directory: {run}")
    if (run / "status.json").is_file():
        previous = json.loads((run / "status.json").read_text())
        pid = previous.get("training_pid")
        if previous.get("state") == "running" and pid:
            proc = Path("/proc") / str(pid)
            try:
                command = (proc / "cmdline").read_bytes()
                environment = (proc / "environ").read_bytes().split(b"\0")
                owned = ("SKILL_RUN_DIR=" + str(run)).encode() in environment
                if owned and b"verl.experimental.skill_harness.train" in command:
                    raise RuntimeError(f"Training process {pid} is still alive; observe the existing run instead")
            except (FileNotFoundError, PermissionError):
                pass
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    run.chmod(0o700)
    archive_previous_attempt(run)
    data = run / "data"
    data.mkdir(exist_ok=True)
    runtime = run / "runtime"
    runtime.mkdir(exist_ok=True)
    prompt_length = int(os.environ.setdefault("MAX_PROMPT_LENGTH", "4096"))
    response_length = int(os.environ.setdefault("MAX_RESPONSE_LENGTH", "32768"))
    capacity = prompt_length + response_length
    os.environ.setdefault("MAX_MODEL_LENGTH", str(capacity))
    if int(os.environ["MAX_MODEL_LENGTH"]) != capacity:
        raise RuntimeError("MAX_MODEL_LENGTH must equal MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH in this recipe")
    os.environ.setdefault("TRAIN_MAX_TOKENS_PER_GPU", str(capacity))
    os.environ.setdefault("RAY_NUM_CPUS", str(cpu_quota()))
    os.environ.setdefault("TRAINING_GPUS", str(gpu_count()))
    if int(os.environ["TRAINING_GPUS"]) < 1:
        raise RuntimeError("Training requires at least one visible GPU")
    if int(os.environ.get("ROLLOUT_N", "4")) < 2:
        raise RuntimeError("The solver GRPO recipe requires ROLLOUT_N >= 2")
    os.environ.update(
        {
            "SKILL_RUN_DIR": str(run),
            "SKILL_DATA_DIR": str(data),
            "ALFWORLD_DATA_DIR": str(data),
            "ALFWORLD_SERVER_CONFIG": str(runtime / "environment.json"),
            "EXPERIMENT_NAME": run.name,
            "RESUME_MODE": "auto" if args.resume else "disable",
        }
    )
    package = (
        "/usr/local/lib/node_modules/@openai/codex/node_modules/"
        "@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl"
    )
    os.environ.setdefault("ALFWORLD_CODEX_PACKAGE_ROOT", package)
    os.environ.setdefault("CODEX_NATIVE_BIN", str(Path(package) / "bin/codex"))
    os.environ.setdefault("MODEL_PATH", "/workspace/models/Qwen3.5-4B-nothink")
    if not (data / "split_manifest.json").is_file():
        command = [
            sys.executable,
            str(source / "verl/experimental/skill_harness/alfworld/prepare_data.py"),
            "--benchmark-root",
            str(benchmark),
            "--output-dir",
            str(data),
            "--run-dir",
            str(run),
            "--dev-fraction",
            os.environ.get("DEV_FRACTION", "0.1"),
            "--seed",
            os.environ.get("DATA_SEED", "42"),
            "--prompt-file",
            str(RECIPE / "prompts/solver_task.txt"),
        ]
        subprocess.run(command, check=True)
    task_config, instructions = install_codex_runtime(run, benchmark, capacity)
    os.environ["ALFWORLD_TASK_CONFIG"] = str(task_config)
    argv = [
        sys.executable,
        "-m",
        "verl.experimental.skill_harness.train",
        "--config-path=" + str(RECIPE / "config"),
        "--config-name=alfworld_train",
        *overrides,
    ]
    write_json(run / "hydra_overrides.json", overrides)
    public_settings = {
        "SKILL_RUN_DIR",
        "SKILL_DATA_DIR",
        "ALFWORLD_DATA_DIR",
        "ALFWORLD_BENCHMARK_ROOT",
        "ALFWORLD_SERVER_CONFIG",
        "ALFWORLD_CODEX_PACKAGE_ROOT",
        "ALFWORLD_SOLVER_INSTRUCTIONS_FILE",
        "ALFWORLD_TASK_CONFIG",
        "ALFWORLD_ENV_PYTHON",
        "UNI_AGENT_SOURCE_ROOT",
        "UNI_AGENT_DEPENDENCY_SITE",
        "MODEL_PATH",
        "VERL_SOURCE_ROOT",
        "CODEX_NATIVE_BIN",
        "TRAINING_GPUS",
        "RAY_NUM_CPUS",
        "MAX_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "MAX_MODEL_LENGTH",
        "TRAIN_MAX_TOKENS_PER_GPU",
        "TRAIN_BATCH_SIZE",
        "ROLLOUT_N",
        "ROLLOUT_CONCURRENCY",
        "TOTAL_TRAINING_STEPS",
        "LEARNING_RATE",
        "DATA_SEED",
        "DEV_FRACTION",
        "DEV_MAX_SAMPLES",
        "TRAINER_MODE",
        "RESUME_MODE",
    }
    safe_env = {name: os.environ[name] for name in public_settings if name in os.environ}
    write_json(
        run / "launch_manifest.json",
        {
            "created_at": stamp,
            "run_dir": str(run),
            "imports": provenance,
            "command": argv,
            "environment_paths": safe_env,
            "solver_system_sha256": hashlib.sha256(instructions.read_bytes()).hexdigest(),
            "solver_task_sha256": hashlib.sha256((RECIPE / "prompts/solver_task.txt").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((RECIPE / "config/alfworld_train.yaml").read_bytes()).hexdigest(),
        },
    )
    if args.prepare_only:
        write_json(run / "status.json", {"state": "prepared", "command": argv})
        print(json.dumps({"state": "prepared", "run_dir": str(run)}, indent=2))
        return 0
    service_pid, started_service = start_environment(run, benchmark, source)
    preflight = [
        sys.executable,
        str(RECIPE / "scripts/preflight.py"),
        "--run-dir",
        str(run),
        "--require-service",
        "--overrides-file",
        str(run / "hydra_overrides.json"),
    ]
    if args.allow_busy_gpus:
        preflight.append("--allow-busy-gpus")
    try:
        subprocess.run(preflight, check=True)
    except BaseException:
        if started_service:
            os.kill(service_pid, signal.SIGTERM)
        raise
    if args.preflight_only:
        if started_service:
            os.kill(service_pid, signal.SIGTERM)
        write_json(run / "status.json", {"state": "preflight_passed", "environment_pid": service_pid})
        return 0
    if args.foreground:
        return supervise(run, argv, service_pid)
    payload = run / "supervisor-input.json"
    write_json(payload, {"run_dir": str(run), "command": argv, "environment_pid": service_pid})
    with (run / "supervisor.log").open("a") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--supervise", str(payload)],
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=source,
            start_new_session=True,
            close_fds=True,
        )
    (run / "supervisor.pid").write_text(str(process.pid) + "\n")
    print(
        json.dumps(
            {
                "state": "submitted",
                "run_dir": str(run),
                "supervisor_pid": process.pid,
                "status_file": str(run / "status.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
