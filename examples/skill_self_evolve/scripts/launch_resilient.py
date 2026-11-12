#!/usr/bin/env python3
"""Prepare and supervise resilient joint training through the existing helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
DEFAULT_ENV = {
    "TRAINING_GPUS": "4", "TRAIN_BATCH_SIZE": "32", "PPO_MINI_BATCH_SIZE": "32", "ROLLOUT_N": "4",
    "ROLLOUT_CONCURRENCY": "32", "TOTAL_TRAINING_STEPS": "64", "TOTAL_EPOCHS": "4",
    "GATEWAY_COUNT": "8", "VAL_BATCH_SIZE": "512", "DEV_MAX_SAMPLES": "-1",
    "DEV_EVAL_FREQUENCY": "8", "SAVE_FREQUENCY": "1", "DATA_SEED": "42", "DEV_FRACTION": "0.1",
    "MAX_PROMPT_LENGTH": "4096", "MAX_RESPONSE_LENGTH": "32768", "MAX_MODEL_LENGTH": "36864",
    "TRAIN_MAX_TOKENS_PER_GPU": "36864", "AGENT_TIMEOUT": "900", "ENFORCE_EAGER": "true",
    "LR_WARMUP_STEPS": "0", "EVALUATE_OFFICIAL_TESTS": "true",
}


def existing_launcher():
    spec = importlib.util.spec_from_file_location("_existing_skill_launcher", RECIPE / "scripts/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--joint", action="store_true", help="Accepted for clarity; this launcher is joint-only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    if any("=" not in value or value.startswith("--") for value in overrides):
        parser.error("Unknown options; remaining arguments must be Hydra key=value overrides")
    for item in overrides:
        key, value = item.split("=", 1)
        normalized = key.lstrip("+~")
        if normalized == "skill_harness":
            parser.error("Use individual skill_harness fields; --run-dir controls artifact_root")
        if normalized == "skill_harness.artifact_root":
            if (
                key.startswith("~") or args.run_dir is None or "${" in value
                or not Path(value).is_absolute() or Path(value).resolve() != args.run_dir.resolve()
            ):
                parser.error("skill_harness.artifact_root must equal the explicit resolved --run-dir")
    fixed = {
        "trainer.val_before_train": "false", "trainer.val_only": "false",
        "skill_harness.evaluation.initial": "false",
        "actor_rollout_ref.rollout.n": "4",
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn":
            "skill_harness_examples.train_resilient_setup.safe_run_task",
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions": "32",
        "trainer.v1.sampler.custom_sampler.path": str(RECIPE / "scripts/train_resilient_setup.py"),
        "trainer.v1.sampler.custom_sampler.name": "ResilientReplayBuffer",
    }
    for item in overrides:
        key, value = item.split("=", 1)
        if key.lstrip("+") in fixed and value != fixed[key.lstrip("+")]:
            parser.error("This joint entrypoint fixes " + key + "=" + fixed[key.lstrip("+")])
    return args, [*overrides, *(key + "=" + value for key, value in fixed.items())]


def refuse_live_run(run, *, proc_root=Path("/proc")):
    if run is None:
        return
    run = Path(run).resolve()
    status = run / "status.json"
    if status.is_file():
        value = json.loads(status.read_bytes())
        pid = value.get("training_pid")
        if value.get("state") == "running" and pid and (proc_root / str(pid)).exists():
            raise RuntimeError("The recorded training process is still present; observe/reconcile this run")
    supervisor = run / "supervisor.pid"
    if supervisor.is_file():
        process = proc_root / supervisor.read_text().strip()
        if process.exists():
            environment = (process / "environ").read_bytes().split(b"\0")
            if ("SKILL_RUN_DIR=" + str(run)).encode() in environment:
                raise RuntimeError("This run already has a live owned supervisor")


def prepare(launcher, args, overrides):
    # The original CLI owns archival-before-rewrite and all environment/path
    # setup. Invoke it in-process so those environment values remain available.
    delegated = [str(RECIPE / "scripts/launch.py"), "--joint", "--prepare-only"]
    if args.run_dir is not None:
        delegated += ["--run-dir", str(args.run_dir)]
    if args.resume:
        delegated.append("--resume")
    original_argv = sys.argv
    try:
        sys.argv = [*delegated, *overrides]
        code = launcher.main()
    finally:
        sys.argv = original_argv
    if code:
        raise RuntimeError("Existing preparation failed with code " + str(code))
    return Path(os.environ["SKILL_RUN_DIR"])


def assert_run_binding(config, run):
    root = Path(config.skill_harness.artifact_root)
    if not root.is_absolute() or root.resolve() != run.resolve():
        raise ValueError("Prepared skill_harness.artifact_root differs from the actual run directory")


def prepared_run_config(run):
    from omegaconf import OmegaConf

    config = OmegaConf.load(run / "resolved-config.preflight.yaml")
    assert_run_binding(config, run)
    return config


def fresh_training_manifest(run, *, config=None):
    """Create the fresh original manifest once; never rewrite a resumed one."""
    if config is not None:
        assert_run_binding(config, run)
    if (run / "training-manifest.json").exists():
        return False
    if (run / "controller").exists() or any((run / "checkpoints").glob("global_step_*")):
        raise RuntimeError("Existing learned state lacks its original training manifest")
    if config is None:
        config = prepared_run_config(run)

    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment
    from verl.experimental.skill_harness.train import save_run_manifest
    from verl.utils.device import auto_set_device

    auto_set_device(config)
    apply_runtime_environment(config)
    save_run_manifest(config)
    return True


def launch_receipt(launcher, run, command, *, fresh_manifest, args):
    root = RECIPE.parent.parent
    assets = {
        "launcher": Path(__file__).resolve(), "shell": RECIPE / "run_resilient.sh",
        "driver": RECIPE / "scripts/train_resilient_setup.py",
        "quiescent_driver": RECIPE / "scripts/train_quiescent_eval.py",
        "namespace": root / "skill_harness_examples/__init__.py",
    }
    folder = run / "launcher_runs"
    folder.mkdir(exist_ok=True, mode=0o700)
    path = folder / (str(time.time_ns()) + ".resilient.json")
    launcher.write_json(path, {
        "schema_version": 1, "kind": "public_resilient_joint_launch",
        "command": command, "run_dir": str(run), "resume_requested": args.resume,
        "fresh_training_manifest_created": fresh_manifest,
        "original_training_manifest_preserved": (run / "training-manifest.json").exists() and not fresh_manifest,
        "training_manifest_sha256": (
            hashlib.sha256((run / "training-manifest.json").read_bytes()).hexdigest()
            if (run / "training-manifest.json").exists() else None
        ),
        "assets": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for name, path in assets.items()},
    })
    return path


def main(argv=None):
    args, overrides = arguments(sys.argv[1:] if argv is None else argv)
    os.umask(0o077)
    for name, default in DEFAULT_ENV.items():
        os.environ.setdefault(name, default)
    if os.environ["ROLLOUT_CONCURRENCY"] != "32" or os.environ["ROLLOUT_N"] != "4":
        raise ValueError("This joint profile requires ROLLOUT_CONCURRENCY=32 and ROLLOUT_N=4")
    refuse_live_run(args.run_dir)
    launcher = existing_launcher()
    run = prepare(launcher, args, overrides)
    original = json.loads((run / "launch_manifest.json").read_bytes())["command"]
    if original[1:3] != ["-m", "verl.experimental.skill_harness.train"]:
        raise ValueError("Unexpected existing preparation command")
    command = [original[0], str(RECIPE / "scripts/train_resilient_setup.py"), *original[3:]]
    if args.prepare_only:
        launcher.write_json(run / "status.json", {"state": "prepared", "command": command})
        receipt = launch_receipt(launcher, run, command, fresh_manifest=False, args=args)
        print(json.dumps({"state": "prepared", "run_dir": str(run), "resilient_launch_receipt": str(receipt)}))
        return 0
    service_pid, started = launcher.start_environment(
        run, Path(os.environ["ALFWORLD_BENCHMARK_ROOT"]), Path(os.environ["VERL_SOURCE_ROOT"])
    )
    preflight = [sys.executable, str(RECIPE / "scripts/preflight.py"), "--run-dir", str(run),
                 "--require-service", "--config-name", "alfworld_joint",
                 "--overrides-file", str(run / "hydra_overrides.json")]
    if args.allow_busy_gpus:
        preflight.append("--allow-busy-gpus")
    try:
        log = run / ("preflight-resilient-" + str(time.time_ns()) + ".log")
        with log.open("w") as output:
            checked = subprocess.run(preflight, stdout=output, stderr=subprocess.STDOUT, check=False)
        print(log.read_text(), end="", flush=True)
        checked.check_returncode()
        if args.preflight_only:
            if started:
                os.kill(service_pid, signal.SIGTERM)
            launcher.write_json(run / "status.json", {"state": "preflight_passed", "environment_pid": service_pid})
            launch_receipt(launcher, run, command, fresh_manifest=False, args=args)
            return 0
        config = prepared_run_config(run)
        created = fresh_training_manifest(run, config=config)
        receipt = launch_receipt(launcher, run, command, fresh_manifest=created, args=args)
        if args.foreground:
            return launcher.supervise(run, command, service_pid)
        payload = run / "supervisor-input-resilient.json"
        launcher.write_json(payload, {"run_dir": str(run), "command": command, "environment_pid": service_pid})
        with (run / "supervisor-resilient.log").open("a") as output:
            process = subprocess.Popen(
                [sys.executable, str(RECIPE / "scripts/launch.py"), "--supervise", str(payload)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                cwd=os.environ["VERL_SOURCE_ROOT"], start_new_session=True, close_fds=True,
            )
        (run / "supervisor.pid").write_text(str(process.pid) + "\n")
        print(json.dumps({"state": "submitted", "run_dir": str(run), "supervisor_pid": process.pid,
                          "status": str(run / "status.json"), "resilient_launch_receipt": str(receipt)}))
        return 0
    except BaseException:
        if started:
            os.kill(service_pid, signal.SIGTERM)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
