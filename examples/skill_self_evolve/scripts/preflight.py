#!/usr/bin/env python3
"""CPU-only import, source-provenance and data-boundary checks for training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import pwd
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-service", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    parser.add_argument("--overrides-file", type=Path)
    parser.add_argument("--config-name", choices=("alfworld_train", "alfworld_joint"), default="alfworld_train")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report: dict[str, object] = {"cpu_only_import_checks": True, "run_dir": str(run)}
    gpu_csv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], text=True
    ).strip()
    report["gpu_inventory"] = gpu_csv.splitlines()
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    active_pids = [line.strip() for line in busy.splitlines() if line.strip() != str(os.getpid()) and line.strip()]
    if not args.allow_busy_gpus:
        check(not active_pids, "GPU jobs are already active; use --allow-busy-gpus only after identifying them")
    report["other_gpu_process_ids"] = active_pids

    # This process only validates imports/configuration. Its child training process
    # is launched separately with the parent's original GPU visibility.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    source = Path(os.environ.get("VERL_SOURCE_ROOT", "/workspace/verl-0.9.1")).resolve()
    sys.path.insert(0, str(source))
    from verl.experimental.skill_harness.bootstrap import configure_runtime, runtime_env_config

    report["imports"] = configure_runtime(register_tasks=True)
    modules = (
        "uni_agent.framework.entry",
        "uni_agent.agents.codex.agent",
        "verl.experimental.skill_harness.alfworld.task",
        "verl.experimental.skill_harness.alfworld.runner",
    )
    report["module_origins"] = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
    entry = importlib.import_module("uni_agent.framework.entry")
    signature = inspect.signature(entry.AgentFrameworkRolloutAdapter.create)
    check({"config", "llm_client"}.issubset(signature.parameters), "Uni-Agent adapter create() contract changed")
    report["package_versions"] = {}
    for package in ("verl", "torch", "vllm", "ray", "transferqueue", "transformers", "uni-agent"):
        try:
            report["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][package] = "source-only"
    check(report["package_versions"]["verl"] == "0.9.1", "Editable distribution metadata shadows verl 0.9.1")

    manifest = json.loads((run / "data/split_manifest.json").read_text())
    phases = manifest["phase_game_ids"]
    train, dev = set(phases["train"]), set(phases["dev"])
    tests = set(phases["test_seen"]) | set(phases["test_unseen"])
    check(bool(train) and bool(dev), "Training and train-derived dev partitions must both be nonempty")
    check(not train & dev and not (train | dev) & tests, "Training/dev/test game identities overlap")
    check(manifest["evaluation_games_used_for_training"] is False, "Official evaluation games entered training")
    for value in manifest["parquet_files"].values():
        p = Path(value["path"])
        check(hashlib.sha256(p.read_bytes()).hexdigest() == value["sha256"], f"Prepared data changed: {p}")
    report["split_game_counts"] = {name: len(ids) for name, ids in phases.items()}
    report["catalog_digest"] = manifest["catalog_digest"]

    model_path = Path(os.environ["MODEL_PATH"]).resolve()
    check((model_path / "config.json").is_file(), f"A local model config is required: {model_path}")
    model = json.loads((model_path / "config.json").read_text())
    report["model"] = {
        "path": str(model_path),
        "architectures": model.get("architectures"),
        "model_type": model.get("model_type"),
    }
    check(Path(os.environ["CODEX_NATIVE_BIN"]).is_file(), "Native Codex executable is missing")
    check(os.access(os.environ["CODEX_NATIVE_BIN"], os.X_OK), "Native Codex executable is not executable")
    check(Path(os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"]).is_file(), "Solver instructions are missing")
    account = pwd.getpwnam("alfworld_eval")
    report["agent_user"] = {"name": account.pw_name, "uid": account.pw_uid}
    check(os.access("/opt/alfworld-tools/alfworld-step", os.X_OK), "The game tool is missing")
    task_config = json.loads(Path(os.environ["ALFWORLD_TASK_CONFIG"]).read_text())
    work_root = Path("/tmp/alfworld-skill-episodes")
    work_root.mkdir(mode=0o711, exist_ok=True)
    check(not work_root.is_symlink() and work_root.stat().st_uid == os.geteuid(), "Unsafe episode work parent")
    work_root.chmod(0o711)
    with tempfile.TemporaryDirectory(prefix="preflight-", dir=work_root) as directory:
        os.chown(directory, account.pw_uid, account.pw_gid)
        os.chmod(directory, 0o700)
        native_check = subprocess.run(
            ["/usr/sbin/runuser", "-u", account.pw_name, "--", "/bin/sh", "-c",
             'cd "$1" && shift && exec "$@"', "native-check", directory,
             task_config[0]["agent"]["codex_bin"], "--version"],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                "CODEX_NATIVE_BIN": os.environ["CODEX_NATIVE_BIN"],
                "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"],
            },
            text=True, capture_output=True, timeout=15,
        )
    check(native_check.returncode == 0, f"Restricted native Codex launch failed: {native_check.stderr.strip()}")
    report["restricted_native_codex_version"] = native_check.stdout.strip()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = json.loads(args.overrides_file.read_text()) if args.overrides_file else []
    with initialize_config_dir(config_dir=str(RECIPE / "config"), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    from verl.checkpoint_engine import CheckpointEngineRegistry

    backend = cfg.actor_rollout_ref.rollout.checkpoint_engine.backend
    CheckpointEngineRegistry.get(backend)
    report["solver_checkpoint_backend"] = backend
    check(cfg.actor_rollout_ref.rollout.n >= 2, "GRPO requires stochastic response groups")
    check(cfg.data.train_files.endswith("/train.parquet"), "The training recipe must consume train.parquet")
    check(cfg.data.val_files.endswith("/dev.parquet"), "Scheduled validation must use train-derived dev.parquet")
    check(cfg.actor_rollout_ref.rollout.calculate_log_probs, "Behavior log probabilities must be retained")
    expected_adapter = (
        "verl.experimental.skill_harness.framework.SkillAwareRolloutAdapter"
        if cfg.skill_harness.mode == "skill_joint"
        else "uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    )
    check(cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class == expected_adapter,
          "The recipe must use the configured official Uni-Agent extension path")
    if cfg.skill_harness.mode == "skill_joint":
        from verl.experimental.skill_harness.curate import provision_native_curator_runtime

        joint = cfg.skill_harness.joint
        needed = int(cfg.trainer.n_gpus_per_node) + int(joint.curator_gpus) + int(joint.credit_gpus)
        check(needed <= len(report["gpu_inventory"]), "Joint policy routes exceed the actual visible GPU inventory")
        check(int(joint.solver_gpus) == int(cfg.trainer.n_gpus_per_node), "Joint solver GPU counts disagree")
        try:
            curator_account = pwd.getpwnam("alfworld_curator")
        except KeyError:
            subprocess.run(
                ["/usr/sbin/useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin",
                 "alfworld_curator"], check=True,
            )
            curator_account = pwd.getpwnam("alfworld_curator")
        check(curator_account.pw_uid != 0, "The curator must use a separate restricted account")
        native_runtime = provision_native_curator_runtime(
            public_root=joint.get("curator_public_root", "/opt/alfworld-tools/skill_self_evolve/curator"),
            work_root=joint.get("curator_work_root", "/tmp/alfworld-curator"),
            native_bin=os.environ["CODEX_NATIVE_BIN"],
            package_root=os.environ["ALFWORLD_CODEX_PACKAGE_ROOT"],
            model_catalog_path=run / "model_catalog.json",
            model_name=joint.get("model_name", "Qwen3.5-4B-nothink"),
        )
        report["curator_runtime"] = native_runtime.manifest()
        report["joint_gpu_allocation"] = {
            "solver": int(cfg.trainer.n_gpus_per_node), "curator": int(joint.curator_gpus),
            "credit": int(joint.credit_gpus),
        }
    (run / "resolved-config.preflight.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    report["ray_worker_setup_hook"] = runtime_env_config()["worker_process_setup_hook"]
    if args.require_service:
        settings = json.loads(Path(os.environ["ALFWORLD_SERVER_CONFIG"]).read_text())
        check(settings["catalog_digest"] == manifest["catalog_digest"], "Service/data catalog digests differ")
        check(
            set(settings.get("allowed_splits", [])) == {"train"}, "Training service permits official evaluation games"
        )
        with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=5) as response:
            health = json.load(response)
        check(health.get("ready") is True, "The ALFWorld service is not ready")
        check(health.get("catalog_digest") == manifest["catalog_digest"], "Health endpoint has a different catalog")
        report["environment_health"] = health
    report["passed"] = True
    (run / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
