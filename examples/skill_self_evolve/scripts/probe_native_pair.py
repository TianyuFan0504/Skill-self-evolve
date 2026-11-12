#!/usr/bin/env python3
"""Run one input-control diagnostic with the archived v4 skill and future task.

Preparation uses only the standard library. --execute explicitly starts one
owned local Ray cluster, a frozen official rollout route and the native Codex
pair. It never creates a learner, optimizer, credit ledger or admission writer.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
SOURCE_RUN = Path("/root/nas_chaohuang/skill_training/runs/alfworld_joint_smoke_v4_20261112")
COLLECTION = "0bb3c733d0963f8a9dc0535b38e912bb8ccf6e8c18aa49c9a15f592015ab75bc"
FUTURE_TASK = "train_6408c0ae5173d1bf3ac38651"
PRODUCER_TASK = "train_13f5d21048f1b70d237bd27c"
CONTEXT_VERSION = "host-skill-context-v2"
INPUT_VERIFIER_VERSION = "exact-native-skill-only-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and "payload" in value:
        expected = value.get("sha256", value.get("payload_hash"))
        if expected != digest(value["payload"]):
            raise ValueError(f"Immutable envelope digest mismatch: {path}")
        return value["payload"]
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def tree_manifest(root):
    """Hash every regular input file; reject links and special files."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"An ordinary input directory is required: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"Input tree contains a link or special file: {path}")
        if stat.S_ISREG(mode):
            content = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    return {"files": files, "tree_sha256": digest(files)}


def clone_immutable_tree(source, target):
    """Copy bytes, then prove both source and clone still match the first read."""
    import shutil

    before = tree_manifest(source)
    shutil.copytree(source, target, symlinks=False)
    if tree_manifest(source) != before or tree_manifest(target) != before:
        raise ValueError("An input tree changed while cloning")
    return before


def archived_inputs(source):
    """Select the actual S0/window and its persisted original future-B row."""
    state = read_json(source / "controller/state.json")
    window_id = "window_" + COLLECTION
    window = state["pending"][window_id]
    if window["record_id"] != "curate-" + COLLECTION or window["producer_task_id"] != PRODUCER_TASK:
        raise ValueError("The source is not the archived v4 S0 producer")
    cohort = state["cohorts"][window["cohort_id"]]
    matches = [
        (key, row)
        for key, row in cohort["assignments"].items()
        if row["tools_kwargs"]["task"]["metadata"].get("stable_game_id") == FUTURE_TASK
    ]
    if len(matches) != 1:
        raise ValueError("Exactly one archived original future-B assignment is required")
    probe_id, fields = matches[0]
    task = fields["tools_kwargs"]["task"]
    metadata = task["metadata"]
    if (
        task["name"] != "alfworld_skill_training"
        or metadata.get("split") != "train"
        or metadata.get("dataset_phase") != "train"
        or metadata.get("game_id") != FUTURE_TASK
        or metadata.get("source_step") != 2
        or window["opened_step"] != 1
        or metadata.get("task_type") != window["task_type"]
        or FUTURE_TASK == PRODUCER_TASK
    ):
        raise ValueError("The diagnostic requires the unchanged actual downstream optimization-train B")
    if not isinstance(fields.get("raw_prompt"), list) or not fields["raw_prompt"]:
        raise ValueError("The archived original B prompt is missing")
    return {
        "window_id": window_id,
        "probe_id": probe_id,
        "window": window,
        "sample_fields": fields,
        "original_assignment_sha256": digest(fields),
    }


def prepare_inputs(source, run):
    import shutil

    source, run = Path(source).resolve(), Path(run).resolve()
    if run == source or run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("Use a fresh sibling diagnostic run, outside the archived source")
    if run.exists():
        raise FileExistsError(f"The one-pair diagnostic never overwrites an existing run: {run}")
    original = archived_inputs(source)
    run.mkdir(parents=True, mode=0o700)
    store = clone_immutable_tree(source / "skill_store", run / "skill_store")
    data = clone_immutable_tree(source / "data", run / "data")
    shutil.copyfile(source / "resolved-config.yaml", run / "source-resolved-config.yaml")
    manifest = read_json(run / "data/split_manifest.json")
    metadata = original["sample_fields"]["tools_kwargs"]["task"]["metadata"]
    if metadata["catalog_digest"] != manifest["catalog_digest"]:
        raise ValueError("The original task and copied service catalog differ")
    fields = copy.deepcopy(original["sample_fields"])
    fields["tools_kwargs"]["task"]["metadata"]["run_dir"] = str(run)
    # The official context pin replaces all old skill-loading metadata per branch.
    payload = {
        "schema_version": 1,
        "source_run": str(source),
        "run_dir": str(run),
        **original,
        "sample_fields": fields,
        "source_store": store,
        "copied_data": data,
        "source_config_sha256": hashlib.sha256((run / "source-resolved-config.yaml").read_bytes()).hexdigest(),
        "metadata_rebinding": ["tools_kwargs.task.metadata.run_dir"],
        "purpose": "One native input-control pair; no utility label, admission or learning claim",
    }
    write_new(run / "probe-input.json", payload)
    for path in (run / "skill_store").rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    (run / "skill_store").chmod(0o500)
    return payload


def validate_prepared(run):
    run = Path(run).resolve()
    payload = read_json(run / "probe-input.json")
    if payload["run_dir"] != str(run):
        raise ValueError("Prepared diagnostic belongs to another path")
    if tree_manifest(run / "skill_store") != payload["source_store"]:
        raise ValueError("The cloned immutable SkillStore changed")
    if tree_manifest(run / "data") != payload["copied_data"]:
        raise ValueError("The copied original task data changed")
    if (
        hashlib.sha256((run / "source-resolved-config.yaml").read_bytes()).hexdigest()
        != payload["source_config_sha256"]
    ):
        raise ValueError("The archived configuration changed")
    original = archived_inputs(Path(payload["source_run"]))
    if original["original_assignment_sha256"] != payload["original_assignment_sha256"]:
        raise ValueError("The archived future-B assignment changed")
    expected = copy.deepcopy(original["sample_fields"])
    expected["tools_kwargs"]["task"]["metadata"]["run_dir"] = str(run)
    if payload["sample_fields"] != expected or payload["window"] != original["window"]:
        raise ValueError("Prepared task/snapshot identity differs from archived S0/B")
    return payload


def launcher_module():
    specification = importlib.util.spec_from_file_location("_native_pair_recipe_launcher", RECIPE / "scripts/launch.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def stop_owned_service(run):
    """Only signal a session-leading ALFWorld service bound to this exact run."""
    path = run / "environment-server.pid"
    if not path.is_file():
        return {"status": "not_started"}
    pid = int(path.read_text())

    def owned():
        try:
            argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            return os.getpgid(pid) == pid and str(run).encode() in argv and any(x.endswith(b"/server.py") for x in argv)
        except (FileNotFoundError, ProcessLookupError):
            return False

    if not owned():
        return {"status": "not_alive_or_not_owned", "pid": pid}
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        if not owned():
            return {"status": "stopped", "pid": pid}
        time.sleep(0.1)
    if owned():
        os.killpg(pid, signal.SIGKILL)
    return {"status": "kill_requested", "pid": pid}


def preserve_ray_logs(scratch, run):
    import shutil

    files = []
    for path in sorted(scratch.rglob("*")):
        if path.is_file() and not path.is_symlink() and "logs" in path.relative_to(scratch).parts:
            target = run / "ray_logs" / path.relative_to(scratch)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files.append({"path": str(target.relative_to(run)), "size": target.stat().st_size})
    return files


def selected_gpu_state():
    """Inspect one explicitly selected physical GPU without creating CUDA state."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].isdigit():
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one physical GPU index")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = dict(line.split(",", 1) for line in inventory.splitlines() if line.strip())
    device = devices[visible[0]].strip()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    pids = [
        int(pid.strip())
        for row in active.splitlines()
        if row.strip()
        for uuid, pid in [row.split(",", 1)]
        if uuid.strip() == device
    ]
    return {"physical_index": int(visible[0]), "gpu_uuid": device, "compute_pids": sorted(pids)}


def cleanup_owned(ray, *, ray_owned, scratch, temporary, run):
    """Finish independent cleanup actions even if another action fails."""
    import shutil

    report = {"ray_owned": ray_owned, "errors": []}

    def attempt(name, callback):
        try:
            report[name] = callback()
        except BaseException as error:
            report["errors"].append({"action": name, "error_type": type(error).__name__})

    if ray_owned:
        attempt("ray_shutdown", ray.shutdown)
    attempt("ray_initialized_after_shutdown", ray.is_initialized)
    attempt("environment", lambda: stop_owned_service(run))
    attempt("preserved_ray_logs", lambda: preserve_ray_logs(scratch, run))
    attempt("scratch_removed", lambda: shutil.rmtree(scratch))
    attempt("temporary_removed", lambda: shutil.rmtree(temporary))

    def released_gpu():
        for _ in range(50):
            state = selected_gpu_state()
            if not state["compute_pids"]:
                return state
            time.sleep(0.2)
        return state

    attempt("gpu_after_shutdown", released_gpu)
    report["cleanup_confirmed"] = bool(
        not report["errors"]
        and report.get("ray_initialized_after_shutdown") is False
        and not report.get("gpu_after_shutdown", {"compute_pids": ["unknown"]})["compute_pids"]
        and report.get("environment", {}).get("status") in ("not_started", "stopped", "not_alive_or_not_owned")
    )
    report["finished_at"] = time.time()
    write_new(run / "cleanup.json", report)
    return report


async def run_pair(config, payload, run):
    """Use the existing frozen route, official Gateway and unmodified collector."""
    from verl.experimental.skill_harness.alfworld import protocol as environment_protocol
    from verl.experimental.skill_harness.context import CONTEXT_BUILDER_VERSION
    from verl.experimental.skill_harness.credit import CreditProtocol
    from verl.experimental.skill_harness.framework import loading_settings
    from verl.experimental.skill_harness.policy_routes import (
        PolicyRoute,
        PolicyRouteSpec,
        build_route_config,
        fingerprint_model,
    )
    from verl.experimental.skill_harness.probes import (
        PAIRED_INPUT_VERIFIER_VERSION,
        ProbeRequest,
        collect_counterfactual_probe,
        create_frozen_probe_gateway,
        probe_physical_id,
        validate_pair_snapshots,
    )
    from verl.experimental.skill_harness.store import SkillRef, SkillStore
    from verl.workers.rollout.llm_server import LLMServerManager

    if CONTEXT_BUILDER_VERSION != CONTEXT_VERSION or PAIRED_INPUT_VERIFIER_VERSION != INPUT_VERIFIER_VERSION:
        raise ValueError("The focused diagnostic requires the new exact-native-input source cut")
    model = await asyncio.to_thread(fingerprint_model, config.actor_rollout_ref.model.path)
    window = payload["window"]
    if model["fingerprint"] != window["protocol"]["evaluator_version"]:
        raise ValueError("The frozen model is not the original v4 evaluator")
    service = read_json(os.environ["ALFWORLD_SERVER_CONFIG"])
    manifest = read_json(config.skill_harness.split_manifest)
    if set(service["allowed_splits"]) != {"train"} or service["catalog_digest"] != manifest["catalog_digest"]:
        raise ValueError("The diagnostic requires the copied catalog and a train-only owned service")
    environment_source = Path(environment_protocol.__file__).resolve().parent
    environment_version = "alfworld:" + digest(
        {
            "catalog_digest": manifest["catalog_digest"],
            "max_steps": service["max_steps"],
            "code": {
                name: hashlib.sha256((environment_source / name).read_bytes()).hexdigest()
                for name in ("server.py", "protocol.py")
            },
        }
    )
    protocol = CreditProtocol(
        **{
            **window["protocol"],
            "probe_protocol_version": "native-paired-greedy:v2",
            "environment_version": environment_version,
        }
    )
    request = ProbeRequest(
        payload["probe_id"],
        payload["window_id"],
        window["record_id"],
        SkillRef(**window["skill_ref"]),
        window["with_snapshot_id"],
        window["without_snapshot_id"],
        model["fingerprint"],
        protocol,
        2,
    )
    store = SkillStore(run / "skill_store", read_only=True)
    validate_pair_snapshots(store, request)
    spec = PolicyRouteSpec(
        "focused_credit",
        "frozen",
        1,
        0,
        model["fingerprint"],
        model_path=model["model_path"],
        tensor_parallel_size=1,
        reference_seed=42,
    )
    route_config = build_route_config(config, spec)
    manager = await LLMServerManager.create(config=route_config, worker_group=None, start_rank=0)
    if len(manager.get_replicas()) != spec.replica_count:
        raise RuntimeError("Official standalone manager did not create exactly one frozen replica")
    route = PolicyRoute(spec, route_config, manager, model_manifest=model)
    gateway = None
    try:
        gateway = await create_frozen_probe_gateway(route_config, route)
        write_new(run / "route-attestation.json", {"route": route.describe(), "model_manifest": model})
        event = await collect_counterfactual_probe(
            request=request,
            sample_fields=payload["sample_fields"],
            store=store,
            loading=loading_settings(route_config),
            gateway_manager=gateway,
            artifact_root=run,
            task_config_path=os.environ["ALFWORLD_TASK_CONFIG"],
            frozen_attestation=route.attestation,
        )
        audit_path = run / "credit_probes" / probe_physical_id(request) / "paired-input-audit.json"
        audit = read_json(audit_path)
        if (
            audit.get("status") != "observed"
            or audit.get("actual_native_inputs_differ_only_by_target_skill") is not True
        ):
            raise ValueError("The actual native inputs did not pass the exact skill-only gate")
        return {
            "schema_version": 1,
            "status": "input_control_passed",
            "input_audit_path": str(audit_path),
            "input_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "event_id": event.event_id,
            "context_builder_version": CONTEXT_BUILDER_VERSION,
            "input_verifier_version": PAIRED_INPUT_VERIFIER_VERSION,
            "raw_outcome_scores": {"with_skill": event.with_skill.score, "without_skill": event.without_skill.score},
            "utility_label_created": False,
            "skill_admission_performed": False,
            "learner_created": False,
            "limitation": "One pair establishes input control, not skill benefit or learning.",
        }
    finally:
        if gateway is not None:
            closed = False
            try:
                await asyncio.wait_for(gateway.shutdown(), timeout=30)
                closed = True
            finally:
                write_new(run / "gateway-shutdown.json", {"shutdown_confirmed": closed, "time": time.time()})


def execute(run, *, timeout_seconds=1800):
    """Allocate only after explicit --execute, and always shut down our local job."""
    run = Path(run).resolve()
    payload = validate_prepared(run)
    write_new(run / "execution-intent.json", {"started_at": time.time(), "pid": os.getpid(), "one_pair_only": True})
    source = Path(os.environ.get("VERL_SOURCE_ROOT", RECIPE.parents[1])).resolve()
    sys.path.insert(0, str(source))
    from verl.experimental.skill_harness.bootstrap import configure_runtime, runtime_env_config

    provenance = configure_runtime()
    import ray
    from omegaconf import OmegaConf, open_dict

    if ray.is_initialized():
        raise RuntimeError("The focused diagnostic must own a new local Ray cluster")
    launcher = launcher_module()
    config = OmegaConf.load(run / "source-resolved-config.yaml")
    benchmark = Path(os.environ.get("ALFWORLD_BENCHMARK_ROOT", "/root/nas_chaohuang/benchmarks/alfworld")).resolve()
    short = hashlib.sha256(str(run).encode()).hexdigest()[:8]
    scratch = Path("/dev/shm") / ("pairray_" + short)
    temporary = Path("/dev/shm") / ("pairtmp_" + short)
    scratch.mkdir(mode=0o700, exist_ok=False)
    temporary.mkdir(mode=0o700, exist_ok=False)
    (run / "runtime").mkdir(exist_ok=True)
    package = (
        "/usr/local/lib/node_modules/@openai/codex/node_modules/"
        "@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl"
    )
    os.environ.update(
        SKILL_RUN_DIR=str(run),
        SKILL_DATA_DIR=str(run / "data"),
        ALFWORLD_DATA_DIR=str(run / "data"),
        ALFWORLD_SERVER_CONFIG=str(run / "runtime/environment.json"),
        TMPDIR=str(temporary),
        RAY_TMPDIR=str(scratch),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        UV_OFFLINE="1",
        ALFWORLD_BENCHMARK_ROOT=str(benchmark),
    )
    os.environ.setdefault("ALFWORLD_CODEX_PACKAGE_ROOT", package)
    os.environ.setdefault("CODEX_NATIVE_BIN", str(Path(package) / "bin/codex"))
    ray_owned = False
    result = None
    try:
        gpu_before = selected_gpu_state()
        write_new(run / "gpu-before.json", gpu_before)
        if gpu_before["compute_pids"]:
            raise RuntimeError("The selected GPU is busy; this diagnostic will not share or stop it")
        capacity = int(
            config.actor_rollout_ref.rollout.prompt_length + config.actor_rollout_ref.rollout.response_length
        )
        task_config, _ = launcher.install_codex_runtime(run, benchmark, capacity)
        os.environ["ALFWORLD_TASK_CONFIG"] = str(task_config)
        with open_dict(config):
            current_rollout = OmegaConf.load(RECIPE / "config/alfworld_common.yaml").actor_rollout_ref.rollout
            config.actor_rollout_ref.rollout.mode = current_rollout.mode
            config.actor_rollout_ref.rollout.engine_kwargs = OmegaConf.merge(
                config.actor_rollout_ref.rollout.get("engine_kwargs", {}), current_rollout.engine_kwargs
            )
            config.skill_harness.artifact_root = str(run)
            config.skill_harness.store_root = str(run / "skill_store")
            config.skill_harness.split_manifest = str(run / "data/split_manifest.json")
            config.skill_harness.terminal.binding_root = str(run / "terminal_bindings")
            config.actor_rollout_ref.rollout.custom.agent_framework.gateway_count = 1
        (run / "diagnostic-config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
        launcher.start_environment(run, benchmark, source)
        input_assets = [
            source / "verl/experimental/skill_harness" / name
            for name in (
                "probes.py", "context.py", "alfworld/accounts.py", "alfworld/task.py", "alfworld/native_files.py",
                "alfworld/backend_terminal.py", "alfworld/budget.py", "alfworld/server.py",
            )
        ]
        input_assets.extend(
            RECIPE / name
            for name in (
                "scripts/codex_entrypoint.sh",
                "scripts/launch.py",
                "prompts/solver_system.md",
                "prompts/solver_task.txt",
                "scripts/probe_native_pair.py",
                "config/alfworld_common.yaml",
            )
        )
        write_new(
            run / "runtime-provenance.json",
            {
                "imports": provenance,
                "scratch": str(scratch),
                "temporary": str(temporary),
                "runtime_env": runtime_env_config(),
                "input_assets": [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in input_assets
                ],
            },
        )
        ray_owned = True
        ray.init(
            address="local",
            namespace="focused_pair_" + short,
            num_cpus=8,
            num_gpus=1,
            _temp_dir=str(scratch),
            runtime_env=runtime_env_config(),
        )
        result = asyncio.run(asyncio.wait_for(run_pair(config, payload, run), timeout=timeout_seconds))
        validate_prepared(run)
        if tree_manifest(Path(payload["source_run"]) / "skill_store") != payload["source_store"]:
            raise ValueError("The archived source SkillStore changed during this diagnostic")
        write_new(run / "result.json", result)
        return result
    except BaseException as error:
        write_new(
            run / "failure.json",
            {
                "error_type": type(error).__name__,
                "time": time.time(),
                "resampling_allowed": False,
                "original_captures_preserved": True,
            },
        )
        raise
    finally:
        original_error = sys.exc_info()[0]
        cleanup = cleanup_owned(ray, ray_owned=ray_owned, scratch=scratch, temporary=temporary, run=run)
        if original_error is None and not cleanup["cleanup_confirmed"]:
            raise RuntimeError("Input audit finished but owned resource cleanup is not confirmed; see cleanup.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly allocate one frozen GPU and run the pair")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.run_dir.exists():
        prepare_inputs(args.source_run, args.run_dir)
    if not args.execute:
        payload = validate_prepared(args.run_dir)
        print(
            json.dumps(
                {
                    "status": "prepared_no_gpu",
                    "run_dir": str(args.run_dir.resolve()),
                    "source_store_tree_sha256": payload["source_store"]["tree_sha256"],
                },
                indent=2,
            )
        )
        return 0

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Owned diagnostic termination requested")

    signal.signal(signal.SIGTERM, terminate)
    print(json.dumps(execute(args.run_dir, timeout_seconds=args.timeout_seconds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
