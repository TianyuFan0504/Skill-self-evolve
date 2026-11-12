#!/usr/bin/env python3
"""Read-only deferred-base versus trained-final official ALFWorld comparison.

No evaluation, model loading, checkpoint RPCs, GPU calls, network, or raw-file
rewrites. PyArrow is needed only to read the existing original parquet files.
Pure source utilities validate model-file hashes and skill reference text.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
BASE_FINGERPRINT = "670423a20133031174a5c4769f8b5aa9cd4ce3f541b503d0cd522dde72173a22"
SELECTOR_AST_SHA256 = "dd37f5e0f8a5214ee62e7c54dbcdeadcb65babab287cdd0781c20cbf14b66568"
SELECTOR_RELATIVE_PATH = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
COUNTS = {"test_seen": 140, "test_unseen": 134}
SPLITS = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}
GREEDY = {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1}
LIFECYCLE_DRIVER = "examples/skill_self_evolve/scripts/train_quiescent_eval.py"
SETUP_DRIVER = "examples/skill_self_evolve/scripts/train_resilient_setup.py"
SETUP_NAMESPACE = "skill_harness_examples/__init__.py"
SETUP_POLICY = "exact-pre-create-no-native-v1"
SETUP_RUNNER = "skill_harness_examples.train_resilient_setup.safe_run_task"
SPAWN_DRIVER = "examples/skill_self_evolve/scripts/train_spawn_safe.py"
SPAWN_MODULE = "skill_harness_examples.native_posix_spawn"
SPAWN_POLICY = "native-posix-spawn-boundary-v1"
SPAWN_RUNNER = "skill_harness_examples.train_spawn_safe.SpawnSafeRunner"
SPAWN_FILES = {
    "python": "examples/skill_self_evolve/scripts/native_posix_spawn.py",
    "c": "examples/skill_self_evolve/scripts/native_posix_spawn.c",
    "helper": "examples/skill_self_evolve/scripts/native_posix_spawn_helper.so",
}
SPAWN_EXPECTED_RECEIPT = "runtime/spawn_safe.expected.json"
PARALLEL_EXPECTED_RECEIPT = "runtime/parallel_client.expected.json"
PARALLEL_DRIVER = "examples/skill_self_evolve/scripts/train_parallel_client.py"
PARALLEL_CLIENT = "examples/skill_self_evolve/scripts/alfworld_request_client.py"
PARALLEL_SERVER = "examples/skill_self_evolve/scripts/parallel_environment_server.py"
PARALLEL_POLICY = "parallel-environment-idempotent-client-v1"
ENVIRONMENT_POLICY = "sticky-process-environments-durable-io-v1"
ACTION_PROTOCOL = "alfworld-step-request-id-v1"
DIRECT_POLICY = "parallel-native-direct-checkpoint-v1"
DIRECT_RESTORE_POLICY = "single-validation-direct-checkpoint-read-v1"
DIRECT_IDENTITY_FIELDS = ("policy", "driver_source_commit", "actual_entry", "files", "core_fingerprint_scope")
DIRECT_FILES = {
    "entry": "examples/skill_self_evolve/scripts/train_direct_checkpoint.py",
    "restore_provider": "examples/skill_self_evolve/scripts/direct_checkpoint_restore.py",
    "parallel_component": PARALLEL_DRIVER,
    "spawn_component": SPAWN_DRIVER,
    "checkpoint_core": "verl/experimental/skill_harness/joint_checkpoint.py",
    "trainer_core": "verl/experimental/skill_harness/joint_trainer.py",
}
ALL_NATIVE_POLICY = "parallel-direct-all-native-spawn-v1"
ALL_NATIVE_TRANSPORT = "solver-and-registered-curator-posix-spawn-v1"
ALL_NATIVE_MODULE = "skill_harness_examples.all_native_spawn"
ALL_NATIVE_HELPER_SHA256 = "18f8ebbba0202bba7d9f871d53fa1517e4f6f70cb061ffe316c45b24b80f520e"
ALL_NATIVE_SOLVER_HELPER_SHA256 = "7905a68b89453e586a777c77cc5e3b0f45e44696f18378ee2ea8cc8d61c35088"
ALL_NATIVE_IDENTITY_FIELDS = (
    "policy",
    "actual_entry",
    "driver_source_commit",
    "files",
    "all_role_native_spawn",
    "solver_spawn_component_identity",
    "core_fingerprint_scope",
)
ALL_NATIVE_FILES = {
    "entry": "examples/skill_self_evolve/scripts/train_all_native_spawn.py",
    "direct_component": DIRECT_FILES["entry"],
    "restore_provider": DIRECT_FILES["restore_provider"],
    "parallel_component": PARALLEL_DRIVER,
    "spawn_component": SPAWN_DRIVER,
}
CURATE_GENERATION_POLICY = "parallel-direct-all-native-spawn-curate-generation-v1"
CURATE_GENERATION_NAMESPACE_POLICY = "queued-source-generation-namespace-v1"
CURATE_GENERATION_MODULE = "skill_harness_examples.curate_generation"
CURATE_GENERATION_CONTROLLER = CURATE_GENERATION_MODULE + ".GenerationController"
CURATE_GENERATION_IDENTITY_FIELDS = (*ALL_NATIVE_IDENTITY_FIELDS, "curate_generation_extension")
CURATE_GENERATION_FILES = {
    **ALL_NATIVE_FILES,
    "entry": "examples/skill_self_evolve/scripts/train_curate_generation.py",
    "all_native_component": ALL_NATIVE_FILES["entry"],
}
CURATE_GENERATION_PROVIDER_FILES = {
    "provider": "examples/skill_self_evolve/scripts/curate_generation.py",
    "lifecycle_hook": SETUP_DRIVER,
}
CURATE_GENERATION_SCOPE = (
    "New current-branch draws for queued sources without curator_jobs; old actions and terminal states are preserved"
)


def all_native_shape(value):
    generation = value.get("policy") == CURATE_GENERATION_POLICY
    return {
        "generation": generation,
        "policy": CURATE_GENERATION_POLICY if generation else ALL_NATIVE_POLICY,
        "fields": CURATE_GENERATION_IDENTITY_FIELDS if generation else ALL_NATIVE_IDENTITY_FIELDS,
        "files": CURATE_GENERATION_FILES if generation else ALL_NATIVE_FILES,
        "stem": "curate_generation" if generation else "all_native_spawn",
        "initialized_kind": "initialized_curate_generation_entry"
        if generation
        else "initialized_all_native_spawn_entry",
        "prelaunch_kind": (
            "prelaunch_curate_generation_parallel_components"
            if generation
            else "prelaunch_all_native_spawn_parallel_components"
        ),
    }


ALL_NATIVE_TRANSPORT_FILES = {
    "python": "examples/skill_self_evolve/scripts/all_native_spawn.py",
    "c": "examples/skill_self_evolve/scripts/curator_posix_spawn.c",
    "helper": "examples/skill_self_evolve/scripts/curator_posix_spawn_helper.so",
}
TASK_PROFILE_FIELD = (
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path"
)
LIFECYCLE_IDENTITY = (
    "driver_path",
    "driver_sha256",
    "driver_source_commit",
    "original_training_manifest_sha256",
)
LIFECYCLE_SEAL = (
    "seal_origin",
    "optimizer_step",
    "checkpoint_cursor",
    "checkpoint_dir",
    "manifest_id",
    "snapshot_id",
    "store_root",
    "checkpoint_core_versions",
)
TRAINING_ONLY_SECTIONS = {
    "actor_rollout_ref.actor",
    "actor_rollout_ref.ref",
    "critic",
    "algorithm",
    "distillation",
    "skill_harness.curate_loss",
}
TRAINING_ONLY_FIELDS = {
    "trainer.save_freq",
    "trainer.test_freq",
    "trainer.critic_warmup",
    "trainer.v1.sync.train_batch_size",
    "trainer.v1.sampler.max_off_policy_threshold",
    "trainer.v1.sampler.max_off_policy_strategy",
    "trainer.v1.sampler.sync_refill_failed_groups",
    "data.train_batch_size",
    "data.gen_batch_size",
    "data.train_max_samples",
    "data.shuffle",
}
TRAINING_ONLY_FIELDS.update(
    "skill_harness.joint." + key
    for key in (
        "background_enabled",
        "curator_concurrency",
        "credit_concurrency",
        "max_source_backlog",
        "sources_per_step",
        "candidates_per_source",
        "probes_per_step",
        "required_exposures",
        "min_delay_steps",
        "replay_batch_size",
        "max_replay_lag_steps",
        "max_pending_windows",
        "window_seconds",
        "lease_seconds",
        "cost_weight",
        "max_label_updates",
        "admission_threshold",
        "frozen_model_path",
    )
)
RUNTIME_FILES = (
    "verl/workers/engine_workers.py",
    "verl/workers/utils/losses.py",
    "verl/workers/utils/padding.py",
    "verl/workers/engine/base.py",
    "verl/workers/engine/fsdp/transformer_impl.py",
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
    "verl/workers/rollout/llm_server.py",
    "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "verl/trainer/ppo/v1/trainer_base.py",
    "verl/trainer/ppo/v1/trainer_sync.py",
    "verl/trainer/ppo/core_algos.py",
    "examples/skill_self_evolve/scripts/codex_entrypoint.sh",
    "examples/skill_self_evolve/scripts/launch.py",
    "examples/skill_self_evolve/prompts/solver_system.md",
    "examples/skill_self_evolve/prompts/solver_task.txt",
    "examples/skill_self_evolve/config/alfworld_common.yaml",
    "examples/skill_self_evolve/config/alfworld_joint.yaml",
)
RUN_PATHS = {
    "data.train_files": "data/train.parquet",
    "data.val_files": "data/dev.parquet",
    "trainer.default_local_dir": "checkpoints",
    "trainer.validation_data_dir": "validation",
    "trainer.rollout_data_dir": "rollouts",
    "skill_harness.artifact_root": "",
    "skill_harness.store_root": "skill_store",
    "skill_harness.replay_root": "curate_replay",
    "skill_harness.ledger_root": "credit_ledger",
    "skill_harness.split_manifest": "data/split_manifest.json",
    "skill_harness.terminal.binding_root": "terminal_bindings",
    "actor_rollout_ref.rollout.custom.agent_framework.log_dir": "agent_logs",
    (
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path"
    ): "task_config.yaml",
    "ray_kwargs.ray_init.runtime_env.env_vars.SKILL_RUN_DIR": "",
    "ray_kwargs.ray_init.runtime_env.env_vars.SKILL_DATA_DIR": "data",
    "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_DATA_DIR": "data",
    "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_TASK_CONFIG": "task_config.yaml",
    "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_SERVER_CONFIG": "runtime/environment.json",
}


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INDEX = load_source("_official_comparison_index", Path(__file__).with_name("audit_training_run.py"))
Error = INDEX.EvidenceError


def pure_module(name):
    package = "_official_comparison_pure"
    if package not in sys.modules:
        value = types.ModuleType(package)
        value.__path__ = [str(REPO / "verl/experimental/skill_harness")]
        sys.modules[package] = value
    return importlib.import_module(package + "." + name)


def _model_fingerprint(path):
    return pure_module("policy_routes").fingerprint_model(path, force_rehash=True)


def need(condition, reason):
    if not condition:
        raise Error(reason)


def field(value, path):
    for key in path.split("."):
        value = value[key]
    return value


def optional_field(value, path, default=None):
    try:
        return field(value, path)
    except (KeyError, TypeError):
        return default


def set_existing_field(value, path, replacement):
    keys = path.split(".")
    parent = field(value, ".".join(keys[:-1]))
    need(keys[-1] in parent, "Missing original configuration field: " + path)
    parent[keys[-1]] = replacement


def exact_keys(value, keys, label):
    need(isinstance(value, dict) and set(value) == set(keys), label + ": unsupported or missing schema fields")


def parquet_rows(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise Error(
            "Existing training Python with PyArrow is required; no package is installed by this tool"
        ) from error
    return pq.read_table(path).to_pylist()


class Reader(INDEX.Reader):
    def hash_file(self, path):
        path = Path(path)
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 << 20):
                digest.update(chunk)
        self.observed[str(path)] = (before.st_size, before.st_mtime_ns)
        return digest.hexdigest()


def runtime_blobs(source, commit):
    need(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit), "Missing full source commit identity")
    try:
        text = subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "ls-tree",
                "-r",
                commit,
                "--",
                "verl/experimental/skill_harness",
                *RUNTIME_FILES,
            ],
            text=True,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Error("Historical runtime source is unavailable; comparison is unsupported") from error
    values = {}
    for line in text.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, identity = metadata.split()
        if path in RUNTIME_FILES or path.startswith("verl/experimental/skill_harness/") and path.endswith(".py"):
            need(kind == "blob" and mode in ("100644", "100755"), "Unexpected runtime source entry")
            values[path] = identity
    need(
        all(path in values for path in RUNTIME_FILES) and len(values) > len(RUNTIME_FILES),
        "Incomplete runtime source tree",
    )
    return values


def tokenizer_version(reader, path):
    digest, found = hashlib.sha256(), False
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        file = Path(path) / name
        if file.is_file():
            found = True
            digest.update(name.encode() + b"\0" + reader.bytes(file))
    need(found, "Missing actual tokenizer bytes")
    return "sha256:" + digest.hexdigest()


def _installed_gdn_selector():
    """Locate installed source through metadata, without importing vLLM."""
    distribution = importlib.metadata.distribution("vllm")
    return distribution.version, Path(distribution.locate_file(SELECTOR_RELATIVE_PATH))


def verified_selector(reader, expected_version):
    version, path = _installed_gdn_selector()
    need(version == expected_version == "0.24.0", "Unsupported installed GDN selector/package identity")
    content = reader.bytes(path)
    functions = [
        value
        for value in ast.parse(content).body
        if isinstance(value, ast.FunctionDef) and value.name == "_resolve_gdn_prefill_backend"
    ]
    need(len(functions) == 1, "Missing installed GDN selector")
    function = functions[0]
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ):
        function.body = function.body[1:]  # Documentation is not selector behavior.
    function.decorator_list = []
    identity = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
    need(identity == SELECTOR_AST_SHA256, "Installed GDN selector differs from the reviewed deterministic selector")
    return {
        "package_version": version,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "function_ast_sha256": identity,
        "requested_backend": "triton",
        "selected_backend_inferred": "triton",
    }


def owned_serve_cli(log):
    """Parse only complete actual pprint lists from owned vLLM server actors."""
    buffers, rows = {}, []
    for number, raw in enumerate(log.splitlines(), 1):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
        match = re.match(r"^\(vLLMHttpServer pid=(\d+)\)\s*(.*)$", line)
        if not match:
            continue
        pid, text = match.groups()
        if text.startswith(("['serve',", '["serve",')):
            buffers[pid] = {"first_line": number, "parts": []}
        if pid not in buffers:
            continue
        buffers[pid]["parts"].append(text)
        if len(buffers[pid]["parts"]) > 512:
            del buffers[pid]
            continue
        try:
            values = ast.literal_eval("\n".join(buffers[pid]["parts"]))
        except (ValueError, SyntaxError):
            continue
        if isinstance(values, list) and values[:1] == ["serve"] and all(isinstance(v, str) for v in values):
            rows.append(
                {
                    "actor_pid": int(pid),
                    "first_line": buffers[pid]["first_line"],
                    "last_line": number,
                    "arguments": values,
                }
            )
        del buffers[pid]
    return rows


def backend_evidence(reader, run_root, model_path, package_version):
    path = Path(run_root) / "training.log"
    log = reader.bytes(path).decode(errors="replace")
    cli = owned_serve_cli(log)
    for row in cli:
        args, choices = row["arguments"], []
        need(
            len(args) >= 2 and Path(args[1]).resolve() == Path(model_path).resolve(),
            "Owned vLLM CLI model identity differs",
        )
        for index, token in enumerate(args):
            if token in ("--gdn-prefill-backend", "--gdn_prefill_backend"):
                choices.append(args[index + 1] if index + 1 < len(args) else None)
            elif token.startswith(("--gdn-prefill-backend=", "--gdn_prefill_backend=")):
                choices.append(token.split("=", 1)[1])
        need(choices == ["triton"], "Missing, ambiguous or mismatched owned GDN CLI argument")
    kernel = [
        number
        for number, raw in enumerate(log.splitlines(), 1)
        if "Using Triton/FLA GDN prefill kernel (requested=triton" in raw
        and re.match(r"^\(vLLMHttpServer pid=\d+\)", re.sub(r"\x1b\[[0-9;]*m", "", raw))
        and re.search(r"\(Worker(?:_[A-Za-z0-9]+)? pid=\d+\)", raw)
    ]
    if kernel:
        return {
            "tier": "worker_kernel_log",
            "kernel_log_observed": True,
            "kernel_lines": kernel,
            "training_log": str(path),
            "owned_cli": cli,
            "active_backend": "triton",
        }
    need(cli, "No owned actual GDN CLI or worker kernel-selection evidence; comparison is unsupported")
    selector = verified_selector(reader, package_version)
    return {
        "tier": "owned_cli_and_verified_selector",
        "kernel_log_observed": False,
        "training_log": str(path),
        "owned_cli": cli,
        "selector_identity": selector,
        "active_backend": "triton",
        "scope": (
            "Inferred from actual owned CLI, matching config and reviewed installed selector; "
            "no per-call kernel trace observed."
        ),
    }


class Stage:
    def __init__(self, path, *, base, reader, read_parquet, expected_fingerprint):
        self.path, self.base, self.reader = Path(path).resolve(), base, reader
        self.name, self.version = ("initial", 0) if base else ("final", 64)
        need(
            self.path.name == self.name and self.path.parent.name == "evaluation",
            "Use explicit base initial and trained final stage paths",
        )
        self.root = self.path.parent.parent
        self.manifest = reader.json(self.root / "training-manifest.json")
        self.config = self.manifest["config"]
        self.launch = reader.json(self.root / "launch_manifest.json")
        self.env = self.launch["environment_paths"]
        self.preflight = reader.json(self.root / "preflight.json")
        self.protocol = reader.json(self.path / "protocol.json")
        self.summary = reader.json(self.path / "summary.json")
        self.index = INDEX.RunAudit(self.root)
        self.index.reader, self.index.config = reader, self.config
        separation = self.index.separation()
        need(
            separation["status"] == "declared_disjoint",
            "Invalid source catalog/split manifest: " + "; ".join(separation["reasons"]),
        )
        self.split = self.index.split
        self.catalog = reader.json(self.index.catalog_path)
        self.games = {row["game_id"]: row for row in self.catalog["games"]}
        self.read_parquet, self.expected_fingerprint = read_parquet, expected_fingerprint
        self.parquets, self.cases, self.allowed_differences = {}, {}, []
        self.parallel_extension, self.parallel_evidence = None, None
        self.direct_observations, self.direct_evidence = [], None
        self.all_native_observations, self.all_native_evidence = [], None
        self.all_native_fresh = False
        self.parallel_test_services, self.parallel_test_evidence = {}, {}
        self.setup_extension = None
        self.initialized_evidence = None
        self.prospective_runtime_evidence = None
        self.restore_bindings = {}
        self.check_protocol()
        self.check_identity()
        self.inference_config = self.config
        self.check_lifecycle_binding()
        if self.setup_extension is None and optional_field(
            self.config, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn"
        ) in (SETUP_RUNNER, SPAWN_RUNNER):
            need(self.base, "A trained safe-setup profile requires its initialized lifecycle receipt")
            source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
            runner = field(
                self.config, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn"
            )
            if runner == SPAWN_RUNNER:
                if self.parallel_extension is not None:
                    identity = self.parallel_extension["spawn_extension"]
                    path = self.root / PARALLEL_EXPECTED_RECEIPT
                else:
                    path = self.root / SPAWN_EXPECTED_RECEIPT
                    receipt = self.reader.json(path)
                    exact_keys(receipt, ("schema_version", "kind", "extension"), "Prospective spawn runtime identity")
                    need(
                        receipt["schema_version"] == 1 and receipt["kind"] == "prospective_spawn_safe_runtime_identity",
                        "Unknown prospective spawn runtime identity",
                    )
                    identity = receipt["extension"]
                need(
                    identity["policy"] == SPAWN_POLICY
                    and identity["driver_source_commit"] == self.manifest["source_commit"],
                    "Base prospective runtime identity differs from its actual source commit",
                )
                self.prospective_runtime_evidence = {
                    "path": str(path),
                    "sha256": self.reader.hash_file(path),
                    "scope": "Prelaunch helper/source identity; actual runner/concurrency use the original manifest.",
                }
            else:
                identity = {
                    "policy": SETUP_POLICY,
                    "driver_path": str(source / SETUP_DRIVER),
                    "driver_sha256": self.reader.hash_file(source / SETUP_DRIVER),
                    "namespace_path": str(source / SETUP_NAMESPACE),
                    "namespace_sha256": self.reader.hash_file(source / SETUP_NAMESPACE),
                    "driver_source_commit": self.manifest["source_commit"],
                }
            self.setup_extension = self.verify_setup_identity(identity)
            self.check_safe_setup_profile(self.config)
        if self.parallel_extension is not None:
            need(
                self.setup_extension is not None and self.setup_extension["policy"] == SPAWN_POLICY,
                "Parallel client profile requires its verified spawn extension",
            )
            self.check_parallel_config(self.inference_config)
        self.signature = self.config_signature(self.inference_config)

    def verified_checkpoint(self, checkpoint, *, manifest_id, clock, cursor):
        """Read COMPLETE publication/files only; no tensor loading or store constructors."""
        envelope = json.loads(self.reader.bytes(checkpoint / "joint_manifest.json"))
        exact_keys(envelope, ("manifest_id", "payload"), "Joint checkpoint envelope")
        payload = envelope["payload"]
        need(
            envelope["manifest_id"] == manifest_id == "sha256:" + INDEX.digest(payload)
            and payload.get("schema_version") == 1
            and payload.get("state") == "complete",
            "Checkpoint is not the intact sealed COMPLETE publication",
        )
        spec = pure_module("joint_checkpoint").JointCheckpointSpec(**payload["spec"])
        need(
            spec.optimizer_step == clock and spec.global_step == cursor and spec.world_size == 4,
            "COMPLETE checkpoint clock/cursor/world differs",
        )
        need(payload["training_state_paths"] == ["data.pt"], "Unsupported synchronous checkpoint data inventory")
        actual = set()
        for component in ("actor", "joint_state", "data.pt"):
            location = checkpoint / component
            for file in location.rglob("*") if location.is_dir() else (location,):
                relative = file.relative_to(checkpoint)
                need(not file.is_symlink(), "Checkpoint file inventory contains a symlink")
                if "locks" not in relative.parts and file.name != ".writer.lock" and file.is_file():
                    actual.add(relative.as_posix())
        need(actual == set(payload["files"]), "COMPLETE checkpoint file inventory differs")
        for relative, identity in payload["files"].items():
            safe = PurePosixPath(relative)
            need(not safe.is_absolute() and ".." not in safe.parts and str(safe) == relative, "Unsafe checkpoint file")
            file = checkpoint / relative
            need(
                identity == {"sha256": self.reader.hash_file(file), "size": file.stat().st_size},
                "COMPLETE checkpoint file digest differs: " + relative,
            )
        need(
            all(
                f"actor/{kind}_world_size_4_rank_{rank}.pt" in actual
                for rank in range(4)
                for kind in ("model", "optim", "extra_state")
            ),
            "Online model/optimizer/RNG rank inventory is incomplete",
        )
        ema = payload["ema_metadata"]
        need(
            len(ema) == 4
            and {row["file"] for row in ema} == {f"ema_rank_{rank:05d}.pt" for rank in range(4)}
            and all(row["last_optimizer_step"] == clock and "joint_state/ema/" + row["file"] in actual for row in ema),
            "COMPLETE checkpoint lacks all four matching EMA acknowledgements",
        )
        saved = self.reader.json(checkpoint / "joint_state/controller.json")
        need(
            saved["optimizer_step"] == clock
            and saved["prepared_update"] is None
            and saved["protocol_fingerprint"] == spec.protocol_fingerprint,
            "Saved controller does not match its sealed applied policy",
        )
        return payload, spec

    def verify_published_source(self, relative, commit, path, digest):
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        need(path == str(source / relative), "Unexpected extension source binding: " + relative)
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(source), "show", commit + ":" + relative],
                stderr=subprocess.PIPE,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise Error("Published extension source is unavailable: " + relative) from error
        need(
            hashlib.sha256(blob).hexdigest() == digest == self.reader.hash_file(path),
            "Extension bytes differ from published source: " + relative,
        )

    def verify_setup_identity(self, identity, *, require_loaded=False):
        spawn = identity.get("policy") == SPAWN_POLICY
        keys = ("policy", "driver_path", "driver_sha256", "namespace_path", "namespace_sha256", "driver_source_commit")
        if spawn:
            keys += ("composed_extensions",)
        exact_keys(identity, keys, "Safe-setup extension identity")
        need(identity["policy"] in (SETUP_POLICY, SPAWN_POLICY), "Unknown safe-setup admission policy")
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        commit = identity["driver_source_commit"]
        need(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit), "Missing setup source commit")
        for kind, relative in (("driver", SPAWN_DRIVER if spawn else SETUP_DRIVER), ("namespace", SETUP_NAMESPACE)):
            self.verify_published_source(relative, commit, identity[kind + "_path"], identity[kind + "_sha256"])
        if spawn:
            composed = identity["composed_extensions"]
            exact_keys(composed, ("resilient_setup", "quiescent_final", "native_spawn"), "Composed spawn extensions")
            resilient = self.verify_setup_identity(composed["resilient_setup"])
            need(
                resilient["policy"] == SETUP_POLICY
                and resilient["driver_source_commit"] == commit
                and resilient["namespace_sha256"] == identity["namespace_sha256"],
                "Composed resilient source identity differs",
            )
            quiescent = composed["quiescent_final"]
            exact_keys(quiescent, ("path", "sha256"), "Composed quiescent source")
            self.verify_published_source(LIFECYCLE_DRIVER, commit, quiescent["path"], quiescent["sha256"])
            native = composed["native_spawn"]
            exact_keys(native, ("policy", *SPAWN_FILES), "Native spawn identity")
            need(native["policy"] == "native-posix-spawn-v1", "Unknown native spawn policy")
            for kind, relative in SPAWN_FILES.items():
                entry = native[kind]
                loaded = kind == "helper" and "loaded_sha256" in entry
                exact_keys(
                    entry,
                    ("path", "sha256", *(("loaded_sha256",) if loaded else ())),
                    "Native spawn " + kind + " identity",
                )
                if kind != "helper":
                    self.verify_published_source(relative, commit, entry["path"], entry["sha256"])
                else:
                    need(
                        entry["path"] == str(source / relative)
                        and self.reader.hash_file(entry["path"]) == entry["sha256"],
                        "Native spawn helper binding/bytes differ",
                    )
                    need(not require_loaded or loaded, "Initialized native spawn helper lacks loaded identity")
                    need(
                        not loaded or entry["loaded_sha256"] == entry["sha256"],
                        "Loaded native spawn helper differs from retained bytes",
                    )
        need(runtime_blobs(source, commit) == self.source_blobs, "Setup extension changed runtime source inventory")
        return dict(identity)

    def check_safe_setup_profile(self, config, *, spawn=None):
        if spawn is None:
            spawn = self.setup_extension["policy"] == SPAWN_POLICY
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        runner = field(config, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task")
        sampler = field(config, "trainer.v1.sampler.custom_sampler")
        need(
            runner["runner_fqn"] == (SPAWN_RUNNER if spawn else SETUP_RUNNER)
            and type(runner["max_concurrent_sessions"]) is int
            and runner["max_concurrent_sessions"] == 32
            and sampler == {"path": str(source / SETUP_DRIVER), "name": "ResilientReplayBuffer"},
            "Actual safe-setup runner/concurrency/sampler profile differs",
        )

    def check_parallel_health(self, health, extension):
        need(
            isinstance(health, dict)
            and health.get("ready") is True
            and health.get("catalog_digest") == self.split["catalog_digest"]
            and health.get("environment_processes") == health.get("io_concurrency") == 4
            and health.get("environment_policy") == ENVIRONMENT_POLICY
            and health.get("action_request_protocol") == ACTION_PROTOCOL
            and health.get("budget_seconds") == 900
            and health.get("max_steps") == 50
            and health.get("environment_extension") == extension
            and isinstance(health.get("server_instance_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", health["server_instance_id"])
            and len(set(health.get("environment_worker_pids", []))) == 4,
            "Parallel environment capability/topology/budget/source identity differs",
        )

    def verify_direct_identity(self, value):
        identity = {key: value[key] for key in DIRECT_IDENTITY_FIELDS}
        need(identity["policy"] == DIRECT_POLICY and not self.base, "Unsupported direct-checkpoint execution policy")
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        commit = identity["driver_source_commit"]
        need(
            isinstance(commit, str)
            and re.fullmatch(r"[0-9a-f]{40,64}", commit)
            and identity["actual_entry"] == str(source / DIRECT_FILES["entry"])
            and identity["core_fingerprint_scope"]
            == "Unchanged legacy core; direct-restore implementation is identified separately",
            "Direct checkpoint entry/source scope differs",
        )
        exact_keys(identity["files"], DIRECT_FILES, "Direct checkpoint implementation sources")
        for name, relative in DIRECT_FILES.items():
            ref = identity["files"][name]
            exact_keys(ref, ("path", "sha256"), "Direct checkpoint " + name + " source")
            self.verify_published_source(relative, commit, ref["path"], ref["sha256"])
        need(runtime_blobs(source, commit) == self.source_blobs, "Direct checkpoint entry changed runtime sources")
        return identity

    def verify_all_native_identity(self, value, *, initialized):
        shape = all_native_shape(value)
        identity = {key: value[key] for key in shape["fields"]}
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        commit = identity["driver_source_commit"]
        need(
            not self.base
            and identity["policy"] == shape["policy"]
            and isinstance(commit, str)
            and re.fullmatch(r"[0-9a-f]{40,64}", commit)
            and identity["actual_entry"] == str(source / shape["files"]["entry"])
            and identity["core_fingerprint_scope"]
            == "Unchanged core; actual all-role spawn and direct-read providers recorded separately",
            "Unsupported AllNative entry/source scope",
        )
        exact_keys(identity["files"], shape["files"], "AllNative entry component sources")
        for name, relative in shape["files"].items():
            ref = identity["files"][name]
            exact_keys(ref, ("path", "sha256"), "AllNative " + name + " source")
            self.verify_published_source(relative, commit, ref["path"], ref["sha256"])
        if shape["generation"]:
            extension = identity["curate_generation_extension"]
            exact_keys(extension, ("policy", "files", "controller_class"), "Curate generation extension")
            need(
                extension["policy"] == CURATE_GENERATION_NAMESPACE_POLICY
                and extension["controller_class"] == CURATE_GENERATION_CONTROLLER,
                "Curate generation policy or controller class differs",
            )
            exact_keys(extension["files"], CURATE_GENERATION_PROVIDER_FILES, "Curate generation source providers")
            for name, relative in CURATE_GENERATION_PROVIDER_FILES.items():
                ref = extension["files"][name]
                exact_keys(ref, ("path", "sha256"), "Curate generation " + name + " source")
                self.verify_published_source(relative, commit, ref["path"], ref["sha256"])
        for name in ("checkpoint_core", "trainer_core"):
            relative = DIRECT_FILES[name]
            path = source / relative
            self.verify_published_source(relative, commit, str(path), self.reader.hash_file(path))
        component = self.verify_setup_identity(identity["solver_spawn_component_identity"], require_loaded=True)
        need(component["driver_source_commit"] == commit, "AllNative solver component source commit differs")
        native = identity["all_role_native_spawn"]
        exact_keys(
            native,
            (
                "policy",
                "files",
                "loaded_curator_helper_sha256",
                "solver_component",
                "loop_class",
                "parent_fd_fork_guard",
                "curator_profiles",
            ),
            "AllNative transport identity",
        )
        need(
            native["policy"] == ALL_NATIVE_TRANSPORT
            and native["loop_class"] == ALL_NATIVE_MODULE + ".AllNativeSelectorLoop"
            and native["parent_fd_fork_guard"]
            == "Python os.register_at_fork exclusion; parent stdio writers close before await"
            and native["solver_component"] == component["composed_extensions"]["native_spawn"],
            "AllNative loop/fork guard or unchanged solver helper identity differs",
        )
        solver_helper = native["solver_component"]["helper"]
        need(
            solver_helper["sha256"] == solver_helper["loaded_sha256"] == ALL_NATIVE_SOLVER_HELPER_SHA256,
            "AllNative primary solver helper differs from the unchanged reviewed four-descriptor helper",
        )
        exact_keys(native["files"], ALL_NATIVE_TRANSPORT_FILES, "AllNative transport sources")
        for name, relative in ALL_NATIVE_TRANSPORT_FILES.items():
            ref = native["files"][name]
            exact_keys(ref, ("path", "sha256"), "AllNative transport " + name)
            if name != "helper":
                self.verify_published_source(relative, commit, ref["path"], ref["sha256"])
            else:
                need(
                    ref["path"] == str(source / relative)
                    and ref["sha256"] == native["loaded_curator_helper_sha256"] == ALL_NATIVE_HELPER_SHA256
                    and self.reader.hash_file(ref["path"]) == ref["sha256"],
                    "AllNative curator helper bytes or loaded identity differ from the reviewed build",
                )
        profiles = native["curator_profiles"]
        need(
            isinstance(profiles, list) and len(profiles) == int(initialized),
            "AllNative registered curator profile count differs",
        )
        if initialized:
            profile = profiles[0]
            exact_keys(profile, ("user", "uid", "gid", "work_root", "package_root"), "Registered curator profile")
            work = Path(profile["work_root"])
            package = Path(profile["package_root"])
            need(
                profile["user"] == self.preflight["curator_runtime"]["unix_user"]
                and profile["user"] not in ("root", "alfworld_eval")
                and type(profile["uid"]) is int
                and profile["uid"] > 0
                and type(profile["gid"]) is int
                and profile["gid"] >= 0
                and work.is_absolute()
                and work == work.resolve()
                and work
                == Path(
                    optional_field(self.config, "skill_harness.joint.curator_work_root", "/tmp/alfworld-curator")
                ).resolve()
                and work != Path("/tmp").resolve()
                and work.is_relative_to(Path("/tmp").resolve())
                and package.is_absolute()
                and package == package.resolve()
                and package.is_dir()
                and package == Path(self.env["ALFWORLD_CODEX_PACKAGE_ROOT"]).resolve(),
                "Registered curator account/work-root/package profile differs from the retained runtime",
            )
        need(
            runtime_blobs(source, commit) == self.source_blobs, "AllNative entry changed core runtime source inventory"
        )
        return identity

    def verify_curate_generation_activation(self, initialized, identity, seal):
        """Bind preparation to the later controller activation and sealed state.

        Prepared-owner fields are retained metadata, not live PID attestation.
        The activation and published provider identify the controller that ran;
        this verifier neither loads models nor reconstructs behavior logprobs.
        """
        ref = initialized["curate_generation_ref"]
        exact_keys(ref, ("path", "sha256"), "Prepared curate generation reference")
        path = Path(ref["path"])
        directory = self.root / "controller/generation_epochs"
        need(
            path == path.resolve()
            and path.parent == directory
            and re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
            and self.reader.hash_file(path) == ref["sha256"],
            "Prepared curate generation path/hash differs",
        )
        prepared = self.reader.json(path)
        exact_keys(
            prepared,
            (
                "schema_version",
                "kind",
                "policy",
                "namespace",
                "created_at_ns",
                "owner",
                "source_checkpoint",
                "provider_ref",
                "scope",
            ),
            "Prepared curate generation namespace",
        )
        namespace = path.stem
        owner = prepared["owner"]
        exact_keys(owner, ("pid", "start_ticks"), "Prepared generation owner metadata")
        need(
            prepared["schema_version"] == 1
            and prepared["kind"] == "prepared_curate_generation_namespace"
            and prepared["policy"] == CURATE_GENERATION_NAMESPACE_POLICY
            and prepared["namespace"] == namespace
            and type(prepared["created_at_ns"]) is int
            and prepared["created_at_ns"] > 0
            and all(type(owner[key]) is int and owner[key] > 0 for key in ("pid", "start_ticks"))
            and prepared["provider_ref"] == identity["curate_generation_extension"]["files"]["provider"]
            and prepared["scope"] == CURATE_GENERATION_SCOPE,
            "Prepared curate generation namespace/provider/scope differs",
        )
        restored = initialized["restored_checkpoint"]
        source_clock = 0 if restored is None else restored["spec"]["optimizer_step"]
        expected_source = (
            None
            if restored is None
            else {
                "checkpoint_dir": restored["checkpoint_dir"],
                "manifest_id": restored["manifest_id"],
                "optimizer_step": source_clock,
            }
        )
        need(prepared["source_checkpoint"] == expected_source, "Curate generation source checkpoint differs")
        need(
            type(source_clock) is int and 0 <= source_clock < seal["optimizer_step"] == 64,
            "Curate generation entry requires a source optimizer step below the final target",
        )
        activated_path = directory / (namespace + ".activated.json")
        need(activated_path == activated_path.resolve(), "Curate generation activation path differs")
        activation_ref = {"path": str(activated_path), "sha256": self.reader.hash_file(activated_path)}
        activated = self.reader.json(activated_path)
        exact_keys(
            activated,
            ("schema_version", "kind", "namespace", "namespace_ref", "controller_class", "optimizer_step"),
            "Activated curate generation controller",
        )
        need(
            activated["schema_version"] == 1
            and activated["kind"] == "activated_curate_generation_controller"
            and activated["namespace"] == namespace
            and activated["namespace_ref"] == ref
            and activated["controller_class"] == CURATE_GENERATION_CONTROLLER
            and activated["optimizer_step"] == source_clock,
            "Curate generation activation namespace/controller/source clock differs",
        )
        checkpoint = Path(seal["checkpoint_dir"])
        for state_path in (self.root / "controller/state.json", checkpoint / "joint_state/controller.json"):
            state = self.reader.json(state_path)
            need(
                state.get("curate_generation_activation_ref") == activation_ref,
                "Final controller/checkpoint does not bind the activated curate namespace",
            )
            queue = state.get("source_queue", {}).get("jobs", {})
            need(isinstance(queue, dict), "Curate generation source queue is malformed")
            for entry in queue.values():
                jobs = entry.get("curator_jobs", [])
                current = entry.get("curate_generation_namespace") == namespace
                if jobs and entry["source_step"] > source_clock:
                    need(current, "Post-restore source jobs lack the active curate generation namespace")
                if current:
                    cohort = "cohort_" + INDEX.digest([entry["update_id"], entry["episode_output_dir"], namespace])
                    need(
                        all(job.get("candidate_group_id") == cohort for job in jobs),
                        "Curate source job cohort differs from its persisted execution namespace",
                    )
        return {
            "prepared_ref": ref,
            "activation_ref": activation_ref,
            "namespace": namespace,
            "controller_class": CURATE_GENERATION_CONTROLLER,
            "source_optimizer_step": source_clock,
        }

    def all_native_initialized(self, extension, ref, seal):
        shape = all_native_shape(extension)
        exact_keys(extension, shape["fields"], "AllNative lifecycle extension")
        identity = self.verify_all_native_identity(extension, initialized=True)
        exact_keys(ref, ("path", "sha256"), "AllNative initialized receipt reference")
        path = Path(ref["path"])
        need(
            path == path.resolve()
            and path.parent == self.root / "driver_runs"
            and re.fullmatch(r"[0-9]+\." + shape["stem"] + r"\.initialized\.json", path.name)
            and self.reader.hash_file(path) == ref["sha256"],
            "AllNative initialized receipt path/hash differs",
        )
        value = self.reader.json(path)
        exact_keys(
            value,
            (
                "schema_version",
                "kind",
                *shape["fields"],
                "imports",
                "effective_config_after_trainer_init",
                "original_training_manifest_sha256",
                "counterfactual_loop_class",
                "event_loop_policy_class",
                "registered_curator_scope",
                "primary_worker_scope",
                "restore_receipt_ref",
                "restored_checkpoint",
                "restore_policy",
                "core_version_binding",
                "component_receipt_scope",
                *(("curate_generation_ref",) if shape["generation"] else ()),
            ),
            "Actual AllNative initialized receipt",
        )
        need(
            value["schema_version"] == 1
            and value["kind"] == shape["initialized_kind"]
            and all(value[key] == extension[key] for key in shape["fields"])
            and value["counterfactual_loop_class"] == ALL_NATIVE_MODULE + ".AllNativeSelectorLoop"
            and value["event_loop_policy_class"] == ALL_NATIVE_MODULE + ".AllNativePolicy"
            and value["registered_curator_scope"]
            == "Actual NativeCuratorRuntime account, work root, package and clean environment"
            and value["primary_worker_scope"]
            == "Unchanged SpawnSafeRunner installs the solver-only component in episode workers"
            and value["component_receipt_scope"]
            == "Frozen final-fit body identity is a component; lifecycle extension uses this receipt"
            and (
                value["restore_policy"] == DIRECT_RESTORE_POLICY
                if value["restored_checkpoint"] is not None
                else value["restore_policy"] is None
                and value["restore_receipt_ref"] is None
                and value["core_version_binding"] is None
            )
            and value["original_training_manifest_sha256"] == seal["original_training_manifest_sha256"]
            and all(seal[key] == item for key, item in identity["solver_spawn_component_identity"].items()),
            "Actual AllNative learner loop/profile or frozen final-fit component binding differs",
        )
        if shape["generation"]:
            self.curate_generation_evidence = self.verify_curate_generation_activation(value, identity, seal)
        return path, value, identity

    def verify_all_native_observation(self, value, path):
        shape = all_native_shape(value)
        exact_keys(
            value,
            (
                "schema_version",
                "kind",
                *shape["fields"],
                "observed_at_ns",
                "imports",
                "parallel_component_ref",
                "client_adapter",
                "environment_health",
                "environment_config_ref",
                "config",
                "original_training_manifest_preserved",
                "not_a_post_init_or_served_model_receipt",
            ),
            "AllNative prelaunch component observation",
        )
        identity = self.verify_all_native_identity(value, initialized=False)
        need(
            path == path.resolve()
            and path.parent == self.root / "driver_runs"
            and re.fullmatch(r"[0-9]+\." + shape["stem"] + r"\.json", path.name)
            and value["kind"] == shape["prelaunch_kind"]
            and value["original_training_manifest_preserved"] is True
            and value["parallel_component_ref"] == identity["files"]["parallel_component"],
            "AllNative prelaunch entry/component binding differs",
        )
        for key in ("verl_source", "verl_version", "uni_agent_source", "uni_agent_adapter_sha256"):
            need(
                value["imports"].get(key) == self.manifest["import_provenance"].get(key),
                "AllNative prelaunch imports differ",
            )
        common = {
            key: value[key]
            for key in (
                "schema_version",
                "observed_at_ns",
                "client_adapter",
                "environment_health",
                "environment_config_ref",
                "not_a_post_init_or_served_model_receipt",
            )
        }
        component = {
            **common,
            "kind": "prelaunch_parallel_environment_client_observation",
            "spawn_extension": identity["solver_spawn_component_identity"],
            "driver_path": value["parallel_component_ref"]["path"],
            "driver_sha256": value["parallel_component_ref"]["sha256"],
            "effective_config_before_spawn_entry": value["config"],
            "core_fingerprint_scope": value["core_fingerprint_scope"],
            "hash_link_scope": "Explicit AllNative parallel inference component reference",
        }
        checked = self.verify_parallel_observation(component, path)
        checked.update(observation=value, all_native_entry=identity)
        return checked

    def verify_direct_observation(self, value, path):
        need(
            path == path.resolve()
            and path.parent == self.root / "driver_runs"
            and re.fullmatch(r"[0-9]+\.direct_checkpoint\.json", path.name),
            "Direct prelaunch observation escaped its explicit run binding",
        )
        exact_keys(
            value,
            (
                "schema_version",
                "kind",
                *DIRECT_IDENTITY_FIELDS,
                "observed_at_ns",
                "imports",
                "parallel_component_ref",
                "spawn_component_identity",
                "client_adapter",
                "environment_health",
                "environment_config_ref",
                "config",
                "original_training_manifest_preserved",
                "not_a_post_init_or_served_model_receipt",
            ),
            "Direct checkpoint prelaunch component observation",
        )
        identity = self.verify_direct_identity(value)
        need(
            value["kind"] == "prelaunch_direct_checkpoint_parallel_components"
            and value["original_training_manifest_preserved"] is True
            and value["parallel_component_ref"] == identity["files"]["parallel_component"]
            and value["spawn_component_identity"]["driver_source_commit"] == identity["driver_source_commit"],
            "Direct checkpoint prelaunch component bindings differ",
        )
        for key in ("verl_source", "verl_version", "uni_agent_source", "uni_agent_adapter_sha256"):
            need(
                value["imports"].get(key) == self.manifest["import_provenance"].get(key),
                "Direct checkpoint prelaunch imports differ",
            )
        # Reuse the exact existing component checks without claiming that the
        # old outer entry was executed. The retained observation stays direct.
        common = {
            key: value[key]
            for key in (
                "schema_version",
                "observed_at_ns",
                "client_adapter",
                "environment_health",
                "environment_config_ref",
                "not_a_post_init_or_served_model_receipt",
            )
        }
        component = {
            **common,
            "kind": "prelaunch_parallel_environment_client_observation",
            "spawn_extension": value["spawn_component_identity"],
            "driver_path": value["parallel_component_ref"]["path"],
            "driver_sha256": value["parallel_component_ref"]["sha256"],
            "effective_config_before_spawn_entry": value["config"],
            "core_fingerprint_scope": value["core_fingerprint_scope"],
            "hash_link_scope": "Explicit direct-entry parallel component reference",
        }
        checked = self.verify_parallel_observation(component, path)
        checked.update(observation=value, direct_entry=identity)
        return checked

    def verify_parallel_observation(self, value, path):
        if value.get("kind") in (
            "prelaunch_all_native_spawn_parallel_components",
            "prelaunch_curate_generation_parallel_components",
        ):
            return self.verify_all_native_observation(value, path)
        if value.get("kind") == "prelaunch_direct_checkpoint_parallel_components":
            return self.verify_direct_observation(value, path)
        prospective = value.get("kind") == "prospective_parallel_environment_runtime_identity"
        common = (
            "schema_version",
            "kind",
            "observed_at_ns",
            "client_adapter",
            "environment_health",
            "environment_config_ref",
            "spawn_extension",
            "not_a_post_init_or_served_model_receipt",
        )
        extra = (
            (
                "entrypoint",
                "source_commit",
                "client_source_ref",
                "original_entry_command",
                "effective_config_before_entry",
            )
            if prospective
            else (
                "driver_path",
                "driver_sha256",
                "effective_config_before_spawn_entry",
                "core_fingerprint_scope",
                "hash_link_scope",
            )
        )
        exact_keys(value, (*common, *extra), "Parallel environment/client observation")
        need(
            value["schema_version"] == 1
            and type(value["observed_at_ns"]) is int
            and value["observed_at_ns"] > 0
            and value["not_a_post_init_or_served_model_receipt"] is True,
            "Parallel observation cannot claim post-init/served-model evidence",
        )
        need(prospective is self.base, "Use a prospective original-entry base and observed trained client entry")
        spawn = self.verify_setup_identity(value["spawn_extension"])
        need(spawn["policy"] == SPAWN_POLICY, "Parallel observation lacks the public spawn boundary")
        commit = spawn["driver_source_commit"]
        if prospective:
            need(
                value["entrypoint"] == "verl.experimental.skill_harness.train"
                and value["source_commit"] == self.manifest["source_commit"] == commit,
                "Base prospective identity must bind its actual original-entry source",
            )
            ref = value["client_source_ref"]
            exact_keys(ref, ("path", "sha256"), "Prospective client source")
            command = value["original_entry_command"]
            need(
                isinstance(command, list) and command[1:3] == ["-m", value["entrypoint"]],
                "Prospective base did not use the supported original zero-epoch entry",
            )
        else:
            need(
                value["kind"] == "prelaunch_parallel_environment_client_observation",
                "Unknown trained parallel observer",
            )
            ref = {"path": value["driver_path"], "sha256": value["driver_sha256"]}
        self.verify_published_source(PARALLEL_DRIVER, commit, ref["path"], ref["sha256"])
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        adapter = value["client_adapter"]
        need(
            adapter.get("policy") == PARALLEL_POLICY
            and adapter.get("action_request_protocol") == ACTION_PROTOCOL
            and adapter.get("instruction_bytes_unchanged") is True
            and adapter.get("only_task_profile_change") == "agent.codex_bin"
            and adapter.get("tool_and_wrapper_same_inode") is True,
            "Unknown public client/Task-profile behavior",
        )
        for key in ("original_task_profile", "task_profile", "instructions", "wrapper", "environment_python_adapter"):
            item = adapter[key]
            exact_keys(item, ("path", "sha256"), "Parallel adapter " + key)
            need(self.reader.hash_file(item["path"]) == item["sha256"], "Parallel adapter file changed: " + key)
        original = self.root / "task_config.yaml"
        need(
            adapter["original_task_profile"]["path"] == str(original)
            and adapter["original_task_profile"]["sha256"] == self.reader.hash_file(original),
            "Parallel adapter does not preserve its original prepared Task profile",
        )
        task_path = self.root / "client_adapter" / ("task_profile_" + adapter["task_profile"]["sha256"] + ".json")
        need(adapter["task_profile"]["path"] == str(task_path), "Parallel Task profile escaped its content binding")
        task = json.loads(self.reader.bytes(task_path))
        original_task = json.loads(self.reader.bytes(original))
        need(
            isinstance(task, list) and len(task) == len(original_task) == 1,
            "Parallel Task profile must preserve the single original task",
        )
        original_task[0]["agent"]["codex_bin"] = adapter["wrapper"]["path"]
        need(task == original_task, "Parallel Task profile changed fields beyond its verified public wrapper")
        instructions = self.reader.bytes(adapter["instructions"]["path"])
        need(
            hashlib.sha256(instructions).hexdigest() == self.launch["solver_system_sha256"],
            "Parallel instructions differ from the prepared solver prompt",
        )
        client, shell = source / PARALLEL_CLIENT, source / "examples/skill_self_evolve/scripts/codex_entrypoint.sh"
        self.verify_published_source(PARALLEL_CLIENT, commit, str(client), adapter["client_source_sha256"])
        self.verify_published_source(
            shell.relative_to(source).as_posix(), commit, str(shell), adapter["native_shell_source_sha256"]
        )
        helper = load_source("_comparison_parallel_client_" + str(id(self)), source / PARALLEL_DRIVER)
        expected_wrapper = helper.public_wrapper(self.reader.bytes(client).decode(), self.reader.bytes(shell).decode())
        need(
            self.reader.bytes(adapter["wrapper"]["path"]) == expected_wrapper,
            "Public Native/client wrapper differs from the published producer",
        )
        route = adapter["environment_python_route"]
        need(
            route["environment_processes"] == route["io_workers"] == 4
            and route["translated_server"] == str(source / "verl/experimental/skill_harness/alfworld/server.py"),
            "Environment interpreter wrapper changed the exact core server route/topology",
        )
        self.verify_published_source(
            PARALLEL_SERVER, commit, route["parallel_server"]["path"], route["parallel_server"]["sha256"]
        )
        need(
            self.reader.hash_file(route["original_python"]["path"]) == route["original_python"]["sha256"],
            "Original ALFWorld interpreter bytes changed",
        )
        # Render with an isolated module-local os view, without changing the
        # comparison process environment or executing either public wrapper.
        helper.os = types.SimpleNamespace(environ={"VERL_SOURCE_ROOT": str(source)}, access=os.access, X_OK=os.X_OK)
        expected_environment, expected_route = helper.environment_python_wrapper(
            route["original_python"]["path"], route["parallel_server"]["path"]
        )
        need(
            expected_route == route
            and self.reader.bytes(adapter["environment_python_adapter"]["path"]) == expected_environment,
            "Environment wrapper bytes/venv route differ from the verified producer",
        )
        public = Path(adapter["wrapper"]["path"]).parent
        asset = hashlib.sha256(
            str(self.root).encode() + expected_wrapper + instructions + expected_environment
        ).hexdigest()[:24]
        need(
            public.name == "idempotent-" + asset
            and Path(adapter["tool_path"]) == public / "alfworld-step"
            and Path(adapter["tool_path"]).is_symlink()
            and os.readlink(adapter["tool_path"]) == "codex-entrypoint"
            and Path(adapter["tool_path"]).stat().st_ino == Path(adapter["wrapper"]["path"]).stat().st_ino
            and Path(adapter["environment_python_adapter"]["path"]) == public / "environment-python"
            and Path(adapter["instructions"]["path"]) == public / "solver_system.md",
            "Public tool does not share the verified allowed wrapper inode/asset binding",
        )
        changes = {
            "ALFWORLD_TASK_CONFIG": str(task_path),
            "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": adapter["instructions"]["path"],
            "ALFWORLD_ENV_PYTHON": adapter["environment_python_adapter"]["path"],
        }
        need(adapter["environment_variable_changes"] == changes, "Unknown parallel environment overrides")
        settings_ref = value["environment_config_ref"]
        exact_keys(settings_ref, ("path", "sha256"), "Observed environment config reference")
        need(
            settings_ref["path"] == str(self.root / "runtime/environment.json")
            and self.reader.hash_file(settings_ref["path"]) == settings_ref["sha256"],
            "Observed training environment configuration changed",
        )
        settings = self.reader.json(settings_ref["path"])
        extension = {
            "source_path": route["parallel_server"]["path"],
            "source_sha256": route["parallel_server"]["sha256"],
            "policy": ENVIRONMENT_POLICY,
        }
        self.check_parallel_health(value["environment_health"], extension)
        need(
            settings.get("environment_extension") == extension
            and settings.get("server_instance_id") == value["environment_health"]["server_instance_id"]
            and set(settings.get("allowed_splits", [])) == {"train"},
            "Parallel observation no longer binds its train-only service",
        )
        return {
            "spawn_extension": spawn,
            "client_adapter": adapter,
            "environment_extension": extension,
            "client_source_ref": ref,
            "source_commit": commit,
            "observation": value,
            "environment_config_sha256": settings_ref["sha256"],
        }

    def load_parallel_identity(self):
        expected_hash = self.protocol["budgets"]["alfworld_task_config_sha256"]
        candidates = (
            [self.root / PARALLEL_EXPECTED_RECEIPT]
            if self.base
            else sorted(
                [
                    *(self.root / "driver_runs").glob("*.parallel_client.json"),
                    *(self.root / "driver_runs").glob("*.direct_checkpoint.json"),
                    *(self.root / "driver_runs").glob("*.all_native_spawn.json"),
                    *(self.root / "driver_runs").glob("*.curate_generation.json"),
                ]
            )
        )
        observed = []
        for path in candidates:
            if path.is_file():
                value = self.reader.json(path)
                if value.get("client_adapter", {}).get("task_profile", {}).get("sha256") == expected_hash:
                    observed.append((path, value))
        direct = [(path, value) for path, value in observed if path.name.endswith(".direct_checkpoint.json")]
        all_native = [
            (path, value)
            for path, value in observed
            if path.name.endswith((".all_native_spawn.json", ".curate_generation.json"))
        ]
        # A direct continuation establishes its own outer-entry/service chain.
        # Historical parallel observers are not substitutes or dependencies.
        if direct or all_native:
            need(
                all(value.get("kind") == "prelaunch_direct_checkpoint_parallel_components" for _, value in direct)
                and all(value.get("kind") == all_native_shape(value)["prelaunch_kind"] for _, value in all_native),
                "Unknown direct checkpoint observation kind",
            )
            lifecycle = self.reader.json(self.path / "lifecycle.json")
            seal_ref = lifecycle["checkpoint_seal"]
            exact_keys(seal_ref, ("path", "sha256"), "Direct lifecycle seal reference")
            seal_path = Path(seal_ref["path"])
            need(
                seal_path == seal_path.resolve()
                and seal_path.name == "checkpoint_seal.json"
                and seal_path.parent.parent == self.root / "evaluation_lifecycle"
                and self.reader.hash_file(seal_path) == seal_ref["sha256"],
                "Direct lifecycle selection lacks its explicit seal binding",
            )
            seal_record = self.reader.json(seal_path)
            component_ref = seal_record["initialized_receipt_ref"]
            current_config_hash = self.reader.hash_file(self.root / "runtime/environment.json")
            if all_native:
                active = self.all_native_initialized(seal_record["lifecycle_extension"], component_ref, lifecycle)
                self.all_native_fresh = active[1]["restored_checkpoint"] is None
                observed = [
                    (path, value)
                    for path, value in all_native
                    if all_native_source_signature(value) == all_native_source_signature(active[2])
                    and value.get("environment_config_ref", {}).get("sha256") == current_config_hash
                ]
            else:
                active = self.direct_initialized_for(component_ref, require_observation=False)
                need(active is not None, "Direct observation lacks the lifecycle-linked initialized entry")
                active_identity = active[2]
                observed = [
                    (path, value)
                    for path, value in direct
                    if {key: value.get(key) for key in DIRECT_IDENTITY_FIELDS} == active_identity
                    and value.get("environment_config_ref", {}).get("sha256") == current_config_hash
                ]
            need(observed, "Direct startup lacks matching active source/service observations")
        matches = [(path, self.verify_parallel_observation(value, path)) for path, value in observed]
        if not matches:
            need(
                not (self.root / "client_adapter").exists(), "Parallel Task artifacts lack matching runtime observation"
            )
            return
        signatures = [parallel_signature(value) for _, value in matches]
        need(all(value == signatures[0] for value in signatures), "Conflicting matching parallel observations")
        self.parallel_extension = matches[0][1]
        self.direct_observations = [(path, value) for path, value in matches if "direct_entry" in value]
        self.all_native_observations = [(path, value) for path, value in matches if "all_native_entry" in value]
        self.parallel_evidence = [
            {
                "path": str(path),
                "sha256": self.reader.hash_file(path),
                "scope": "Prelaunch source/helper/client/service observation; not post-init or served-model proof",
            }
            for path, _ in matches
        ]
        adapter = self.parallel_extension["client_adapter"]
        self.env = {**self.env, **adapter["environment_variable_changes"]}
        for path in sorted(self.path.glob("service-*/environment.json")):
            settings = self.reader.json(path)
            instance = settings.get("server_instance_id")
            if isinstance(instance, str):
                need(instance not in self.parallel_test_services, "Duplicate official environment service identity")
                self.parallel_test_services[instance] = (path, settings)
        self.restore_bindings.update(
            {
                TASK_PROFILE_FIELD: adapter["task_profile"]["path"],
                "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_TASK_CONFIG": adapter["task_profile"]["path"],
            }
        )

    def apply_parallel_config(self, expected):
        adapter = self.parallel_extension["client_adapter"]
        set_existing_field(
            expected,
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path",
            adapter["task_profile"]["path"],
        )
        environment = expected["ray_kwargs"]["ray_init"]["runtime_env"].setdefault("env_vars", {})
        environment.update(adapter["environment_variable_changes"])

    def check_parallel_config(self, config):
        adapter = self.parallel_extension["client_adapter"]
        if self.base:
            need(
                field(config, "trainer.total_epochs") == 0 and field(config, "trainer.resume_mode") == "disable",
                "Prospective parallel base must use the fresh zero-epoch route",
            )
        need(
            field(
                config,
                "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path",
            )
            == adapter["task_profile"]["path"],
            "Actual runner does not bind the observed public client profile",
        )
        environment = field(config, "ray_kwargs.ray_init.runtime_env.env_vars")
        need(
            all(environment.get(key) == value for key, value in adapter["environment_variable_changes"].items()),
            "Actual worker environment does not bind the observed client/interpreter/instructions",
        )
        for key in (
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        ):
            need(optional_field(config, key) == 1, "Matched parallel profile must retain measured microbatch1: " + key)

    def check_parallel_case_service(self, registration):
        instance = registration["server_instance_id"]
        need(instance in self.parallel_test_services, "Official case lacks its actual request-ID test-service config")
        path, settings = self.parallel_test_services[instance]
        need(
            set(settings.get("allowed_splits", [])) == set(SPLITS.values())
            and settings.get("catalog_digest") == self.split["catalog_digest"]
            and settings.get("environment_processes")
            == settings.get("io_workers")
            == settings.get("io_concurrency")
            == 4
            and settings.get("environment_policy") == ENVIRONMENT_POLICY
            and settings.get("action_request_protocol") == ACTION_PROTOCOL
            and settings.get("environment_extension") == self.parallel_extension["environment_extension"]
            and settings.get("max_steps") == 50
            and settings.get("budget_seconds") == 900
            and len(set(settings.get("environment_worker_pids", []))) == 4,
            "Actual official service differs from the observed parallel/client profile",
        )
        pure_module("alfworld.protocol").validate_loopback_url(settings["url"])
        self.parallel_test_evidence[instance] = {
            "path": str(path),
            "sha256": self.reader.hash_file(path),
            "environment_extension": settings["environment_extension"],
            "environment_processes": 4,
            "io_concurrency": 4,
            "action_request_protocol": ACTION_PROTOCOL,
            "scope": "Retained test-only service configuration matched to each private budget registration",
        }

    def direct_initialized_for(self, component_ref, *, require_observation=True):
        candidates = []
        for path in sorted((self.root / "driver_runs").glob("*.direct_checkpoint.initialized.json")):
            value = self.reader.json(path)
            if value.get("spawn_component_initialized_ref") == component_ref:
                candidates.append((path, value))
        if not self.direct_observations and not candidates:
            return None
        need(
            len(candidates) == 1 and (self.direct_observations or not require_observation),
            "Direct entry lacks its matching initialized chain",
        )
        path, value = candidates[0]
        need(
            path == path.resolve() and re.fullmatch(r"[0-9]+\.direct_checkpoint\.initialized\.json", path.name),
            "Direct initialized receipt escaped its explicit run binding",
        )
        exact_keys(
            value,
            (
                "schema_version",
                "kind",
                *DIRECT_IDENTITY_FIELDS,
                "effective_config_after_trainer_init",
                "original_training_manifest_sha256",
                "restore_receipt_ref",
                "spawn_component_initialized_ref",
                "restored_checkpoint",
                "restore_policy",
                "core_version_binding",
            ),
            "Initialized direct checkpoint entry",
        )
        identity = self.verify_direct_identity(value)
        need(
            value["schema_version"] == 1
            and value["kind"] == "initialized_direct_checkpoint_entry"
            and value["restore_policy"] == DIRECT_RESTORE_POLICY
            and (
                not require_observation
                or any(observation["direct_entry"] == identity for _, observation in self.direct_observations)
            ),
            "Direct initialized entry differs from its actual prelaunch sources",
        )
        return path, value, identity

    def check_direct_restore(self, direct, initialized, seal, source_payload, expected_prelaunch, *, all_native=False):
        path, value, identity = direct
        need(
            value["original_training_manifest_sha256"] == seal["original_training_manifest_sha256"]
            and value["restored_checkpoint"] == initialized["restored_checkpoint"]
            and value["effective_config_after_trainer_init"] == initialized["effective_config_after_trainer_init"]
            and value["core_version_binding"] == source_payload["spec"]["versions"],
            "Direct and spawn initialized checkpoint/configuration proofs disagree",
        )
        observations = self.all_native_observations if all_native else self.direct_observations
        need(
            any(
                (
                    all_native_source_signature(observation["all_native_entry"])
                    == all_native_source_signature(identity)
                    if all_native
                    else observation["direct_entry"] == identity
                )
                and observation["observation"]["config"] == expected_prelaunch
                for _, observation in observations
            ),
            "Direct initialized profile does not follow its recorded prelaunch configuration",
        )
        ref = value["restore_receipt_ref"]
        exact_keys(ref, ("path", "sha256"), "Direct restore receipt reference")
        mutable = Path(seal["store_root"]).parent
        receipt_path = mutable / "DIRECT_READ_RESTORE.json"
        need(
            ref["path"] == str(receipt_path) and self.reader.hash_file(receipt_path) == ref["sha256"],
            "Direct restore receipt path/hash differs",
        )
        receipt = self.reader.json(receipt_path)
        exact_keys(
            receipt,
            (
                "schema_version",
                "policy",
                "manifest_id",
                "checkpoint_dir",
                "mutable_state_root",
                "online_source",
                "ema_source",
                "training_state_sources",
                "full_validation_passes",
                "ema_files_copied",
                "copied_components",
                "mutable_copy_bytes",
                "source_guard",
                "source_guard_sha256",
                "restore_seconds_after_validation",
            ),
            "Direct read restore evidence",
        )
        restored = initialized["restored_checkpoint"]
        checkpoint = Path(restored["checkpoint_dir"])
        need(
            receipt["schema_version"] == 1
            and receipt["policy"] == DIRECT_RESTORE_POLICY
            and receipt["manifest_id"] == restored["manifest_id"]
            and receipt["checkpoint_dir"] == str(checkpoint)
            and receipt["mutable_state_root"] == str(mutable)
            and receipt["online_source"] == str(checkpoint / "actor")
            and receipt["ema_source"] == str(checkpoint / "joint_state/ema")
            and receipt["training_state_sources"] == [str(checkpoint / "data.pt")]
            and type(receipt["full_validation_passes"]) is int
            and receipt["full_validation_passes"] == 1
            and type(receipt["ema_files_copied"]) is int
            and receipt["ema_files_copied"] == 0
            and receipt["copied_components"] == ["skill_store", "replay", "ledger"],
            "Direct restore source paths, validation count or copy policy differs",
        )
        copied = sum(
            item["size"]
            for relative, item in source_payload["files"].items()
            if relative in ("joint_state/controller.json", "joint_state/evaluation.json")
            or any(relative.startswith("joint_state/" + name + "/") for name in receipt["copied_components"])
        )
        need(
            type(receipt["mutable_copy_bytes"]) is int
            and receipt["mutable_copy_bytes"] == copied
            and receipt["source_guard"]
            == "manifest and file metadata/inventory before/after validation and before loaders"
            and isinstance(receipt["source_guard_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt["source_guard_sha256"])
            and INDEX.finite(receipt["restore_seconds_after_validation"])
            and receipt["restore_seconds_after_validation"] >= 0
            and self.reader.json(mutable / "RESTORE_COMPLETE.json") == {"manifest_id": restored["manifest_id"]}
            and not (mutable / "ema").exists(),
            "Direct restore completion, copy inventory or source guard evidence differs",
        )
        for name in receipt["copied_components"]:
            destination = mutable / name
            need(destination.is_dir() and not destination.is_symlink(), "Missing independent mutable component")
            for source in (checkpoint / "joint_state" / name).rglob("*"):
                relative = source.relative_to(checkpoint / "joint_state" / name)
                target = destination / relative
                if source.is_file() and target.exists():
                    need(not target.is_symlink() and not source.samefile(target), "Mutable state aliases checkpoint")
        self.direct_evidence = {
            "initialized_receipt": {"path": str(path), "sha256": self.reader.hash_file(path)},
            "restore_receipt_ref": ref,
            "source_checkpoint": restored,
            "entry_identity": identity,
            "restore_policy": receipt["policy"],
            "full_validation_passes": 1,
            "ema_files_copied": 0,
            "scope": "Published operational entry and linked restore records; unchanged inference component signature",
        }
        if not all_native:
            self.direct_evidence["spawn_component_initialized_ref"] = value["spawn_component_initialized_ref"]

    def check_initialized_setup(self, extension, ref, seal, final_spec):
        all_native = extension.get("policy") in (ALL_NATIVE_POLICY, CURATE_GENERATION_POLICY)
        if all_native:
            direct = self.all_native_initialized(extension, ref, seal)
            path, initialized, identity = direct
            extension = identity["solver_spawn_component_identity"]
            spawn = True  # Common primary-worker component, not the learner loop.
        else:
            spawn = extension.get("policy") == SPAWN_POLICY
            extension = self.verify_setup_identity(extension, require_loaded=spawn)
            exact_keys(ref, ("path", "sha256"), "Initialized setup receipt reference")
            path = Path(ref["path"])
            need(
                path.is_absolute()
                and path == path.resolve()
                and path.parent == self.root / "driver_runs"
                and re.fullmatch(
                    r"[0-9]+\." + ("spawn_safe" if spawn else "resilient") + r"\.initialized\.json", path.name
                ),
                "Initialized setup receipt escaped its explicit run directory",
            )
            need(self.reader.hash_file(path) == ref["sha256"], "Initialized setup receipt hash differs")
            initialized = self.reader.json(path)
            direct = self.direct_initialized_for(ref)
            spawn_fields = (
                ("counterfactual_loop_class", "event_loop_policy_class", "primary_runner_policy_install")
                if spawn
                else ()
            )
            exact_keys(
                initialized,
                (
                    *extension,
                    "imports",
                    "effective_config_after_trainer_init",
                    "original_training_manifest_sha256",
                    "restored_checkpoint",
                    *spawn_fields,
                ),
                "Initialized setup receipt",
            )
            need(
                all(initialized[key] == value for key, value in extension.items())
                and initialized["original_training_manifest_sha256"] == seal["original_training_manifest_sha256"],
                "Initialized setup source/original-manifest identity differs",
            )
            if spawn:
                need(
                    initialized["counterfactual_loop_class"] == SPAWN_MODULE + ".NativeSpawnSelectorLoop"
                    and initialized["event_loop_policy_class"] == SPAWN_MODULE + ".NativeSpawnPolicy"
                    and initialized["primary_runner_policy_install"]
                    == "synchronous runner constructor before SDK asyncio.run",
                    "Initialized native spawn policy/loop/primary-runner installation differs",
                )
                need(
                    all(seal[key] == value for key, value in extension.items()),
                    "Spawn lifecycle and initialized composite identities differ",
                )
        for key in ("verl_source", "verl_version", "uni_agent_source", "uni_agent_adapter_sha256"):
            need(
                initialized["imports"].get(key) == self.manifest["import_provenance"].get(key),
                "Initialized setup runtime/adapter import identity differs",
            )
        restored = initialized["restored_checkpoint"]
        fresh_all_native = all_native and restored is None
        if fresh_all_native:
            source_step, source_payload, checkpoint = 0, None, None
            effective = initialized["effective_config_after_trainer_init"]
            need(
                initialized["restore_receipt_ref"] is None
                and initialized["restore_policy"] is None
                and initialized["core_version_binding"] is None
                and optional_field(self.config, "trainer.resume_mode") in ("disable", "auto")
                and optional_field(effective, "trainer.resume_mode") == field(self.config, "trainer.resume_mode")
                and optional_field(self.config, "trainer.resume_from_path") is None
                and optional_field(effective, "trainer.resume_from_path") is None
                and seal["seal_origin"] == "final_checkpoint_saved_after_quiescence",
                "Fresh AllNative source0 contradicts its original configuration or restore fields",
            )
            fingerprint = _model_fingerprint(self.model_path)
            need(
                fingerprint["fingerprint"] == self.expected_fingerprint,
                "Fresh AllNative initial HF source differs from the unchanged base model",
            )
            for item in fingerprint["files"]:
                model_file = self.model_path / item["path"]
                info = model_file.stat()
                self.reader.observed[str(model_file)] = (info.st_size, info.st_mtime_ns)
        else:
            exact_keys(restored, ("checkpoint_dir", "manifest_id", "spec"), "Explicit source checkpoint")
            checkpoint = Path(restored["checkpoint_dir"])
            source_step = restored["spec"]["optimizer_step"] if direct else 4
            need(
                type(source_step) is int
                and 1 <= source_step <= 64
                and (not direct or restored["spec"]["global_step"] == source_step),
                "Unsupported direct source checkpoint clock/cursor",
            )
            need(
                checkpoint == Path(self.config["trainer"]["default_local_dir"]) / f"global_step_{source_step}"
                and checkpoint == checkpoint.resolve(),
                "Continuation did not bind its explicit retained source checkpoint",
            )
            source_payload, source_spec = self.verified_checkpoint(
                checkpoint, manifest_id=restored["manifest_id"], clock=source_step, cursor=source_step
            )
            need(
                source_spec.to_dict() == restored["spec"]
                and dict(source_spec.versions) == dict(final_spec.versions)
                and source_spec.protocol_fingerprint == final_spec.protocol_fingerprint,
                "Source checkpoint does not preserve final core/protocol compatibility",
            )
            if direct:
                need(
                    source_spec.compatibility() == final_spec.compatibility()
                    and (
                        source_step < 64
                        and seal["seal_origin"] == "final_checkpoint_saved_after_quiescence"
                        or source_step == 64
                        and seal["seal_origin"] == "restored_complete_final_checkpoint"
                        and restored["manifest_id"] == seal["manifest_id"]
                    ),
                    "Direct continuation source/final seal compatibility differs",
                )
        effective = initialized["effective_config_after_trainer_init"]
        expected = copy.deepcopy(self.config)
        if self.parallel_extension is not None:
            need(
                extension_signature(extension) == extension_signature(self.parallel_extension["spawn_extension"]),
                "Parallel observation and initialized loaded spawn sources differ",
            )
            self.apply_parallel_config(expected)
        runner = field(expected, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task")
        sampler = field(expected, "trainer.v1.sampler.custom_sampler")
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        need(
            runner["runner_fqn"]
            in ("verl.experimental.skill_harness.alfworld.runner.run_task", SETUP_RUNNER, SPAWN_RUNNER)
            and runner["max_concurrent_sessions"] in (32, 64)
            and sampler
            in (
                {
                    "path": str(source / "verl/experimental/skill_harness/group_sampler.py"),
                    "name": "ArchiveAwareReplayBuffer",
                },
                {"path": str(source / SETUP_DRIVER), "name": "ResilientReplayBuffer"},
            ),
            "Unknown original profile cannot be silently replaced by setup extension",
        )
        runner.update(runner_fqn=SPAWN_RUNNER if spawn else SETUP_RUNNER, max_concurrent_sessions=32)
        sampler.update(path=str(source / SETUP_DRIVER), name="ResilientReplayBuffer")
        store = Path(seal["store_root"])
        if fresh_all_native:
            need(
                store == self.root / "skill_store" == Path(self.config["skill_harness"]["store_root"]),
                "Fresh AllNative final seal does not bind the original prepared store",
            )
        else:
            need(
                re.fullmatch(rf"step_{source_step}_[0-9a-f]{{32}}", store.parent.name),
                "Restored store is not bound to its explicit source checkpoint",
            )
            for key, value in (
                ("store_root", store),
                ("replay_root", store.parent / "replay"),
                ("ledger_root", store.parent / "ledger"),
            ):
                set_existing_field(expected, "skill_harness." + key, str(value))
                self.restore_bindings["skill_harness." + key] = str(value)
        need(
            field(effective, "trainer.resume_mode")
            in (("auto", "disable") if fresh_all_native else ("auto", "resume_path")),
            "Unknown initialized resume mode",
        )
        set_existing_field(expected, "trainer.resume_mode", field(effective, "trainer.resume_mode"))
        resume_from = optional_field(effective, "trainer.resume_from_path")
        need(
            resume_from is None or not fresh_all_native and resume_from == str(checkpoint),
            "Initialized resume path differs from source",
        )
        if "resume_from_path" in expected["trainer"]:
            set_existing_field(expected, "trainer.resume_from_path", resume_from)
        if direct:
            original_save = optional_field(expected, "trainer.save_freq")
            initialized_save = optional_field(effective, "trainer.save_freq")
            if initialized_save != original_save:
                need(
                    type(original_save) is int
                    and original_save == 1
                    and type(initialized_save) is int
                    and initialized_save == 10,
                    "Unknown direct checkpoint save-frequency change",
                )
                set_existing_field(expected, "trainer.save_freq", initialized_save)
        expected_prelaunch = copy.deepcopy(expected)
        for name in ("store_root", "replay_root", "ledger_root"):
            expected_prelaunch["skill_harness"][name] = self.config["skill_harness"][name]
        if optional_field(expected, "data.gen_batch_size") in (None, 1):
            need(
                optional_field(effective, "data.gen_batch_size") == 1
                and optional_field(expected, "trainer.v1.sampler.sync_refill_failed_groups") is True,
                "Unknown exact-refill dataloader normalization",
            )
            set_existing_field(expected, "data.gen_batch_size", 1)
        horizon = expected["trainer"]["total_training_steps"] * optional_field(
            expected, "trainer.v1.sync.parameter_sync_step", 1
        )
        for prefix in ("actor_rollout_ref.actor.optim", "critic.optim"):
            key = prefix + ".total_training_steps"
            old = optional_field(expected, key)
            if old is not None:
                need(old in (-1, horizon), "Unknown optimizer schedule initialization")
                set_existing_field(expected, key, horizon)
        if optional_field(expected, "reward.reward_model.enable") is False:
            need(
                optional_field(expected, "reward.reward_model.enable_resource_pool") is False,
                "Separate reward-model topology cannot be normalized",
            )
            for name, value in (
                ("n_gpus_per_node", expected["trainer"]["n_gpus_per_node"]),
                ("nnodes", expected["trainer"]["nnodes"]),
            ):
                set_existing_field(expected, "reward.reward_model." + name, value)
        need(
            effective == expected,
            "Unknown initialized configuration change outside explicit setup/restore/constructor bindings",
        )
        self.check_safe_setup_profile(effective, spawn=spawn)
        self.setup_extension = extension
        self.inference_config = effective
        self.initialized_evidence = {
            "path": str(path),
            "sha256": ref["sha256"],
            "restored_checkpoint": restored,
            "effective_config_sha256": INDEX.digest(effective),
        }
        if direct:
            if fresh_all_native:
                need(
                    any(
                        all_native_source_signature(observation["all_native_entry"])
                        == all_native_source_signature(identity)
                        and observation["observation"]["config"] == expected_prelaunch
                        for _, observation in self.all_native_observations
                    ),
                    "Fresh AllNative initialized config differs from its actual prelaunch observation",
                )
            else:
                self.check_direct_restore(
                    direct, initialized, seal, source_payload, expected_prelaunch, all_native=all_native
                )
            cadence = {
                "original": optional_field(self.config, "trainer.save_freq"),
                "initialized": optional_field(effective, "trainer.save_freq"),
                "prelaunch": optional_field(expected_prelaunch, "trainer.save_freq"),
                "scope": "Verified operational cadence; excluded from the common inference signature",
            }
            if self.direct_evidence is not None:
                self.direct_evidence["checkpoint_save_frequency"] = cadence
            if all_native:
                self.initialized_evidence.update(
                    operational_policy=identity["policy"],
                    actual_learner_loop=initialized["counterfactual_loop_class"],
                )
                self.all_native_evidence = {
                    "initialized_receipt_ref": ref,
                    "entry_identity": identity,
                    "actual_learner_loop": initialized["counterfactual_loop_class"],
                    "actual_event_loop_policy": initialized["event_loop_policy_class"],
                    "initial_source_step": source_step,
                    "initial_hf_fingerprint": fingerprint["fingerprint"] if fresh_all_native else None,
                    "restore_receipt_ref": initialized["restore_receipt_ref"],
                    "checkpoint_save_frequency": cadence,
                    "final_fit_component": extension,
                    "scope": "Actual learner transport; unchanged primary-worker inference component signature",
                }
                if identity["policy"] == CURATE_GENERATION_POLICY:
                    self.all_native_evidence["curate_generation"] = self.curate_generation_evidence

    def check_lifecycle_binding(self):
        """Bind a resumed final stage to its explicit, successful sealed-store receipt."""
        self.lifecycle = None
        self.evaluation_store_root = Path(self.config["skill_harness"]["store_root"]).resolve()
        self.sealed_snapshot_id = None
        path = self.path / "lifecycle.json"
        if self.base:
            need(not path.exists(), "An initial base stage cannot use a final lifecycle receipt")
            return
        if not path.exists():
            need(
                not any(
                    (self.root / name).exists() for name in ("restored_state", "driver_runs", "evaluation_lifecycle")
                ),
                "Resumed/external final evaluation lacks its explicit lifecycle receipt",
            )
            return  # Original fresh runs keep their original configured store.
        receipt = self.reader.json(path)
        links = ("checkpoint_seal", "evaluation_entry", "evaluation_complete")
        spawn = receipt.get("policy") == SPAWN_POLICY
        identity_fields = LIFECYCLE_IDENTITY + (
            ("policy", "namespace_path", "namespace_sha256", "composed_extensions") if spawn else ()
        )
        exact_keys(
            receipt,
            ("schema_version", "kind", *identity_fields, *LIFECYCLE_SEAL, "evaluation_cursor", *links),
            "Final lifecycle receipt",
        )
        need(
            receipt["schema_version"] == 1
            and receipt["kind"] == "quiescent_final_evaluation"
            and type(receipt["optimizer_step"]) is int
            and receipt["optimizer_step"] == self.version
            and type(receipt["checkpoint_cursor"]) is int
            and receipt["checkpoint_cursor"] == 64
            and isinstance(receipt["snapshot_id"], str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt["snapshot_id"])
            and receipt["evaluation_cursor"] == self.protocol["global_step"]
            and receipt["seal_origin"]
            in ("final_checkpoint_saved_after_quiescence", "restored_complete_final_checkpoint"),
            "Final lifecycle stage/clock/seal differs",
        )
        need(
            receipt["original_training_manifest_sha256"] == self.reader.hash_file(self.root / "training-manifest.json"),
            "Lifecycle receipt does not bind the preserved original training manifest",
        )
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        driver = Path(receipt["driver_path"])
        relative = SPAWN_DRIVER if spawn else LIFECYCLE_DRIVER
        need(driver == source / relative, "Unexpected lifecycle driver source binding")
        commit = receipt["driver_source_commit"]
        need(
            isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit),
            "Missing lifecycle driver Git identity",
        )
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(source), "show", commit + ":" + relative],
                stderr=subprocess.PIPE,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise Error("Published lifecycle driver source is unavailable") from error
        need(
            hashlib.sha256(blob).hexdigest() == receipt["driver_sha256"] == self.reader.hash_file(driver)
            and runtime_blobs(source, commit) == self.source_blobs,
            "Lifecycle driver bytes or unchanged runtime source inventory differ",
        )
        records, directory = {}, None
        for key in links:
            ref = receipt[key]
            exact_keys(ref, ("path", "sha256"), "Lifecycle record reference")
            target = Path(ref["path"])
            need(
                target.is_absolute()
                and target == target.resolve()
                and target.name == key + ".json"
                and target.parent.parent == self.root / "evaluation_lifecycle"
                and re.fullmatch(r"driver_[0-9a-f]{32}", target.parent.name),
                "Lifecycle record escaped its explicit run/driver directory",
            )
            need(
                directory is None or directory == target.parent, "Lifecycle records belong to different driver sessions"
            )
            directory = target.parent
            need(self.reader.hash_file(target) == ref["sha256"], "Lifecycle record file hash differs")
            record = self.reader.json(target)
            need(
                record.get("schema_version") == 1
                and all(record.get(field) == receipt[field] for field in (*identity_fields, *LIFECYCLE_SEAL)),
                "Lifecycle identity/store/head/clock copies disagree",
            )
            records[key] = record
        need(
            records["evaluation_entry"].get("evaluation_cursor") == receipt["evaluation_cursor"]
            and records["evaluation_entry"].get("background_stopped_before_test_service") is True
            and records["evaluation_entry"].get("validation_factory_uses_config_copy") is True
            and records["evaluation_complete"].get("evaluation_cursor") == receipt["evaluation_cursor"]
            and records["evaluation_complete"].get("optimizer_updates_during_test") == 0,
            "Final evaluation lifecycle lacks its quiescent/config-copy/completion evidence",
        )
        store = Path(receipt["store_root"])
        if self.all_native_fresh:
            need(
                store == self.root / "skill_store" == Path(self.config["skill_harness"]["store_root"]),
                "Fresh AllNative sealed store escaped its original run binding",
            )
        else:
            need(
                store.is_absolute()
                and store == store.resolve()
                and store.name == "skill_store"
                and store.parent.parent == self.root / "restored_state"
                and re.fullmatch(r"step_[0-9]+_[0-9a-f]{32}", store.parent.name),
                "Sealed restored skill store is outside its explicit run restore binding",
            )
        checkpoint = Path(receipt["checkpoint_dir"])
        need(
            checkpoint == Path(self.config["trainer"]["default_local_dir"]) / "global_step_64"
            and checkpoint == checkpoint.resolve(),
            "Final COMPLETE checkpoint path differs from the original configured run",
        )
        payload, spec = self.verified_checkpoint(
            checkpoint, manifest_id=receipt["manifest_id"], clock=self.version, cursor=receipt["checkpoint_cursor"]
        )
        need(
            spec.optimizer_step == self.version
            and spec.global_step == receipt["checkpoint_cursor"]
            and spec.world_size == 4
            and dict(spec.versions) == receipt["checkpoint_core_versions"]
            and spec.versions["verl"] == "0.9.1"
            and spec.versions["model"] == self.expected_fingerprint
            and spec.versions["tokenizer"] == self.tokenizer
            and spec.protocol_fingerprint == self.controller["protocol_fingerprint"],
            "Final checkpoint sealed policy/core/protocol identity differs",
        )
        head = self.reader.json(store / "refs/main.json")
        snapshot_id = receipt["snapshot_id"]
        snapshot = self.reader.json(store / "snapshots" / (snapshot_id + ".json"))
        captured_snapshot = self.reader.json(checkpoint / "joint_state/skill_store/snapshots" / (snapshot_id + ".json"))
        need(
            head == {"branch": "main", "snapshot_id": snapshot_id}
            and INDEX.digest(snapshot) == snapshot_id
            and snapshot == captured_snapshot
            and payload["snapshots"]["skill_heads"]["main"] == snapshot_id,
            "Final restored store/head differs from its COMPLETE checkpoint seal",
        )
        self.evaluation_store_root, self.sealed_snapshot_id = store, snapshot_id
        extensions = [record.get("lifecycle_extension") for record in records.values()]
        initialized_refs = [record.get("initialized_receipt_ref") for record in records.values()]
        if any(value is not None for value in (*extensions, *initialized_refs)):
            need(
                all(value == extensions[0] for value in extensions)
                and extensions[0] is not None
                and all(value == initialized_refs[0] for value in initialized_refs)
                and initialized_refs[0] is not None,
                "Lifecycle setup extension/initialized receipt copies disagree",
            )
            self.check_initialized_setup(extensions[0], initialized_refs[0], receipt, spec)
        else:
            need(
                not (self.root / "setup_adapter").exists()
                and not spawn
                and not any((self.root / "driver_runs").glob("*.resilient.initialized.json"))
                and not any((self.root / "driver_runs").glob("*.spawn_safe.initialized.json")),
                "Safe-setup run lacks its lifecycle extension/initialized receipt",
            )
        self.lifecycle = {"receipt_path": str(path), "receipt_sha256": self.reader.hash_file(path), **receipt}

    def check_protocol(self):
        p, summary = self.protocol, self.summary
        exact_keys(
            p,
            (
                "schema_version",
                "stage",
                "global_step",
                "policy",
                "agent",
                "optimizer_updates",
                "curation",
                "dataset",
                "budgets",
            ),
            "Official protocol",
        )
        need(
            p["schema_version"] == 1 and p["stage"] == self.name and type(p["global_step"]) is int,
            "Wrong official stage/schema/cursor",
        )
        need(
            p["agent"] == "native_uni_agent_codex" and p["policy"] == "already_initialized_training_policy",
            "Unsupported native policy protocol",
        )
        need(p["optimizer_updates"] is False and p["curation"] is False, "Official evaluation is not isolated")
        exact_keys(
            summary,
            ("schema_version", "stage", "global_step", "completed", "metrics", "case_audits", "authoritative"),
            "Official summary",
        )
        need(
            summary["schema_version"] == 1
            and summary["stage"] == self.name
            and summary["completed"] is True
            and summary["global_step"] == p["global_step"],
            "Official stage is incomplete or mismatched",
        )
        exact_keys(p["dataset"], ("manifest_sha256", "catalog_digest", "catalog_sha256", "splits"), "Dataset protocol")
        exact_keys(p["dataset"]["splits"], COUNTS, "Official partition mapping")
        budgets = p["budgets"]
        exact_keys(
            budgets,
            (
                "sampling",
                "prompt_length",
                "response_length",
                "max_model_len",
                "alfworld_task_config_sha256",
                "alfworld_solver_instructions_file_sha256",
                "environment_max_steps",
            ),
            "Budget protocol",
        )
        need(
            budgets["sampling"] == GREEDY
            and [budgets[k] for k in ("prompt_length", "response_length", "max_model_len", "environment_max_steps")]
            == [4096, 32768, 36864, 50],
            "Changed sampling or token/world budget",
        )
        need(
            p["dataset"]["manifest_sha256"] == self.reader.hash_file(self.index.split_manifest_path)
            and p["dataset"]["catalog_sha256"] == self.reader.hash_file(self.index.catalog_path)
            and p["dataset"]["catalog_digest"] == self.split["catalog_digest"],
            "Underlying dataset protocol hashes differ",
        )
        need(self.split["run_dir"] == str(self.root) and self.split["seed"] == 42, "Split host identity/seed differs")
        exact_keys(
            self.split,
            (
                "schema_version",
                "benchmark_root",
                "run_dir",
                "catalog_digest",
                "seed",
                "dev_fraction",
                "development_grouping",
                "task_prompt_sha256",
                "available",
                "selected",
                "phase_game_ids",
                "parquet_files",
                "no_synthetic_tasks",
                "evaluation_games_used_for_training",
            ),
            "Split manifest",
        )
        need(self.split["no_synthetic_tasks"] is True, "Synthetic benchmark preparation is unsupported")
        self.split_signature = {
            key: self.split[key]
            for key in (
                "schema_version",
                "benchmark_root",
                "catalog_digest",
                "seed",
                "task_prompt_sha256",
                "no_synthetic_tasks",
                "evaluation_games_used_for_training",
            )
        }
        self.split_signature["official_ids"] = {phase: self.split["phase_game_ids"][phase] for phase in COUNTS}
        self.split_signature["official_counts"] = {
            key: {phase: self.split[key][phase] for phase in COUNTS} for key in ("available", "selected")
        }
        for phase in COUNTS:
            need(self.split["available"][phase] == self.split["selected"][phase], "Official preparation is truncated")
        phases = tuple(COUNTS)
        for phase in phases:
            path = self.root / "data" / (phase + ".parquet")
            prepared = self.split["parquet_files"][phase]
            exact_keys(prepared, ("path", "sha256"), "Prepared parquet file identity")
            actual_hash = self.reader.hash_file(path)
            need(
                Path(prepared["path"]).resolve() == path and prepared["sha256"] == actual_hash,
                "Prepared parquet path/hash differs",
            )
            count = COUNTS.get(phase, len(self.index.phase_ids[phase]))
            if phase in COUNTS:
                spec = p["dataset"]["splits"][phase]
                exact_keys(spec, ("path", "sha256", "games"), "Official parquet protocol")
                need(
                    Path(spec["path"]).resolve() == path and spec["games"] == count and spec["sha256"] == actual_hash,
                    "Official parquet identity/count/hash differs",
                )
            rows = self.read_parquet(path)
            mapped = {}
            for row in rows:
                data = copy.deepcopy(row)
                metadata = data["extra_info"]["tools_kwargs"]["task"]["metadata"]
                game = metadata["game_id"]
                need(game not in mapped and game in self.index.phase_ids[phase], "Unknown or duplicate parquet game")
                need(
                    metadata["run_dir"] == str(self.root)
                    and metadata["split"] == SPLITS.get(phase, "train")
                    and metadata["dataset_phase"] == phase
                    and metadata["catalog_digest"] == self.split["catalog_digest"],
                    "Parquet Task scope differs",
                )
                need(
                    all(
                        metadata.get(k) == self.games[game].get(k)
                        for k in ("relative_game_path", "game_sha256", "task_type", "scene_id")
                    ),
                    "Parquet/source game identity differs",
                )
                metadata["run_dir"] = "<explicit-run-root>"
                mapped[game] = data
            need(
                len(rows) == count and set(mapped) == self.index.phase_ids[phase],
                "Official parquet lacks exact full game set",
            )
            self.parquets[phase] = mapped

    def check_identity(self):
        reader, config = self.reader, self.config
        provenance = self.manifest["import_provenance"]
        need(provenance["verl_version"] == "0.9.1", "Wrong verl version")
        self.source_blobs = runtime_blobs(provenance["verl_source"], self.manifest["source_commit"])
        self.load_parallel_identity()
        need(
            re.fullmatch(r"[0-9a-f]{64}", self.manifest["dirty_diff_sha256"]) is not None,
            "Missing dirty source identity",
        )
        self.adapter = provenance["uni_agent_adapter_sha256"]
        need(re.fullmatch(r"[0-9a-f]{64}", self.adapter) is not None, "Missing adapter identity")
        model = Path(config["actor_rollout_ref"]["model"]["path"])
        self.model_path = model
        self.model_fingerprint = None
        if self.base:
            fingerprint = _model_fingerprint(model)
            need(fingerprint["fingerprint"] == self.expected_fingerprint, "Untouched HF model fingerprint differs")
            self.model_fingerprint = fingerprint["fingerprint"]
            for entry in fingerprint["files"]:
                path = model / entry["path"]
                info = path.stat()
                reader.observed[str(path)] = (info.st_size, info.st_mtime_ns)
        self.model_definition = reader.json(model / "config.json")
        self.model_definition.pop("_name_or_path", None)
        tokenizer_path = config["actor_rollout_ref"]["model"].get("tokenizer_path") or model
        self.tokenizer = tokenizer_version(reader, tokenizer_path)
        need(self.preflight["passed"] is True, "Native preflight did not pass")
        self.package_versions = self.preflight["package_versions"]
        need(
            isinstance(self.package_versions, dict)
            and all(
                k in self.package_versions
                for k in ("verl", "torch", "vllm", "ray", "transferqueue", "transformers", "uni-agent")
            ),
            "Missing recorded inference/runtime package identities",
        )
        native = self.preflight["curator_runtime"]
        need(native["native_version"] == "0.167.0", "Native Codex version differs")
        self.native_identity = {k: native[k] for k in ("native_version", "template_version", "files")}
        native_bin = Path(self.env["CODEX_NATIVE_BIN"])
        need(reader.hash_file(native_bin) == native["files"]["codex_bin"], "Native binary bytes differ from preflight")
        self.native_hash = reader.hash_file(native_bin)
        self.instructions_hash = reader.hash_file(self.env["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"])
        need(
            self.launch["solver_system_sha256"]
            == self.instructions_hash
            == self.protocol["budgets"]["alfworld_solver_instructions_file_sha256"],
            "Solver instruction bytes differ",
        )
        task_path = Path(self.env["ALFWORLD_TASK_CONFIG"])
        need(
            task_path.resolve()
            == (
                Path(self.parallel_extension["client_adapter"]["task_profile"]["path"])
                if self.parallel_extension is not None
                else self.root / "task_config.yaml"
            )
            and reader.hash_file(task_path) == self.protocol["budgets"]["alfworld_task_config_sha256"],
            "Task config is not the recorded run-local file",
        )
        self.task_config = json.loads(reader.bytes(task_path))
        need(isinstance(self.task_config, list) and len(self.task_config) == 1, "Unsupported Task config list")
        task = self.task_config[0]
        exact_keys(task, ("name", "sandbox", "agent"), "Task config")
        exact_keys(task["sandbox"], ("provider", "runtime_timeout"), "Sandbox config")
        exact_keys(task["agent"], ("name", "run_timeout", "codex_bin", "model_catalog_path"), "Native agent config")
        need(
            task["name"] == "alfworld_skill_training"
            and task["sandbox"] == {"provider": "local", "runtime_timeout": 900}
            and task["agent"]["name"] == "codex"
            and task["agent"]["run_timeout"] == 900,
            "Native agent or declared900-second budget differs",
        )
        need(
            Path(task["agent"]["model_catalog_path"]).resolve() == self.root / "model_catalog.json",
            "Unapproved catalog path binding",
        )
        self.wrapper_hash = reader.hash_file(task["agent"]["codex_bin"])
        self.model_catalog = reader.json(self.root / "model_catalog.json")
        need(
            reader.hash_file(self.root / "model_catalog.json") == native["files"]["model_catalog_path"],
            "Catalog bytes differ from preflight",
        )
        need(self.model_catalog.get("models"), "Missing Native model catalog")
        for row in self.model_catalog["models"]:
            need(
                row["slug"] == "Qwen3.5-4B-nothink"
                and row["context_window"] == 36864
                and row["input_modalities"] == ["text"],
                "Changed Native model capabilities",
            )
            need(
                row["base_instructions"].encode() == reader.bytes(self.env["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"]),
                "Catalog system prompt differs",
            )
        self.task_signature = copy.deepcopy(self.task_config)
        self.task_signature[0]["agent"].update(
            codex_bin="<validated-wrapper-bytes>", model_catalog_path="<run>/model_catalog.json"
        )
        rollout = config["actor_rollout_ref"]["rollout"]
        expected = {
            "seed": 42,
            "mode": "async",
            "prompt_length": 4096,
            "response_length": 32768,
            "max_model_len": 36864,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.55,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 32,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "calculate_log_probs": True,
        }
        need(all(rollout.get(k) == v for k, v in expected.items()), "Changed rollout seed/backend/capacity settings")
        need({k: rollout["val_kwargs"][k] for k in GREEDY} == GREEDY, "Changed greedy evaluation settings")
        need(
            rollout["engine_kwargs"]["vllm"]["gdn_prefill_backend"] == "triton"
            and rollout["engine_kwargs"]["vllm"]["async_scheduling"] is False,
            "Changed GDN/scheduler backend",
        )
        joint = config["skill_harness"]["joint"]
        need(
            [joint[k] for k in ("solver_gpus", "curator_gpus", "credit_gpus", "available_gpus")] == [4, 2, 2, 8]
            and config["trainer"]["n_gpus_per_node"] == 4,
            "Changed4+2+2 topology",
        )
        need(
            config["skill_harness"]["mode"] == "skill_joint"
            and config["skill_harness"]["loading"]
            == {"enabled": True, "branch": "main", "top_k": 6, "skill_token_budget": 1024},
            "Changed skill loading protocol",
        )
        need(
            config["trainer"]["val_before_train"] is False and config["trainer"]["val_only"] is False,
            "Unsupported baseline/final execution mode",
        )
        self.backend_evidence = backend_evidence(reader, self.root, model, self.package_versions["vllm"])
        controller = reader.json(self.root / "controller/state.json")
        need(
            controller["optimizer_step"] == self.version and controller["prepared_update"] is None,
            "Evaluation does not match intended applied policy version",
        )
        if self.base:
            need(
                config["trainer"].get("total_training_steps") == 0 or config["trainer"].get("total_epochs") == 0,
                "Base evaluation must select zero optimizer iterations",
            )
            need(
                all(controller[k] == {} for k in ("collections", "pending", "applied_update_ids")),
                "Base run contains learned controller history",
            )
            store = Path(config["skill_harness"]["store_root"])
            head = reader.json(store / "refs/main.json")
            snapshot = reader.json(store / "snapshots" / (head["snapshot_id"] + ".json"))
            need(
                head["branch"] == "main"
                and INDEX.digest(snapshot) == head["snapshot_id"]
                and snapshot
                == {"schema_version": 1, "parent_id": None, "skills": [], "provenance": {"operation": "genesis"}},
                "Untouched base does not retain an empty genesis main library",
            )
        self.controller = controller

    def config_signature(self, value, path=""):
        if path in TRAINING_ONLY_SECTIONS or path in TRAINING_ONLY_FIELDS:
            return "<unused-training-configuration>"
        if path in RUN_PATHS:
            expected = self.restore_bindings.get(path, str(self.root / RUN_PATHS[path]))
            need(value == expected, "Unknown host path change: " + path)
            return "<run>/" + RUN_PATHS[path]
        if (
            path in ("reward.reward_model.n_gpus_per_node", "reward.reward_model.nnodes")
            and optional_field(self.inference_config, "reward.reward_model.enable") is False
        ):
            need(
                optional_field(self.inference_config, "reward.reward_model.enable_resource_pool") is False
                and type(value) is int
                and value >= 0,
                "Unknown disabled reward-model placement",
            )
            return "<unused-disabled-reward-model-placement>"
        if path in ("trainer.experiment_name", "actor_rollout_ref.rollout.trace.experiment_name"):
            need(value == self.root.name, "Unknown run-name binding")
            return "<run-name>"
        if path == "ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR":
            need(
                value == "/dev/shm/skill-tmp-" + hashlib.sha256(str(self.root).encode()).hexdigest()[:8],
                "Unknown private temporary path binding",
            )
            return "<derived-host-temporary-path>"
        if path == "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_SOLVER_INSTRUCTIONS_FILE":
            need(value == self.env["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"], "Runtime instruction path differs")
            return "<validated-solver-instruction-bytes>"
        if (
            path == "ray_kwargs.ray_init.runtime_env.env_vars.ALFWORLD_ENV_PYTHON"
            and self.parallel_extension is not None
        ):
            need(value == self.env["ALFWORLD_ENV_PYTHON"], "Parallel environment interpreter path differs")
            return "<validated-parallel-environment-interpreter-wrapper>"
        source = self.manifest["import_provenance"]["verl_source"]
        if path in (
            "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL",
            "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL",
        ):
            need(value in ("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"), "Unknown logging level metadata")
            return "<logging-only-metadata>"
        if path == "trainer.v1.sampler.custom_sampler.path":
            if self.setup_extension is not None:
                need(value == str(Path(source) / SETUP_DRIVER), "Unsupported safe-setup sampler source binding")
                return "<validated-source>/" + SETUP_DRIVER
            need(
                value == str(Path(source) / "verl/experimental/skill_harness/group_sampler.py"),
                "Unsupported sampler source binding",
            )
            return "<validated-source>/verl/experimental/skill_harness/group_sampler.py"
        if path == "ray_kwargs.ray_init.runtime_env.env_vars.VERL_SOURCE_ROOT":
            need(value == source, "Source root binding differs")
            return "<validated-runtime-source-tree>"
        if path == "ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH":
            need(isinstance(value, str) and value.split(":", 1)[0] == source, "PYTHONPATH source binding differs")
            return ["<validated-runtime-source-tree>", *value.split(":")[1:]]
        if path == "trainer.total_epochs":
            need(
                type(value) is int and value >= 0,
                "Invalid training epoch metadata",
            )
            return "<explicit-evaluation-execution-mode>"
        if path == "trainer.total_training_steps":
            need(type(value) is int and value >= 0, "Invalid training horizon metadata")
            return "<explicit-training-horizon-not-an-inference-budget>"
        if path in ("actor_rollout_ref.model.path", "ray_kwargs.ray_init.runtime_env.env_vars.MODEL_PATH"):
            need(Path(value).resolve() == self.model_path.resolve(), "Model source path binding differs")
            return "<intended-base-or-trained-model-source>"
        if path in (
            "actor_rollout_ref.model.local_path",
            "actor_rollout_ref.model.hf_config_path",
            "actor_rollout_ref.model.local_hf_config_path",
        ):
            need(value is None or Path(value).resolve() == self.model_path.resolve(), "Unknown model metadata path")
            return "<intended-model-metadata-source>"
        if path == "trainer.resume_mode":
            need(
                value == "disable"
                if self.base
                else value in ("auto", "disable") or value == "resume_path" and self.initialized_evidence is not None,
                "Unsupported resume mode",
            )
            return "<explicit-zero-base-or-trained-run>"
        if path == "trainer.resume_from_path":
            need(
                value is None
                or self.initialized_evidence is not None
                and value == self.initialized_evidence["restored_checkpoint"]["checkpoint_dir"],
                "Unknown resume checkpoint binding",
            )
            return "<explicit-zero-base-or-retained-source-checkpoint>"
        if path in ("skill_harness.evaluation.initial", "skill_harness.evaluation.final"):
            need(
                value is (self.base if path.endswith("initial") else not self.base), "Stage evaluation switches differ"
            )
            return "<explicit-base-initial-or-trained-final>"
        if isinstance(value, dict):
            return {k: self.config_signature(v, path + "." + k if path else k) for k, v in value.items()}
        if isinstance(value, list):
            return [self.config_signature(v, path + "." + str(i)) for i, v in enumerate(value)]
        return value

    def collect(self):
        reader = self.reader
        trace_dirs = {}
        for log in (self.root / "agent_logs" / ("step_" + str(self.protocol["global_step"]))).glob("*/task.log"):
            text = reader.bytes(log).decode(errors="replace")
            match = re.search(r"codex: launch in (\S+)", text)
            if match:
                trace_dirs.setdefault(match.group(1), []).append(log.parent)
        for phase, count in COUNTS.items():
            joined, _, issues, sidecar = self.index._case_audit_rows(
                self.name, phase, self.path / phase, self.protocol, self.summary
            )
            need(not issues and sidecar is not None, "Incomplete official case audit: " + "; ".join(issues))
            report = reader.json(sidecar)
            need(
                report["observed_weight_versions"] == [self.version], "Missing or wrong observed policy weight version"
            )
            metrics = reader.json(self.path / phase / "metrics.json")
            need(
                metrics["case_audit_path"] == sidecar
                and metrics["metrics"] == self.summary["metrics"][phase] == report["parent_metrics"],
                "Forged or conflicting official metric copies",
            )
            authoritative = {
                key: report[key]
                for key in (
                    "expected_cases",
                    "authoritative_wins",
                    "authoritative_reward_mean",
                    "authoritative_accuracy",
                    "native_error_cases",
                    "observed_weight_versions",
                )
            }
            need(
                metrics["authoritative"] == self.summary["authoritative"][phase] == authoritative,
                "Forged official authoritative aggregates",
            )
            cases = {}
            for _, row in joined:
                scope = row["_case_audit"]
                case = scope["case"]
                game, output = case["game_id"], Path(case["episode_output_dir"]).resolve()
                need(
                    row["score"] == row["acc"] == row["reward"] == float(case["engine_won"]),
                    "Forged official prefix score/accuracy",
                )
                if game in cases:
                    continue  # All prefix scores/UIDs were already checked by the scoped sidecar join.
                need(output.is_relative_to(self.root / "episodes"), "Raw episode escaped its run")
                item = INDEX.episode(reader, output)
                need(item["valid"], "Invalid raw official outcome: " + str(item.get("reason")))
                task, engine = item["result"], reader.json(output / "engine-result.json")
                need(
                    task["attempt_id"] == case["attempt_id"]
                    and task["game_id"] == game
                    and task["game_sha256"] == self.games[game]["game_sha256"],
                    "Raw/source game identity differs",
                )
                expected = {
                    "rollout_uid": case["rollout_uid"],
                    "session_index": 0,
                    "evaluation_audit_id": scope["audit_id"],
                    "evaluation_stage": self.name,
                    "global_step": self.protocol["global_step"],
                    "dataset_phase": phase,
                    "split": SPLITS[phase],
                }
                attempt = reader.json(output / "attempt.json")
                need(
                    all(
                        task.get(k) == engine.get(k) == attempt.get(k) == self.games[game].get(k)
                        for k in ("game_id", "game_sha256", "relative_game_path", "split", "task_type", "scene_id")
                    ),
                    "Raw/source game record fields differ",
                )
                need(
                    all(task.get(k) == v and attempt.get(k) == v for k, v in expected.items()),
                    "Raw Task/attempt evaluation scope differs",
                )
                need(
                    type(engine["won"]) is bool
                    and case["engine_won"] == engine["won"]
                    and row["score"] == row["acc"] == row["reward"] == float(engine["won"]),
                    "Forged case/dump reward or accuracy",
                )
                initial = reader.json(output / "initial-state.json")
                need(
                    engine.get("max_steps") == task.get("max_steps") == initial.get("max_steps") == 50
                    and type(initial.get("steps")) is int
                    and initial["steps"] == 0,
                    "Actual world action budget/initial state differs",
                )
                metadata = self.parquets[phase][game]["extra_info"]["tools_kwargs"]["task"]["metadata"]
                source_root = (Path(self.split["benchmark_root"]) / "data").resolve()
                relative = Path(metadata["relative_game_path"])
                need(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source game path")
                source_game = (source_root / relative).resolve()
                need(source_game.is_relative_to(source_root), "Source game escaped the benchmark")
                need(reader.hash_file(source_game) == metadata["game_sha256"], "Underlying benchmark game bytes differ")
                budget = pure_module("alfworld.budget")
                registration = reader.json(output / "budget-registration.json")
                if self.parallel_extension is not None:
                    self.check_parallel_case_service(registration)
                need(registration["account"]["owner"]["run_root"] == str(self.root), "Budget registered to another run")
                budget.validate_budget_registration(
                    registration,
                    payload={
                        "schema_version": 1,
                        "budget_seconds": 900,
                        "binding": registration["binding"],
                        "account": registration["account"],
                    },
                    session_id=task["environment_session_id"],
                    run_root=self.root,
                    output=output,
                )
                need(
                    task["backend_stop_confirmed"] is True
                    and not task.get("backend_terminal_error")
                    and not task.get("account_isolation_error"),
                    "Backend/UID outcome is not confirmed",
                )
                archive = reader.json(output / "native-files.json")
                for name in ("archive_path", "source_manifest_path"):
                    path = Path(archive[name])
                    info = path.stat()
                    reader.observed[str(path)] = (info.st_size, info.st_mtime_ns)
                budget._verify_archive(output, registration, task)
                context = reader.json(output / "skill-context.json")
                need(
                    context == task["skill_context"]
                    and context["enabled"] is True
                    and context["tokenizer_version"] == self.tokenizer,
                    "Skill context/tokenizer evidence differs",
                )
                need(
                    context["context_id"] == INDEX.digest({k: v for k, v in context.items() if k != "context_id"}),
                    "Skill context content hash differs",
                )
                original = pure_module("alfworld.protocol").agent_messages(
                    self.parquets[phase][game]["prompt"], initial, instruction=metadata.get("instruction")
                )
                need(
                    context["original_messages_sha256"] == INDEX.digest(original),
                    "Task prompt differs from original parquet/world",
                )
                store = self.evaluation_store_root
                need(
                    self.sealed_snapshot_id is None or context["snapshot_id"] == self.sealed_snapshot_id,
                    "Final Task context did not use the single receipt-sealed skill snapshot",
                )
                snapshot = reader.json(store / "snapshots" / (context["snapshot_id"] + ".json"))
                need(INDEX.digest(snapshot) == context["snapshot_id"], "Pinned skill snapshot hash differs")
                artifacts = []
                for ref in context["injected_refs"]:
                    need(ref in snapshot["skills"], "Injected skill absent from pinned snapshot")
                    artifact = reader.json(store / "artifacts" / (ref["content_hash"] + ".json"))
                    value = pure_module("store").SkillArtifact.from_dict(artifact)
                    need(value.ref.to_dict() == ref, "Injected skill version differs")
                    artifacts.append(value)
                if self.base:
                    need(
                        snapshot["parent_id"] is None
                        and snapshot["skills"] == []
                        and snapshot["provenance"] == {"operation": "genesis"}
                        and context["injected_refs"] == [],
                        "Base reference contains learned skills",
                    )
                messages = json.loads(reader.bytes(output / "agent-messages.json"))
                expected_messages = copy.deepcopy(original)
                block, _ = pure_module("context")._skill_block(artifacts)
                expected_messages[-1]["content"] += block
                need(
                    messages == expected_messages and context["injected_messages_sha256"] == INDEX.digest(messages),
                    "Unexpected prompt change beyond intended skill text",
                )
                need(
                    reader.bytes(output / "agent-prompt.txt").decode()
                    == "\n\n".join(m["content"] for m in messages) + "\n",
                    "Saved native Task prompt differs",
                )
                traces = trace_dirs.get(task.get("workdir"), [])
                need(
                    len(traces) == 1 and INDEX.gateway_trace(reader, traces[0]) > 0,
                    "Missing or ambiguous real numeric Gateway trajectory",
                )
                cases[game] = {
                    "won": engine["won"],
                    "initial_state": initial,
                    "game_sha256": task["game_sha256"],
                    "episode": str(output),
                    "snapshot_id": context["snapshot_id"],
                }
            need(
                len(cases) == count and set(cases) == self.index.phase_ids[phase],
                "Official cases lack full unique denominator",
            )
            wins = sum(value["won"] for value in cases.values())
            need(authoritative["authoritative_wins"] == wins, "Authoritative wins differ from original engine results")
            for key in (
                "val-core/alfworld/acc/mean@1",
                "val-aux/alfworld/reward/mean@1",
                "val-aux/alfworld/score/mean@1",
            ):
                if key in metrics["metrics"]:
                    need(
                        math.isclose(metrics["metrics"][key], wins / count, rel_tol=0, abs_tol=1e-12),
                        "Forged aggregate metric: " + key,
                    )
            self.cases[phase] = cases


def extension_signature(identity):
    """Compare source/helper bytes while retaining each run's full path/commit evidence."""
    result = {key: identity[key] for key in ("policy", "driver_sha256", "namespace_sha256")}
    if identity["policy"] == SPAWN_POLICY:
        composed = identity["composed_extensions"]
        native = composed["native_spawn"]
        result["composed_extensions"] = {
            "resilient_setup": extension_signature(composed["resilient_setup"]),
            "quiescent_final": {"sha256": composed["quiescent_final"]["sha256"]},
            "native_spawn": {
                "policy": native["policy"],
                **{key: {"sha256": native[key]["sha256"]} for key in SPAWN_FILES},
            },
        }
    return result


def all_native_source_signature(value):
    identity = {key: copy.deepcopy(value.get(key)) for key in all_native_shape(value)["fields"]}
    native = identity["all_role_native_spawn"]
    if isinstance(native, dict):
        native.pop("curator_profiles", None)  # Prelaunch precedes actual learner profile registration.
    return identity


def parallel_signature(value):
    adapter = value["client_adapter"]
    route = adapter["environment_python_route"]
    return {
        "policy": adapter["policy"],
        "action_request_protocol": adapter["action_request_protocol"],
        "spawn_extension": extension_signature(value["spawn_extension"]),
        "client_driver_sha256": value["client_source_ref"]["sha256"],
        "client_source_sha256": adapter["client_source_sha256"],
        "native_shell_source_sha256": adapter["native_shell_source_sha256"],
        **{name: adapter[name]["sha256"] for name in ("wrapper", "instructions", "environment_python_adapter")},
        "original_python_sha256": route["original_python"]["sha256"],
        "parallel_server_sha256": route["parallel_server"]["sha256"],
        "environment_processes": route["environment_processes"],
        "io_workers": route["io_workers"],
    }


def compare_stages(base_stage, trained_stage, *, expected_base_fingerprint=BASE_FINGERPRINT, parquet_reader=None):
    reader = Reader()
    report = {
        "schema_version": 1,
        "kind": "deferred_official_cross_run_comparison",
        "generated_at": time.time(),
        "comparison_supported": False,
        "improvement_supported": False,
        "reasons": [],
        "comparisons": {},
        "pooled": {},
        "base_stage": str(Path(base_stage).resolve()),
        "trained_stage": str(Path(trained_stage).resolve()),
        "raw_artifacts_rewritten": False,
        "baseline_backdated_or_transplanted": False,
    }
    try:
        base = Stage(
            base_stage,
            base=True,
            reader=reader,
            read_parquet=parquet_reader or parquet_rows,
            expected_fingerprint=expected_base_fingerprint,
        )
        trained = Stage(
            trained_stage,
            base=False,
            reader=reader,
            read_parquet=parquet_reader or parquet_rows,
            expected_fingerprint=expected_base_fingerprint,
        )
        need(base.root != trained.root, "Use two distinct original run roots")
        need(
            base.signature == trained.signature,
            "Unknown resolved configuration difference beyond explicit host paths/execution modes",
        )
        need(
            (base.setup_extension is None) == (trained.setup_extension is None),
            "Base and final safe-setup execution policies differ",
        )
        if base.setup_extension is not None:
            need(
                extension_signature(base.setup_extension) == extension_signature(trained.setup_extension),
                "Base and final execution driver/namespace/composed-source/helper identities differ",
            )
        need(
            (base.parallel_extension is None) == (trained.parallel_extension is None),
            "Base and final parallel environment/client policies differ",
        )
        if base.parallel_extension is not None:
            need(
                parallel_signature(base.parallel_extension) == parallel_signature(trained.parallel_extension),
                "Base and final environment/client/interpreter/source bytes differ",
            )
        need(
            base.source_blobs == trained.source_blobs
            and base.manifest["dirty_diff_sha256"] == trained.manifest["dirty_diff_sha256"]
            and base.adapter == trained.adapter,
            "Runtime source/adapter/dirty-state identities differ",
        )
        need(
            base.catalog == trained.catalog
            and base.parquets == trained.parquets
            and base.split_signature == trained.split_signature,
            "Source catalog or original parquet prompt/Task fields differ beyond run_dir",
        )
        need(
            all(list(base.parquets[phase]) == list(trained.parquets[phase]) for phase in base.parquets),
            "Original official parquet row order differs",
        )
        need(
            base.task_signature == trained.task_signature
            and base.wrapper_hash == trained.wrapper_hash
            and base.native_identity == trained.native_identity
            and base.model_catalog == trained.model_catalog
            and base.instructions_hash == trained.instructions_hash
            and base.tokenizer == trained.tokenizer
            and base.model_definition == trained.model_definition
            and base.package_versions == trained.package_versions
            and base.launch["solver_task_sha256"] == trained.launch["solver_task_sha256"],
            "Native tool/prompt/tokenizer identity differs",
        )
        base.collect()
        trained.collect()
        for phase, count in COUNTS.items():
            for game in base.cases[phase]:
                a, b = base.cases[phase][game], trained.cases[phase][game]
                need(
                    a["game_sha256"] == b["game_sha256"] and a["initial_state"] == b["initial_state"],
                    "Source game/world identity changed: " + game,
                )
            before, after = (
                sum(v["won"] for v in base.cases[phase].values()),
                sum(v["won"] for v in trained.cases[phase].values()),
            )
            report["comparisons"][phase] = {
                "games": count,
                "base_wins": before,
                "trained_wins": after,
                "base_reward": before / count,
                "trained_reward": after / count,
                "base_accuracy": before / count,
                "trained_accuracy": after / count,
                "reward_delta": (after - before) / count,
                "accuracy_delta": (after - before) / count,
            }
        before = sum(v["base_wins"] for v in report["comparisons"].values())
        after = sum(v["trained_wins"] for v in report["comparisons"].values())
        report["pooled"] = {
            "games": 274,
            "base_wins": before,
            "trained_wins": after,
            "base_reward": before / 274,
            "trained_reward": after / 274,
            "base_accuracy": before / 274,
            "trained_accuracy": after / 274,
            "reward_delta": (after - before) / 274,
            "accuracy_delta": (after - before) / 274,
        }
        report["comparison_supported"], report["improvement_supported"] = True, after > before
        report["explicit_allowed_differences"] = [
            "Original distinct run/stage/cursor/episode/UID/audit identities",
            "Only RUN_PATHS and validated native wrapper/instruction/catalog/source-path bindings",
            "Zero-optimizer base/initial-hook versus trained policy/final-hook; unused training budgets may differ",
            "Source commits may differ only if their complete enumerated runtime Git blob maps and dirty hashes match",
            "Applied model version0 versus64; empty genesis versus integrity-checked learned skill snapshots",
        ]
        if trained.initialized_evidence and not trained.all_native_fresh:
            report["explicit_allowed_differences"].append(
                "Resumed original manifest is preserved; inference uses only its verified initialized32/safe-runner "
                "profile and explicit source-checkpoint/store bindings, matched to the base's original actual profile."
            )
        if trained.direct_evidence:
            report["explicit_allowed_differences"].append(
                "The trained direct-checkpoint entry changes only verified restore operations; its published "
                "outer/provider sources and source checkpoint are retained separately from the common "
                "parallel/spawn inference signature. The zero-epoch base does not execute restore operations."
            )
        if trained.all_native_evidence:
            report["explicit_allowed_differences"].append(
                "The actual AllNative learner loop and reviewed curator helper/profile are verified as operational "
                "evidence. The untouched zero-epoch base retains the identical primary SpawnSafeRunner/solver/client "
                "inference profile and does not need to load or execute the unused curator helper."
            )
        if base.prospective_runtime_evidence:
            report["explicit_allowed_differences"].append(
                "Base uses its original zero-epoch driver with actual SpawnSafeRunner/concurrency32 configuration; "
                "its prospective helper/source receipt is matched to the trained initialized loaded-helper identity. "
                "Base policy installation is inferred from the verified runner constructor; "
                "there is no baseline loaded-helper receipt."
            )
        report["identity_scope"] = {
            "untouched_base_hf_fingerprint": base.model_fingerprint,
            "trained_parameter_fingerprint_compared_to_base": False,
            "trained_weight_identity_scope": (
                "Observed final policy version64; trained and base weight bytes are not required to match."
            ),
            "tokenizer_version": base.tokenizer,
            "runtime_source_blobs": base.source_blobs,
            "recorded_weight_versions": {"base": 0, "trained": 64},
            "clock_scope": (
                "Exact aggregate sidecar weight tags; saved numeric metadata has no independent per-case weight spans."
            ),
        }
        report["backend_evidence"] = {"base": base.backend_evidence, "trained": trained.backend_evidence}
        report["lifecycle_evidence"] = {"base": base.lifecycle, "trained": trained.lifecycle}
        report["execution_extension_evidence"] = {
            "base": base.setup_extension,
            "trained": trained.setup_extension,
            "base_prospective_runtime_receipt": base.prospective_runtime_evidence,
            "trained_initialized_receipt": trained.initialized_evidence,
            "inference_configuration_sources": {
                "base": "original_manifest_actual_profile",
                "trained": "validated_initialized_receipt" if trained.initialized_evidence else "original_manifest",
            },
        }
        report["parallel_environment_client_evidence"] = {
            "base": base.parallel_evidence,
            "trained": trained.parallel_evidence,
            "matched_signature": None
            if base.parallel_extension is None
            else parallel_signature(base.parallel_extension),
            "scope": "Source/profile/real prelaunch observations; baseline loading remains source-linked inference",
            "actual_official_service_configs": {
                "base": base.parallel_test_evidence,
                "trained": trained.parallel_test_evidence,
            },
        }
        report["direct_checkpoint_restore_evidence"] = trained.direct_evidence
        report["all_native_spawn_evidence"] = trained.all_native_evidence
    except (Error, KeyError, TypeError, ValueError, OSError, AttributeError, ImportError) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
    changed = reader.changed()
    report["inputs_changed_during_comparison"] = changed
    if changed:
        report["comparison_supported"] = report["improvement_supported"] = False
        report["reasons"].append("Inputs changed during comparison; repeat only after a durable cut")
    report["status"] = "complete" if report["comparison_supported"] else "incomplete_or_unsupported"
    report["limitations"] = [
        "No model evaluation or checkpoint loading was performed.",
        "Declared source cuts and immutable model/tool bytes do not independently attest every in-memory operator.",
        "This compares the final joint system, not separate causal effects of parameter learning and skill memory.",
    ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-stage", required=True, type=Path)
    parser.add_argument("--trained-stage", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON only; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    report = compare_stages(args.base_stage, args.trained_stage)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0 if report["comparison_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
