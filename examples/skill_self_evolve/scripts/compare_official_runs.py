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
LIFECYCLE_IDENTITY = (
    "driver_path", "driver_sha256", "driver_source_commit", "original_training_manifest_sha256",
)
LIFECYCLE_SEAL = (
    "seal_origin", "optimizer_step", "checkpoint_cursor", "checkpoint_dir", "manifest_id",
    "snapshot_id", "store_root", "checkpoint_core_versions",
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
        self.check_protocol()
        self.check_identity()
        self.signature = self.config_signature(self.config)
        self.check_lifecycle_binding()

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
        exact_keys(
            receipt, ("schema_version", "kind", *LIFECYCLE_IDENTITY, *LIFECYCLE_SEAL, "evaluation_cursor", *links),
            "Final lifecycle receipt",
        )
        need(
            receipt["schema_version"] == 1 and receipt["kind"] == "quiescent_final_evaluation"
            and type(receipt["optimizer_step"]) is int and receipt["optimizer_step"] == self.version
            and type(receipt["checkpoint_cursor"]) is int and receipt["checkpoint_cursor"] == 64
            and isinstance(receipt["snapshot_id"], str) and re.fullmatch(r"[0-9a-f]{64}", receipt["snapshot_id"])
            and receipt["evaluation_cursor"] == self.protocol["global_step"]
            and receipt["seal_origin"] in (
                "final_checkpoint_saved_after_quiescence", "restored_complete_final_checkpoint"
            ),
            "Final lifecycle stage/clock/seal differs",
        )
        need(
            receipt["original_training_manifest_sha256"] == self.reader.hash_file(self.root / "training-manifest.json"),
            "Lifecycle receipt does not bind the preserved original training manifest",
        )
        source = Path(self.manifest["import_provenance"]["verl_source"]).resolve()
        driver = Path(receipt["driver_path"])
        need(driver == source / LIFECYCLE_DRIVER, "Unexpected lifecycle driver source binding")
        commit = receipt["driver_source_commit"]
        need(
            isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit),
            "Missing lifecycle driver Git identity",
        )
        try:
            blob = subprocess.check_output(
                ["git", "-C", str(source), "show", commit + ":" + LIFECYCLE_DRIVER],
                stderr=subprocess.PIPE, timeout=15,
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
                target.is_absolute() and target == target.resolve()
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
                and all(record.get(field) == receipt[field] for field in (*LIFECYCLE_IDENTITY, *LIFECYCLE_SEAL)),
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
        need(
            store.is_absolute() and store == store.resolve() and store.name == "skill_store"
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
        manifest_path = checkpoint / "joint_manifest.json"
        envelope = json.loads(self.reader.bytes(manifest_path))
        exact_keys(envelope, ("manifest_id", "payload"), "Joint checkpoint envelope")
        payload = envelope["payload"]
        need(
            envelope["manifest_id"] == receipt["manifest_id"] == "sha256:" + INDEX.digest(payload)
            and payload.get("schema_version") == 1 and payload.get("state") == "complete",
            "Final checkpoint is not the intact sealed COMPLETE publication",
        )
        spec = pure_module("joint_checkpoint").JointCheckpointSpec(**payload["spec"])
        need(
            spec.optimizer_step == self.version and spec.global_step == receipt["checkpoint_cursor"]
            and spec.world_size == 4 and dict(spec.versions) == receipt["checkpoint_core_versions"]
            and spec.versions["verl"] == "0.9.1" and spec.versions["model"] == self.expected_fingerprint
            and spec.versions["tokenizer"] == self.tokenizer
            and spec.protocol_fingerprint == self.controller["protocol_fingerprint"],
            "Final checkpoint sealed policy/core/protocol identity differs",
        )
        need(payload["training_state_paths"] == ["data.pt"], "Unsupported synchronous checkpoint data-state inventory")
        actual = set()
        for component in ("actor", "joint_state", "data.pt"):
            location = checkpoint / component
            for file in location.rglob("*") if location.is_dir() else (location,):
                relative = file.relative_to(checkpoint)
                need(not file.is_symlink(), "Checkpoint file inventory contains a symlink")
                if "locks" not in relative.parts and file.name != ".writer.lock" and file.is_file():
                    actual.add(relative.as_posix())
        need(actual == set(payload["files"]), "Final COMPLETE checkpoint file inventory differs")
        for relative, identity in payload["files"].items():
            safe = PurePosixPath(relative)
            need(not safe.is_absolute() and ".." not in safe.parts and str(safe) == relative, "Unsafe checkpoint file")
            file = checkpoint / relative
            need(
                identity == {"sha256": self.reader.hash_file(file), "size": file.stat().st_size},
                "Final COMPLETE checkpoint file digest differs: " + relative,
            )
        for rank in range(4):
            need(
                all(
                    f"actor/{kind}_world_size_4_rank_{rank}.pt" in actual for kind in ("model", "optim", "extra_state")
                ),
                "Final online model/optimizer/RNG rank inventory is incomplete",
            )
        ema = payload["ema_metadata"]
        need(
            len(ema) == 4 and {row["file"] for row in ema} == {f"ema_rank_{rank:05d}.pt" for rank in range(4)}
            and all(
                row["last_optimizer_step"] == self.version and "joint_state/ema/" + row["file"] in actual for row in ema
            ),
            "Final COMPLETE checkpoint lacks all four EMA64 acknowledgements",
        )
        saved_controller = self.reader.json(checkpoint / "joint_state/controller.json")
        need(
            saved_controller["optimizer_step"] == self.version and saved_controller["prepared_update"] is None
            and saved_controller["protocol_fingerprint"] == spec.protocol_fingerprint,
            "Final saved controller does not match the sealed applied policy",
        )
        head = self.reader.json(store / "refs/main.json")
        snapshot_id = receipt["snapshot_id"]
        snapshot = self.reader.json(store / "snapshots" / (snapshot_id + ".json"))
        captured_snapshot = self.reader.json(checkpoint / "joint_state/skill_store/snapshots" / (snapshot_id + ".json"))
        need(
            head == {"branch": "main", "snapshot_id": snapshot_id}
            and INDEX.digest(snapshot) == snapshot_id and snapshot == captured_snapshot
            and payload["snapshots"]["skill_heads"]["main"] == snapshot_id,
            "Final restored store/head differs from its COMPLETE checkpoint seal",
        )
        self.evaluation_store_root, self.sealed_snapshot_id = store, snapshot_id
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
            task_path.resolve() == self.root / "task_config.yaml"
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
            need(value == str(self.root / RUN_PATHS[path]), "Unknown host path change: " + path)
            return "<run>/" + RUN_PATHS[path]
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
        source = self.manifest["import_provenance"]["verl_source"]
        if path in (
            "ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL",
            "ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL",
        ):
            need(value in ("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"), "Unknown logging level metadata")
            return "<logging-only-metadata>"
        if path == "trainer.v1.sampler.custom_sampler.path":
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
            need(value == "disable" if self.base else value in ("auto", "disable"), "Unsupported resume mode")
            return "<explicit-zero-base-or-trained-run>"
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
