# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in standalone curator/evaluator routes using official rollout APIs.

Importing this module starts no Ray actors, process groups, or models. Creation
uses ``LLMServerManager.create(worker_group=None)``: each standalone replica
reserves its own exclusive Ray GPU pool. Reserve fewer GPUs for the solver
before creating these routes; fractional sharing of its training pool is not
used. A route is a token-generation endpoint for the existing native agent
adapter, not another agent framework or an unsupported teacher client.

EMA synchronization must run at the trainer's rollout boundary, while online
optimizer updates are stopped. The existing EMA worker keeps its temporary
parameter swap alive through the complete awaited NCCL send. A failed transfer
poisons the route; its owner must stop the job and resume from a coherent
checkpoint. The official manager has no public teardown API, so job shutdown
and partially created resource cleanup remain the owning trainer's job.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

_FILE_HASH_CACHE: dict[tuple, str] = {}
_FILE_HASH_LOCK = threading.Lock()
_MODEL_METADATA = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
    "added_tokens.json", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
}


def _stat_key(path: Path) -> tuple:
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def fingerprint_model(model_path: str | Path, *, force_rehash: bool = False) -> dict[str, Any]:
    """Hash local HF weights and inference metadata, caching by validated file stat.

    The fingerprint contains relative file names, sizes, and SHA256s, so copying
    identical files to another directory preserves its identity. Every call
    re-enumerates selected files and stats them; changed/new/deleted files cannot
    silently reuse the old manifest. This cache assumes an ordinary local/NAS
    filesystem with reliable modification/change timestamps, not adversarial
    preservation of all filesystem metadata. Recently touched files bypass the
    cache because NAS timestamp precision can be coarse. ``force_rehash=True``
    explicitly verifies content even when filesystem stat metadata is unchanged.
    """
    root = Path(model_path).resolve(strict=True)
    if not root.is_dir() or not (root / "config.json").is_file():
        raise ValueError("A local full HF model directory with config.json is required")
    selected = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        name = path.name
        if (
            name in _MODEL_METADATA or name.endswith((".safetensors", ".safetensors.index.json", ".bin.index.json"))
            or (name.startswith(("pytorch_model", "model")) and name.endswith(".bin"))
            or (name.startswith("chat_template") and name.endswith(".jinja"))
            or name.endswith(".tiktoken")
        ):
            selected.append(path)
    selected.sort(key=lambda path: path.relative_to(root).as_posix())
    relative_names = {path.relative_to(root).as_posix() for path in selected}
    if not any(name.endswith((".safetensors", ".bin")) for name in relative_names):
        raise ValueError("The model directory has no supported full-model weight files")
    if not any(name in relative_names for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json")):
        raise ValueError("The pinned model must include its tokenizer files")
    index_hashes = {}
    for path in selected:
        if path.name.endswith(".index.json"):
            content = path.read_bytes()
            index = json.loads(content)
            index_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(content).hexdigest()
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("A model shard index requires a nonempty weight_map")
            for shard in weight_map.values():
                if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError("Model index shard paths must stay inside the model directory")
                if shard not in relative_names:
                    raise ValueError(f"Missing model weight shard referenced by index: {shard}")
    files = []
    observed_stats = []
    for path in selected:
        before = _stat_key(path)
        old_enough = max(before[4], before[5]) < time.time_ns() - 5_000_000_000
        with _FILE_HASH_LOCK:
            digest = _FILE_HASH_CACHE.get(before) if old_enough and not force_rehash else None
        if digest is None:
            hasher = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(8 << 20):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if _stat_key(path) != before:
                raise RuntimeError("Model files changed while computing a frozen fingerprint")
            if old_enough:
                with _FILE_HASH_LOCK:
                    _FILE_HASH_CACHE[before] = digest
        files.append({"path": path.relative_to(root).as_posix(), "size": before[3], "sha256": digest})
        observed_stats.append((path, before))
    if any(_stat_key(path) != before for path, before in observed_stats):
        raise RuntimeError("Model files changed before the fingerprint could be published")
    if any(file["sha256"] != index_hashes[file["path"]] for file in files if file["path"] in index_hashes):
        raise RuntimeError("A model shard index changed while validating its referenced files")
    canonical = json.dumps({"schema_version": 1, "files": files}, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "model_path": str(root),
        "files": files,
    }


@dataclass(frozen=True)
class PolicyRouteSpec:
    """A dedicated single-node dense-model route with immutable provenance.

    ``model_version`` identifies the checkpoint/manifest used to initialize the
    route. For EMA routes, the actual generated policy additionally has an
    observed optimizer-step version. Frozen routes never accept weight updates.
    ``transport_group`` must be distinct from all other transports in the job.
    """

    name: str
    kind: str
    num_gpus: int
    start_rank: int
    model_version: str
    model_path: str | None = None
    tensor_parallel_size: int = 1
    transport_group: str | None = None
    bucket_size_mb: int = 256
    reference_seed: int = 42

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.name):
            raise ValueError("Route names must be simple nonempty identifiers")
        if self.kind not in ("ema", "frozen"):
            raise ValueError("Policy route kind must be ema or frozen")
        for name in ("num_gpus", "tensor_parallel_size", "bucket_size_mb"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.start_rank) is not int or self.start_rank < 0:
            raise ValueError("start_rank must be a nonnegative integer")
        if self.num_gpus % self.tensor_parallel_size:
            raise ValueError("Route GPU count must be divisible by tensor parallel size")
        if type(self.reference_seed) is not int or not 0 <= self.reference_seed < 2**32:
            raise ValueError("reference_seed must be an unsigned 32-bit integer")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("An immutable model/checkpoint provenance identifier is required")
        if self.kind == "frozen" and not self.model_path:
            raise ValueError("A frozen evaluator requires an explicit pinned model path")
        if self.model_path is not None and not Path(self.model_path).is_absolute():
            raise ValueError("Policy route model paths must be absolute local paths")
        if self.transport_group is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", self.transport_group):
            raise ValueError("transport_group must be a simple nonempty NCCL group name")

    @property
    def replica_count(self) -> int:
        return self.num_gpus // self.tensor_parallel_size

    @property
    def group_name(self) -> str:
        return self.transport_group or f"skill_{self.name}_ema"


def validate_route_topology(
    specs: Sequence[PolicyRouteSpec], *, solver_gpus: int, available_gpus: int, solver_replica_count: int | None = None
) -> None:
    """Validate all reservations/rank ranges before launching any new replica."""
    if type(solver_gpus) is not int or solver_gpus <= 0:
        raise ValueError("solver_gpus must be positive")
    if type(available_gpus) is not int or available_gpus <= 0:
        raise ValueError("available_gpus must be positive")
    if not specs or solver_gpus + sum(spec.num_gpus for spec in specs) > available_gpus:
        raise ValueError("Solver and standalone routes exceed the available GPU budget")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Policy route names must be unique")
    ema_specs = [spec for spec in specs if spec.kind == "ema"]
    if len(ema_specs) > 1:
        raise ValueError("The current EMA worker supports one dedicated curator transport")
    occupied = set(range(solver_gpus if solver_replica_count is None else solver_replica_count))
    for spec in specs:
        ranks = set(range(spec.start_rank, spec.start_rank + spec.replica_count))
        if ranks & occupied:
            raise ValueError("Solver and policy route replica rank ranges must not overlap")
        occupied.update(ranks)


def build_route_config(config: Any, spec: PolicyRouteSpec):
    """Clone the main config, leaving the online solver configuration untouched."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    is_omega = OmegaConf is not None and OmegaConf.is_config(config)
    data = OmegaConf.to_container(config, resolve=False) if is_omega else copy.deepcopy(config)
    if not isinstance(data, dict):
        raise TypeError("A route configuration must be a mapping or OmegaConf DictConfig")
    actor = data["actor_rollout_ref"]
    model = actor["model"]
    rollout = actor["rollout"]
    if rollout.get("name") != "vllm":
        raise NotImplementedError("The initial policy route supports the verified vLLM backend only")
    if rollout.get("disaggregation", {}).get("enabled", False):
        raise NotImplementedError("Policy routes do not combine standalone replicas with PD disaggregation")
    if model.get("lora_rank", 0) or model.get("lora", {}).get("rank", 0):
        raise NotImplementedError("EMA routes currently support dense non-LoRA policies")
    rollout.update(
        nnodes=1,
        n_gpus_per_node=spec.num_gpus,
        tensor_model_parallel_size=spec.tensor_parallel_size,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        calculate_log_probs=spec.kind == "ema",
        load_format="dummy" if spec.kind == "ema" else "auto",
        # vLLM 0.24's batch-invariant mode rejects Qwen3.5 GDN_ATTN. Reference
        # reproducibility uses frozen weights, greedy decoding and an explicit
        # per-request seed instead; this is not a bitwise batch-invariance claim.
        full_determinism=False,
    )
    rollout.setdefault("prometheus", {})["served_model_name"] = f"skill-{spec.name}"
    checkpoint = rollout.setdefault("checkpoint_engine", {})
    checkpoint.update(
        backend="nccl" if spec.kind == "ema" else "naive",
        update_weights_bucket_megabytes=spec.bucket_size_mb,
    )
    if spec.kind == "ema":
        checkpoint.setdefault("engine_kwargs", {})["nccl"] = {
            "group_name": spec.group_name,
            "multi_sender": False,
        }
    if spec.model_path is not None:
        # Avoid retaining a cached solver path/config after pinning another checkpoint.
        model.update(path=spec.model_path, hf_config_path=spec.model_path, tokenizer_path=spec.model_path)
        for field in (
            "local_path", "local_hf_config_path", "local_tokenizer_path", "hf_config", "generation_config",
            "tokenizer", "processor", "architectures",
        ):
            if field in model:
                model[field] = None
    return OmegaConf.create(data, flags={"allow_objects": True}) if is_omega else data


async def _resolved(value):
    """Accept normal local values, awaitables, and lists of Ray ObjectRefs."""
    if inspect.isawaitable(value):
        return await value
    if isinstance(value, list | tuple):
        return await asyncio.gather(*[_resolved(item) for item in value])
    return value


class EMASenderView:
    """Adapt only the EMA RPCs to the stock CheckpointEngineManager protocol.

    No online update_weights or online execute_checkpoint_engine method is used.
    Rank-specific topology dispatch remains the existing DP_COMPUTE worker RPC.
    """

    def __init__(self, actor_worker_group):
        self.actor_worker_group = actor_worker_group
        self.world_size = actor_worker_group.world_size

    def update_weights(self, global_steps: int | None = None, mode: str = "nccl"):
        if mode != "nccl":
            raise ValueError("The dedicated EMA sender only accepts NCCL transfer")
        return self.actor_worker_group.send_ema_weights(global_steps=global_steps)

    def execute_checkpoint_engine(self, method, *args, **kwargs):
        methods = method if isinstance(method, list | tuple) else [method]
        if not methods or any(item not in ("prepare", "init_process_group", "finalize") for item in methods):
            raise ValueError("Unsupported EMA transport lifecycle operation")
        return self.actor_worker_group.execute_ema_transport(method, *args, **kwargs)


class PolicyRouteClient:
    """Serializable token client pinned to one published policy version.

    An old client deliberately fails when servers have newer weights. Recreate
    clients after each EMA publication. This avoids silently recording curate
    tokens against another behavior policy. The native agent still runs through
    its usual LLMServerClient-compatible generate interface.
    """

    def __init__(self, delegate, spec: PolicyRouteSpec, optimizer_step: int | None, attestation: Mapping[str, Any]):
        self.delegate = delegate
        self.spec = spec
        self.optimizer_step = optimizer_step
        self.config = delegate.config
        self._attestation = copy.deepcopy(dict(attestation))
        self._frozen_clock_seen = False
        self._frozen_observed_clock = None

    @property
    def frozen_version(self) -> str | None:
        return self.spec.model_version if self.spec.kind == "frozen" else None

    @property
    def attestation(self) -> dict[str, Any]:
        return copy.deepcopy(self._attestation)

    async def generate(self, request_id, **kwargs):
        if self.spec.kind == "frozen" and (
            self._attestation.get("policy_kind") != "frozen_evaluator"
            or self._attestation.get("frozen_model_version") != self.frozen_version
            or self._attestation.get("model_fingerprint") != self.frozen_version
            or not self._attestation.get("lifecycle_id")
        ):
            raise RuntimeError("The frozen client has invalid model or server-lifecycle provenance")
        if self.spec.kind == "frozen":
            sampling = dict(kwargs.get("sampling_params") or {})
            if sampling.get("temperature", 0.0) != 0.0:
                raise ValueError("The reference evaluator requires greedy temperature-zero decoding")
            if sampling.get("seed", self.spec.reference_seed) != self.spec.reference_seed:
                raise ValueError("Reference sampling seed differs from the declared evaluation protocol")
            sampling.update(temperature=0.0, seed=self.spec.reference_seed)
            kwargs["sampling_params"] = sampling
        output = await self.delegate.generate(request_id=request_id, **kwargs)
        if output.stop_reason in ("abort", "aborted"):
            raise RuntimeError("Policy route generation was interrupted; retry with the current pinned client")
        extra = output.extra_fields
        if self.spec.kind == "ema":
            observed = extra.get("global_steps")
            minimum = extra.get("min_global_steps", observed)
            maximum = extra.get("max_global_steps", observed)
            if any(value != self.optimizer_step for value in (observed, minimum, maximum)):
                raise RuntimeError("Curator response has a missing, mixed, or stale EMA policy version")
            if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
                raise RuntimeError("Curator output must preserve one behavior logprob for each generated token")
            extra["policy_kind"] = "ema_curator"
            extra["ema_optimizer_step"] = self.optimizer_step
        else:
            observed = extra.get("global_steps")
            clocks = [observed, extra.get("min_global_steps", observed), extra.get("max_global_steps", observed)]
            if any(value != clocks[0] for value in clocks):
                raise RuntimeError("A frozen evaluator response contains mixed observed weight clocks")
            if self._frozen_clock_seen and self._frozen_observed_clock != observed:
                raise RuntimeError("The frozen evaluator weight clock changed during its private lifecycle")
            self._frozen_clock_seen = True
            self._frozen_observed_clock = observed
            extra["policy_kind"] = "frozen_evaluator"
            extra["frozen_model_version"] = self.frozen_version
            extra["observed_global_steps"] = observed
            extra["policy_route_lifecycle_id"] = self._attestation["lifecycle_id"]
        extra["skill_policy_route"] = {
            "name": self.spec.name,
            "kind": self.spec.kind,
            "model_version": self.spec.model_version,
            "optimizer_step": self.optimizer_step,
        }
        return output


class PolicyRoute:
    """A standalone inference manager plus an optional isolated EMA weight sender."""

    def __init__(self, spec, config, manager, actor_worker_group=None, checkpoint_manager=None, model_manifest=None):
        self.spec = spec
        self.config = config
        self.manager = manager
        self.actor_worker_group = actor_worker_group
        self.checkpoint_manager = checkpoint_manager
        self.optimizer_step = None
        self.ready = spec.kind == "frozen"
        self.failed = False
        self._syncing = False
        self._attestation = {
            "schema_version": 1,
            "policy_kind": "frozen_evaluator" if spec.kind == "frozen" else "ema_curator",
            "frozen_model_version": spec.model_version if spec.kind == "frozen" else None,
            "model_path": model_manifest["model_path"] if model_manifest is not None else spec.model_path,
            "model_fingerprint": model_manifest["fingerprint"] if model_manifest is not None else None,
            "lifecycle_id": uuid4().hex,
            "reference_seed": spec.reference_seed if spec.kind == "frozen" else None,
            "batch_invariant": False,
        }

    @property
    def frozen_version(self) -> str | None:
        return self.spec.model_version if self.spec.kind == "frozen" else None

    @property
    def attestation(self) -> dict[str, Any]:
        """Return immutable-by-copy provenance captured beside native trajectories."""
        return copy.deepcopy(self._attestation)

    def get_client(self) -> PolicyRouteClient:
        if not self.ready or self.failed or self._syncing:
            raise RuntimeError("The policy route has no completely published policy available")
        terminal = self.config.get("skill_harness", {}).get("terminal", {})
        if self.spec.kind == "frozen" and terminal.get("enabled", False):
            from .alfworld.backend_terminal import TerminalAwareLLMServerClient

            delegate = self.manager.get_client(
                client_cls=TerminalAwareLLMServerClient, binding_root=terminal["binding_root"],
            )
        else:
            delegate = self.manager.get_client()
        return PolicyRouteClient(delegate, self.spec, self.optimizer_step, self.attestation)

    async def sync_ema(self, optimizer_step: int):
        """Publish one observed EMA version, then reopen replica submission gates."""
        if self.spec.kind != "ema":
            raise RuntimeError("A frozen evaluator never accepts online or EMA weight updates")
        if type(optimizer_step) is not int or optimizer_step < 0:
            raise ValueError("optimizer_step must be an actual nonnegative optimizer-update count")
        if self.failed or self._syncing:
            raise RuntimeError("EMA route is failed or another publication is active")
        if self.optimizer_step is not None and optimizer_step < self.optimizer_step:
            raise ValueError("EMA route versions cannot move backwards")
        self._syncing = True
        try:
            states = await _resolved(self.actor_worker_group.get_ema_metadata())
            if not isinstance(states, list) or len(states) != self.actor_worker_group.world_size:
                raise RuntimeError("EMA metadata must cover every online actor rank")
            if any(state["last_optimizer_step"] != optimizer_step for state in states):
                raise RuntimeError("Requested publication differs from an actor rank's actual EMA update count")
            if self.ready and self.optimizer_step == optimizer_step:
                return self.describe()
            self.ready = False
            await _resolved(self.checkpoint_manager.update_weights(global_steps=optimizer_step))
            # vLLM pause_generation remains closed after the stock manager restores KV cache.
            await asyncio.gather(*[replica.resume_generation() for replica in self.manager.get_replicas()])
            self.optimizer_step = optimizer_step
            self.ready = True
            return self.describe()
        except BaseException:
            self.failed = True
            self.ready = False
            raise
        finally:
            self._syncing = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "kind": self.spec.kind,
            "model_version": self.spec.model_version,
            "frozen_version": self.frozen_version,
            "model_path": self.spec.model_path,
            "num_gpus": self.spec.num_gpus,
            "replica_ranks": list(range(self.spec.start_rank, self.spec.start_rank + self.spec.replica_count)),
            "optimizer_step": self.optimizer_step,
            "ready": self.ready,
            "failed": self.failed,
            "addresses": list(self.manager.get_addresses()),
            "attestation": self.attestation,
        }


async def create_policy_routes(
    config,
    specs: Sequence[PolicyRouteSpec],
    *,
    actor_worker_group,
    solver_gpus: int,
    available_gpus: int,
    solver_replica_count: int | None = None,
    manager_factory=None,
    checkpoint_manager_factory=None,
    fingerprint_factory=None,
) -> dict[str, PolicyRoute]:
    """Launch disjoint standalone pools after the solver reserves its GPU pool.

    Factories are injectable for CPU contract tests. Production defaults use the
    official manager/checkpoint APIs, with no model-engine or trainer patch.
    Initialize/load actor EMA before calling this function. The EMA client is
    unavailable until the caller completes the initial ``sync_ema(step)``.
    """
    validate_route_topology(
        specs, solver_gpus=solver_gpus, available_gpus=available_gpus, solver_replica_count=solver_replica_count
    )
    fingerprint_factory = fingerprint_factory or fingerprint_model
    prepared = []
    for spec in specs:
        route_config = build_route_config(config, spec)
        manifest = None
        if spec.kind == "frozen":
            manifest = await asyncio.to_thread(fingerprint_factory, spec.model_path)
            if manifest["fingerprint"] != spec.model_version:
                raise ValueError("Pinned evaluator model fingerprint differs from its declared version")
        prepared.append((spec, route_config, manifest))
    if manager_factory is None:
        from omegaconf import OmegaConf

        from verl.workers.rollout.llm_server import LLMServerManager

        def manager_factory(*, config, **kwargs):
            if not OmegaConf.is_config(config):
                config = OmegaConf.create(config)
            return LLMServerManager.create(config=config, **kwargs)
    if checkpoint_manager_factory is None:
        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.utils.config import omega_conf_to_dataclass

        def checkpoint_manager_factory(route_config, sender, replicas):
            return CheckpointEngineManager(
                config=omega_conf_to_dataclass(route_config["actor_rollout_ref"]["rollout"]["checkpoint_engine"]),
                actor_wg=sender,
                replicas=replicas,
            )

    routes = {}
    for spec, route_config, manifest in prepared:
        checkpoint_manager = None
        if spec.kind == "ema":
            if actor_worker_group is None:
                raise ValueError("An EMA route requires the online actor worker group")
            await _resolved(
                actor_worker_group.configure_ema_transport(
                    "nccl", bucket_size_mb=spec.bucket_size_mb,
                    engine_kwargs={"group_name": spec.group_name, "multi_sender": False},
                )
            )
        manager = await _resolved(
            manager_factory(config=route_config, worker_group=None, start_rank=spec.start_rank)
        )
        if len(manager.get_replicas()) != spec.replica_count:
            raise RuntimeError("Official standalone manager created an unexpected number of replicas")
        if spec.kind == "ema":
            checkpoint_manager = checkpoint_manager_factory(
                route_config, EMASenderView(actor_worker_group), manager.get_replicas()
            )
        routes[spec.name] = PolicyRoute(spec, route_config, manager, actor_worker_group, checkpoint_manager, manifest)
    return routes
