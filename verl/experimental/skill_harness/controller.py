# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""One learner's native solver/EMA-curator/delayed-credit coordination.

The trainer owns the exclusive learner fence and the single actual optimizer
RPC. ``prepare_update`` runs only after ordinary solver GRPO advantages exist;
``after_update_and_sync`` runs after that RPC returns and before source TQ
cleanup. Curate actions are never added to solver GRPO groups or solver metrics.

Delayed probes use only later, original train tasks encountered in real solver
batches. A missing exposure is pending/censored, never a zero utility label.
Expired optimizer leases require explicit checkpoint/worker reconciliation.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .alfworld.data import GameRecord
from .context import TokenizerRenderer
from .credit import CreditLedger, CreditProtocol
from .curate import (
    SKILL_FIELDS,
    SKILL_SCHEMA,
    CurateAbstention,
    CurateCollection,
    CurateOutputError,
    CurateRequest,
    CurateSourceCensored,
    CuratorBehavior,
    collect_curate_candidate,
    load_completed_solver_episode,
    provision_native_curator_runtime,
    redact_completed_episode,
)
from .framework import loading_settings
from .policy_routes import PolicyRouteSpec, create_policy_routes, fingerprint_model
from .probes import ProbeNotObserved, ProbeRequest, collect_counterfactual_probe
from .replay import CurateReplay
from .store import SkillRef, SkillStore


class ControllerRecoveryRequired(RuntimeError):
    """An unresolved update must be reconciled against a coherent checkpoint."""


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    for name in ("tolist", "item"):
        method = getattr(value, name, None)
        if callable(method):
            return _plain(method())
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _write_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".controller-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema_version": 1, "sha256": _digest(payload), "payload": _plain(payload)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_state(path):
    envelope = json.loads(path.read_text())
    if envelope.get("schema_version") != 1 or envelope.get("sha256") != _digest(envelope["payload"]):
        raise ControllerRecoveryRequired("Controller state failed its content integrity check")
    return envelope["payload"]


async def _resolve(value):
    if inspect.isawaitable(value):
        return await value
    if isinstance(value, list | tuple):
        return await asyncio.gather(*[_resolve(item) for item in value])
    if type(value).__name__ == "DataProtoFuture" and type(value).__module__ == "verl.protocol":
        return await asyncio.to_thread(value.get)
    return value


@dataclass(frozen=True)
class JointSettings:
    solver_gpus: int = 4
    available_gpus: int = 8
    curator_gpus: int = 2
    credit_gpus: int = 2
    route_tensor_parallel_size: int = 1
    gateway_count: int = 2
    candidates_per_source: int = 2
    probes_per_step: int = 2
    required_exposures: int = 1
    min_delay_steps: int = 1
    replay_batch_size: int = 2
    max_replay_lag_steps: int = 64
    max_pending_windows: int = 64
    window_seconds: float = 86400.0
    lease_seconds: float = 3600.0
    cost_weight: float = 0.0
    max_label_updates: int = 1
    admission_threshold: float = 0.0
    seed: int = 42
    curator_native_prefix_reserve_tokens: int = 4096
    curator_output_headroom_tokens: int = 2048
    curator_concurrency: int = 8
    credit_concurrency: int = 8
    max_source_backlog: int = 16384

    @classmethod
    def from_config(cls, config):
        requested = _plain(_get(config.skill_harness, "joint", {}) or {})
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in requested.items() if key in allowed})

    def __post_init__(self):
        for key in (
            "solver_gpus",
            "available_gpus",
            "curator_gpus",
            "credit_gpus",
            "route_tensor_parallel_size",
            "gateway_count",
            "candidates_per_source",
            "required_exposures",
            "min_delay_steps",
            "max_pending_windows",
            "max_label_updates",
            "curator_output_headroom_tokens",
            "curator_concurrency",
            "credit_concurrency",
            "max_source_backlog",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"joint.{key} must be a positive integer")
        for key in (
            "probes_per_step",
            "replay_batch_size",
            "max_replay_lag_steps",
            "curator_native_prefix_reserve_tokens",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"joint.{key} must be a nonnegative integer")
        if self.window_seconds <= 0 or self.lease_seconds <= 0 or self.cost_weight < 0:
            raise ValueError("Credit windows/leases must be positive and costs nonnegative")
        if self.solver_gpus + self.curator_gpus + self.credit_gpus > self.available_gpus:
            raise ValueError("Joint policy reservations exceed the actual GPU budget")
        if self.probes_per_step and self.probes_per_step < self.candidates_per_source:
            raise ValueError("Probe budget must cover one logical exposure for every candidate in a cohort")


@dataclass(frozen=True)
class PreparedJointUpdate:
    data: Any
    update_id: str
    global_step: int
    before_optimizer_step: int
    expected_optimizer_step: int
    acknowledgments: tuple[dict[str, str], ...]
    anchor_policy_version: str
    anchor_input_fingerprint: str


@dataclass(frozen=True)
class ControllerIdentity:
    frozen_version: str
    protocol_fingerprint: str
    tokenizer_version: str
    template_version: str
    environment_version: str
    distribution_version: str
    frozen_model_path: str
    runtime: Any
    settings: JointSettings
    loading: dict

    def manifest(self):
        return {
            key: getattr(self, key)
            for key in (
                "frozen_version",
                "protocol_fingerprint",
                "tokenizer_version",
                "template_version",
                "environment_version",
                "distribution_version",
                "frozen_model_path",
            )
        }


def preflight_controller_identity(config, *, native_runtime=None):
    """Compute current compatibility before any extra GPU/model allocation."""
    settings = JointSettings.from_config(config)
    loading = loading_settings(config)
    if not loading["enabled"]:
        raise ValueError("Joint training requires the host skill loading path")
    options = _get(config.skill_harness, "joint", {})
    frozen_path = Path(_get(options, "frozen_model_path", str(config.actor_rollout_ref.model.path))).resolve()
    frozen_version = fingerprint_model(frozen_path)["fingerprint"]
    if native_runtime is None:
        task_config = json.loads(Path(os.environ["ALFWORLD_TASK_CONFIG"]).read_text())
        native_task = next(row for row in task_config if row["name"] == "alfworld_skill_training")
        native_runtime = provision_native_curator_runtime(
            public_root=_get(options, "curator_public_root"),
            work_root=_get(options, "curator_work_root"),
            native_bin=os.environ["CODEX_NATIVE_BIN"],
            package_root=os.environ["ALFWORLD_CODEX_PACKAGE_ROOT"],
            model_catalog_path=native_task["agent"]["model_catalog_path"],
            model_name=_get(options, "model_name"),
            unix_user=_get(options, "curator_unix_user"),
        )
    identity = object.__new__(SkillJointController)
    identity.config, identity.settings, identity.loading = config, settings, loading
    identity.runtime, identity.frozen_version = native_runtime, frozen_version
    identity._configure_protocol()
    return ControllerIdentity(
        frozen_version,
        identity.protocol_fingerprint,
        loading["tokenizer_version"],
        native_runtime.template_version,
        identity.environment_version,
        identity.distribution_version,
        str(frozen_path),
        native_runtime,
        settings,
        loading,
    )


class SkillJointController:
    """Exclusive coordinator; factories replace only native/route dependencies."""

    @classmethod
    async def create(
        cls,
        trainer,
        *,
        config=None,
        route_factory=None,
        gateway_factory=None,
        native_runtime=None,
        restored_state=None,
    ):
        config = config or trainer.config
        settings = JointSettings.from_config(config)
        if trainer.actor_rollout_wg.world_size != settings.solver_gpus:
            raise ValueError("Reserve the configured solver GPU pool before creating joint policy routes")
        self = cls()
        self.trainer, self.config, self.settings = trainer, config, settings
        self.actor = trainer.actor_rollout_wg
        self.artifact_root = Path(config.skill_harness.artifact_root).resolve()
        self.state_path = self.artifact_root / "controller/state.json"
        restored_bundle = restored_state if hasattr(restored_state, "skill_store") else None
        if restored_bundle is not None:
            self.store, self.replay, self.ledger = (
                restored_bundle.skill_store,
                restored_bundle.replay,
                restored_bundle.ledger,
            )
            restored_state = restored_bundle.controller_state
            if self.store.root.resolve() != Path(config.skill_harness.store_root).resolve():
                raise ControllerRecoveryRequired(
                    "Bind the restored SkillStore in rollout config before creating its manager"
                )
        else:
            self.store = SkillStore(config.skill_harness.store_root)
            self.replay = CurateReplay(_get(config.skill_harness, "replay_root", self.artifact_root / "curate_replay"))
            self.ledger = CreditLedger(
                _get(config.skill_harness, "ledger_root", self.artifact_root / "credit_ledger"), self.replay
            )
        identity = await asyncio.to_thread(preflight_controller_identity, config, native_runtime=native_runtime)
        self.loading, self.runtime = identity.loading, identity.runtime
        self.frozen_version = identity.frozen_version
        self.protocol_fingerprint = identity.protocol_fingerprint
        self.environment_version, self.distribution_version = (
            identity.environment_version,
            identity.distribution_version,
        )
        self.dp_size, self.pad_token_id = self.actor.world_size, trainer.tokenizer.pad_token_id
        self._busy = False
        self._writes_allowed = asyncio.Event()
        self._writes_allowed.set()
        self._background_tasks = {}
        self._background_pump = None
        self._background_stopping = False
        self._ema_syncing = False
        self._learner_active = True
        self._ema_refresh_requested = False
        self._ema_publication_task = None
        self._gateway_factory = gateway_factory
        self._collect_curate = collect_curate_candidate
        self._collect_probe = collect_counterfactual_probe
        self._load_source = load_completed_solver_episode
        self.ema_gateway = None
        model_path = Path(config.actor_rollout_ref.model.path).resolve()
        frozen_path = Path(identity.frozen_model_path)
        specifications = [
            PolicyRouteSpec(
                "curator",
                "ema",
                settings.curator_gpus,
                10000,
                "initial:" + identity.frozen_version,
                model_path=str(model_path),
                tensor_parallel_size=settings.route_tensor_parallel_size,
                transport_group="skill_curator_ema",
            ),
            PolicyRouteSpec(
                "credit",
                "frozen",
                settings.credit_gpus,
                20000,
                self.frozen_version,
                model_path=str(frozen_path),
                tensor_parallel_size=settings.route_tensor_parallel_size,
                reference_seed=settings.seed,
            ),
        ]
        self.routes = await _resolve(
            (route_factory or create_policy_routes)(
                config,
                specifications,
                actor_worker_group=self.actor,
                solver_gpus=settings.solver_gpus,
                available_gpus=settings.available_gpus,
            )
        )
        clock = await self._optimizer_clock()
        await self.routes["curator"].sync_ema(clock)
        self.task_config_path = os.environ["ALFWORLD_TASK_CONFIG"]
        if restored_state is not None:
            self.restore_state(restored_state, expected_optimizer_step=clock)
        elif self.state_path.exists():
            self.restore_state(_read_state(self.state_path), expected_optimizer_step=clock)
        else:
            self.state = {
                "schema_version": 1,
                "optimizer_step": clock,
                "protocol_fingerprint": self.protocol_fingerprint,
                "pending": {},
                "cohorts": {},
                "prepared_update": None,
                "applied_update_ids": {},
                "last_global_step": 0,
                "collections": {},
                "configuration": asdict(settings),
            }
            self.save_state()
        from .probes import create_frozen_probe_gateway

        self.frozen_gateway = await create_frozen_probe_gateway(
            self._gateway_config(self.routes["credit"].config),
            self.routes["credit"],
            gateway_factory=gateway_factory,
        )
        await self._refresh_ema_gateway()
        self._initialize_background()
        return self

    def _configure_protocol(self):
        manifest_path = Path(self.config.skill_harness.split_manifest)
        manifest = json.loads(manifest_path.read_text())
        service = json.loads(Path(os.environ["ALFWORLD_SERVER_CONFIG"]).read_text())
        if (
            set(service.get("allowed_splits", [])) != {"train"}
            or service["catalog_digest"] != manifest["catalog_digest"]
        ):
            raise ValueError("Joint learning and probes must attach only to this run's train-only environment")
        self.environment_version = "alfworld:" + _digest(
            {
                "catalog_digest": manifest["catalog_digest"],
                "max_steps": service["max_steps"],
                "protocol": service.get("protocol_version", "alfworld-v1"),
                "budget_seconds": service.get("budget_seconds", 900),
            }
        )
        self.distribution_version = "train:" + manifest["parquet_files"]["train"]["sha256"]
        self.protocol_fingerprint = _digest(
            {
                "schema_version": 1,
                "settings": asdict(self.settings),
                "frozen_version": self.frozen_version,
                "environment_version": self.environment_version,
                "distribution_version": self.distribution_version,
                "loading": self.loading,
                "native_runtime": self.runtime.manifest(),
                "curate_admission_policy": "complete-cohort-semantic-gates-v1",
                "curate_source_policy": "whole-redacted-source-or-censor-v1",
                "frozen_generation_policy": {
                    "temperature": 0.0, "seed": self.settings.seed, "batch_invariant": False,
                },
            }
        )

    def _gateway_config(self, config):
        from omegaconf import open_dict

        result = copy.deepcopy(config)
        framework = result.actor_rollout_ref.rollout.custom.agent_framework
        with open_dict(framework):
            framework.allowed_request_sampling_param_keys = []
            framework.gateway_count = self.settings.gateway_count
        return result

    async def _refresh_ema_gateway(self):
        if self.ema_gateway is not None:
            await _resolve(self.ema_gateway.shutdown())
        factory = self._gateway_factory
        if factory is None:
            from uni_agent.framework.entry import build_gateway_manager

            factory = build_gateway_manager
        route = self.routes["curator"]
        self.ema_gateway = await _resolve(
            factory(config=self._gateway_config(route.config), llm_client=route.get_client())
        )

    async def _optimizer_clock(self):
        states = await _resolve(self.actor.get_ema_metadata())
        if not isinstance(states, list) or len(states) != self.actor.world_size:
            raise RuntimeError("Actual optimizer/EMA metadata must cover every actor rank")
        clocks = [row["last_optimizer_step"] for row in states]
        if not clocks or type(clocks[0]) is not int or clocks[0] < 0 or any(value != clocks[0] for value in clocks):
            raise RuntimeError("Online optimizer and EMA clocks differ across actor ranks")
        return clocks[0]

    def export_state(self):
        return json.loads(_canonical(self.state))

    def save_state(self):
        _write_state(self.state_path, self.state)

    def restore_state(self, payload, *, expected_optimizer_step):
        payload = json.loads(_canonical(payload))
        if (
            payload.get("schema_version") != 1
            or payload.get("optimizer_step") != expected_optimizer_step
            or payload.get("protocol_fingerprint") != self.protocol_fingerprint
        ):
            raise ControllerRecoveryRequired(
                "Controller queue/protocol must be restored with its matching model checkpoint"
            )
        for key in ("pending", "cohorts", "collections", "applied_update_ids"):
            if not isinstance(payload.get(key), dict):
                raise ControllerRecoveryRequired("Incomplete joint controller checkpoint state")
        self.state = payload
        self.save_state()

    def _sample_fields(self, data):
        from verl.utils import tensordict_utils as tu

        if "raw_prompt" not in data:
            raise ValueError("Materialize original raw_prompt for actual future-train credit probes")
        columns = {
            key: _plain(tu.get(data, key))
            for key in ("uid", "session_id", "raw_prompt", "tools_kwargs", "extra_fields")
        }
        unique, sources = {}, {}
        for index in range(len(data)):
            if not bool(data["response_mask"][index].any()):
                continue
            tools = columns["tools_kwargs"][index]
            task = tools.get("task", {})
            metadata = task.get("metadata", {})
            if (
                task.get("name") != "alfworld_skill_training"
                or metadata.get("split") != "train"
                or metadata.get("dataset_phase") != "train"
            ):
                raise ValueError("Joint update/probe sources must be identifiable original optimization-train tasks")
            GameRecord.from_dict(metadata)
            stable_id = metadata.get("stable_game_id", metadata["game_id"])
            fields = {
                "uid": str(columns["uid"][index]),
                "raw_prompt": columns["raw_prompt"][index],
                "tools_kwargs": tools,
            }
            unique.setdefault(stable_id, fields)
            extra = columns["extra_fields"][index]
            context = extra.get("runner_reward_info", {}).get("reward_context", {})
            output = context.get("episode_output_dir")
            if not output:
                continue
            path = Path(output).resolve()
            if not path.is_relative_to(self.artifact_root / "episodes"):
                raise ValueError("Native solver source episode escaped this run's private artifacts")
            sources.setdefault(
                str(path),
                {
                    "producer_uid": fields["uid"],
                    "episode_output_dir": str(path),
                    "task_id": stable_id,
                    "task_type": metadata["task_type"],
                },
            )
        selected = [sources[key] for key in sorted(sources)]
        return list(unique.values()), selected

    def _protocol(self, baseline):
        return CreditProtocol(
            evaluator_version=self.frozen_version,
            probe_protocol_version="native-paired-greedy:v2",
            task_distribution_version=self.distribution_version,
            environment_version=self.environment_version,
            baseline_snapshot_id=baseline,
            required_exposures=self.settings.required_exposures,
            cost_weight=self.settings.cost_weight,
            max_updates=self.settings.max_label_updates,
        )

    async def _observe_one(self, window_id, pending, logical_id, assignment, step):
        request = ProbeRequest(
            logical_id,
            window_id,
            pending["record_id"],
            SkillRef(**pending["skill_ref"]),
            pending["with_snapshot_id"],
            pending["without_snapshot_id"],
            self.frozen_version,
            CreditProtocol(**pending["protocol"]),
            step,
        )
        try:
            event = await self._collect_probe(
                request=request,
                sample_fields=assignment,
                store=self.store,
                loading=self.loading,
                gateway_manager=self.frozen_gateway,
                artifact_root=self.artifact_root,
                task_config_path=self.task_config_path,
                frozen_attestation=self.frozen_gateway.skill_route_attestation,
            )
        except ProbeNotObserved as error:
            await self._writes_allowed.wait()
            pending["unobserved"][logical_id] = type(error).__name__
            self.save_state()
            return
        await self._writes_allowed.wait()
        self.ledger.add_usage(event)
        pending["observed"].append(logical_id)
        self.save_state()
        if self.ledger.window_status(window_id)["state"] == "matured":
            self._settle_window(window_id, pending)

    def _settle_window(self, window_id, pending):
        label = self.ledger.settle(window_id)
        pending.update(status="settled", settlement_id=label.settlement_id, utility=label.utility)
        reason = "Delayed marginal utility " + str(label.utility) + " under " + label.protocol_id
        if label.utility > self.settings.admission_threshold:
            self.store.admit(pending["candidate_id"], reason=reason)
            artifact = self.store.load_artifact(SkillRef(**pending["skill_ref"]))
            current = self.store.get_snapshot()
            if artifact.ref not in current.skills:
                staged = self.store.stage_candidate(artifact)
                self.store.admit(staged.candidate_id, reason=reason)
                published = self.store.publish(
                    [staged.candidate_id],
                    expected_parent=staged.base_snapshot,
                    provenance={"window_id": window_id, "settlement_id": label.settlement_id},
                )
                pending["published_snapshot_id"] = published.snapshot_id
            else:
                pending["published_snapshot_id"] = current.snapshot_id
        else:
            self.store.reject(pending["candidate_id"], reason=reason)
        self.save_state()

    async def prepare_update(self, solver_TensorDict, global_step):
        from .replay_batch import (
            finalize_joint_batch,
            infer_position_id_planes,
            prepare_curate_anchor_inputs,
            prepare_solver_only_batch,
        )

        if self._busy:
            raise RuntimeError("Joint controller operations cannot overlap")
        self._busy = True
        try:
            await self._writes_allowed.wait()
            if self._background_pump is not None and self._background_pump.done():
                self._background_pump.result()
                raise RuntimeError("The background skill dispatcher stopped unexpectedly")
            if self.state["prepared_update"] is not None:
                raise ControllerRecoveryRequired(
                    "Finish/reconcile the previous single optimizer update before preparing another"
                )
            clock = await self._optimizer_clock()
            if clock != self.state["optimizer_step"]:
                raise ControllerRecoveryRequired(
                    "Actor/controller optimizer clocks require coherent checkpoint reconciliation"
                )
            fields, sources = self._sample_fields(solver_TensorDict)
            self._enqueue_probe_jobs(fields, global_step)
            update_id = "update_" + _digest([self.protocol_fingerprint, global_step, clock + 1])
            prepared = {
                "update_id": update_id,
                "global_step": global_step,
                "before_optimizer_step": clock,
                "expected_optimizer_step": clock + 1,
                "status": "preparing",
                "acknowledgments": [],
                "sources": sources,
                "curate_jobs": [],
            }
            self.state["prepared_update"] = prepared
            self.save_state()
            ready = self.ledger.sample_ready(len(self.state["pending"]), seed=self.settings.seed + global_step)
            labels = []
            for label in ready:
                record = self.replay.get(label.curate_record_id)
                source_clock = record.context_snapshot["observed_max_weight_version"]
                if not 0 <= clock - source_clock <= self.settings.max_replay_lag_steps:
                    continue
                labels.append(label)
                if len(labels) >= self.settings.replay_batch_size:
                    break
            if self.settings.replay_batch_size == 0:
                labels = []
            for label in labels:
                lease = self.ledger.lease(
                    label.settlement_id,
                    update_id,
                    owner="skill_joint_controller",
                    lease_seconds=self.settings.lease_seconds,
                )
                prepared["acknowledgments"].append(
                    {"settlement_id": label.settlement_id, "update_id": update_id, "lease_token": lease["lease_token"]}
                )
                self.save_state()
            solver_temperature = float(self.config.actor_rollout_ref.rollout.temperature)
            if labels:
                anchor_version = "online:" + str(clock)
                plan = prepare_curate_anchor_inputs(
                    self.replay,
                    self.ledger,
                    [label.settlement_id for label in labels],
                    update_id=update_id,
                    owner="skill_joint_controller",
                    anchor_policy_version=anchor_version,
                    tokenizer_version=self.loading["tokenizer_version"],
                    template_version=self.runtime.template_version,
                    dp_size=self.dp_size,
                    pad_token_id=self.pad_token_id,
                    position_id_planes=infer_position_id_planes(solver_TensorDict),
                )
                anchor = await _resolve(self.actor.compute_log_prob(plan.data))
                if await self._optimizer_clock() != clock:
                    raise ControllerRecoveryRequired("The learner fence was broken during actual anchor inference")
                joint = finalize_joint_batch(
                    solver_TensorDict,
                    plan,
                    anchor,
                    ledger=self.ledger,
                    anchor_policy_version=anchor_version,
                    anchor_input_fingerprint=plan.fingerprint,
                    solver_temperature=solver_temperature,
                )
            else:
                joint = prepare_solver_only_batch(
                    solver_TensorDict,
                    dp_size=self.dp_size,
                    pad_token_id=self.pad_token_id,
                    solver_temperature=solver_temperature,
                )
            prepared.update(
                status="prepared",
                anchor_policy_version=joint.anchor_policy_version,
                anchor_input_fingerprint=joint.anchor_input_fingerprint,
                joint_rows=len(joint.data),
            )
            self.save_state()
            return PreparedJointUpdate(
                joint.data,
                update_id,
                global_step,
                clock,
                clock + 1,
                joint.acknowledgments,
                joint.anchor_policy_version,
                joint.anchor_input_fingerprint,
            )
        finally:
            self._busy = False

    def _register_collection(self, result, job, step):
        window_id = "window_" + result.collection_id
        if window_id in self.state["pending"]:
            return
        baseline = job["skill_snapshot_id"]
        probe_branch = "probe-" + result.collection_id
        if not (self.store.root / "refs" / (probe_branch + ".json")).exists():
            self.store.create_branch(probe_branch, from_snapshot=baseline)
        snapshot = self.store.get_snapshot(branch=probe_branch)
        if result.artifact.ref not in snapshot.skills:
            candidate = self.store.stage_candidate(result.artifact, branch=probe_branch)
            self.store.admit(candidate.candidate_id, reason="Basic integrity for isolated counterfactual snapshot only")
            snapshot = self.store.publish(
                [candidate.candidate_id],
                expected_parent=baseline,
                branch=probe_branch,
                provenance={"evaluation_only": True, "record_id": result.record_id},
            )
        protocol = self._protocol(baseline)
        cohort_id = job["candidate_group_id"]
        cohort = self.state["cohorts"].setdefault(
            cohort_id,
            {
                "probe_ids": [
                    "probe_" + _digest([cohort_id, index]) for index in range(self.settings.required_exposures)
                ],
                "assignments": {},
            },
        )
        opened = job.setdefault("window_opened_at", time.time())
        self.save_state()
        self.ledger.register_window(
            window_id,
            result.record_id,
            protocol,
            opened_at=opened,
            deadline_at=opened + self.settings.window_seconds,
            skill_cost=sum(self.replay.get(result.record_id).curate_mask),
            expected_probe_ids=tuple(cohort["probe_ids"]),
            cohort_id=cohort_id,
        )
        self.state["pending"][window_id] = {
            "window_id": window_id,
            "record_id": result.record_id,
            "candidate_id": result.candidate.candidate_id,
            "skill_ref": result.artifact.ref.to_dict(),
            "opened_step": step,
            "producer_task_id": job["task_id"],
            "task_type": job["task_type"],
            "cohort_id": cohort_id,
            "status": "pending",
            "with_snapshot_id": snapshot.snapshot_id,
            "without_snapshot_id": baseline,
            "protocol": protocol.to_dict(),
            "observed": [],
            "unobserved": {},
        }
        self.save_state()

    def _curate_source_gate(self, source, request):
        """Measure the exact host input before sampling; never truncate a trace."""
        redacted = redact_completed_episode(source)
        template = Path(self.runtime.task_prompt_path).read_text()
        prompt = template.replace("{{SKILL_ID}}", request.skill_id).replace("{{TASK_FAMILY}}", redacted["task_family"])
        prompt = prompt.replace("{{EPISODE_JSON}}", json.dumps(redacted, ensure_ascii=False, indent=2))
        messages = [
            {"role": "system", "content": Path(self.runtime.instructions_path).read_text()},
            {"role": "user", "content": prompt},
        ]
        renderer = TokenizerRenderer(self.trainer.tokenizer, render_kwargs=self.loading["render_kwargs"])
        rendered = renderer.render(messages)
        rollout = self.routes["curator"].config.actor_rollout_ref.rollout
        capacity = int(rollout.prompt_length) + int(rollout.response_length)
        model_capacity = _get(rollout, "max_model_len")
        if model_capacity is not None:
            capacity = min(capacity, int(model_capacity))
        budget = (
            capacity - self.settings.curator_native_prefix_reserve_tokens - self.settings.curator_output_headroom_tokens
        )
        if budget <= 0:
            raise ValueError("Curator route has no source budget after declared native/output reserves")
        return {
            "policy_version": "whole-redacted-source-or-censor-v1",
            "accepted": len(rendered.token_ids) <= budget,
            "host_prompt_tokens": len(rendered.token_ids),
            "host_prompt_budget": budget,
            "gateway_total_capacity": capacity,
            "native_prefix_reserve_tokens": self.settings.curator_native_prefix_reserve_tokens,
            "output_headroom_tokens": self.settings.curator_output_headroom_tokens,
            "source_redacted_sha256": _digest(redacted),
            "host_messages_sha256": _digest(messages),
            "host_token_ids_sha256": _digest(rendered.token_ids),
            "budget_scope": "complete host chat including curator instructions; native additions reserved separately",
            "actual_native_request_verified": False,
            "source_truncated": False,
        }

    @staticmethod
    def _semantic_output_reason(content, request, family):
        if not isinstance(content, str):
            return "invalid_json_content_type"
        try:
            value = json.loads(content)
        except ValueError:
            return "invalid_json"
        if not isinstance(value, dict) or set(value) != SKILL_FIELDS:
            return "invalid_skill_schema"
        if value["skill_id"] != request.skill_id or value["task_families"] != [family]:
            return "invalid_skill_identity_or_family"
        for key in ("title", "principle", "when_to_apply"):
            field = value[key]
            if (
                not isinstance(field, str)
                or not field.strip()
                or len(field) > SKILL_SCHEMA["properties"][key]["maxLength"]
            ):
                return "invalid_skill_text_field"
            if re.search(r"https?://|(?<!\w)/(?:\w+/)+", field):
                return "forbidden_skill_path_or_endpoint"
        return None

    @staticmethod
    def _valid_observed_capture(captured, behavior):
        rows = captured.get("trajectories")
        if not isinstance(rows, list) or not rows:
            return False
        for row in rows:
            for key in ("observed_min_weight_version", "observed_max_weight_version"):
                if type(row.get(key)) is not int or row[key] != behavior.rollout_weight_version:
                    return False
            prompt, actions, mask, probabilities = (
                row.get(key)
                for key in (
                    "prompt_ids",
                    "action_ids",
                    "curate_mask",
                    "behavior_logprobs",
                )
            )
            if not prompt or not actions or probabilities is None or mask is None:
                return False
            if len(actions) != len(mask) or len(actions) != len(probabilities) or not any(mask):
                return False
            if any(type(value) is not int or value < 0 for value in [*prompt, *actions]):
                return False
            for selected, probability in zip(mask, probabilities, strict=True):
                if (
                    type(selected) is not int
                    or selected not in (0, 1)
                    or isinstance(probability, bool)
                    or not isinstance(probability, int | float)
                    or not math.isfinite(probability)
                    or probability > 0
                    or (not selected and probability != 0)
                ):
                    return False
        return True

    def _semantic_curate_rejection(self, error, source, request, behavior):
        """Recognize only proved model-output errors, never catch generic infra."""
        if not isinstance(error, CurateOutputError):
            return None
        directory = self.artifact_root / "curate" / request.collection_id
        capture_path = directory / "generated.json"
        if not capture_path.is_file():
            return None
        captured, intent = _read_state(capture_path), _read_state(directory / "intent.json")
        if (
            captured.get("input_hash") != intent.get("input_hash")
            or intent.get("request") != _plain(asdict(request))
            or intent.get("behavior") != _plain(asdict(behavior))
            or captured.get("session_id") != request.session_id
            or not self._valid_observed_capture(captured, behavior)
        ):
            return None
        try:
            self.replay.get_ref(request.record_id)
        except FileNotFoundError:
            pass
        else:
            return None
        if captured.get("tool_events_seen") is True:
            reason = "native_tool_invocation"
        elif captured.get("agent_ok") is True and captured.get("agent_finished") is True:
            reason = self._semantic_output_reason(captured.get("content"), request, source["task_type"])
        else:
            return None
        if reason is None:
            return None
        return {
            "reason": reason,
            "sampled": True,
            "collector_error_type": type(error).__name__,
            "generated_capture": str(capture_path),
            "generated_capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "old_mu_observed": True,
            "reward_event_created": False,
            "training_label_created": False,
        }

    def _job_audit(self, job, request, disposition):
        payload = {
            "schema_version": 1,
            "collection_id": request.collection_id,
            "candidate_group_id": request.candidate_group_id,
            "candidate_index": request.candidate_index,
            "optimizer_step": self.state["optimizer_step"],
            "status": job["status"],
            **disposition,
        }
        path = self.artifact_root / "controller/curate_jobs" / (request.collection_id + ".json")
        if path.exists() and _read_state(path) != payload:
            raise ControllerRecoveryRequired("A terminal curator job disposition cannot be rewritten")
        _write_state(path, payload)
        job["disposition"] = payload
        self.state.setdefault("curate_job_dispositions", {})[request.collection_id] = payload
        self.save_state()

    def _retained_collection(self, job, request):
        candidate = self.store.get_candidate(job["candidate_id"])
        record, ref = self.replay.get(job["record_id"]), self.replay.get_ref(job["record_id"])
        if record.record_id != request.record_id or dict(record.skill_ref) != candidate.artifact.to_dict():
            raise ControllerRecoveryRequired("Retained cohort identity differs from its durable replay")
        return CurateCollection(
            request.collection_id,
            request.session_id,
            record.record_id,
            candidate,
            self.store.load_artifact(candidate.artifact),
            ref,
            job["output_dir"],
        )

    def _censor_cohort(self, jobs, rejecting_job):
        cohort_id = rejecting_job["candidate_group_id"]
        cohort = self.state["cohorts"].setdefault(cohort_id, {"probe_ids": [], "assignments": {}})
        cohort.update(
            status="censored", reason="incomplete_curate_cohort", rejected_index=rejecting_job["candidate_index"]
        )
        for job in jobs:
            if job is rejecting_job:
                continue
            request = CurateRequest(**{key: job[key] for key in CurateRequest.__dataclass_fields__})
            sampled = "record_id" in job
            if sampled:
                self.store.reject(
                    job["candidate_id"], reason="Administrative cohort censorship; no negative utility was observed"
                )
                window = self.state["pending"].get("window_" + request.collection_id)
                if window is not None:
                    if window.get("observed") or window.get("settlement_id"):
                        raise ControllerRecoveryRequired(
                            "An incomplete cohort already acquired credit; reconcile its history"
                        )
                    window.update(status="censored", reason="incomplete_curate_cohort")
            job["status"] = "censored"
            self._job_audit(
                job,
                request,
                {
                    "reason": "incomplete_curate_cohort",
                    "sampled": sampled,
                    "reward_event_created": False,
                    "training_label_created": False,
                    **(
                        {"retained_record_id": job["record_id"], "retained_candidate_id": job["candidate_id"]}
                        if sampled
                        else {}
                    ),
                },
            )
        self.save_state()

    async def _collect_cohorts(self, current, behavior, step):
        grouped = {}
        for job in current["curate_jobs"]:
            grouped.setdefault(job["candidate_group_id"], []).append(job)
        for cohort_id, jobs in grouped.items():
            if len(jobs) != self.settings.candidates_per_source:
                raise ControllerRecoveryRequired("Incomplete declared candidate group in the durable curate queue")
            rejecting = next((job for job in jobs if job["status"] in {"semantic_rejected", "source_censored"}), None)
            if rejecting is not None:
                self._censor_cohort(jobs, rejecting)
                continue
            for job in jobs:
                if job["status"] in {"retained", "registered", "abstained"}:
                    continue
                if job["status"] != "queued":
                    raise ControllerRecoveryRequired("Unknown curator job lifecycle requires reconciliation")
                request = CurateRequest(**{key: job[key] for key in CurateRequest.__dataclass_fields__})
                try:
                    source = self._load_source(job["episode_output_dir"])
                except CurateSourceCensored as error:
                    job["status"] = "source_censored"
                    self._job_audit(
                        job,
                        request,
                        {
                            "reason": error.reason,
                            "source_evidence": error.evidence,
                            "sampled": False,
                            "reward_event_created": False,
                            "training_label_created": False,
                        },
                    )
                    rejecting = job
                    break
                gate = self._curate_source_gate(source, request)
                job["input_policy"] = gate
                self.save_state()
                if not gate["accepted"]:
                    job["status"] = "source_censored"
                    self._job_audit(
                        job,
                        request,
                        {
                            "reason": "source_input_over_capacity_budget",
                            "sampled": False,
                            "input_policy": gate,
                            "reward_event_created": False,
                            "training_label_created": False,
                        },
                    )
                    rejecting = job
                    break
                try:
                    result = await self._collect_curate(
                        gateway_manager=self.ema_gateway,
                        behavior=behavior,
                        source_episode=source,
                        store=self.store,
                        replay=self.replay,
                        request=request,
                        runtime=self.runtime,
                        output_root=self.artifact_root / "curate",
                        before_commit=self._writes_allowed.wait,
                    )
                except Exception as error:
                    await self._writes_allowed.wait()
                    rejection = self._semantic_curate_rejection(error, source, request, behavior)
                    if rejection is None:
                        raise
                    job["status"] = "semantic_rejected"
                    self._job_audit(job, request, rejection)
                    rejecting = job
                    break
                await self._writes_allowed.wait()
                if isinstance(result, CurateAbstention):
                    job["status"] = "abstained"
                    self._job_audit(job, request, {
                        "reason": result.reason,
                        "sampled": True,
                        "action_record_path": result.action_record_path,
                        "action_record_sha256": result.action_record_sha256,
                        "reward_event_created": False,
                        "training_label_created": False,
                    })
                    continue
                job.update(
                    status="retained",
                    collection_id=result.collection_id,
                    record_id=result.record_id,
                    candidate_id=result.candidate.candidate_id,
                    output_dir=result.output_dir,
                )
                self.save_state()
            if rejecting is not None:
                self._censor_cohort(jobs, rejecting)
                continue
            # All actions are durably retained before any window/probe admission.
            for job in jobs:
                if job["status"] in {"registered", "abstained"}:
                    continue
                request = CurateRequest(**{key: job[key] for key in CurateRequest.__dataclass_fields__})
                result = self._retained_collection(job, request)
                self._register_collection(result, job, step)
                job["status"] = "registered"
                self.state["collections"][result.collection_id] = {
                    "record_id": result.record_id,
                    "window_id": "window_" + result.collection_id,
                    "output_dir": result.output_dir,
                }
                self.save_state()
            cohort = self.state["cohorts"].setdefault(cohort_id, {"probe_ids": [], "assignments": {}})
            cohort["status"] = "abstained" if all(job["status"] == "abstained" for job in jobs) else "registered"
            self.save_state()

    def _complete_post_update(self, current, prepared, optimizer_step):
        jobs = current["curate_jobs"]
        report = {
            "schema_version": 1,
            "update_id": prepared.update_id,
            "optimizer_step": optimizer_step,
            "global_step": prepared.global_step,
            "new_curate_windows": sum(job["status"] == "registered" for job in jobs),
            "curate_semantic_rejections": sum(job["status"] == "semantic_rejected" for job in jobs),
            "curate_censored_jobs": sum(job["status"] in {"censored", "source_censored"} for job in jobs),
            "curate_retained_actions": sum("record_id" in job for job in jobs),
            "curate_censored_cohorts": len(
                {
                    job["candidate_group_id"]
                    for job in jobs
                    if job["status"] in {"censored", "source_censored", "semantic_rejected"}
                }
            ),
        }
        path = self.artifact_root / "controller/updates" / (prepared.update_id + ".json")
        if path.exists() and _read_state(path) != report:
            raise ControllerRecoveryRequired("An optimizer post-update disposition cannot be rewritten")
        _write_state(path, report)
        # Keep live background job/queue references attached to the state.
        final_state = dict(self.state)
        final_state["post_update_reports"] = dict(final_state.get("post_update_reports", {}))
        final_state["post_update_reports"][prepared.update_id] = report
        totals = final_state["curate_metrics"] = dict(final_state.get("curate_metrics", {}))
        for key in (
            "new_curate_windows",
            "curate_semantic_rejections",
            "curate_censored_jobs",
            "curate_retained_actions",
            "curate_censored_cohorts",
        ):
            totals[key] = totals.get(key, 0) + report[key]
        final_state["prepared_update"] = None
        _write_state(self.state_path, final_state)
        self.state = final_state
        return report

    def _initialize_background(self):
        from .source_queue import SourceQueue

        # New collection identities after restore; uncertain sampled actions
        # remain censored and are never relabeled as a new behavior draw.
        self.collection_generation = uuid.uuid4().hex

        self.source_queue = SourceQueue(
            self.state.setdefault("source_queue", {}),
            persist=self.save_state,
            max_backlog=self.settings.max_source_backlog,
        )
        self.state.setdefault("probe_jobs", {})
        # A checkpoint may cut through a generation. Never silently resample an
        # uncertain model action with a new EMA and call it the original draw.
        for key, _entry in self.source_queue.recovery_required():
            self.source_queue.finish(key, "failed", reason="interrupted_background_generation")
        for job in self.state["probe_jobs"].values():
            if job["status"] == "running":
                job.update(status="unobserved", reason="interrupted_background_pair")
                self.state["pending"][job["window_id"]]["unobserved"][job["logical_id"]] = job["reason"]
        self.save_state()
        self._background_pump = asyncio.create_task(self._pump_background(), name="skill-background-pump")

    def _offer_sources(self, sources, step, update_id):
        prepared = [{**source, "update_id": update_id} for source in sources]
        self.source_queue.offer_many(prepared, source_step=step)

    def _enqueue_probe_jobs(self, fields, step):
        """Bind real future B inputs now; generation is performed off the learner path."""
        self.state["latest_train_fields"] = copy.deepcopy(fields)
        self.state["latest_train_step"] = step
        if self.state.get("probe_budget_step") != step:
            self.state.update(probe_budget_step=step, probe_budget_used=0)
        jobs = self.state["probe_jobs"]
        for window_id, pending in list(self.state["pending"].items()):
            if pending["status"] != "pending" or step < pending["opened_step"] + self.settings.min_delay_steps:
                continue
            status = self.ledger.window_status(window_id)
            if status["state"] == "censored":
                pending["status"] = "censored"
                continue
            if status["state"] in {"matured", "settled"}:
                self._settle_window(window_id, pending)
                continue
            cohort = self.state["cohorts"][pending["cohort_id"]]
            for logical_id in cohort["probe_ids"]:
                identity = window_id + ":" + logical_id
                if identity in jobs or logical_id in pending["observed"] or logical_id in pending["unobserved"]:
                    continue
                if self.state["probe_budget_used"] >= self.settings.probes_per_step:
                    self.save_state()
                    return
                assignment = cohort["assignments"].get(logical_id)
                if assignment is None:
                    used = {row["tools_kwargs"]["task"]["metadata"]["game_id"]
                            for row in cohort["assignments"].values()}
                    eligible = [row for row in fields if (
                        row["tools_kwargs"]["task"]["metadata"]["game_id"] != pending["producer_task_id"]
                        and row["tools_kwargs"]["task"]["metadata"]["game_id"] not in used
                        and row["tools_kwargs"]["task"]["metadata"]["task_type"] == pending["task_type"]
                    )]
                    if not eligible:
                        continue
                    assignment = copy.deepcopy(sorted(eligible, key=lambda row: row["uid"])[0])
                    assignment["tools_kwargs"]["task"]["metadata"]["source_step"] = step
                    cohort["assignments"][logical_id] = assignment
                jobs[identity] = {
                    "status": "queued", "window_id": window_id,
                    "logical_id": logical_id, "assignment": copy.deepcopy(assignment), "source_step": step,
                }
                self.state["probe_budget_used"] += 1
        self.save_state()

    async def _run_background_source(self, key, entry):
        await self._writes_allowed.wait()
        behavior = CuratorBehavior(
            "ema:" + str(self.routes["curator"].optimizer_step),
            self.routes["curator"].optimizer_step,
            self.loading["tokenizer_version"], self.runtime.template_version,
        )
        cohort = "cohort_" + _digest([self.collection_generation, entry["update_id"], entry["episode_output_dir"]])
        if not entry["read"] and entry.get("collection_generation") != self.collection_generation:
            entry["curator_jobs"] = []
            entry["collection_generation"] = self.collection_generation
        jobs = entry["curator_jobs"]
        if not jobs:
            for index in range(self.settings.candidates_per_source):
                jobs.append({
                    **{field: entry[field] for field in ("producer_uid", "episode_output_dir", "task_id", "task_type")},
                    "skill_snapshot_id": self.store.head(), "candidate_group_id": cohort,
                    "candidate_index": index, "candidate_group_size": self.settings.candidates_per_source,
                    "created_at": time.time(), "status": "queued",
                })
            entry["behavior"] = asdict(behavior)
            self.save_state()
        # Mark actual source reads at the loader boundary, not merely enqueue.
        original_loader = self._load_source
        # No shared monkeypatch: each source is explicitly checked before the
        # collector loads it again and starts a model request.
        try:
            original_loader(entry["episode_output_dir"])
            self.source_queue.mark_read(key)
            await self._collect_cohorts({"curate_jobs": jobs}, behavior, self.state["optimizer_step"])
            await self._writes_allowed.wait()
            if any(job["status"] == "registered" for job in jobs):
                terminal = "generated"
            elif all(job["status"] == "abstained" for job in jobs):
                terminal = "abstained"
            else:
                terminal = "source_censored"
            if terminal == "generated":
                self.source_queue.finish(key, terminal)
            elif terminal == "abstained":
                self.source_queue.finish(key, terminal, reason="model_abstained")
            else:
                reason = next((job.get("disposition", {}).get("reason") for job in jobs
                               if job.get("disposition", {}).get("reason")), "source_not_admitted")
                self.source_queue.finish(key, terminal, reason=reason)
        except asyncio.CancelledError:
            raise
        except CurateSourceCensored as error:
            await self._writes_allowed.wait()
            self.source_queue.finish(key, "source_censored", reason=error.reason, evidence=error.evidence)
        except Exception as error:
            await self._writes_allowed.wait()
            self.source_queue.finish(key, "failed", reason=type(error).__name__, message=str(error)[:2000])

    async def _run_background_probe(self, job):
        try:
            await self._observe_one(
                job["window_id"], self.state["pending"][job["window_id"]],
                job["logical_id"], job["assignment"], job["source_step"],
            )
            await self._writes_allowed.wait()
            pending = self.state["pending"][job["window_id"]]
            job["status"] = "complete" if job["logical_id"] in pending["observed"] else "unobserved"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._writes_allowed.wait()
            job.update(status="failed", reason=type(error).__name__, message=str(error)[:2000])
            self.state["pending"][job["window_id"]]["unobserved"][job["logical_id"]] = type(error).__name__
        self.save_state()

    async def _pump_background(self):
        while not self._background_stopping:
            await self._writes_allowed.wait()
            for identity, task in list(self._background_tasks.items()):
                if task.done():
                    task.result()  # Bookkeeping failures remain visible, never silently discarded.
                    del self._background_tasks[identity]
            if self._ema_publication_task is not None and self._ema_publication_task.done():
                self._ema_publication_task.result()
                self._ema_publication_task = None
            curator_active = any(key.startswith("curate:") for key in self._background_tasks)
            if (self._ema_refresh_requested and not curator_active
                    and not self._learner_active and not self._ema_syncing):
                self._ema_syncing = True
                self._ema_publication_task = asyncio.create_task(self._publish_idle_ema())
            if not self._ema_syncing and not self._ema_refresh_requested:
                active = sum(identity.startswith("curate:") for identity in self._background_tasks)
                pending = sum(row["status"] == "pending" for row in self.state["pending"].values())
                capacity = min(self.settings.curator_concurrency - active,
                               max(0, self.settings.max_pending_windows - pending - active))
                for key, entry in self.source_queue.claim_ready(capacity):
                    self._background_tasks["curate:" + key] = asyncio.create_task(
                        self._run_background_source(key, entry), name=key,
                    )
            active = sum(identity.startswith("probe:") for identity in self._background_tasks)
            for identity, job in self.state["probe_jobs"].items():
                if active >= self.settings.credit_concurrency:
                    break
                if job["status"] == "queued":
                    job["status"] = "running"
                    self._background_tasks["probe:" + identity] = asyncio.create_task(
                        self._run_background_probe(job), name="probe:" + identity,
                    )
                    active += 1
                    self.save_state()
            await asyncio.sleep(0.1)

    def background_metrics(self):
        metrics = {"source_" + key.removeprefix("sources_"): value
                   for key, value in self.source_queue.metrics().items()}
        for status in ("queued", "running", "complete", "unobserved", "failed"):
            metrics["probe_" + status] = sum(job["status"] == status for job in self.state["probe_jobs"].values())
        metrics["source_sampled"] = sum(
            any("record_id" in job or job.get("disposition", {}).get("sampled") is True
                for job in entry["curator_jobs"])
            for entry in self.source_queue.state["jobs"].values()
        )
        metrics["curator_weight_version"] = self.routes["curator"].optimizer_step
        return metrics

    async def pause_background_writes(self):
        """Take a consistent storage cut without waiting for model generations."""
        if self._busy:
            raise ControllerRecoveryRequired("Checkpoint cannot cut through an optimizer transaction")
        was_active = self._learner_active
        await self.enter_learner()
        self._writes_allowed.clear()
        return was_active

    async def resume_background_writes(self, learner_was_active=True):
        self._writes_allowed.set()
        self._learner_active = learner_was_active

    async def enter_learner(self):
        """Fence online RPCs against a short background EMA tensor transfer."""
        self._learner_active = True
        if self._ema_publication_task is not None:
            await self._ema_publication_task

    async def leave_learner(self):
        self._learner_active = False

    async def _publish_idle_ema(self):
        try:
            await self.routes["curator"].sync_ema(self.state["optimizer_step"])
            await self._refresh_ema_gateway()
            self._ema_refresh_requested = False
        finally:
            self._ema_syncing = False

    async def stop_background(self):
        self._background_stopping = True
        self._writes_allowed.set()
        failures = []
        if self._background_pump is not None:
            failures.extend(await asyncio.gather(self._background_pump, return_exceptions=True))
        tasks = list(self._background_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            failures.extend(await asyncio.gather(*tasks, return_exceptions=True))
        if self._ema_publication_task is not None:
            failures.extend(await asyncio.gather(self._ema_publication_task, return_exceptions=True))
        self.save_state()
        for error in failures:
            if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
                raise error

    async def after_update_and_sync(self, solver_data, step, *, prepared, optimizer_step):
        if self._busy:
            raise RuntimeError("Joint controller operations cannot overlap")
        self._busy = True
        try:
            current = self.state["prepared_update"]
            if current is None:
                if prepared.update_id in self.state["applied_update_ids"]:
                    report = self.state.get("post_update_reports", {}).get(prepared.update_id, {})
                    return {**report, "update_id": prepared.update_id, "already_completed": True}
                raise ControllerRecoveryRequired("No prepared update owns this optimizer acknowledgement")
            if (
                current["update_id"] != prepared.update_id
                or current["global_step"] != step
                or optimizer_step != current["expected_optimizer_step"]
            ):
                raise ControllerRecoveryRequired(
                    "Optimizer acknowledgement differs from the prepared single-update identity"
                )
            actual = await self._optimizer_clock()
            if actual != optimizer_step:
                raise ControllerRecoveryRequired("Actor/EMA did not observe the claimed optimizer step")
            model_update_id = "optimizer:" + str(actual) + ":" + prepared.update_id
            for ack in current["acknowledgments"]:
                self.ledger.mark_applied(**ack, model_update_id=model_update_id)
            self.state["optimizer_step"] = actual
            self.state["applied_update_ids"][prepared.update_id] = model_update_id
            self.state["last_global_step"] = step
            current["status"] = "optimizer_applied"
            self.save_state()
            # Publication is a short learner boundary operation and is
            # skipped while an already pinned curator generation is live.
            # Background workers never issue online optimizer/model RPCs.
            self._ema_refresh_requested = True
            if not any(key.startswith("curate:") for key in self._background_tasks):
                self._ema_syncing = True
                try:
                    await self.routes["curator"].sync_ema(actual)
                    await self._refresh_ema_gateway()
                    self._ema_refresh_requested = False
                finally:
                    self._ema_syncing = False
            self._offer_sources(current["sources"], step, prepared.update_id)
            report = self._complete_post_update(current, prepared, actual)
            return {**report, **self.background_metrics()}
        finally:
            self._busy = False
