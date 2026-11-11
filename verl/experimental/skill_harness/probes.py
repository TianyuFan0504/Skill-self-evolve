# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Paired native-Codex credit probes over original, downstream train tasks.

These sessions never enter the solver GRPO buffer. Their environment outcomes
become versioned marginal-utility labels for an already archived curate action.
All generated token trajectories and failed branches remain experiment artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import pin_prompt_group
from .credit import CreditProtocol, EvaluationOutcome, UsageEvent
from .replay import _identifier, _plain, _publish_immutable, _read_payload
from .store import SkillRef, SkillStore


class ProbeNotObserved(RuntimeError):
    """No valid paired exposure was observed; this is not a zero utility label."""


def validate_frozen_attestation(value, *, evaluator_version=None):
    """Validate immutable route provenance without fabricating a weight clock."""
    if not isinstance(value, Mapping):
        raise ValueError("A frozen evaluator route attestation is required")
    result = copy.deepcopy(dict(value))
    fingerprint = result.get("model_fingerprint")
    if (
        type(result.get("schema_version")) is not int or result["schema_version"] != 1
        or result.get("policy_kind") != "frozen_evaluator"
        or not isinstance(fingerprint, str) or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or result.get("frozen_model_version") != fingerprint
        or not isinstance(result.get("model_path"), str) or not Path(result["model_path"]).is_absolute()
    ):
        raise ValueError(
            "Frozen provenance requires schema1, a canonical model fingerprint, and an absolute model path"
        )
    _identifier(result.get("lifecycle_id"), "frozen lifecycle_id")
    if evaluator_version is not None and result["frozen_model_version"] != evaluator_version:
        raise ValueError("Frozen evaluator fingerprint differs from the registered credit protocol")
    return result


def _gateway_attestation(manager, supplied):
    expected = validate_frozen_attestation(supplied)
    actual = validate_frozen_attestation(getattr(manager, "skill_route_attestation", None))
    if actual != expected:
        raise ValueError("Probe attestation differs from the frozen client bound to its Gateway manager")
    return expected


async def create_frozen_probe_gateway(config, route, *, gateway_factory=None):
    """Bind the official native Gateway to this route's validated frozen client.

    Pass the dedicated route configuration, so tokenizer and model identity
    match its pinned checkpoint. The sampling allowlist is disabled on a clone:
    native agent requests cannot override the paired session's greedy settings.
    """
    attestation = validate_frozen_attestation(route.attestation)
    if route.frozen_version != attestation["frozen_model_version"]:
        raise ValueError("The route does not expose the attested frozen model version")
    client = route.get_client()
    if (
        getattr(client, "frozen_version", None) != route.frozen_version
        or validate_frozen_attestation(getattr(client, "attestation", None)) != attestation
    ):
        raise ValueError("Frozen route and generated client do not share one immutable server lifecycle")
    gateway_config = copy.deepcopy(config)
    path = "actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys"
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(gateway_config):
        OmegaConf.update(gateway_config, path, [], force_add=True)
    else:
        target = gateway_config
        for key in path.split(".")[:-1]:
            target = target.setdefault(key, {})
        target[path.split(".")[-1]] = []
    if gateway_factory is None:
        from uni_agent.framework.entry import build_gateway_manager

        gateway_factory = build_gateway_manager
    manager = gateway_factory(config=gateway_config, llm_client=client)
    if inspect.isawaitable(manager):
        manager = await manager
    manager.skill_route_attestation = copy.deepcopy(attestation)
    return manager


def probe_physical_id(request):
    """Separate physical sessions/artifacts while retaining a cohort probe ID."""
    identity = json.dumps([request.window_id, request.probe_id], separators=(",", ":"))
    return "probe_" + hashlib.sha256(identity.encode()).hexdigest()


def probe_group_id(request):
    return probe_physical_id(request).removeprefix("probe_")


def _verify_paired_workdir(capture, request, *, loaded, output):
    info = capture.get("extra_info", {})
    manifest = info.get("paired_workdir", {})
    group = probe_group_id(request)
    variant = "with_skill" if loaded else "without_skill"
    workdir = str(Path("/tmp/alfworld-skill-episodes") / ("credit_" + group))
    if (
        type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
        or manifest.get("kind") != "paired_empty_workdir"
        or manifest.get("credit_probe_group") != group or manifest.get("variant") != variant
        or manifest.get("workdir") != workdir or manifest.get("empty_before_run") is not True
        or manifest.get("cleanup_completed") is not True
        or info.get("role") != "counterfactual_probe" or info.get("credit_probe_group") != group
        or info.get("credit_probe_variant") != variant
        or any(
            type(manifest.get(key)) is not int or manifest[key] < 0 for key in ("host_uid", "agent_uid", "agent_gid")
        )
    ):
        raise ProbeNotObserved("Native branch lacks controlled empty-workdir and completed-cleanup provenance")
    homes = manifest.get("codex_homes", [])
    if not isinstance(homes, list) or len(homes) != 1 or not isinstance(homes[0], str):
        raise ProbeNotObserved("Native paired branch lacks its separate per-episode Codex home")
    if Path(homes[0]).parent != Path("/tmp") or not re.fullmatch(r"codex-home-[0-9a-f]{32}", Path(homes[0]).name):
        raise ProbeNotObserved("The paired branch has an invalid native Codex home namespace")
    archives = manifest.get("archives", [])
    if len(archives) != 2 or {row.get("source") for row in archives} != {workdir, homes[0]}:
        raise ProbeNotObserved("Paired workspace and native home were not both archived")
    for row in archives:
        path = Path(row["path"]).resolve()
        if not path.is_relative_to(output) or row.get("removed") is not True:
            raise ProbeNotObserved("Paired branch archive escaped its attempt or cleanup is incomplete")
        hasher = hashlib.sha256()
        if not path.is_file():
            raise ValueError("A retained paired archive is missing")
        with path.open("rb") as stream:
            while chunk := stream.read(1 << 20):
                hasher.update(chunk)
        if hasher.hexdigest() != row.get("sha256"):
            raise ValueError("The retained paired workspace archive differs from its host manifest")
    return manifest


def _observed_frozen_clock(capture):
    clocks = []
    for row in capture.get("trajectories", []):
        extra = row.get("extra_fields", {})
        lower, upper = extra.get("min_global_steps"), extra.get("max_global_steps")
        if lower is None and upper is None:
            clocks.append(None)
        elif type(lower) is not int or type(upper) is not int or lower < 0 or lower != upper:
            raise ValueError("A frozen probe trajectory contains a mixed or invalid observed weight clock")
        else:
            clocks.append(lower)
    if clocks and any(clock != clocks[0] for clock in clocks):
        raise ValueError("A frozen probe changed its observed weight clock across native trajectories")
    return clocks[0] if clocks else None


@dataclass(frozen=True)
class ProbeRequest:
    probe_id: str
    window_id: str
    curate_record_id: str
    skill_ref: SkillRef
    with_snapshot_id: str
    without_snapshot_id: str
    evaluator_version: str
    protocol: CreditProtocol
    global_step: int

    def __post_init__(self):
        for name in ("probe_id", "window_id", "curate_record_id", "evaluator_version"):
            _identifier(getattr(self, name), name)
        if self.evaluator_version != self.protocol.evaluator_version:
            raise ValueError("Probe evaluator differs from its frozen credit protocol")
        if self.without_snapshot_id != self.protocol.baseline_snapshot_id:
            raise ValueError("Probe baseline differs from its registered credit window")
        if type(self.global_step) is not int or self.global_step < 0:
            raise ValueError("A probe requires its downstream training-step identity")


def validate_pair_snapshots(store: SkillStore, request: ProbeRequest):
    baseline = store.get_snapshot(request.without_snapshot_id)
    treated = store.get_snapshot(request.with_snapshot_id)
    before = {ref.skill_id: ref.to_dict() for ref in baseline.skills}
    after = {ref.skill_id: ref.to_dict() for ref in treated.skills}
    ref = request.skill_ref.to_dict()
    if ref["skill_id"] in before or after != {**before, ref["skill_id"]: ref}:
        raise ValueError("Paired libraries must differ only by the exact target skill version")


def _task_identity(sample_fields):
    task = sample_fields.get("tools_kwargs", {}).get("task", {})
    metadata = task.get("metadata", {})
    if (
        task.get("name") != "alfworld_skill_training"
        or metadata.get("split") != "train"
        or metadata.get("dataset_phase") != "train"
    ):
        raise ValueError("Credit probes may only use original optimization-train tasks")
    return str(metadata["stable_game_id"]), metadata


def _trajectory_payload(trajectory):
    return {
        name: _plain(getattr(trajectory, name))
        for name in (
            "prompt_ids", "response_ids", "response_mask", "response_logprobs",
            "num_turns", "finished", "extra_fields",
        )
    }


async def native_probe_branch(
    *, gateway_manager, session_id, sample_fields, artifact_dir, task_config_path,
    frozen_attestation,
    model_name="Qwen3.5-4B-nothink",
):
    """Run the unchanged official Task runner and persist exact Gateway tokens."""
    frozen_attestation = _gateway_attestation(gateway_manager, frozen_attestation)
    from .alfworld.runner import run_task

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = await gateway_manager.create_session(
        session_id,
        metadata={
            "role": "counterfactual_probe", "_log_dir": str(artifact_dir),
            **{key: sample_fields["tools_kwargs"]["task"]["metadata"][key]
               for key in ("credit_probe_group", "credit_probe_variant")},
        },
        sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": -1},
    )
    finalized = False
    try:
        result = await run_task(
            session=session,
            tools_kwargs=sample_fields["tools_kwargs"],
            raw_prompt=sample_fields["raw_prompt"],
            sample_index=0,
            task_config_path=task_config_path,
            model_name=model_name,
        )
        trajectories = await gateway_manager.finalize_session(session_id)
        finalized = True
        _gateway_attestation(gateway_manager, frozen_attestation)
        capture = {
            "schema_version": 1,
            "session_id": session_id,
            "reward": float(result.reward),
            "accuracy": float(result.accuracy),
            "finished": bool(result.finished),
            "policy_attestation": _plain(frozen_attestation),
            "extra_info": _plain(result.extra_info),
            "trajectories": [_trajectory_payload(row) for row in trajectories],
        }
        _publish_immutable(artifact_dir / "native-capture.json", capture)
        return capture
    finally:
        if not finalized:
            await gateway_manager.abort_session(session_id)


def _verify_branch(capture, *, request, sample_fields, loaded, artifact_root, frozen_attestation):
    task_id, metadata = _task_identity(sample_fields)
    if capture.get("finished") is not True or capture.get("reward") not in (0.0, 1.0):
        raise ProbeNotObserved("Probe did not finish with an authoritative binary environment result")
    info = capture.get("extra_info", {})
    if info.get("error") or info.get("eval_completed") is not True:
        raise ProbeNotObserved("A failed native agent branch cannot become a counterfactual label")
    if capture.get("accuracy") != capture["reward"]:
        raise ValueError("Probe reward and authoritative engine accuracy differ")
    context = info.get("skill_context", {})
    expected_snapshot = request.with_snapshot_id if loaded else request.without_snapshot_id
    if context.get("snapshot_id") != expected_snapshot:
        raise ValueError("Probe used a different library snapshot")
    injected = context.get("injected_refs", [])
    contains = request.skill_ref.to_dict() in injected
    if contains != loaded:
        raise ProbeNotObserved("The target skill was not actually injected as assigned")
    rows = capture.get("trajectories", [])
    if not rows:
        raise ProbeNotObserved("Native Gateway returned no token trajectory")
    attestation = validate_frozen_attestation(
        capture.get("policy_attestation"), evaluator_version=request.evaluator_version
    )
    if attestation != frozen_attestation:
        raise ValueError("Native capture belongs to another frozen model or server lifecycle")
    _observed_frozen_clock(capture)
    output = Path(info["episode_output_dir"]).resolve()
    if not output.is_relative_to(Path(artifact_root).resolve()):
        raise ValueError("Probe episode artifacts escaped this experiment")
    _verify_paired_workdir(capture, request, loaded=loaded, output=output)
    engine = json.loads((output / "engine-result.json").read_text())
    if (
        engine["game_sha256"] != metadata["game_sha256"]
        or engine["split"] != "train"
        or bool(engine["won"]) != bool(capture["reward"])
    ):
        raise ValueError("Probe outcome differs from the private environment result")
    initial = json.loads((output / "initial-state.json").read_text())
    state_hash = hashlib.sha256(
        json.dumps(initial, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return EvaluationOutcome(
        task_id=task_id,
        model_version=request.evaluator_version,
        environment_version=request.protocol.environment_version,
        environment_state_hash=state_hash,
        baseline_snapshot_id=request.without_snapshot_id,
        score=float(capture["reward"]),
        trajectory_ref=str(output),
        loaded_skill_ref=request.skill_ref if loaded else None,
    )


async def collect_counterfactual_probe(
    *, request: ProbeRequest, sample_fields: dict[str, Any], store: SkillStore,
    loading: dict[str, Any], gateway_manager, artifact_root, task_config_path,
    frozen_attestation,
    run_branch=native_probe_branch,
):
    """Return one paired event; the ledger performs durable causal settlement.

    A deterministic journal prevents an uncertain retry from silently drawing a
    second pair. Completed captures can be revalidated after a caller crashes.
    Branch assignment is alternated by a stable hash to avoid fixed-order drift.
    """
    task_id, metadata = _task_identity(sample_fields)
    frozen_attestation = validate_frozen_attestation(frozen_attestation, evaluator_version=request.evaluator_version)
    _gateway_attestation(gateway_manager, frozen_attestation)
    if metadata.get("source_step", request.global_step) > request.global_step:
        raise ValueError("Probe source task belongs to a future training step")
    validate_pair_snapshots(store, request)
    physical_id = probe_physical_id(request)
    root = Path(artifact_root) / "credit_probes" / physical_id
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    intent = {
        "schema_version": 1, "probe_id": request.probe_id, "window_id": request.window_id,
        "curate_record_id": request.curate_record_id, "skill_ref": request.skill_ref.to_dict(),
        "with_snapshot_id": request.with_snapshot_id, "without_snapshot_id": request.without_snapshot_id,
        "evaluator_version": request.evaluator_version, "protocol_id": request.protocol.protocol_id,
        "global_step": request.global_step, "task_id": task_id,
        "task_game_sha256": metadata["game_sha256"],
        "physical_probe_id": physical_id, "policy_attestation": frozen_attestation,
        "paired_workdir_protocol": "paired_empty_workdir_v1", "credit_probe_group": probe_group_id(request),
    }
    # The lock is acquired nonblocking: a duplicated orchestrator must not hang
    # its event loop behind another in-flight copy of the same native session.
    import fcntl

    with (root / ".probe.lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent_path = root / "intent.json"
        pair_attestation = frozen_attestation
        if intent_path.exists():
            previous_intent = _read_payload(intent_path)
            previous_attestation = validate_frozen_attestation(
                previous_intent.get("policy_attestation"), evaluator_version=request.evaluator_version
            )
            if previous_attestation != frozen_attestation:
                complete = all(
                    (root / name / "native-capture.json").is_file() for name in ("with_skill", "without_skill")
                )
                if not complete:
                    raise ProbeNotObserved("A partial old-lifecycle pair requires explicit reconciliation")
                pair_attestation = previous_attestation
                intent["policy_attestation"] = previous_attestation
        _publish_immutable(intent_path, intent)
        outcomes = {}
        observed_clocks = {}
        workspace_provenance = {}
        order = [True, False]
        if int(hashlib.sha256(request.probe_id.encode()).hexdigest()[:2], 16) % 2:
            order.reverse()
        for loaded in order:
            name = "with_skill" if loaded else "without_skill"
            branch_root = root / name
            branch_root.mkdir(exist_ok=True, mode=0o700)
            captured = branch_root / "native-capture.json"
            if captured.exists():
                capture = _read_payload(captured)
            else:
                started = branch_root / "started.json"
                if started.exists():
                    raise ProbeNotObserved("An uncaptured native branch requires explicit reconciliation")
                fields = copy.deepcopy(sample_fields)
                fields["uid"] = physical_id + "_" + name
                fields["tools_kwargs"]["task"]["metadata"].update(
                    role="counterfactual_probe", credit_probe_group=probe_group_id(request), credit_probe_variant=name,
                )
                settings = dict(loading, enabled=True)
                settings["snapshot_id"] = request.with_snapshot_id if loaded else request.without_snapshot_id
                fields = pin_prompt_group(
                    fields, loading=settings, store_root=store.root,
                    global_step=request.global_step, partition_id="credit",
                    pin_directory=Path(artifact_root) / "context_groups",
                )
                _publish_immutable(started, {"started_at": time.time(), "session_id": fields["uid"]})
                capture = await run_branch(
                    gateway_manager=gateway_manager,
                    session_id=fields["uid"], sample_fields=fields, artifact_dir=branch_root,
                    task_config_path=task_config_path, frozen_attestation=frozen_attestation,
                )
                _publish_immutable(captured, capture)
            outcomes[name] = _verify_branch(
                capture, request=request, sample_fields=sample_fields,
                loaded=loaded, artifact_root=artifact_root, frozen_attestation=pair_attestation,
            )
            observed_clocks[name] = _observed_frozen_clock(capture)
            workspace_provenance[name] = capture["extra_info"]["paired_workdir"]
        if observed_clocks["with_skill"] != observed_clocks["without_skill"]:
            raise ValueError("Paired branches changed the frozen evaluator's observed weight clock")
        with_workdir, without_workdir = (workspace_provenance[name] for name in ("with_skill", "without_skill"))
        if (
            with_workdir["workdir"] != without_workdir["workdir"]
            or with_workdir["codex_homes"] == without_workdir["codex_homes"]
            or any(with_workdir[key] != without_workdir[key] for key in ("host_uid", "agent_uid", "agent_gid"))
        ):
            raise ValueError("Counterfactual branches must share an empty cwd and use distinct native Codex homes")
        _publish_immutable(root / "paired-workdirs.json", {
            "schema_version": 1, "credit_probe_group": probe_group_id(request),
            "physical_probe_id": physical_id, "branches": workspace_provenance,
        })
        completed = root / "paired-event.json"
        previous = _read_payload(completed) if completed.exists() else None
        event = UsageEvent(
            event_id="event_" + physical_id,
            window_id=request.window_id, curate_record_id=request.curate_record_id,
            skill_ref=request.skill_ref, probe_id=request.probe_id,
            protocol_id=request.protocol.protocol_id,
            with_skill=outcomes["with_skill"], without_skill=outcomes["without_skill"],
            occurred_at=previous["occurred_at"] if previous else time.time(),
        )
        if event.with_skill.environment_state_hash != event.without_skill.environment_state_hash:
            raise ProbeNotObserved("Paired native branches started from different environment states")
        _publish_immutable(root / "paired-event.json", event.to_dict())
        return event
