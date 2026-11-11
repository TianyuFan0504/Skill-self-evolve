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
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import pin_prompt_group
from .credit import CreditProtocol, EvaluationOutcome, UsageEvent
from .replay import _identifier, _plain, _publish_immutable, _read_payload
from .store import SkillRef, SkillStore


class ProbeNotObserved(RuntimeError):
    """No valid paired exposure was observed; this is not a zero utility label."""


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
    from .alfworld.runner import run_task

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = await gateway_manager.create_session(
        session_id,
        metadata={"role": "counterfactual_probe", "_log_dir": str(artifact_dir)},
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


def _verify_branch(capture, *, request, sample_fields, loaded, artifact_root):
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
    attestation = capture.get("policy_attestation", {})
    if (
        attestation.get("policy_kind") != "frozen_evaluator"
        or attestation.get("frozen_model_version") != request.evaluator_version
        or not attestation.get("lifecycle_id")
    ):
        raise ValueError("Probe lacks the frozen backend route's model/lifecycle attestation")
    output = Path(info["episode_output_dir"]).resolve()
    if not output.is_relative_to(Path(artifact_root).resolve()):
        raise ValueError("Probe episode artifacts escaped this experiment")
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
    if (
        frozen_attestation.get("policy_kind") != "frozen_evaluator"
        or frozen_attestation.get("frozen_model_version") != request.evaluator_version
    ):
        raise ValueError("A paired probe requires the actual immutable evaluator route attestation")
    if metadata.get("source_step", request.global_step) > request.global_step:
        raise ValueError("Probe source task belongs to a future training step")
    validate_pair_snapshots(store, request)
    root = Path(artifact_root) / "credit_probes" / request.probe_id
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    intent = {
        "schema_version": 1, "probe_id": request.probe_id, "window_id": request.window_id,
        "curate_record_id": request.curate_record_id, "skill_ref": request.skill_ref.to_dict(),
        "with_snapshot_id": request.with_snapshot_id, "without_snapshot_id": request.without_snapshot_id,
        "evaluator_version": request.evaluator_version, "protocol_id": request.protocol.protocol_id,
        "global_step": request.global_step, "task_id": task_id,
        "task_game_sha256": metadata["game_sha256"],
    }
    # The lock is acquired nonblocking: a duplicated orchestrator must not hang
    # its event loop behind another in-flight copy of the same native session.
    import fcntl

    with (root / ".probe.lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _publish_immutable(root / "intent.json", intent)
        outcomes = {}
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
                fields["uid"] = request.probe_id + "_" + name
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
                loaded=loaded, artifact_root=artifact_root,
            )
        completed = root / "paired-event.json"
        previous = _read_payload(completed) if completed.exists() else None
        event = UsageEvent(
            event_id="event_" + request.probe_id,
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
