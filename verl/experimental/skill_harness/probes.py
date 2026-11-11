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

from .alfworld.terminal_guard import _validated_state
from .context import CONTEXT_BUILDER_VERSION, _local_tokenizer, _skill_block, _user_index, pin_prompt_group
from .credit import CreditProtocol, EvaluationOutcome, UsageEvent
from .replay import _identifier, _plain, _publish_immutable, _read_payload
from .store import SkillRef, SkillStore, SkillStoreError


class ProbeNotObserved(RuntimeError):
    """No valid paired exposure was observed; this is not a zero utility label."""


PAIRED_INPUT_VERIFIER_VERSION = "exact-native-skill-only-v1"


def _input_hash(value):
    serialized = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def verify_pair_inputs(captures, *, request, store, loading, decode_prompt_ids=None):
    """Attest the actual initial native inputs before a credit event can exist.

    Reconstruct each Task's complete injected block from immutable artifacts,
    prove that retained refs/order and base messages differ only by target S,
    then compare decoded, actually sampled first-request prompts. The sole
    allowed replacement is the already-proved complete Task user string. No
    HOME/path/time normalization or token/probability rewriting is performed.
    ``decode_prompt_ids`` is an explicit CPU test boundary; production resolves
    a cached, local-only tokenizer using the recorded loading configuration.
    """
    proof = {
        "schema_version": 1,
        "kind": "paired_native_input_audit",
        "verifier_version": PAIRED_INPUT_VERIFIER_VERSION,
        "context_builder_version": CONTEXT_BUILDER_VERSION,
        "window_id": request.window_id,
        "probe_id": request.probe_id,
        "skill_ref": request.skill_ref.to_dict(),
        "status": "unobserved",
        "branches": {},
    }

    def reject(reason):
        error = ProbeNotObserved(reason)
        error.input_audit = {**proof, "reason": reason}
        raise error

    messages, user_indices, refs, blocks, ranges, bases, native = {}, {}, {}, {}, {}, {}, {}
    target = request.skill_ref.to_dict()
    for name, snapshot_id in (
        ("with_skill", request.with_snapshot_id),
        ("without_skill", request.without_snapshot_id),
    ):
        capture = captures.get(name)
        if not isinstance(capture, dict):
            reject("Paired actual input lacks one durable native capture")
        info = capture.get("extra_info", {})
        if not isinstance(info, dict):
            reject("Paired actual input lacks protected Task metadata")
        context = info.get("skill_context", {})
        if not isinstance(context, dict) or not isinstance(info.get("episode_output_dir"), str):
            reject("Paired actual input lacks its Task context or raw message path")
        if (
            context.get("enabled") is not True
            or context.get("builder_version") != CONTEXT_BUILDER_VERSION
            or context.get("snapshot_id") != snapshot_id
        ):
            reject("Paired Task context does not use the current exact-input builder and snapshot")
        path = Path(info.get("episode_output_dir", "/missing-task")) / "agent-messages.json"
        try:
            task_messages = json.loads(path.read_text())
            if not isinstance(task_messages, list) or any(not isinstance(message, dict) for message in task_messages):
                raise ValueError("Invalid native Task message structure")
            user_index = _user_index(task_messages)
            injected = context["injected_refs"]
            if not isinstance(injected, list) or any(not isinstance(ref, dict) for ref in injected):
                raise ValueError("Invalid injected refs")
            if len({_input_hash(ref) for ref in injected}) != len(injected):
                raise ValueError("Duplicate injected ref")
            view = store.read_only_view(snapshot_id)
            artifacts = [view.load_artifact(SkillRef(**ref)) for ref in injected]
            block, spans = _skill_block(artifacts)
            user = task_messages[user_index]["content"]
            if block and not user.endswith(block):
                reject("Actual Task messages do not contain the exact immutable injected skill block")
            original = copy.deepcopy(task_messages)
            original[user_index]["content"] = user[: -len(block)] if block else user
            if (
                context.get("skill_text_sha256") != hashlib.sha256(block.encode()).hexdigest()
                or context.get("original_messages_sha256") != _input_hash(original)
                or context.get("injected_messages_sha256") != _input_hash(task_messages)
            ):
                reject("Raw Task messages differ from the recorded skill context hashes")
        except ProbeNotObserved:
            raise
        except (OSError, ValueError, TypeError, KeyError, SkillStoreError) as error:
            proof["failure_type"] = type(error).__name__
            reject("Paired Task messages or immutable injected artifacts are unobserved")
        messages[name], user_indices[name], refs[name], blocks[name], bases[name] = (
            task_messages,
            user_index,
            injected,
            block,
            original,
        )
        ranges[name] = spans
        proof["branches"][name] = {
            "task_messages_path": str(path),
            "task_messages_sha256": _input_hash(task_messages),
            "base_messages_sha256": _input_hash(original),
            "injected_refs": injected,
            "skill_block_sha256": hashlib.sha256(block.encode()).hexdigest(),
        }
    treated, control = refs["with_skill"], refs["without_skill"]
    if treated.count(target) != 1 or target in control or [ref for ref in treated if ref != target] != control:
        reject("Paired loading displaced, reordered, or changed a non-target skill")
    if bases["with_skill"] != bases["without_skill"]:
        reject("Actual Task messages changed beyond the target skill and its required wrapper")
    proof["retained_refs_and_order_preserved"] = True
    proof["same_original_task_messages"] = True

    if decode_prompt_ids is None:
        try:
            tokenizer = _local_tokenizer(loading["tokenizer_path"], loading.get("tokenizer_revision"))
        except (ImportError, OSError, TypeError, KeyError, ValueError) as error:
            proof["failure_type"] = type(error).__name__
            reject("The recorded local tokenizer for actual native prompts is unavailable")

        def decode_prompt_ids(ids):
            return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    for name in ("with_skill", "without_skill"):
        rows = captures[name].get("trajectories", [])
        ids = rows[0].get("prompt_ids") if rows and isinstance(rows[0], dict) else None
        if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
            reject("An actual first native prompt has no valid numeric token IDs")
        try:
            decoded = decode_prompt_ids(list(ids))
        except Exception as error:
            proof["failure_type"] = type(error).__name__
            reject("An actual first native prompt cannot be decoded with its recorded tokenizer")
        if not isinstance(decoded, str) or not decoded:
            reject("The actual first native prompt decoder returned no text")
        user = messages[name][user_indices[name]]["content"]
        if not user or decoded.count(user) != 1:
            reject("The verified Task user string is not unique in the actual native prompt")
        native[name] = decoded
        proof["branches"][name].update(
            first_prompt_ids_sha256=_input_hash(ids),
            first_prompt_tokens=len(ids),
            decoded_native_prompt_sha256=hashlib.sha256(decoded.encode()).hexdigest(),
        )
    left, right = ranges["with_skill"][treated.index(target)]["block_char_range"]
    target_text = blocks["with_skill"][left:right]
    if target_text in native["without_skill"]:
        reject("The target skill is already present in the actual control prompt")
    with_user = messages["with_skill"][user_indices["with_skill"]]["content"]
    without_user = messages["without_skill"][user_indices["without_skill"]]["content"]
    substituted = native["with_skill"].replace(with_user, without_user, 1)
    if substituted.encode() != native["without_skill"].encode():
        reject("Actual native prefixes differ beyond the verified target skill Task string")
    proof.update(status="observed", actual_native_inputs_differ_only_by_target_skill=True)
    return proof


def _backend_drained(capture):
    info = capture.get("extra_info", {})
    proof, bound = info.get("backend_terminal", {}), info.get("backend_binding", {})
    return bool(
        isinstance(proof, dict)
        and isinstance(bound, dict)
        and info.get("backend_stop_confirmed") is True
        and proof.get("schema_version") == 1
        and proof.get("backend_stop_confirmed") is True
        and proof.get("active_requests") == 0
        and (
            proof.get("fenced") is True
            or (
                proof.get("status") == "drained"
                and info.get("agent_ok") is True
                and not info.get("error")
                and info.get("expected_host_terminal_stop") is not True
                and info.get("expected_backend_terminal_stop") is not True
            )
        )
        and proof.get("status") in ("drained", "terminal_refused", "terminal_aborted")
        and proof.get("attempt_id") == info.get("attempt_id")
        and proof.get("gateway_session_id") == capture.get("session_id")
        and proof.get("environment_session_id") == info.get("environment_session_id")
        and isinstance(proof.get("binding_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", proof["binding_sha256"])
        and all(
            bound.get(key) is not None and proof.get(key) == bound[key]
            for key in ("binding_id", "binding_sha256", "attempt_id", "gateway_session_id", "environment_session_id")
        )
    )


def _proven_terminal_stop(info, engine, output, rows, *, backend=False):
    proof = info.get("backend_terminal" if backend else "terminal_stop", {})
    if not isinstance(proof, dict):
        return False
    observed = _validated_state(proof.get("terminal_state"))
    requested = (
        (
            info.get("expected_backend_terminal_stop") is True
            and proof.get("expected_backend_terminal_stop") is True
            and proof.get("fenced") is True
            and proof.get("status") in ("terminal_refused", "terminal_aborted")
        )
        if backend
        else (
            info.get("expected_host_terminal_stop") is True
            and proof.get("status") == "terminal_stop"
            and proof.get("termination_requested") is True
            and proof.get("termination_completed") is True
            and proof.get("alive_after_termination") is False
        )
    )
    if not (
        requested
        and observed
        and (observed[0]["done"] or observed[0]["won"])
        and proof.get("terminal_state_hash") == observed[1]
        and observed[0]["won"] == engine.get("won")
        and all(observed[0].get(key) == engine.get(key) for key in ("steps", "max_steps"))
    ):
        return False
    private_done = engine.get("won") is True or engine.get("step_limit") is True
    trace = output / "environment-trace.jsonl"
    if not private_done and trace.is_file():
        history = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        private_done = bool(history and history[-1].get("after", {}).get("done") is True)
    if not private_done:
        return False
    for row in rows:
        prompt, response, mask = row.get("prompt_ids"), row.get("response_ids"), row.get("response_mask")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(response, list)
            or not response
            or not isinstance(mask, list)
            or len(response) != len(mask)
            or not any(mask)
            or any(type(token) is not int or token < 0 for token in [*prompt, *response])
            or any(type(value) is not int or value not in (0, 1) for value in mask)
        ):
            return False
    return bool(rows)


def validate_frozen_attestation(value, *, evaluator_version=None):
    """Validate immutable route provenance without fabricating a weight clock."""
    if not isinstance(value, Mapping):
        raise ValueError("A frozen evaluator route attestation is required")
    result = copy.deepcopy(dict(value))
    fingerprint = result.get("model_fingerprint")
    if (
        type(result.get("schema_version")) is not int
        or result["schema_version"] != 1
        or result.get("policy_kind") != "frozen_evaluator"
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or result.get("frozen_model_version") != fingerprint
        or not isinstance(result.get("model_path"), str)
        or not Path(result["model_path"]).is_absolute()
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
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("kind") != "paired_empty_workdir"
        or manifest.get("credit_probe_group") != group
        or manifest.get("variant") != variant
        or manifest.get("workdir") != workdir
        or manifest.get("empty_before_run") is not True
        or manifest.get("cleanup_completed") is not True
        or info.get("role") != "counterfactual_probe"
        or info.get("credit_probe_group") != group
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
            "prompt_ids",
            "response_ids",
            "response_mask",
            "response_logprobs",
            "num_turns",
            "finished",
            "extra_fields",
        )
    }


async def native_probe_branch(
    *,
    gateway_manager,
    session_id,
    sample_fields,
    artifact_dir,
    task_config_path,
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
            "role": "counterfactual_probe",
            "_log_dir": str(artifact_dir),
            **{
                key: sample_fields["tools_kwargs"]["task"]["metadata"][key]
                for key in ("credit_probe_group", "credit_probe_variant")
            },
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
    if (
        info.get("error")
        and not (info.get("expected_host_terminal_stop") is True or info.get("expected_backend_terminal_stop") is True)
    ) or info.get("eval_completed") is not True:
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
    if not _backend_drained(capture):
        raise ProbeNotObserved("Counterfactual backend drain is unknown, pending, or not bound to this exact attempt")
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
    if info.get("expected_backend_terminal_stop") is True:
        if not _proven_terminal_stop(info, engine, output, rows, backend=True):
            raise ProbeNotObserved("Backend terminal stop lacks matching private environment proof or numeric TITO")
    elif info.get("expected_host_terminal_stop") is True:
        if not _proven_terminal_stop(info, engine, output, rows):
            raise ProbeNotObserved("Host terminal stop lacks matching private environment proof or numeric TITO")
    elif info.get("error"):
        raise ProbeNotObserved("A native bookkeeping error without proven terminal stop cannot become a credit label")
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
    *,
    request: ProbeRequest,
    sample_fields: dict[str, Any],
    store: SkillStore,
    loading: dict[str, Any],
    gateway_manager,
    artifact_root,
    task_config_path,
    frozen_attestation,
    run_branch=native_probe_branch,
    decode_prompt_ids=None,
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
        "schema_version": 1,
        "probe_id": request.probe_id,
        "window_id": request.window_id,
        "curate_record_id": request.curate_record_id,
        "skill_ref": request.skill_ref.to_dict(),
        "with_snapshot_id": request.with_snapshot_id,
        "without_snapshot_id": request.without_snapshot_id,
        "evaluator_version": request.evaluator_version,
        "protocol_id": request.protocol.protocol_id,
        "global_step": request.global_step,
        "task_id": task_id,
        "task_game_sha256": metadata["game_sha256"],
        "physical_probe_id": physical_id,
        "policy_attestation": frozen_attestation,
        "paired_workdir_protocol": "paired_empty_workdir_v1",
        "credit_probe_group": probe_group_id(request),
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
        outcomes, captures = {}, {}
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
                    role="counterfactual_probe",
                    credit_probe_group=probe_group_id(request),
                    credit_probe_variant=name,
                )
                settings = dict(loading, enabled=True)
                settings["snapshot_id"] = request.with_snapshot_id if loaded else request.without_snapshot_id
                fields = pin_prompt_group(
                    fields,
                    loading=settings,
                    store_root=store.root,
                    global_step=request.global_step,
                    partition_id="credit",
                    pin_directory=Path(artifact_root) / "context_groups",
                )
                _publish_immutable(started, {"started_at": time.time(), "session_id": fields["uid"]})
                capture = await run_branch(
                    gateway_manager=gateway_manager,
                    session_id=fields["uid"],
                    sample_fields=fields,
                    artifact_dir=branch_root,
                    task_config_path=task_config_path,
                    frozen_attestation=frozen_attestation,
                )
                _publish_immutable(captured, capture)
            outcomes[name] = _verify_branch(
                capture,
                request=request,
                sample_fields=sample_fields,
                loaded=loaded,
                artifact_root=artifact_root,
                frozen_attestation=pair_attestation,
            )
            captures[name] = capture
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
        _publish_immutable(
            root / "paired-workdirs.json",
            {
                "schema_version": 1,
                "credit_probe_group": probe_group_id(request),
                "physical_probe_id": physical_id,
                "branches": workspace_provenance,
            },
        )
        try:
            input_audit = verify_pair_inputs(
                captures,
                request=request,
                store=store,
                loading=loading,
                decode_prompt_ids=decode_prompt_ids,
            )
        except ProbeNotObserved as error:
            _publish_immutable(root / "paired-input-audit.json", error.input_audit)
            raise
        _publish_immutable(root / "paired-input-audit.json", input_audit)
        completed = root / "paired-event.json"
        previous = _read_payload(completed) if completed.exists() else None
        event = UsageEvent(
            event_id="event_" + physical_id,
            window_id=request.window_id,
            curate_record_id=request.curate_record_id,
            skill_ref=request.skill_ref,
            probe_id=request.probe_id,
            protocol_id=request.protocol.protocol_id,
            with_skill=outcomes["with_skill"],
            without_skill=outcomes["without_skill"],
            occurred_at=previous["occurred_at"] if previous else time.time(),
        )
        if event.with_skill.environment_state_hash != event.without_skill.environment_state_hash:
            raise ProbeNotObserved("Paired native branches started from different environment states")
        _publish_immutable(root / "paired-event.json", event.to_dict())
        return event
