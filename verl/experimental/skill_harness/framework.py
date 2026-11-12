# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Small official Uni-Agent extension hooks for host-side skill context.

Native Codex and the gateway's sampled token/logprob path are unchanged. The
default disabled adapter delegates directly to the original framework factory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from uni_agent.framework.framework import GatewayAgentFramework

from .context import CONTEXT_BUILDER_VERSION, pin_prompt_group, token_ids_hash
from .validation_audit import stamp_task_identity

FRAMEWORK_FQN = "verl.experimental.skill_harness.framework.SkillAwareAgentFramework"
TEXT_ONLY_MEDIA_FIELDS = (
    "multi_modal_inputs", "multi_modal_data", "pixel_values", "pixel_values_videos",
    "image_grid_thw", "video_grid_thw", "mm_token_type_ids",
)


class NonTextAlfworldTrajectoryError(ValueError):
    """Media cannot be trained by this text-only ALFWorld joint harness."""


def record_nontext_incident(artifact_root, *, source, context=None):
    """Persist a host-owned run-wide failure before a per-case exception escapes."""
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("Unsupported-media incidents require the trusted absolute config artifact_root")
    directory = root / "protocol_incidents" / "unsupported_media"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Directory existence is sticky too: an interrupted/failed JSON write must
    # not let a learner treat the remaining text cases as an acceptable batch.
    payload = {"schema_version": 1, "kind": "unsupported_alfworld_media", "source": source,
               "recorded_at": time.time(), "context": dict(context or {})}
    descriptor, temporary = tempfile.mkstemp(prefix=".incident-", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        path = directory / (uuid.uuid4().hex + ".json")
        os.replace(temporary, path)
        for parent in (directory, directory.parent, root):
            handle = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(handle)
            finally:
                os.close(handle)
        return path
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assert_no_nontext_incidents(artifact_root):
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ValueError("Unsupported-media checks require the trusted absolute config artifact_root")
    directory = root / "protocol_incidents" / "unsupported_media"
    if directory.exists() or directory.is_symlink():
        raise NonTextAlfworldTrajectoryError(
            "This run has an unsupported-media incident; refusing an optimizer update even if TQ kept only text cases"
        )


def _unwrap_media_carrier(value):
    # TQ returns actual per-row NonTensorStack/NonTensorData carriers. Avoid a
    # global Torch import so the framework's disabled CPU boundary stays light.
    if type(value).__name__ == "NonTensorData":
        return value.data
    if type(value).__name__ == "NonTensorStack":
        return value.tolist()
    return value


def _nontext_modality_flags(value):
    value = _unwrap_media_carrier(value)
    if value is None:
        return False
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_nontext_modality_flags(item) for item in value)
    if callable(getattr(value, "numel", None)):
        return value.numel() > 0 and bool(value.count_nonzero())
    if type(value).__module__.split(".")[0] == "numpy":
        return bool(value.any())
    return value != 0 if isinstance(value, int | bool) else True


def _contains_media(value):
    value = _unwrap_media_carrier(value)
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(_nontext_modality_flags(item) if key == "mm_token_type_ids" else _contains_media(item)
                   for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return any(_contains_media(item) for item in value)
    if callable(getattr(value, "numel", None)):
        # A black image is still media: test element count, not pixel truthiness.
        return value.numel() > 0
    if type(value).__module__.split(".")[0] == "numpy":
        return _contains_media(value.tolist()) if value.dtype.kind == "O" else value.size > 0
    return True  # PIL images, paths/URIs, bytes and unknown nonempty payloads.


def require_text_only_media(value, *, source, artifact_root=None, incident_context=None):
    if _contains_media(value):
        if artifact_root is not None:
            record_nontext_incident(artifact_root, source=source, context=incident_context)
        raise NonTextAlfworldTrajectoryError(
            f"ALFWorld joint training is text-only: nonempty media in {source}; refusing loss without image context"
        )


def guard_text_only_tq_batch(queue, batch, *, artifact_root=None):
    """Check authoritative per-key media columns before selecting text fields.

    kv_batch_get returns only common ready columns. kv_describe also exposes
    columns present in just one key or still unfinalized; never let those hide.
    Only empty carriers may subsequently be omitted from mixed-role packing.
    """
    description = queue.kv_describe(keys=batch.keys, partition_id=batch.partition_id)
    if not isinstance(description, Mapping) or set(description) != set(batch.keys):
        raise ValueError("Cannot prove a text-only batch: TQ per-key metadata is incomplete")
    for key, row in description.items():
        if (not isinstance(row, Mapping) or not isinstance(row.get("fields"), list)
                or not isinstance(row.get("unfinalized"), list)
                or any(not isinstance(name, str) for name in (*row["fields"], *row["unfinalized"]))
                or not set(row["unfinalized"]).issubset(row["fields"])):
            raise ValueError("Cannot prove a text-only batch: invalid TQ optional-column metadata")
    for column in TEXT_ONLY_MEDIA_FIELDS:
        keys = [key for key in batch.keys if column in description[key]["fields"]]
        if not keys:
            continue
        if any(column in description[key]["unfinalized"] for key in keys):
            raise ValueError("Cannot prove a text-only batch: a media carrier column is unfinalized")
        data = queue.kv_batch_get(keys=keys, partition_id=batch.partition_id, select_fields=[column])
        if column not in data:
            raise ValueError("Cannot prove a text-only batch: a described media carrier was not returned")
        require_text_only_media({column: data[column]}, source="retained TransferQueue media column",
                                artifact_root=artifact_root, incident_context={"column": column, "tq_keys": keys})
    for column in TEXT_ONLY_MEDIA_FIELDS:
        if column in (batch.extra_info or {}):
            require_text_only_media({column: batch.extra_info[column]}, source="batch-level media metadata",
                                    artifact_root=artifact_root, incident_context={"column": column})


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_plain(item) for item in value]
    return value


def _core_import_guard():
    import verl

    source = Path(__file__).resolve().parents[3]
    if Path(verl.__file__).resolve() != source / "verl/__init__.py" or verl.__version__ != "0.9.1":
        raise RuntimeError("Skill-aware rollout requires this core verl 0.9.1 checkout before benchmark dependencies")


def _tokenizer_version(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError("Skill loading requires an existing local tokenizer directory")
    digest = hashlib.sha256()
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        file = root / name
        if file.is_file():
            found = True
            digest.update(name.encode() + b"\0")
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    if not found:
        raise FileNotFoundError("The skill tokenizer directory has no tokenizer metadata")
    return "sha256:" + digest.hexdigest()


def loading_settings(config):
    """Normalize trusted run settings; loading is opt-in and baseline-safe."""
    harness = _get(config, "skill_harness", {})
    requested = _get(harness, "loading", {}) or {}
    enabled = _get(requested, "enabled", False)
    if type(enabled) is not bool:
        raise ValueError("skill_harness.loading.enabled must be a boolean")
    if not enabled:
        return {"enabled": False}
    if _get(harness, "mode") == "solver_baseline":
        raise ValueError("solver_baseline must keep host skill loading disabled")
    actor = _get(config, "actor_rollout_ref")
    rollout, model = _get(actor, "rollout"), _get(actor, "model")
    tokenizer_path = _get(requested, "tokenizer_path") or _get(model, "tokenizer_path") or _get(model, "path")
    result = {
        "enabled": True,
        "branch": _get(requested, "branch", "main"),
        "snapshot_id": _get(requested, "snapshot_id"),
        "top_k": _get(requested, "top_k", 6),
        "skill_token_budget": _get(requested, "skill_token_budget", 1024),
        "max_prompt_tokens": _get(requested, "max_prompt_tokens", _get(rollout, "prompt_length")),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "tokenizer_revision": _get(requested, "tokenizer_revision"),
        "tokenizer_version": _tokenizer_version(tokenizer_path),
        "render_kwargs": _plain(_get(requested, "render_kwargs", _get(config.data, "apply_chat_template_kwargs", {}))),
    }
    for name in ("top_k", "skill_token_budget", "max_prompt_tokens"):
        value = result[name]
        if type(value) is not int or value < 0 or (name == "max_prompt_tokens" and value == 0):
            raise ValueError(f"skill_harness.loading.{name} has an invalid limit")
    if not _get(harness, "store_root"):
        raise ValueError("Enabled host skill loading requires skill_harness.store_root")
    return result


class SkillAwareRolloutAdapter(AgentFrameworkRolloutAdapter):
    """Use the official adapter factory and select one small framework subclass."""

    @classmethod
    def create(cls, *, config, **kwargs):
        loading = loading_settings(config)
        effective = config
        if loading["enabled"]:
            _core_import_guard()
            from omegaconf import open_dict

            effective = copy.deepcopy(config)
            framework_config = effective.actor_rollout_ref.rollout.custom.agent_framework
            existing = framework_config.get("framework_class_fqn")
            if existing not in (None, FRAMEWORK_FQN):
                raise ValueError("Host skill loading cannot silently replace another custom framework subclass")
            with open_dict(framework_config):
                framework_config.framework_class_fqn = FRAMEWORK_FQN
        return super().create(config=effective, **kwargs)


class SkillAwareAgentFramework(GatewayAgentFramework):
    """Pin metadata before the upstream prompt hook performs rollout.n fanout."""

    @classmethod
    def from_config(cls, *, config, **kwargs):
        instance = super().from_config(config=config, **kwargs)
        instance._skill_strict_train_groups = _get(config.skill_harness, "mode") == "skill_joint"
        instance._skill_loading = loading_settings(config)
        instance._skill_store_root = _get(config.skill_harness, "store_root")
        instance._skill_pin_directory = None
        artifact_root = _get(config.skill_harness, "artifact_root")
        if artifact_root is not None and not Path(artifact_root).is_absolute():
            raise ValueError("Text-only joint training requires an absolute trusted config artifact_root")
        instance._text_protocol_artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None
        if _get(config.skill_harness, "mode") == "skill_joint" and instance._text_protocol_artifact_root is None:
            raise ValueError("Text-only joint training requires a trusted config artifact_root")
        if instance._skill_loading["enabled"]:
            _core_import_guard()
            instance._skill_pin_directory = Path(config.skill_harness.artifact_root) / "context_groups"
        return instance

    async def _run_prompt_rollouts(
        self,
        *,
        sample_fields,
        sample_index,
        global_steps,
        partition_id,
        num_sessions,
    ):
        pinned = pin_prompt_group(
            sample_fields,
            loading=self._skill_loading,
            store_root=self._skill_store_root,
            global_step=global_steps,
            partition_id=partition_id,
            pin_directory=getattr(self, "_skill_pin_directory", None),
        )
        stats = await super()._run_prompt_rollouts(
            sample_fields=pinned,
            sample_index=sample_index,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
        )
        if (
            getattr(self, "_skill_strict_train_groups", False)
            and partition_id == "train"
            and pinned.get("tools_kwargs", {}).get("task", {}).get("name") == "alfworld_skill_training"
            and isinstance(stats, Mapping)
            and stats.get("num_failed_sessions", 0) > 0
        ):
            import transfer_queue as tq

            # The parent has awaited every sibling. Preserve its actual session
            # statistics, but never call an incomplete training group finished.
            # The custom sampler also checks the pre-exception local archive
            # registry, closing the parent's earlier provisional-finished gap.
            await tq.async_kv_put(key=str(pinned["uid"]), partition_id=partition_id, tag={"status": "failure"})
            stats = {**stats, "num_failed_uids": 1}
        return stats

    async def _run_agent_episode(
        self, *, sample_fields, sample_index, session_index, global_steps, runner_name, runner_config, sampling_params,
    ):
        # This supported host hook runs after fanout. No model message or token
        # is changed, and each session gets its real identity rather than n=0.
        fields = dict(sample_fields)
        fields["tools_kwargs"] = stamp_task_identity(
            sample_fields.get("tools_kwargs"), rollout_uid=str(sample_fields.get("uid", "")),
            session_index=session_index, global_step=global_steps,
        )
        return await super()._run_agent_episode(
            sample_fields=fields, sample_index=sample_index, session_index=session_index,
            global_steps=global_steps, runner_name=runner_name, runner_config=runner_config,
            sampling_params=sampling_params,
        )

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory,
        sample_fields,
        session_index,
        global_steps,
        uid,
    ):
        task = sample_fields.get("tools_kwargs", {}).get("task", {})
        alfworld = task.get("name") == "alfworld_skill_training"
        protocol_root = getattr(self, "_text_protocol_artifact_root", None)
        incident_context = {"uid": str(uid), "session_index": int(session_index),
                            "global_steps": None if global_steps is None else int(global_steps)}
        if alfworld:
            require_text_only_media(getattr(trajectory, "multi_modal_data", None), source="native trajectory media",
                                    artifact_root=protocol_root, incident_context=incident_context)
        field, tag = super()._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields=sample_fields,
            session_index=session_index,
            global_steps=global_steps,
            uid=uid,
        )
        if alfworld:
            for name in TEXT_ONLY_MEDIA_FIELDS:
                if name in field:
                    require_text_only_media({name: field[name]}, source="official serializer media field",
                                            artifact_root=protocol_root, incident_context=incident_context)
        if not self._skill_loading["enabled"]:
            return field, tag
        pinned = task.get("metadata", {})
        runner = trajectory.extra_fields.get("runner_reward_info", {})
        context = runner.get("reward_context", {}).get("skill_context")
        if not isinstance(context, dict) or context.get("enabled") is not True:
            raise ValueError("An enabled skill episode did not return its actual injected context provenance")
        if (
            context.get("snapshot_id") != pinned.get("skill_snapshot_id")
            or context.get("context_version") != pinned.get("skill_context_version")
            or context.get("group_id") != pinned.get("skill_context_group_id")
            or context.get("builder_version") != CONTEXT_BUILDER_VERSION
        ):
            raise ValueError("Injected skill context differs from its trusted prompt-group pin")
        # Upstream produced every tensor, response mask and behavior logprob.
        # This hook changes only extension metadata after calling that path.
        extra = dict(field.get("extra_fields", {}))
        extra["skill_context"] = copy.deepcopy(context)
        extra["skill_prompt_provenance"] = {
            "actual_prompt_tokens": len(trajectory.prompt_ids),
            "actual_prompt_token_ids_sha256": token_ids_hash(trajectory.prompt_ids),
            "actual_prompt_budget_scope": "gateway_prompt_ids; includes_native_codex_context",
            "host_task_prompt_tokens": context["rendered_prompt_tokens"],
            "host_budget_scope": context["budget_scope"],
            "native_overhead_is_in_host_count": False,
        }
        field["extra_fields"] = extra
        for key, value in {
            "skill_snapshot_id": context["snapshot_id"],
            "skill_context_version": context["context_version"],
            "skill_context_id": context["context_id"],
        }.items():
            tag.setdefault(key, value)
        return field, tag
