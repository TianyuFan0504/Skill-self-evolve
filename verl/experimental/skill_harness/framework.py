# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Small official Uni-Agent extension hooks for host-side skill context.

Native Codex and the gateway's sampled token/logprob path are unchanged. The
default disabled adapter delegates directly to the original framework factory.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
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
logger = logging.getLogger(__name__)


class NonTextAlfworldTrajectoryError(ValueError):
    """Media cannot be trained by this text-only ALFWorld joint harness."""


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
    """Normalize host-side skill loading settings."""
    harness = _get(config, "skill_harness", {})
    requested = _get(harness, "loading", {}) or {}
    enabled = _get(requested, "enabled", False)
    if type(enabled) is not bool:
        raise ValueError("skill_harness.loading.enabled must be a boolean")
    if not enabled:
        return {"enabled": False}
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
        instance._skill_strict_train_groups = True
        instance._skill_loading = loading_settings(config)
        instance._skill_store_root = _get(config.skill_harness, "store_root")
        instance._skill_pin_directory = None
        artifact_root = _get(config.skill_harness, "artifact_root")
        if artifact_root is not None and not Path(artifact_root).is_absolute():
            raise ValueError("Text-only joint training requires an absolute trusted config artifact_root")
        instance._text_protocol_artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None
        if instance._text_protocol_artifact_root is None:
            raise ValueError("Text-only joint training requires a trusted config artifact_root")
        if instance._skill_loading["enabled"]:
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
        strict_group = (
            getattr(self, "_skill_strict_train_groups", False)
            and partition_id == "train"
            and pinned.get("tools_kwargs", {}).get("task", {}).get("name") == "alfworld_skill_training"
        )
        if not strict_group:
            return await super()._run_prompt_rollouts(
                sample_fields=pinned, sample_index=sample_index, global_steps=global_steps,
                partition_id=partition_id, num_sessions=num_sessions,
            )
        import transfer_queue as tq

        if pinned.get("uid") is None:
            raise ValueError("ALFWorld prompt groups require a TransferQueue uid")
        uid = str(pinned["uid"])
        sampling = self._build_session_sampling_params(partition_id=partition_id, sample_fields=pinned)
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        outcomes = await asyncio.gather(*(
            self._run_agent_episode_with_concurrency_limit(
                sample_fields=pinned, sample_index=sample_index, session_index=index,
                global_steps=global_steps, sampling_params=sampling,
            )
            for index in range(num_sessions)
        ), return_exceptions=True)
        successful = failed = outputs = unfinished = 0
        reasons = []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed += 1
                reasons.append(f"{type(outcome).__name__}: {outcome}")
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            trajectories, fields = outcome
            if not trajectories:
                failed += 1
                reasons.append(f"empty trajectories for uid={uid} session_index={index}")
                continue
            try:
                await self._write_session_trajectories_to_tq(
                    uid=uid, session_index=index, trajectories=trajectories, sample_fields=fields,
                    global_steps=global_steps, partition_id=partition_id,
                )
            except Exception as error:
                logger.exception("ALFWorld group serialization failed for uid=%s session=%s", uid, index)
                failed += 1
                reasons.append(f"TQ write error: {error}")
            else:
                successful += 1
                outputs += len(trajectories)
                unfinished += int(any(trajectory.finished is False for trajectory in trajectories))
        rejected = failed > 0 or successful == 0
        # Publish one terminal state after every sibling serializer has settled.
        # Incomplete groups never become briefly sampleable as finished.
        await tq.async_kv_put(
            key=uid, partition_id=partition_id, tag={"status": "failure" if rejected else "finished"},
        )
        return {
            "num_success_sessions": successful, "num_failed_sessions": failed,
            "num_success_outputs": outputs, "num_unfinished_episodes": unfinished,
            "num_failed_uids": int(rejected), "failure_reasons": reasons,
        }

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
