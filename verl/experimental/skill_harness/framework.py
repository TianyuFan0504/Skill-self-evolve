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
from collections.abc import Mapping, Sequence
from pathlib import Path

from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
from uni_agent.framework.framework import GatewayAgentFramework

from .context import CONTEXT_BUILDER_VERSION, pin_prompt_group, token_ids_hash

FRAMEWORK_FQN = "verl.experimental.skill_harness.framework.SkillAwareAgentFramework"


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
        instance._skill_loading = loading_settings(config)
        instance._skill_store_root = _get(config.skill_harness, "store_root")
        instance._skill_pin_directory = None
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
        return await super()._run_prompt_rollouts(
            sample_fields=pinned,
            sample_index=sample_index,
            global_steps=global_steps,
            partition_id=partition_id,
            num_sessions=num_sessions,
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
        field, tag = super()._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields=sample_fields,
            session_index=session_index,
            global_steps=global_steps,
            uid=uid,
        )
        if not self._skill_loading["enabled"]:
            return field, tag
        task = sample_fields.get("tools_kwargs", {}).get("task", {})
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
