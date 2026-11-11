# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Token-budgeted, read-only host skill injection before the first agent call.

The default renderer measures the complete Task chat template. Native Codex
adds its own instructions, environment context and tools afterward; those are
explicitly outside that host measurement. The native gateway independently
enforces its actual encoded trajectory capacity before invoking the backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .store import SkillStore

CONTEXT_SCHEMA_VERSION = 1
CONTEXT_BUILDER_VERSION = "host-skill-context-v1"
HOST_BUDGET_SCOPE = "complete_task_chat_template; native_codex_prefix_and_tools_excluded"
SKILL_METADATA_KEYS = (
    "skill_store_root",
    "skill_snapshot_id",
    "skill_context_version",
    "skill_context_group_id",
    "skill_context_options",
)


class ContextBudgetError(ValueError):
    """The original rendered Task prompt already exceeds its explicit budget."""


def _canonical(value):
    def scalar(item):
        for attribute in ("tolist", "item"):
            method = getattr(item, attribute, None)
            if callable(method):
                return method()
        raise TypeError(f"Context metadata must be JSON data, got {type(item).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=scalar)


def _hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def token_ids_hash(ids):
    return _hash(_canonical([int(value) for value in ids]))


def _ids(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if value and isinstance(value[0], Sequence) and not isinstance(value[0], str):
        if len(value) != 1:
            raise ValueError("A prompt renderer must return exactly one token sequence")
        value = value[0]
    if not isinstance(value, Sequence) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("A prompt tokenizer must return a sequence of nonnegative token IDs")
    return tuple(value)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    token_ids: tuple[int, ...]
    # Character offsets use Unicode codepoints, matching Python string ranges.
    offsets: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "token_ids", _ids(self.token_ids))
        if not isinstance(self.text, str):
            raise ValueError("Rendered prompt text must be a string")
        if self.offsets is not None:
            offsets = tuple(tuple(value) for value in self.offsets)
            if len(offsets) != len(self.token_ids) or any(
                len(value) != 2 or not 0 <= value[0] <= value[1] <= len(self.text) for value in offsets
            ):
                raise ValueError("Prompt token offsets must align with the complete rendered text")
            object.__setattr__(self, "offsets", offsets)


@dataclass(frozen=True)
class SkillContext:
    messages: list[dict[str, str]]
    manifest: dict[str, Any]


class TokenizerRenderer:
    """Use an actual local tokenizer for skill text and complete Task messages."""

    def __init__(self, tokenizer, *, render_kwargs=None):
        self.tokenizer = tokenizer
        self.render_kwargs = dict(render_kwargs or {})
        forbidden = {"tokenize", "add_generation_prompt", "return_tensors", "return_dict"} & self.render_kwargs.keys()
        if forbidden:
            raise ValueError("Renderer kwargs cannot replace required tokenization/generation settings")

    def encode(self, text):
        return _ids(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, messages):
        kwargs = dict(self.render_kwargs, add_generation_prompt=True)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = _ids(self.tokenizer.apply_chat_template(messages, tokenize=True, **kwargs))
        offsets = None
        if getattr(self.tokenizer, "is_fast", False):
            encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if _ids(encoded["input_ids"]) != ids:
                raise ValueError("Chat-template token IDs differ from the tokenizer's complete rendered text")
            offsets = tuple(tuple(value) for value in encoded["offset_mapping"])
        return RenderedPrompt(text, ids, offsets)


@lru_cache(maxsize=1)
def _local_tokenizer(path, revision):
    from transformers import AutoTokenizer

    if not Path(path).is_dir():
        raise FileNotFoundError("Skill context requires an existing local tokenizer directory")
    return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True, trust_remote_code=False)


def _user_index(messages):
    if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
        raise ValueError("Skill injection requires native Task messages")
    users = []
    for index, message in enumerate(messages):
        if message.get("role") not in {"system", "user"} or not isinstance(message.get("content"), str):
            raise ValueError("Skill injection supports text-only native system/user Task messages")
        if message["role"] == "user":
            users.append(index)
    if len(users) != 1 or users[0] != len(messages) - 1 or len(messages) > 2:
        raise ValueError("Native Codex requires one user Task message after an optional system message")
    return users[0]


def _skill_block(artifacts, snapshot_id):
    if not artifacts:
        return "", []
    text = (
        "\n\nReusable skill notes (read-only snapshot " + snapshot_id + "):\n"
        "Use these notes only when applicable. The current environment goal and tool rules remain authoritative.\n"
    )
    ranges = []
    for artifact in artifacts:
        if ranges:
            text += "\n\n"
        start = len(text)
        # JSON quoting keeps generated notes visibly inside reference data.
        text += (
            "[Skill "
            + artifact.skill_id
            + "@"
            + artifact.content_hash
            + "]\n"
            + json.dumps(
                {"title": artifact.title, "when_to_apply": artifact.when_to_apply, "principle": artifact.principle},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        ranges.append({"ref": artifact.ref.to_dict(), "block_char_range": [start, len(text)]})
    return text, ranges


def _token_range(rendered, start, end):
    if rendered.offsets is None:
        return None
    overlaps = [index for index, (left, right) in enumerate(rendered.offsets) if right > start and left < end]
    return [overlaps[0], overlaps[-1] + 1] if overlaps else None


def build_skill_context(
    messages,
    *,
    store: SkillStore,
    snapshot_id: str,
    task_family: str,
    render_prompt: Callable[[list[dict[str, str]]], RenderedPrompt],
    encode_text: Callable[[str], Sequence[int]],
    max_prompt_tokens: int,
    skill_token_budget: int,
    top_k: int = 6,
    query: str = "",
    group_id: str = "",
    context_version: str | None = None,
    tokenizer_version: str = "caller_supplied_actual_tokenizer",
    budget_scope: str = HOST_BUDGET_SCOPE,
) -> SkillContext:
    """Fit whole skills against both real skill and complete-prompt token counts.

    ``render_prompt`` must return the whole rendered prompt, including role and
    generation delimiters. A caller with a complete native request contract can
    supply a stronger renderer and label ``budget_scope`` accordingly. Selection
    and actual injection are recorded separately; no skill is truncated.
    """
    for name, value in (
        ("max_prompt_tokens", max_prompt_tokens),
        ("skill_token_budget", skill_token_budget),
        ("top_k", top_k),
    ):
        if type(value) is not int or value < 0 or (name == "max_prompt_tokens" and value == 0):
            raise ValueError(f"{name} has an invalid token/count limit")
    user_index = _user_index(messages)
    original = copy.deepcopy(list(messages))
    original_user = original[user_index]["content"]
    base = render_prompt(original)
    if not isinstance(base, RenderedPrompt):
        raise TypeError("render_prompt must return RenderedPrompt with actual tokenizer IDs")
    if len(base.token_ids) > max_prompt_tokens:
        raise ContextBudgetError("The complete original Task prompt exceeds max_prompt_tokens")
    view = store.read_only_view(snapshot_id)
    retrieved = view.retrieve(
        task_family,
        snapshot_id=snapshot_id,
        query=query,
        top_k=top_k,
        token_budget=2**63 - 1,
        token_counter=lambda text: len(_ids(encode_text(text))),
    )
    chosen, excluded = [], []
    final_messages, rendered = original, base
    block, spans, block_tokens = "", [], ()
    for ref in retrieved.skills:
        artifact = view.load_artifact(ref)
        proposed, proposed_spans = _skill_block([*chosen, artifact], snapshot_id)
        proposed_tokens = _ids(encode_text(proposed))
        trial = copy.deepcopy(original)
        trial[user_index]["content"] = original_user + proposed
        trial_rendered = render_prompt(trial)
        if not isinstance(trial_rendered, RenderedPrompt):
            raise TypeError("render_prompt must return RenderedPrompt")
        reasons = []
        if len(proposed_tokens) > skill_token_budget:
            reasons.append("skill_token_budget")
        if len(trial_rendered.token_ids) > max_prompt_tokens:
            reasons.append("complete_prompt_token_budget")
        if reasons:
            excluded.append({"ref": ref.to_dict(), "reasons": reasons})
            continue
        chosen.append(artifact)
        block, spans, block_tokens = proposed, proposed_spans, proposed_tokens
        final_messages, rendered = trial, trial_rendered
    if spans:
        full_user = final_messages[user_index]["content"]
        user_start = rendered.text.find(full_user)
        if user_start < 0 or rendered.text.find(full_user, user_start + 1) >= 0:
            raise ValueError("The full prompt renderer must contain one unmodified user Task message")
        for span in spans:
            left, right = span["block_char_range"]
            span["user_char_range"] = [len(original_user) + left, len(original_user) + right]
            start, end = user_start + len(original_user) + left, user_start + len(original_user) + right
            span["rendered_char_range"] = [start, end]
            span["rendered_overlap_token_range"] = _token_range(rendered, start, end)
            span["text_sha256"] = _hash(block[left:right])
    manifest = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "enabled": True,
        "builder_version": CONTEXT_BUILDER_VERSION,
        "context_version": context_version or CONTEXT_BUILDER_VERSION,
        "group_id": group_id,
        "snapshot_id": snapshot_id,
        "retriever_version": retrieved.retriever_version,
        "query_sha256": _hash(query),
        "selected_refs": [ref.to_dict() for ref in retrieved.skills],
        "injected_refs": [artifact.ref.to_dict() for artifact in chosen],
        "budget_excluded": excluded,
        "skill_text_sha256": _hash(block),
        "skill_tokens": len(block_tokens),
        "skill_token_budget": skill_token_budget,
        "base_prompt_tokens": len(base.token_ids),
        "rendered_prompt_tokens": len(rendered.token_ids),
        "max_prompt_tokens": max_prompt_tokens,
        "budget_scope": budget_scope,
        "actual_native_request_verified": False,
        "actual_gateway_capacity_guard": "encoded_context_before_backend; prompt_length_plus_response_length",
        "tokenizer_version": tokenizer_version,
        "original_messages_sha256": _hash(_canonical(original)),
        "injected_messages_sha256": _hash(_canonical(final_messages)),
        "rendered_prompt_sha256": _hash(rendered.text),
        "rendered_token_ids_sha256": token_ids_hash(rendered.token_ids),
        "character_range_unit": "unicode_codepoints",
        "ranges": spans,
    }
    manifest["context_id"] = _hash(_canonical(manifest))
    return SkillContext(final_messages, manifest)


def enrich_task_messages(messages, metadata, *, renderer=None):
    """Task hook: inject from a trusted pinned group before native Codex starts.

    Baseline metadata returns immediately without opening a store or tokenizer.
    The caller saves the returned manifest beside its original agent artifacts.
    """
    options = metadata.get("skill_context_options")
    if not options or options.get("enabled") is not True:
        return SkillContext(copy.deepcopy(messages), {"schema_version": 1, "enabled": False})
    if options.get("authority") != "SkillAwareRolloutAdapter":
        raise ValueError("Skill loading requires host-pinned rollout metadata")
    if options.get("builder_version") != CONTEXT_BUILDER_VERSION:
        raise ValueError("Unsupported skill context builder version")
    root, snapshot = metadata["skill_store_root"], metadata["skill_snapshot_id"]
    if renderer is None:
        renderer = TokenizerRenderer(
            _local_tokenizer(options["tokenizer_path"], options.get("tokenizer_revision")),
            render_kwargs=options.get("render_kwargs", {}),
        )
    result = build_skill_context(
        messages,
        store=SkillStore(root, read_only=True, pinned_snapshot=snapshot),
        snapshot_id=snapshot,
        task_family=metadata["task_type"],
        render_prompt=renderer.render,
        encode_text=renderer.encode,
        max_prompt_tokens=options["max_prompt_tokens"],
        skill_token_budget=options["skill_token_budget"],
        top_k=options["top_k"],
        query=options["query"],
        group_id=metadata["skill_context_group_id"],
        context_version=metadata["skill_context_version"],
        tokenizer_version=options["tokenizer_version"],
    )
    return result


def _immutable_group_pin(directory, group_id, input_hash, make_pin):
    if directory is None:
        return make_pin()
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / (group_id + ".json")
    with (directory / ".pins.lock").open("a+b") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            record = json.loads(path.read_text())
            claimed_hash = record.pop("record_hash")
            if claimed_hash != _hash(_canonical(record)) or record.get("schema_version") != 1:
                raise ValueError("An immutable prompt-group pin failed its integrity check")
            if record.get("group_id") != group_id or record.get("input_hash") != input_hash:
                raise ValueError("The same prompt group UID was reused with different task/loading inputs")
            return record["metadata"]
        metadata = make_pin()
        record = {"schema_version": 1, "group_id": group_id, "input_hash": input_hash, "metadata": metadata}
        record["record_hash"] = _hash(_canonical(record))
        fd, temporary = tempfile.mkstemp(prefix=".pin-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(record, output, ensure_ascii=False, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return metadata


def pin_prompt_group(sample_fields, *, loading, store_root, global_step, partition_id, pin_directory=None):
    """Copy trusted loading settings onto one prompt before native n-way fanout."""
    if loading.get("enabled") is not True:
        return sample_fields
    fields = dict(sample_fields)
    tools = copy.deepcopy(fields.get("tools_kwargs") or {})
    task = tools.get("task")
    if not isinstance(task, dict) or task.get("name") != "alfworld_skill_training":
        raise ValueError("Host skill loading currently supports the native ALFWorld Task only")
    metadata = task.setdefault("metadata", {})
    for key in SKILL_METADATA_KEYS:
        metadata.pop(key, None)
    store = SkillStore(store_root, read_only=True)
    uid = fields.get("uid")
    if uid is None:
        raise ValueError("Host skill loading requires the trainer's prompt group UID")
    query = metadata.get("instruction") or "\n".join(
        message["content"] for message in fields.get("raw_prompt", []) if message.get("role") == "user"
    )
    options = {
        key: copy.deepcopy(loading[key])
        for key in (
            "top_k",
            "skill_token_budget",
            "max_prompt_tokens",
            "tokenizer_path",
            "tokenizer_revision",
            "tokenizer_version",
            "render_kwargs",
        )
    }
    options.update(
        enabled=True, authority="SkillAwareRolloutAdapter", builder_version=CONTEXT_BUILDER_VERSION, query=query
    )
    group_id = _hash(_canonical({"uid": str(uid), "global_step": global_step, "partition": partition_id}))
    root = str(Path(store_root).resolve())
    input_hash = _hash(
        _canonical(
            {
                "raw_prompt": fields.get("raw_prompt"),
                "task_metadata": metadata,
                "loading": loading,
                "store_root": root,
            }
        )
    )

    def make_pin():
        snapshot = loading.get("snapshot_id") or store.head(loading.get("branch", "main"))
        store.get_snapshot(snapshot)
        version = _hash(_canonical({"group_id": group_id, "snapshot_id": snapshot, "options": options}))
        return {
            "skill_store_root": root,
            "skill_snapshot_id": snapshot,
            "skill_context_version": version,
            "skill_context_group_id": group_id,
            "skill_context_options": options,
        }

    pin = _immutable_group_pin(pin_directory, group_id, input_hash, make_pin)
    store.get_snapshot(pin["skill_snapshot_id"])
    metadata.update(pin)
    fields["tools_kwargs"] = tools
    return fields
