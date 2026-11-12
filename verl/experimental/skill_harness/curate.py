# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Native Codex EMA curation with immutable generation/replay acknowledgements.

The owner supplies a dedicated EMA GatewayManager and holds its weight-sync
barrier for the entire collection call. This module never loads a model, updates
an optimizer, admits a skill, or publishes a main-branch snapshot. Only existing
official training episodes can become curator inputs.

Native Codex 0.167.0 exposes an output-schema CLI option, but the pinned Uni-Agent
Responses adapter rejects structured text formats. Consequently the native
request remains ordinary text with unrestricted sampling; JSON is validated on
the host. The schema is an artifact, not a constrained-decoding request.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import inspect
import ipaddress
import json
import math
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .alfworld.data import TASK_TYPES, validate_relative_game_path
from .replay import CurateReplay, CurateReplayRecord, ReplayRef
from .store import SkillArtifact, SkillCandidate, SkillStore

COLLECTOR_VERSION = "native-codex-curate-v3"
EVENT_CLASSIFIER_VERSION = "codex-startup-diagnostics-v1"
NATIVE_CODEX_VERSION = "0.167.0"
NATIVE_CONFIG_OVERRIDES = (
    "skills.include_instructions=false",
    'web_search="disabled"',
    "tools.experimental_request_user_input.enabled=false",
)
SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "min_p": 0.0,
    "logprobs": True,
}
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "unified_exec_tty",
    "multi_agent",
    "multi_agent_v2",
    "apps",
    "plugins",
    "hooks",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "view_image",
    "skill_search",
    "workspace_dependencies",
    "sleep_tool",
    "goals",
)
SKILL_FIELDS = {"skill_id", "title", "principle", "when_to_apply", "task_families"}
SKILL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(SKILL_FIELDS),
    "properties": {
        "skill_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"},
        "title": {"type": "string", "minLength": 1, "maxLength": 256},
        "principle": {"type": "string", "minLength": 1, "maxLength": 32768},
        "when_to_apply": {"type": "string", "minLength": 1, "maxLength": 8192},
        "task_families": {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "string"}},
    },
}
ABSTENTION_REASON_MAX_CHARS = 2048
ABSTENTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "reason"],
    "properties": {
        "decision": {"const": "abstain"},
        "reason": {"type": "string", "minLength": 1, "maxLength": ABSTENTION_REASON_MAX_CHARS},
    },
}
CURATE_ACTION_SCHEMA = {"oneOf": [SKILL_SCHEMA, ABSTENTION_SCHEMA]}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CurateProtocolError(ValueError):
    """The observed generation cannot safely enter the current curate objective."""


class CurateSourceCensored(RuntimeError):
    """A consistent completed source has no environment actions to summarize."""

    reason = "source_has_no_environment_actions"

    def __init__(self, evidence):
        super().__init__("Completed solver source has zero observed environment actions")
        self.evidence = evidence


class CurateRecoveryRequired(RuntimeError):
    """An earlier launch lacks a durable capture; do not silently resample it."""


class CurateNativeExecutionError(RuntimeError):
    """An unknown native error is infrastructure failure, never a tool reward."""


def classify_native_events(stdout, *, expected_codex_home=None):
    """Classify native JSON events without changing the raw event/token stream.

    Only two exact, observed Codex 0.167 startup warning forms are diagnostic.
    The feature warning must name this transport's isolated home. They cease to
    be startup diagnostics after the first turn/model item. Unknown error/item
    types fail closed as execution failures instead of fabricated tool calls.
    """
    ignored_config = (
        "Codex is ignoring 1 unrecognized configuration setting. Check for typos or deprecated settings.\n"
        "  session-flags: `disable_response_storage` is ignored."
    )
    feature_warning = None
    if expected_codex_home is not None:
        feature_warning = (
            "Under-development features enabled: skip_host_skill_discovery. "
            "Under-development features are incomplete and may behave unpredictably. "
            "To suppress this warning, set `suppress_unstable_features_warning = true` in "
            + str(Path(expected_codex_home) / "config.toml")
            + "."
        )
    tool_types = {
        "command_execution",
        "mcp_tool_call",
        "web_search",
        "web_search_call",
        "file_change",
        "plan",
        "tool_call",
        "function_call",
        "custom_tool_call",
        "computer_call",
        "collab_agent_tool_call",
        "shell_command",
        "exec_command",
        "apply_patch",
        "image_generation",
    }
    content_types = {"agent_message", "message", "reasoning"}
    controls = {"thread.started", "turn.started", "turn.completed", "response.completed", "process.completed"}
    result = {
        "schema_version": 1,
        "classifier_version": EVENT_CLASSIFIER_VERSION,
        "startup_diagnostics": [],
        "tool_events": [],
        "fatal_events": [],
    }
    started = False
    for number, line in enumerate(stdout.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            result["fatal_events"].append({"line": number, "kind": "non_json_stdout", "text": line})
            continue
        if not isinstance(event, dict):
            result["fatal_events"].append({"line": number, "kind": "invalid_event", "event": event})
            continue
        event_type, item = event.get("type"), event.get("item")
        entry = {"line": number, "event": event}
        if event_type in {"error", "turn.failed"}:
            result["fatal_events"].append({**entry, "kind": "native_execution_error"})
            continue
        if isinstance(item, dict) and event_type not in {"item.started", "item.updated", "item.completed"}:
            result["fatal_events"].append({**entry, "kind": "unknown_event_type"})
            continue
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type in content_types:
                started = True
            elif item_type in tool_types:
                started = True
                result["tool_events"].append({**entry, "kind": "native_tool_invocation"})
            elif item_type == "error":
                message = item.get("message")
                known = message == ignored_config or (feature_warning is not None and message == feature_warning)
                if not started and event_type == "item.completed" and known:
                    result["startup_diagnostics"].append({**entry, "kind": "known_startup_warning"})
                else:
                    result["fatal_events"].append({**entry, "kind": "unrecognized_native_error"})
            else:
                result["fatal_events"].append({**entry, "kind": "unknown_item_type"})
        elif event_type not in controls:
            result["fatal_events"].append({**entry, "kind": "unknown_event_type"})
        if event_type == "turn.started":
            started = True
        if event_type == "process.completed" and event.get("exit_code", 0) != 0:
            result["fatal_events"].append({**entry, "kind": "native_process_failed"})
    return result


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identifier(value, name):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a short filesystem-safe identifier")


def _read_immutable(path):
    envelope = json.loads(Path(path).read_text())
    if envelope.get("schema_version") != 1 or envelope.get("sha256") != _digest(envelope["payload"]):
        raise CurateProtocolError("Curator journal content failed its immutable hash check")
    return envelope["payload"]


def _write_immutable(path, payload):
    payload = json.loads(_canonical(payload))
    path = Path(path)
    if path.exists():
        if _read_immutable(path) != payload:
            raise CurateProtocolError("Curator identity was reused for different immutable data")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".curate-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema_version": 1, "sha256": _digest(payload), "payload": payload}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_immutable(path) != payload:
                raise CurateProtocolError("Concurrent curator journal identity conflict") from None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


@contextlib.contextmanager
def _collection_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(directory / ".collector.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CurateRecoveryRequired("This deterministic curator collection is already active") from error
        yield
    finally:
        os.close(fd)


def _redact_text(value):
    if not isinstance(value, str):
        raise CurateProtocolError("Public environment trajectory text must be a string")
    value = re.sub(r"https?://[^\s\"'<>]+", "[REDACTED_URL]", value, flags=re.IGNORECASE)
    value = re.sub(r"(?<!\w)/(?:[^\s/\"'<>]+/)*[^\s\"'<>]*", "[REDACTED_PATH]", value)
    return re.sub(
        r"(?i)\b(?:api[_-]?key|admin[_-]?token|access[_-]?token|password|authorization)\s*[:=]\s*\S+",
        "[REDACTED_CREDENTIAL]",
        value,
    )


def _completed_episode_identity(source):
    """Validate the shared source identity before action or censorship checks."""
    if source.get("split") != "train" or source.get("dataset_phase") != "train" or source.get("finished") is not True:
        raise CurateProtocolError("Curator generation accepts completed official training episodes only")
    family = source.get("task_type")
    if family not in TASK_TYPES:
        raise CurateProtocolError("Source must identify a supported official ALFWorld task family")
    for key in ("game_id", "stable_game_id", "attempt_id"):
        _identifier(source.get(key), key)
    if source["game_id"] != source["stable_game_id"]:
        raise CurateProtocolError("Source game identities disagree")
    for key in ("game_sha256", "catalog_digest"):
        if not isinstance(source.get(key), str) or not _SHA256.fullmatch(source[key]):
            raise CurateProtocolError("Source requires immutable game/catalog content identities")
    relative = validate_relative_game_path(source["relative_game_path"])
    if relative.split("/")[1] != "train":
        raise CurateProtocolError("Source game path does not belong to the official training partition")
    won = source.get("won")
    if type(won) is not bool or source.get("reward") != float(won) or source.get("accuracy") != float(won):
        raise CurateProtocolError("Source reward/accuracy must match the authoritative boolean environment outcome")
    return family, won


def redact_completed_episode(source):
    """Copy only observed train actions/states/outcomes, never expert/private data."""
    source = json.loads(_canonical(source))
    family, won = _completed_episode_identity(source)
    initial = source.get("initial_state", {})
    steps = []
    for index, entry in enumerate(source.get("transitions", []), 1):
        after = entry.get("after", {})
        if (
            entry.get("step") != index
            or type(entry.get("step")) is not int
            or type(entry.get("action_in_exposed_commands")) is not bool
            or type(after.get("won")) is not bool
            or type(after.get("done")) is not bool
        ):
            raise CurateProtocolError("Source environment transitions must be ordered public engine states")
        steps.append(
            {
                "step": index,
                "action": _redact_text(entry["action"]),
                "observation": _redact_text(after["observation"]),
                "action_was_admissible": bool(entry["action_in_exposed_commands"]),
                "won": after["won"],
                "done": after["done"],
            }
        )
    if not steps or source.get("steps") != len(steps) or steps[-1]["won"] != won:
        raise CurateProtocolError("A complete nonempty observed solver action sequence is required")
    max_steps = source.get("max_steps")
    if type(max_steps) is not int or not 1 <= len(steps) <= max_steps:
        raise CurateProtocolError("The source trajectory exceeds its authoritative environment budget")
    return {
        "schema_version": 1,
        "task_family": family,
        "initial_observation": _redact_text(initial["observation"]),
        "steps": steps,
        "outcome": {"won": won, "reward": float(won), "steps": len(steps), "max_steps": max_steps},
    }


def load_completed_solver_episode(directory):
    """Host-only convenience loader for one completed native ALFWorld attempt."""
    directory = Path(directory)
    paths = [
        directory / name
        for name in (
            "result.json",
            "engine-result.json",
            "initial-state.json",
            "environment-trace.jsonl",
        )
    ]
    result, engine, initial = [json.loads(path.read_text()) for path in paths[:3]]
    for key in ("game_id", "game_sha256", "catalog_digest", "split", "won", "done", "reward", "steps", "max_steps"):
        if result.get(key) != engine.get(key):
            raise CurateProtocolError("Completed solver artifacts disagree with the authoritative engine result")
    if type(result.get("steps")) is int and result["steps"] == 0:
        source = json.loads(_canonical({**result, "finished": True, "initial_state": initial}))
        _completed_episode_identity(source)
        if (
            type(engine.get("steps")) is not int
            or type(engine.get("max_steps")) is not int
            or engine["max_steps"] < 1
            or type(engine.get("done")) is not bool
            or not isinstance(initial, dict)
            or any(
                type(initial.get(key)) is not type(engine.get(key)) or initial.get(key) != engine.get(key)
                for key in ("steps", "max_steps", "won", "done")
            )
        ):
            raise CurateProtocolError("Zero-action source must preserve its authoritative initial world state")
        _redact_text(initial.get("observation"))
        trace_present = os.path.lexists(paths[3])
        if trace_present and any(line.strip() for line in paths[3].read_text().splitlines()):
            raise CurateProtocolError("Zero-action source cannot contain environment trace entries")
        raise CurateSourceCensored(
            {
                "game_id": result["game_id"],
                "attempt_id": result["attempt_id"],
                "steps": 0,
                "trace_present": trace_present,
                "artifact_sha256": {path.name: _file_hash(path) for path in paths if path != paths[3] or trace_present},
            }
        )
    source = {
        **result,
        "finished": True,
        "initial_state": initial,
        "transitions": [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()],
        "artifact_sha256": {path.name: _file_hash(path) for path in paths},
    }
    redact_completed_episode(source)
    return source


@dataclass(frozen=True)
class CurateRequest:
    producer_uid: str
    skill_snapshot_id: str
    candidate_group_id: str
    candidate_index: int
    candidate_group_size: int
    created_at: float

    def __post_init__(self):
        for key in ("producer_uid", "candidate_group_id"):
            _identifier(getattr(self, key), key)
        if not isinstance(self.skill_snapshot_id, str) or not _SHA256.fullmatch(self.skill_snapshot_id):
            raise ValueError("skill_snapshot_id must identify an immutable SkillStore snapshot")
        if type(self.candidate_group_size) is not int or self.candidate_group_size < 1:
            raise ValueError("candidate_group_size must be positive")
        if type(self.candidate_index) is not int or not 0 <= self.candidate_index < self.candidate_group_size:
            raise ValueError("candidate_index must belong to the original candidate group")
        if isinstance(self.created_at, bool) or not math.isfinite(self.created_at) or self.created_at < 0:
            raise ValueError("created_at must preserve a finite original collection timestamp")

    @property
    def collection_id(self):
        return _digest([self.producer_uid, self.candidate_group_id, self.candidate_index])

    @property
    def record_id(self):
        return "curate-" + self.collection_id

    @property
    def skill_id(self):
        return "skill-" + self.collection_id[:40]

    @property
    def session_id(self):
        return "curate-" + self.collection_id


@dataclass(frozen=True)
class CuratorBehavior:
    policy_version: str
    rollout_weight_version: int
    tokenizer_version: str
    template_version: str
    gateway_policy_name: str | None = None
    logprobs_mode: str = "processed_logprobs"
    # The gateway always allows max_tokens/stop; probability-changing overrides
    # must not be enabled for this dedicated curator gateway.
    request_sampling_param_keys: tuple[str, ...] = ()

    def __post_init__(self):
        for key in ("policy_version", "tokenizer_version", "template_version"):
            _identifier(getattr(self, key), key)
        if type(self.rollout_weight_version) is not int or self.rollout_weight_version < 0:
            raise ValueError("A nonnegative observed EMA rollout weight clock is required")
        if self.logprobs_mode != "processed_logprobs":
            raise ValueError("Curator collection requires the backend's actual processed behavior logprobs")
        forbidden = set(self.request_sampling_param_keys) - {"max_tokens", "stop"}
        if forbidden:
            raise ValueError("The dedicated EMA gateway must not allow native request sampling overrides")
        object.__setattr__(self, "request_sampling_param_keys", tuple(self.request_sampling_param_keys))

    def sampling_config(self):
        return {
            **SAMPLING,
            "logprobs_mode": self.logprobs_mode,
            "request_sampling_param_keys": list(self.request_sampling_param_keys),
        }


@dataclass(frozen=True)
class NativeCuratorRuntime:
    model_name: str
    unix_user: str
    uid: int
    gid: int
    work_root: str
    codex_bin: str
    package_root: str
    model_catalog_path: str
    instructions_path: str
    task_prompt_path: str
    schema_path: str
    native_version: str
    template_version: str
    run_timeout: float = 300.0

    def manifest(self):
        return {
            "native_version": self.native_version,
            "template_version": self.template_version,
            "model_name": self.model_name,
            "unix_user": self.unix_user,
            "disabled_features": list(DISABLED_FEATURES),
            "native_config_overrides": list(NATIVE_CONFIG_OVERRIDES),
            "native_output_schema_enabled": False,
            "isolation": "dedicated_nonroot_account; clean_environment; no_game_session_capability",
            "files": {
                key: _file_hash(getattr(self, key))
                for key in (
                    "codex_bin",
                    "model_catalog_path",
                    "instructions_path",
                    "task_prompt_path",
                    "schema_path",
                )
            },
        }


def _public_file(path, text, mode=0o644):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise CurateProtocolError("A versioned native curator runtime file was modified")
        return
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(fd, "w") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    path.chmod(mode)


def _runtime_directory(path, *, public):
    """Repair traversal only inside explicit public/runtime roots, never NAS."""
    path = Path(path).resolve()
    anchors = (Path("/opt/alfworld-tools"), Path("/tmp").resolve())
    anchor = next((root for root in anchors if path == root or root in path.parents), None)
    if anchor is None or path == anchor:
        raise CurateProtocolError("Native runtime trees must be children of /opt/alfworld-tools or /tmp")
    path.mkdir(parents=True, exist_ok=True, mode=0o755 if public else 0o711)
    nodes = [node for node in [path, *path.parents] if node != anchor and anchor in node.parents]
    # /tmp retains its sticky permissions. /opt/alfworld-tools is the explicit
    # public tooling root and may itself have been created under umask 077.
    if anchor == Path("/opt/alfworld-tools"):
        nodes.append(anchor)
    for node in reversed(nodes):
        node.chmod((node.stat().st_mode & 0o777) | (0o055 if public else 0o011))
    path.chmod(0o755 if public else 0o711)
    return path


def provision_native_curator_runtime(
    *,
    public_root,
    work_root,
    native_bin,
    package_root,
    model_catalog_path,
    model_name,
    unix_user="alfworld_curator",
    run_timeout=300.0,
    prompt_root=None,
):
    """Provision public instructions/wrapper; root provisions the Unix account.

    No model server or environment is started. The only subprocess is native
    ``--version``. Runtime parent directories must be traversable by the account.
    The existing solver account must never be reused for curator execution.
    """
    account = pwd.getpwnam(unix_user)
    if account.pw_uid == 0:
        raise CurateProtocolError("Native curator must never run as root")
    try:
        solver_uid = pwd.getpwnam("alfworld_eval").pw_uid
    except KeyError:
        solver_uid = None
    if account.pw_uid == solver_uid:
        raise CurateProtocolError("Curator requires an account distinct from the solver account")
    native_bin, package_root = Path(native_bin).resolve(), Path(package_root).resolve()
    if not native_bin.is_file() or not os.access(native_bin, os.X_OK) or not package_root.is_dir():
        raise FileNotFoundError("The managed native Codex installation is unavailable")
    native_version = (
        subprocess.check_output([str(native_bin), "--version"], text=True).strip().removeprefix("codex-cli ")
    )
    if native_version != NATIVE_CODEX_VERSION:
        raise CurateProtocolError(
            "Inspect native feature compatibility before changing the pinned curator Codex version"
        )
    prompts = (
        Path(prompt_root) if prompt_root else Path(__file__).resolve().parents[3] / "examples/skill_self_evolve/prompts"
    )
    system, task = [(prompts / filename).read_text() for filename in ("curator_system.md", "curator_task.txt")]
    if "{{SKILL_ID}}" not in task or "{{TASK_FAMILY}}" not in task or "{{EPISODE_JSON}}" not in task:
        raise CurateProtocolError("The curator task template is missing its explicit data placeholders")
    catalog = json.loads(Path(model_catalog_path).read_text())
    models = [row for row in catalog["models"] if row["slug"] == model_name]
    if len(models) != 1:
        raise CurateProtocolError("The native model catalog must identify one exact curator model")
    model = copy.deepcopy(models[0])
    # Native built-in tool declarations can remain even with execution
    # features disabled. False is supported only for an empty effective set.
    model.update(base_instructions=system, supports_parallel_tool_calls=True, experimental_supported_tools=[])
    template_version = "sha256:" + _digest(
        {
            "system": system,
            "task": task,
            "schema": CURATE_ACTION_SCHEMA,
            "model": model,
            "native_version": native_version,
            "disabled_features": DISABLED_FEATURES,
            "native_config_overrides": NATIVE_CONFIG_OVERRIDES,
        }
    )
    public = _runtime_directory(Path(public_root) / template_version.removeprefix("sha256:"), public=True)
    for parent in [public, *public.parents]:
        if not parent.stat().st_mode & 0o001:
            raise CurateProtocolError("Public curator runtime parents must permit account traversal")
    instructions, task_path, schema_path = (
        public / "curator_system.md",
        public / "curator_task.txt",
        public / "schema.json",
    )
    catalog_path, wrapper_path = public / "model_catalog.json", public / "codex-curator"
    _public_file(instructions, system)
    _public_file(task_path, task)
    _public_file(schema_path, json.dumps(CURATE_ACTION_SCHEMA, indent=2) + "\n")
    _public_file(catalog_path, json.dumps({"models": [model]}, indent=2) + "\n")
    command = [
        str(native_bin),
        "-c",
        "model_instructions_file=" + json.dumps(str(instructions)),
        "-c",
        "check_for_update_on_startup=false",
        "--enable",
        "skip_host_skill_discovery",
    ]
    for override in NATIVE_CONFIG_OVERRIDES:
        command.extend(["-c", override])
    for feature in DISABLED_FEATURES:
        command.extend(["--disable", feature])
    command.extend(["exec", "--ignore-user-config", "--ignore-rules"])
    wrapper = (
        "#!/bin/sh\nset -eu\n"
        "if [ ! -r "
        + shlex.quote(str(instructions))
        + " ] || [ ! -r "
        + shlex.quote(str(catalog_path))
        + " ]; then echo 'Curator runtime files are not readable.' >&2; exit 78; fi\n"
        'if [ "${1:-}" = --version ]; then exec ' + shlex.quote(str(native_bin)) + " --version; fi\n"
        "if [ \"${1:-}\" != exec ]; then echo 'Curator wrapper only supports native exec.' >&2; exit 78; fi\n"
        'shift\n: "${CURATOR_CODEX_HOME:?An isolated curator home is required}"\n'
        'export CODEX_HOME="$CURATOR_CODEX_HOME"\nexec ' + shlex.join(command) + ' "$@"\n'
    )
    _public_file(wrapper_path, wrapper, mode=0o755)
    work_root = _runtime_directory(work_root, public=False)
    probe_version = (
        subprocess.check_output(
            [
                "/usr/sbin/runuser",
                "-u",
                unix_user,
                "--",
                "/usr/bin/env",
                "-i",
                "PATH=/usr/bin:/bin",
                str(wrapper_path),
                "--version",
            ],
            text=True,
        )
        .strip()
        .removeprefix("codex-cli ")
    )
    if probe_version != native_version:
        raise CurateProtocolError("The dedicated account could not run the real native wrapper and instruction files")
    return NativeCuratorRuntime(
        model_name,
        unix_user,
        account.pw_uid,
        account.pw_gid,
        str(work_root),
        str(wrapper_path),
        str(package_root),
        str(catalog_path),
        str(instructions),
        str(task_path),
        str(schema_path),
        native_version,
        template_version,
        run_timeout,
    )


class CuratorSandbox:
    """Minimal native-agent transport with no inherited solver/environment state.

    This is an account and capability boundary, not a container/network namespace.
    Known tool features are disabled and any emitted native tool event is rejected
    before replay admission. The account still has ordinary public host access.
    """

    def __init__(self, runtime, output_dir, collection_id):
        if runtime.uid == 0 or pwd.getpwnam(runtime.unix_user).pw_uid != runtime.uid:
            raise CurateProtocolError("The provisioned non-root curator account changed")
        try:
            if pwd.getpwnam("alfworld_eval").pw_uid == runtime.uid:
                raise CurateProtocolError("The native curator must not reuse the solver account")
        except KeyError:
            pass
        self.runtime, self.output_dir = runtime, Path(output_dir)
        self.collection_id = collection_id
        self.workdir = Path(runtime.work_root) / collection_id
        self.workdir.mkdir(mode=0o700)
        os.chown(self.workdir, runtime.uid, runtime.gid)
        self.home = self.workdir / ".home"
        self.home.mkdir(mode=0o700)
        os.chown(self.home, runtime.uid, runtime.gid)
        self.created_files = []
        self.native_bootstrap_homes = []
        self.tool_events_seen = False
        self.native_event_classification = None
        self.process = None

    async def write_file(self, path, content):
        path = Path(path)
        if path.parent != Path("/tmp") or not re.fullmatch(r"codex-model-catalog-[0-9a-f]{32}\.json", path.name):
            raise CurateProtocolError("Native curator may stage only its isolated temporary model catalog")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.created_files.append(path)  # Retain ownership even if the following write fails partway through.
        with os.fdopen(fd, "wb") as stream:
            stream.write(content.encode() if isinstance(content, str) else content)
        os.chown(path, self.runtime.uid, self.runtime.gid)

    def process_environment(self):
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.home),
            "CURATOR_CODEX_HOME": str(self.home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "CODEX_MANAGED_PACKAGE_ROOT": self.runtime.package_root,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }

    async def _terminate(self):
        if self.process is None or self.process.returncode is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()

    async def exec_shell(self, script, *, timeout=None, workdir=None, env=None):
        from uni_agent.sandbox.base import ExecResult

        if env or Path(workdir).resolve() != self.workdir:
            raise CurateProtocolError("Native curator transport cannot inherit caller environment or other workspaces")
        for token in shlex.split(script):
            if re.fullmatch(r"CODEX_HOME=/tmp/codex-home-[0-9a-f]{32}", token):
                home = token.split("=", 1)[1]
                if Path(home).exists() or Path(home).is_symlink():
                    raise CurateProtocolError("Native curator must start with a fresh bootstrap Codex home")
                self.native_bootstrap_homes.append(home)
        self.process = await asyncio.create_subprocess_exec(
            "/usr/sbin/runuser",
            "-u",
            self.runtime.unix_user,
            "--",
            "/bin/bash",
            "-c",
            script,
            cwd=self.workdir,
            env=self.process_environment(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def capture(reader, name):
            chunks = []
            with (self.output_dir / name).open("ab") as output:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    output.write(chunk)
                    output.flush()
                    chunks.append(chunk)
            return b"".join(chunks)

        readers = [
            asyncio.create_task(capture(self.process.stdout, "codex-events.jsonl")),
            asyncio.create_task(capture(self.process.stderr, "codex-stderr.log")),
        ]
        timed_out = False
        try:
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await self._terminate()
        except BaseException:
            await self._terminate()
            await asyncio.gather(*readers)
            raise
        stdout, stderr = await asyncio.wait_for(asyncio.gather(*readers), timeout=10)
        self.native_event_classification = classify_native_events(
            stdout.decode(errors="replace"),
            expected_codex_home=self.home,
        )
        self.tool_events_seen = bool(self.native_event_classification["tool_events"])
        _write_immutable(self.output_dir / "native-events-classified.json", self.native_event_classification)
        return ExecResult(
            -1 if timed_out else self.process.returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def close(self):
        await self._terminate()
        # Preserve logs/workspaces, but prevent later curator sessions reusing
        # another episode's files as external memory.
        from .alfworld.native_files import archive_native_files

        archive_native_files(
            self.output_dir, codex_homes=getattr(self, "native_bootstrap_homes", ()),
            model_catalogs=self.created_files, agent_uid=self.runtime.uid,
            curator_workdir=self.workdir, curator_root=self.runtime.work_root, curator_collection_id=self.collection_id,
        )


def _native_agent(runtime, handle):
    from uni_agent.agents.codex.agent import CodexAgent, CodexConfig

    return CodexAgent(
        CodexConfig(
            name="codex",
            model={"base_url": handle.base_url, "model_name": runtime.model_name},
            run_timeout=runtime.run_timeout,
            model_catalog_path=runtime.model_catalog_path,
            codex_bin=runtime.codex_bin,
        )
    )


def _validate_session(handle, session_id):
    url = urlsplit(handle.base_url or "")
    try:
        local = ipaddress.ip_address(url.hostname).is_private
    except ValueError as error:
        raise CurateProtocolError("EMA gateway must use a literal private/local address") from error
    if (
        url.scheme != "http"
        or not local
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path != "/sessions/" + session_id + "/v1"
        or handle.session_id != session_id
    ):
        raise CurateProtocolError("The returned native session is not the requested local EMA gateway route")


def _capture_trajectory(trajectory):
    extras = trajectory.extra_fields or {}
    result = {
        "prompt_ids": list(trajectory.prompt_ids),
        "action_ids": list(trajectory.response_ids),
        "curate_mask": list(trajectory.response_mask),
        "behavior_logprobs": None if trajectory.response_logprobs is None else list(trajectory.response_logprobs),
        "observed_min_weight_version": extras.get("min_global_steps"),
        "observed_max_weight_version": extras.get("max_global_steps"),
        "chain_id": trajectory.chain_id,
        "materialization_reason": extras.get("materialization_reason"),
    }
    positions = extras.get("native_position_ids", extras.get("position_ids"))
    if positions is not None:
        result["native_position_ids"] = positions.tolist() if hasattr(positions, "tolist") else positions
    return result


def _validate_capture(tokens, behavior):
    if (
        tokens["observed_min_weight_version"] != behavior.rollout_weight_version
        or tokens["observed_max_weight_version"] != behavior.rollout_weight_version
        or type(tokens["observed_min_weight_version"]) is not int
        or type(tokens["observed_max_weight_version"]) is not int
    ):
        raise CurateProtocolError(
            "Actual gateway weight versions are absent, mixed, or differ from the pinned EMA version"
        )
    if tokens["materialization_reason"] == "max_trajectory_length":
        raise CurateProtocolError("A capacity-truncated curator action is not a complete trainable candidate")
    if tokens["behavior_logprobs"] is None:
        raise CurateProtocolError("Actual processed behavior logprobs are required; recomputation is forbidden")
    if not tokens["prompt_ids"] or not tokens["action_ids"] or not any(tokens["curate_mask"]):
        raise CurateProtocolError("Initial role loss requires a complete nonempty authored curate action")
    prompt, actions, mask, probabilities = (
        tokens[key] for key in ("prompt_ids", "action_ids", "curate_mask", "behavior_logprobs")
    )
    if (
        len(actions) != len(mask)
        or len(actions) != len(probabilities)
        or any(type(token) is not int or token < 0 for token in [*prompt, *actions])
        or any(type(selected) is not int or selected not in (0, 1) for selected in mask)
        or any(
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or probability > 0
            or (not selected and probability != 0)
            for selected, probability in zip(mask, probabilities, strict=True)
        )
    ):
        raise CurateProtocolError(
            "Curator action tokens, masks and processed behavior logprobs must be aligned and valid"
        )


def _parse_abstention(content):
    """Recognize a valid model abstention without inventing a SkillArtifact."""
    try:
        value = json.loads(content)
    except ValueError:
        return None  # The existing skill parser supplies the malformed-JSON error.
    if not isinstance(value, dict) or value.get("decision") != "abstain":
        return None
    if set(value) != {"decision", "reason"}:
        raise CurateProtocolError("Curator abstention must contain only decision and reason")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > ABSTENTION_REASON_MAX_CHARS:
        raise CurateProtocolError("Curator abstention reason must be nonempty text of at most 2048 characters")
    if re.search(r"https?://|(?<!\w)/(?:\w+/)+", reason):
        raise CurateProtocolError("Curator abstention reason must not contain host paths or network endpoints")
    return reason


def _parse_artifact(content, *, request, family, provenance):
    try:
        value = json.loads(content)
    except ValueError as error:
        raise CurateProtocolError("Curator must return exactly one plain JSON SkillArtifact object") from error
    if not isinstance(value, dict) or set(value) != SKILL_FIELDS:
        raise CurateProtocolError("Curator JSON must contain only the five allowed skill fields")
    if value["skill_id"] != request.skill_id or value["task_families"] != [family]:
        raise CurateProtocolError("Generated skill identity/family differs from the trusted curator request")
    if any(re.search(r"https?://|(?<!\w)/(?:\w+/)+", value[key]) for key in ("title", "principle", "when_to_apply")):
        raise CurateProtocolError("Reusable skill text must not contain host paths or network endpoints")
    return SkillArtifact(**value, provenance=provenance)


def _find_or_stage(store, artifact, branch, baseline):
    if not (store.root / "refs" / (branch + ".json")).exists():
        store.create_branch(branch, from_snapshot=baseline)
    matches = [store.get_candidate(path.stem) for path in sorted((store.root / "candidates").glob("*.json"))]
    matches = [item for item in matches if item.branch == branch and item.artifact == artifact.ref]
    if matches:
        return matches[0]
    if store.head(branch) != baseline:
        raise CurateProtocolError("Unacknowledged candidate branch changed before its durable replay commit")
    candidate = store.stage_candidate(artifact, branch=branch)
    if candidate.base_snapshot != baseline:
        raise CurateProtocolError("Candidate staged against another baseline skill snapshot")
    return candidate


@dataclass(frozen=True)
class CurateCollection:
    collection_id: str
    session_id: str
    record_id: str
    candidate: SkillCandidate
    artifact: SkillArtifact
    replay_ref: ReplayRef
    output_dir: str
    # The learner must register the credit window before clearing source/TQ data.
    credit_registration_required: bool = True


@dataclass(frozen=True)
class CurateAbstention:
    """A retained neutral model action, without a candidate or skill replay ref."""

    collection_id: str
    session_id: str
    reason: str
    action_record_path: str
    action_record_sha256: str
    output_dir: str
    credit_registration_required: bool = False


def _completed_abstention(path, value, *, input_hash):
    action_path = path.parent / "abstention.json"
    if (
        value.get("action_record_path") != str(action_path)
        or value.get("action_record_sha256") != _file_hash(action_path)
        or value.get("credit_registration_required") is not False
    ):
        raise CurateProtocolError("Completed abstention acknowledgement disagrees with its durable action record")
    record = _read_immutable(action_path)
    capture_path = path.parent / "generated.json"
    captured = _read_immutable(capture_path)
    if (
        record.get("kind") != "curate_abstention_action"
        or record.get("input_hash") != input_hash
        or record.get("collection_id") != value["collection_id"]
        or record.get("session_id") != value["session_id"]
        or record.get("reason") != value["reason"]
        or record.get("decision") != "abstain"
        or record.get("generated_capture_sha256") != _file_hash(capture_path)
        or captured.get("input_hash") != input_hash
        or captured.get("trajectories") != [record.get("trajectory")]
        or captured.get("content") != record.get("content")
        or _parse_abstention(record.get("content", "")) != record.get("reason")
        or record.get("reward_event_created") is not False
        or record.get("training_label_created") is not False
    ):
        raise CurateProtocolError("Completed abstention differs from the retained native capture")
    _validate_capture(record["trajectory"], CuratorBehavior(**record["behavior"]))
    return CurateAbstention(
        value["collection_id"], value["session_id"], value["reason"], str(action_path),
        value["action_record_sha256"], str(path.parent),
    )


def _completed_collection(path, *, input_hash, store, replay):
    value = _read_immutable(path)
    if value["input_hash"] != input_hash:
        raise CurateProtocolError("Completed curate identity does not match the requested source/behavior")
    if value.get("decision") == "abstain":
        return _completed_abstention(path, value, input_hash=input_hash)
    candidate = store.get_candidate(value["candidate_id"])
    artifact = store.load_artifact(candidate.artifact)
    ref = replay.get_ref(value["record_id"])
    record = replay.get(value["record_id"])
    if ref.to_dict() != value["replay_ref"] or dict(record.skill_ref) != artifact.ref.to_dict():
        raise CurateProtocolError("Completed curate acknowledgement disagrees with durable skill/replay data")
    return CurateCollection(
        value["collection_id"], value["session_id"], value["record_id"], candidate, artifact, ref, str(path.parent)
    )


async def collect_curate_candidate(
    *,
    gateway_manager,
    behavior: CuratorBehavior,
    source_episode,
    store: SkillStore,
    replay: CurateReplay,
    request: CurateRequest,
    runtime: NativeCuratorRuntime,
    output_root,
    _agent_factory=None,
    _sandbox_factory=None,
    before_commit=None,
):
    """Retain one actual native EMA skill or abstention action and its old mu.

    The underscore factories exist for offline protocol tests. Production uses
    unchanged Uni-Agent CodexAgent and the dedicated non-root transport above.
    The caller owns the EMA manager, synchronization barrier, and credit window.
    No model or environment is initialized here, including on retry.
    """
    source_episode = json.loads(_canonical(source_episode))
    redacted = redact_completed_episode(source_episode)
    if runtime.uid == 0 or runtime.template_version != behavior.template_version:
        raise CurateProtocolError("A matching native prompt template and non-root curator runtime are required")
    store.get_snapshot(request.skill_snapshot_id)
    runtime_manifest = runtime.manifest()
    source_hash, redacted_hash = _digest(source_episode), _digest(redacted)
    template = Path(runtime.task_prompt_path).read_text()
    prompt = template.replace("{{SKILL_ID}}", request.skill_id).replace("{{TASK_FAMILY}}", redacted["task_family"])
    prompt = prompt.replace("{{EPISODE_JSON}}", json.dumps(redacted, ensure_ascii=False, indent=2))
    messages = [{"role": "user", "content": prompt}]
    identity = {
        "collector_version": COLLECTOR_VERSION,
        "request": asdict(request),
        "behavior": asdict(behavior),
        "runtime": runtime_manifest,
        "source_sha256": source_hash,
        "redacted_source_sha256": redacted_hash,
        "messages_sha256": _digest(messages),
    }
    input_hash = _digest(identity)
    directory = Path(output_root).resolve() / request.collection_id
    with _collection_lock(directory):
        complete = directory / "complete.json"
        if complete.exists():
            return _completed_collection(complete, input_hash=input_hash, store=store, replay=replay)
        _write_immutable(directory / "intent.json", {"input_hash": input_hash, **identity})
        _write_immutable(directory / "redacted-source.json", redacted)
        _write_immutable(directory / "agent-messages.json", messages)
        captured_path = directory / "generated.json"
        if not captured_path.exists():
            launch = directory / "launch-intent.json"
            if launch.exists():
                raise CurateRecoveryRequired(
                    "A previous native launch lacks durable generation capture; reconcile its session before retry"
                )
            _write_immutable(launch, {"input_hash": input_hash, "session_id": request.session_id})
            sandbox = (_sandbox_factory or CuratorSandbox)(runtime, directory, request.collection_id)
            session_created = False
            try:
                arguments = dict(
                    metadata={
                        "role": "curator",
                        "behavior_policy_version": behavior.policy_version,
                        "producer_uid": request.producer_uid,
                        "collection_id": request.collection_id,
                    },
                    sampling_params=dict(SAMPLING),
                )
                if behavior.gateway_policy_name is not None:
                    arguments["policy_name"] = behavior.gateway_policy_name
                handle = await gateway_manager.create_session(request.session_id, **arguments)
                session_created = True
                _validate_session(handle, request.session_id)
                agent = (_agent_factory or _native_agent)(runtime, handle)
                agent_result = await agent.run(sandbox=sandbox, messages=messages, workdir=str(sandbox.workdir))
                trajectories = await gateway_manager.finalize_session(request.session_id)
                session_created = False
                # Persist all returned segments even if the initial loss later
                # rejects their count. Failed generations are never resampled.
                captures = [_capture_trajectory(trajectory) for trajectory in trajectories]
                _write_immutable(
                    captured_path,
                    {
                        "input_hash": input_hash,
                        "session_id": request.session_id,
                        "trajectories": captures,
                        "content": (agent_result.output or {}).get("content", ""),
                        "agent_ok": (agent_result.info or {}).get("ok") is True,
                        "agent_finished": agent_result.finished is True,
                        "tool_events_seen": sandbox.tool_events_seen,
                        "native_event_classification": getattr(sandbox, "native_event_classification", None),
                    },
                )
            except BaseException as error:
                _write_immutable(
                    directory / "launch-failure.json",
                    {"input_hash": input_hash, "session_id": request.session_id, "error_type": type(error).__name__},
                )
                raise
            finally:
                primary_error = sys.exc_info()[0] is not None
                if session_created:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(gateway_manager.abort_session(request.session_id))
                try:
                    await sandbox.close()
                except Exception as cleanup_error:
                    with contextlib.suppress(Exception):
                        _write_immutable(directory / "cleanup-failure.json", {
                            "error_type": type(cleanup_error).__name__, "primary_error_retained": primary_error,
                        })
                    if not primary_error:
                        raise
        captured = _read_immutable(captured_path)
        if captured["input_hash"] != input_hash:
            raise CurateProtocolError("Captured curate behavior/source differs from this retry")
        event_classification = captured.get("native_event_classification") or {}
        if event_classification.get("fatal_events"):
            raise CurateNativeExecutionError(
                "Native Codex emitted an unknown execution error; see retained event classification"
            )
        if captured["agent_ok"] is not True or captured["agent_finished"] is not True or captured["tool_events_seen"]:
            raise CurateProtocolError("Curator must complete successfully without any native tool invocation")
        if len(captured["trajectories"]) != 1:
            raise CurateProtocolError(
                "Initial role loss requires exactly one complete nonempty TITO trajectory per candidate"
            )
        tokens = captured["trajectories"][0]
        _validate_capture(tokens, behavior)
        provenance = {
            "collector_version": COLLECTOR_VERSION,
            "curate_record_id": request.record_id,
            "producer_uid": request.producer_uid,
            "source_redacted_sha256": redacted_hash,
            "source_sha256": source_hash,
            "behavior_policy_version": behavior.policy_version,
            "observed_ema_weight_version": behavior.rollout_weight_version,
            "skill_snapshot_id": request.skill_snapshot_id,
            "candidate_group_id": request.candidate_group_id,
            "candidate_index": request.candidate_index,
            "session_id": request.session_id,
            "prompt_token_ids_sha256": _digest(tokens["prompt_ids"]),
            "action_token_ids_sha256": _digest(tokens["action_ids"]),
        }
        abstention_reason = _parse_abstention(captured["content"])
        context_snapshot = {
            "collector_version": COLLECTOR_VERSION,
            "input_hash": input_hash,
            "source_redacted_sha256": redacted_hash,
            "source_sha256": source_hash,
            "messages_sha256": _digest(messages),
            "runtime": runtime_manifest,
            "skill_snapshot_id": request.skill_snapshot_id,
            "session_id": request.session_id,
            "observed_min_weight_version": tokens["observed_min_weight_version"],
            "observed_max_weight_version": tokens["observed_max_weight_version"],
            "actual_logprobs_source": "EMA_gateway_TITO; never_recomputed",
            "native_event_classification": event_classification,
            "task_metadata": {
                "official_split": "train",
                "dataset_phase": "train",
                "producer_uid": request.producer_uid,
                **{
                    key: source_episode[key]
                    for key in (
                        "game_id",
                        "stable_game_id",
                        "task_type",
                        "scene_id",
                        "relative_game_path",
                        "game_sha256",
                        "catalog_digest",
                        "attempt_id",
                        "global_step",
                        "parameter_version",
                    )
                    if key in source_episode
                },
            },
            **({"native_position_ids": tokens["native_position_ids"]} if "native_position_ids" in tokens else {}),
        }
        if abstention_reason is not None:
            action_record = {
                "schema_version": 1,
                "kind": "curate_abstention_action",
                "decision": "abstain",
                "reason": abstention_reason,
                "input_hash": input_hash,
                "collection_id": request.collection_id,
                "session_id": request.session_id,
                "request": asdict(request),
                "behavior": asdict(behavior),
                "sampling_config": behavior.sampling_config(),
                "context_snapshot": context_snapshot,
                "trajectory": tokens,
                "content": captured["content"],
                "generated_capture_sha256": _file_hash(captured_path),
                "reward_event_created": False,
                "training_label_created": False,
                "credit_registration_required": False,
            }
        else:
            artifact = _parse_artifact(
                captured["content"], request=request, family=redacted["task_family"], provenance=provenance
            )
            record = CurateReplayRecord(
                record_id=request.record_id,
                producer_uid=request.producer_uid,
                skill_ref=artifact.ref,
                prompt_ids=tokens["prompt_ids"],
                action_ids=tokens["action_ids"],
                curate_mask=tokens["curate_mask"],
                behavior_logprobs=tokens["behavior_logprobs"],
                behavior_policy_version=behavior.policy_version,
                sampling_config=behavior.sampling_config(),
                tokenizer_version=behavior.tokenizer_version,
                template_version=behavior.template_version,
                context_snapshot=context_snapshot,
                skill_snapshot_id=request.skill_snapshot_id,
                candidate_group_id=request.candidate_group_id,
                candidate_index=request.candidate_index,
                candidate_group_size=request.candidate_group_size,
                created_at=request.created_at,
                trajectory_index=0,
                trajectory_count=1,
            )
        if before_commit is not None:
            permission = before_commit()
            if inspect.isawaitable(permission):
                await permission
        # No await after the gate: a checkpoint cannot interleave with this
        # synchronous store/replay or neutral-action acknowledgement commit.
        if abstention_reason is not None:
            action_path = directory / "abstention.json"
            _write_immutable(action_path, action_record)
            acknowledgment = {
                "decision": "abstain",
                "input_hash": input_hash,
                "collection_id": request.collection_id,
                "session_id": request.session_id,
                "reason": abstention_reason,
                "action_record_path": str(action_path),
                "action_record_sha256": _file_hash(action_path),
                "credit_registration_required": False,
            }
            _write_immutable(complete, acknowledgment)
            return CurateAbstention(
                request.collection_id, request.session_id, abstention_reason, str(action_path),
                acknowledgment["action_record_sha256"], str(directory),
            )
        candidate = _find_or_stage(store, artifact, "curate-" + request.collection_id, request.skill_snapshot_id)
        ref = replay.append(record)
        _write_immutable(
            complete,
            {
                "input_hash": input_hash,
                "collection_id": request.collection_id,
                "session_id": request.session_id,
                "record_id": request.record_id,
                "candidate_id": candidate.candidate_id,
                "replay_ref": ref.to_dict(),
                "artifact_ref": artifact.ref.to_dict(),
                "credit_registration_required": True,
            },
        )
        return CurateCollection(
            request.collection_id, request.session_id, request.record_id, candidate, artifact, ref, str(directory)
        )
