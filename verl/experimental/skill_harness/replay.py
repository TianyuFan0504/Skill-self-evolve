# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Immutable, dependency-free retention of the actual EMA curate actions.

Only CPU JSON/token data are retained, never an autograd graph. An append is
acknowledged after its immutable payload and causal index have been fsynced.
The TQ cleanup caller must wait for that acknowledgement.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReplayError(RuntimeError):
    """Base replay/credit persistence error."""


class RecordConflict(ReplayError):
    """An immutable identity has already been used for different data."""


class IntegrityError(ReplayError):
    """A stored payload does not match its content digest."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _identifier(value: str, kind: str = "identifier") -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {kind}: use a short identifier without path separators")
    return value


def _finite(value: float, kind: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{kind} must be finite")
    return float(value)


def _skill_ref(value: Any) -> Mapping[str, str]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise ValueError("skill_ref must be a SkillRef or a plain reference mapping")
    ref = {key: value.get(key) for key in ("skill_id", "version", "content_hash")}
    _identifier(ref["skill_id"], "skill ID")
    if not isinstance(ref["content_hash"], str) or not _SHA256.fullmatch(ref["content_hash"]):
        raise ValueError("skill_ref requires a SHA-256 content_hash")
    if ref["version"] != "sha256:" + ref["content_hash"]:
        raise ValueError("skill_ref version must identify its immutable content hash")
    return MappingProxyType(ref)


def _json_copy(value: Any) -> Any:
    return _freeze(json.loads(_canonical(value)))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_payload(path: Path) -> Any:
    try:
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        if envelope.get("schema_version") != SCHEMA_VERSION or envelope["payload_hash"] != _digest(payload):
            raise IntegrityError(f"Payload digest mismatch: {path}")
        return payload
    except (ValueError, KeyError, TypeError) as exc:
        raise IntegrityError(f"Malformed persisted object: {path}") from exc


def _publish_immutable(path: Path, payload: Any) -> str:
    digest = _digest(payload)
    if path.exists():
        if _digest(_read_payload(path)) != digest:
            raise RecordConflict(f"Immutable identity conflicts with existing data: {path.name}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    envelope = {"schema_version": SCHEMA_VERSION, "payload_hash": digest, "payload": _plain(payload)}
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical(envelope) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link publishes a complete file without replacing another ID.
            os.link(temporary, path)
        except FileExistsError:
            if _digest(_read_payload(path)) != digest:
                raise RecordConflict(f"Immutable identity conflicts with existing data: {path.name}") from None
        _fsync_directory(path.parent)
        return digest
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _writer_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(root / ".writer.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass(frozen=True)
class CurateReplayRecord:
    record_id: str
    producer_uid: str
    skill_ref: Any
    prompt_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    curate_mask: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    behavior_policy_version: str
    sampling_config: Mapping[str, Any]
    tokenizer_version: str
    template_version: str
    context_snapshot: Any
    skill_snapshot_id: str
    candidate_group_id: str
    candidate_index: int
    candidate_group_size: int
    created_at: float
    role: str = "curator"
    behavior_policy_kind: str = "ema_curator"
    trajectory_index: int = 0
    trajectory_count: int = 1

    def __post_init__(self):
        for name in (
            "record_id", "producer_uid", "behavior_policy_version", "tokenizer_version", "template_version",
            "skill_snapshot_id", "candidate_group_id",
        ):
            _identifier(getattr(self, name), name)
        if self.role != "curator" or self.behavior_policy_kind != "ema_curator":
            raise ValueError("This replay stores curator actions generated by an identified EMA behavior policy")
        object.__setattr__(self, "skill_ref", _skill_ref(self.skill_ref))
        for name in ("prompt_ids", "action_ids"):
            values = tuple(getattr(self, name))
            if not values or any(type(token) is not int or token < 0 for token in values):
                raise ValueError(f"{name} must be a non-empty sequence of nonnegative token IDs")
            object.__setattr__(self, name, values)
        mask = tuple(self.curate_mask)
        probabilities = tuple(_finite(value, "behavior logprob") for value in self.behavior_logprobs)
        if len(mask) != len(self.action_ids) or len(probabilities) != len(self.action_ids):
            raise ValueError("Action tokens, curate mask and behavior logprobs must have matching lengths")
        if any(type(value) is not int or value not in (0, 1) for value in mask) or not any(mask):
            raise ValueError("curate_mask must contain binary integers and at least one generated action")
        if any(value > 0 for value in probabilities):
            raise ValueError("Discrete behavior logprobs cannot be positive")
        if any(not selected and probability != 0 for selected, probability in zip(mask, probabilities, strict=True)):
            raise ValueError("Non-action context tokens must have zero behavior logprob")
        object.__setattr__(self, "curate_mask", mask)
        object.__setattr__(self, "behavior_logprobs", probabilities)
        if not isinstance(self.sampling_config, Mapping):
            raise ValueError("sampling_config must be a complete mapping of generation settings")
        sampling = dict(self.sampling_config)
        for field in ("temperature", "top_p", "top_k", "logprobs_mode"):
            if field not in sampling:
                raise ValueError(f"sampling_config must preserve {field}")
        if _finite(sampling["temperature"], "temperature") < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < _finite(sampling["top_p"], "top_p") <= 1:
            raise ValueError("top_p must lie in (0, 1]")
        if type(sampling["top_k"]) is not int or sampling["top_k"] < -1:
            raise ValueError("top_k must be an integer >= -1")
        if sampling["logprobs_mode"] != "processed_logprobs":
            raise ValueError("Behavior probabilities must use the actual processed sampling distribution")
        object.__setattr__(self, "sampling_config", _json_copy(sampling))
        if not self.context_snapshot or not isinstance(self.context_snapshot, str | Mapping):
            raise ValueError("A non-empty original context snapshot is required")
        object.__setattr__(self, "context_snapshot", _json_copy(self.context_snapshot))
        if type(self.candidate_group_size) is not int or self.candidate_group_size < 1:
            raise ValueError("candidate_group_size must be positive")
        if type(self.candidate_index) is not int or not 0 <= self.candidate_index < self.candidate_group_size:
            raise ValueError("candidate_index must name a member of the original candidate group")
        if type(self.trajectory_count) is not int or self.trajectory_count < 1:
            raise ValueError("trajectory_count must be positive")
        if type(self.trajectory_index) is not int or not 0 <= self.trajectory_index < self.trajectory_count:
            raise ValueError("trajectory_index must name a retained segment of the producer episode")
        created = _finite(self.created_at, "created_at")
        if created < 0:
            raise ValueError("created_at cannot be negative")
        object.__setattr__(self, "created_at", created)

    def to_dict(self) -> dict[str, Any]:
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CurateReplayRecord:
        return cls(**dict(value))


@dataclass(frozen=True)
class ReplayRef:
    record_id: str
    payload_hash: str
    payload_uri: str

    def to_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class DurableSnapshotRef:
    snapshot_id: str
    manifest_uri: str
    kind: str
    item_count: int

    def to_dict(self):
        return dict(self.__dict__)


def _snapshot_payload(ref: DurableSnapshotRef, expected_kind: str) -> tuple[dict[str, Any], Path]:
    path = Path(ref.manifest_uri)
    payload = _read_payload(path)
    if payload.get("kind") != expected_kind or ref.kind != expected_kind:
        raise IntegrityError("Snapshot kind mismatch")
    if ref.snapshot_id != "sha256:" + _digest(payload):
        raise IntegrityError("Snapshot reference does not match its immutable payload")
    return payload, path.parent.parent


class CurateReplay:
    """Append-only record objects; snapshots select a fixed retained record set."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _record_path(self, record_id: str) -> Path:
        return self.root / "records" / (_identifier(record_id, "record ID") + ".json")

    def append(self, record: CurateReplayRecord) -> ReplayRef:
        payload = record.to_dict()
        producer = _digest([record.producer_uid, record.candidate_group_id, record.candidate_index])
        provenance = _digest([producer, record.trajectory_index])
        index = self.root / "provenance" / (provenance + ".json")
        producer_path = self.root / "producers" / (producer + ".json")
        bundle = {
            "producer_uid": record.producer_uid,
            "candidate_group_id": record.candidate_group_id,
            "candidate_index": record.candidate_index,
            "trajectory_count": record.trajectory_count,
            "skill_ref": _plain(record.skill_ref),
            "records": {},
        }
        with _writer_lock(self.root):
            if index.exists() and _read_payload(index)["record_id"] != record.record_id:
                raise RecordConflict("The same producer/group/candidate already has a different record ID")
            if producer_path.exists():
                previous = _read_payload(producer_path)
                if {k: v for k, v in previous.items() if k != "records"} != {
                    k: v for k, v in bundle.items() if k != "records"
                }:
                    raise RecordConflict("Producer segments disagree about their immutable skill or segment count")
                bundle = previous
            digest = _publish_immutable(self._record_path(record.record_id), payload)
            _publish_immutable(index, {"record_id": record.record_id, "payload_hash": digest})
            segment = {"record_id": record.record_id, "payload_hash": digest}
            key = str(record.trajectory_index)
            if key in bundle["records"] and bundle["records"][key] != segment:
                raise RecordConflict("Producer segment already refers to another immutable action")
            bundle["records"][key] = segment
            _write_atomic(
                producer_path,
                {"schema_version": SCHEMA_VERSION, "payload_hash": _digest(bundle), "payload": bundle},
            )
        return ReplayRef(record.record_id, digest, str(self._record_path(record.record_id)))

    def _committed_payload(self, record_id):
        payload = _read_payload(self._record_path(record_id))
        record = CurateReplayRecord.from_dict(payload)
        producer = _digest([record.producer_uid, record.candidate_group_id, record.candidate_index])
        provenance = _digest([producer, record.trajectory_index])
        index = self.root / "provenance" / (provenance + ".json")
        if not index.exists():
            raise IntegrityError("Replay object was not causally committed before the crash")
        if _read_payload(index) != {"record_id": record_id, "payload_hash": _digest(payload)}:
            raise IntegrityError("Replay causal index does not match its payload")
        producer_path = self.root / "producers" / (producer + ".json")
        if not producer_path.exists() or _read_payload(producer_path)["records"].get(str(record.trajectory_index)) != {
            "record_id": record_id, "payload_hash": _digest(payload)
        }:
            raise IntegrityError("Replay producer segment was not committed")
        return payload

    def get(self, record_id: str) -> CurateReplayRecord:
        return CurateReplayRecord.from_dict(self._committed_payload(record_id))

    def get_ref(self, record_id: str) -> ReplayRef:
        payload = self._committed_payload(record_id)
        return ReplayRef(record_id, _digest(payload), str(self._record_path(record_id)))

    def get_candidate_records(self, record_id: str) -> tuple[CurateReplayRecord, ...]:
        record = self.get(record_id)
        producer = _digest([record.producer_uid, record.candidate_group_id, record.candidate_index])
        bundle = _read_payload(self.root / "producers" / (producer + ".json"))
        if set(bundle["records"]) != {str(i) for i in range(record.trajectory_count)}:
            raise IntegrityError(
                "The producer episode is incomplete; retain all TITO segments before credit registration"
            )
        return tuple(self.get(bundle["records"][str(i)]["record_id"]) for i in range(record.trajectory_count))

    def export_snapshot(self) -> DurableSnapshotRef:
        with _writer_lock(self.root):
            refs = []
            for path in sorted((self.root / "records").glob("*.json")):
                payload = _read_payload(path)
                record = CurateReplayRecord.from_dict(payload)
                producer = _digest([record.producer_uid, record.candidate_group_id, record.candidate_index])
                provenance = _digest([producer, record.trajectory_index])
                index = self.root / "provenance" / (provenance + ".json")
                if not index.exists():
                    # Unacknowledged object left by a crash before causal commit.
                    continue
                committed = _read_payload(index)
                if committed != {"record_id": record.record_id, "payload_hash": _digest(payload)}:
                    raise IntegrityError("Replay provenance index disagrees with its payload")
                producer_path = self.root / "producers" / (producer + ".json")
                if not producer_path.exists() or _read_payload(producer_path)["records"].get(
                    str(record.trajectory_index)
                ) != committed:
                    continue
                refs.append({"record_id": record.record_id, "payload_hash": _digest(payload)})
            payload = {"schema_version": SCHEMA_VERSION, "kind": "curate_replay", "records": refs}
            digest = _digest(payload)
            path = self.root / "snapshots" / (digest + ".json")
            _publish_immutable(path, payload)
        return DurableSnapshotRef("sha256:" + digest, str(path), "curate_replay", len(refs))

    def restore_snapshot(self, ref: DurableSnapshotRef) -> tuple[ReplayRef, ...]:
        """Materialize referenced immutable data; never delete later append data.

        The joint-checkpoint coordinator restores the ledger's active references.
        Keeping later replay objects does not authorize them for training.
        """
        payload, source_root = _snapshot_payload(ref, "curate_replay")
        restored = []
        for item in payload["records"]:
            record_id = _identifier(item["record_id"], "record ID")
            value = _read_payload(source_root / "records" / (record_id + ".json"))
            if _digest(value) != item["payload_hash"]:
                raise IntegrityError("Replay snapshot record hash mismatch")
            restored.append(self.append(CurateReplayRecord.from_dict(value)))
        if len(restored) != ref.item_count:
            raise IntegrityError("Replay snapshot record count mismatch")
        return tuple(restored)
