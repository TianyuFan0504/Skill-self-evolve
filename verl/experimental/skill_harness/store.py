# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""A dependency-free, versioned store for structured text skills.

Artifacts, candidates, admission decisions, and snapshots are immutable JSON
objects. Only branch heads change. Writers serialize with a POSIX file lock and
publish a head with an atomic replace; readers never create or update files.
The store is suitable for a single filesystem service wrapped by a Ray actor,
but does not import Ray or the verl package at runtime.

Basic admission checks structure and integrity, not downstream usefulness.
The publisher must run its own utility tests before admitting a candidate.
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
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1
RETRIEVER_VERSION = "family-lexical-v1"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class SkillStoreError(RuntimeError):
    """Base exception for persistent store errors."""


class SnapshotConflict(SkillStoreError):
    """A branch no longer points to the expected parent snapshot."""


class AdmissionError(SkillStoreError):
    """A candidate is not eligible for publication."""


class IntegrityError(SkillStoreError):
    """A persisted object's bytes or identity do not match its manifest."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
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


def _name(value: str, kind: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"Invalid {kind}: use a short identifier without path separators")
    return value


def _hash(value: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("Expected a SHA-256 identifier")
    return value


@dataclass(frozen=True)
class SkillRef:
    skill_id: str
    version: str
    content_hash: str

    def __post_init__(self) -> None:
        _name(self.skill_id, "skill ID")
        _hash(self.content_hash)
        if self.version != f"sha256:{self.content_hash}":
            raise ValueError("Skill versions must name their complete immutable content hash")

    def to_dict(self) -> dict[str, str]:
        return {"skill_id": self.skill_id, "version": self.version, "content_hash": self.content_hash}


@dataclass(frozen=True)
class SkillArtifact:
    skill_id: str
    title: str
    principle: str
    when_to_apply: str
    task_families: tuple[str, ...] = ("*",)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    version: str = field(init=False)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _name(self.skill_id, "skill ID")
        for key, limit in (("title", 256), ("principle", 32768), ("when_to_apply", 8192)):
            text = getattr(self, key)
            if not isinstance(text, str) or not text.strip() or len(text) > limit:
                raise ValueError(f"{key} must be nonempty text of at most {limit} characters")
            object.__setattr__(self, key, text.strip())
        if (not isinstance(self.task_families, list | tuple) or not self.task_families
                or any(not isinstance(family, str) for family in self.task_families)):
            raise ValueError("task_families must be a nonempty sequence")
        families = tuple(sorted(set(self.task_families)))
        for family in families:
            if family != "*":
                _name(family, "task family")
        object.__setattr__(self, "task_families", families)
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a JSON object")
        # Canonical round-trip rejects non-JSON objects and copies caller-owned data.
        provenance = json.loads(_canonical(self.provenance))
        if len(_canonical(provenance)) > 65536:
            raise ValueError("provenance must be at most 64 KiB")
        object.__setattr__(self, "provenance", _freeze(provenance))
        digest = _digest(self._payload())
        object.__setattr__(self, "content_hash", digest)
        object.__setattr__(self, "version", f"sha256:{digest}")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "skill_id": self.skill_id,
            "title": self.title,
            "principle": self.principle,
            "when_to_apply": self.when_to_apply,
            "task_families": list(self.task_families),
            "provenance": _plain(self.provenance),
        }

    @property
    def ref(self) -> SkillRef:
        return SkillRef(self.skill_id, self.version, self.content_hash)

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "version": self.version, "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SkillArtifact:
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise IntegrityError("Unsupported skill schema version")
        artifact = cls(
            skill_id=data["skill_id"], title=data["title"], principle=data["principle"],
            when_to_apply=data["when_to_apply"], task_families=data.get("task_families", ("*",)),
            provenance=data.get("provenance", {}),
        )
        for key in ("version", "content_hash"):
            if key in data and data[key] != getattr(artifact, key):
                raise IntegrityError(f"Skill {key} does not match its content")
        return artifact


@dataclass(frozen=True)
class SkillSnapshot:
    snapshot_id: str
    parent_id: str | None
    skills: tuple[SkillRef, ...]
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "parent_id": self.parent_id,
                "skills": [ref.to_dict() for ref in self.skills], "provenance": _plain(self.provenance)}


@dataclass(frozen=True)
class SkillCandidate:
    candidate_id: str
    branch: str
    base_snapshot: str
    artifact: SkillRef
    replace: bool

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "branch": self.branch, "base_snapshot": self.base_snapshot,
                "artifact": self.artifact.to_dict(), "replace": self.replace}


@dataclass(frozen=True)
class RetrievedSkills:
    snapshot_id: str
    skills: tuple[SkillRef, ...]
    text: str
    token_count: int
    token_count_is_estimate: bool
    context_hash: str
    query_hash: str
    budget_excluded: tuple[SkillRef, ...]
    retriever_version: str = RETRIEVER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, "skills": [ref.to_dict() for ref in self.skills],
                "text": self.text, "token_count": self.token_count,
                "token_count_is_estimate": self.token_count_is_estimate, "context_hash": self.context_hash,
                "query_hash": self.query_hash, "budget_excluded": [ref.to_dict() for ref in self.budget_excluded],
                "retriever_version": self.retriever_version}


def estimate_tokens(text: str) -> int:
    """An English-text estimate; production callers should supply their tokenizer."""
    return math.ceil(len(text.encode("utf-8")) / 4)


class SkillStore:
    def __init__(self, root: str | Path, *, read_only: bool = False, pinned_snapshot: str | None = None):
        self.root = Path(root)
        self.read_only = read_only
        self.pinned_snapshot = _hash(pinned_snapshot) if pinned_snapshot is not None else None
        self._pinned_refs: frozenset[SkillRef] | None = None
        if self.pinned_snapshot is not None and not read_only:
            raise ValueError("A pinned evaluation view must be read-only")
        if read_only:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
            if self.pinned_snapshot:
                self._pinned_refs = frozenset(self.get_snapshot(self.pinned_snapshot).skills)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("artifacts", "candidates", "admissions", "snapshots", "refs", "locks"):
            (self.root / name).mkdir(exist_ok=True)
        with self._writer():
            if not (self.root / "refs/main.json").exists():
                manifest = {"schema_version": SCHEMA_VERSION, "parent_id": None, "skills": [],
                            "provenance": {"operation": "genesis"}}
                snapshot_id = _digest(manifest)
                other_snapshots = list((self.root / "snapshots").glob("*.json"))
                if any(p.stem != snapshot_id for p in other_snapshots):
                    raise IntegrityError("Missing main head in a nonempty store; recover an explicit snapshot")
                self._immutable(self.root / "snapshots" / f"{snapshot_id}.json", manifest)
                self._atomic(self.root / "refs/main.json", {"branch": "main", "snapshot_id": snapshot_id})

    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError("This SkillStore view is read-only")

    @contextlib.contextmanager
    def _writer(self):
        self._writable()
        with (self.root / "locks/store.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic(path: Path, value: Any) -> None:
        payload = _canonical(value)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _immutable(self, path: Path, value: Any) -> None:
        if path.exists():
            if path.read_bytes() != _canonical(value):
                raise IntegrityError(f"Attempted to change immutable object {path.name}")
            return
        self._atomic(path, value)

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise IntegrityError(f"Invalid JSON object: {path.name}") from exc
        if not isinstance(data, dict):
            raise IntegrityError(f"Expected a JSON object: {path.name}")
        return data

    def head(self, branch: str = "main") -> str:
        if self.pinned_snapshot is not None:
            return self.pinned_snapshot
        branch = _name(branch, "branch")
        data = self._read(self.root / "refs" / f"{branch}.json")
        if data.get("branch") != branch:
            raise IntegrityError("Branch identity does not match its path")
        return _hash(data["snapshot_id"])

    def get_snapshot(self, snapshot_id: str | None = None, *, branch: str = "main") -> SkillSnapshot:
        snapshot_id = _hash(snapshot_id) if snapshot_id is not None else self.head(branch)
        if self.pinned_snapshot is not None and snapshot_id != self.pinned_snapshot:
            raise PermissionError("An evaluation view cannot switch its pinned snapshot")
        data = self._read(self.root / "snapshots" / f"{snapshot_id}.json")
        if data.get("schema_version") != SCHEMA_VERSION or _digest(data) != snapshot_id:
            raise IntegrityError("Snapshot content does not match its address")
        refs = tuple(SkillRef(**ref) for ref in data["skills"])
        if len({ref.skill_id for ref in refs}) != len(refs):
            raise IntegrityError("A snapshot cannot contain two active versions of the same skill")
        return SkillSnapshot(snapshot_id, data["parent_id"], refs, _freeze(data["provenance"]))

    def read_only_view(self, snapshot_id: str | None = None, *, branch: str = "main") -> SkillStore:
        return SkillStore(self.root, read_only=True, pinned_snapshot=snapshot_id or self.head(branch))

    def load_artifact(self, ref: SkillRef) -> SkillArtifact:
        if self._pinned_refs is not None and ref not in self._pinned_refs:
            raise PermissionError("Skill is outside the pinned evaluation snapshot")
        data = self._read(self.root / "artifacts" / f"{ref.content_hash}.json")
        artifact = SkillArtifact.from_dict(data)
        if artifact.ref != ref:
            raise IntegrityError("Artifact reference does not match stored content")
        return artifact

    def create_branch(self, branch: str, *, from_snapshot: str | None = None) -> str:
        branch = _name(branch, "branch")
        with self._writer():
            path = self.root / "refs" / f"{branch}.json"
            if path.exists():
                raise SnapshotConflict(f"Branch already exists: {branch}")
            snapshot_id = from_snapshot or self.head()
            self.get_snapshot(snapshot_id)
            self._atomic(path, {"branch": branch, "snapshot_id": snapshot_id})
        return snapshot_id

    def stage_candidate(self, artifact: SkillArtifact | Mapping[str, Any], *, branch: str = "main",
                        replace: bool = False) -> SkillCandidate:
        self._writable()
        artifact = artifact if isinstance(artifact, SkillArtifact) else SkillArtifact.from_dict(artifact)
        branch = _name(branch, "branch")
        with self._writer():
            base = self.head(branch)
            self._immutable(self.root / "artifacts" / f"{artifact.content_hash}.json", artifact.to_dict())
            candidate = SkillCandidate(uuid.uuid4().hex, branch, base, artifact.ref, bool(replace))
            self._immutable(self.root / "candidates" / f"{candidate.candidate_id}.json", {
                "candidate_id": candidate.candidate_id, "branch": branch, "base_snapshot": base,
                "artifact": artifact.ref.to_dict(), "replace": candidate.replace,
            })
        return candidate

    def get_candidate(self, candidate_id: str) -> SkillCandidate:
        _name(candidate_id, "candidate ID")
        data = self._read(self.root / "candidates" / f"{candidate_id}.json")
        if data["candidate_id"] != candidate_id:
            raise IntegrityError("Candidate identity does not match its path")
        return SkillCandidate(data["candidate_id"], data["branch"], data["base_snapshot"],
                              SkillRef(**data["artifact"]), data["replace"])

    def _decide(self, candidate_id: str, approved: bool, reason: str) -> SkillCandidate:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Admission decisions require an explicit reason")
        with self._writer():
            candidate = self.get_candidate(candidate_id)
            self.load_artifact(candidate.artifact)
            self._immutable(self.root / "admissions" / f"{candidate_id}.json", {
                "candidate_id": candidate_id, "approved": approved, "reason": reason.strip(),
                "artifact": candidate.artifact.to_dict(), "kind": "basic",
            })
        return candidate

    def admit(self, candidate_id: str, *,
              reason: str = "Basic structure and integrity checks passed") -> SkillCandidate:
        return self._decide(candidate_id, True, reason)

    def reject(self, candidate_id: str, *, reason: str) -> SkillCandidate:
        return self._decide(candidate_id, False, reason)

    def publish(self, candidate_ids: Sequence[str], *, expected_parent: str, branch: str = "main",
                provenance: Mapping[str, Any] | None = None) -> SkillSnapshot:
        """Atomically activate explicitly admitted candidates against their staged parent.

        A stale candidate must be restaged and readmitted; publication never silently
        rebases it onto a changed skill library. All records stay available by hash.
        """
        branch = _name(branch, "branch")
        _hash(expected_parent)
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("Publication provenance must be a JSON object")
        if isinstance(candidate_ids, str) or not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("Provide a nonempty list of distinct candidate IDs")
        with self._writer():
            actual = self.head(branch)
            if actual != expected_parent:
                raise SnapshotConflict(f"Expected parent {expected_parent}; branch head is {actual}")
            parent = self.get_snapshot(actual)
            active = {ref.skill_id: ref for ref in parent.skills}
            candidates = [self.get_candidate(key) for key in candidate_ids]
            if len({c.artifact.skill_id for c in candidates}) != len(candidates):
                raise AdmissionError("A publication cannot contain competing versions of one skill")
            for candidate in candidates:
                if candidate.branch != branch or candidate.base_snapshot != expected_parent:
                    raise SnapshotConflict("Candidate belongs to another branch or staged parent")
                admission_path = self.root / "admissions" / f"{candidate.candidate_id}.json"
                if not admission_path.exists():
                    raise AdmissionError("Candidate requires explicit admission before publication")
                decision = self._read(admission_path)
                if not decision.get("approved") or decision.get("artifact") != candidate.artifact.to_dict():
                    raise AdmissionError("Candidate was rejected or its admission does not match")
                self.load_artifact(candidate.artifact)
                exists = candidate.artifact.skill_id in active
                if exists != candidate.replace:
                    raise AdmissionError("Use replace=True only when explicitly revising an active skill")
                active[candidate.artifact.skill_id] = candidate.artifact
            manifest = {
                "schema_version": SCHEMA_VERSION, "parent_id": expected_parent,
                "skills": [active[key].to_dict() for key in sorted(active)],
                "provenance": {"operation": "publish", "branch": branch,
                               "candidate_ids": sorted(candidate_ids), "metadata": _plain(provenance or {})},
            }
            snapshot_id = _digest(manifest)
            self._immutable(self.root / "snapshots" / f"{snapshot_id}.json", manifest)
            self._atomic(self.root / "refs" / f"{branch}.json", {"branch": branch, "snapshot_id": snapshot_id})
            return self.get_snapshot(snapshot_id)

    @staticmethod
    def _render(artifacts: Sequence[SkillArtifact], snapshot_id: str) -> str:
        if not artifacts:
            return ""
        sections = [f"Reusable skills (snapshot {snapshot_id}):"]
        for artifact in artifacts:
            sections.append(
                f"[Skill {artifact.skill_id}@{artifact.content_hash[:16]}]\n"
                f"Title: {artifact.title}\nWhen to apply: {artifact.when_to_apply}\n"
                f"Principle: {artifact.principle}"
            )
        return "\n\n".join(sections)

    def retrieve(self, task_family: str, *, snapshot_id: str | None = None, branch: str = "main",
                 token_budget: int = 1024, top_k: int = 6, query: str = "",
                 token_counter: Callable[[str], int] | None = None) -> RetrievedSkills:
        """Select whole skills deterministically within the rendered-context budget.

        The callback measures the complete rendered skill block, including headers
        and separators. It does not include surrounding chat-template overhead.
        Default counts are estimates, explicitly marked in the result.
        """
        _name(task_family, "task family")
        if type(token_budget) is not int or token_budget < 0 or type(top_k) is not int or top_k < 0:
            raise ValueError("token_budget and top_k must be nonnegative integers")
        snapshot = self.get_snapshot(snapshot_id, branch=branch)
        terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        eligible = []
        for ref in snapshot.skills:
            artifact = self.load_artifact(ref)
            if task_family not in artifact.task_families and not {"*", "general"}.intersection(artifact.task_families):
                continue
            searchable = f"{artifact.title} {artifact.principle} {artifact.when_to_apply}"
            words = set(re.findall(r"[a-z0-9]+", searchable.casefold()))
            rank = (
                0 if task_family in artifact.task_families else 1,
                -len(terms & words), artifact.skill_id, artifact.version,
            )
            eligible.append((rank, artifact))
        eligible.sort(key=lambda pair: pair[0])
        counter = token_counter or estimate_tokens
        selected, excluded = [], []
        text, count = "", 0
        for _, artifact in eligible:
            if len(selected) >= top_k:
                break
            proposed = self._render([*selected, artifact], snapshot.snapshot_id)
            measured = counter(proposed)
            if type(measured) is not int or measured < 0:
                raise ValueError("token_counter must return a nonnegative integer")
            if measured > token_budget:
                excluded.append(artifact.ref)
                continue
            selected.append(artifact)
            text, count = proposed, measured
        return RetrievedSkills(
            snapshot.snapshot_id, tuple(a.ref for a in selected), text, count, token_counter is None,
            hashlib.sha256(text.encode()).hexdigest(), hashlib.sha256(query.encode()).hexdigest(), tuple(excluded),
        )
