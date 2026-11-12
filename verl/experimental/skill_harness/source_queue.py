"""Controller-owned source opportunities; no thread, loop, model or file writes.

The caller serializes mutations and persists its enclosing controller state.
A running job is never retried automatically after restoration: reconcile its
Native launch/capture or explicitly censor it before considering another launch.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping

TERMINAL_STATUSES = frozenset({"generated", "abstained", "source_censored", "failed"})
STATUSES = TERMINAL_STATUSES | {"queued", "running"}
_RESERVED = frozenset({"key", "sequence", "status", "read", "curator_jobs", "details", "source_identity_hash"})


class SourceQueueConflict(ValueError):
    """An episode identity or terminal disposition cannot silently change."""


class SourceQueueFull(RuntimeError):
    """Explicit backpressure; no source opportunity was silently dropped."""


class SourceQueueRecoveryRequired(RuntimeError):
    """An uncertain persistence result requires owner reconciliation."""


def _plain(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Source queue fields must have string keys")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _json_copy(value):
    return json.loads(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    )


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def _episode_path(value):
    if not isinstance(value, str) or not os.path.isabs(value) or ".." in value.split(os.sep):
        raise ValueError("An absolute host-resolved episode_output_dir is required")
    return os.path.normpath(value)


class SourceQueue:
    """FIFO jobs backed directly by controller.state['source_queue']."""

    def __init__(self, state: dict, *, persist: Callable[[], None], max_backlog: int = 8192):
        if not isinstance(state, dict) or not callable(persist):
            raise TypeError("SourceQueue requires its state dict and enclosing persistence callback")
        if type(max_backlog) is not int or max_backlog < 1:
            raise ValueError("max_backlog must be positive")
        self.state, self.persist, self.max_backlog = state, persist, max_backlog
        self._uncertain = False
        if not state:
            state.update(schema_version=1, max_backlog=max_backlog, next_sequence=0, jobs={})
            self._commit()
        self.validate()

    def _commit(self):
        try:
            self.persist()
        except BaseException:
            # The enclosing write may have committed before failing its fsync.
            # Reverting a running claim could cause a duplicate Native launch.
            self._uncertain = True
            raise

    def _ready(self):
        if self._uncertain:
            raise SourceQueueRecoveryRequired("Source persistence is uncertain; reload/reconcile before mutation")

    def validate(self):
        state = self.state
        if state.get("schema_version") != 1 or state.get("max_backlog") != self.max_backlog:
            raise ValueError("Source queue schema or configured bound differs")
        _integer(state.get("next_sequence"), "next_sequence")
        jobs = state.get("jobs")
        if not isinstance(jobs, dict):
            raise ValueError("Source queue jobs must be a mapping")
        sequences = []
        for key, entry in jobs.items():
            path = _episode_path(entry.get("episode_output_dir"))
            if key != "source_" + hashlib.sha256(path.encode()).hexdigest() or entry.get("key") != key:
                raise ValueError("Source key does not identify its exact episode directory")
            sequences.append(_integer(entry.get("sequence"), "sequence"))
            _integer(entry.get("source_step"), "source_step")
            version = entry.get("observed_parameter_version")
            if version is not None:
                _integer(version, "observed_parameter_version")
            if (
                entry.get("status") not in STATUSES
                or type(entry.get("read")) is not bool
                or not isinstance(entry.get("curator_jobs"), list)
                or not isinstance(entry.get("details"), dict)
                or not isinstance(entry.get("source_identity_hash"), str)
            ):
                raise ValueError("Unsupported source lifecycle record")
            if entry["status"] == "generated" and not entry["read"]:
                raise ValueError("A generated source must have been read")
        if len(set(sequences)) != len(sequences) or any(x >= state["next_sequence"] for x in sequences):
            raise ValueError("Source queue sequence is inconsistent")
        if sum(x["status"] in {"queued", "running"} for x in jobs.values()) > self.max_backlog:
            raise ValueError("Source queue exceeds its explicit bound")
        _json_copy(state)
        return True

    def offer_many(
        self, sources: Iterable[dict], *, source_step: int | None = None, observed_parameter_version: int | None = None
    ) -> int:
        """Offer all completed logical episodes, deduplicating TITO prefixes.

        An optional observed version is source evidence, never inferred from its
        step. Curator behavior/version pins belong in curator_jobs and replay.
        """
        self._ready()
        additions = {}
        for value in sources:
            if not isinstance(value, Mapping) or _RESERVED & set(value):
                raise ValueError("Source opportunities must not contain lifecycle fields")
            source = _json_copy(value)
            source["episode_output_dir"] = _episode_path(source.get("episode_output_dir"))
            source["source_step"] = _integer(source.get("source_step", source_step), "source_step")
            version = source.get("observed_parameter_version", observed_parameter_version)
            if version is not None:
                _integer(version, "observed_parameter_version")
            source["observed_parameter_version"] = version
            if not isinstance(source.get("task_id"), str) or not source["task_id"]:
                raise ValueError("A source task_id is required")
            key = "source_" + hashlib.sha256(source["episode_output_dir"].encode()).hexdigest()
            identity = _digest(source)
            previous = self.state["jobs"].get(key) or additions.get(key)
            if previous is not None:
                if previous["source_identity_hash"] != identity:
                    raise SourceQueueConflict("One episode directory has conflicting source identity")
                continue
            additions[key] = {
                **source,
                "key": key,
                "source_identity_hash": identity,
                "sequence": self.state["next_sequence"] + len(additions),
                "status": "queued",
                "read": False,
                "curator_jobs": [],
                "details": {},
            }
        if self.metrics()["sources_backlog"] + len(additions) > self.max_backlog:
            raise SourceQueueFull("Source backlog bound reached; no offers were committed or dropped")
        if additions:
            self.state["jobs"].update(additions)
            self.state["next_sequence"] += len(additions)
            self._commit()
        return len(additions)

    def claim_ready(self, limit: int) -> list[tuple[str, dict]]:
        """Persist claims before launch; return actual entries owned by the caller."""
        self._ready()
        _integer(limit, "limit")
        selected = sorted(
            ((key, entry) for key, entry in self.state["jobs"].items() if entry["status"] == "queued"),
            key=lambda pair: pair[1]["sequence"],
        )[:limit]
        for _, entry in selected:
            entry["status"] = "running"
        if selected:
            self._commit()
        return selected

    def mark_read(self, key: str) -> None:
        self._ready()
        entry = self.state["jobs"][key]
        if entry["status"] != "running":
            raise SourceQueueConflict("Only an active source can be marked read")
        if not entry["read"]:
            entry["read"] = True
            self._commit()

    def finish(self, key: str, status: str, **details) -> None:
        self._ready()
        if status not in TERMINAL_STATUSES:
            raise ValueError("Unknown terminal source disposition")
        entry = self.state["jobs"][key]
        detail = _json_copy(details)
        if entry["status"] in TERMINAL_STATUSES:
            if entry["status"] != status or entry["details"] != detail:
                raise SourceQueueConflict("A terminal source disposition cannot be rewritten")
            return
        if status == "generated" and (entry["status"] != "running" or not entry["read"]):
            raise SourceQueueConflict("Generated sources require an active, read opportunity")
        if status != "generated" and not detail.get("reason"):
            raise ValueError("Abstained/censored/failed sources require an explicit reason")
        entry.update(status=status, details=detail)
        self._commit()

    def recovery_required(self) -> list[tuple[str, dict]]:
        """Expose running records for restore reconciliation; never requeue them."""
        return [
            (key, copy.deepcopy(entry)) for key, entry in self.state["jobs"].items() if entry["status"] == "running"
        ]

    def metrics(self) -> dict[str, int]:
        jobs = list(self.state["jobs"].values())
        counts = {status: sum(x["status"] == status for x in jobs) for status in STATUSES}
        return {
            "sources_opportunities": len(jobs),
            "sources_enqueued": len(jobs),
            "sources_pending": counts["queued"],
            "sources_active": counts["running"],
            "sources_read": sum(x["read"] for x in jobs),
            "sources_generated": counts["generated"],
            "sources_abstained": counts["abstained"],
            "sources_censored": counts["source_censored"],
            "sources_failed": counts["failed"],
            "sources_backlog": counts["queued"] + counts["running"],
        }
