# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Versioned delayed counterfactual credit, separate from transient training TQ.

This module computes labels and coordinates leases; it never updates a model.
Lease/CAS state alone does NOT make optimizer updates exactly-once. A learner
must reconcile uncertain updates against its applied-update IDs or restore a
joint model/optimizer/ledger checkpoint before releasing an expired lease.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .replay import (
    SCHEMA_VERSION,
    CurateReplay,
    DurableSnapshotRef,
    IntegrityError,
    RecordConflict,
    ReplayError,
    _digest,
    _finite,
    _identifier,
    _plain,
    _publish_immutable,
    _read_payload,
    _skill_ref,
    _snapshot_payload,
    _write_atomic,
    _writer_lock,
)


class CreditError(ReplayError):
    """A credit event or transition violates the registered protocol."""


class WindowNotMature(CreditError):
    """A pending or censored window cannot be turned into a zero label."""


class LeaseConflict(CreditError):
    """A model update is leased or requires explicit reconciliation."""


@dataclass(frozen=True)
class CreditProtocol:
    evaluator_version: str
    probe_protocol_version: str
    task_distribution_version: str
    environment_version: str
    baseline_snapshot_id: str
    required_exposures: int
    aggregation: str = "mean"
    cost_weight: float = 0.0
    max_updates: int = 1
    score_min: float = 0.0
    score_max: float = 1.0

    def __post_init__(self):
        for name in (
            "evaluator_version", "probe_protocol_version", "task_distribution_version",
            "environment_version", "baseline_snapshot_id",
        ):
            _identifier(getattr(self, name), name)
        if type(self.required_exposures) is not int or self.required_exposures < 1:
            raise ValueError("required_exposures must be positive")
        if self.aggregation not in ("mean", "sum"):
            raise ValueError("aggregation must be mean or sum")
        if type(self.max_updates) is not int or self.max_updates < 1:
            raise ValueError("max_updates must be positive")
        for name in ("cost_weight", "score_min", "score_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.cost_weight < 0 or self.score_min > self.score_max:
            raise ValueError("Invalid cost weight or score bounds")

    def to_dict(self):
        return dict(self.__dict__)

    @property
    def protocol_id(self):
        return "sha256:" + _digest(self.to_dict())


@dataclass(frozen=True)
class EvaluationOutcome:
    task_id: str
    model_version: str
    environment_version: str
    environment_state_hash: str
    baseline_snapshot_id: str
    score: float
    trajectory_ref: str
    loaded_skill_ref: Any = None

    def __post_init__(self):
        for name in ("task_id", "model_version", "environment_version", "baseline_snapshot_id"):
            _identifier(getattr(self, name), name)
        if (
            not isinstance(self.environment_state_hash, str)
            or len(self.environment_state_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.environment_state_hash)
        ):
            raise ValueError("environment_state_hash must be SHA-256")
        if not isinstance(self.trajectory_ref, str) or not self.trajectory_ref:
            raise ValueError("Each evaluation branch requires a trajectory reference")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        if self.loaded_skill_ref is not None:
            object.__setattr__(self, "loaded_skill_ref", _skill_ref(self.loaded_skill_ref))

    def to_dict(self):
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class UsageEvent:
    event_id: str
    window_id: str
    curate_record_id: str
    skill_ref: Any
    probe_id: str
    protocol_id: str
    with_skill: EvaluationOutcome
    without_skill: EvaluationOutcome
    occurred_at: float
    live_solver_version: str | None = None
    observed_live_score: float | None = None

    def __post_init__(self):
        for name in ("event_id", "window_id", "curate_record_id", "probe_id", "protocol_id"):
            _identifier(getattr(self, name), name)
        object.__setattr__(self, "skill_ref", _skill_ref(self.skill_ref))
        for name in ("with_skill", "without_skill"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = EvaluationOutcome(**dict(value))
                object.__setattr__(self, name, value)
            if not isinstance(value, EvaluationOutcome):
                raise ValueError(f"{name} must be an EvaluationOutcome")
        object.__setattr__(self, "occurred_at", _finite(self.occurred_at, "occurred_at"))
        if self.occurred_at < 0:
            raise ValueError("occurred_at cannot be negative")
        if self.live_solver_version is not None:
            _identifier(self.live_solver_version, "live_solver_version")
        if self.observed_live_score is not None:
            object.__setattr__(self, "observed_live_score", _finite(self.observed_live_score, "observed_live_score"))

    def to_dict(self):
        return {
            **{
                name: _plain(getattr(self, name))
                for name in self.__dataclass_fields__
                if name not in ("with_skill", "without_skill")
            },
            "with_skill": self.with_skill.to_dict(),
            "without_skill": self.without_skill.to_dict(),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class CreditSettlement:
    settlement_id: str
    label_version: str
    window_id: str
    curate_record_id: str
    curate_record_ids: tuple[str, ...]
    skill_ref: Any
    protocol_id: str
    event_ids: tuple[str, ...]
    contributions: tuple[float, ...]
    aggregation: str
    raw_utility: float
    skill_cost: float
    cost_weight: float
    utility: float
    max_updates: int

    def __post_init__(self):
        object.__setattr__(self, "skill_ref", _skill_ref(self.skill_ref))
        object.__setattr__(self, "event_ids", tuple(self.event_ids))
        object.__setattr__(self, "contributions", tuple(_finite(v, "contribution") for v in self.contributions))
        object.__setattr__(self, "curate_record_ids", tuple(self.curate_record_ids))
        if not self.event_ids or len(self.event_ids) != len(self.contributions):
            raise IntegrityError("Settlement exposures and contributions must match")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise IntegrityError("Settlement cannot count the same usage event twice")
        if not self.curate_record_ids or self.curate_record_id not in self.curate_record_ids:
            raise IntegrityError("Settlement must retain its primary producer and all trajectory segments")
        if len(set(self.curate_record_ids)) != len(self.curate_record_ids):
            raise IntegrityError("Settlement contains duplicate producer segments")
        if self.aggregation not in ("mean", "sum"):
            raise IntegrityError("Unknown settlement aggregation")
        for name in ("raw_utility", "skill_cost", "cost_weight", "utility"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        expected = sum(self.contributions)
        if self.aggregation == "mean":
            expected /= len(self.contributions)
        if not math.isclose(expected, self.raw_utility, abs_tol=1e-12):
            raise IntegrityError("Settlement aggregate disagrees with its exposures")
        if self.skill_cost < 0 or self.cost_weight < 0 or not math.isclose(
            self.utility, self.raw_utility - self.cost_weight * self.skill_cost, abs_tol=1e-12
        ):
            raise IntegrityError("Settlement cost/utility arithmetic is inconsistent")
        if type(self.max_updates) is not int or self.max_updates < 1:
            raise IntegrityError("Settlement update budget must be positive")
        base = self.to_dict()
        base.pop("settlement_id")
        base.pop("label_version")
        digest = _digest(base)
        if self.settlement_id != "credit_" + digest or self.label_version != "sha256:" + digest:
            raise IntegrityError("Settlement version does not match its immutable label")

    def to_dict(self):
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


def _empty_state():
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "credit_ledger_state",
        "windows": {},
        "events": {},
        "settlements": {},
        "candidate_cohorts": {},
        "probe_bindings": {},
        "reconciliations": [],
        "revision": 0,
    }


class CreditLedger:
    """Single filesystem-writer ledger with immutable outcomes and label versions.

    Credit is a versioned evaluator utility, not an asserted unbiased estimate
    of the complete moving solver/library system's historical return.
    """

    def __init__(self, root: str | Path, replay: CurateReplay):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.replay = replay

    def _state(self):
        path = self.root / "state.json"
        state = _read_payload(path) if path.exists() else _empty_state()
        if state.get("schema_version") != SCHEMA_VERSION or state.get("kind") != "credit_ledger_state":
            raise IntegrityError("Unsupported credit ledger state")
        return state

    def _save_state(self, state):
        state["revision"] += 1
        _write_atomic(
            self.root / "state.json",
            {"schema_version": SCHEMA_VERSION, "payload_hash": _digest(state), "payload": state},
        )

    def _object(self, kind, identity, expected_hash=None):
        path = self.root / kind / (_identifier(identity) + ".json")
        payload = _read_payload(path)
        if expected_hash is not None and _digest(payload) != expected_hash:
            raise IntegrityError(f"Ledger index hash mismatch: {kind}/{identity}")
        return payload

    def _window(self, state, window_id):
        item = state["windows"].get(window_id)
        if item is None:
            raise CreditError(f"Unknown credit window: {window_id}")
        return self._object("windows", window_id, item["payload_hash"])

    def register_window(
        self,
        window_id: str,
        curate_record_id: str,
        protocol: CreditProtocol,
        *,
        opened_at: float,
        deadline_at: float,
        skill_cost: float = 0.0,
        expected_probe_ids: tuple[str, ...] = (),
        cohort_id: str = "default",
    ) -> dict[str, Any]:
        _identifier(window_id, "window_id")
        _identifier(cohort_id, "cohort_id")
        record = self.replay.get(curate_record_id)
        producer_records = self.replay.get_candidate_records(curate_record_id)
        opened = _finite(opened_at, "opened_at")
        deadline = _finite(deadline_at, "deadline_at")
        cost = _finite(skill_cost, "skill_cost")
        if opened < max(r.created_at for r in producer_records) or deadline <= opened or cost < 0:
            raise ValueError("Credit window times/cost violate producer causality")
        probes = tuple(expected_probe_ids)
        if probes:
            if len(probes) != protocol.required_exposures or len(set(probes)) != len(probes):
                raise ValueError("Expected probes must uniquely cover required_exposures")
            for probe in probes:
                _identifier(probe, "probe_id")
        window = {
            "window_id": window_id,
            "curate_record_id": curate_record_id,
            "replay_payload_hash": self.replay.get_ref(curate_record_id).payload_hash,
            "replay_records": [
                {"record_id": r.record_id, "payload_hash": self.replay.get_ref(r.record_id).payload_hash}
                for r in producer_records
            ],
            "skill_ref": _plain(record.skill_ref),
            "protocol": protocol.to_dict(),
            "protocol_id": protocol.protocol_id,
            "opened_at": opened,
            "deadline_at": deadline,
            # This is an incremental per-window cost; callers must not charge a
            # one-off creation cost again in each later window.
            "skill_cost": cost,
            "expected_probe_ids": list(probes),
            "cohort_id": cohort_id,
            "candidate_cohort_key": _digest([record.candidate_group_id, cohort_id]),
        }
        group_key = _digest([record.candidate_group_id, cohort_id])
        group = {
            "protocol_id": protocol.protocol_id,
            "expected_probe_ids": list(probes),
            "candidate_group_size": record.candidate_group_size,
        }
        with _writer_lock(self.root):
            state = self._state()
            existing = state["candidate_cohorts"].get(group_key)
            if existing is not None and existing != group:
                raise CreditError("Candidates in one credit cohort require the same evaluator/probe protocol")
            digest = _publish_immutable(self.root / "windows" / (window_id + ".json"), window)
            if window_id not in state["windows"]:
                state["windows"][window_id] = {"payload_hash": digest, "event_ids": [], "settlement_id": None}
                state["candidate_cohorts"][group_key] = group
                self._save_state(state)
        return window

    def _status(self, state, window_id, now):
        window = self._window(state, window_id)
        item = state["windows"][window_id]
        count = len(item["event_ids"])
        required = window["protocol"]["required_exposures"]
        if item["settlement_id"] is not None:
            phase = "settled"
        elif count >= required:
            phase = "matured"
        elif now > window["deadline_at"]:
            phase = "censored"
        else:
            phase = "pending"
        return {
            "window_id": window_id,
            "state": phase,
            "observed_exposures": count,
            "required_exposures": required,
            "deadline_at": window["deadline_at"],
            "settlement_id": item["settlement_id"],
        }

    def window_status(self, window_id: str, *, now: float | None = None):
        return self._status(self._state(), window_id, time.time() if now is None else _finite(now, "now"))

    def get_window(self, window_id: str):
        return self._window(self._state(), window_id)

    def add_usage(self, event: UsageEvent, *, now: float | None = None) -> dict[str, Any]:
        payload = event.to_dict()
        timestamp = time.time() if now is None else _finite(now, "now")
        with _writer_lock(self.root):
            state = self._state()
            window = self._window(state, event.window_id)
            item = state["windows"][event.window_id]
            if event.event_id in state["events"]:
                previous = self._object("events", event.event_id, state["events"][event.event_id]["payload_hash"])
                if _digest(previous) != _digest(payload):
                    raise RecordConflict("A usage_event_id cannot be reused for different outcomes")
                return self._status(state, event.window_id, timestamp)
            if item["settlement_id"] is not None:
                raise CreditError("This immutable window is settled; register a new attribution window")
            protocol = CreditProtocol(**window["protocol"])
            record = self.replay.get(event.curate_record_id)
            if (
                event.curate_record_id != window["curate_record_id"]
                or _plain(event.skill_ref) != window["skill_ref"]
                or _plain(record.skill_ref) != window["skill_ref"]
                or self.replay.get_ref(record.record_id).payload_hash != window["replay_payload_hash"]
            ):
                raise CreditError("Usage event is not linked to the registered producer and skill version")
            if event.protocol_id != protocol.protocol_id:
                raise CreditError("Credit evaluator/probe protocol mismatch")
            for ref in window["replay_records"]:
                if self.replay.get_ref(ref["record_id"]).payload_hash != ref["payload_hash"]:
                    raise IntegrityError("A retained producer segment changed after credit registration")
            if not window["opened_at"] <= event.occurred_at <= window["deadline_at"]:
                raise CreditError("Usage event occurred outside its attribution window")
            if event.occurred_at < record.created_at:
                raise CreditError("A consumer event cannot precede its producer")
            if window["expected_probe_ids"] and event.probe_id not in window["expected_probe_ids"]:
                raise CreditError("Unexpected probe identity")
            for event_id in item["event_ids"]:
                previous = self._object("events", event_id, state["events"][event_id]["payload_hash"])
                if previous["probe_id"] == event.probe_id:
                    raise CreditError("The same exposure/probe cannot count twice under different event IDs")
            if len(item["event_ids"]) >= protocol.required_exposures:
                raise CreditError("Attribution window is full; use a new window for further exposures")
            positive, control = event.with_skill, event.without_skill
            for name in (
                "task_id", "model_version", "environment_version", "environment_state_hash", "baseline_snapshot_id"
            ):
                if getattr(positive, name) != getattr(control, name):
                    raise CreditError(f"Counterfactual pair differs in {name}")
            if (
                positive.model_version != protocol.evaluator_version
                or positive.environment_version != protocol.environment_version
                or positive.baseline_snapshot_id != protocol.baseline_snapshot_id
            ):
                raise CreditError("Counterfactual outcome does not use the registered evaluator/environment/baseline")
            if _plain(positive.loaded_skill_ref) != _plain(event.skill_ref) or control.loaded_skill_ref is not None:
                raise CreditError("Only the with-skill branch may load the attributed candidate version")
            if not all(protocol.score_min <= outcome.score <= protocol.score_max for outcome in (positive, control)):
                raise CreditError("Counterfactual score lies outside the declared reward bounds")
            binding_key = _digest([window["candidate_cohort_key"], event.probe_id])
            binding = {"task_id": positive.task_id, "environment_state_hash": positive.environment_state_hash}
            previous_binding = state["probe_bindings"].get(binding_key)
            if previous_binding is not None and previous_binding != binding:
                raise CreditError("Candidates in one cohort must share the same probe task and initial state")
            digest = _publish_immutable(self.root / "events" / (event.event_id + ".json"), payload)
            state["events"][event.event_id] = {"payload_hash": digest, "window_id": event.window_id}
            state["probe_bindings"][binding_key] = binding
            item["event_ids"].append(event.event_id)
            self._save_state(state)
            return self._status(state, event.window_id, timestamp)

    def settle(self, window_id: str, *, now: float | None = None) -> CreditSettlement:
        with _writer_lock(self.root):
            state = self._state()
            status = self._status(state, window_id, time.time() if now is None else _finite(now, "now"))
            item = state["windows"][window_id]
            if status["state"] == "settled":
                return self._settlement(state, item["settlement_id"])
            if status["state"] != "matured":
                raise WindowNotMature("Unexposed/pending/censored skills cannot receive an implicit zero utility label")
            window = self._window(state, window_id)
            protocol = CreditProtocol(**window["protocol"])
            events = [
                UsageEvent.from_dict(self._object("events", event_id, state["events"][event_id]["payload_hash"]))
                for event_id in item["event_ids"]
            ]
            if window["expected_probe_ids"]:
                order = {probe: i for i, probe in enumerate(window["expected_probe_ids"])}
                events.sort(key=lambda event: order[event.probe_id])
            else:
                events.sort(key=lambda event: (event.occurred_at, event.event_id))
            contributions = [event.with_skill.score - event.without_skill.score for event in events]
            raw = sum(contributions)
            if protocol.aggregation == "mean":
                raw /= len(contributions)
            label = {
                "window_id": window_id,
                "curate_record_id": window["curate_record_id"],
                "curate_record_ids": [ref["record_id"] for ref in window["replay_records"]],
                "skill_ref": window["skill_ref"],
                "protocol_id": protocol.protocol_id,
                "event_ids": [event.event_id for event in events],
                "contributions": contributions,
                "aggregation": protocol.aggregation,
                "raw_utility": raw,
                "skill_cost": window["skill_cost"],
                "cost_weight": protocol.cost_weight,
                "utility": raw - protocol.cost_weight * window["skill_cost"],
                "max_updates": protocol.max_updates,
            }
            digest = _digest(label)
            settlement = CreditSettlement("credit_" + digest, "sha256:" + digest, **label)
            object_hash = _publish_immutable(
                self.root / "settlements" / (settlement.settlement_id + ".json"), settlement.to_dict()
            )
            item["settlement_id"] = settlement.settlement_id
            state["settlements"][settlement.settlement_id] = {
                "payload_hash": object_hash, "window_id": window_id, "lease": None, "applied_updates": {}
            }
            self._save_state(state)
            return settlement

    def _settlement(self, state, settlement_id):
        item = state["settlements"].get(settlement_id)
        if item is None:
            raise CreditError(f"Unknown settlement: {settlement_id}")
        return CreditSettlement.from_dict(self._object("settlements", settlement_id, item["payload_hash"]))

    def get_settlement(self, settlement_id: str):
        return self._settlement(self._state(), settlement_id)

    def training_status(self, settlement_id: str, *, now: float | None = None):
        state = self._state()
        settlement = self._settlement(state, settlement_id)
        item = state["settlements"][settlement_id]
        timestamp = time.time() if now is None else _finite(now, "now")
        if item["lease"] is not None:
            phase = "leased" if item["lease"]["expires_at"] > timestamp else "needs_reconciliation"
        elif len(item["applied_updates"]) >= settlement.max_updates:
            phase = "applied"
        else:
            phase = "ready"
        return {"state": phase, "applied_update_ids": sorted(item["applied_updates"]), "lease": item["lease"]}

    def sample_ready(self, batch_size: int, *, seed: int = 0) -> list[CreditSettlement]:
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        state = self._state()
        ready = []
        for identity in sorted(state["settlements"]):
            item = state["settlements"][identity]
            settlement = self._settlement(state, identity)
            # Expired leases remain excluded until the learner reconciles them.
            if item["lease"] is None and len(item["applied_updates"]) < settlement.max_updates:
                ready.append(settlement)
        return random.Random(seed).sample(ready, min(batch_size, len(ready)))

    def lease(self, settlement_id: str, update_id: str, *, owner: str, lease_seconds=300.0, now=None):
        _identifier(update_id, "update_id")
        _identifier(owner, "lease owner")
        duration = _finite(lease_seconds, "lease_seconds")
        timestamp = time.time() if now is None else _finite(now, "now")
        if duration <= 0:
            raise ValueError("lease_seconds must be positive")
        with _writer_lock(self.root):
            state = self._state()
            settlement = self._settlement(state, settlement_id)
            item = state["settlements"][settlement_id]
            if update_id in item["applied_updates"]:
                return {"state": "already_applied", "model_update_id": item["applied_updates"][update_id]}
            if item["lease"] is not None:
                current = item["lease"]
                if current["expires_at"] <= timestamp:
                    raise LeaseConflict("Expired optimizer lease requires explicit model/checkpoint reconciliation")
                if current["update_id"] == update_id and current["owner"] == owner:
                    return dict(current)
                raise LeaseConflict("Settlement is already leased")
            if len(item["applied_updates"]) >= settlement.max_updates:
                raise LeaseConflict("The immutable label's update budget is exhausted")
            lease = {
                "update_id": update_id, "owner": owner, "lease_token": uuid.uuid4().hex,
                "expires_at": timestamp + duration,
            }
            item["lease"] = lease
            self._save_state(state)
            return dict(lease)

    def mark_applied(self, settlement_id, update_id, *, lease_token, model_update_id):
        if not isinstance(model_update_id, str) or not model_update_id:
            raise ValueError("A verifiable model update/checkpoint reference is required")
        with _writer_lock(self.root):
            state = self._state()
            self._settlement(state, settlement_id)
            item = state["settlements"][settlement_id]
            if update_id in item["applied_updates"]:
                if item["applied_updates"][update_id] != model_update_id:
                    raise RecordConflict("Applied update identity refers to a different model update")
                return
            lease = item["lease"]
            if not lease or lease["update_id"] != update_id or lease["lease_token"] != lease_token:
                raise LeaseConflict("The model update does not own this settlement lease")
            item["applied_updates"][update_id] = model_update_id
            item["lease"] = None
            self._save_state(state)

    def reconcile_update(self, settlement_id, update_id, *, was_applied, model_update_id=None):
        """Reconcile against a fenced/stopped actor or a restored joint checkpoint.

        A timeout alone is never evidence that an optimizer update did not run.
        """
        if type(was_applied) is not bool:
            raise ValueError("was_applied must be an explicit boolean decision")
        if was_applied and (not isinstance(model_update_id, str) or not model_update_id):
            raise ValueError("Applied reconciliation needs a verifiable model update reference")
        with _writer_lock(self.root):
            state = self._state()
            self._settlement(state, settlement_id)
            item = state["settlements"][settlement_id]
            lease = item["lease"]
            if not lease or lease["update_id"] != update_id:
                raise LeaseConflict("There is no matching uncertain optimizer update to reconcile")
            if was_applied:
                item["applied_updates"][update_id] = model_update_id
            item["lease"] = None
            state["reconciliations"].append(
                {"settlement_id": settlement_id, "update_id": update_id,
                 "was_applied": was_applied, "model_update_id": model_update_id}
            )
            self._save_state(state)

    def export_snapshot(self) -> DurableSnapshotRef:
        with _writer_lock(self.root):
            state = self._state()
            objects = []
            for kind in ("windows", "events", "settlements"):
                for identity, item in sorted(state[kind].items()):
                    self._object(kind, identity, item["payload_hash"])
                    objects.append({"relative_path": f"{kind}/{identity}.json", "payload_hash": item["payload_hash"]})
            replay_ref = self.replay.export_snapshot()
            payload = {
                "schema_version": SCHEMA_VERSION, "kind": "credit_ledger", "state": state,
                "objects": objects, "replay_snapshot": replay_ref.to_dict(),
            }
            digest = _digest(payload)
            path = self.root / "snapshots" / (digest + ".json")
            _publish_immutable(path, payload)
        return DurableSnapshotRef("sha256:" + digest, str(path), "credit_ledger", len(state["windows"]))

    def restore_snapshot(self, ref: DurableSnapshotRef, *, require_empty=True):
        """Restore only together with the matching actor/optimizer checkpoint.

        The default refuses an active state. Explicit rollback requires stopped
        writers; immutable later objects remain available for audit, not training.
        """
        payload, source_root = _snapshot_payload(ref, "credit_ledger")
        state = payload["state"]
        if state.get("kind") != "credit_ledger_state" or len(state["windows"]) != ref.item_count:
            raise IntegrityError("Ledger snapshot state/count mismatch")
        with _writer_lock(self.root):
            if require_empty and self._state()["windows"]:
                raise CreditError("Restore requires an empty ledger or an explicitly coordinated rollback")
            self.replay.restore_snapshot(DurableSnapshotRef(**payload["replay_snapshot"]))
            for item in payload["objects"]:
                relative = PurePosixPath(item["relative_path"])
                if len(relative.parts) != 2 or relative.parts[0] not in ("windows", "events", "settlements"):
                    raise IntegrityError("Invalid ledger snapshot object path")
                _identifier(Path(relative.name).stem)
                value = _read_payload(source_root / str(relative))
                if _digest(value) != item["payload_hash"]:
                    raise IntegrityError("Ledger snapshot object hash mismatch")
                _publish_immutable(self.root / str(relative), value)
            for kind in ("windows", "events", "settlements"):
                for identity, item in state[kind].items():
                    self._object(kind, identity, item["payload_hash"])
            _write_atomic(
                self.root / "state.json",
                {"schema_version": SCHEMA_VERSION, "payload_hash": _digest(state), "payload": state},
            )
        return {"restored_windows": len(state["windows"]), "snapshot_id": ref.snapshot_id}
