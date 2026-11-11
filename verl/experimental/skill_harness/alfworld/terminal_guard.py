# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Host-only terminal-MDP supervision of one owned native CLI process group.

The caller binds observe/is_alive/terminate to the SAME private episode and
owned process handle. terminate must target only that handle's process group;
this module never discovers processes or sends OS signals itself. A trusted,
absorbing terminal state arms a grace period. Missing/malformed observations,
transport failures and elapsed wall time alone never authorize termination.

Evidence is control-plane metadata: it invents no reward, accuracy, finished
trajectory, native exit code or behavior logprob. The caller preserves those
independently. An intentional terminal stop can still have a nonzero raw native
exit code. Cancellation belongs to the caller's existing process cleanup.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class TerminalStopEvidence:
    status: str
    process_id: int | None
    started_at_epoch: float
    finished_at_epoch: float
    terminal_observed_at_epoch: float | None
    terminal_state: Mapping[str, Any] | None
    terminal_state_hash: str | None
    grace_seconds: float
    elapsed_after_terminal_seconds: float | None
    observations: int
    unknown_observations: int
    observation_error_count: int
    observation_error_types: tuple[str, ...]
    termination_requested: bool
    termination_completed: bool
    alive_after_termination: bool | None
    error_type: str | None
    schema_version: int = 1

    def __post_init__(self):
        if self.terminal_state is not None:
            object.__setattr__(self, "terminal_state", MappingProxyType(dict(self.terminal_state)))

    def to_dict(self):
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["terminal_state"] = None if self.terminal_state is None else dict(self.terminal_state)
        result["observation_error_types"] = list(self.observation_error_types)
        return result


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


def _validated_state(value):
    """Return None for UNKNOWN, not a fabricated nonterminal/terminal state."""
    if not isinstance(value, Mapping) or any(type(value.get(key)) is not bool for key in ("done", "won")):
        return None
    selected = {"done": value["done"], "won": value["won"]}
    for key in ("steps", "max_steps"):
        if key in value:
            if type(value[key]) is not int or value[key] < (1 if key == "max_steps" else 0):
                return None
            selected[key] = value[key]
    if "steps" in selected and "max_steps" in selected and selected["steps"] > selected["max_steps"]:
        return None
    # Hash only the exact validated terminal control fields. Do not persist
    # credentials, observations, answers or arbitrary observer payloads.
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return selected, digest


def _attach(error, name, value):
    try:
        setattr(error, name, value)
    except (AttributeError, TypeError):
        pass  # Never replace the original exception just to attach diagnostics.


async def monitor_terminal_process(
    observe,
    is_alive,
    terminate,
    *,
    process_id: int | None = None,
    grace_seconds: float = 5.0,
    poll_seconds: float = 0.25,
    observation_timeout_seconds: float = 2.0,
    on_evidence=None,
) -> TerminalStopEvidence:
    """Observe until normal exit or proven terminal state plus bounded grace.

    Callbacks may return values or awaitables and must be nonblocking when
    synchronous. observe returns this fixed episode's public state mapping.
    is_alive returns a strict bool; terminate performs the caller's bounded,
    idempotent owned-process-group cleanup. It is called at most once here.

    Transient observer exceptions/timeouts are counted and retried. Liveness or
    termination failures propagate unchanged. Cancellation propagates unchanged
    and does NOT invoke terminate: the owner retains its cancellation cleanup.
    Evidence is returned normally, and attached as ``terminal_stop_evidence``
    to propagated exceptions. Optional on_evidence receives it exactly once.
    A failing evidence sink never masks an already-propagating exception; its
    error is attached as ``terminal_stop_evidence_delivery_error`` instead.
    """
    for name, value, minimum in (("grace_seconds", grace_seconds, 0), ("poll_seconds", poll_seconds, 0),
                                 ("observation_timeout_seconds", observation_timeout_seconds, 0)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < minimum or name != "grace_seconds" and value == 0:
            raise ValueError(f"{name} must be {'nonnegative' if name == 'grace_seconds' else 'positive'}")
    if process_id is not None and (type(process_id) is not int or process_id <= 0):
        raise ValueError("process_id must identify the caller's owned process")
    if not all(callable(callback) for callback in (observe, is_alive, terminate)):
        raise TypeError("observe/is_alive/terminate must be callable")
    if on_evidence is not None and not callable(on_evidence):
        raise TypeError("on_evidence must be callable")

    started = time.time()
    terminal_epoch = terminal_monotonic = None
    terminal_state = terminal_hash = None
    observations = unknown = error_count = 0
    error_types = []
    requested = completed = False
    alive_after = None
    phase = "liveness"

    def evidence(status, error=None):
        return TerminalStopEvidence(
            status=status, process_id=process_id, started_at_epoch=started, finished_at_epoch=time.time(),
            terminal_observed_at_epoch=terminal_epoch, terminal_state=terminal_state,
            terminal_state_hash=terminal_hash, grace_seconds=grace_seconds,
            elapsed_after_terminal_seconds=(
                None if terminal_monotonic is None else time.monotonic() - terminal_monotonic
            ),
            observations=observations, unknown_observations=unknown, observation_error_count=error_count,
            observation_error_types=tuple(error_types), termination_requested=requested,
            termination_completed=completed, alive_after_termination=alive_after,
            error_type=None if error is None else type(error).__name__,
        )

    async def alive():
        value = await _resolve(is_alive())
        if type(value) is not bool:
            raise TypeError("is_alive must return a strict bool; unknown liveness cannot authorize a stop")
        return value

    try:
        while True:
            phase = "liveness"
            if not await alive():
                status = "native_completed" if terminal_state is None else "native_completed_after_terminal"
                break
            if terminal_monotonic is not None:
                remaining = terminal_monotonic + grace_seconds - time.monotonic()
                if remaining <= 0:
                    # Recheck after the grace deadline; natural completion wins
                    # the race and must not be reclassified as a terminal stop.
                    if not await alive():
                        status = "native_completed_after_terminal"
                        break
                    phase = "termination"
                    requested = True
                    await _resolve(terminate())
                    completed = True
                    phase = "liveness"
                    alive_after = await alive()
                    status = "terminal_stop_unconfirmed" if alive_after else "terminal_stop"
                    break
                await asyncio.sleep(min(poll_seconds, remaining))
                continue
            phase = "observation"
            observations += 1
            try:
                value = await asyncio.wait_for(_resolve(observe()), timeout=observation_timeout_seconds)
            except Exception as error:
                error_count += 1
                if type(error).__name__ not in error_types and len(error_types) < 8:
                    error_types.append(type(error).__name__)
            else:
                state = _validated_state(value)
                if state is None:
                    unknown += 1
                elif state[0]["done"] or state[0]["won"]:
                    terminal_state, terminal_hash = state
                    terminal_epoch, terminal_monotonic = time.time(), time.monotonic()
                    continue
            await asyncio.sleep(poll_seconds)
    except BaseException as error:
        result = evidence("cancelled" if isinstance(error, asyncio.CancelledError) else phase + "_error", error)
        _attach(error, "terminal_stop_evidence", result)
        if on_evidence is not None:
            try:
                await _resolve(on_evidence(result))
            except BaseException as delivery_error:
                _attach(error, "terminal_stop_evidence_delivery_error", delivery_error)
        raise
    result = evidence(status)
    if on_evidence is not None:
        try:
            await _resolve(on_evidence(result))
        except BaseException as error:
            _attach(error, "terminal_stop_evidence", result)
            raise
    return result
