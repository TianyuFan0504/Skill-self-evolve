"""Cancel only an owned ALFWorld-terminal vLLM request, without synthetic EOS.

Completed preterminal requests retain the ordinary Gateway TITO path. A refused
request or a cancelled postterminal tail raises a control-plane exception; exact
partial IDs/logprobs/raw stop reasons are archived, never returned through the
Gateway codec's aborted-to-stop conversion. No global abort/cache reset is used.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import hashlib
import ipaddress
import json
import os
import secrets
import stat
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from verl.workers.rollout.llm_server import LLMServerClient

_CURRENT = contextvars.ContextVar("alfworld_owned_backend_request", default=None)
_DONE = {"completed", "failed", "terminal_refused", "terminal_tail_archived", "host_refused", "host_tail_archived"}
_HOST_STOP_KINDS = {"budget_timeout", "task_cancel", "host_error"}


class BackendControlStopped(RuntimeError):
    """A durable bound stop; never synthetic EOS or a task-result reward."""


class HostBackendStopped(BackendControlStopped):
    """Budget/cancellation/host-error control, distinct from absorbing MDP done."""


class TerminalMDPStopped(BackendControlStopped):
    """Proven absorbing MDP terminal control, not model EOS or native success."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".terminal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump({"schema_version": 1, "sha256": _digest(payload), "payload": payload}, output)
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


def _read(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("Terminal control files must be private and host-owned")
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("sha256") != _digest(value["payload"]):
        raise RuntimeError("Terminal control file integrity mismatch")
    return value["payload"]


def _session(url, prefix, suffix=""):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Terminal bindings require trusted plain HTTP session routes")
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_private
    except ValueError:
        local = False
    if not local:
        raise ValueError("Terminal bindings require literal private/local hosts")
    parts = parsed.path.split("/")
    expected = ["", prefix, "ID"] + ([suffix] if suffix else [])
    if len(parts) != len(expected) or parts[1] != prefix or (suffix and parts[-1] != suffix):
        raise ValueError("Unexpected trusted terminal session route")
    identity = parts[2]
    if not identity or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in identity
    ):
        raise ValueError("Invalid terminal session identity")
    return identity


@dataclass(frozen=True)
class TerminalBinding:
    binding_id: str
    binding_sha256: str
    gateway_session_id: str
    environment_session_id: str
    attempt_id: str
    environment_session_url: str = field(repr=False)
    root: str = field(repr=False)
    capability: str = field(repr=False)

    @property
    def directory(self):
        return Path(self.root) / self.binding_id


@contextlib.contextmanager
def _admission_lock(directory):
    import fcntl

    descriptor = os.open(directory / ".admission.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def bind_episode(binding_root, *, gateway_base_url, environment_session_url, attempt_id):
    """Bind only trusted host-created session/config objects before native launch."""
    gateway = _session(gateway_base_url, "sessions", "v1")
    environment = _session(environment_session_url, "session")
    if urlsplit(environment_session_url).hostname != "127.0.0.1" or not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("ALFWorld terminal observer must be the private loopback episode")
    root = Path(binding_root).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_uid != os.geteuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise RuntimeError("Terminal binding root must be private and host-owned")
    identity = hashlib.sha256(gateway.encode()).hexdigest()
    directory = root / identity
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "binding.json"
    values = {
        "binding_id": identity,
        "gateway_session_id": gateway,
        "environment_session_id": environment,
        "attempt_id": attempt_id,
        "environment_session_url": environment_session_url,
    }
    with _admission_lock(directory):
        if path.exists():
            payload = _read(path)
            if {key: payload[key] for key in values} != values:
                raise RuntimeError("A Gateway session cannot be rebound to another private attempt")
        else:
            payload = {**values, "capability": secrets.token_hex(32)}
            _write(path, payload)
    return TerminalBinding(**payload, root=str(root), binding_sha256=_digest(payload))


def _load(binding_root, session_id):
    root = Path(binding_root)
    path = root / hashlib.sha256(str(session_id).encode()).hexdigest() / "binding.json"
    if not path.exists():
        return None
    payload = _read(path)
    if payload["gateway_session_id"] != session_id:
        raise RuntimeError("Terminal binding belongs to another Gateway session")
    return TerminalBinding(**payload, root=str(root), binding_sha256=_digest(payload))


def _terminal_state(value):
    if not isinstance(value, dict) or any(type(value.get(key)) is not bool for key in ("done", "won")):
        return None
    if not value["done"] and not value["won"]:
        return None
    if any(type(value.get(key)) is not int for key in ("steps", "max_steps")):
        return None
    if not 0 <= value["steps"] <= value["max_steps"] or value["max_steps"] < 1:
        return None
    return {key: value[key] for key in ("done", "won", "steps", "max_steps")}


def _fenced(binding):
    path = binding.directory / "terminal.json"
    if not path.exists():
        return None
    proof = _read(path)
    if proof["binding_sha256"] != binding.binding_sha256 or proof["capability"] != binding.capability:
        raise RuntimeError("Terminal fence capability/attempt mismatch")
    return proof


def _stop_kind(proof):
    return proof.get("stop_kind", "mdp_terminal") if proof is not None else None


def _stop_exception(proof, message):
    return TerminalMDPStopped(message) if _stop_kind(proof) == "mdp_terminal" else HostBackendStopped(message)


def _fence(binding, state, *, stop_kind="mdp_terminal", reason=None):
    old = _fenced(binding)
    if old is not None:
        return old  # First bound stop is immutable; never upgrade a budget stop to MDP done.
    proof = {
        "binding_sha256": binding.binding_sha256,
        "capability": binding.capability,
        "attempt_id": binding.attempt_id,
        "gateway_session_id": binding.gateway_session_id,
        "environment_session_id": binding.environment_session_id,
        "stop_kind": stop_kind,
        "reason": reason,
        "terminal_state": state,
        "terminal_state_hash": _digest(state) if state is not None else None,
        "observed_at": time.time(),
    }
    # Retain the original private filename for existing MDP proof readers. Kind
    # and null state distinguish host control; no non-MDP stop invents done/EOS.
    _write(binding.directory / "terminal.json", proof)
    return proof


def fence_episode(binding, validated_private_terminal_state):
    """Atomically close admission using proof from this private observer only."""
    state = _terminal_state(validated_private_terminal_state)
    if state is None:
        raise ValueError("A terminal fence requires validated absorbing private environment state")
    if _load(binding.root, binding.gateway_session_id) != binding:
        raise RuntimeError("Terminal fence caller does not own the bound attempt")
    with _admission_lock(binding.directory):
        return _fence(binding, state)


def fence_host_stop(binding, stop_kind, *, reason=None):
    """Durably fence this attempt before its CLI is terminated or UID drained."""
    if stop_kind not in _HOST_STOP_KINDS:
        raise ValueError("Host stop kind must be budget_timeout, task_cancel, or host_error")
    if _load(binding.root, binding.gateway_session_id) != binding:
        raise RuntimeError("Host stop caller does not own the bound attempt")
    with _admission_lock(binding.directory):
        return _fence(binding, None, stop_kind=stop_kind, reason=reason)


def read_first_observed_input(binding):
    """Read only actual admission-0 input paired with a completed backend output."""
    if _load(binding.root, binding.gateway_session_id) != binding:
        raise RuntimeError("First-input reader does not own this binding")
    path = binding.directory / "first-observed-input.json"
    if not path.exists():
        return None
    payload = _read(path)
    if (
        payload.get("binding_sha256") != binding.binding_sha256
        or payload.get("attempt_id") != binding.attempt_id
        or payload.get("admission_index") != 0
        or payload.get("observed_on_completed_output") is not True
        or payload.get("raw_stop_reason") != "completed"
    ):
        raise RuntimeError("First-input proof does not describe this attempt's first completed backend request")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "payload": payload}


async def _observe(binding, timeout):
    def observe():
        with urllib.request.urlopen(binding.environment_session_url + "/observe", timeout=timeout) as response:
            return json.load(response)

    return await asyncio.to_thread(observe)


async def wait_for_terminal_drain(binding, timeout_seconds=15.0):
    """Return private identity-bound proof; timeout is explicitly unconfirmed."""
    current = _load(binding.root, binding.gateway_session_id)
    if current != binding:
        raise RuntimeError("Terminal drain caller does not own this bound attempt")
    if _fenced(binding) is None:
        try:
            state = _terminal_state(await _observe(binding, min(2.0, timeout_seconds)))
        except Exception:
            state = None
        if state is not None:
            fence_episode(binding, state)
    deadline = time.monotonic() + timeout_seconds
    while True:
        with _admission_lock(binding.directory):
            requests = [_read(path) for path in sorted((binding.directory / "requests").glob("*.json"))]
            active = [row for row in requests if row["phase"] not in _DONE or not row["backend_rpc_finished"]]
        if not active or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.05)
    proof = _fenced(binding)
    reasons = [row.get("reason") for row in requests]
    reason = (
        "terminal_abort"
        if "terminal_abort" in reasons
        else "terminal_refusal"
        if "terminal_refusal" in reasons
        else None
    )
    backend_errors = [row for row in requests if row.get("phase") == "failed" or row.get("error_type")]
    confirmed = not active and not backend_errors
    stop_kind = _stop_kind(proof)
    expected = bool(confirmed and stop_kind == "mdp_terminal" and reason)
    expected_host = bool(confirmed and stop_kind in _HOST_STOP_KINDS)
    status = (
        ("terminal_aborted" if reason == "terminal_abort" else "terminal_refused")
        if expected
        else "host_stopped"
        if expected_host
        else "drained"
        if confirmed
        else "backend_failed"
        if backend_errors
        else "unconfirmed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "binding_id": binding.binding_id,
        "binding_sha256": binding.binding_sha256,
        "attempt_id": binding.attempt_id,
        "gateway_session_id": binding.gateway_session_id,
        "environment_session_id": binding.environment_session_id,
        "fenced": proof is not None,
        "stop_kind": stop_kind,
        "host_stop_requested": stop_kind in _HOST_STOP_KINDS,
        "expected_host_stop": expected_host,
        "backend_error_types": sorted({row["error_type"] for row in backend_errors if row.get("error_type")}),
        "terminal_state": proof["terminal_state"] if proof else None,
        "terminal_state_hash": proof["terminal_state_hash"] if proof else None,
        "active_requests": len(active),
        "backend_stop_confirmed": confirmed,
        "expected_backend_terminal_stop": expected,
        "expected_terminal_stop": expected,
        "reason": reason,
        "requests": requests,
        "tail_archive_refs": [row["tail_archive_ref"] for row in requests if row.get("tail_archive_ref")],
    }


@dataclass
class _Lease:
    binding: TerminalBinding
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex)
    server: object = None
    server_id: str | None = None
    actual_request_id: str | None = None
    terminal: bool = False
    abort_attempts: int = 0
    abort_acknowledged: bool = False
    phase: str = "checking"
    reason: str | None = None
    stop_kind: str | None = None
    admission_index: int | None = None
    failed_input_ref: dict | None = None
    failed_input_archive_error: str | None = None


class TerminalAwareLLMServerClient(LLMServerClient):
    """Use official client hooks; retain the backend future across disconnects."""

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        binding_root,
        poll_seconds=0.25,
        abort_retry_seconds=0.10,
        abort_timeout_seconds=15.0,
        observation_timeout_seconds=2.0,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self.binding_root, self.poll_seconds = str(Path(binding_root).resolve()), poll_seconds
        self.abort_retry_seconds, self.abort_timeout_seconds = abort_retry_seconds, abort_timeout_seconds
        self.observation_timeout_seconds = observation_timeout_seconds
        self._owned_tasks = set()

    def _record(self, lease, *, finished=False, tail=None, raw_stop_reason=None, error_type=None):
        data = {
            "binding_sha256": lease.binding.binding_sha256,
            "request_nonce": lease.nonce,
            "admission_index": lease.admission_index,
            "stop_kind": lease.stop_kind,
            "gateway_session_id": lease.binding.gateway_session_id,
            "attempt_id": lease.binding.attempt_id,
            "server_id": lease.server_id,
            "actual_request_id": lease.actual_request_id,
            "phase": lease.phase,
            "reason": lease.reason,
            "backend_rpc_finished": finished,
            "abort_attempts": lease.abort_attempts,
            "abort_acknowledged": lease.abort_acknowledged,
            "raw_stop_reason": raw_stop_reason,
            "tail_archive_ref": tail,
            "error_type": error_type,
        }
        if lease.failed_input_ref is not None:
            data["failed_input_ref"] = lease.failed_input_ref
        if lease.failed_input_archive_error is not None:
            data["failed_input_archive_error"] = lease.failed_input_archive_error
        _write(lease.binding.directory / "requests" / (lease.nonce + ".json"), data)

    @staticmethod
    def _archive_failed_client_input(lease, kwargs, error):
        """Retain client submission evidence, never a model-observed input proof."""
        prompt_ids = list(kwargs["prompt_ids"])
        sampling_params = kwargs["sampling_params"]
        payload = {
            "schema_version": 1,
            "kind": "unexpected_backend_rpc_client_input",
            "evidence_stage": "CLIENT_INPUT_PRE_SUBMIT",
            "binding_id": lease.binding.binding_id,
            "binding_sha256": lease.binding.binding_sha256,
            "attempt_id": lease.binding.attempt_id,
            "gateway_session_id": lease.binding.gateway_session_id,
            "environment_session_id": lease.binding.environment_session_id,
            "request_nonce": lease.nonce,
            "admission_index": lease.admission_index,
            "server_id": lease.server_id,
            "actual_request_id": lease.actual_request_id,
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": _digest(prompt_ids),
            "sampling_params": sampling_params,
            "sampling_params_sha256": _digest(sampling_params),
            "sampling_params_scope": "client_input_before_server_defaults_or_clamping",
            "backend_exception_type": type(error).__name__,
            "actual_model_observed": False,
            "cache_state_captured": False,
            "trajectory_complete": False,
            "archive_only": True,
            "reward_eligible": False,
            "behavior_gradient": False,
            "model_eos_invented": False,
        }
        path = lease.binding.directory / "failed-inputs" / (lease.nonce + ".json")
        _write(path, payload)
        lease.failed_input_ref = {
            "path": str(path),
            "sha256": _digest(payload),
            "prompt_ids_sha256": payload["prompt_ids_sha256"],
            "evidence_stage": payload["evidence_stage"],
            "actual_model_observed": False,
            "archive_only": True,
        }

    @staticmethod
    def _note_secondary_failure(error, detail):
        # Python 3.10 lacks add_note but still permits diagnostic attributes.
        if callable(getattr(error, "add_note", None)):
            error.add_note(detail)
        else:
            error.__notes__ = [*getattr(error, "__notes__", []), detail]

    async def _acquire_server(self, request_id, **extra):
        result = await super()._acquire_server(request_id, **extra)
        lease = _CURRENT.get()
        if lease is not None:
            lease.server_id, lease.server = result
            lease.phase = "acquired"
            self._record(lease)
        return result

    def _vllm_request_id(self, request_id):
        actual = super()._vllm_request_id(request_id)
        lease = _CURRENT.get()
        if lease is not None:
            lease.actual_request_id = actual
            proof = _fenced(lease.binding)
            if proof:
                self._mark_stopped(lease, proof, "refusal")
                raise _stop_exception(proof, "A bound stop refused a new backend submission")
            lease.phase = "submitting"
            self._record(lease)
        return actual

    async def _watch(self, lease, future):
        while not future.done():
            proof = _fenced(lease.binding)
            if proof is None:
                try:
                    state = _terminal_state(await _observe(lease.binding, self.observation_timeout_seconds))
                except Exception:
                    state = None  # Unknown observation never authorizes cancellation.
                if future.done():
                    return
                if state is not None:
                    proof = fence_episode(lease.binding, state)
            if proof is not None:
                lease.terminal = True
                lease.stop_kind = _stop_kind(proof)
                if lease.server is not None and lease.actual_request_id is not None:
                    prefix = "terminal" if lease.stop_kind == "mdp_terminal" else lease.stop_kind
                    lease.reason, lease.phase = prefix + "_abort", "cancelling"
                    lease.abort_attempts += 1
                    self._record(lease)
                    try:
                        result = await asyncio.wait_for(
                            lease.server.abort_request.remote(lease.actual_request_id, reset_prefix_cache=False),
                            timeout=min(self.observation_timeout_seconds, self.abort_timeout_seconds),
                        )
                        lease.abort_acknowledged |= result.get("aborted") is True
                    except Exception:
                        pass  # Keep ownership and retry until the real generate future resolves.
                await asyncio.sleep(self.abort_retry_seconds)
            else:
                await asyncio.sleep(self.poll_seconds)

    def _mark_stopped(self, lease, proof, action):
        lease.terminal, lease.stop_kind = True, _stop_kind(proof)
        prefix = "terminal" if lease.stop_kind == "mdp_terminal" else lease.stop_kind
        lease.reason = prefix + "_" + action
        lease.phase = "terminal_refused" if lease.stop_kind == "mdp_terminal" else "host_refused"

    def _first_observed(self, lease, kwargs, output):
        if lease.admission_index != 0 or output.stop_reason != "completed":
            return
        if not lease.actual_request_id:
            raise RuntimeError("A completed first-input proof requires the actual backend request identity")
        prompt_ids = list(kwargs["prompt_ids"])
        raw_output = {
            "token_ids": list(output.token_ids),
            "log_probs": None if output.log_probs is None else list(output.log_probs),
            "stop_reason": output.stop_reason,
            "extra_fields": dict(output.extra_fields),
        }
        payload = {
            "binding_sha256": lease.binding.binding_sha256,
            "attempt_id": lease.binding.attempt_id,
            "gateway_session_id": lease.binding.gateway_session_id,
            "environment_session_id": lease.binding.environment_session_id,
            "admission_index": 0,
            "actual_request_id": lease.actual_request_id,
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": _digest(prompt_ids),
            "output_sha256": _digest(raw_output),
            "raw_stop_reason": output.stop_reason,
            "observed_on_completed_output": True,
            "trajectory_complete": False,
            "archive_only": True,
            "behavior_gradient": False,
            "observed_at": time.time(),
        }
        with _admission_lock(lease.binding.directory):
            path = lease.binding.directory / "first-observed-input.json"
            if path.exists():
                old = _read(path)
                if any(old.get(k) != payload[k] for k in payload if k != "observed_at"):
                    raise RuntimeError("The immutable actual first-input proof changed")
            else:
                _write(path, payload)

    async def generate(self, request_id, **kwargs):
        binding = _load(self.binding_root, str(request_id))
        if binding is None:
            return await super().generate(request_id, **kwargs)
        kwargs = dict(
            kwargs,
            prompt_ids=list(kwargs["prompt_ids"]),
            sampling_params=copy.deepcopy(kwargs["sampling_params"]),
        )
        lease = _Lease(binding)
        with _admission_lock(binding.directory):
            proof = _fenced(binding)
            if proof is not None:
                self._mark_stopped(lease, proof, "refusal")
                self._record(lease, finished=True)
                raise _stop_exception(proof, "A durable bound stop closed backend admission")
            counter_path = binding.directory / "admission-counter.json"
            counter = (
                _read(counter_path)
                if counter_path.exists()
                else {"binding_sha256": binding.binding_sha256, "next_index": 0}
            )
            if counter["binding_sha256"] != binding.binding_sha256:
                raise RuntimeError("Backend admission counter belongs to another attempt")
            lease.admission_index = counter["next_index"]
            _write(counter_path, {**counter, "next_index": lease.admission_index + 1})
            self._record(lease)  # Durable before the owned task can submit any GPU work.
        task = asyncio.create_task(self._generate_owned(lease, request_id, kwargs))
        self._owned_tasks.add(task)

        def completed(value):
            self._owned_tasks.discard(value)
            if not value.cancelled():
                value.exception()

        task.add_done_callback(completed)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            fence_host_stop(binding, "task_cancel", reason="gateway_frontend_cancelled")
            raise  # The owned backend future and watcher retain their ownership.

    async def _generate_owned(self, lease, request_id, kwargs):
        binding = lease.binding
        token = _CURRENT.set(lease)
        watcher = future = None
        rpc_failure = False
        backend_failure_error = None
        self._record(lease)
        try:
            proof = _fenced(binding)
            if proof is None:
                try:
                    state = _terminal_state(await _observe(binding, self.observation_timeout_seconds))
                except Exception:
                    state = None
                if state is not None:
                    proof = fence_episode(binding, state)
            if proof is not None:
                self._mark_stopped(lease, proof, "refusal")
                raise _stop_exception(proof, "A bound stop refused an unnecessary decode")
            future = asyncio.create_task(super().generate(request_id, **kwargs))
            watcher = asyncio.create_task(self._watch(lease, future))
            while True:
                try:
                    output = await asyncio.shield(future)
                    break
                except asyncio.CancelledError:
                    if future.cancelled():
                        rpc_failure = True
                        raise RuntimeError("The original backend future was cancelled without drain proof") from None
                    fence_host_stop(binding, "task_cancel", reason="owned_client_task_cancelled")
                    # Cancellation of this control task does not prove kernel drain.
                    # Keep the original future plus its UUID-targeted watcher alive.
                except Exception:
                    rpc_failure = True
                    raise
            self._first_observed(lease, kwargs, output)
            proof = _fenced(binding)
            if lease.terminal or proof or output.stop_reason != "completed":
                if proof:
                    lease.stop_kind = _stop_kind(proof)
                lease.reason = lease.reason or (
                    ("terminal_refusal" if lease.stop_kind == "mdp_terminal" else lease.stop_kind + "_refusal")
                    if lease.stop_kind
                    else "unconfirmed_backend_stop"
                )
                payload = {
                    "binding_sha256": binding.binding_sha256,
                    "attempt_id": binding.attempt_id,
                    "gateway_session_id": binding.gateway_session_id,
                    "actual_request_id": lease.actual_request_id,
                    "stop_kind": lease.stop_kind,
                    "prompt_ids": list(kwargs["prompt_ids"]),
                    "token_ids": list(output.token_ids),
                    "log_probs": None if output.log_probs is None else list(output.log_probs),
                    "raw_stop_reason": output.stop_reason,
                    "extra_fields": dict(output.extra_fields),
                    "archive_only": True,
                    "model_eos_invented": False,
                    "behavior_gradient": False,
                }
                path = binding.directory / "tails" / (lease.nonce + ".json")
                _write(path, payload)
                tail = {
                    "path": str(path),
                    "sha256": _digest(payload),
                    "token_count": len(output.token_ids),
                    "raw_stop_reason": output.stop_reason,
                    "archive_only": True,
                }
                if proof is None or output.stop_reason not in ("completed", "aborted"):
                    lease.phase = "failed"
                    self._record(
                        lease,
                        finished=True,
                        tail=tail,
                        raw_stop_reason=output.stop_reason,
                        error_type="UnconfirmedBackendStop",
                    )
                    raise RuntimeError("An unconfirmed backend stop is not a normal task outcome")
                lease.phase = "terminal_tail_archived" if lease.stop_kind == "mdp_terminal" else "host_tail_archived"
                self._record(lease, finished=True, tail=tail, raw_stop_reason=output.stop_reason)
                raise _stop_exception(proof, "Bound decode stopped; exact raw tail was archived")
            lease.phase = "completed"
            self._record(lease, finished=True, raw_stop_reason=output.stop_reason)
            return output
        except BackendControlStopped:
            if lease.phase not in ("terminal_tail_archived", "host_tail_archived"):
                lease.phase = "terminal_refused" if lease.stop_kind == "mdp_terminal" else "host_refused"
                self._record(lease, finished=True)
            raise
        except BaseException as error:
            if rpc_failure:
                backend_failure_error = error
                try:
                    self._archive_failed_client_input(lease, kwargs, error)
                except Exception as archive_error:
                    lease.failed_input_archive_error = type(archive_error).__name__ + ": " + str(archive_error)
                    self._note_secondary_failure(
                        error, "Failed client-input archive: " + lease.failed_input_archive_error
                    )
            previous = {}
            try:
                previous = _read(binding.directory / "requests" / (lease.nonce + ".json"))
            except Exception as journal_error:
                self._note_secondary_failure(error, "Failure journal read: " + str(journal_error))
            lease.phase = "failed"
            try:
                self._record(
                    lease,
                    finished=future is None or future.done(),
                    tail=previous.get("tail_archive_ref"),
                    raw_stop_reason=previous.get("raw_stop_reason"),
                    error_type=type(error).__name__,
                )
            except Exception as journal_error:
                self._note_secondary_failure(error, "Failure journal write: " + str(journal_error))
            raise
        finally:
            if watcher is not None:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
                except Exception as watcher_error:
                    if backend_failure_error is None:
                        raise
                    self._note_secondary_failure(
                        backend_failure_error, "Failure watcher cleanup: " + str(watcher_error)
                    )
            _CURRENT.reset(token)
