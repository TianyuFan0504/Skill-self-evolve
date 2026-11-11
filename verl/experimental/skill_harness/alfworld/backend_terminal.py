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
_DONE = {"completed", "failed", "terminal_refused", "terminal_tail_archived"}


class TerminalMDPStopped(RuntimeError):
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


def _fence(binding, state):
    old = _fenced(binding)
    if old is not None:
        return old
    proof = {
        "binding_sha256": binding.binding_sha256,
        "capability": binding.capability,
        "terminal_state": state,
        "terminal_state_hash": _digest(state),
        "observed_at": time.time(),
    }
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
    confirmed = not active
    expected = bool(confirmed and proof and reason)
    status = (
        ("terminal_aborted" if reason == "terminal_abort" else "terminal_refused")
        if expected
        else "drained"
        if confirmed
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
        _write(lease.binding.directory / "requests" / (lease.nonce + ".json"), data)

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
            if _fenced(lease.binding):
                lease.terminal, lease.reason, lease.phase = True, "terminal_refusal", "terminal_refused"
                raise TerminalMDPStopped("ALFWorld terminal fence refused a new backend submission")
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
                if lease.server is not None and lease.actual_request_id is not None:
                    lease.reason, lease.phase = "terminal_abort", "cancelling"
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

    async def generate(self, request_id, **kwargs):
        binding = _load(self.binding_root, str(request_id))
        if binding is None:
            return await super().generate(request_id, **kwargs)
        lease = _Lease(binding)
        # Admission is durable BEFORE scheduling the owned task. A concurrent
        # drain/fence uses the same interprocess lock, so empty scans cannot
        # miss a queued backend task that could later submit GPU work.
        with _admission_lock(binding.directory):
            self._record(lease)
        task = asyncio.create_task(self._generate_owned(lease, request_id, kwargs))
        self._owned_tasks.add(task)

        def completed(value):
            self._owned_tasks.discard(value)
            if not value.cancelled():
                value.exception()  # Retrieve an orphaned exception; artifacts retain its cause.

        task.add_done_callback(completed)
        return await asyncio.shield(task)

    async def _generate_owned(self, lease, request_id, kwargs):
        binding = lease.binding
        token = _CURRENT.set(lease)
        watcher = None
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
                lease.terminal, lease.reason, lease.phase = True, "terminal_refusal", "terminal_refused"
                raise TerminalMDPStopped("ALFWorld absorbing terminal state refused an unnecessary decode")
            future = asyncio.create_task(super().generate(request_id, **kwargs))
            watcher = asyncio.create_task(self._watch(lease, future))
            output = await future
            if lease.terminal or _fenced(binding):
                lease.reason = lease.reason or "terminal_refusal"
                payload = {
                    "binding_sha256": binding.binding_sha256,
                    "attempt_id": binding.attempt_id,
                    "gateway_session_id": binding.gateway_session_id,
                    "actual_request_id": lease.actual_request_id,
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
                lease.phase = "terminal_tail_archived"
                self._record(lease, finished=True, tail=tail, raw_stop_reason=output.stop_reason)
                raise TerminalMDPStopped("ALFWorld terminal decode stopped; exact raw tail was archived")
            lease.phase = "completed"
            self._record(lease, finished=True, raw_stop_reason=output.stop_reason)
            return output
        except TerminalMDPStopped:
            if lease.phase != "terminal_tail_archived":
                lease.phase = "terminal_refused"
                self._record(lease, finished=True)
            raise
        except BaseException as error:
            lease.phase = "failed"
            self._record(lease, finished=True, error_type=type(error).__name__)
            raise
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            _CURRENT.reset(token)
