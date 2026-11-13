"""Configured native runtime providers and local task failure types.

The harness does not own a server account roster. A trusted deployment selects
an isolation provider using an importable factory or an explicitly registered
factory name. Providers own account validation, launch isolation and cleanup.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

POLICY_VERSION = "native-runtime-v1"


class TaskSetupError(RuntimeError):
    """An invalid or unavailable setup: discard this task and refill its group."""


class TaskExecutionError(RuntimeError):
    """A task runtime failure, without an observed environment outcome."""


class IsolationError(TaskExecutionError):
    """The configured native process/backend isolation could not be confirmed."""


# Compatibility for consumers that already distinguish runtime isolation errors.
AccountIsolationError = IsolationError


@runtime_checkable
class RuntimeProvider(Protocol):
    def acquire(self, *, run_root, scope_id, kind="ordinary"): ...
    def begin_attempt(self, ref, *, attempt_id, output, variant="ordinary"): ...
    def account(self, ref): ...
    def public(self, ref): ...
    def register_resource(self, ref, attempt_id, path, kind): ...
    def mark_launcher(self, ref, attempt_id, proof): ...
    def quarantine(self, ref, reason, *, attempt_id=None): ...
    def prepare_resource(self, path, *, account, kind): ...
    def prepare_work_root(self): ...
    def command_environment(self): ...
    def close_empty_attempt(self, ref, attempt_id, *, backend_drained): ...
    def drain(self, ref, *, timeout=5.0): ...
    def close_attempt(self, ref, attempt_id, *, receipt_path, backend_drained, native_started): ...
    def finish(self, ref, *, abandoned=False): ...
    def launch_arguments(self, account, argv, **kwargs): ...
    def decode_launcher_proof(self, data, **kwargs): ...


_FACTORIES = {}


def register_runtime_provider(name, factory):
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("A runtime provider requires a nonempty name and callable factory")
    if name in _FACTORIES and _FACTORIES[name] is not factory:
        raise ValueError("A different runtime factory is already registered")
    _FACTORIES[name] = factory


def deployment_runtime_config():
    """Read only an explicitly selected host deployment file."""
    path = os.environ.get("ALFWORLD_RUNTIME_CONFIG") or os.environ.get("ALFWORLD_SERVER_CONFIG")
    if not path:
        raise TaskSetupError("Configure a native runtime provider before running a task")
    try:
        return json.loads(Path(path).read_text())["account_pool"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TaskSetupError("The deployment must declare account_pool factory and settings") from error


class AccountPool:
    """Stable provider facade; no implicit host identities or fallback runtime."""

    def __init__(self, config=None):
        config = deployment_runtime_config() if config is None else config
        if not isinstance(config, Mapping) or not isinstance(config.get("factory"), str):
            raise TaskSetupError("account_pool must contain a configured factory")
        name = config["factory"]
        factory = _FACTORIES.get(name)
        if factory is None:
            module, separator, attribute = name.replace(":", ".").rpartition(".")
            if not separator:
                raise TaskSetupError("Runtime factory is neither registered nor fully qualified")
            try:
                factory = getattr(importlib.import_module(module), attribute)
            except (ImportError, AttributeError) as error:
                raise TaskSetupError("The configured runtime provider cannot be imported") from error
        kwargs = config.get("kwargs", {})
        if not callable(factory) or not isinstance(kwargs, Mapping):
            raise TaskSetupError("The runtime factory and its arguments are invalid")
        try:
            self.provider = factory(**dict(kwargs))
        except IsolationError:
            raise
        except Exception as error:
            raise TaskSetupError("The native runtime provider could not be configured") from error
        if not isinstance(self.provider, RuntimeProvider):
            raise TaskSetupError("The runtime provider does not implement the native lifecycle")

    def __getattr__(self, name):
        return getattr(self.provider, name)
