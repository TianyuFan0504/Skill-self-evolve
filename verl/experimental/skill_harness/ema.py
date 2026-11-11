# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Parameter-only EMA of local tensors, independent of Ray and model engines.

Floating parameters have detached FP32 shadows. Buffers are either copied from
the most recent optimizer update, without averaging, or excluded explicitly.
DTensors are inspected through ``to_local()``; no collective or full gather is
performed here. State files therefore belong to one particular worker shard.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

TensorSource = Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]]
TensorProvider = TensorSource | Callable[[], TensorSource]


def _tensors(source: TensorProvider | None) -> dict[str, torch.Tensor]:
    if source is None:
        return {}
    source = source() if callable(source) else source
    pairs = list(source.items()) if isinstance(source, Mapping) else list(source)
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("Tensor names must be unique")
    result = {}
    for name, value in pairs:
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise TypeError("EMA sources must contain named torch tensors")
        result[name] = value
    return result


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Return only this rank's data; never call DTensor.full_tensor()."""
    result = value.to_local() if hasattr(value, "to_local") else value
    if result.is_meta or result.layout != torch.strided:
        raise ValueError("EMA requires materialized, strided local tensors")
    return result.detach()


def tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    local = local_tensor(value)
    result = {"global_shape": list(value.shape), "local_shape": list(local.shape), "dtype": str(local.dtype)}
    if hasattr(value, "device_mesh"):
        mesh = value.device_mesh
        result.update(
            placements=[str(item) for item in value.placements],
            mesh_shape=list(mesh.mesh.shape),
            mesh_coordinate=mesh.get_coordinate(),
            mesh_device_type=mesh.device_type,
        )
    return result


def describe_local_layout(
    parameters: TensorProvider, buffers: TensorProvider | None = None, *, buffer_policy: str = "copy",
) -> dict[str, Any]:
    """Inspect current local shards without copying values or creating an EMA clock."""
    if buffer_policy not in {"copy", "exclude"}:
        raise ValueError("buffer_policy must be copy or exclude")
    params = _tensors(parameters)
    bufs = _tensors(buffers) if buffer_policy == "copy" else {}
    descriptor = {
        "parameters": {name: tensor_layout(value) for name, value in params.items()},
        "buffers": {name: tensor_layout(value) for name, value in bufs.items()},
    }
    return {
        "layout_fingerprint": hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "parameter_count": len(params),
        "local_parameter_elements": sum(local_tensor(value).numel() for value in params.values()),
    }


def require_single_optimizer_step(
    local_batch_size: int,
    dp_size: int,
    *,
    mini_batch_size: int | None = None,
    num_mini_batch: int | None = None,
    epochs: int = 1,
) -> None:
    """Fail before training if one RPC would contain multiple optimizer steps."""
    if type(local_batch_size) is not int or local_batch_size <= 0 or type(dp_size) is not int or dp_size <= 0:
        raise ValueError("Batch and data-parallel sizes must be positive integers")
    if type(epochs) is not int or epochs != 1:
        raise ValueError("EMA foundation requires exactly one epoch per update RPC")
    if num_mini_batch is not None and (type(num_mini_batch) is not int or num_mini_batch != 1):
        raise ValueError("EMA foundation requires exactly one mini-batch per update RPC")
    if mini_batch_size is None:
        if num_mini_batch != 1:
            raise ValueError("Specify a single mini-batch explicitly")
    elif type(mini_batch_size) is not int or mini_batch_size != local_batch_size * dp_size:
        raise ValueError("Global mini_batch_size must equal the actual distributed batch size")


class ParameterEMA:
    """A detached local-shard EMA with an explicit optimizer-step clock.

    ``optimizer_step`` is supplied by an observer of actual optimizer.step()
    calls. Duplicate, skipped, or reversed clocks fail before modifying shadows.
    The tracker does not own an optimizer and does not inspect gradients.
    """

    SCHEMA_VERSION = 1

    def __init__(self, decay: float = 0.95, *, shadow_device: str = "cpu", buffer_policy: str = "copy"):
        if not isinstance(decay, float | int) or not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        if buffer_policy not in ("copy", "exclude"):
            raise ValueError("buffer_policy must be 'copy' or 'exclude'")
        self.decay = float(decay)
        self.shadow_device = torch.device(shadow_device)
        self.buffer_policy = buffer_policy
        self.parameters: dict[str, torch.Tensor] = {}
        self.buffers: dict[str, torch.Tensor] = {}
        self.parameter_layouts: dict[str, dict] = {}
        self.buffer_layouts: dict[str, dict] = {}
        self.initial_optimizer_step: int | None = None
        self.last_optimizer_step: int | None = None
        self.num_updates = 0
        self._swapped = False
        self._guard = threading.Lock()

    @property
    def initialized(self) -> bool:
        return self.last_optimizer_step is not None

    @property
    def swapped(self) -> bool:
        return self._swapped

    def _idle(self) -> None:
        if self._swapped:
            raise RuntimeError("EMA export is active; online updates and checkpointing must wait")

    @staticmethod
    def _clock(step: int) -> int:
        if type(step) is not int or step < 0:
            raise ValueError("optimizer_step must be a nonnegative integer")
        return step

    def _capture(self, source: Mapping[str, torch.Tensor], *, parameters: bool) -> dict[str, torch.Tensor]:
        result = {}
        with torch.no_grad():
            for name, value in source.items():
                local = local_tensor(value)
                if parameters and not local.is_floating_point():
                    raise TypeError(f"EMA parameter {name} is not floating point")
                dtype = torch.float32 if parameters else local.dtype
                copied = local.to(device=self.shadow_device, dtype=dtype, copy=True).detach()
                if copied.is_floating_point() and not torch.isfinite(copied).all():
                    raise ValueError(f"Nonfinite tensor in EMA source: {name}")
                result[name] = copied
        return result

    def _validate(self, parameters: Mapping[str, torch.Tensor], buffers: Mapping[str, torch.Tensor]) -> None:
        if not self.initialized:
            raise RuntimeError("EMA has not been initialized")
        current = {name: tensor_layout(value) for name, value in parameters.items()}
        if current != self.parameter_layouts:
            raise ValueError("EMA parameter keys, shapes, dtypes, or local shard layouts changed")
        if self.buffer_policy == "copy":
            current_buffers = {name: tensor_layout(value) for name, value in buffers.items()}
            if current_buffers != self.buffer_layouts:
                raise ValueError("EMA buffer keys, shapes, dtypes, or layouts changed; use exclude for dynamic caches")

    def initialize(self, parameters: TensorProvider, buffers: TensorProvider | None = None, *, optimizer_step: int = 0):
        self._idle()
        if self.initialized:
            raise RuntimeError("Create a new tracker to explicitly reset EMA state")
        step = self._clock(optimizer_step)
        params = _tensors(parameters)
        if not params:
            raise ValueError("EMA requires at least one parameter")
        bufs = _tensors(buffers) if self.buffer_policy == "copy" else {}
        captured = self._capture(params, parameters=True)
        captured_buffers = self._capture(bufs, parameters=False)
        self.parameter_layouts = {name: tensor_layout(value) for name, value in params.items()}
        self.buffer_layouts = {name: tensor_layout(value) for name, value in bufs.items()}
        self.parameters = captured
        self.buffers = captured_buffers
        self.initial_optimizer_step = self.last_optimizer_step = step
        return self.metadata()

    def update(self, parameters: TensorProvider, buffers: TensorProvider | None = None, *, optimizer_step: int):
        self._idle()
        step = self._clock(optimizer_step)
        if not self.initialized or step != self.last_optimizer_step + 1:
            raise ValueError("EMA must advance exactly once for the next observed optimizer update")
        params = _tensors(parameters)
        bufs = _tensors(buffers) if self.buffer_policy == "copy" else {}
        self._validate(params, bufs)
        # Validate all incoming tensors before changing any existing shadow.
        incoming = self._capture(params, parameters=True)
        incoming_buffers = self._capture(bufs, parameters=False)
        with torch.no_grad():
            for name, value in incoming.items():
                self.parameters[name].mul_(self.decay).add_(value, alpha=1 - self.decay)
        self.buffers = incoming_buffers
        self.last_optimizer_step = step
        self.num_updates += 1
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "decay": self.decay,
            "buffer_policy": self.buffer_policy,
            "initial_optimizer_step": self.initial_optimizer_step,
            "last_optimizer_step": self.last_optimizer_step,
            "num_updates": self.num_updates,
            "shadow_dtype": "torch.float32",
            "shadow_device": str(self.shadow_device),
            "parameter_count": len(self.parameters),
            "local_parameter_elements": sum(v.numel() for v in self.parameters.values()),
            "layout_fingerprint": hashlib.sha256(
                json.dumps(
                    {"parameters": self.parameter_layouts, "buffers": self.buffer_layouts},
                    sort_keys=True, separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }

    def state_dict(self) -> dict[str, Any]:
        self._idle()
        if not self.initialized:
            raise RuntimeError("Cannot save an uninitialized EMA")
        # Serialize only detached ordinary CPU tensors, never DTensors or optimizers.
        return {
            **self.metadata(), "parameter_layouts": copy.deepcopy(self.parameter_layouts),
            "buffer_layouts": copy.deepcopy(self.buffer_layouts),
            "parameters": {key: value.detach().cpu().clone() for key, value in self.parameters.items()},
            "buffers": {key: value.detach().cpu().clone() for key, value in self.buffers.items()},
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, parameters: TensorProvider, buffers: TensorProvider | None = None
    ):
        self._idle()
        if state.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported EMA schema version")
        if state.get("decay") != self.decay or state.get("buffer_policy") != self.buffer_policy:
            raise ValueError("Saved EMA decay/buffer policy differs from the configured tracker")
        initial = self._clock(state["initial_optimizer_step"])
        last = self._clock(state["last_optimizer_step"])
        updates = self._clock(state["num_updates"])
        if last - initial != updates:
            raise ValueError("Inconsistent EMA optimizer clock")
        params = _tensors(parameters)
        bufs = _tensors(buffers) if self.buffer_policy == "copy" else {}
        layouts = {name: tensor_layout(value) for name, value in params.items()}
        buffer_layouts = {name: tensor_layout(value) for name, value in bufs.items()}
        if layouts != state["parameter_layouts"] or buffer_layouts != state["buffer_layouts"]:
            raise ValueError("Saved EMA belongs to different model keys/shapes or worker shard layouts")
        saved_params = _tensors(state["parameters"])
        saved_buffers = _tensors(state["buffers"])
        if saved_params.keys() != params.keys() or saved_buffers.keys() != bufs.keys():
            raise ValueError("Saved EMA tensor keys are incomplete")
        for name, value in saved_params.items():
            if list(value.shape) != layouts[name]["local_shape"] or value.dtype != torch.float32:
                raise ValueError(f"Invalid saved FP32 parameter shadow: {name}")
        for name, value in saved_buffers.items():
            layout = buffer_layouts[name]
            if list(value.shape) != layout["local_shape"] or str(value.dtype) != layout["dtype"]:
                raise ValueError(f"Invalid saved buffer: {name}")
        captured = self._capture(saved_params, parameters=True)
        captured_buffers = self._capture(saved_buffers, parameters=False)
        self.parameters, self.buffers = captured, captured_buffers
        self.parameter_layouts, self.buffer_layouts = layouts, buffer_layouts
        self.initial_optimizer_step, self.last_optimizer_step, self.num_updates = initial, last, updates
        return self.metadata()

    @contextlib.contextmanager
    def swap_into(self, parameters: TensorProvider, buffers: TensorProvider | None = None):
        """Temporarily copy EMA values into live parameters and always restore them.

        A callable provider is resolved again at exit so offload/export operations
        that replace tensor storage do not leave restoration writing stale views.
        The context must cover the entire lifetime of any lazy export consumer.
        """
        with self._guard:
            self._idle()
            params = _tensors(parameters)
            bufs = _tensors(buffers) if self.buffer_policy == "copy" else {}
            self._validate(params, bufs)
            self._swapped = True
        backups, buffer_backups = {}, {}
        try:
            with torch.no_grad():
                backups = {name: local_tensor(value).cpu().clone() for name, value in params.items()}
                buffer_backups = {name: local_tensor(value).cpu().clone() for name, value in bufs.items()}
                for name, value in params.items():
                    local_tensor(value).copy_(self.parameters[name])
                for name, value in bufs.items():
                    local_tensor(value).copy_(self.buffers[name])
            yield
        finally:
            try:
                restored_params = _tensors(parameters)
                restored_buffers = _tensors(buffers) if self.buffer_policy == "copy" else {}
                self._validate(restored_params, restored_buffers)
                with torch.no_grad():
                    for name, value in backups.items():
                        local_tensor(restored_params[name]).copy_(value)
                    for name, value in buffer_backups.items():
                        local_tensor(restored_buffers[name]).copy_(value)
            finally:
                self._swapped = False

    async def consume_export(
        self,
        parameters: TensorProvider,
        export_factory: Callable,
        consumer: Callable,
        buffers: TensorProvider | None = None,
    ):
        """Await a complete lazy weight transfer while EMA values remain installed.

        A transport that returns before consuming its iterator is rejected. It
        must not retain live aliases after its awaited send operation completes.
        """
        with self.swap_into(parameters, buffers):
            source = iter(export_factory())
            exhausted = False

            def tracked():
                nonlocal exhausted
                while True:
                    try:
                        value = next(source)
                    except StopIteration:
                        exhausted = True
                        return
                    yield value

            stream = tracked()
            try:
                result = consumer(stream)
                result = await result if inspect.isawaitable(result) else result
                if not exhausted:
                    raise RuntimeError("EMA transport returned before consuming the complete lazy export")
                return result
            finally:
                stream.close()
                if hasattr(source, "close"):
                    source.close()
