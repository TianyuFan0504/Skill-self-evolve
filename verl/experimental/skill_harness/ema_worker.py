# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in official-worker subclasses providing local-shard EMA state and export.

No worker, process group, model, or GPU is started by importing this module.
The trainer must initialize EMA after loading the online checkpoint. The normal
solver weight-sync path remains untouched. Curator transport is separate and
must consume its complete weight stream before its awaited send completes.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import torch

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from .ema import ParameterEMA, require_single_optimizer_step


class EMATrainingWorker(TrainingWorker):
    """Count real optimizer calls and update detached EMA after one completed step."""

    def _sources(self):
        if not isinstance(getattr(self.engine, "module", None), torch.nn.Module):
            raise TypeError("EMA foundation requires an engine exposing a torch.nn.Module as module")
        return self.engine.module.named_parameters(), self.engine.module.named_buffers()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """Rebuilding the online model invalidates its optimizer observer and EMA."""
        old = getattr(self, "parameter_ema", None)
        if old is not None:
            old._idle()
        hook = getattr(self, "_ema_step_hook", None)
        if hook is not None:
            hook.remove()
        self._ema_step_hook = None
        self.parameter_ema = None
        return super().reset()

    def _ready(self):
        ema = getattr(self, "parameter_ema", None)
        if ema is None or not ema.initialized:
            raise RuntimeError("Call initialize_ema after loading the online checkpoint and before training")
        ema._idle()
        if ema.last_optimizer_step != self._observed_optimizer_step:
            raise RuntimeError("An observed optimizer step has not been incorporated into EMA")
        return ema

    def _install_step_observer(self, optimizer_step: int):
        old = getattr(self, "_ema_step_hook", None)
        if old is not None:
            old.remove()
        optimizer = self.engine.optimizer
        if optimizer is None or not hasattr(optimizer, "register_step_post_hook"):
            raise TypeError("EMA requires a PyTorch optimizer with register_step_post_hook support")
        self._observed_optimizer_step = optimizer_step

        def observed_step(_optimizer, _args, _kwargs):
            self._observed_optimizer_step += 1

        self._ema_step_hook = optimizer.register_step_post_hook(observed_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize_ema(
        self,
        decay: float = 0.95,
        *,
        optimizer_step: int = 0,
        shadow_device: str = "cpu",
        buffer_policy: str = "copy",
        reset: bool = False,
    ):
        old = getattr(self, "parameter_ema", None)
        if old is not None:
            old._idle()
            if not reset:
                raise RuntimeError("EMA already exists; reset=True is required for explicit reinitialization")
        tracker = ParameterEMA(decay, shadow_device=shadow_device, buffer_policy=buffer_policy)
        parameters, buffers = self._sources()
        tracker.initialize(parameters, buffers, optimizer_step=optimizer_step)
        self._install_step_observer(optimizer_step)
        self.parameter_ema = tracker
        return tracker.metadata()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_ema(self, expected_optimizer_step: int):
        """An idempotent synchronization barrier; callers cannot invent optimizer steps."""
        ema = getattr(self, "parameter_ema", None)
        if ema is None:
            raise RuntimeError("EMA is not initialized")
        if expected_optimizer_step != self._observed_optimizer_step:
            raise ValueError("Requested EMA clock does not match the observed optimizer clock")
        if ema.last_optimizer_step == expected_optimizer_step:
            return ema.metadata()
        parameters, buffers = self._sources()
        return ema.update(parameters, buffers, optimizer_step=expected_optimizer_step)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data):
        self._ready()
        require_single_optimizer_step(
            data.shape[0], self.engine.get_data_parallel_size(),
            mini_batch_size=tu.get(data, "mini_batch_size"),
            num_mini_batch=tu.get(data, "num_mini_batch"), epochs=tu.get(data, "epochs", 1),
        )
        before = self._observed_optimizer_step
        output = super().train_mini_batch(data)
        if self._observed_optimizer_step != before + 1:
            raise RuntimeError("Expected exactly one actual optimizer.step call during this update RPC")
        self._ready()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_batch(self, data):
        self._ready()
        before = self._observed_optimizer_step
        output = super().train_batch(data)
        if self._observed_optimizer_step != before + 1:
            raise RuntimeError("Optimizer skipped or performed multiple updates; EMA was not advanced")
        self.update_ema(self._observed_optimizer_step)
        return output

    def ema_state(self):
        ema = self._ready()
        return {
            "worker_rank": self.rank, "world_size": torch.distributed.get_world_size(),
            "engine_class": f"{type(self.engine).__module__}.{type(self.engine).__qualname__}",
            "ema": ema.state_dict(),
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_ema(self, directory: str):
        payload = self.ema_state()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"ema_rank_{self.rank:05d}.pt"
        temporary = root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as output:
                torch.save(payload, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"path": str(destination), **self.parameter_ema.metadata()}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_ema(self, directory: str, *, expected_decay: float | None = None, shadow_device: str = "cpu"):
        old = getattr(self, "parameter_ema", None)
        if old is not None:
            old._idle()
        path = Path(directory) / f"ema_rank_{self.rank:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        engine_class = f"{type(self.engine).__module__}.{type(self.engine).__qualname__}"
        if payload["worker_rank"] != self.rank or payload["world_size"] != torch.distributed.get_world_size():
            raise ValueError("EMA checkpoint belongs to another worker rank or world size")
        if payload["engine_class"] != engine_class:
            raise ValueError("EMA checkpoint belongs to a different model engine")
        state = payload["ema"]
        if expected_decay is not None and state["decay"] != expected_decay:
            raise ValueError("EMA decay does not match the requested resume configuration")
        tracker = ParameterEMA(state["decay"], shadow_device=shadow_device, buffer_policy=state["buffer_policy"])
        parameters, buffers = self._sources()
        tracker.load_state_dict(state, parameters=parameters, buffers=buffers)
        self._install_step_observer(tracker.last_optimizer_step)
        self.parameter_ema = tracker
        return tracker.metadata()


class EMAActorRolloutRefWorker(ActorRolloutRefWorker):
    """An opt-in actor worker; its default online solver sync is not changed."""

    actor_worker_cls = EMATrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize_ema(self, *args, **kwargs):
        if not self._is_actor or self.actor is None:
            raise RuntimeError("EMA initialization requires an initialized online actor")
        if getattr(self, "_ema_checkpoint_engine", None) is not None:
            raise RuntimeError("Reconfiguring EMA while a transport exists requires a new worker")
        return self.actor.initialize_ema(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_ema(self, expected_optimizer_step: int):
        return self.actor.update_ema(expected_optimizer_step)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_ema_metadata(self):
        return self.actor._ready().metadata()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_ema(self, directory: str):
        return self.actor.save_ema(directory)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_ema(self, directory: str, **kwargs):
        return self.actor.load_ema(directory, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_ema_transport(self, backend: str, *, bucket_size_mb: int = 512, engine_kwargs: dict | None = None):
        """Create a separate transport for a dedicated curator replica topology.

        The ordinary solver checkpoint engine is deliberately not reused. Naive
        transport defers generator consumption and delta_sharded owns an engine
        state machine, so neither is supported by this conservative foundation.
        """
        self.actor._ready()
        if backend in ("naive", "delta_sharded"):
            raise ValueError("EMA export requires a transport that consumes the weight iterator during send")
        if type(bucket_size_mb) is not int or bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive")
        if getattr(self, "_ema_checkpoint_engine", None) is not None:
            raise RuntimeError("EMA transport is already configured")
        self._ema_checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=torch.distributed.get_rank() == 0, bucket_size=bucket_size_mb << 20,
            **(engine_kwargs or {}),
        )
        return {"backend": backend, "bucket_size_mb": bucket_size_mb, "source": "ema"}

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_ema_transport(self, method: str, *args, **kwargs):
        if method.startswith("_"):
            raise ValueError("Private transport methods cannot be dispatched")
        transport = getattr(self, "_ema_checkpoint_engine", None)
        if transport is None:
            raise RuntimeError("Configure the dedicated EMA transport first")
        return getattr(transport, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def send_ema_weights(self, global_steps: int | None = None):
        """Keep temporary EMA values installed until the entire asynchronous send ends."""
        ema = self.actor._ready()
        transport = getattr(self, "_ema_checkpoint_engine", None)
        if transport is None:
            raise RuntimeError("Configure a dedicated curator transport before exporting EMA weights")
        if self.config.model.get("lora", {}).get("rank", 0) or self.config.model.get("lora_rank", 0):
            raise NotImplementedError("EMA foundation supports the non-LoRA dense actor path only")

        def export():
            generator, peft = self.actor.engine.get_per_tensor_param()
            if peft is not None:
                raise NotImplementedError("EMA export cannot silently mix adapter and full-model states")
            return generator

        async def send(generator):
            return await transport.send_weights(generator, global_steps=global_steps)

        result = await ema.consume_export(
            lambda: self.actor.engine.module.named_parameters(), export, send,
            lambda: self.actor.engine.module.named_buffers(),
        )
        return {"transport_metrics": result or {}, "ema": ema.metadata()}
