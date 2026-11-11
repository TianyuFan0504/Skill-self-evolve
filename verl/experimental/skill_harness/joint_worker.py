# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in joint solver/curate worker using public verl Engine extension points.

Select ``JointEMAActorRolloutRefWorker`` instead of the ordinary actor worker.
Its actor subclass replaces init_model's stock ``partial(ppo_loss, config=...)``
through the public ``set_loss_fn`` method. No init_model, TQ bridge, rollout,
minibatch iterator or Engine implementation is copied or monkey-patched.

The exact update boundary is TrainingWorker.train_mini_batch -> self.train_batch.
Here role counts are prepared AFTER minibatch selection and BEFORE the Engine
splits microbatches. The inherited EMA single-step guard allows one epoch and
one optimizer minibatch per update RPC. The inherited optimizer post-hook counts
actual optimizer.step calls, and EMA advances once after that call succeeds.

Usage after normal ``init_model``: optionally ``configure_joint_loss(...)``, then
``initialize_ema(...)``, then the ordinary ``update_actor(mixed_batch)``. Set
global mini_batch_size to the actual mixed distributed batch size (or explicitly
num_mini_batch=1), epochs=1. The trainer owns replay leases, immutable archive
validation, frozen anchor preparation, checkpoint coordination and applied IDs.
This worker does not claim optimizer/replay crash atomicity.

Importing this file starts no workers, process groups, models or GPU jobs.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss

from .ema_worker import EMAActorRolloutRefWorker, EMATrainingWorker
from .losses import CurateLossConfig, prepare_role_batch_metadata, skill_harness_loss


class JointEMATrainingWorker(EMATrainingWorker):
    """One online optimizer, one joint backward, followed by parameter-only EMA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._joint_control_lock = threading.RLock()
        self._joint_update_active = False

    @contextmanager
    def _joint_control(self, *, training=False):
        if not self._joint_control_lock.acquire(blocking=False):
            raise RuntimeError("Another joint update or loss reconfiguration is in progress")
        try:
            if self._joint_update_active:
                raise RuntimeError("Joint training cannot overlap another update or loss reconfiguration")
            ema = getattr(self, "parameter_ema", None)
            if ema is not None:
                ema._idle()
            if training:
                self._joint_update_active = True
            try:
                yield
            finally:
                if training:
                    self._joint_update_active = False
        finally:
            self._joint_control_lock.release()

    @staticmethod
    def _loss_settings(loss_fn):
        if not isinstance(loss_fn, partial) or loss_fn.args:
            raise TypeError("Joint training requires a keyword-configured public PPO/skill loss partial")
        if loss_fn.func not in (ppo_loss, skill_harness_loss):
            raise ValueError("Joint worker cannot silently replace an unrelated or distillation loss")
        allowed = {"config"} if loss_fn.func is ppo_loss else {"config", "curate_config"}
        if set(loss_fn.keywords) - allowed:
            raise ValueError("Unexpected pre-bound loss arguments could change the joint loss contract")
        config = loss_fn.keywords.get("config")
        if not isinstance(config, ActorConfig):
            raise TypeError("The joint loss requires the actual initialized ActorConfig dataclass")
        if config.strategy not in ("fsdp", "fsdp2"):
            raise ValueError("Joint role normalization is currently audited only for FSDP/FSDP2")
        if config.policy_loss.get("loss_mode", "vanilla") != "vanilla":
            raise ValueError("Only the audited vanilla solver GRPO/PPO loss is supported")
        curate = loss_fn.keywords.get("curate_config", CurateLossConfig())
        if not isinstance(curate, CurateLossConfig):
            raise TypeError("curate_config must be an immutable CurateLossConfig")
        return config, curate

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        """Install through stock init_model's existing public call; no silent fallback."""
        with self._joint_control():
            config, curate = self._loss_settings(loss_fn)
            super().set_loss_fn(partial(skill_harness_loss, config=config, curate_config=curate))
            return self.joint_training_metadata()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_joint_loss(self, **settings):
        """Set CurateLossConfig fields after init_model, before the next update.

        Settings must be recorded/restored by the trainer alongside its run and
        checkpoint configuration. Reconfiguration never mutates EMA weights.
        """
        with self._joint_control():
            loss_fn = getattr(self, "loss_fn", None)
            if not isinstance(loss_fn, partial) or loss_fn.func is not skill_harness_loss:
                raise RuntimeError("Initialize the actor and its joint loss before configuring curate loss")
            config, current = self._loss_settings(loss_fn)
            curate = CurateLossConfig(**{**asdict(current), **settings})
            return self.set_loss_fn(partial(skill_harness_loss, config=config, curate_config=curate))

    def joint_training_metadata(self):
        loss_fn = getattr(self, "loss_fn", None)
        if not isinstance(loss_fn, partial) or loss_fn.func is not skill_harness_loss:
            raise RuntimeError("Joint loss is not installed")
        config, curate = self._loss_settings(loss_fn)
        return {
            "schema_version": 1,
            "loss": "skill_harness_loss",
            "solver_role_id": 0,
            "curator_role_id": 1,
            "solver_loss_mode": "vanilla",
            "strategy": config.strategy,
            "curate": asdict(curate),
            "single_optimizer_step_per_rpc": True,
            "complete_tito_trajectories_per_candidate": 1,
            "ema_initialized": getattr(self, "parameter_ema", None) is not None,
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_batch(self, data):
        """Exact stock optimizer-minibatch boundary; preserve every native field."""
        with self._joint_control(training=True):
            self._ready()
            loss_fn = getattr(self, "loss_fn", None)
            if not isinstance(loss_fn, partial) or loss_fn.func is not skill_harness_loss:
                raise RuntimeError("Refusing a joint update without the role-aware loss")
            self._loss_settings(loss_fn)
            prepare_role_batch_metadata(data, self.engine.get_data_parallel_group())
            return super().train_batch(data)


class JointEMAActorRolloutRefWorker(EMAActorRolloutRefWorker):
    """Public actor worker preserving native update_actor, inference and TQ paths."""

    actor_worker_cls = JointEMATrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def configure_joint_loss(self, **settings):
        if not self._is_actor or self.actor is None:
            raise RuntimeError("Joint loss configuration requires an initialized actor")
        return self.actor.configure_joint_loss(**settings)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_joint_training_metadata(self):
        if not self._is_actor or self.actor is None:
            raise RuntimeError("Joint training metadata requires an initialized actor")
        return self.actor.joint_training_metadata()
