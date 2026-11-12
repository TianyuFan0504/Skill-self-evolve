# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Synchronous native verl controller integration for joint skill training.

Solver GRPO is computed by the unchanged parent. This extension materializes
that exact native batch, prepares delayed replay, issues ONE actor update, then
awaits durable credit acknowledgement and EMA transport before the parent's
normal source-batch metrics and TQ cleanup. No token IDs are re-encoded here.

All learner/controller/checkpoint operations share a driver RLock and one
persistent event loop. Resume validates the COMPLETE joint snapshot before any
live load and never adopts the native trainer's early latest pointer. Failed
updates/restores abort startup/training; no lease is cleared on a timeout.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path

import ray
import torch
from omegaconf import open_dict

import verl
from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.metric import Metric
from verl.utils.transferqueue_utils import tq

from .alfworld.accounts import assert_no_isolation_incidents
from .controller import SkillJointController, preflight_controller_identity
from .framework import (
    TEXT_ONLY_MEDIA_FIELDS,
    assert_no_nontext_incidents,
    guard_text_only_tq_batch,
    require_text_only_media,
)
from .joint_checkpoint import (
    CompletedOnlineCheckpoint,
    JointCheckpointCoordinator,
    JointCheckpointError,
    JointCheckpointSpec,
)
from .joint_worker import JointEMAActorRolloutRefWorker
from .policy_routes import fingerprint_model
from .replay_batch import required_solver_fields
from .store import SkillStore
from .trainer import SkillSyncTrainer
from .uniagent_compat import actual_adapter_sha256

CALLBACK_FQN = "verl.experimental.skill_harness.joint_trainer.SkillJointCheckpointCallback"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ordered_layout_fingerprint(rows, world_size):
    """Bind physical checkpoint ranks, not merely the actor DP count."""
    if not isinstance(rows, list | tuple) or len(rows) != world_size:
        raise RuntimeError("Online layout metadata must cover every physical actor rank")
    ordered = sorted(rows, key=lambda row: row["worker_rank"])
    if [row["worker_rank"] for row in ordered] != list(range(world_size)):
        raise RuntimeError("Online layout ranks are missing or duplicated")
    if any(row["world_size"] != world_size or not row.get("layout_fingerprint") for row in ordered):
        raise RuntimeError("Online layout metadata world/layout mismatch")
    return _digest([{key: row[key] for key in ("worker_rank", "world_size", "layout_fingerprint")}
                    for row in ordered])


def _source_fingerprint():
    root = Path(verl.__file__).resolve().parent.parent
    paths = list((root / "verl/experimental/skill_harness").rglob("*.py"))
    paths.extend(root / relative for relative in (
        "verl/workers/engine_workers.py", "verl/workers/utils/losses.py", "verl/workers/utils/padding.py",
        "verl/workers/engine/base.py", "verl/workers/engine/fsdp/transformer_impl.py",
        "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
        "verl/utils/checkpoint/fsdp_checkpoint_manager.py", "verl/trainer/ppo/v1/trainer_base.py",
        "verl/trainer/ppo/v1/trainer_sync.py", "verl/trainer/ppo/core_algos.py",
        "examples/skill_self_evolve/scripts/codex_entrypoint.sh",
        "examples/skill_self_evolve/scripts/launch.py",
        "examples/skill_self_evolve/prompts/solver_system.md",
        "examples/skill_self_evolve/prompts/solver_task.txt",
    ))
    payload = [(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
               for path in sorted(set(paths))]
    payload.append(("external/uni_agent/gateway/adapters/openai_response.py", actual_adapter_sha256()))
    return _digest(payload)


async def _resolve(value):
    if inspect.isawaitable(value):
        return await _resolve(await value)
    if type(value).__name__ == "DataProtoFuture" and callable(getattr(value, "get", None)):
        return await _resolve(await asyncio.to_thread(value.get))
    if isinstance(value, list | tuple):
        return list(await asyncio.gather(*(_resolve(item) for item in value)))
    return value


def _metric_number(value, *, name=""):
    if isinstance(value, Metric):
        number = float(value.aggregate())
    elif isinstance(value, list | tuple):
        if not value:
            raise ValueError("Empty actor metric")
        if all(isinstance(item, Metric) for item in value):
            number = float(Metric.aggregate_dp(list(value)))
        else:
            numbers = [_metric_number(item, name=name) for item in value]
            number = max(numbers) if "max" in name else min(numbers) if "min" in name else sum(numbers) / len(numbers)
    else:
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError("Nonfinite joint actor metric")
    return number


def normalize_actor_metrics(output):
    """Keep loss namespaces once; preserve the stock perf/mfu/actor mapping."""
    raw = tu.get(output, "metrics") if not isinstance(output, dict) else output["metrics"]
    result = {}
    for key, value in raw.items():
        if key == "mfu":
            name = "perf/mfu/actor"
        else:
            name = key if key.startswith(("actor/", "skill/", "perf/")) else "actor/" + key
        if name in result:
            raise ValueError("Actor metric namespace collision")
        result[name] = _metric_number(value, name=name)
    return result


class SkillJointSyncTrainer(SkillSyncTrainer):
    supported_modes = ("skill_joint",)

    def __init__(self, config):
        self._joint_lock = threading.RLock()
        self._joint_loop = asyncio.new_event_loop()
        self._joint_controller = None
        self._joint_identity = None
        self._joint_compatibility = None
        self._joint_restored = None
        self._joint_deferred_controller = None
        self._joint_coordinator = JointCheckpointCoordinator()
        self._validate_joint_config(config)
        callback = config.trainer.get("checkpoint_callback_class")
        if callback not in (None, CALLBACK_FQN):
            raise ValueError("Joint training requires the official joint checkpoint callback")
        with open_dict(config.trainer):
            config.trainer.checkpoint_callback_class = CALLBACK_FQN
        super().__init__(config)
        if self.use_critic:
            raise ValueError("The initial joint harness requires solver GRPO without a critic optimizer")

    @staticmethod
    def _validate_joint_config(config):
        if config.skill_harness.mode != "skill_joint" or config.trainer.v1.trainer_mode != "sync":
            raise ValueError("Joint skill training requires the synchronous skill_joint mode")
        ema = config.skill_harness.ema
        if not ema.enabled or ema.get("update_every", 1) != 1:
            raise ValueError("Joint training requires EMA after every observed optimizer update")
        actor = config.actor_rollout_ref.actor
        if actor.ppo_epochs != 1 or actor.get("checkpoint", {}).get("async_save", False):
            raise ValueError("Joint training requires one PPO epoch and synchronous actor checkpoints")
        if (config.trainer.get("max_actor_ckpt_to_keep") is not None
                or config.trainer.get("remove_previous_ckpt_in_save", False)):
            raise ValueError("Disable native actor pruning until COMPLETE joint checkpoint publication")
        if config.trainer.get("del_local_ckpt_after_load", False):
            raise ValueError("Joint checkpoint files must remain available for complete verification")

    def _await(self, value):
        if self._joint_loop.is_closed() or self._joint_loop.is_running():
            raise RuntimeError("Joint driver operations require its single stopped persistent event loop")
        return self._joint_loop.run_until_complete(_resolve(value))

    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        selected = [role for role in (Role.Actor, Role.ActorRollout, Role.ActorRolloutRef)
                    if role in self.role_worker_mapping]
        if len(selected) != 1:
            raise RuntimeError("The joint learner requires exactly one actual actor role")
        self.role_worker_mapping[selected[0]] = ray.remote(JointEMAActorRolloutRefWorker)

    def _get_n_gpus_for_throughput(self):
        """Native numerator is solver tokens; denominator includes ALL reserved GPUs."""
        solver = super()._get_n_gpus_for_throughput()
        controller = self._joint_controller
        if controller is None:
            return solver
        return solver + controller.settings.curator_gpus + controller.settings.credit_gpus

    def _identity(self):
        if self._joint_identity is None:
            self._joint_identity = preflight_controller_identity(self.config)
        return self._joint_identity

    def _compatibility(self):
        if self._joint_compatibility is not None:
            bound = self._joint_compatibility.get("versions", {}).get("uni_agent_adapter")
            if bound is not None and bound != actual_adapter_sha256():
                raise RuntimeError("Uni-Agent adapter changed after checkpoint compatibility was initialized")
        if self._joint_compatibility is None:
            identity = self._identity()
            ema = self.config.skill_harness.ema
            ranks = self._await(self.actor_rollout_wg.get_online_layout_metadata(
                buffer_policy=ema.get("buffer_policy", "copy"),
            ))
            layout = ordered_layout_fingerprint(ranks, self.actor_rollout_wg.world_size)
            self._joint_compatibility = {
                "world_size": self.actor_rollout_wg.world_size, "layout_fingerprint": layout,
                "versions": {"code": _source_fingerprint(), "verl": verl.__version__,
                             "uni_agent_adapter": actual_adapter_sha256(),
                             "model": fingerprint_model(self.config.actor_rollout_ref.model.path)["fingerprint"],
                             "tokenizer": identity.tokenizer_version, "template": identity.template_version},
                "evaluator_fingerprint": identity.frozen_version, "protocol_fingerprint": identity.protocol_fingerprint,
            }
        return self._joint_compatibility

    def _ema_clock(self):
        rows = self._await(self.actor_rollout_wg.get_ema_metadata())
        if not isinstance(rows, list) or len(rows) != self.actor_rollout_wg.world_size:
            raise RuntimeError("Actual EMA/optimizer clocks must cover all actor ranks")
        clocks = [row["last_optimizer_step"] for row in rows]
        if not clocks or type(clocks[0]) is not int or clocks[0] < 0 or any(value != clocks[0] for value in clocks):
            raise RuntimeError("Actual EMA/optimizer clocks differ across actor ranks")
        actual_layout = ordered_layout_fingerprint([
            {"worker_rank": rank, "world_size": len(rows), "layout_fingerprint": row["layout_fingerprint"]}
            for rank, row in enumerate(rows)
        ], len(rows))
        if actual_layout != self._compatibility()["layout_fingerprint"]:
            raise RuntimeError("Actual EMA layout differs from the initialized online shards")
        return clocks[0]

    def on_init_end(self):
        with self._joint_lock:
            self._joint_store = self._joint_restored.skill_store if self._joint_restored else SkillStore(
                self.config.skill_harness.store_root
            )
            self._await(self.actor_rollout_wg.configure_joint_loss(**dict(self.config.skill_harness.curate_loss)))
            if self._joint_restored is None:
                if self.global_steps != 0:
                    raise RuntimeError("A trained online checkpoint cannot initialize a fabricated fresh EMA clock")
                ema = self.config.skill_harness.ema
                self._await(self.actor_rollout_wg.initialize_ema(
                    decay=ema.decay, optimizer_step=0, shadow_device="cpu",
                    buffer_policy=ema.get("buffer_policy", "copy"),
                ))
            expected = (0 if self._joint_restored is None
                        else self._joint_restored.checkpoint.payload["spec"]["optimizer_step"])
            if self._ema_clock() != expected:
                raise RuntimeError("Restored/fresh EMA clock does not match its online checkpoint")
            super().on_init_end()

    def on_train_begin(self):
        with self._joint_lock:
            if self._joint_controller is not None:
                raise RuntimeError("The joint controller cannot be initialized twice")
            identity = self._identity()
            self._joint_controller = self._await(SkillJointController.create(
                self, native_runtime=identity.runtime, restored_state=self._joint_restored,
            ))
            if (self._joint_controller.protocol_fingerprint != identity.protocol_fingerprint
                    or self._joint_controller.frozen_version != identity.frozen_version):
                raise RuntimeError("The live controller differs from the checkpoint compatibility preflight")
            # Parent's official initial tests run only after dedicated routes and
            # immutable evaluator/curator identities have been established.
            super().on_train_begin()

    def _materialize_solver(self, batch):
        guard_text_only_tq_batch(tq, batch, artifact_root=self.config.skill_harness.artifact_root)
        fields = list(required_solver_fields()) + ["raw_prompt"]
        actor = self.config.actor_rollout_ref.actor
        if actor.use_kl_loss or self.config.algorithm.get("use_kl_in_reward", False):
            fields.append("ref_log_prob")
        correction = self.config.algorithm.get("rollout_correction")
        if correction and not correction.get("bypass_mode", False) and correction.get("rollout_is") is not None:
            fields.append("rollout_is_weights")
        if self.config.actor_rollout_ref.rollout.calculate_log_probs:
            fields.append("rollout_log_probs")
        source = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                                select_fields=list(dict.fromkeys(fields)))
        # Inspect before omitting only proved text-only optional carriers. A nonempty
        # payload raises; image-conditioned likelihood must never lose pixels.
        for key in TEXT_ONLY_MEDIA_FIELDS:
            if key in source:
                require_text_only_media({key: source[key]}, source="materialized solver media field",
                                        artifact_root=self.config.skill_harness.artifact_root)
                source.pop(key)
        for key, value in (batch.extra_info or {}).items():
            if key in TEXT_ONLY_MEDIA_FIELDS:
                continue  # Already proved empty by the authoritative media guard.
            if key not in source:
                tu.assign_non_tensor_data(source, key, value)
        tu.assign_non_tensor(source, calculate_entropy=actor.calculate_entropy or actor.entropy_coeff != 0,
                             distillation_only=False, distillation_use_topk=False,
                             seed=actor.data_loader_seed, dataloader_kwargs={"shuffle": actor.shuffle},
                             use_fused_kernels=self.config.actor_rollout_ref.model.get("use_fused_kernels", False))
        return source

    def _update_actor(self, batch, metrics):
        with self._joint_lock:
            assert_no_isolation_incidents(self.config.skill_harness.artifact_root)
            assert_no_nontext_incidents(self.config.skill_harness.artifact_root)
            if self._joint_controller is None:
                raise RuntimeError("Joint controller must be initialized before any optimizer update")
            source = self._materialize_solver(batch)
            prepared = self._await(self._joint_controller.prepare_update(source, self.global_steps))
            assert_no_isolation_incidents(self.config.skill_harness.artifact_root)
            assert_no_nontext_incidents(self.config.skill_harness.artifact_root)
            output = self._await(self.actor_rollout_wg.update_actor(prepared.data))
            actual = self._ema_clock()
            if actual != prepared.expected_optimizer_step:
                raise RuntimeError("The mixed update did not produce exactly its expected observed optimizer step")
            completed = self._await(self._joint_controller.after_update_and_sync(
                source, self.global_steps, prepared=prepared, optimizer_step=actual,
            ))
            metrics.update(normalize_actor_metrics(output))
            metrics["skill/optimizer_step"] = actual
            metrics["skill/replay_candidates"] = len(prepared.acknowledgments)
            metrics["skill/solver_gpu_count"] = self.actor_rollout_wg.world_size
            metrics["skill/total_reserved_gpu_count"] = (
                self.actor_rollout_wg.world_size + self._joint_controller.settings.curator_gpus
                + self._joint_controller.settings.credit_gpus
            )
            if isinstance(completed, Mapping):
                for key in (
                    "new_curate_windows", "curate_semantic_rejections", "curate_censored_jobs",
                    "curate_retained_actions", "curate_censored_cohorts",
                ):
                    metrics["skill/" + key] = completed.get(key, 0)
            return batch  # Parent metrics and cleanup retain ONLY the original solver KVBatchMeta.

    def _save_checkpoint(self):
        # The post-save callback alone is too late to fence a concurrent actor
        # update. Hold the same lock around the native model/data save as well.
        with self._joint_lock:
            return super()._save_checkpoint()

    def _load_checkpoint(self):
        self.global_steps = 0
        mode = self.config.trainer.resume_mode
        if mode == "disable":
            return
        expected = self._compatibility()
        if mode == "auto":
            root = Path(self.config.trainer.default_local_dir).resolve()
            if not (root / "latest_joint_checkpoint.json").exists():
                if (root / "latest_checkpointed_iteration.txt").exists() or list(root.glob("global_step_*")):
                    raise JointCheckpointError("Existing checkpoints lack a COMPLETE joint resume pointer")
                return
            checkpoint = self._joint_coordinator.latest(root, expected=expected)
        elif mode == "resume_path":
            checkpoint = self._joint_coordinator.validate(self.config.trainer.resume_from_path, expected=expected)
        else:
            raise ValueError("Unknown joint checkpoint resume mode")
        destination = Path(self.config.skill_harness.artifact_root) / "restored_state" / (
            f"step_{checkpoint.payload['spec']['global_step']}_" + uuid.uuid4().hex
        )

        def load_online(path, _spec):
            return self._await(self.actor_rollout_wg.load_checkpoint(local_path=str(path), del_local_after_load=False))

        def load_ema(path, _spec):
            return self._await(self.actor_rollout_wg.load_ema(
                str(path), expected_decay=self.config.skill_harness.ema.decay, shadow_device="cpu",
            ))

        def load_training(paths, _spec):
            if len(paths) != 1 or paths[0].name != "data.pt":
                raise JointCheckpointError("The synchronous joint trainer expects its complete dataloader state")
            self.train_dataloader.load_state_dict(torch.load(paths[0], weights_only=False, map_location="cpu"))

        def load_controller(payload, *, expected_optimizer_step):
            if payload["optimizer_step"] != expected_optimizer_step:
                raise JointCheckpointError("Deferred controller restore clock mismatch")
            self._joint_deferred_controller = payload

        restored = self._joint_coordinator.restore(
            checkpoint.checkpoint_dir, expected=expected, destination_root=destination,
            load_online=load_online, load_ema=load_ema, load_training_state=load_training,
            load_controller=load_controller, fence=lambda: self._joint_lock,
        )
        with open_dict(self.config.skill_harness):
            self.config.skill_harness.store_root = str(restored.skill_store.root)
            self.config.skill_harness.replay_root = str(restored.replay.root)
            self.config.skill_harness.ledger_root = str(restored.ledger.root)
        self._joint_restored = restored
        self.global_steps = restored.checkpoint.payload["spec"]["global_step"]

    def fit(self, agent_loop_manager):
        try:
            return super().fit(agent_loop_manager)
        finally:
            try:
                if self._joint_controller is not None:
                    for name in ("ema_gateway", "frozen_gateway"):
                        gateway = getattr(self._joint_controller, name, None)
                        if gateway is not None:
                            self._await(gateway.shutdown())
            finally:
                self._joint_loop.close()


class SkillJointCheckpointCallback(CheckpointCallback):
    """Official post-save hook; reuse the completed core save, never save it twice."""

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        if async_save:
            raise JointCheckpointError("The joint trainer forbids incomplete asynchronous native checkpoints")
        with trainer._joint_lock:
            controller = trainer._joint_controller
            if controller is None or controller._busy:
                raise JointCheckpointError("Checkpoint requires the initialized, quiescent joint controller")
            identity = trainer._identity()
            clock = trainer._ema_clock()
            spec = JointCheckpointSpec(global_step=global_step, optimizer_step=clock, **trainer._compatibility())

            def completed_online(path):
                if path.resolve() != (Path(checkpoint_dir) / "actor").resolve():
                    raise JointCheckpointError("Core checkpoint completion path differs from the joint manifest")
                return CompletedOnlineCheckpoint(global_step, clock, spec.world_size, spec.layout_fingerprint,
                                                  f"native-sync-post-save:{global_step}:{clock}")

            evaluation = {**identity.manifest(), "evaluator_fingerprint": identity.frozen_version,
                          "protocol_fingerprint": identity.protocol_fingerprint}
            trainer._last_joint_checkpoint = trainer._joint_coordinator.save(
                checkpoint_dir, spec=spec, save_online=completed_online,
                save_ema=lambda path: trainer._await(trainer.actor_rollout_wg.save_ema(str(path))),
                skill_store=controller.store, replay=controller.replay, ledger=controller.ledger,
                controller_state=controller.export_state, evaluation_state=evaluation,
                fence=lambda: trainer._joint_lock, training_state_paths=("data.pt",),
            )
