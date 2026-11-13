# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Native solver updates with independently progressing skill collection.

Solver GRPO is computed by the unchanged parent. This extension materializes
that exact native batch, prepares delayed replay, issues ONE actor update, then
acknowledges consumed credit and queues skill work before the parent's normal
source-batch metrics and TQ cleanup. Background model generations are never
awaited on this update path. No token IDs are re-encoded here.

Learner updates and checkpoints share a driver RLock. Controller writes are
serialized on one continuously running loop and gated during a checkpoint cut.
Resume validates the COMPLETE joint snapshot before any
live load and never adopts the native trainer's early latest pointer. Failed
updates/restores abort startup/training; no lease is cleared on a timeout.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import sys
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path

import ray
import torch
from omegaconf import open_dict

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import Metric
from verl.utils.transferqueue_utils import tq

from .controller import SkillJointController, preflight_controller_identity
from .framework import (
    TEXT_ONLY_MEDIA_FIELDS,
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
from .telemetry import append_metrics, episode_metrics

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


class SkillJointSyncTrainer(PPOTrainerSync):
    """One synchronous solver/curator optimizer with EMA and delayed replay."""

    def __init__(self, config):
        self._joint_lock = threading.RLock()
        self._joint_loop = asyncio.new_event_loop()
        self._joint_loop_ready = threading.Event()
        self._joint_thread = None
        self._joint_controller = None
        self._joint_identity = None
        self._joint_compatibility = None
        self._joint_restored = None
        self._joint_coordinator = JointCheckpointCoordinator()
        self._background_stopped = False
        self._final_evaluation_enabled = False
        self._final_checkpoint_seal = None
        self.artifact_root = Path(config.skill_harness.artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._validate_joint_config(config)
        callback = config.trainer.get("checkpoint_callback_class")
        if callback not in (None, CALLBACK_FQN):
            raise ValueError("Joint training requires the official joint checkpoint callback")
        with open_dict(config.trainer):
            config.trainer.checkpoint_callback_class = CALLBACK_FQN
        super().__init__(config)
        if self.use_critic:
            raise ValueError("Joint training requires solver GRPO without a critic optimizer")

    @staticmethod
    def _validate_joint_config(config):
        if config.trainer.v1.trainer_mode != "sync" or config.algorithm.adv_estimator != "grpo":
            raise ValueError("Joint training requires synchronous solver GRPO")
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
        if self._joint_loop.is_closed():
            raise RuntimeError("The joint event loop is closed")
        if hasattr(self, "_joint_loop_ready") and self._joint_thread is None:
            self._joint_thread = threading.Thread(
                target=self._run_joint_loop, name="skill-background-loop", daemon=True,
            )
            self._joint_thread.start()
            self._joint_loop_ready.wait()
        if self._joint_loop.is_running():
            if threading.current_thread() is getattr(self, "_joint_thread", None):
                raise RuntimeError("A background callback cannot synchronously wait on its own event loop")
            return asyncio.run_coroutine_threadsafe(_resolve(value), self._joint_loop).result()
        return self._joint_loop.run_until_complete(_resolve(value))

    def _run_joint_loop(self):
        asyncio.set_event_loop(self._joint_loop)
        self._joint_loop.call_soon(self._joint_loop_ready.set)
        self._joint_loop.run_forever()

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

    def on_sample_end(self):
        if self._joint_controller is not None:
            self._await(self._joint_controller.enter_learner())
        return super().on_sample_end()

    def on_step_end(self):
        result = super().on_step_end()
        if self._joint_controller is not None:
            self._await(self._joint_controller.leave_learner())
        return result

    def _identity(self):
        if self._joint_identity is None:
            self._joint_identity = preflight_controller_identity(self.config)
        return self._joint_identity

    def _compatibility(self):
        if self._joint_compatibility is None:
            identity = self._identity()
            ema = self.config.skill_harness.ema
            ranks = self._await(self.actor_rollout_wg.get_online_layout_metadata(
                buffer_policy=ema.get("buffer_policy", "copy"),
            ))
            layout = ordered_layout_fingerprint(ranks, self.actor_rollout_wg.world_size)
            self._joint_compatibility = {
                "world_size": self.actor_rollout_wg.world_size, "layout_fingerprint": layout,
                "versions": {"model": fingerprint_model(self.config.actor_rollout_ref.model.path)["fingerprint"],
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
            if self.global_steps > self.total_training_steps:
                return super().on_train_begin()
            identity = self._identity()
            self._joint_controller = self._await(SkillJointController.create(
                self, native_runtime=identity.runtime, restored_state=self._joint_restored,
            ))
            if (self._joint_controller.protocol_fingerprint != identity.protocol_fingerprint
                    or self._joint_controller.frozen_version != identity.frozen_version):
                raise RuntimeError("The live controller differs from the checkpoint compatibility preflight")
            super().on_train_begin()

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        fields = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["uid", "session_id", "rm_scores", "extra_fields"],
        )
        uids = tu.get(fields, "uid")
        sessions = tu.get(fields, "session_id")
        scores = fields["rm_scores"].to_padded_tensor(0.0).sum(dim=-1).tolist()
        extras = tu.get(fields, "extra_fields")
        rows = []
        for i, (uid, session, score, extra) in enumerate(zip(uids, sessions, scores, extras, strict=True)):
            info = extra.get("reward_extra_info", {})
            if "acc" not in info:
                raise RuntimeError("Native ALFWorld runner did not provide authoritative TaskResult.accuracy")
            rows.append(
                {
                    "uid": uid,
                    "session_id": session,
                    "reward": score,
                    "accuracy": info["acc"],
                    "error": info.get("error"),
                    "padding": batch.tags[i].get("is_padding", False),
                }
            )
        metrics.update(episode_metrics(rows))
        for key in ("actor/grad_norm", "actor/pg_loss"):
            value = metrics.get(key)
            if value is not None and not math.isfinite(float(value)):
                raise FloatingPointError(f"Nonfinite optimizer metric: {key}={value}")
        append_metrics(self.artifact_root, step=global_steps, phase="train", metrics=metrics)

    def get_llm_client(self):
        terminal = self.config.skill_harness.get("terminal", {})
        if not terminal.get("enabled", False):
            return super().get_llm_client()
        from .alfworld.backend_terminal import TerminalAwareLLMServerClient

        return self.llm_server_manager.get_client(
            client_cls=TerminalAwareLLMServerClient,
            binding_root=terminal.get("binding_root", str(self.artifact_root / "terminal_bindings")),
        )

    def _validate(self):
        metrics = super()._validate()
        append_metrics(
            self.artifact_root,
            step=self.global_steps,
            phase=getattr(self, "_evaluation_phase", "dev"),
            stage=getattr(self, "_evaluation_stage", None),
            metrics=metrics,
        )
        return metrics

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
            if self._joint_controller is None:
                raise RuntimeError("Joint controller must be initialized before any optimizer update")
            source = self._materialize_solver(batch)
            prepared = self._await(self._joint_controller.prepare_update(source, self.global_steps))
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
                for key, value in completed.items():
                    if key.startswith(("source_", "probe_", "curator_weight_")):
                        metrics["skill/" + key] = value
            return batch  # Parent metrics and cleanup retain ONLY the original solver KVBatchMeta.

    def _stop_background(self):
        if self._joint_controller is not None and not getattr(self, "_background_stopped", False):
            self._await(self._joint_controller.stop_background())
            self._background_stopped = True

    def _checkpoint_seal(self, checkpoint):
        spec = checkpoint.payload["spec"]
        return {
            "optimizer_step": spec["optimizer_step"],
            "checkpoint_step": spec["global_step"],
            "checkpoint_dir": checkpoint.checkpoint_dir,
            "manifest_id": checkpoint.manifest_id,
            "snapshot_id": checkpoint.payload["snapshots"]["skill_heads"]["main"],
        }

    def _save_checkpoint(self):
        # The native model/optimizer save and joint state share the learner fence.
        with self._joint_lock:
            final = (getattr(self, "_final_evaluation_enabled", False)
                     and self.global_steps >= self.total_training_steps)
            if final:
                self._stop_background()
            controller = self._joint_controller
            was_active = None
            if controller is not None:
                was_active = self._await(controller.pause_background_writes())
            try:
                result = super()._save_checkpoint()
                if final:
                    self._final_checkpoint_seal = self._checkpoint_seal(self._last_joint_checkpoint)
                    self._assert_final_seal()
                return result
            finally:
                if controller is not None and not getattr(self, "_background_stopped", False):
                    self._await(controller.resume_background_writes(was_active))

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

        restored = self._joint_coordinator.restore(
            checkpoint.checkpoint_dir, expected=expected, destination_root=destination,
            load_online=load_online, load_ema=load_ema, load_training_state=load_training,
            fence=lambda: self._joint_lock,
        )
        with open_dict(self.config.skill_harness):
            self.config.skill_harness.store_root = str(restored.skill_store.root)
            self.config.skill_harness.replay_root = str(restored.replay.root)
            self.config.skill_harness.ledger_root = str(restored.ledger.root)
        self._joint_restored = restored
        self.global_steps = restored.checkpoint.payload["spec"]["global_step"]

    def _assert_final_seal(self):
        seal = self._final_checkpoint_seal
        if seal is None or seal["optimizer_step"] <= 0:
            raise RuntimeError("Final evaluation requires a completed trained joint checkpoint")
        if self._ema_clock() != seal["optimizer_step"] or self._joint_store.head() != seal["snapshot_id"]:
            raise RuntimeError("The policy or library changed after the final joint checkpoint")

    def _pinned_validation_manager(self, snapshot_id):
        config = copy.deepcopy(self.config)
        with open_dict(config.skill_harness.loading):
            config.skill_harness.loading.snapshot_id = snapshot_id
        manager_class = load_class_from_fqn(
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager",
        )
        return manager_class.create(
            config=config,
            llm_client=self.get_llm_client(),
            teacher_client=self.get_teacher_client(),
            reward_loop_worker_handles=self.get_reward_handles(),
        )

    def _evaluate_final(self):
        from .evaluation import evaluate_official_splits

        self._assert_final_seal()
        self.agent_loop_manager = self._pinned_validation_manager(self._final_checkpoint_seal["snapshot_id"])
        lifecycle = {
            "schema_version": 1,
            "kind": "quiescent_final_evaluation",
            **self._final_checkpoint_seal,
            "evaluation_step": self.global_steps,
            "background_stopped_before_test_service": True,
        }
        root = self.artifact_root / "evaluation" / "final"
        root.mkdir(parents=True, exist_ok=True)
        with (root / "lifecycle.json").open("x", encoding="utf-8") as stream:
            json.dump(lifecycle, stream, indent=2, sort_keys=True)
            stream.write("\n")
        self._init_dump_executor()
        try:
            evaluate_official_splits(self)
            self._assert_final_seal()
            with (root / "complete.json").open("x", encoding="utf-8") as stream:
                json.dump({**lifecycle, "optimizer_updates_during_test": 0}, stream, indent=2, sort_keys=True)
                stream.write("\n")
        finally:
            self._shutdown_dump_executor()

    def _shutdown_joint_services(self):
        errors = []
        try:
            self._stop_background()
        except BaseException as error:
            errors.append(error)
        if self._joint_controller is not None:
            for name in ("ema_gateway", "frozen_gateway"):
                gateway = getattr(self._joint_controller, name, None)
                if gateway is not None:
                    try:
                        self._await(gateway.shutdown())
                        setattr(self._joint_controller, name, None)
                    except BaseException as error:
                        errors.append(error)
        if errors:
            raise errors[0]

    def fit(self, agent_loop_manager):
        try:
            self._final_evaluation_enabled = self.config.skill_harness.get("evaluation", {}).get("final", False)
            self._final_checkpoint_seal = None
            if self._final_evaluation_enabled:
                root = self.artifact_root / "evaluation" / "final"
                if root.exists() and any(root.iterdir()):
                    raise FileExistsError("Completed or partial final evaluation artifacts already exist")
                if self._joint_restored is not None:
                    self._final_checkpoint_seal = self._checkpoint_seal(self._joint_restored.checkpoint)
            result = super().fit(agent_loop_manager)
            self._shutdown_joint_services()
            if self._final_evaluation_enabled:
                self._evaluate_final()
            return result
        finally:
            primary_error = sys.exc_info()[1]
            try:
                self._shutdown_joint_services()
            except BaseException as error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note("Background cleanup also failed: " + repr(error))
            finally:
                if self._joint_loop.is_running():
                    self._joint_loop.call_soon_threadsafe(self._joint_loop.stop)
                    self._joint_thread.join()
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
