#!/usr/bin/env python3
"""Parallel Native joint training with one validation and direct checkpoint reads.

The existing parallel client/server and spawn/final-evaluation components are
used unchanged. Separate receipts identify this actual outer entry and restore
provider; the frozen component receipts retain their original identities.
"""

from __future__ import annotations

import asyncio
import faulthandler
import hashlib
import inspect
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

POLICY = "parallel-native-direct-checkpoint-v1"
_RETAINED_FAULT_FILES = []  # A failed timer cancellation must never leave a reusable, closed descriptor.


def _await_frames(coroutine, limit=12):
    """Describe code locations only: never serialize arguments, locals or reprs."""
    result, seen = [], set()
    while coroutine is not None and len(result) < limit and id(coroutine) not in seen:
        seen.add(id(coroutine))
        if inspect.iscoroutine(coroutine):
            frame, following = coroutine.cr_frame, coroutine.cr_await
        elif inspect.isgenerator(coroutine):
            frame, following = coroutine.gi_frame, coroutine.gi_yieldfrom
        elif inspect.isasyncgen(coroutine):
            frame, following = coroutine.ag_frame, coroutine.ag_await
        else:
            break
        if frame is not None:
            result.append({"file": frame.f_code.co_filename, "function": frame.f_code.co_name,
                           "line": frame.f_lineno})
        coroutine = following
    return result


class PassiveControlDiagnostics:
    """One learner-local observer; disk I/O never runs on its asyncio loop.

    The periodic fault dump remains useful if synchronous code stops the loop.
    Loop callbacks only inspect code locations and enqueue bounded metadata.
    Local files survive close so an operator can archive them after shutdown.
    """

    def __init__(self, run, *, interval=30.0, stack_interval=120.0, local_parent="/tmp", faults=faulthandler):
        self.interval, self.stack_interval, self.faults = interval, stack_interval, faults
        self.directory = Path(tempfile.mkdtemp(prefix=f"skill-control-{os.getpid()}-", dir=local_parent))
        self.directory.chmod(0o700)
        self.fault_path, self.snapshot_path = self.directory / "threads.log", self.directory / "async-stages.jsonl"
        self._fault_file = self.fault_path.open("xb")
        os.fchmod(self._fault_file.fileno(), 0o600)
        descriptor = os.open(self.snapshot_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self._queue, self._lock, self._stop = queue.Queue(maxsize=4), threading.Lock(), threading.Event()
        self._trainer, self._stage, self._pending = None, "entry_initialized", None
        self._closed = self._timer_started = False
        self._thread = None
        self._dropped = 0
        index = {"schema_version": 1, "kind": "passive_local_control_diagnostics", "pid": os.getpid(),
                 "local_directory": str(self.directory), "thread_stacks": str(self.fault_path),
                 "async_stages": str(self.snapshot_path), "interval_seconds": interval,
                 "stack_interval_seconds": stack_interval,
                 "scope": "code locations and control flags only; no prompts/tokens/locals/model requests"}
        path = self.directory / "index.json"
        path.write_text(json.dumps(index, indent=2) + "\n")
        path.chmod(0o600)
        # This single reference is written from the runner's main thread,
        # before fit; ongoing diagnostic output stays on the local filesystem.
        self._reference = Path(run) / "diagnostics" / (str(time.time_ns()) + ".passive_control.json")
        self._index_reference = {"schema_version": 1, "kind": "local_passive_control_index",
                                 "pid": os.getpid(), "index_path": str(path),
                                 "index_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def start(self):
        if self._closed or self._thread is not None:
            raise RuntimeError("Passive diagnostics can only start once")
        self.faults.dump_traceback_later(self.stack_interval, repeat=True, file=self._fault_file, exit=False)
        self._timer_started = True
        self._thread = threading.Thread(target=self._run, name="passive-control-diagnostics", daemon=True)
        self._thread.start()
        # Arm local diagnostics before touching NAS so even a slow one-time
        # artifact reference cannot prevent a local fault-stack capture.
        try:
            self._reference.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._reference.open("x") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(self._index_reference, stream)
                stream.write("\n")
        except OSError as error:
            self._enqueue({"kind": "artifact_reference_error", "observed_at_ns": time.time_ns(),
                           "error_type": type(error).__name__, "local_index": self._index_reference["index_path"]})
        return self

    def attach(self, trainer):
        self._trainer = trainer

    def stage(self, value):
        self._stage = value

    def _enqueue(self, event):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped += 1

    def _loop_snapshot(self, submitted):
        if self._closed:
            return
        try:
            event = self._collect_loop_snapshot(submitted)
        except Exception as error:
            event = {"kind": "observer_snapshot_error", "observed_at_ns": time.time_ns(),
                     "stage": self._stage, "error_type": type(error).__name__}
        finally:
            with self._lock:
                if self._pending == submitted:
                    self._pending = None
        self._enqueue(event)

    def _collect_loop_snapshot(self, submitted):
        now = time.monotonic()
        controller = getattr(self._trainer, "_joint_controller", None)
        writes = getattr(controller, "_writes_allowed", None)
        tasks = sorted(asyncio.all_tasks(), key=id)
        return {"kind": "async_loop_snapshot", "observed_at_ns": time.time_ns(),
                "stage": self._stage, "loop_lag_ms": (now - submitted) * 1000,
                "controller_created": controller is not None,
                "learner_active": getattr(controller, "_learner_active", None),
                "ema_syncing": getattr(controller, "_ema_syncing", None),
                "ema_refresh_requested": getattr(controller, "_ema_refresh_requested", None),
                "writes_allowed": writes.is_set() if writes is not None else None,
                "task_count": len(tasks),
                "tasks": [{"task_id": id(task), "done": task.done(),
                           "await_chain": _await_frames(task.get_coro())} for task in tasks[:128]]}

    def _tick(self):
        now = time.monotonic()
        loop = getattr(self._trainer, "_joint_loop", None)
        running = loop is not None and loop.is_running() and not loop.is_closed()
        with self._lock:
            pending = self._pending
            if running and pending is None:
                self._pending = pending = now
                try:
                    loop.call_soon_threadsafe(self._loop_snapshot, pending)
                except RuntimeError:
                    self._pending = pending = None
        self._enqueue({"kind": "watchdog_tick", "observed_at_ns": time.time_ns(), "stage": self._stage,
                       "loop_running": running,
                       "loop_response_pending_ms": None if pending is None else (now - pending) * 1000,
                       "dropped_snapshots": self._dropped})

    def _run(self):
        try:
            with self.snapshot_path.open("a") as stream:
                while True:
                    if not self._stop.is_set():
                        self._tick()
                    while True:
                        try:
                            event = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    if self._stop.wait(self.interval):
                        break
        except Exception:
            # Diagnostics must never cancel training or mutate its state.
            pass

    def close(self):
        self._closed = True
        if self._timer_started:
            try:
                self.faults.cancel_dump_traceback_later()  # Cancel before closing its still-owned descriptor.
                self._timer_started = False
            except Exception:
                if self._fault_file not in _RETAINED_FAULT_FILES:
                    _RETAINED_FAULT_FILES.append(self._fault_file)
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            try:
                self._thread.join(timeout=5)
            except RuntimeError:
                pass
        if not self._timer_started:
            try:
                self._fault_file.close()
            except OSError:
                pass
        self._trainer = None


class _NoDiagnostics:
    def attach(self, _trainer):
        pass

    def stage(self, _value):
        pass

    def close(self):
        pass


def passive_diagnostics(run):
    observer = None
    try:
        observer = PassiveControlDiagnostics(run)
        return observer.start()
    except Exception:
        if observer is not None:
            try:
                observer.close()
            except Exception:
                pass
        return _NoDiagnostics()


def extension_identity():
    from skill_harness_examples import train_spawn_safe

    path = Path(__file__).resolve()
    repository = path.parents[3]
    files = {
        "entry": path,
        "restore_provider": path.with_name("direct_checkpoint_restore.py"),
        "parallel_component": path.with_name("train_parallel_client.py"),
        "spawn_component": path.with_name("train_spawn_safe.py"),
        "checkpoint_core": repository / "verl/experimental/skill_harness/joint_checkpoint.py",
        "trainer_core": repository / "verl/experimental/skill_harness/joint_trainer.py",
    }
    return {
        "policy": POLICY,
        "driver_source_commit": train_spawn_safe._source_commit(repository),
        "actual_entry": str(path),
        "files": {
            name: {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
            for name, file in files.items()
        },
        "core_fingerprint_scope": "Unchanged legacy core; direct-restore implementation is identified separately",
    }


def trainer_class():
    from skill_harness_examples.direct_checkpoint_restore import DirectReadRestoreMixin

    from skill_harness_examples import train_spawn_safe

    class DirectReadJointTrainer(DirectReadRestoreMixin, train_spawn_safe.trainer_class()):
        pass

    return DirectReadJointTrainer


def initialized_receipt(trainer, config):
    from omegaconf import OmegaConf

    from skill_harness_examples import train_spawn_safe
    from verl.experimental.skill_harness.train import _redact

    restored = getattr(trainer, "_joint_restored", None)
    checkpoint = getattr(restored, "checkpoint", None)
    root = Path(config.skill_harness.artifact_root)
    restore = getattr(trainer._joint_coordinator, "last_restore_receipt", None)
    restore_ref = None
    if restore is not None:
        path = Path(restore["mutable_state_root"]) / "DIRECT_READ_RESTORE.json"
        restore_ref = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return train_spawn_safe._base()._publish(
        root / "driver_runs" / (str(time.time_ns()) + ".direct_checkpoint.initialized.json"),
        {
            "schema_version": 1,
            "kind": "initialized_direct_checkpoint_entry",
            **extension_identity(),
            "effective_config_after_trainer_init": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_sha256": hashlib.sha256(
                (root / "training-manifest.json").read_bytes()
            ).hexdigest(),
            "restore_receipt_ref": restore_ref,
            "spawn_component_initialized_ref": trainer._resilient_initialized_receipt,
            "restored_checkpoint": None
            if checkpoint is None
            else {
                "checkpoint_dir": checkpoint.checkpoint_dir,
                "manifest_id": checkpoint.manifest_id,
                "spec": checkpoint.payload["spec"],
            },
            "restore_policy": None if restore is None else restore["policy"],
            "core_version_binding": None if checkpoint is None else checkpoint.payload["spec"]["versions"],
        },
    )


def runner_class():
    import ray

    @ray.remote
    class DirectReadTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf
            from skill_harness_examples.native_posix_spawn import install_spawn_policy

            from skill_harness_examples import train_spawn_safe
            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("Direct checkpoint entry requires the synchronous joint trainer")
            train_spawn_safe.configure_extensions(config)
            install_spawn_policy()
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            diagnostics = passive_diagnostics(config.skill_harness.artifact_root)
            try:
                diagnostics.stage("trainer_construct")
                trainer = trainer_class()(config)
                diagnostics.attach(trainer)
                diagnostics.stage("trainer_init")
                trainer.init()
                diagnostics.stage("initialized_receipts")
                trainer._resilient_extension_identity = train_spawn_safe.extension_identity()
                trainer._resilient_initialized_receipt = train_spawn_safe.initialized_receipt(trainer, config, imports)
                initialized_receipt(trainer, config)
                diagnostics.stage("manager_create")
                factory = load_class_from_fqn(
                    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
                )
                manager = factory.create(
                    config=config,
                    llm_client=trainer.get_llm_client(),
                    teacher_client=trainer.get_teacher_client(),
                    reward_loop_worker_handles=trainer.get_reward_handles(),
                )
                diagnostics.stage("trainer_fit")
                trainer.fit(manager)
                succeeded = True
            except BaseException as error:
                diagnostics.stage("failure_cleanup")
                try:
                    preserve_failed_queue(config, trainer, tq, error)
                except Exception as secondary:
                    error.add_note("Failure capture also failed: " + str(secondary))
                raise
            finally:
                try:
                    tracking = getattr(trainer, "logger", None)
                    if tracking is not None:
                        tracking.finish(exit_code=0 if succeeded else 1)
                finally:
                    try:
                        diagnostics.stage("queue_close")
                        tq.close()
                    finally:
                        diagnostics.close()

    return DirectReadTaskRunner


def entry(config):
    from omegaconf import OmegaConf, open_dict

    from skill_harness_examples import train_parallel_client as parallel
    from skill_harness_examples import train_spawn_safe as spawn
    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment, configure_runtime

    run = Path(config.skill_harness.artifact_root)
    settings_path = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
    settings = json.loads(settings_path.read_bytes())
    health = parallel.environment_observation(settings)
    adapter = parallel.prepare_adapter(
        run, os.environ["ALFWORLD_TASK_CONFIG"], os.environ["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"]
    )
    os.environ.update(adapter["environment_variable_changes"])
    with open_dict(config):
        config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path = (
            adapter["task_profile"]["path"]
        )
    imports = configure_runtime()
    import ray

    from verl.experimental.skill_harness.train import _redact
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    if config.trainer.val_before_train or config.skill_harness.evaluation.initial:
        raise ValueError("Keep pretraining evaluation disabled in this continuation")
    spawn.configure_extensions(config)
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    identity = extension_identity()
    spawn._base()._publish(
        run / "driver_runs" / (str(time.time_ns()) + ".direct_checkpoint.json"),
        {
            "schema_version": 1,
            "kind": "prelaunch_direct_checkpoint_parallel_components",
            **identity,
            "observed_at_ns": time.time_ns(),
            "imports": imports,
            "parallel_component_ref": identity["files"]["parallel_component"],
            "spawn_component_identity": spawn.extension_identity(),
            "client_adapter": adapter,
            "environment_health": health,
            "environment_config_ref": {
                "path": str(settings_path),
                "sha256": parallel.digest(settings_path.read_bytes()),
            },
            "config": _redact(OmegaConf.to_container(config, resolve=True)),
            "original_training_manifest_preserved": True,
            "not_a_post_init_or_served_model_receipt": True,
        },
    )
    try:
        run_ppo(config, task_runner_class=runner_class())
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
