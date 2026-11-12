#!/usr/bin/env python3
"""Examples-only continuation with narrowly invalidated pre-Native setup data.

The original runtime, reward, optimizer and checkpoint code is unchanged. A
late ALFWorld create may still exist: this extension never calls it closed.
Runtime imports are lazy so metadata tests do not initialize model libraries.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import os
import re
import stat
import tempfile
import textwrap
import time
from pathlib import Path

MODULE = "skill_harness_examples.train_resilient_setup"
ORIGINAL_TASK = "alfworld_skill_training"
SAFE_TASK = "alfworld_skill_training_safe_setup_v1"
POLICY = "exact-pre-create-no-native-v1"
MARKERS = "diagnostics/no_native_startup_invalid"
_safe_class = _error_class = _sampler_class = None


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _publish(path, value):
    """Publish one private immutable record, including concurrent identical use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.parent.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077 or path.parent.is_symlink():
        raise ValueError("Setup adapter records require a private host-owned directory")
    data = _bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=".setup-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError("Conflicting setup adapter record") from None
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def extension_identity():
    path = Path(__file__).resolve()
    namespace = path.parents[3] / "skill_harness_examples/__init__.py"
    import subprocess

    return {
        "policy": POLICY,
        "driver_path": str(path),
        "driver_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "namespace_path": str(namespace),
        "namespace_sha256": hashlib.sha256(namespace.read_bytes()).hexdigest(),
        "driver_source_commit": subprocess.check_output(
            ["git", "-C", str(path.parents[3]), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def alias_sample(tools_kwargs):
    copied = copy.deepcopy(tools_kwargs)
    task = copied.get("task") if isinstance(copied, dict) else None
    if not isinstance(task, dict) or task.get("name") != ORIGINAL_TASK:
        raise ValueError("Safe setup runner requires the original ALFWorld Task sample")
    task["name"] = SAFE_TASK
    return copied


def alias_task_profile(task_config_path, run_root):
    # The existing launcher writes JSON (a YAML subset); reject other formats
    # rather than interpreting or mutating an arbitrary profile.
    source = Path(task_config_path)
    rows = json.loads(source.read_bytes())
    selected = [row for row in rows if row.get("name") == ORIGINAL_TASK]
    if len(selected) != 1:
        raise ValueError("Expected one trusted original ALFWorld Task profile")
    profile = copy.deepcopy(selected[0])
    profile["name"] = SAFE_TASK
    data = [profile]
    digest = hashlib.sha256(_bytes(data)).hexdigest()
    path = Path(run_root) / "setup_adapter" / ("task_profile_" + digest + ".json")
    _publish(path, data)
    return str(path)


def exact_setup_timeout(error, task):
    """Recognize only the old core's first create await before local bindings."""
    from verl.experimental.skill_harness.alfworld.task import AlfworldSkillTrainingTask

    if not isinstance(error, TimeoutError):
        return None
    original = AlfworldSkillTrainingTask._run_episode
    lines, start = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    create_lines = [
        start + node.lineno - 1
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncWith)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "post"
            and any(isinstance(value, ast.Constant) and value.value == "/admin/create"
                    for value in ast.walk(item.context_expr))
            for item in node.items
        )
    ]
    if len(create_lines) != 1:
        return None
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is original.__code__:
            frame = trace.tb_frame.f_locals
            forbidden = ("created", "session_id", "account", "account_pool", "account_lease", "sandbox",
                         "agent_started", "backend_binding", "gateway_session_id")
            if trace.tb_lineno != create_lines[0] or any(name in frame for name in forbidden):
                return None
            metadata = frame.get("metadata")
            if metadata is not task.config.metadata:
                return None
            root = task._trusted_run_root()
            attempt, output = frame.get("attempt_id"), frame.get("output")
            if (
                Path(metadata["run_dir"]).resolve() != root
                or not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt)
                or not isinstance(metadata.get("rollout_uid"), str) or not metadata["rollout_uid"]
                or not isinstance(output, Path) or output != output.resolve()
                or output.name != "attempt_" + attempt or root / "episodes" not in output.parents
                or json.loads((output / "attempt.json").read_bytes()) != frame.get("manifest")
            ):
                return None
            return {
                "schema_version": 1, "kind": "NoNativeStartupInvalid", "policy": POLICY,
                "run_root": str(root), "attempt_id": attempt, "episode_output_dir": str(output),
                "rollout_uid": metadata["rollout_uid"], "role": metadata.get("role"),
                "dataset_phase": metadata.get("dataset_phase"), "game_id": metadata.get("game_id"),
                "session_index": metadata.get("session_index"), "global_step": metadata.get("global_step"),
                "training_data_valid": False, "native_started": False,
                "uid_acquired_by_task": False, "backend_binding_created": False,
                "host_pair_account_preleased": metadata.get("host_account_lease") is not None,
                "environment_session_id_received": False, "world_closure": "unknown",
                "error_type": type(error).__name__, "source_path": original.__code__.co_filename,
                "source_line": trace.tb_lineno,
            }
        trace = trace.tb_next
    return None


def ensure_safe_task():
    global _safe_class, _error_class
    if _safe_class is not None:
        return _safe_class
    from uni_agent.tasks.registry import get_task_cls, register_task

    from verl.experimental.skill_harness.alfworld.accounts import LocalArchiveDataError
    from verl.experimental.skill_harness.alfworld.task import AlfworldSkillTrainingTask

    class NoNativeSetupDataError(LocalArchiveDataError):
        """Nonsticky data-invalid carrier; does NOT assert archive/UID/world closure."""

    try:
        existing = get_task_cls(SAFE_TASK)
    except ValueError:
        existing = None
    if existing is not None:
        if getattr(existing, "setup_adapter_policy", None) != POLICY:
            raise ValueError("Safe Task alias belongs to a different extension")
        _safe_class, _error_class = existing, existing.setup_error_class
        return existing

    @register_task(SAFE_TASK)
    class NoNativeSafeTask(AlfworldSkillTrainingTask):
        setup_adapter_policy = POLICY
        setup_error_class = NoNativeSetupDataError

        async def _run_episode(self):
            try:
                return await super()._run_episode()
            except TimeoutError as error:
                proof = exact_setup_timeout(error, self)
                if proof is None:
                    raise
                marker = Path(proof["run_root"]) / MARKERS / (proof["attempt_id"] + ".json")
                reference = _publish(marker, proof)
                raise NoNativeSetupDataError(
                    "Pre-Native environment creation timed out; no outcome observed; world closure unknown",
                    {**proof, "marker": reference},
                ) from error

    _safe_class, _error_class = NoNativeSafeTask, NoNativeSetupDataError
    return _safe_class


async def safe_run_task(**kwargs):
    from uni_agent.framework.task_runner import run_task

    ensure_safe_task()
    forwarded = dict(kwargs)
    forwarded["tools_kwargs"] = alias_sample(kwargs.get("tools_kwargs"))
    task = forwarded["tools_kwargs"]["task"]
    if kwargs.get("sample_index") is not None:
        task.setdefault("metadata", {})["sample_index"] = kwargs["sample_index"]
    forwarded["task_config_path"] = alias_task_profile(kwargs["task_config_path"], task["metadata"]["run_dir"])
    return await run_task(**forwarded)


async def safe_native_probe_branch(**kwargs):
    from verl.experimental.skill_harness.probes import native_probe_branch

    ensure_safe_task()
    forwarded = dict(kwargs)
    fields = copy.deepcopy(kwargs["sample_fields"])
    fields["tools_kwargs"] = alias_sample(fields["tools_kwargs"])
    forwarded["sample_fields"] = fields
    forwarded["task_config_path"] = alias_task_profile(
        kwargs["task_config_path"], fields["tools_kwargs"]["task"]["metadata"]["run_dir"]
    )
    return await native_probe_branch(**forwarded)


async def safe_collect_counterfactual_probe(**kwargs):
    from verl.experimental.skill_harness.probes import collect_counterfactual_probe

    return await collect_counterfactual_probe(**kwargs, run_branch=safe_native_probe_branch)


def no_native_failed_groups(run_root):
    root, failed = Path(run_root).resolve(), set()
    for path in (root / MARKERS).glob("*.json"):
        info = path.lstat()
        if path.is_symlink() or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError("No-Native invalid marker is not private and host-owned")
        row = json.loads(path.read_bytes())
        if (
            row.get("kind") != "NoNativeStartupInvalid" or row.get("policy") != POLICY
            or row.get("run_root") != str(root) or row.get("training_data_valid") is not False
            or row.get("native_started") is not False or row.get("uid_acquired_by_task") is not False
            or row.get("backend_binding_created") is not False or row.get("world_closure") != "unknown"
        ):
            raise ValueError("Invalid No-Native startup marker")
        if row.get("dataset_phase") == "train" and row.get("role") in (None, "solver"):
            if not isinstance(row.get("rollout_uid"), str) or not row["rollout_uid"]:
                raise ValueError("Startup marker has no prompt-group UID")
            failed.add(row["rollout_uid"])
    return failed


class NoNativeSamplerMixin:
    def _terminal_eviction_reasons(self, global_steps, partition_id):
        stale, filtered, failed, counts = super()._terminal_eviction_reasons(global_steps, partition_id)
        if partition_id != "val":
            terminal = self.finished_keys[partition_id] | self.failure_keys[partition_id]
            failed |= no_native_failed_groups(self.archive_run_root) & terminal
        return stale, filtered, failed, counts


def __getattr__(name):
    global _sampler_class
    if name == "NoNativeSetupDataError":
        ensure_safe_task()
        return _error_class
    if name == "ResilientReplayBuffer":
        if _sampler_class is None:
            from verl.experimental.skill_harness.group_sampler import ArchiveAwareReplayBuffer

            _sampler_class = type(name, (NoNativeSamplerMixin, ArchiveAwareReplayBuffer), {"__module__": __name__})
        return _sampler_class
    raise AttributeError(name)


def controller_class():
    from verl.experimental.skill_harness.controller import SkillJointController

    class SafeSetupController(SkillJointController):
        def _initialize_background(self):
            self._collect_probe = safe_collect_counterfactual_probe
            return super()._initialize_background()

    return SafeSetupController


class ResilientLifecycleMixin:
    def _lifecycle_record(self, name, payload):
        return super()._lifecycle_record(name, {
            **payload, "lifecycle_extension": self._resilient_extension_identity,
            "initialized_receipt_ref": self._resilient_initialized_receipt,
        })

    def on_train_begin(self):
        from verl.experimental.skill_harness.trainer import SkillSyncTrainer

        with self._joint_lock:
            if self._joint_controller is not None:
                raise RuntimeError("The joint controller cannot be initialized twice")
            identity = self._identity()
            self._joint_controller = self._await(controller_class().create(
                self, native_runtime=identity.runtime, restored_state=self._joint_restored,
            ))
            if (self._joint_controller.protocol_fingerprint != identity.protocol_fingerprint
                    or self._joint_controller.frozen_version != identity.frozen_version):
                raise RuntimeError("The live controller differs from checkpoint compatibility")
            return SkillSyncTrainer.on_train_begin(self)


def trainer_class():
    from skill_harness_examples.train_quiescent_eval import QuiescentFinalMixin
    from verl.experimental.skill_harness.joint_trainer import SkillJointSyncTrainer

    class ResilientJointTrainer(ResilientLifecycleMixin, QuiescentFinalMixin, SkillJointSyncTrainer):
        pass

    return ResilientJointTrainer


def configure_extensions(config):
    from omegaconf import open_dict

    with open_dict(config):
        runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task
        runner.runner_fqn = MODULE + ".safe_run_task"
        runner.max_concurrent_sessions = 32
        config.trainer.v1.sampler.custom_sampler.path = str(Path(__file__).resolve())
        config.trainer.v1.sampler.custom_sampler.name = "ResilientReplayBuffer"
    os.environ["ROLLOUT_CONCURRENCY"] = "32"


def initialized_receipt(trainer, config, imports):
    from omegaconf import OmegaConf

    from verl.experimental.skill_harness.train import _redact

    root = Path(config.skill_harness.artifact_root)
    restored = getattr(trainer, "_joint_restored", None)
    checkpoint = getattr(restored, "checkpoint", None)
    return _publish(root / "driver_runs" / (str(time.time_ns()) + ".resilient.initialized.json"), {
        **trainer._resilient_extension_identity, "imports": imports,
        "effective_config_after_trainer_init": _redact(OmegaConf.to_container(config, resolve=True)),
        "original_training_manifest_sha256": hashlib.sha256((root / "training-manifest.json").read_bytes()).hexdigest(),
        "restored_checkpoint": None if checkpoint is None else {
            "checkpoint_dir": checkpoint.checkpoint_dir, "manifest_id": checkpoint.manifest_id,
            "spec": checkpoint.payload["spec"],
        },
    })


def runner_class():
    import ray

    @ray.remote
    class ResilientTaskRunner:
        def run(self, config):
            import transfer_queue as tq
            from omegaconf import OmegaConf

            from verl.experimental.skill_harness.bootstrap import configure_runtime
            from verl.experimental.skill_harness.train import preserve_failed_queue
            from verl.utils.import_utils import load_class_from_fqn
            from verl.utils.logging_utils import configure_verl_logging

            imports = configure_runtime()
            configure_verl_logging()
            if config.trainer.v1.trainer_mode != "sync" or config.skill_harness.mode != "skill_joint":
                raise ValueError("Resilient continuation requires the original synchronous joint trainer")
            configure_extensions(config)
            config.transfer_queue.enable = True
            OmegaConf.resolve(config)
            trainer, succeeded = None, False
            tq.init(config.transfer_queue)
            try:
                trainer = trainer_class()(config)
                trainer.init()
                trainer._resilient_extension_identity = extension_identity()
                trainer._resilient_initialized_receipt = initialized_receipt(trainer, config, imports)
                factory = load_class_from_fqn(
                    config.actor_rollout_ref.rollout.agent.agent_loop_manager_class, "AgentLoopManager"
                )
                manager = factory.create(config=config, llm_client=trainer.get_llm_client(),
                                         teacher_client=trainer.get_teacher_client(),
                                         reward_loop_worker_handles=trainer.get_reward_handles())
                trainer.fit(manager)
                succeeded = True
            except BaseException as error:
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
                    tq.close()

    return ResilientTaskRunner


def entry(config):
    from omegaconf import OmegaConf

    from verl.experimental.skill_harness.bootstrap import apply_runtime_environment, configure_runtime

    imports = configure_runtime()
    import ray

    from verl.experimental.skill_harness.train import _redact
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    if config.trainer.val_before_train or config.skill_harness.evaluation.initial:
        raise ValueError("Keep pretraining evaluation disabled in this continuation")
    configure_extensions(config)
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    apply_runtime_environment(config)
    _publish(Path(config.skill_harness.artifact_root) / "driver_runs" / (str(time.time_ns()) + ".resilient.json"), {
        **extension_identity(), "imports": imports,
        "config": _redact(OmegaConf.to_container(config, resolve=True)),
        "original_training_manifest_preserved": True,
    })
    try:
        run_ppo(config, task_runner_class=runner_class())
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    import hydra

    hydra.main(config_path=None, config_name=None, version_base=None)(entry)()
