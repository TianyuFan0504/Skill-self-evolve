"""Namespace new queued curator draws without rewriting checkpoint history.

This examples-only controller keeps the frozen collector, replay and credit
protocol. A resumed process gets a new durable execution namespace; a source
that was still queued at the checkpoint can then have a new draw without
colliding with orphan journals produced after that checkpoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path

POLICY = "queued-source-generation-namespace-v1"
SCOPE = ("New current-branch draws for queued sources without curator_jobs; "
         "old actions and terminal states are preserved")
_controller_class = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_ref(ref):
    path = Path(ref["path"])
    data = path.read_bytes()
    if path != path.resolve() or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Curator generation reference path/hash differs")
    return json.loads(data)


def publish(path, value):
    """Publish one complete immutable JSON document, then confirm persistence."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(".pending-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("Curator generation identity cannot be rewritten") from None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def process_identity():
    text = (Path("/proc") / str(os.getpid()) / "stat").read_text()
    return {"pid": os.getpid(), "start_ticks": int(text[text.rfind(")") + 2:].split()[19])}


def source_checkpoint(trainer):
    checkpoint = getattr(getattr(trainer, "_joint_restored", None), "checkpoint", None)
    return None if checkpoint is None else {
        "checkpoint_dir": checkpoint.checkpoint_dir,
        "manifest_id": checkpoint.manifest_id,
        "optimizer_step": checkpoint.payload["spec"]["optimizer_step"],
    }


def namespace_for(trainer):
    ref = trainer._curate_generation_ref
    value = read_ref(ref)
    provider = Path(__file__).resolve()
    namespace = value.get("namespace", "")
    root = Path(trainer.config.skill_harness.artifact_root).resolve()
    if (not re.fullmatch("[0-9a-f]{64}", namespace)
            or Path(ref["path"]) != root / "controller/generation_epochs" / (namespace + ".json")
            or value.get("schema_version") != 1 or value.get("kind") != "prepared_curate_generation_namespace"
            or value.get("policy") != POLICY or value.get("scope") != SCOPE
            or value.get("source_checkpoint") != source_checkpoint(trainer)
            or value.get("owner") != process_identity()
            or value.get("provider_ref") != {
                "path": str(provider), "sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
            }):
        raise ValueError("Curator namespace does not belong to this restored execution/provider")
    return value


def prepare_namespace(trainer):
    """Run after actual checkpoint restore, before controller/model curation."""
    source = source_checkpoint(trainer)
    target = getattr(trainer, "total_training_steps", None)
    clock = 0 if source is None else source["optimizer_step"]
    if target is not None and clock >= target:
        raise ValueError(
            "Curate generation entry requires remaining optimizer updates; final-only recovery is unsupported"
        )
    if getattr(trainer, "_curate_generation_ref", None) is not None:
        namespace_for(trainer)
        return trainer._curate_generation_ref
    root = Path(trainer.config.skill_harness.artifact_root).resolve()
    namespace = digest([POLICY, uuid.uuid4().hex])
    provider = Path(__file__).resolve()
    value = {
        "schema_version": 1, "kind": "prepared_curate_generation_namespace", "policy": POLICY,
        "namespace": namespace, "created_at_ns": time.time_ns(),
        "owner": process_identity(),
        "source_checkpoint": source,
        "provider_ref": {"path": str(provider), "sha256": hashlib.sha256(provider.read_bytes()).hexdigest()},
        "scope": SCOPE,
    }
    trainer._curate_generation_ref = publish(root / "controller/generation_epochs" / (namespace + ".json"), value)
    return trainer._curate_generation_ref


class GenerationControllerMixin:
    def _initialize_background(self):
        from skill_harness_examples import learner_runtime

        ref = self.trainer._curate_generation_ref
        namespace = namespace_for(self.trainer)
        source = namespace["source_checkpoint"]
        expected_clock = 0 if source is None else source["optimizer_step"]
        if self.state["optimizer_step"] != expected_clock:
            raise ValueError("Curator generation namespace and restored controller clocks differ")
        if (learner_runtime.multiprocessing.get_start_method() != learner_runtime.START_METHOD
                or self.settings.credit_concurrency != learner_runtime.CONFIGURED_PROBE_CONCURRENCY):
            raise ValueError("Actual learner start method or configured credit ceiling differs before activation")
        self._curate_generation_namespace = namespace["namespace"]
        self._physical_probe_gate = asyncio.Semaphore(learner_runtime.PROBE_CONCURRENCY)
        activation = {
            "schema_version": 1, "kind": "activated_curate_generation_controller",
            "namespace": namespace["namespace"], "namespace_ref": ref,
            "controller_class": type(self).__module__ + "." + type(self).__qualname__,
            "optimizer_step": self.state["optimizer_step"],
            "runtime_guards": {
                "learner_runtime_ref": self.trainer._learner_runtime_ref,
                "observed_default_start_method": learner_runtime.multiprocessing.get_start_method(),
                "probe_semaphore_class": type(self._physical_probe_gate).__module__ + "."
                + type(self._physical_probe_gate).__qualname__,
                "probe_semaphore_initial_value": self._physical_probe_gate._value,
                "configured_credit_concurrency": self.settings.credit_concurrency,
            },
        }
        path = Path(ref["path"]).with_name(namespace["namespace"] + ".activated.json")
        self.state["curate_generation_activation_ref"] = publish(path, activation)
        self.save_state()
        # The frozen initializer preserves running-action uncertainty and only
        # creates its background pump after this durable activation record.
        return super()._initialize_background()

    async def _run_background_probe(self, job):
        # The frozen pump can retain its configured ceiling of eight logical
        # jobs. Only this many confirmed-lifecycle probe bodies enter the lease
        # and RPC path together; waiting acquires neither and never blocks the loop.
        async with self._physical_probe_gate:
            return await super()._run_background_probe(job)

    async def _run_background_source(self, key, entry):
        from verl.experimental.skill_harness.curate import CuratorBehavior
        from verl.experimental.skill_harness.source_queue import TERMINAL_STATUSES

        await self._writes_allowed.wait()
        if entry["status"] in TERMINAL_STATUSES:
            return
        if entry["status"] != "running":
            raise ValueError("Only an already persisted source claim can create curator jobs")
        behavior = CuratorBehavior(
            "ema:" + str(self.routes["curator"].optimizer_step), self.routes["curator"].optimizer_step,
            self.loading["tokenizer_version"], self.runtime.template_version,
        )
        namespace = self._curate_generation_namespace
        if entry["curator_jobs"]:
            # Existing actions belong to their original request/behavior. In
            # particular, an old queued job cannot be renamed as a new draw.
            if (entry.get("curate_generation_namespace") != namespace
                    or digest(entry.get("behavior")) != digest(asdict(behavior))):
                self.source_queue.finish(key, "failed", reason="existing_curate_jobs_require_reconciliation")
                return
        else:
            cohort = "cohort_" + digest([entry["update_id"], entry["episode_output_dir"], namespace])
            entry["curator_jobs"] = [{
                **{field: entry[field] for field in ("producer_uid", "episode_output_dir", "task_id", "task_type")},
                "skill_snapshot_id": self.store.head(), "candidate_group_id": cohort,
                "candidate_index": index, "candidate_group_size": self.settings.candidates_per_source,
                "created_at": time.time(), "status": "queued",
            } for index in range(self.settings.candidates_per_source)]
            entry["behavior"] = asdict(behavior)
            entry["curate_generation_namespace"] = namespace
            self.save_state()
        # No suspension between request persistence and the frozen method's
        # initial open gate: the pump prevents EMA publication while this task
        # is active. The frozen collector still records actual processed mu.
        return await super()._run_background_source(key, entry)


def controller_class():
    global _controller_class
    if _controller_class is None:
        from skill_harness_examples.train_resilient_setup import controller_class as base_class

        _controller_class = type("GenerationController", (GenerationControllerMixin, base_class()),
                                 {"__module__": __name__})
    return _controller_class


class GenerationTrainerMixin:
    def _joint_controller_class(self):
        return controller_class()
