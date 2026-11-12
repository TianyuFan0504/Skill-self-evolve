"""CPU recovery contracts; model and native OS boundaries remain inert."""

from __future__ import annotations

import ast
import asyncio
import copy
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/experimental/skill_harness"))
protocol = importlib.import_module("test_curate")
queue_fixture = importlib.import_module("test_source_queue")

SPEC = importlib.util.spec_from_file_location(
    "_curate_generation_cpu", ROOT / "examples/skill_self_evolve/scripts/curate_generation.py"
)
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)
OWNER = {"pid": 123, "start_ticks": 456}


class FrozenBoundary:
    """Observe calls to the unchanged collector boundary without a model."""

    def _initialize_background(self):
        self.events.append(("pump_start", copy.deepcopy(self.state)))

    async def _run_background_source(self, key, entry):
        self.events.append(("delegate", copy.deepcopy(entry)))

    async def _run_background_probe(self, job):
        return await self.probe_body(job)


Controller = type("GenerationController", (generation.GenerationControllerMixin, FrozenBoundary),
                  {"__module__": generation.__name__})


class GenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.owner = patch.object(generation, "process_identity", return_value=OWNER)
        self.owner.start()
        self.addCleanup(self.owner.stop)
        runtime = types.ModuleType("skill_harness_examples.learner_runtime")
        runtime.PROBE_CONCURRENCY, runtime.CONFIGURED_PROBE_CONCURRENCY, runtime.START_METHOD = 2, 8, "spawn"
        runtime.multiprocessing = types.SimpleNamespace(get_start_method=lambda: "spawn")
        examples = types.ModuleType("skill_harness_examples")
        examples.__path__ = []
        examples.learner_runtime = runtime
        self.modules = patch.dict(sys.modules, {
            "verl.experimental.skill_harness.curate": protocol.curate,
            "verl.experimental.skill_harness.source_queue": queue_fixture.queue,
            "skill_harness_examples": examples,
            "skill_harness_examples.learner_runtime": runtime,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def trainer(self, name="run", clock=10):
        root = self.root / name
        checkpoint = types.SimpleNamespace(
            checkpoint_dir=str(self.root / "checkpoints/global_step_10"), manifest_id="sha256:" + "c" * 64,
            payload={"spec": {"optimizer_step": clock}},
        )
        return types.SimpleNamespace(
            config=types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(root))),
            _joint_restored=types.SimpleNamespace(checkpoint=checkpoint),
            _learner_runtime_ref={"path": "runtime-reference", "sha256": "a" * 64},
        )

    def controller(self, trainer=None):
        trainer = trainer or self.trainer()
        generation.prepare_namespace(trainer)
        controller = Controller()
        controller.trainer = trainer
        controller.state = {"optimizer_step": 10}
        controller.events = []
        controller.save_state = lambda: controller.events.append(("persist", copy.deepcopy(controller.state)))
        controller._writes_allowed = asyncio.Event()
        controller._writes_allowed.set()
        controller.routes = {"curator": types.SimpleNamespace(optimizer_step=10)}
        controller.loading = {"tokenizer_version": "tokenizer:one"}
        controller.runtime = types.SimpleNamespace(template_version="template:one")
        controller.settings = types.SimpleNamespace(candidates_per_source=1, credit_concurrency=8)
        controller.store = types.SimpleNamespace(head=lambda: "a" * 64)
        controller._initialize_background()
        controller.source_queue = queue_fixture.queue.SourceQueue(
            {}, persist=lambda: controller.events.append(("source_persist",)),
        )
        source = {**queue_fixture.source(1), "update_id": "update10"}
        controller.source_queue.offer_many([source], source_step=10)
        key, entry = controller.source_queue.claim_ready(1)[0]
        controller.state["source_queue"] = controller.source_queue.state
        return controller, key, entry

    def test_namespace_is_persisted_once_per_execution_and_new_after_restore(self):
        trainer = self.trainer()
        first = generation.prepare_namespace(trainer)
        self.assertEqual(generation.prepare_namespace(trainer), first)
        self.assertTrue(Path(first["path"]).is_file())
        second = generation.prepare_namespace(self.trainer())
        self.assertNotEqual(first, second)
        self.assertEqual(generation.read_ref(first)["source_checkpoint"],
                         generation.read_ref(second)["source_checkpoint"])

    def test_namespace_and_activation_precede_pump_and_bind_actual_class_clock(self):
        controller, _, _ = self.controller()
        self.assertEqual([row[0] for row in controller.events[:2]], ["persist", "pump_start"])
        activation = generation.read_ref(controller.state["curate_generation_activation_ref"])
        self.assertEqual(activation["controller_class"], Controller.__module__ + ".GenerationController")
        self.assertEqual(activation["optimizer_step"], 10)
        self.assertEqual(activation["namespace_ref"], controller.trainer._curate_generation_ref)

    def test_different_owner_cannot_reuse_an_execution_namespace(self):
        trainer = self.trainer()
        generation.prepare_namespace(trainer)
        with patch.object(generation, "process_identity", return_value={"pid": 124, "start_ticks": 900}):
            with self.assertRaisesRegex(ValueError, "restored execution"):
                generation.prepare_namespace(trainer)

    def test_changed_checkpoint_cannot_reuse_namespace(self):
        trainer = self.trainer()
        generation.prepare_namespace(trainer)
        trainer._joint_restored.checkpoint.payload["spec"]["optimizer_step"] = 11
        with self.assertRaisesRegex(ValueError, "restored execution"):
            generation.prepare_namespace(trainer)

    def test_failed_namespace_persistence_prevents_prepared_reference(self):
        trainer = self.trainer()
        with patch.object(generation.os, "fsync", side_effect=OSError("storage unavailable")):
            with self.assertRaises(OSError):
                generation.prepare_namespace(trainer)
        self.assertFalse(hasattr(trainer, "_curate_generation_ref"))

    def test_final_only_restore_rejects_before_namespace_publication(self):
        trainer = self.trainer()
        trainer.total_training_steps = 10
        with self.assertRaisesRegex(ValueError, "final-only recovery"):
            generation.prepare_namespace(trainer)
        self.assertFalse(hasattr(trainer, "_curate_generation_ref"))
        self.assertFalse(Path(trainer.config.skill_harness.artifact_root).exists())
        trainer._joint_restored = None
        trainer.total_training_steps = 0
        with self.assertRaisesRegex(ValueError, "remaining optimizer updates"):
            generation.prepare_namespace(trainer)

    def test_all_request_fields_persist_before_delegate_and_source_identity_is_unchanged(self):
        controller, key, entry = self.controller()
        original = copy.deepcopy(entry)
        asyncio.run(controller._run_background_source(key, entry))
        persisted = next(row[1]["source_queue"]["jobs"][key] for row in controller.events
                         if row[0] == "persist" and "source_queue" in row[1])
        self.assertEqual(persisted, entry)
        for name in ("source_identity_hash", "producer_uid", "episode_output_dir", "source_step", "update_id"):
            self.assertEqual(entry[name], original[name])
        job = entry["curator_jobs"][0]
        request = protocol.curate.CurateRequest(
            **{k: job[k] for k in protocol.curate.CurateRequest.__dataclass_fields__})
        self.assertEqual(request.candidate_group_size, 1)
        self.assertEqual(controller.events[-1][0], "delegate")

    def test_retry_same_execution_preserves_exact_jobs_context_behavior_and_time(self):
        controller, key, entry = self.controller()
        asyncio.run(controller._run_background_source(key, entry))
        saved = copy.deepcopy(entry)
        with patch.object(generation.time, "time", side_effect=AssertionError("retry cannot retimestamp")):
            asyncio.run(controller._run_background_source(key, entry))
        self.assertEqual(entry, saved)

    def test_new_execution_has_disjoint_collection_skill_session_and_replay_ids(self):
        requests = []
        for _ in range(2):
            controller, key, entry = self.controller()
            asyncio.run(controller._run_background_source(key, entry))
            job = entry["curator_jobs"][0]
            requests.append(protocol.curate.CurateRequest(
                **{k: job[k] for k in protocol.curate.CurateRequest.__dataclass_fields__}))
        for attribute in ("candidate_group_id", "collection_id", "skill_id", "session_id", "record_id"):
            self.assertNotEqual(getattr(requests[0], attribute), getattr(requests[1], attribute))

    def test_same_execution_json_roundtrip_preserves_behavior_identity(self):
        controller, key, entry = self.controller()
        asyncio.run(controller._run_background_source(key, entry))
        restored = json.loads(json.dumps(entry))
        controller.source_queue.state["jobs"][key] = restored
        asyncio.run(controller._run_background_source(key, restored))
        self.assertEqual(restored["status"], "running")
        self.assertEqual(sum(row[0] == "delegate" for row in controller.events), 2)

    def test_nonempty_legacy_jobs_are_not_renamed_or_launched(self):
        controller, key, entry = self.controller()
        entry["curator_jobs"] = [{"candidate_group_id": "original", "created_at": 123, "mu": [-.1]}]
        old = copy.deepcopy(entry["curator_jobs"])
        asyncio.run(controller._run_background_source(key, entry))
        self.assertEqual(entry["curator_jobs"], old)
        self.assertEqual(entry["status"], "failed")
        self.assertFalse(any(row[0] == "delegate" for row in controller.events))

    def test_terminal_failure_remains_immutable(self):
        controller, key, entry = self.controller()
        controller.source_queue.finish(key, "failed", reason="old_interrupted_action")
        old = copy.deepcopy(entry)
        asyncio.run(controller._run_background_source(key, entry))
        self.assertEqual(entry, old)
        self.assertFalse(any(row[0] == "delegate" for row in controller.events))

    def test_changed_ema_cannot_retry_an_existing_action(self):
        controller, key, entry = self.controller()
        asyncio.run(controller._run_background_source(key, entry))
        old = copy.deepcopy(entry["curator_jobs"])
        count = sum(row[0] == "delegate" for row in controller.events)
        controller.routes["curator"].optimizer_step = 11
        asyncio.run(controller._run_background_source(key, entry))
        self.assertEqual(entry["curator_jobs"], old)
        self.assertEqual(sum(row[0] == "delegate" for row in controller.events), count)
        self.assertEqual(entry["status"], "failed")

    def test_actual_frozen_initializer_never_requeues_running_uncertain_actions(self):
        tree = ast.parse((ROOT / "verl/experimental/skill_harness/controller.py").read_text())
        controller_node = next(node for node in tree.body if getattr(node, "name", None) == "SkillJointController")
        initializer = next(node for node in controller_node.body
                           if getattr(node, "name", None) == "_initialize_background")
        events = []
        scope = {"__package__": protocol.PACKAGE, "asyncio": types.SimpleNamespace(
            create_task=lambda value, **kwargs: events.append("pump"))}
        exec(compile(ast.Module(body=[initializer], type_ignores=[]), "frozen_controller_init", "exec"), scope)
        frozen = type("ActualFrozenInitializer", (), {"_initialize_background": scope["_initialize_background"]})
        controller_type = type("GenerationController", (generation.GenerationControllerMixin, frozen), {})
        controller = controller_type()
        controller.trainer = self.trainer()
        generation.prepare_namespace(controller.trainer)
        queue = queue_fixture.queue.SourceQueue({}, persist=lambda: None)
        queue.offer_many([{**queue_fixture.source(1), "update_id": "update10"}], source_step=10)
        key, entry = queue.claim_ready(1)[0]
        entry["curator_jobs"] = [{"created_at": 12, "candidate_group_id": "original", "mu": [-.3]}]
        jobs = copy.deepcopy(entry["curator_jobs"])
        controller.state = {"optimizer_step": 10, "source_queue": queue.state, "probe_jobs": {}}
        controller.settings = types.SimpleNamespace(max_source_backlog=8192, credit_concurrency=8)
        controller.save_state = lambda: events.append("persist")
        controller._pump_background = lambda: None
        controller._initialize_background()
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["details"]["reason"], "interrupted_background_generation")
        self.assertEqual(entry["curator_jobs"], jobs)
        self.assertLess(events.index("persist"), events.index("pump"))

    def test_probe_gate_bounds_eight_logical_jobs_without_blocking_loop_or_waiter_leases(self):
        async def scenario():
            controller, _, _ = self.controller()
            release = asyncio.Event()
            active, peak, heartbeats, entered = 0, 0, 0, []

            async def body(job):
                nonlocal active, peak
                entered.append(job["id"])
                active += 1
                peak = max(peak, active)
                try:
                    await release.wait()
                finally:
                    active -= 1

            controller.probe_body = body
            jobs = [{"id": index, "status": "running"} for index in range(8)]
            tasks = [asyncio.create_task(controller._run_background_probe(job)) for job in jobs]
            for _ in range(5):
                await asyncio.sleep(.001)
                heartbeats += 1
            self.assertEqual(entered, [0, 1])
            self.assertEqual(peak, 2)
            self.assertEqual(heartbeats, 5)
            tasks[-1].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[-1]
            self.assertEqual(jobs[-1], {"id": 7, "status": "running"})
            release.set()
            await asyncio.gather(*tasks[:-1])
            self.assertEqual(len(entered), 7)
            self.assertEqual(controller._physical_probe_gate._value, 2)
            self.assertEqual(controller.settings.credit_concurrency, 8)

        asyncio.run(scenario())

    def test_frozen_probe_unknown_and_cancellation_do_not_create_zero_reward(self):
        tree = ast.parse((ROOT / "verl/experimental/skill_harness/controller.py").read_text())
        controller_node = next(node for node in tree.body if getattr(node, "name", None) == "SkillJointController")
        probe = next(node for node in controller_node.body if getattr(node, "name", None) == "_run_background_probe")
        namespace = {"asyncio": asyncio}
        exec(compile(ast.Module(body=[probe], type_ignores=[]), "frozen_probe", "exec"), namespace)
        base = type("FrozenProbe", (), {"_run_background_probe": namespace["_run_background_probe"]})
        cls = type("ProbeController", (generation.GenerationControllerMixin, base), {})

        async def scenario():
            controller = cls()
            controller._physical_probe_gate = asyncio.Semaphore(2)
            controller._writes_allowed = asyncio.Event()
            controller._writes_allowed.set()
            controller.state = {"pending": {"w": {"observed": {}, "unobserved": {}}}}
            controller.save_state = lambda: None
            job = {"window_id": "w", "logical_id": "p", "assignment": {}, "source_step": 12,
                   "status": "running"}

            async def unknown(*args):
                raise ValueError("unknown tail provenance")

            controller._observe_one = unknown
            await controller._run_background_probe(job)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(controller.state["pending"]["w"]["observed"], {})
            self.assertEqual(controller.state["pending"]["w"]["unobserved"], {"p": "ValueError"})
            entered = asyncio.Event()

            async def blocked(*args):
                entered.set()
                await asyncio.Event().wait()

            controller._observe_one = blocked
            job = {**job, "logical_id": "cancel", "status": "running"}
            task = asyncio.create_task(controller._run_background_probe(job))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(job["status"], "running")
            self.assertNotIn("cancel", controller.state["pending"]["w"]["observed"])
            self.assertEqual(controller._physical_probe_gate._value, 2)

        asyncio.run(scenario())

    def test_new_branch_abstention_still_has_no_replay_or_credit_label(self):
        fixture = protocol.CurateTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        controller, key, entry = self.controller()
        controller.store = fixture.store
        asyncio.run(controller._run_background_source(key, entry))
        job = entry["curator_jobs"][0]
        fixture.request = protocol.curate.CurateRequest(
            **{k: job[k] for k in protocol.curate.CurateRequest.__dataclass_fields__})
        fixture.content = json.dumps({"decision": "abstain", "reason": "No stable reusable rule is supported."})
        result = fixture.collect()
        self.assertIsInstance(result, protocol.curate.CurateAbstention)
        with self.assertRaises(FileNotFoundError):
            fixture.replay.get(fixture.request.record_id)
        complete = protocol.curate._read_immutable(Path(result.output_dir) / "complete.json")
        self.assertFalse(complete["credit_registration_required"])

    def test_real_collector_keeps_orphan_immutable_and_preserves_actual_mu_on_new_branch(self):
        fixture = protocol.CurateTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        old = fixture.collect()
        old_path = Path(old.output_dir) / "complete.json"
        old_bytes = old_path.read_bytes()
        fixture.store = protocol.store_module.SkillStore(fixture.root / "restored-store")
        fixture.replay = protocol.replay_module.CurateReplay(fixture.root / "restored-replay")
        controller, key, entry = self.controller()
        controller.store = fixture.store
        controller.routes["curator"].optimizer_step = fixture.behavior.rollout_weight_version
        controller.loading["tokenizer_version"] = fixture.behavior.tokenizer_version
        controller.runtime.template_version = fixture.behavior.template_version
        asyncio.run(controller._run_background_source(key, entry))
        job = entry["curator_jobs"][0]
        fixture.request = protocol.curate.CurateRequest(
            **{k: job[k] for k in protocol.curate.CurateRequest.__dataclass_fields__})
        new = fixture.collect()
        self.assertNotEqual(old.collection_id, new.collection_id)
        self.assertEqual(old_path.read_bytes(), old_bytes)
        record = fixture.replay.get(new.record_id)
        self.assertEqual(tuple(record.behavior_logprobs), (-.3, 0, -.7))
        self.assertEqual(tuple(record.prompt_ids), (101, 102, 103))
        self.assertEqual(tuple(record.action_ids), (501, 502, 503))
        self.assertEqual(fixture.collect(), new)
        self.assertEqual(len(fixture.agent_calls), 2)  # Old orphan draw + one new draw; retry is read-only.


if __name__ == "__main__":
    unittest.main()
