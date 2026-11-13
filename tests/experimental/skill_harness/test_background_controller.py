"""Bounded CPU contracts for background source/credit work and single learner."""

from __future__ import annotations

import asyncio
import json
import unittest

import test_controller as fixture


class BackgroundControllerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fx = fixture.ControllerFixture()
        await self.fx.asyncSetUp()
        # Transfer only cleanup callbacks, not another TestCase/asyncio runner.
        for function, args, kwargs in self.fx._cleanups:
            self.addCleanup(function, *args, **kwargs)
        self.fx._cleanups.clear()
        self.controllers = []
        self.addAsyncCleanup(self.stop_all)
        self.fx.config.skill_harness.joint = {
            "candidates_per_source": 1,
            "curator_concurrency": 2,
            "credit_concurrency": 2,
            "max_pending_windows": 8192,
            "max_source_backlog": 8192,
            "required_exposures": 1,
            "min_delay_steps": 2,
            "probes_per_step": 4,
        }

    async def stop_all(self):
        for controller in reversed(self.controllers):
            await asyncio.wait_for(controller.stop_background(), timeout=3)

    async def create(self):
        controller = await self.fx.create()
        self.controllers.append(controller)
        return controller

    async def until(self, predicate, timeout=3):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(poll(), timeout=timeout)

    async def update(self, controller, names, step):
        await asyncio.wait_for(controller.enter_learner(), timeout=2)
        try:
            return await asyncio.wait_for(self.fx.update(controller, self.fx.solver(names), step), timeout=2)
        finally:
            await controller.leave_learner()

    async def test_blocked_curator_does_not_block_updates_or_hide_source_opportunities(self):
        controller = await self.create()
        entered, release = asyncio.Event(), asyncio.Event()
        versions = []
        pinned_records = []
        original = self.fx.native_curate

        async def blocked(**kwargs):
            versions.append(kwargs["behavior"].rollout_weight_version)
            pinned_records.append(kwargs["request"].record_id)
            entered.set()
            await release.wait()
            return await original(**kwargs)

        controller._collect_curate = blocked
        await self.update(controller, ("first-d", "first-c", "first-b", "first-a"), 1)
        await asyncio.wait_for(entered.wait(), timeout=2)
        await self.until(lambda: len(versions) == 2)
        self.assertEqual(controller.source_queue.metrics()["sources_enqueued"], 4)
        self.assertEqual(controller.source_queue.metrics()["sources_active"], 2)
        await self.update(controller, ("next-a", "next-b", "next-c", "next-d"), 2)
        self.assertEqual(self.fx.actor.clock, 2)
        self.assertEqual(len(self.fx.actor.update_calls), 2)
        self.assertEqual(controller.state["optimizer_step"], 2)
        self.assertEqual(controller.source_queue.metrics()["sources_enqueued"], 8)
        self.assertEqual(controller.routes["curator"].optimizer_step, 1)
        self.assertEqual(controller.routes["curator"].syncs, [0, 1])
        self.assertEqual(versions, [1, 1])
        self.assertIs(controller.state["source_queue"], controller.source_queue.state)
        original_pinned = list(pinned_records)
        release.set()
        await self.until(lambda: controller.source_queue.metrics()["sources_generated"] >= 2)
        first_records = [controller.replay.get(record_id) for record_id in original_pinned]
        self.assertTrue(all(record.behavior_policy_version == "ema:1" for record in first_records))
        await self.until(
            lambda: controller.routes["curator"].optimizer_step == 2
            and controller.source_queue.metrics()["sources_generated"] == 8
        )
        self.assertIn(2, versions)  # Queued work must not starve the requested refresh.

    async def test_blocked_probe_does_not_block_update3_and_late_credit_enters_update4(self):
        controller = await self.create()
        await self.update(controller, ("producer-a",), 1)
        await self.until(lambda: len(controller.state["collections"]) == 1)
        record_id = next(iter(controller.state["collections"].values()))["record_id"]
        old_record = controller.replay.get(record_id)
        old_mu = tuple(old_record.behavior_logprobs)
        entered, release = asyncio.Event(), asyncio.Event()
        held_sources = asyncio.Event()

        async def hold_source(**kwargs):
            await held_sources.wait()
            return await self.fx.native_curate(**kwargs)

        original_probe = self.fx.native_probe

        async def blocked_probe(**kwargs):
            entered.set()
            await release.wait()
            return await original_probe(**kwargs)

        controller._collect_curate = hold_source
        controller._collect_probe = blocked_probe
        await self.update(controller, ("future-step2",), 2)
        third = await self.update(controller, ("future-step3",), 3)
        await asyncio.wait_for(entered.wait(), timeout=2)
        self.assertEqual(self.fx.actor.clock, 3)
        self.assertEqual(third.acknowledgments, ())
        self.assertEqual(len(self.fx.actor.anchor_calls), 0)
        self.assertTrue(all(not row["observed"] for row in controller.state["pending"].values()))
        self.assertEqual(list((self.fx.root / "credit_ledger/settlements").glob("*.json")), [])
        release.set()
        await self.until(lambda: any(row["status"] == "settled" for row in controller.state["pending"].values()))
        await asyncio.wait_for(controller.enter_learner(), timeout=2)
        try:
            prepared = await asyncio.wait_for(
                controller.prepare_update(self.fx.solver(("future-step4",)), 4), timeout=2
            )
            self.assertEqual(len(prepared.acknowledgments), 1)
            self.assertEqual(prepared.anchor_policy_version, "online:3")
            self.assertEqual(max(prepared.data["curate_advantages"].tolist()), 1)
            self.assertEqual(tuple(controller.replay.get(record_id).behavior_logprobs), old_mu)
            self.assertEqual(controller.replay.get(record_id).behavior_policy_version, "ema:1")
            self.assertEqual(len(self.fx.actor.update_calls), 3)
            self.fx.actor.update_actor(prepared.data)
            await controller.after_update_and_sync(
                self.fx.solver(("future-step4",)), 4, prepared=prepared, optimizer_step=self.fx.actor.clock
            )
        finally:
            await controller.leave_learner()
        self.assertEqual(self.fx.actor.clock, 4)
        self.assertEqual(len(self.fx.actor.update_calls), 4)
        self.assertEqual(controller.state["optimizer_step"], 4)

    async def test_checkpoint_gate_does_not_wait_model_but_delays_store_replay_commit(self):
        controller = await self.create()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.fx.native_curate

        async def blocked(**kwargs):
            entered.set()
            await release.wait()
            return await original(**kwargs)

        controller._collect_curate = blocked
        await self.update(controller, ("checkpoint-source",), 1)
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.wait_for(controller.pause_background_writes(), timeout=0.5)
        release.set()
        await self.until(lambda: bool(list((self.fx.root / "curate").glob("*/generated.json"))))
        self.assertEqual(list((self.fx.root / "curate_replay/records").glob("*.json")), [])
        self.assertEqual(controller.state["collections"], {})
        self.assertEqual(controller.source_queue.metrics()["sources_generated"], 0)
        await controller.resume_background_writes()
        await self.until(lambda: controller.source_queue.metrics()["sources_generated"] == 1)
        self.assertEqual(len(list((self.fx.root / "curate_replay/records").glob("*.json"))), 1)

    async def test_model_abstention_is_read_and_terminal_without_failed_or_generated_skill(self):
        controller = await self.create()
        self.fx.curate_bad_content[0] = json.dumps({"decision": "abstain", "reason": "No reusable evidence."})
        await self.update(controller, ("abstaining-source",), 1)
        await self.until(lambda: controller.source_queue.metrics()["sources_abstained"] == 1)
        metrics = controller.source_queue.metrics()
        self.assertEqual(metrics["sources_read"], 1)
        self.assertEqual(metrics["sources_failed"], 0)
        self.assertEqual(metrics["sources_generated"], 0)
        self.assertEqual(controller.state["pending"], {})
        records = list((self.fx.root / "curate").glob("*/abstention.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text())["payload"]
        self.assertFalse(record["training_label_created"])
        self.assertEqual(list((self.fx.root / "curate_replay/records").glob("*.json")), [])

    async def test_one_bad_source_is_recorded_without_blocking_other_sources_or_next_update(self):
        controller = await self.create()
        _, bad = self.fx.episode("host-bad-source")
        original = controller._load_source

        def source_loader(path):
            if str(path) == str(bad):
                raise OSError("injected source evidence read failure")
            return original(path)

        controller._load_source = source_loader
        await self.update(controller, ("host-bad-source", "host-good-source"), 1)
        await self.until(
            lambda: controller.source_queue.metrics()["sources_failed"] == 1
            and controller.source_queue.metrics()["sources_generated"] == 1
        )
        await self.update(controller, ("later-source",), 2)
        self.assertEqual(self.fx.actor.clock, 2)
        self.assertEqual(controller.state["optimizer_step"], 2)
        failed = [row for row in controller.source_queue.state["jobs"].values() if row["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("source evidence read failure", failed[0]["details"]["message"])


    async def producer(self, controller):
        first = await self.update(controller, ("producer",), 1)
        self.assertEqual(first.acknowledgments, ())
        self.assertTrue((first.data["role_id"] == 0).all())
        await self.until(lambda: len(controller.state["collections"]) == 1)
        record_id = next(iter(controller.state["collections"].values()))["record_id"]
        held = asyncio.Event()

        async def hold_later_sources(**_kwargs):
            await held.wait()

        controller._collect_curate = hold_later_sources
        return record_id

    async def downstream(self, controller):
        second = await self.update(controller, ("future-step2",), 2)
        self.assertEqual(second.acknowledgments, ())
        self.assertEqual(self.fx.probe_calls, [])
        third = await self.update(controller, ("future-step3",), 3)
        self.assertEqual(third.acknowledgments, ())
        return third

    async def matured(self, controller):
        record_id = await self.producer(controller)
        mu = tuple(controller.replay.get(record_id).behavior_logprobs)
        await self.downstream(controller)
        await self.until(lambda: any(row["status"] == "settled" for row in controller.state["pending"].values()))
        return record_id, mu

    async def test_signed_negative_delayed_credit_preserves_behavior_and_main_library(self):
        self.fx.negative_probe = True
        controller = await self.create()
        record_id, mu = await self.matured(controller)
        await controller.enter_learner()
        prepared = await controller.prepare_update(self.fx.solver(("future-step4",)), 4)
        advantages = prepared.data["curate_advantages"].tolist()
        self.assertIn(-1, advantages)
        self.assertEqual(max(advantages), 0)
        self.assertEqual(controller.store.get_snapshot().skills, ())
        self.assertEqual(tuple(controller.replay.get(record_id).behavior_logprobs), mu)
        self.assertTrue(all(row["utility"] == -1 for row in controller.state["pending"].values()))
        self.assertEqual(len(self.fx.actor.anchor_calls), 1)
        self.assertEqual(len(self.fx.actor.update_calls), 3)

    async def test_unknown_backend_is_unobserved_and_never_a_zero_credit_label(self):
        controller = await self.create()
        await self.producer(controller)
        self.fx.unknown_backend = True
        await self.downstream(controller)
        await self.until(lambda: any(job["status"] == "unobserved" for job in controller.state["probe_jobs"].values()))
        await controller.enter_learner()
        prepared = await controller.prepare_update(self.fx.solver(("future-step4",)), 4)
        self.assertEqual(prepared.acknowledgments, ())
        self.assertTrue((prepared.data["role_id"] == 0).all())
        self.assertEqual(controller.ledger.sample_ready(100), [])
        self.assertTrue(all(not row["observed"] for row in controller.state["pending"].values()))
        self.assertEqual(self.fx.actor.anchor_calls, [])
        self.assertEqual(list((self.fx.root / "credit_ledger/settlements").glob("*.json")), [])

    async def assert_input_confound(self, attack):
        controller = await self.create()
        if attack == "displaced":
            artifact = fixture.store_module.SkillArtifact(
                "zz-existing", "Existing note", "Observe before acting.", "Household tasks",
                ("pick_clean_then_place_in_recep",),
            )
            candidate = controller.store.stage_candidate(artifact)
            controller.store.admit(candidate.candidate_id)
            controller.store.publish([candidate.candidate_id], expected_parent=controller.store.head())
        await self.producer(controller)
        self.fx.cf_input_attack = attack
        await self.downstream(controller)
        await self.until(lambda: any(job["status"] == "unobserved" for job in controller.state["probe_jobs"].values()))
        await controller.enter_learner()
        prepared = await controller.prepare_update(self.fx.solver(("future-step4",)), 4)
        self.assertEqual(prepared.acknowledgments, ())
        self.assertEqual(controller.ledger.sample_ready(100), [])
        reports = list((self.fx.root / "credit_probes").glob("*/paired-input-audit.json"))
        self.assertTrue(reports)
        self.assertTrue(all(fixture.probes._read_payload(path)["status"] == "unobserved" for path in reports))
        self.assertFalse(list((self.fx.root / "credit_probes").glob("*/paired-event.json")))

    async def test_actual_native_home_difference_cannot_become_credit(self):
        await self.assert_input_confound("home")

    async def test_actual_native_path_difference_cannot_become_credit(self):
        await self.assert_input_confound("path")

    async def test_displaced_existing_skill_cannot_become_credit(self):
        await self.assert_input_confound("displaced")

    async def test_multiple_exposures_use_distinct_future_batches_before_replay(self):
        self.fx.config.skill_harness.joint.update(required_exposures=2, min_delay_steps=1)
        controller = await self.create()
        await self.producer(controller)
        await self.update(controller, ("first-future",), 2)
        await self.until(lambda: len(next(iter(controller.state["pending"].values()))["observed"]) == 1)
        self.assertEqual(controller.ledger.sample_ready(100), [])
        await self.update(controller, ("second-future",), 3)
        await self.until(lambda: next(iter(controller.state["pending"].values()))["status"] == "settled")
        await controller.enter_learner()
        prepared = await controller.prepare_update(self.fx.solver(("later-batch",)), 4)
        self.assertEqual(len(prepared.acknowledgments), 1)
        self.assertEqual(len(next(iter(controller.state["pending"].values()))["observed"]), 2)
        self.assertEqual({request.global_step for request in self.fx.probe_calls}, {2, 3})

    async def test_stale_positive_label_remains_ready_instead_of_becoming_zero(self):
        self.fx.config.skill_harness.joint.max_replay_lag_steps = 0
        controller = await self.create()
        await self.matured(controller)
        await controller.enter_learner()
        prepared = await controller.prepare_update(self.fx.solver(("future-step4",)), 4)
        self.assertEqual(prepared.acknowledgments, ())
        self.assertTrue((prepared.data["role_id"] == 0).all())
        settled = [row for row in controller.state["pending"].values() if row["status"] == "settled"]
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]["utility"], 1)
        self.assertEqual(controller.ledger.training_status(settled[0]["settlement_id"])["state"], "ready")

    async def test_anchor_clock_failure_keeps_original_mu_and_uncertain_credit_lease(self):
        controller = await self.create()
        record_id, mu = await self.matured(controller)
        await controller.enter_learner()
        self.fx.actor.break_anchor_fence = True
        with self.assertRaises(fixture.controller_module.ControllerRecoveryRequired):
            await controller.prepare_update(self.fx.solver(("future-step4",)), 4)
        acknowledgments = controller.state["prepared_update"]["acknowledgments"]
        self.assertEqual(len(acknowledgments), 1)
        for ack in acknowledgments:
            self.assertEqual(controller.ledger.training_status(ack["settlement_id"])["state"], "leased")
        self.assertEqual(tuple(controller.replay.get(record_id).behavior_logprobs), mu)
        self.assertEqual(len(self.fx.actor.update_calls), 3)

    async def test_unverified_optimizer_receipt_does_not_acknowledge_existing_leases(self):
        controller = await self.create()
        record_id, mu = await self.matured(controller)
        await controller.enter_learner()
        data = self.fx.solver(("future-step4",))
        prepared = await controller.prepare_update(data, 4)
        self.fx.actor.update_actor(prepared.data)
        with self.assertRaises(fixture.controller_module.ControllerRecoveryRequired):
            await controller.after_update_and_sync(data, 4, prepared=prepared, optimizer_step=5)
        self.assertEqual(len(prepared.acknowledgments), 1)
        for ack in prepared.acknowledgments:
            self.assertEqual(controller.ledger.training_status(ack["settlement_id"])["state"], "leased")
        self.assertEqual(tuple(controller.replay.get(record_id).behavior_logprobs), mu)
        self.assertEqual(len(self.fx.actor.update_calls), 4)

    async def test_invalid_candidate_censors_the_whole_cohort_without_fake_negative_labels(self):
        self.fx.config.skill_harness.joint.candidates_per_source = 2
        self.fx.curate_bad_content[1] = "This is not JSON."
        controller = await self.create()
        await self.update(controller, ("invalid-cohort",), 1)
        await self.until(lambda: controller.source_queue.metrics()["sources_censored"] == 1)
        self.assertEqual(self.fx.actor.clock, 1)
        self.assertEqual(len(self.fx.actor.update_calls), 1)
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(controller.ledger.sample_ready(100), [])
        self.assertEqual(len(list((controller.replay.root / "records").glob("*.json"))), 1)
        jobs = next(iter(controller.source_queue.state["jobs"].values()))["curator_jobs"]
        self.assertEqual({job["status"] for job in jobs}, {"semantic_rejected", "censored"})
        self.assertTrue(all(row["status"] == "censored" for row in controller.state["cohorts"].values()))

    async def test_invalid_field_type_has_no_retained_action_or_reward_label(self):
        self.fx.curate_bad_fields[0] = {"title": []}
        controller = await self.create()
        await self.update(controller, ("invalid-field",), 1)
        await self.until(lambda: controller.source_queue.metrics()["sources_censored"] == 1)
        self.assertEqual(len(self.fx.curate_calls), 1)
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(list((controller.replay.root / "records").glob("*.json")), [])

    async def test_proved_tool_invocation_censors_before_sampling_another_candidate(self):
        self.fx.config.skill_harness.joint.candidates_per_source = 2
        self.fx.curate_tool_indices.add(0)
        controller = await self.create()
        await self.update(controller, ("unexpected-tool",), 1)
        await self.until(lambda: controller.source_queue.metrics()["sources_censored"] == 1)
        self.assertEqual(len(self.fx.curate_calls), 1)
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(list((controller.replay.root / "records").glob("*.json")), [])

    async def test_oversize_whole_source_is_censored_without_truncation_or_sampling(self):
        controller = await self.create()
        data = self.fx.solver(("oversize-source",))
        path = next((self.fx.root / "episodes").glob("*/*/initial-state.json"))
        path.write_text(json.dumps({"observation": "Long observed feedback. " * 3000}))
        original = path.read_bytes()
        await self.fx.update(controller, data, 1)
        await controller.leave_learner()
        await self.until(lambda: controller.source_queue.metrics()["sources_censored"] == 1)
        self.assertEqual(self.fx.curate_calls, [])
        job = next(iter(controller.source_queue.state["jobs"].values()))["curator_jobs"][0]
        policy = job["input_policy"]
        self.assertGreater(policy["host_prompt_tokens"], policy["host_prompt_budget"])
        self.assertEqual(policy["host_prompt_budget"], 36864 - 4096 - 2048)
        self.assertFalse(policy["source_truncated"])
        self.assertEqual(path.read_bytes(), original)

    async def test_zero_action_source_is_censored_without_sampling_or_invented_credit(self):
        controller = await self.create()
        data = self.fx.solver(("zero-action",))
        for directory in (self.fx.root / "episodes").glob("*/*"):
            for name in ("result.json", "engine-result.json"):
                path = directory / name
                result = json.loads(path.read_text())
                result.update(steps=0, won=False, done=False, reward=0.0, accuracy=0.0)
                path.write_text(json.dumps(result))
            (directory / "initial-state.json").write_text(json.dumps({
                "observation": "No action was taken.", "steps": 0, "max_steps": 50, "won": False, "done": False,
            }))
            (directory / "environment-trace.jsonl").unlink()
        await self.fx.update(controller, data, 1)
        await controller.leave_learner()
        await self.until(lambda: controller.source_queue.metrics()["sources_censored"] == 1)
        self.assertEqual(self.fx.curate_calls, [])
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(controller.state["collections"], {})
        self.assertEqual(list((controller.replay.root / "records").glob("*.json")), [])
        entry = next(iter(controller.source_queue.state["jobs"].values()))
        self.assertEqual(entry["details"]["reason"], "source_has_no_environment_actions")
        self.assertFalse(entry["details"]["evidence"]["trace_present"])
        self.assertEqual(controller.ledger.sample_ready(100), [])

    async def assert_failed_source(self, mutate, error_fragment):
        controller = await self.create()
        data = self.fx.solver(("failed-source",))
        mutate(controller)
        await self.fx.update(controller, data, 1)
        await controller.leave_learner()
        await self.until(lambda: controller.source_queue.metrics()["sources_failed"] == 1)
        self.assertEqual(controller.state["optimizer_step"], 1)
        self.assertIsNone(controller.state["prepared_update"])
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(controller.ledger.sample_ready(100), [])
        failed = next(iter(controller.source_queue.state["jobs"].values()))
        self.assertIn(error_fragment, failed["details"]["message"])
        return controller

    async def test_missing_nonzero_source_trace_is_local_failure_after_the_single_update(self):
        def mutate(_controller):
            for path in (self.fx.root / "episodes").glob("*/*/environment-trace.jsonl"):
                path.unlink()
        await self.assert_failed_source(mutate, "environment-trace")
        self.assertEqual(self.fx.curate_calls, [])

    async def test_source_identity_mismatch_does_not_create_or_zero_a_label(self):
        def mutate(_controller):
            for path in (self.fx.root / "episodes").glob("*/*/engine-result.json"):
                value = json.loads(path.read_text())
                value["game_sha256"] = "f" * 64
                path.write_text(json.dumps(value))
        await self.assert_failed_source(mutate, "authoritative engine result")
        self.assertEqual(self.fx.curate_calls, [])

    async def test_behavior_version_failure_is_local_and_preserves_original_sample_probabilities(self):
        def mutate(controller):
            native = controller._collect_curate
            async def broken(**kwargs):
                kwargs["gateway_manager"].client.optimizer_step += 1
                return await native(**kwargs)
            controller._collect_curate = broken
        controller = await self.assert_failed_source(mutate, "weight versions")
        self.assertEqual(list((controller.replay.root / "records").glob("*.json")), [])
        self.assertEqual(len(self.fx.actor.update_calls), 1)

    async def test_replay_append_failure_is_local_and_never_retries_the_optimizer(self):
        def mutate(controller):
            controller.replay.append = lambda *_args: (_ for _ in ()).throw(OSError("durable append failed"))
        controller = await self.assert_failed_source(mutate, "durable append failed")
        self.assertEqual(len(self.fx.actor.update_calls), 1)
        self.assertEqual(controller.state["pending"], {})

if __name__ == "__main__":
    unittest.main()
