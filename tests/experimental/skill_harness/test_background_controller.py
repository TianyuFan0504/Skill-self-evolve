"""Bounded CPU contracts for background source/credit work and single learner."""

from __future__ import annotations

import asyncio
import json
import unittest

import test_controller as fixture


class BackgroundControllerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fx = fixture.ControllerTest(
            methodName="test_first_step_solver_only_then_future_native_credit_enters_same_optimizer"
        )
        await self.fx.asyncSetUp()
        # Transfer only cleanup callbacks, not another TestCase/asyncio runner.
        for function, args, kwargs in self.fx._cleanups:
            self.addCleanup(function, *args, **kwargs)
        self.fx._cleanups.clear()
        self.controllers = []
        self.addAsyncCleanup(self.stop_all)
        self.fx.config.skill_harness.joint = {
            "background_enabled": True,
            "sources_per_step": 1,
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


if __name__ == "__main__":
    unittest.main()
