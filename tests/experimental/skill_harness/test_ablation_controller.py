"""Experience and optimizer clocks stay distinct in the two ablations."""

from __future__ import annotations

import asyncio
import unittest

import test_controller as fixture


class AblationControllerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fx = fixture.ControllerFixture()
        await self.fx.asyncSetUp()
        for function, args, kwargs in self.fx._cleanups:
            self.addCleanup(function, *args, **kwargs)
        self.fx._cleanups.clear()
        self.controllers = []
        self.addAsyncCleanup(self.stop_all)
        self.fx.config.skill_harness.joint = {
            "candidates_per_source": 1,
            "curator_concurrency": 2,
            "credit_concurrency": 2,
            "required_exposures": 1,
            "min_delay_steps": 2,
            "probes_per_step": 4,
        }

    async def stop_all(self):
        for controller in reversed(self.controllers):
            await controller.stop_background()

    async def until(self, predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(poll(), timeout=3)

    async def create(self):
        controller = await self.fx.create()
        self.controllers.append(controller)
        return controller

    async def test_skill_only_generates_and_admits_at_clock_zero_without_replay_updates(self):
        self.fx.config.skill_harness.learning = {"update_parameters": False, "update_skills": True}
        controller = await self.create()
        first = await controller.advance_experience(self.fx.solver(("producer",)), 1)
        self.assertEqual(first["optimizer_step"], 0)
        await self.until(lambda: len(controller.state["collections"]) == 1)
        record_id = next(iter(controller.state["collections"].values()))["record_id"]
        self.assertEqual(controller.replay.get(record_id).behavior_policy_version, "ema:0")
        hold = asyncio.Event()

        async def hold_later(**_kwargs):
            await hold.wait()

        controller._collect_curate = hold_later
        await controller.advance_experience(self.fx.solver(("future2",)), 2)
        self.assertEqual(self.fx.probe_calls, [])
        third = await controller.advance_experience(self.fx.solver(("future3",)), 3)
        await self.until(lambda: any(row["status"] == "settled" for row in controller.state["pending"].values()))
        self.assertTrue(controller.store.get_snapshot().skills)
        self.assertEqual(self.fx.actor.clock, 0)
        self.assertEqual(self.fx.actor.update_calls, [])
        self.assertEqual(self.fx.actor.anchor_calls, [])
        self.assertEqual(controller.routes["curator"].syncs, [0])
        self.assertEqual(controller.state["experience_round"], 3)
        self.assertEqual(controller.state["last_global_step"], 3)
        self.assertEqual(controller.state["applied_update_ids"], {})
        self.assertIsNone(controller.state["prepared_update"])
        label = controller.ledger.sample_ready(1)[0]
        self.assertEqual(label.utility, 1)
        self.assertEqual(controller.ledger.training_status(label.settlement_id)["state"], "ready")
        self.assertEqual(third["optimizer_step"], 0)
        with self.assertRaisesRegex(RuntimeError, "Frozen parameters"):
            await controller.prepare_update(self.fx.solver(("invalid-learner",)), 4)

    async def test_parameter_only_updates_without_auxiliary_models_or_skill_changes(self):
        self.fx.config.skill_harness.learning = {"update_parameters": True, "update_skills": False}
        self.fx.config.skill_harness.loading.enabled = False
        self.fx.config.skill_harness.joint.update(curator_gpus=0, credit_gpus=0, available_gpus=4)

        async def no_routes(*_args, **_kwargs):
            self.fail("A parameter-only controller must not allocate auxiliary policy models")

        self.fx.routes = no_routes
        controller = await self.create()
        initial = controller.store.head()
        for step in (1, 2):
            await self.fx.update(controller, self.fx.solver(("task-" + str(step),)), step)
        self.assertEqual(self.fx.actor.clock, 2)
        self.assertEqual(len(self.fx.actor.update_calls), 2)
        self.assertEqual(controller.state["experience_round"], 2)
        self.assertEqual(controller.store.head(), initial)
        self.assertEqual(controller.store.get_snapshot().skills, ())
        self.assertEqual(controller.routes, {})
        self.assertIsNone(controller.ema_gateway)
        self.assertIsNone(controller.frozen_gateway)
        self.assertIsNone(controller._background_pump)
        self.assertEqual(controller.source_queue.state["jobs"], {})
        self.assertEqual(controller.state["probe_jobs"], {})
        self.assertEqual(self.fx.curate_calls, [])
        self.assertEqual(self.fx.probe_calls, [])

    async def test_delayed_curation_opens_its_window_at_completion_round(self):
        self.fx.config.skill_harness.learning = {"update_parameters": False, "update_skills": True}
        controller = await self.create()
        entered, release = asyncio.Event(), asyncio.Event()
        native = self.fx.native_curate

        async def delayed(**kwargs):
            entered.set()
            await release.wait()
            return await native(**kwargs)

        controller._collect_curate = delayed
        await controller.advance_experience(self.fx.solver(("slow-source",)), 1)
        await entered.wait()
        await controller.advance_experience(self.fx.solver(("second-round",)), 2)
        release.set()
        source_id = self.fx.metadata("slow-source")["game_id"]
        await self.until(
            lambda: any(row["producer_task_id"] == source_id for row in controller.state["pending"].values())
        )
        pending = next(row for row in controller.state["pending"].values() if row["producer_task_id"] == source_id)
        self.assertEqual(pending["opened_step"], 2)
        await controller.advance_experience(self.fx.solver(("third-round",)), 3)
        self.assertEqual(self.fx.probe_calls, [])
        await controller.advance_experience(self.fx.solver(("fourth-round",)), 4)
        await self.until(lambda: pending["status"] == "settled")
        self.assertEqual(self.fx.actor.clock, 0)

    async def test_declared_default_learning_mode_preserves_protocol_identity(self):
        original = fixture.controller_module.preflight_controller_identity(
            self.fx.config, native_runtime=self.fx.runtime
        )
        self.fx.config.skill_harness.learning = {"update_parameters": True, "update_skills": True}
        explicit = fixture.controller_module.preflight_controller_identity(
            self.fx.config, native_runtime=self.fx.runtime
        )
        self.assertEqual(original.protocol_fingerprint, explicit.protocol_fingerprint)
        self.fx.config.skill_harness.learning.update_parameters = False
        frozen = fixture.controller_module.preflight_controller_identity(self.fx.config, native_runtime=self.fx.runtime)
        self.assertNotEqual(original.protocol_fingerprint, frozen.protocol_fingerprint)

    async def test_frozen_round_rejects_optimizer_clock_change(self):
        self.fx.config.skill_harness.learning = {"update_parameters": False, "update_skills": True}
        controller = await self.create()
        self.fx.actor.clock = 1
        with self.assertRaisesRegex(fixture.controller_module.ControllerRecoveryRequired, "clock"):
            await controller.advance_experience(self.fx.solver(("unexpected-update",)), 1)


if __name__ == "__main__":
    unittest.main()
