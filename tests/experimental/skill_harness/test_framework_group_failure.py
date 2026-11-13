"""Exercise group completion with real media validation and inherited hook seams."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_group_framework_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


class NativeFramework:
    async def _run_prompt_rollouts(self, **kwargs):
        self.delegated = True
        return {"native_delegate": True}

    def _build_session_sampling_params(self, **kwargs):
        return {"temperature": 1.0}

    async def _run_agent_episode_with_concurrency_limit(self, *, session_index, sample_fields, **kwargs):
        self.entered.add(session_index)
        if len(self.entered) == self.sessions:
            self.all_entered.set()
        await self.release.wait()
        self.settled.add(session_index)
        if session_index in self.agent_failures:
            raise RuntimeError("agent transport failed")
        trajectory = types.SimpleNamespace(
            finished=True, extra_fields={},
            multi_modal_data={"images": [b"image"]} if session_index in self.media_failures else None,
        )
        return [trajectory], sample_fields

    def _trajectory_to_tq_field_and_tag(self, **kwargs):
        return {"extra_fields": {}}, {}

    async def _write_session_trajectories_to_tq(self, *, session_index, trajectories, **kwargs):
        self.serializers.append(session_index)
        for trajectory in trajectories:
            self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory, session_index=session_index,
                sample_fields=kwargs["sample_fields"], uid=kwargs["uid"], global_steps=kwargs["global_steps"],
            )
        self.materialized.add(session_index)


entry = types.ModuleType("uni_agent.framework.entry")
entry.AgentFrameworkRolloutAdapter = type("Adapter", (), {})
upstream = types.ModuleType("uni_agent.framework.framework")
upstream.GatewayAgentFramework = NativeFramework
with patch.dict(sys.modules, {"uni_agent.framework.entry": entry, "uni_agent.framework.framework": upstream}):
    spec = importlib.util.spec_from_file_location(PACKAGE + ".framework", ROOT / "framework.py")
    framework = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = framework
    spec.loader.exec_module(framework)


class FrameworkGroupFailureTest(unittest.IsolatedAsyncioTestCase):
    async def run_hook(self, *, media_failures=(), agent_failures=(), partition="train", strict=True,
                       task="alfworld_skill_training"):
        instance = framework.SkillAwareAgentFramework()
        instance._skill_loading, instance._skill_store_root = {"enabled": False}, None
        instance._skill_strict_train_groups = strict
        instance.sessions, instance.entered, instance.settled = 2, set(), set()
        instance.all_entered, instance.release = asyncio.Event(), asyncio.Event()
        instance.media_failures, instance.agent_failures = set(media_failures), set(agent_failures)
        instance.materialized, instance.serializers = set(), []
        instance.delegated = False
        calls = []
        queue = types.ModuleType("transfer_queue")

        async def put(**kwargs):
            if kwargs["tag"]["status"] != "running":
                self.assertEqual(instance.settled, {0, 1})
            calls.append(kwargs)

        queue.async_kv_put = put
        with patch.dict(sys.modules, {"transfer_queue": queue}):
            future = asyncio.create_task(instance._run_prompt_rollouts(
                sample_fields={"uid": "group-a", "tools_kwargs": {"task": {"name": task}}},
                sample_index=0, global_steps=1, partition_id=partition, num_sessions=2,
            ))
            if strict and partition == "train" and task == "alfworld_skill_training":
                await instance.all_entered.wait()
                self.assertEqual([call["tag"]["status"] for call in calls], ["running"])
                instance.release.set()
            result = await future
        return instance, result, [call["tag"]["status"] for call in calls]

    async def test_media_failure_after_valid_sibling_never_publishes_finished(self):
        instance, stats, statuses = await self.run_hook(media_failures={1})
        self.assertEqual(instance.serializers, [0, 1])
        self.assertEqual(instance.materialized, {0})
        self.assertEqual(statuses, ["running", "failure"])
        self.assertEqual(stats["num_failed_uids"], 1)
        self.assertEqual(stats["num_success_sessions"], 1)
        self.assertEqual(stats["num_failed_sessions"], 1)
        self.assertIn("text-only", stats["failure_reasons"][0])

    async def test_transport_failure_refuses_partial_group(self):
        instance, stats, statuses = await self.run_hook(agent_failures={0})
        self.assertEqual(statuses, ["running", "failure"])
        self.assertEqual(instance.materialized, {1})
        self.assertEqual(stats["num_success_outputs"], 1)
        self.assertEqual(stats["num_failed_sessions"], 1)

    async def test_complete_group_publishes_finished_once(self):
        instance, stats, statuses = await self.run_hook()
        self.assertEqual(statuses, ["running", "finished"])
        self.assertEqual(instance.materialized, {0, 1})
        self.assertEqual(stats["num_failed_uids"], 0)
        self.assertEqual(stats["num_success_sessions"], 2)

    async def test_validation_and_other_tasks_delegate_unchanged(self):
        for options in ({"partition": "val"}, {"strict": False}, {"task": "other_task"}):
            with self.subTest(options=options):
                instance, stats, statuses = await self.run_hook(**options)
                self.assertTrue(instance.delegated)
                self.assertEqual(stats, {"native_delegate": True})
                self.assertEqual(statuses, [])


if __name__ == "__main__":
    unittest.main()
