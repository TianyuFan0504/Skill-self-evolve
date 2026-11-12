"""CPU contract for the supported prompt hook after all sibling sessions settle."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_archive_framework_group_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


class NativeFramework:
    async def _run_prompt_rollouts(self, **kwargs):
        self.parent_entered.set()
        await self.parent_release.wait()
        self.parent_settled = True
        return self.parent_stats


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
    async def run_hook(self, *, failures, partition="train", joint=True, task="alfworld_skill_training"):
        instance = framework.SkillAwareAgentFramework()
        instance._skill_loading = {"enabled": False}
        instance._skill_store_root = None
        instance._skill_strict_train_groups = joint
        instance.parent_entered, instance.parent_release = asyncio.Event(), asyncio.Event()
        instance.parent_settled = False
        instance.parent_stats = {
            "num_success_sessions": 4 - failures, "num_failed_sessions": failures,
            "num_success_outputs": 4 - failures, "num_unfinished_episodes": 0,
            "num_failed_uids": int(failures == 4), "failure_reasons": ["archive failure"] * failures,
        }
        before = copy.deepcopy(instance.parent_stats)
        calls = []
        queue = types.ModuleType("transfer_queue")

        async def put(**kwargs):
            self.assertTrue(instance.parent_settled)
            calls.append(kwargs)

        queue.async_kv_put = put
        with patch.dict(sys.modules, {"transfer_queue": queue}):
            future = asyncio.create_task(instance._run_prompt_rollouts(
                sample_fields={"uid": "group-a", "tools_kwargs": {"task": {"name": task}}},
                sample_index=0, global_steps=1, partition_id=partition, num_sessions=4,
            ))
            await asyncio.wait_for(instance.parent_entered.wait(), timeout=2)
            self.assertEqual(calls, [])
            instance.parent_release.set()
            result = await asyncio.wait_for(future, timeout=2)
        self.assertEqual(instance.parent_stats, before)
        return result, calls

    async def test_partial_failed_training_group_is_marked_failure_after_all_siblings_settle(self):
        stats, calls = await self.run_hook(failures=1)
        self.assertEqual(calls, [{"key": "group-a", "partition_id": "train", "tag": {"status": "failure"}}])
        self.assertEqual(stats["num_failed_uids"], 1)
        self.assertEqual(stats["num_success_sessions"], 3)
        self.assertEqual(stats["num_failed_sessions"], 1)
        self.assertNotIn("reward", stats)

    async def test_healthy_training_group_is_unchanged(self):
        stats, calls = await self.run_hook(failures=0)
        self.assertEqual(calls, [])
        self.assertEqual(stats["num_failed_uids"], 0)
        self.assertEqual(stats["num_success_sessions"], 4)

    async def test_validation_baseline_and_other_tasks_delegate_unchanged(self):
        for options in ({"partition": "val"}, {"joint": False}, {"task": "other_task"}):
            with self.subTest(options=options):
                stats, calls = await self.run_hook(failures=1, **options)
                self.assertEqual(calls, [])
                self.assertEqual(stats["num_failed_uids"], 0)


if __name__ == "__main__":
    unittest.main()
