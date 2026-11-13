"""Exercise the real TensorDict non-tensor representation used by live TQ."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

try:
    importlib.import_module("uni_agent")
except ModuleNotFoundError:
    entry, framework = ModuleType("uni_agent.framework.entry"), ModuleType("uni_agent.framework.framework")
    entry.AgentFrameworkRolloutAdapter, framework.GatewayAgentFramework = object, object
    sys.modules.update({entry.__name__: entry, framework.__name__: framework})
    try:
        importlib.import_module("verl.experimental.skill_harness.framework")
    finally:
        sys.modules.pop(entry.__name__)
        sys.modules.pop(framework.__name__)

import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from tensordict.tensorclass import NonTensorData, NonTensorStack  # noqa: E402

from verl.experimental.skill_harness.joint_trainer import SkillJointSyncTrainer  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402


class EpisodeMetricsAdapterTest(unittest.TestCase):
    def test_linked_list_fields_use_supported_unwrapping_and_deduplicate_sessions(self):
        rows = [
            {"uid": "a", "session_id": 0, "reward_extra_info": {"acc": 1.0}},
            {"uid": "a", "session_id": 0, "reward_extra_info": {"acc": 1.0}},
            {"uid": "b", "session_id": 0, "reward_extra_info": {"acc": 0.0}},
        ]
        data = TensorDict(
            {
                "uid": NonTensorStack.from_list([NonTensorData(row["uid"]) for row in rows]),
                "session_id": NonTensorStack.from_list([NonTensorData(row["session_id"]) for row in rows]),
                "extra_fields": NonTensorStack.from_list(
                    [NonTensorData({"reward_extra_info": row["reward_extra_info"]}) for row in rows]
                ),
                "rm_scores": tu.nested_tensor_from_tensor_list(
                    [torch.tensor([0.0, 1.0]), torch.tensor([1.0]), torch.tensor([0.0, 0.0, 0.0])]
                ),
            },
            batch_size=[3],
        )
        # Current TensorDict indexing exposes LinkedList, while .get retains
        # the NonTensorStack wrapper understood by the supported verl helper.
        self.assertEqual(type(data["uid"]).__name__, "LinkedList")
        with tempfile.TemporaryDirectory() as directory:
            trainer = object.__new__(SkillJointSyncTrainer)
            trainer.artifact_root = Path(directory)
            batch = SimpleNamespace(keys=["a_0_0", "a_0_1", "b_0_0"], partition_id="train", tags=[{}, {}, {}])
            metrics = {"actor/grad_norm": 1.25, "actor/pg_loss": 0.0}
            with (
                patch("verl.experimental.skill_harness.joint_trainer.PPOTrainerSync._compute_metrics"),
                patch("verl.experimental.skill_harness.joint_trainer.tq.kv_batch_get", return_value=data),
            ):
                trainer._compute_metrics(batch, metrics, {}, global_steps=1, epoch=0)
            self.assertEqual(metrics["train/episodes"], 2)
            self.assertEqual(metrics["train/reward"], 0.5)
            self.assertEqual(metrics["train/accuracy"], 0.5)
            saved = json.loads((Path(directory) / "metrics.jsonl").read_text())
            self.assertEqual(saved["metrics"]["actor/grad_norm"], 1.25)


if __name__ == "__main__":
    unittest.main()
