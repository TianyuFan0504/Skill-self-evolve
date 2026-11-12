"""Exercise the actual core sampler loop with an in-memory TQ transport.

The sync non-DAPO path performs no NumPy operations. Its optional dependency
imports and SkipManager instrumentation are stubbed; eviction/refill/sample
remain the unmodified verl ReplayBuffer implementation.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


class BatchMeta:
    def __init__(self, *, partition_id, keys, tags):
        self.partition_id, self.keys, self.tags = partition_id, keys, tags


class Queue(types.ModuleType):
    def __init__(self):
        super().__init__("transfer_queue")
        self.KVBatchMeta = BatchMeta
        self.reset()

    def reset(self):
        self.metadata, self.values, self.created_values, self.clears = {}, {}, {}, []

    def group(self, uid, *, status="finished", rewards=(0.0, 1.0, 0.0, 1.0)):
        tags = self.metadata.setdefault("train", {})
        tags[uid] = {"is_prompt": True, "status": status, "global_steps": 1}
        for session, reward in enumerate(rewards):
            if reward is None:
                continue  # An unobserved sibling is absent, never reward zero.
            key = f"{uid}_{session}_0"
            tags[key] = {"global_steps": 1}
            self.values[key] = {"reward": reward}
            self.created_values[key] = copy.deepcopy(self.values[key])

    def kv_list(self):
        return copy.deepcopy(self.metadata)

    def kv_clear(self, *, partition_id, keys):
        self.clears.append((partition_id, set(keys)))
        for key in keys:
            self.metadata.get(partition_id, {}).pop(key, None)
            self.values.pop(key, None)


QUEUE = Queue()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


optional_numpy = types.ModuleType("numpy")
optional_omega = types.ModuleType("omegaconf")
optional_omega.DictConfig = dict
skip = types.ModuleType("verl.utils.skip")
skip.SkipManager = types.SimpleNamespace(annotate_tq=lambda **_kwargs: lambda method: method)
package = types.ModuleType("_archive_sampler_contract")
package.__path__ = [str(ROOT / "verl/experimental/skill_harness")]
sys.modules[package.__name__] = package
accounts = load_module(
    "_actual_archive_sampler_accounts", ROOT / "verl/experimental/skill_harness/alfworld/accounts.py"
)
import_utils = load_module("_actual_verl_archive_import_utils", ROOT / "verl/utils/import_utils.py")
with patch.dict(sys.modules, {"numpy": optional_numpy, "omegaconf": optional_omega,
                              "transfer_queue": QUEUE, "verl.utils.skip": skip}):
    core = load_module("_actual_core_archive_replay_buffer", ROOT / "verl/trainer/ppo/v1/replay_buffer.py")
    with patch.dict(sys.modules, {
        "verl.trainer.ppo.v1.replay_buffer": core,
        "verl.experimental.skill_harness.alfworld.accounts": accounts,
    }):
        sampler = load_module(package.__name__ + ".group_sampler",
                              ROOT / "verl/experimental/skill_harness/group_sampler.py")


class GroupSamplerTest(unittest.TestCase):
    def setUp(self):
        QUEUE.reset()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run_root = str(Path(temporary.name).resolve())
        self.environment = patch.dict(os.environ, {"SKILL_RUN_DIR": self.run_root})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.archive_failed = set()
        self.registry = patch.object(sampler, "local_archive_failed_groups", side_effect=self.lookup)
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.refills = []

    def lookup(self, run_root):
        self.assertEqual(run_root, self.run_root)
        return set(self.archive_failed)

    def refill(self, count):
        self.refills.append(count)
        for index in range(count):
            QUEUE.group(f"replacement-{len(self.refills)}-{index}")

    def buffer(self, **overrides):
        kwargs = {
            "trainer_mode": "sync", "trainer_config": {"train_batch_size": 2},
            "max_off_policy_threshold": None, "max_off_policy_strategy": "drop",
            "sampler_kwargs": {}, "refill_fn": self.refill, "poll_interval": 0.001,
        }
        kwargs.update(overrides)
        return sampler.ArchiveAwareReplayBuffer(**kwargs)

    def test_partial_failed_group_and_registry_overlap_are_evicted_and_refilled_once(self):
        QUEUE.group("healthy", rewards=(0.0, 0.0, 0.0, 0.0))
        QUEUE.group("broken", status="failure", rewards=(1.0, 0.0, None, 1.0))
        self.archive_failed.add("broken")
        original = copy.deepcopy(QUEUE.created_values)
        batch, metrics = self.buffer().sample(global_steps=1, partition_id="train", batch_size=2)
        self.assertEqual({key.split("_")[0] for key in batch.keys}, {"healthy", "replacement-1-0"})
        self.assertEqual(self.refills, [1])
        self.assertEqual(metrics["training/rollout_failure/evicted_samples"], 1)
        removed_broken = [keys for _, keys in QUEUE.clears if "broken" in keys]
        self.assertEqual(removed_broken, [{"broken", "broken_0_0", "broken_1_0", "broken_3_0"}])
        self.assertNotIn("broken_2_0", QUEUE.created_values)
        self.assertTrue(all(QUEUE.created_values[key] == value for key, value in original.items()))

    def test_preexception_registry_rejects_parent_provisional_finished_group(self):
        QUEUE.group("healthy")
        QUEUE.group("broken", status="finished", rewards=(1.0, None, 0.0, 1.0))
        self.archive_failed.add("broken")
        batch, _ = self.buffer().sample(global_steps=1, partition_id="train", batch_size=2)
        self.assertNotIn("broken", {key.split("_")[0] for key in batch.keys})
        self.assertEqual(self.refills, [1])

    def test_marker_does_not_evict_running_siblings_before_terminal_status(self):
        QUEUE.group("broken", status="running", rewards=(1.0, None, 0.0, None))
        self.archive_failed.add("broken")
        buffer = self.buffer()
        buffer._sync_metadata_from_transfer_queue()
        reasons = buffer._terminal_eviction_reasons(global_steps=1, partition_id="train")
        self.assertEqual(reasons[2], set())
        self.assertEqual(buffer._sampleable_terminal_keys("train", reasons), set())
        self.assertEqual(QUEUE.clears, [])
        self.assertEqual(self.refills, [])

    def test_healthy_zero_reward_group_is_retained_without_refill(self):
        QUEUE.group("healthy", rewards=(0.0, 0.0, 0.0, 0.0))
        batch, metrics = self.buffer().sample(global_steps=1, partition_id="train", batch_size=1)
        self.assertEqual(set(batch.keys), {f"healthy_{index}_0" for index in range(4)})
        self.assertEqual(self.refills, [])
        self.assertEqual(metrics, {})

    def test_validation_keeps_stock_policy_and_does_not_read_training_registry(self):
        buffer = self.buffer()
        buffer.failure_keys["val"].add("failed-validation")
        with patch.object(sampler, "local_archive_failed_groups", side_effect=AssertionError("train registry")):
            self.assertEqual(buffer._terminal_eviction_reasons(1, "val")[2], set())

    def test_supported_constructor_sets_exact_refill_granularity(self):
        buffer = self.buffer()
        self.assertEqual(buffer.train_batch_size, 2)
        self.assertEqual(buffer.gen_batch_size, 1)
        self.assertTrue(buffer.sync_refill_failed_groups)
        with self.assertRaisesRegex(ValueError, "train_batch_size"):
            self.buffer(trainer_config={})
        with self.assertRaisesRegex(ValueError, "synchronous"):
            self.buffer(trainer_mode="separate_async")
        with self.assertRaisesRegex(ValueError, "gen_batch_size"):
            self.buffer(gen_batch_size=2)

    def test_real_official_file_loader_loads_sampler_and_constructs_it(self):
        # Run the actual production loader, including its anonymous file-module
        # namespace. Only TQ transport/dependency boundaries remain replaced.
        with patch.dict(sys.modules, {
            "verl.trainer.ppo.v1.replay_buffer": core,
            "verl.experimental.skill_harness.alfworld.accounts": accounts,
        }):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                loaded = import_utils.load_extern_type(
                    str(ROOT / "verl/experimental/skill_harness/group_sampler.py"), "ArchiveAwareReplayBuffer"
                )
            self.assertTrue(loaded.__module__.startswith("custom_module_"))
            instance = loaded(
                trainer_mode="sync", trainer_config={"train_batch_size": 2},
                max_off_policy_threshold=None, max_off_policy_strategy="drop",
                sampler_kwargs={}, refill_fn=self.refill,
            )
        self.assertIsInstance(instance, core.ReplayBuffer)
        self.assertEqual(instance.train_batch_size, 2)
        self.assertEqual(instance.gen_batch_size, 1)
        self.assertTrue(instance.sync_refill_failed_groups)
        self.assertEqual(instance.archive_run_root, self.run_root)

    def test_real_official_loader_rejects_a_bare_dotted_path(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with self.assertRaisesRegex(FileNotFoundError, "Custom module file not found"):
                import_utils.load_extern_type(
                    "verl.experimental.skill_harness.group_sampler", "ArchiveAwareReplayBuffer"
                )


if __name__ == "__main__":
    unittest.main()
