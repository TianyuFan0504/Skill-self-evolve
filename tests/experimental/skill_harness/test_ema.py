# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU-only numerical and lifecycle tests; no Ray workers or process groups."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = ROOT / "verl" / "experimental" / "skill_harness"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if torch is not None:
    ema_module = load_module("_skill_ema_test", MODULE_ROOT / "ema.py")
    ParameterEMA = ema_module.ParameterEMA


@unittest.skipIf(torch is None, "CPU PyTorch is required")
class ParameterEMATest(unittest.TestCase):
    def setUp(self):
        self.parameters = {"weight": torch.nn.Parameter(torch.tensor([2.0, -2.0]))}
        self.buffers = {"counter": torch.tensor(1, dtype=torch.int64), "running": torch.tensor([3.0])}
        self.ema = ParameterEMA()
        self.ema.initialize(self.parameters, self.buffers, optimizer_step=7)

    def test_initial_identity_fp32_and_no_alias_or_autograd(self):
        torch.testing.assert_close(self.ema.parameters["weight"], self.parameters["weight"])
        self.assertFalse(self.ema.parameters["weight"].requires_grad)
        self.assertIsNone(self.ema.parameters["weight"].grad_fn)
        self.assertNotEqual(self.ema.parameters["weight"].data_ptr(), self.parameters["weight"].data_ptr())
        half = {"half": torch.tensor([1.5], dtype=torch.bfloat16, requires_grad=True)}
        tracker = ParameterEMA()
        tracker.initialize(half)
        self.assertEqual(tracker.parameters["half"].dtype, torch.float32)
        self.assertEqual(tracker.parameters["half"].item(), 1.5)

    def test_numerical_update_and_buffers_are_copied(self):
        with torch.no_grad():
            self.parameters["weight"].copy_(torch.tensor([4.0, 2.0]))
            self.buffers["counter"].fill_(5)
            self.buffers["running"].fill_(9)
        result = self.ema.update(self.parameters, self.buffers, optimizer_step=8)
        torch.testing.assert_close(self.ema.parameters["weight"], torch.tensor([2.1, -1.8]))
        self.assertEqual(self.ema.buffers["counter"].item(), 5)
        self.assertEqual(self.ema.buffers["counter"].dtype, torch.int64)
        self.assertEqual(self.ema.buffers["running"].item(), 9)
        self.assertEqual(result["num_updates"], 1)
        self.assertEqual(result["last_optimizer_step"], 8)
        self.assertIsNone(self.parameters["weight"].grad)

    def test_excluded_dynamic_buffers_are_never_swapped(self):
        tracker = ParameterEMA(buffer_policy="exclude")
        tracker.initialize(self.parameters, self.buffers)
        self.buffers["new_cache"] = torch.ones(5)
        tracker.update(self.parameters, self.buffers, optimizer_step=1)
        with tracker.swap_into(self.parameters, self.buffers):
            self.assertEqual(self.buffers["counter"].item(), 1)
        self.assertEqual(tracker.buffers, {})

    def test_invalid_clock_layout_and_nonfinite_updates_are_atomic(self):
        original = self.ema.state_dict()
        for step in (7, 6, 9, True, -1):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.ema.update(self.parameters, self.buffers, optimizer_step=step)
        candidates = (
            {"renamed": torch.ones(2)},
            {"weight": torch.ones(3)},
            {"weight": torch.ones(2, dtype=torch.float64)},
            {"weight": torch.tensor([float("nan"), 1.0])},
        )
        for candidate in candidates:
            with self.assertRaises(ValueError):
                self.ema.update(candidate, self.buffers, optimizer_step=8)
        self.assertEqual(self.ema.last_optimizer_step, 7)
        torch.testing.assert_close(self.ema.parameters["weight"], original["parameters"]["weight"])

    def test_serialization_roundtrip_resume_and_snapshot_aliases(self):
        self.parameters["weight"].data.fill_(4)
        self.ema.update(self.parameters, self.buffers, optimizer_step=8)
        state = self.ema.state_dict()
        state["parameters"]["weight"].zero_()
        state["parameter_layouts"]["weight"]["local_shape"][0] = 99
        self.assertNotEqual(self.ema.parameters["weight"][0].item(), 0)
        self.assertEqual(self.ema.parameter_layouts["weight"]["local_shape"], [2])
        output = io.BytesIO()
        torch.save(self.ema.state_dict(), output)
        output.seek(0)
        saved = torch.load(output, map_location="cpu", weights_only=True)
        resumed = ParameterEMA()
        resumed.load_state_dict(saved, parameters=self.parameters, buffers=self.buffers)
        resumed.update(self.parameters, self.buffers, optimizer_step=9)
        self.ema.update(self.parameters, self.buffers, optimizer_step=9)
        torch.testing.assert_close(resumed.parameters["weight"], self.ema.parameters["weight"])
        self.assertEqual(resumed.metadata(), self.ema.metadata())
        saved["parameters"]["weight"].zero_()
        self.assertNotEqual(resumed.parameters["weight"][0].item(), 0)

    def test_reject_incompatible_or_corrupted_checkpoint(self):
        state = self.ema.state_dict()
        for field, value in (("schema_version", 3), ("decay", 0.8), ("num_updates", 1)):
            bad = copy.deepcopy(state)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ParameterEMA().load_state_dict(bad, parameters=self.parameters, buffers=self.buffers)
        bad = copy.deepcopy(state)
        bad["parameters"]["weight"] = torch.ones(3)
        with self.assertRaises(ValueError):
            ParameterEMA().load_state_dict(bad, parameters=self.parameters, buffers=self.buffers)

    def test_swap_restores_online_even_when_body_raises(self):
        self.parameters["weight"].data.fill_(10)
        self.buffers["counter"].fill_(20)
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with self.ema.swap_into(self.parameters, self.buffers):
                torch.testing.assert_close(self.parameters["weight"], torch.tensor([2.0, -2.0]))
                self.assertEqual(self.buffers["counter"].item(), 1)
                with self.assertRaises(RuntimeError):
                    self.ema.update(self.parameters, self.buffers, optimizer_step=8)
                with self.assertRaises(RuntimeError):
                    self.ema.state_dict()
                with self.assertRaises(RuntimeError):
                    with self.ema.swap_into(self.parameters, self.buffers):
                        pass
                raise RuntimeError("body failed")
        torch.testing.assert_close(self.parameters["weight"], torch.tensor([10.0, 10.0]))
        self.assertEqual(self.buffers["counter"].item(), 20)
        self.assertFalse(self.ema.swapped)

    def test_swap_reresolves_replaced_live_parameter_storage(self):
        self.parameters["weight"].data.fill_(10)
        with self.ema.swap_into(lambda: self.parameters, lambda: self.buffers):
            self.parameters["weight"] = torch.nn.Parameter(self.parameters["weight"].detach().clone())
        torch.testing.assert_close(self.parameters["weight"], torch.tensor([10.0, 10.0]))

    def test_local_shard_never_requests_a_full_tensor(self):
        class Mesh:
            mesh = torch.empty(2)
            device_type = "cpu"

            @staticmethod
            def get_coordinate():
                return [0]

        class LocalShard(torch.Tensor):
            @staticmethod
            def __new__(cls):
                return torch.Tensor._make_subclass(cls, torch.zeros(4), False)

            def __init__(self):
                self.local_value = torch.tensor([1.0, 2.0])
                self.device_mesh = Mesh()
                self.placements = ["Shard(0)"]

            def to_local(self):
                return self.local_value

            def full_tensor(self):
                raise AssertionError("EMA must not gather a global tensor")

        shard = LocalShard()
        tracker = ParameterEMA(decay=0.5)
        tracker.initialize({"shard": shard})
        shard.local_value.add_(2)
        tracker.update({"shard": shard}, optimizer_step=1)
        torch.testing.assert_close(tracker.parameters["shard"], torch.tensor([2.0, 3.0]))
        self.assertEqual(tracker.parameter_layouts["shard"]["local_shape"], [2])
        self.assertEqual(tracker.parameter_layouts["shard"]["global_shape"], [4])
        shard.placements = ["Replicate()"]
        with self.assertRaises(ValueError):
            tracker.update({"shard": shard}, optimizer_step=2)

    def test_complete_lazy_async_export_keeps_ema_until_send_finishes(self):
        self.parameters["weight"].data.fill_(10)
        events = []

        def export():
            events.append("factory")
            torch.testing.assert_close(self.parameters["weight"], torch.tensor([2.0, -2.0]))

            def stream():
                try:
                    yield ("weight", self.parameters["weight"].detach().clone())
                finally:
                    events.append("closed")

            return stream()

        async def send(iterator):
            await asyncio.sleep(0)
            values = list(iterator)
            await asyncio.sleep(0)
            self.assertTrue(self.ema.swapped)
            torch.testing.assert_close(self.parameters["weight"], torch.tensor([2.0, -2.0]))
            torch.testing.assert_close(values[0][1], torch.tensor([2.0, -2.0]))
            events.append("sent")
            return {"tensors": len(values)}

        result = asyncio.run(self.ema.consume_export(self.parameters, export, send, self.buffers))
        self.assertEqual(result, {"tensors": 1})
        self.assertEqual(events, ["factory", "closed", "sent"])
        torch.testing.assert_close(self.parameters["weight"], torch.tensor([10.0, 10.0]))

    def test_partial_export_and_cancellation_restore_online(self):
        self.parameters["weight"].data.fill_(10)
        closed = []

        def export():
            try:
                yield "one", self.parameters["weight"]
                yield "two", self.parameters["weight"]
            finally:
                closed.append(True)

        async def partial(iterator):
            next(iterator)

        with self.assertRaisesRegex(RuntimeError, "complete lazy export"):
            asyncio.run(self.ema.consume_export(self.parameters, export, partial, self.buffers))
        self.assertEqual(closed, [True])
        self.assertFalse(self.ema.swapped)

        async def cancelled(iterator):
            next(iterator)
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(self.ema.consume_export(self.parameters, export, cancelled, self.buffers))
        self.assertEqual(closed, [True, True])
        torch.testing.assert_close(self.parameters["weight"], torch.tensor([10.0, 10.0]))
        self.assertFalse(self.ema.swapped)

    def test_exact_one_optimizer_step_guard(self):
        guard = ema_module.require_single_optimizer_step
        guard(4, 2, mini_batch_size=8)
        guard(4, 2, num_mini_batch=1)
        for kwargs in ({}, {"mini_batch_size": 4}, {"num_mini_batch": 2}, {"epochs": 2, "num_mini_batch": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                guard(4, 2, **kwargs)

    def test_suite_never_initialized_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


@unittest.skipIf(torch is None, "CPU PyTorch is required")
class WorkerObserverTest(unittest.TestCase):
    """Use a CPU engine double around the real worker extension and optimizer."""

    @classmethod
    def setUpClass(cls):
        modules = {}
        for name in (
            "verl", "verl.checkpoint_engine", "verl.single_controller", "verl.single_controller.base",
            "verl.single_controller.base.decorator", "verl.utils", "verl.utils.tensordict_utils",
            "verl.workers", "verl.workers.engine_workers", "_ema_worker_test_package",
        ):
            modules[name] = types.ModuleType(name)
        modules["_ema_worker_test_package"].__path__ = [str(MODULE_ROOT)]
        modules["_ema_worker_test_package.ema"] = ema_module
        modules["verl.checkpoint_engine"].CheckpointEngineRegistry = object
        decorator = modules["verl.single_controller.base.decorator"]
        decorator.Dispatch = types.SimpleNamespace(ONE_TO_ALL="all", DP_COMPUTE="per_rank")
        decorator.register = lambda **_kwargs: lambda method: method
        decorator.make_nd_compute_dataproto_dispatch_fn = lambda **_kwargs: "train"
        modules["verl.utils.tensordict_utils"].get = lambda data, key, default=None: data.metadata.get(key, default)

        class BaseTrainingWorker:
            def train_batch(self, data):
                if not data.skip:
                    self.engine.module.weight.grad = torch.ones_like(self.engine.module.weight)
                    self.engine.optimizer.step()
                return {"trained": True}

            def train_mini_batch(self, data):
                return self.train_batch(data)

            def reset(self):
                return "reset"

        modules["verl.workers.engine_workers"].TrainingWorker = BaseTrainingWorker
        modules["verl.workers.engine_workers"].ActorRolloutRefWorker = type("BaseActor", (), {})
        with patch.dict(sys.modules, modules):
            cls.worker_module = load_module("_ema_worker_test_package.ema_worker", MODULE_ROOT / "ema_worker.py")

    def setUp(self):
        self.worker = self.worker_module.EMATrainingWorker()
        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.data.fill_(2)
        self.worker.engine = types.SimpleNamespace(
            module=module, optimizer=torch.optim.SGD(module.parameters(), lr=1), get_data_parallel_size=lambda: 1,
        )
        self.worker.rank = 0
        self.worker.initialize_ema()
        self.batch = types.SimpleNamespace(shape=(4,), metadata={"mini_batch_size": 4}, skip=False)

    def test_observer_counts_actual_step_and_ema_applies_once(self):
        self.worker.train_mini_batch(self.batch)
        self.assertEqual(self.worker._observed_optimizer_step, 1)
        self.assertEqual(self.worker.parameter_ema.num_updates, 1)
        self.assertAlmostEqual(self.worker.parameter_ema.parameters["weight"].item(), 1.95, places=6)
        self.worker.update_ema(1)
        self.assertEqual(self.worker.parameter_ema.num_updates, 1)
        with self.assertRaises(ValueError):
            self.worker.update_ema(2)

    def test_skipped_step_and_multiple_steps_fail_without_fake_clock(self):
        self.batch.skip = True
        with self.assertRaisesRegex(RuntimeError, "Optimizer skipped"):
            self.worker.train_mini_batch(self.batch)
        self.assertEqual(self.worker.parameter_ema.num_updates, 0)
        self.assertEqual(self.worker._observed_optimizer_step, 0)
        self.batch.metadata = {"mini_batch_size": 2}
        with self.assertRaises(ValueError):
            self.worker.train_mini_batch(self.batch)
        self.assertEqual(self.worker._observed_optimizer_step, 0)

    def test_reset_invalidates_state_and_removes_optimizer_observer(self):
        self.assertEqual(self.worker.reset(), "reset")
        with self.assertRaises(RuntimeError):
            self.worker._ready()
        self.worker.engine.module.weight.grad = torch.ones_like(self.worker.engine.module.weight)
        self.worker.engine.optimizer.step()
        self.assertEqual(self.worker._observed_optimizer_step, 0)

    def test_rank_checkpoint_save_load_and_observer_resume(self):
        self.worker.train_mini_batch(self.batch)
        saved = self.worker.parameter_ema.parameters["weight"].clone()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(torch.distributed, "get_world_size", return_value=1),
        ):
            result = self.worker.save_ema(directory)
            self.assertTrue(Path(result["path"]).is_file())
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.worker.initialize_ema(reset=True)
            self.worker.load_ema(directory, expected_decay=0.95)
            torch.testing.assert_close(self.worker.parameter_ema.parameters["weight"], saved)
            self.assertEqual(self.worker._observed_optimizer_step, 1)
            self.worker.train_mini_batch(self.batch)
            self.assertEqual(self.worker.parameter_ema.last_optimizer_step, 2)
            self.assertEqual(self.worker.parameter_ema.num_updates, 2)


if __name__ == "__main__":
    unittest.main()
