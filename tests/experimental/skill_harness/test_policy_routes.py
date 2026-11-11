# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU contract tests for policy isolation and versioned standalone routing."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "verl/experimental/skill_harness/policy_routes.py"
SPEC = importlib.util.spec_from_file_location("_policy_routes_cpu_test", PATH)
routes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = routes
SPEC.loader.exec_module(routes)


def config():
    return {
        "actor_rollout_ref": {
            "model": {
                "path": "/models/solver", "local_path": "/cache/solver", "hf_config_path": "/models/solver",
                "hf_config": {"solver": True}, "lora": {"rank": 0},
            },
            "rollout": {
                "name": "vllm", "nnodes": 0, "n_gpus_per_node": 8, "tensor_model_parallel_size": 1,
                "data_parallel_size": 1, "pipeline_model_parallel_size": 1, "load_format": "dummy",
                "calculate_log_probs": False, "checkpoint_engine": {"backend": "naive", "engine_kwargs": {}},
            },
        },
    }


def curator():
    return routes.PolicyRouteSpec("curator", "ema", 2, 10000, "initial-checkpoint", transport_group="curator_test")


def evaluator():
    return routes.PolicyRouteSpec("credit", "frozen", 2, 20000, "canonical-model-digest", model_path="/models/frozen")


class FakeClient:
    def __init__(self, cfg, clock=None):
        self.config = cfg
        self.clock = clock
        self.minimum = clock
        self.maximum = clock
        self.log_probs = [-0.2, -0.3]
        self.stop_reason = "stop"
        self.calls = []

    async def generate(self, request_id, **kwargs):
        self.calls.append((request_id, kwargs))
        return types.SimpleNamespace(
            token_ids=[12, 34], log_probs=self.log_probs, stop_reason=self.stop_reason,
            extra_fields={
                "global_steps": self.clock, "min_global_steps": self.minimum, "max_global_steps": self.maximum,
            },
        )


class FakeReplica:
    def __init__(self, events):
        self.events = events
        self.resumed = False

    async def resume_generation(self):
        self.events.append("resume-generation")
        self.resumed = True


class FakeManager:
    def __init__(self, cfg, start_rank, events):
        self.config = cfg
        self.start_rank = start_rank
        self.events = events
        rollout = cfg["actor_rollout_ref"]["rollout"]
        count = rollout["n_gpus_per_node"] // rollout["tensor_model_parallel_size"]
        self.replicas = [FakeReplica(events) for _ in range(count)]
        self.client = FakeClient(cfg)

    def get_replicas(self):
        return self.replicas

    def get_client(self):
        return self.client

    def get_addresses(self):
        return [f"fake-{self.start_rank + index}:8000" for index in range(len(self.replicas))]


class FakeActor:
    world_size = 4

    def __init__(self, events):
        self.events = events
        self.step = 0
        self.configured = []
        self.sent = []
        self.transport_calls = []
        self.mismatched_rank = False

    def configure_ema_transport(self, backend, **kwargs):
        self.configured.append((backend, kwargs))
        return [{"backend": backend} for _ in range(self.world_size)]

    def get_ema_metadata(self):
        return [
            {"last_optimizer_step": self.step + int(self.mismatched_rank and rank == 3)}
            for rank in range(self.world_size)
        ]

    def send_ema_weights(self, global_steps=None):
        self.events.append("ema-send")
        self.sent.append(global_steps)
        return [{"ema": {"last_optimizer_step": self.step}} for _ in range(self.world_size)]

    def execute_ema_transport(self, method, *args, **kwargs):
        self.transport_calls.append((method, args, kwargs))
        return [{"ema_transport": True} for _ in range(self.world_size)]

    def update_weights(self, **kwargs):
        raise AssertionError("The EMA route must never invoke online solver synchronization")

    def execute_checkpoint_engine(self, *args, **kwargs):
        raise AssertionError("The EMA route must never invoke the online solver transport")


class FakeCheckpointManager:
    def __init__(self, cfg, sender, replicas, events, manager):
        self.config = cfg
        self.sender = sender
        self.replicas = replicas
        self.events = events
        self.manager = manager
        self.fail = False
        self.sent = []

    async def update_weights(self, global_steps):
        self.events.append("transfer-start")
        self.sender.execute_checkpoint_engine(["prepare"] * self.sender.world_size)
        self.sender.update_weights(global_steps=global_steps, mode="nccl")
        if self.fail:
            raise RuntimeError("transfer failed")
        self.sender.execute_checkpoint_engine(["finalize"] * self.sender.world_size)
        self.events.append("kv-restored")
        self.sent.append(global_steps)
        self.manager.client.clock = global_steps
        self.manager.client.minimum = global_steps
        self.manager.client.maximum = global_steps


class PolicyRoutesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.actor = FakeActor(self.events)
        self.managers = []
        self.checkpoints = []

        async def manager_factory(*, config, worker_group, start_rank):
            self.assertIsNone(worker_group)
            manager = FakeManager(config, start_rank, self.events)
            self.managers.append(manager)
            return manager

        def checkpoint_factory(cfg, sender, replicas):
            manager = self.managers[-1]
            result = FakeCheckpointManager(cfg, sender, replicas, self.events, manager)
            self.checkpoints.append(result)
            return result

        self.manager_factory = manager_factory
        self.checkpoint_factory = checkpoint_factory

    async def create(self, specs=None, available_gpus=8):
        return await routes.create_policy_routes(
            config(), specs or [curator(), evaluator()], actor_worker_group=self.actor, solver_gpus=4,
            available_gpus=available_gpus, manager_factory=self.manager_factory,
            checkpoint_manager_factory=self.checkpoint_factory,
            fingerprint_factory=lambda path: {"fingerprint": "canonical-model-digest", "model_path": path, "files": []},
        )

    async def test_topology_and_configuration_are_isolated(self):
        original = config()
        saved = copy.deepcopy(original)
        ema = routes.build_route_config(original, curator())
        frozen = routes.build_route_config(original, evaluator())
        self.assertEqual(original, saved)
        self.assertEqual(ema["actor_rollout_ref"]["rollout"]["checkpoint_engine"]["backend"], "nccl")
        self.assertEqual(frozen["actor_rollout_ref"]["rollout"]["load_format"], "auto")
        self.assertEqual(frozen["actor_rollout_ref"]["model"]["path"], "/models/frozen")
        self.assertIsNone(frozen["actor_rollout_ref"]["model"]["local_path"])
        created = await self.create()
        self.assertEqual(set(created), {"curator", "credit"})
        self.assertEqual([manager.start_rank for manager in self.managers], [10000, 20000])
        self.assertEqual(len(self.actor.configured), 1)
        self.assertEqual(len(self.checkpoints), 1)
        self.assertEqual(
            self.actor.configured[0][1]["engine_kwargs"], {"group_name": "curator_test", "multi_sender": False}
        )

    async def test_budget_and_rank_overlap_fail_before_starting_resources(self):
        with self.assertRaises(ValueError):
            await self.create(available_gpus=7)
        overlap = routes.PolicyRouteSpec("credit", "frozen", 2, 10001, "digest", model_path="/models/pinned")
        with self.assertRaises(ValueError):
            await self.create([curator(), overlap])
        with self.assertRaises(ValueError):
            await self.create([curator(), curator()])
        self.assertEqual(self.managers, [])
        self.assertEqual(self.actor.configured, [])

    async def test_first_ema_publication_is_required_and_reopens_after_transfer(self):
        route = (await self.create())["curator"]
        with self.assertRaises(RuntimeError):
            route.get_client()
        result = await route.sync_ema(0)
        self.assertTrue(result["ready"])
        self.assertEqual(
            self.events, ["transfer-start", "ema-send", "kv-restored", "resume-generation", "resume-generation"]
        )
        client = route.get_client()
        output = await client.generate("curate-A", prompt_ids=[1], sampling_params={"seed": 31})
        self.assertEqual(output.extra_fields["ema_optimizer_step"], 0)
        self.assertEqual(output.extra_fields["skill_policy_route"]["kind"], "ema")
        self.assertEqual(client.delegate.calls[-1][0], "curate-A")
        await route.sync_ema(0)
        self.assertEqual(self.actor.sent, [0])

    async def test_old_client_rejects_new_published_version(self):
        route = (await self.create())["curator"]
        await route.sync_ema(0)
        old_client = route.get_client()
        self.actor.step = 1
        await route.sync_ema(1)
        with self.assertRaisesRegex(RuntimeError, "stale EMA"):
            await old_client.generate("old")
        await route.get_client().generate("new")

    async def test_missing_mixed_aborted_or_missing_logprob_response_rejected(self):
        route = (await self.create())["curator"]
        await route.sync_ema(0)
        client = route.get_client()
        for field, value in (("clock", None), ("maximum", 1), ("log_probs", None), ("stop_reason", "aborted")):
            previous = getattr(client.delegate, field)
            setattr(client.delegate, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                await client.generate("invalid")
            setattr(client.delegate, field, previous)

    async def test_frozen_attestation_does_not_invent_an_optimizer_clock(self):
        route = (await self.create())["credit"]
        client = route.get_client()
        client.delegate.log_probs = None
        output = await client.generate("paired-B", prompt_ids=[1], sampling_params={"seed": 44})
        self.assertEqual(route.frozen_version, "canonical-model-digest")
        self.assertEqual(client.frozen_version, route.frozen_version)
        self.assertEqual(output.extra_fields["policy_kind"], "frozen_evaluator")
        self.assertEqual(output.extra_fields["frozen_model_version"], route.frozen_version)
        self.assertIsNone(output.extra_fields["global_steps"])
        self.assertIsNone(output.extra_fields["observed_global_steps"])
        with self.assertRaises(RuntimeError):
            await route.sync_ema(1)

    async def test_serializable_client_preserves_policy_pin(self):
        route = (await self.create())["credit"]
        client = pickle.loads(pickle.dumps(route.get_client()))
        output = await client.generate("after-ray-serialization")
        self.assertEqual(output.extra_fields["frozen_model_version"], route.frozen_version)

    async def test_frozen_attestation_is_defensive_and_observed_clock_cannot_change(self):
        route = (await self.create())["credit"]
        first = route.attestation
        first["frozen_model_version"] = "changed-copy"
        self.assertEqual(route.attestation["frozen_model_version"], route.frozen_version)
        client = route.get_client()
        await client.generate("first")
        client.delegate.clock = client.delegate.minimum = client.delegate.maximum = 1
        with self.assertRaisesRegex(RuntimeError, "clock changed"):
            await client.generate("unexpected-update")

    async def test_wrong_frozen_fingerprint_fails_before_any_launch(self):
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            await routes.create_policy_routes(
                config(), [curator(), evaluator()], actor_worker_group=self.actor, solver_gpus=4,
                available_gpus=8, manager_factory=self.manager_factory,
                checkpoint_manager_factory=self.checkpoint_factory,
                fingerprint_factory=lambda path: {"fingerprint": "different", "model_path": path, "files": []},
            )
        self.assertEqual(self.managers, [])
        self.assertEqual(self.actor.configured, [])

    async def test_transfer_failure_does_not_publish_or_resume(self):
        route = (await self.create())["curator"]
        self.checkpoints[0].fail = True
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            await route.sync_ema(0)
        self.assertTrue(route.failed)
        self.assertFalse(route.ready)
        self.assertNotIn("resume-generation", self.events)
        with self.assertRaises(RuntimeError):
            route.get_client()
        with self.assertRaises(RuntimeError):
            await route.sync_ema(0)

    async def test_actor_rank_clock_mismatch_fails_before_transferring(self):
        route = (await self.create())["curator"]
        self.actor.mismatched_rank = True
        with self.assertRaisesRegex(RuntimeError, "actual EMA"):
            await route.sync_ema(0)
        self.assertEqual(self.actor.sent, [])

    async def test_sender_view_only_calls_dedicated_ema_methods(self):
        sender = routes.EMASenderView(self.actor)
        sender.update_weights(global_steps=3)
        sender.execute_checkpoint_engine(["init_process_group"] * 4, rank=[0, 1, 2, 3])
        self.assertEqual(self.actor.sent, [3])
        self.assertEqual(self.actor.transport_calls[-1][2], {"rank": [0, 1, 2, 3]})
        with self.assertRaises(ValueError):
            sender.update_weights(mode="naive")
        with self.assertRaises(ValueError):
            sender.execute_checkpoint_engine(["_private"] * 4)

    async def test_invalid_specs_and_unsupported_backends_fail(self):
        with self.assertRaises(ValueError):
            routes.PolicyRouteSpec("credit", "frozen", 1, 200, "digest")
        with self.assertRaises(ValueError):
            routes.PolicyRouteSpec("curator", "ema", 3, 200, "digest", tensor_parallel_size=2)
        bad = config()
        bad["actor_rollout_ref"]["rollout"]["disaggregation"] = {"enabled": True}
        with self.assertRaises(NotImplementedError):
            routes.build_route_config(bad, curator())


class ModelFingerprintTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text('{"model_type":"test"}')
        (self.root / "tokenizer.json").write_text('{"model":{}}')
        (self.root / "model.safetensors").write_bytes(b"small-weight-fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def test_content_identity_survives_copy_and_cached_reads_open_no_files(self):
        with patch.object(routes.time, "time_ns", return_value=time.time_ns() + 10_000_000_000):
            first = routes.fingerprint_model(self.root)
            with patch.object(Path, "open", side_effect=AssertionError("Stable old files should use cached hashes")):
                self.assertEqual(routes.fingerprint_model(self.root), first)
        with tempfile.TemporaryDirectory() as target:
            copy_path = Path(target) / "copied"
            shutil.copytree(self.root, copy_path)
            second = routes.fingerprint_model(copy_path)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertNotEqual(first["model_path"], second["model_path"])

    def test_modified_weights_and_new_tokenizer_metadata_invalidate_identity(self):
        first = routes.fingerprint_model(self.root)
        weight = self.root / "model.safetensors"
        stat = weight.stat()
        weight.write_bytes(b"other-weight-fixture")
        os.utime(weight, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        second = routes.fingerprint_model(self.root)
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        (self.root / "tokenizer_config.json").write_text('{"chat_template":"changed"}')
        third = routes.fingerprint_model(self.root)
        self.assertNotEqual(second["fingerprint"], third["fingerprint"])

    def test_missing_or_escaping_indexed_shard_is_rejected(self):
        index = self.root / "model.safetensors.index.json"
        for shard in ("missing.safetensors", "../outside.safetensors"):
            index.write_text(json.dumps({"weight_map": {"weight": shard}}))
            with self.subTest(shard=shard), self.assertRaises(ValueError):
                routes.fingerprint_model(self.root)


if __name__ == "__main__":
    unittest.main()
