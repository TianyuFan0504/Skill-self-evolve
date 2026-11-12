"""Concurrency, durability and request replay contracts without model resources."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from aiohttp import ClientTimeout
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from skill_harness_examples import parallel_environment_server as server  # noqa: E402

_TOY_COUNTS = {}


def toy_process_operation(operation, payload):
    entered = time.monotonic_ns()
    if operation == "ping":
        result = {"pid": os.getpid()}
    else:
        time.sleep(payload.get("delay", 0))
        key = payload["session_id"]
        _TOY_COUNTS[key] = _TOY_COUNTS.get(key, 0) + 1
        result = {"count": _TOY_COUNTS[key]}
    return {**result, "worker_pid": os.getpid(), "entered_ns": entered, "completed_ns": time.monotonic_ns()}


class FakeEnginePool:
    def __init__(self):
        self.executors, self.pids, self.cursor = [None] * 4, [1101, 1102, 1103, 1104], 0
        self.counts, self.delay, self.entered, self.concurrent = {}, 0, asyncio.Event(), {}
        self.closed = False

    def assign(self):
        value = self.cursor % 4
        self.cursor += 1
        return value

    async def call(self, shard, operation, payload, timing=None):
        entered = time.monotonic_ns()
        key = payload["session_id"]
        if operation == "create":
            self.counts[key] = 0
            result = {"state": {"feedback": "test goal", "admissible_commands": ["look", "inventory"], "won": False}}
        elif operation == "step":
            self.entered.set()
            self.concurrent[key] = self.concurrent.get(key, 0) + 1
            assert self.concurrent[key] == 1, "One session concurrently entered the engine"
            await asyncio.sleep(self.delay)
            self.counts[key] += 1
            self.concurrent[key] -= 1
            result = {
                "admitted": True,
                "state": {
                    "feedback": str(self.counts[key]),
                    "admissible_commands": ["look", "inventory"],
                    "won": False,
                },
                "engine_score": 0,
                "done": False,
            }
        elif operation == "close":
            result = {}
        else:
            raise AssertionError(operation)
        if timing is not None:
            timing.update(
                engine_queue_ms=0,
                engine_execute_ms=(time.monotonic_ns() - entered) / 1e6,
                engine_shard=shard,
                engine_pid=self.pids[shard],
            )
        return {**result, "worker_pid": self.pids[shard], "entered_ns": entered, "completed_ns": time.monotonic_ns()}

    def shutdown(self):
        self.closed = True


class ParallelServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run, self.benchmark = self.root / "run", self.root / "benchmark"
        self.relative = "json_2.1.1/train/task/trial/game.tw-pddl"
        game = self.benchmark / "data" / self.relative
        game.parent.mkdir(parents=True)
        game.write_text("test fixture\n")
        from data import GameRecord, stable_game_id

        digest = hashlib.sha256(game.read_bytes()).hexdigest()
        self.row = GameRecord(
            stable_game_id(self.relative, digest), "train", "pick_and_place_simple", "scene", self.relative, digest
        )
        self.pool = FakeEnginePool()
        self.service = server.ParallelGameService(
            [self.row], self.benchmark, self.run, allowed_splits=["train"], engine_pool=self.pool
        )
        self.client = TestClient(TestServer(server.application(self.service)))
        await self.client.start_server()
        self.headers = {"X-Admin-Token": self.service.admin_token}

    async def asyncTearDown(self):
        await self.client.close()

    async def create(self):
        attempt = uuid.uuid4().hex
        output = self.run / "episodes" / self.row.game_id / ("attempt_" + attempt)
        response = await self.client.post(
            "/admin/create",
            headers=self.headers,
            json={**self.row.to_dict(), "catalog_digest": self.service.catalog_digest, "output_dir": str(output)},
        )
        self.assertEqual(response.status, 200, await response.text())
        session_id = (await response.json())["session_id"]
        # Pure protocol fixture: no account is allocated and no backend exists.
        payload = {
            "schema_version": 1,
            "budget_seconds": 900,
            "binding": {
                "binding_id": "a" * 64,
                "binding_sha256": "b" * 64,
                "attempt_id": attempt,
                "gateway_session_id": "cpu-fixture",
                "environment_session_id": session_id,
            },
            "account": {
                "lease_id": "c" * 32,
                "uid": 20000,
                "gid": 20000,
                "name": "fixture",
                "owner": {"kind": "ordinary", "run_root": str(self.run.resolve()), "scope_id": attempt},
            },
        }
        response = await self.client.post("/admin/budget/" + session_id + "/start", headers=self.headers, json=payload)
        self.assertEqual(response.status, 200, await response.text())
        return session_id, output

    async def test_four_real_processes_run_independently_and_keep_sticky_state(self):
        pool = server.StickyEnginePool(4, operation=toy_process_operation)
        try:
            await pool.start()
            self.assertEqual(len(set(pool.pids)), 4)
            results = await asyncio.gather(
                *(pool.call(index, "toy", {"session_id": str(index), "delay": 0.12}) for index in range(4))
            )
            self.assertLess(max(row["entered_ns"] for row in results), min(row["completed_ns"] for row in results))
            self.assertEqual([row["count"] for row in results], [1] * 4)
            again = await asyncio.gather(*(pool.call(index, "toy", {"session_id": str(index)}) for index in range(4)))
            self.assertEqual([row["count"] for row in again], [2] * 4)
            self.assertEqual([row["worker_pid"] for row in results], [row["worker_pid"] for row in again])
        finally:
            pool.shutdown()

    async def test_timeout_query_and_duplicate_execute_only_once_and_survive_close(self):
        session, output = await self.create()
        identity = uuid.uuid4().hex
        self.pool.delay = 0.15
        with self.assertRaises(asyncio.TimeoutError):
            await self.client.post(
                "/session/" + session + "/step",
                json={"action": "look", "request_id": identity},
                timeout=ClientTimeout(total=0.03),
            )
        query = "/session/" + session + "/requests/" + identity
        response = await self.client.get(query)
        self.assertEqual(response.status, 202)
        deadline = time.monotonic() + 3
        while True:
            response = await self.client.get(query)
            record = await response.json()
            if record["status"] == "completed" or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["response_status"], 200)
        self.assertEqual(record["result"]["steps"], 1)
        repeated = await self.client.post(
            "/session/" + session + "/step", json={"action": "look", "request_id": identity}
        )
        self.assertEqual((await repeated.json())["steps"], 1)
        changed = await self.client.post(
            "/session/" + session + "/step", json={"action": "inventory", "request_id": identity}
        )
        self.assertEqual(changed.status, 409)
        self.assertEqual(self.pool.counts[session], 1)
        self.assertEqual(len((output / "budget-events.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((output / "environment-trace.jsonl").read_text().splitlines()), 1)
        self.assertTrue((output / ("action-result-" + identity + ".json")).exists())
        closed = await self.client.post("/admin/close/" + session, headers=self.headers)
        self.assertEqual(closed.status, 200)
        response = await self.client.get(query)
        self.assertEqual((await response.json())["result"]["steps"], 1)

    async def test_slow_durable_write_does_not_block_health_or_another_session(self):
        session, _ = await self.create()
        other, _ = await self.create()
        original = self.service._budget_event

        def slow(*args, **kwargs):
            time.sleep(0.18)
            return original(*args, **kwargs)

        self.service._budget_event = slow
        action = asyncio.create_task(
            self.client.post("/session/" + session + "/step", json={"action": "look", "request_id": uuid.uuid4().hex})
        )
        await self.pool.entered.wait()
        await asyncio.sleep(0.02)
        started = time.monotonic()
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        observed = await self.client.get("/session/" + other + "/observe")
        self.assertEqual(observed.status, 200)
        self.assertLess(time.monotonic() - started, 0.12)
        result = await action
        self.assertEqual(result.status, 200)

    async def test_session_lock_orders_actions_and_ack_waits_for_durable_result(self):
        session, output = await self.create()
        self.pool.delay = 0.04
        results = await asyncio.gather(
            *(
                self.client.post(
                    "/session/" + session + "/step", json={"action": "look", "request_id": uuid.uuid4().hex}
                )
                for _ in range(3)
            )
        )
        self.assertEqual([(await response.json())["steps"] for response in results], [1, 2, 3])
        traces = [json.loads(line) for line in (output / "environment-trace.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in traces], [1, 2, 3])
        self.assertEqual(len(list(output.glob("action-result-*.json"))), 3)
        health = await self.client.get("/health")
        self.assertEqual((await health.json())["action_request_protocol"], "alfworld-step-request-id-v1")
        self.assertIn("X-Environment-Timing", results[0].headers)

    async def test_header_body_identity_conflict_and_missing_lookup(self):
        session, _ = await self.create()
        response = await self.client.post(
            "/session/" + session + "/step",
            headers={"X-Request-ID": "a" * 32},
            json={"action": "look", "request_id": "b" * 32},
        )
        self.assertEqual(response.status, 409)
        response = await self.client.get("/session/" + session + "/requests/" + "d" * 32)
        self.assertEqual(response.status, 404)
        self.assertEqual(self.pool.counts[session], 0)

    async def test_unknown_engine_error_is_cached_and_never_reexecuted(self):
        session, output = await self.create()
        original = self.pool.call

        async def failed(shard, operation, payload, timing=None):
            if operation == "step":
                self.pool.counts[session] += 1
                raise RuntimeError("Injected unknown engine result")
            return await original(shard, operation, payload, timing)

        self.pool.call = failed
        identity = uuid.uuid4().hex
        payload = {"action": "look", "request_id": identity}
        first = await self.client.post("/session/" + session + "/step", json=payload)
        repeated = await self.client.post("/session/" + session + "/step", json=payload)
        self.assertEqual(first.status, 500)
        self.assertEqual(repeated.status, 500)
        self.assertEqual(self.pool.counts[session], 1)
        queried = await self.client.get("/session/" + session + "/requests/" + identity)
        self.assertEqual((await queried.json())["response_status"], 500)
        durable = json.loads((output / ("action-result-" + identity + ".json")).read_text())
        self.assertEqual(durable["response_status"], 500)
        self.assertEqual(len((output / "budget-events.jsonl").read_text().splitlines()), 1)

    async def test_cleanup_failure_still_attempts_other_world_and_closes_all_pools(self):
        first, _ = await self.create()
        second, _ = await self.create()
        original, attempted = self.service._finish, []

        async def failed(session, **kwargs):
            attempted.append(session["id"])
            if session["id"] == first:
                raise OSError("Injected close failure")
            return await original(session, **kwargs)

        self.service._finish = failed
        with self.assertRaisesRegex(RuntimeError, "Injected close failure"):
            await self.service.close_all(None)
        self.assertEqual(attempted, [first, second])
        self.assertTrue(self.pool.closed)
        for executor in (self.service.io, self.service.telemetry):
            with self.assertRaises(RuntimeError):
                executor.submit(lambda: None)
        # The fake failed world's unknown closure remains an explicit error;
        # remove only this in-memory fixture before test-server teardown.
        self.service.sessions.clear()


if __name__ == "__main__":
    unittest.main()
