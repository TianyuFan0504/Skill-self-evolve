"""CPU failure archives; no native process, Ray actor, backend or reward is run."""

from __future__ import annotations

import asyncio
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_probe_failure_capture_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
probes = importlib.import_module(PACKAGE + ".probes")
accounts = importlib.import_module(PACKAGE + ".alfworld.accounts")


class Trajectory:
    def __init__(self, **fields):
        self.__dict__.update(dict(finished=False, reward=0.0, accuracy=0.0, extra_fields={}), **fields)


def lifecycle_manager(session_id, rows, *, before_finalize=None):
    """Exercise capture against the public async lifecycle interface only."""
    session = types.SimpleNamespace(
        handle=types.SimpleNamespace(
            session_id=session_id, base_url="http://127.0.0.1:1/sessions/" + session_id + "/v1"
        ),
        request_lock=asyncio.Lock(),
        phase="ACTIVE",
        materialized_chains=[types.SimpleNamespace(order_seq=i, trajectory=row) for i, row in enumerate(rows)],
    )
    calls = []
    manager = types.SimpleNamespace(_session_to_gateway_index={session_id: 0}, active_sessions_per_gateway=[1])

    async def create(identity, **kwargs):
        if identity != session_id:
            raise AssertionError("Unexpected session")
        return session.handle

    async def finalize(identity):
        calls.append("finalize")
        if before_finalize:
            before_finalize()
        async with session.request_lock:
            if session.phase == "ABORTED":
                raise RuntimeError("The session was aborted")
            result = [row.trajectory for row in session.materialized_chains]
            session.phase = "FINALIZED"
            manager._session_to_gateway_index.pop(identity, None)
            manager.active_sessions_per_gateway[0] = 0
            return result

    async def abort(identity):
        calls.append("abort")
        async with session.request_lock:
            session.phase = "ABORTED"
            manager._session_to_gateway_index.pop(identity, None)
            manager.active_sessions_per_gateway[0] = 0

    manager.create_session, manager.finalize_session, manager.abort_session = create, finalize, abort
    return manager, session, Trajectory, calls


class NativeFailureCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.attestation = {
            "schema_version": 1,
            "policy_kind": "frozen_evaluator",
            "frozen_model_version": "b" * 64,
            "model_fingerprint": "b" * 64,
            "model_path": "/models/frozen",
            "lifecycle_id": "public-lifecycle-CPU-fixture",
        }
        self.sample = {
            "raw_prompt": [{"role": "user", "content": "Original task"}],
            "tools_kwargs": {
                "task": {"metadata": {"credit_probe_group": "group", "credit_probe_variant": "with_skill"}}
            },
        }

    def manager(self, *, empty=False, before_finalize=None):
        manager, session, trajectory, calls = lifecycle_manager("session_1", [], before_finalize=before_finalize)
        row = trajectory(
            prompt_ids=[10, 20],
            response_ids=[30, 40, 50],
            response_mask=[1, 0, 1],
            response_logprobs=[-0.25, 0.0, -0.75],
            num_turns=2,
            extra_fields={"min_global_steps": 7, "max_global_steps": 7},
        )
        if not empty:
            session.materialized_chains = [types.SimpleNamespace(order_seq=0, trajectory=row)]
        manager.skill_route_attestation = self.attestation.copy()
        return manager, session, row, calls

    async def run_branch(self, manager, runner, **kwargs):
        runner_module = types.ModuleType(PACKAGE + ".alfworld.runner")
        runner_module.run_task = runner
        with patch.dict(sys.modules, {runner_module.__name__: runner_module}):
            return await probes.native_probe_branch(
                gateway_manager=manager,
                session_id="session_1",
                sample_fields=self.sample,
                artifact_dir=self.root,
                task_config_path="unused.yaml",
                frozen_attestation=self.attestation,
                capture_timeout_seconds=0.03,
                cleanup_timeout_seconds=0.03,
                **kwargs,
            )

    def payload(self, name):
        return probes._read_payload(self.root / name)

    async def test_public_finalize_preserves_token_fields_and_original_cause(self):
        def before_finalize():
            self.assertEqual(self.payload("native-failure.json")["error"]["type"], "IsolationError")

        manager, _, row, calls = self.manager(before_finalize=before_finalize)
        cause = RuntimeError("owned backend drain remained unknown")
        primary = accounts.IsolationError("Task isolation failed")

        async def runner(**kwargs):
            raise primary from cause

        with self.assertRaises(accounts.IsolationError) as caught:
            await self.run_branch(manager, runner)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        capture = self.payload("native-failure-capture.json")
        self.assertEqual(capture["trajectories"], [probes._trajectory_payload(row)])
        self.assertEqual(capture["capture_status"], "observed_committed_trajectories")
        self.assertEqual(capture["finalization"]["status"], "returned")
        self.assertTrue(capture["archive_only"])
        self.assertFalse(capture["finished"])
        self.assertIsNone(capture["reward"])
        self.assertIsNone(capture["accuracy"])
        self.assertEqual(calls, ["finalize"])
        self.assertEqual(manager.active_sessions_per_gateway, [0])
        self.assertFalse((self.root / "native-capture.json").exists())
        self.assertFalse(list(self.root.rglob("paired-event.json")))

    async def test_empty_finalize_is_not_submitted_or_model_observed_evidence(self):
        manager, _, _, _ = self.manager(empty=True)
        primary = RuntimeError("first request still uncommitted")

        async def runner(**kwargs):
            raise primary

        with self.assertRaises(RuntimeError) as caught:
            await self.run_branch(manager, runner)
        self.assertIs(caught.exception, primary)
        capture = self.payload("native-failure-capture.json")
        self.assertEqual(capture["capture_status"], "empty")
        self.assertEqual(capture["trajectories"], [])

    async def test_locked_actual_finalize_and_abort_wait_are_bounded_and_unknown(self):
        manager, session, _, calls = self.manager()
        await session.request_lock.acquire()
        primary = RuntimeError("Task raised while Gateway lock held")

        async def runner(**kwargs):
            raise primary

        try:
            started = asyncio.get_running_loop().time()
            with self.assertRaises(RuntimeError) as caught:
                await self.run_branch(manager, runner)
            self.assertLess(asyncio.get_running_loop().time() - started, 0.5)
        finally:
            session.request_lock.release()
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.payload("native-failure-capture.json")["capture_status"], "unknown")
        self.assertIsNone(self.payload("native-cleanup-failure.json")["backend_stop_confirmed"])
        self.assertEqual(calls, ["finalize", "abort"])
        self.assertIn("session_1", manager._session_to_gateway_index)
        self.assertTrue(any("abort" in note for note in primary.__notes__))

    async def test_finalize_exception_then_actual_abort_preserves_primary_and_clears_only_gateway(self):
        manager, session, _, calls = self.manager()
        session.phase = "ABORTED"
        primary = ValueError("original runner failure")

        async def runner(**kwargs):
            raise primary

        with self.assertRaises(ValueError) as caught:
            await self.run_branch(manager, runner)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ["finalize", "abort"])
        self.assertEqual(self.payload("native-failure-capture.json")["capture_status"], "unknown")
        self.assertEqual(manager.active_sessions_per_gateway, [0])

    async def test_diagnostic_io_failure_does_not_replace_original_or_skip_capture(self):
        manager, _, _, _ = self.manager()
        primary = ValueError("original Task Python error")
        publish = probes._publish_immutable

        def failed_write(path, payload):
            if path.name == "native-failure.json":
                raise OSError("disk full")
            return publish(path, payload)

        async def runner(**kwargs):
            raise primary

        with patch.object(probes, "_publish_immutable", failed_write), self.assertRaises(ValueError) as caught:
            await self.run_branch(manager, runner)
        self.assertIs(caught.exception, primary)
        self.assertTrue((self.root / "native-failure-capture.json").is_file())
        self.assertTrue(any("disk full" in note for note in primary.__notes__))

    async def test_normal_success_retains_token_payload_and_no_failure_files(self):
        manager, _, row, calls = self.manager()

        async def runner(**kwargs):
            return types.SimpleNamespace(reward=0.0, accuracy=0.0, finished=True, extra_info={"agent_ok": True})

        capture = await self.run_branch(manager, runner)
        self.assertEqual(capture["trajectories"], [probes._trajectory_payload(row)])
        self.assertTrue(capture["finished"])
        self.assertEqual(capture, self.payload("native-capture.json"))
        self.assertFalse(list(self.root.glob("*failure*.json")))
        self.assertEqual(calls, ["finalize"])

    async def test_normal_finalize_timeout_is_not_retried_or_returned_as_success(self):
        manager, session, _, calls = self.manager()
        await session.request_lock.acquire()

        async def runner(**kwargs):
            return types.SimpleNamespace(reward=1.0, accuracy=1.0, finished=True, extra_info={})

        try:
            with self.assertRaises(TimeoutError):
                await self.run_branch(manager, runner)
        finally:
            session.request_lock.release()
        self.assertEqual(calls, ["finalize", "abort"])
        self.assertEqual(self.payload("native-failure.json")["stage"], "finalize")
        capture = self.payload("native-failure-capture.json")
        self.assertFalse(capture["finished"])
        self.assertIsNone(capture["reward"])
        self.assertEqual(capture["capture_status"], "unknown")

    async def test_cancellation_is_same_original_instance_after_failure_archive(self):
        manager, _, _, _ = self.manager()
        primary = asyncio.CancelledError("caller cancelled native branch")

        async def runner(**kwargs):
            raise primary

        with self.assertRaises(asyncio.CancelledError) as caught:
            await self.run_branch(manager, runner)
        self.assertIs(caught.exception, primary)
        self.assertEqual(self.payload("native-failure.json")["error"]["type"], "CancelledError")
        self.assertFalse(self.payload("native-failure-capture.json")["finished"])


class PairPrimaryPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = types.SimpleNamespace(window_id="window_1", probe_id="probe_1")
        self.cleanup_error = accounts.IsolationError("UID remains quarantined")

        def finish(*args, **kwargs):
            raise self.cleanup_error

        self.pool = types.SimpleNamespace(
            acquire=lambda **kwargs: {"lease": "private CPU fixture"},
            public=lambda ref: {"lease_id": "public_lease"},
            finish=finish,
        )

    def test_cleanup_failure_keeps_original_exception_and_writes_secondary_audit(self):
        primary = RuntimeError("native runner Python failure")
        with self.assertRaises(RuntimeError) as caught:
            with probes._pair_account_scope(self.root, request=self.request, run_root=self.root, pool=self.pool):
                raise primary
        self.assertIs(caught.exception, primary)
        payload = probes._read_payload(self.root / "pair-cleanup-failure.json")
        self.assertEqual(payload["primary"]["message"], str(primary))
        self.assertEqual(payload["cleanup"]["message"], str(self.cleanup_error))
        self.assertFalse(payload["closed"])
        self.assertNotIn("private CPU fixture", str(payload))

    def test_cleanup_alone_is_fatal(self):
        with self.assertRaises(accounts.IsolationError) as caught:
            with probes._pair_account_scope(self.root, request=self.request, run_root=self.root, pool=self.pool):
                pass
        self.assertIs(caught.exception, self.cleanup_error)

    def test_cancelled_primary_is_not_replaced_by_account_closure_failure(self):
        primary = asyncio.CancelledError("original cancellation")
        with self.assertRaises(asyncio.CancelledError) as caught:
            with probes._pair_account_scope(self.root, request=self.request, run_root=self.root, pool=self.pool):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertTrue((self.root / "pair-cleanup-failure.json").exists())


if __name__ == "__main__":
    unittest.main()
