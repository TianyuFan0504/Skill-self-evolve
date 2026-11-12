"""CPU failure archives; no native process, Ray actor, backend or reward is run."""

from __future__ import annotations

import ast
import asyncio
import importlib
import importlib.util
import logging
import os
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
SOURCE = Path(os.environ["UNI_AGENT_SOURCE_ROOT"]) if os.environ.get("UNI_AGENT_SOURCE_ROOT") else None


def official_class(path, original, methods, *, namespace):
    """Execute unchanged lifecycle method bodies without importing Ray/HTTP."""
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == original)
    selected = [node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name in methods]
    if len(selected) != len(methods):
        raise AssertionError("Installed public Gateway lifecycle API changed")
    fixture = ast.ClassDef(name=original, bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fixture], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[original]


def installed_lifecycle_manager(session_id, rows, *, before_finalize=None):
    """Use the pinned real Trajectory and three actual public lifecycle layers."""
    spec = importlib.util.spec_from_file_location(
        PACKAGE + "._official_types", SOURCE / "uni_agent/gateway/session/types.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    namespace = {
        "SessionPhase": types.SimpleNamespace(ACTIVE="ACTIVE", ABORTED="ABORTED", FINALIZED="FINALIZED"),
        "Trajectory": module.Trajectory,
        "logger": logging.getLogger(__name__),
    }
    session_cls = official_class(
        SOURCE / "uni_agent/gateway/session/session.py", "GatewaySession", {"finalize", "abort"}, namespace=namespace
    )
    actor_cls = official_class(
        SOURCE / "uni_agent/gateway/gateway.py",
        "_GatewayActor",
        {"finalize_session", "abort_session"},
        namespace=namespace,
    )
    manager_cls = official_class(
        SOURCE / "uni_agent/gateway/manager.py",
        "GatewayManager",
        {"finalize_session", "abort_session"},
        namespace=namespace,
    )
    session = session_cls()
    session.handle = module.SessionHandle(session_id, "http://127.0.0.1:1/sessions/" + session_id + "/v1")
    session.request_lock = asyncio.Lock()
    session.phase = "ACTIVE"
    session.active_chains = []
    session.materialized_chains = [types.SimpleNamespace(order_seq=i, trajectory=row) for i, row in enumerate(rows)]
    session.reserved_chain_ids = set()
    session._coalesced_request_count = session._rollback_count = 0
    session._touch = lambda: None
    session._materialize_active_chains = lambda: None
    actor = actor_cls()
    actor._sessions = {session_id: session}
    actor._session_policies = {session_id: "frozen"}
    actor._get_session = lambda identity: actor._sessions[identity]
    calls = []

    async def finalize(**kwargs):
        calls.append("finalize")
        if before_finalize:
            before_finalize()
        return await actor.finalize_session(**kwargs)

    async def abort(**kwargs):
        calls.append("abort")
        return await actor.abort_session(**kwargs)

    remote = types.SimpleNamespace(
        finalize_session=types.SimpleNamespace(remote=finalize),
        abort_session=types.SimpleNamespace(remote=abort),
    )
    manager = manager_cls()
    manager._session_to_gateway_index = {session_id: 0}
    manager.active_sessions_per_gateway = [1]
    manager._get_gateway = lambda identity: (remote, manager._session_to_gateway_index[identity])

    async def create(identity, **kwargs):
        if identity != session_id:
            raise AssertionError("Unexpected session")
        return session.handle

    manager.create_session = create
    return manager, session, module.Trajectory, calls


@unittest.skipIf(SOURCE is None, "Set UNI_AGENT_SOURCE_ROOT to the verified offline UniAgent source")
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
            "lifecycle_id": "actual-source-CPU-fixture",
        }
        self.sample = {
            "raw_prompt": [{"role": "user", "content": "Original task"}],
            "tools_kwargs": {
                "task": {"metadata": {"credit_probe_group": "group", "credit_probe_variant": "with_skill"}}
            },
        }

    def manager(self, *, empty=False, before_finalize=None):
        manager, session, trajectory, calls = installed_lifecycle_manager(
            "session_1", [], before_finalize=before_finalize
        )
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

    async def test_actual_public_finalize_preserves_real_type_tokens_mu_and_original_cause(self):
        def before_finalize():
            self.assertEqual(self.payload("native-failure.json")["error"]["type"], "AccountIsolationError")

        manager, _, row, calls = self.manager(before_finalize=before_finalize)
        cause = RuntimeError("owned backend drain remained unknown")
        primary = probes.AccountIsolationError("Task isolation failed")

        async def runner(**kwargs):
            raise primary from cause

        with self.assertRaises(probes.AccountIsolationError) as caught:
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

    async def test_normal_success_retains_official_token_payload_and_no_failure_files(self):
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
        self.cleanup_error = probes.AccountIsolationError("UID remains quarantined")

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
        with self.assertRaises(probes.AccountIsolationError) as caught:
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
