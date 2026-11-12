"""Host-only Task clock registration, ordered stop race, and censor contracts."""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_native_capture as capture

DIRECTORY = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld"
PACKAGE = "_task_budget_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE] = package
budget = importlib.import_module(PACKAGE + ".budget")
guard = importlib.import_module(PACKAGE + ".terminal_guard")
accounts = importlib.import_module(PACKAGE + ".accounts")
base_stub = types.ModuleType("verl.workers.rollout.llm_server")
base_stub.LLMServerClient = object
with patch.dict(sys.modules, {base_stub.__name__: base_stub}):
    backend = importlib.import_module(PACKAGE + ".backend_terminal")
sys.modules[PACKAGE + ".backend_terminal"] = backend

tree = ast.parse((DIRECTORY / "task.py").read_text())
nodes = [
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef | ast.AsyncFunctionDef)
    and node.name
    in ("AlfworldUserSandbox", "AlfworldSkillTrainingTask", "_register_native_budget", "_observe_native_budget")
]
scope = {
    "__package__": PACKAGE,
    "Task": object,
    "TaskConfig": object,
    "LocalSandbox": object,
    "register_task": lambda name: lambda cls: cls,
    "TASK_NAME": "cpu",
    "Path": Path,
    "time": time,
    "json": json,
    "os": os,
    "asyncio": asyncio,
    "AccountIsolationError": accounts.AccountIsolationError,
    "record_isolation_incident": accounts.record_isolation_incident,
    "_validated_state": guard._validated_state,
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DIRECTORY / "task.py"), "exec"), scope)


class Response:
    def __init__(self, payload):
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.payload


class TaskBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.attempt = "c" * 32
        self.session = "e" * 32
        self.output = self.root / "episodes" / "game" / ("attempt_" + self.attempt)
        self.output.mkdir(parents=True, mode=0o700)
        self.binding = backend.bind_episode(
            self.root / "terminal_bindings",
            gateway_base_url="http://127.0.0.1:9/sessions/cpu_gateway/v1",
            environment_session_url="http://127.0.0.1:8/session/" + self.session,
            attempt_id=self.attempt,
        )
        self.account = {
            "lease_id": "d" * 32,
            "uid": 19002,
            "gid": 19002,
            "name": "alfworld_ep_002",
            "owner": {"kind": "ordinary", "run_root": str(self.root.resolve()), "scope_id": self.attempt},
        }
        self.payloads = []
        self.registration = None
        self.snapshot = None
        test = self

        class Client:
            def post(self, url, *, headers, json):
                test.payloads.append((url, headers, json))
                value = budget.make_registration(
                    json,
                    session_id=test.session,
                    run_root=test.root,
                    output=test.output,
                    server_instance_id="f" * 32,
                    started_ns=time.monotonic_ns(),
                )
                test.registration = value
                return Response(value)

            def get(self, url, *, headers):
                test.payloads.append((url, headers))
                return Response(test.snapshot)

        self.client = Client()
        self.headers = {"X-Admin-Token": "host-only-fixture"}

    async def register(self, timeout=900):
        return await scope["_register_native_budget"](
            self.client,
            base="http://127.0.0.1:8",
            session_id=self.session,
            headers=self.headers,
            binding=self.binding,
            account=self.account,
            run_root=self.root,
            output=self.output,
            timeout=timeout,
            budget_seconds=900,
        )

    async def sandbox(self, *, won=False, done=False):
        registration = await self.register()
        state = {"won": won, "done": done, "steps": 3, "max_steps": 50}
        self.snapshot = {
            **registration,
            "server_closed": False,
            "deadline_reached": True,
            "now_ns": registration["deadline_ns"] + 1,
            "world_state": state,
            "world_state_sha256": budget.digest(state),
        }
        sandbox = object.__new__(scope["AlfworldUserSandbox"])
        sandbox.backend_binding = self.binding
        sandbox.budget_registration = registration
        sandbox.output = self.output
        sandbox.host_stop_evidence = []

        async def observe():
            return await scope["_observe_native_budget"](
                self.client, base="http://127.0.0.1:8", session_id=self.session, headers=self.headers
            )

        sandbox.observe_native_budget = observe
        return sandbox

    async def test_registration_validates_exact_protocol_binding_and_private_lease(self):
        registration = await self.register()
        self.assertEqual(registration["budget_seconds"], 900)
        self.assertEqual(registration["deadline_ns"] - registration["started_ns"], 900_000_000_000)
        self.assertEqual(registration["account"], self.account)
        self.assertEqual(self.payloads[0][0], "http://127.0.0.1:8/admin/budget/" + self.session + "/start")
        self.assertEqual(self.payloads[0][1], self.headers)
        self.assertEqual(registration["binding"]["binding_sha256"], self.binding.binding_sha256)
        with self.assertRaises(accounts.AccountIsolationError):
            await self.register(timeout=901)
        self.assertEqual(len(self.payloads), 1)

    async def test_changed_server_deadline_ack_is_rejected_before_native(self):
        original = self.client.post

        def changed(*args, **kwargs):
            response = original(*args, **kwargs)
            return Response({**response.payload, "deadline_ns": response.payload["deadline_ns"] + 1})

        self.client.post = changed
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            await self.register()

    async def test_nonterminal_budget_fence_binds_the_actual_ordered_snapshot(self):
        sandbox = await self.sandbox()
        evidence = await sandbox._budget_deadline_stop()
        proof = backend._fenced(self.binding)
        self.assertEqual(proof["stop_kind"], "budget_timeout")
        self.assertIsNone(proof["terminal_state"])
        self.assertEqual(proof["reason"], budget.make_budget_stop_reason(self.snapshot))
        self.assertEqual(evidence["effective_stop_kind"], "budget_timeout")

    async def test_ordered_predeadline_success_stays_real_mdp_success(self):
        sandbox = await self.sandbox(won=True, done=True)
        evidence = await sandbox._budget_deadline_stop()
        proof = backend._fenced(self.binding)
        self.assertEqual(proof["stop_kind"], "mdp_terminal")
        self.assertIs(proof["terminal_state"]["won"], True)
        self.assertEqual(evidence["requested_stop_kind"], "budget_timeout")
        self.assertEqual(evidence["effective_stop_kind"], "mdp_terminal")

    async def test_unconfirmed_clock_does_not_write_a_budget_fence(self):
        sandbox = await self.sandbox()
        self.snapshot["deadline_reached"] = False
        self.snapshot["now_ns"] = self.registration["deadline_ns"] - 1
        with self.assertRaises(accounts.AccountIsolationError):
            await sandbox._budget_deadline_stop()
        self.assertIsNone(backend._fenced(self.binding))

    async def test_only_safely_closed_cf_unknown_may_censor_without_global_incident(self):
        for role, safe, fatal in (
            ("solver", True, True),
            ("counterfactual_probe", False, True),
            ("counterfactual_probe", True, False),
        ):
            with self.subTest(role=role, safe=safe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = object.__new__(scope["AlfworldSkillTrainingTask"])
                task.config = types.SimpleNamespace(metadata={"run_dir": directory, "role": role})

                async def unknown(safely_closed=safe):
                    error = budget.BudgetOutcomeUnknown("actual closure proof is insufficient")
                    error.safely_closed_budget_censor = safely_closed
                    raise error

                task._run_episode = unknown
                with patch.dict(os.environ, {"SKILL_RUN_DIR": directory}):
                    with self.assertRaises(accounts.AccountIsolationError if fatal else budget.BudgetOutcomeUnknown):
                        await task.run()
                self.assertEqual((root / accounts.INCIDENT_DIRECTORY).exists(), fatal)


class NativeBudgetOrderTest(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = capture.NativeCaptureTest.asyncSetUp
    asyncTearDown = capture.NativeCaptureTest.asyncTearDown
    execute = capture.NativeCaptureTest.execute
    stubborn_script = capture.NativeCaptureTest.stubborn_script
    diagnostic = capture.NativeCaptureTest.diagnostic

    async def test_registration_ack_precedes_popen_and_timer_uses_remaining_deadline(self):
        calls = []
        original_exec = asyncio.create_subprocess_exec

        async def register(timeout):
            calls.append("registered")
            now = time.monotonic_ns()
            return {
                "started_ns": now - 1_800_000_000,
                "deadline_ns": now + 200_000_000,
                "budget_seconds": timeout,
                "registration_sha256": "a" * 64,
            }

        async def launch(*args, **kwargs):
            self.assertEqual(calls, ["registered"])
            calls.append("native_popen")
            return await original_exec(*args, **kwargs)

        self.sandbox.register_native_budget = register
        started = time.monotonic()
        with patch.object(asyncio, "create_subprocess_exec", side_effect=launch):
            result = await self.execute(self.stubborn_script(self.root / "budget-child.pid"), timeout=2)
        self.assertEqual(result.exit_code, -1)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(calls, ["registered", "native_popen"])
        self.assertEqual(self.requested_host_stops, ["budget_timeout"])
        self.assertEqual(self.diagnostic()["declared_budget_seconds"], 2)  # Fixture scale only; product is900.

    async def test_unknown_ordered_snapshot_falls_back_to_fatal_host_stop_before_signal(self):
        async def unconfirmed():
            raise RuntimeError("private server clock unavailable")

        self.sandbox._budget_deadline_stop = unconfirmed
        self.sandbox._request_backend_stop = lambda kind, **kwargs: self.requested_host_stops.append(kind)
        original_signal = os.killpg

        def signal_after_host_fence(pid, number):
            self.assertEqual(self.requested_host_stops, ["host_error"])
            return original_signal(pid, number)

        with patch.object(os, "killpg", side_effect=signal_after_host_fence):
            with self.assertRaises(RuntimeError):
                await self.execute(self.stubborn_script(self.root / "unknown-clock.pid"), timeout=0.2)
        self.assertEqual(self.requested_host_stops, ["host_error"])


if __name__ == "__main__":
    unittest.main()
