"""Host clock registration and ordered budget stop tests without native launches."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from _native_runtime_test_support import PACKAGE, accounts, budget, linux, task


class TaskBudgetTest(unittest.IsolatedAsyncioTestCase):
    async def test_provider_public_account_is_adapted_to_strict_budget_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            attempt, session_id = "1" * 32, "2" * 32
            output = root / "episodes" / ("attempt_" + attempt)
            binding = types.SimpleNamespace(
                binding_id="3" * 64,
                binding_sha256="4" * 64,
                attempt_id=attempt,
                gateway_session_id="unit_gateway",
                environment_session_id=session_id,
            )
            lease = {
                "lease_id": "5" * 32,
                "token": "private-lease-capability",
                "uid": 12345,
                "gid": 12345,
                "name": "unit_runtime_account",
                "owner": {"kind": "ordinary", "run_root": str(root), "scope_id": attempt},
            }
            public = linux.LinuxAccountPool.public(lease)
            self.assertEqual(public["policy_version"], linux.POLICY_VERSION)
            payload = {
                "schema_version": 1,
                "budget_seconds": 900,
                "binding": {key: getattr(binding, key) for key in budget.BINDING_KEYS},
                "account": public,
            }
            service_arguments = {
                "session_id": session_id,
                "run_root": root,
                "output": output,
                "server_instance_id": "6" * 32,
                "started_ns": 1_000_000_000,
            }
            # The service protocol must keep rejecting provider-specific fields.
            with self.assertRaisesRegex(budget.BudgetOutcomeUnknown, "Budget lease fields differ"):
                budget.make_registration(payload, **service_arguments)

            posted = []

            class Response:
                def __init__(self, registration):
                    self.registration = registration

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

                def raise_for_status(self):
                    pass

                async def json(self):
                    return self.registration

            def post(url, *, headers, json):
                self.assertEqual(url, "http://127.0.0.1/admin/budget/" + session_id + "/start")
                self.assertEqual(headers, {"X-Admin-Token": "unit-admin"})
                self.assertEqual(set(json["account"]), set(budget.ACCOUNT_KEYS))
                self.assertNotIn("token", json["account"])
                posted.append(json)
                registration = budget.make_registration(json, **service_arguments)
                if len(posted) == 2:
                    registration["deadline_ns"] += 1
                return Response(registration)

            arguments = {
                "base": "http://127.0.0.1",
                "session_id": session_id,
                "headers": {"X-Admin-Token": "unit-admin"},
                "binding": binding,
                "account": public,
                "run_root": root,
                "output": output,
                "timeout": 900,
                "budget_seconds": 900,
            }
            client = types.SimpleNamespace(post=post)
            registration = await task._register_native_budget(client, **arguments)
            self.assertEqual(registration["account"], {key: lease[key] for key in budget.ACCOUNT_KEYS})
            self.assertIn("policy_version", public)
            self.assertNotIn("token", public)
            # The host still verifies the real ACK rather than trusting HTTP success.
            with self.assertRaisesRegex(budget.BudgetOutcomeUnknown, "registration ACK differs"):
                await task._register_native_budget(client, **arguments)

    async def test_native_timeout_and_server_budget_must_agree(self):
        client = types.SimpleNamespace(post=lambda *args, **kwargs: self.fail("invalid budget reached service"))
        with self.assertRaises(accounts.IsolationError):
            await task._register_native_budget(
                client,
                base="http://127.0.0.1",
                session_id="episode",
                headers={},
                binding=None,
                account={},
                run_root="/tmp",
                output="/tmp",
                timeout=20,
                budget_seconds=30,
            )

    async def test_ordered_unknown_budget_snapshot_cannot_authorize_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = task.AlfworldUserSandbox(
                "http://127.0.0.1",
                Path(directory),
                account=None,
                account_pool=None,
                account_lease=None,
                attempt_id="unit",
            )
            sandbox.budget_registration = {"deadline_ns": 10}
            sandbox.observe_native_budget = AsyncMock(
                return_value={
                    "deadline_ns": 10,
                    "deadline_reached": True,
                    "server_closed": False,
                    "now_ns": 10,
                    "world_state": {"won": False},
                }
            )
            backend = types.SimpleNamespace(fence_episode=Mock())
            with patch.dict(__import__("sys").modules, {PACKAGE + ".backend_terminal": backend}):
                with self.assertRaises(accounts.IsolationError):
                    await sandbox._budget_deadline_stop()
            backend.fence_episode.assert_not_called()
            self.assertEqual(sandbox.host_stop_evidence, [])


if __name__ == "__main__":
    unittest.main()
