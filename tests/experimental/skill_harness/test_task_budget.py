"""Host clock registration and ordered budget stop tests without native launches."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from _native_runtime_test_support import PACKAGE, accounts, task


class TaskBudgetTest(unittest.IsolatedAsyncioTestCase):
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
