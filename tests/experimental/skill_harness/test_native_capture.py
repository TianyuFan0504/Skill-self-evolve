"""CPU task lifecycle tests; external agent and service calls are interfaces only."""

import asyncio
import json
import os
import shutil
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from _native_runtime_test_support import PACKAGE, accounts, task

Sandbox = task.AlfworldUserSandbox


class Response:
    def __init__(self, value, entered=None):
        self.value, self.entered = value, entered

    async def __aenter__(self):
        if self.entered:
            self.entered()
        return self

    async def __aexit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.value


class NativePolicyCommandTest(unittest.TestCase):
    def test_policy_keeps_native_binary_prompt_pipe_and_home(self):
        command = (
            "mkdir -p /tmp/codex-home-unit && printf %s dGFzaw== | base64 -d | "
            "env CODEX_HOME=/tmp/codex-home-unit /native/codex exec --json --ephemeral"
        )
        result = task._native_policy_command(
            ["bash", "-c", command], native_binary="/native/codex", instructions_file="/policy/solver instructions.md"
        )
        self.assertEqual(result[:2], ["bash", "-c"])
        prefix, native = result[2].split("/native/codex", 1)
        self.assertEqual(prefix, command.split("/native/codex", 1)[0])
        self.assertIn("skip_host_skill_discovery", native)
        self.assertIn("skills.include_instructions=false", native)
        self.assertIn("web_search=", native)
        self.assertIn(" exec --json --ephemeral", native)
        tokens = __import__("shlex").split(result[2])
        self.assertIn('model_instructions_file="/policy/solver instructions.md"', tokens)
        self.assertIn("image_generation", tokens)

    def test_unknown_command_shape_cannot_bypass_native_policy(self):
        for command in (["/native/codex", "exec"], ["bash", "-c", "/different/codex exec --json"]):
            with self.assertRaises(accounts.TaskSetupError):
                task._native_policy_command(
                    command, native_binary="/native/codex", instructions_file="/policy/solver.md"
                )


class NativeLaunchCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_budget_registration_cancellation_closes_pipe_without_starting_process(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
            self.addCleanup(shutil.rmtree, home, True)
            pool = Mock()
            pool.prepare_resource.side_effect = lambda path, **kwargs: Path(path).mkdir(mode=0o700)
            pool.command_environment.return_value = {"PATH": os.defpath}
            pool.launch_arguments.return_value = (["/usr/bin/false"], {}, None)
            sandbox = Sandbox(
                "http://127.0.0.1",
                Path(directory),
                account=types.SimpleNamespace(pw_uid=os.geteuid(), pw_gid=os.getegid()),
                account_pool=pool,
                account_lease={"lease_id": "unit"},
                attempt_id="unit",
            )
            sandbox.native_binary = "/native/codex"
            sandbox.solver_instructions = "/policy/solver.md"
            sandbox.disabled_native_features = []
            cancelled = asyncio.CancelledError("cancelled during native budget registration")
            sandbox.register_native_budget = AsyncMock(side_effect=cancelled)
            pipe_fds = []
            open_pipe = os.pipe

            def owned_pipe():
                descriptors = open_pipe()
                pipe_fds.extend(descriptors)
                return descriptors

            argv = ["bash", "-c", f"env CODEX_HOME={home} /native/codex exec --json"]
            with patch.object(task.os, "pipe", side_effect=owned_pipe):
                with patch.object(task.asyncio, "create_subprocess_exec", new_callable=AsyncMock) as launch:
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        await sandbox._exec(argv, timeout=900, workdir=directory)
                    self.assertIs(caught.exception, cancelled)
                    launch.assert_not_called()
            self.assertFalse(sandbox.native_started)
            self.assertEqual(len(pipe_fds), 2)
            for descriptor in pipe_fds:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)


class NativeCaptureFinalizationTest(unittest.IsolatedAsyncioTestCase):
    async def test_finalization_joins_pending_event_reader_without_discarding_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Sandbox(
                "http://127.0.0.1",
                Path(directory),
                account=None,
                account_pool=None,
                account_lease=None,
                attempt_id="unit",
            )
            release = asyncio.Event()
            events = []

            async def reader():
                await release.wait()
                events.append("last native event")

            sandbox.native_capture_tasks = [asyncio.create_task(reader())]
            finalizing = asyncio.create_task(sandbox.finish_capture())
            await asyncio.sleep(0)
            self.assertFalse(finalizing.done())
            release.set()
            await finalizing
            self.assertEqual(events, ["last native event"])

    async def test_reader_failure_remains_execution_error_without_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Sandbox(
                "http://127.0.0.1",
                Path(directory),
                account=None,
                account_pool=None,
                account_lease=None,
                attempt_id="unit",
            )

            async def reader():
                raise OSError("event output unavailable")

            sandbox.native_capture_tasks = [asyncio.create_task(reader())]
            with self.assertRaises(accounts.TaskExecutionError):
                await sandbox.finish_capture()


class NativeReadinessPipeTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_pipe_proof_is_joined_without_executor_threads(self):
        read_fd, write_fd = os.pipe()
        try:
            reading = asyncio.create_task(task._read_launcher_ready(read_fd))
            await asyncio.sleep(0)
            os.write(write_fd, b'{"uid":')
            await asyncio.sleep(0)
            self.assertFalse(reading.done())
            os.write(write_fd, b"12345}\n")
            self.assertEqual(await reading, b'{"uid":12345}\n')
        finally:
            os.close(read_fd)
            os.close(write_fd)

    async def test_cancelled_pipe_reader_cannot_consume_reused_descriptor(self):
        read_fd, write_fd = os.pipe()
        reused = read_fd
        reading = asyncio.create_task(task._read_launcher_ready(read_fd))
        await asyncio.sleep(0)
        reading.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reading
        os.close(read_fd)
        os.close(write_fd)
        next_read, next_write = os.pipe()
        try:
            if next_read != reused:
                os.dup2(next_read, reused)
                os.close(next_read)
            os.write(next_write, b"next-owned-proof\n")
            await asyncio.sleep(0)
            self.assertEqual(os.read(reused, 65536), b"next-owned-proof\n")
        finally:
            os.close(reused)
            os.close(next_write)

    async def test_timeout_and_oversize_proof_release_reader_registration(self):
        read_fd, write_fd = os.pipe()
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await task._read_launcher_ready(read_fd, timeout_seconds=0.001)
            os.write(write_fd, b"proof-too-long")
            with self.assertRaises(accounts.IsolationError):
                await task._read_launcher_ready(read_fd, max_bytes=4)
            self.assertEqual(os.read(read_fd, 65536), b"f-too-long")
        finally:
            os.close(read_fd)
            os.close(write_fd)


class NativeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.settings = self.root / "environment.json"
        self.instructions = self.root / "solver.md"
        self.instructions.write_text("Only solve the assigned text environment task.")
        self.settings.write_text(
            json.dumps(
                {
                    "url": "http://127.0.0.1:1",
                    "admin_token": "unit",
                    "catalog_digest": "catalog",
                    "allowed_splits": ["train"],
                    "native_policy": {"instructions_file": str(self.instructions)},
                }
            )
        )
        self.metadata = dict(
            run_dir=str(self.root),
            game_id="unit_game",
            stable_game_id="unit_game",
            split="train",
            dataset_phase="train",
            task_type="unit",
            scene_id="scene",
            relative_game_path="game",
            game_sha256="game-digest",
            catalog_digest="catalog",
        )
        self.closed = False
        self.started = asyncio.Event()
        self.draining = asyncio.Event()
        self.release_drain = asyncio.Event()
        self.release_drain.set()
        self.agent_error = None
        self.cancel_agent = False
        self.receipt_confirmed = True
        self.engine_won = False
        self.pool = Mock()
        self.pool.prepare_work_root.return_value = self.work
        self.pool.acquire.return_value = dict(
            lease_id="lease",
            token="token",
            uid=os.geteuid(),
            gid=os.getegid(),
            name="unit",
            owner={"run_root": str(self.root), "scope_id": "scope", "kind": "ordinary"},
        )
        self.pool.public.side_effect = lambda ref: {key: value for key, value in ref.items() if key != "token"}
        self.pool.account.return_value = types.SimpleNamespace(pw_uid=os.geteuid(), pw_gid=os.getegid())
        self.pool.drain.return_value = {"drained": True, "uid": os.geteuid(), "remaining_pids": []}
        self.pool.close_attempt.return_value = {
            "status": "attempt_closed",
            "backend_drained": True,
            "uid_drained": True,
        }
        self.binding = None
        backend = types.ModuleType(PACKAGE + ".backend_terminal")

        def bind_episode(directory, **kwargs):
            self.binding = types.SimpleNamespace(
                binding_id="binding",
                binding_sha256="digest",
                attempt_id=kwargs["attempt_id"],
                gateway_session_id="gateway",
                environment_session_id="environment",
            )
            return self.binding

        async def wait_for_terminal_drain(binding, **kwargs):
            self.draining.set()
            await self.release_drain.wait()
            return {
                **vars(binding),
                "schema_version": 1,
                "backend_stop_confirmed": self.receipt_confirmed,
                "active_requests": 0,
                "backend_error_types": [],
                "fenced": True,
                "status": "drained",
            }

        backend.bind_episode = bind_episode
        backend.fence_host_stop = Mock()
        backend.wait_for_terminal_drain = wait_for_terminal_drain
        self.backend = backend
        fixture = self

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

            def post(self, url, **kwargs):
                if url.endswith("/admin/create"):
                    return Response(
                        {
                            "session_id": "environment",
                            "catalog_digest": "catalog",
                            "state": {
                                "observation": "initial observation",
                                "admissible_commands": ["look"],
                                "steps": 0,
                                "max_steps": 50,
                                "won": False,
                                "done": False,
                            },
                        }
                    )
                return Response(
                    {
                        **fixture.metadata,
                        "won": fixture.engine_won,
                        "reward": float(fixture.engine_won),
                        "steps": 1,
                        "max_steps": 50,
                        "invalid_actions": 0,
                    },
                    lambda: setattr(fixture, "closed", True),
                )

        self.client = Client()

        class Agent:
            config = types.SimpleNamespace(
                model=types.SimpleNamespace(base_url="http://127.0.0.1:2/sessions/gateway/v1"),
                codex_bin="/usr/bin/codex",
            )

            async def run(self, **kwargs):
                fixture.started.set()
                if fixture.cancel_agent:
                    await asyncio.Event().wait()
                if fixture.agent_error:
                    raise fixture.agent_error
                return types.SimpleNamespace(info={"ok": True, "error": None, "exit_code": 0}, finished=True)

        self.agent = Agent()
        self.instance = task.AlfworldSkillTrainingTask()
        self.instance.config = types.SimpleNamespace(
            metadata=self.metadata, prompt=[{"role": "user", "content": "solve task"}]
        )
        self.instance.build_agent = lambda: self.agent
        self.patches = [
            patch.dict(os.environ, {"ALFWORLD_SERVER_CONFIG": str(self.settings)}),
            patch.object(task, "AccountPool", return_value=self.pool),
            patch.object(task.aiohttp, "ClientSession", return_value=self.client, create=True),
            patch.dict(__import__("sys").modules, {backend.__name__: backend}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    async def test_completed_task_preserves_authoritative_reward_without_optional_home_archive(self):
        self.engine_won = True
        with patch("builtins.print"):
            result = await self.instance.run()
        self.assertTrue(result.finished)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(result.accuracy, 1.0)
        self.assertTrue(self.closed)
        self.pool.close_attempt.assert_called_once()
        output = next((self.root / "episodes" / "unit_game").iterdir())
        self.assertEqual(json.loads((output / "native-cleanup.json").read_text())["status"], "cleaned")
        self.assertFalse((output / "native-files.tar.gz").exists())

    async def test_service_create_error_is_typed_setup_failure_without_reward(self):
        self.client.post = Mock(side_effect=OSError("environment unavailable"))
        with self.assertRaises(accounts.TaskSetupError):
            await self.instance.run()
        self.pool.acquire.assert_not_called()
        self.assertEqual(list(self.root.glob("episodes/*/*/result.json")), [])

    async def test_repeated_cancellation_waits_for_backend_environment_and_account_cleanup(self):
        self.cancel_agent = True
        self.release_drain.clear()
        running = asyncio.create_task(self.instance.run())
        await self.started.wait()
        running.cancel()
        await self.draining.wait()
        running.cancel()
        await asyncio.sleep(0)
        self.assertFalse(running.done())
        self.assertFalse(self.closed)
        self.release_drain.set()
        with self.assertRaises(asyncio.CancelledError):
            await running
        self.assertTrue(self.closed)
        self.pool.close_attempt.assert_called_once()
        self.assertEqual(list(self.work.iterdir()), [])
        output = next((self.root / "episodes" / "unit_game").iterdir())
        self.assertFalse((output / "result.json").exists())
        self.assertIn("initial observation", (output / "agent-prompt.txt").read_text())

    async def test_invalid_agent_setup_closes_resources_and_raises_refillable_setup_error(self):
        self.instance.build_agent = Mock(side_effect=ValueError("invalid agent configuration"))
        with self.assertRaises(accounts.TaskSetupError):
            await self.instance.run()
        self.assertTrue(self.closed)
        self.pool.close_attempt.assert_called_once()
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertEqual(list(self.root.glob("episodes/*/*/result.json")), [])

    async def test_execution_error_is_not_a_zero_reward(self):
        self.agent_error = RuntimeError("native transport failed")
        with self.assertRaises(accounts.TaskExecutionError):
            await self.instance.run()
        self.assertTrue(self.closed)
        self.pool.close_attempt.assert_called_once()
        self.assertEqual(list(self.root.glob("episodes/*/*/result.json")), [])

    async def test_native_setup_error_inside_agent_remains_refillable(self):
        self.agent_error = accounts.TaskSetupError("invalid native command configuration")
        with self.assertRaises(accounts.TaskSetupError):
            await self.instance.run()
        self.assertTrue(self.closed)
        self.pool.close_attempt.assert_called_once()

    async def test_unconfirmed_backend_drain_remains_isolation_error(self):
        self.receipt_confirmed = False
        with self.assertRaises(accounts.IsolationError):
            await self.instance.run()
        self.assertTrue(self.closed)
        self.pool.quarantine.assert_called_once()
        self.pool.close_attempt.assert_not_called()
        self.assertEqual(list(self.root.glob("episodes/*/*/result.json")), [])


if __name__ == "__main__":
    unittest.main()
