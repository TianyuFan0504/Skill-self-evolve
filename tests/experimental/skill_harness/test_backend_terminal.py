"""CPU-only owned-request cancellation races over the real core client hooks."""

import asyncio
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_PATH = Path(
    os.environ.get(
        "SKILL_BACKEND_TERMINAL_PATH",
        Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/backend_terminal.py",
    )
)
SPEC = importlib.util.spec_from_file_location("verl.experimental.skill_harness.alfworld.backend_terminal", MODULE_PATH)
terminal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = terminal
SPEC.loader.exec_module(terminal)


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class Server:
    def __init__(self):
        self.requests, self.aborts = {}, []
        self.registration = asyncio.Event()
        self.registration.set()
        self.generate = Remote(self._generate)
        self.abort_request = Remote(self._abort)

    async def _generate(self, request_id, **kwargs):
        state = {"kwargs": kwargs, "finished": asyncio.Event(), "registered": False, "stop": "completed"}
        self.requests[request_id] = state
        await self.registration.wait()
        state["registered"] = True
        await state["finished"].wait()
        return SimpleNamespace(
            token_ids=[17, 19], log_probs=[-0.25, -0.75], stop_reason=state["stop"], extra_fields={"global_steps": 7}
        )

    async def _abort(self, request_id, reset_prefix_cache):
        self.aborts.append((request_id, reset_prefix_cache))
        state = self.requests.get(request_id)
        if state is None or not state["registered"]:
            return {"aborted": False}
        state["stop"] = "aborted"
        state["finished"].set()
        return {"aborted": True}


class Balancer:
    def __init__(self, server):
        self.server, self.releases = server, []
        self.acquire_gate = asyncio.Event()
        self.acquire_gate.set()
        self.require_acquire_fields = Remote(self.fields)
        self.require_release_fields = Remote(self.fields)
        self.acquire_server = Remote(self.acquire)
        self.release_server = Remote(lambda **kwargs: self.releases.append(kwargs))

    async def fields(self):
        return []

    async def acquire(self, **kwargs):
        await self.acquire_gate.wait()
        return "owned-server", self.server


class BackendTerminalTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = Server()
        self.balancer = Balancer(self.server)
        config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm", full_determinism=False))
        )
        self.client = terminal.TerminalAwareLLMServerClient(
            config,
            self.balancer,
            binding_root=self.root,
            poll_seconds=0.001,
            abort_retry_seconds=0.001,
            observation_timeout_seconds=0.1,
        )
        self.states = {}

        async def observe(binding, timeout):
            value = self.states[binding.environment_session_id]
            if isinstance(value, Exception):
                raise value
            return value

        self.observer = patch.object(terminal, "_observe", side_effect=observe)
        self.observer.start()
        self.addCleanup(self.observer.stop)

    def bind(self, name):
        self.states[name] = {"done": False, "won": False, "steps": 1, "max_steps": 50}
        return terminal.bind_episode(
            self.root,
            gateway_base_url="http://127.0.0.1:8080/sessions/" + name + "/v1",
            environment_session_url="http://127.0.0.1:8081/session/" + name,
            attempt_id="attempt-" + name,
        )

    def done(self, name):
        self.states[name] = {"done": True, "won": True, "steps": 2, "max_steps": 50}

    def generate(self, name, first=1):
        return asyncio.create_task(
            self.client.generate(name, prompt_ids=[first, 2, 3], sampling_params={"logprobs": True})
        )

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait(), timeout=2)

    async def test_terminal_before_generation_refuses_without_backend_or_fake_output(self):
        binding = self.bind("early")
        self.done("early")
        with self.assertRaises(terminal.TerminalMDPStopped):
            await self.generate("early")
        self.assertEqual(self.server.requests, {})
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertTrue(receipt["expected_backend_terminal_stop"])
        self.assertTrue(receipt["backend_stop_confirmed"])
        self.assertEqual(receipt["reason"], "terminal_refusal")
        self.assertEqual(receipt["tail_archive_refs"], [])
        self.assertEqual(receipt["attempt_id"], binding.attempt_id)

    async def test_registration_race_retries_only_actual_uuid_and_preserves_tail_mu(self):
        binding = self.bind("race")
        self.server.registration.clear()
        task = self.generate("race")
        await self.until(lambda: bool(self.server.requests))
        actual = next(iter(self.server.requests))
        self.assertNotEqual(actual, "race")
        self.done("race")
        await self.until(lambda: len(self.server.aborts) >= 2)
        self.assertFalse(task.done())
        self.server.registration.set()
        with self.assertRaises(terminal.TerminalMDPStopped):
            await task
        self.assertTrue(all(request == actual and reset is False for request, reset in self.server.aborts))
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertTrue(receipt["expected_backend_terminal_stop"])
        tail = terminal._read(Path(receipt["tail_archive_refs"][0]["path"]))
        self.assertEqual(tail["prompt_ids"], [1, 2, 3])
        self.assertEqual(tail["token_ids"], [17, 19])
        self.assertEqual(tail["log_probs"], [-0.25, -0.75])
        self.assertEqual(tail["raw_stop_reason"], "aborted")
        self.assertTrue(tail["archive_only"])
        self.assertFalse(tail["behavior_gradient"])

    async def test_terminal_during_acquire_blocks_submission_and_releases_balancer(self):
        binding = self.bind("acquire")
        self.balancer.acquire_gate.clear()
        task = self.generate("acquire")
        await asyncio.sleep(0.01)
        self.done("acquire")
        await self.until(lambda: (binding.directory / "terminal.json").exists())
        self.balancer.acquire_gate.set()
        with self.assertRaises(terminal.TerminalMDPStopped):
            await task
        self.assertEqual(self.server.requests, {})
        self.assertEqual(len(self.balancer.releases), 1)

    async def test_other_owned_session_is_not_aborted(self):
        one, two = self.bind("one"), self.bind("two")
        first, second = self.generate("one", 101), self.generate("two", 202)
        await self.until(lambda: len(self.server.requests) == 2)
        target = next(key for key, value in self.server.requests.items() if value["kwargs"]["prompt_ids"][0] == 101)
        other = next(key for key in self.server.requests if key != target)
        self.done("one")
        with self.assertRaises(terminal.TerminalMDPStopped):
            await first
        self.assertFalse(second.done())
        self.assertTrue(all(key == target for key, _ in self.server.aborts))
        self.server.requests[other]["finished"].set()
        result = await second
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.log_probs, [-0.25, -0.75])
        self.assertTrue((await terminal.wait_for_terminal_drain(one, 0.2))["expected_backend_terminal_stop"])
        self.assertFalse((await terminal.wait_for_terminal_drain(two, 0.2))["expected_backend_terminal_stop"])

    async def test_frontend_cancellation_does_not_drop_owned_backend_future(self):
        binding = self.bind("disconnect")
        task = self.generate("disconnect")
        await self.until(lambda: bool(self.server.requests))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.done("disconnect")
        receipt = await terminal.wait_for_terminal_drain(binding, 1)
        self.assertTrue(receipt["backend_stop_confirmed"])
        self.assertEqual(len(receipt["tail_archive_refs"]), 1)

    async def test_timeout_never_claims_backend_stop_then_real_completion_can_confirm(self):
        binding = self.bind("timeout")
        self.server.registration.clear()
        task = self.generate("timeout")
        await self.until(lambda: bool(self.server.requests))
        self.done("timeout")
        receipt = await terminal.wait_for_terminal_drain(binding, 0.02)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertFalse(receipt["expected_backend_terminal_stop"])
        self.assertEqual(receipt["active_requests"], 1)
        self.server.registration.set()
        with self.assertRaises(terminal.TerminalMDPStopped):
            await task
        self.assertTrue((await terminal.wait_for_terminal_drain(binding, 0.2))["backend_stop_confirmed"])

    async def test_paused_owned_task_is_admitted_before_fenced_drain_can_scan(self):
        binding = self.bind("paused")
        proceed = asyncio.Event()
        original = self.client._generate_owned

        async def paused(*args):
            await proceed.wait()
            return await original(*args)

        with patch.object(self.client, "_generate_owned", side_effect=paused):
            task = self.generate("paused")
            await self.until(lambda: bool(list((binding.directory / "requests").glob("*.json"))))
            self.done("paused")
            receipt = await terminal.wait_for_terminal_drain(binding, 0.01)
            self.assertFalse(receipt["backend_stop_confirmed"])
            self.assertEqual(receipt["active_requests"], 1)
            proceed.set()
            with self.assertRaises(terminal.TerminalMDPStopped):
                await task
        self.assertEqual(self.server.requests, {})
        self.assertTrue((await terminal.wait_for_terminal_drain(binding, 0.2))["backend_stop_confirmed"])

    async def test_public_binding_rejects_external_urls_and_cross_attempt_rebinding(self):
        binding = self.bind("bound")
        with self.assertRaises(RuntimeError):
            terminal.bind_episode(
                self.root,
                gateway_base_url="http://127.0.0.1:8080/sessions/bound/v1",
                environment_session_url=binding.environment_session_url,
                attempt_id="other-attempt",
            )
        with self.assertRaises(ValueError):
            terminal.bind_episode(
                self.root,
                gateway_base_url="http://8.8.8.8/sessions/external/v1",
                environment_session_url=binding.environment_session_url,
                attempt_id="external",
            )


if __name__ == "__main__":
    unittest.main()
