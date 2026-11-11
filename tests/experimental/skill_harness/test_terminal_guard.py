"""Real owned CPU subprocess tests; no endpoint, model, GPU, Ray or training."""

import asyncio
import importlib.util
import json
import os
import signal
import sys
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/terminal_guard.py"
SPEC = importlib.util.spec_from_file_location("_alfworld_terminal_guard_contract", MODULE)
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


@unittest.skipUnless(os.name == "posix", "Owned process-group contract requires POSIX")
class TerminalGuardTests(unittest.IsolatedAsyncioTestCase):
    async def spawn(self, seconds=60, exit_code=0):
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", f"import time,sys;time.sleep({seconds});sys.exit({exit_code})",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )

        async def cleanup():
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await proc.wait()

        self.addAsyncCleanup(cleanup)
        return proc

    def callbacks(self, proc, state):
        signals, received = [], []

        async def observe():
            return state

        async def terminate():
            signals.append(signal.SIGTERM)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            await proc.wait()

        return {"observe": observe, "is_alive": lambda: proc.returncode is None, "terminate": terminate,
                "process_id": proc.pid, "poll_seconds": 0.005, "observation_timeout_seconds": 0.02,
                "on_evidence": received.append}, signals, received

    async def test_proven_terminal_stops_only_the_owned_group_after_grace(self):
        proc = await self.spawn()
        callbacks, signals, received = self.callbacks(proc, {"done": True, "won": False, "steps": 50, "max_steps": 50})
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0.03)
        self.assertEqual(result.status, "terminal_stop")
        self.assertEqual(signals, [signal.SIGTERM])
        self.assertEqual(proc.returncode, -signal.SIGTERM)
        self.assertTrue(result.termination_requested)
        self.assertTrue(result.termination_completed)
        self.assertFalse(result.alive_after_termination)
        self.assertGreaterEqual(result.elapsed_after_terminal_seconds, 0.03)
        self.assertEqual(result.observations, 1)
        self.assertEqual(received, [result])
        encoded = result.to_dict()
        json.dumps(encoded, allow_nan=False)
        for fabricated in ("reward", "accuracy", "finished", "native_exit_code", "behavior_logprobs"):
            self.assertNotIn(fabricated, encoded)
        with self.assertRaises(TypeError):
            result.terminal_state["done"] = False

    async def test_normal_native_completion_within_grace_is_not_a_host_stop(self):
        proc = await self.spawn(0.04)
        callbacks, signals, _ = self.callbacks(proc, {"done": True, "won": True, "steps": 6, "max_steps": 50})
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0.25)
        self.assertEqual(result.status, "native_completed_after_terminal")
        self.assertEqual(proc.returncode, 0)
        self.assertFalse(result.termination_requested)
        self.assertEqual(signals, [])

    async def test_nonterminal_native_exit_preserves_actual_nonzero_exit(self):
        proc = await self.spawn(0.04, exit_code=7)
        callbacks, signals, _ = self.callbacks(proc, {"done": False, "won": False, "steps": 1, "max_steps": 50})
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0)
        self.assertEqual(result.status, "native_completed")
        self.assertEqual(proc.returncode, 7)
        self.assertIsNone(result.terminal_state)
        self.assertEqual(signals, [])

    async def test_unknown_or_malformed_observations_never_authorize_stop(self):
        proc = await self.spawn(0.06)
        values = iter((None, {"done": "true", "won": False}, {"done": 1, "won": True}, {"done": True}))
        callbacks, signals, _ = self.callbacks(proc, None)

        async def observe():
            return next(values, None)

        callbacks["observe"] = observe
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0)
        self.assertEqual(result.status, "native_completed")
        self.assertGreater(result.unknown_observations, 0)
        self.assertFalse(result.termination_requested)
        self.assertEqual(signals, [])

    async def test_observer_failures_are_retained_and_only_later_proof_arms_stop(self):
        proc = await self.spawn()
        callbacks, signals, _ = self.callbacks(proc, None)
        calls = 0

        async def observe():
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise OSError("temporary observer failure")
            return {"done": True, "won": False}

        callbacks["observe"] = observe
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0.01)
        self.assertEqual(result.status, "terminal_stop")
        self.assertEqual(result.observation_error_count, 2)
        self.assertEqual(result.observation_error_types, ("OSError",))
        self.assertEqual(result.observations, 3)
        self.assertEqual(signals, [signal.SIGTERM])

    async def test_observer_timeout_cannot_replace_native_completion(self):
        proc = await self.spawn(0.06)
        callbacks, signals, _ = self.callbacks(proc, None)

        async def observe():
            await asyncio.sleep(60)

        callbacks.update(observe=observe, observation_timeout_seconds=0.01)
        result = await guard.monitor_terminal_process(**callbacks, grace_seconds=0)
        self.assertEqual(result.status, "native_completed")
        self.assertGreater(result.observation_error_count, 0)
        self.assertIn("TimeoutError", result.observation_error_types)
        self.assertEqual(signals, [])

    async def test_owner_cancellation_propagates_without_stealing_cleanup(self):
        proc = await self.spawn()
        callbacks, signals, received = self.callbacks(proc, {"done": False, "won": False})
        task = asyncio.create_task(guard.monitor_terminal_process(**callbacks))
        await asyncio.sleep(0.02)
        task.cancel("owner cancellation")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(str(raised.exception), "owner cancellation")
        self.assertEqual(raised.exception.terminal_stop_evidence.status, "cancelled")
        self.assertEqual(received[0].status, "cancelled")
        self.assertEqual(signals, [])
        self.assertIsNone(proc.returncode)  # Owner/test cleanup remains responsible.

    async def test_termination_error_propagates_same_exception_with_evidence(self):
        proc = await self.spawn()
        callbacks, _, received = self.callbacks(proc, {"done": True, "won": False})
        original = RuntimeError("owned cleanup failed")

        async def terminate():
            raise original

        callbacks["terminate"] = terminate
        with self.assertRaises(RuntimeError) as raised:
            await guard.monitor_terminal_process(**callbacks, grace_seconds=0)
        self.assertIs(raised.exception, original)
        self.assertEqual(original.terminal_stop_evidence.status, "termination_error")
        self.assertTrue(original.terminal_stop_evidence.termination_requested)
        self.assertFalse(original.terminal_stop_evidence.termination_completed)
        self.assertEqual(received, [original.terminal_stop_evidence])
        self.assertIsNone(proc.returncode)

    async def test_liveness_error_does_not_authorize_any_termination(self):
        calls = []
        original = LookupError("process ownership unavailable")

        def alive():
            raise original

        with self.assertRaises(LookupError) as raised:
            await guard.monitor_terminal_process(lambda: {"done": True, "won": True}, alive, lambda: calls.append(1))
        self.assertIs(raised.exception, original)
        self.assertEqual(original.terminal_stop_evidence.status, "liveness_error")
        self.assertEqual(calls, [])

    async def test_a_returned_cleanup_without_exit_is_explicitly_unconfirmed(self):
        calls = []
        result = await guard.monitor_terminal_process(
            lambda: {"done": False, "won": True}, lambda: True, lambda: calls.append(1), grace_seconds=0,
        )
        self.assertEqual(result.status, "terminal_stop_unconfirmed")
        self.assertEqual(result.terminal_state, {"done": False, "won": True})
        self.assertEqual(calls, [1])
        self.assertTrue(result.alive_after_termination)

    async def test_evidence_sink_error_cannot_mask_original_cancellation(self):
        proc = await self.spawn()
        callbacks, signals, _ = self.callbacks(proc, {"done": False, "won": False})
        delivery_error = OSError("evidence filesystem unavailable")

        def sink(_evidence):
            raise delivery_error

        callbacks["on_evidence"] = sink
        task = asyncio.create_task(guard.monitor_terminal_process(**callbacks))
        await asyncio.sleep(0.02)
        task.cancel("owner")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertIs(raised.exception.terminal_stop_evidence_delivery_error, delivery_error)
        self.assertEqual(str(raised.exception), "owner")
        self.assertEqual(signals, [])

    async def test_normal_evidence_sink_error_is_not_silently_swallowed(self):
        original = OSError("evidence write failed")

        def sink(_evidence):
            raise original

        with self.assertRaises(OSError) as raised:
            await guard.monitor_terminal_process(lambda: None, lambda: False, lambda: None, on_evidence=sink)
        self.assertIs(raised.exception, original)
        self.assertEqual(original.terminal_stop_evidence.status, "native_completed")


if __name__ == "__main__":
    unittest.main()
