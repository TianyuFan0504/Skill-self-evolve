"""Host-budget/cancel fences and actual first-input evidence; CPU only."""

import asyncio
import dataclasses
import errno
import unittest
from pathlib import Path
from unittest.mock import patch

import test_backend_terminal as base

terminal = base.terminal


class HostBackendStopTest(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = base.BackendTerminalTest.asyncSetUp
    bind = base.BackendTerminalTest.bind
    generate = base.BackendTerminalTest.generate
    until = base.BackendTerminalTest.until

    async def test_budget_stop_refuses_admission_without_fabricating_mdp_done(self):
        binding = self.bind("budget_early")
        proof = terminal.fence_host_stop(binding, "budget_timeout", reason="native_cli_deadline")
        self.assertIsNone(proof["terminal_state"])
        self.assertIsNone(proof["terminal_state_hash"])
        with self.assertRaises(terminal.HostBackendStopped):
            await self.generate("budget_early")
        self.assertEqual(self.server.requests, {})
        self.assertIsNone(terminal.read_first_observed_input(binding))
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertTrue(receipt["expected_host_stop"])
        self.assertEqual(receipt["stop_kind"], "budget_timeout")
        self.assertTrue(receipt["backend_stop_confirmed"])
        self.assertFalse(receipt["expected_backend_terminal_stop"])
        self.assertIsNone(receipt["terminal_state"])

    async def test_host_stop_aborts_only_owned_uuid_and_archives_partial_mu(self):
        one, two = self.bind("budget_one"), self.bind("budget_two")
        first, second = self.generate("budget_one", 101), self.generate("budget_two", 202)
        await self.until(lambda: len(self.server.requests) == 2)
        target = next(k for k, v in self.server.requests.items() if v["kwargs"]["prompt_ids"][0] == 101)
        other = next(k for k in self.server.requests if k != target)
        terminal.fence_host_stop(one, "budget_timeout")
        with self.assertRaises(terminal.HostBackendStopped):
            await first
        self.assertFalse(second.done())
        self.assertTrue(all(key == target and reset is False for key, reset in self.server.aborts))
        receipt = await terminal.wait_for_terminal_drain(one, 0.2)
        self.assertTrue(receipt["expected_host_stop"])
        self.assertFalse(receipt["expected_backend_terminal_stop"])
        tail = terminal._read(Path(receipt["tail_archive_refs"][0]["path"]))
        self.assertEqual(tail["stop_kind"], "budget_timeout")
        self.assertEqual(tail["token_ids"], [17, 19])
        self.assertEqual(tail["log_probs"], [-0.25, -0.75])
        self.assertEqual(tail["raw_stop_reason"], "aborted")
        self.assertTrue(tail["archive_only"])
        self.assertFalse(tail["behavior_gradient"])
        self.assertFalse(tail["model_eos_invented"])
        self.assertIsNone(terminal.read_first_observed_input(one))
        self.assertFalse((one.directory / "failed-inputs").exists())
        self.server.requests[other]["finished"].set()
        await second
        self.assertTrue((await terminal.wait_for_terminal_drain(two, 0.2))["backend_stop_confirmed"])

    async def test_abort_ack_is_not_original_future_or_kernel_drain(self):
        binding = self.bind("ack_only")
        task = self.generate("ack_only")
        await self.until(lambda: bool(self.server.requests))
        actual = next(iter(self.server.requests))

        async def ack_without_finish(request_id, reset_prefix_cache):
            self.server.aborts.append((request_id, reset_prefix_cache))
            return {"aborted": True}

        self.server.abort_request = base.Remote(ack_without_finish)
        terminal.fence_host_stop(binding, "task_cancel")
        await self.until(lambda: bool(self.server.aborts))
        receipt = await terminal.wait_for_terminal_drain(binding, 0.02)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertFalse(receipt["expected_host_stop"])
        self.assertEqual(receipt["active_requests"], 1)
        self.assertTrue(receipt["requests"][0]["abort_acknowledged"])
        self.assertFalse(task.done())
        self.server.requests[actual]["stop"] = "aborted"
        self.server.requests[actual]["finished"].set()
        with self.assertRaises(terminal.HostBackendStopped):
            await task
        self.assertTrue((await terminal.wait_for_terminal_drain(binding, 0.2))["backend_stop_confirmed"])

    async def test_frontend_cancel_fences_automatically_without_mdp_terminal(self):
        binding = self.bind("cancel_without_done")
        task = self.generate("cancel_without_done")
        await self.until(lambda: bool(self.server.requests))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        receipt = await terminal.wait_for_terminal_drain(binding, 1)
        self.assertTrue(receipt["backend_stop_confirmed"])
        self.assertTrue(receipt["expected_host_stop"])
        self.assertEqual(receipt["stop_kind"], "task_cancel")
        self.assertFalse(receipt["expected_backend_terminal_stop"])
        self.assertEqual(len(receipt["tail_archive_refs"]), 1)
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_first_actual_completed_input_is_durable_before_later_task_failure(self):
        binding = self.bind("first_proof")
        first = self.generate("first_proof", 711)
        await self.until(lambda: bool(self.server.requests))
        actual = next(iter(self.server.requests))
        self.assertIsNone(terminal.read_first_observed_input(binding))  # Submission is not observation.
        self.server.requests[actual]["finished"].set()
        output = await first
        proof = terminal.read_first_observed_input(binding)
        self.assertEqual(proof["payload"]["prompt_ids"], [711, 2, 3])
        self.assertEqual(proof["payload"]["actual_request_id"], actual)
        self.assertEqual(proof["payload"]["admission_index"], 0)
        self.assertEqual(
            proof["payload"]["output_sha256"],
            terminal._digest(
                {
                    "token_ids": output.token_ids,
                    "log_probs": output.log_probs,
                    "stop_reason": output.stop_reason,
                    "extra_fields": output.extra_fields,
                }
            ),
        )
        self.assertTrue(proof["payload"]["observed_on_completed_output"])
        self.assertFalse(proof["payload"]["trajectory_complete"])
        self.assertFalse(proof["payload"]["behavior_gradient"])
        second = self.generate("first_proof", 812)
        await self.until(lambda: len(self.server.requests) == 2)
        other = next(k for k in self.server.requests if k != actual)
        self.server.requests[other]["finished"].set()
        await second
        terminal.fence_host_stop(binding, "host_error", reason="later_task_python_failure")
        self.assertEqual(terminal.read_first_observed_input(binding), proof)
        self.assertFalse((binding.directory / "failed-inputs").exists())
        with self.assertRaises(terminal.HostBackendStopped):
            await self.generate("first_proof", 913)
        self.assertEqual(terminal.read_first_observed_input(binding), proof)

    async def test_backend_exception_is_not_successful_drain_or_zero_reward(self):
        binding = self.bind("engine_failed")

        async def failed(*args, **kwargs):
            raise RuntimeError("EngineDeadError fixture; no real backend")

        self.server.generate = base.Remote(failed)
        with self.assertRaisesRegex(RuntimeError, "EngineDeadError"):
            await self.generate("engine_failed")
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertEqual(receipt["status"], "backend_failed")
        self.assertEqual(receipt["backend_error_types"], ["RuntimeError"])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_engine_failure_retains_exact_client_input_without_claiming_observation(self):
        class EngineDeadError(RuntimeError):
            pass

        binding = self.bind("retained_failed_input")
        primary = EngineDeadError("injected RPC failure, not a real model")
        entered, release = asyncio.Event(), asyncio.Event()

        async def failed(*args, **kwargs):
            entered.set()
            await release.wait()
            raise primary

        self.server.generate = base.Remote(failed)
        prompts = [194, 27, 98, 6]
        params = {"temperature": 0.0, "logprobs": True, "max_tokens": 29418, "stop_token_ids": [248044]}
        task = asyncio.create_task(
            self.client.generate("retained_failed_input", prompt_ids=prompts, sampling_params=params)
        )
        await entered.wait()
        prompts[0] = 999  # Caller mutations cannot rewrite the retained submission.
        params["stop_token_ids"].append(999)
        release.set()
        with self.assertRaises(EngineDeadError) as caught:
            await task
        self.assertIs(caught.exception, primary)
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertFalse(receipt["backend_stop_confirmed"])
        request = receipt["requests"][0]
        ref = request["failed_input_ref"]
        payload = terminal._read(Path(ref["path"]))
        self.assertEqual(Path(ref["path"]), binding.directory / "failed-inputs" / (request["request_nonce"] + ".json"))
        self.assertEqual(payload["prompt_ids"], [194, 27, 98, 6])
        self.assertEqual(
            payload["sampling_params"],
            {"temperature": 0.0, "logprobs": True, "max_tokens": 29418, "stop_token_ids": [248044]},
        )
        self.assertEqual(payload["prompt_ids_sha256"], terminal._digest(payload["prompt_ids"]))
        self.assertEqual(payload["sampling_params_sha256"], terminal._digest(payload["sampling_params"]))
        self.assertEqual(ref["sha256"], terminal._digest(payload))
        self.assertEqual(payload["actual_request_id"], request["actual_request_id"])
        self.assertEqual(payload["request_nonce"], request["request_nonce"])
        self.assertEqual(payload["binding_sha256"], binding.binding_sha256)
        self.assertEqual(payload["environment_session_id"], binding.environment_session_id)
        self.assertEqual(payload["evidence_stage"], "CLIENT_INPUT_PRE_SUBMIT")
        for field in (
            "actual_model_observed",
            "behavior_gradient",
            "model_eos_invented",
            "reward_eligible",
            "cache_state_captured",
            "trajectory_complete",
        ):
            self.assertIs(payload[field], False)
        self.assertIs(payload["archive_only"], True)
        self.assertIsNone(terminal.read_first_observed_input(binding))
        self.assertEqual(receipt["tail_archive_refs"], [])

    async def test_failed_input_archive_io_error_does_not_replace_backend_exception(self):
        binding = self.bind("archive_disk_full")
        primary = RuntimeError("EngineDeadError injected primary")

        async def failed(*args, **kwargs):
            raise primary

        self.server.generate = base.Remote(failed)
        original = terminal._write

        def disk_full(path, payload):
            if path.parent.name == "failed-inputs":
                raise OSError(errno.ENOSPC, "injected evidence disk full")
            return original(path, payload)

        with patch.object(terminal, "_write", side_effect=disk_full):
            with self.assertRaises(RuntimeError) as caught:
                await self.generate("archive_disk_full")
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("evidence disk full" in note for note in getattr(primary, "__notes__", [])))
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        request = receipt["requests"][0]
        self.assertIn("evidence disk full", request["failed_input_archive_error"])
        self.assertNotIn("failed_input_ref", request)
        self.assertEqual(request["error_type"], "RuntimeError")
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_acquire_rpc_failure_archives_client_input_without_an_actual_request_id(self):
        binding = self.bind("acquire_rpc_failed")
        primary = RuntimeError("injected load-balancer RPC failure")

        async def failed_acquire(**kwargs):
            raise primary

        self.balancer.acquire_server = base.Remote(failed_acquire)
        with self.assertRaises(RuntimeError) as caught:
            await self.generate("acquire_rpc_failed", 461)
        self.assertIs(caught.exception, primary)
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        request = receipt["requests"][0]
        payload = terminal._read(Path(request["failed_input_ref"]["path"]))
        self.assertEqual(payload["prompt_ids"], [461, 2, 3])
        self.assertIsNone(payload["actual_request_id"])
        self.assertIsNone(payload["server_id"])
        self.assertIs(payload["actual_model_observed"], False)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_failure_journal_io_error_preserves_primary_and_retained_input(self):
        binding = self.bind("failure_journal_full")
        primary = RuntimeError("EngineDeadError injected primary")

        async def failed(*args, **kwargs):
            raise primary

        self.server.generate = base.Remote(failed)
        original = terminal._write

        def journal_full(path, payload):
            if path.parent.name == "requests" and payload.get("error_type"):
                raise OSError(errno.ENOSPC, "injected failure journal full")
            return original(path, payload)

        with patch.object(terminal, "_write", side_effect=journal_full):
            with self.assertRaises(RuntimeError) as caught:
                await self.generate("failure_journal_full")
        self.assertIs(caught.exception, primary)
        self.assertTrue(any("failure journal full" in note for note in getattr(primary, "__notes__", [])))
        files = list((binding.directory / "failed-inputs").glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(terminal._read(files[0])["prompt_ids"], [1, 2, 3])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_unknown_raw_stop_retains_mu_but_never_becomes_completed_output(self):
        binding = self.bind("unknown_stop")
        task = self.generate("unknown_stop")
        await self.until(lambda: bool(self.server.requests))
        actual = next(iter(self.server.requests))
        self.server.requests[actual]["stop"] = "unknown_error"
        self.server.requests[actual]["finished"].set()
        with self.assertRaisesRegex(RuntimeError, "unconfirmed backend stop"):
            await task
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertEqual(receipt["status"], "backend_failed")
        tail = terminal._read(Path(receipt["tail_archive_refs"][0]["path"]))
        self.assertEqual(tail["raw_stop_reason"], "unknown_error")
        self.assertEqual(tail["token_ids"], [17, 19])
        self.assertEqual(tail["log_probs"], [-0.25, -0.75])
        self.assertTrue(tail["archive_only"])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_unknown_stop_with_existing_budget_fence_is_still_backend_failed(self):
        binding = self.bind("fenced_unknown_stop")
        task = self.generate("fenced_unknown_stop")
        await self.until(lambda: bool(self.server.requests))
        actual = next(iter(self.server.requests))

        async def ack_only(*args, **kwargs):
            return {"aborted": True}  # ACK cannot rewrite the raw backend output.

        self.server.abort_request = base.Remote(ack_only)
        terminal.fence_host_stop(binding, "budget_timeout")
        self.server.requests[actual]["stop"] = "unknown_error"
        self.server.requests[actual]["finished"].set()
        with self.assertRaisesRegex(RuntimeError, "unconfirmed backend stop"):
            await task
        receipt = await terminal.wait_for_terminal_drain(binding, 0.2)
        self.assertFalse(receipt["backend_stop_confirmed"])
        self.assertFalse(receipt["expected_host_stop"])
        self.assertEqual(receipt["status"], "backend_failed")
        self.assertEqual(receipt["stop_kind"], "budget_timeout")
        tail = terminal._read(Path(receipt["tail_archive_refs"][0]["path"]))
        self.assertEqual(tail["raw_stop_reason"], "unknown_error")
        self.assertEqual(tail["log_probs"], [-0.25, -0.75])
        self.assertIsNone(terminal.read_first_observed_input(binding))

    async def test_first_host_stop_cannot_be_upgraded_to_mdp_terminal(self):
        binding = self.bind("immutable_stop")
        original = terminal.fence_host_stop(binding, "budget_timeout")
        later = terminal.fence_episode(binding, {"done": True, "won": True, "steps": 2, "max_steps": 50})
        self.assertEqual(later, original)
        self.assertIsNone(later["terminal_state"])
        with self.assertRaises(RuntimeError):
            terminal.fence_host_stop(dataclasses.replace(binding, attempt_id="other"), "task_cancel")


if __name__ == "__main__":
    unittest.main()
