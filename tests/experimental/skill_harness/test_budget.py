"""Synthetic CPU budget proofs and engine admission; never training evidence."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

DIRECTORY = Path(
    os.environ.get(
        "SKILL_BUDGET_MODULE_ROOT", Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld"
    )
)
PACKAGE = "_private_budget_cpu"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE] = package
budget = importlib.import_module(PACKAGE + ".budget")


def write(path, value, *, wrapped=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrapped:
        value = {"schema_version": 1, "sha256": budget.digest(value), "payload": value}
    path.write_text(json.dumps(value) + "\n")


class BudgetProofTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        attempt, environment, binding_id = "1" * 32, "2" * 32, "3" * 64
        output = root / "episodes/game" / ("attempt_" + attempt)
        output.mkdir(parents=True)
        private_binding = {
            "binding_id": binding_id,
            "attempt_id": attempt,
            "environment_session_id": environment,
            "gateway_session_id": "CPU_session",
            "capability": "4" * 64,
            "environment_session_url": "http://127.0.0.1:1/session/" + environment,
        }
        binding = {key: private_binding[key] for key in budget.BINDING_KEYS if key != "binding_sha256"}
        binding["binding_sha256"] = budget.digest(private_binding)
        account = {
            "lease_id": "5" * 32,
            "uid": 19000,
            "gid": 19000,
            "name": "alfworld_ep_000",
            "owner": {"run_root": str(root), "kind": "ordinary", "scope_id": attempt},
        }
        payload = {"schema_version": 1, "budget_seconds": 900, "binding": binding, "account": account}
        registration = budget.make_registration(
            payload,
            session_id=environment,
            run_root=root,
            output=output,
            server_instance_id="6" * 32,
            started_ns=1_000_000_000,
        )
        write(output / "budget-registration.json", registration)
        event = {
            "schema_version": 1,
            "registration_sha256": registration["registration_sha256"],
            "sequence": 1,
            "kind": "admitted",
            "checked_ns": registration["started_ns"] + 1,
            "completed_ns": registration["started_ns"] + 2,
            "step": 1,
            "action_sha256": hashlib.sha256(b"look").hexdigest(),
            "won_after": False,
            "done_after": False,
        }
        (output / "budget-events.jsonl").write_text(json.dumps(event) + "\n")
        state = {"won": False, "done": False, "steps": 1, "max_steps": 50}
        ordered = {
            **registration,
            "now_ns": registration["deadline_ns"],
            "server_closed": False,
            "closed_ns": None,
            "close_reason": None,
            "deadline_reached": True,
            "world_state": state,
            "world_state_sha256": budget.digest(state),
            "admitted_actions": 1,
            "rejected_after_deadline": 0,
            "last_admitted_ns": event["checked_ns"],
            "last_completed_ns": event["completed_ns"],
            "last_sequence": 1,
            "action_error_count": 0,
            "events_sha256": budget.sha256_file(output / "budget-events.jsonl"),
        }
        final = {
            **ordered,
            "now_ns": registration["deadline_ns"] + 1,
            "server_closed": True,
            "closed_ns": registration["deadline_ns"] + 1,
            "close_reason": "agent_finished",
        }
        write(output / "budget-ordered-state.json", ordered)
        write(output / "budget-environment.json", final)
        engine = {
            "game_id": "train_cpu",
            "game_sha256": "a" * 64,
            "split": "train",
            "catalog_digest": "b" * 64,
            **state,
            "reward": 0.0,
            "episode_budget": final,
        }
        directory = root / "terminal_bindings" / binding_id
        write(directory / "binding.json", private_binding, wrapped=True)
        write(
            directory / "terminal.json",
            {
                **{key: binding[key] for key in budget.BINDING_KEYS if key != "binding_id"},
                "capability": private_binding["capability"],
                "stop_kind": "budget_timeout",
                "reason": budget.make_budget_stop_reason(ordered),
                "terminal_state": None,
                "terminal_state_hash": None,
            },
            wrapped=True,
        )
        request = {
            "request_nonce": "7" * 32,
            "actual_request_id": "8" * 32,
            "admission_index": 0,
            "binding_sha256": binding["binding_sha256"],
            "attempt_id": attempt,
            "gateway_session_id": binding["gateway_session_id"],
            "phase": "completed",
            "backend_rpc_finished": True,
            "error_type": None,
            "raw_stop_reason": "completed",
        }
        write(directory / "requests" / (request["request_nonce"] + ".json"), request, wrapped=True)
        write(
            directory / "admission-counter.json",
            {"binding_sha256": binding["binding_sha256"], "next_index": 1},
            wrapped=True,
        )
        receipt = {
            "schema_version": 1,
            **binding,
            "stop_kind": "budget_timeout",
            "status": "host_stopped",
            "fenced": True,
            "host_stop_requested": True,
            "expected_host_stop": True,
            "backend_stop_confirmed": True,
            "active_requests": 0,
            "backend_error_types": [],
            "requests": [request],
        }
        write(output / "backend-terminal-drain.json", receipt)
        archive = output / "native-files.tar.gz"
        archive.write_bytes(b"CPU retained native-byte fixture; no model executed")
        manifest = output / "native-files-manifest.json"
        write(manifest, {"CPU": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}})
        removed = [
            {"kind": kind, "path": str(root / ("removed_" + kind)), "removed": True}
            for kind in ("workspace", "codex_home", "model_catalog")
        ]
        native_files = {
            "status": "archived_and_removed",
            "source_stats_stable": True,
            "account_lease_id": account["lease_id"],
            "agent_uid": account["uid"],
            "archive_path": str(archive),
            "archive_sha256": budget.sha256_file(archive),
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": budget.sha256_file(manifest),
            "sources": removed,
        }
        write(output / "native-files.json", native_files)
        write(
            output / "native-capture-diagnostics.jsonl",
            {
                "timed_out": True,
                "cancelled": False,
                "pid": 12345,
                "drain_timed_out": False,
                "terminal_monitor_error": None,
                "returned_exit_code": -1,
                "declared_budget_seconds": 900,
                "budget_registration_sha256": registration["registration_sha256"],
                "budget_started_ns": registration["started_ns"],
                "budget_deadline_ns": registration["deadline_ns"],
                "native_started_ns": registration["started_ns"] + 1,
            },
        )
        (output / "codex-events.jsonl").write_text(
            json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}) + "\n"
        )
        task = {
            **engine,
            "accuracy": 0.0,
            "attempt_id": attempt,
            "environment_session_id": environment,
            "budget_exhausted": True,
            "task_cancelled": False,
            "native_finished": False,
            "agent_ok": False,
            "agent_error": "agent process timed out",
            "agent_exit_code": -1,
            "agent_exit_status": "timeout",
            "budget_reward_eligible": False,
            "backend_terminal": receipt,
            "backend_binding": binding,
            "backend_stop_confirmed": True,
            "account_isolation": {
                **account,
                "policy_version": "alfworld-uid-landlock-v1",
                "status": "attempt_closed",
                "backend_drained": True,
                "uid_drained": True,
                "archive_receipt": str(output / "native-files.json"),
            },
        }
        return types.SimpleNamespace(
            root=root,
            output=output,
            task=task,
            engine=engine,
            registration=registration,
            ordered=ordered,
            final=final,
            directory=directory,
            request=request,
            receipt=receipt,
            native_files=native_files,
            payload=payload,
            event=event,
        )

    def verify(self, value):
        return budget.verify_budget_failure(task=value.task, engine=value.engine, output_dir=value.output)

    def test_proven_budget_zero_preserves_native_error_finished_and_raw_world(self):
        value = self.fixture()
        original = copy.deepcopy(value.task)
        before = {str(p): budget.sha256_file(p) for p in value.root.rglob("*") if p.is_file()}
        proof = self.verify(value)
        self.assertEqual(proof["score"], 0.0)
        self.assertEqual(proof["status"], "observed_budget_failure")
        self.assertEqual(value.task, original)
        self.assertFalse(value.engine["done"])
        self.assertFalse(value.task["native_finished"])
        self.assertEqual(before, {str(p): budget.sha256_file(p) for p in value.root.rglob("*") if p.is_file()})

    def test_cancellation_infrastructure_error_and_world_success_never_become_zero(self):
        for field, changed in (
            ("task_cancelled", True),
            ("native_finished", None),
            ("backend_terminal_error", "EngineDeadError"),
            ("account_isolation_error", "retained"),
        ):
            value = self.fixture()
            value.task[field] = changed
            with self.subTest(field=field), self.assertRaises(budget.BudgetOutcomeUnknown):
                self.verify(value)
        value = self.fixture()
        value.task["won"] = value.engine["won"] = True
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)

    def test_unconfirmed_or_missing_owned_requests_do_not_pass_boolean_flags(self):
        value = self.fixture()
        value.receipt.update(active_requests=1, backend_stop_confirmed=False)
        write(value.output / "backend-terminal-drain.json", value.receipt)
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)
        value = self.fixture()
        (value.directory / "requests" / (value.request["request_nonce"] + ".json")).unlink()
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)

    def test_unknown_raw_stop_and_native_transport_errors_are_censored(self):
        value = self.fixture()
        value.request["raw_stop_reason"] = "unknown"
        write(value.directory / "requests" / (value.request["request_nonce"] + ".json"), value.request, wrapped=True)
        write(value.output / "backend-terminal-drain.json", value.receipt)
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)
        value = self.fixture()
        (value.output / "codex-events.jsonl").write_text(
            json.dumps({"type": "error", "message": "network unavailable"})
        )
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)

    def test_missing_ordered_clock_or_retained_sources_cannot_retroactively_complete(self):
        value = self.fixture()
        (value.output / "budget-ordered-state.json").unlink()
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)
        value = self.fixture()
        Path(value.native_files["sources"][0]["path"]).mkdir()
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)

    def test_a_late_action_is_rejected_even_after_rehashing_private_files(self):
        value = self.fixture()
        value.event["checked_ns"] = value.registration["deadline_ns"]
        value.event["completed_ns"] = value.registration["deadline_ns"] + 1
        (value.output / "budget-events.jsonl").write_text(json.dumps(value.event) + "\n")
        for receipt in (value.ordered, value.final):
            receipt.update(
                events_sha256=budget.sha256_file(value.output / "budget-events.jsonl"),
                last_admitted_ns=value.event["checked_ns"],
                last_completed_ns=value.event["completed_ns"],
            )
        write(value.output / "budget-ordered-state.json", value.ordered)
        write(value.output / "budget-environment.json", value.final)
        with self.assertRaises(budget.BudgetOutcomeUnknown):
            self.verify(value)

    def test_archive_only_aborted_tail_preserves_exact_tokens_and_probabilities(self):
        value = self.fixture()
        tail = {
            "binding_sha256": value.request["binding_sha256"],
            "attempt_id": value.request["attempt_id"],
            "gateway_session_id": value.request["gateway_session_id"],
            "actual_request_id": value.request["actual_request_id"],
            "stop_kind": "budget_timeout",
            "prompt_ids": [10, 11],
            "token_ids": [20, 21],
            "log_probs": [-0.2, -0.4],
            "raw_stop_reason": "aborted",
            "archive_only": True,
            "behavior_gradient": False,
            "model_eos_invented": False,
        }
        path = value.directory / "tails" / (value.request["request_nonce"] + ".json")
        write(path, tail, wrapped=True)
        value.request.update(
            phase="host_tail_archived",
            stop_kind="budget_timeout",
            raw_stop_reason="aborted",
            abort_acknowledged=True,
            abort_attempts=1,
            tail_archive_ref={
                "path": str(path),
                "sha256": budget.digest(tail),
                "token_count": 2,
                "raw_stop_reason": "aborted",
                "archive_only": True,
            },
        )
        write(value.directory / "requests" / (value.request["request_nonce"] + ".json"), value.request, wrapped=True)
        write(value.output / "backend-terminal-drain.json", value.receipt)
        self.assertEqual(self.verify(value)["status"], "observed_budget_failure")
        original = path.read_bytes()
        for wrong in ([-0.2], [0.1, -0.4]):
            tail["log_probs"] = wrong
            write(path, tail, wrapped=True)
            value.request["tail_archive_ref"]["sha256"] = budget.digest(tail)
            write(
                value.directory / "requests" / (value.request["request_nonce"] + ".json"), value.request, wrapped=True
            )
            write(value.output / "backend-terminal-drain.json", value.receipt)
            with self.subTest(probabilities=wrong), self.assertRaises(budget.BudgetOutcomeUnknown):
                self.verify(value)
        self.assertNotEqual(path.read_bytes(), original)

    def test_official_audit_rechecks_budget_proof_before_including_zero_in_full_denominator(self):
        harness_package = PACKAGE + "_harness"
        package = types.ModuleType(harness_package)
        package.__path__ = [str(DIRECTORY.parent)]
        sys.modules[harness_package] = package
        audit_module = importlib.import_module(harness_package + ".validation_audit")
        for forged in (False, True):
            value = self.fixture()
            audit = audit_module.ValidationCaseAudit(
                run_root=value.root,
                output_dir=value.root / "evaluation",
                stage="initial",
                phase="test_seen",
                split="train",
                global_step=0,
                expected_ids={"train_cpu"},
                catalog_digest=value.engine["catalog_digest"],
            )
            tools = {"task": {"name": "alfworld_skill_training", "metadata": {
                "game_id": "train_cpu", "game_sha256": value.engine["game_sha256"],
                "split": "train", "dataset_phase": "test_seen", "catalog_digest": value.engine["catalog_digest"],
                "run_dir": str(value.root),
            }}}
            metadata = audit.submit(["uid-budget"], [tools])[0]["task"]["metadata"]
            value.task.update(metadata)
            value.task.update(budget_reward_eligible=True, budget_failure_proof=self.verify(value))
            write(value.output / "attempt.json", value.task)
            write(value.output / "engine-result.json", value.engine)
            if forged:
                value.task["backend_stop_confirmed"] = False
            write(value.output / "result.json", value.task)
            row = {
                "rollout_uid": "uid-budget", "session_index": 0, "trajectory_key": "uid-budget_0_0",
                "score": 0.0, "sampled_action_tokens": 1, "min_global_steps": 0, "max_global_steps": 0,
                "reward_context": {**metadata, "episode_output_dir": str(value.output)},
            }
            audit.capture([row])
            dump = value.root / "validation.jsonl"
            write(dump, {"uid": "uid-budget_0_0", "score": 0.0, "acc": 0.0})
            report = audit.finish(dump_path=dump, require_complete=False)
            with self.subTest(forged=forged):
                self.assertEqual(report["complete"], not forged)
                self.assertEqual(report["observed_budget_failure_cases"], 0 if forged else 1)
                if not forged:
                    self.assertEqual(report["unexpected_native_error_cases"], 0)
                    self.assertEqual(report["authoritative_accuracy"], 0.0)


class BudgetServerTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = importlib.import_module(PACKAGE + ".server")
        except ModuleNotFoundError as error:
            if error.name != "aiohttp":
                raise
            raise unittest.SkipTest("Use the existing server CPU environment for aiohttp contracts") from None

    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.now = 1_000_000_000
        self.service = self.server.GameService([], self.root, self.root, clock_ns=lambda: self.now)
        self.identity = "2" * 32
        self.output = self.root / ("attempt_" + "1" * 32)
        self.output.mkdir()
        self.calls = []

        def step(action):
            self.calls.append(action)
            return {"feedback": "unchanged", "admissible_commands": ["look"], "won": False}, 0.0, False

        self.engine = types.SimpleNamespace(step=step, close=lambda: None)
        self.session = {
            "id": self.identity,
            "game": types.SimpleNamespace(to_dict=lambda: {"game_id": "train_CPU"}),
            "output": self.output,
            "env": self.engine,
            "state": {"feedback": "initial", "admissible_commands": ["look"], "won": False},
            "steps": 0,
            "won": False,
            "done": False,
            "invalid_actions": 0,
            "started": 0.0,
            "lock": asyncio.Lock(),
            "budget": None,
        }
        self.service.sessions[self.identity] = self.session
        self.payload = {
            "schema_version": 1,
            "budget_seconds": 900,
            "binding": {
                "binding_id": "3" * 64,
                "binding_sha256": "4" * 64,
                "attempt_id": "1" * 32,
                "gateway_session_id": "CPU_session",
                "environment_session_id": self.identity,
            },
            "account": {
                "lease_id": "5" * 32,
                "uid": 19000,
                "gid": 19000,
                "name": "alfworld_ep_000",
                "owner": {"kind": "ordinary", "run_root": str(self.root), "scope_id": "CPU"},
            },
        }

    def request(self, payload=None, *, authorized=True):
        async def body():
            return copy.deepcopy(self.payload if payload is None else payload)

        return types.SimpleNamespace(
            match_info={"session": self.identity},
            json=body,
            headers={"X-Admin-Token": self.service.admin_token} if authorized else {},
        )

    async def start(self):
        return json.loads((await self.service.start_budget(self.request())).body)

    async def test_registration_is_private_immutable_and_preserves_public_initial_input(self):
        before = self.service.state(self.session)
        with self.assertRaises(self.server.web.HTTPForbidden):
            await self.service.start_budget(self.request(authorized=False))
        registration = await self.start()
        self.now += 30_000_000_000
        self.assertEqual(await self.start(), registration)
        changed = copy.deepcopy(self.payload)
        changed["budget_seconds"] = 899
        with self.assertRaises(self.server.web.HTTPConflict):
            await self.service.start_budget(self.request(changed))
        self.assertEqual(self.service.state(self.session), before)
        self.assertEqual(self.service.budget_seconds, 900)
        with self.assertRaises(self.server.web.HTTPForbidden):
            await self.service.observe_budget(self.request(authorized=False))

    async def test_late_actions_cannot_increment_steps_or_fake_world_done(self):
        with self.assertRaises(self.server.web.HTTPConflict):
            await self.service.step(self.request({"action": "look"}))
        registration = await self.start()
        self.now = registration["deadline_ns"] - 1
        await self.service.step(self.request({"action": "look"}))
        self.now = registration["deadline_ns"]
        with self.assertRaises(self.server.web.HTTPConflict):
            await self.service.step(self.request({"action": "look"}))
        self.assertEqual(self.calls, ["look"])
        self.assertEqual(self.session["steps"], 1)
        self.assertFalse(self.session["done"])
        snapshot = json.loads((await self.service.observe_budget(self.request())).body)
        self.assertTrue(snapshot["deadline_reached"])
        self.assertEqual(snapshot["rejected_after_deadline"], 1)

    async def test_predeadline_observation_is_readonly_and_first_deadline_view_is_immutable(self):
        registration = await self.start()
        initial = json.loads((await self.service.observe_budget(self.request())).body)
        self.assertFalse(initial["deadline_reached"])
        proof_path = self.output / "budget-ordered-state.json"
        self.assertFalse(proof_path.exists())
        self.now = registration["deadline_ns"]
        first = json.loads((await self.service.observe_budget(self.request())).body)
        original_bytes = proof_path.read_bytes()
        self.now += 123
        with self.assertRaises(self.server.web.HTTPConflict):
            await self.service.step(self.request({"action": "look"}))
        later = json.loads((await self.service.observe_budget(self.request())).body)
        self.assertEqual(first, later)
        self.assertEqual(proof_path.read_bytes(), original_bytes)
        self.assertEqual(self.session["budget"]["rejected"], 1)
        self.assertEqual(later["rejected_after_deadline"], 0)

    async def test_executor_queue_delay_is_checked_at_actual_engine_entry(self):
        registration = await self.start()

        async def delayed(function, *args):
            self.now = registration["deadline_ns"]
            return function(*args)

        with mock.patch.object(self.server.asyncio, "to_thread", side_effect=delayed):
            with self.assertRaises(self.server.web.HTTPConflict):
                await self.service.step(self.request({"action": "look"}))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.session["steps"], 0)

    async def test_predeadline_admitted_success_resolves_before_ordered_deadline_snapshot(self):
        registration = await self.start()
        self.now = registration["deadline_ns"] - 1

        def winning(_action):
            self.now = registration["deadline_ns"] + 1
            return {"feedback": "won", "admissible_commands": [], "won": True}, 1.0, True

        self.engine.step = winning
        await self.service.step(self.request({"action": "look"}))
        snapshot = json.loads((await self.service.observe_budget(self.request())).body)
        self.assertTrue(snapshot["world_state"]["won"])
        self.assertTrue(snapshot["deadline_reached"])
        self.assertLess(snapshot["last_admitted_ns"], registration["deadline_ns"])
        engine = json.loads((await self.service.close(self.request())).body)
        self.assertTrue(engine["won"])
        self.assertEqual(engine["reward"], 1.0)
        self.assertTrue(engine["done"])

    async def test_cancelled_handler_keeps_lock_until_engine_resolves_and_marks_unknown(self):
        registration = await self.start()
        self.now = registration["deadline_ns"] - 1
        entered, release = threading.Event(), threading.Event()

        def blocked(_action):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("CPU engine fixture was not released")
            return {"feedback": "won after admitted action", "admissible_commands": [], "won": True}, 1.0, True

        self.engine.step = blocked
        action = asyncio.create_task(self.service.step(self.request({"action": "look"})))
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        self.assertTrue(entered.is_set())
        action.cancel()
        self.now = registration["deadline_ns"] + 1
        observation = asyncio.create_task(self.service.observe_budget(self.request()))
        await asyncio.sleep(0.01)
        self.assertFalse(action.done())
        self.assertFalse(observation.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await action
        snapshot = json.loads((await observation).body)
        self.assertEqual(snapshot["admitted_actions"], 1)
        self.assertTrue(snapshot["world_state"]["won"])
        self.assertEqual(snapshot["world_state"]["steps"], 1)
        self.assertEqual(snapshot["action_error_count"], 1)


if __name__ == "__main__":
    unittest.main()
