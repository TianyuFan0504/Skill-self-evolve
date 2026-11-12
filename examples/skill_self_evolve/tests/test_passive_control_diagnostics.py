"""Bounded local-only observer checks; no Ray, GPU, model or signals are used."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_direct_checkpoint.py"
spec = importlib.util.spec_from_file_location("_passive_control_cpu", SCRIPT)
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


class Faults:
    def __init__(self):
        self.events, self.file = [], None

    def dump_traceback_later(self, timeout, *, repeat, file, exit):
        self.file = file
        self.events.append(("start", timeout, repeat, exit, file.closed))

    def cancel_dump_traceback_later(self):
        self.events.append(("cancel", self.file.closed))


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.faults = Faults()
        self.observer = entry.PassiveControlDiagnostics(
            self.run, local_parent=self.root, interval=.01, stack_interval=123, faults=self.faults
        )
        self.addCleanup(self.observer.close)

    def test_files_are_private_and_artifact_reference_has_no_runtime_payload(self):
        observer = self.observer
        observer.start()
        self.assertEqual(stat.S_IMODE(observer.directory.stat().st_mode), 0o700)
        for path in observer.directory.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        reference = json.loads(next((self.run / "diagnostics").glob("*.json")).read_text())
        self.assertEqual(reference["index_path"], str(observer.directory / "index.json"))
        self.assertEqual(reference["pid"], os.getpid())
        self.assertEqual(set(reference), {"schema_version", "kind", "pid", "index_path", "index_sha256"})

    def test_reference_failure_keeps_the_already_armed_local_observer(self):
        blocked = self.run / "not-a-directory"
        blocked.write_text("fixture")
        self.observer._reference = blocked / "index.json"
        self.assertIs(self.observer.start(), self.observer)
        self.assertTrue(self.observer._timer_started)
        self.assertTrue(self.observer._thread.is_alive())
        self.assertFalse(self.observer._fault_file.closed)

    def test_timer_is_cancelled_before_fault_descriptor_closes_and_files_survive(self):
        self.observer.start()
        time.sleep(.03)
        self.observer.close()
        self.assertEqual(self.faults.events, [("start", 123, True, False, False), ("cancel", False)])
        self.assertTrue(self.faults.file.closed)
        self.assertTrue(self.observer.fault_path.exists())
        self.assertTrue(self.observer.snapshot_path.exists())
        self.assertFalse(self.observer._thread.is_alive())

    def test_blocked_loop_keeps_one_pending_callback_and_records_lag(self):
        scheduled = []
        loop = types.SimpleNamespace(
            is_running=lambda: True, is_closed=lambda: False,
            call_soon_threadsafe=lambda callback, submitted: scheduled.append((callback, submitted)),
        )
        self.observer.attach(types.SimpleNamespace(_joint_loop=loop))
        self.observer._tick()
        time.sleep(.005)
        self.observer._tick()
        events = [self.observer._queue.get_nowait(), self.observer._queue.get_nowait()]
        self.assertEqual(len(scheduled), 1)
        self.assertGreater(events[-1]["loop_response_pending_ms"], 0)

    def test_late_controller_and_await_frames_are_collected_without_locals_or_task_names(self):
        secret = "UNTRUSTED_PROMPT_AND_TOKEN_CONTENT_MUST_NOT_APPEAR"
        trainer = types.SimpleNamespace(_joint_controller=None)
        self.observer.attach(trainer)

        async def exercise():
            blocker = asyncio.Event()

            async def inner():
                local_prompt = secret
                await blocker.wait()
                return local_prompt

            task = asyncio.create_task(inner(), name=secret)
            await asyncio.sleep(0)
            self.observer._loop_snapshot(time.monotonic())
            first = self.observer._queue.get_nowait()
            self.assertFalse(first["controller_created"])
            trainer._joint_controller = types.SimpleNamespace(
                _learner_active=True, _ema_syncing=False, _ema_refresh_requested=True, _writes_allowed=blocker
            )
            self.observer.stage("trainer_fit")
            self.observer._loop_snapshot(time.monotonic() - .1)
            second = self.observer._queue.get_nowait()
            self.assertTrue(second["controller_created"])
            self.assertTrue(second["learner_active"])
            self.assertFalse(second["ema_syncing"])
            self.assertFalse(second["writes_allowed"])
            self.assertGreater(second["loop_lag_ms"], 90)
            self.assertNotIn(secret, json.dumps(second))
            frames = next(row["await_chain"] for row in second["tasks"] if row["task_id"] == id(task))
            self.assertTrue(frames)
            self.assertEqual(set(frames[0]), {"file", "function", "line"})
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(exercise())

    def test_loop_callback_performs_no_file_writes(self):
        async def exercise():
            with patch.object(Path, "open", side_effect=AssertionError("event-loop file write")):
                self.observer._loop_snapshot(time.monotonic())

        asyncio.run(exercise())

    def test_snapshot_failure_is_metadata_and_cannot_poison_the_loop(self):
        submitted = time.monotonic()
        self.observer._pending = submitted
        with patch.object(entry.asyncio, "all_tasks", side_effect=RuntimeError("not a training failure")):
            self.observer._loop_snapshot(submitted)
        value = self.observer._queue.get_nowait()
        self.assertEqual(value["kind"], "observer_snapshot_error")
        self.assertEqual(value["error_type"], "RuntimeError")
        self.assertIsNone(self.observer._pending)

    def test_start_failure_closes_observer_and_leaves_training_without_a_probe(self):
        events = []

        def fail():
            raise OSError("diagnostic setup fixture")

        fake = types.SimpleNamespace(start=fail, close=lambda: events.append("closed"))
        with patch.object(entry, "PassiveControlDiagnostics", return_value=fake):
            result = entry.passive_diagnostics(self.run)
        self.assertIsInstance(result, entry._NoDiagnostics)
        self.assertEqual(events, ["closed"])

    def test_real_thread_start_failure_cancels_timer_without_joining_unstarted_thread(self):
        with patch.object(entry, "PassiveControlDiagnostics", return_value=self.observer), \
                patch.object(entry.threading.Thread, "start", side_effect=RuntimeError("thread fixture")):
            result = entry.passive_diagnostics(self.run)
        self.assertIsInstance(result, entry._NoDiagnostics)
        self.assertEqual(self.faults.events[-1], ("cancel", False))
        self.assertTrue(self.observer._fault_file.closed)

    def test_failed_timer_cancel_retains_open_descriptor_instead_of_reusing_it(self):
        self.observer.start()
        with patch.object(self.faults, "cancel_dump_traceback_later", side_effect=RuntimeError("timer fixture")):
            self.observer.close()
        self.assertFalse(self.observer._fault_file.closed)
        self.assertIn(self.observer._fault_file, entry._RETAINED_FAULT_FILES)
        # The fixture timer was inert; ordinary cleanup can now cancel and close.
        self.observer.close()
        self.assertTrue(self.observer._fault_file.closed)

    def test_writer_uses_a_dedicated_thread_and_metadata_queue_is_bounded(self):
        self.observer.stage("trainer_init")
        for _ in range(20):
            self.observer._enqueue({"kind": "test_metadata"})
        self.assertEqual(self.observer._queue.qsize(), 4)
        self.assertEqual(self.observer._dropped, 16)
        self.observer.start()
        deadline = time.monotonic() + 1
        while not self.observer.snapshot_path.stat().st_size and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertGreater(self.observer.snapshot_path.stat().st_size, 0)
        self.assertNotEqual(self.observer._thread.ident, threading.get_ident())


if __name__ == "__main__":
    unittest.main()
