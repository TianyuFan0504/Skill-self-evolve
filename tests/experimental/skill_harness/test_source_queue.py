"""Source coverage, durable lifecycle and restore uncertainty; stdlib only."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

PATH = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/source_queue.py"
SPEC = importlib.util.spec_from_file_location("_source_queue_cpu", PATH)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def source(index, **extra):
    return {
        "episode_output_dir": f"/private/run/episodes/game{index // 4}/attempt{index}",
        "producer_uid": f"prompt{index // 4}",
        "task_id": f"game{index // 4}",
        "task_type": "pick_and_place_simple",
        **extra,
    }


class SourceQueueTest(unittest.TestCase):
    def make(self, *, bound=8192):
        state, snapshots = {}, []
        persist = Mock(side_effect=lambda: snapshots.append(copy.deepcopy(state)))
        return queue.SourceQueue(state, persist=persist, max_backlog=bound), snapshots, persist

    def test_all128_episode_opportunities_and_prefix_dedup_without_path_sort_cap(self):
        q, snapshots, persist = self.make()
        sources = [source(index) for index in reversed(range(128))]
        self.assertEqual(q.offer_many([*sources, *sources, *sources], source_step=1), 128)
        self.assertEqual(persist.call_count, 2)  # Initialization plus one atomic offer group.
        claimed = q.claim_ready(128)
        self.assertEqual(
            [entry["episode_output_dir"] for _, entry in claimed], [entry["episode_output_dir"] for entry in sources]
        )
        self.assertEqual(len({key for key, _ in claimed}), 128)
        self.assertEqual(q.offer_many(sources, source_step=1), 0)
        self.assertEqual(len(snapshots[-1]["jobs"]), 128)
        self.assertEqual(q.metrics()["sources_opportunities"], 128)
        self.assertEqual(q.metrics()["sources_active"], 128)

    def test_terminal_lifecycle_and_read_coverage_metrics(self):
        q, _, _ = self.make()
        q.offer_many([source(i) for i in range(5)], source_step=2, observed_parameter_version=1)
        jobs = q.claim_ready(4)
        generated, abstained, censored, failed = [key for key, _ in jobs]
        q.mark_read(generated)
        q.finish(generated, "generated", record_id="curate-real-reference", observed_curator_version=2)
        q.mark_read(abstained)
        q.finish(abstained, "abstained", reason="no_reusable_lesson")
        q.finish(censored, "source_censored", reason="verified_zero_actions")
        q.finish(failed, "failed", reason="host_transport_error")
        metrics = q.metrics()
        self.assertEqual(
            metrics,
            {
                "sources_opportunities": 5,
                "sources_enqueued": 5,
                "sources_pending": 1,
                "sources_active": 0,
                "sources_read": 2,
                "sources_generated": 1,
                "sources_abstained": 1,
                "sources_censored": 1,
                "sources_failed": 1,
                "sources_backlog": 1,
            },
        )
        self.assertEqual(q.state["jobs"][generated]["source_step"], 2)
        self.assertEqual(q.state["jobs"][generated]["observed_parameter_version"], 1)
        self.assertEqual(q.state["jobs"][generated]["details"]["observed_curator_version"], 2)

    def test_bound_is_explicit_atomic_backpressure_not_silent_discard(self):
        q, _, persist = self.make(bound=2)
        q.offer_many([source(0)], source_step=1)
        before = copy.deepcopy(q.state)
        with self.assertRaises(queue.SourceQueueFull):
            q.offer_many([source(1), source(2)], source_step=1)
        self.assertEqual(q.state, before)
        self.assertEqual(persist.call_count, 2)
        key, _ = q.claim_ready(1)[0]
        q.finish(key, "abstained", reason="explicit_owner_abstention")
        self.assertEqual(q.offer_many([source(1), source(2)], source_step=1), 2)
        self.assertEqual(q.metrics()["sources_backlog"], 2)
        self.assertEqual(q.metrics()["sources_opportunities"], 3)

    def test_restore_preserves_running_jobs_and_never_resamples_uncaptured_launch(self):
        q, _, _ = self.make()
        q.offer_many([source(0), source(1)], source_step=3)
        active_key, active = q.claim_ready(1)[0]
        active["curator_jobs"].append({"collection_id": "pinned-job", "behavior_version": "ema:3"})
        q.persist()
        restored = json.loads(json.dumps(q.state))
        before = copy.deepcopy(restored)
        persisted = Mock()
        resumed = queue.SourceQueue(restored, persist=persisted)
        self.assertEqual(restored, before)
        persisted.assert_not_called()
        recovery = resumed.recovery_required()
        self.assertEqual([key for key, _ in recovery], [active_key])
        self.assertEqual(recovery[0][1]["curator_jobs"][0]["behavior_version"], "ema:3")
        recovery[0][1]["status"] = "queued"
        self.assertEqual(resumed.state["jobs"][active_key]["status"], "running")
        ready = resumed.claim_ready(10)
        self.assertEqual(len(ready), 1)
        self.assertNotEqual(ready[0][0], active_key)
        resumed.finish(active_key, "source_censored", reason="uncaptured_old_launch_reconciled")
        self.assertEqual(resumed.metrics()["sources_censored"], 1)

    def test_uncertain_persist_does_not_roll_claim_back_or_allow_duplicate_launch(self):
        q, snapshots, _ = self.make()
        q.offer_many([source(0)], source_step=1)

        def committed_then_failed():
            snapshots.append(copy.deepcopy(q.state))
            raise OSError("directory fsync failed after replace")

        q.persist = committed_then_failed
        with self.assertRaises(OSError):
            q.claim_ready(1)
        self.assertEqual(q.metrics()["sources_active"], 1)
        with self.assertRaises(queue.SourceQueueRecoveryRequired):
            q.claim_ready(1)
        restored = queue.SourceQueue(snapshots[-1], persist=lambda: None)
        self.assertEqual(restored.claim_ready(1), [])
        self.assertEqual(len(restored.recovery_required()), 1)

    def test_duplicate_identity_and_terminal_outcomes_cannot_be_rewritten(self):
        q, _, persist = self.make()
        q.offer_many([source(0)], source_step=1)
        with self.assertRaises(queue.SourceQueueConflict):
            q.offer_many([source(1), source(0)], source_step=2)
        self.assertEqual(q.metrics()["sources_opportunities"], 1)
        key, _ = q.claim_ready(1)[0]
        with self.assertRaises(queue.SourceQueueConflict):
            q.finish(key, "generated", record_id="not-read")
        q.mark_read(key)
        q.finish(key, "generated", record_id="original")
        calls = persist.call_count
        q.finish(key, "generated", record_id="original")
        self.assertEqual(persist.call_count, calls)
        with self.assertRaises(queue.SourceQueueConflict):
            q.finish(key, "generated", record_id="changed")

    def test_owner_can_persist_actual_entry_and_unknown_versions_stay_unknown(self):
        outer = {"source_queue": {}}
        persisted = []
        q = queue.SourceQueue(outer["source_queue"], persist=lambda: persisted.append(copy.deepcopy(outer)))
        q.offer_many([source(0)], source_step=1)
        key, entry = q.claim_ready(1)[0]
        self.assertIs(entry, outer["source_queue"]["jobs"][key])
        self.assertIsNone(entry["observed_parameter_version"])
        entry["curator_jobs"].append({"behavior_policy_version": "ema:1", "capture": None})
        q.persist()
        self.assertIsNone(persisted[-1]["source_queue"]["jobs"][key]["observed_parameter_version"])
        self.assertEqual(len(persisted[-1]["source_queue"]["jobs"][key]["curator_jobs"]), 1)

    def test_malformed_or_nonfinite_offers_do_not_partially_commit(self):
        q, _, _ = self.make()
        for bad in (
            source(0, source_step=True),
            source(0, observed_parameter_version=float("nan")),
            source(0, episode_output_dir="relative"),
            source(0, episode_output_dir="/private/../ambiguous"),
            source(0, task_id=""),
            source(0, status="generated"),
        ):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                q.offer_many([source(1), bad], source_step=1)
            self.assertEqual(q.state["jobs"], {})

    def test_restore_validation_rejects_duplicate_sequence_and_fake_generated_coverage(self):
        q, _, _ = self.make()
        q.offer_many([source(0), source(1)], source_step=1)
        invalid = copy.deepcopy(q.state)
        rows = list(invalid["jobs"].values())
        rows[1]["sequence"] = rows[0]["sequence"]
        with self.assertRaises(ValueError):
            queue.SourceQueue(invalid, persist=lambda: None)
        invalid = copy.deepcopy(q.state)
        next(iter(invalid["jobs"].values()))["status"] = "generated"
        with self.assertRaises(ValueError):
            queue.SourceQueue(invalid, persist=lambda: None)


if __name__ == "__main__":
    unittest.main()
