"""Never report a faster distributed configuration by hiding a slow/failed rank."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_microbatch.py"
SPEC = importlib.util.spec_from_file_location("microbatch_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for role in ("actor", "reference"):
            folder = self.root / role
            folder.mkdir()
            for rank in range(4):
                record = {
                    "rank": rank,
                    "status": "completed",
                    "selected_segments": [{"tokens": 100}],
                    "trials": [
                        {
                            "phase": role + "_logprob_mb1_repeat0",
                            "microbatch": 1,
                            "seconds": rank + 1,
                            "memory": {"peak_allocated_gib": rank + 2, "peak_reserved_gib": rank + 3},
                            "logprob_comparison": {"passed": True},
                        }
                    ],
                }
                (folder / f"rank_{rank}.json").write_text(json.dumps(record))

    def change(self, role, rank, **updates):
        path = self.root / role / f"rank_{rank}.json"
        data = json.loads(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data))

    def test_straggler_controls_wall_time_and_aggregate_throughput(self):
        summary = MODULE.summarize(self.root)
        group = summary["roles"]["actor"]["groups"]["actor_logprob_mb1"]
        self.assertEqual(group["last_repeat_seconds"], 4)
        self.assertEqual(group["last_repeat_tokens_per_second"], 25)
        self.assertEqual(group["peak_allocated_gib"], 5)
        self.assertTrue(summary["all_completed"] and summary["all_parity_passed"])

    def test_a_failed_rank_prevents_a_passing_summary(self):
        self.change("actor", 2, status="failed", failed_phase="ppo_forward_backward_mb2_repeat0", oom=True)
        summary = MODULE.summarize(self.root)
        self.assertFalse(summary["all_completed"])
        self.assertFalse(summary["all_parity_passed"])
        self.assertTrue(summary["roles"]["actor"]["failures"][0]["oom"])

    def test_missing_rank_does_not_turn_partial_success_into_a_result(self):
        (self.root / "actor/rank_3.json").unlink()
        with self.assertRaises(ValueError):
            MODULE.summarize(self.root)

    def test_different_input_corpora_are_not_comparable(self):
        self.change("reference", 1, selected_segments=[{"tokens": 99}])
        with self.assertRaises(ValueError):
            MODULE.summarize(self.root)


if __name__ == "__main__":
    unittest.main()
