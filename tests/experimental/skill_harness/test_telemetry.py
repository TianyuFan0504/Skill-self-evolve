"""Verify episode-level accounting rather than trajectory-row inflation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/telemetry.py"
SPEC = importlib.util.spec_from_file_location("skill_telemetry", PATH)
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


class TelemetryTest(unittest.TestCase):
    def test_prefix_splits_do_not_inflate_success(self):
        rows = [dict(uid="a", session_id=0, reward=1, accuracy=1) for _ in range(5)]
        rows += [dict(uid="b", session_id=0, reward=0, accuracy=0, error=True)]
        result = telemetry.episode_metrics(rows)
        self.assertEqual(result["train/episodes"], 2)
        self.assertEqual(result["train/accuracy"], 0.5)
        self.assertEqual(result["train/infrastructure_error_rate"], 0.5)

    def test_conflicting_terminal_outcomes_are_rejected(self):
        with self.assertRaises(ValueError):
            telemetry.episode_metrics([dict(uid="a", session_id=0, reward=r, accuracy=r) for r in (0, 1)])

    def test_padding_is_excluded_and_sessions_are_distinct(self):
        result = telemetry.episode_metrics(
            [
                dict(uid="a", session_id=0, reward=1, accuracy=1),
                dict(uid="a", session_id=1, reward=0, accuracy=0),
                dict(uid="p", session_id=0, reward=1, accuracy=1, padding=True),
            ]
        )
        self.assertEqual(result["train/accuracy"], 0.5)

    def test_jsonl_has_phases_and_no_nonstandard_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            telemetry.append_metrics(folder, step=1, phase="train", metrics={"loss": float("nan"), "acc": 0.5})
            telemetry.append_metrics(folder, step=1, phase="dev", metrics={"acc": 0.25})
            rows = [json.loads(line) for line in (Path(folder) / "metrics.jsonl").read_text().splitlines()]
            self.assertIsNone(rows[0]["metrics"]["loss"])
            self.assertEqual([r["phase"] for r in rows], ["train", "dev"])


if __name__ == "__main__":
    unittest.main()
