"""CPU boundaries for the focused native-pair driver; no Ray/model is started."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_native_pair.py"
SPEC = importlib.util.spec_from_file_location("focused_native_pair_test", SCRIPT)
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class FocusedPairBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.run = self.root / "source", self.root / "diagnostic"
        (self.source / "controller").mkdir(parents=True)
        (self.source / "skill_store/artifacts").mkdir(parents=True)
        (self.source / "data").mkdir()
        (self.source / "skill_store/artifacts/fixture.json").write_text('{"fixture":"immutable"}\n')
        (self.source / "data/split_manifest.json").write_text('{"catalog_digest":"original-catalog"}')
        (self.source / "data/train.parquet").write_bytes(b"original copied data fixture; never parsed as tasks")
        (self.source / "resolved-config.yaml").write_text("original: config\n")
        self.fields = {
            "uid": "original-row",
            "raw_prompt": [{"role": "user", "content": "Original B instruction; no invented task"}],
            "tools_kwargs": {
                "task": {
                    "name": "alfworld_skill_training",
                    "metadata": {
                        "stable_game_id": PAIR.FUTURE_TASK,
                        "game_id": PAIR.FUTURE_TASK,
                        "split": "train",
                        "dataset_phase": "train",
                        "source_step": 2,
                        "task_type": "pick_clean_then_place_in_recep",
                        "run_dir": str(self.source),
                        "catalog_digest": "original-catalog",
                    },
                }
            },
        }
        self.state = {
            "pending": {
                "window_" + PAIR.COLLECTION: {
                    "record_id": "curate-" + PAIR.COLLECTION,
                    "producer_task_id": PAIR.PRODUCER_TASK,
                    "cohort_id": "original-cohort",
                    "opened_step": 1,
                    "task_type": "pick_clean_then_place_in_recep",
                }
            },
            "cohorts": {"original-cohort": {"assignments": {"original-probe": self.fields}}},
        }
        self.write_state()

    def write_state(self):
        (self.source / "controller/state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sha256": PAIR.digest(self.state),
                    "payload": self.state,
                }
            )
        )

    def test_default_cli_only_clones_real_input_and_never_calls_execute(self):
        before = PAIR.tree_manifest(self.source)
        with patch.object(PAIR, "execute", side_effect=AssertionError("GPU execution is forbidden")) as execute:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(PAIR.main(["--source-run", str(self.source), "--run-dir", str(self.run)]), 0)
        execute.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["status"], "prepared_no_gpu")
        self.assertEqual(PAIR.tree_manifest(self.source), before)
        prepared = PAIR.validate_prepared(self.run)
        self.assertEqual(prepared["sample_fields"]["raw_prompt"], self.fields["raw_prompt"])
        self.assertEqual(prepared["original_assignment_sha256"], PAIR.digest(self.fields))
        self.assertEqual(prepared["source_store"], PAIR.tree_manifest(self.run / "skill_store"))
        self.assertFalse((self.run / "execution-intent.json").exists())

    def test_only_the_archived_downstream_train_task_is_allowed(self):
        for field, value in (
            ("split", "valid_seen"),
            ("dataset_phase", "dev"),
            ("source_step", 1),
            ("game_id", PAIR.PRODUCER_TASK),
        ):
            with self.subTest(field=field):
                metadata = self.fields["tools_kwargs"]["task"]["metadata"]
                original = metadata[field]
                metadata[field] = value
                self.write_state()
                with self.assertRaises(ValueError):
                    PAIR.archived_inputs(self.source)
                metadata[field] = original

    def test_input_symlinks_are_rejected_without_following_them(self):
        (self.source / "skill_store/escape").symlink_to(self.source / "resolved-config.yaml")
        with self.assertRaisesRegex(ValueError, "link or special"):
            PAIR.prepare_inputs(self.source, self.run)

    def test_changed_copy_and_changed_original_assignment_are_rejected(self):
        PAIR.prepare_inputs(self.source, self.run)
        path = self.run / "skill_store/artifacts/fixture.json"
        path.chmod(0o600)
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SkillStore changed"):
            PAIR.validate_prepared(self.run)
        path.write_bytes(original)
        self.fields["raw_prompt"][0]["content"] = "Changed source task"
        self.write_state()
        with self.assertRaisesRegex(ValueError, "assignment changed"):
            PAIR.validate_prepared(self.run)

    def test_existing_runs_and_second_execute_intents_never_silently_resample(self):
        PAIR.prepare_inputs(self.source, self.run)
        with self.assertRaises(FileExistsError):
            PAIR.prepare_inputs(self.source, self.run)
        PAIR.write_new(self.run / "execution-intent.json", {"already_attempted": True})
        with self.assertRaises(FileExistsError):
            PAIR.execute(self.run)

    def test_gpu_check_requires_one_explicit_idle_device(self):
        with patch.dict(PAIR.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
            with self.assertRaises(ValueError):
                PAIR.selected_gpu_state()
        with patch.dict(PAIR.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with patch.object(
                PAIR.subprocess,
                "check_output",
                side_effect=[
                    "0, GPU-fixture0\n1, GPU-fixture1\n",
                    "GPU-fixture0, 100\nGPU-fixture1, 200\n",
                ],
            ):
                self.assertEqual(PAIR.selected_gpu_state()["compute_pids"], [200])

    def test_cleanup_still_stops_owned_service_and_preserves_logs_after_ray_failure(self):
        self.run.mkdir()
        scratch, temporary = self.root / "ray", self.root / "tmp"
        (scratch / "session/logs").mkdir(parents=True)
        (scratch / "session/logs/native.log").write_text("preserved CPU fixture\n")
        temporary.mkdir()
        ray = types.SimpleNamespace(
            shutdown=Mock(side_effect=RuntimeError("CPU injected failure")), is_initialized=lambda: False
        )
        with patch.object(PAIR, "stop_owned_service", return_value={"status": "stopped"}) as stop:
            with patch.object(PAIR, "selected_gpu_state", return_value={"compute_pids": []}):
                report = PAIR.cleanup_owned(ray, ray_owned=True, scratch=scratch, temporary=temporary, run=self.run)
        stop.assert_called_once_with(self.run)
        self.assertFalse(report["cleanup_confirmed"])
        self.assertEqual(report["errors"], [{"action": "ray_shutdown", "error_type": "RuntimeError"}])
        self.assertEqual((self.run / "ray_logs/session/logs/native.log").read_text(), "preserved CPU fixture\n")
        self.assertFalse(scratch.exists())
        self.assertFalse(temporary.exists())


if __name__ == "__main__":
    unittest.main()
