"""CPU-only checks of official split isolation and protected service routing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/evaluation.py"
SPEC = importlib.util.spec_from_file_location("skill_evaluation", PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class Options(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key, value):
        self[key] = value


class EvaluationProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        self.data = self.run / "data"
        self.data.mkdir()
        games = [
            {"game_id": "train-a", "split": "train", "relative_game_path": "train/a"},
            {"game_id": "seen-a", "split": "valid_seen", "relative_game_path": "valid_seen/a"},
            {"game_id": "seen-b", "split": "valid_seen", "relative_game_path": "valid_seen/b"},
            {"game_id": "unseen-a", "split": "valid_unseen", "relative_game_path": "valid_unseen/a"},
        ]
        canonical = json.dumps(games, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode()).hexdigest()
        self.catalog = self.data / "catalog.json"
        self.catalog.write_text(json.dumps({"schema_version": 1, "catalog_digest": self.digest, "games": games}))
        counts = {"test_seen": {"games": 2}, "test_unseen": {"games": 1}}
        files = {}
        for phase in evaluation.TEST_SPLITS:
            path = self.data / (phase + ".parquet")
            path.write_bytes(phase.encode())
            files[phase] = {"path": str(path), "sha256": evaluation._sha256(path)}
        self.manifest = {
            "schema_version": 1,
            "benchmark_root": str(self.run / "benchmark"),
            "run_dir": str(self.run),
            "catalog_digest": self.digest,
            "evaluation_games_used_for_training": False,
            "available": counts,
            "selected": counts,
            "phase_game_ids": {
                "train": ["train-a"],
                "dev": [],
                "test_seen": ["seen-a", "seen-b"],
                "test_unseen": ["unseen-a"],
            },
            "parquet_files": files,
        }
        self.manifest_path = self.data / "split_manifest.json"
        self.write_manifest()
        self.backend = self.run / "backend.py"
        self.backend.write_text("# Explicit backend fixture\n")
        self.config = self.run / "environment.json"
        self.config.write_text(
            json.dumps(
                {
                    "url": "http://127.0.0.1:1234",
                    "admin_token": "TRAINING-SECRET",
                    "pid": 123,
                    "catalog_digest": self.digest,
                    "allowed_splits": ["train"],
                    "max_steps": 50,
                    "budget_seconds": 900,
                    "action_request_protocol": evaluation.ACTION_REQUEST_PROTOCOL,
                }
            )
            + "\n"
        )
        self.config.chmod(0o600)
        self.environment_patch = patch.dict(evaluation.os.environ, {
            "ALFWORLD_SERVER_CONFIG": str(self.config), "ALFWORLD_ENV_PYTHON": "ENV-PYTHON",
        })
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def trainer(self):
        trainer = SimpleNamespace(
            config=SimpleNamespace(
                data=Options(
                    val_files="dev.parquet",
                    val_max_samples=64,
                    validation_shuffle=True,
                    filter_overlong_prompts=True,
                    dataloader_num_workers=4,
                    val_batch_size=16,
                ),
                trainer=Options(validation_data_dir="dev-output", log_val_generations=5),
                skill_harness=Options(
                    split_manifest=str(self.manifest_path), artifact_root=str(self.run),
                    evaluation=Options(server_script=str(self.backend), server_args=[]),
                ),
                actor_rollout_ref=SimpleNamespace(
                    rollout=SimpleNamespace(
                        val_kwargs=Options(evaluation.GREEDY_SETTINGS),
                        prompt_length=4096,
                        response_length=32768,
                        max_model_len=36864,
                    )
                ),
            ),
            val_dataset=object(),
            val_dataloader=object(),
            agent_loop_manager=object(),
            tokenizer=object(),
            processor=object(),
            global_steps=7,
            on_validate_begin=Mock(),
            on_validate_end=Mock(),
            _dump_futures=[Mock()],
            _assert_final_seal=Mock(),
        )
        trainer._validate = Mock(return_value={"val-core/alfworld/acc/mean@1": 0.5})
        return trainer

    def dataset_components(self):
        def create(path, data, tokenizer, processor, *, is_train, max_samples):
            self.assertFalse(is_train)
            self.assertEqual(max_samples, -1)
            self.assertFalse(data.filter_overlong_prompts)
            return [object()] * (2 if Path(path).stem == "test_seen" else 1)

        return create, lambda **kwargs: kwargs, object()

    def service_kwargs(self):
        return dict(
            config_path=self.config,
            artifact_root=self.run / "evaluation",
            benchmark_root=self.run / "benchmark",
            run_root=self.run,
            catalog_path=self.catalog,
            digest=self.digest,
            environment_python="ENV-PYTHON",
            environment_server=self.backend,
            environment_args=["--environment-processes", "2", "--io-workers", "1"],
        )

    def fake_process_factory(self, *, ready):
        processes = []

        def start(argv, **kwargs):
            self.assertEqual(argv[0:2], ["ENV-PYTHON", str(self.backend)])
            self.assertIn("--budget-seconds", argv)
            self.assertEqual(argv[-4:], ["--environment-processes", "2", "--io-workers", "1"])
            self.assertTrue(kwargs["start_new_session"])
            self.assertTrue(kwargs["close_fds"])
            process = Mock(pid=456, returncode=0)
            process.poll.return_value = None
            process.terminate.side_effect = lambda: setattr(process.poll, "return_value", 0)
            if ready:
                settings_path = Path(argv[argv.index("--config-path") + 1])
                settings_path.write_text(
                    json.dumps(
                        {
                            "url": "http://127.0.0.1:4567",
                            "admin_token": "TEST-SECRET",
                            "pid": 456,
                            "catalog_digest": self.digest,
                            "allowed_splits": ["valid_seen", "valid_unseen"],
                            "max_steps": 50,
                            "budget_seconds": 900,
                            "action_request_protocol": evaluation.ACTION_REQUEST_PROTOCOL,
                        }
                    )
                )
                settings_path.chmod(0o600)
            processes.append(process)
            return process

        return start, processes

    def test_complete_manifest_and_changed_parquet(self):
        _, _, files = evaluation._read_protocol(self.manifest_path)
        self.assertEqual(files["test_seen"]["games"], 2)
        (self.data / "test_seen.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "parquet changed"):
            evaluation._read_protocol(self.manifest_path)

    def test_truncated_official_partition_is_rejected(self):
        self.manifest["phase_game_ids"]["test_seen"] = ["seen-a"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "every official eligible game"):
            evaluation._read_protocol(self.manifest_path)

    def test_parquet_rows_must_use_exact_native_task_and_split(self):
        rows = [
            {
                "tools_kwargs": {
                    "task": {
                        "name": "alfworld_skill_training",
                        "metadata": {
                            "game_id": "seen-a",
                            "split": "train",
                            "catalog_digest": self.digest,
                            "run_dir": str(self.run),
                        },
                    }
                }
            }
        ]
        with self.assertRaisesRegex(ValueError, "incompatible native-task metadata"):
            evaluation._assert_test_rows(
                rows,
                expected_ids={"seen-a"},
                split="valid_seen",
                digest=self.digest,
                run_root=self.run,
            )

    def test_service_swap_restores_exact_training_bytes_after_failure(self):
        before = self.config.read_bytes()
        start, processes = self.fake_process_factory(ready=True)
        with (
            patch.object(evaluation.subprocess, "Popen", side_effect=start),
            patch.object(evaluation, "_health", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "episode failure"):
                with evaluation.official_test_service(**self.service_kwargs()) as public:
                    self.assertEqual(json.loads(self.config.read_bytes())["pid"], 456)
                    self.assertNotIn("admin_token", public)
                    raise RuntimeError("episode failure")
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o600)
        processes[0].terminate.assert_called_once()
        processes[0].kill.assert_not_called()

    def test_service_timeout_stops_only_new_process_without_config_swap(self):
        before = self.config.read_bytes()
        start, processes = self.fake_process_factory(ready=False)
        with (
            patch.object(evaluation.subprocess, "Popen", side_effect=start),
            patch.object(evaluation, "_atomic_bytes") as replace,
        ):
            with self.assertRaises(TimeoutError):
                with evaluation.official_test_service(**self.service_kwargs(), startup_timeout=0):
                    self.fail("An unhealthy service must never be exposed to tasks")
        replace.assert_not_called()
        processes[0].terminate.assert_called_once()
        self.assertEqual(self.config.read_bytes(), before)

    def test_real_configured_backend_routes_final_tests_and_preserves_provider(self):
        self.backend.write_text('''
import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
for field in ("benchmark-root", "run-root", "catalog-path", "config-path"):
    p.add_argument("--" + field)
p.add_argument("--allowed-splits", nargs="+")
for field in ("max-steps", "budget-seconds", "port", "environment-processes", "io-workers"):
    p.add_argument("--" + field, type=int)
a = p.parse_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(settings).encode())
    def log_message(self, *_args):
        pass

server = HTTPServer(("127.0.0.1", a.port), Handler)
settings = {
    "url": "http://127.0.0.1:" + str(server.server_port),
    "admin_token": "test-only-capability", "pid": os.getpid(), "ready": True,
    "catalog_digest": json.loads(Path(a.catalog_path).read_text())["catalog_digest"],
    "allowed_splits": a.allowed_splits, "max_steps": a.max_steps,
    "budget_seconds": a.budget_seconds, "action_request_protocol": "alfworld-step-request-id-v1",
}
path = Path(a.config_path)
path.write_text(json.dumps(settings))
path.chmod(0o600)
server.serve_forever()
''')
        training = json.loads(self.config.read_bytes())
        training["account_pool"] = {"factory": "fixture:Pool", "kwargs": {"root": "fixture"}}
        training["native_policy"] = {"instructions_file": str(self.run / "instructions.md"),
                                     "disabled_features": ["fixture_feature"]}
        self.config.write_text(json.dumps(training))
        original = self.config.read_bytes()
        kwargs = self.service_kwargs()
        kwargs["environment_python"] = sys.executable
        with evaluation.official_test_service(**kwargs, startup_timeout=5):
            routed = json.loads(self.config.read_bytes())
            self.assertEqual(set(routed["allowed_splits"]), set(evaluation.TEST_SPLITS.values()))
            self.assertEqual(routed["budget_seconds"], 900)
            self.assertEqual(routed["account_pool"], training["account_pool"])
            self.assertEqual(routed["native_policy"], training["native_policy"])
            self.assertNotEqual(routed["pid"], training["pid"])
        self.assertEqual(self.config.read_bytes(), original)
        with self.assertRaises(ProcessLookupError):
            os.kill(routed["pid"], 0)

    def test_validation_context_restores_config_loader_and_phase_on_failure(self):
        trainer = self.trainer()
        trainer._evaluation_phase = "dev"
        originals = trainer.config.data, trainer.config.trainer, trainer.val_dataset, trainer.val_dataloader
        with patch.object(evaluation, "_dataset_components", side_effect=self.dataset_components):
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                with evaluation._validation_inputs(
                    trainer,
                    path=self.data / "test_seen.parquet",
                    expected_games=2,
                    phase="test_seen",
                    stage="final",
                    output_dir=self.run / "output",
                ):
                    self.assertEqual(trainer._evaluation_phase, "test_seen")
                    self.assertEqual(trainer._evaluation_stage, "final")
                    self.assertEqual(trainer.config.data.val_max_samples, -1)
                    self.assertEqual(trainer.config.trainer.log_val_generations, 0)
                    raise RuntimeError("failed validation")
        self.assertEqual(
            (trainer.config.data, trainer.config.trainer, trainer.val_dataset, trainer.val_dataloader), originals
        )
        self.assertEqual(trainer._evaluation_phase, "dev")
        self.assertFalse(hasattr(trainer, "_evaluation_stage"))

    def test_truthy_metrics_without_raw_cases_do_not_complete_and_restore_same_manager(self):
        trainer = self.trainer()
        original_manager, original_loader = trainer.agent_loop_manager, trainer.val_dataloader
        phases = []

        def validate():
            self.assertIs(trainer.agent_loop_manager.delegate, original_manager)
            self.assertEqual(trainer.config.data.val_max_samples, -1)
            phases.append((trainer._evaluation_phase, trainer._evaluation_stage, len(trainer.val_dataset)))
            return {"val-core/alfworld/acc/mean@1": 0.5}

        trainer._validate.side_effect = validate

        @contextmanager
        def service(**kwargs):
            self.assertEqual(kwargs["run_root"], self.run)
            yield {"catalog_digest": self.digest, "max_steps": 50}

        with (
            patch.object(evaluation, "_dataset_components", side_effect=self.dataset_components),
            patch.object(evaluation, "_verify_parquet_rows") as verify,
            patch.object(evaluation, "official_test_service", side_effect=service),
        ):
            with self.assertRaisesRegex(RuntimeError, "Official cases/scores are incomplete"):
                evaluation.evaluate_official_splits(trainer)
        self.assertEqual(phases, [("test_seen", "final", 2)])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(trainer.on_validate_begin.call_count, 1)
        self.assertEqual(trainer.on_validate_end.call_count, 1)
        self.assertEqual(trainer._dump_futures[0].result.call_count, 1)
        self.assertIs(trainer.val_dataloader, original_loader)
        self.assertIs(trainer.agent_loop_manager, original_manager)
        self.assertFalse((self.run / "evaluation/final/summary.json").exists())
        audit = next((self.run / "evaluation/final/test_seen").glob("case-session-audit-*.json"))
        self.assertFalse(json.loads(audit.read_text())["complete"])
        reports = "".join(path.read_text() for path in (self.run / "evaluation").rglob("*.json"))
        self.assertNotIn("TRAINING-SECRET", reports)

    def test_evaluation_failure_always_ends_hook_and_restores_dev(self):
        trainer = self.trainer()
        original_data, original_loader = trainer.config.data, trainer.val_dataloader
        trainer._validate.side_effect = RuntimeError("failed native episode collection")

        @contextmanager
        def service(**kwargs):
            yield {"catalog_digest": self.digest}

        with (
            patch.object(evaluation, "_dataset_components", side_effect=self.dataset_components),
            patch.object(evaluation, "_verify_parquet_rows"),
            patch.object(evaluation, "official_test_service", side_effect=service),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed native episode"):
                evaluation.evaluate_official_splits(trainer)
        trainer.on_validate_end.assert_called_once()
        self.assertIs(trainer.config.data, original_data)
        self.assertIs(trainer.val_dataloader, original_loader)
        self.assertFalse(hasattr(trainer, "_evaluation_phase"))
        self.assertFalse((self.run / "evaluation/final/summary.json").exists())

    def test_non_greedy_workers_are_rejected_before_service_start(self):
        trainer = self.trainer()
        trainer.config.actor_rollout_ref.rollout.val_kwargs.temperature = 0.7
        with patch.object(evaluation, "official_test_service") as service:
            with self.assertRaisesRegex(ValueError, "greedy n=1"):
                evaluation.evaluate_official_splits(trainer)
        service.assert_not_called()

    def test_evaluation_requires_completed_trained_checkpoint_before_service(self):
        trainer = self.trainer()
        trainer._assert_final_seal.side_effect = RuntimeError("A trained checkpoint is required")
        with patch.object(evaluation, "official_test_service") as service:
            with self.assertRaisesRegex(RuntimeError, "trained checkpoint"):
                evaluation.evaluate_official_splits(trainer)
        service.assert_not_called()
        self.assertFalse((self.run / "evaluation").exists())

    def test_dataset_filtering_cannot_hide_missing_official_cases(self):
        trainer = self.trainer()
        original_dataset = trainer.val_dataset
        components = lambda: (lambda *args, **kwargs: [object()], Mock(), object())
        with patch.object(evaluation, "_dataset_components", side_effect=components):
            with self.assertRaisesRegex(ValueError, "dropped or duplicated official test cases"):
                with evaluation._validation_inputs(
                    trainer,
                    path=self.data / "test_seen.parquet",
                    expected_games=2,
                    phase="test_seen",
                    stage="final",
                    output_dir=self.run / "output",
                ):
                    self.fail("A partial official dataset must never be evaluated")
        self.assertIs(trainer.val_dataset, original_dataset)
        self.assertEqual(trainer.config.data.val_files, "dev.parquet")


if __name__ == "__main__":
    unittest.main()
