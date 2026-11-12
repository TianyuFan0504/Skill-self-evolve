"""Local launcher metadata contracts; all process/service/model actions stubbed."""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_resilient.py"
spec = importlib.util.spec_from_file_location("_resilient_launcher_cpu", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeLauncher:
    def __init__(self, root, events):
        self.root, self.events, self.started = root, events, True
        self.original = module.existing_launcher()
        self.write_json = self.original.write_json

    def main(self):
        self.events.append("prepare")
        assert "--joint" in sys.argv and "--prepare-only" in sys.argv
        self.root.mkdir(parents=True, exist_ok=True)
        # Exercise the existing real archival helper, before our simulated
        # preparation replaces command/status metadata.
        self.original.archive_previous_attempt(self.root)
        os.environ["SKILL_RUN_DIR"] = str(self.root)
        os.environ["ALFWORLD_SERVER_CONFIG"] = str(self.root / "runtime/environment.json")
        self.write_json(self.root / "launch_manifest.json", {"command": [
            sys.executable, "-m", "verl.experimental.skill_harness.train",
            "--config-path=actual-config", "--config-name=alfworld_joint",
        ]})
        self.write_json(self.root / "status.json", {"state": "prepared"})
        self.write_json(self.root / "hydra_overrides.json", sys.argv[3:])
        return 0

    def start_environment(self, *args):
        self.events.append("service")
        return 987654, self.started

    def supervise(self, run, command, pid):
        self.events.append("supervise")
        assert run == self.root and command[1].endswith("train_resilient_setup.py") and pid == 987654
        return 17


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve() / "run"
        self.events = []
        self.launcher = FakeLauncher(self.run, self.events)
        environment = {**module.DEFAULT_ENV, "VERL_SOURCE_ROOT": str(SCRIPT.parents[3]),
                       "ALFWORLD_BENCHMARK_ROOT": "/unused-benchmark"}
        self.environment = patch.dict(os.environ, environment)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def execute(self, *flags, preflight_code=0):
        def checked(*args, **kwargs):
            self.events.append("preflight")
            return subprocess.CompletedProcess(args[0], preflight_code)

        def manifest(run, *, config=None):
            self.events.append("fresh-manifest")
            if (run / "training-manifest.json").exists():
                return False
            (run / "training-manifest.json").write_text('{"fresh":true}\n')
            return True

        with patch.object(module, "existing_launcher", return_value=self.launcher), \
                patch.object(module, "fresh_training_manifest", side_effect=manifest), \
                patch.object(module, "prepared_run_config", return_value=types.SimpleNamespace(
                    skill_harness=types.SimpleNamespace(artifact_root=str(self.run)))), \
                patch.object(module.subprocess, "run", side_effect=checked), \
                contextlib.redirect_stdout(io.StringIO()):
            return module.main(["--run-dir", str(self.run), *flags])

    def test_defaults_are_joint32_n4_no_initial_and_actual_safe_paths(self):
        _, overrides = module.arguments([])
        values = dict(value.split("=", 1) for value in overrides)
        self.assertEqual(values["actor_rollout_ref.rollout.n"], "4")
        self.assertEqual(values["skill_harness.evaluation.initial"], "false")
        self.assertEqual(values["trainer.val_before_train"], "false")
        self.assertEqual(values[
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions"], "32")
        self.assertEqual(values["trainer.v1.sampler.custom_sampler.name"], "ResilientReplayBuffer")

    def test_spawn_safe_mode_fixes_the_public_runner_class(self):
        args, overrides = module.arguments(["--spawn-safe"])
        self.assertTrue(args.spawn_safe)
        values = dict(value.split("=", 1) for value in overrides)
        key = "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn"
        self.assertEqual(values[key], "skill_harness_examples.train_spawn_safe.SpawnSafeRunner")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.arguments(["--spawn-safe", key + "=skill_harness_examples.train_resilient_setup.safe_run_task"])

    def test_parallel_profile_also_selects_the_spawn_safe_runner(self):
        args, overrides = module.arguments(["--parallel-environment"])
        self.assertTrue(args.parallel_environment)
        self.assertTrue(args.spawn_safe)
        values = dict(value.split("=", 1) for value in overrides)
        self.assertEqual(values[
            "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn"],
            "skill_harness_examples.train_spawn_safe.SpawnSafeRunner")

    def test_parallel_submission_selects_service_and_client_driver_together(self):
        spawn_module = types.ModuleType("skill_harness_examples.native_posix_spawn")
        spawn_module.NativeSpawnPolicy = lambda _: None
        service_module = types.ModuleType("skill_harness_examples.parallel_environment_launcher")

        def start(*args):
            self.events.append("parallel-service")
            return 56789, True

        service_module.start_environment = start
        with patch.dict(sys.modules, {spawn_module.__name__: spawn_module, service_module.__name__: service_module}), \
                patch.object(module, "launch_receipt", return_value=self.run / "receipt.json"), \
                patch.object(module.subprocess, "Popen", return_value=types.SimpleNamespace(pid=1234)) as spawn:
            self.assertEqual(self.execute("--parallel-environment"), 0)
        command = spawn.call_args.args[0]
        payload = json.loads(Path(command[3]).read_text())
        self.assertTrue(payload["command"][1].endswith("train_parallel_client.py"))
        self.assertEqual(payload["environment_pid"], 56789)
        self.assertIn("parallel-service", self.events)
        self.assertNotIn("service", self.events)

    def test_parallel_resume_selects_direct_checkpoint_reader(self):
        spawn_module = types.ModuleType("skill_harness_examples.native_posix_spawn")
        spawn_module.NativeSpawnPolicy = lambda _: None
        service_module = types.ModuleType("skill_harness_examples.parallel_environment_launcher")
        service_module.start_environment = lambda *args: (56789, False)
        with patch.dict(sys.modules, {spawn_module.__name__: spawn_module, service_module.__name__: service_module}), \
                patch.object(module, "launch_receipt", return_value=self.run / "receipt.json"), \
                patch.object(module.subprocess, "Popen", return_value=types.SimpleNamespace(pid=1234)) as spawn:
            self.assertEqual(self.execute("--parallel-environment", "--resume"), 0)
        payload = json.loads(Path(spawn.call_args.args[0][3]).read_text())
        self.assertTrue(payload["command"][1].endswith("train_direct_checkpoint.py"))
        self.assertEqual(payload["environment_pid"], 56789)

    def test_spawn_helper_failure_precedes_preparation(self):
        fake = types.ModuleType("skill_harness_examples.native_posix_spawn")

        def unavailable(_):
            raise FileNotFoundError("CPU fixture missing helper")

        fake.NativeSpawnPolicy = unavailable
        with patch.dict(sys.modules, {fake.__name__: fake}), self.assertRaises(FileNotFoundError):
            self.execute("--spawn-safe", "--prepare-only")
        self.assertEqual(self.events, [])
        self.assertFalse(self.run.exists())

    def test_spawn_safe_submission_uses_new_driver_and_preserves_supervision(self):
        fake = types.ModuleType("skill_harness_examples.native_posix_spawn")
        seen = []
        fake.NativeSpawnPolicy = lambda path: seen.append(path)
        with patch.dict(sys.modules, {fake.__name__: fake}), \
                patch.object(module, "launch_receipt", return_value=self.run / "receipt.json"), \
                patch.object(module.subprocess, "Popen", return_value=types.SimpleNamespace(pid=1234)) as spawn:
            self.assertEqual(self.execute("--spawn-safe"), 0)
        self.assertEqual(seen, [module.RECIPE / "scripts/native_posix_spawn_helper.so"])
        command = spawn.call_args.args[0]
        self.assertEqual(command[2], "--supervise")
        payload = json.loads(Path(command[3]).read_text())
        self.assertTrue(payload["command"][1].endswith("train_spawn_safe.py"))
        self.assertEqual(payload["environment_pid"], 987654)

    def test_spawn_receipt_binds_driver_sources_and_built_helper(self):
        recipe = Path(self.temporary.name) / "project/examples/skill_self_evolve"
        root = recipe.parent.parent
        paths = [recipe / name for name in [
            "run_resilient.sh", "scripts/train_resilient_setup.py", "scripts/train_spawn_safe.py",
            "scripts/train_quiescent_eval.py", "scripts/native_posix_spawn.py",
            "scripts/native_posix_spawn.c", "scripts/native_posix_spawn_helper.so",
        ]] + [root / "skill_harness_examples/__init__.py"]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        self.run.mkdir()
        args = types.SimpleNamespace(spawn_safe=True, resume=False)
        with patch.object(module, "RECIPE", recipe):
            receipt = module.launch_receipt(self.launcher, self.run, ["cpu-fixture"],
                                            fresh_manifest=False, args=args)
        record = json.loads(receipt.read_text())
        self.assertTrue(record["native_spawn_extension"])
        self.assertEqual(record["assets"]["driver"]["path"], str(recipe / "scripts/train_spawn_safe.py"))
        self.assertEqual(record["assets"]["spawn_helper"]["sha256"],
                         hashlib.sha256(b"native_posix_spawn_helper.so").hexdigest())

    def test_invalid_flags_or_initial_eval_cannot_silently_change_semantics(self):
        for flags in (["--unknown"], ["skill_harness.evaluation.initial=true"],
                      ["trainer.val_before_train=true"], ["actor_rollout_ref.rollout.n=2"],
                      ["--prepare-only", "--preflight-only"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.arguments(flags)

    def test_prepare_only_keeps_same_process_environment_and_starts_nothing(self):
        previous = sys.argv
        self.assertEqual(self.execute("--prepare-only"), 0)
        self.assertIs(sys.argv, previous)
        self.assertEqual(self.events, ["prepare"])
        self.assertEqual(os.environ["ALFWORLD_SERVER_CONFIG"], str(self.run / "runtime/environment.json"))
        receipt = json.loads(next((self.run / "launcher_runs").glob("*.json")).read_text())
        self.assertTrue(receipt["command"][1].endswith("train_resilient_setup.py"))
        self.assertIsNone(receipt["training_manifest_sha256"])
        self.assertFalse(receipt["original_training_manifest_preserved"])
        self.assertEqual(set(receipt["assets"]), {"launcher", "shell", "driver", "quiescent_driver", "namespace"})

    def test_cli_argument_routing_is_restored_after_preparation_exception(self):
        previous = sys.argv
        with patch.object(self.launcher, "main", side_effect=RuntimeError("prepare failed")):
            with self.assertRaises(RuntimeError):
                module.prepare(self.launcher, types.SimpleNamespace(run_dir=self.run, resume=False), [])
        self.assertIs(sys.argv, previous)

    def test_resume_archives_original_manifest_before_rewrites_and_keeps_it_unchanged(self):
        self.run.mkdir()
        original = b'{"original_actual_run":true}\n'
        (self.run / "training-manifest.json").write_bytes(original)
        (self.run / "training.log").write_text("historical training log\n")
        self.launcher.write_json(self.run / "status.json", {"state": "failed"})
        self.launcher.write_json(self.run / "launch_manifest.json", {"old_actual_command": True})
        self.assertEqual(self.execute("--resume", "--prepare-only"), 0)
        archive = next((self.run / "attempt_history").iterdir())
        self.assertEqual((archive / "training-manifest.json").read_bytes(), original)
        self.assertEqual((self.run / "training-manifest.json").read_bytes(), original)
        self.assertTrue(json.loads((archive / "launch_manifest.json").read_text())["old_actual_command"])

    def test_live_resilient_process_is_refused_before_preparation(self):
        self.run.mkdir()
        self.launcher.write_json(self.run / "status.json", {"state": "running", "training_pid": 42})
        proc = Path(self.temporary.name) / "proc"
        (proc / "42").mkdir(parents=True)
        with self.assertRaises(RuntimeError):
            module.refuse_live_run(self.run, proc_root=proc)
        self.assertEqual(self.events, [])

    def test_live_owned_supervisor_is_refused_during_submission_window(self):
        self.run.mkdir()
        (self.run / "supervisor.pid").write_text("42")
        proc = Path(self.temporary.name) / "proc"
        (proc / "42").mkdir(parents=True)
        (proc / "42/environ").write_bytes(("SKILL_RUN_DIR=" + str(self.run)).encode() + b"\0")
        with self.assertRaises(RuntimeError):
            module.refuse_live_run(self.run, proc_root=proc)

    def test_foreground_order_is_prepare_service_preflight_manifest_then_existing_supervisor(self):
        self.assertEqual(self.execute("--foreground"), 17)
        self.assertEqual(self.events, ["prepare", "service", "preflight", "fresh-manifest", "supervise"])
        receipt = json.loads(next((self.run / "launcher_runs").glob("*.json")).read_text())
        self.assertTrue(receipt["fresh_training_manifest_created"])

    def test_preflight_only_stops_only_its_owned_service_and_creates_no_training_manifest(self):
        with patch.object(module.os, "kill") as kill:
            self.assertEqual(self.execute("--preflight-only"), 0)
        kill.assert_called_once_with(987654, module.signal.SIGTERM)
        self.assertEqual(self.events, ["prepare", "service", "preflight"])
        self.assertFalse((self.run / "training-manifest.json").exists())

    def test_preflight_failure_leaves_attached_service_untouched(self):
        self.launcher.started = False
        with patch.object(module.os, "kill") as kill, self.assertRaises(subprocess.CalledProcessError):
            self.execute(preflight_code=2)
        kill.assert_not_called()
        self.assertFalse((self.run / "training-manifest.json").exists())

    def test_detached_mode_uses_existing_supervise_entry_with_exact_resilient_payload(self):
        with patch.object(module.subprocess, "Popen", return_value=types.SimpleNamespace(pid=1234)) as spawn:
            self.assertEqual(self.execute(), 0)
        command = spawn.call_args.args[0]
        self.assertTrue(command[1].endswith("scripts/launch.py"))
        self.assertEqual(command[2], "--supervise")
        payload = json.loads(Path(command[3]).read_text())
        self.assertTrue(payload["command"][1].endswith("train_resilient_setup.py"))
        self.assertEqual(payload["environment_pid"], 987654)

    def test_real_fresh_manifest_helper_does_not_import_or_overwrite_existing_run(self):
        self.run.mkdir()
        original = b'{"preserve":true}\n'
        (self.run / "training-manifest.json").write_bytes(original)
        self.assertFalse(module.fresh_training_manifest(self.run))
        self.assertEqual((self.run / "training-manifest.json").read_bytes(), original)

    def test_redirected_artifact_root_is_rejected_before_prepare_and_other_manifest_is_unchanged(self):
        other = Path(self.temporary.name).resolve() / "existing-experiment"
        other.mkdir()
        original = b'{"existing_actual_experiment":true}\n'
        (other / "training-manifest.json").write_bytes(original)
        for override in ("skill_harness.artifact_root=" + str(other),
                         "skill_harness={artifact_root:" + str(other) + "}"):
            with self.subTest(override=override), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.execute("--prepare-only", override)
        self.assertEqual(self.events, [])
        self.assertFalse(self.run.exists())
        self.assertEqual((other / "training-manifest.json").read_bytes(), original)

    def test_readonly_data_and_model_paths_remain_configurable(self):
        _, overrides = module.arguments(["data.train_files=/existing/read-only/train.parquet",
                                         "actor_rollout_ref.model.path=/existing/read-only/model"])
        self.assertIn("data.train_files=/existing/read-only/train.parquet", overrides)
        self.assertIn("actor_rollout_ref.model.path=/existing/read-only/model", overrides)

    def test_matching_artifact_override_still_runs_live_destination_guard(self):
        self.run.mkdir()
        original = b'{"live_actual_experiment":true}\n'
        (self.run / "training-manifest.json").write_bytes(original)
        self.launcher.write_json(self.run / "status.json", {"state": "running", "training_pid": 42})
        proc = Path(self.temporary.name) / "proc"
        (proc / "42").mkdir(parents=True)
        guard = module.refuse_live_run
        with patch.object(module, "refuse_live_run", side_effect=lambda run: guard(run, proc_root=proc)):
            with self.assertRaises(RuntimeError):
                self.execute("--resume", "--prepare-only", "skill_harness.artifact_root=" + str(self.run))
        self.assertEqual(self.events, [])
        self.assertEqual((self.run / "training-manifest.json").read_bytes(), original)

    def test_resolved_prepared_config_cannot_redirect_manifest_writer(self):
        self.run.mkdir()
        other = Path(self.temporary.name).resolve() / "other"
        other.mkdir()
        original = b'{"preserve_other":true}\n'
        (other / "training-manifest.json").write_bytes(original)
        config = types.SimpleNamespace(skill_harness=types.SimpleNamespace(artifact_root=str(other)))
        with self.assertRaises(ValueError):
            module.fresh_training_manifest(self.run, config=config)
        self.assertEqual((other / "training-manifest.json").read_bytes(), original)
        self.assertFalse((self.run / "training-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
