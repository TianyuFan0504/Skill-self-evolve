"""CPU-only deployment orchestration checks; no engine, account or GPU starts."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "scripts/launch.py"
SPEC = importlib.util.spec_from_file_location("_joint_launch_cpu", PATH)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class FakeSupervisor:
    instances = []

    def __init__(self, run_dir, environment):
        self.run_dir, self.environment = run_dir, environment
        self.commands, self.cleaned = [], False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.cleaned = True

    def run(self, name, command):
        self.commands.append((name, command))
        if name == "prepare-data":
            data = self.run_dir / "data"
            data.mkdir()
            (data / "split_manifest.json").write_text(json.dumps({"catalog_digest": "catalog-fixture"}))

    def spawn(self, name, command):
        self.commands.append((name, command))
        return SimpleNamespace(pid=123, name=name)

    def wait(self, process, *, backend):
        self.waited = (process.name, backend.name)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "run-one"
        self.asset = self.root / "instructions.md"
        self.asset.write_text("Public instructions\n")
        self.profile = {
            "environment": {
                "TRAINING_PYTHON": sys.executable,
                "ALFWORLD_ENV_PYTHON": "/engine/bin/python",
                "MODEL_PATH": "/models/fixture",
                "MODEL_NAME": "fixture",
                "ALFWORLD_BENCHMARK_ROOT": "/benchmark",
                "CODEX_NATIVE_BIN": "/native/codex",
                "ALFWORLD_CODEX_PACKAGE_ROOT": "/native/package",
                "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": str(self.root / "public/{run_id}/instructions.md"),
                "CURATOR_USER": "deployment-curator",
                "CURATOR_PUBLIC_ROOT": "/public/curator",
                "CURATOR_WORK_ROOT": "/work/curator/{run_id}",
            },
            "runtime": {
                "account_pool": {"factory": launch.LINUX_FACTORY, "kwargs": {"root": "{run_dir}/registry"}},
                "initialize_if_missing": True,
                "native_policy": {"instructions_file": str(self.root / "public/{run_id}/instructions.md")},
            },
            "task_config": [{"name": "alfworld_skill_training", "native": "/native/codex"}],
            "public_assets": [
                {"source": str(self.asset), "destination": str(self.root / "public/{run_id}/instructions.md")}
            ],
            "tool_dir": str(self.root / "public/{run_id}/tools"),
            "tool_python": "/usr/bin/python3",
            "hydra_overrides": ["trainer.experiment_name=literal run", "skill_harness.joint.curator_concurrency=2"],
        }
        FakeSupervisor.instances.clear()

    def mocked_launch(self, **kwargs):
        with (
            patch.object(launch, "Supervisor", FakeSupervisor),
            patch.object(launch, "wait_for_backend", return_value={"pid": 123, "url": "http://127.0.0.1:7"}) as ready,
            patch.object(launch, "process_start_ticks", return_value=456),
        ):
            launch.launch(self.profile, self.run_dir, **kwargs)
        return FakeSupervisor.instances[-1], ready

    def test_joint_smoke_prepares_fresh_run_then_trains_with_private_policy_and_explicit_interpreters(self):
        supervisor, ready = self.mocked_launch(smoke=True, overrides=["trainer.total_training_steps=7"])
        names = [name for name, _command in supervisor.commands]
        self.assertEqual(names, ["initialize-native-runtime", "prepare-data", "environment", "preflight", "training"])
        commands = dict(supervisor.commands)
        self.assertEqual(commands["prepare-data"][0], sys.executable)
        self.assertEqual(commands["environment"][0], "/engine/bin/python")
        self.assertEqual(commands["environment"][-8:-4], ["--environment-processes", "4", "--io-workers", "4"])
        self.assertEqual(
            commands["training"][:4],
            [
                sys.executable,
                "-m",
                "verl.experimental.skill_harness.train",
                f"--config-path={launch.RECIPE / 'config'}",
            ],
        )
        self.assertIn("skill_harness.joint.probes_per_step=16", commands["training"])
        self.assertIn("skill_harness.joint.replay_batch_size=8", commands["training"])
        self.assertIn("skill_harness.joint.curator_concurrency=2", commands["training"])
        self.assertNotIn("skill_harness.joint.curator_concurrency=8", commands["training"])
        self.assertIn("skill_harness.evaluation.final=false", commands["training"])
        self.assertEqual(commands["training"][-1], "trainer.total_training_steps=7")
        self.assertEqual(supervisor.waited, ("training", "environment"))
        self.assertTrue(supervisor.cleaned)
        self.assertEqual(ready.call_args.kwargs["digest"], "catalog-fixture")
        settings = json.loads((self.run_dir / "config/environment.json").read_text())
        self.assertEqual(settings["process_start_ticks"], 456)
        self.assertEqual(settings["account_pool"]["kwargs"]["root"], str(self.run_dir / "registry"))
        self.assertEqual(
            settings["native_policy"]["instructions_file"], str(self.root / "public/run-one/instructions.md")
        )
        self.assertEqual((self.run_dir / "config/environment.json").stat().st_mode & 0o777, 0o600)
        wrapper = (self.root / "public/run-one/tools/alfworld-step").read_text()
        self.assertIn("exec /usr/bin/python3", wrapper)
        self.assertNotIn("conda", wrapper)
        self.assertEqual((self.root / "public/run-one/instructions.md").read_text(), self.asset.read_text())
        self.assertEqual(self.profile["runtime"]["account_pool"]["kwargs"]["root"], "{run_dir}/registry")

    def test_existing_registry_is_not_initialized_and_existing_run_is_never_overwritten(self):
        registry = self.root / "existing-registry"
        registry.mkdir()
        marker = registry / "state.json"
        marker.write_text("old state\n")
        self.profile["runtime"]["account_pool"]["kwargs"]["root"] = str(registry)
        supervisor, _ = self.mocked_launch()
        self.assertNotIn("initialize-native-runtime", dict(supervisor.commands))
        self.assertEqual(marker.read_text(), "old state\n")
        with self.assertRaises(FileExistsError):
            self.mocked_launch()
        self.assertEqual(marker.read_text(), "old state\n")

    def test_recipe_overrides_follow_smoke_and_precede_cli_for_both_checks_and_training(self):
        recipe = ["trainer.total_training_steps=3", "data.train_batch_size=8"]
        cli = ["trainer.total_training_steps=5", "data.train_batch_size=4"]
        supervisor, _ = self.mocked_launch(smoke=True, recipe_overrides=recipe, overrides=cli)
        commands = dict(supervisor.commands)
        expected = [*self.profile["hydra_overrides"], *launch.SMOKE_OVERRIDES, *recipe, *cli]
        self.assertEqual(commands["preflight"][2:], expected)
        self.assertEqual(commands["training"][5:], expected)

    def test_dry_run_prints_the_same_command_without_preparing_or_starting_anything(self):
        before = sorted(self.root.rglob("*"))
        output = io.StringIO()
        with (
            patch.object(launch, "Supervisor", side_effect=AssertionError("dry run started a process")),
            patch.object(launch, "publish_assets", side_effect=AssertionError("dry run deployed assets")),
            patch("sys.stdout", output),
        ):
            launch.launch(
                self.profile,
                self.run_dir,
                smoke=True,
                recipe_overrides=["trainer.total_training_steps=3"],
                overrides=["trainer.total_training_steps=5"],
                dry_run=True,
            )
        import shlex

        command = shlex.split(output.getvalue().splitlines()[-1])
        self.assertEqual(
            command[:4],
            [
                sys.executable,
                "-m",
                "verl.experimental.skill_harness.train",
                f"--config-path={launch.RECIPE / 'config'}",
            ],
        )
        self.assertEqual(command[-2:], ["trainer.total_training_steps=3", "trainer.total_training_steps=5"])
        self.assertEqual(sorted(self.root.rglob("*")), before)
        self.assertFalse(self.run_dir.exists())

    def test_backend_readiness_failure_never_starts_training_and_cleans_owned_children(self):
        with (
            patch.object(launch, "Supervisor", FakeSupervisor),
            patch.object(launch, "wait_for_backend", side_effect=TimeoutError("not ready")),
        ):
            with self.assertRaisesRegex(TimeoutError, "not ready"):
                launch.launch(self.profile, self.run_dir)
        supervisor = FakeSupervisor.instances[-1]
        self.assertTrue(supervisor.cleaned)
        self.assertNotIn("training", dict(supervisor.commands))

    def test_installed_environment_drops_inherited_python_overlays(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/unrelated/checkout", "CONDA_PREFIX": "/old/environment"}):
            environment = launch.deployment_environment(self.profile, self.run_dir)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("CONDA_PREFIX", environment)
        self.assertTrue(environment["PATH"].startswith(str(Path(sys.executable).parent)))
        profile = copy.deepcopy(self.profile)
        profile["environment"]["PYTHONPATH"] = "/overlay"
        with self.assertRaisesRegex(ValueError, "Install dependencies"):
            launch.deployment_environment(profile, self.run_dir)

    def test_readiness_checks_owned_pid_and_parallel_workers(self):
        config = self.root / "service.json"
        config.write_text(json.dumps({"pid": 111, "url": "http://127.0.0.1:7"}))
        process = Mock(pid=111)
        process.poll.return_value = None
        supervisor = Mock()
        health = {
            "ready": True,
            "catalog_digest": "fixture",
            "environment_processes": 4,
            "io_concurrency": 4,
            "action_request_ids": True,
            "environment_worker_pids": [1, 2, 3, 4],
        }
        response = io.StringIO(json.dumps(health))
        with patch.object(launch.urllib.request, "urlopen", return_value=response):
            settings = launch.wait_for_backend(
                supervisor, process, config, processes=4, io_workers=4, digest="fixture", timeout=1
            )
        self.assertEqual(settings["pid"], 111)
        config.write_text(json.dumps({"pid": 222, "url": "http://127.0.0.1:7"}))
        with patch.object(launch.time, "monotonic", side_effect=[0, 0, 2]), patch.object(launch.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "owned backend PID"):
                launch.wait_for_backend(
                    supervisor, process, config, processes=4, io_workers=4, digest="fixture", timeout=1
                )

    def test_supervisor_allows_owned_environment_leader_to_clean_workers(self):
        self.run_dir.mkdir()
        process = Mock(pid=111, args=["fixture"])
        process.poll.side_effect = [None, None, 0]
        with (
            patch.object(launch.subprocess, "Popen", return_value=process),
            patch.object(launch, "process_start_ticks", return_value=456),
            patch.object(launch.os, "kill") as leader_signal,
            patch.object(launch.os, "killpg") as killpg,
        ):
            with launch.Supervisor(self.run_dir, {}) as supervisor:
                supervisor.spawn("environment", ["fixture"])
        leader_signal.assert_called_once_with(111, signal.SIGTERM)
        killpg.assert_not_called()
        process.wait.assert_called_once_with(timeout=20)
        recorded = json.loads((self.run_dir / "launch-processes.json").read_text())
        self.assertEqual(recorded["children"][0]["returncode"], 0)

    def test_requested_stop_interrupts_training_to_allow_python_cleanup(self):
        self.run_dir.mkdir()
        process = Mock(pid=111, args=["fixture"])
        process.poll.side_effect = [None, None, -signal.SIGINT]
        with (
            patch.object(launch.subprocess, "Popen", return_value=process),
            patch.object(launch, "process_start_ticks", return_value=456),
            patch.object(launch.os, "kill") as leader_signal,
            patch.object(launch.os, "killpg") as killpg,
        ):
            with self.assertRaises(InterruptedError):
                with launch.Supervisor(self.run_dir, {}) as supervisor:
                    supervisor.spawn("training", ["fixture"])
                    supervisor.request_stop(signal.SIGTERM, None)
                    supervisor.check_stop()
        leader_signal.assert_called_once_with(111, signal.SIGINT)
        killpg.assert_not_called()
        process.wait.assert_called_once_with(timeout=20)

    def test_training_stop_escalates_only_its_owned_group_after_graceful_timeout(self):
        self.run_dir.mkdir()
        process = Mock(pid=111, args=["fixture"])
        process.poll.side_effect = [None, None, None, None, -signal.SIGKILL]
        process.wait.side_effect = [
            subprocess.TimeoutExpired("fixture", 20),
            subprocess.TimeoutExpired("fixture", 15),
            None,
        ]
        with (
            patch.object(launch.subprocess, "Popen", return_value=process),
            patch.object(launch, "process_start_ticks", return_value=456),
            patch.object(launch.os, "kill") as leader_signal,
            patch.object(launch.os, "killpg") as killpg,
        ):
            with launch.Supervisor(self.run_dir, {}) as supervisor:
                supervisor.spawn("training", ["fixture"])
        leader_signal.assert_called_once_with(111, signal.SIGINT)
        self.assertEqual(
            [call.args for call in killpg.call_args_list],
            [(111, signal.SIGTERM), (111, signal.SIGKILL)],
        )
        self.assertEqual([call.kwargs["timeout"] for call in process.wait.call_args_list], [20, 15, 5])


if __name__ == "__main__":
    unittest.main()
