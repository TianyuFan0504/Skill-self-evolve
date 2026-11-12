"""Service profile/ownership checks; no processes or sockets are started."""

import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/parallel_environment_launcher.py"
spec = importlib.util.spec_from_file_location("_parallel_launcher_cpu", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        (self.run / "data").mkdir()
        (self.run / "data/split_manifest.json").write_text('{"catalog_digest":"catalog"}')
        self.config = self.run / "environment.json"
        self.settings = {"url": "http://127.0.0.1:9999", "pid": 42, "allowed_splits": ["train"],
                         "environment_policy": module.POLICY, "environment_processes": 4, "io_workers": 4,
                         "action_request_protocol": module.PROTOCOL}
        self.health = {"ready": True, "catalog_digest": "catalog", "environment_policy": module.POLICY,
                       "environment_processes": 4, "io_concurrency": 4,
                       "action_request_protocol": module.PROTOCOL}
        environment = patch.dict(os.environ, {"ALFWORLD_SERVER_CONFIG": str(self.config)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_legacy_service_is_refused_without_spawning_or_rewriting(self):
        self.config.write_text(json.dumps(self.settings))
        original = self.config.read_bytes()
        with patch.object(module, "read_health", return_value={"ready": True}), \
                patch.object(module.subprocess, "Popen") as spawn, self.assertRaises(RuntimeError):
            module.start_environment(self.run, "/benchmark", "/source")
        spawn.assert_not_called()
        self.assertEqual(self.config.read_bytes(), original)

    def test_matching_train_only_service_attaches_without_signal_or_spawn(self):
        self.config.write_text(json.dumps(self.settings))
        with patch.object(module, "read_health", return_value=self.health), \
                patch.object(module.os, "kill") as check, patch.object(module.subprocess, "Popen") as spawn:
            self.assertEqual(module.start_environment(self.run, "/benchmark", "/source"), (42, False))
        check.assert_called_once_with(42, 0)
        spawn.assert_not_called()

    def test_fresh_service_has_four_workers_separate_io_and_no_cuda(self):
        child = types.SimpleNamespace(pid=42, poll=lambda: None)

        def create(*args, **kwargs):
            self.config.write_text(json.dumps(self.settings))
            return child

        with patch.object(module, "read_health", return_value=self.health), \
                patch.object(module.subprocess, "Popen", side_effect=create) as spawn:
            self.assertEqual(module.start_environment(self.run, "/benchmark", "/source"), (42, True))
        command = spawn.call_args.args[0]
        self.assertTrue(command[1].endswith("parallel_environment_server.py"))
        self.assertEqual(command[command.index("--environment-processes") + 1], "4")
        self.assertEqual(command[command.index("--io-workers") + 1], "4")
        self.assertEqual(command[command.index("--allowed-splits") + 1], "train")
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(spawn.call_args.kwargs["env"]["OMP_NUM_THREADS"], "1")

    def test_incompatible_new_service_stops_only_owned_child(self):
        child = types.SimpleNamespace(pid=42, poll=lambda: None)
        from unittest.mock import Mock

        child.terminate, child.wait = Mock(), Mock()

        def create(*args, **kwargs):
            self.config.write_text(json.dumps(self.settings))
            return child

        with patch.object(module, "read_health", return_value={"ready": True}), \
                patch.object(module.subprocess, "Popen", side_effect=create), self.assertRaises(RuntimeError):
            module.start_environment(self.run, "/benchmark", "/source")
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=30)

    def test_partial_startup_config_is_observed_again_before_cleanup(self):
        child = types.SimpleNamespace(pid=42, poll=lambda: None)

        def create(*args, **kwargs):
            self.config.write_text('{"url":')
            return child

        def finish_write(_):
            self.config.write_text(json.dumps(self.settings))

        with patch.object(module, "read_health", return_value=self.health), \
                patch.object(module.subprocess, "Popen", side_effect=create), \
                patch.object(module.time, "sleep", side_effect=finish_write):
            self.assertEqual(module.start_environment(self.run, "/benchmark", "/source"), (42, True))


if __name__ == "__main__":
    unittest.main()
