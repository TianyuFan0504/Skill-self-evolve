"""CPU checks for the native ALFWorld solver's text-only tool declaration."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RECIPE = Path(__file__).resolve().parents[1]
WRAPPER = RECIPE / "scripts/codex_entrypoint.sh"


def load_launcher():
    spec = importlib.util.spec_from_file_location("_solver_tools_launcher", RECIPE / "scripts/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SolverNativeToolsTests(unittest.TestCase):
    def test_wrapper_preserves_native_exec_and_text_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instructions = root / "instructions with spaces.md"
            instructions.write_text("Text ALFWorld instructions.\n")
            native = root / "native"
            capture = root / "argv.json"
            native.write_text(
                "#!/usr/bin/env python3\n"
                "import json,os,sys\n"
                "with open(os.environ['ARG_CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
            )
            native.chmod(0o755)
            tail = ["exec", "--json", "--ephemeral", "plain task text\nwith a second line"]
            environment = {
                **os.environ,
                "CODEX_NATIVE_BIN": str(native),
                "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": str(instructions),
                "ARG_CAPTURE": str(capture),
            }
            result = subprocess.run(["bash", str(WRAPPER), *tail], env=environment, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            arguments = json.loads(capture.read_text())
            self.assertEqual(arguments[-len(tail) :], tail)
            disabled = {arguments[i + 1] for i, word in enumerate(arguments) if word == "--disable"}
            enabled = {arguments[i + 1] for i, word in enumerate(arguments) if word == "--enable"}
            self.assertTrue(
                {
                    "multi_agent",
                    "multi_agent_v2",
                    "view_image",
                    "image_generation",
                    "browser_use",
                    "browser_use_external",
                    "computer_use",
                    "apps",
                    "plugins",
                    "hooks",
                    "skill_search",
                    "workspace_dependencies",
                    "sleep_tool",
                    "goals",
                }.issubset(disabled)
            )
            self.assertTrue({"shell_tool", "unified_exec", "unified_exec_tty"}.issubset(enabled))
            self.assertFalse(disabled & {"shell_tool", "unified_exec", "unified_exec_tty"})
            overrides = [arguments[i + 1] for i, word in enumerate(arguments) if word == "-c"]
            self.assertIn('web_search="disabled"', overrides)
            self.assertIn("tools.experimental_request_user_input.enabled=false", overrides)
            self.assertIn("allow_login_shell=false", overrides)
            self.assertIn("skills.include_instructions=false", overrides)
            self.assertIn("model_instructions_file=" + json.dumps(str(instructions)), overrides)

    def test_wrapper_rejects_missing_native_or_instructions(self):
        result = subprocess.run(
            ["bash", str(WRAPPER), "exec"],
            env={
                **os.environ,
                "CODEX_NATIVE_BIN": "/missing/native",
                "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": "/missing/prompt",
            },
            capture_output=True,
        )
        self.assertEqual(result.returncode, 78)

    def test_catalog_declares_text_without_reducing_capacity_or_native_tools(self):
        launcher = load_launcher()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark, run = root / "benchmark", root / "run"
            (benchmark / "adapter").mkdir(parents=True)
            run.mkdir()
            original = {
                "models": [
                    {
                        "slug": "old",
                        "display_name": "old",
                        "context_window": 65536,
                        "input_modalities": ["text", "image"],
                        "shell_type": "shell_command",
                        "experimental_supported_tools": [],
                        "truncation_policy": {"mode": "tokens", "limit": 10000},
                    }
                ]
            }
            source = benchmark / "adapter/model_catalog_qwen35_4b.json"
            source.write_text(json.dumps(original))
            (benchmark / "adapter/alfworld_step.py").write_text("# text environment helper\n")
            mapping = {
                "/opt/alfworld-tools/skill_self_evolve": root / "public",
                "/opt/alfworld-tools/alfworld-step": root / "alfworld-step",
            }
            with mock.patch.object(launcher, "Path", side_effect=lambda value: mapping.get(str(value), Path(value))):
                with mock.patch.dict(os.environ):
                    task_path, _ = launcher.install_codex_runtime(run, benchmark, 36864)
            model = json.loads((run / "model_catalog.json").read_text())["models"][0]
            self.assertEqual(model["input_modalities"], ["text"])
            self.assertEqual(model["context_window"], 36864)
            self.assertTrue(model["supports_parallel_tool_calls"])
            for key in ("shell_type", "experimental_supported_tools", "truncation_policy"):
                self.assertEqual(model[key], original["models"][0][key])
            self.assertEqual(json.loads(source.read_text()), original)
            self.assertEqual(json.loads(task_path.read_text())[0]["agent"]["name"], "codex")


if __name__ == "__main__":
    unittest.main()
