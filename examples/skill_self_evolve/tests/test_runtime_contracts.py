"""CPU-only checks for source pinning and the unmodified native Codex boundary."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "skill_runtime_bootstrap_test", ROOT / "verl/experimental/skill_harness/bootstrap.py"
)
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class RuntimeContractsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "core"
        (self.source / "verl/version").mkdir(parents=True)
        (self.source / "verl/version/version").write_text("0.9.1\n")
        (self.source / "verl/__init__.py").write_text("")
        self.uni = self.root / "uni-agent"
        (self.uni / "uni_agent").mkdir(parents=True)
        (self.uni / "uni_agent/__init__.py").write_text("")
        self.venv = self.root / "dependencies"
        self.site = self.venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        self.site.mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text(
            f"implementation = CPython\nversion_info = {sys.version_info.major}.{sys.version_info.minor}.0\n"
        )
        self.verl = types.ModuleType("verl")
        self.verl.__file__ = str(self.source / "verl/__init__.py")
        self.verl.__version__ = "0.9.1"
        self.uni_module = types.ModuleType("uni_agent")
        self.uni_module.__file__ = str(self.uni / "uni_agent/__init__.py")
        self.environment = {
            "VERL_SOURCE_ROOT": str(self.source),
            "UNI_AGENT_SOURCE_ROOT": str(self.uni),
            "UNI_AGENT_DEPENDENCY_SITE": str(self.site),
        }

    def configure(self):
        old_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        environment = patch.dict(os.environ, self.environment)
        environment.start()
        self.addCleanup(environment.stop)
        modules = patch.dict(sys.modules, {"verl": self.verl, "uni_agent": self.uni_module})
        modules.start()
        self.addCleanup(modules.stop)
        return BOOTSTRAP.configure_runtime(register_tasks=False)

    def test_dependency_pth_never_executes_and_core_stays_first(self):
        (self.site / "unexpected.pth").write_text("import os; os.environ['UNEXPECTED_PTH_EXECUTION']='yes'\n")
        result = self.configure()
        self.assertEqual(sys.path[:2], [str(self.source), str(self.uni)])
        self.assertEqual(sys.path[-1], str(self.site))
        self.assertNotIn("UNEXPECTED_PTH_EXECUTION", os.environ)
        self.assertEqual(result["verl_version"], "0.9.1")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_wrong_loaded_source_is_rejected(self):
        self.verl.__file__ = str(self.root / "bundled-verl/verl/__init__.py")
        with self.assertRaisesRegex(RuntimeError, "Wrong verl import"):
            self.configure()

    def test_wrong_loaded_version_is_rejected(self):
        self.verl.__version__ = "0.10.0.dev0"
        with self.assertRaisesRegex(RuntimeError, "Wrong verl import"):
            self.configure()

    def test_another_python_minor_abi_is_rejected(self):
        (self.venv / "pyvenv.cfg").write_text(
            f"implementation = CPython\nversion_info = {sys.version_info.major}.{sys.version_info.minor + 1}.0\n"
        )
        with self.assertRaisesRegex(RuntimeError, "ABI"):
            self.configure()

    def test_native_wrapper_preserves_arguments_and_quotes_instruction_path(self):
        native = self.root / "native-codex"
        native.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        native.chmod(0o700)
        instructions = self.root / 'instructions with "quotes".md'
        instructions.write_text("Use only the ALFWorld game tool.\n")
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "examples/skill_self_evolve/scripts/codex_entrypoint.sh"),
                "exec",
                "--model",
                "Qwen3.5-4B-nothink",
            ],
            env={**os.environ, "CODEX_NATIVE_BIN": str(native), "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": str(instructions)},
            text=True,
            capture_output=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "-c")
        self.assertEqual(json.loads(lines[1].split("=", 1)[1]), str(instructions))
        self.assertEqual(lines[2:4], ["-c", "allow_login_shell=false"])
        self.assertEqual(lines[4:6], ["--enable", "skip_host_skill_discovery"])
        self.assertEqual(lines[6:8], ["-c", "skills.include_instructions=false"])
        self.assertEqual(lines[8:], ["exec", "--model", "Qwen3.5-4B-nothink"])

    def test_checkpoint_fingerprint_binds_solver_input_assets(self):
        tree = ast.parse((ROOT / "verl/experimental/skill_harness/joint_trainer.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_source_fingerprint"
        )
        namespace = {
            "Path": Path,
            "verl": self.verl,
            "hashlib": hashlib,
            "actual_adapter_sha256": lambda: "verified-adapter",
            "_digest": lambda value: hashlib.sha256(repr(value).encode()).hexdigest(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_source_fingerprint", "exec"), namespace)
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(("verl/", "examples/")) and value.endswith((".py", ".sh", ".md", ".txt")):
                    path = self.source / value
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("original bytes\n")
        initial = namespace["_source_fingerprint"]()
        for relative in (
            "examples/skill_self_evolve/scripts/codex_entrypoint.sh",
            "examples/skill_self_evolve/scripts/launch.py",
            "examples/skill_self_evolve/prompts/solver_system.md",
            "examples/skill_self_evolve/prompts/solver_task.txt",
        ):
            with self.subTest(asset=relative):
                path = self.source / relative
                original = path.read_bytes()
                path.write_bytes(original + b"changed\n")
                self.assertNotEqual(namespace["_source_fingerprint"](), initial)
                path.write_bytes(original)
                self.assertEqual(namespace["_source_fingerprint"](), initial)
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    namespace["_source_fingerprint"]()
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
