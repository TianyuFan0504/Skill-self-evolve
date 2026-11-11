"""Stdlib-only isolated installer, runtime gate and checkpoint provenance tests."""

import ast
import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "verl/experimental/skill_harness"


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


compat = load("compat_offline_test", DIRECTORY / "uniagent_compat.py")
bootstrap = load("compat_bootstrap_test", DIRECTORY / "bootstrap.py")


class CompatibilityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.adapter = self.root / compat.ADAPTER_RELATIVE_PATH
        self.adapter.parent.mkdir(parents=True)
        # Minimal contexts, not a vendored dependency. Real pinned source is
        # tested separately when the explicitly supplied staging path exists.
        self.before = b"# isolated fixture\n" + b"\n".join(before for before, after in compat.EDITS)
        self.after = self.before
        for before, after in compat.EDITS:
            self.after = self.after.replace(before, after, 1)
        self.adapter.write_bytes(self.before)
        self.adapter.chmod(0o644)
        for name, value in (
            ("PRISTINE_SHA256", hashlib.sha256(self.before).hexdigest()),
            ("PATCHED_SHA256", hashlib.sha256(self.after).hexdigest()),
        ):
            context = patch.object(compat, name, value)
            context.start()
            self.addCleanup(context.stop)

    def test_verification_does_not_apply_and_prints_explicit_offline_remediation(self):
        before = self.adapter.read_bytes()
        with self.assertRaisesRegex(compat.UniAgentCompatibilityError, "--apply"):
            compat.require_patched_adapter(self.root)
        self.assertEqual(self.adapter.read_bytes(), before)
        self.assertEqual(list(self.adapter.parent.glob("*.bak")), [])

    def test_apply_is_atomic_backed_up_mode_preserving_and_idempotent(self):
        first = compat.apply_patch(self.root)
        self.assertEqual(self.adapter.read_bytes(), self.after)
        self.assertEqual(Path(first["backup"]).read_bytes(), self.before)
        self.assertEqual(self.adapter.stat().st_mode & 0o777, 0o644)
        again = compat.apply_patch(self.root)
        self.assertEqual(again["action"], "already_applied")
        self.assertEqual(again["adapter_sha256"], first["adapter_sha256"])
        self.assertEqual(len(list(self.adapter.parent.glob("*.bak"))), 1)

    def test_unknown_source_and_unknown_backup_are_never_overwritten(self):
        self.adapter.write_bytes(b"unrecognized external edit")
        with self.assertRaises(compat.UniAgentCompatibilityError):
            compat.apply_patch(self.root)
        self.assertEqual(self.adapter.read_bytes(), b"unrecognized external edit")
        self.adapter.write_bytes(self.before)
        backup = self.adapter.with_name(self.adapter.name + ".before-" + compat.PATCH_ID + ".bak")
        backup.write_bytes(b"unknown backup")
        with self.assertRaises(compat.UniAgentCompatibilityError):
            compat.apply_patch(self.root)
        self.assertEqual(self.adapter.read_bytes(), self.before)
        self.assertEqual(backup.read_bytes(), b"unknown backup")

    def test_interrupted_atomic_publish_keeps_pristine_source_and_retryable_backup(self):
        with patch.object(compat.os, "replace", side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError):
                compat.apply_patch(self.root)
        self.assertEqual(self.adapter.read_bytes(), self.before)
        self.assertEqual(list(self.adapter.parent.glob(".adapter-compat-*")), [])
        self.assertEqual(compat.apply_patch(self.root)["action"], "applied")

    def test_actual_imported_dependency_wins_over_desired_environment_path(self):
        module = types.ModuleType("uni_agent")
        module.__file__ = str(self.root / "uni_agent/__init__.py")
        with (
            patch.dict(sys.modules, {"uni_agent": module}),
            patch.dict(os.environ, {"UNI_AGENT_SOURCE_ROOT": "/wrong"}),
        ):
            self.assertEqual(compat.actual_adapter_sha256(), hashlib.sha256(self.before).hexdigest())

    def test_non_task_cpu_provenance_cannot_hide_external_hash_from_code_fingerprint(self):
        tree = ast.parse((DIRECTORY / "joint_trainer.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_source_fingerprint"
        )
        namespace = {
            "Path": Path,
            "hashlib": hashlib,
            "_digest": compat._sha256,
            "verl": types.SimpleNamespace(__file__=str(ROOT / "verl/__init__.py")),
            "actual_adapter_sha256": compat.actual_adapter_sha256,
        }
        namespace["_digest"] = lambda value: hashlib.sha256(repr(value).encode()).hexdigest()
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_source_fingerprint", "exec"), namespace)
        module = types.ModuleType("uni_agent")
        module.__file__ = str(self.root / "uni_agent/__init__.py")
        with patch.dict(sys.modules, {"uni_agent": module}):
            first = namespace["_source_fingerprint"]()
            compat.apply_patch(self.root)
            self.assertNotEqual(namespace["_source_fingerprint"](), first)

    def test_cached_compatibility_refuses_changed_actual_adapter(self):
        tree = ast.parse((DIRECTORY / "joint_trainer.py").read_text())
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkillJointSyncTrainer"
        )
        function = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compatibility"
        )
        namespace = {"actual_adapter_sha256": compat.actual_adapter_sha256}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_compatibility", "exec"), namespace)
        holder = types.SimpleNamespace(
            _joint_compatibility={"versions": {"uni_agent_adapter": hashlib.sha256(self.before).hexdigest()}}
        )
        module = types.ModuleType("uni_agent")
        module.__file__ = str(self.root / "uni_agent/__init__.py")
        with patch.dict(sys.modules, {"uni_agent": module}):
            self.assertIs(namespace["_compatibility"](holder), holder._joint_compatibility)
            compat.apply_patch(self.root)
            with self.assertRaisesRegex(RuntimeError, "adapter changed"):
                namespace["_compatibility"](holder)


class PinnedSourceTest(unittest.TestCase):
    def setUp(self):
        value = os.environ.get("SKILL_UNIAGENT_PRISTINE_ADAPTER")
        if not value:
            self.skipTest("Supply the independently archived pristine adapter for exact pinned-source checks")
        self.pristine = Path(value).read_bytes()
        self.assertEqual(hashlib.sha256(self.pristine).hexdigest(), compat.PRISTINE_SHA256)

    def test_real_frozen_source_produces_exact_approved_hash(self):
        self.assertEqual(hashlib.sha256(compat._patched_bytes(self.pristine)).hexdigest(), compat.PATCHED_SHA256)

    def test_runtime_false_records_original_sha_but_task_and_worker_require_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            core, uni = root / "core", root / "uni"
            (core / "verl/version").mkdir(parents=True)
            (core / "verl/version/version").write_text("0.9.1\n")
            (core / "verl/__init__.py").write_text("")
            target = uni / compat.ADAPTER_RELATIVE_PATH
            target.parent.mkdir(parents=True)
            (uni / "uni_agent/__init__.py").write_text("")
            target.write_bytes(self.pristine)
            modules = {
                name: types.ModuleType(name)
                for name in ("verl", "uni_agent", "verl.experimental.skill_harness.alfworld.task")
            }
            modules["verl"].__file__, modules["verl"].__version__ = str(core / "verl/__init__.py"), "0.9.1"
            modules["uni_agent"].__file__ = str(uni / "uni_agent/__init__.py")
            imported = []

            def import_module(name):
                imported.append(name)
                return modules[name]

            old_path = list(sys.path)
            try:
                with (
                    patch.dict(
                        os.environ,
                        {
                            "VERL_SOURCE_ROOT": str(core),
                            "UNI_AGENT_SOURCE_ROOT": str(uni),
                            "UNI_AGENT_DEPENDENCY_SITE": "",
                        },
                    ),
                    patch.object(bootstrap.importlib, "import_module", side_effect=import_module),
                ):
                    provenance = bootstrap.configure_runtime(register_tasks=False)
                    self.assertEqual(provenance["uni_agent_adapter_sha256"], compat.PRISTINE_SHA256)
                    with self.assertRaises(RuntimeError):
                        bootstrap.configure_runtime(register_tasks=True)
                    self.assertNotIn("verl.experimental.skill_harness.alfworld.task", imported)
                    compat.apply_patch(uni)
                    self.assertTrue(bootstrap.configure_runtime(register_tasks=True)["tasks_registered"])
                    bootstrap.configure_worker()
            finally:
                sys.path[:] = old_path
