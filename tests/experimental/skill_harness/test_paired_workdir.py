# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Real CPU directory/subprocess tests for controlled counterfactual cwd reuse."""

from __future__ import annotations

import ast
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


def load_workspace_class():
    path = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/task.py"
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "PairedWorkspace")
    scope = {
        "fcntl": fcntl, "hashlib": hashlib, "json": json, "os": os, "re": re, "shutil": shutil,
        "stat": stat, "tarfile": tarfile, "uuid": uuid, "Path": Path,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["PairedWorkspace"]


Workspace = load_workspace_class()


def load_sandbox_class():
    path = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld/task.py"
    tree = ast.parse(path.read_text())
    spec = importlib.util.spec_from_file_location("_paired_capture_guard", path.with_name("terminal_guard.py"))
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "AlfworldUserSandbox")
    scope = {
        "asyncio": asyncio, "json": json, "os": os, "signal": __import__("signal"), "time": time, "Path": Path,
        "LocalSandbox": object, "ExecResult": types.SimpleNamespace,
        "monitor_terminal_process": guard.monitor_terminal_process,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["AlfworldUserSandbox"]


class PairedWorkdirTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.work_root = self.root / "work"
        self.work_root.mkdir(mode=0o711)
        self.group = "a" * 64
        self.homes = []

    def tearDown(self):
        for home in self.homes:
            shutil.rmtree(home, ignore_errors=True)
        self.temporary.cleanup()

    def workspace(self, variant="with_skill", group=None):
        output = self.root / (variant + "_" + uuid.uuid4().hex)
        output.mkdir(mode=0o700)
        return Workspace(
            group=group or self.group, variant=variant, work_root=self.work_root, output=output,
            agent_uid=os.geteuid(), agent_gid=os.getegid(),
        )

    def native_home(self):
        home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        home.mkdir(mode=0o700)
        (home / "native-state.json").write_text('{"history":"private prior branch"}')
        self.homes.append(home)
        return home

    def child(self, workdir, name):
        script = (
            "import json,pathlib,sys\n"
            "root=pathlib.Path.cwd()\n"
            "print(json.dumps({'cwd':str(root),'initial_files':sorted(p.name for p in root.iterdir())}))\n"
            "(root/sys.argv[1]).write_text('generated branch file')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, name], cwd=workdir, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_pair_has_identical_empty_cwd_and_distinct_removed_native_homes(self):
        first = self.workspace()
        first_path = first.acquire()
        first_observed = self.child(first_path, "skill-canary.txt")
        first_home = self.native_home()
        first_manifest = first.finish([first_home])
        self.assertFalse(first_path.exists())
        self.assertFalse(first_home.exists())
        second = self.workspace("without_skill")
        second_path = second.acquire()
        second_observed = self.child(second_path, "other-branch.txt")
        second_home = self.native_home()
        second_manifest = second.finish([second_home])
        self.assertEqual(first_path, second_path)
        self.assertEqual(first_observed["cwd"], second_observed["cwd"])
        self.assertEqual(first_observed["initial_files"], [])
        self.assertEqual(second_observed["initial_files"], [])
        self.assertNotEqual(first_manifest["codex_homes"], second_manifest["codex_homes"])
        for manifest in (first_manifest, second_manifest):
            self.assertTrue(manifest["cleanup_completed"])
            self.assertTrue(manifest["empty_before_run"])
            self.assertEqual(len(manifest["archives"]), 2)
            for archive in manifest["archives"]:
                self.assertTrue(archive["removed"])
                self.assertEqual(hashlib.sha256(Path(archive["path"]).read_bytes()).hexdigest(), archive["sha256"])
        with tarfile.open(first.output / "paired-workspace.tar") as archive:
            self.assertIn("workspace/skill-canary.txt", archive.getnames())

    def test_other_pairs_are_isolated_and_concurrent_same_pair_is_rejected(self):
        first = self.workspace()
        first.acquire()
        other = self.workspace(group="b" * 64)
        other.acquire()
        self.assertNotEqual(first.workdir, other.workdir)
        child_output = self.root / "concurrent"
        child_output.mkdir()
        script = (
            "import os,runpy\n"
            f"W=runpy.run_path({str(Path(__file__).resolve())!r},run_name='_paired_loader')['Workspace']\n"
            f"w=W(group={'a' * 64!r},variant='without_skill',work_root={str(self.work_root)!r},"
            f"output={str(child_output)!r},agent_uid=os.geteuid(),agent_gid=os.getegid())\n"
            "try: w.acquire()\n"
            "except BlockingIOError: raise SystemExit(23)\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 23, result.stderr)
        first.finish([self.native_home()])
        other.finish([self.native_home()])

    def test_wrong_uid_marker_and_unowned_or_dirty_path_are_rejected(self):
        first = self.workspace()
        first.acquire()
        first.finish([self.native_home()])
        marker = json.loads(first.marker_path.read_text())
        marker["agent_uid"] += 1
        first.marker_path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, "UID"):
            self.workspace("without_skill").acquire()
        orphan = self.work_root / ("credit_" + "c" * 64)
        orphan.mkdir()
        (orphan / "stale.txt").write_text("must not be reused")
        with self.assertRaises(ValueError):
            self.workspace(group="c" * 64).acquire()

    def test_cleanup_failure_preserves_archive_quarantines_and_blocks_reuse(self):
        workspace = self.workspace()
        workspace.acquire()
        self.child(workspace.workdir, "retained.txt")
        home = self.native_home()
        with patch.object(shutil, "rmtree", side_effect=OSError("simulated cleanup failure")):
            with self.assertRaisesRegex(OSError, "cleanup failure"):
                workspace.finish([home])
        manifest = json.loads((workspace.output / "paired-workdir.json").read_text())
        self.assertFalse(manifest["cleanup_completed"])
        self.assertTrue((workspace.output / "paired-workspace.tar").is_file())
        self.assertEqual(workspace.workdir.stat().st_uid, os.geteuid())
        self.assertEqual(stat.S_IMODE(workspace.workdir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            self.workspace("without_skill").acquire()

    def test_archive_keeps_symlink_without_following_private_target(self):
        workspace = self.workspace()
        workspace.acquire()
        secret = self.root / "host-private.txt"
        secret.write_text("DO_NOT_DEREFERENCE_SECRET")
        (workspace.workdir / "link").symlink_to(secret)
        workspace.finish([self.native_home()])
        archive_path = workspace.output / "paired-workspace.tar"
        with tarfile.open(archive_path) as archive:
            self.assertTrue(archive.getmember("workspace/link").issym())
        self.assertNotIn(b"DO_NOT_DEREFERENCE_SECRET", archive_path.read_bytes())

    def test_native_home_path_cannot_be_reused_by_the_other_branch(self):
        first = self.workspace()
        first.acquire()
        home = self.native_home()
        first.finish([home])
        home.mkdir(mode=0o700)
        second = self.workspace("without_skill")
        second.acquire()
        with self.assertRaisesRegex(ValueError, "distinct native Codex homes"):
            second.finish([home])

    def test_invalid_group_or_variant_never_creates_control_paths(self):
        for group in ("../escape", "a" * 63, "A" * 64):
            with self.subTest(group=group), self.assertRaises(ValueError):
                self.workspace(group=group)
        with self.assertRaises(ValueError):
            self.workspace(variant="unexpected")
        self.assertEqual(list(self.work_root.iterdir()), [])


class NativeHomeTrackingTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_host_command_records_fresh_home_without_agent_prompt_changes(self):
        sandbox_class = load_sandbox_class()
        sandbox = object.__new__(sandbox_class)
        home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        original = asyncio.create_subprocess_exec

        async def direct_exec(*argv, **kwargs):
            return await original(*argv[4:], **kwargs)

        try:
            with tempfile.TemporaryDirectory() as directory:
                sandbox.output = Path(directory)
                sandbox.session_url = "http://127.0.0.1/unused"

                async def nonterminal():
                    return {"done": False, "won": False}

                sandbox._observe_terminal_state = nonterminal
                sandbox.TERMINAL_POLL_SECONDS = .005
                command = f"mkdir -p {home} && env CODEX_HOME={home} /usr/bin/printf 'native-output\\n'"
                with patch.object(asyncio, "create_subprocess_exec", side_effect=direct_exec):
                    result = await sandbox._exec(["bash", "-c", command], timeout=2, workdir=directory)
                    self.assertEqual(result.exit_code, 0)
                    self.assertEqual(result.stdout, "native-output\n")
                    self.assertEqual(sandbox.native_codex_homes, [str(home)])
                    with self.assertRaisesRegex(ValueError, "fresh per-episode home"):
                        await sandbox._exec(["bash", "-c", command], timeout=2, workdir=directory)
        finally:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
