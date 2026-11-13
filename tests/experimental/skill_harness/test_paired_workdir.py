"""CPU paired workspace tests with direct module imports and real directories."""

import json
import os
import shutil
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from _native_runtime_test_support import task

Workspace = task.PairedWorkspace


class PairedWorkdirTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.work.mkdir()

    def workspace(self, variant):
        output = self.root / variant
        output.mkdir()
        return Workspace(
            group="a" * 64,
            variant=variant,
            work_root=self.work,
            output=output,
            agent_uid=os.geteuid(),
            agent_gid=os.getegid(),
        )

    def home(self):
        home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        home.mkdir()
        self.addCleanup(shutil.rmtree, home, True)
        return home

    def test_pair_reuses_empty_cwd_with_distinct_removed_homes_without_archive(self):
        first = self.workspace("with_skill")
        first_path = first.acquire()
        (first_path / "private-state").write_text("must not cross branches")
        first_home = self.home()
        first.finish([first_home])
        second = self.workspace("without_skill")
        second_path = second.acquire()
        self.assertEqual(first_path, second_path)
        self.assertEqual(list(second_path.iterdir()), [])
        second_home = self.home()
        provenance = second.finish([second_home])
        self.assertTrue(provenance["cleanup_completed"])
        self.assertTrue(provenance["empty_before_run"])
        self.assertFalse(first_home.exists())
        self.assertFalse(second_home.exists())
        self.assertEqual(list(self.root.glob("*/native-files.tar.gz")), [])

    def test_same_pair_lock_rejects_concurrent_branch(self):
        first = self.workspace("with_skill")
        first.acquire()
        try:
            with self.assertRaises(BlockingIOError):
                self.workspace("without_skill").acquire()
        finally:
            first.finish([])

    def test_failed_cleanup_marks_only_pair_unavailable(self):
        first = self.workspace("with_skill")
        workdir = first.acquire()
        home = self.home()
        with patch.object(shutil, "rmtree", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                first.finish([home])
        self.assertTrue(workdir.exists())
        self.assertEqual(json.loads(first.marker_path.read_text())["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "reconciliation"):
            self.workspace("without_skill").acquire()

    def test_unconfirmed_drain_retains_sources(self):
        first = self.workspace("with_skill")
        workdir = first.acquire()
        (workdir / "state").write_text("still owned")
        first.retain("drain unconfirmed")
        self.assertTrue((workdir / "state").exists())
        self.assertFalse(first.active)
        self.assertFalse(first.provenance["cleanup_completed"])


if __name__ == "__main__":
    unittest.main()
