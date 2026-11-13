"""CPU checks of resource identity cleanup, independent of optional archives."""

import json
import os
import shutil
import tarfile
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from _native_runtime_test_support import native_files as files


class NativeCleanupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.workspace = self.root / uuid.uuid4().hex
        self.workspace.mkdir()
        (self.workspace / "state").write_text("private state")
        self.home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        self.home.mkdir()
        self.addCleanup(shutil.rmtree, self.home, True)
        (self.home / "history").write_text("native diagnostic")

    def args(self):
        return dict(
            agent_uid=os.geteuid(),
            account_lease_id="lease",
            workspace=self.workspace,
            workspace_root=self.root,
            codex_homes=[self.home],
        )

    def test_cleanup_has_ownership_receipt_without_archive(self):
        before = self.workspace.stat()
        receipt = files.cleanup_native_files(self.output, **self.args())
        self.assertEqual(receipt["status"], "cleaned")
        self.assertFalse(self.home.exists())
        self.assertFalse(self.workspace.exists())
        self.assertFalse((self.output / "native-files.tar.gz").exists())
        row = next(row for row in receipt["sources"] if row["kind"] == "workspace")
        self.assertEqual((row["device"], row["inode"]), (before.st_dev, before.st_ino))
        self.assertTrue(row["removed"])
        self.assertEqual(json.loads((self.output / "native-cleanup.json").read_text()), receipt)

    def test_optional_archive_preserves_sources_and_cleanup_survives_archive_failure(self):
        receipt = files.archive_native_files(self.output, **self.args())
        self.assertTrue(self.home.exists())
        self.assertTrue(self.workspace.exists())
        with tarfile.open(receipt["archive_path"]) as archive:
            self.assertTrue(any(name.endswith("/history") for name in archive.getnames()))
        with patch.object(files.tarfile, "open", side_effect=OSError("no diagnostic space")):
            with self.assertRaises(OSError):
                files.archive_native_files(self.output, **self.args())
            closed = files.cleanup_native_files(self.output, **self.args())
        self.assertEqual(closed["status"], "cleaned")

    def test_symlink_root_rejected_without_removing_target(self):
        linked = self.root / uuid.uuid4().hex
        linked.symlink_to(self.workspace, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "identity"):
            files.cleanup_native_files(self.output, agent_uid=os.geteuid(), workspace=linked, workspace_root=self.root)
        self.assertTrue((self.workspace / "state").exists())

    def test_workspace_symlinks_do_not_follow_external_targets(self):
        victim = self.root / "keep"
        victim.write_text("kept")
        (self.workspace / "link").symlink_to(victim)
        files.cleanup_native_files(self.output, **self.args())
        self.assertEqual(victim.read_text(), "kept")

    def test_failed_cleanup_retains_explicit_incomplete_receipt(self):
        with patch.object(files.shutil, "rmtree", side_effect=OSError("cleanup failed")):
            with self.assertRaises(OSError):
                files.cleanup_native_files(self.output, **self.args())
        receipt = json.loads((self.output / "native-cleanup.json").read_text())
        self.assertEqual(receipt["status"], "cleaning")
        self.assertFalse(any(row["removed"] for row in receipt["sources"]))
        self.assertTrue(self.workspace.exists())


if __name__ == "__main__":
    unittest.main()
