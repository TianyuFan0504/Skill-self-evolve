"""CPU-only storage failure boundaries; no native app, GPU, Ray, or model."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

RECIPE = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, RECIPE / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("_native_storage_preflight", "preflight.py")
launch = load("_native_storage_launch", "launch.py")
BYTE_LIMIT = 10 * 1024**3
INODE_LIMIT = 650_000


class StorageTest(unittest.TestCase):
    def space(self, available_bytes=BYTE_LIMIT, available_inodes=INODE_LIMIT):
        return SimpleNamespace(f_bavail=available_bytes, f_frsize=1, f_favail=available_inodes)

    def inspect(self, space, *, devices=(7, 7), environ=None):
        with (
            patch.object(preflight.os, "statvfs", return_value=space) as query,
            patch.object(preflight.os, "stat", side_effect=[SimpleNamespace(st_dev=value) for value in devices]),
        ):
            result = preflight.native_tmp_storage_preflight(environ=environ or {})
            query.assert_called_once_with("/tmp")
            return result

    def test_exact_default_limits_pass_on_same_or_separate_tmp_filesystem(self):
        for devices in ((7, 7), (8, 7)):
            with self.subTest(devices=devices):
                result = self.inspect(self.space(), devices=devices)
                self.assertTrue(result["passed"])
                self.assertEqual(result["shares_root_filesystem"], devices[0] == devices[1])
                self.assertEqual(result["available_bytes"], BYTE_LIMIT)
                self.assertEqual(result["available_inodes"], INODE_LIMIT)
                self.assertEqual(result["default_budget_basis"]["concurrent_native_homes"], 64)

    def test_large_other_storage_does_not_hide_byte_or_inode_exhaustion_in_tmp(self):
        for space in (self.space(BYTE_LIMIT - 1), self.space(100 * BYTE_LIMIT, INODE_LIMIT - 1)):
            with self.subTest(space=space), self.assertRaises(preflight.NativeTmpStorageError) as raised:
                self.inspect(space, devices=(8, 7), environ={"TMPDIR": "/large-shm", "NAS_FREE_BYTES": str(10**15)})
            self.assertIn("/tmp", str(raised.exception))
            self.assertIn("MIN_NATIVE_TMP_FREE_INODES", str(raised.exception))
            self.assertFalse(raised.exception.report["passed"])
            self.assertEqual(raised.exception.report["available_inodes"], space.f_favail)

    def test_limits_can_be_explicitly_overridden_as_positive_integer_values(self):
        report = self.inspect(
            self.space(16384, 9),
            environ={
                "MIN_NATIVE_TMP_FREE_BYTES": "16384",
                "MIN_NATIVE_TMP_FREE_INODES": "9",
                "ROLLOUT_CONCURRENCY": "64",
                "VAL_BATCH_SIZE": "512",
            },
        )
        self.assertEqual(report["minimum_free_bytes"], 16384)
        self.assertEqual(report["minimum_free_inodes"], 9)
        self.assertEqual(report["default_budget_basis"]["concurrent_native_homes"], 64)

    def test_invalid_limits_fail_before_any_filesystem_or_native_query(self):
        for key in ("MIN_NATIVE_TMP_FREE_BYTES", "MIN_NATIVE_TMP_FREE_INODES"):
            for value in ("0", "-1", "10GiB", "1.5", True):
                with self.subTest(key=key, value=value), patch.object(preflight.os, "statvfs") as query:
                    with self.assertRaisesRegex(ValueError, key):
                        preflight.native_tmp_storage_preflight(environ={key: value})
                    query.assert_not_called()

    def test_main_rejects_storage_before_gpu_inventory_or_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(sys, "argv", ["preflight.py", "--run-dir", directory]),
                patch.object(preflight.os, "statvfs", return_value=self.space(1, 1)),
                patch.object(preflight.subprocess, "check_output") as gpu_query,
            ):
                with self.assertRaises(preflight.NativeTmpStorageError):
                    preflight.main()
                gpu_query.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_resource_telemetry_retains_legacy_bytes_and_reports_real_tmp_inodes(self):
        def statvfs(path):
            return self.space(333, 22) if path == "/tmp" else self.space(111, 11)

        with (
            patch.dict(os.environ, {"TMPDIR": "/large-shm"}),
            patch.object(launch.os, "statvfs", side_effect=statvfs),
            patch.object(
                launch.shutil, "disk_usage", side_effect=lambda path: SimpleNamespace(free=111 if path == "/" else 999)
            ),
        ):
            result = launch.storage_resource_fields()
        self.assertEqual(
            result,
            {
                "root_available_bytes": 111,
                "scratch_available_bytes": 999,
                "root_available_inodes": 11,
                "tmp_available_bytes": 333,
                "tmp_available_inodes": 22,
            },
        )

    def test_gpu_query_failure_still_writes_storage_telemetry(self):
        sampled = threading.Event()
        fields = {
            "root_available_bytes": 100,
            "scratch_available_bytes": 200,
            "root_available_inodes": 11,
            "tmp_available_bytes": 100,
            "tmp_available_inodes": 11,
        }

        def storage():
            sampled.set()
            return fields

        class Child:
            pid = 99999999

            def poll(self):
                return 0

            def wait(self):
                if not sampled.wait(3):
                    raise AssertionError("Storage sampler did not run after the GPU query failed")
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.dict(os.environ, {"VERL_SOURCE_ROOT": directory, "RAY_TMPDIR": directory}),
                patch.object(launch.subprocess, "Popen", return_value=Child()),
                patch.object(launch.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 8)),
                patch.object(launch.signal, "signal"),
                patch.object(launch, "storage_resource_fields", side_effect=storage),
            ):
                self.assertEqual(launch.supervise(root, ["unused-fixture"], None), 0)
            rows = [json.loads(line) for line in (root / "resources.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("sampling_error", rows[0])
            for key, value in fields.items():
                self.assertEqual(rows[0][key], value)


if __name__ == "__main__":
    unittest.main()
