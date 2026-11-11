"""CPU filesystem and ENOSPC lifecycle regressions; no models or real services."""

from __future__ import annotations

import ast
import asyncio
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld"
SPEC = importlib.util.spec_from_file_location("_native_files_cpu", DIRECTORY / "native_files.py")
FILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILES)


class NativeFilesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        self.home.mkdir(mode=0o700)
        self.addCleanup(lambda: shutil.rmtree(self.home, ignore_errors=True))
        (self.home / "state.sqlite").write_bytes(b"exact original native state\x00")
        (self.home / "events.jsonl").write_text('{"event":"native event"}\n')
        self.catalog = Path("/tmp") / ("codex-model-catalog-" + uuid.uuid4().hex + ".json")
        self.catalog.write_bytes(b'{"model":"original model config"}\n')
        self.addCleanup(lambda: self.catalog.unlink(missing_ok=True))

    def archive(self, **kwargs):
        return FILES.archive_native_files(
            self.output, codex_homes=[self.home], model_catalogs=[self.catalog], agent_uid=os.geteuid(), **kwargs
        )

    def test_all_original_state_events_catalog_and_workspace_are_archived_before_removal(self):
        root = self.output / "scratch"
        root.mkdir()
        work = root / uuid.uuid4().hex
        work.mkdir()
        (work / "generated.txt").write_bytes(b"original workspace file")
        result = self.archive(workspace=work, workspace_root=root)
        self.assertEqual(result["status"], "archived_and_removed")
        self.assertTrue(all(row["removed"] for row in result["sources"]))
        self.assertFalse(self.home.exists())
        self.assertFalse(self.catalog.exists())
        self.assertFalse(work.exists())
        path = Path(result["archive_path"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result["archive_sha256"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with tarfile.open(path) as archive:
            self.assertEqual(
                archive.extractfile("codex_home-0/state.sqlite").read(), b"exact original native state\x00"
            )
            self.assertEqual(archive.extractfile("codex_home-0/events.jsonl").read(), b'{"event":"native event"}\n')
            self.assertEqual(archive.extractfile("model_catalog-1").read(), b'{"model":"original model config"}\n')
            self.assertEqual(archive.extractfile("workspace-2/generated.txt").read(), b"original workspace file")
        manifest = Path(result["source_manifest_path"]).read_bytes()
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), result["source_manifest_sha256"])
        self.assertEqual(len(json.loads(manifest)), result["source_entry_count"])

    def test_archive_or_fsync_failure_keeps_every_local_source(self):
        with patch.object(FILES.tarfile, "open", side_effect=OSError(errno.ENOSPC, "archive disk full")):
            with self.assertRaises(OSError):
                self.archive()
        self.assertTrue(self.home.exists())
        self.assertTrue(self.catalog.exists())
        with patch.object(FILES.os, "fsync", side_effect=OSError("fsync failed")):
            with self.assertRaises(OSError):
                self.archive()
        self.assertTrue(self.home.exists())
        self.assertTrue(self.catalog.exists())

    def test_catalog_only_cleanup_keeps_existing_paired_home_and_workspace_archives_untouched(self):
        before = self.home.joinpath("state.sqlite").read_bytes()
        paired = self.output / "paired-codex-home-0.tar"
        paired.write_bytes(b"prior independently checked paired archive")
        result = FILES.archive_native_files(self.output, model_catalogs=[self.catalog], agent_uid=os.geteuid())
        self.assertEqual([row["kind"] for row in result["sources"]], ["model_catalog"])
        self.assertEqual(paired.read_bytes(), b"prior independently checked paired archive")
        self.assertEqual(self.home.joinpath("state.sqlite").read_bytes(), before)

    def test_corrupt_or_omitted_archive_preserves_all_sources(self):
        original_open = tarfile.open

        def corrupt(path, mode, *args, **kwargs):
            if mode == "r:gz":
                Path(path).write_bytes(b"corrupt archive bytes")
            return original_open(path, mode, *args, **kwargs)

        with patch.object(FILES.tarfile, "open", side_effect=corrupt):
            with self.assertRaises(tarfile.TarError):
                self.archive()
        self.assertTrue(self.home.exists())
        self.assertTrue(self.catalog.exists())
        for path in self.output.glob("native-*"):
            path.unlink()
        original_add = tarfile.TarFile.add

        def omit(target, name, *args, **kwargs):
            if Path(name).name != "events.jsonl":
                return original_add(target, name, *args, **kwargs)

        with patch.object(tarfile.TarFile, "add", omit):
            with self.assertRaisesRegex(ValueError, "entry inventory"):
                self.archive()
        self.assertTrue(self.home.exists())
        self.assertTrue(self.catalog.exists())

    def test_source_change_during_readback_preserves_every_source(self):
        original = tarfile.open

        def changed(path, mode, *args, **kwargs):
            if mode == "r:gz":
                self.catalog.write_bytes(b"changed native source")
            return original(path, mode, *args, **kwargs)

        with patch.object(FILES.tarfile, "open", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "source changed during archival"):
                self.archive()
        self.assertTrue(self.home.exists())
        self.assertTrue(self.catalog.exists())
        self.assertFalse((self.output / "native-files.json").exists())

    def test_nested_links_preserve_representation_without_following_external_target(self):
        victim = self.output / "external-private-fixture"
        victim.write_bytes(b"not part of the native archive")
        (self.home / "external-link").symlink_to(victim)
        os.link(self.home / "events.jsonl", self.home / "events-hardlink.jsonl")
        result = self.archive()
        with tarfile.open(result["archive_path"]) as archive:
            member = archive.getmember("codex_home-0/external-link")
            self.assertTrue(member.issym())
            self.assertEqual(member.linkname, str(victim))
        self.assertEqual(victim.read_bytes(), b"not part of the native archive")

    def test_unregistered_path_or_symlink_never_removes_other_files(self):
        victim = self.output / "unrelated"
        victim.write_text("keep")
        with self.assertRaises(ValueError):
            FILES.archive_native_files(self.output, model_catalogs=[victim], agent_uid=os.geteuid())
        self.catalog.unlink()
        self.catalog.symlink_to(victim)
        with self.assertRaises(ValueError):
            self.archive()
        self.assertEqual(victim.read_text(), "keep")

    def test_public_solver_write_hook_retains_partial_catalog_ownership(self):
        class LocalWriter:
            async def write_file(self, path, content):
                Path(path).write_bytes(content[:4])
                raise OSError(errno.ENOSPC, "primary catalog disk full")

        tree = ast.parse((DIRECTORY / "task.py").read_text())
        node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "AlfworldUserSandbox")
        scope = {"LocalSandbox": LocalWriter, "Path": Path, "re": re, "os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_catalog_write_hook", "exec"), scope)
        sandbox = object.__new__(scope["AlfworldUserSandbox"])
        with self.assertRaisesRegex(OSError, "primary catalog disk full"):
            asyncio.run(sandbox.write_file(str(self.catalog), b"original catalog bytes"))
        self.assertEqual(sandbox.native_model_catalogs, [str(self.catalog)])
        result = FILES.archive_native_files(
            self.output, model_catalogs=sandbox.native_model_catalogs, agent_uid=os.geteuid()
        )
        with tarfile.open(result["archive_path"]) as archive:
            self.assertEqual(archive.extractfile("model_catalog-0").read(), b"orig")
        self.assertFalse(self.catalog.exists())

    def test_curator_guard_rejects_root_or_collection_mismatch(self):
        with self.assertRaisesRegex(ValueError, "trusted runtime root"):
            FILES.archive_native_files(
                self.output,
                curator_workdir=self.home,
                curator_root=Path("/tmp"),
                curator_collection_id="a" * 64,
                agent_uid=os.geteuid(),
            )
        self.assertTrue(self.home.exists())


class MissingWorkspaceTaskTest(unittest.IsolatedAsyncioTestCase):
    async def test_enospc_mkdir_retains_primary_error_and_writes_raw_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, work = root / "attempt", root / "work"
            output.mkdir()
            work.mkdir()
            settings = root / "environment.json"
            settings.write_text(
                json.dumps(
                    {
                        "url": "http://127.0.0.1",
                        "catalog_digest": "catalog",
                        "allowed_splits": ["train"],
                        "admin_token": "CPU-fixture",
                    }
                )
            )
            engine = {"won": False, "steps": 0, "invalid_actions": 0, "max_steps": 50, "step_limit": False}

            class Response:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    return False

                def raise_for_status(self):
                    pass

                async def json(self):
                    return self.payload

            class Client(Response):
                def post(self, url, **_):
                    response = Response()
                    response.payload = (
                        {"session_id": "session", "catalog_digest": "catalog", "state": {}}
                        if url.endswith("/create")
                        else engine
                    )
                    return response

            class WorkRoot:
                def mkdir(self, **_):
                    pass

                def is_symlink(self):
                    return False

                def stat(self):
                    return work.stat()

                def __fspath__(self):
                    return str(work)

                def __truediv__(self, name):
                    return work / name

            real_mkdir = Path.mkdir

            def full_disk(path, *args, **kwargs):
                if path.parent == work:
                    raise OSError(errno.ENOSPC, "primary episode disk full")
                return real_mkdir(path, *args, **kwargs)

            node = next(
                x
                for x in ast.parse((DIRECTORY / "task.py").read_text()).body
                if isinstance(x, ast.ClassDef) and x.name == "AlfworldSkillTrainingTask"
            )
            package = "_task_enospc_cpu"
            scope = {
                "__package__": package,
                "Task": object,
                "TaskConfig": object,
                "TASK_NAME": "fixture",
                "register_task": lambda _: lambda cls: cls,
                "json": json,
                "os": os,
                "time": time,
                "uuid": uuid,
                "Path": lambda value: WorkRoot() if str(value) == "/tmp/alfworld-skill-episodes" else Path(value),
                "pwd": types.SimpleNamespace(
                    getpwnam=lambda _: types.SimpleNamespace(pw_uid=os.geteuid(), pw_gid=os.getegid())
                ),
                "aiohttp": types.SimpleNamespace(ClientSession=lambda **_: Client(), ClientTimeout=lambda **_: None),
                "validate_loopback_url": lambda value: value,
                "new_attempt_directory": lambda *_: ("attempt", output),
                "validate_game_reference": lambda *_: None,
                "authoritative_reward": lambda _: 0.0,
                "TaskResult": lambda **kwargs: types.SimpleNamespace(**kwargs),
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_enospc_task", "exec"), scope)
            task = scope["AlfworldSkillTrainingTask"]()
            task.config = types.SimpleNamespace(
                metadata={
                    "run_dir": str(root),
                    "game_id": "game",
                    "stable_game_id": "game",
                    "split": "train",
                    "dataset_phase": "dev",
                    "task_type": "type",
                    "scene_id": "scene",
                    "relative_game_path": "game.tw-pddl",
                    "game_sha256": "sha",
                    "catalog_digest": "catalog",
                }
            )
            with (
                patch.dict(os.environ, {"ALFWORLD_SERVER_CONFIG": str(settings)}),
                patch.object(Path, "mkdir", full_disk),
            ):
                with patch.dict(sys.modules, {package + ".native_files": FILES}):
                    result = await task.run()
            saved = json.loads((output / "result.json").read_text())
            self.assertIn("primary episode disk full", saved["agent_error"])
            self.assertIn("Native workspace missing during cleanup", saved["agent_error"])
            self.assertNotIn("FileNotFoundError", saved["agent_error"])
            self.assertFalse(result.extra_info["eval_completed"])
            self.assertEqual(saved["reward"], 0.0)  # The explicit CPU fixture engine outcome, not a fallback.


if __name__ == "__main__":
    unittest.main()
