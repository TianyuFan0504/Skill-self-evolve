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
from unittest.mock import Mock, patch

DIRECTORY = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld"
SPEC = importlib.util.spec_from_file_location("_native_files_cpu", DIRECTORY / "native_files.py")
FILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILES)
ACCOUNT_SPEC = importlib.util.spec_from_file_location("_native_files_accounts_cpu", DIRECTORY / "accounts.py")
ACCOUNTS = importlib.util.module_from_spec(ACCOUNT_SPEC)
ACCOUNT_SPEC.loader.exec_module(ACCOUNTS)
BUDGET_SPEC = importlib.util.spec_from_file_location("_native_files_budget_cpu", DIRECTORY / "budget.py")
BUDGET = importlib.util.module_from_spec(BUDGET_SPEC)
BUDGET_SPEC.loader.exec_module(BUDGET)
GUARD_SPEC = importlib.util.spec_from_file_location("_native_files_guard_cpu", DIRECTORY / "terminal_guard.py")
GUARD = importlib.util.module_from_spec(GUARD_SPEC)
sys.modules[GUARD_SPEC.name] = GUARD
GUARD_SPEC.loader.exec_module(GUARD)


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
        incident = Mock()
        scope = {
            "LocalSandbox": LocalWriter,
            "Path": Path,
            "re": re,
            "os": os,
            "AccountIsolationError": ACCOUNTS.AccountIsolationError,
            "record_isolation_incident": incident,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_catalog_write_hook", "exec"), scope)
        sandbox = object.__new__(scope["AlfworldUserSandbox"])
        sandbox.native_model_catalogs = []
        sandbox.account_lease = {"lease_id": "cpu", "owner": {"run_root": str(self.output)}}
        sandbox.attempt_id = "cpu"
        self.catalog.unlink()
        with self.assertRaisesRegex(ACCOUNTS.AccountIsolationError, "catalog preparation failed") as raised:
            asyncio.run(sandbox.write_file(str(self.catalog), b"original catalog bytes"))
        self.assertIn("primary catalog disk full", str(raised.exception.__cause__))
        incident.assert_called_once()
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
        await self._task_case()

    async def test_unconfirmed_uid_drain_preserves_sources_and_never_archives(self):
        await self._task_case(drain_failure=True)

    async def test_early_admin_create_python_error_is_sticky_before_any_lease(self):
        await self._task_case(host_failure="early_create")

    async def test_post_start_sdk_python_error_is_fatal_after_raw_result(self):
        await self._task_case(host_failure="sdk")

    async def test_proven_mdp_stop_closes_task_without_rewriting_raw_native_finish(self):
        await self._task_case(completed_native_stop=True)

    async def _task_case(self, *, drain_failure=False, host_failure=None, completed_native_stop=False):
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
            terminal_state = {"won": True, "done": True, "steps": 3, "max_steps": 50}
            if completed_native_stop:
                engine.update(terminal_state)

            class Response:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    return False

                def raise_for_status(self):
                    if host_failure == "early_create" and getattr(self, "url", "").endswith("/create"):
                        raise OSError("injected admin create transport failure")

                async def json(self):
                    return self.payload

            class Client(Response):
                def post(self, url, **_):
                    response = Response()
                    response.url = url
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
                if path.parent == work and not drain_failure and host_failure is None and not completed_native_stop:
                    raise OSError(errno.ENOSPC, "primary episode disk full")
                return real_mkdir(path, *args, **kwargs)

            node = next(
                x
                for x in ast.parse((DIRECTORY / "task.py").read_text()).body
                if isinstance(x, ast.ClassDef) and x.name == "AlfworldSkillTrainingTask"
            )
            package = "_task_enospc_cpu"
            lease = {
                "lease_id": "cpu",
                "uid": os.getuid(),
                "gid": os.getgid(),
                "name": "cpu",
                "owner": {"run_root": str(root)},
            }
            pool = types.SimpleNamespace(
                acquire=Mock(return_value=lease),
                begin_attempt=Mock(),
                account=Mock(return_value=types.SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())),
                public=lambda ref: ref,
                drain=Mock(
                    side_effect=OSError("injected residual UID process") if drain_failure else None,
                    return_value={"uid": os.getuid(), "remaining_pids": [], "drained": True},
                ),
                register_resource=Mock(),
                quarantine=Mock(),
                close_attempt=Mock(
                    return_value={"status": "attempt_closed", "uid_drained": True, "backend_drained": True}
                ),
                close_empty_attempt=Mock(),
            )
            incidents = Mock()
            scope = {
                "__package__": package,
                "Task": object,
                "AccountPool": lambda: pool,
                "AccountIsolationError": ACCOUNTS.AccountIsolationError,
                "POLICY_VERSION": ACCOUNTS.POLICY_VERSION,
                "record_isolation_incident": incidents,
                "asyncio": asyncio,
                "PairedWorkspace": types.SimpleNamespace(_json=lambda path, value: path.write_text(json.dumps(value))),
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
                "authoritative_reward": lambda state: float(state["won"]),
                "TaskResult": lambda **kwargs: types.SimpleNamespace(**kwargs),
                "agent_messages": lambda *args, **kwargs: [],
                "_validated_state": GUARD._validated_state,
                "AlfworldUserSandbox": lambda *args, **kwargs: sandbox,
            }
            exec(compile(ast.Module(body=[node], type_ignores=[]), "actual_enospc_task", "exec"), scope)
            task = scope["AlfworldSkillTrainingTask"]()
            native_home = Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
            sandbox = types.SimpleNamespace(
                native_codex_homes=[], native_model_catalogs=[], native_started=False, terminal_stop_evidence=[]
            )

            async def run_agent(**kwargs):
                native_home.mkdir(mode=0o700)
                (native_home / "inflight.txt").write_text("must survive unconfirmed drain")
                (Path(kwargs["workdir"]) / "inflight.txt").write_text("must remain")
                sandbox.native_codex_homes.append(str(native_home))
                if host_failure == "sdk":
                    raise OSError("injected SDK stream transport failure")
                if completed_native_stop:
                    return types.SimpleNamespace(
                        finished=False,
                        info={"ok": False, "exit_code": -1, "error": "agent process stopped after terminal"},
                    )
                return types.SimpleNamespace(info={"ok": True, "exit_code": 0})

            agent = types.SimpleNamespace(
                config=types.SimpleNamespace(
                    model=types.SimpleNamespace(base_url="http://127.0.0.1/sessions/gateway/v1")
                ),
                run=run_agent,
            )
            task.build_agent = lambda: agent
            binding = types.SimpleNamespace(binding_sha256="sha", binding_id="binding")
            backend = types.ModuleType(package + ".backend_terminal")
            backend.bind_episode = lambda *args, **kwargs: binding
            backend.fence_host_stop = Mock()

            async def wait_for_drain(*args, **kwargs):
                receipt = {
                    "schema_version": 1,
                    "binding_id": "binding",
                    "binding_sha256": "sha",
                    "attempt_id": "attempt",
                    "gateway_session_id": "gateway",
                    "environment_session_id": "session",
                    "active_requests": 0,
                    "backend_stop_confirmed": True,
                    "status": "drained",
                }
                if completed_native_stop:
                    receipt.update(
                        status="terminal_aborted",
                        fenced=True,
                        stop_kind="mdp_terminal",
                        expected_backend_terminal_stop=True,
                        terminal_state=terminal_state,
                        terminal_state_hash=GUARD._validated_state(terminal_state)[1],
                    )
                return receipt

            backend.wait_for_terminal_drain = wait_for_drain
            task.config = types.SimpleNamespace(
                prompt=[],
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
                },
            )
            original_archive = FILES.archive_native_files

            def fixture_archive(*args, **kwargs):
                return original_archive(*args, workspace_root=work, **kwargs)

            try:
                with (
                    patch.dict(os.environ, {"ALFWORLD_SERVER_CONFIG": str(settings), "SKILL_RUN_DIR": str(root)}),
                    patch.object(Path, "mkdir", full_disk),
                    patch.object(FILES, "archive_native_files", side_effect=fixture_archive) as archived,
                ):
                    with patch.dict(
                        sys.modules,
                        {
                            package + ".native_files": FILES,
                            package + ".backend_terminal": backend,
                            package + ".budget": BUDGET,
                        },
                    ):
                        if host_failure == "early_create":
                            with self.assertRaisesRegex(OSError, "admin create transport failure"):
                                await task.run()
                            self.assertFalse((output / "result.json").exists())
                            pool.begin_attempt.assert_not_called()
                            pool.quarantine.assert_not_called()
                            archived.assert_not_called()
                            incidents.assert_called_once()
                            self.assertEqual(Path(incidents.call_args.args[0]), root.resolve())
                            return
                        if completed_native_stop:
                            result = await task.run()
                            saved = json.loads((output / "result.json").read_text())
                            self.assertIs(result.finished, True)
                            self.assertIs(result.extra_info["native_finished"], False)
                            self.assertIs(saved["native_finished"], False)
                            self.assertEqual(result.reward, 1.0)
                            self.assertIs(result.extra_info["expected_backend_terminal_stop"], True)
                            incidents.assert_not_called()
                            self.assertFalse(native_home.exists())
                            archived.assert_called_once()
                            return
                        message = (
                            "Host attempt execution failed"
                            if host_failure == "sdk"
                            else "UID process drain failed"
                            if drain_failure
                            else "boundary setup failed"
                        )
                        with self.assertRaisesRegex(ACCOUNTS.AccountIsolationError, message):
                            await task.run()
                    if drain_failure:
                        archived.assert_not_called()
                saved = json.loads((output / "result.json").read_text())
                if drain_failure:
                    self.assertTrue((native_home / "inflight.txt").is_file())
                    self.assertTrue((Path(saved["workdir"]) / "inflight.txt").is_file())
                elif host_failure == "sdk":
                    self.assertIn("injected SDK stream transport failure", saved["agent_error"])
                    self.assertIn("Host attempt execution failed", saved["account_isolation_error"])
                    archived.assert_called_once()
                    self.assertFalse(native_home.exists())
                    self.assertTrue((output / "native-files.json").is_file())
                    self.assertGreaterEqual(incidents.call_count, 2)
                    self.assertEqual(Path(incidents.call_args.args[0]), root.resolve())
                else:
                    self.assertIn("primary episode disk full", saved["agent_error"])
                    self.assertIn("Native workspace missing during cleanup", saved["agent_error"])
                    self.assertNotIn("FileNotFoundError", saved["agent_error"])
                    self.assertGreaterEqual(incidents.call_count, 2)
                pool.quarantine.assert_called_once()
                pool.close_attempt.assert_not_called()
                pool.close_empty_attempt.assert_not_called()
                self.assertEqual(saved["reward"], 0.0)  # Fixture engine outcome, not fallback training data.
            finally:
                shutil.rmtree(native_home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
