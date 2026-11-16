"""Planner lifetime and failed initialization cleanup without model resources."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.skill_self_evolve.scripts import parallel_environment_server as server  # noqa: E402


class Environment:
    def close(self):
        pass


class PddlEnv(Environment):
    def __init__(self):
        self.downward_lib = types.SimpleNamespace(cleanup=self.release)
        self._pddl_state = types.SimpleNamespace(downward_lib=self.downward_lib)

    def load(self, path):
        self.game_path = path
        self._pddl_state = types.SimpleNamespace(downward_lib=self.downward_lib)

    def seed(self, seed):
        pass

    def reset(self):
        raise RuntimeError("reset failed")


class Wrapper:
    def __init__(self, root=None, error=None, **kwargs):
        self.unwrapped = getattr(root, "unwrapped", root)
        self._wrapped = root
        self.error = error
        self.close = Mock(side_effect=self._close)

    def _close(self):
        if self.error:
            raise self.error
        self._wrapped.close()

    def load(self, path):
        self._wrapped.load(path)

    def seed(self, seed):
        self.unwrapped.seed(seed)

    def reset(self):
        return self.unwrapped.reset()


class EnvironmentCloseTests(unittest.TestCase):
    def setUp(self):
        self.release = Mock()
        PddlEnv.release = self.release
        server.core._IDLE_TEXTWORLD_ENVIRONMENTS.clear()
        server.core._CLOSED_TEXTWORLD_ENVIRONMENTS.clear()
        self.addCleanup(server.core._IDLE_TEXTWORLD_ENVIRONMENTS.clear)
        self.addCleanup(server.core._CLOSED_TEXTWORLD_ENVIRONMENTS.clear)
        textworld = types.ModuleType("textworld")
        textworld.Environment = Environment
        textworld.EnvInfos = Mock()
        textworld.start = Mock()
        pddl = types.ModuleType("textworld.envs.pddl.pddl")
        pddl.PddlEnv = PddlEnv
        downward = types.ModuleType("fast_downward")
        downward.close_lib = Mock(side_effect=AssertionError("Planner libraries must remain mapped"))
        self.addCleanup(downward.close_lib.assert_not_called)
        alfred = types.ModuleType("alfworld.agents.environment.alfred_tw_env")
        alfred.AlfredDemangler = Wrapper
        alfred.AlfredInfos = Wrapper
        self.textworld = textworld
        modules = {
            "textworld": textworld,
            "textworld.envs.pddl.pddl": pddl,
            "fast_downward": downward,
            "alfworld.agents.environment.alfred_tw_env": alfred,
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_wrapped_noop_close_releases_planner_exactly_once(self):
        root = PddlEnv()
        library = root.downward_lib
        env = Wrapper(root)
        server.core.close_textworld_environment(env)
        server.core.close_textworld_environment(env)
        self.release.assert_called_once_with()
        self.assertIs(root.downward_lib, library)
        self.assertIsNone(root._pddl_state)
        self.assertEqual(env.close.call_count, 1)
        self.assertEqual(server.core._IDLE_TEXTWORLD_ENVIRONMENTS, [root])

    def test_planner_release_runs_even_if_wrapper_close_fails(self):
        root = PddlEnv()
        library = root.downward_lib
        primary = RuntimeError("wrapper close failed")
        with self.assertRaises(RuntimeError) as caught:
            server.core.close_textworld_environment(Wrapper(root, primary))
        self.assertIs(caught.exception, primary)
        self.release.assert_called_once_with()
        self.assertIs(root.downward_lib, library)
        self.assertIsNone(root._pddl_state)

    def test_secondary_release_failure_preserves_close_error_and_does_not_retry(self):
        root = PddlEnv()
        primary = RuntimeError("wrapper close failed")
        env = Wrapper(root, primary)
        self.release.side_effect = ValueError("release failed")
        with self.assertRaises(RuntimeError) as caught:
            server.core.close_textworld_environment(env)
        self.assertIs(caught.exception, primary)
        if hasattr(primary, "__notes__"):
            self.assertIn("release failed", primary.__notes__[0])
        server.core.close_textworld_environment(env)
        self.release.assert_called_once()
        self.assertEqual(server.core._IDLE_TEXTWORLD_ENVIRONMENTS, [])

    def test_release_error_surfaces_without_a_primary_error(self):
        root = PddlEnv()
        primary = RuntimeError("release failed")
        self.release.side_effect = primary
        with self.assertRaises(RuntimeError) as caught:
            server.core.close_textworld_environment(root)
        self.assertIs(caught.exception, primary)
        self.assertEqual(server.core._IDLE_TEXTWORLD_ENVIRONMENTS, [])

    def test_upstream_close_override_owns_planner_lifetime(self):
        class UpstreamPddlEnv(PddlEnv):
            def close(self):
                self.closed = True

        env = UpstreamPddlEnv()
        server.core.close_textworld_environment(Wrapper(env))
        self.assertTrue(env.closed)
        self.release.assert_not_called()

    def test_unrelated_environment_with_similarly_named_library_is_untouched(self):
        env = Environment()
        env.downward_lib = object()
        server.core.close_textworld_environment(env)
        self.release.assert_not_called()

    def test_parallel_engine_close_releases_its_session(self):
        root = PddlEnv()
        with patch.dict(server._ENGINES, {"session": Wrapper(root)}, clear=True):
            server._engine_operation("close", {"session_id": "session"})
            self.assertNotIn("session", server._ENGINES)
        self.release.assert_called_once_with()

    def test_pool_reuses_base_with_fresh_wrappers_and_closed_session_guard(self):
        root = PddlEnv()
        self.textworld.start.return_value = Wrapper(root)
        first = server.core.start_textworld_environment("first")
        server.core.close_textworld_environment(first)
        second = server.core.start_textworld_environment("second")
        self.assertIs(second.unwrapped, root)
        self.assertIsNot(first._environment, second._environment)
        self.assertEqual(root.game_path, "second")
        for operation, args in [("step", ("look",)), ("reset", ()), ("load", ("stale",)), ("seed", (42,))]:
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, "session is closed"):
                getattr(first, operation)(*args)
        server.core.close_textworld_environment(first)
        self.assertEqual(root.game_path, "second")
        self.assertEqual(self.release.call_count, 1)
        server.core.close_textworld_environment(second)
        self.assertEqual(self.release.call_count, 2)
        self.assertEqual(server.core._IDLE_TEXTWORLD_ENVIRONMENTS, [root])

    def test_active_sessions_never_share_a_base(self):
        first_root, second_root = PddlEnv(), PddlEnv()
        self.textworld.start.side_effect = [Wrapper(first_root), Wrapper(second_root)]
        first = server.core.start_textworld_environment("first")
        second = server.core.start_textworld_environment("second")
        self.assertIsNot(first.unwrapped, second.unwrapped)
        server.core.close_textworld_environment(first)
        server.core.close_textworld_environment(second)
        self.assertEqual(len(server.core._IDLE_TEXTWORLD_ENVIRONMENTS), 2)

    def test_reused_load_failure_cleans_and_returns_base(self):
        root = PddlEnv()
        self.textworld.start.return_value = Wrapper(root)
        first = server.core.start_textworld_environment("first")
        server.core.close_textworld_environment(first)
        root.load = Mock(side_effect=RuntimeError("load failed"))
        with self.assertRaisesRegex(RuntimeError, "^load failed$"):
            server.core.start_textworld_environment("second")
        self.assertEqual(self.release.call_count, 2)
        self.assertIsNone(root._pddl_state)
        self.assertEqual(server.core._IDLE_TEXTWORLD_ENVIRONMENTS, [root])

    def test_parallel_reset_failure_preserves_primary_and_releases_planner(self):
        root = PddlEnv()
        env = Wrapper(root, RuntimeError("wrapper close failed"))
        self.textworld.start.return_value = env
        with self.assertRaisesRegex(RuntimeError, "^reset failed$"):
            server._engine_operation("create", {"session_id": "session", "game_path": "fixture"})
        self.release.assert_called_once()
        self.assertIsNone(root._pddl_state)

    def test_serial_reset_failure_preserves_primary_and_releases_planner(self):
        from data import GameRecord, stable_game_id

        root = PddlEnv()
        self.textworld.start.return_value = Wrapper(root, RuntimeError("wrapper close failed"))
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            relative = "json_2.1.1/train/task/trial/game.tw-pddl"
            game = base / "benchmark/data" / relative
            game.parent.mkdir(parents=True)
            game.write_text("fixture")
            digest = hashlib.sha256(game.read_bytes()).hexdigest()
            row = GameRecord(stable_game_id(relative, digest), "train", "task", "scene", relative, digest)
            service = server.core.GameService([row], base / "benchmark", base / "run", allowed_splits=["train"])
            payload = {
                **row.to_dict(),
                "catalog_digest": service.catalog_digest,
                "output_dir": str(base / "run/episode"),
            }

            class Request:
                headers = {"X-Admin-Token": service.admin_token}

                async def json(self):
                    return payload

            with self.assertRaisesRegex(RuntimeError, "^reset failed$"):
                asyncio.run(service.create(Request()))
        self.release.assert_called_once()
        self.assertIsNone(root._pddl_state)


if __name__ == "__main__":
    unittest.main()
