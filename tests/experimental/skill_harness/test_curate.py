"""CPU-only checks of native curator retention, isolation, and safe retry."""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_curate_offline_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "verl/experimental/skill_harness")]
sys.modules[PACKAGE] = package
curate = importlib.import_module(PACKAGE + ".curate")
store_module = importlib.import_module(PACKAGE + ".store")
replay_module = importlib.import_module(PACKAGE + ".replay")


ACTUAL_CURATE_ID = "b37deeb7aee9b8102c0342f37982fd7fd8eacef6ed8e978ac4f4feca60d7350a"
ACTUAL_CURATE_HOME = "/tmp/alfworld-curator/" + ACTUAL_CURATE_ID + "/.home"


def actual_startup_events():
    """Exact three warning items from the first real EMA curator capture."""
    ignored = (
        "Codex is ignoring 1 unrecognized configuration setting. Check for typos or deprecated settings.\n"
        "  session-flags: `disable_response_storage` is ignored."
    )
    unstable = (
        "Under-development features enabled: skip_host_skill_discovery. "
        "Under-development features are incomplete and may behave unpredictably. "
        "To suppress this warning, set `suppress_unstable_features_warning = true` in "
        + ACTUAL_CURATE_HOME
        + "/config.toml."
    )
    return [
        {"type": "thread.started"},
        *[
            {"type": "item.completed", "item": {"id": "item_" + str(index), "type": "error", "message": message}}
            for index, message in enumerate((ignored, ignored, unstable))
        ],
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
        {"type": "turn.completed"},
    ]


def classify(events):
    return curate.classify_native_events(
        "\n".join(json.dumps(event) for event in events), expected_codex_home=ACTUAL_CURATE_HOME
    )


class FakeGateway:
    def __init__(self):
        self.created, self.finalized, self.aborted = [], [], []
        self.trajectories = [
            SimpleNamespace(
                prompt_ids=[101, 102, 103],
                response_ids=[501, 502, 503],
                response_mask=[1, 0, 1],
                response_logprobs=[-0.3, 0.0, -0.7],
                chain_id=17,
                extra_fields={"min_global_steps": 3, "max_global_steps": 3},
            )
        ]

    async def create_session(self, identity, **kwargs):
        self.created.append((identity, copy.deepcopy(kwargs)))
        return SimpleNamespace(session_id=identity, base_url="http://127.0.0.1:12345/sessions/" + identity + "/v1")

    async def finalize_session(self, identity):
        self.finalized.append(identity)
        return copy.deepcopy(self.trajectories)

    async def abort_session(self, identity):
        self.aborted.append(identity)


class FakeSandbox:
    instances = []

    def __init__(self, runtime, output_dir, collection_id):
        self.workdir = Path(runtime.work_root) / collection_id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.tool_events_seen, self.closed = False, False
        self.instances.append(self)

    async def close(self):
        self.closed = True


class CurateTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o755)
        self.store = store_module.SkillStore(self.root / "store")
        self.replay = replay_module.CurateReplay(self.root / "replay")
        self.gateway = FakeGateway()
        self.request = curate.CurateRequest("solver:group0", self.store.head(), "curate:group0", 0, 2, 10.0)
        runtime_root = self.root / "runtime"
        runtime_root.mkdir()
        paths = {}
        for key, name in (
            ("codex_bin", "codex"),
            ("model_catalog_path", "catalog.json"),
            ("instructions_path", "system.md"),
            ("schema_path", "schema.json"),
        ):
            path = runtime_root / name
            path.write_text(name)
            paths[key] = str(path)
        paths["task_prompt_path"] = str(ROOT / "examples/skill_self_evolve/prompts/curator_task.txt")
        self.runtime = curate.NativeCuratorRuntime(
            model_name="Qwen3.5-4B-nothink",
            unix_user="alfworld_curator",
            uid=1201,
            gid=1201,
            work_root=str(self.root / "work"),
            package_root=str(runtime_root),
            native_version="0.167.0",
            template_version="template:curator1",
            **paths,
        )
        self.behavior = curate.CuratorBehavior("ema:3", 3, "tokenizer:v1", self.runtime.template_version)
        self.source = {
            "split": "train",
            "dataset_phase": "train",
            "finished": True,
            "task_type": "pick_clean_then_place_in_recep",
            "game_id": "train_example_a",
            "stable_game_id": "train_example_a",
            "game_sha256": "a" * 64,
            "catalog_digest": "b" * 64,
            "attempt_id": "attempt_a",
            "scene_id": "FloorPlan1",
            "relative_game_path": "json_2.1.1/train/x/y/game.tw-pddl",
            "won": True,
            "reward": 1.0,
            "accuracy": 1.0,
            "steps": 1,
            "max_steps": 50,
            "initial_state": {"observation": "A mug is on the counter.", "expert_plan": "PRIVATE_EXPERT"},
            "admin_token": "PRIVATE_ADMIN",
            "expert_plan": "PRIVATE_EXPERT",
            "transitions": [
                {
                    "step": 1,
                    "action": "clean mug 1 with sinkbasin 1",
                    "action_in_exposed_commands": True,
                    "after": {"observation": "You clean the mug.", "won": True, "done": True},
                }
            ],
        }
        self.agent_calls, self.content, self.agent_error = [], None, None
        FakeSandbox.instances = []

    def agent_factory(self, runtime, handle):
        fixture = self

        class Agent:
            async def run(self, **kwargs):
                fixture.agent_calls.append(copy.deepcopy(kwargs["messages"]))
                if fixture.agent_error:
                    raise fixture.agent_error
                content = (
                    fixture.content
                    if fixture.content is not None
                    else json.dumps(
                        {
                            "skill_id": fixture.request.skill_id,
                            "title": "Take before cleaning",
                            "principle": "Take the target before cleaning it and check the feedback.",
                            "when_to_apply": "A goal requires a clean portable target.",
                            "task_families": [fixture.source["task_type"]],
                        }
                    )
                )
                return SimpleNamespace(output={"content": content}, info={"ok": True}, finished=True)

        return Agent()

    def collect(self, **kwargs):
        arguments = dict(
            gateway_manager=self.gateway,
            behavior=self.behavior,
            source_episode=self.source,
            store=self.store,
            replay=self.replay,
            request=self.request,
            runtime=self.runtime,
            output_root=self.root / "collections",
            _agent_factory=self.agent_factory,
            _sandbox_factory=FakeSandbox,
        )
        arguments.update(kwargs)
        return asyncio.run(curate.collect_curate_candidate(**arguments))

    def test_exact_native_tito_old_mu_retained_with_train_identity(self):
        self.gateway.trajectories[0].extra_fields["native_position_ids"] = [[0, 1, 2, 3, 4, 5]] * 4
        original = copy.deepcopy(self.source)
        result = self.collect()
        record = self.replay.get(result.record_id)
        self.assertEqual(record.prompt_ids, (101, 102, 103))
        self.assertEqual(record.action_ids, (501, 502, 503))
        self.assertEqual(record.curate_mask, (1, 0, 1))
        self.assertEqual(record.behavior_logprobs, (-0.3, 0.0, -0.7))
        self.assertEqual(record.behavior_policy_version, "ema:3")
        self.assertEqual(record.context_snapshot["task_metadata"]["official_split"], "train")
        self.assertEqual(record.context_snapshot["task_metadata"]["dataset_phase"], "train")
        self.assertEqual(record.context_snapshot["task_metadata"]["game_id"], self.source["game_id"])
        self.assertEqual(len(record.context_snapshot["native_position_ids"]), 4)
        self.assertEqual(self.store.head(), self.request.skill_snapshot_id)
        self.assertEqual(self.store.retrieve(self.source["task_type"]).skills, ())
        self.assertTrue(result.credit_registration_required)
        self.assertEqual(self.source, original)
        self.assertNotIn("PRIVATE", self.agent_calls[0][0]["content"])
        self.assertEqual(self.gateway.created[0][1]["sampling_params"], curate.SAMPLING)
        self.assertTrue(FakeSandbox.instances[0].closed)

    def test_completed_retry_is_idempotent_without_another_native_call(self):
        first, again = self.collect(), self.collect()
        self.assertEqual(first, again)
        self.assertEqual(len(self.agent_calls), 1)
        self.assertEqual(len(self.gateway.created), 1)
        self.assertEqual(len(list((self.store.root / "candidates").glob("*.json"))), 1)

    def test_captured_generation_recovers_interrupted_replay_commit(self):
        with patch.object(self.replay, "append", side_effect=OSError("interrupted retention")):
            with self.assertRaises(OSError):
                self.collect()
        result = self.collect()
        self.assertEqual(len(self.agent_calls), 1)
        self.assertEqual(len(self.gateway.created), 1)
        self.assertEqual(self.replay.get(result.record_id).skill_ref["skill_id"], result.artifact.skill_id)
        self.assertEqual(len(list((self.store.root / "candidates").glob("*.json"))), 1)

    def test_replay_ack_interruption_recovers_without_new_generation(self):
        write = curate._write_immutable

        def interrupted(path, payload):
            if path.name == "complete.json":
                raise OSError("ack interrupted")
            return write(path, payload)

        with patch.object(curate, "_write_immutable", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.collect()
        self.assertIsNotNone(self.replay.get(self.request.record_id))
        self.assertEqual(self.collect().record_id, self.request.record_id)
        self.assertEqual(len(self.agent_calls), 1)

    def test_dev_and_test_episodes_never_reach_ema_gateway(self):
        for split, phase in (("train", "dev"), ("valid_seen", "test_seen"), ("valid_unseen", "test_unseen")):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(curate.CurateProtocolError, "training episodes only"):
                    self.collect(source_episode={**self.source, "split": split, "dataset_phase": phase})
        self.assertEqual(self.gateway.created, [])

    def test_unversioned_or_mixed_weights_are_captured_but_never_labeled_by_desired_version(self):
        for index, versions in enumerate(({}, {"min_global_steps": 2, "max_global_steps": 3})):
            with self.subTest(versions=versions):
                self.request = replace(self.request, candidate_group_id="versions:" + str(index))
                self.gateway.trajectories[0].extra_fields = versions
                with self.assertRaisesRegex(curate.CurateProtocolError, "weight versions"):
                    self.collect()
                folder = self.root / "collections" / self.request.collection_id
                self.assertTrue((folder / "generated.json").exists())
                self.assertFalse((folder / "complete.json").exists())

    def test_zero_or_multiple_tito_segments_fail_closed(self):
        original = copy.deepcopy(self.gateway.trajectories[0])
        for index, trajectories in enumerate(([], [original, original])):
            with self.subTest(count=len(trajectories)):
                self.request = replace(self.request, candidate_group_id="segments:" + str(index))
                self.gateway.trajectories = trajectories
                with self.assertRaisesRegex(curate.CurateProtocolError, "exactly one complete"):
                    self.collect()
        self.assertFalse((self.replay.root / "records").exists())

    def test_missing_behavior_probabilities_are_never_recomputed(self):
        self.gateway.trajectories[0].response_logprobs = None
        with self.assertRaisesRegex(curate.CurateProtocolError, "recomputation is forbidden"):
            self.collect()
        self.assertFalse((self.replay.root / "records").exists())

    def test_tool_execution_is_not_accepted_as_pure_curation(self):
        def sandbox(*args):
            instance = FakeSandbox(*args)
            instance.tool_events_seen = True
            return instance

        with self.assertRaisesRegex(curate.CurateProtocolError, "without any native tool"):
            self.collect(_sandbox_factory=sandbox)
        self.assertFalse((self.replay.root / "records").exists())

    def test_uncertain_launch_aborts_owned_session_and_does_not_resample(self):
        self.agent_error = TimeoutError("no durable capture")
        with self.assertRaises(TimeoutError):
            self.collect()
        self.assertEqual(self.gateway.aborted, [self.request.session_id])
        self.assertTrue(FakeSandbox.instances[0].closed)
        self.agent_error = None
        with self.assertRaises(curate.CurateRecoveryRequired):
            self.collect()
        self.assertEqual(len(self.gateway.created), 1)

    def test_cleanup_failure_preserves_primary_native_error_and_captured_tokens(self):
        class BrokenClose(FakeSandbox):
            async def close(self):
                self.closed = True
                raise OSError("CPU injected archive failure")

        self.agent_error = TimeoutError("primary native timeout")
        with self.assertRaisesRegex(TimeoutError, "primary native timeout"):
            self.collect(_sandbox_factory=BrokenClose)
        directory = self.root / "collections" / self.request.collection_id
        self.assertTrue(curate._read_immutable(directory / "cleanup-failure.json")["primary_error_retained"])
        self.request = replace(self.request, candidate_group_id="close-failure-with-capture")
        self.agent_error = None
        with self.assertRaisesRegex(OSError, "archive failure"):
            self.collect(_sandbox_factory=BrokenClose)
        directory = self.root / "collections" / self.request.collection_id
        self.assertTrue((directory / "generated.json").exists())
        self.assertFalse((directory / "complete.json").exists())
        self.assertFalse((self.replay.root / "records").exists())

    def test_bad_format_and_fabricated_provenance_are_not_staged(self):
        for index, content in enumerate(("```json\n{}\n```", json.dumps({"provenance": "invented"}))):
            with self.subTest(content=content):
                self.request = replace(self.request, candidate_group_id="format:" + str(index))
                self.content = content
                with self.assertRaises(curate.CurateProtocolError):
                    self.collect()
        self.assertEqual(list((self.store.root / "candidates").glob("*.json")), [])

    def test_redaction_removes_credentials_paths_urls_and_expert_data(self):
        source = copy.deepcopy(self.source)
        source["initial_state"]["observation"] += (
            " http://127.0.0.1:1234/session/admin /root/private/file api_key=PRIVATE"
        )
        redacted = json.dumps(curate.redact_completed_episode(source))
        for value in ("PRIVATE", "expert_plan", "127.0.0.1", "/root/private", "train_example_a", "FloorPlan1"):
            self.assertNotIn(value, redacted)
        self.assertIn("REDACTED_URL", redacted)
        self.assertIn("REDACTED_PATH", redacted)

    def test_sampling_overrides_and_raw_logprobs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sampling overrides"):
            replace(self.behavior, request_sampling_param_keys=("temperature",))
        with self.assertRaisesRegex(ValueError, "processed behavior logprobs"):
            replace(self.behavior, logprobs_mode="raw_logprobs")

    def test_runtime_uses_observed_feature_flags_and_no_unsupported_structured_format(self):
        native, package, public, work = [self.root / name for name in ("native", "package", "public/a/b", "work-root")]
        native.write_text("native binary placeholder")
        native.chmod(0o755)
        package.mkdir()
        catalog = self.root / "native-catalog.json"
        catalog.write_text(json.dumps({"models": [{"slug": "Qwen3.5-4B-nothink", "base_instructions": "old"}]}))

        def account(name):
            return SimpleNamespace(pw_uid=1201 if name == "alfworld_curator" else 1202, pw_gid=1201)

        with (
            patch.object(curate.pwd, "getpwnam", side_effect=account),
            patch.object(curate.subprocess, "check_output", return_value="codex-cli 0.167.0\n"),
        ):
            previous_umask = os.umask(0o077)
            try:
                runtime = curate.provision_native_curator_runtime(
                    public_root=public,
                    work_root=work,
                    native_bin=native,
                    package_root=package,
                    model_catalog_path=catalog,
                    model_name="Qwen3.5-4B-nothink",
                )
            finally:
                os.umask(previous_umask)
        for parent in [public, public.parent, public.parent.parent]:
            self.assertTrue(parent.stat().st_mode & 0o001)
        wrapper = Path(runtime.codex_bin).read_text()
        for feature in ("shell_tool", "unified_exec", "multi_agent", "goals"):
            self.assertIn("--disable " + feature, wrapper)
        for override in curate.NATIVE_CONFIG_OVERRIDES:
            self.assertIn(override, wrapper)
        self.assertIn("--ignore-user-config", wrapper)
        self.assertIn("CURATOR_CODEX_HOME", wrapper)
        self.assertNotIn("--output-schema", wrapper)
        model = json.loads(Path(runtime.model_catalog_path).read_text())["models"][0]
        self.assertTrue(model["supports_parallel_tool_calls"])
        self.assertIn("skill curator", model["base_instructions"])
        sandbox = object.__new__(curate.CuratorSandbox)
        sandbox.runtime, sandbox.home = runtime, work / "isolated-home"
        with patch.dict(
            curate.os.environ, {"ALFWORLD_SESSION_URL": "SECRET_GAME", "ALFWORLD_ADMIN_TOKEN": "SECRET_ADMIN"}
        ):
            env = sandbox.process_environment()
        self.assertFalse(any("ALFWORLD" in name for name in env))
        self.assertNotIn("SECRET", json.dumps(env))
        self.assertNotIn("/opt/alfworld-tools", env["PATH"])

    def test_actual_three_startup_warnings_are_retained_as_diagnostics_not_tools(self):
        events = actual_startup_events()
        result = classify(events)
        self.assertEqual(len(result["startup_diagnostics"]), 3)
        self.assertEqual(result["tool_events"], [])
        self.assertEqual(result["fatal_events"], [])
        self.assertEqual([item["event"] for item in result["startup_diagnostics"]], events[1:4])

    def test_command_mcp_web_file_and_plan_events_remain_proved_tools(self):
        for kind in ("command_execution", "mcp_tool_call", "web_search", "file_change", "plan"):
            with self.subTest(kind=kind):
                result = classify([{"type": "turn.started"}, {"type": "item.completed", "item": {"type": kind}}])
                self.assertEqual(len(result["tool_events"]), 1)
                self.assertEqual(result["startup_diagnostics"], [])
                self.assertEqual(result["fatal_events"], [])

    def test_unknown_errors_and_misbound_or_late_warning_are_fatal_not_semantic_tools(self):
        known = actual_startup_events()[1]
        variants = [
            [{"type": "item.completed", "item": {"type": "error", "message": "Local EMA server failed with HTTP 500"}}],
            [{"type": "error", "message": "Transport disconnected"}],
            [{"type": "turn.started"}, known],
            [{"type": "item.completed", "item": {"type": "future_unknown_item"}}],
            [
                {
                    "type": "item.completed",
                    "item": {
                        "type": "error",
                        "message": actual_startup_events()[3]["item"]["message"].replace(
                            ACTUAL_CURATE_HOME, "/tmp/untrusted-home"
                        ),
                    },
                }
            ],
        ]
        for events in variants:
            with self.subTest(events=events):
                result = classify(events)
                self.assertTrue(result["fatal_events"])
                self.assertEqual(result["tool_events"], [])
                self.assertEqual(result["startup_diagnostics"], [])

    def test_proved_startup_diagnostics_do_not_reject_valid_json_or_change_old_mu(self):
        def sandbox(*args):
            instance = FakeSandbox(*args)
            instance.native_event_classification = classify(actual_startup_events())
            return instance

        result = self.collect(_sandbox_factory=sandbox)
        record = self.replay.get(result.record_id)
        self.assertEqual(record.prompt_ids, (101, 102, 103))
        self.assertEqual(record.action_ids, (501, 502, 503))
        self.assertEqual(record.behavior_logprobs, (-0.3, 0.0, -0.7))
        self.assertEqual(len(record.context_snapshot["native_event_classification"]["startup_diagnostics"]), 3)

    def test_unknown_native_error_preserves_capture_then_raises_infrastructure_failure(self):
        def sandbox(*args):
            instance = FakeSandbox(*args)
            instance.native_event_classification = classify(
                [
                    {"type": "item.completed", "item": {"type": "error", "message": "EMA gateway transport failed"}},
                ]
            )
            return instance

        with self.assertRaises(curate.CurateNativeExecutionError):
            self.collect(_sandbox_factory=sandbox)
        capture = curate._read_immutable(self.root / "collections" / self.request.collection_id / "generated.json")
        self.assertEqual(capture["trajectories"][0]["behavior_logprobs"], [-0.3, 0, -0.7])
        self.assertFalse(capture["tool_events_seen"])
        self.assertTrue(capture["native_event_classification"]["fatal_events"])
        self.assertFalse((self.replay.root / "records").exists())

    def test_missing_task_families_is_rejected_without_filling_or_rewriting_output(self):
        self.content = json.dumps(
            {
                "skill_id": self.request.skill_id,
                "title": "Clean target",
                "principle": "Clean the held target.",
                "when_to_apply": "A clean object is required.",
            }
        )
        with self.assertRaisesRegex(curate.CurateProtocolError, "five allowed skill fields"):
            self.collect()
        capture = curate._read_immutable(self.root / "collections" / self.request.collection_id / "generated.json")
        self.assertEqual(capture["content"], self.content)
        self.assertNotIn("task_families", json.loads(capture["content"]))
        self.assertFalse((self.replay.root / "records").exists())
        template = Path(self.runtime.task_prompt_path).read_text()
        self.assertGreater(template.rfind('"task_families"'), template.index("{{EPISODE_JSON}}"))


class CuratorCatalogCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_catalog_and_owned_home_are_archived_then_removed(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            sandbox = object.__new__(curate.CuratorSandbox)
            sandbox.collection_id = "a" * 64
            sandbox.workdir, sandbox.output_dir = root / sandbox.collection_id, root / "output"
            sandbox.workdir.mkdir()
            sandbox.output_dir.mkdir()
            sandbox.home = sandbox.workdir / ".home"
            sandbox.home.mkdir()
            state = sandbox.home / "native-state.json"
            state.write_bytes(b"original curator native home state")
            sandbox.runtime = SimpleNamespace(uid=os.geteuid(), gid=os.getegid(), work_root=str(root))
            sandbox.process, sandbox.created_files = None, []
            catalog = Path("/tmp") / ("codex-model-catalog-" + uuid.uuid4().hex + ".json")
            try:
                await sandbox.write_file(str(catalog), '{"model":"original curator config"}')
                self.assertEqual(sandbox.created_files, [catalog])
                await sandbox.close()
                receipt = json.loads((sandbox.output_dir / "native-files.json").read_text())
                self.assertEqual(receipt["status"], "archived_and_removed")
                self.assertFalse(catalog.exists())
                self.assertFalse(sandbox.workdir.exists())
                import tarfile

                with tarfile.open(receipt["archive_path"]) as archive:
                    self.assertEqual(archive.extractfile("curator_workspace-1/.home/native-state.json").read(),
                                     b"original curator native home state")
            finally:
                catalog.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
