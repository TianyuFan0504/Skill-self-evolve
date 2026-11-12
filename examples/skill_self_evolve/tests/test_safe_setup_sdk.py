"""Real installed Uni-Agent alias/resolver/runner integration, CPU only.

No registry, resolver, Pydantic config, runner loader or core exception handler
is replaced. Terminal episode execution or the HTTP transport is the explicit
fixture boundary. Nothing sends HTTP, starts Native, allocates a UID or a model.
Run in the existing training interpreter after bootstrap; an unavailable SDK
is reported as an integration skip and is never treated as SDK proof.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
MODULE = "skill_harness_examples.train_resilient_setup"


class SafeSetupSDKTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        sdk_source = Path(os.environ.get("UNI_AGENT_SOURCE_ROOT", "/root/nas_chaohuang/benchmarks/alfworld/uni-agent"))
        if not sdk_source.is_dir() and importlib.util.find_spec("uni_agent") is None:
            raise unittest.SkipTest("Actual Uni-Agent SDK unavailable; this skip supplies no integration evidence")
        sys.path.insert(0, str(ROOT))
        from verl.experimental.skill_harness.bootstrap import configure_runtime

        provenance = configure_runtime(register_tasks=False)
        cls.sdk = importlib.import_module("uni_agent")
        if not Path(cls.sdk.__file__).resolve().is_relative_to(Path(provenance["uni_agent_source"]).resolve()):
            raise RuntimeError("Actual SDK origin differs from the pinned bootstrap")
        cls.registry = importlib.import_module("uni_agent.tasks.registry")
        cls.sdk_config = importlib.import_module("uni_agent.tasks.config")
        cls.sdk_tasks = importlib.import_module("uni_agent.tasks")
        cls.sdk_gateway = importlib.import_module("uni_agent.gateway.session")
        cls.sdk_framework = importlib.import_module("uni_agent.framework.framework")
        cls.core = importlib.import_module("verl.experimental.skill_harness.alfworld.task")
        cls.accounts = importlib.import_module("verl.experimental.skill_harness.alfworld.accounts")
        cls.context = importlib.import_module("verl.experimental.skill_harness.context")
        cls.store_module = importlib.import_module("verl.experimental.skill_harness.store")
        cls.protocol = importlib.import_module("verl.experimental.skill_harness.alfworld.protocol")
        cls.driver = importlib.import_module(MODULE)
        cls.original_class = cls.core.AlfworldSkillTrainingTask

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)
        self.environment = patch.dict(os.environ, {"SKILL_RUN_DIR": str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.profile = self.root / "task_config.json"
        self.profile_data = [{
            "name": self.driver.ORIGINAL_TASK,
            "sandbox": {"provider": "local", "runtime_timeout": 900},
            "agent": {"name": "codex", "run_timeout": 900, "codex_bin": "/unused/native",
                      "model_catalog_path": "/unused/catalog.json"},
        }]
        self.profile.write_text(json.dumps(self.profile_data))
        self.raw_prompt = [{"role": "user", "content": "Complete the goal using the current environment."}]
        metadata = {
            "run_dir": str(self.root), "game_id": "train_sdk_fixture", "stable_game_id": "train_sdk_fixture",
            "catalog_digest": "b" * 64, "game_sha256": "a" * 64,
            "relative_game_path": "json_2.1.1/train/fixture/trial/game.tw-pddl",
            "split": "train", "dataset_phase": "train", "task_type": "pick_and_place_simple",
            "scene_id": "FloorPlan1", "rollout_uid": "original-logical-group", "session_index": 2,
            "global_step": 5, "role": "solver", "sample_index": 99,
        }
        self.tools = {"task": {"name": self.driver.ORIGINAL_TASK, "metadata": metadata,
                               "prompt": [{"role": "user", "content": "Serialized prompt must lose to raw_prompt."}]}}
        self.session = self.sdk_gateway.SessionHandle(
            session_id="sdk-fixture", base_url="http://127.0.0.1:1/sessions/sdk-fixture/v1"
        )
        self.observed = []

    def observe_episode(self):
        observed, result_type = self.observed, self.sdk_tasks.TaskResult

        async def run(task):
            observed.append({"instance_type": type(task), "config": task.config.model_dump()})
            return result_type(reward=1.0, accuracy=1.0, finished=True, extra_info={"sdk_fixture": True})

        return run

    async def test_public_registry_alias_preserves_the_original_task_class(self):
        original = self.registry.get_task_cls(self.driver.ORIGINAL_TASK)
        safe = self.driver.ensure_safe_task()
        self.assertIs(original, self.original_class)
        self.assertIs(self.registry.get_task_cls(self.driver.ORIGINAL_TASK), original)
        self.assertIs(self.registry.get_task_cls(self.driver.SAFE_TASK), safe)
        self.assertEqual(original.name, self.driver.ORIGINAL_TASK)
        self.assertEqual(safe.name, self.driver.SAFE_TASK)
        self.assertIs(safe.run, original.run)
        self.assertTrue(issubclass(safe.setup_error_class, self.accounts.LocalArchiveDataError))
        self.assertIs(self.driver.ensure_safe_task(), safe)

    async def test_real_resolver_requires_the_alias_profile_and_retains_900_budgets(self):
        before, tools_before = self.profile.read_bytes(), copy.deepcopy(self.tools)
        alias = self.driver.alias_sample(self.tools)
        with self.assertRaisesRegex(ValueError, "no Task Config"):
            self.sdk_config.TaskConfigResolver.from_file(str(self.profile)).resolve(alias["task"])
        profile = Path(self.driver.alias_task_profile(self.profile, self.root))
        resolved = self.sdk_config.TaskConfigResolver.from_file(str(profile)).resolve(
            alias["task"], runtime_model={"base_url": self.session.base_url, "model_name": "Qwen3.5-4B-nothink"}
        )
        task = self.registry.get_task(resolved)
        self.assertIsInstance(task, self.driver.ensure_safe_task())
        self.assertEqual(task.config.name, self.driver.SAFE_TASK)
        self.assertEqual(task.config.sandbox.runtime_timeout, 900)
        self.assertEqual(task.config.agent.run_timeout, 900)
        self.assertEqual(task.config.metadata, self.tools["task"]["metadata"])
        self.assertEqual(stat.S_IMODE(profile.stat().st_mode) & 0o077, 0)
        self.assertEqual(self.profile.read_bytes(), before)
        self.assertEqual(self.tools, tools_before)

    async def test_actual_sdk_runner_FQN_parser_and_Pydantic_resolution_preserve_source_identity(self):
        from omegaconf import OmegaConf

        config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": {
            "agent_runners": {"task": {"runner_fqn": "verl.experimental.skill_harness.alfworld.runner.run_task",
                "runner_kwargs": {"task_config_path": str(self.profile), "model_name": "Qwen3.5-4B-nothink"},
                "dispatch_mode": "inline_async", "max_concurrent_sessions": 64}}
        }}}}, "trainer": {"v1": {"sampler": {"custom_sampler": {
            "path": "/unused/group_sampler.py", "name": "ArchiveAwareReplayBuffer"
        }}}}})
        self.driver.configure_extensions(config)
        parsed = self.sdk_framework.GatewayAgentFramework._parse_framework_config(config)
        runner_config = parsed["runner_registry"]["task"]
        self.assertEqual(runner_config.runner_fqn, MODULE + ".safe_run_task")
        self.assertEqual(runner_config.max_concurrent_sessions, 32)
        self.assertEqual(config.trainer.v1.sampler.custom_sampler.path, str(Path(self.driver.__file__).resolve()))
        runner = self.sdk_framework._materialize_runner(runner_config.runner_fqn, runner_config.runner_kwargs)
        original = copy.deepcopy(self.tools)
        with patch.object(self.original_class, "_run_episode", self.observe_episode()):
            result = await runner(session=self.session, tools_kwargs=self.tools,
                                  raw_prompt=self.raw_prompt, sample_index=7)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(self.tools, original)
        captured = self.observed[0]["config"]
        self.assertEqual(captured["prompt"], self.raw_prompt)
        self.assertEqual(captured["metadata"]["sample_index"], 7)
        self.assertEqual(captured["metadata"]["rollout_uid"], "original-logical-group")
        self.assertEqual(captured["metadata"]["session_index"], 2)
        self.assertEqual(captured["agent"]["model"]["base_url"], self.session.base_url)
        self.assertEqual(self.tools["task"]["name"], self.driver.ORIGINAL_TASK)

    async def test_alias_copy_leaves_actual_skill_prompt_budget_and_pinned_metadata_identical(self):
        store = self.store_module.SkillStore(self.root / "store")
        loading = {"enabled": True, "snapshot_id": store.head(), "top_k": 6, "skill_token_budget": 1024,
                   "max_prompt_tokens": 4096, "tokenizer_path": str(self.root / "unused-tokenizer"),
                   "tokenizer_revision": None, "tokenizer_version": "sdk-fixture-tokenizer", "render_kwargs": {}}
        fields = {"uid": "prompt-a", "raw_prompt": self.raw_prompt, "tools_kwargs": copy.deepcopy(self.tools)}
        pinned = self.context.pin_prompt_group(fields, loading=loading, store_root=store.root,
                                               global_step=5, partition_id="train")
        copied = self.driver.alias_sample(pinned["tools_kwargs"])
        original_metadata = pinned["tools_kwargs"]["task"]["metadata"]
        self.assertEqual(copied["task"]["metadata"], original_metadata)
        module = self.context

        class ByteRenderer:
            def encode(self, text):
                return list(text.encode())

            def render(self, messages):
                text = "".join("<" + row["role"] + ">" + row["content"] + "</turn>" for row in messages)
                return module.RenderedPrompt(text, tuple(text.encode()))

        messages = self.protocol.agent_messages(self.raw_prompt, {
            "observation": "A movable target is visible.", "admissible_commands": ["look"],
            "steps": 0, "max_steps": 50, "won": False, "done": False,
        })
        before = self.context.enrich_task_messages(messages, original_metadata, renderer=ByteRenderer())
        after = self.context.enrich_task_messages(messages, copied["task"]["metadata"], renderer=ByteRenderer())
        self.assertEqual(before.messages, after.messages)
        self.assertEqual(before.manifest, after.manifest)
        self.assertEqual(fields["tools_kwargs"]["task"]["name"], self.driver.ORIGINAL_TASK)

    async def test_counterfactual_uses_real_core_branch_and_SDK_alias_resolution(self):
        fields = {"uid": "cf-source", "raw_prompt": self.raw_prompt, "tools_kwargs": copy.deepcopy(self.tools)}
        fields["tools_kwargs"]["task"]["metadata"].update(
            role="counterfactual_probe", credit_probe_group="c" * 64, credit_probe_variant="with_skill"
        )
        original = copy.deepcopy(fields)
        attestation = {"schema_version": 1, "policy_kind": "frozen_evaluator", "model_path": "/unused/model",
                       "model_fingerprint": "a" * 64, "frozen_model_version": "a" * 64, "lifecycle_id": "b" * 32}
        sdk_gateway = self.sdk_gateway

        class Gateway:
            skill_route_attestation = attestation

            async def create_session(self, session_id, **_kwargs):
                return sdk_gateway.SessionHandle(session_id, "http://127.0.0.1:1/sessions/" + session_id + "/v1")

            async def finalize_session(self, _session_id):
                return [sdk_gateway.Trajectory([11, 12], [13], [1], [-0.4])]

            async def abort_session(self, _session_id):
                raise AssertionError("A completed fixture must not abort its finalized session")

        with patch.object(self.original_class, "_run_episode", self.observe_episode()):
            captured = await self.driver.safe_native_probe_branch(
                gateway_manager=Gateway(), session_id="cf-fixture", sample_fields=fields,
                artifact_dir=self.root / "probe", task_config_path=str(self.profile), frozen_attestation=attestation,
            )
        self.assertEqual(fields, original)
        self.assertTrue(captured["extra_info"]["sdk_fixture"])
        self.assertIs(self.observed[0]["instance_type"], self.driver.ensure_safe_task())
        self.assertEqual(self.observed[0]["config"]["metadata"]["credit_probe_group"], "c" * 64)

    async def test_real_core_precreate_timeout_uses_typed_bypass_and_never_claims_world_closed(self):
        settings = self.root / "runtime/environment.json"
        settings.parent.mkdir()
        settings.write_text(json.dumps({"url": "http://127.0.0.1:1", "admin_token": "UNUSED_CPU_FIXTURE",
            "catalog_digest": "b" * 64, "allowed_splits": ["train"], "max_steps": 50}))

        class Request:
            async def __aenter__(self):
                raise TimeoutError("CPU transport fixture: no request dispatched")

            async def __aexit__(self, *_args):
                return False

        class Session:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, url, **_kwargs):
                if not url.endswith("/admin/create"):
                    raise AssertionError("Unexpected HTTP path in CPU fixture")
                return Request()

        with patch.dict(os.environ, {"ALFWORLD_SERVER_CONFIG": str(settings)}):
            with patch.object(self.core.aiohttp, "ClientSession", Session):
                with self.assertRaises(self.driver.NoNativeSetupDataError) as caught:
                    await self.driver.safe_run_task(session=self.session, tools_kwargs=self.tools,
                        raw_prompt=self.raw_prompt, sample_index=7, task_config_path=str(self.profile))
        proof = caught.exception.disposition
        self.assertEqual(proof["kind"], "NoNativeStartupInvalid")
        self.assertEqual(proof["world_closure"], "unknown")
        self.assertFalse(proof["native_started"])
        self.assertFalse(proof["uid_acquired_by_task"])
        self.assertFalse(proof["backend_binding_created"])
        self.assertNotIn("reward", proof)
        self.assertNotIn("uid_drained", proof)
        self.assertNotIn("backend_drained", proof)
        self.assertFalse((self.root / "diagnostics/account_isolation_incidents").exists())
        self.assertEqual(self.driver.no_native_failed_groups(self.root), {"original-logical-group"})

    async def test_real_official_file_loader_supports_the_lazy_resilient_sampler(self):
        from verl.utils.import_utils import load_extern_type

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            sampler = load_extern_type(str(Path(self.driver.__file__).resolve()), "ResilientReplayBuffer")
        value = sampler(trainer_mode="sync", trainer_config={"train_batch_size": 32},
            max_off_policy_threshold=None, max_off_policy_strategy="drop", sampler_kwargs={}, refill_fn=lambda _n: None)
        self.assertEqual(value.gen_batch_size, 1)
        self.assertTrue(value.sync_refill_failed_groups)
        self.assertEqual(value.archive_run_root, str(self.root))


if __name__ == "__main__":
    unittest.main()
