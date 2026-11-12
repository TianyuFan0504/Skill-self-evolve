"""CPU adapter contracts; no server, model, Native process or GPU calls."""

import ast
import asyncio
import importlib
import importlib.util
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "examples/skill_self_evolve/scripts/train_resilient_setup.py"
spec = importlib.util.spec_from_file_location("_resilient_setup_cpu", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class LocalArchiveDataError(RuntimeError):
    def __init__(self, reason, disposition=None):
        super().__init__(reason)
        self.disposition = disposition


class Context:
    def __init__(self, fail):
        self.fail = fail

    async def __aenter__(self):
        if self.fail:
            raise TimeoutError("fixture original create await")
        return self

    async def __aexit__(self, *args):
        pass


class OriginalTask:
    name = driver.ORIGINAL_TASK

    def __init__(self, root, *, mode="create", role=None):
        self.root, self.mode, self.incidents = root, mode, []
        self.config = types.SimpleNamespace(metadata={
            "run_dir": str(root), "rollout_uid": "original-uid", "dataset_phase": "train",
            "game_id": "original-game", "role": role,
        })

    def _trusted_run_root(self):
        return self.root

    async def _run_episode(self):
        metadata = self.config.metadata  # noqa: F841 -- classifier attests the original frame binding
        attempt_id = "a" * 32
        output = self.root / "episodes/game" / ("attempt_" + attempt_id)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"attempt_id": attempt_id}
        (output / "attempt.json").write_text(json.dumps(manifest))
        base, headers, payload = "http://127.0.0.1", {}, {}
        client = types.SimpleNamespace(post=lambda url, **kwargs: Context(self.mode != "close"))
        if self.mode == "after_binding":
            created = {"session_id": "known"}  # noqa: F841
        if self.mode == "value_error":
            raise ValueError("not a timeout")
        if self.mode == "cancel":
            raise asyncio.CancelledError()
        async with client.post(base + "/admin/create", headers=headers, json=payload) as response:  # noqa: F841
            pass
        created = {}  # noqa: F841
        session_id = "known"  # noqa: F841
        async with Context(True):
            pass


# Compile the actual core outer catch unchanged, rather than mirror its logic.
tree = ast.parse((ROOT / "verl/experimental/skill_harness/alfworld/task.py").read_text())
task_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "AlfworldSkillTrainingTask")
run = next(node for node in task_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run")
namespace = {
    "__name__": "verl.experimental.skill_harness.alfworld.task",
    "__package__": "verl.experimental.skill_harness.alfworld",
    "Path": Path, "asyncio": asyncio, "LocalArchiveDataError": LocalArchiveDataError,
    "AccountIsolationError": type("AccountIsolationError", (RuntimeError,), {}),
    "record_isolation_incident": lambda root, reason: OriginalTask.current.incidents.append(reason),
}
compiled = compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])), "<actual-core-run>", "exec")
exec(compiled, namespace)
OriginalTask.run = namespace["run"]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        registry = {driver.ORIGINAL_TASK: OriginalTask}

        def register(name):
            def decorate(cls):
                if name in registry:
                    raise ValueError("duplicate")
                cls.name = name
                registry[name] = cls
                return cls
            return decorate

        def get(name):
            if name not in registry:
                raise ValueError("unknown")
            return registry[name]

        self.registry = registry
        self.modules = {
            "uni_agent.tasks.registry": types.SimpleNamespace(register_task=register, get_task_cls=get),
            "verl.experimental.skill_harness.alfworld.task": types.SimpleNamespace(
                AlfworldSkillTrainingTask=OriginalTask),
            "verl.experimental.skill_harness.alfworld.accounts": types.SimpleNamespace(
                LocalArchiveDataError=LocalArchiveDataError),
            "verl.experimental.skill_harness.alfworld.budget": types.SimpleNamespace(
                BudgetOutcomeUnknown=type("BudgetOutcomeUnknown", (RuntimeError,), {})),
        }
        self.imports = patch.dict("sys.modules", self.modules)
        self.imports.start()
        self.addCleanup(self.imports.stop)
        driver._safe_class = driver._error_class = None
        self.safe = driver.ensure_safe_task()
        self.addCleanup(setattr, driver, "_safe_class", None)
        self.addCleanup(setattr, driver, "_error_class", None)
        self.profile = self.root / "task_config.json"
        self.profile.write_text(json.dumps([{"name": driver.ORIGINAL_TASK,
                                            "sandbox": {"runtime_timeout": 900},
                                            "agent": {"run_timeout": 900, "model_catalog_path": "unchanged"}}]))

    def task(self, **kwargs):
        task = self.safe(self.root, **kwargs)
        OriginalTask.current = task
        return task

    def test_exact_precreate_timeout_uses_actual_outer_nonsticky_catch_and_unknown_world(self):
        task = self.task()
        with self.assertRaises(driver.NoNativeSetupDataError) as captured:
            asyncio.run(task.run())
        self.assertIsInstance(captured.exception, LocalArchiveDataError)
        self.assertEqual(task.incidents, [])
        proof = captured.exception.disposition
        self.assertEqual(proof["world_closure"], "unknown")
        self.assertFalse(proof["native_started"])
        self.assertFalse(proof["training_data_valid"])
        self.assertEqual(driver.no_native_failed_groups(self.root), {"original-uid"})
        self.assertNotIn("reward", proof)
        self.assertFalse((self.root / "diagnostics/account_isolation_incidents").exists())

    def test_after_create_binding_timeout_keeps_original_fatal_marker(self):
        task = self.task(mode="after_binding")
        with self.assertRaises(TimeoutError):
            asyncio.run(task.run())
        self.assertEqual(len(task.incidents), 1)
        self.assertEqual(driver.no_native_failed_groups(self.root), set())

    def test_real_core_episode_precreate_prefix_is_classified_without_http_or_native(self):
        # Execute the actual complete method, but its first HTTP await is a
        # local context stub. No post-create statement or runtime import runs.
        core = next(node for node in task_class.body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_episode")

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                return Context(True)

        def attempt(root, game):
            destination = root / "episodes" / game / ("attempt_" + "c" * 32)
            destination.mkdir(parents=True)
            return "c" * 32, destination

        settings = self.root / "environment.json"
        settings.write_text(json.dumps({"url": "http://127.0.0.1", "catalog_digest": "catalog",
                                        "allowed_splits": ["train"], "admin_token": "fixture-only"}))
        scope = {"Path": Path, "json": json, "time": time, "os": os,
                 "validate_loopback_url": lambda value: value, "new_attempt_directory": attempt,
                 "aiohttp": types.SimpleNamespace(ClientSession=lambda **kwargs: Client(),
                                                  ClientTimeout=lambda **kwargs: object())}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[core], type_ignores=[])),
                     str(ROOT / "verl/experimental/skill_harness/alfworld/task.py"), "exec"), scope)
        task = self.task()
        task.config.metadata.update({key: "fixture" for key in (
            "stable_game_id", "split", "task_type", "scene_id", "relative_game_path", "game_sha256")})
        task.config.metadata.update(split="train", catalog_digest="catalog")
        with patch.object(OriginalTask, "_run_episode", scope["_run_episode"]), \
                patch.dict("os.environ", {"ALFWORLD_SERVER_CONFIG": str(settings)}):
            with self.assertRaises(driver.NoNativeSetupDataError) as captured:
                asyncio.run(task.run())
        self.assertEqual(captured.exception.disposition["source_line"], 820)
        self.assertEqual(captured.exception.disposition["world_closure"], "unknown")
        self.assertEqual(task.incidents, [])

    def test_later_close_timeout_and_non_timeout_remain_fatal(self):
        for mode, kind in (("close", TimeoutError), ("value_error", ValueError)):
            with self.subTest(mode=mode):
                task = self.task(mode=mode)
                with self.assertRaises(kind):
                    asyncio.run(task.run())
                self.assertEqual(len(task.incidents), 1)

    def test_cancellation_is_not_reclassified(self):
        task = self.task(mode="cancel")
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(task.run())
        self.assertEqual(task.incidents, [])

    def test_alias_registration_and_profile_preserve_original_and_budgets(self):
        self.assertIs(self.registry[driver.ORIGINAL_TASK], OriginalTask)
        self.assertEqual(OriginalTask.name, driver.ORIGINAL_TASK)
        self.assertEqual(self.safe.name, driver.SAFE_TASK)
        path = driver.alias_task_profile(self.profile, self.root)
        expected = json.loads(self.profile.read_text())
        expected[0]["name"] = driver.SAFE_TASK
        self.assertEqual(json.loads(Path(path).read_text()), expected)
        self.assertEqual(path, driver.alias_task_profile(self.profile, self.root))

    def test_primary_runner_aliases_only_copy_then_delegates_sdk(self):
        original = {"task": {"name": driver.ORIGINAL_TASK, "metadata": {"run_dir": str(self.root)}}}
        before = json.loads(json.dumps(original))
        seen = {}

        async def upstream(**kwargs):
            seen.update(kwargs)
            return "sdk-result"

        with patch.dict("sys.modules", {"uni_agent.framework.task_runner": types.SimpleNamespace(run_task=upstream)}):
            result = asyncio.run(driver.safe_run_task(
                tools_kwargs=original, task_config_path=str(self.profile), raw_prompt=[{"content": "original"}],
                session="session", sample_index=3,
            ))
        self.assertEqual(result, "sdk-result")
        self.assertEqual(original, before)
        self.assertEqual(seen["tools_kwargs"]["task"]["name"], driver.SAFE_TASK)
        self.assertEqual(seen["tools_kwargs"]["task"]["metadata"]["sample_index"], 3)
        self.assertEqual(seen["raw_prompt"], [{"content": "original"}])

    def test_cf_branch_delegates_existing_capture_cleanup_on_aliased_copy(self):
        fields = {"uid": "unchanged", "tools_kwargs": {"task": {
            "name": driver.ORIGINAL_TASK, "metadata": {"run_dir": str(self.root), "role": "counterfactual_probe"}}}}
        before, seen = json.loads(json.dumps(fields)), {}

        async def native(**kwargs):
            seen.update(kwargs)
            return "original-native-result"

        with patch.dict("sys.modules", {"verl.experimental.skill_harness.probes":
                                        types.SimpleNamespace(native_probe_branch=native)}):
            result = asyncio.run(driver.safe_native_probe_branch(
                sample_fields=fields, task_config_path=str(self.profile), gateway_manager="samegateway",
                session_id="sameuid", artifact_dir="sameartifacts", frozen_attestation={"version": 0},
            ))
        self.assertEqual(result, "original-native-result")
        self.assertEqual(fields, before)
        self.assertEqual(seen["sample_fields"]["tools_kwargs"]["task"]["name"], driver.SAFE_TASK)
        self.assertEqual(seen["gateway_manager"], "samegateway")
        self.assertEqual(seen["session_id"], "sameuid")

    def test_cf_marker_cannot_evict_solver_group(self):
        task = self.task(role="counterfactual_probe")
        with self.assertRaises(driver.NoNativeSetupDataError):
            asyncio.run(task.run())
        self.assertEqual(driver.no_native_failed_groups(self.root), set())

    def test_sampler_waits_for_terminal_uid_then_rejects_complete_group(self):
        task = self.task()
        with self.assertRaises(driver.NoNativeSetupDataError):
            asyncio.run(task.run())

        class Parent:
            def _terminal_eviction_reasons(self, *args):
                return set(), set(), {"other-failed"}, {}

        class Sampler(driver.NoNativeSamplerMixin, Parent):
            archive_run_root = str(self.root)
            finished_keys = {"train": set(), "val": set()}
            failure_keys = {"train": set(), "val": set()}

        sampler = Sampler()
        self.assertEqual(sampler._terminal_eviction_reasons(5, "train")[2], {"other-failed"})
        sampler.finished_keys["train"].add("original-uid")
        self.assertEqual(sampler._terminal_eviction_reasons(5, "train")[2], {"other-failed", "original-uid"})
        self.assertEqual(sampler._terminal_eviction_reasons(5, "val")[2], {"other-failed"})

    def test_forged_native_started_marker_is_rejected(self):
        task = self.task()
        with self.assertRaises(driver.NoNativeSetupDataError):
            asyncio.run(task.run())
        path = next((self.root / driver.MARKERS).glob("*.json"))
        value = json.loads(path.read_text())
        value["native_started"] = True
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            driver.no_native_failed_groups(self.root)

    def test_controller_injection_precedes_background_pump_creation(self):
        events = []

        class Parent:
            def _initialize_background(self):
                events.append(self._collect_probe)

        with patch.dict("sys.modules", {"verl.experimental.skill_harness.controller":
                                        types.SimpleNamespace(SkillJointController=Parent)}):
            driver.controller_class()()._initialize_background()
        self.assertEqual(events, [driver.safe_collect_counterfactual_probe])

    def test_lifecycle_links_effective_initialized_receipt_without_mutating_payload(self):
        seen = {}

        class Parent:
            def _lifecycle_record(self, name, payload):
                seen.update(name=name, payload=payload)

        class Trainer(driver.ResilientLifecycleMixin, Parent):
            _resilient_initialized_receipt = {"path": "actual-restored-config", "sha256": "a" * 64}
            _resilient_extension_identity = {"policy": driver.POLICY, "driver_sha256": "b" * 64}

        original = {"optimizer_step": 64}
        Trainer()._lifecycle_record("evaluation_entry.json", original)
        self.assertEqual(original, {"optimizer_step": 64})
        self.assertEqual(seen["payload"]["initialized_receipt_ref"], Trainer._resilient_initialized_receipt)
        self.assertEqual(seen["payload"]["lifecycle_extension"]["policy"], driver.POLICY)

    def test_dedicated_namespace_resolves_both_examples_drivers(self):
        import sys

        with patch.object(sys, "path", [str(ROOT), *sys.path]):
            resilient = importlib.import_module("skill_harness_examples.train_resilient_setup")
            quiescent = importlib.import_module("skill_harness_examples.train_quiescent_eval")
        self.assertEqual(Path(resilient.__file__).resolve(), SCRIPT)
        self.assertEqual(Path(quiescent.__file__).resolve(), SCRIPT.with_name("train_quiescent_eval.py"))
        self.assertEqual(resilient.MODULE, "skill_harness_examples.train_resilient_setup")
        identity = resilient.extension_identity()
        namespace = ROOT / "skill_harness_examples/__init__.py"
        self.assertEqual(identity["namespace_path"], str(namespace))
        self.assertEqual(identity["namespace_sha256"], driver.hashlib.sha256(namespace.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
