"""CPU-only production coverage scenarios; no models, Ray, or evaluation calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
SPEC = importlib.util.spec_from_file_location("_validation_case_test", ROOT / "validation_audit.py")
audit_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_module)


class CaseAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.digest = "a" * 64

    def audit(self, count=3, phase="test_seen"):
        identities = {f"{phase}-case-{index:03d}" for index in range(count)}
        return audit_module.ValidationCaseAudit(
            run_root=self.root,
            output_dir=self.root / phase,
            stage="initial",
            phase=phase,
            split="valid_seen" if phase == "test_seen" else "valid_unseen",
            global_step=1,
            expected_ids=identities,
            catalog_digest=self.digest,
        )

    def submissions(self, audit):
        games = sorted(audit.expected_ids)
        uids = ["uid-" + game for game in games]
        kwargs = [
            {
                "task": {
                    "name": "alfworld_skill_training",
                    "prompt": [{"role": "user", "content": game}],
                    "metadata": {
                        "game_id": game,
                        "split": audit.split,
                        "dataset_phase": audit.phase,
                        "catalog_digest": self.digest,
                        "run_dir": str(self.root),
                    },
                }
            }
            for game in games
        ]
        stamped = audit.submit(uids, kwargs)
        return [(uid, value["task"]["metadata"]) for uid, value in zip(uids, stamped, strict=True)]

    def raw(self, uid, metadata, *, won, native_error=None):
        directory = self.root / "episodes" / metadata["game_id"] / ("attempt_" + uid)
        directory.mkdir(parents=True)
        identity = {**metadata, "game_sha256": "b" * 64, "attempt_id": uid}
        (directory / "attempt.json").write_text(json.dumps(identity))
        result = {
            **identity,
            "won": won,
            "reward": float(won),
            "accuracy": float(won),
            "agent_ok": native_error is None,
            "agent_error": native_error,
        }
        (directory / "result.json").write_text(json.dumps(result))
        engine = {key: identity[key] for key in ("game_id", "game_sha256", "split", "catalog_digest")}
        (directory / "engine-result.json").write_text(json.dumps({**engine, "won": won}))
        return directory

    def materialized(self, uid, metadata, directory, won, *, prefix=0, score=None):
        return {
            "rollout_uid": uid,
            "session_index": 0,
            "trajectory_key": f"{uid}_0_{prefix}",
            "score": float(won) if score is None else score,
            "sampled_action_tokens": 1,
            "min_global_steps": 0,
            "max_global_steps": 0,
            "reward_context": {**metadata, "episode_output_dir": str(directory), "native_string": "host-only"},
        }

    def dump(self, audit, materialized):
        path = audit.output_dir / "1.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "uid": row["trajectory_key"],
                        "score": row["score"],
                        "acc": row["score"],
                        "input": "task",
                        "output": "",
                    }
                )
                + "\n"
                for row in materialized
            )
        )
        return path

    def test_full140_and134_with_prefix_duplicates_reconcile_raw_engine_cases(self):
        for phase, count in (("test_seen", 140), ("test_unseen", 134)):
            with self.subTest(phase=phase):
                audit = self.audit(count, phase)
                materialized = []
                for index, (uid, metadata) in enumerate(self.submissions(audit)):
                    won = index % 3 == 0
                    directory = self.raw(uid, metadata, won=won)
                    materialized.extend(
                        self.materialized(uid, metadata, directory, won, prefix=prefix) for prefix in (0, 1)
                    )
                audit.capture(materialized)
                report = audit.finish(dump_path=self.dump(audit, materialized))
                self.assertTrue(report["complete"])
                self.assertEqual(report["raw_task_cases"], count)
                self.assertEqual(report["materialized_cases"], count)
                self.assertEqual(report["expected_case_denominator"], count)
                self.assertEqual(report["observed_weight_versions"], [0])
                self.assertEqual(report["global_cursor"], 1)
                self.assertFalse(report["optimizer_version_inferred_from_cursor"])

    def test_failed_empty_uid_exposes139_reward_vs140_accuracy_denominator_trap(self):
        audit = self.audit(140)
        materialized = []
        for index, (uid, metadata) in enumerate(self.submissions(audit)):
            won = index < 70
            directory = self.raw(uid, metadata, won=won, native_error="no sampled output" if index == 139 else None)
            if index != 139:
                materialized.append(self.materialized(uid, metadata, directory, won))
        audit.capture(materialized)
        parent = {"val-aux/alfworld/reward/mean@1": 70 / 139, "val-core/alfworld/acc/mean@1": 70 / 140}
        original = copy.deepcopy(parent)
        with self.assertRaises(audit_module.ValidationCoverageError):
            audit.finish(dump_path=self.dump(audit, materialized), parent_metrics=parent)
        report = json.loads(audit.report_path.read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["raw_task_cases"], 140)
        self.assertEqual(report["materialized_cases"], 139)
        self.assertEqual(report["authoritative_reward_mean"], 70 / 140)
        self.assertEqual(report["parent_observed_reward_mean"], 70 / 139)
        self.assertEqual(report["native_error_cases"], 1)
        self.assertEqual(parent, original)
        self.assertEqual(report["cases"][-1]["engine_won"], False)

    def test_unobserved_task_is_unknown_not_synthetic_zero(self):
        audit = self.audit(1)
        self.submissions(audit)
        with self.assertRaises(audit_module.ValidationCoverageError):
            audit.finish(dump_path=self.dump(audit, []))
        report = json.loads(audit.report_path.read_text())
        self.assertIsNone(report["authoritative_reward_mean"])
        self.assertIsNone(report["cases"][0]["engine_won"])
        self.assertIsNone(report["cases"][0]["authoritative_accuracy"])

    def test_native_bookkeeping_error_does_not_rewrite_an_observed_engine_win(self):
        audit = self.audit(1)
        uid, metadata = self.submissions(audit)[0]
        directory = self.raw(uid, metadata, won=True, native_error="actual CLI diagnostic")
        materialized = [self.materialized(uid, metadata, directory, True)]
        audit.capture(materialized)
        report = audit.finish(dump_path=self.dump(audit, materialized))
        self.assertTrue(report["complete"])
        self.assertEqual(report["authoritative_wins"], 1)
        self.assertEqual(report["authoritative_accuracy"], 1)
        self.assertEqual(report["native_error_cases"], 1)

    def test_observed_empty_eos_tokens_are_valid_but_mismatched_score_is_not(self):
        audit = self.audit(1)
        uid, metadata = self.submissions(audit)[0]
        directory = self.raw(uid, metadata, won=False)
        materialized = [self.materialized(uid, metadata, directory, False)]
        audit.capture(materialized)
        self.assertTrue(audit.finish(dump_path=self.dump(audit, materialized))["complete"])
        other = self.audit(1, "test_unseen")
        uid, metadata = self.submissions(other)[0]
        directory = self.raw(uid, metadata, won=True)
        materialized = [self.materialized(uid, metadata, directory, True, score=0)]
        other.capture(materialized)
        with self.assertRaises(audit_module.ValidationCoverageError):
            other.finish(dump_path=self.dump(other, materialized))

    def test_metadata_stamp_preserves_model_messages_and_caller_data(self):
        tools = {
            "task": {
                "name": "alfworld_skill_training",
                "prompt": [{"role": "user", "content": "exact task"}],
                "metadata": {"rollout_uid": "forged", "session_index": 99, "global_step": 99},
            }
        }
        original = copy.deepcopy(tools)
        stamped = audit_module.stamp_task_identity(tools, rollout_uid="host-uid", session_index=2, global_step=3)
        self.assertEqual(tools, original)
        self.assertEqual(stamped["task"]["prompt"], original["task"]["prompt"])
        self.assertEqual(stamped["task"]["metadata"], {"rollout_uid": "host-uid", "session_index": 2, "global_step": 3})

    def test_scoped_proxies_preserve_results_and_keep_context_outside_reward_metrics(self):
        audit = self.audit(1)
        uid, metadata = self.submissions(audit)[0]
        directory = self.raw(uid, metadata, won=False)
        source_extra = {
            "reward_extra_info": {"acc": 0},
            "runner_reward_info": {
                "reward_context": {**metadata, "episode_output_dir": str(directory), "error": "host-only-string"},
            },
        }
        result = (SimpleNamespace(keys=[uid + "_0_0"], partition_id="val"), {"original": 1})
        sampler = SimpleNamespace(sample=lambda **kwargs: result)
        reader = lambda batch: [self.materialized(uid, metadata, directory, False)]
        proxy = audit_module.SamplerAuditProxy(sampler, audit, reader)
        self.assertIs(proxy.sample(global_steps=1, partition_id="val", batch_size=1), result)
        self.assertEqual(source_extra["reward_extra_info"], {"acc": 0})
        self.assertNotIn("host-only-string", json.dumps(source_extra["reward_extra_info"]))


class RuntimeAdapterTest(unittest.TestCase):
    def test_runtime_proxies_use_tu_get_with_linkedlist_columns_without_tolist(self):
        spec = importlib.util.spec_from_file_location("_eval_runtime_case_test", ROOT / "evaluation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class LinkedList(list):
            pass

        class Vector:
            def __init__(self, values):
                self.values = values

            def tolist(self):
                return self.values

        class TensorRows:
            is_nested = False

            def __init__(self, values):
                self.values = values

            def sum(self, dim):
                return Vector([sum(row) for row in self.values])

        tu = types.ModuleType("verl.utils.tensordict_utils")
        tu.get = lambda data, key, default=None: data.get(key, default)
        tu.assign_non_tensor_stack = lambda data, key, value: data.__setitem__(key, LinkedList(value))
        utils = types.ModuleType("verl.utils")
        utils.tensordict_utils = tu
        transfer = types.ModuleType("verl.utils.transferqueue_utils")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            audit = audit_module.ValidationCaseAudit(
                run_root=root,
                output_dir=root / "output",
                stage="initial",
                phase="test_seen",
                split="valid_seen",
                global_step=1,
                expected_ids={"game"},
                catalog_digest="a" * 64,
            )
            tools = {
                "task": {
                    "name": "alfworld_skill_training",
                    "metadata": {
                        "game_id": "game",
                        "split": "valid_seen",
                        "dataset_phase": "test_seen",
                        "catalog_digest": "a" * 64,
                        "run_dir": str(root),
                    },
                }
            }
            prompt_ids = [[1, 2, 3]]
            batch = {
                "uid": LinkedList(["actual-uid"]),
                "extra_info": LinkedList([{"tools_kwargs": tools}]),
                "prompts": prompt_ids,
            }
            context = {"episode_output_dir": str(root / "episode"), "host_string": "not a reward"}
            extra = {"reward_extra_info": {"acc": 0.0}, "runner_reward_info": {"reward_context": context}}
            fields = {
                "uid": LinkedList(["actual-uid"]),
                "session_id": LinkedList([0]),
                "extra_fields": LinkedList([extra]),
                "rm_scores": TensorRows([[0.0]]),
                "response_mask": TensorRows([[1]]),
            }
            transfer.tq = SimpleNamespace(kv_batch_get=lambda **kwargs: fields)
            with patch.dict(
                sys.modules,
                {"verl.utils": utils, "verl.utils.tensordict_utils": tu, "verl.utils.transferqueue_utils": transfer},
            ):
                module._prepare_validation_batch(batch, audit)
                captured = module._validation_tq_rows(SimpleNamespace(keys=["actual-uid_0_0"], partition_id="val"))
            self.assertIs(batch["prompts"], prompt_ids)
            self.assertEqual(batch["tools_kwargs"][0]["task"]["metadata"]["rollout_uid"], "actual-uid")
            self.assertEqual(captured[0]["reward_context"], context)
            self.assertEqual(extra["reward_extra_info"], {"acc": 0.0})
            self.assertEqual(captured[0]["sampled_action_tokens"], 1)

    def test_framework_episode_hook_stamps_only_tools_metadata(self):
        package_name = "_case_framework_scope"
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT)]

        class Parent:
            async def _run_agent_episode(self, **kwargs):
                return kwargs

        entry = types.ModuleType("uni_agent.framework.entry")
        entry.AgentFrameworkRolloutAdapter = type("Adapter", (), {})
        framework = types.ModuleType("uni_agent.framework.framework")
        framework.GatewayAgentFramework = Parent
        with patch.dict(
            sys.modules,
            {package_name: package, "uni_agent.framework.entry": entry, "uni_agent.framework.framework": framework},
        ):
            spec = importlib.util.spec_from_file_location(package_name + ".framework", ROOT / "framework.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            instance = module.SkillAwareAgentFramework()
            raw_prompt = [{"role": "user", "content": "same native task"}]
            fields = {
                "uid": "uid",
                "raw_prompt": raw_prompt,
                "tools_kwargs": {"task": {"name": "alfworld_skill_training", "metadata": {}}},
            }
            import asyncio

            result = asyncio.run(
                instance._run_agent_episode(
                    sample_fields=fields,
                    sample_index=0,
                    session_index=1,
                    global_steps=7,
                    runner_name="native",
                    runner_config=object(),
                    sampling_params={"temperature": 0},
                )
            )
        self.assertIs(result["sample_fields"]["raw_prompt"], raw_prompt)
        self.assertEqual(fields["tools_kwargs"]["task"]["metadata"], {})
        self.assertEqual(result["sample_fields"]["tools_kwargs"]["task"]["metadata"]["session_index"], 1)


if __name__ == "__main__":
    unittest.main()
