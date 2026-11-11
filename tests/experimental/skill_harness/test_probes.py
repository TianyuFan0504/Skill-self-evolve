"""Offline checks of native credit orchestration, not benchmark score evidence."""

from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_native_credit_probe_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
probes = importlib.import_module(PACKAGE + ".probes")
store_module = importlib.import_module(PACKAGE + ".store")
credit = importlib.import_module(PACKAGE + ".credit")


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = store_module.SkillStore(self.root / "store")
        self.base = self.store.head()
        self.store.create_branch("probe", from_snapshot=self.base)
        artifact = store_module.SkillArtifact(
            "heat_skill", "Heat then place", "Take before heating; verify heated state before placing.",
            "When a task requests a heated object", ("pick_heat_then_place_in_recep",),
        )
        candidate = self.store.stage_candidate(artifact, branch="probe")
        self.store.admit(candidate.candidate_id)
        self.treated = self.store.publish([candidate.candidate_id], branch="probe", expected_parent=self.base)
        self.protocol = credit.CreditProtocol(
            evaluator_version="frozen:v1", probe_protocol_version="paired:v1",
            task_distribution_version="original_train:v1", environment_version="alfworld:v1",
            baseline_snapshot_id=self.base, required_exposures=1,
        )
        self.request = probes.ProbeRequest(
            "probe_1", "window_1", "curate_1", artifact.ref,
            self.treated.snapshot_id, self.base, "frozen:v1", self.protocol, 3,
        )
        self.sample = {
            "uid": "source_b", "raw_prompt": [{"role": "user", "content": "Heat a potato and place it."}],
            "tools_kwargs": {"task": {
                "name": "alfworld_skill_training",
                "metadata": {"split": "train", "dataset_phase": "train", "stable_game_id": "original_b",
                             "game_sha256": "a" * 64, "task_type": "pick_heat_then_place_in_recep"},
            }},
        }
        self.loading = {
            "enabled": True, "branch": "main", "top_k": 6, "skill_token_budget": 1024,
            "max_prompt_tokens": 4096, "tokenizer_path": "/local/tokenizer", "tokenizer_revision": None,
            "tokenizer_version": "tokenizer:v1", "render_kwargs": {"enable_thinking": False},
        }
        self.attestation = {
            "policy_kind": "frozen_evaluator", "frozen_model_version": "frozen:v1", "lifecycle_id": "route_1",
        }
        self.calls = []
        self.rewards = {True: 1.0, False: 0.0}
        self.changed_state = False
        self.fail_branch = False
        self.missing_injection = False

    async def branch(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_branch:
            raise RuntimeError("native session crashed before durable token capture")
        metadata = kwargs["sample_fields"]["tools_kwargs"]["task"]["metadata"]
        loaded = metadata["skill_snapshot_id"] == self.request.with_snapshot_id
        output = kwargs["artifact_dir"] / "episode"
        output.mkdir()
        reward = self.rewards[loaded]
        initial = {"feedback": "Same kitchen.", "admissible_commands": ["look"], "steps": 0}
        if self.changed_state and loaded:
            initial["feedback"] = "Different kitchen."
        (output / "initial-state.json").write_text(json.dumps(initial))
        (output / "engine-result.json").write_text(json.dumps({
            "game_sha256": "a" * 64, "split": "train", "won": bool(reward),
        }))
        return {
            "reward": reward, "accuracy": reward, "finished": True,
            "policy_attestation": kwargs["frozen_attestation"],
            "trajectories": [{"prompt_ids": [1, 2], "response_ids": [3], "extra_fields": {}}],
            "extra_info": {
                "eval_completed": True, "error": None, "episode_output_dir": str(output),
                "skill_context": {"snapshot_id": metadata["skill_snapshot_id"],
                                  "injected_refs": [self.request.skill_ref.to_dict()]
                                  if loaded and not self.missing_injection else []},
            },
        }

    async def collect(self, **changes):
        kwargs = dict(
            request=self.request, sample_fields=self.sample, store=self.store, loading=self.loading,
            gateway_manager=object(), artifact_root=self.root, task_config_path="task_config.yaml",
            frozen_attestation=self.attestation, run_branch=self.branch,
        )
        kwargs.update(changes)
        return await probes.collect_counterfactual_probe(**kwargs)

    async def test_paired_result_and_completed_retry_reuse_original_native_captures(self):
        event = await self.collect()
        retry = await self.collect()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(event.to_dict(), retry.to_dict())
        self.assertEqual(event.with_skill.score - event.without_skill.score, 1.0)
        self.assertEqual(event.with_skill.environment_state_hash, event.without_skill.environment_state_hash)
        self.assertNotIn("skill_snapshot_id", self.sample["tools_kwargs"]["task"]["metadata"])

    async def test_negative_and_observed_zero_utility_are_preserved(self):
        self.rewards = {True: 0.0, False: 1.0}
        event = await self.collect()
        self.assertEqual(event.with_skill.score - event.without_skill.score, -1.0)
        second = replace(self.request, probe_id="probe_2")
        self.rewards = {True: 0.0, False: 0.0}
        event = await self.collect(request=second)
        self.assertEqual(event.with_skill.score - event.without_skill.score, 0.0)

    async def test_dev_and_test_tasks_cannot_generate_training_credit(self):
        for phase in ("dev", "test_seen", "test_unseen"):
            sample = copy.deepcopy(self.sample)
            sample["tools_kwargs"]["task"]["metadata"]["dataset_phase"] = phase
            with self.assertRaises(ValueError):
                await self.collect(sample_fields=sample)
        self.assertEqual(self.calls, [])

    async def test_unobserved_injection_does_not_become_zero_reward(self):
        self.missing_injection = True
        with self.assertRaises(probes.ProbeNotObserved):
            await self.collect()
        self.assertFalse((self.root / "credit_probes/probe_1/paired-event.json").exists())

    async def test_different_environment_initialization_does_not_settle(self):
        self.changed_state = True
        with self.assertRaises(probes.ProbeNotObserved):
            await self.collect()
        self.assertFalse((self.root / "credit_probes/probe_1/paired-event.json").exists())

    async def test_uncertain_native_branch_is_not_silently_resampled(self):
        self.fail_branch = True
        with self.assertRaises(RuntimeError):
            await self.collect()
        self.fail_branch = False
        with self.assertRaises(probes.ProbeNotObserved):
            await self.collect()
        self.assertEqual(len(self.calls), 1)

    async def test_library_changes_beyond_target_are_rejected_before_model_call(self):
        self.store.create_branch("extra", from_snapshot=self.treated.snapshot_id)
        extra = self.store.stage_candidate(
            store_module.SkillArtifact("other", "Other", "An unrelated procedure.", "Other tasks"), branch="extra",
        )
        self.store.admit(extra.candidate_id)
        snapshot = self.store.publish([extra.candidate_id], branch="extra", expected_parent=self.treated.snapshot_id)
        with self.assertRaises(ValueError):
            await self.collect(request=replace(self.request, with_snapshot_id=snapshot.snapshot_id))
        self.assertEqual(self.calls, [])

    async def test_moving_or_unattested_evaluator_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.collect(frozen_attestation={"policy_kind": "online_solver", "frozen_model_version": "frozen:v1"})
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
