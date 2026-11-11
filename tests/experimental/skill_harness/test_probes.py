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
            evaluator_version="b" * 64, probe_protocol_version="paired:v1",
            task_distribution_version="original_train:v1", environment_version="alfworld:v1",
            baseline_snapshot_id=self.base, required_exposures=1,
        )
        self.request = probes.ProbeRequest(
            "probe_1", "window_1", "curate_1", artifact.ref,
            self.treated.snapshot_id, self.base, "b" * 64, self.protocol, 3,
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
            "schema_version": 1, "policy_kind": "frozen_evaluator", "frozen_model_version": "b" * 64,
            "model_fingerprint": "b" * 64, "model_path": "/models/frozen", "lifecycle_id": "route_1",
        }
        self.manager = types.SimpleNamespace(skill_route_attestation=copy.deepcopy(self.attestation))
        self.calls = []
        self.rewards = {True: 1.0, False: 0.0}
        self.changed_state = False
        self.fail_branch = False
        self.missing_injection = False
        self.fail_after = None
        self.clock_by_loaded = {True: None, False: None}
        self.mixed_clock = False
        self.capture_lifecycle = None

    def probe_root(self, request=None):
        return self.root / "credit_probes" / probes.probe_physical_id(request or self.request)

    async def branch(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_branch or (self.fail_after is not None and len(self.calls) > self.fail_after):
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
        attestation = copy.deepcopy(kwargs["frozen_attestation"])
        if self.capture_lifecycle is not None:
            attestation["lifecycle_id"] = self.capture_lifecycle
        clock = self.clock_by_loaded[loaded]
        extra = {} if clock is None else {"min_global_steps": clock, "max_global_steps": clock}
        if self.mixed_clock:
            extra = {"min_global_steps": 3, "max_global_steps": 4}
        return {
            "reward": reward, "accuracy": reward, "finished": True,
            "policy_attestation": attestation,
            "trajectories": [{"prompt_ids": [1, 2], "response_ids": [3], "extra_fields": extra}],
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
            gateway_manager=self.manager, artifact_root=self.root, task_config_path="task_config.yaml",
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
        self.assertFalse((self.probe_root() / "paired-event.json").exists())

    async def test_different_environment_initialization_does_not_settle(self):
        self.changed_state = True
        with self.assertRaises(probes.ProbeNotObserved):
            await self.collect()
        self.assertFalse((self.probe_root() / "paired-event.json").exists())

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

    async def test_malformed_or_misbound_attestation_fails_before_generation(self):
        for field, value in (
            ("schema_version", True), ("model_fingerprint", "c" * 64),
            ("model_path", "relative/model"), ("lifecycle_id", None),
        ):
            bad = {**self.attestation, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                await self.collect(frozen_attestation=bad)
        self.manager.skill_route_attestation["lifecycle_id"] = "different_gateway"
        with self.assertRaisesRegex(ValueError, "Gateway manager"):
            await self.collect()
        self.assertEqual(self.calls, [])

    async def test_captured_attestation_must_match_bound_private_lifecycle(self):
        self.capture_lifecycle = "another_lifecycle"
        with self.assertRaisesRegex(ValueError, "another frozen"):
            await self.collect()
        self.assertFalse((self.probe_root() / "paired-event.json").exists())

    async def test_all_none_and_matching_observed_clocks_are_valid(self):
        await self.collect()
        self.request = replace(self.request, probe_id="observed_clock")
        self.clock_by_loaded = {True: 7, False: 7}
        await self.collect()

    async def test_mixed_or_changed_native_weight_clocks_are_rejected(self):
        self.mixed_clock = True
        with self.assertRaisesRegex(ValueError, "observed weight clock"):
            await self.collect()
        self.mixed_clock = False
        self.request = replace(self.request, probe_id="changed_clock")
        self.clock_by_loaded = {True: 7, False: 8}
        with self.assertRaisesRegex(ValueError, "observed weight clock"):
            await self.collect()
        self.request = replace(self.request, probe_id="missing_one_clock")
        self.clock_by_loaded = {True: None, False: 8}
        with self.assertRaises(ValueError):
            await self.collect()

    async def test_partial_pair_cannot_be_finished_under_new_server_lifecycle(self):
        self.fail_after = 1
        with self.assertRaises(RuntimeError):
            await self.collect()
        self.attestation["lifecycle_id"] = "restarted_route"
        self.manager.skill_route_attestation = copy.deepcopy(self.attestation)
        self.fail_after = None
        with self.assertRaisesRegex(probes.ProbeNotObserved, "old-lifecycle"):
            await self.collect()
        self.assertEqual(len(self.calls), 2)

    async def test_complete_pair_can_be_revalidated_after_lifecycle_restart_without_resampling(self):
        first = await self.collect()
        self.attestation["lifecycle_id"] = "restarted_route"
        self.manager.skill_route_attestation = copy.deepcopy(self.attestation)
        retry = await self.collect()
        self.assertEqual(first.to_dict(), retry.to_dict())
        self.assertEqual(len(self.calls), 2)

    async def test_candidate_windows_share_logical_probe_but_not_physical_artifacts(self):
        first_request = self.request
        first = await self.collect()
        self.store.create_branch("candidate_two", from_snapshot=self.base)
        artifact = store_module.SkillArtifact(
            "second_skill", "Another heat procedure", "Check the object after heating.",
            "When a task requests a heated object", ("pick_heat_then_place_in_recep",),
        )
        candidate = self.store.stage_candidate(artifact, branch="candidate_two")
        self.store.admit(candidate.candidate_id)
        snapshot = self.store.publish([candidate.candidate_id], branch="candidate_two", expected_parent=self.base)
        self.request = replace(
            first_request, window_id="window_two", curate_record_id="curate_two", skill_ref=artifact.ref,
            with_snapshot_id=snapshot.snapshot_id,
        )
        second = await self.collect()
        self.assertEqual(first.probe_id, second.probe_id)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertNotEqual(self.probe_root(first_request), self.probe_root(self.request))
        self.assertEqual(len({call["session_id"] for call in self.calls}), 4)
        self.assertEqual(len(list((self.root / "credit_probes").glob("*/paired-event.json"))), 2)

    async def test_gateway_factory_binds_validated_client_and_disables_request_sampling_overrides(self):
        client = types.SimpleNamespace(frozen_version="b" * 64, attestation=copy.deepcopy(self.attestation))
        route = types.SimpleNamespace(
            frozen_version="b" * 64, attestation=copy.deepcopy(self.attestation), get_client=lambda: client,
        )
        config = {"actor_rollout_ref": {"rollout": {"custom": {"agent_framework": {
            "allowed_request_sampling_param_keys": ["temperature", "top_p"],
        }}}}}
        calls = []

        def factory(*, config, llm_client):
            calls.append((config, llm_client))
            return types.SimpleNamespace()

        manager = await probes.create_frozen_probe_gateway(config, route, gateway_factory=factory)
        self.assertIs(calls[0][1], client)
        self.assertEqual(
            calls[0][0]["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]
            ["allowed_request_sampling_param_keys"], [],
        )
        self.assertEqual(config["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]
                         ["allowed_request_sampling_param_keys"], ["temperature", "top_p"])
        self.assertEqual(manager.skill_route_attestation, self.attestation)
        client.attestation["lifecycle_id"] = "wrong_client_lifecycle"
        with self.assertRaises(ValueError):
            await probes.create_frozen_probe_gateway(config, route, gateway_factory=factory)
        self.assertEqual(len(calls), 1)

    async def test_native_branch_refuses_unbound_manager_before_loading_task_runtime(self):
        with self.assertRaises(ValueError):
            await probes.native_probe_branch(
                gateway_manager=types.SimpleNamespace(), session_id="bad", sample_fields=self.sample,
                artifact_dir=self.root / "bad", task_config_path="unused.yaml", frozen_attestation=self.attestation,
            )


if __name__ == "__main__":
    unittest.main()
