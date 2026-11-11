"""Offline persistence, causality and delayed-credit contracts; no training stack."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_credit_replay_offline_contract"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(ROOT)]
    sys.modules[PACKAGE] = package
replay = importlib.import_module(PACKAGE + ".replay")
credit = importlib.import_module(PACKAGE + ".credit")


def skill_ref(name):
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {"skill_id": name, "version": "sha256:" + digest, "content_hash": digest}


def record(identity="curate_0", *, group=None, index=0, size=1, trajectory_index=0, trajectory_count=1):
    return replay.CurateReplayRecord(
        record_id=identity,
        producer_uid="producer_" + (group or identity),
        skill_ref=skill_ref("skill_" + (group or identity) + "_" + str(index)),
        prompt_ids=(1, 2, 3),
        action_ids=(4, 5, 6),
        curate_mask=(1, 0, 1),
        behavior_logprobs=(-0.25, 0.0, -0.75),
        behavior_policy_version="ema:3",
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs_mode": "processed_logprobs"},
        tokenizer_version="tokenizer:v1",
        template_version="template:v1",
        context_snapshot={"snapshot_id": "context:v1", "loaded_skills": []},
        skill_snapshot_id="skills:v1",
        candidate_group_id=group or ("group_" + identity),
        candidate_index=index,
        candidate_group_size=size,
        created_at=10.0,
        trajectory_index=trajectory_index,
        trajectory_count=trajectory_count,
    )


def protocol(exposures=2, **kwargs):
    return credit.CreditProtocol(
        evaluator_version="evaluator:7",
        probe_protocol_version="paired:v1",
        task_distribution_version="train-probes:v1",
        environment_version="alfworld:v1",
        baseline_snapshot_id="skills:base",
        required_exposures=exposures,
        **kwargs,
    )


def usage(item, proto, window, probe, *, positive=1.0, control=0.0, event_id=None, occurred_at=20.0, task=None):
    task_id = task or ("task:" + probe)
    state_hash = hashlib.sha256(task_id.encode()).hexdigest()
    common = dict(
        task_id=task_id,
        model_version=proto.evaluator_version,
        environment_version=proto.environment_version,
        environment_state_hash=state_hash,
        baseline_snapshot_id=proto.baseline_snapshot_id,
    )
    return credit.UsageEvent(
        event_id=event_id or (window + "_" + probe),
        window_id=window,
        curate_record_id=item.record_id,
        skill_ref=item.skill_ref,
        probe_id=probe,
        protocol_id=proto.protocol_id,
        with_skill=credit.EvaluationOutcome(
            **common, score=positive, trajectory_ref="/traces/with/" + probe, loaded_skill_ref=item.skill_ref
        ),
        without_skill=credit.EvaluationOutcome(**common, score=control, trajectory_ref="/traces/control/" + probe),
        occurred_at=occurred_at,
        live_solver_version="online:99",
        observed_live_score=0.5,
    )


class CreditFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.replay = replay.CurateReplay(self.root / "replay")
        self.ledger = credit.CreditLedger(self.root / "ledger", self.replay)

    def register(self, identity="curate_0", *, proto=None, cost=0.0, deadline=100.0):
        item = record(identity)
        self.replay.append(item)
        proto = proto or protocol()
        window = "window_" + identity
        self.ledger.register_window(
            window, item.record_id, proto, opened_at=11.0, deadline_at=deadline,
            skill_cost=cost, expected_probe_ids=tuple("p" + str(i) for i in range(proto.required_exposures)),
        )
        return item, proto, window

    def mature(self, identity="curate_0", *, positive=1.0, control=0.0, proto=None, cost=0.0):
        item, proto, window = self.register(identity, proto=proto, cost=cost)
        for index in range(proto.required_exposures):
            self.ledger.add_usage(
                usage(item, proto, window, "p" + str(index), positive=positive, control=control), now=30.0
            )
        return self.ledger.settle(window, now=30.0)


class TestCurateReplay(CreditFixture):
    def test_store_skill_ref_is_accepted_without_a_runtime_dependency(self):
        store = importlib.import_module(PACKAGE + ".store")
        artifact = store.SkillArtifact(
            skill_id="clean_skill", title="Clean an object", principle="Take, clean, then place.",
            when_to_apply="A cleaning task", task_families=("clean",),
        )
        item = replace(record(), skill_ref=artifact.ref)
        self.replay.append(item)
        self.assertEqual(dict(self.replay.get(item.record_id).skill_ref), artifact.ref.to_dict())

    def test_record_is_frozen_and_validates_tokens_probabilities_and_provenance(self):
        item = record()
        with self.assertRaises(TypeError):
            item.sampling_config["temperature"] = 2.0
        with self.assertRaises(TypeError):
            item.context_snapshot["snapshot_id"] = "changed"
        invalid = (
            {"behavior_logprobs": (-0.1,)},
            {"behavior_logprobs": (0.1, 0.0, -0.1)},
            {"behavior_logprobs": (float("nan"), 0.0, -0.1)},
            {"behavior_logprobs": (-0.1, -0.1, -0.1)},
            {"prompt_ids": (True, 2)},
            {"curate_mask": (0, 0, 0)},
            {"candidate_index": 2},
            {"trajectory_index": 1},
            {"behavior_policy_kind": "online"},
            {"tokenizer_version": ""},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(item, **values)
        with self.assertRaises(ValueError):
            replace(item, sampling_config={**dict(item.sampling_config), "logprobs_mode": "raw_logprobs"})

    def test_append_is_atomic_idempotent_and_conflict_checked_under_concurrency(self):
        item = record()
        with ThreadPoolExecutor(max_workers=6) as pool:
            refs = list(pool.map(lambda _: self.replay.append(item), range(20)))
        self.assertEqual(len({r.payload_hash for r in refs}), 1)
        self.assertEqual(self.replay.get(item.record_id).to_dict(), item.to_dict())
        self.assertEqual(self.replay.export_snapshot().item_count, 1)
        with self.assertRaises(replay.RecordConflict):
            self.replay.append(replace(item, behavior_logprobs=(-0.5, 0.0, -0.75)))
        with self.assertRaises(replay.RecordConflict):
            self.replay.append(replace(item, record_id="new_id_for_same_action"))

    def test_crash_before_causal_commit_is_not_acknowledged_and_retry_repairs_it(self):
        item = record()
        publish = replay._publish_immutable

        def interrupt(path, payload):
            if path.parent.name == "provenance":
                raise OSError("simulated process crash before causal index commit")
            return publish(path, payload)

        with mock.patch.object(replay, "_publish_immutable", side_effect=interrupt):
            with self.assertRaises(OSError):
                self.replay.append(item)
        with self.assertRaises(replay.IntegrityError):
            self.replay.get(item.record_id)
        self.assertEqual(self.replay.export_snapshot().item_count, 0)
        self.replay.append(item)
        self.assertEqual(self.replay.export_snapshot().item_count, 1)

    def test_multisegment_producer_must_be_complete_before_credit_registration(self):
        first = record("first_segment", group="one_candidate", trajectory_count=2)
        second = replace(
            first, record_id="second_segment", trajectory_index=1, prompt_ids=(1, 2, 3, 4), created_at=12.0
        )
        self.replay.append(first)
        with self.assertRaises(replay.IntegrityError):
            self.ledger.register_window("window", first.record_id, protocol(), opened_at=13, deadline_at=100)
        self.replay.append(second)
        self.assertEqual([r.record_id for r in self.replay.get_candidate_records(first.record_id)],
                         ["first_segment", "second_segment"])
        window = self.ledger.register_window("window", first.record_id, protocol(), opened_at=13, deadline_at=100)
        self.assertEqual(len(window["replay_records"]), 2)

    def test_snapshot_restore_retains_original_behavior_and_rejects_tampering(self):
        item = record()
        original_ref = self.replay.append(item)
        snapshot = self.replay.export_snapshot()
        other = replay.CurateReplay(self.root / "restored_replay")
        restored = other.restore_snapshot(snapshot)
        self.assertEqual(restored[0].payload_hash, original_ref.payload_hash)
        self.assertEqual(other.get(item.record_id).behavior_logprobs, item.behavior_logprobs)
        path = Path(original_ref.payload_uri)
        value = json.loads(path.read_text())
        value["payload"]["action_ids"][0] = 999
        path.write_text(json.dumps(value))
        with self.assertRaises(replay.IntegrityError):
            self.replay.get(item.record_id)


class TestCreditLedger(CreditFixture):
    def test_pending_censored_and_observed_zero_have_distinct_semantics(self):
        item, proto, window = self.register(deadline=25.0)
        self.assertEqual(self.ledger.window_status(window, now=15)["state"], "pending")
        self.assertEqual(self.ledger.window_status(window, now=30)["state"], "censored")
        with self.assertRaises(credit.WindowNotMature):
            self.ledger.settle(window, now=30)
        # Feedback arriving after the deadline is valid when the use event itself
        # happened inside the attribution window. Missing exposure is never zero.
        self.ledger.add_usage(usage(item, proto, window, "p0", positive=0, control=0, occurred_at=20), now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["state"], "censored")
        self.ledger.add_usage(usage(item, proto, window, "p1", positive=0, control=0, occurred_at=21), now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["state"], "matured")
        label = self.ledger.settle(window, now=30)
        self.assertEqual(label.utility, 0.0)
        self.assertEqual(self.ledger.sample_ready(1)[0].settlement_id, label.settlement_id)

    def test_pairing_protocol_causality_and_duplicate_exposure_are_checked(self):
        item, proto, window = self.register()
        event = usage(item, proto, window, "p0")
        invalid = (
            replace(event, protocol_id="wrong_protocol"),
            replace(event, occurred_at=9.0),
            replace(event, occurred_at=101.0),
            replace(event, without_skill=replace(event.without_skill, model_version="wrong:model")),
            replace(event, without_skill=replace(event.without_skill, task_id="wrong:task")),
            replace(event, without_skill=replace(event.without_skill, environment_state_hash="1" * 64)),
            replace(event, with_skill=replace(event.with_skill, baseline_snapshot_id="wrong:baseline")),
            replace(event, without_skill=replace(event.without_skill, loaded_skill_ref=item.skill_ref)),
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate), self.assertRaises(credit.CreditError):
                self.ledger.add_usage(candidate, now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["observed_exposures"], 0)
        self.ledger.add_usage(event, now=30)
        self.ledger.add_usage(event, now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["observed_exposures"], 1)
        with self.assertRaises(replay.RecordConflict):
            self.ledger.add_usage(replace(event, with_skill=replace(event.with_skill, score=0)), now=30)
        with self.assertRaises(credit.CreditError):
            self.ledger.add_usage(replace(event, event_id="duplicate_probe_new_event_id"), now=30)

    def test_crash_after_event_publish_does_not_double_count_on_retry(self):
        item, proto, window = self.register()
        event = usage(item, proto, window, "p0")
        with mock.patch.object(self.ledger, "_save_state", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                self.ledger.add_usage(event, now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["observed_exposures"], 0)
        self.ledger.add_usage(event, now=30)
        self.ledger.add_usage(event, now=30)
        self.assertEqual(self.ledger.window_status(window, now=30)["observed_exposures"], 1)

    def test_candidates_in_one_cohort_cannot_use_different_probe_tasks(self):
        proto = protocol()
        first = record("candidate0", group="same_group", index=0, size=2)
        second = record("candidate1", group="same_group", index=1, size=2)
        for i, item in enumerate((first, second)):
            self.replay.append(item)
            self.ledger.register_window(
                "window" + str(i), item.record_id, proto, opened_at=11, deadline_at=100,
                expected_probe_ids=("p0", "p1"),
            )
        self.ledger.add_usage(usage(first, proto, "window0", "p0"), now=30)
        with self.assertRaises(credit.CreditError):
            self.ledger.add_usage(usage(second, proto, "window1", "p0", task="different:task"), now=30)

    def test_labels_preserve_signed_utility_cost_and_immutable_versions(self):
        labels = [
            self.mature("positive", positive=1, control=0),
            self.mature("zero", positive=0.5, control=0.5),
            self.mature("negative", positive=0, control=1),
        ]
        self.assertEqual({label.utility for label in labels}, {-1.0, 0.0, 1.0})
        self.assertEqual(self.ledger.sample_ready(3, seed=17), self.ledger.sample_ready(3, seed=17))
        self.assertEqual({label.utility for label in self.ledger.sample_ready(3, seed=19)}, {-1.0, 0.0, 1.0})
        label = self.mature("costed", positive=0.75, control=0.25, proto=protocol(cost_weight=0.2), cost=0.5)
        self.assertAlmostEqual(label.raw_utility, 0.5)
        self.assertAlmostEqual(label.utility, 0.4)
        self.assertEqual(self.ledger.settle(label.window_id, now=31), label)
        item = self.replay.get(label.curate_record_id)
        with self.assertRaises(credit.CreditError):
            self.ledger.add_usage(usage(item, protocol(cost_weight=0.2), label.window_id, "p2"), now=31)

    def test_expired_optimizer_lease_is_not_silently_replayed(self):
        label = self.mature()
        lease = self.ledger.lease(label.settlement_id, "update_1", owner="learner", lease_seconds=5, now=30)
        self.assertEqual(self.ledger.sample_ready(10), [])
        self.assertEqual(self.ledger.training_status(label.settlement_id, now=36)["state"], "needs_reconciliation")
        with self.assertRaises(credit.LeaseConflict):
            self.ledger.lease(label.settlement_id, "update_2", owner="replacement", now=36)
        # The coordinator has checked a model-applied ID or restored its joint
        # checkpoint. CAS/timeout itself is deliberately not this evidence.
        self.ledger.reconcile_update(label.settlement_id, "update_1", was_applied=True, model_update_id="model:42")
        self.assertEqual(self.ledger.training_status(label.settlement_id, now=36)["state"], "applied")
        self.ledger.mark_applied(
            label.settlement_id, "update_1", lease_token=lease["lease_token"], model_update_id="model:42"
        )
        self.assertEqual(self.ledger.lease(label.settlement_id, "update_1", owner="learner", now=37)["state"],
                         "already_applied")

    def test_explicit_update_budget_allows_planned_gradient_reuse_not_new_reward(self):
        label = self.mature(proto=protocol(max_updates=2))
        for i in (1, 2):
            lease = self.ledger.lease(label.settlement_id, "update_" + str(i), owner="learner", now=30)
            self.ledger.mark_applied(
                label.settlement_id, "update_" + str(i), lease_token=lease["lease_token"],
                model_update_id="model:" + str(i),
            )
        self.assertEqual(self.ledger.sample_ready(10), [])
        self.assertEqual(self.ledger.get_settlement(label.settlement_id).utility, 1.0)
        with self.assertRaises(credit.LeaseConflict):
            self.ledger.lease(label.settlement_id, "update_3", owner="learner", now=31)

    def test_joint_snapshot_restores_pending_ready_applied_and_uncertain_states(self):
        pending, _, pending_window = self.register("pending")
        ready = self.mature("ready", positive=0, control=1)
        uncertain = self.mature("uncertain", positive=1, control=0)
        self.ledger.lease(uncertain.settlement_id, "uncertain_update", owner="learner", now=30, lease_seconds=5)
        applied = self.mature("applied", positive=0.5, control=0.5)
        lease = self.ledger.lease(applied.settlement_id, "done_update", owner="learner", now=30)
        self.ledger.mark_applied(
            applied.settlement_id, "done_update", lease_token=lease["lease_token"], model_update_id="model:checkpoint"
        )
        snapshot = self.ledger.export_snapshot()
        restored_replay = replay.CurateReplay(self.root / "restored_replay")
        restored = credit.CreditLedger(self.root / "restored_ledger", restored_replay)
        restored.restore_snapshot(snapshot)
        self.assertEqual(restored_replay.get(pending.record_id).to_dict(), pending.to_dict())
        self.assertEqual(restored.window_status(pending_window, now=20)["state"], "pending")
        self.assertEqual(restored.training_status(uncertain.settlement_id, now=36)["state"], "needs_reconciliation")
        self.assertEqual(restored.training_status(applied.settlement_id, now=36)["state"], "applied")
        self.assertEqual([label.settlement_id for label in restored.sample_ready(10)], [ready.settlement_id])
        with self.assertRaises(credit.CreditError):
            restored.restore_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
