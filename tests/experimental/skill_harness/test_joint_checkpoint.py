"""Stdlib crash/recovery tests: no Torch, Ray, network or external processes."""

import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_joint_checkpoint_offline_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
joint = importlib.import_module(PACKAGE + ".joint_checkpoint")
store_module = importlib.import_module(PACKAGE + ".store")
replay_module = importlib.import_module(PACKAGE + ".replay")
credit_module = importlib.import_module(PACKAGE + ".credit")


class JointCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.live = self.root / "live"
        self.store = store_module.SkillStore(self.live / "skills")
        self.replay = replay_module.CurateReplay(self.live / "replay")
        self.ledger = credit_module.CreditLedger(self.live / "ledger", self.replay)
        self.artifact = store_module.SkillArtifact(
            "skill", "Find objects", "Inspect plausible receptacles", "When locating"
        )
        candidate = self.store.stage_candidate(self.artifact)
        self.store.admit(candidate.candidate_id)
        self.store.publish([candidate.candidate_id], expected_parent=candidate.base_snapshot)
        self.store.create_branch("evaluation")
        self.pending = self.window("pending")
        self.uncertain = self.mature("uncertain")
        self.ledger.lease(self.uncertain.settlement_id, "uncertain-update", owner="learner", lease_seconds=1, now=40)
        self.applied = self.mature("applied")
        lease = self.ledger.lease(self.applied.settlement_id, "applied-update", owner="learner", now=40)
        self.ledger.mark_applied(self.applied.settlement_id, "applied-update", lease_token=lease["lease_token"],
                                 model_update_id="online:4")
        self.spec = joint.JointCheckpointSpec(7, 4, 2, "layout:fsdp2:2-ranks", {
            "model": "Qwen3.5-4B", "tokenizer": "tok:v1", "template": "t:v1",
        }, "eval:frozen-1", "protocol:fixed-1")
        self.controller = {"schema_version": 1, "optimizer_step": 4,
                           "protocol_fingerprint": self.spec.protocol_fingerprint,
                           "pending": {"producer-a": {"window": self.pending[1]}},
                           "prepared_update": {"update_id": "uncertain-update", "expected_before": 4,
                                               "expected_after": 5, "settlement_id": self.uncertain.settlement_id},
                           "applied_update_ids": {"applied-update": "online:4"}, "last_global_step": 7}
        self.evaluation = {"evaluator_fingerprint": self.spec.evaluator_fingerprint,
                           "protocol_fingerprint": self.spec.protocol_fingerprint,
                           "pinned_model": "eval:1", "probe_seed": 3}
        self.coordinator = joint.JointCheckpointCoordinator()
        self.inside_fence = False
        self.events = []

    def window(self, identity):
        record = replay_module.CurateReplayRecord(
            identity, "producer-" + identity, self.artifact.ref, (1, 2), (3, 4), (1, 1), (-0.2, -0.3),
            "ema:3", {"temperature": 1., "top_p": 1., "top_k": -1, "logprobs_mode": "processed_logprobs"},
            "tok:v1", "template:v1", {"context": "immutable"}, self.store.head(), "group-" + identity, 0, 1, 10,
        )
        self.replay.append(record)
        proto = credit_module.CreditProtocol("eval:1", "probe:1", "train:1", "env:1", "baseline:1", 1)
        window = "window-" + identity
        self.ledger.register_window(window, identity, proto, opened_at=11, deadline_at=1000)
        return record, window, proto

    def mature(self, identity):
        record, window, proto = self.window(identity)
        common = {"task_id": "probe-task", "model_version": proto.evaluator_version,
                  "environment_version": proto.environment_version, "environment_state_hash": "a" * 64,
                  "baseline_snapshot_id": proto.baseline_snapshot_id}
        event = credit_module.UsageEvent(
            "event-" + identity, window, identity, record.skill_ref, "probe-1", proto.protocol_id,
            credit_module.EvaluationOutcome(**common, score=1, trajectory_ref="with.json",
                                           loaded_skill_ref=record.skill_ref),
            credit_module.EvaluationOutcome(**common, score=0, trajectory_ref="without.json"), 20,
        )
        self.ledger.add_usage(event, now=30)
        return self.ledger.settle(window, now=30)

    @contextmanager
    def fence(self):
        self.assertFalse(self.inside_fence)
        self.inside_fence = True
        try:
            yield
        finally:
            self.inside_fence = False

    def online_saver(self, directory):
        self.assertTrue(self.inside_fence)
        self.assertFalse((directory.parent / joint.MANIFEST_NAME).exists())
        directory.mkdir(exist_ok=True)
        for rank in range(self.spec.world_size):
            for kind in ("model", "optim", "extra_state"):
                (directory / f"{kind}_world_size_{self.spec.world_size}_rank_{rank}.pt").write_bytes(
                    f"{kind}:{rank}:{self.spec.optimizer_step}".encode()
                )
        (directory.parent / "data.pt").write_bytes(b"dataloader-position-and-rng")
        # Model the native trainer's early pointer. Joint resume must ignore it.
        (directory.parent.parent / "latest_checkpointed_iteration.txt").write_text("999")
        self.events.append("online-complete")
        return joint.CompletedOnlineCheckpoint(self.spec.global_step, self.spec.optimizer_step,
                                                self.spec.world_size, self.spec.layout_fingerprint, "rpc-barrier:done")

    def ema_saver(self, directory):
        self.assertTrue(self.inside_fence)
        result = []
        for rank in range(self.spec.world_size):
            path = directory / f"ema_rank_{rank:05d}.pt"
            path.write_bytes(f"EMA:{rank}:{self.spec.optimizer_step}".encode())
            result.append({"path": str(path), "last_optimizer_step": self.spec.optimizer_step, "decay": 0.95})
        self.events.append("ema-complete")
        return result

    def save(self, name="global_step_7", **changes):
        arguments = {"spec": self.spec, "save_online": self.online_saver, "save_ema": self.ema_saver,
                     "skill_store": self.store, "replay": self.replay, "ledger": self.ledger,
                     "controller_state": self.controller, "evaluation_state": self.evaluation, "fence": self.fence}
        arguments.update(changes)
        return self.coordinator.save(self.root / "checkpoints" / name, **arguments)

    def restore(self, checkpoint, **changes):
        arguments = {"expected": self.spec.compatibility(), "destination_root": self.root / "restored",
                     "load_online": Mock(return_value=None), "load_ema": Mock(return_value=None),
                     "load_training_state": Mock(return_value=None),
                     "fence": self.fence}
        arguments.update(changes)
        return self.coordinator.restore(checkpoint.checkpoint_dir, **arguments)

    def test_complete_manifest_binds_all_components_and_separate_latest(self):
        saved = self.save()
        self.assertEqual(self.events, ["online-complete", "ema-complete"])
        self.assertEqual(saved.payload["state"], "complete")
        files = saved.payload["files"]
        for prefix in ("actor/optim_", "actor/model_", "actor/extra_state_", "joint_state/ema/",
                       "joint_state/skill_store/", "joint_state/replay/", "joint_state/ledger/"):
            self.assertTrue(any(name.startswith(prefix) for name in files), prefix)
        self.assertIn("data.pt", files)
        self.assertIn("joint_state/controller.json", files)
        latest = self.coordinator.latest(self.root / "checkpoints", expected=self.spec.compatibility())
        self.assertEqual(latest.manifest_id, saved.manifest_id)
        self.assertEqual((self.root / "checkpoints/latest_checkpointed_iteration.txt").read_text(), "999")
        self.assertEqual(latest.payload["spec"]["global_step"], 7)

    def test_portable_restore_preserves_pending_applied_and_uncertain_state(self):
        saved = self.save()
        original_controller = json.loads(json.dumps(self.controller))
        original_head = self.store.head()
        relocated = self.root / "relocated"
        shutil.copytree(saved.checkpoint_dir, relocated)
        shutil.rmtree(self.live)  # historical absolute export URIs no longer exist
        saved = joint.VerifiedJointCheckpoint(str(relocated), saved.manifest_id, saved.payload)
        online, ema = Mock(return_value=None), Mock(return_value=None)
        state = self.restore(saved, load_online=online, load_ema=ema)
        self.assertEqual(state.skill_store.head(), original_head)
        self.assertEqual(state.skill_store.head("evaluation"), original_head)
        self.assertEqual(state.replay.get("uncertain").producer_uid, "producer-uncertain")
        self.assertEqual(state.ledger.window_status(self.pending[1], now=100)["state"], "pending")
        self.assertEqual(state.ledger.training_status(self.uncertain.settlement_id, now=100)["state"],
                         "needs_reconciliation")
        self.assertEqual(state.ledger.training_status(self.applied.settlement_id, now=100)["state"], "applied")
        self.assertEqual(state.controller_state, original_controller)
        self.assertEqual(state.evaluation_state, self.evaluation)
        online.assert_called_once_with(relocated.resolve() / "actor", saved.payload["spec"])
        ema.assert_called_once_with(relocated.resolve() / "joint_state/ema", saved.payload["spec"])
        # Large immutable tensors and metadata stay at the original checkpoint.
        self.assertFalse((self.root / "restored/ema").exists())
        self.assertFalse((self.root / "restored/controller.json").exists())
        self.assertFalse((self.root / "restored/evaluation.json").exists())
        self.assertTrue((self.root / "restored/RESTORE_COMPLETE.json").is_file())

    def test_failure_before_ema_completion_never_publishes_latest_or_complete(self):
        previous = self.save()

        def broken_ema(directory):
            (directory / "ema_rank_00000.pt").write_bytes(b"partial")
            raise OSError("simulated disk failure")

        with self.assertRaisesRegex(OSError, "simulated"):
            self.save("global_step_7_incomplete", save_ema=broken_ema)
        partial = self.root / "checkpoints/global_step_7_incomplete"
        self.assertFalse((partial / joint.MANIFEST_NAME).exists())
        with self.assertRaises(joint.JointCheckpointError):
            self.coordinator.validate(partial, expected=self.spec.compatibility())
        latest = self.coordinator.latest(self.root / "checkpoints", expected=self.spec.compatibility())
        self.assertEqual(latest.manifest_id, previous.manifest_id)

    def test_crash_after_components_before_manifest_rejects_resume(self):
        original = joint._atomic_json

        def crash_manifest(path, value):
            if path.name == joint.MANIFEST_NAME:
                raise OSError("crash before COMPLETE publication")
            original(path, value)

        with patch.object(joint, "_atomic_json", side_effect=crash_manifest):
            with self.assertRaisesRegex(OSError, "COMPLETE"):
                self.save()
        root = self.root / "checkpoints/global_step_7"
        self.assertTrue((root / "joint_state").is_dir())
        self.assertFalse((root / joint.MANIFEST_NAME).exists())
        loader = Mock()
        with self.assertRaises(joint.JointCheckpointError):
            self.coordinator.restore(root, expected=self.spec.compatibility(),
                                     destination_root=self.root / "bad-restore",
                                     load_online=loader, load_ema=loader, fence=self.fence)
        with self.assertRaises(joint.JointCheckpointError):
            self.coordinator.latest(root.parent, expected=self.spec.compatibility())
        loader.assert_not_called()
        self.assertFalse((self.root / "bad-restore").exists())

    def test_crash_after_manifest_before_pointer_can_be_recovered_idempotently(self):
        with patch.object(self.coordinator, "_publish_latest", side_effect=OSError("pointer write failed")):
            with self.assertRaises(OSError):
                self.save()
        saved = self.coordinator.validate(self.root / "checkpoints/global_step_7", expected=self.spec.compatibility())
        with patch.object(self, "online_saver", side_effect=AssertionError("must not save again")):
            again = self.save()
        self.assertEqual(saved.manifest_id, again.manifest_id)
        latest = self.coordinator.latest(self.root / "checkpoints", expected=self.spec.compatibility())
        self.assertEqual(latest.manifest_id, saved.manifest_id)

    def test_digest_corruption_fails_before_any_live_actor_or_storage_load(self):
        saved = self.save()
        (Path(saved.checkpoint_dir) / "actor/optim_world_size_2_rank_0.pt").write_bytes(b"wrong optimizer")
        loader = Mock()
        with self.assertRaisesRegex(joint.JointCheckpointError, "digest"):
            self.restore(saved, load_online=loader, load_ema=loader)
        loader.assert_not_called()
        self.assertFalse((self.root / "restored").exists())

    def test_world_layout_runtime_evaluator_and_protocol_must_match(self):
        saved = self.save()
        for key, value in (("world_size", 4), ("layout_fingerprint", "other"),
                           ("versions", {**self.spec.compatibility()["versions"], "tokenizer": "other"}),
                           ("evaluator_fingerprint", "other"), ("protocol_fingerprint", "other")):
            with self.subTest(key=key), self.assertRaisesRegex(joint.JointCheckpointError, "mismatch"):
                self.coordinator.validate(saved.checkpoint_dir, expected={**self.spec.compatibility(), key: value})

    def test_state_identities_exclude_source_versions_and_require_token_interpretation(self):
        self.assertEqual(set(self.spec.versions), {"model", "tokenizer", "template"})
        for versions in ({**self.spec.versions, "code": "source-bytes"},
                         {key: value for key, value in self.spec.versions.items() if key != "tokenizer"}):
            with self.subTest(versions=versions), self.assertRaises(ValueError):
                replace(self.spec, versions=versions)

    def test_controller_state_schema_is_checked_before_publication(self):
        with self.assertRaisesRegex(joint.JointCheckpointError, "Controller"):
            self.save("bad-controller-schema", controller_state={**self.controller, "schema_version": 99})

    def test_async_checkpoint_requires_explicit_completion_waiter(self):
        def asynchronous(directory):
            return replace(self.online_saver(directory), async_save=True)

        with self.assertRaisesRegex(joint.JointCheckpointError, "completion waiter"):
            self.save("async-rejected", save_online=asynchronous)
        waiter = Mock(side_effect=lambda receipt: receipt)
        saved = self.save("async-completed", save_online=asynchronous, wait_online=waiter)
        waiter.assert_called_once()
        self.assertEqual(saved.payload["state"], "complete")

    def test_incomplete_model_optimizer_or_ema_clock_cannot_publish(self):
        def missing_optimizer(directory):
            receipt = self.online_saver(directory)
            (directory / "optim_world_size_2_rank_1.pt").unlink()
            return receipt

        with self.assertRaisesRegex(joint.JointCheckpointError, "model/optimizer/RNG"):
            self.save("missing-optimizer", save_online=missing_optimizer)

        def wrong_ema(directory):
            result = self.ema_saver(directory)
            result[1]["last_optimizer_step"] = 3
            return result

        with self.assertRaisesRegex(joint.JointCheckpointError, "EMA.*clock"):
            self.save("wrong-ema", save_ema=wrong_ema)

    def test_controller_or_evaluator_clock_mismatch_cannot_publish(self):
        with self.assertRaisesRegex(joint.JointCheckpointError, "Controller"):
            self.save("bad-controller", controller_state={**self.controller, "optimizer_step": 3})
        with self.assertRaisesRegex(joint.JointCheckpointError, "Evaluator"):
            self.save("bad-evaluator", evaluation_state={**self.evaluation, "evaluator_fingerprint": "other"})

    def test_restore_failure_does_not_mark_live_state_complete_or_reconcile_leases(self):
        saved = self.save()
        with self.assertRaisesRegex(RuntimeError, "EMA loader failed"):
            self.restore(saved, load_ema=Mock(side_effect=RuntimeError("EMA loader failed")))
        self.assertFalse((self.root / "restored/RESTORE_COMPLETE.json").exists())
        replay = replay_module.CurateReplay(self.root / "restored/replay")
        ledger = credit_module.CreditLedger(self.root / "restored/ledger", replay)
        self.assertEqual(ledger.training_status(self.uncertain.settlement_id, now=100)["state"], "needs_reconciliation")

    def test_unresolved_rpc_results_and_existing_live_restore_destinations_rejected(self):
        with self.assertRaisesRegex(joint.JointCheckpointError, "explicitly completed"):
            self.save("unresolved-online", save_online=lambda _directory: object())
        saved = self.save()
        with self.assertRaisesRegex(joint.JointCheckpointError, "dataloader/TQ"):
            self.restore(saved, load_training_state=None)
        self.assertFalse((self.root / "restored").exists())
        with self.assertRaisesRegex(joint.JointCheckpointError, "unresolved"):
            self.restore(saved, load_online=Mock(return_value=object()))
        with self.assertRaisesRegex(joint.JointCheckpointError, "fresh destination"):
            self.restore(saved)


if __name__ == "__main__":
    unittest.main()
