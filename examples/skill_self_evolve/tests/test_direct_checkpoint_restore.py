"""Small real-file CPU lifecycle tests; no Torch, Ray, model or live checkpoint."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import shutil
import sys
import threading
import types
import unittest
import uuid
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


fixture = module("_direct_checkpoint_fixture", ROOT / "tests/experimental/skill_harness/test_joint_checkpoint.py")
direct = module("_direct_checkpoint_adapter", ROOT / "examples/skill_self_evolve/scripts/direct_checkpoint_restore.py")
core = fixture.joint
DirectCoordinator = direct.coordinator_class(core)


class DirectReadTests(unittest.TestCase):
    setUp = fixture.JointCheckpointTests.setUp
    window = fixture.JointCheckpointTests.window
    mature = fixture.JointCheckpointTests.mature
    fence = fixture.JointCheckpointTests.fence
    online_saver = fixture.JointCheckpointTests.online_saver
    ema_saver = fixture.JointCheckpointTests.ema_saver
    save = fixture.JointCheckpointTests.save

    def arguments(self, destination="restored", **changes):
        value = {
            "expected": self.spec.compatibility(),
            "destination_root": self.root / destination,
            "load_online": Mock(return_value=None),
            "load_ema": Mock(return_value=None),
            "load_controller": Mock(return_value=None),
            "load_training_state": Mock(return_value=None),
            "fence": self.fence,
        }
        value.update(changes)
        return value

    def test_latest_then_restore_has_one_full_validation_and_zero_ema_copies(self):
        saved = self.save()
        coordinator = DirectCoordinator()
        original_validate = core.JointCheckpointCoordinator.validate
        original_copy = shutil.copyfile
        original_digest = core._file_digest
        with (
            patch.object(
                core.JointCheckpointCoordinator, "validate", autospec=True, side_effect=original_validate
            ) as validate,
            patch.object(core.shutil, "copyfile", wraps=original_copy) as copied,
            patch.object(core, "_file_digest", wraps=original_digest) as digested,
        ):
            selected = coordinator.latest(Path(saved.checkpoint_dir).parent, expected=self.spec.compatibility())
            state = coordinator.restore(selected.checkpoint_dir, **self.arguments())
        self.assertEqual(validate.call_count, 1)
        source = Path(saved.checkpoint_dir).resolve()
        hashed = Counter(
            str(Path(call.args[0]).resolve().relative_to(source))
            for call in digested.call_args_list
            if source in Path(call.args[0]).resolve().parents
        )
        self.assertEqual(hashed, Counter({name: 1 for name in saved.payload["files"]}))
        self.assertFalse((self.root / "restored/ema").exists())
        self.assertTrue(copied.call_args_list)
        for call in copied.call_args_list:
            relative = Path(call.args[0]).relative_to(Path(saved.checkpoint_dir) / "joint_state")
            self.assertIn(relative.parts[0], (*direct.MUTABLE_COMPONENTS, *direct.SMALL_STATE_FILES))
        self.assertEqual(state.checkpoint.manifest_id, saved.manifest_id)
        receipt = coordinator.last_restore_receipt
        self.assertEqual((receipt["full_validation_passes"], receipt["ema_files_copied"]), (1, 0))

    def test_direct_restore_alone_validates_once(self):
        saved = self.save()
        original = core.JointCheckpointCoordinator.validate
        with patch.object(core.JointCheckpointCoordinator, "validate", autospec=True, side_effect=original) as validate:
            DirectCoordinator().restore(saved.checkpoint_dir, **self.arguments())
        self.assertEqual(validate.call_count, 1)

    def test_original_source_paths_callback_order_fence_and_controller_identity(self):
        saved = self.save()
        coordinator, events = DirectCoordinator(), []

        def callback(name):
            def run(path, spec):
                self.assertTrue(self.inside_fence)
                events.append((name, path, spec))

            return run

        def controller(payload, *, expected_optimizer_step):
            self.assertTrue(self.inside_fence)
            self.assertEqual(payload, self.controller)
            self.assertEqual(expected_optimizer_step, 4)
            events.append(("controller", payload, expected_optimizer_step))

        coordinator.restore(
            saved.checkpoint_dir,
            **self.arguments(
                load_online=callback("online"),
                load_ema=callback("ema"),
                load_training_state=callback("training"),
                load_controller=controller,
            ),
        )
        root = Path(saved.checkpoint_dir)
        self.assertEqual([row[0] for row in events], ["online", "ema", "training", "controller"])
        self.assertEqual(events[0][1], root / "actor")
        self.assertEqual(events[1][1], root / "joint_state/ema")
        self.assertEqual(events[2][1], (root / "data.pt",))

    def test_mutable_state_independent_inodes_and_writes_preserve_source_bytes(self):
        saved = self.save()
        source = Path(saved.checkpoint_dir)
        before = {
            p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in core._tree_files(source)
        }
        state = DirectCoordinator().restore(saved.checkpoint_dir, **self.arguments())
        destination = self.root / "restored"
        for component in direct.MUTABLE_COMPONENTS:
            for path in core._tree_files(source / "joint_state" / component):
                target = destination / component / path.relative_to(source / "joint_state" / component)
                self.assertNotEqual(path.stat().st_ino, target.stat().st_ino)
        state.skill_store.create_branch("after_resume")
        (destination / "replay/new-working-state.json").write_text('{"local":true}')
        (destination / "ledger/new-working-state.json").write_text('{"local":true}')
        after = {
            p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in core._tree_files(source)
        }
        self.assertEqual(before, after)
        self.assertEqual(
            state.ledger.training_status(self.uncertain.settlement_id, now=100)["state"], "needs_reconciliation"
        )

    def test_unissued_forged_cross_coordinator_and_consumed_proofs_are_rejected(self):
        saved = self.save()
        owner, foreign = DirectCoordinator(), DirectCoordinator()
        selected = owner.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
        forged = core.VerifiedJointCheckpoint(
            selected.checkpoint_dir, selected.manifest_id, copy.deepcopy(selected.payload)
        )
        loader = Mock()
        for coordinator, value in ((owner, forged), (foreign, selected)):
            with self.assertRaises(core.JointCheckpointError):
                coordinator.restore_verified(value, **self.arguments(load_online=loader))
        loader.assert_not_called()
        owner.restore_verified(selected, **self.arguments())
        with self.assertRaises(core.JointCheckpointError):
            owner.restore_verified(selected, **self.arguments(destination="second"))

    def test_mutated_proof_payload_and_manifest_identity_are_rejected(self):
        for changed in ("payload", "manifest_id"):
            with self.subTest(changed=changed):
                saved = self.save("checkpoint_" + changed)
                coordinator = DirectCoordinator()
                selected = coordinator.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
                if changed == "payload":
                    selected.payload["spec"]["optimizer_step"] = 900
                else:
                    object.__setattr__(selected, "manifest_id", "forged")
                loader = Mock()
                with self.assertRaises(core.JointCheckpointError):
                    coordinator.restore_verified(selected, **self.arguments(destination=changed, load_online=loader))
                loader.assert_not_called()

    def test_changed_actor_ema_or_data_after_validation_fails_before_loaders(self):
        for relative in ("actor/optim_world_size_2_rank_0.pt", "joint_state/ema/ema_rank_00000.pt", "data.pt"):
            with self.subTest(relative=relative):
                saved = self.save("checkpoint_" + str(uuid.uuid4()))
                coordinator = DirectCoordinator()
                selected = coordinator.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
                path = Path(saved.checkpoint_dir) / relative
                path.write_bytes(path.read_bytes() + b"changed")
                loader = Mock()
                with self.assertRaises(core.JointCheckpointError):
                    coordinator.restore_verified(selected, **self.arguments(load_online=loader))
                loader.assert_not_called()
                self.assertFalse((self.root / "restored").exists())

    def test_changed_during_mutable_copy_fails_before_live_loaders(self):
        saved = self.save()
        coordinator = DirectCoordinator()
        selected = coordinator.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
        original = core._validate_storage

        def validate(destination, snapshots):
            original(destination, snapshots)
            if destination == (self.root / "restored").resolve():
                (Path(saved.checkpoint_dir) / "data.pt").write_bytes(b"changed during clone")

        loader = Mock()
        with (
            patch.object(core, "_validate_storage", side_effect=validate),
            self.assertRaises(core.JointCheckpointError),
        ):
            coordinator.restore_verified(selected, **self.arguments(load_online=loader))
        loader.assert_not_called()
        self.assertFalse((self.root / "restored/RESTORE_COMPLETE.json").exists())

    def test_added_deleted_and_symlink_source_entries_invalidate_proof(self):
        for change in ("added", "deleted", "symlink"):
            with self.subTest(change=change):
                saved = self.save("checkpoint_" + change)
                coordinator = DirectCoordinator()
                selected = coordinator.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
                source = Path(saved.checkpoint_dir)
                path = source / "actor/model_world_size_2_rank_0.pt"
                if change == "added":
                    (source / "actor/unlisted.pt").write_bytes(b"new")
                elif change == "deleted":
                    path.unlink()
                else:
                    path.unlink()
                    path.symlink_to(source / "data.pt")
                loader = Mock()
                with self.assertRaises(core.JointCheckpointError):
                    coordinator.restore_verified(selected, **self.arguments(load_online=loader))
                loader.assert_not_called()

    def test_corrupt_small_state_copy_is_rejected_before_callbacks(self):
        saved = self.save()
        original = shutil.copyfile

        def copy_file(source, target):
            result = original(source, target)
            if Path(target).name == "controller.json":
                Path(target).write_text("{}")
            return result

        loader = Mock()
        with patch.object(core.shutil, "copyfile", side_effect=copy_file), self.assertRaises(core.JointCheckpointError):
            DirectCoordinator().restore(saved.checkpoint_dir, **self.arguments(load_online=loader))
        loader.assert_not_called()

    def test_initial_corruption_rejects_before_destination_or_callback(self):
        saved = self.save()
        (Path(saved.checkpoint_dir) / "data.pt").write_bytes(b"corrupt")
        loader = Mock()
        with self.assertRaises(core.JointCheckpointError):
            DirectCoordinator().restore(saved.checkpoint_dir, **self.arguments(load_online=loader))
        loader.assert_not_called()
        self.assertFalse((self.root / "restored").exists())

    def test_source_changed_after_its_hash_during_validation_cannot_receive_proof(self):
        saved = self.save()
        path = Path(saved.checkpoint_dir).resolve() / "actor/model_world_size_2_rank_0.pt"
        original = core._file_digest

        def digest(file, **kwargs):
            result = original(file, **kwargs)
            if Path(file).resolve() == path:
                path.write_bytes(path.read_bytes() + b"changed after hashing")
            return result

        coordinator = DirectCoordinator()
        with patch.object(core, "_file_digest", side_effect=digest), self.assertRaises(core.JointCheckpointError):
            coordinator.validate(saved.checkpoint_dir, expected=self.spec.compatibility())
        self.assertEqual(coordinator._proofs, {})

    def test_unresolved_or_failed_callbacks_do_not_publish_restore_complete(self):
        for mode in ("awaitable", "failed"):
            with self.subTest(mode=mode):
                saved = self.save("checkpoint_" + mode)
                coordinator = DirectCoordinator()

                class Unresolved:
                    def __await__(self):
                        yield

                callback = (
                    Mock(return_value=Unresolved())
                    if mode == "awaitable"
                    else Mock(side_effect=RuntimeError("RPC failed"))
                )
                with self.assertRaises((core.JointCheckpointError, RuntimeError)):
                    coordinator.restore(saved.checkpoint_dir, **self.arguments(destination=mode, load_ema=callback))
                self.assertFalse((self.root / mode / "RESTORE_COMPLETE.json").exists())
                self.assertIsNone(coordinator.last_restore_receipt)

    def test_frozen_trainer_method_uses_adapter_without_copying_its_body(self):
        saved = self.save()
        tree = ast.parse((ROOT / "verl/experimental/skill_harness/joint_trainer.py").read_text())
        trainer = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkillJointSyncTrainer"
        )
        method = next(
            node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_load_checkpoint"
        )
        scope = {
            "Path": Path,
            "uuid": uuid,
            "JointCheckpointError": core.JointCheckpointError,
            "open_dict": lambda _: nullcontext(),
            "torch": types.SimpleNamespace(load=Mock(return_value={"cursor": 128})),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual frozen trainer method>", "exec"), scope)
        Frozen = type("FrozenTrainerMethod", (), {"_load_checkpoint": scope["_load_checkpoint"]})
        Combined = type("Combined", (direct.DirectReadRestoreMixin, Frozen), {})
        instance = Combined()
        instance.config = types.SimpleNamespace(
            trainer=types.SimpleNamespace(resume_mode="auto", default_local_dir=str(Path(saved.checkpoint_dir).parent)),
            skill_harness=types.SimpleNamespace(artifact_root=str(self.root), ema=types.SimpleNamespace(decay=0.95)),
        )
        instance._compatibility = self.spec.compatibility
        instance._await = lambda value: value
        instance._joint_lock = threading.RLock()
        instance.actor_rollout_wg = types.SimpleNamespace(
            load_checkpoint=Mock(return_value=None), load_ema=Mock(return_value=None)
        )
        instance.train_dataloader = types.SimpleNamespace(load_state_dict=Mock())
        original = core.JointCheckpointCoordinator.validate
        with (
            patch.object(direct, "coordinator_class", return_value=DirectCoordinator),
            patch.object(core.JointCheckpointCoordinator, "validate", autospec=True, side_effect=original) as validate,
        ):
            instance._load_checkpoint()
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(instance.global_steps, self.spec.global_step)
        self.assertEqual(instance._joint_deferred_controller, self.controller)
        instance.actor_rollout_wg.load_ema.assert_called_once_with(
            str(Path(saved.checkpoint_dir) / "joint_state/ema"), expected_decay=0.95, shadow_device="cpu"
        )
        self.assertFalse((instance._joint_restored.skill_store.root.parent / "ema").exists())


if __name__ == "__main__":
    unittest.main()
