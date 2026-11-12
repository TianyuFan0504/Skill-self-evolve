"""Read immutable model/EMA checkpoint files directly and clone mutable state only.

The frozen trainer already selects and validates a COMPLETE checkpoint before
calling restore. This coordinator reuses that single-use validation proof;
it does not hash the model/optimizer/EMA files a second time or copy EMA files.
The source checkpoint must remain immutable throughout startup, as with the
original restore callback contract.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

POLICY = "single-validation-direct-checkpoint-read-v1"
MUTABLE_COMPONENTS = ("skill_store", "replay", "ledger")
SMALL_STATE_FILES = ("controller.json", "evaluation.json")


@dataclass(frozen=True)
class _Proof:
    checkpoint: object
    expected: bytes
    payload: bytes
    identity: tuple
    source_guard: tuple


def _stat_identity(path, *, directory=False):
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode):
        raise ValueError("Checkpoint source cannot contain symbolic links")
    common = (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)
    # Lock directories/files are excluded by the frozen inventory. Their
    # creation may change a parent's mtime without changing checkpoint data.
    return common if directory else common + (value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def coordinator_class(core=None):
    """Return the adapter class; dependency injection keeps its CPU tests pure."""
    core = core or importlib.import_module("verl.experimental.skill_harness.joint_checkpoint")

    class DirectReadCheckpointCoordinator(core.JointCheckpointCoordinator):
        def __init__(self):
            self._proofs = {}
            self._selected = {}
            self.last_restore_receipt = None

        def _guard(self, root):
            root = Path(root).resolve()
            manifest = root / core.MANIFEST_NAME
            raw = manifest.read_bytes()
            payload = json.loads(raw)["payload"]
            entries = []
            for component in ("actor", "joint_state", *payload["training_state_paths"]):
                path = root / str(core._relative(component))
                if path.is_dir():
                    entries.append((path.relative_to(root).as_posix(), _stat_identity(path, directory=True)))
                    for child in sorted(path.rglob("*")):
                        relative = child.relative_to(root)
                        if "locks" in relative.parts or child.name == ".writer.lock":
                            continue
                        if child.is_symlink() or not (child.is_file() or child.is_dir()):
                            raise core.JointCheckpointError("Unsafe checkpoint source inventory")
                        entries.append((relative.as_posix(), _stat_identity(child, directory=child.is_dir())))
                else:
                    entries.append((path.relative_to(root).as_posix(), _stat_identity(path)))
            return (hashlib.sha256(raw).hexdigest(), _stat_identity(manifest), tuple(sorted(entries)))

        def _checked_guard(self, root):
            try:
                return self._guard(root)
            except (OSError, ValueError, KeyError) as error:
                raise core.JointCheckpointError("Checkpoint source changed or is malformed") from error

        def validate(self, checkpoint_dir, *, expected):
            root = Path(checkpoint_dir).resolve()
            try:
                before = self._checked_guard(root)
                checkpoint = super().validate(root, expected=expected)
                after = self._checked_guard(root)
            except (OSError, ValueError, KeyError) as error:
                raise core.JointCheckpointError("Checkpoint source changed or is malformed") from error
            if before != after:
                raise core.JointCheckpointError("Checkpoint source changed during full validation")
            proof = _Proof(
                checkpoint,
                core._canonical(expected),
                core._canonical(checkpoint.payload),
                (checkpoint.checkpoint_dir, checkpoint.manifest_id),
                after,
            )
            self._proofs[id(checkpoint)] = proof
            self._selected[str(root)] = checkpoint
            return checkpoint

        def restore(self, checkpoint_dir, *, expected, **kwargs):
            # The original trainer's latest()/validate() call issued this proof.
            # A direct caller still gets exactly one full validation here.
            root = str(Path(checkpoint_dir).resolve())
            checkpoint = self._selected.get(root)
            if checkpoint is None:
                checkpoint = self.validate(root, expected=expected)
            return self.restore_verified(checkpoint, expected=expected, **kwargs)

        def restore_verified(
            self,
            checkpoint,
            *,
            expected,
            destination_root,
            load_online,
            load_ema,
            load_controller,
            fence,
            load_training_state=None,
        ):
            """Consume only this coordinator's intact proof; never trust a public dataclass alone."""
            with fence():
                proof = self._proofs.pop(id(checkpoint), None)
                if proof is None or proof.checkpoint is not checkpoint:
                    raise core.JointCheckpointError("Restore requires this coordinator's unused validation proof")
                root = Path(checkpoint.checkpoint_dir).resolve()
                if self._selected.get(str(root)) is checkpoint:
                    self._selected.pop(str(root))
                if (
                    core._canonical(expected) != proof.expected
                    or core._canonical(checkpoint.payload) != proof.payload
                    or (checkpoint.checkpoint_dir, checkpoint.manifest_id) != proof.identity
                    or self._checked_guard(root) != proof.source_guard
                ):
                    raise core.JointCheckpointError("Verified checkpoint source, payload or compatibility changed")
                if checkpoint.payload["training_state_paths"] and load_training_state is None:
                    raise core.JointCheckpointError("A completed dataloader/TQ restore callback is required")
                destination = Path(destination_root).resolve()
                if destination.exists() or destination == root or root in destination.parents:
                    raise core.JointCheckpointError("Joint restore requires a fresh independent mutable-state root")
                destination.mkdir(parents=True)
                source = root / "joint_state"
                copied_bytes = 0
                started = time.monotonic()
                for component in MUTABLE_COMPONENTS:
                    core._copy_tree(source / component, destination / component)
                    copied_bytes += sum(path.stat().st_size for path in core._tree_files(destination / component))
                for name in SMALL_STATE_FILES:
                    target = destination / name
                    shutil.copyfile(source / name, target)
                    if core._file_digest(target) != checkpoint.payload["files"]["joint_state/" + name]:
                        raise core.JointCheckpointError("Copied small controller/evaluation state differs")
                    copied_bytes += target.stat().st_size
                core._validate_storage(destination, checkpoint.payload["snapshots"])
                controller = core._read_json(destination / "controller.json")
                evaluation = core._read_json(destination / "evaluation.json")
                # Mutable-state validation can take time. Check source identity
                # again before any external model/optimizer/EMA loader runs.
                if self._checked_guard(root) != proof.source_guard:
                    raise core.JointCheckpointError("Checkpoint source changed while copying mutable state")
                spec = checkpoint.payload["spec"]

                def finished(callback, *args, **callback_kwargs):
                    result = callback(*args, **callback_kwargs)
                    if inspect.isawaitable(result):
                        raise core.JointCheckpointError("Restore callbacks must resolve all worker RPCs")
                    try:
                        core._canonical(result)
                    except (TypeError, ValueError) as error:
                        raise core.JointCheckpointError("Restore returned an unresolved/non-JSON RPC result") from error

                finished(load_online, root / "actor", spec)
                finished(load_ema, source / "ema", spec)
                if load_training_state is not None:
                    paths = tuple(
                        root / str(core._relative(path)) for path in checkpoint.payload["training_state_paths"]
                    )
                    finished(load_training_state, paths, spec)
                finished(load_controller, controller, expected_optimizer_step=spec["optimizer_step"])
                store = core.SkillStore(destination / "skill_store")
                replay = core.CurateReplay(destination / "replay")
                ledger = core.CreditLedger(destination / "ledger", replay)
                receipt = {
                    "schema_version": 1,
                    "policy": POLICY,
                    "manifest_id": checkpoint.manifest_id,
                    "checkpoint_dir": str(root),
                    "mutable_state_root": str(destination),
                    "online_source": str(root / "actor"),
                    "ema_source": str(source / "ema"),
                    "training_state_sources": [
                        str(root / str(core._relative(p))) for p in checkpoint.payload["training_state_paths"]
                    ],
                    "full_validation_passes": 1,
                    "ema_files_copied": 0,
                    "copied_components": list(MUTABLE_COMPONENTS),
                    "mutable_copy_bytes": copied_bytes,
                    "source_guard": "manifest and file metadata/inventory before/after validation and before loaders",
                    "source_guard_sha256": hashlib.sha256(repr(proof.source_guard).encode()).hexdigest(),
                    "restore_seconds_after_validation": time.monotonic() - started,
                }
                core._atomic_json(destination / "DIRECT_READ_RESTORE.json", receipt)
                core._atomic_json(destination / "RESTORE_COMPLETE.json", {"manifest_id": checkpoint.manifest_id})
                self.last_restore_receipt = receipt
                return core.RestoredJointState(checkpoint, store, replay, ledger, controller, evaluation)

    return DirectReadCheckpointCoordinator


class DirectReadRestoreMixin:
    """Keep the frozen trainer's load order; change only its coordinator provider."""

    def _load_checkpoint(self):
        self._joint_coordinator = coordinator_class()()
        return super()._load_checkpoint()
