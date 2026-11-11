# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Complete joint snapshots over explicitly fenced, fully finished checkpoints.

This module never uses verl's early ``latest_checkpointed_iteration.txt``.
Callers stop actor updates AND SkillStore/replay/ledger/controller writers using
one fence, resolve actual checkpoint RPCs, then publish a separate COMPLETE
manifest only after every component is copied, hashed and fsynced. An async
online saver requires an explicit completion waiter. No Torch/Ray import occurs.

There is no optimizer/ledger transaction here. A snapshot is a consistent cut
provided by the caller's fence, not exactly-once training via CAS. Failed actor
or controller loads require the caller to keep training stopped; this module
cannot roll back external model/optimizer side effects. Uncertain leases remain
uncertain and are never cleared or reconciled based on elapsed time.

Replay/ledger export manifests are retained byte-for-byte, together with their
complete fenced trees. Their active states use IDs/hashes, so copying verified
trees to new roots avoids dereferencing absolute historical transport URIs in a
ledger export. Controller state and historical URI strings are preserved for
audit; consumers resolve replay objects by ID through the restored backend.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .credit import CreditLedger
from .replay import CurateReplay, _plain, _read_payload
from .store import SkillStore

MANIFEST_NAME = "joint_manifest.json"
LATEST_NAME = "latest_joint_checkpoint.json"
SCHEMA_VERSION = 1


class JointCheckpointError(RuntimeError):
    """A joint snapshot is incomplete, corrupt, incompatible or not safely fenced."""


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".joint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise JointCheckpointError(f"Missing or malformed joint checkpoint object: {path}") from error


def _relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or not path.parts:
        raise JointCheckpointError("Checkpoint file references must be normalized relative paths")
    return path


def _tree_files(root):
    result = []
    if root.is_symlink() or not root.is_dir():
        raise JointCheckpointError(f"Missing checkpoint component directory: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise JointCheckpointError(f"Checkpoint components cannot contain symbolic links: {path}")
        relative = path.relative_to(root)
        if "locks" in relative.parts or path.name == ".writer.lock":
            continue
        if path.is_file():
            result.append(path)
        elif not path.is_dir():
            raise JointCheckpointError(f"Unsupported checkpoint filesystem object: {path}")
    return result


def _copy_tree(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise JointCheckpointError("Checkpoint components cannot be copied recursively into themselves")
    destination.mkdir(parents=True, exist_ok=False)
    for path in _tree_files(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _file_digest(path, *, durable=False):
    if path.is_symlink() or not path.is_file():
        raise JointCheckpointError(f"Missing/unsafe checkpoint file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        if durable:
            os.fsync(source.fileno())
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


@dataclass(frozen=True)
class JointCheckpointSpec:
    global_step: int
    optimizer_step: int
    world_size: int
    layout_fingerprint: str
    versions: Mapping[str, str]
    evaluator_fingerprint: str
    protocol_fingerprint: str

    def __post_init__(self):
        for name in ("global_step", "optimizer_step", "world_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "world_size" else 0):
                raise ValueError(f"Invalid joint checkpoint {name}")
        for name in ("layout_fingerprint", "evaluator_fingerprint", "protocol_fingerprint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"A named {name} is required")
        required = {"code", "verl", "model", "tokenizer", "template"}
        if not required <= self.versions.keys() or any(not isinstance(v, str) or not v for v in self.versions.values()):
            raise ValueError("Explicit code/verl/model/tokenizer/template versions are required")
        object.__setattr__(self, "versions", MappingProxyType(dict(self.versions)))

    def compatibility(self):
        return {name: _plain(getattr(self, name)) for name in (
            "world_size", "layout_fingerprint", "versions", "evaluator_fingerprint", "protocol_fingerprint",
        )}

    def to_dict(self):
        return {"global_step": self.global_step, "optimizer_step": self.optimizer_step, **self.compatibility()}


@dataclass(frozen=True)
class CompletedOnlineCheckpoint:
    global_step: int
    optimizer_step: int
    world_size: int
    layout_fingerprint: str
    durability_id: str
    completed: bool = True
    async_save: bool = False


@dataclass(frozen=True)
class VerifiedJointCheckpoint:
    checkpoint_dir: str
    manifest_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class RestoredJointState:
    checkpoint: VerifiedJointCheckpoint
    skill_store: SkillStore
    replay: CurateReplay
    ledger: CreditLedger
    controller_state: Mapping[str, Any]
    evaluation_state: Mapping[str, Any]


def _validate_controller(payload, spec):
    required = {"optimizer_step", "protocol_fingerprint", "pending", "prepared_update", "applied_update_ids"}
    if not isinstance(payload, Mapping) or not required <= payload.keys():
        raise JointCheckpointError("Controller snapshot must retain clocks, pending/prepared queues and applied IDs")
    if (type(payload["optimizer_step"]) is not int or payload["optimizer_step"] != spec.optimizer_step
            or payload["protocol_fingerprint"] != spec.protocol_fingerprint):
        raise JointCheckpointError("Controller protocol/optimizer clock differs from the online checkpoint")
    if (not isinstance(payload["pending"], Mapping) or not isinstance(payload["applied_update_ids"], Mapping)
            or payload["prepared_update"] is not None and not isinstance(payload["prepared_update"], Mapping)):
        raise JointCheckpointError("Controller pending/prepared/applied state has unsupported structure")


def _validate_storage(state_root, snapshots):
    store = SkillStore(state_root / "skill_store", read_only=True)
    heads = {}
    for reference in sorted((store.root / "refs").glob("*.json")):
        heads[reference.stem] = store.head(reference.stem)
    if heads != snapshots["skill_heads"]:
        raise JointCheckpointError("Skill branch heads disagree with the joint snapshot")
    for path in (store.root / "snapshots").glob("*.json"):
        for skill in store.get_snapshot(path.stem).skills:
            store.load_artifact(skill)
    replay = CurateReplay(state_root / "replay")
    replay_data = _read_payload(state_root / "replay" / str(_relative(snapshots["replay"]["relative_manifest"])))
    if "sha256:" + _digest(replay_data) != snapshots["replay"]["snapshot_id"]:
        raise JointCheckpointError("Replay export identity mismatch")
    for item in replay_data["records"]:
        if replay.get_ref(item["record_id"]).payload_hash != item["payload_hash"]:
            raise JointCheckpointError("Replay export record mismatch")
    ledger = CreditLedger(state_root / "ledger", replay)
    ledger_data = _read_payload(state_root / "ledger" / str(_relative(snapshots["ledger"]["relative_manifest"])))
    if "sha256:" + _digest(ledger_data) != snapshots["ledger"]["snapshot_id"]:
        raise JointCheckpointError("Credit export identity mismatch")
    if (ledger_data["state"] != ledger._state()
            or ledger_data["replay_snapshot"]["snapshot_id"] != snapshots["replay"]["snapshot_id"]):
        raise JointCheckpointError("Copied credit/replay trees are not the same fenced exported state")
    for item in ledger_data["objects"]:
        relative = _relative(item["relative_path"])
        if _digest(_read_payload(ledger.root / str(relative))) != item["payload_hash"]:
            raise JointCheckpointError("Credit export object mismatch")


class JointCheckpointCoordinator:
    def save(
        self, checkpoint_dir, *, spec: JointCheckpointSpec, save_online: Callable,
        save_ema: Callable, skill_store: SkillStore, replay: CurateReplay, ledger: CreditLedger,
        controller_state: Mapping | Callable, evaluation_state: Mapping | Callable,
        fence: Callable, wait_online: Callable | None = None, training_state_paths=("data.pt",),
    ) -> VerifiedJointCheckpoint:
        """Callbacks must resolve actual RPCs; the fence must stop ALL state writers.

        save_online(actor_dir) returns CompletedOnlineCheckpoint, possibly after
        wait_online(result). save_ema(ema_dir) returns one save_ema metadata dict
        per worker rank, including path and last_optimizer_step. The saved actor
        must include FSDP model/optim/extra-state shards for every rank. The
        supplied callback also saves data.pt/TQ state under checkpoint_dir.
        """
        root = Path(checkpoint_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with fence():
            if (root / MANIFEST_NAME).exists():
                verified = self.validate(root, expected=spec.compatibility())
                if verified.payload["spec"] != spec.to_dict():
                    raise JointCheckpointError("Completed checkpoint identity cannot be reused for another clock")
                self._publish_latest(verified)
                return verified
            if (root / "joint_state").exists():
                raise JointCheckpointError("Incomplete joint_state exists; resume must not adopt a partial snapshot")
            receipt = save_online(root / "actor")
            if wait_online is not None:
                receipt = wait_online(receipt)
            if inspect.isawaitable(receipt) or not isinstance(receipt, CompletedOnlineCheckpoint):
                raise JointCheckpointError("Online checkpoint RPC/async save has not been explicitly completed")
            expected_receipt = (spec.global_step, spec.optimizer_step, spec.world_size, spec.layout_fingerprint)
            actual = (receipt.global_step, receipt.optimizer_step, receipt.world_size, receipt.layout_fingerprint)
            if actual != expected_receipt or not receipt.completed or not receipt.durability_id:
                raise JointCheckpointError("Online checkpoint completion receipt has wrong clocks/layout")
            if receipt.async_save and wait_online is None:
                raise JointCheckpointError("Async online saves require an explicit completion waiter")
            for rank in range(spec.world_size):
                for kind in ("model", "optim", "extra_state"):
                    if not (root / "actor" / f"{kind}_world_size_{spec.world_size}_rank_{rank}.pt").is_file():
                        raise JointCheckpointError("Complete per-rank online model/optimizer/RNG shards are required")
            stage = root / (".joint-state-" + uuid.uuid4().hex)
            stage.mkdir()
            try:
                ema_dir = stage / "ema"
                ema_dir.mkdir()
                ema = save_ema(ema_dir)
                if not isinstance(ema, list | tuple) or len(ema) != spec.world_size:
                    raise JointCheckpointError("Every rank must acknowledge a durable EMA checkpoint")
                expected_names = {f"ema_rank_{rank:05d}.pt" for rank in range(spec.world_size)}
                actual_names = set()
                ema_metadata = []
                for item in ema:
                    path = Path(item["path"]).resolve()
                    if path.parent != ema_dir.resolve() or item["last_optimizer_step"] != spec.optimizer_step:
                        raise JointCheckpointError("EMA rank checkpoint clock/path differs from the online checkpoint")
                    actual_names.add(path.name)
                    ema_metadata.append({"file": path.name, **{
                        key: value for key, value in item.items() if key != "path"
                    }})
                if actual_names != expected_names:
                    raise JointCheckpointError("EMA rank files are missing or duplicated")
                replay_ref, ledger_ref = replay.export_snapshot(), ledger.export_snapshot()
                _copy_tree(skill_store.root, stage / "skill_store")
                _copy_tree(replay.root, stage / "replay")
                _copy_tree(ledger.root, stage / "ledger")
                heads = {path.stem: skill_store.head(path.stem) for path in (skill_store.root / "refs").glob("*.json")}
                snapshots = {"skill_heads": heads}
                for kind, ref, source in (("replay", replay_ref, replay.root), ("ledger", ledger_ref, ledger.root)):
                    snapshots[kind] = {"snapshot_id": ref.snapshot_id, "kind": ref.kind, "item_count": ref.item_count,
                                       "relative_manifest": Path(ref.manifest_uri).relative_to(source).as_posix()}
                _validate_storage(stage, snapshots)
                controller = controller_state() if callable(controller_state) else controller_state
                evaluation = evaluation_state() if callable(evaluation_state) else evaluation_state
                _validate_controller(controller, spec)
                if (evaluation.get("evaluator_fingerprint") != spec.evaluator_fingerprint
                        or evaluation.get("protocol_fingerprint") != spec.protocol_fingerprint):
                    raise JointCheckpointError("Evaluator/protocol snapshot fingerprint mismatch")
                _atomic_json(stage / "controller.json", controller)
                _atomic_json(stage / "evaluation.json", evaluation)
                os.replace(stage, root / "joint_state")
                _fsync_dir(root)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            files = {}
            for component in ("actor", "joint_state", *training_state_paths):
                relative = _relative(component)
                path = root / str(relative)
                paths = _tree_files(path) if path.is_dir() else [path]
                for item in paths:
                    files[item.relative_to(root).as_posix()] = _file_digest(item, durable=True)
            for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
                _fsync_dir(directory)
            payload = {"schema_version": SCHEMA_VERSION, "state": "complete", "spec": spec.to_dict(),
                       "snapshots": snapshots, "files": files, "online_durability_id": receipt.durability_id,
                       "ema_metadata": sorted(ema_metadata, key=lambda item: item["file"]),
                       "training_state_paths": list(training_state_paths)}
            envelope = {"manifest_id": "sha256:" + _digest(payload), "payload": payload}
            _atomic_json(root / MANIFEST_NAME, envelope)
            verified = self.validate(root, expected=spec.compatibility())
            self._publish_latest(verified)
            return verified

    @staticmethod
    def _publish_latest(checkpoint):
        path = Path(checkpoint.checkpoint_dir)
        _atomic_json(path.parent / LATEST_NAME, {"checkpoint_dir": path.name,
                                               "manifest_id": checkpoint.manifest_id})

    def validate(self, checkpoint_dir, *, expected: Mapping) -> VerifiedJointCheckpoint:
        root = Path(checkpoint_dir).resolve()
        manifest = _read_json(root / MANIFEST_NAME)
        payload = manifest.get("payload", {})
        if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("state") != "complete"
                or manifest.get("manifest_id") != "sha256:" + _digest(payload)):
            raise JointCheckpointError("Joint checkpoint lacks an intact COMPLETE manifest")
        spec = JointCheckpointSpec(**payload["spec"])
        if dict(expected) != spec.compatibility():
            raise JointCheckpointError("Joint checkpoint world/layout/version/evaluator/protocol mismatch")
        for relative, identity in payload["files"].items():
            if _file_digest(root / str(_relative(relative))) != identity:
                raise JointCheckpointError(f"Checkpoint file digest mismatch: {relative}")
        actual_files = set()
        for component in ("actor", "joint_state", *payload["training_state_paths"]):
            path = root / str(_relative(component))
            paths = _tree_files(path) if path.is_dir() else [path]
            actual_files.update(item.relative_to(root).as_posix() for item in paths)
        if actual_files != set(payload["files"]):
            raise JointCheckpointError("Checkpoint component file inventory changed after COMPLETE publication")
        names = {item["file"] for item in payload["ema_metadata"]}
        if (len(payload["ema_metadata"]) != spec.world_size
                or names != {f"ema_rank_{rank:05d}.pt" for rank in range(spec.world_size)}
                or any(item["last_optimizer_step"] != spec.optimizer_step for item in payload["ema_metadata"])):
            raise JointCheckpointError("EMA checkpoint metadata world/clock mismatch")
        _validate_storage(root / "joint_state", payload["snapshots"])
        _validate_controller(_read_json(root / "joint_state/controller.json"), spec)
        evaluation = _read_json(root / "joint_state/evaluation.json")
        if (evaluation.get("evaluator_fingerprint") != spec.evaluator_fingerprint
                or evaluation.get("protocol_fingerprint") != spec.protocol_fingerprint):
            raise JointCheckpointError("Evaluator/protocol state does not match the COMPLETE manifest")
        return VerifiedJointCheckpoint(str(root), manifest["manifest_id"], payload)

    def latest(self, root, *, expected: Mapping):
        base = Path(root).resolve()
        pointer = _read_json(base / LATEST_NAME)
        name = str(_relative(pointer["checkpoint_dir"]))
        if "/" in name:
            raise JointCheckpointError("Latest joint pointer must identify a direct checkpoint child")
        checkpoint = self.validate(base / name, expected=expected)
        if checkpoint.manifest_id != pointer["manifest_id"]:
            raise JointCheckpointError("Latest joint pointer does not match the completed manifest")
        return checkpoint

    def restore(
        self, checkpoint_dir, *, expected: Mapping, destination_root,
        load_online: Callable, load_ema: Callable, load_controller: Callable, fence: Callable,
        load_training_state: Callable | None = None,
    ) -> RestoredJointState:
        """Validate everything before any live load; restore into an EMPTY new root.

        load_online(actor_path, spec), load_ema(ema_path, spec), then
        load_training_state(paths, spec) restores declared dataloader/TQ files,
        then load_controller(payload, expected_optimizer_step=step) finishes.
        All callbacks must resolve worker RPCs. External side effects cannot be rolled back. Do not
        release a training fence after failure; abort startup/reconcile first.
        """
        with fence():
            checkpoint = self.validate(checkpoint_dir, expected=expected)
            if checkpoint.payload["training_state_paths"] and load_training_state is None:
                raise JointCheckpointError("A completed dataloader/TQ restore callback is required")
            destination = Path(destination_root).resolve()
            if destination.exists():
                raise JointCheckpointError(
                    "Joint restore requires a fresh destination; active state is never overwritten"
                )
            source = Path(checkpoint.checkpoint_dir) / "joint_state"
            _copy_tree(source, destination)
            _validate_storage(destination, checkpoint.payload["snapshots"])
            controller = _read_json(destination / "controller.json")
            evaluation = _read_json(destination / "evaluation.json")
            spec = checkpoint.payload["spec"]
            for callback, path in ((load_online, Path(checkpoint.checkpoint_dir) / "actor"),
                                   (load_ema, Path(checkpoint.checkpoint_dir) / "joint_state/ema")):
                result = callback(path, spec)
                if inspect.isawaitable(result):
                    raise JointCheckpointError("Restore callbacks must explicitly await all worker RPCs")
                try:
                    _canonical(result)
                except (TypeError, ValueError) as error:
                    raise JointCheckpointError("Restore callback returned an unresolved/non-JSON RPC result") from error
            if load_training_state is not None:
                paths = tuple(Path(checkpoint.checkpoint_dir) / str(_relative(path))
                              for path in checkpoint.payload["training_state_paths"])
                result = load_training_state(paths, spec)
                if inspect.isawaitable(result):
                    raise JointCheckpointError("Training dataloader/TQ restore has not finished")
                try:
                    _canonical(result)
                except (TypeError, ValueError) as error:
                    raise JointCheckpointError(
                        "Training state restore returned an unresolved/non-JSON result"
                    ) from error
            result = load_controller(controller, expected_optimizer_step=spec["optimizer_step"])
            if inspect.isawaitable(result):
                raise JointCheckpointError("Controller restore has not finished")
            try:
                _canonical(result)
            except (TypeError, ValueError) as error:
                raise JointCheckpointError("Controller restore returned an unresolved/non-JSON result") from error
            store = SkillStore(destination / "skill_store")
            replay = CurateReplay(destination / "replay")
            ledger = CreditLedger(destination / "ledger", replay)
            _atomic_json(destination / "RESTORE_COMPLETE.json", {"manifest_id": checkpoint.manifest_id})
            return RestoredJointState(checkpoint, store, replay, ledger, controller, evaluation)
