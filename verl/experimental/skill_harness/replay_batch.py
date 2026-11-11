# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Native text-only TITO tensors for delayed-credit replay; never re-tokenize.

Prepare anchor inputs, call the ordinary actor.compute_log_prob with a fenced
online policy, then finalize AFTER solver GRPO advantages have been computed.
The caller must attest the actual anchor version and request fingerprint. The
caller also owns lease acquire/apply and optimizer/checkpoint reconciliation.
No ledger state is changed here. Only complete single-trajectory candidates are
supported. ALFWorld train producers and train solver rows are required; dev/test
or unidentifiable data fail closed. Padding is a masked computational row only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.utils import tensordict_utils as tu
from verl.workers.utils.padding import response_from_nested

from .alfworld.data import GameRecord
from .credit import CreditLedger
from .replay import CurateReplay, _plain

_REQUIRED = ("input_ids", "position_ids", "prompts", "responses", "response_mask", "loss_mask")
_RESPONSE_FIELDS = {
    "old_log_probs", "advantages", "ref_log_prob", "rollout_log_probs", "rollout_is_weights",
    "rm_scores", "entropy", "token_level_scores", "token_level_rewards", "returns", "values",
}


@dataclass(frozen=True)
class ReplayLineage:
    row_index: int
    settlement_id: str
    label_version: str
    record_id: str
    record_payload_hash: str
    producer_uid: str
    candidate_id: int
    update_id: str
    owner: str
    lease_token: str
    expires_at: float


@dataclass(frozen=True)
class CurateAnchorBatch:
    data: TensorDict
    lineages: tuple[ReplayLineage, ...]
    fingerprint: str
    anchor_policy_version: str
    dp_size: int
    pad_token_id: int
    position_id_planes: int

    @property
    def leased_ids(self):
        return tuple(item.settlement_id for item in self.lineages)


@dataclass(frozen=True)
class JointReplayBatch:
    data: TensorDict
    lineages: tuple[ReplayLineage, ...]
    anchor_policy_version: str
    anchor_input_fingerprint: str

    @property
    def leased_ids(self):
        return tuple(item.settlement_id for item in self.lineages)

    @property
    def acknowledgments(self):
        """Inputs for mark_applied only AFTER an independently verified update."""
        return tuple({"settlement_id": row.settlement_id, "update_id": row.update_id,
                      "lease_token": row.lease_token} for row in self.lineages)


def required_solver_fields():
    """Fetch these from TQ AFTER solver advantage; preserve additional native fields."""
    return (*_REQUIRED, "old_log_probs", "advantages", "uid", "session_id", "tools_kwargs", "extra_fields")


def _rows(value):
    return list(value.unbind()) if value.is_nested else list(value)


def _nested(rows):
    return tu.nested_tensor_from_tensor_list(rows)


def _task_metadata(value):
    if not isinstance(value, dict) or value.get("dataset_phase") != "train":
        raise ValueError("training requires native train task_metadata; dev/test producers are excluded")
    native = dict(value)
    split = native.get("split", native.get("official_split"))
    if split != "train" or native.get("official_split", split) != split:
        raise ValueError("training source official split must be train")
    native["split"] = split
    if not native.get("catalog_digest"):
        raise ValueError("training source requires an immutable catalog identity")
    GameRecord.from_dict(native)
    return native


def infer_position_id_planes(solver: TensorDict) -> int:
    """Infer and validate actual native text-only position layout, including Qwen M-RoPE."""
    if "position_ids" not in solver or "input_ids" not in solver:
        raise ValueError("native solver input_ids and position_ids are required")
    planes = None
    for ids, positions in zip(_rows(solver["input_ids"]), _rows(solver["position_ids"]), strict=True):
        current = 1 if positions.ndim == 1 else positions.shape[0] if positions.ndim == 2 else 0
        if current not in (1, 4) or positions.shape[-1] != ids.numel():
            raise ValueError("unsupported native text-only position_ids layout")
        expected = torch.arange(ids.numel(), dtype=positions.dtype, device=positions.device)
        if not torch.equal(positions, expected if positions.ndim == 1 else expected.expand_as(positions)):
            raise ValueError("non-sequential/multimodal positions require an explicitly validated replay adapter")
        if planes is not None and planes != current:
            raise ValueError("inconsistent native position planes")
        planes = current
    if planes is None:
        raise ValueError("an actual nonempty native solver batch is required to infer position layout")
    return planes


def _position_ids(length, planes):
    position = torch.arange(length, dtype=torch.int64)
    return position if planes == 1 else position.repeat(planes, 1)


def _padding_row(token, planes, identity):
    return {
        "prompts": torch.tensor([token]), "responses": torch.tensor([token]),
        "input_ids": torch.tensor([token, token]), "position_ids": _position_ids(2, planes),
        "response_mask": torch.zeros(1, dtype=torch.int64), "loss_mask": torch.zeros(1, dtype=torch.int64),
        "role_id": 0, "curate_advantages": 0.0, "curate_candidate_id": -1, "curate_trajectory_count": 0,
        "curate_behavior_log_probs": torch.zeros(1), "curate_anchor_log_probs": torch.zeros(1),
        "solver_token_mask": torch.zeros(1, dtype=torch.bool), "curate_token_mask": torch.zeros(1, dtype=torch.bool),
        "uid": identity, "session_id": identity, "extra_fields": {"skill_replay_padding": True},
        "tools_kwargs": {}, "multi_modal_inputs": {}, "is_padding": True,
    }


def _pack(rows):
    fields = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        first = values[0]
        if isinstance(first, torch.Tensor):
            fields[key] = _nested(values)
        elif key in ("role_id", "curate_candidate_id", "curate_trajectory_count"):
            fields[key] = torch.tensor(values, dtype=torch.int64)
        elif key == "curate_advantages":
            fields[key] = torch.tensor(values, dtype=torch.float32)
        elif key == "is_padding":
            fields[key] = torch.tensor(values, dtype=torch.bool)
        else:
            fields[key] = NonTensorStack.from_list([NonTensorData(copy.deepcopy(value)) for value in values])
    return TensorDict(fields, batch_size=[len(rows)])


def _fingerprint(data, version):
    payload = {"policy_version": version}
    for key in (*_REQUIRED, "curate_behavior_log_probs", "curate_token_mask", "solver_token_mask",
                "curate_advantages", "curate_candidate_id", "curate_trajectory_count", "role_id"):
        value = data[key]
        payload[key] = [item.tolist() for item in _rows(value)]
    for key in ("tools_kwargs", "extra_fields", "uid", "session_id"):
        payload[key] = tu.get(data, key)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _active_lease(ledger, settlement_id, update_id, owner, now, expected_token=None):
    status = ledger.training_status(settlement_id, now=now)
    lease = status["lease"]
    if status["state"] != "leased" or lease["update_id"] != update_id or lease["owner"] != owner:
        raise ValueError("a live matching learner lease is required for every curate label")
    if expected_token is not None and lease["lease_token"] != expected_token:
        raise ValueError("curate lease ownership changed after anchor preparation")
    return lease


def prepare_curate_anchor_inputs(
    replay: CurateReplay,
    ledger: CreditLedger,
    settlement_ids,
    *,
    update_id: str,
    owner: str,
    anchor_policy_version: str,
    tokenizer_version: str,
    template_version: str,
    dp_size: int,
    pad_token_id: int,
    position_id_planes: int,
    now: float | None = None,
) -> CurateAnchorBatch:
    """Build immutable-token inference inputs from active leased settlements.

    Pass position_id_planes=infer_position_id_planes(actual_solver_batch).
    The caller must fence online updates throughout anchor inference and attest
    its policy version/fingerprint when finalizing. No calls or GPU work here.
    """
    if not anchor_policy_version or type(dp_size) is not int or dp_size <= 0:
        raise ValueError("a named anchor policy and positive DP size are required")
    if type(pad_token_id) is not int or pad_token_id < 0 or position_id_planes not in (1, 4):
        raise ValueError("invalid native padding token or inferred position plane count")
    identities = list(settlement_ids)
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("a nonempty unique set of settled labels is required")
    timestamp = time.time() if now is None else now
    rows, lineages, candidate_ids = [], [], set()
    for identity in identities:
        lease = _active_lease(ledger, identity, update_id, owner, timestamp)
        label = ledger.get_settlement(identity)
        if len(label.curate_record_ids) != 1:
            raise ValueError("multi-TITO candidate replay requires full-candidate gradient aggregation")
        record = replay.get(label.curate_record_id)
        ref = replay.get_ref(record.record_id)
        if record.trajectory_count != 1 or record.trajectory_index != 0:
            raise ValueError("only complete single-trajectory curate records are supported")
        if dict(record.skill_ref) != dict(label.skill_ref):
            raise ValueError("settled skill does not match the immutable producer record")
        if record.tokenizer_version != tokenizer_version or record.template_version != template_version:
            raise ValueError("replay tokenizer/template version does not match the current native runtime")
        context = _plain(record.context_snapshot)
        if not isinstance(context, dict):
            raise ValueError("replay context must retain its train task_metadata")
        metadata = _task_metadata(context.get("task_metadata"))
        candidate_id = int.from_bytes(hashlib.sha256(record.record_id.encode()).digest()[:8], "big") & (2**63 - 1)
        if candidate_id in candidate_ids:
            raise ValueError("duplicate candidate or stable integer candidate-ID collision")
        candidate_ids.add(candidate_id)
        prompt, action = torch.tensor(record.prompt_ids), torch.tensor(record.action_ids)
        mask = torch.tensor(record.curate_mask, dtype=torch.int64)
        ids = torch.cat((prompt, action))
        positions = _position_ids(ids.numel(), position_id_planes)
        if "native_position_ids" in context:
            archived = torch.tensor(context["native_position_ids"], dtype=torch.int64)
            if not torch.equal(archived, positions):
                raise ValueError("archived native positions disagree with validated text-only position layout")
            positions = archived
        row = _padding_row(pad_token_id, position_id_planes, "curate:" + record.record_id)
        row.update({
            "prompts": prompt, "responses": action, "input_ids": ids, "position_ids": positions,
            "response_mask": mask, "loss_mask": mask.clone(), "role_id": 1,
            "curate_behavior_log_probs": torch.tensor(record.behavior_logprobs, dtype=torch.float32),
            "curate_anchor_log_probs": torch.zeros_like(mask, dtype=torch.float32),
            "solver_token_mask": torch.zeros_like(mask, dtype=torch.bool), "curate_token_mask": mask.bool(),
            "curate_advantages": label.utility, "curate_candidate_id": candidate_id, "curate_trajectory_count": 1,
            "is_padding": False, "tools_kwargs": {"task": {"metadata": metadata}},
            "extra_fields": {"curate_record_id": record.record_id, "producer_uid": record.producer_uid,
                             "settlement_id": label.settlement_id, "label_version": label.label_version,
                             "behavior_policy_version": record.behavior_policy_version,
                             "behavior_policy_kind": record.behavior_policy_kind,
                             "tokenizer_version": record.tokenizer_version, "template_version": record.template_version,
                             "candidate_group_id": record.candidate_group_id,
                             "candidate_index": record.candidate_index,
                             "candidate_group_size": record.candidate_group_size,
                             "trajectory_index": record.trajectory_index, "trajectory_count": record.trajectory_count,
                             "created_at": record.created_at, "replay_ref": ref.to_dict(),
                             "skill_ref": _plain(record.skill_ref), "context_snapshot": context,
                             "skill_snapshot_id": record.skill_snapshot_id,
                             "sampling_config": _plain(record.sampling_config)},
        })
        lineages.append(ReplayLineage(len(rows), identity, label.label_version, record.record_id, ref.payload_hash,
                                     record.producer_uid, candidate_id, update_id, owner,
                                     lease["lease_token"], lease["expires_at"]))
        rows.append(row)
    for index in range((-len(rows)) % dp_size):
        rows.append(_padding_row(pad_token_id, position_id_planes, f"anchor-pad:{update_id}:{index}"))
    data = _pack(rows)
    tu.assign_non_tensor(data, compute_loss=False, calculate_entropy=False, temperature=1.0,
                         pad_mode="no_padding", pad_token_id=pad_token_id)
    return CurateAnchorBatch(data, tuple(lineages), _fingerprint(data, anchor_policy_version),
                             anchor_policy_version, dp_size, pad_token_id, position_id_planes)


def finalize_joint_batch(
    solver_after_grpo: TensorDict,
    anchor_plan: CurateAnchorBatch,
    anchor_output: TensorDict,
    *,
    ledger: CreditLedger,
    anchor_policy_version: str,
    anchor_input_fingerprint: str,
    solver_temperature: float,
    now: float | None = None,
) -> JointReplayBatch:
    """Combine native solver rows and externally anchored, uncentered curate rows.

    ``anchor_output['log_probs']`` must be the actual full-sequence nested native
    compute_log_prob output. Never provide zeros or loss-masked approximations.
    The supplied anchor attestation must describe the policy fenced across the
    actual inference, not merely repeat a hoped-for version after another step.
    """
    if (anchor_policy_version != anchor_plan.anchor_policy_version
            or anchor_input_fingerprint != anchor_plan.fingerprint
            or (anchor_plan.lineages
                and _fingerprint(anchor_plan.data, anchor_policy_version) != anchor_plan.fingerprint)):
        raise ValueError("anchor policy version/input fingerprint attestation mismatch")
    if not anchor_plan.lineages and (len(anchor_plan.data) or anchor_policy_version or anchor_input_fingerprint):
        raise ValueError("A solver-only update must not claim replay rows or an anchor evaluation")
    if not math.isfinite(solver_temperature) or solver_temperature <= 0:
        raise ValueError("solver likelihood temperature must be finite and positive")
    for field in (*_REQUIRED, "old_log_probs", "advantages"):
        if field not in solver_after_grpo or not solver_after_grpo[field].is_nested:
            raise ValueError(f"native nested solver {field} required AFTER solver GRPO advantage")
    if infer_position_id_planes(solver_after_grpo) != anchor_plan.position_id_planes:
        raise ValueError("anchor position layout differs from the actual native solver")
    if any(key in solver_after_grpo for key in ("role_id", "skill_role_normalization")):
        raise ValueError("solver input already contains joint/stale role metadata")
    timestamp = time.time() if now is None else now
    for lineage in anchor_plan.lineages:
        _active_lease(ledger, lineage.settlement_id, lineage.update_id, lineage.owner, timestamp, lineage.lease_token)
    native = anchor_output.get("log_probs")
    if anchor_plan.lineages and (native is None or not native.is_nested):
        raise ValueError("actual native full-sequence nested anchor log_probs are required")
    if anchor_plan.lineages and not torch.equal(
        native.offsets().diff().cpu(), anchor_plan.data["input_ids"].offsets().diff().cpu()
    ):
        raise ValueError("anchor output does not cover every original prompt/action token")
    anchored = _rows(response_from_nested(native, anchor_plan.data["response_mask"])) if anchor_plan.lineages else []
    request = anchor_plan.data
    replay_rows = []
    for index in range(len(anchor_plan.lineages)):
        row = {}
        for key, value in request.items():
            if isinstance(value, torch.Tensor):
                row[key] = _rows(value)[index].clone()
                if not value.is_nested:
                    row[key] = row[key].item()
            elif isinstance(value, NonTensorStack):
                row[key] = copy.deepcopy(value.tolist()[index])
        mask = row["curate_token_mask"]
        probabilities = anchored[index].detach().cpu()
        if not torch.isfinite(probabilities[mask]).all() or (probabilities[mask] > 1e-6).any():
            raise ValueError("missing/nonfinite/invalid anchor likelihood on an authored curate action")
        row["curate_anchor_log_probs"] = torch.where(mask, probabilities, 0)
        replay_rows.append(row)
    rows, globals_ = [], {}
    for index in range(len(solver_after_grpo)):
        row = {}
        for key, value in solver_after_grpo.items():
            if isinstance(value, torch.Tensor):
                if key not in _REQUIRED and key not in _RESPONSE_FIELDS and key != "attention_mask":
                    raise ValueError(f"unaudited solver tensor field {key!r}; provide an explicit replay adapter")
                row[key] = _rows(value)[index].clone()
            elif isinstance(value, NonTensorStack):
                row[key] = copy.deepcopy(value.tolist()[index])
            elif isinstance(value, NonTensorData):
                globals_[key] = copy.deepcopy(value.data)
            else:
                raise ValueError(f"unsupported native solver field {key!r}")
        response = row["responses"]
        mask = row["response_mask"].bool()
        original_mask = row["loss_mask"].bool()
        if (response.numel() != mask.numel() or original_mask.shape != mask.shape
                or (mask & ~original_mask).any()):
            raise ValueError("native solver response/loss masks are inconsistent")
        if not torch.equal(row["input_ids"], torch.cat((row["prompts"], response))):
            raise ValueError("native solver prompt/response IDs do not match its actual full context")
        for key in _RESPONSE_FIELDS & row.keys():
            if row[key].shape != response.shape:
                raise ValueError(f"native solver {key} must retain the full response-token alignment")
        if row.get("multi_modal_inputs"):
            raise ValueError("multimodal replay is unsupported; ALFWorld must remain text-only")
        tools = row.get("tools_kwargs", {})
        task_metadata = tools.get("task", {}).get("metadata")
        if mask.any() or task_metadata is not None:
            _task_metadata(task_metadata)
        row.update({"role_id": 0, "solver_token_mask": mask, "curate_token_mask": torch.zeros_like(mask),
                    "curate_behavior_log_probs": torch.zeros_like(response, dtype=torch.float32),
                    "curate_anchor_log_probs": torch.zeros_like(response, dtype=torch.float32),
                    "curate_advantages": 0.0, "curate_candidate_id": -1, "curate_trajectory_count": 0,
                    "is_padding": not bool(mask.any())})
        rows.append(row)
    solver_count = len(rows)
    rows.extend(replay_rows)
    padding_group = anchor_plan.lineages[0].update_id if anchor_plan.lineages else "solver_only"
    for index in range((-len(rows)) % anchor_plan.dp_size):
        rows.append(_padding_row(anchor_plan.pad_token_id, anchor_plan.position_id_planes,
                                 f"joint-pad:{padding_group}:{index}"))
    # Preserve every solver field. Only explicitly audited response-shaped loss
    # fields receive curator/padding placeholders, all excluded from solver loss.
    all_keys = set().union(*(row.keys() for row in rows))
    for row in rows:
        for key in all_keys - row.keys():
            if key in _RESPONSE_FIELDS:
                prototype = _rows(solver_after_grpo[key])[0] if key in solver_after_grpo else torch.zeros(1)
                row[key] = torch.zeros(row["responses"].numel(), dtype=prototype.dtype)
            elif key == "attention_mask":
                row[key] = torch.ones_like(row["input_ids"])
            elif key == "multi_modal_inputs":
                row[key] = {}
            else:
                row[key] = None
    data = _pack(rows)
    for key, value in globals_.items():
        tu.assign_non_tensor_data(data, key, value)
    temperatures = [solver_temperature] * solver_count + [1.0] * (len(rows) - solver_count)
    if solver_temperature != 1 and globals_.get("use_fused_kernels", False):
        raise ValueError("mixed role temperatures require unfused kernels supporting per-sample temperature")
    if solver_temperature == 1:
        tu.assign_non_tensor_data(data, "temperature", 1.0)
    else:
        data["temperature"] = torch.tensor(temperatures, dtype=torch.float32)
    tu.assign_non_tensor(data, global_batch_size=len(rows), mini_batch_size=len(rows), num_mini_batch=1,
                         epochs=1, compute_loss=True, pad_mode="no_padding", pad_token_id=anchor_plan.pad_token_id)
    lineage = tuple(replace(item, row_index=solver_count + i) for i, item in enumerate(anchor_plan.lineages))
    return JointReplayBatch(data, lineage, anchor_policy_version, anchor_input_fingerprint)


def prepare_solver_only_batch(
    solver_after_grpo: TensorDict, *, dp_size: int, pad_token_id: int, solver_temperature: float = 1.0,
) -> JointReplayBatch:
    """Use the same native validation/normalization before any credit is ready.

    This is an ordinary solver update with role-zero metadata and optional
    masked DP padding. No curate trajectory, label, anchor or lease is invented.
    """
    if type(dp_size) is not int or dp_size < 1 or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("A real DP size and nonnegative tokenizer padding ID are required")
    planes = infer_position_id_planes(solver_after_grpo)
    empty = TensorDict({}, batch_size=[0])
    plan = CurateAnchorBatch(empty, (), "", "", dp_size, pad_token_id, planes)
    return finalize_joint_batch(
        solver_after_grpo, plan, empty, ledger=None,
        anchor_policy_version="", anchor_input_fingerprint="", solver_temperature=solver_temperature,
    )
