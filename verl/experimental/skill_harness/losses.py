# Copyright 2026 The skill harness contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Joint online-solver and delayed-curate objectives for the public Engine loss hook.

Only ``log_probs`` from the online model carry gradients. The EMA curator supplies
immutable behavior probabilities; it is never an optimizer target. Before EACH
``engine.train_batch`` (after minibatch selection, before microbatch splitting),
call :func:`prepare_role_batch_metadata` on every DP rank. Engine's ordinary
``batch_num_tokens`` mixes roles and cannot normalize these two objectives.

The initial implementation deliberately accepts one COMPLETE TITO trajectory per
curate candidate. Splitting a candidate across rows, microbatches or ranks needs a
differentiable candidate-level likelihood reduction, not tokenwise PPO clipping.
Such input fails closed. All authored curate tokens in its one trajectory must
remain in ``curate_token_mask``; context/tool tokens remain visible but masked.

Curate loss is sequence PPO times a detached, capped historical IS weight:
``w = pi_anchor(action|context) / mu_EMA(action|context)`` and
``r = pi_online(action|context) / pi_anchor(action|context)``. Both likelihoods
are PRODUCTS over the complete macro action (implemented as SUMS of logprobs).
Clipping/capping introduce bias; this is not an unbiased off-policy estimator.
The externally settled advantage is used verbatim, with no GRPO centering.
Truncated/greedy behavior also lacks full target-policy support; prefer full
support sampling for curate collection. Importance weights cannot recover
actions that the behavior distribution could never sample.

This module imports only Torch eagerly so its numerical contract can be tested on
CPU without importing verl's Ray/accelerator infrastructure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace

import torch

SOLVER_ROLE = 0
CURATOR_ROLE = 1
ROLE_NORMALIZATION_KEY = "skill_role_normalization"


@dataclass(frozen=True)
class RoleNormalization:
    """Counts over one complete optimizer minibatch, across its DP group."""

    dp_size: int
    solver_tokens: int
    solver_sequences: int
    curate_sequences: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "dp_size" else 0):
                raise ValueError(f"invalid role normalization {name}={value!r}")


@dataclass(frozen=True)
class CurateLossConfig:
    coefficient: float = 1.0
    clip_low: float = 0.2
    clip_high: float = 0.2
    historical_is_cap: float = 10.0
    max_log_ratio: float = 20.0

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("curate loss settings must be finite")
        if self.coefficient < 0 or not 0 <= self.clip_low < 1 or self.clip_high < 0:
            raise ValueError("invalid curate coefficient or PPO clip interval")
        if self.historical_is_cap <= 0 or not 0 < self.max_log_ratio <= 80:
            raise ValueError("invalid curate IS cap or numerical log-ratio bound")


@dataclass(frozen=True)
class SolverLossConfig:
    """CPU vanilla-PPO settings; the Engine callback uses stock policy dispatch."""

    clip_low: float = 0.2
    clip_high: float = 0.2
    dual_clip: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: float | None = None
    entropy_coefficient: float = 0.0
    kl_coefficient: float = 0.0

    def __post_init__(self):
        if not 0 <= self.clip_low < 1 or self.clip_high < 0 or self.dual_clip <= 1:
            raise ValueError("invalid solver PPO clipping settings")
        if not all(math.isfinite(v) for v in (self.clip_low, self.clip_high, self.dual_clip,
                                               self.entropy_coefficient, self.kl_coefficient)):
            raise ValueError("solver settings must be finite")
        if self.loss_agg_mode not in {
            "token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        }:
            raise ValueError("unsupported solver aggregation")
        if self.loss_agg_mode == "seq-mean-token-sum-norm" and (
            self.loss_scale_factor is None or not math.isfinite(self.loss_scale_factor)
            or self.loss_scale_factor <= 0
        ):
            raise ValueError("a fixed positive loss_scale_factor is required across microbatches")


def _padded(value: torch.Tensor, padding: float = 0) -> torch.Tensor:
    if not value.is_nested:
        return value
    # to_padded_tensor does not support every jagged layout on every Torch release.
    rows = list(value.unbind())
    return torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=padding)


def _masks(role_id, response_mask, solver_token_mask=None, curate_token_mask=None):
    response_mask = _padded(response_mask).bool()
    role_id = role_id.reshape(-1)
    if response_mask.ndim != 2 or role_id.shape != response_mask.shape[:1]:
        raise ValueError("role_id must be [B] and response_mask [B,L]")
    if role_id.dtype == torch.bool or role_id.is_floating_point():
        raise ValueError("role_id must contain integer role identifiers")
    if not ((role_id == SOLVER_ROLE) | (role_id == CURATOR_ROLE)).all():
        raise ValueError("unknown role_id")
    masks = []
    for role, supplied in ((SOLVER_ROLE, solver_token_mask), (CURATOR_ROLE, curate_token_mask)):
        allowed = response_mask & (role_id == role).unsqueeze(-1)
        mask = allowed if supplied is None else _padded(supplied).bool()
        if mask.shape != allowed.shape or (mask & ~allowed).any():
            raise ValueError("role token masks must be response-mask subsets of their own role")
        masks.append(mask)
    if ((role_id == CURATOR_ROLE) & ~masks[1].any(-1)).any():
        raise ValueError("every curator row must contain a complete nonempty curate action")
    return role_id, masks[0], masks[1]


def compute_role_normalization(
    role_id: torch.Tensor,
    response_mask: torch.Tensor,
    solver_token_mask: torch.Tensor | None = None,
    curate_token_mask: torch.Tensor | None = None,
    dp_group=None,
) -> RoleNormalization:
    """Collective called once per optimizer minibatch, NEVER inside the loss.

    All ranks, including ranks without either role, must participate. A supplied
    DP group is mandatory when WORLD is not the engine's data-parallel group.
    For NCCL only the three scalar counts move to the current CUDA device;
    the full CPU minibatch can stay off GPU until Engine microbatch transfer.
    The worker must have selected its ordinary engine device beforehand.
    """
    _, solver_mask, curate_mask = _masks(role_id, response_mask, solver_token_mask, curate_token_mask)
    counts = torch.stack((solver_mask.sum(), solver_mask.any(-1).sum(), curate_mask.any(-1).sum())).long()
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    dp_size = torch.distributed.get_world_size(dp_group) if distributed else 1
    if distributed:
        if torch.distributed.get_backend(dp_group) == "nccl" and counts.device.type != "cuda":
            counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return RoleNormalization(dp_size, *(int(v) for v in counts.tolist()))


def prepare_role_batch_metadata(data, dp_group=None) -> RoleNormalization:
    """Attach scalar metadata that survives TensorDict slicing/microbatching.

    Call AFTER each optimizer-minibatch selection. Attaching once to the full
    rollout batch before ``train_mini_batch`` selects several optimizer batches
    is incorrect. A custom worker can use the public engine ``train_batch`` API
    with this preparation immediately before each call.
    """
    normalization = compute_role_normalization(
        data["role_id"], data["response_mask"], data.get("solver_token_mask"),
        data.get("curate_token_mask"), dp_group,
    )
    # Check candidate uniqueness over the WHOLE update, including candidates on
    # other ranks or later microbatches. Never silently count a TITO segment as
    # a separate complete macro action. Gather errors as well so all DP ranks
    # take the same validation branch after this collective.
    local_candidates, error = [], None
    try:
        roles = data["role_id"].reshape(-1)
        rows = roles == CURATOR_ROLE
        if rows.any():
            counts, ids = data.get("curate_trajectory_count"), data.get("curate_candidate_id")
            if (counts is None or counts.shape != roles.shape or counts.dtype == torch.bool
                    or counts.is_floating_point() or not (counts[rows] == 1).all()):
                raise ValueError("only one complete trajectory per curator candidate is supported")
            if ids is None or ids.shape != roles.shape or ids.dtype == torch.bool or ids.is_floating_point():
                raise ValueError("integer curate_candidate_id [B] is required")
            if (ids[rows] < 0).any():
                raise ValueError("curate candidate IDs must be nonnegative")
            local_candidates = ids[rows].tolist()
    except ValueError as exc:
        error = str(exc)
    summaries = [(local_candidates, error)]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        summaries = [None] * normalization.dp_size
        torch.distributed.all_gather_object(summaries, (local_candidates, error), group=dp_group)
    errors = [item[1] for item in summaries if item[1] is not None]
    candidates = [candidate for item in summaries for candidate in item[0]]
    if errors:
        raise ValueError("curate candidate validation: " + "; ".join(errors))
    if len(candidates) != len(set(candidates)):
        raise ValueError("duplicate or segmented curate candidate IDs across optimizer minibatch")
    if isinstance(data, dict):
        data[ROLE_NORMALIZATION_KEY] = asdict(normalization)
        data["skill_curate_candidates_verified"] = True
    else:
        from verl.utils.tensordict_utils import assign_non_tensor_data

        assign_non_tensor_data(data, ROLE_NORMALIZATION_KEY, asdict(normalization))
        assign_non_tensor_data(data, "skill_curate_candidates_verified", True)
    return normalization


def _active_tensor(value, mask, name):
    if value is None or value.shape != mask.shape:
        raise ValueError(f"{name} must have response shape {tuple(mask.shape)}")
    if not torch.isfinite(value[mask]).all():
        raise ValueError(f"{name} has nonfinite active entries")
    return torch.where(mask, value, 0)


def _aggregate_solver(values, mask, norm, config):
    values = torch.where(mask, values, 0)
    if not mask.any():
        return values.sum() * 0
    if config.loss_agg_mode == "token-sum":
        return values.sum() * norm.dp_size
    if config.loss_agg_mode == "token-mean":
        return values.sum() * norm.dp_size / norm.solver_tokens
    rows = values.sum(-1)
    if config.loss_agg_mode == "seq-mean-token-mean":
        rows = rows / mask.sum(-1).clamp_min(1)
    result = rows.sum() * norm.dp_size / norm.solver_sequences
    if config.loss_agg_mode == "seq-mean-token-sum-norm":
        result = result / config.loss_scale_factor
    return result


def _validate_local_counts(norm, solver_mask, curate_mask):
    if not isinstance(norm, RoleNormalization):
        raise ValueError("whole-optimizer-minibatch RoleNormalization is required")
    local = (int(solver_mask.sum()), int(solver_mask.any(-1).sum()), int(curate_mask.any(-1).sum()))
    totals = (norm.solver_tokens, norm.solver_sequences, norm.curate_sequences)
    if any(current > total for current, total in zip(local, totals, strict=True)):
        raise ValueError("role counts exceed global optimizer-minibatch normalization")


def role_aware_ppo_loss(
    log_probs: torch.Tensor,
    *,
    role_id: torch.Tensor,
    response_mask: torch.Tensor,
    normalization: RoleNormalization,
    old_log_probs: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    solver_token_mask: torch.Tensor | None = None,
    curate_token_mask: torch.Tensor | None = None,
    curate_behavior_log_probs: torch.Tensor | None = None,
    curate_anchor_log_probs: torch.Tensor | None = None,
    curate_advantages: torch.Tensor | None = None,
    curate_trajectory_count: torch.Tensor | None = None,
    curate_candidate_id: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    solver_kl_values: torch.Tensor | None = None,
    solver_config: SolverLossConfig | None = None,
    curate_config: CurateLossConfig | None = None,
    solver_policy_loss: Callable | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return additive DP-scaled microbatch loss; no optimizer/EMA mutation.

    ``curate_advantages`` is [B], one settled counterfactual delta per candidate.
    The anchor is a FROZEN online-policy snapshot at update preparation, while
    behavior probabilities are those actually used by the historical EMA actor.
    Contexts, tokenizer, sampling transform, masks and candidate identities must
    have passed the durable replay validation before this numerical boundary.
    ``solver_policy_loss`` optionally dispatches to stock verl's selected loss.
    """
    solver_config = solver_config if solver_config is not None else SolverLossConfig()
    curate_config = curate_config if curate_config is not None else CurateLossConfig()
    role_id, solver_mask, curate_mask = _masks(
        role_id, response_mask, solver_token_mask, curate_token_mask,
    )
    _validate_local_counts(normalization, solver_mask, curate_mask)
    active = solver_mask | curate_mask
    current = _active_tensor(log_probs, active, "log_probs")
    zero = current.sum() * 0
    solver_loss = solver_pg_loss = zero
    # Additive global token means remain defined on ranks/microbatches without
    # solver rows; no local masked_mean(0/0) enters logging.
    metrics = {name: zero.detach() for name in (
        "actor/pg_clipfrac", "actor/ppo_kl", "actor/pg_clipfrac_lower",
        "skill/solver_entropy", "skill/solver_kl", "skill/curate_advantage",
        "skill/curate_sequence_ratio", "skill/curate_historical_is", "skill/curate_clip_fraction",
        "skill/curate_is_cap_fraction", "skill/curate_numerical_clip_fraction",
    )}
    if solver_mask.any():
        old = _active_tensor(old_log_probs, solver_mask, "old_log_probs").detach()
        adv = _active_tensor(advantages, solver_mask, "advantages").detach()
        weights = None
        if rollout_is_weights is not None:
            weights = _active_tensor(rollout_is_weights, solver_mask, "rollout_is_weights").detach()
            if (weights[solver_mask] < 0).any():
                raise ValueError("rollout IS weights must be nonnegative")
        solver_current = torch.where(solver_mask, current, 0)
        log_ratio = (solver_current - old).clamp(-20, 20)
        ratio = log_ratio.exp()
        ordinary = -adv * ratio
        clipped = -adv * ratio.clamp(1 - solver_config.clip_low, 1 + solver_config.clip_high)
        pessimistic = torch.maximum(ordinary, clipped)
        dual_bound = -adv * solver_config.dual_clip
        metric_scale = normalization.dp_size / normalization.solver_tokens
        metrics.update({
            "actor/pg_clipfrac": ((clipped > ordinary) & solver_mask).sum().detach() * metric_scale,
            "actor/ppo_kl": -log_ratio[solver_mask].sum().detach() * metric_scale,
            "actor/pg_clipfrac_lower": ((pessimistic > dual_bound) & (adv < 0) & solver_mask)
            .sum().detach() * metric_scale,
        })
        if solver_policy_loss is not None:
            solver_loss, _ = solver_policy_loss(old, solver_current, adv, solver_mask, weights)
        else:
            dual = torch.minimum(dual_bound, pessimistic)
            values = torch.where(adv < 0, dual, pessimistic)
            if weights is not None:
                values = values * weights
            solver_loss = _aggregate_solver(values, solver_mask, normalization, solver_config)
        solver_pg_loss = solver_loss
        if entropy is not None:
            entropy_values = _active_tensor(entropy, solver_mask, "entropy")
            entropy_loss = _aggregate_solver(entropy_values, solver_mask, normalization, solver_config)
            solver_loss = solver_loss - solver_config.entropy_coefficient * entropy_loss
            metrics["skill/solver_entropy"] = entropy_loss.detach()
        if solver_config.kl_coefficient:
            kl = _active_tensor(solver_kl_values, solver_mask, "solver_kl_values")
            kl_loss = _aggregate_solver(kl, solver_mask, normalization, solver_config)
            solver_loss = solver_loss + solver_config.kl_coefficient * kl_loss
            metrics["skill/solver_kl"] = kl_loss.detach()
    curate_loss = zero
    rows = curate_mask.any(-1)
    if rows.any():
        if (curate_trajectory_count is None or curate_trajectory_count.shape != role_id.shape
                or curate_trajectory_count.dtype == torch.bool or curate_trajectory_count.is_floating_point()):
            raise ValueError("explicit curate_trajectory_count [B] is required")
        if not (curate_trajectory_count[rows] == 1).all():
            raise ValueError("multi-TITO candidates require a differentiable full-candidate reduction; unsupported")
        if curate_candidate_id is None or curate_candidate_id.shape != role_id.shape:
            raise ValueError("explicit unique curate_candidate_id [B] is required")
        if curate_candidate_id.dtype == torch.bool or curate_candidate_id.is_floating_point():
            raise ValueError("curate_candidate_id must be integer")
        if (curate_candidate_id[rows] < 0).any() or curate_candidate_id[rows].unique().numel() != int(rows.sum()):
            raise ValueError("each complete curator row must have a distinct nonnegative candidate ID")
        behavior = _active_tensor(curate_behavior_log_probs, curate_mask, "curate_behavior_log_probs").detach()
        anchor = _active_tensor(curate_anchor_log_probs, curate_mask, "curate_anchor_log_probs").detach()
        if (behavior[curate_mask] > 1e-6).any() or (anchor[curate_mask] > 1e-6).any():
            raise ValueError("curate likelihood inputs must be log probabilities")
        if curate_advantages is None or curate_advantages.shape != role_id.shape:
            raise ValueError("curate_advantages must be [B], with one uncentered external label per candidate")
        if not torch.isfinite(curate_advantages[rows]).all():
            raise ValueError("curate advantages must be finite")
        advantage = curate_advantages.detach()[rows]
        current_total = torch.where(curate_mask, current, 0).sum(-1)[rows]
        anchor_total = anchor.sum(-1)[rows]
        behavior_total = behavior.sum(-1)[rows]
        log_ratio = current_total - anchor_total
        ratio = log_ratio.clamp(-curate_config.max_log_ratio, curate_config.max_log_ratio).exp()
        historical_log_ratio = anchor_total - behavior_total
        historical_weight = historical_log_ratio.clamp(max=math.log(curate_config.historical_is_cap)).exp().detach()
        unclipped = -advantage * ratio
        clipped = -advantage * ratio.clamp(1 - curate_config.clip_low, 1 + curate_config.clip_high)
        values = historical_weight * torch.maximum(unclipped, clipped)
        scale = normalization.dp_size / normalization.curate_sequences
        curate_loss = values.sum() * scale
        metrics.update({
            "skill/curate_advantage": advantage.sum().detach() * scale,
            "skill/curate_sequence_ratio": ratio.sum().detach() * scale,
            "skill/curate_historical_is": historical_weight.sum().detach() * scale,
            "skill/curate_clip_fraction": (clipped > unclipped).float().sum().detach() * scale,
            "skill/curate_is_cap_fraction": (historical_log_ratio > math.log(curate_config.historical_is_cap))
            .float().sum().detach() * scale,
            "skill/curate_numerical_clip_fraction": (log_ratio.abs() > curate_config.max_log_ratio)
            .float().sum().detach() * scale,
        })
    loss = solver_loss + curate_config.coefficient * curate_loss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite joint loss; check likelihoods, weights and advantages")
    metrics.update({"actor/pg_loss": solver_pg_loss.detach(), "skill/solver_loss": solver_loss.detach(),
                    "skill/curate_loss": curate_loss.detach(),
                    "skill/joint_loss": loss.detach()})
    return loss, metrics


def response_model_output(value: torch.Tensor, data) -> torch.Tensor:
    """Align native Engine full-sequence output or already padded response output.

    NO_PADDING outputs are flattened/nested full sequences and require the
    official causal one-token shift in ``no_padding_2_padding``. A 2D ordinary
    tensor must already have the exact padded response shape; ambiguous full
    sequence padded outputs fail rather than silently slice the wrong tokens.
    """
    if not value.is_nested and value.ndim == 2:
        response_shape = _padded(data["response_mask"]).shape
        if value.shape != response_shape:
            raise ValueError("padded model output must already be response-aligned")
        return value
    from verl.workers.utils.padding import no_padding_2_padding

    return no_padding_2_padding(value, data)


def skill_harness_loss(config, model_output, data, dp_group=None, *, curate_config=None):
    """Inject using ``set_loss_fn(partial(skill_harness_loss, config=config))``.

    This is an FSDP/FSDP2 Engine callback. Other backends have additional pipeline
    scaling contracts and require their own validated adapter. It performs no
    collectives and no optimizer step; the Engine accumulates microbatch losses
    then performs its one ordinary online-parameter update. EMA update belongs
    AFTER a successful optimizer step, outside this callback.
    """
    from verl.trainer.ppo.core_algos import get_policy_loss_fn, kl_penalty
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import AggregationType, Metric

    curate_config = curate_config if curate_config is not None else CurateLossConfig()
    raw_norm = tu.get_non_tensor_data(data, ROLE_NORMALIZATION_KEY, None)
    if not isinstance(raw_norm, Mapping):
        raise ValueError("prepare_role_batch_metadata must run at the optimizer-minibatch boundary")
    norm = RoleNormalization(**dict(raw_norm))
    if norm.curate_sequences and not tu.get_non_tensor_data(data, "skill_curate_candidates_verified", False):
        raise ValueError("whole-update curator candidate validation is required before microbatch splitting")
    actual_dp = torch.distributed.get_world_size(dp_group) if torch.distributed.is_initialized() else 1
    if norm.dp_size != actual_dp:
        raise ValueError("role normalization DP size differs from the loss DP group")
    log_probs = response_model_output(model_output["log_probs"], data)
    fields = {
        name: _padded(data[name]) for name in (
            "role_id", "response_mask", "solver_token_mask", "curate_token_mask", "old_log_probs",
            "advantages", "curate_behavior_log_probs", "curate_anchor_log_probs", "curate_advantages",
            "curate_trajectory_count", "curate_candidate_id", "rollout_is_weights",
        ) if name in data
    }
    scale_factor = config.loss_scale_factor
    settings = SolverLossConfig(
        clip_low=config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio,
        clip_high=config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio,
        dual_clip=config.get("clip_ratio_c", 3.0), loss_agg_mode=config.loss_agg_mode,
        loss_scale_factor=scale_factor, entropy_coefficient=config.entropy_coeff,
        kl_coefficient=config.kl_loss_coef if config.use_kl_loss else 0.0,
    )
    # Avoid mutating the shared ActorConfig; role-specific global counts differ
    # from Engine's total token counts and must not leak to another callback.
    global_batch_info = {
        "dp_size": norm.dp_size, "batch_num_tokens": norm.solver_tokens,
        "global_batch_size": norm.solver_sequences, "loss_scale_factor": scale_factor,
    }
    # ActorConfig freezes attribute assignment after construction while keeping
    # the dict contents mutable. replace constructs a new config with its own
    # dict without bypassing that frozen-field contract.
    solver_config = replace(config, global_batch_info=global_batch_info)
    mode = config.policy_loss.get("loss_mode", "vanilla")
    # Stock loss plugins with internal local means can violate global role
    # normalization. Admit only the audited vanilla GRPO/PPO implementation.
    if mode != "vanilla":
        raise ValueError(f"solver policy loss {mode!r} needs a role-normalization audit before use")
    policy_fn = get_policy_loss_fn(mode)

    def solver_policy(old, current, advantage, mask, weights):
        return policy_fn(old_log_prob=old, log_prob=current, advantages=advantage,
                         response_mask=mask, loss_agg_mode=config.loss_agg_mode,
                         config=solver_config, rollout_is_weights=weights)

    entropy = model_output.get("entropy")
    if entropy is not None:
        entropy = response_model_output(entropy, data)
    kl_values = None
    if config.use_kl_loss:
        _, solver_mask, _ = _masks(fields["role_id"], fields["response_mask"],
                                     fields.get("solver_token_mask"), fields.get("curate_token_mask"))
        if solver_mask.any():
            reference = _padded(data["ref_log_prob"]).detach()
            current = _active_tensor(log_probs, solver_mask, "log_probs")
            reference = _active_tensor(reference, solver_mask, "ref_log_prob")
            kl_values = kl_penalty(current, reference, config.kl_loss_type)
    loss, metrics = role_aware_ppo_loss(
        log_probs, normalization=norm, solver_config=settings, curate_config=curate_config,
        solver_policy_loss=solver_policy, entropy=entropy, solver_kl_values=kl_values, **fields,
    )
    return loss, {name: Metric(value=value, aggregation=AggregationType.SUM) for name, value in metrics.items()}
