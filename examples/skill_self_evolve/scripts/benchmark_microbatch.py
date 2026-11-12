#!/usr/bin/env python3
"""Isolated FSDP microbatch benchmark on archived Native trajectory tokens.

Launch with torchrun on four otherwise idle GPUs. This creates fresh actor and
reference engines, never resumes or writes a training checkpoint, and creates
no rollout/reward. One zero-LR diagnostic step initializes AdamW state without
changing weights so candidate memory is not measured with an empty optimizer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from datetime import timedelta
from functools import partial
from pathlib import Path


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packed_nested(values):
    """Use the public ragged-dimension constructor, including mRoPE planes."""
    import torch

    if values[0].dim() == 1:
        return torch.nested.as_nested_tensor(values, layout=torch.jagged)
    if values[0].dim() != 2:
        raise ValueError("Only text tokens and text mRoPE planes are supported")
    lengths = torch.tensor([value.shape[1] for value in values], device=values[0].device)
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    return torch.nested.nested_tensor_from_jagged(torch.cat(values, dim=1), offsets, jagged_dim=2)


def fixed_microbatches(data, size):
    """A diagnostic packing adapter; model/loss/backward use the official engine."""
    import torch
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    tensor_rows = {key: list(value.unbind()) for key, value in data.items() if isinstance(value, torch.Tensor)}
    metadata = {key: tu.get_non_tensor_data(data, key, None) for key in data.keys() if key not in tensor_rows}
    if len(data) % size:
        raise ValueError("The local batch must be divisible by the fixed microbatch")
    result = []
    for start in range(0, len(data), size):
        item = TensorDict(
            {key: packed_nested(rows[start : start + size]) for key, rows in tensor_rows.items()}, batch_size=[size]
        )
        tu.assign_non_tensor(item, **metadata)
        result.append(item)
    return result


def engine_batch(engine, data, *, loss_function=None, forward_only=True):
    """Mirror the production outer loop with public, validated text packing.

    The current no_sync=False setting means every backward synchronizes.
    Global loss normalization and result aggregation use the production rules.
    This adapter is explicitly not validation of the TransferQueue chunk path.
    """
    from contextlib import nullcontext

    import torch
    import torch.distributed as dist

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.utils import postprocess_batch_func

    tokens = data["loss_mask"].values().sum().to("cuda")
    dist.all_reduce(tokens, group=engine.get_data_parallel_group())
    tu.assign_non_tensor(data, batch_num_tokens=int(tokens), dp_size=engine.get_data_parallel_size(), sp_size=1)
    microbatches = fixed_microbatches(data, tu.get_non_tensor_data(data, "micro_batch_size_per_gpu", None))
    outputs = []
    for micro in microbatches:
        with torch.no_grad() if forward_only else nullcontext():
            loss, output = engine.forward_step(micro, loss_function=loss_function, forward_only=forward_only)
            if not forward_only:
                loss.backward()
                output.pop("model_output", None)
            outputs.append(output)
    return postprocess_batch_func(outputs, None, data)


def select_segments(run, count=16):
    import numpy as np

    segments = []
    for path in sorted((run / "agent_logs/step_4").glob("*/trajectory.npz")):
        with np.load(path, allow_pickle=False) as saved:
            for key in saved.files:
                if not key.endswith("_prompt_ids"):
                    continue
                prefix = key.removesuffix("_prompt_ids")
                p, r, mask = (saved[prefix + suffix] for suffix in ("_prompt_ids", "_response_ids", "_response_mask"))
                if len(p) and len(r) and mask.any() and len(r) == len(mask):
                    segments.append(
                        {
                            "path": str(path),
                            "prefix": prefix,
                            "prompt_tokens": len(p),
                            "response_tokens": len(r),
                            "tokens": len(p) + len(r),
                            "action_tokens": int(mask.sum()),
                        }
                    )
    segments.sort(key=lambda row: (row["tokens"], row["path"], row["prefix"]))
    if len(segments) < count:
        raise ValueError("Not enough real archived segments")
    selected = [segments[round(i * (len(segments) - 1) / (count - 1))] for i in range(count)]
    if len({(row["path"], row["prefix"]) for row in selected}) != count:
        raise ValueError("The fixed benchmark must use distinct archived segments")
    for row in selected:
        row["npz_sha256"] = digest(Path(row["path"]))
    return selected


def make_data(rows, global_batch_size, *, processor=None, tokenizer=None, short=False):
    from types import SimpleNamespace

    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    columns = {key: [] for key in ("input_ids", "position_ids", "prompts", "responses", "response_mask")}
    for row in rows:
        with np.load(row["path"], allow_pickle=False) as saved:
            p = saved[row["prefix"] + "_prompt_ids"].astype(np.int64)
            r = saved[row["prefix"] + "_response_ids"].astype(np.int64)
            mask = saved[row["prefix"] + "_response_mask"].astype(np.int64)
        if short:
            p, r, mask = p[-64:], r[:64], mask[:64]
            mask = np.ones_like(mask)
        ids = torch.from_numpy(np.concatenate((p, r)))
        columns["input_ids"].append(ids)
        builder = SimpleNamespace(processor=processor, tokenizer=tokenizer, _get_mm_processor_kwargs=lambda audios: {})
        multimodal = AgentLoopWorker._compute_multi_modal_inputs(
            builder, SimpleNamespace(multi_modal_data={}, mm_processor_kwargs={}), ids
        )
        positions = AgentLoopWorker._compute_position_ids(
            builder, ids.unsqueeze(0), torch.ones_like(ids).unsqueeze(0), multimodal
        ).squeeze(0)
        columns["position_ids"].append(positions)
        for name, value in (("prompts", p), ("responses", r), ("response_mask", mask)):
            columns[name].append(torch.from_numpy(value.copy()))
    # Public jagged_dim preserves mRoPE's [planes, sequence] physical layout.
    data = TensorDict(
        {key: packed_nested(value) for key, value in columns.items()},
        batch_size=[len(rows)],
    )
    data["loss_mask"] = data["response_mask"]
    tu.assign_non_tensor(
        data,
        pad_mode=DatasetPadMode.NO_PADDING,
        temperature=1.0,
        global_batch_size=global_batch_size,
        use_remove_padding=False,
        use_dynamic_bsz=False,
        max_token_len_per_gpu=36864,
        micro_batch_size_per_gpu=1,
        use_fused_kernels=True,
        calculate_entropy=False,
        pad_token_id=248044,
    )
    return data


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def gradient_snapshot(module):
    return {
        name: local_tensor(p.grad).detach().cpu().clone() for name, p in module.named_parameters() if p.grad is not None
    }


def gradient_comparison(module, baseline, torch, dist):
    totals = torch.zeros(5, dtype=torch.float64, device="cuda")
    seen = set()
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        seen.add(name)
        current = local_tensor(p.grad).detach().reshape(-1)
        expected = baseline[name].reshape(-1)
        for start in range(0, current.numel(), 2_000_000):
            a = current[start : start + 2_000_000].float()
            b = expected[start : start + 2_000_000].to(a.device).float()
            delta = a - b
            totals[0] += delta.double().square().sum()
            totals[1] += a.double().square().sum()
            totals[2] += b.double().square().sum()
            totals[3] += (a.double() * b.double()).sum()
            totals[4] = torch.maximum(totals[4], delta.abs().max().double())
    if seen != set(baseline):
        raise ValueError("Gradient parameter coverage changed")
    maximum = totals[4].clone()
    dist.all_reduce(totals[:4], op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return {
        "relative_l2_error": float((totals[0] / totals[2].clamp_min(1e-30)).sqrt()),
        "cosine_similarity": float(totals[3] / (totals[1] * totals[2]).clamp_min(1e-30).sqrt()),
        "max_abs_error": float(maximum),
        "parameters_with_grad_per_rank": len(seen),
        "candidate_local_shard_l2": float(totals[1].sqrt()),
        "baseline_local_shard_l2": float(totals[2].sqrt()),
    }


def memory(torch):
    return {
        name: getattr(torch.cuda, function)() / 2**30
        for name, function in (
            ("allocated_gib", "memory_allocated"),
            ("reserved_gib", "memory_reserved"),
            ("peak_allocated_gib", "max_memory_allocated"),
            ("peak_reserved_gib", "max_memory_reserved"),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("actor", "reference"), required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("At least one measured repeat is required")
    import sys

    source = Path(os.environ.get("VERL_SOURCE_ROOT", "/workspace/verl-0.9.1"))
    sys.path.insert(0, str(source))
    from verl.experimental.skill_harness.bootstrap import configure_runtime

    configure_runtime(register_tasks=False)
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf, open_dict

    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine import EngineRegistry
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise ValueError("This benchmark requires the production four-rank FSDP group")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    torch.manual_seed(42)
    report_path = args.output / f"rank_{rank}.json"
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if report_path.exists():
        raise FileExistsError("Choose a new benchmark output directory")
    report = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "rank": rank,
        "role": args.role,
        "started_at": time.time(),
        "training_optimizer_updates": 0,
        "diagnostic_zero_lr_optimizer_steps": 0,
        "new_rollouts": 0,
        "co_resident_reference_and_actor": True,
        "co_resident_vllm": False,
        "scope": "official FSDP2 engines, archived tokens; independent diagnostic, no training checkpoint",
        "trials": [],
    }
    publish(report_path, report)
    phase = "prepare"
    try:
        cfg = OmegaConf.load(args.run / "resolved-config.yaml")
        with open_dict(cfg.actor_rollout_ref.actor):
            cfg.actor_rollout_ref.actor.rollout_n = cfg.actor_rollout_ref.rollout.n
            cfg.actor_rollout_ref.actor.optim.total_training_steps = 64
        actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        model_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
        actor_cfg.model_config = model_cfg
        actor_engine_cfg = actor_cfg.engine
        actor_engine_cfg.use_dynamic_bsz = False
        actor_engine_cfg.use_remove_padding = False
        actor_engine_cfg.use_fused_kernels = True
        actor_engine_cfg.micro_batch_size_per_gpu = 1
        actor_engine_cfg.infer_micro_batch_size_per_gpu = 1
        actor_engine_cfg.max_token_len_per_gpu = 36864
        actor_engine_cfg.infer_max_token_len_per_gpu = 36864
        ref_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.ref.fsdp_config)
        object.__setattr__(ref_cfg, "strategy", cfg.actor_rollout_ref.ref.strategy)
        ref_cfg.forward_only = True
        ref_cfg.use_dynamic_bsz = False
        ref_cfg.use_remove_padding = False
        ref_cfg.use_fused_kernels = True
        ref_cfg.infer_micro_batch_size_per_gpu = 1
        ref_cfg.infer_max_token_len_per_gpu = 36864
        selected = select_segments(args.run)
        rows = selected[rank::4]
        report.update(
            selected_segments=selected,
            rank_segments=rows,
            total_unpadded_tokens=sum(row["tokens"] for row in selected),
            actor_strategy=actor_engine_cfg.strategy,
            no_sync=actor_engine_cfg.use_no_sync_for_gradient_accumulation,
            packing_adapter="public jagged_dim=2 + fixed_microbatches; official engine forward_step/backward/loss",
            transferqueue_chunk_path_validated=False,
            config_sha256=digest(args.run / "resolved-config.yaml"),
            model_path=model_cfg.path,
            hardware=torch.cuda.get_device_name(),
            capacity_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        )
        assert not actor_engine_cfg.use_no_sync_for_gradient_accumulation
        for name, engine_cfg, optim_cfg in (("reference", ref_cfg, None), ("actor", actor_engine_cfg, actor_cfg.optim)):
            phase = "initialize_" + name
            report["active_phase"] = phase
            publish(report_path, report)
            engine = EngineRegistry.new(
                model_type="language_model",
                backend=engine_cfg.strategy,
                model_config=model_cfg,
                engine_config=engine_cfg,
                optimizer_config=optim_cfg,
                checkpoint_config=CheckpointConfig(),
            )
            engine.initialize()
            if name == "reference":
                reference = engine
            else:
                actor = engine
        phase = "initialize_diagnostic_optimizer_state"
        report["active_phase"] = phase
        publish(report_path, report)
        rates = [group["lr"] for group in actor.optimizer.param_groups]
        for group in actor.optimizer.param_groups:
            group["lr"] = 0.0
        with actor.train_mode():
            for p in actor.module.parameters():
                if p.requires_grad:
                    p.grad = torch.zeros_like(p)
            actor.optimizer_step()
        for group, rate in zip(actor.optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        report["diagnostic_zero_lr_optimizer_steps"] = 1
        report["optimizer_state_tensors"] = sum(len(state) for state in actor.optimizer.state.values())
        data = make_data(rows, len(selected), processor=model_cfg.processor, tokenizer=model_cfg.tokenizer)
        short = make_data(rows, len(selected), processor=model_cfg.processor, tokenizer=model_cfg.tokenizer, short=True)
        from verl.experimental.skill_harness.replay_batch import infer_position_id_planes

        report["position_id_planes"] = infer_position_id_planes(data)
        report["position_layout_provenance"] = "actual AgentLoopWorker position builder with actual model processor"
        report["processor_class"] = None if model_cfg.processor is None else type(model_cfg.processor).__name__
        target = actor if args.role == "actor" else reference
        # Warm up kernels outside measured full-trajectory trials.
        phase = "short_warmup"
        report["active_phase"] = phase
        publish(report_path, report)
        with target.eval_mode():
            warm = engine_batch(target, short)
        del warm, short
        torch.cuda.synchronize()
        baseline_logp = None
        all_infer = {}
        for mb in (1, 2, 4):
            tu.assign_non_tensor(data, micro_batch_size_per_gpu=mb)
            trials = []
            for repeat in range(args.repeats):
                phase = f"{args.role}_logprob_mb{mb}_repeat{repeat}"
                report["active_phase"] = phase
                publish(report_path, report)
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with target.eval_mode():
                    output = engine_batch(target, data)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                padded = no_padding_2_padding(output["model_output"]["log_probs"], data.to("cuda")).float().cpu()
                if not torch.isfinite(padded).all():
                    raise FloatingPointError("Non-finite diagnostic logprob")
                if baseline_logp is None:
                    baseline_logp = padded.clone()
                mask = data["response_mask"].to_padded_tensor(0).bool()
                delta = (padded - baseline_logp)[mask]
                comparison = {
                    "max_abs_error": float(delta.abs().max()),
                    "mean_abs_error": float(delta.abs().mean()),
                    "rmse": float(delta.square().mean().sqrt()),
                    "tolerance_max_abs": 0.10,
                    "tolerance_mean_abs": 0.01,
                }
                comparison["passed"] = comparison["max_abs_error"] <= 0.10 and comparison["mean_abs_error"] <= 0.01
                trial = {
                    "phase": phase,
                    "microbatch": mb,
                    "seconds": seconds,
                    "memory": memory(torch),
                    "logprob_comparison": comparison,
                }
                report["trials"].append(trial)
                trials.append(trial)
                publish(report_path, report)
                del output, padded
            all_infer[mb] = trials
        if args.role == "actor":
            # Actor likelihoods are recomputed on these exact fixed tokens.
            # Synthetic +/- advantages only probe the production PPO kernel;
            # they are not task rewards, GRPO estimates, or main-run updates.
            response_lengths = [row["response_tokens"] for row in rows]
            data["old_log_probs"] = torch.nested.as_nested_tensor(
                [baseline_logp[i, :n].clone() for i, n in enumerate(response_lengths)], layout=torch.jagged
            )
            data["ref_log_prob"] = data["old_log_probs"].clone()
            data["advantages"] = torch.nested.as_nested_tensor(
                [torch.full((n,), 0.5 if i % 2 else -0.5) for i, n in enumerate(response_lengths)], layout=torch.jagged
            )
            loss_function = partial(ppo_loss, config=actor_cfg)
            baseline_grad = None
            baseline_loss = None
            for mb in (1, 2):
                tu.assign_non_tensor(data, micro_batch_size_per_gpu=mb)
                for repeat in range(args.repeats):
                    phase = f"ppo_forward_backward_mb{mb}_repeat{repeat}"
                    report["active_phase"] = phase
                    publish(report_path, report)
                    gc.collect()
                    torch.cuda.empty_cache()
                    dist.barrier()
                    actor.optimizer_zero_grad()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with actor.train_mode(zero_grad_on_exit=False):
                        output = engine_batch(actor, data, loss_function=loss_function, forward_only=False)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - started
                    loss = sum(float(value) for value in output["loss"])
                    loss_tensor = torch.tensor(loss, device="cuda", dtype=torch.float64)
                    dist.all_reduce(loss_tensor)
                    loss = float(loss_tensor / 4)
                    if baseline_grad is None:
                        baseline_grad = gradient_snapshot(actor.module)
                        baseline_loss = loss
                    comparison = gradient_comparison(actor.module, baseline_grad, torch, dist)
                    comparison.update(
                        loss=loss,
                        baseline_loss=baseline_loss,
                        loss_abs_error=abs(loss - baseline_loss),
                        tolerance_relative_l2=0.03,
                        tolerance_cosine=0.999,
                    )
                    comparison["passed"] = (
                        comparison["relative_l2_error"] <= 0.03
                        and comparison["cosine_similarity"] >= 0.999
                        and comparison["loss_abs_error"] <= 0.005
                    )
                    trial = {
                        "phase": phase,
                        "microbatch": mb,
                        "seconds": seconds,
                        "memory": memory(torch),
                        "gradient_comparison": comparison,
                    }
                    report["trials"].append(trial)
                    publish(report_path, report)
                    actor.optimizer_zero_grad()
                    del output
            del baseline_grad
        report["status"] = "completed"
        report["finished_at"] = time.time()
        report["all_parity_checks_passed"] = all(
            trial.get("logprob_comparison", trial.get("gradient_comparison"))["passed"] for trial in report["trials"]
        )
        publish(report_path, report)
        dist.barrier()
    except BaseException as error:
        report.update(
            status="failed",
            failed_phase=phase,
            error_type=type(error).__name__,
            error=str(error)[:2000],
            traceback=traceback.format_exc()[-5000:],
            oom=isinstance(error, torch.cuda.OutOfMemoryError),
            finished_at=time.time(),
        )
        publish(report_path, report)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
