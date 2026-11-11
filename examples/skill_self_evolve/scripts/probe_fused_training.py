#!/usr/bin/env python3
"""Probe official fused PPO numerics on archived native Codex TITO tokens.

This is a training-kernel diagnostic, not an inference substitute or an RL run.
It performs no optimizer step, creates no new rollout, and never re-encodes the
archived token IDs. Only physical GPU 0 is permitted for the authorized probe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def gradient_comparison(torch, actual, expected):
    squared_error, squared_actual, squared_expected, dot, maximum = 0.0, 0.0, 0.0, 0.0, 0.0
    left, right = actual.detach().reshape(-1), expected.detach().reshape(-1)
    for start in range(0, left.numel(), 4_000_000):
        a, b = left[start : start + 4_000_000].float(), right[start : start + 4_000_000].float()
        delta = a - b
        squared_error += float(delta.square().sum())
        squared_actual += float(a.square().sum())
        squared_expected += float(b.square().sum())
        dot += float((a * b).sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {
        "max_absolute_error": maximum,
        "relative_l2_error": (squared_error / max(squared_expected, 1e-30)) ** 0.5,
        "cosine_similarity": dot / max((squared_actual * squared_expected) ** 0.5, 1e-30),
    }


def unfused_head(torch, hidden, weights, labels):
    logits = (hidden @ weights.t()).float()
    logp = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = (logits.logsumexp(-1) - (logits.softmax(-1) * logits).sum(-1)).to(hidden.dtype)
    return logp, entropy


def short_parity(torch, model, ids, positions, prefix_length):
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    length = min(prefix_length, ids.shape[-1])
    small, pos = ids[:, :length], positions[..., :length]
    mask = torch.ones_like(small)
    labels = small.roll(-1, -1)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        fused_output = model(
            input_ids=small, position_ids=pos, attention_mask=mask, use_cache=False, return_dict=True, temperature=1.0
        )
        hidden = model.model(input_ids=small, position_ids=pos, attention_mask=mask, use_cache=False, return_dict=True)[
            0
        ]
        reference_logp, reference_entropy = unfused_head(torch, hidden, model.lm_head.weight, labels)
        forward = {
            "logprob_max_abs_error": float((fused_output.log_probs - reference_logp).abs().max()),
            "entropy_bf16_max_abs_error": float((fused_output.entropy.float() - reference_entropy.float()).abs().max()),
        }
        hidden = hidden.detach()
    del fused_output, reference_logp, reference_entropy
    actual_hidden = hidden.clone().requires_grad_(True)
    actual_weights = model.lm_head.weight.detach().clone().requires_grad_(True)
    actual_logp, actual_entropy = FusedLinearForPPO(impl_backend="torch")(actual_hidden, actual_weights, labels)
    actual_loss = -actual_logp.float().mean() - 0.01 * actual_entropy.float().mean()
    actual_loss.backward()
    hidden_gradient, weight_gradient = actual_hidden.grad, actual_weights.grad
    del actual_loss, actual_logp, actual_entropy, actual_hidden, actual_weights
    expected_hidden = hidden.clone().requires_grad_(True)
    expected_weights = model.lm_head.weight.detach().clone().requires_grad_(True)
    expected_logp, expected_entropy = unfused_head(torch, expected_hidden, expected_weights, labels)
    (-expected_logp.float().mean() - 0.01 * expected_entropy.float().mean()).backward()
    hidden_error = gradient_comparison(torch, hidden_gradient, expected_hidden.grad)
    weight_error = gradient_comparison(torch, weight_gradient, expected_weights.grad)
    torch.cuda.synchronize()
    result = {
        "prefix_tokens": length,
        "forward": forward,
        "hidden_gradient": hidden_error,
        "weight_gradient": weight_error,
        "seconds": time.perf_counter() - started,
        "memory": memory(torch),
        "scope": "same real token prefix; complete patched model forward and exact same hidden-state head gradients",
    }
    result["passed"] = (
        forward["logprob_max_abs_error"] <= 0.02
        and forward["entropy_bf16_max_abs_error"] <= 0.125
        and hidden_error["relative_l2_error"] <= 0.03
        and weight_error["relative_l2_error"] <= 0.03
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/Qwen3.5-4B-nothink")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-prefix", default="traj0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-tokens", type=int, default=128)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES=0; this authorized diagnostic permits physical GPU 0 only")
    if args.parity_tokens < 2 or args.parity_tokens > 256:
        raise SystemExit("Unfused parity is restricted to 2..256 prefix tokens")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "report.json").exists():
        raise SystemExit("Preserve earlier evidence; choose a new diagnostic output directory")
    report = {
        "schema_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
        "scope": "official fused training-kernel diagnostic on archived native Codex trajectory",
        "optimizer_steps": 0,
        "new_rollouts": 0,
        "token_reencoding": False,
        "model_path": args.model,
        "physical_gpu": 0,
        "phases": {},
    }
    write_json(output / "report.json", report)
    torch, model = None, None
    phase = "bootstrap"
    try:
        source = Path(os.environ.get("VERL_SOURCE_ROOT", "/workspace/verl-0.9.1"))
        sys.path.insert(0, str(source))
        from verl.experimental.skill_harness.bootstrap import configure_runtime

        report["runtime"] = configure_runtime(register_tasks=False)
        import numpy as np
        import torch
        import transformers
        from transformers import AutoConfig

        from verl.models.transformers.monkey_patch import apply_monkey_patch
        from verl.utils.model import get_hf_auto_model_class

        if torch.cuda.device_count() != 1:
            raise RuntimeError("The diagnostic must see exactly one authorized GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(42)
        report.update(
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_capacity_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        )
        saved = np.load(args.trajectory, allow_pickle=False)
        prefix = args.trajectory_prefix
        prompt = saved[prefix + "_prompt_ids"]
        response = saved[prefix + "_response_ids"]
        response_mask = saved[prefix + "_response_mask"]
        if len(response) != len(response_mask) or not response_mask.any():
            raise ValueError("An archived trajectory must preserve aligned nonempty model-authored action masks")
        original_ids = np.concatenate([prompt, response]).astype(np.int64)
        ids = torch.from_numpy(original_ids).unsqueeze(0).to("cuda:0")
        positions = torch.arange(ids.shape[-1], device="cuda:0").view(1, 1, -1).expand(4, 1, -1)
        attention = torch.ones_like(ids)
        loss_mask = torch.zeros_like(ids, dtype=torch.float32)
        loss_mask[:, len(prompt) - 1 : -1] = torch.from_numpy(response_mask.copy()).to("cuda:0")
        report["trajectory"] = {
            "path": str(args.trajectory),
            "npz_sha256": sha256(args.trajectory),
            "prefix": prefix,
            "prompt_tokens": len(prompt),
            "response_tokens": len(response),
            "total_tokens": len(original_ids),
            "authored_action_tokens": int(response_mask.sum()),
            "input_ids_int64_sha256": hashlib.sha256(original_ids.tobytes()).hexdigest(),
            "response_mask_sha256": hashlib.sha256(response_mask.tobytes()).hexdigest(),
            "loss_alignment": (
                "output position prompt_len+j-1 predicts original response token j; final wrap position excluded"
            ),
        }
        phase = "model_load"
        report["active_phase"] = phase
        write_json(output / "report.json", report)
        started = time.perf_counter()
        config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        auto_class = get_hf_auto_model_class(config)
        model = auto_class.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": "cuda:0"},
            local_files_only=True,
            trust_remote_code=False,
        )
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=True, fused_kernels_backend="torch")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        report["phases"][phase] = {"seconds": time.perf_counter() - started, "memory": memory(torch)}
        report["model"] = {
            "class": type(model).__name__,
            "forward": model.forward.__func__.__module__ + "." + model.forward.__func__.__name__,
            "parameter_dtype": str(next(model.parameters()).dtype),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "gradient_checkpointing": model.is_gradient_checkpointing,
            "use_remove_padding": False,
            "fused_backend": model._verl_fused_kernels_backend,
            "official_fused_linear_chunk_size": 512,
        }
        phase = "short_prefix_parity"
        report["active_phase"] = phase
        write_json(output / "report.json", report)
        report["phases"][phase] = short_parity(torch, model, ids, positions, args.parity_tokens)
        if not report["phases"][phase]["passed"]:
            raise AssertionError("Short-prefix fused/unfused forward/backward exceeded the recorded BF16 tolerances")
        gc.collect()
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        model.train()
        phase = "long_forward"
        report["active_phase"] = phase
        write_json(output / "report.json", report)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = model(
            input_ids=ids,
            position_ids=positions,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
            temperature=1.0,
        )
        torch.cuda.synchronize()
        if result.logits is not None or result.log_probs.shape != ids.shape or result.entropy.shape != ids.shape:
            raise AssertionError(
                "Official fused model must return per-token likelihood and entropy without full logits"
            )
        if not torch.isfinite(result.log_probs).all() or not torch.isfinite(result.entropy).all():
            raise FloatingPointError("Nonfinite full-trajectory likelihood/entropy")
        report["phases"][phase] = {
            "seconds": time.perf_counter() - started,
            "memory": memory(torch),
            "logprob_finite": True,
            "entropy_finite": True,
            "logprob_dtype": str(result.log_probs.dtype),
            "entropy_dtype": str(result.entropy.dtype),
        }
        denominator = loss_mask.sum()
        nll = -(result.log_probs.float() * loss_mask).sum() / denominator
        entropy = (result.entropy.float() * loss_mask).sum() / denominator
        loss = nll - args.entropy_coefficient * entropy
        report["objective"] = {
            "name": "diagnostic authored-token masked NLL minus entropy coefficient times H",
            "nll": float(nll.detach()),
            "entropy": float(entropy.detach()),
            "entropy_coefficient": args.entropy_coefficient,
            "loss": float(loss.detach()),
            "is_grpo_update": False,
        }
        phase = "long_backward"
        report["active_phase"] = phase
        write_json(output / "report.json", report)
        started = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        report["phases"][phase] = {"seconds": time.perf_counter() - started, "memory": memory(torch)}
        squared_norm, parameters_with_grad, tensors_with_grad = 0.0, 0, 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Nonfinite model gradient after actual masked-loss backward")
                squared_norm += float(parameter.grad.float().square().sum())
                parameters_with_grad += parameter.numel()
                tensors_with_grad += 1
        report["gradients"] = {
            "all_present_gradients_finite": True,
            "global_l2_norm": squared_norm**0.5,
            "parameters_with_grad": parameters_with_grad,
            "tensors_with_grad": tensors_with_grad,
        }
        if not parameters_with_grad or squared_norm <= 0:
            raise AssertionError("Backward must produce a nonzero model gradient")
        report.update(status="passed", finished_at=time.time(), active_phase="complete")
    except BaseException as error:
        (output / "traceback.txt").write_text(traceback.format_exc())
        report.update(
            status="failed",
            failed_phase=phase,
            error_type=type(error).__name__,
            error=str(error),
            finished_at=time.time(),
        )
        if torch is not None and torch.cuda.is_initialized():
            report["failure_memory"] = memory(torch)
        raise
    finally:
        write_json(output / "report.json", report)
        if model is not None:
            model.zero_grad(set_to_none=True)
        model = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        print(
            json.dumps({"status": report["status"], "pid": os.getpid(), "report": str(output / "report.json")}),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
