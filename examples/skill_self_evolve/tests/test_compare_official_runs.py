"""Synthetic full-denominator CPU evidence; never model or benchmark results.

JSON rows deliberately replace only the parquet parser boundary. All case,
dump, raw Task/engine, numeric trajectory, budget and content-hash checks run
against retained ordinary temporary files.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import pprint
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


legacy = module("_crossrun_schema_fixture", DIRECTORY / "test_run_audit.py")
budget = module("_crossrun_budget_fixture", ROOT / "verl/experimental/skill_harness/alfworld/budget.py")
package = types.ModuleType("_crossrun_real_pure_harness")
package.__path__ = [str(ROOT / "verl/experimental/skill_harness")]
sys.modules[package.__name__] = package
context = importlib.import_module(package.__name__ + ".context")
public_protocol = importlib.import_module(package.__name__ + ".alfworld.protocol")
store_module = importlib.import_module(package.__name__ + ".store")

# Copied function source from the installed vLLM0.24.0 selector. This fixture
# is parsed as AST only; its body is never executed and vLLM is never imported.
# Normalized AST SHA256: dd37f5e0f8a5214ee62e7c54dbcdeadcb65babab287cdd0781c20cbf14b66568.
SELECTOR_SOURCE = """def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()
    if not current_platform.is_cuda():
        return backend, "triton"
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    supports_flashinfer = False
    supports_cutedsl = False
    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = _is_libs_cu13_install_intact()
        supports_cutedsl = True
        if not supports_flashinfer:
            logger.warning_once(
                "FlashInfer Blackwell GDN requires an intact nvidia-cutlass-dsl"
                "-libs-cu13 install, but some on-disk files do not match the "
                "SHA-256 declared in its RECORD (install-order race in "
                "nvidia-cutlass-dsl packaging -- see "
                "https://github.com/NVIDIA/cutlass/issues/3170 and "
                "https://github.com/NVIDIA/cutlass/issues/3259). Falling back "
                "to Triton/FLA. Repair with: pip install --force-reinstall "
                "--no-deps nvidia-cutlass-dsl-libs-cu13"
            )
    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"
"""


def owned_cli_log(model, *, flag="--gdn-prefill-backend", value="triton", complete=True, owned=True):
    selection = [flag] if value is None else [flag, value]
    args = ["serve", str(model), *selection, "--max_model_len", "36864", "--no-async_scheduling"]
    lines = pprint.pformat(args, width=32).splitlines()
    if not complete:
        lines = lines[:-1]
    prefix = "\x1b[36m(vLLMHttpServer pid=321)\x1b[0m " if owned else "(untrusted_actor pid=321) "
    return "WARN fixture: INFO-level selection was not emitted.\n" + "\n".join(prefix + line for line in lines) + "\n"


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, *, wrapped=False):
    legacy.write(Path(path), value, wrapped=wrapped)


def read(path):
    value = json.loads(Path(path).read_text())
    return value.get("payload", value) if isinstance(value, dict) else value


def json_parquet_reader(path, *args, **kwargs):
    """Explicit CPU fixture boundary; production defaults to real PyArrow."""
    return json.loads(Path(path).read_text())


class OfficialFixture(legacy.Fixture):
    def __init__(self, root, benchmark):
        super().__init__(root)
        self.benchmark = benchmark
        ids, games = {}, []
        for phase, names in self.ids.items():
            split = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}.get(phase, "train")
            ids[phase] = []
            for old in names:
                relative = f"json_2.1.1/{split}/pick_and_place_simple-{old}/trial/game.tw-pddl"
                content = old.encode()
                digest = hashlib.sha256(content).hexdigest()
                identity = split + "_" + hashlib.sha256((relative + "\0" + digest).encode()).hexdigest()[:24]
                game = {
                    "game_id": identity,
                    "game_sha256": digest,
                    "relative_game_path": relative,
                    "split": split,
                    "scene_id": "FixtureScene-" + old,
                    "task_type": "pick_and_place_simple",
                }
                source = benchmark / "data" / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(content)
                games.append(game)
                ids[phase].append(identity)
        self.ids, self.games = ids, games
        self.by_game = {game["game_id"]: game for game in games}
        self.catalog_digest = sha(sorted(games, key=lambda row: row["relative_game_path"]))
        write(root / "data/catalog.json", {"schema_version": 1, "games": games, "catalog_digest": self.catalog_digest})
        write(
            root / "data/manifest.json",
            {
                "schema_version": 1,
                "phase_game_ids": ids,
                "evaluation_games_used_for_training": False,
                "catalog_digest": self.catalog_digest,
            },
        )

    def episode(self, game, phase, attempt, won, **kwargs):
        directory, result = super().episode(game, phase, attempt, won, **kwargs)
        for name in ("attempt.json", "result.json", "engine-result.json"):
            value = read(directory / name)
            value.update(self.by_game[game])
            write(directory / name, value)
        return directory, read(directory / "result.json")


class MatchedPair:
    def __init__(self, root, compare):
        root = root.resolve()
        self.root = root
        self.benchmark = root / "benchmark"
        self.model = root / "shared-model"
        self.model.mkdir()
        write(
            self.model / "config.json", {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}
        )
        write(self.model / "tokenizer.json", {"model": {"type": "FixtureOnly", "vocab": {"fixture": 1}}})
        write(self.model / "tokenizer_config.json", {"chat_template": "fixture template; never executed"})
        write(self.model / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
        (self.model / "model.safetensors").write_bytes(b"Synthetic immutable weight identity, never loaded")
        tokenizer = hashlib.sha256()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
            path = self.model / name
            if path.exists():
                tokenizer.update(name.encode() + b"\0")
                tokenizer.update(path.read_bytes())
        self.tokenizer_version = "sha256:" + tokenizer.hexdigest()
        self.native = root / "shared-native-bin"
        self.native.write_bytes(b"Synthetic native binary identity, never executed")
        self.native.chmod(0o755)
        self.uni = root / "shared-uni/uni_agent"
        self.uni.mkdir(parents=True)
        (self.uni / "__init__.py").write_text("# Synthetic dependency identity, never imported\n")
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.dirty = hashlib.sha256(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)).hexdigest()
        fingerprint = compare._model_fingerprint(self.model)
        self.fingerprint = fingerprint["fingerprint"] if isinstance(fingerprint, dict) else fingerprint
        self.fixtures = {}
        for role, stage, cursor, version in (("base", "initial", 1, 0), ("trained", "final", 65, 64)):
            run = root / role
            fixture = OfficialFixture(run, self.benchmark)
            fixture.sidecar_evaluations()
            self.fixtures[role] = fixture
            self._adapt(fixture, stage, cursor, version)
        self.base = root / "base/evaluation/initial"
        self.trained = root / "trained/evaluation/final"

    def _adapt(self, fixture, stage, cursor, version):
        run = fixture.root
        selected = run / "evaluation" / stage
        other = "final" if stage == "initial" else "initial"
        shutil.rmtree(run / "evaluation" / other)
        other_step = 3 if other == "final" else 1
        shutil.rmtree(run / "agent_logs" / f"step_{other_step}")
        old_cursor = 1 if stage == "initial" else 3
        if cursor != old_cursor:
            (run / "agent_logs" / f"step_{old_cursor}").rename(run / "agent_logs" / f"step_{cursor}")
        for directory in (run / "episodes").glob("*/attempt_*"):
            if read(directory / "result.json").get("evaluation_stage") == other:
                shutil.rmtree(directory)

        public = run / "public-runtime"
        public.mkdir()
        wrapper = public / "codex-entrypoint"
        wrapper.write_bytes((ROOT / "examples/skill_self_evolve/scripts/codex_entrypoint.sh").read_bytes())
        instructions = public / "solver_system.md"
        instructions.write_bytes((ROOT / "examples/skill_self_evolve/prompts/solver_system.md").read_bytes())
        prompt = (ROOT / "examples/skill_self_evolve/prompts/solver_task.txt").read_text()
        catalog = {
            "models": [
                {
                    "slug": "Qwen3.5-4B-nothink",
                    "display_name": "Qwen3.5-4B-nothink",
                    "context_window": 36864,
                    "supports_parallel_tool_calls": True,
                    "input_modalities": ["text"],
                    "base_instructions": instructions.read_text(),
                }
            ]
        }
        write(run / "model_catalog.json", catalog)
        task_config = [
            {
                "name": "alfworld_skill_training",
                "sandbox": {"provider": "local", "runtime_timeout": 900},
                "agent": {
                    "name": "codex",
                    "run_timeout": 900,
                    "codex_bin": str(wrapper),
                    "model_catalog_path": str(run / "model_catalog.json"),
                },
            }
        ]
        write(run / "task_config.yaml", task_config)
        genesis = {"schema_version": 1, "parent_id": None, "skills": [], "provenance": {"operation": "genesis"}}
        snapshot = sha(genesis)
        write(run / "skill_store/refs/main.json", {"branch": "main", "snapshot_id": snapshot})
        write(run / "skill_store/snapshots" / (snapshot + ".json"), genesis)
        loading = {"enabled": True, "snapshot_id": snapshot, "injected_refs": []}
        store = store_module.SkillStore(run / "skill_store")

        parquet_files = {}
        for phase, identities in fixture.ids.items():
            values = []
            for index, game in enumerate(identities):
                metadata = {
                    **fixture.by_game[game],
                    "stable_game_id": game,
                    "catalog_digest": fixture.catalog_digest,
                    "dataset_phase": phase,
                    "run_dir": str(run),
                }
                values.append(
                    {
                        "data_source": "alfworld",
                        "prompt": [{"role": "user", "content": prompt}],
                        "ability": "agent",
                        "reward_model": {"style": "rule", "ground_truth": "environment_won"},
                        "extra_info": {
                            "index": index,
                            "split": phase,
                            "game_id": game,
                            "tools_kwargs": {"task": {"name": "alfworld_skill_training", "metadata": metadata}},
                        },
                    }
                )
            path = run / "data" / (phase + ".parquet")
            write(path, values)
            parquet_files[phase] = {"path": str(path), "sha256": file_sha(path)}
        manifest = {
            "schema_version": 1,
            "benchmark_root": str(self.benchmark),
            "run_dir": str(run),
            "catalog_digest": fixture.catalog_digest,
            "seed": 42,
            "dev_fraction": 0.1,
            "development_grouping": "whole training scenes, deterministic seeded hash order",
            "task_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "phase_game_ids": fixture.ids,
            "parquet_files": parquet_files,
            "available": {p: {"games": len(ids)} for p, ids in fixture.ids.items()},
            "selected": {p: {"games": len(ids)} for p, ids in fixture.ids.items()},
            "no_synthetic_tasks": True,
            "evaluation_games_used_for_training": False,
        }
        write(run / "data/split_manifest.json", manifest)

        config = {
            "actor_rollout_ref": {
                "model": {"path": str(self.model), "tokenizer_path": str(self.model)},
                "rollout": {
                    "mode": "async",
                    "seed": 42,
                    "n": 4,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "prompt_length": 4096,
                    "response_length": 32768,
                    "max_model_len": 36864,
                    "enforce_eager": True,
                    "gpu_memory_utilization": 0.55,
                    "max_num_batched_tokens": 8192,
                    "max_num_seqs": 32,
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                    "calculate_log_probs": True,
                    "full_determinism": False,
                    "val_kwargs": {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
                    "engine_kwargs": {
                        "vllm": {"async_scheduling": False, "gdn_prefill_backend": "triton"},
                        "sglang": {},
                        "trtllm": {},
                    },
                    "agent": {
                        "agent_loop_manager_class": "verl.experimental.skill_harness.framework.SkillAwareRolloutAdapter"
                    },
                    "custom": {
                        "agent_framework": {
                            "gateway_count": 8,
                            "max_concurrent_tasks": 64,
                            "allowed_request_sampling_param_keys": [],
                        }
                    },
                },
            },
            "data": {
                "train_files": str(run / "data/train.parquet"),
                "val_files": str(run / "data/dev.parquet"),
                "train_max_samples": -1,
                "val_max_samples": -1,
                "train_batch_size": 32,
                "val_batch_size": 512,
                "apply_chat_template_kwargs": {"enable_thinking": False},
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 4,
                "total_epochs": 0 if version == 0 else 4,
                "total_training_steps": 64,
                "resume_mode": "disable",
                "val_only": False,
                "val_before_train": False,
                "experiment_name": run.name,
                "default_local_dir": str(run / "checkpoints"),
            },
            "skill_harness": {
                "mode": "skill_joint",
                "artifact_root": str(run),
                "store_root": str(run / "skill_store"),
                "replay_root": str(run / "curate_replay"),
                "ledger_root": str(run / "credit_ledger"),
                "split_manifest": str(run / "data/split_manifest.json"),
                "terminal": {"enabled": True, "binding_root": str(run / "terminal_bindings")},
                "evaluation": {"initial": stage == "initial", "final": stage == "final"},
                "loading": {"enabled": True, "branch": "main", "top_k": 6, "skill_token_budget": 1024},
                "joint": {
                    "solver_gpus": 4,
                    "available_gpus": 8,
                    "curator_gpus": 2,
                    "credit_gpus": 2,
                    "route_tensor_parallel_size": 1,
                    "gateway_count": 2,
                    "seed": 42,
                    "required_exposures": 2,
                    "min_delay_steps": 2,
                    "probes_per_step": 4,
                    "frozen_model_path": str(self.model),
                },
            },
        }
        provenance = {
            "verl_version": "0.9.1",
            "verl_source": str(ROOT),
            "verl_origin": str(ROOT / "verl/__init__.py"),
            "uni_agent_source": str(self.uni.parent),
            "uni_agent_origin": str(self.uni / "__init__.py"),
            "uni_agent_adapter_sha256": "a" * 64,
            "python_version": "3.11.13",
        }
        write(
            run / "training-manifest.json",
            {
                "schema_version": 1,
                "source_commit": self.commit,
                "dirty_diff_sha256": self.dirty,
                "import_provenance": provenance,
                "config": config,
            },
        )
        env = {
            "MODEL_PATH": str(self.model),
            "VERL_SOURCE_ROOT": str(ROOT),
            "CODEX_NATIVE_BIN": str(self.native),
            "ALFWORLD_TASK_CONFIG": str(run / "task_config.yaml"),
            "ALFWORLD_SOLVER_INSTRUCTIONS_FILE": str(instructions),
            "ALFWORLD_BENCHMARK_ROOT": str(self.benchmark),
            "ALFWORLD_SERVER_CONFIG": str(run / "runtime/environment.json"),
        }
        write(
            run / "launch_manifest.json",
            {
                "run_dir": str(run),
                "environment_paths": env,
                "solver_system_sha256": file_sha(instructions),
                "solver_task_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            },
        )
        runtime = {
            "native_version": "0.167.0",
            "template_version": "sha256:" + "b" * 64,
            "model_name": "Qwen3.5-4B-nothink",
            "disabled_features": ["multi_agent", "view_image"],
            "files": {
                "codex_bin": file_sha(self.native),
                "model_catalog_path": file_sha(run / "model_catalog.json"),
                "instructions_path": file_sha(instructions),
                "task_prompt_path": "c" * 64,
                "schema_path": "d" * 64,
            },
        }
        write(
            run / "preflight.json",
            {
                "passed": True,
                "imports": provenance,
                "model": {"path": str(self.model)},
                "curator_runtime": runtime,
                "joint_gpu_allocation": {"solver": 4, "curator": 2, "credit": 2},
                "package_versions": {
                    "verl": "0.9.1",
                    "vllm": "0.24.0",
                    "torch": "2.11.0+cu130",
                    "transformers": "5.10.4",
                    "ray": "2.50.1",
                    "transferqueue": "0.1.12",
                    "uni-agent": "fixture-source-tree",
                },
            },
        )
        write(
            run / "experiment_plan.json",
            {
                "source_commit": self.commit,
                "model": {"path": str(self.model), "initial_model_fingerprint": self.fingerprint},
            },
        )
        state = {
            "schema_version": 1,
            "optimizer_step": version,
            "prepared_update": None,
            "collections": {},
            "pending": {},
            "cohorts": {},
            "configuration": config["skill_harness"]["joint"],
            "applied_update_ids": {f"update_{i}": f"optimizer:{i}:update_{i}" for i in range(1, version + 1)},
        }
        write(run / "controller/state.json", state, wrapped=True)
        write(run / "status.json", {"state": "completed", "exit_code": 0})
        (run / "training.log").write_text(
            "Synthetic CPU log schema only; no worker or kernel executed.\n"
            "\x1b[36m(vLLMHttpServer pid=321)\x1b[0m (Worker pid=654) INFO "
            "[qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).\n"
        )

        protocol = read(selected / "protocol.json")
        protocol.update(
            global_step=cursor,
            dataset={
                "catalog_digest": fixture.catalog_digest,
                "manifest_sha256": file_sha(run / "data/split_manifest.json"),
                "catalog_sha256": file_sha(run / "data/catalog.json"),
                "splits": {p: {**parquet_files[p], "games": len(fixture.ids[p])} for p in ("test_seen", "test_unseen")},
            },
            budgets={
                "sampling": config["actor_rollout_ref"]["rollout"]["val_kwargs"],
                "prompt_length": 4096,
                "response_length": 32768,
                "max_model_len": 36864,
                "environment_max_steps": 50,
                "alfworld_task_config_sha256": file_sha(run / "task_config.yaml"),
                "alfworld_solver_instructions_file_sha256": file_sha(instructions),
            },
        )
        write(selected / "protocol.json", protocol)
        summary = read(selected / "summary.json")
        summary.update(schema_version=1, global_step=cursor, authoritative={}, metrics={})
        for phase in ("test_seen", "test_unseen"):
            report_path = Path(summary["case_audits"][phase])
            report = read(report_path)
            old_dump = Path(report["dump_path"])
            dumped = [json.loads(line) for line in old_dump.read_text().splitlines()]
            old_dump.unlink()
            report.update(
                global_cursor=cursor,
                observed_weight_versions=[version],
                dump_path=str(report_path.parent / f"{cursor}.jsonl"),
            )
            for index, case in enumerate(report["cases"]):
                old = Path(case["episode_output_dir"])
                attempt = sha([run.name, phase, case["game_id"]])[:32]
                output = old.parent / ("attempt_" + attempt)
                old.rename(output)
                case.update(
                    attempt_id=attempt,
                    episode_output_dir=str(output),
                    backend_stop_confirmed=True,
                    observed_budget_failure=False,
                )
                for reward_context in case["reward_context"]:
                    reward_context["episode_output_dir"] = str(output)
                for row in dumped:
                    if row["uid"] in case["trajectory_keys"]:
                        row["step"] = cursor
                for name in ("attempt.json", "result.json", "engine-result.json"):
                    value = read(output / name)
                    value.update(attempt_id=attempt, global_step=cursor, run_dir=str(run))
                    if name != "attempt.json":
                        value.update(done=bool(value["won"]), max_steps=50)
                    if name == "result.json":
                        value.update(
                            native_finished=True,
                            skill_context=loading,
                            backend_stop_confirmed=True,
                            environment_session_id=sha([attempt, "env"])[:32],
                            budget_exhausted=False,
                            task_cancelled=False,
                            budget_reward_eligible=False,
                        )
                    write(output / name, value)
                initial = public_protocol.public_state(
                    {"feedback": "Same fixture kitchen", "admissible_commands": ["look", "inventory"]},
                    steps=0,
                    max_steps=50,
                    won=False,
                    done=False,
                )
                write(output / "initial-state.json", initial)
                original_messages = public_protocol.agent_messages([{"role": "user", "content": prompt}], initial)

                def render(messages):
                    text = json.dumps(messages, sort_keys=True, ensure_ascii=False)
                    return context.RenderedPrompt(text, tuple(text.encode()))

                built = context.build_skill_context(
                    original_messages,
                    store=store,
                    snapshot_id=snapshot,
                    task_family="pick_and_place_simple",
                    render_prompt=render,
                    encode_text=lambda text: list(text.encode()),
                    max_prompt_tokens=4096,
                    skill_token_budget=1024,
                    tokenizer_version=self.tokenizer_version,
                    group_id=sha([run.name, case["game_id"]]),
                )
                write(output / "skill-context.json", built.manifest)
                write(output / "agent-messages.json", built.messages)
                (output / "agent-prompt.txt").write_text("\n\n".join(m["content"] for m in built.messages) + "\n")
                result = read(output / "result.json")
                result["skill_context"] = built.manifest
                write(output / "result.json", result)
                self._registration(run, output, result)
                gateway = run / "agent_logs" / f"step_{cursor}" / f"{stage}-{phase}-{index}"
                metadata = read(gateway / "trajectory.json")
                metadata["trajectories"] = [
                    {
                        "finished": True,
                        "reward_score": case["authoritative_reward"],
                        "reward_metrics": {"acc": case["authoritative_accuracy"]},
                        "model_token_count": 1,
                        "prompt_len": 2,
                        "response_len": 1,
                        "has_logprobs": True,
                    }
                ]
                write(gateway / "trajectory.json", metadata)
                with zipfile.ZipFile(gateway / "trajectory.npz", "a") as archive:
                    header = repr({"descr": "<f8", "fortran_order": False, "shape": (1,)}).encode()
                    header += b" " * ((-len(header) - 11) % 16) + b"\n"
                    archive.writestr(
                        "traj0_response_logprobs.npy",
                        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<d", -0.2),
                    )
            legacy.rows(Path(report["dump_path"]), dumped)
            report["parent_metrics"] = {
                "val-core/alfworld/acc/mean@1": report["authoritative_accuracy"],
                "val-aux/alfworld/reward/mean@1": report["authoritative_reward_mean"],
            }
            write(report_path, report)
            summary["authoritative"][phase] = {
                key: report[key]
                for key in (
                    "expected_cases",
                    "authoritative_wins",
                    "authoritative_reward_mean",
                    "authoritative_accuracy",
                    "native_error_cases",
                    "observed_weight_versions",
                )
            }
            summary["metrics"][phase] = {
                "val-core/alfworld/acc/mean@1": report["authoritative_accuracy"],
                "val-aux/alfworld/reward/mean@1": report["authoritative_reward_mean"],
            }
            write(
                report_path.parent / "metrics.json",
                {
                    "schema_version": 1,
                    "stage": stage,
                    "phase": phase,
                    "global_step": cursor,
                    "expected_games": len(fixture.ids[phase]),
                    "metrics": summary["metrics"][phase],
                    "case_audit_path": str(report_path),
                    "authoritative": summary["authoritative"][phase],
                },
            )
        write(selected / "summary.json", summary)
        write(
            selected / "environment-protocol.json",
            {
                "catalog_digest": fixture.catalog_digest,
                "max_steps": 50,
                "allowed_splits": ["valid_seen", "valid_unseen"],
            },
        )

    def _registration(self, run, output, task):
        attempt = task["attempt_id"]
        binding = {
            "binding_id": sha([attempt, "binding"]),
            "binding_sha256": sha([attempt, "private_binding"]),
            "attempt_id": attempt,
            "gateway_session_id": "session_" + attempt,
            "environment_session_id": task["environment_session_id"],
        }
        account = {
            "lease_id": sha([attempt, "lease"])[:32],
            "uid": 20000,
            "gid": 20000,
            "name": "alfworld_ep2_000",
            "owner": {"kind": "ordinary", "run_root": str(run), "scope_id": attempt},
        }
        reg = budget.make_registration(
            {"schema_version": 1, "budget_seconds": 900, "binding": binding, "account": account},
            session_id=binding["environment_session_id"],
            run_root=run,
            output=output,
            server_instance_id=sha([run.name, "server"])[:32],
            started_ns=1_000_000_000,
        )
        write(output / "budget-registration.json", reg)
        private_binding = {key: value for key, value in binding.items() if key != "binding_sha256"}
        private_binding.update(
            capability="a" * 64,
            environment_session_url="http://127.0.0.1:1/session/" + binding["environment_session_id"],
        )
        binding["binding_sha256"] = sha(private_binding)
        # Recreate the registration with the exact retained private binding.
        reg = budget.make_registration(
            {"schema_version": 1, "budget_seconds": 900, "binding": binding, "account": account},
            session_id=binding["environment_session_id"],
            run_root=run,
            output=output,
            server_instance_id=sha([run.name, "server"])[:32],
            started_ns=1_000_000_000,
        )
        write(output / "budget-registration.json", reg)
        binding_dir = run / "terminal_bindings" / binding["binding_id"]
        write(binding_dir / "binding.json", private_binding, wrapped=True)
        request = {
            "binding_sha256": binding["binding_sha256"],
            "request_nonce": sha([attempt, "nonce"])[:32],
            "admission_index": 0,
            "attempt_id": attempt,
            "gateway_session_id": binding["gateway_session_id"],
            "server_id": "127.0.0.1:1",
            "actual_request_id": sha([attempt, "request"])[:32],
            "phase": "completed",
            "backend_rpc_finished": True,
            "error_type": None,
            "stop_kind": None,
            "raw_stop_reason": "completed",
            "abort_attempts": 0,
            "abort_acknowledged": False,
            "tail_archive_ref": None,
        }
        write(binding_dir / "requests" / (request["request_nonce"] + ".json"), request, wrapped=True)
        write(
            binding_dir / "admission-counter.json",
            {"binding_sha256": binding["binding_sha256"], "next_index": 1},
            wrapped=True,
        )
        backend = {
            "schema_version": 1,
            **binding,
            "status": "drained",
            "stop_kind": None,
            "fenced": False,
            "active_requests": 0,
            "backend_stop_confirmed": True,
            "backend_error_types": [],
            "requests": [request],
            "tail_archive_refs": [],
        }
        write(output / "backend-terminal-drain.json", backend)
        state = {"won": task["won"], "done": task["done"], "steps": 1, "max_steps": 50}
        event = {
            "schema_version": 1,
            "registration_sha256": reg["registration_sha256"],
            "sequence": 1,
            "kind": "admitted",
            "checked_ns": reg["started_ns"] + 1,
            "completed_ns": reg["started_ns"] + 2,
            "step": 1,
            "action_sha256": hashlib.sha256(b"look").hexdigest(),
            "won_after": task["won"],
            "done_after": task["done"],
        }
        legacy.rows(output / "budget-events.jsonl", [event])
        final = {
            **reg,
            "now_ns": reg["started_ns"] + 100,
            "server_closed": True,
            "closed_ns": reg["started_ns"] + 100,
            "close_reason": "agent_finished",
            "deadline_reached": False,
            "world_state": state,
            "world_state_sha256": sha(state),
            "admitted_actions": 1,
            "rejected_after_deadline": 0,
            "last_admitted_ns": event["checked_ns"],
            "last_completed_ns": event["completed_ns"],
            "last_sequence": 1,
            "action_error_count": 0,
            "events_sha256": file_sha(output / "budget-events.jsonl"),
        }
        write(output / "budget-environment.json", final)
        engine = read(output / "engine-result.json")
        engine["episode_budget"] = final
        write(output / "engine-result.json", engine)
        write(
            output / "native-capture-diagnostics.jsonl",
            {
                "schema_version": 1,
                "pid": 12345,
                "native_started_ns": reg["started_ns"] + 1,
                "declared_budget_seconds": 900,
                "budget_registration_sha256": reg["registration_sha256"],
                "budget_started_ns": reg["started_ns"],
                "budget_deadline_ns": reg["deadline_ns"],
                "native_exit_code": 0,
                "returned_exit_code": 0,
                "timed_out": False,
                "cancelled": False,
                "drain_timed_out": False,
                "terminal_monitor_error": None,
            },
        )
        archive = output / "native-files.tar.gz"
        archive.write_bytes(b"Synthetic archive receipt bytes only, no native process executed")
        manifest = output / "native-files-manifest.json"
        write(manifest, {"fixture": "CPU-only readback schema"})
        receipt = {
            "schema_version": 1,
            "status": "archived_and_removed",
            "source_stats_stable": True,
            "account_lease_id": account["lease_id"],
            "agent_uid": account["uid"],
            "archive_path": str(archive),
            "archive_sha256": file_sha(archive),
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": file_sha(manifest),
            "sources": [
                {"kind": kind, "path": str(output / ("removed-" + kind)), "removed": True}
                for kind in ("codex_home", "workspace", "model_catalog")
            ],
        }
        write(output / "native-files.json", receipt)
        task.update(
            backend_binding=binding,
            backend_terminal=backend,
            native_files=receipt,
            backend_terminal_error=None,
            account_isolation_error=None,
            account_isolation={
                **account,
                "policy_version": "alfworld-uid-landlock-v1",
                "status": "attempt_closed",
                "backend_drained": True,
                "uid_drained": True,
                "archive_receipt": str(output / "native-files.json"),
            },
        )
        write(output / "result.json", task)


class CompareOfficialRunsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = module("_compare_official_cpu", DIRECTORY.parent / "scripts/compare_official_runs.py")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.pair = MatchedPair(Path(cls.temporary.name), cls.compare)

    def compare_pair(self):
        return self.compare.compare_stages(
            self.pair.base,
            self.pair.trained,
            expected_base_fingerprint=self.pair.fingerprint,
            parquet_reader=json_parquet_reader,
        )

    def change(self, path, callback):
        path = Path(path)
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        value = read(path)
        callback(value)
        write(path, value)

    def assert_incomplete(self, report):
        self.assertFalse(report["comparison_supported"])
        self.assertFalse(report["improvement_supported"])
        self.assertTrue(report["reasons"])

    def test_distinct_host_roots_full_cases_and_original_binary_gains_are_supported(self):
        before = {str(path): file_sha(path) for path in self.pair.root.rglob("*") if path.is_file()}
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertTrue(report["improvement_supported"])
        self.assertEqual(report["pooled"]["base_wins"], 137)
        self.assertEqual(report["pooled"]["trained_wins"], 139)
        self.assertAlmostEqual(report["pooled"]["reward_delta"], 2 / 274)
        self.assertEqual(report["pooled"]["reward_delta"], report["pooled"]["accuracy_delta"])
        self.assertEqual(before, {str(path): file_sha(path) for path in self.pair.root.rglob("*") if path.is_file()})

    def test_declared_900_budget_mismatch_is_not_ignored_as_host_metadata(self):
        path = self.pair.root / "trained/task_config.yaml"
        self.change(path, lambda value: value[0]["agent"].update(run_timeout=899))
        # Keep the recorded hash internally consistent: rejection must come
        # from the actual declared budget, not merely a stale file checksum.
        self.change(
            self.pair.trained / "protocol.json",
            lambda value: value["budgets"].update(alfworld_task_config_sha256=file_sha(path)),
        )
        self.assert_incomplete(self.compare_pair())

    def test_backend_drift_is_not_an_allowed_policy_change(self):
        path = self.pair.root / "trained/training-manifest.json"
        self.change(
            path,
            lambda value: value["config"]["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"].update(
                gdn_prefill_backend="flashinfer"
            ),
        )
        self.assert_incomplete(self.compare_pair())

    def test_actual_initial_world_drift_is_rejected(self):
        summary = read(self.pair.trained / "summary.json")
        report = read(summary["case_audits"]["test_seen"])
        output = Path(report["cases"][0]["episode_output_dir"])
        for name in (
            "initial-state.json",
            "skill-context.json",
            "agent-messages.json",
            "agent-prompt.txt",
            "result.json",
        ):
            path = output / name
            self.addCleanup(path.write_bytes, path.read_bytes())
        initial = read(output / "initial-state.json")
        initial["observation"] = "A different private world"
        write(output / "initial-state.json", initial)
        task = read(output / "result.json")
        records = json_parquet_reader(self.pair.root / "trained/data/test_seen.parquet")
        row = next(value for value in records if value["extra_info"]["game_id"] == task["game_id"])
        original = public_protocol.agent_messages(row["prompt"], initial)
        previous = task["skill_context"]

        def render(messages):
            text = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            return context.RenderedPrompt(text, tuple(text.encode()))

        built = context.build_skill_context(
            original,
            store=store_module.SkillStore(self.pair.root / "trained/skill_store", read_only=True),
            snapshot_id=previous["snapshot_id"],
            task_family="pick_and_place_simple",
            render_prompt=render,
            encode_text=lambda text: list(text.encode()),
            max_prompt_tokens=4096,
            skill_token_budget=1024,
            tokenizer_version=self.pair.tokenizer_version,
            group_id=previous["group_id"],
        )
        write(output / "skill-context.json", built.manifest)
        write(output / "agent-messages.json", built.messages)
        (output / "agent-prompt.txt").write_text("\n\n".join(m["content"] for m in built.messages) + "\n")
        task["skill_context"] = built.manifest
        write(output / "result.json", task)
        # Every source-prompt/context hash now agrees internally; the mismatch
        # is the privately observed world across the two original runs.
        self.assert_incomplete(self.compare_pair())

    def test_game_identity_drift_cannot_hide_in_complete_case_counts(self):
        summary = read(self.pair.trained / "summary.json")
        self.change(
            summary["case_audits"]["test_seen"],
            lambda value: value["cases"][0].update(game_id=value["cases"][1]["game_id"]),
        )
        self.assert_incomplete(self.compare_pair())

    def test_forged_authoritative_summary_does_not_replace_raw_outcomes(self):
        self.change(
            self.pair.trained / "summary.json",
            lambda value: value["authoritative"]["test_seen"].update(
                authoritative_wins=140, authoritative_reward_mean=1.0, authoritative_accuracy=1.0
            ),
        )
        self.assert_incomplete(self.compare_pair())

    def test_forged_dump_metric_does_not_replace_raw_engine_won(self):
        report = read(read(self.pair.trained / "summary.json")["case_audits"]["test_seen"])
        path = Path(report["dump_path"])
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        rows = [json.loads(line) for line in original.splitlines()]
        rows[0].update(score=0.123, reward=0.123, acc=0.123)
        legacy.rows(path, rows)
        self.assert_incomplete(self.compare_pair())

    def test_unknown_backend_config_is_rejected(self):
        path = self.pair.root / "trained/training-manifest.json"
        self.change(
            path,
            lambda value: value["config"]["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"].update(
                unknown_context_rewriter=True
            ),
        )
        self.assert_incomplete(self.compare_pair())

    def test_kernel_package_version_drift_is_rejected(self):
        self.change(
            self.pair.root / "trained/preflight.json",
            lambda value: value["package_versions"].update(vllm="0.24.1-fixture"),
        )
        self.assert_incomplete(self.compare_pair())

    def test_official_parquet_order_drift_is_rejected_after_rehashing(self):
        run = self.pair.root / "trained"
        path = run / "data/test_seen.parquet"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        values = json.loads(original)
        values.reverse()
        write(path, values)
        manifest_path = run / "data/split_manifest.json"
        self.change(manifest_path, lambda value: value["parquet_files"]["test_seen"].update(sha256=file_sha(path)))

        def update_protocol(value):
            value["dataset"]["manifest_sha256"] = file_sha(manifest_path)
            value["dataset"]["splits"]["test_seen"]["sha256"] = file_sha(path)

        self.change(self.pair.trained / "protocol.json", update_protocol)
        self.assert_incomplete(self.compare_pair())

    def test_unlisted_host_path_change_is_rejected(self):
        path = self.pair.root / "trained/training-manifest.json"
        self.change(path, lambda value: value["config"]["data"].update(val_files="/another-host/dev.parquet"))
        self.assert_incomplete(self.compare_pair())

    def test_unknown_protocol_field_is_not_normalized_away(self):
        self.change(
            self.pair.trained / "protocol.json", lambda value: value.update(unrecognized_context_policy="ignore")
        )
        self.assert_incomplete(self.compare_pair())

    def test_unknown_split_metadata_drift_is_rejected_after_rehashing(self):
        path = self.pair.root / "trained/data/split_manifest.json"
        self.change(path, lambda value: value.update(unknown_curriculum_policy="changed"))
        self.change(
            self.pair.trained / "protocol.json",
            lambda value: value["dataset"].update(manifest_sha256=file_sha(path)),
        )
        self.assert_incomplete(self.compare_pair())

    def test_missing_case_audit_is_incomplete_rather_than_zero(self):
        self.change(self.pair.trained / "summary.json", lambda value: value["case_audits"].pop("test_unseen"))
        self.assert_incomplete(self.compare_pair())

    def test_zero_step_horizon_with_nonzero_epochs_is_an_untouched_base(self):
        self.change(
            self.pair.root / "base/training-manifest.json",
            lambda value: value["config"]["trainer"].update(total_training_steps=0, total_epochs=4),
        )
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertTrue(report["improvement_supported"])

    def test_trained_weight_bytes_need_not_equal_immutable_base_bytes(self):
        temporary = tempfile.TemporaryDirectory(dir=self.pair.root)
        self.addCleanup(temporary.cleanup)
        trained_model = Path(temporary.name) / "trained-model"
        shutil.copytree(self.pair.model, trained_model)
        (trained_model / "model.safetensors").write_bytes(b"Different synthetic trained parameter bytes; never loaded")
        self.assertNotEqual(self.compare._model_fingerprint(trained_model)["fingerprint"], self.pair.fingerprint)
        self.change(
            self.pair.root / "trained/training-manifest.json",
            lambda value: value["config"]["actor_rollout_ref"]["model"].update(path=str(trained_model)),
        )
        self.change(
            self.pair.root / "trained/launch_manifest.json",
            lambda value: value["environment_paths"].update(MODEL_PATH=str(trained_model)),
        )
        self.change(
            self.pair.root / "trained/preflight.json", lambda value: value["model"].update(path=str(trained_model))
        )
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertFalse(report["identity_scope"]["trained_parameter_fingerprint_compared_to_base"])

    def test_warn_only_owned_cli_supports_comparison_without_claiming_kernel_log(self):
        temporary = tempfile.TemporaryDirectory(dir=self.pair.root)
        self.addCleanup(temporary.cleanup)
        selector = Path(temporary.name) / "qwen_gdn_linear_attn.py"
        selector.write_text(SELECTOR_SOURCE)
        for role, flag in (("base", "--gdn_prefill_backend"), ("trained", "--gdn-prefill-backend")):
            path = self.pair.root / role / "training.log"
            self.addCleanup(path.write_bytes, path.read_bytes())
            path.write_text(owned_cli_log(self.pair.model, flag=flag))
        with mock.patch.object(self.compare, "_installed_gdn_selector", return_value=("0.24.0", selector)):
            report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        for role in ("base", "trained"):
            evidence = report["backend_evidence"][role]
            self.assertFalse(evidence["kernel_log_observed"])
            self.assertEqual(evidence["tier"], "owned_cli_and_verified_selector")
            self.assertEqual(evidence["selector_identity"]["function_ast_sha256"], self.compare.SELECTOR_AST_SHA256)


class BackendEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = module("_compare_backend_cpu", DIRECTORY.parent / "scripts/compare_official_runs.py")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.selector = self.root / "qwen_gdn_linear_attn.py"
        self.selector.write_text(SELECTOR_SOURCE)

    def evidence(self, log, *, installed_version="0.24.0", recorded_version="0.24.0"):
        (self.root / "training.log").write_text(log)
        with mock.patch.object(
            self.compare, "_installed_gdn_selector", return_value=(installed_version, self.selector)
        ):
            return self.compare.backend_evidence(self.compare.Reader(), self.root, self.model, recorded_version)

    def test_real_shape_owned_worker_info_is_direct_kernel_evidence(self):
        log = (
            "\x1b[36m(vLLMHttpServer pid=321)\x1b[0m (Worker_TP0 pid=654) INFO "
            "[qwen_gdn_linear_attn.py:228] Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).\n"
        )
        (self.root / "training.log").write_text(log)
        with mock.patch.object(self.compare, "_installed_gdn_selector", side_effect=AssertionError("Not needed")):
            evidence = self.compare.backend_evidence(self.compare.Reader(), self.root, self.model, "0.24.0")
        self.assertTrue(evidence["kernel_log_observed"])
        self.assertEqual(evidence["tier"], "worker_kernel_log")

    def test_warn_only_split_cli_accepts_both_flag_spellings_and_equals_forms(self):
        for flag, value in (
            ("--gdn_prefill_backend", "triton"),
            ("--gdn-prefill-backend", "triton"),
            ("--gdn_prefill_backend=triton", None),
            ("--gdn-prefill-backend=triton", None),
        ):
            with self.subTest(flag=flag):
                evidence = self.evidence(owned_cli_log(self.model, flag=flag, value=value))
                self.assertFalse(evidence["kernel_log_observed"])
                self.assertEqual(evidence["tier"], "owned_cli_and_verified_selector")
                self.assertEqual(evidence["active_backend"], "triton")
                self.assertEqual(evidence["selector_identity"]["function_ast_sha256"], self.compare.SELECTOR_AST_SHA256)

    def test_absent_unowned_incomplete_or_mismatched_cli_is_rejected(self):
        for log in (
            "WARN only: no owned CLI or INFO selection\n",
            owned_cli_log(self.model, owned=False),
            owned_cli_log(self.model, complete=False),
            owned_cli_log(self.model, value="flashinfer"),
            owned_cli_log(self.root / "another-model"),
        ):
            with self.subTest(log=log), self.assertRaises(self.compare.Error):
                self.evidence(log)

    def test_bare_kernel_text_is_not_an_owned_worker_selection(self):
        with self.assertRaises(self.compare.Error):
            self.evidence("Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128).\n")

    def test_selector_behavior_ast_drift_is_rejected(self):
        self.selector.write_text(SELECTOR_SOURCE.replace('return backend, "triton"', 'return backend, "flashinfer"'))
        with self.assertRaisesRegex(self.compare.Error, "selector differs"):
            self.evidence(owned_cli_log(self.model))

    def test_installed_or_recorded_package_identity_mismatch_is_rejected(self):
        for installed, recorded in (("0.24.1", "0.24.0"), ("0.24.0", "0.24.1"), ("0.24.1", "0.24.1")):
            with (
                self.subTest(installed=installed, recorded=recorded),
                self.assertRaisesRegex(self.compare.Error, "selector/package identity"),
            ):
                self.evidence(owned_cli_log(self.model), installed_version=installed, recorded_version=recorded)


if __name__ == "__main__":
    unittest.main()
