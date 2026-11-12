#!/usr/bin/env python3
"""CPU proof for the real four-worker dataset/checkpoint transport.

Reads original train parquet and local tokenizer/processor only. It never
initializes a model, Ray, a benchmark environment, or a training optimizer.
All worker reports/cache/output are written below a newly created /tmp root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import types
from functools import partial
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import datasets  # noqa: E402 - the isolated source/cache environment precedes third-party imports.
import torch  # noqa: E402
import torchdata  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torchdata.stateful_dataloader import StatefulDataLoader  # noqa: E402

from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler  # noqa: E402
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn  # noqa: E402
from verl.utils.tokenizer import hf_processor, hf_tokenizer  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [canonical(item) for item in value]
    if hasattr(value, "tolist"):
        return canonical(value.tolist())
    return value


def state_hash(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def worker_report(worker_id, *, directory, sentinel):
    """Measure only our deliberately inheritable marker descriptor."""
    matches = 0
    for path in Path("/proc/self/fd").iterdir():
        try:
            stat = os.fstat(int(path.name))
            matches += (stat.st_dev, stat.st_ino) == tuple(sentinel)
        except OSError:
            pass
    value = {
        "pid": os.getpid(),
        "worker_id": worker_id,
        "observed_start_method": mp.get_start_method(),
        "sentinel_matches": matches,
    }
    path = Path(directory) / (str(worker_id) + ".json")
    path.write_text(json.dumps(value) + "\n")


def batch_identity(batch):
    """Keep only stable original-game IDs and a raw-prompt digest, no prompt text."""
    values = []
    for index, extra in enumerate(batch["extra_info"]):
        task = extra.get("tools_kwargs", {}).get("task", {}).get("metadata", {})
        game_id = task.get("stable_game_id", task.get("game_id", extra.get("index")))
        prompt_hash = hashlib.sha256(
            json.dumps(canonical(batch["raw_prompt"][index]), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        values.append({"game_id": game_id, "prompt_sha256": prompt_hash})
    return values


def shutdown(iterator):
    workers = list(getattr(iterator, "_workers", ()))
    iterator._shutdown_workers()
    result = [{"pid": process.pid, "alive": process.is_alive(), "exitcode": process.exitcode} for process in workers]
    if any(row["alive"] for row in result):
        raise AssertionError("Owned dataloader workers did not exit during normal shutdown")
    return result


def make_loader(dataset, config, state, method, directory, sentinel):
    mp.set_start_method(method, force=True)
    directory.mkdir()
    loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.gen_batch_size,
        num_workers=4,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=create_rl_sampler(config, dataset),
        worker_init_fn=partial(worker_report, directory=str(directory), sentinel=sentinel),
    )
    assert loader.multiprocessing_context is None
    loader.load_state_dict(copy.deepcopy(state))
    return loader


def reports(directory):
    deadline = time.monotonic() + 15
    while len(list(directory.glob("*.json"))) < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
    result = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    assert {row["worker_id"] for row in result} == {0, 1, 2, 3}, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--manual-processor-detach",
        action="store_true",
        help="Prove the narrowly guarded text-only detachment before its provider integration",
    )
    args = parser.parse_args()
    assert sys.platform == "linux" and args.samples >= 4 and args.samples % 4 == 0
    run = args.run_root.resolve()
    checkpoint = run / "checkpoints/global_step_10/data.pt"
    before_hash = digest(checkpoint)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert state["_snapshot"]["_main_snapshot"]["_num_workers"] == 4
    initialized = sorted((run / "driver_runs").glob("*.all_native_spawn.initialized.json"))[-1]
    config = json.loads(initialized.read_text())["effective_config_after_trainer_init"]
    data = OmegaConf.create(config["data"])
    assert data.dataloader_num_workers == 4 and data.gen_batch_size == 1
    assert data.train_max_samples == -1 and data.seed == 42 and data.shuffle is True
    work = Path(tempfile.mkdtemp(prefix="stateful-spawn-proof-", dir="/tmp"))
    os.environ["HF_DATASETS_CACHE"] = str(work / "hf-datasets")
    datasets.config.HF_DATASETS_CACHE = work / "hf-datasets"
    datasets.config.DOWNLOADED_DATASETS_PATH = work / "hf-datasets/downloads"
    data.cache_dir = str(work / "parquet-cache")
    model = config["actor_rollout_ref"]["model"]["path"]
    tokenizer = hf_tokenizer(model, trust_remote_code=False, local_files_only=True)
    processor = hf_processor(model, trust_remote_code=False, local_files_only=True)
    if processor is not None and not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    dataset = create_rl_dataset(data.train_files, data, tokenizer, processor, is_train=True, max_samples=-1)
    val_dataset = create_rl_dataset(data.val_files, data, tokenizer, processor, is_train=False, max_samples=-1)
    assert len(dataset) == 3154 and dataset.serialize_dataset is False
    assert len(val_dataset) == 399 and val_dataset.serialize_dataset is False
    filtered_fingerprint = dataset.dataframe._fingerprint
    sentinel_path = work / "inheritable-sentinel"
    sentinel_fd = os.open(sentinel_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.set_inheritable(sentinel_fd, True)
    sentinel_stat = os.fstat(sentinel_fd)
    sentinel = (sentinel_stat.st_dev, sentinel_stat.st_ino)
    result = {
        "schema_version": 1,
        "scope": __doc__,
        "work_root": str(work),
        "torch_version": torch.__version__,
        "torchdata_version": torchdata.__version__,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": before_hash,
        "actual_data_state_loaded": True,
        "source_snapshot_step": state["_snapshot"]["_snapshot_step"],
        "dataset_size": len(dataset),
        "filtered_dataframe_fingerprint": filtered_fingerprint,
        "train_parquet_sha256": digest(data.train_files),
        "processor_class": None if processor is None else type(processor).__name__,
        "samples_compared_per_stage": args.samples,
        "operational_cache_override_only": True,
        "source_files": {
            name: digest(path)
            for name, path in {
                "rl_dataset": SOURCE / "verl/utils/dataset/rl_dataset.py",
                "ppo_dataset_factories": SOURCE / "verl/trainer/ppo/utils.py",
                "stateful_dataloader": Path(sys.modules[StatefulDataLoader.__module__].__file__),
            }.items()
        },
    }
    baseline_it = failed_it = spawn_it = roundtrip_it = None
    try:
        fork_dir = work / "fork-original"
        baseline = make_loader(dataset, data, state, "fork", fork_dir, sentinel)
        baseline_it = iter(baseline)
        fork_ids = [batch_identity(next(baseline_it)) for _ in range(args.samples)]
        fork_state = baseline.state_dict()
        baseline_tail = [batch_identity(next(baseline_it)) for _ in range(args.samples)]
        result["fork_original"] = {
            "workers": reports(fork_dir),
            "next_prompt_ids": fork_ids,
            "state_after_samples_sha256": state_hash(fork_state),
        }
        assert all(row["sentinel_matches"] == 1 for row in result["fork_original"]["workers"])
        print("FORK4 actual checkpoint restore completed", flush=True)

        try:
            pickle.loads(pickle.dumps(dataset))
        except AttributeError as error:
            assert "get_rope_index" in str(error)
            result["spawn_unadapted"] = {
                "expected_failure": "Qwen3VLProcessor dynamic bound method cannot be unpickled",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "boundary": "Actual dataset pickle/unpickle; prior real four-worker spawn failure corroborates this",
            }
        else:
            raise AssertionError("Unadapted spawn unexpectedly succeeded; re-examine dataset transport contract")
        old_processor = dataset.processor
        dataset.processor = None
        without_processor = pickle.loads(pickle.dumps(dataset))
        assert not hasattr(without_processor, "dataframe")
        try:
            without_processor[0]
        except AttributeError as error:
            assert "dataframe" in str(error)
            result["spawn_without_processor_only"] = {"error_type": type(error).__name__, "error_message": str(error)}
        else:
            raise AssertionError("Unadapted dataset unexpectedly restored its omitted dataframe")
        dataset.processor = old_processor
        print("Unadapted processor + dataframe pickle failures confirmed", flush=True)

        namespace = types.ModuleType("skill_harness_examples")
        namespace.__path__ = [str(SOURCE / "examples/skill_self_evolve/scripts")]
        sys.modules[namespace.__name__] = namespace
        from skill_harness_examples import learner_runtime

        guard_config = OmegaConf.create(config)
        guard_config.data = data
        guard_config.skill_harness.artifact_root = str(work / "operational-runtime")
        runtime_ref = learner_runtime.prepare_learner(guard_config)
        spawn_dir = work / "spawn-filtered-frame"
        spawned = make_loader(dataset, data, state, "spawn", spawn_dir, sentinel)
        val_loader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=data.val_batch_size,
            num_workers=4,
            shuffle=data.validation_shuffle,
            drop_last=False,
            collate_fn=collate_fn,
        )
        trainer = types.SimpleNamespace(
            config=guard_config,
            _learner_runtime_ref=runtime_ref,
            train_dataloader=spawned,
            val_dataloader=val_loader,
            train_dataloader_it=None,
            processor=processor,
        )
        if args.manual_processor_detach:
            assert type(dataset) is RLHFDataset and type(val_dataset) is RLHFDataset
            for value in (dataset, val_dataset):
                assert not set(value.dataframe.column_names) & {data.image_key, data.video_key, data.audio_key}
                assert all(
                    all(isinstance(message.get("content"), str) for message in row[data.prompt_key])
                    for row in value.dataframe
                )
                value.processor = None
        observed = learner_runtime.prepare_loaders(trainer)
        assert dataset.processor is None and val_dataset.processor is None and trainer.processor is processor
        result["operational_helper"] = {
            "learner_runtime_ref": runtime_ref,
            "loader_observation": observed,
            "learner_runtime_source_sha256": digest(Path(learner_runtime.__file__)),
            "manual_processor_detach": args.manual_processor_detach,
            "trainer_processor_unchanged": trainer.processor is processor,
        }
        assert len(dataset) == 3154 and dataset.dataframe._fingerprint == filtered_fingerprint
        assert dataset.serialize_dataset is True and val_dataset.serialize_dataset is True
        spawn_it = iter(spawned)
        spawn_ids = [batch_identity(next(spawn_it)) for _ in range(args.samples)]
        spawn_state = spawned.state_dict()
        assert spawn_ids == fork_ids
        assert canonical(spawn_state) == canonical(fork_state)
        result["spawn_filtered_frame"] = {
            "workers": reports(spawn_dir),
            "next_prompt_ids_match": True,
            "exact_loader_state_matches_fork": True,
            "state_after_samples_sha256": state_hash(spawn_state),
        }
        assert all(row["sentinel_matches"] == 0 for row in result["spawn_filtered_frame"]["workers"])
        assert all(row["observed_start_method"] == "spawn" for row in result["spawn_filtered_frame"]["workers"])
        saved = work / "spawn-state.pt"
        torch.save(spawn_state, saved)
        result["spawn_filtered_frame"]["shutdown"] = shutdown(spawn_it)
        spawn_it = None
        print("SPAWN4 adapted exact checkpoint order/state parity completed", flush=True)

        back_dir = work / "fork-roundtrip"
        roundtrip = make_loader(
            dataset, data, torch.load(saved, weights_only=False, map_location="cpu"), "fork", back_dir, sentinel
        )
        roundtrip_it = iter(roundtrip)
        tail = [batch_identity(next(roundtrip_it)) for _ in range(args.samples)]
        assert tail == baseline_tail
        result["fork_spawn_fork_roundtrip"] = {
            "workers": reports(back_dir),
            "continued_prompt_ids_match": True,
            "shutdown": shutdown(roundtrip_it),
        }
        roundtrip_it = None
        result["fork_original"]["shutdown"] = shutdown(baseline_it)
        baseline_it = None
        result["checkpoint_sha256_after"] = digest(checkpoint)
        assert result["checkpoint_sha256_after"] == before_hash
        result["cuda_initialized"] = torch.cuda.is_initialized()
        assert result["cuda_initialized"] is False
        result["passed"] = True
    finally:
        for iterator in (roundtrip_it, spawn_it, failed_it, baseline_it):
            if iterator is not None:
                shutdown(iterator)
        os.close(sentinel_fd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output),
                "output_sha256": digest(args.output),
                "work_root": str(work),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
