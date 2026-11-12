"""Observed learner spawn default and transport-only DataLoader preparation.

No model, benchmark environment, optimizer or scientific config is changed.
Spawn avoids inheriting unrelated parent descriptors; this is not a diagnosis
of a previous learner exit or an assertion about every worker's live state.
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing
import time
from pathlib import Path

POLICY = "spawn-default-and-credit-probe-semaphore-v1"
START_METHOD = "spawn"
PROBE_CONCURRENCY = 2
CONFIGURED_PROBE_CONCURRENCY = 8
DEFAULT_SCOPE = "Actual learner default before trainer construction; no per-worker or root-cause attestation"
LOADER_SCOPE = (
    "Loader transport observed before iteration; worker start method is source-linked inference, "
    "not per-worker attestation"
)


def _field(value, key):
    return value[key] if hasattr(value, "__getitem__") else getattr(value, key)


def _configured(config):
    workers = _field(_field(config, "data"), "dataloader_num_workers")
    credit = _field(_field(_field(config, "skill_harness"), "joint"), "credit_concurrency")
    if workers != 4 or credit != CONFIGURED_PROBE_CONCURRENCY:
        raise ValueError("This operational profile requires configured DataLoader workers4 and credit ceiling8")
    return workers, credit


def provider_ref():
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_learner(config):
    """Call in the actual Ray learner before constructing its trainer."""
    from skill_harness_examples import curate_generation

    workers, credit = _configured(config)
    previous = multiprocessing.get_start_method(allow_none=True)
    # Ray may already have selected the platform default. Only future process
    # creation in this owned learner is changed; existing Ray resources stay as-is.
    multiprocessing.set_start_method(START_METHOD, force=True)
    observed = multiprocessing.get_start_method()
    if observed != START_METHOD:
        raise RuntimeError("The actual learner multiprocessing default is not spawn")
    root = Path(_field(_field(config, "skill_harness"), "artifact_root")).resolve()
    return curate_generation.publish(root / "driver_runs" / (str(time.time_ns()) + ".learner_runtime.json"), {
        "schema_version": 1, "kind": "observed_spawn_learner_default", "policy": POLICY,
        "owner": curate_generation.process_identity(), "observed_before": "trainer_construct",
        "previous_default_start_method": previous, "observed_default_start_method": observed,
        "configured_dataloader_workers": workers, "configured_credit_concurrency": credit,
        "physical_credit_probe_concurrency": PROBE_CONCURRENCY, "provider_ref": provider_ref(),
        "scope": DEFAULT_SCOPE,
    })


def prepare_loaders(trainer):
    """Keep the existing filtered frames pickleable before their first iterator.

    RLHFDataset's default __getstate__ removes its dataframe without rebuilding
    it on unpickle. Its existing serialize_dataset flag preserves that frame.
    Its text-only __getitem__ does not use the filtered processor, whose dynamic
    model methods are not pickleable. Detach that dataset reference only.
    """
    from skill_harness_examples import curate_generation

    workers, credit = _configured(trainer.config)
    before = curate_generation.read_ref(trainer._learner_runtime_ref)
    if (before["owner"] != curate_generation.process_identity()
            or before["provider_ref"] != provider_ref()
            or before["observed_default_start_method"] != START_METHOD
            or multiprocessing.get_start_method() != START_METHOD
            or before["configured_dataloader_workers"] != workers
            or before["configured_credit_concurrency"] != credit):
        raise ValueError("Prepared learner runtime differs from the actual process/provider/config")
    if getattr(trainer, "train_dataloader_it", None) is not None:
        raise ValueError("Spawn dataset preparation requires an unstarted training iterator")
    data_config = copy.deepcopy(trainer.config.data)
    trainer_processor = getattr(trainer, "processor", None)
    result = {"default_start_method": multiprocessing.get_start_method(), "scope": LOADER_SCOPE}
    loaders = {name: getattr(trainer, name + "_dataloader") for name in ("train", "val")}
    # Validate both sides before changing either dataset instance.
    for loader in loaders.values():
        context = loader.multiprocessing_context
        method = None if context is None else context.get_start_method()
        if loader.num_workers != 4 or method not in (None, START_METHOD) or loader._iterator is not None:
            raise ValueError("Spawn dataset preparation requires four unstarted workers and no fork override")
        if not hasattr(loader.dataset, "serialize_dataset") or not hasattr(loader.dataset, "dataframe"):
            raise TypeError("The configured dataset lacks the supported dataframe serialization transport")
        _require_text_only_dataset(loader.dataset)
    for name, loader in loaders.items():
        dataset, sampler, frame = loader.dataset, loader.sampler, loader.dataset.dataframe
        previous = dataset.serialize_dataset
        fingerprint, length = frame._fingerprint, len(dataset)
        processor = dataset.processor
        dataset.serialize_dataset = True
        dataset.processor = None
        unchanged = (dataset.dataframe is frame and frame._fingerprint == fingerprint
                     and len(dataset) == length and loader.sampler is sampler)
        processor_unchanged = getattr(trainer, "processor", None) is trainer_processor
        if not unchanged or not processor_unchanged or trainer.config.data != data_config:
            raise RuntimeError("Spawn transport preparation changed data or sampler/config")
        context = loader.multiprocessing_context
        result[name] = {
            "dataset_class": type(dataset).__module__ + "." + type(dataset).__qualname__,
            "num_workers": loader.num_workers,
            "multiprocessing_context": None if context is None else context.get_start_method(),
            "iterator_started": loader._iterator is not None,
            "serialize_dataset_before": previous, "serialize_dataset_after": dataset.serialize_dataset,
            "dataframe_fingerprint": fingerprint, "dataset_length": length,
            "filtered_dataframe_unchanged": unchanged,
            "processor_before_class": None if processor is None else type(processor).__module__ + "."
            + type(processor).__qualname__,
            "processor_after_class": None,
            "trainer_processor_unchanged": processor_unchanged,
            "text_only_transport": True,
        }
    return result


def _require_text_only_dataset(dataset):
    actual_class = type(dataset).__module__ + "." + type(dataset).__qualname__
    if actual_class != "verl.utils.dataset.rl_dataset.RLHFDataset":
        raise TypeError("Spawn processor detachment supports the exact default RLHFDataset only")
    frame = dataset.dataframe
    for key in (dataset.image_key, dataset.video_key, dataset.audio_key):
        if key in frame.column_names:
            if any(value is not None and (not isinstance(value, list) or value) for value in frame[key]):
                raise ValueError("Spawn processor detachment requires text-only data without media payloads")
    for messages in frame[dataset.prompt_key]:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            or any(token in message["content"] for token in ("<image>", "<video>", "<audio>"))
            for message in messages
        ):
            raise ValueError("Spawn processor detachment requires plain text message content")
