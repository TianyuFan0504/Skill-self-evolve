"""Official ALFWorld tests through the initialized training agent loop.

The synchronous trainer must be idle while this module temporarily points the
per-run environment configuration at a separate test-only service. Neither the
training service nor the model/optimizer is restarted or replaced.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import time
import urllib.request
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

TEST_SPLITS = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}
GREEDY_SETTINGS = {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1}
_MISSING = object()


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _set(value, key, item):
    if isinstance(value, Mapping):
        value[key] = item
    else:
        setattr(value, key, item)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path, content, *, mode=0o600):
    """Publish complete protected contents without changing a worker's path."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(path, value):
    def scalar(item):
        # NumPy metric scalars have item(); no general configuration is dumped.
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Unsupported evaluation report value: {type(item).__name__}")

    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=scalar) + "\n"
    _atomic_bytes(path, payload.encode())


def _read_protocol(manifest_path):
    """Reject sample-limited test preparations before starting any process."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_bytes())
    catalog_path = manifest_path.parent / "catalog.json"
    catalog = json.loads(catalog_path.read_bytes())
    if manifest.get("schema_version") != 1 or catalog.get("schema_version") != 1:
        raise ValueError("Unsupported evaluation dataset schema")
    games = catalog["games"]
    ids = [row["game_id"] for row in games]
    if len(ids) != len(set(ids)):
        raise ValueError("Official catalog contains duplicate game identities")
    canonical = json.dumps(
        sorted(games, key=lambda row: row["relative_game_path"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if catalog.get("catalog_digest") != digest or manifest.get("catalog_digest") != digest:
        raise ValueError("Official evaluation catalog digest mismatch")
    if manifest.get("evaluation_games_used_for_training") is not False:
        raise ValueError("The split manifest does not prove test isolation")
    training_ids = set(manifest["phase_game_ids"]["train"]) | set(manifest["phase_game_ids"]["dev"])
    files = {}
    for phase, split in TEST_SPLITS.items():
        expected = {row["game_id"] for row in games if row["split"] == split}
        selected = manifest["phase_game_ids"][phase]
        if not expected or len(selected) != len(set(selected)) or set(selected) != expected:
            raise ValueError(f"{phase} must contain every official eligible game exactly once")
        if expected & training_ids:
            raise ValueError("Official test games occur in the optimization/development partition")
        for key in ("available", "selected"):
            if manifest[key][phase]["games"] != len(expected):
                raise ValueError(f"{phase} is a truncated official evaluation partition")
        path = manifest_path.parent / f"{phase}.parquet"
        spec = manifest["parquet_files"][phase]
        if Path(spec["path"]).resolve() != path or _sha256(path) != spec["sha256"]:
            raise ValueError(f"The prepared {phase} parquet changed")
        files[phase] = {"path": str(path), "sha256": spec["sha256"], "games": len(expected), "ids": expected}
    return manifest, catalog_path, files


def _assert_test_rows(rows, *, expected_ids, split, digest, run_root):
    identities = []
    for row in rows:
        task = row["tools_kwargs"]["task"]
        metadata = task["metadata"]
        identities.append(metadata["game_id"])
        if (
            task["name"] != "alfworld_skill_training"
            or metadata["split"] != split
            or metadata["catalog_digest"] != digest
            or Path(metadata["run_dir"]).resolve() != Path(run_root).resolve()
        ):
            raise ValueError("Official test row has incompatible native-task metadata")
    if len(identities) != len(expected_ids) or set(identities) != expected_ids:
        raise ValueError("Official parquet does not contain every catalog test identity exactly once")


def _verify_parquet_rows(path, **kwargs):
    import pyarrow.parquet as pq

    _assert_test_rows(pq.read_table(path, columns=["extra_info"]).column("extra_info").to_pylist(), **kwargs)


def _health(settings, *, digest, max_steps):
    from urllib.parse import urlsplit

    url = urlsplit(settings["url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password:
        raise ValueError("The evaluation environment must be a local loopback service")
    with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=1) as response:
        value = json.load(response)
    return value.get("ready") is True and value.get("catalog_digest") == digest and value.get("max_steps") == max_steps


def _stop_owned_service(process):
    """Operate only on our Popen handle; never signal the training config PID."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def official_test_service(
    *,
    config_path,
    artifact_root,
    benchmark_root,
    run_root,
    catalog_path,
    digest,
    environment_python,
    startup_timeout=30.0,
):
    """Temporarily route native tasks to an owned test-only environment service.

    A local lock prevents concurrent configuration swaps. Health checks and all
    initialization finish before the shared protected file changes. The exact
    training bytes and permission bits are restored even when evaluation fails.
    """
    import fcntl

    config_path = Path(config_path)
    if config_path.is_symlink():
        raise ValueError("The shared environment config must be a protected regular file")
    original_mode = stat.S_IMODE(config_path.stat().st_mode)
    if original_mode & 0o077:
        raise ValueError("The shared environment config exposes an administration capability")
    lock_path = config_path.with_name(config_path.name + ".evaluation.lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    process = None
    swapped = False
    original = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = config_path.read_bytes()
        training = json.loads(original)
        if set(training.get("allowed_splits", [])) != {"train"} or training.get("catalog_digest") != digest:
            raise ValueError("The original service must be train-only and use this immutable catalog")
        max_steps = int(training["max_steps"])
        runtime = Path(artifact_root) / ("service-" + uuid.uuid4().hex)
        runtime.mkdir(parents=True, mode=0o700)
        settings_path = runtime / "environment.json"
        argv = [
            str(environment_python),
            str(Path(__file__).with_name("alfworld") / "server.py"),
            "--benchmark-root",
            str(benchmark_root),
            "--run-root",
            str(run_root),
            "--catalog-path",
            str(catalog_path),
            "--config-path",
            str(settings_path),
            "--allowed-splits",
            *TEST_SPLITS.values(),
            "--max-steps",
            str(max_steps),
            "--port",
            "0",
        ]
        with (runtime / "server.log").open("ab") as output:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"The owned evaluation service exited; see {runtime / 'server.log'}")
            if settings_path.is_file():
                try:
                    settings_bytes = settings_path.read_bytes()
                    settings = json.loads(settings_bytes)
                    compatible = (
                        settings.get("catalog_digest") == digest
                        and set(settings.get("allowed_splits", [])) == set(TEST_SPLITS.values())
                        and settings.get("max_steps") == max_steps
                        and settings.get("pid") == process.pid
                    )
                    if compatible and _health(settings, digest=digest, max_steps=max_steps):
                        break
                except (OSError, ValueError, KeyError):
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError("The owned evaluation service did not become healthy; training service is unchanged")
            time.sleep(0.1)
        # Tasks retain this exact path in existing Ray worker environments.
        _atomic_bytes(config_path, settings_bytes, mode=original_mode)
        swapped = True
        yield {"catalog_digest": digest, "max_steps": max_steps, "allowed_splits": list(TEST_SPLITS.values())}
    finally:
        try:
            if swapped:
                _atomic_bytes(config_path, original, mode=original_mode)
        finally:
            try:
                if process is not None:
                    _stop_owned_service(process)
            finally:
                os.close(lock_fd)


def _dataset_components():
    from torchdata.stateful_dataloader import StatefulDataLoader

    from verl.trainer.ppo.utils import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn

    return create_rl_dataset, StatefulDataLoader, collate_fn


@contextmanager
def _validation_inputs(trainer, *, path, expected_games, phase, stage, output_dir):
    create_rl_dataset, loader_class, collate_fn = _dataset_components()
    config = trainer.config
    originals = {
        name: getattr(trainer, name, _MISSING)
        for name in (
            "val_dataset",
            "val_dataloader",
            "_evaluation_phase",
            "_evaluation_stage",
        )
    }
    original_data, original_trainer_config = config.data, config.trainer
    data = copy.deepcopy(original_data)
    trainer_config = copy.deepcopy(original_trainer_config)
    for key, value in {
        "val_files": str(path),
        "val_max_samples": -1,
        "validation_shuffle": False,
        "filter_overlong_prompts": False,
        "dataloader_num_workers": 0,
    }.items():
        _set(data, key, value)
    _set(trainer_config, "validation_data_dir", str(output_dir))
    _set(trainer_config, "log_val_generations", 0)
    try:
        config.data, config.trainer = data, trainer_config
        # Use the attached config so OmegaConf references resolve in the same
        # root as the already initialized rollout/learner configuration.
        data = config.data
        trainer.val_dataset = create_rl_dataset(
            str(path),
            data,
            trainer.tokenizer,
            trainer.processor,
            is_train=False,
            max_samples=-1,
        )
        if len(trainer.val_dataset) != expected_games:
            raise ValueError(f"{phase} dataset construction dropped or duplicated official test cases")
        trainer.val_dataloader = loader_class(
            dataset=trainer.val_dataset,
            batch_size=_get(data, "val_batch_size") or expected_games,
            num_workers=0,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        trainer._evaluation_phase, trainer._evaluation_stage = phase, stage
        yield
    finally:
        config.data, config.trainer = original_data, original_trainer_config
        for name, value in originals.items():
            if value is _MISSING:
                if hasattr(trainer, name):
                    delattr(trainer, name)
            else:
                setattr(trainer, name, value)


def _budgets(config):
    rollout = config.actor_rollout_ref.rollout
    values = {key: _get(rollout.val_kwargs, key) for key in GREEDY_SETTINGS}
    if values != GREEDY_SETTINGS:
        raise ValueError("Initialize native agent workers with greedy n=1 validation settings before official tests")
    budgets = {
        "sampling": values,
        "prompt_length": _get(rollout, "prompt_length"),
        "response_length": _get(rollout, "response_length"),
        "max_model_len": _get(rollout, "max_model_len"),
    }
    for name in ("ALFWORLD_TASK_CONFIG", "ALFWORLD_SOLVER_INSTRUCTIONS_FILE"):
        path = os.environ.get(name)
        if path:
            budgets[name.lower() + "_sha256"] = _sha256(path)
    settings = json.loads(Path(os.environ["ALFWORLD_SERVER_CONFIG"]).read_bytes())
    budgets["environment_max_steps"] = settings["max_steps"]
    return budgets


def evaluate_official_splits(trainer, *, phase_prefix="initial"):
    """Evaluate complete official test splits without actor updates or curation.

    The caller installs ``trainer.agent_loop_manager`` before an initial call
    made ahead of the base trainer's ``fit``. Scheduled validation remains on the
    train-derived development set because every temporary setting is restored.
    """
    if phase_prefix not in {"initial", "final"}:
        raise ValueError("Official evaluation stage must be initial or final")
    if getattr(trainer, "agent_loop_manager", None) is None:
        raise RuntimeError("Official tests require the already initialized native agent loop manager")
    harness = trainer.config.skill_harness
    manifest_path = Path(harness.split_manifest).resolve()
    manifest, catalog_path, files = _read_protocol(manifest_path)
    run_root = Path(manifest["run_dir"]).resolve()
    artifact_root = Path(harness.artifact_root).resolve()
    if artifact_root != run_root:
        raise ValueError("Official tests must use the same prepared run root as training")
    stage_root = artifact_root / "evaluation" / phase_prefix
    stage_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (stage_root / "summary.json").exists():
        raise FileExistsError(f"Completed {phase_prefix} test artifacts already exist; preserve the original baseline")
    budgets = _budgets(trainer.config)
    dataset_protocol = {
        "manifest_sha256": _sha256(manifest_path),
        "catalog_digest": manifest["catalog_digest"],
        "catalog_sha256": _sha256(catalog_path),
        "splits": {phase: {key: value for key, value in spec.items() if key != "ids"} for phase, spec in files.items()},
    }
    if phase_prefix == "final":
        initial_path = artifact_root / "evaluation/initial/protocol.json"
        if initial_path.is_file():
            initial = json.loads(initial_path.read_bytes())
            if initial["dataset"] != dataset_protocol or initial["budgets"] != budgets:
                raise ValueError("Final tests must retain the initial test identities, prompts, and generation budgets")
    for phase, spec in files.items():
        _verify_parquet_rows(
            spec["path"],
            expected_ids=spec["ids"],
            split=TEST_SPLITS[phase],
            digest=manifest["catalog_digest"],
            run_root=run_root,
        )
    _write_report(
        stage_root / "protocol.json",
        {
            "schema_version": 1,
            "stage": phase_prefix,
            "global_step": trainer.global_steps,
            "policy": "already_initialized_training_policy",
            "agent": "native_uni_agent_codex",
            "optimizer_updates": False,
            "curation": False,
            "dataset": dataset_protocol,
            "budgets": budgets,
        },
    )
    results = {}
    try:
        with official_test_service(
            config_path=os.environ["ALFWORLD_SERVER_CONFIG"],
            artifact_root=stage_root,
            benchmark_root=manifest["benchmark_root"],
            run_root=run_root,
            catalog_path=catalog_path,
            digest=manifest["catalog_digest"],
            environment_python=os.environ.get(
                "ALFWORLD_ENV_PYTHON",
                str(Path(manifest["benchmark_root"]) / "env/bin/python"),
            ),
        ) as service:
            _write_report(stage_root / "environment-protocol.json", service)
            for phase, spec in files.items():
                output_dir = stage_root / phase
                output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                with _validation_inputs(
                    trainer,
                    path=spec["path"],
                    expected_games=spec["games"],
                    phase=phase,
                    stage=phase_prefix,
                    output_dir=output_dir,
                ):
                    trainer.on_validate_begin()
                    try:
                        metrics = trainer._validate()
                        # The base trainer writes full trajectories asynchronously.
                        # A completed evaluation must mean these files reached disk.
                        for future in list(getattr(trainer, "_dump_futures", [])):
                            future.result()
                    finally:
                        trainer.on_validate_end()
                if not metrics:
                    raise RuntimeError(f"{phase} did not produce official evaluation metrics")
                results[phase] = metrics
                _write_report(
                    output_dir / "metrics.json",
                    {
                        "schema_version": 1,
                        "stage": phase_prefix,
                        "phase": phase,
                        "global_step": trainer.global_steps,
                        "expected_games": spec["games"],
                        "metrics": metrics,
                    },
                )
    except BaseException as error:
        _write_report(
            stage_root / "failure.json",
            {
                "schema_version": 1,
                "stage": phase_prefix,
                "completed_splits": list(results),
                "error_type": type(error).__name__,
            },
        )
        raise
    _write_report(
        stage_root / "summary.json",
        {
            "schema_version": 1,
            "stage": phase_prefix,
            "global_step": trainer.global_steps,
            "completed": True,
            "metrics": results,
        },
    )
    return results
