#!/usr/bin/env python3
# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Read-only stdlib evidence index; this is not an experiment acceptance oracle.

No model/package imports, network, subprocesses, pickle loading, or evaluations.
Reports distinguish declarations from linked Task/native/token evidence. Outputs
are new files only; raw metrics and artifacts are never repaired or overwritten.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import struct
import time
import zipfile
from pathlib import Path

OFFICIAL_COUNTS = {"test_seen": 140, "test_unseen": 134}
OFFICIAL_SPLITS = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class EvidenceError(ValueError):
    pass


class Reader:
    def __init__(self):
        self.observed = {}

    def bytes(self, path):
        path = Path(path)
        try:
            content = path.read_bytes()
            info = path.stat()
        except OSError as error:
            raise EvidenceError(f"{path}: missing/unreadable evidence ({type(error).__name__})") from error
        self.observed[str(path)] = (info.st_size, info.st_mtime_ns)
        return content

    def json(self, path):
        try:
            value = json.loads(self.bytes(path), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            if isinstance(value, dict) and "payload" in value:
                expected = value.get("payload_hash", value.get("sha256"))
                if expected is None or expected != digest(value["payload"]):
                    raise ValueError("envelope content hash differs")
                return value["payload"]
            return value
        except (ValueError, TypeError) as error:
            raise EvidenceError(f"{path}: invalid JSON/content hash ({error})") from error

    def rows(self, path):
        rows = []
        for number, line in enumerate(self.bytes(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if not isinstance(row, dict):
                    raise ValueError("expected a JSONL object")
                rows.append(row)
            except (ValueError, TypeError) as error:
                raise EvidenceError(f"{path}:{number}: invalid or still-being-written JSONL") from error
        return rows

    def changed(self):
        changed = []
        for name, observed in self.observed.items():
            try:
                info = Path(name).stat()
                if (info.st_size, info.st_mtime_ns) != observed:
                    changed.append(name)
            except OSError:
                changed.append(name)
        return changed


def section(status, reasons=(), paths=(), **facts):
    return {"status": status, "reasons": list(reasons), "evidence_paths": sorted(set(map(str, paths))), **facts}


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _tokens(capture):
    rows = capture.get("trajectories", [])
    return bool(rows) and all(
        row.get("prompt_ids")
        and row.get("response_ids", row.get("action_ids"))
        and any(row.get("response_mask", row.get("curate_mask", [])))
        for row in rows
    )


def _npy_numbers(content):
    """Read ordinary numeric NPY arrays without NumPy or executing pickle."""
    if content[:6] != b"\x93NUMPY":
        raise ValueError("not an NPY array")
    major = content[6]
    start = 10 if major == 1 else 12
    length = struct.unpack("<H" if major == 1 else "<I", content[8:start])[0]
    header = ast.literal_eval(content[start : start + length].decode("latin1"))
    shape, descr = header["shape"], header["descr"]
    if not isinstance(descr, str) or not re.fullmatch(r"[<>=|][biuf?][1248]?", descr):
        raise ValueError("non-numeric/object arrays are not evidence")
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("invalid numeric array shape")
    size = int(descr[2:]) if descr[2:] else 1
    formats = {
        ("i", 1): "b",
        ("i", 2): "h",
        ("i", 4): "i",
        ("i", 8): "q",
        ("u", 1): "B",
        ("u", 2): "H",
        ("u", 4): "I",
        ("u", 8): "Q",
        ("f", 2): "e",
        ("f", 4): "f",
        ("f", 8): "d",
        ("b", 1): "?",
        ("?", 1): "?",
    }
    code = formats[(descr[1], size)]
    data = content[start + length :]
    if len(data) != math.prod(shape) * size:
        raise ValueError("truncated numeric trajectory")
    endian = ">" if descr[0] == ">" else "<"
    return [value[0] for value in struct.iter_unpack(endian + code, data)]


def gateway_trace(reader, directory):
    metadata = reader.json(directory / "trajectory.json")
    reader.bytes(directory / "trajectory.npz")
    count = 0
    try:
        with zipfile.ZipFile(directory / "trajectory.npz") as archive:
            for name in archive.namelist():
                if not name.endswith("_response_mask.npy"):
                    continue
                mask = _npy_numbers(archive.read(name))
                prefix = name.removesuffix("response_mask.npy")
                prompt = _npy_numbers(archive.read(prefix + "prompt_ids.npy"))
                response = _npy_numbers(archive.read(prefix + "response_ids.npy"))
                if not prompt or len(mask) != len(response) or not any(value == 1 for value in mask):
                    raise ValueError("no aligned sampled model action tokens")
                count += sum(value == 1 for value in mask)
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, struct.error) as error:
        raise EvidenceError(f"{directory}: invalid/missing numeric Gateway trajectory ({error})") from error
    if not count or not metadata.get("num_trajectories"):
        raise EvidenceError(f"{directory}: Gateway trajectory contains no sampled model tokens")
    return count


def episode(reader, directory):
    directory = Path(directory).resolve()
    files = {
        name: directory / name
        for name in (
            "attempt.json",
            "result.json",
            "engine-result.json",
            "initial-state.json",
            "environment-trace.jsonl",
            "codex-events.jsonl",
            "codex-stderr.log",
        )
    }
    try:
        result, engine, attempt = (
            reader.json(files[name]) for name in ("result.json", "engine-result.json", "attempt.json")
        )
        reader.json(files["initial-state.json"])
        transitions = reader.rows(files["environment-trace.jsonl"])
        events = reader.rows(files["codex-events.jsonl"])
        reader.bytes(files["codex-stderr.log"])
        won = engine.get("won")
        if type(won) is not bool or result.get("reward") != float(won) or result.get("accuracy") != float(won):
            raise EvidenceError(f"{directory}: Task reward/accuracy disagree with binary engine won")
        for key in ("game_id", "game_sha256", "split", "catalog_digest"):
            if not result.get(key) or engine.get(key) != result[key] or attempt.get(key) != result[key]:
                raise EvidenceError(f"{directory}: raw attempt/Task/engine differ in {key}")
        if len(transitions) != engine.get("steps"):
            raise EvidenceError(f"{directory}: environment transitions do not cover engine steps")
        if result.get("agent_ok") is not True or result.get("agent_error") or result.get("agent_exit_code") != 0:
            raise EvidenceError(f"{directory}: environment outcome exists but native completion is not clean")
        if not any(event.get("type") == "turn.completed" for event in events):
            raise EvidenceError(f"{directory}: no raw native turn.completed event")
        if any(event.get("type") in ("turn.failed", "error") for event in events):
            raise EvidenceError(f"{directory}: raw native execution contains an error")
        return {
            "valid": True,
            "path": str(directory),
            "result": result,
            "engine_win": won,
            "evidence_paths": list(map(str, files.values())),
        }
    except (EvidenceError, KeyError, TypeError) as error:
        result = result if "result" in locals() else {}
        engine = engine if "engine" in locals() else {}
        return {
            "valid": False,
            "path": str(directory),
            "result": result,
            "engine_win": engine.get("won"),
            "reason": str(error),
            "evidence_paths": list(map(str, files.values())),
        }


class RunAudit:
    def __init__(self, root):
        self.root, self.reader = Path(root).resolve(), Reader()
        self.episodes = {}
        self.manifest = {}
        self.config = {}
        self.train_updates = {}
        self.train_games = set()
        self.token_supported_episodes = set()
        self.trace_directories = {}
        self.trace_cache = {}
        self.split_manifest_path = None
        self.phase_ids, self.split = {}, {}

    def raw_episode(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root / "episodes"):
            return {
                "valid": False,
                "path": str(path),
                "result": {},
                "reason": "Task artifact escaped this run",
                "evidence_paths": [],
            }
        if str(path) not in self.episodes:
            self.episodes[str(path)] = episode(self.reader, path)
        return self.episodes[str(path)]

    def provenance(self):
        paths = [self.root / "training-manifest.json", self.root / "launch_manifest.json"]
        reasons, facts = [], {}
        try:
            self.manifest = self.reader.json(paths[0])
            launch = self.reader.json(paths[1])
            self.config = self.manifest.get("config", {})
            provenance = self.manifest.get("import_provenance", {})
            source = Path(provenance.get("verl_source", "/missing-core-source"))
            origin = Path(provenance.get("verl_origin", "/missing-core-origin"))
            if provenance.get("verl_version") != "0.9.1" or origin.resolve() != (source / "verl/__init__.py").resolve():
                reasons.append("Declared core import is not the pinned verl0.9.1 checkout")
            for path in (origin, Path(provenance.get("uni_agent_origin", "/missing-uni-agent"))):
                self.reader.bytes(path)
                paths.append(path)
            model = Path(self.config.get("actor_rollout_ref", {}).get("model", {}).get("path", "/missing-model"))
            model_config = self.reader.json(model / "config.json")
            index = self.reader.json(model / "model.safetensors.index.json")
            weights = sorted(set(index.get("weight_map", {}).values()))
            if not weights or any(
                not (model / name).is_file() or (model / name).stat().st_size == 0 for name in weights
            ):
                reasons.append("Configured model lacks actual nonempty indexed weight shards")
            paths.extend(
                [model / "config.json", model / "model.safetensors.index.json", *[model / name for name in weights]]
            )
            native = launch.get("environment_paths", {}).get("CODEX_NATIVE_BIN")
            if not native or not Path(native).is_file() or not Path(native).stat().st_size:
                reasons.append("No accessible native Codex executable is bound by the launch artifact")
            else:
                paths.append(Path(native))
            facts = {
                "declared_source_commit": self.manifest.get("source_commit"),
                "declared_imports": provenance,
                "configured_model_path": str(model),
                "model_architectures": model_config.get("architectures"),
                "indexed_weight_shards": len(weights),
                "native_binary": native,
                "weight_content_rehashed": False,
                "runtime_loaded_identity_independently_attested": False,
            }
        except (EvidenceError, TypeError, KeyError) as error:
            reasons.append(str(error))
        return section("incomplete" if reasons else "declared_with_files", reasons, paths, **facts)

    def separation(self):
        harness = self.config.get("skill_harness", {})
        path = _path(self.root, harness.get("split_manifest", "data/manifest.json"))
        reasons = []
        try:
            split = self.reader.json(path)
            catalog_path = path.parent / "catalog.json"
            catalog = self.reader.json(catalog_path)
            self.phase_ids = {phase: set(ids) for phase, ids in split["phase_game_ids"].items()}
            train, dev = self.phase_ids["train"], self.phase_ids["dev"]
            tests = self.phase_ids["test_seen"] | self.phase_ids["test_unseen"]
            if train & dev or (train | dev) & tests or split.get("evaluation_games_used_for_training") is not False:
                reasons.append("Declared optimization/development/official-test identities overlap or are not isolated")
            calculated = digest(sorted(catalog["games"], key=lambda row: row["relative_game_path"]))
            if calculated != catalog.get("catalog_digest") or calculated != split.get("catalog_digest"):
                reasons.append("Catalog and split manifest content hashes differ")
            for phase, count in OFFICIAL_COUNTS.items():
                catalog_ids = {row["game_id"] for row in catalog["games"] if row["split"] == OFFICIAL_SPLITS[phase]}
                if self.phase_ids[phase] != catalog_ids or len(catalog_ids) != count:
                    reasons.append(f"{phase} must contain all {count} official catalog games")
            self.split, self.catalog_path = split, catalog_path
            self.split_manifest_path = path
            return section(
                "mismatch" if reasons else "declared_disjoint",
                reasons,
                [path, catalog_path],
                declared_counts={key: len(value) for key, value in self.phase_ids.items()},
            )
        except (EvidenceError, KeyError, TypeError) as error:
            self.phase_ids, self.split = {}, {}
            return section("incomplete", [str(error)], [path])

    def training(self):
        path = self.root / "metrics.jsonl"
        reasons, rows, valid = [], [], []
        try:
            rows = self.reader.rows(path)
        except EvidenceError as error:
            reasons.append(str(error))
        traces = {}
        for log in (self.root / "agent_logs").glob("step_*/*/task.log"):
            try:
                text = self.reader.bytes(log).decode(errors="replace")
                match = re.search(r"codex: launch in (\S+)", text)
                if match:
                    traces.setdefault(match.group(1), []).append(log.parent)
            except EvidenceError:
                continue
        self.trace_directories = traces
        supporting = {}
        native_failures = []
        for result in (self.root / "episodes").glob("*/attempt_*/result.json"):
            item = self.raw_episode(result.parent)
            data = item["result"]
            if data.get("dataset_phase") != "train" or data.get("role") == "counterfactual_probe":
                continue
            self.train_games.add(data.get("game_id"))
            if not item["valid"]:
                native_failures.append(
                    {"episode": item["path"], "engine_win": item["engine_win"], "reason": item["reason"]}
                )
                continue
            for directory in traces.get(data.get("workdir"), []):
                try:
                    step = int(directory.parent.name.removeprefix("step_"))
                    tokens = gateway_trace(self.reader, directory)
                    self.trace_cache[str(directory)] = tokens
                    self.token_supported_episodes.add(item["path"])
                    supporting.setdefault(step, []).append(
                        {"episode": item["path"], "gateway": str(directory), "sampled_action_tokens": tokens}
                    )
                except (EvidenceError, ValueError):
                    continue
        eligible = []
        for row in rows:
            if row.get("phase") != "train" or type(row.get("step")) is not int or row["step"] <= 0:
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                reasons.append(f"Train cursor{row['step']} lacks an object of actual metrics")
                continue
            clock = metrics.get("skill/optimizer_step")
            if clock is not None and (type(clock) is not int or clock <= 0):
                reasons.append(f"Invalid observed optimizer clock at train cursor{row['step']}")
                continue
            eligible.append(row)
        step_counts = {step: sum(row["step"] == step for row in eligible) for step in {row["step"] for row in eligible}}
        clock_counts = {}
        for row in eligible:
            clock = row.get("metrics", {}).get("skill/optimizer_step")
            if clock is not None:
                clock_counts[clock] = clock_counts.get(clock, 0) + 1
        for row in eligible:
            if row.get("phase") != "train" or type(row.get("step")) is not int or row["step"] <= 0:
                continue
            metrics = row.get("metrics", {})
            clock = metrics.get("skill/optimizer_step")
            if step_counts[row["step"]] > 1 or (clock is not None and clock_counts[clock] > 1):
                reasons.append(
                    f"Ambiguous duplicate train cursor/optimizer clock at step{row['step']}; not counted twice"
                )
                continue
            if clock is not None and (type(clock) is not int or clock <= 0):
                reasons.append(f"Invalid observed optimizer clock at train cursor{row['step']}")
                continue
            loss = metrics.get("actor/pg_loss", metrics.get("actor/loss"))
            if (
                not finite(metrics.get("actor/grad_norm"))
                or metrics["actor/grad_norm"] <= 0
                or not finite(loss)
                or not finite(metrics.get("train/reward"))
                or not finite(metrics.get("train/accuracy"))
                or not finite(metrics.get("train/episodes"))
                or metrics["train/episodes"] <= 0
            ):
                reasons.append(f"train step{row['step']} lacks finite nonzero gradient and finite episode/loss metrics")
                continue
            if not supporting.get(row["step"]):
                reasons.append(f"train step{row['step']} lacks linked clean native Task and numeric Gateway trajectory")
                continue
            valid.append(row)
            self.train_updates[row["step"]] = row
        if not valid:
            reasons.append("No supported real training update; diagnostic/system/kernel results are not substitutes")
        leak = self.train_games - self.phase_ids.get("train", set())
        if leak:
            reasons.append("Observed optimization Task game identities are outside the declared train partition")
        return section(
            "observed" if valid and not leak else "incomplete",
            reasons,
            [path],
            verified_update_count=len(valid),
            verified_steps=[row["step"] for row in valid],
            supporting_native_trajectories=supporting,
            native_failures=native_failures,
            observed_optimizer_clocks=[row["metrics"].get("skill/optimizer_step") for row in valid],
            batch_link_caveat=("Task cwd/Gateway/cursor are linked; train telemetry lacks exact TQ batch membership"),
            reward_trend_is_not_controlled_improvement=True,
            observed_train_game_ids=sorted(value for value in self.train_games if value),
        )

    def official_evaluation(self):
        stages, protocols = {}, {}
        for stage in ("initial", "final"):
            root = self.root / "evaluation" / stage
            reasons, splits, paths = [], {}, [root / "protocol.json", root / "summary.json"]
            try:
                protocol = self.reader.json(paths[0])
                summary = self.reader.json(paths[1])
                protocols[stage] = protocol
                if summary.get("completed") is not True or protocol.get("stage") != stage:
                    reasons.append("Stage summary/protocol is not completed and consistently named")
                if protocol.get("optimizer_updates") is not False or protocol.get("curation") is not False:
                    reasons.append("Official evaluation does not declare optimizer/curation isolation")
                dataset, budgets = protocol.get("dataset", {}), protocol.get("budgets", {})
                if protocol.get("agent") != "native_uni_agent_codex":
                    reasons.append("Stage protocol does not identify the native Codex agent")
                if protocol.get("policy") != "already_initialized_training_policy":
                    reasons.append("Stage protocol does not bind evaluation to the initialized training policy")
                if self.split_manifest_path is None:
                    reasons.append("Stage protocol cannot be bound to an accessible split manifest")
                else:
                    expected_hashes = {
                        "manifest_sha256": hashlib.sha256(self.reader.bytes(self.split_manifest_path)).hexdigest(),
                        "catalog_sha256": hashlib.sha256(self.reader.bytes(self.catalog_path)).hexdigest(),
                        "catalog_digest": self.split["catalog_digest"],
                    }
                    if any(dataset.get(key) != value for key, value in expected_hashes.items()):
                        reasons.append("Stage dataset hashes do not match the underlying split/catalog files")
                for phase, expected_count in OFFICIAL_COUNTS.items():
                    spec = dataset.get("splits", {}).get(phase, {})
                    if spec.get("games") != expected_count or not spec.get("path") or not spec.get("sha256"):
                        reasons.append(f"Stage protocol lacks the complete {phase} parquet identity/count")
                    elif (
                        hashlib.sha256(self.reader.bytes(_path(self.root, spec["path"]))).hexdigest() != spec["sha256"]
                    ):
                        reasons.append(f"The underlying {phase} parquet differs from the stage protocol")
                greedy = {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1}
                if budgets.get("sampling") != greedy:
                    reasons.append("Stage protocol lacks the complete fixed greedy n1 sampling contract")
                for name in ("prompt_length", "response_length", "max_model_len", "environment_max_steps"):
                    if type(budgets.get(name)) is not int or budgets[name] <= 0:
                        reasons.append(f"Stage protocol lacks a valid {name} budget")
                for name in ("alfworld_task_config_sha256", "alfworld_solver_instructions_file_sha256"):
                    if not isinstance(budgets.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", budgets[name]):
                        reasons.append(f"Stage protocol lacks its immutable {name}")
            except EvidenceError as error:
                reasons.append(str(error))
            for phase, expected_count in OFFICIAL_COUNTS.items():
                split_root = root / phase
                issues, outcomes, attempts = [], {}, {}
                ambiguous = set()
                observed_engine = {}
                files = sorted(split_root.glob("*.jsonl"))
                paths.extend(files)
                if not files:
                    issues.append("Missing underlying per-game validation trajectory dump")
                for file in files:
                    try:
                        for row in self.reader.rows(file):
                            pointer = row.get("episode_output_dir")
                            if (
                                not pointer
                                or not isinstance(row.get("input"), str)
                                or not row["input"]
                                or not isinstance(row.get("output"), str)
                            ):
                                issues.append(f"{file}: missing raw episode pointer or decoded input/output trajectory")
                                continue
                            item = self.raw_episode(pointer)
                            data = item["result"]
                            if type(item.get("engine_win")) is bool:
                                observed_engine[item["path"]] = item["engine_win"]
                            signature = (row.get("game_id"), row.get("score"), row.get("acc"), row.get("error"))
                            if pointer in attempts and attempts[pointer] != signature:
                                issues.append(f"{pointer}: conflicting outcomes across prefix trajectories")
                            attempts[pointer] = signature
                            if not item["valid"]:
                                issues.append(item["reason"])
                                continue
                            game = data.get("game_id")
                            if (
                                data.get("split") != OFFICIAL_SPLITS[phase]
                                or data.get("dataset_phase") != phase
                                or row.get("game_id") != game
                                or row.get("score") != data["reward"]
                                or row.get("acc") != data["accuracy"]
                                or row.get("error")
                                or row.get("eval_completed") is not True
                                or data.get("catalog_digest") != self.split.get("catalog_digest")
                            ):
                                issues.append(f"{pointer}: validation row/Task/catalog identities or outcomes differ")
                                continue
                            token_count = 0
                            for trace in self.trace_directories.get(data.get("workdir"), []):
                                try:
                                    token_count = self.trace_cache.get(str(trace)) or gateway_trace(self.reader, trace)
                                    self.trace_cache[str(trace)] = token_count
                                    break
                                except EvidenceError:
                                    continue
                            if not token_count:
                                issues.append(f"{pointer}: no linked numeric Gateway model trajectory")
                                continue
                            budget = protocols.get(stage, {}).get("budgets", {}).get("environment_max_steps")
                            if data.get("max_steps") != budget:
                                issues.append(f"{pointer}: actual environment step budget differs from stage protocol")
                            if game in outcomes and outcomes[game]["episode"] != item["path"]:
                                issues.append(f"{game}: more than one distinct attempt in this stage/split")
                                ambiguous.add(game)
                                outcomes.pop(game)
                            if game in ambiguous:
                                continue
                            outcomes[game] = {"won": item["engine_win"], "episode": item["path"]}
                    except EvidenceError as error:
                        issues.append(str(error))
                expected = self.phase_ids.get(phase, set())
                if len(outcomes) != expected_count or set(outcomes) != expected:
                    issues.append(
                        f"Verified unique games {len(outcomes)}/{expected_count}; prefix rows are not new games"
                    )
                splits[phase] = section(
                    "observed" if not issues else "incomplete",
                    issues,
                    files,
                    expected_games=expected_count,
                    verified_games=len(outcomes),
                    outcomes=outcomes,
                    engine_wins=sum(row["won"] for row in outcomes.values()),
                    observed_raw_engine_wins=sum(observed_engine.values()),
                    observed_raw_engine_attempts=len(observed_engine),
                    reward_mean=sum(row["won"] for row in outcomes.values()) / len(outcomes) if outcomes else None,
                    accuracy=sum(row["won"] for row in outcomes.values()) / len(outcomes) if outcomes else None,
                )
            complete = not reasons and all(split["status"] == "observed" for split in splits.values())
            stages[stage] = section(
                "observed" if complete else "incomplete",
                reasons,
                paths,
                splits=splits,
                reported_global_cursor=protocols.get(stage, {}).get("global_step"),
                optimizer_version_inferred_from_cursor=False,
            )
        same = (
            len(protocols) == 2
            and protocols["initial"].get("dataset") == protocols["final"].get("dataset")
            and protocols["initial"].get("budgets") == protocols["final"].get("budgets")
        )
        complete = same and all(stage["status"] == "observed" for stage in stages.values())
        comparisons = {}
        if all(
            stage["splits"][phase]["verified_games"] == expected
            for stage in stages.values()
            for phase, expected in OFFICIAL_COUNTS.items()
        ):
            for phase in OFFICIAL_COUNTS:
                before, after = (stages[name]["splits"][phase] for name in ("initial", "final"))
                comparisons[phase] = {
                    "initial_wins": before["engine_wins"],
                    "final_wins": after["engine_wins"],
                    "reward_delta": after["reward_mean"] - before["reward_mean"],
                    "accuracy_delta": after["accuracy"] - before["accuracy"],
                }
        delta = sum(row["final_wins"] - row["initial_wins"] for row in comparisons.values())
        return section(
            "observed" if complete else "incomplete",
            [] if complete else ["Complete raw140+134 outcomes and identical recorded protocols are required"],
            stages=stages,
            same_recorded_protocol=same,
            comparisons=comparisons,
            controlled_improvement_supported=bool(complete and delta > 0),
            overall_win_delta=delta if comparisons else None,
            runtime_identity_caveat="Stage protocols do not independently attest unchanged in-memory code/operators",
        )

    def delayed_lineage(self):
        harness = self.config.get("skill_harness", {})
        replay = _path(self.root, harness.get("replay_root", "curate_replay"))
        ledger = _path(self.root, harness.get("ledger_root", "credit_ledger"))
        store = _path(self.root, harness.get("store_root", "skill_store"))
        paths, reasons, lineages = [ledger / "state.json", self.root / "controller/state.json"], [], []
        try:
            state = self.reader.json(paths[0])
            controller = self.reader.json(paths[1])
        except EvidenceError as error:
            return section("incomplete", [str(error)], paths, complete_lineages=0, lineages=[])
        collections = {}
        for path in (self.root / "curate").glob("*/complete.json"):
            try:
                value = self.reader.json(path)
                collections[value["record_id"]] = path.parent
            except (EvidenceError, KeyError):
                continue
        probe_events = {}
        for path in (self.root / "credit_probes").glob("*/paired-event.json"):
            try:
                event = self.reader.json(path)
                probe_events[event["event_id"]] = path.parent
            except (EvidenceError, KeyError):
                continue
        for identity, application in state.get("settlements", {}).items():
            evidence = []
            try:
                settlement_path = ledger / "settlements" / (identity + ".json")
                settlement = self.reader.json(settlement_path)
                evidence.append(str(settlement_path))
                if digest(settlement) != application.get("payload_hash"):
                    raise EvidenceError("Ledger settlement state/object hash differs")
                record_path = replay / "records" / (settlement["curate_record_id"] + ".json")
                record = self.reader.json(record_path)
                evidence.append(str(record_path))
                window_path = ledger / "windows" / (settlement["window_id"] + ".json")
                window = self.reader.json(window_path)
                if (
                    window.get("curate_record_id") != record.get("record_id")
                    or window.get("replay_payload_hash") != digest(record)
                    or application.get("window_id") != settlement["window_id"]
                ):
                    raise EvidenceError("Attribution window is not bound to this exact retained producer archive")
                evidence.append(str(window_path))
                ref = record["skill_ref"]
                context = record["context_snapshot"]
                tokens, mask, probabilities = record["action_ids"], record["curate_mask"], record["behavior_logprobs"]
                if (
                    not record.get("prompt_ids")
                    or not tokens
                    or len(tokens) != len(mask)
                    or len(tokens) != len(probabilities)
                    or not any(mask)
                    or not all(finite(value) and value <= 0 for value in probabilities)
                    or record.get("behavior_policy_kind") != "ema_curator"
                    or settlement["skill_ref"] != ref
                    or context.get("task_metadata", {}).get("game_id") not in self.phase_ids.get("train", set())
                    or context.get("task_metadata", {}).get("official_split") != "train"
                    or context.get("task_metadata", {}).get("dataset_phase") != "train"
                ):
                    raise EvidenceError("No aligned native EMA curate archive from an optimization-train source")
                source_identity = context["task_metadata"]
                source_episodes = [
                    item
                    for item in self.episodes.values()
                    if (
                        item["valid"]
                        and item["path"] in self.token_supported_episodes
                        and item["result"].get("game_id") == source_identity["game_id"]
                        and item["result"].get("attempt_id") == source_identity.get("attempt_id")
                        and item["result"].get("dataset_phase") == "train"
                        and item["result"].get("started_at_epoch", math.inf) < record["created_at"]
                    )
                ]
                if len(source_episodes) != 1:
                    raise EvidenceError(
                        "Curate producer lacks its exact prior native solver Task/attempt/token trajectory"
                    )
                evidence.extend(source_episodes[0]["evidence_paths"])
                directory = collections.get(record["record_id"])
                if directory is None:
                    raise EvidenceError("Replay has no completed native curate collection")
                generated = self.reader.json(directory / "generated.json")
                complete = self.reader.json(directory / "complete.json")
                if (
                    complete.get("replay_ref", {}).get("payload_hash") != digest(record)
                    or complete.get("artifact_ref") != ref
                ):
                    raise EvidenceError("Native collection completion does not bind this exact replay/skill content")
                native = self.reader.rows(directory / "codex-events.jsonl")
                capture = generated.get("trajectories", [{}])[0]
                if (
                    generated.get("agent_ok") is not True
                    or generated.get("agent_finished") is not True
                    or capture.get("action_ids") != tokens
                    or capture.get("prompt_ids") != record["prompt_ids"]
                    or capture.get("curate_mask") != mask
                    or capture.get("behavior_logprobs") != probabilities
                    or not any(event.get("type") == "turn.completed" for event in native)
                ):
                    raise EvidenceError("Curate archive lacks matching actual native generated token evidence")
                artifact_path = store / "artifacts" / (ref["content_hash"] + ".json")
                artifact = self.reader.json(artifact_path)
                if (
                    digest({key: value for key, value in artifact.items() if key not in ("content_hash", "version")})
                    != ref["content_hash"]
                ):
                    raise EvidenceError("Skill content differs from its archived immutable version")
                evidence.extend(
                    map(str, (directory / "generated.json", directory / "codex-events.jsonl", artifact_path))
                )
                deltas = []
                for event_id in settlement["event_ids"]:
                    event_path = ledger / "events" / (event_id + ".json")
                    event = self.reader.json(event_path)
                    probe_dir = probe_events.get(event_id)
                    if probe_dir is None or self.reader.json(probe_dir / "paired-event.json") != event:
                        raise EvidenceError("Ledger credit has no identical durable native paired-probe event")
                    positive, control = event["with_skill"], event["without_skill"]
                    group = digest([event["window_id"], event["probe_id"]])
                    expected_cwd = "/tmp/alfworld-skill-episodes/credit_" + group
                    if (
                        event["skill_ref"] != ref
                        or positive.get("loaded_skill_ref") != ref
                        or event["window_id"] != settlement["window_id"]
                        or event["curate_record_id"] != record["record_id"]
                        or control.get("loaded_skill_ref") is not None
                        or event["occurred_at"] <= record["created_at"]
                        or positive["task_id"] not in self.phase_ids.get("train", set())
                        or any(
                            positive[key] != control[key]
                            for key in (
                                "task_id",
                                "model_version",
                                "environment_version",
                                "environment_state_hash",
                                "baseline_snapshot_id",
                            )
                        )
                    ):
                        raise EvidenceError("Counterfactual event is not a delayed matched train-only exposure")
                    attestations = []
                    workspaces, frozen_clocks = [], []
                    for name, outcome, loaded in (("with_skill", positive, True), ("without_skill", control, False)):
                        native_capture = self.reader.json(probe_dir / name / "native-capture.json")
                        raw = self.raw_episode(outcome["trajectory_ref"])
                        skill_context = raw["result"].get("skill_context", {})
                        prompt = self.reader.bytes(Path(raw["path"]) / "agent-prompt.txt").decode()
                        contains = ref in skill_context.get("injected_refs", [])
                        if (
                            not raw["valid"]
                            or float(raw["engine_win"]) != outcome["score"]
                            or not _tokens(native_capture)
                            or contains != loaded
                            or (ref["content_hash"] in prompt) != loaded
                            or raw["result"].get("game_id") != outcome["task_id"]
                            or raw["result"].get("split") != "train"
                            or raw["result"].get("dataset_phase") != "train"
                            or native_capture.get("reward") != outcome["score"]
                            or native_capture.get("accuracy") != outcome["score"]
                            or Path(
                                native_capture.get("extra_info", {}).get("episode_output_dir", "/missing")
                            ).resolve()
                            != Path(raw["path"])
                        ):
                            raise EvidenceError(
                                "Paired label lacks matching raw Task, native tokens, or actual skill injection"
                            )
                        initial = self.reader.json(Path(raw["path"]) / "initial-state.json")
                        if digest(initial) != outcome["environment_state_hash"]:
                            raise EvidenceError("Paired label's environment state hash differs from raw initialization")
                        workspace = raw["result"].get("paired_workdir", {})
                        if (
                            workspace.get("empty_before_run") is not True
                            or workspace.get("cleanup_completed") is not True
                            or workspace.get("kind") != "paired_empty_workdir"
                            or workspace.get("schema_version") != 1
                            or workspace.get("credit_probe_group") != group
                            or workspace.get("variant") != name
                            or workspace.get("workdir") != expected_cwd
                            or len(workspace.get("codex_homes", [])) != 1
                            or not re.fullmatch(r"/tmp/codex-home-[0-9a-f]{32}", workspace["codex_homes"][0])
                            or len(workspace.get("archives", [])) != 2
                        ):
                            raise EvidenceError(
                                "Counterfactual pair lacks controlled empty-cwd/archive/cleanup evidence"
                            )
                        if {row.get("source") for row in workspace["archives"]} != {
                            expected_cwd,
                            workspace["codex_homes"][0],
                        }:
                            raise EvidenceError(
                                "Counterfactual workspace and native home archives are not both retained"
                            )
                        for archived in workspace["archives"]:
                            archive = Path(archived["path"]).resolve()
                            if (
                                not archive.is_relative_to(Path(raw["path"]))
                                or archived.get("removed") is not True
                                or hashlib.sha256(self.reader.bytes(archive)).hexdigest() != archived.get("sha256")
                            ):
                                raise EvidenceError(
                                    "Counterfactual workspace archive is absent, changed, or not cleaned"
                                )
                        workspaces.append(workspace)
                        for trajectory in native_capture["trajectories"]:
                            extra = trajectory.get("extra_fields", {})
                            low, high = extra.get("min_global_steps"), extra.get("max_global_steps")
                            if low != high or (low is not None and (type(low) is not int or low < 0)):
                                raise EvidenceError("Frozen trajectory contains a mixed/invalid observed weight clock")
                            frozen_clocks.append(low)
                        attestations.append(native_capture.get("policy_attestation", {}))
                    if (
                        attestations[0] != attestations[1]
                        or attestations[0].get("policy_kind") != "frozen_evaluator"
                        or attestations[0].get("schema_version") != 1
                        or not attestations[0].get("lifecycle_id")
                        or attestations[0].get("frozen_model_version") != positive["model_version"]
                        or attestations[0].get("model_fingerprint") != positive["model_version"]
                        or not Path(attestations[0].get("model_path", "relative")).is_absolute()
                    ):
                        raise EvidenceError("Counterfactual branches lack one matching frozen evaluator lifecycle")
                    if (
                        workspaces[0].get("workdir") != workspaces[1].get("workdir")
                        or not workspaces[0].get("codex_homes")
                        or not workspaces[1].get("codex_homes")
                        or workspaces[0]["codex_homes"] == workspaces[1]["codex_homes"]
                        or any(clock != frozen_clocks[0] for clock in frozen_clocks)
                    ):
                        raise EvidenceError(
                            "Counterfactual cwd, native home isolation, or frozen weight clocks changed"
                        )
                    deltas.append(positive["score"] - control["score"])
                    evidence.extend(map(str, (event_path, probe_dir / "paired-event.json")))
                if deltas != settlement["contributions"]:
                    raise EvidenceError("Settled counterfactual contributions differ from raw paired outcomes")
                raw_utility = sum(deltas) / len(deltas) if settlement["aggregation"] == "mean" else sum(deltas)
                expected_utility = raw_utility - settlement["cost_weight"] * settlement["skill_cost"]
                if not finite(settlement["utility"]) or not math.isclose(expected_utility, settlement["utility"]):
                    raise EvidenceError(
                        "Settlement utility cannot be reconstructed from the observed outcome pairs/cost"
                    )
                applied = []
                for update_id, model_update in application.get("applied_updates", {}).items():
                    match = re.fullmatch(r"optimizer:(\d+):(.+)", model_update)
                    report_path = self.root / "controller/updates" / (update_id + ".json")
                    report = self.reader.json(report_path)
                    metrics = self.train_updates.get(report.get("global_step"), {}).get("metrics", {})
                    if (
                        not match
                        or match.group(2) != update_id
                        or int(match.group(1)) != report.get("optimizer_step")
                        or metrics.get("skill/optimizer_step") != report.get("optimizer_step")
                        or controller.get("applied_update_ids", {}).get(update_id) != model_update
                        or not finite(metrics.get("skill/curate_loss"))
                        or not finite(metrics.get("skill/replay_candidates"))
                        or metrics["skill/replay_candidates"] <= 0
                    ):
                        raise EvidenceError(
                            "Reported replay application has no matching real optimizer clock and consumed curate batch"
                        )
                    applied.append(update_id)
                    evidence.append(str(report_path))
                if not applied:
                    raise EvidenceError("Settlement exists but no independently supported applied update")
                loaded = []
                for item in list(self.episodes.values()):
                    data = item["result"]
                    if (
                        item["valid"]
                        and item["path"] in self.token_supported_episodes
                        and data.get("dataset_phase") == "train"
                        and data.get("role") != "counterfactual_probe"
                    ):
                        path = Path(item["path"])
                        if ref in data.get("skill_context", {}).get("injected_refs", []):
                            prompt = self.reader.bytes(path / "agent-prompt.txt").decode()
                            if ref["content_hash"] in prompt and data.get("started_at_epoch", 0) > record["created_at"]:
                                snapshot_id = data["skill_context"]["snapshot_id"]
                                snapshot = self.reader.json(store / "snapshots" / (snapshot_id + ".json"))
                                if digest(snapshot) != snapshot_id or ref not in snapshot.get("skills", []):
                                    raise EvidenceError(
                                        "Actual loaded SkillContext does not belong to its immutable snapshot"
                                    )
                                loaded.append(item["path"])
                if not loaded:
                    raise EvidenceError("No later native solver Task actually loaded this archived skill version")
                lineages.append(
                    {
                        "status": "observed",
                        "record_id": record["record_id"],
                        "skill_ref": ref,
                        "settlement_id": identity,
                        "utility": settlement["utility"],
                        "counterfactual_deltas": deltas,
                        "applied_update_ids": applied,
                        "curate_branch_gradient": "not separately measured; zero scalar loss can have nonzero gradient",
                        "loaded_solver_episodes": loaded,
                        "evidence_paths": sorted(set(evidence)),
                    }
                )
            except (EvidenceError, KeyError, TypeError, ValueError, AttributeError, OSError) as error:
                lineages.append(
                    {
                        "status": "incomplete",
                        "settlement_id": identity,
                        "reason": str(error),
                        "evidence_paths": sorted(set(evidence)),
                    }
                )
        complete = sum(row["status"] == "observed" for row in lineages)
        if not complete:
            reasons.append(
                "No complete archived-curate -> delayed paired credit -> real applied update -> solver-load lineage"
            )
        return section(
            "observed" if complete else "incomplete",
            reasons,
            paths,
            archived_record_files=len(list((replay / "records").glob("*.json"))),
            complete_lineages=complete,
            lineages=lineages,
            tokenization_caveat="Host injection and native tokens linked; skill text not independently token-decoded",
        )

    def run(self):
        report = {
            "schema_version": 1,
            "kind": "training_evidence_index",
            "run_root": str(self.root),
            "generated_at_epoch": time.time(),
            "acceptance_decision": None,
            "purpose": "Reviewer aid; neither manifest declarations nor green unit tests establish experiment success",
        }
        for name, collect in (
            ("provenance", self.provenance),
            ("separation", self.separation),
            ("training_activity", self.training),
            ("official_evaluation", self.official_evaluation),
            ("delayed_skill_lineage", self.delayed_lineage),
        ):
            before = set(self.reader.observed)
            try:
                report[name] = collect()
            except (EvidenceError, KeyError, TypeError, ValueError, AttributeError, OSError) as error:
                report[name] = section(
                    "incomplete",
                    [f"Malformed/unavailable artifact data: {type(error).__name__}: {error}"],
                    set(self.reader.observed) - before,
                )
        changed = self.reader.changed()
        report["inputs_changed_during_audit"] = changed
        report["objective_support"] = {
            "native_training_observed": report["training_activity"].get("verified_update_count", 0) > 0,
            "delayed_skill_update_and_load_observed": report["delayed_skill_lineage"].get("complete_lineages", 0) > 0,
            "complete_recorded_protocol_evaluation": report["official_evaluation"]["status"] == "observed",
            "positive_controlled_delta_observed": report["official_evaluation"].get(
                "controlled_improvement_supported", False
            ),
        }
        report["review_status"] = (
            "incomplete_evidence"
            if changed
            or any(
                report[key]["status"] in ("incomplete", "mismatch")
                for key in (
                    "provenance",
                    "separation",
                    "training_activity",
                    "official_evaluation",
                    "delayed_skill_lineage",
                )
            )
            else "review_required"
        )
        report["limitations"] = [
            "Artifacts cannot independently attest the loaded in-memory model or optimizer gradients.",
            "Live runs may advance; changed or partially written artifacts require another audit after a durable cut.",
            "A training-reward rise on different tasks is never used as controlled improvement evidence.",
        ]
        return report


def audit_run(root):
    return RunAudit(root).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output", type=Path, help="New evidence-index JSON path; existing files are never overwritten"
    )
    args = parser.parse_args(argv)
    text = json.dumps(audit_run(args.run_root), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
