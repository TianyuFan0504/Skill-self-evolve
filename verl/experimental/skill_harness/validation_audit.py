# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Host-only validation coverage and score reconciliation, independent of Ray.

Scoped proxies observe the existing rollout/sampler calls; they never resample,
change rewards, or place private Task context in numeric reward metrics.
"""

from __future__ import annotations

import copy
import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path


class ValidationCoverageError(RuntimeError):
    pass


def stamp_task_identity(tools_kwargs, *, rollout_uid, session_index, global_step, evaluation=None):
    """Copy trusted host identity into Task metadata, leaving prompts untouched."""
    tools = copy.deepcopy(tools_kwargs or {})
    task = tools.get("task")
    if not isinstance(task, dict) or task.get("name") != "alfworld_skill_training":
        return tools
    if not isinstance(rollout_uid, str) or not rollout_uid:
        raise ValueError("A native Task requires its actual nonempty rollout UID")
    if type(session_index) is not int or session_index < 0:
        raise ValueError("Native Task session_index must be a nonnegative host integer")
    if global_step is not None and (type(global_step) is not int or global_step < 0):
        raise ValueError("Native Task global_step is a host cursor, never an inferred optimizer version")
    metadata = task.setdefault("metadata", {})
    metadata.update(rollout_uid=rollout_uid, session_index=session_index, global_step=global_step)
    if evaluation is not None:
        metadata.update(evaluation_audit_id=evaluation["audit_id"], evaluation_stage=evaluation["stage"])
    return tools


def plain(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    for name in ("tolist", "item"):
        method = getattr(value, name, None)
        if callable(method):
            return plain(method())
    return value


class ValidationCaseAudit:
    def __init__(self, *, run_root, output_dir, stage, phase, split, global_step, expected_ids, catalog_digest):
        self.run_root, self.output_dir = Path(run_root).resolve(), Path(output_dir).resolve()
        self.stage, self.phase, self.split = stage, phase, split
        self.global_step, self.expected_ids, self.catalog_digest = global_step, set(expected_ids), catalog_digest
        self.audit_id = uuid.uuid4().hex
        self.scope = {"audit_id": self.audit_id, "stage": stage}
        self.submitted, self.materialized = {}, {}
        self.report_path = self.output_dir / ("case-session-audit-" + self.audit_id + ".json")

    def submit(self, uids, tools, rollout_ns=None):
        if len(uids) != len(tools):
            raise ValueError("Validation UID and Task columns differ in length")
        output = []
        for index, (uid, kwargs) in enumerate(zip(uids, tools, strict=True)):
            uid = str(uid)
            task = kwargs.get("task", {})
            metadata = task.get("metadata", {})
            game = metadata.get("game_id")
            if (
                task.get("name") != "alfworld_skill_training"
                or game not in self.expected_ids
                or metadata.get("split") != self.split
                or metadata.get("dataset_phase") != self.phase
                or metadata.get("catalog_digest") != self.catalog_digest
                or Path(metadata.get("run_dir", "/missing")).resolve() != self.run_root
                or (rollout_ns is not None and int(rollout_ns[index]) != 1)
            ):
                raise ValueError("Official validation submission differs from its phase/catalog/n1 protocol")
            if uid in self.submitted or any(row["game_id"] == game for row in self.submitted.values()):
                raise ValueError("An official game/UID cannot be submitted twice in one evaluation")
            self.submitted[uid] = {"game_id": game, "session_index": 0}
            self.submitted[uid]["game_sha256"] = metadata.get("game_sha256")
            output.append(
                stamp_task_identity(
                    kwargs,
                    rollout_uid=uid,
                    session_index=0,
                    global_step=self.global_step,
                    evaluation=self.scope,
                )
            )
        return output

    def capture(self, rows):
        for row in rows:
            value = plain(row)
            uid, session = str(value["rollout_uid"]), value["session_index"]
            if uid not in self.submitted or session != 0:
                raise ValueError("Materialized validation session does not match an expected n1 UID")
            key = value["trajectory_key"]
            if key in self.materialized and self.materialized[key] != value:
                raise ValueError("Materialized validation trajectory identity changed")
            self.materialized[key] = value

    def _attempts(self):
        matches = {}
        for path in (self.run_root / "episodes").glob("*/attempt_*/attempt.json"):
            try:
                metadata = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            uid = metadata.get("rollout_uid")
            if (
                uid in self.submitted
                and metadata.get("session_index") == 0
                and metadata.get("global_step") == self.global_step
                and metadata.get("dataset_phase") == self.phase
                and metadata.get("split") == self.split
                and metadata.get("evaluation_audit_id") == self.audit_id
                and metadata.get("evaluation_stage") == self.stage
            ):
                matches.setdefault(uid, []).append(path.parent)
        return matches

    def finish(self, *, dump_path, parent_metrics=None, parent_error=None, require_complete=True):
        """Persist all expected cases, including unobserved ones; never invent zero."""
        reasons, cases, dumps = [], [], {}
        try:
            for line in Path(dump_path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                uid, session, _ = row["uid"].rsplit("_", 2)
                dumps.setdefault((uid, int(session)), []).append(row)
        except (OSError, ValueError, KeyError) as error:
            reasons.append(f"Validation dump missing or invalid: {type(error).__name__}")
        attempts = self._attempts()
        by_game = {row["game_id"]: uid for uid, row in self.submitted.items()}
        observed_weights = set()
        for game in sorted(self.expected_ids):
            uid = by_game.get(game)
            case = {
                "game_id": game,
                "rollout_uid": uid,
                "session_index": 0,
                "engine_won": None,
                "authoritative_reward": None,
                "authoritative_accuracy": None,
                "native_error": None,
                "trajectory_keys": [],
                "parent_scores": [],
                "reasons": [],
            }
            if uid is None:
                case["reasons"].append("Official game was not submitted")
            directories = attempts.get(uid, [])
            if len(directories) != 1:
                case["reasons"].append(f"Expected exactly one raw Task attempt, observed {len(directories)}")
            else:
                directory = directories[0]
                case["episode_output_dir"] = str(directory)
                try:
                    task = json.loads((directory / "result.json").read_text())
                    engine = json.loads((directory / "engine-result.json").read_text())
                    if any(
                        task.get(key) != engine.get(key)
                        for key in ("game_id", "game_sha256", "split", "catalog_digest")
                    ):
                        raise ValueError("Raw Task/engine identity differs")
                    if (
                        task.get("game_id") != game
                        or task.get("catalog_digest") != self.catalog_digest
                        or task.get("rollout_uid") != uid
                        or task.get("session_index") != 0
                        or task.get("global_step") != self.global_step
                        or task.get("dataset_phase") != self.phase
                        or task.get("split") != self.split
                        or (
                            self.submitted[uid]["game_sha256"] is not None
                            and task.get("game_sha256") != self.submitted[uid]["game_sha256"]
                        )
                        or task.get("evaluation_audit_id") != self.audit_id
                        or task.get("evaluation_stage") != self.stage
                        or type(engine.get("won")) is not bool
                    ):
                        raise ValueError("Raw Task does not match the submitted evaluation identity")
                    won = engine["won"]
                    case.update(
                        engine_won=won,
                        authoritative_reward=float(won),
                        authoritative_accuracy=float(won),
                        native_error=task.get("agent_error")
                        or (None if task.get("agent_ok") is True else "native_not_ok"),
                        attempt_id=task.get("attempt_id"),
                        expected_host_terminal_stop=task.get("expected_host_terminal_stop") is True,
                        expected_backend_terminal_stop=task.get("expected_backend_terminal_stop") is True,
                        backend_stop_confirmed=task.get("backend_stop_confirmed"),
                        terminal_stop=task.get("terminal_stop"),
                    )
                    if task.get("reward") != float(won) or task.get("accuracy") != float(won):
                        case["reasons"].append("Task reward/accuracy differs from authoritative engine won")
                except (OSError, ValueError) as error:
                    case["reasons"].append(f"Raw Task/engine unobserved or inconsistent: {error}")
            materialized = [value for value in self.materialized.values() if value["rollout_uid"] == uid]
            case["trajectory_keys"] = [value["trajectory_key"] for value in materialized]
            case["parent_scores"] = [value["score"] for value in materialized]
            case["reward_context"] = [value.get("reward_context", {}) for value in materialized]
            if not materialized or sum(value.get("sampled_action_tokens", 0) for value in materialized) <= 0:
                case["reasons"].append("No observed numeric model trajectory; parent padding is not an outcome")
            dumped = dumps.get((uid, 0), [])
            if not dumped:
                case["reasons"].append("No emitted validation score row for this expected session")
            for value in materialized:
                context = value.get("reward_context", {})
                if (
                    not context
                    or context.get("rollout_uid") != uid
                    or context.get("session_index") != 0
                    or context.get("game_id") != game
                    or Path(context.get("episode_output_dir", "/missing")).resolve()
                    != Path(case.get("episode_output_dir", "/unobserved")).resolve()
                ):
                    case["reasons"].append("Protected runner context does not match the raw Task/session")
                low, high = value.get("min_global_steps"), value.get("max_global_steps")
                if low != high or (low is not None and (type(low) is not int or low < 0)):
                    case["reasons"].append("Materialized trajectory contains a mixed/invalid observed weight span")
                elif low is not None:
                    observed_weights.add(low)
            for score in [*case["parent_scores"], *[row.get("score") for row in dumped]]:
                if type(score) not in (float, int) or not math.isfinite(score) or score != case["authoritative_reward"]:
                    case["reasons"].append("Materialized/dumped score does not reconcile with raw engine won")
            if any(row.get("acc") != case["authoritative_accuracy"] for row in dumped):
                case["reasons"].append("Dumped accuracy does not reconcile with raw engine won")
            case["status"] = "observed" if not case["reasons"] else "incomplete"
            cases.append(case)
        if len(observed_weights) > 1:
            reasons.append("Official evaluation changed observed model weight versions")
        expected_uid_sessions = {(uid, 0) for uid in self.submitted}
        if set(dumps) - expected_uid_sessions:
            reasons.append("Validation dump contains sessions outside this scoped submission")
        if parent_error:
            reasons.append("Parent validation raised " + str(parent_error))
        raw_cases = [case for case in cases if case["engine_won"] is not None]
        materialized_cases = [case for case in cases if case["trajectory_keys"]]
        observed_scores = [
            case["parent_scores"][0]
            for case in materialized_cases
            if case["parent_scores"] and type(case["parent_scores"][0]) in (int, float)
        ]
        complete = not reasons and all(case["status"] == "observed" for case in cases)
        report = {
            "schema_version": 1,
            "kind": "official_validation_case_audit",
            "audit_id": self.audit_id,
            "stage": self.stage,
            "phase": self.phase,
            "global_cursor": self.global_step,
            "optimizer_version_inferred_from_cursor": False,
            "observed_weight_versions": sorted(observed_weights),
            "expected_cases": len(self.expected_ids),
            "submitted_cases": len(self.submitted),
            "raw_task_cases": len(raw_cases),
            "materialized_cases": len(materialized_cases),
            "authoritative_wins": sum(case["engine_won"] for case in raw_cases),
            "authoritative_reward_mean": sum(case["engine_won"] for case in raw_cases) / len(cases)
            if len(raw_cases) == len(cases)
            else None,
            "authoritative_accuracy": sum(case["engine_won"] for case in raw_cases) / len(cases)
            if len(raw_cases) == len(cases)
            else None,
            "parent_observed_reward_mean": sum(observed_scores) / len(observed_scores) if observed_scores else None,
            "parent_observed_reward_denominator": len(observed_scores),
            "expected_case_denominator": len(cases),
            "native_error_cases": sum(bool(case["native_error"]) for case in cases),
            "expected_host_terminal_stop_cases": sum(case.get("expected_host_terminal_stop", False) for case in cases),
            "expected_backend_terminal_stop_cases": sum(
                case.get("expected_backend_terminal_stop", False) for case in cases
            ),
            "backend_unconfirmed_cases": sum(case.get("backend_stop_confirmed") is not True for case in raw_cases),
            "unexpected_native_error_cases": sum(
                bool(case["native_error"]) and not case.get("expected_host_terminal_stop", False)
                and not case.get("expected_backend_terminal_stop", False) for case in cases
            ),
            "complete": complete,
            "reasons": reasons,
            "cases": cases,
            "parent_metrics": plain(parent_metrics or {}),
            "dump_path": str(dump_path),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.report_path.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if require_complete and not complete:
            raise ValidationCoverageError(f"Official cases/scores are incomplete; see {self.report_path}")
        return report


class SubmissionAuditProxy:
    def __init__(self, delegate, audit, prepare_batch):
        self.delegate, self.audit, self.prepare_batch = delegate, audit, prepare_batch

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate_sequences(self, batch):
        self.prepare_batch(batch, self.audit)
        return self.delegate.generate_sequences(batch)


class SamplerAuditProxy:
    def __init__(self, delegate, audit, read_rows):
        self.delegate, self.audit, self.read_rows = delegate, audit, read_rows

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def sample(self, *args, **kwargs):
        result = self.delegate.sample(*args, **kwargs)
        partition = kwargs.get("partition_id", args[1] if len(args) > 1 else None)
        if partition == "val" and result[0].keys:
            self.audit.capture(self.read_rows(result[0]))
        return result
