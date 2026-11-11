"""Evidence scenarios; fixtures are schema examples, never benchmark results."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_training_run.py"
SPEC = importlib.util.spec_from_file_location("_run_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def sha(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value, *, wrapped=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrapped:
        value = {"schema_version": 1, "payload_hash": sha(value), "payload": value}
    path.write_text(json.dumps(value) + "\n")


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def npy(values):
    header = repr({"descr": "<i8", "fortran_order": False, "shape": (len(values),)}).encode()
    header += b" " * ((-len(header) - 11) % 16) + b"\n"
    return (
        b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + "q" * len(values), *values)
    )


class Fixture:
    def __init__(self, root):
        self.root = root
        source, model = root / "source", root / "model"
        (source / "verl").mkdir(parents=True)
        (source / "uni_agent").mkdir()
        (source / "verl/__init__.py").write_text("# Fixture core declaration only\n")
        (source / "uni_agent/__init__.py").write_text("# Fixture runtime declaration only\n")
        model.mkdir()
        write(model / "config.json", {"architectures": ["Qwen3_5ForConditionalGeneration"]})
        write(model / "model.safetensors.index.json", {"weight_map": {"weight": "model.safetensors"}})
        (model / "model.safetensors").write_bytes(b"fixture bytes; no model execution")
        (root / "native-binary-fixture").write_text("not executed\n")
        self.ids = {
            "train": ["train_a", "train_b"],
            "dev": ["train_dev"],
            "test_seen": [f"seen_{index:03d}" for index in range(140)],
            "test_unseen": [f"unseen_{index:03d}" for index in range(134)],
        }
        self.games = []
        for phase, identities in self.ids.items():
            split = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}.get(phase, "train")
            self.games.extend(
                {"game_id": game, "split": split, "relative_game_path": f"{split}/{game}/game.tw-pddl"}
                for game in identities
            )
        self.catalog_digest = sha(sorted(self.games, key=lambda row: row["relative_game_path"]))
        write(
            root / "data/catalog.json",
            {"schema_version": 1, "games": self.games, "catalog_digest": self.catalog_digest},
        )
        split = {
            "schema_version": 1,
            "phase_game_ids": self.ids,
            "evaluation_games_used_for_training": False,
            "catalog_digest": self.catalog_digest,
        }
        write(root / "data/manifest.json", split)
        config = {
            "actor_rollout_ref": {"model": {"path": str(model)}},
            "skill_harness": {
                "split_manifest": str(root / "data/manifest.json"),
                "store_root": str(root / "skill_store"),
                "replay_root": str(root / "curate_replay"),
                "ledger_root": str(root / "credit_ledger"),
            },
        }
        write(
            root / "training-manifest.json",
            {
                "schema_version": 1,
                "source_commit": "a" * 40,
                "config": config,
                "import_provenance": {
                    "verl_version": "0.9.1",
                    "verl_source": str(source),
                    "verl_origin": str(source / "verl/__init__.py"),
                    "uni_agent_origin": str(source / "uni_agent/__init__.py"),
                },
            },
        )
        write(
            root / "launch_manifest.json",
            {"environment_paths": {"CODEX_NATIVE_BIN": str(root / "native-binary-fixture")}},
        )
        self.metric_rows = []

    def episode(self, game, phase, attempt, won, *, role=None, skill_context=None):
        directory = self.root / "episodes" / game / ("attempt_" + attempt)
        split = {"test_seen": "valid_seen", "test_unseen": "valid_unseen"}.get(phase, "train")
        identity = {
            "game_id": game,
            "stable_game_id": game,
            "game_sha256": hashlib.sha256(game.encode()).hexdigest(),
            "split": split,
            "dataset_phase": phase,
            "catalog_digest": self.catalog_digest,
            "attempt_id": attempt,
            "started_at_epoch": 101,
            "workdir": "/tmp/fixture-" + attempt,
        }
        if role:
            identity["role"] = role
        result = {
            **identity,
            "won": won,
            "reward": float(won),
            "accuracy": float(won),
            "agent_ok": True,
            "agent_error": None,
            "agent_exit_code": 0,
            "steps": 1,
            "max_steps": 50,
            "harness": "Uni-Agent native CodexAgent",
        }
        if skill_context:
            result["skill_context"] = skill_context
        write(directory / "attempt.json", identity)
        write(directory / "result.json", result)
        write(
            directory / "engine-result.json",
            {**identity, "won": won, "reward": float(won), "steps": 1, "max_steps": 50},
        )
        write(directory / "initial-state.json", {"observation": "Same fixture kitchen", "steps": 0})
        rows(directory / "environment-trace.jsonl", [{"step": 1, "action": "look", "after": {"won": won}}])
        rows(
            directory / "codex-events.jsonl",
            [
                {"type": "item.completed", "item": {"type": "agent_message", "text": "Fixture terminal message"}},
                {"type": "turn.completed"},
            ],
        )
        (directory / "codex-stderr.log").write_text("")
        (directory / "agent-prompt.txt").write_text(json.dumps(skill_context or {}) + "\nOriginal task")
        return directory, result

    def numeric_trace(self, result, step, session):
        gateway = self.root / "agent_logs" / ("step_" + str(step)) / session
        gateway.mkdir(parents=True)
        (gateway / "task.log").write_text("codex: launch in " + result["workdir"] + "\n")
        write(gateway / "trajectory.json", {"num_trajectories": 1, "trajectories": [{"model_token_count": 1}]})
        with zipfile.ZipFile(gateway / "trajectory.npz", "w") as archive:
            for name, values in {"prompt_ids": [10, 20], "response_ids": [30], "response_mask": [1]}.items():
                archive.writestr("traj0_" + name + ".npy", npy(values))

    def training(self, step=1, *, gradient=1.25, skill_context=None, optimizer_step=None):
        directory, result = self.episode("train_b", "train", f"train{step}", True, skill_context=skill_context)
        self.numeric_trace(result, step, "session_" + str(step))
        self.metric_rows.append(
            {
                "schema_version": 1,
                "phase": "train",
                "step": step,
                "metrics": {
                    "actor/grad_norm": gradient,
                    "actor/pg_loss": -0.2,
                    "train/episodes": 1,
                    "train/reward": 1.0,
                    "train/accuracy": 1.0,
                    "skill/optimizer_step": optimizer_step or step,
                    "skill/curate_loss": -0.125,
                    "skill/replay_candidates": 1,
                },
            }
        )
        rows(self.root / "metrics.jsonl", self.metric_rows)
        return directory

    def evaluations(self):
        files = {}
        for phase in ("test_seen", "test_unseen"):
            path = self.root / "data" / (phase + ".parquet")
            path.write_bytes(b"PAR1-synthetic-schema-fixture-only-PAR1")
            files[phase] = {
                "games": len(self.ids[phase]),
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        dataset = {
            "catalog_digest": self.catalog_digest,
            "manifest_sha256": hashlib.sha256((self.root / "data/manifest.json").read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256((self.root / "data/catalog.json").read_bytes()).hexdigest(),
            "splits": files,
        }
        budgets = {
            "sampling": {"n": 1, "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1},
            "environment_max_steps": 50,
            "prompt_length": 512,
            "response_length": 512,
            "max_model_len": 1024,
            "alfworld_task_config_sha256": "a" * 64,
            "alfworld_solver_instructions_file_sha256": "b" * 64,
        }
        for stage in ("initial", "final"):
            root = self.root / "evaluation" / stage
            write(
                root / "protocol.json",
                {
                    "schema_version": 1,
                    "stage": stage,
                    "dataset": dataset,
                    "budgets": budgets,
                    "optimizer_updates": False,
                    "curation": False,
                    "agent": "native_uni_agent_codex",
                    "policy": "already_initialized_training_policy",
                    "global_step": 1 if stage == "initial" else 3,
                },
            )
            write(root / "summary.json", {"completed": True, "stage": stage})
            for phase in ("test_seen", "test_unseen"):
                dump = []
                initial_wins = len(self.ids[phase]) // 2
                for index, game in enumerate(self.ids[phase]):
                    won = index < initial_wins + int(stage == "final")
                    directory, result = self.episode(game, phase, f"{stage}-{phase}-{index}", won)
                    self.numeric_trace(result, 1 if stage == "initial" else 3, f"{stage}-{phase}-{index}")
                    dump.append(
                        {
                            "episode_output_dir": str(directory),
                            "attempt_id": result["attempt_id"],
                            "game_id": game,
                            "score": float(won),
                            "acc": float(won),
                            "error": None,
                            "eval_completed": True,
                            "input": "Fixture full native task",
                            "output": "Fixture response",
                        }
                    )
                rows(root / phase / "0.jsonl", dump)

    def sidecar_evaluations(self, *, prefixes=False):
        """Production format: only a scoped sidecar contains raw Task pointers."""
        self.evaluations()
        self.case_reports = {}
        for stage in ("initial", "final"):
            root = self.root / "evaluation" / stage
            cursor = 1 if stage == "initial" else 3
            summary = {"completed": True, "stage": stage, "global_step": cursor, "case_audits": {}}
            for phase in ("test_seen", "test_unseen"):
                old_path = root / phase / "0.jsonl"
                old = [json.loads(line) for line in old_path.read_text().splitlines()]
                old_path.unlink()
                cases, dump = [], []
                audit_id = sha([stage, phase])[:32]
                for index, row in enumerate(old):
                    directory = Path(row["episode_output_dir"])
                    uid = f"rollout_{stage}_{phase}_{index}"
                    identity = {
                        "rollout_uid": uid,
                        "session_index": 0,
                        "global_step": cursor,
                        "evaluation_audit_id": audit_id,
                        "evaluation_stage": stage,
                    }
                    for name in ("attempt.json", "result.json"):
                        payload = json.loads((directory / name).read_text())
                        write(directory / name, {**payload, **identity})
                    keys, contexts, scores = [], [], []
                    for prefix in range(2 if prefixes and index == 0 else 1):
                        key = uid + "_0_" + str(prefix)
                        keys.append(key)
                        scores.append(row["score"])
                        contexts.append(
                            {
                                "rollout_uid": uid,
                                "session_index": 0,
                                "game_id": row["game_id"],
                                "episode_output_dir": str(directory),
                            }
                        )
                        dump.append(
                            {
                                "uid": key,
                                "step": cursor,
                                "input": row["input"],
                                "output": row["output"],
                                "gts": "Fixture reward",
                                "score": row["score"],
                                "reward": row["score"],
                                "acc": row["acc"],
                            }
                        )
                    cases.append(
                        {
                            "game_id": row["game_id"],
                            "rollout_uid": uid,
                            "session_index": 0,
                            "engine_won": bool(row["score"]),
                            "authoritative_reward": row["score"],
                            "authoritative_accuracy": row["acc"],
                            "native_error": None,
                            "status": "observed",
                            "trajectory_keys": keys,
                            "parent_scores": scores,
                            "reasons": [],
                            "episode_output_dir": str(directory),
                            "attempt_id": row["attempt_id"],
                            "reward_context": contexts,
                            "expected_host_terminal_stop": False,
                            "expected_backend_terminal_stop": False,
                            "backend_stop_confirmed": None,
                            "terminal_stop": None,
                        }
                    )
                dump_path = root / phase / f"{cursor}.jsonl"
                rows(dump_path, dump)
                count, wins = len(cases), sum(case["engine_won"] for case in cases)
                report = {
                    "schema_version": 1,
                    "kind": "official_validation_case_audit",
                    "audit_id": audit_id,
                    "stage": stage,
                    "phase": phase,
                    "global_cursor": cursor,
                    "optimizer_version_inferred_from_cursor": False,
                    "observed_weight_versions": [cursor - 1],
                    "expected_cases": count,
                    "submitted_cases": count,
                    "raw_task_cases": count,
                    "materialized_cases": count,
                    "authoritative_wins": wins,
                    "authoritative_reward_mean": wins / count,
                    "authoritative_accuracy": wins / count,
                    "parent_observed_reward_mean": wins / count,
                    "parent_observed_reward_denominator": count,
                    "expected_case_denominator": count,
                    "native_error_cases": 0,
                    "complete": True,
                    "reasons": [],
                    "cases": cases,
                    "parent_metrics": {},
                    "dump_path": str(dump_path),
                }
                path = root / phase / ("case-session-audit-" + audit_id + ".json")
                write(path, report)
                self.case_reports[stage, phase] = path
                summary["case_audits"][phase] = str(path)
            write(root / "summary.json", summary)

    def lineage(self, *, actual_update=True):
        source, source_result = self.episode("train_a", "train", "producer", False)
        for name in ("result.json", "attempt.json", "engine-result.json"):
            payload = json.loads((source / name).read_text())
            payload["started_at_epoch"] = 99
            write(source / name, payload)
        self.numeric_trace(source_result, 1, "producer")
        artifact = {
            "schema_version": 1,
            "skill_id": "heat",
            "title": "Heat procedure",
            "principle": "Take then heat",
            "when_to_apply": "Heating tasks",
            "task_families": ["heat"],
            "provenance": {"record": "record_1"},
        }
        content_hash = sha(artifact)
        ref = {"skill_id": "heat", "version": "sha256:" + content_hash, "content_hash": content_hash}
        write(self.root / "skill_store/artifacts" / (content_hash + ".json"), {**artifact, **ref})
        snapshot = {"schema_version": 1, "parent_id": None, "skills": [ref], "provenance": {"operation": "fixture"}}
        snapshot_id = sha(snapshot)
        write(self.root / "skill_store/snapshots" / (snapshot_id + ".json"), snapshot)
        context = {"enabled": True, "snapshot_id": snapshot_id, "injected_refs": [ref]}
        self.training(2, skill_context=context)
        if not actual_update:
            self.metric_rows[-1]["metrics"]["skill/optimizer_step"] = 1
            rows(self.root / "metrics.jsonl", self.metric_rows)
        record = {
            "record_id": "record_1",
            "producer_uid": "source_a",
            "skill_ref": ref,
            "prompt_ids": [1, 2],
            "action_ids": [3],
            "curate_mask": [1],
            "behavior_logprobs": [-0.2],
            "behavior_policy_kind": "ema_curator",
            "created_at": 100,
            "context_snapshot": {
                "task_metadata": {
                    "game_id": "train_a",
                    "official_split": "train",
                    "dataset_phase": "train",
                    "attempt_id": "producer",
                }
            },
        }
        write(self.root / "curate_replay/records/record_1.json", record, wrapped=True)
        collection = self.root / "curate/collection_1"
        write(
            collection / "complete.json",
            {"record_id": "record_1", "replay_ref": {"payload_hash": sha(record)}, "artifact_ref": ref},
            wrapped=True,
        )
        capture = {key: record[key] for key in ("prompt_ids", "action_ids", "curate_mask", "behavior_logprobs")}
        write(
            collection / "generated.json",
            {"agent_ok": True, "agent_finished": True, "trajectories": [capture]},
            wrapped=True,
        )
        rows(collection / "codex-events.jsonl", [{"type": "turn.completed"}])
        probe = self.root / "credit_probes/probe_1"
        outcomes = {}
        attestation = {
            "schema_version": 1,
            "policy_kind": "frozen_evaluator",
            "frozen_model_version": "f" * 64,
            "model_fingerprint": "f" * 64,
            "model_path": str(self.root / "model"),
            "lifecycle_id": "one",
        }
        group = sha(["window_1", "probe_1"])
        for loaded in (True, False):
            name = "with_skill" if loaded else "without_skill"
            episode_dir, _ = self.episode(
                "train_b",
                "train",
                name,
                loaded,
                role="counterfactual_probe",
                skill_context=context if loaded else {"enabled": True, "injected_refs": []},
            )
            initial = json.loads((episode_dir / "initial-state.json").read_text())
            cwd = "/tmp/alfworld-skill-episodes/credit_" + group
            home = "/tmp/codex-home-" + hashlib.sha256(name.encode()).hexdigest()[:32]
            archives = []
            for filename, source in (("workspace.tar", cwd), ("home.tar", home)):
                archive = episode_dir / filename
                archive.write_bytes(b"fixture controlled workspace archive")
                archives.append(
                    {
                        "source": source,
                        "path": str(archive),
                        "removed": True,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                )
            task_result = json.loads((episode_dir / "result.json").read_text())
            task_result["paired_workdir"] = {
                "schema_version": 1,
                "kind": "paired_empty_workdir",
                "credit_probe_group": group,
                "variant": name,
                "empty_before_run": True,
                "cleanup_completed": True,
                "workdir": cwd,
                "codex_homes": [home],
                "archives": archives,
            }
            write(episode_dir / "result.json", task_result)
            outcomes[name] = {
                "task_id": "train_b",
                "model_version": "f" * 64,
                "environment_version": "fixture-env",
                "environment_state_hash": sha(initial),
                "baseline_snapshot_id": "base",
                "score": float(loaded),
                "trajectory_ref": str(episode_dir),
                "loaded_skill_ref": ref if loaded else None,
            }
            write(
                probe / name / "native-capture.json",
                {
                    "reward": float(loaded),
                    "accuracy": float(loaded),
                    "policy_attestation": attestation,
                    "trajectories": [{"prompt_ids": [1], "response_ids": [2], "response_mask": [1]}],
                    "extra_info": {"episode_output_dir": str(episode_dir)},
                },
                wrapped=True,
            )
        event = {
            "event_id": "event_1",
            "window_id": "window_1",
            "curate_record_id": "record_1",
            "skill_ref": ref,
            "probe_id": "probe_1",
            "occurred_at": 102,
            "with_skill": outcomes["with_skill"],
            "without_skill": outcomes["without_skill"],
        }
        write(probe / "paired-event.json", event, wrapped=True)
        write(self.root / "credit_ledger/events/event_1.json", event, wrapped=True)
        window = {"curate_record_id": "record_1", "replay_payload_hash": sha(record)}
        write(self.root / "credit_ledger/windows/window_1.json", window, wrapped=True)
        settlement = {
            "curate_record_id": "record_1",
            "window_id": "window_1",
            "skill_ref": ref,
            "event_ids": ["event_1"],
            "contributions": [1.0],
            "utility": 1.0,
            "aggregation": "mean",
            "cost_weight": 0.0,
            "skill_cost": 0.0,
        }
        write(self.root / "credit_ledger/settlements/credit_1.json", settlement, wrapped=True)
        update = "optimizer:2:update_2"
        write(
            self.root / "credit_ledger/state.json",
            {
                "settlements": {
                    "credit_1": {
                        "payload_hash": sha(settlement),
                        "window_id": "window_1",
                        "applied_updates": {"update_2": update},
                    }
                }
            },
            wrapped=True,
        )
        write(self.root / "controller/state.json", {"applied_update_ids": {"update_2": update}}, wrapped=True)
        write(self.root / "controller/updates/update_2.json", {"global_step": 2, "optimizer_step": 2}, wrapped=True)


class RunAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_missing_or_manifest_only_run_is_explicitly_incomplete(self):
        empty = audit.audit_run(self.root)
        self.assertEqual(empty["review_status"], "incomplete_evidence")
        self.assertIsNone(empty["acceptance_decision"])
        fixture = Fixture(self.root)
        for stage in ("initial", "final"):
            write(self.root / "evaluation" / stage / "summary.json", {"completed": True, "accuracy": 1})
        report = audit.audit_run(self.root)
        self.assertEqual(report["training_activity"]["verified_update_count"], 0)
        self.assertFalse(report["official_evaluation"]["controlled_improvement_supported"])
        self.assertEqual(report["provenance"]["status"], "declared_with_files")
        self.assertEqual(fixture.ids["train"], ["train_a", "train_b"])

    def test_diagnostic_gradients_and_null_or_zero_train_gradients_are_not_optimizer_proof(self):
        fixture = Fixture(self.root)
        fixture.training(gradient=None)
        fixture.metric_rows.append({"phase": "system", "step": 1, "metrics": {"actor/grad_norm": 99}})
        rows(self.root / "metrics.jsonl", fixture.metric_rows)
        write(self.root / "diagnostics/kernel-pass.json", {"phase": "train", "actor/grad_norm": 4, "passed": True})
        report = audit.audit_run(self.root)
        self.assertEqual(report["training_activity"]["verified_update_count"], 0)
        fixture.metric_rows[0]["metrics"]["actor/grad_norm"] = 0
        rows(self.root / "metrics.jsonl", fixture.metric_rows)
        self.assertEqual(audit.audit_run(self.root)["training_activity"]["verified_update_count"], 0)

    def test_training_needs_real_numeric_tokens_and_does_not_rewrite_metrics(self):
        fixture = Fixture(self.root)
        fixture.training()
        original = (self.root / "metrics.jsonl").read_bytes()
        report = audit.audit_run(self.root)
        self.assertEqual(report["training_activity"]["verified_update_count"], 1)
        self.assertTrue(report["training_activity"]["reward_trend_is_not_controlled_improvement"])
        self.assertEqual((self.root / "metrics.jsonl").read_bytes(), original)
        (self.root / "agent_logs/step_1/session_1/trajectory.npz").unlink()
        self.assertEqual(audit.audit_run(self.root)["training_activity"]["verified_update_count"], 0)
        with self.assertRaises(FileExistsError):
            audit.main([str(self.root), "--output", str(self.root / "metrics.jsonl")])
        self.assertEqual((self.root / "metrics.jsonl").read_bytes(), original)

    def test_complete_140_and_134_same_protocol_raw_outcomes_support_descriptive_improvement(self):
        fixture = Fixture(self.root)
        fixture.evaluations()
        fixture.lineage()
        report = audit.audit_run(self.root)
        comparison = report["official_evaluation"]
        self.assertTrue(comparison["controlled_improvement_supported"])
        self.assertEqual(comparison["overall_win_delta"], 2)
        self.assertEqual(comparison["stages"]["initial"]["splits"]["test_seen"]["verified_games"], 140)
        self.assertEqual(comparison["stages"]["final"]["splits"]["test_unseen"]["verified_games"], 134)
        self.assertEqual(report["delayed_skill_lineage"]["complete_lineages"], 1)
        self.assertEqual(report["review_status"], "review_required")
        self.assertIsNone(report["acceptance_decision"])

    def test_prefix_duplicates_cannot_replace_a_missing_game_and_protocol_drift_blocks_claim(self):
        fixture = Fixture(self.root)
        fixture.evaluations()
        path = self.root / "evaluation/initial/test_seen/0.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        rows(path, original + [copy.deepcopy(original[0])])
        self.assertTrue(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])
        rows(path, original[:-1] + [copy.deepcopy(original[0])])
        report = audit.audit_run(self.root)
        self.assertFalse(report["official_evaluation"]["controlled_improvement_supported"])
        self.assertEqual(
            report["official_evaluation"]["stages"]["initial"]["splits"]["test_seen"]["verified_games"], 139
        )
        rows(path, original)
        protocol_path = self.root / "evaluation/final/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["budgets"]["environment_max_steps"] = 60
        write(protocol_path, protocol)
        changed = audit.audit_run(self.root)["official_evaluation"]
        self.assertEqual(changed["overall_win_delta"], 2)
        self.assertFalse(changed["same_recorded_protocol"])
        self.assertFalse(changed["controlled_improvement_supported"])

    def test_ledger_applied_claim_requires_matching_actual_optimizer_clock(self):
        fixture = Fixture(self.root)
        fixture.lineage(actual_update=False)
        report = audit.audit_run(self.root)
        self.assertEqual(report["delayed_skill_lineage"]["complete_lineages"], 0)
        self.assertIn("matching real optimizer", report["delayed_skill_lineage"]["lineages"][0]["reason"])

    def test_engine_win_with_native_failure_is_not_silently_counted_as_a_clean_win(self):
        fixture = Fixture(self.root)
        fixture.training()
        path = self.root / "episodes/train_b/attempt_train1/result.json"
        result = json.loads(path.read_text())
        result.update(agent_ok=False, agent_error="missing terminal event")
        write(path, result)
        report = audit.audit_run(self.root)
        self.assertEqual(report["training_activity"]["verified_update_count"], 0)
        raw = audit.episode(audit.Reader(), path.parent)
        self.assertTrue(raw["engine_win"])
        self.assertFalse(raw["valid"])

    def test_duplicate_retry_cursor_and_optimizer_clock_are_ambiguous_not_two_updates(self):
        fixture = Fixture(self.root)
        fixture.training()
        fixture.metric_rows.append(copy.deepcopy(fixture.metric_rows[0]))
        rows(self.root / "metrics.jsonl", fixture.metric_rows)
        report = audit.audit_run(self.root)["training_activity"]
        self.assertEqual(report["verified_update_count"], 0)
        self.assertTrue(any("Ambiguous duplicate" in reason for reason in report["reasons"]))

    def test_identically_empty_protocols_do_not_establish_a_controlled_comparison(self):
        fixture = Fixture(self.root)
        fixture.evaluations()
        for stage in ("initial", "final"):
            path = self.root / "evaluation" / stage / "protocol.json"
            protocol = json.loads(path.read_text())
            protocol.update(dataset={}, budgets={})
            write(path, protocol)
        result = audit.audit_run(self.root)["official_evaluation"]
        self.assertTrue(result["same_recorded_protocol"])
        self.assertFalse(result["controlled_improvement_supported"])
        self.assertEqual(result["overall_win_delta"], 2)
        self.assertTrue(result["stages"]["initial"]["reasons"])
        self.assertFalse(result["stages"]["initial"]["optimizer_version_inferred_from_cursor"])

    def test_archive_and_context_claims_without_actual_prompt_injection_are_incomplete(self):
        fixture = Fixture(self.root)
        fixture.lineage()
        prompt = self.root / "episodes/train_b/attempt_train2/agent-prompt.txt"
        prompt.write_text("Original task with no retained skill text")
        result = audit.audit_run(self.root)["delayed_skill_lineage"]
        self.assertEqual(result["complete_lineages"], 0)
        self.assertIn("actually loaded", result["lineages"][0]["reason"])

    def test_live_partial_jsonl_is_reported_without_altering_it(self):
        Fixture(self.root)
        path = self.root / "metrics.jsonl"
        partial = b'{"phase":"train","step":1'
        path.write_bytes(partial)
        report = audit.audit_run(self.root)
        self.assertEqual(report["training_activity"]["verified_update_count"], 0)
        self.assertTrue(any("still-being-written" in value for value in report["training_activity"]["reasons"]))
        self.assertEqual(path.read_bytes(), partial)

    def test_empty_eos_completion_with_numeric_tokens_remains_a_valid_benchmark_failure(self):
        fixture = Fixture(self.root)
        fixture.evaluations()
        for stage in ("initial", "final"):
            path = self.root / "evaluation" / stage / "test_seen/0.jsonl"
            dump = [json.loads(line) for line in path.read_text().splitlines()]
            row = dump[-1]
            row["output"] = ""
            rows(path, dump)
            directory = Path(row["episode_output_dir"])
            for name in ("result.json", "engine-result.json"):
                result = json.loads((directory / name).read_text())
                result["steps"] = 0
                write(directory / name, result)
            rows(directory / "environment-trace.jsonl", [])
            rows(directory / "codex-events.jsonl", [{"type": "turn.completed"}])
        report = audit.audit_run(self.root)["official_evaluation"]
        self.assertTrue(report["controlled_improvement_supported"])
        self.assertEqual(report["stages"]["initial"]["splits"]["test_seen"]["verified_games"], 140)

    def test_zero_centered_curate_scalar_does_not_prove_that_its_gradient_was_zero(self):
        fixture = Fixture(self.root)
        fixture.lineage()
        fixture.metric_rows[0]["metrics"]["skill/curate_loss"] = 0.0
        rows(self.root / "metrics.jsonl", fixture.metric_rows)
        report = audit.audit_run(self.root)["delayed_skill_lineage"]
        self.assertEqual(report["complete_lineages"], 1)
        self.assertIn("not separately measured", report["lineages"][0]["curate_branch_gradient"])

    def test_pointerless_production_dump_joins_full_official_cases_without_rewriting_it(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations(prefixes=True)
        path = self.root / "evaluation/initial/test_seen/1.jsonl"
        original = path.read_bytes()
        self.assertNotIn("episode_output_dir", json.loads(original.splitlines()[0]))
        # An unrelated dump/retry must not be selected by globbing or modification time.
        rows(path.parent / "999.jsonl", [{"uid": "dev_only_0_0", "score": 1}])
        result = audit.audit_run(self.root)["official_evaluation"]
        self.assertEqual(result["status"], "observed")
        self.assertTrue(result["controlled_improvement_supported"])
        self.assertEqual(result["overall_win_delta"], 2)
        for stage in ("initial", "final"):
            for phase, count in audit.OFFICIAL_COUNTS.items():
                split = result["stages"][stage]["splits"][phase]
                self.assertEqual(split["verified_games"], count)
                self.assertEqual(split["case_audit_path"], str(fixture.case_reports[stage, phase].resolve()))
        self.assertEqual(path.read_bytes(), original)

    def test_sidecar_complete_cannot_hide_missing_uid_raw_task_or_numeric_trajectory(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations()
        path = self.root / "evaluation/initial/test_seen/1.jsonl"
        dump = [json.loads(line) for line in path.read_text().splitlines()]
        # Parent reward can still average 139 observed rows; a claimed denominator140
        # cannot turn the unobserved last task into a zero or a complete evaluation.
        rows(path, dump[:-1])
        report = audit.audit_run(self.root)["official_evaluation"]
        split = report["stages"]["initial"]["splits"]["test_seen"]
        self.assertEqual(split["verified_games"], 139)
        self.assertFalse(report["controlled_improvement_supported"])
        rows(path, dump)
        case_path = fixture.case_reports["initial", "test_seen"]
        case = json.loads(case_path.read_text())["cases"][-1]
        raw_path = Path(case["episode_output_dir"]) / "engine-result.json"
        original = raw_path.read_bytes()
        raw_path.unlink()
        self.assertEqual(
            audit.audit_run(self.root)["official_evaluation"]["stages"]["initial"]["splits"]["test_seen"][
                "verified_games"
            ],
            139,
        )
        raw_path.write_bytes(original)
        (self.root / "agent_logs/step_1/initial-test_seen-139/trajectory.npz").unlink()
        self.assertFalse(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])

    def test_sidecar_scope_raw_identity_and_score_must_each_match(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations()
        path = fixture.case_reports["initial", "test_seen"]
        original = json.loads(path.read_text())
        for changes in ({"stage": "final"}, {"phase": "dev"}, {"global_cursor": 0}):
            write(path, {**original, **changes})
            self.assertFalse(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])
        write(path, original)
        case = original["cases"][0]
        result_path = Path(case["episode_output_dir"]) / "result.json"
        task = json.loads(result_path.read_text())
        write(result_path, {**task, "evaluation_audit_id": "f" * 32})
        self.assertFalse(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])
        write(result_path, task)
        dump_path = Path(original["dump_path"])
        dump = [json.loads(line) for line in dump_path.read_text().splitlines()]
        dump[0]["score"] = 0.5
        rows(dump_path, dump)
        self.assertFalse(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])

    def test_duplicate_sidecar_games_or_dump_uid_cannot_substitute_for_an_official_case(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations()
        path = fixture.case_reports["initial", "test_seen"]
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed["cases"][-1] = copy.deepcopy(changed["cases"][0])
        write(path, changed)
        report = audit.audit_run(self.root)["official_evaluation"]
        self.assertLess(report["stages"]["initial"]["splits"]["test_seen"]["verified_games"], 140)
        self.assertFalse(report["controlled_improvement_supported"])
        write(path, original)
        dump_path = Path(original["dump_path"])
        dump = [json.loads(line) for line in dump_path.read_text().splitlines()]
        rows(dump_path, dump + [copy.deepcopy(dump[0])])
        self.assertFalse(audit.audit_run(self.root)["official_evaluation"]["controlled_improvement_supported"])

    def test_phase_sidecar_without_completed_stage_is_useful_but_never_complete(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations()
        (self.root / "evaluation/initial/summary.json").unlink()
        for phase in audit.OFFICIAL_COUNTS:
            write(
                self.root / "evaluation/initial" / phase / "metrics.json",
                {
                    "stage": "initial",
                    "phase": phase,
                    "global_step": 1,
                    "case_audit_path": str(fixture.case_reports["initial", phase]),
                },
            )
        result = audit.audit_run(self.root)["official_evaluation"]
        self.assertEqual(result["stages"]["initial"]["splits"]["test_seen"]["verified_games"], 140)
        self.assertEqual(result["stages"]["initial"]["status"], "incomplete")
        self.assertFalse(result["controlled_improvement_supported"])

    def test_sidecar_cannot_convert_dev_only_dump_into_official_evaluation(self):
        Fixture(self.root)
        rows(self.root / "validation/0.jsonl", [{"uid": "dev_0_0", "score": 1, "acc": 1}])
        report = audit.audit_run(self.root)["official_evaluation"]
        self.assertEqual(report["status"], "incomplete")
        for stage in report["stages"].values():
            self.assertEqual(stage["splits"]["test_seen"]["verified_games"], 0)

    def test_native_bookkeeping_error_does_not_drop_observed_engine_win_from_denominator(self):
        fixture = Fixture(self.root)
        fixture.sidecar_evaluations()
        path = fixture.case_reports["initial", "test_seen"]
        sidecar = json.loads(path.read_text())
        case = sidecar["cases"][0]
        directory = Path(case["episode_output_dir"])
        result = json.loads((directory / "result.json").read_text())
        result.update(agent_ok=False, agent_error="Unexplained missing native completion")
        write(directory / "result.json", result)
        rows(directory / "codex-events.jsonl", [])
        case["native_error"] = result["agent_error"]
        sidecar["native_error_cases"] = 1
        write(path, sidecar)
        report = audit.audit_run(self.root)["official_evaluation"]
        split = report["stages"]["initial"]["splits"]["test_seen"]
        self.assertEqual(split["verified_games"], 140)
        self.assertEqual(split["engine_wins"], 70)
        self.assertEqual(split["reward_mean"], 70 / 140)
        self.assertTrue(split["native_errors"])
        self.assertEqual(split["status"], "incomplete")
        self.assertFalse(report["controlled_improvement_supported"])

    def test_expected_terminal_stop_preserves_raw_native_error_and_needs_matching_private_proof(self):
        fixture = Fixture(self.root)
        directory, result = fixture.episode("train_a", "train", "terminal", True)
        state = {"done": True, "won": True, "steps": 1, "max_steps": 50}
        binding = {
            "binding_id": "binding_fixture",
            "binding_sha256": "a" * 64,
            "attempt_id": "terminal",
            "gateway_session_id": "gateway_fixture",
            "environment_session_id": "environment_fixture",
        }
        receipt = {
            **binding,
            "schema_version": 1,
            "status": "terminal_aborted",
            "fenced": True,
            "terminal_state": state,
            "terminal_state_hash": sha(state),
            "active_requests": 0,
            "backend_stop_confirmed": True,
            "expected_backend_terminal_stop": True,
            "reason": "terminal_abort",
        }
        result.update(
            agent_ok=False,
            agent_error="Native CLI terminated after proven terminal state",
            agent_exit_code=-15,
            expected_host_terminal_stop=True,
            expected_backend_terminal_stop=True,
            terminal_stop={
                "status": "terminal_stop",
                "termination_requested": True,
                "termination_completed": True,
                "alive_after_termination": False,
                "terminal_state": state,
                "terminal_state_hash": sha(state),
            },
            backend_binding=binding,
            backend_terminal=receipt,
            backend_stop_confirmed=True,
            environment_session_id=binding["environment_session_id"],
        )
        write(directory / "result.json", result)
        rows(directory / "codex-events.jsonl", [{"type": "error", "message": "Actual native failure retained"}])
        observed = audit.episode(audit.Reader(), directory)
        self.assertTrue(observed["valid"])
        self.assertTrue(observed["expected_terminal_stop_validated"])
        self.assertEqual(observed["result"]["agent_exit_code"], -15)
        self.assertEqual(observed["native_bookkeeping_error"], result["agent_error"])
        self.assertFalse(observed["native_completion_event_observed"])
        # Flags never substitute for a drained backend, exact identities, or
        # agreement with the private environment outcome.
        for changes in (
            {"active_requests": 1},
            {"attempt_id": "some_other_attempt"},
            {"terminal_state_hash": "f" * 64},
            {"terminal_state": {**state, "won": False}},
        ):
            write(directory / "result.json", {**result, "backend_terminal": {**receipt, **changes}})
            failed = audit.episode(audit.Reader(), directory)
            self.assertFalse(failed["valid"])
            self.assertTrue(failed["engine_win"])


if __name__ == "__main__":
    unittest.main()
