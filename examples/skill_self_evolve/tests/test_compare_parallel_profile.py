"""Synthetic complete official stages for the parallel source/profile contract.

No model, service, UID, wrapper or shared library is executed. Real temporary
Git objects, generated public files, COMPLETE metadata and all274 case joins
exercise the offline comparer. The binary helper is deliberately inert bytes.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import test_compare_official_runs as fixture

ROOT = Path(__file__).resolve().parents[3]
read, write, sha = fixture.read, fixture.write, fixture.file_sha


class ParallelProfileTests(unittest.TestCase):
    compare_pair = fixture.CompareOfficialRunsTests.compare_pair
    change = fixture.CompareOfficialRunsTests.change
    assert_incomplete = fixture.CompareOfficialRunsTests.assert_incomplete
    mutated_initialized = fixture.SafeSetupEffectiveProfileTests.mutated_initialized

    @classmethod
    def setUpClass(cls):
        fixture.SafeSetupEffectiveProfileTests.setUpClass.__func__(cls)
        compare, source = cls.compare, Path(cls.extension["driver_path"]).parents[3]
        files = [
            compare.SPAWN_DRIVER,
            *[p for k, p in compare.SPAWN_FILES.items() if k != "helper"],
            compare.PARALLEL_DRIVER,
            compare.PARALLEL_CLIENT,
            compare.PARALLEL_SERVER,
            "examples/skill_self_evolve/scripts/codex_entrypoint.sh",
        ]
        for relative in files:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
        subprocess.check_call(["git", "-C", str(source), "add", *files])
        if subprocess.run(["git", "-C", str(source), "diff", "--cached", "--quiet"]).returncode:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "commit",
                    "-q",
                    "-m",
                    "Parallel metadata fixture",
                ],
                check=True,
                capture_output=True,
            )
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        helper_path = source / compare.SPAWN_FILES["helper"]
        helper_path.write_bytes(b"Inert CPU metadata fixture; never loaded or executed")
        resilient = {**cls.extension, "driver_source_commit": commit}
        native = {
            "policy": "native-posix-spawn-v1",
            **{
                key: {"path": str(source / relative), "sha256": sha(source / relative)}
                for key, relative in compare.SPAWN_FILES.items()
            },
        }
        cls.spawn = {
            "policy": compare.SPAWN_POLICY,
            "driver_path": str(source / compare.SPAWN_DRIVER),
            "driver_sha256": sha(source / compare.SPAWN_DRIVER),
            "namespace_path": cls.extension["namespace_path"],
            "namespace_sha256": cls.extension["namespace_sha256"],
            "driver_source_commit": commit,
            "composed_extensions": {
                "resilient_setup": resilient,
                "quiescent_final": {
                    "path": str(source / compare.LIFECYCLE_DRIVER),
                    "sha256": sha(source / compare.LIFECYCLE_DRIVER),
                },
                "native_spawn": native,
            },
        }
        cls.loaded_spawn = copy.deepcopy(cls.spawn)
        cls.loaded_spawn["composed_extensions"]["native_spawn"]["helper"]["loaded_sha256"] = native["helper"]["sha256"]
        adapter_module = fixture.module("_parallel_fixture_producer", source / compare.PARALLEL_DRIVER)
        cls.observations, cls.adapters = {}, {}
        for role in ("base", "trained"):
            run = cls.pair.root / role
            manifest = read(run / "training-manifest.json")
            config = manifest["config"]
            config.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {}).setdefault(
                "env_vars", {}
            )
            config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
            config["actor_rollout_ref"]["ref"] = {"log_prob_micro_batch_size_per_gpu": 1}
            config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
            launch = read(run / "launch_manifest.json")
            with mock.patch.dict(os.environ, {"VERL_SOURCE_ROOT": str(source)}):
                adapter = adapter_module.prepare_adapter(
                    run,
                    run / "task_config.yaml",
                    launch["environment_paths"]["ALFWORLD_SOLVER_INSTRUCTIONS_FILE"],
                    public_root=Path(cls.temporary.name) / "public",
                    environment_python=sys.executable,
                    server_script=source / compare.PARALLEL_SERVER,
                )
            cls.adapters[role] = adapter
            extension = {
                "source_path": str(source / compare.PARALLEL_SERVER),
                "source_sha256": sha(source / compare.PARALLEL_SERVER),
                "policy": compare.ENVIRONMENT_POLICY,
            }
            health = {
                "ready": True,
                "catalog_digest": read(run / "data/split_manifest.json")["catalog_digest"],
                "max_steps": 50,
                "budget_seconds": 900,
                "environment_processes": 4,
                "io_concurrency": 4,
                "environment_policy": compare.ENVIRONMENT_POLICY,
                "action_request_protocol": compare.ACTION_PROTOCOL,
                "server_instance_id": ("1" if role == "base" else "2") * 32,
                "environment_worker_pids": [1001, 1002, 1003, 1004],
                "environment_extension": extension,
            }
            settings = {
                **health,
                "allowed_splits": ["train"],
                "url": "http://127.0.0.1:12345",
                "pid": 1000,
                "io_workers": 4,
                "admin_token": "private-fixture-token",
            }
            settings_path = run / "runtime/environment.json"
            write(settings_path, settings)
            official = {
                **settings,
                "allowed_splits": list(compare.SPLITS.values()),
                "server_instance_id": fixture.sha([run.name, "server"])[:32],
            }
            write(
                (cls.pair.base if role == "base" else cls.pair.trained) / ("service-" + "a" * 32) / "environment.json",
                official,
            )
            actual = copy.deepcopy(config)
            runner = actual["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]["agent_runners"]["task"]
            runner.update(runner_fqn=compare.SPAWN_RUNNER, max_concurrent_sessions=32)
            runner["runner_kwargs"]["task_config_path"] = adapter["task_profile"]["path"]
            actual["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"].update(adapter["environment_variable_changes"])
            if role == "base":
                manifest["source_commit"], manifest["config"] = commit, actual
            write(run / "training-manifest.json", manifest)
            stage = cls.pair.base if role == "base" else cls.pair.trained
            protocol = read(stage / "protocol.json")
            protocol["budgets"]["alfworld_task_config_sha256"] = adapter["task_profile"]["sha256"]
            write(stage / "protocol.json", protocol)
            common = {
                "schema_version": 1,
                "observed_at_ns": 99,
                "client_adapter": adapter,
                "environment_health": health,
                "environment_config_ref": {"path": str(settings_path), "sha256": sha(settings_path)},
                "spawn_extension": cls.spawn,
                "not_a_post_init_or_served_model_receipt": True,
            }
            if role == "base":
                value = {
                    **common,
                    "kind": "prospective_parallel_environment_runtime_identity",
                    "entrypoint": "verl.experimental.skill_harness.train",
                    "source_commit": commit,
                    "client_source_ref": {
                        "path": str(source / compare.PARALLEL_DRIVER),
                        "sha256": sha(source / compare.PARALLEL_DRIVER),
                    },
                    "original_entry_command": [sys.executable, "-m", "verl.experimental.skill_harness.train"],
                    "effective_config_before_entry": actual,
                }
                path = run / compare.PARALLEL_EXPECTED_RECEIPT
            else:
                value = {
                    **common,
                    "kind": "prelaunch_parallel_environment_client_observation",
                    "driver_path": str(source / compare.PARALLEL_DRIVER),
                    "driver_sha256": sha(source / compare.PARALLEL_DRIVER),
                    "effective_config_before_spawn_entry": actual,
                    "core_fingerprint_scope": "Explicitly supplemented core identity",
                    "hash_link_scope": "Immutable Task profile path and bytes",
                }
                path = run / "driver_runs/99.parallel_client.json"
            write(path, value)
            cls.observations[role] = path
        trained = cls.pair.root / "trained"
        initialized = read(cls.initialized_path)
        effective = initialized["effective_config_after_trainer_init"]
        effective.setdefault("ray_kwargs", {}).setdefault("ray_init", {}).setdefault("runtime_env", {}).setdefault(
            "env_vars", {}
        )
        manifest = read(trained / "training-manifest.json")
        for section, key in (
            ("actor", "ppo_micro_batch_size_per_gpu"),
            ("rollout", "log_prob_micro_batch_size_per_gpu"),
        ):
            effective["actor_rollout_ref"][section][key] = 1
        effective["actor_rollout_ref"]["ref"] = {"log_prob_micro_batch_size_per_gpu": 1}
        runner = effective["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]["agent_runners"]["task"]
        runner.update(runner_fqn=compare.SPAWN_RUNNER, max_concurrent_sessions=32)
        runner["runner_kwargs"]["task_config_path"] = cls.adapters["trained"]["task_profile"]["path"]
        effective["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"].update(
            cls.adapters["trained"]["environment_variable_changes"]
        )
        initialized.update(cls.loaded_spawn)
        initialized.update(
            counterfactual_loop_class=compare.SPAWN_MODULE + ".NativeSpawnSelectorLoop",
            event_loop_policy_class=compare.SPAWN_MODULE + ".NativeSpawnPolicy",
            primary_runner_policy_install="synchronous runner constructor before SDK asyncio.run",
            original_training_manifest_sha256=sha(trained / "training-manifest.json"),
        )
        old = cls.initialized_path
        cls.initialized_path = old.with_name("100.spawn_safe.initialized.json")
        write(cls.initialized_path, initialized)
        old.unlink()
        receipt = read(cls.receipt_path)
        receipt.update(cls.loaded_spawn)
        receipt["original_training_manifest_sha256"] = sha(trained / "training-manifest.json")
        for key in ("checkpoint_seal", "evaluation_entry", "evaluation_complete"):
            path = Path(receipt[key]["path"])
            value = read(path)
            value.update(cls.loaded_spawn)
            value.update(
                original_training_manifest_sha256=sha(trained / "training-manifest.json"),
                lifecycle_extension=cls.loaded_spawn,
                initialized_receipt_ref={"path": str(cls.initialized_path), "sha256": sha(cls.initialized_path)},
            )
            write(path, value)
            receipt[key]["sha256"] = sha(path)
        write(cls.receipt_path, receipt)

    def test_parallel_actual_profile_matches_prospective_original_entry_without_loading_helper(self):
        report = self.compare_pair()
        self.assertTrue(report["comparison_supported"], report["reasons"])
        self.assertIsNotNone(report["parallel_environment_client_evidence"]["base"])
        self.assertNotIn("private-fixture-token", str(report))

    def test_missing_parallel_observation_cannot_fall_back_to_legacy_manifest(self):
        path = self.observations["base"]
        data = path.read_bytes()
        self.addCleanup(path.write_bytes, data)
        path.unlink()
        self.assert_incomplete(self.compare_pair())

    def test_wrong_process_count_or_capability_is_rejected(self):
        self.change(
            self.observations["base"], lambda value: value["environment_health"].update(environment_processes=1)
        )
        self.assert_incomplete(self.compare_pair())

    def test_nonmatching_generated_wrapper_is_rejected_even_with_updated_reference_hash(self):
        ref = self.adapters["base"]["wrapper"]
        path = Path(ref["path"])
        data = path.read_bytes()
        self.addCleanup(path.write_bytes, data)
        path.write_bytes(data + b"# foreign wrapper\n")
        self.change(
            self.observations["base"], lambda value: value["client_adapter"]["wrapper"].update(sha256=sha(path))
        )
        self.assert_incomplete(self.compare_pair())

    def test_loaded_helper_and_microbatch_drift_cannot_be_normalized(self):
        changes = [
            lambda value: value["composed_extensions"]["native_spawn"]["helper"].update(loaded_sha256="f" * 64),
            lambda value: value["effective_config_after_trainer_init"]["actor_rollout_ref"]["actor"].update(
                ppo_micro_batch_size_per_gpu=2
            ),
        ]
        for callback in changes:
            with self.subTest(callback=callback), self.mutated_initialized(callback):
                self.assert_incomplete(self.compare_pair())

    def test_same_legacy_core_fingerprint_cannot_hide_changed_environment_source(self):
        path = self.observations["trained"]
        self.change(
            path,
            lambda value: value["client_adapter"]["environment_python_route"]["parallel_server"].update(
                sha256="f" * 64
            ),
        )
        self.assert_incomplete(self.compare_pair())

    def test_actual_official_service_config_is_bound_to_case_budget_instance(self):
        path = self.pair.base / ("service-" + "a" * 32) / "environment.json"
        self.change(path, lambda value: value.update(environment_processes=1))
        self.assert_incomplete(self.compare_pair())


if __name__ == "__main__":
    unittest.main()
