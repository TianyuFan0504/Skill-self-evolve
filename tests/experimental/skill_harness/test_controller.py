"""CPU end-to-end controller contracts with native/model boundaries replaced."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

MODULE_ROOT = Path(
    os.environ.get(
        "SKILL_HARNESS_MODULE_ROOT",
        Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness",
    )
)
PACKAGE = "_controller_cpu_contract"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MODULE_ROOT)]
sys.modules[PACKAGE] = package
controller_module = importlib.import_module(PACKAGE + ".controller")
curate = importlib.import_module(PACKAGE + ".curate")
probes = importlib.import_module(PACKAGE + ".probes")
data_module = importlib.import_module(PACKAGE + ".alfworld.data")
batcher = importlib.import_module(PACKAGE + ".replay_batch")


class FakeActor:
    world_size = 4

    def __init__(self):
        self.clock, self.anchor_calls, self.update_calls = 0, [], []
        self.break_anchor_fence = False

    def get_ema_metadata(self):
        return [{"last_optimizer_step": self.clock} for _ in range(self.world_size)]

    def compute_log_prob(self, data):
        self.anchor_calls.append(data.clone())
        if self.break_anchor_fence:
            self.clock += 1
        return TensorDict(
            {
                "log_probs": batcher._nested(
                    [torch.full_like(ids, -0.25, dtype=torch.float32) for ids in data["input_ids"].unbind()]
                )
            },
            batch_size=[len(data)],
        )

    def update_actor(self, data):
        self.update_calls.append(data.clone())
        self.clock += 1


class FakeRoute:
    def __init__(self, config, spec, actor):
        self.config, self.spec, self.actor = copy.deepcopy(config), spec, actor
        self.optimizer_step, self.syncs = None, []
        self.frozen_version = spec.model_version if spec.kind == "frozen" else None
        self.attestation = {
            "schema_version": 1,
            "policy_kind": "frozen_evaluator" if spec.kind == "frozen" else "ema_curator",
            "frozen_model_version": self.frozen_version,
            "model_fingerprint": self.frozen_version,
            "model_path": spec.model_path,
            "lifecycle_id": "private-native-lifecycle",
        }

    async def sync_ema(self, step):
        if step != self.actor.clock:
            raise AssertionError("EMA must use the actual observed optimizer clock")
        self.optimizer_step = step
        self.syncs.append(step)

    def get_client(self):
        return SimpleNamespace(
            optimizer_step=self.optimizer_step,
            attestation=copy.deepcopy(self.attestation),
            frozen_version=self.frozen_version,
            spec=self.spec,
        )


class FakeGateway:
    def __init__(self, client):
        self.client, self.created, self.shutdowns = client, [], 0

    async def create_session(self, identity, **kwargs):
        self.created.append((identity, kwargs))
        return SimpleNamespace(session_id=identity, base_url="http://127.0.0.1:12345/sessions/" + identity + "/v1")

    async def finalize_session(self, identity):
        return [
            SimpleNamespace(
                prompt_ids=[11, 12],
                response_ids=[31, 32, 33],
                response_mask=[1, 0, 1],
                response_logprobs=[-0.5, 0.0, -0.7],
                chain_id=1,
                extra_fields={
                    "min_global_steps": self.client.optimizer_step,
                    "max_global_steps": self.client.optimizer_step,
                },
            )
        ]

    async def abort_session(self, identity):
        return None

    async def shutdown(self):
        self.shutdowns += 1


class FakeSandbox:
    def __init__(self, runtime, output, identity):
        self.workdir, self.tool_events_seen = Path(runtime.work_root) / identity, False

    async def close(self):
        pass


class CallbackTokenizer:
    pad_token_id = 0
    is_fast = False

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = "".join("<" + row["role"] + ">" + row["content"] + "</turn>" for row in messages)
        rendered += "<assistant>" if add_generation_prompt else ""
        return self.encode(rendered) if tokenize else rendered


class ControllerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        for name, content in (("config.json", "{}"), ("tokenizer.json", "{}"), ("model.safetensors", "weights")):
            (self.model / name).write_text(content)
        runtime_files = {}
        for key in ("codex_bin", "model_catalog_path", "instructions_path", "schema_path", "task_prompt_path"):
            path = self.root / key
            path.write_text("{{SKILL_ID}} {{TASK_FAMILY}} {{EPISODE_JSON}}" if key == "task_prompt_path" else key)
            runtime_files[key] = str(path)
        self.runtime = curate.NativeCuratorRuntime(
            "Qwen3.5-4B-nothink",
            "alfworld_curator",
            1201,
            1201,
            str(self.root / "work"),
            package_root=str(self.root),
            native_version="0.167.0",
            template_version="template:curator1",
            **runtime_files,
        )
        self.manifest = self.root / "split_manifest.json"
        self.manifest.write_text(
            json.dumps({"catalog_digest": "b" * 64, "parquet_files": {"train": {"sha256": "c" * 64}}})
        )
        environment = self.root / "environment.json"
        environment.write_text(json.dumps({"catalog_digest": "b" * 64, "allowed_splits": ["train"], "max_steps": 50}))
        task_config = self.root / "task_config.json"
        task_config.write_text("[]")
        self.environment = patch.dict(
            os.environ,
            {
                "ALFWORLD_SERVER_CONFIG": str(environment),
                "ALFWORLD_TASK_CONFIG": str(task_config),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = OmegaConf.create(
            {
                "skill_harness": {
                    "mode": "skill_joint",
                    "artifact_root": str(self.root),
                    "store_root": str(self.root / "skill_store"),
                    "split_manifest": str(self.manifest),
                    "loading": {"enabled": True, "skill_token_budget": 4096},
                    "joint": {},
                },
                "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
                "actor_rollout_ref": {
                    "model": {"path": str(self.model)},
                    "rollout": {
                        "temperature": 1.0,
                        "prompt_length": 4096,
                        "response_length": 32768,
                        "max_model_len": 36864,
                        "custom": {"agent_framework": {"gateway_count": 2}},
                    },
                },
            }
        )
        self.actor, self.gateways, self.probe_calls, self.curate_calls = FakeActor(), [], [], []
        self.negative_probe = False
        self.curate_bad_content = {}
        self.curate_bad_fields = {}
        self.curate_tool_indices = set()
        self.trainer = SimpleNamespace(config=self.config, actor_rollout_wg=self.actor, tokenizer=CallbackTokenizer())

    async def routes(self, config, specs, **kwargs):
        self.assertEqual(kwargs["solver_gpus"], 4)
        self.assertEqual(kwargs["available_gpus"], 8)
        return {spec.name: FakeRoute(config, spec, self.actor) for spec in specs}

    def gateway(self, *, config, llm_client):
        self.assertEqual(
            config.actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys, []
        )
        result = FakeGateway(llm_client)
        self.gateways.append(result)
        return result

    async def create(self):
        result = await controller_module.SkillJointController.create(
            self.trainer,
            route_factory=self.routes,
            gateway_factory=self.gateway,
            native_runtime=self.runtime,
        )
        result._collect_curate = self.native_curate
        result._collect_probe = self.native_probe
        return result

    def metadata(self, name):
        path, digest = "json_2.1.1/train/" + name + "/trial/game.tw-pddl", "a" * 64
        game = data_module.stable_game_id(path, digest)
        return {
            "game_id": game,
            "stable_game_id": game,
            "split": "train",
            "dataset_phase": "train",
            "relative_game_path": path,
            "game_sha256": digest,
            "catalog_digest": "b" * 64,
            "task_type": "pick_clean_then_place_in_recep",
            "scene_id": "FloorPlan1",
            "run_dir": str(self.root),
        }

    def episode(self, name):
        metadata = self.metadata(name)
        output = self.root / "episodes" / metadata["game_id"] / "attempt_native"
        output.mkdir(parents=True, exist_ok=True)
        result = {
            **metadata,
            "attempt_id": "native_attempt",
            "won": True,
            "reward": 1.0,
            "accuracy": 1.0,
            "steps": 1,
            "max_steps": 50,
        }
        for filename in ("result.json", "engine-result.json"):
            (output / filename).write_text(json.dumps(result))
        (output / "initial-state.json").write_text(json.dumps({"observation": "You carry a mug near a sink."}))
        (output / "environment-trace.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "action": "clean mug 1 with sinkbasin 1",
                    "action_in_exposed_commands": True,
                    "after": {"observation": "You clean the mug.", "won": True, "done": True},
                }
            )
            + "\n"
        )
        return metadata, output

    def solver(self, names=("task-a", "task-a-second")):
        entries = [self.episode(name) for name in names]
        prompts = [torch.tensor([10, 11])] * len(entries)
        responses = [torch.tensor([20, 21])] * len(entries)
        ids = [torch.cat((p, r)) for p, r in zip(prompts, responses, strict=True)]
        return tu.get_tensordict(
            {
                "prompts": batcher._nested(prompts),
                "responses": batcher._nested(responses),
                "input_ids": batcher._nested(ids),
                "position_ids": batcher._nested([torch.arange(4).repeat(4, 1) for _ in entries]),
                "response_mask": batcher._nested([torch.tensor([1, 1])] * len(entries)),
                "loss_mask": batcher._nested([torch.tensor([1, 1])] * len(entries)),
                "old_log_probs": batcher._nested([torch.tensor([-0.4, -0.5])] * len(entries)),
                "advantages": batcher._nested([torch.tensor([1.0, 1.0])] * len(entries)),
                "uid": ["uid-" + name for name in names],
                "session_id": list(range(len(entries))),
                "raw_prompt": [[{"role": "user", "content": "Complete the household goal using the environment."}]]
                * len(entries),
                "tools_kwargs": [
                    {"task": {"name": "alfworld_skill_training", "metadata": meta}} for meta, _ in entries
                ],
                "extra_fields": [
                    {"runner_reward_info": {"reward_context": {"episode_output_dir": str(path)}}} for _, path in entries
                ],
                "multi_modal_inputs": [{}] * len(entries),
            }
        )

    async def native_curate(self, **kwargs):
        request = kwargs["request"]
        self.curate_calls.append(request)

        def agent(runtime, handle):
            async def run(**arguments):
                content = {
                    "skill_id": request.skill_id,
                    "title": "Hold before cleaning",
                    "principle": "Hold the target before cleaning it and confirm the feedback.",
                    "when_to_apply": "An object must be cleaned before placement.",
                    "task_families": [kwargs["source_episode"]["task_type"]],
                }
                content.update(self.curate_bad_fields.get(request.candidate_index, {}))
                text = self.curate_bad_content.get(request.candidate_index, json.dumps(content))
                return SimpleNamespace(output={"content": text}, info={"ok": True}, finished=True)

            return SimpleNamespace(run=run)

        def sandbox(*arguments):
            value = FakeSandbox(*arguments)
            value.tool_events_seen = request.candidate_index in self.curate_tool_indices
            return value

        return await curate.collect_curate_candidate(**kwargs, _agent_factory=agent, _sandbox_factory=sandbox)

    async def native_probe(self, **kwargs):
        self.probe_calls.append(kwargs["request"])
        store = kwargs["store"]

        async def branch(**arguments):
            fields, session = arguments["sample_fields"], arguments["session_id"]
            metadata = fields["tools_kwargs"]["task"]["metadata"]
            loaded = session.endswith("_with_skill")
            reward = float(not loaded if self.negative_probe else loaded)
            output = self.root / "episodes" / session
            output.mkdir(parents=True, exist_ok=True)
            (output / "engine-result.json").write_text(
                json.dumps({"game_sha256": metadata["game_sha256"], "split": "train", "won": bool(reward)})
            )
            (output / "initial-state.json").write_text(json.dumps({"observation": "Identical paired game state"}))
            refs = store.get_snapshot(metadata["skill_snapshot_id"]).skills
            group, variant = metadata["credit_probe_group"], metadata["credit_probe_variant"]
            workdir = "/tmp/alfworld-skill-episodes/credit_" + group
            home = "/tmp/codex-home-" + hashlib.sha256(session.encode()).hexdigest()[:32]
            archives = []
            for name, source in (("workspace.tar", workdir), ("codex-home.tar", home)):
                archive = output / name
                archive.write_bytes(("CPU native-boundary fixture: " + session).encode())
                archives.append(
                    {
                        "source": source,
                        "path": str(archive),
                        "removed": True,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                )
            return {
                "finished": True,
                "reward": reward,
                "accuracy": reward,
                "policy_attestation": arguments["frozen_attestation"],
                "extra_info": {
                    "eval_completed": True,
                    "episode_output_dir": str(output),
                    "role": "counterfactual_probe",
                    "credit_probe_group": group,
                    "credit_probe_variant": variant,
                    "paired_workdir": {
                        "schema_version": 1,
                        "kind": "paired_empty_workdir",
                        "credit_probe_group": group,
                        "variant": variant,
                        "workdir": workdir,
                        "empty_before_run": True,
                        "cleanup_completed": True,
                        "codex_homes": [home],
                        "archives": archives,
                        "host_uid": 0,
                        "agent_uid": 1001,
                        "agent_gid": 1001,
                    },
                    "skill_context": {
                        "snapshot_id": metadata["skill_snapshot_id"],
                        "injected_refs": [ref.to_dict() for ref in refs],
                    },
                },
                "trajectories": [{"extra_fields": {}, "prompt_ids": [10], "response_ids": [20]}],
            }

        return await probes.collect_counterfactual_probe(**kwargs, run_branch=branch)

    async def update(self, controller, data, step):
        prepared = await controller.prepare_update(data, step)
        self.actor.update_actor(prepared.data)
        await controller.after_update_and_sync(data, step, prepared=prepared, optimizer_step=self.actor.clock)
        return prepared

    async def test_first_step_solver_only_then_future_native_credit_enters_same_optimizer(self):
        controller = await self.create()
        first = await self.update(controller, self.solver(), 1)
        self.assertEqual(first.data["role_id"].tolist(), [0, 0, 0, 0])
        self.assertEqual(first.acknowledgments, ())
        self.assertEqual(len(self.probe_calls), 0)
        self.assertEqual(len(self.curate_calls), 2)
        old_labels = list(controller.state["pending"])
        second_data = self.solver(("future-b", "future-c"))
        second = await self.update(controller, second_data, 2)
        self.assertEqual(second.data["role_id"].tolist(), [0, 0, 1, 1])
        self.assertEqual(second.data["curate_advantages"].tolist(), [0, 0, 1, 1])
        self.assertEqual(len(self.actor.anchor_calls), 1)
        self.assertEqual(len(self.actor.update_calls), 2)
        self.assertEqual(len(controller.store.get_snapshot().skills), 2)
        self.assertTrue(all(request.global_step == 2 for request in self.probe_calls))
        self.assertEqual(len({request.probe_id for request in self.probe_calls}), 1)
        for window in old_labels:
            label = controller.state["pending"][window]["settlement_id"]
            self.assertEqual(controller.ledger.training_status(label)["state"], "applied")
        self.assertIsNone(controller.export_state()["prepared_update"])
        self.assertEqual(controller.export_state()["optimizer_step"], 2)

    async def test_negative_delayed_utility_stays_signed_and_does_not_publish_main(self):
        self.negative_probe = True
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        prepared = await controller.prepare_update(self.solver(("future-b", "future-c")), 2)
        self.assertEqual(prepared.data["curate_advantages"].tolist(), [0, 0, -1, -1])
        self.assertEqual(controller.store.get_snapshot().skills, ())
        self.assertTrue(all(row["utility"] == -1 for row in controller.state["pending"].values()))

    async def test_multiple_exposures_cover_whole_cohort_before_leaving_current_batch(self):
        self.config.skill_harness.joint.required_exposures = 2
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        first_windows = list(controller.state["pending"])
        second = await self.update(controller, self.solver(("future-b", "future-c")), 2)
        self.assertFalse((second.data["role_id"] == 1).any())
        for identity in first_windows:
            self.assertEqual(len(controller.state["pending"][identity]["observed"]), 1)
        third = await controller.prepare_update(self.solver(("future-d", "future-e")), 3)
        self.assertEqual(int((third.data["role_id"] == 1).sum()), 2)
        for identity in first_windows:
            self.assertEqual(len(controller.state["pending"][identity]["observed"]), 2)
            self.assertEqual(controller.state["pending"][identity]["status"], "settled")

    async def test_minimum_delay_uses_later_real_batches_and_stale_labels_are_not_zeroed(self):
        self.config.skill_harness.joint.update(min_delay_steps=2, max_replay_lag_steps=0)
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        second = await self.update(controller, self.solver(("future-b", "future-c")), 2)
        self.assertFalse((second.data["role_id"] == 1).any())
        self.assertEqual(self.probe_calls, [])
        third = await controller.prepare_update(self.solver(("future-d", "future-e")), 3)
        self.assertFalse((third.data["role_id"] == 1).any())
        settled = [row for row in controller.state["pending"].values() if row["status"] == "settled"]
        self.assertEqual(len(settled), 2)
        self.assertTrue(all(row["utility"] == 1 for row in settled))
        self.assertTrue(
            all(controller.ledger.training_status(row["settlement_id"])["state"] == "ready" for row in settled)
        )

    async def test_claimed_update_clock_cannot_acknowledge_an_optimizer_that_did_not_run(self):
        controller = await self.create()
        data = self.solver()
        prepared = await controller.prepare_update(data, 1)
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertEqual(controller.state["optimizer_step"], 0)
        self.assertEqual(self.curate_calls, [])
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.prepare_update(data, 1)

    async def test_anchor_fence_break_preserves_uncertain_leases_for_reconciliation(self):
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        self.actor.break_anchor_fence = True
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.prepare_update(self.solver(("future-b", "future-c")), 2)
        self.assertIsNotNone(controller.state["prepared_update"])
        for acknowledgment in controller.state["prepared_update"]["acknowledgments"]:
            self.assertEqual(controller.ledger.training_status(acknowledgment["settlement_id"])["state"], "leased")

    async def test_resume_preserves_pending_and_rejects_wrong_protocol_or_clock(self):
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        state = controller.export_state()
        controller.restore_state(state, expected_optimizer_step=1)
        self.assertEqual(len(controller.state["pending"]), 2)
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            controller.restore_state(state, expected_optimizer_step=0)
        altered = copy.deepcopy(state)
        altered["protocol_fingerprint"] = "wrong"
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            controller.restore_state(altered, expected_optimizer_step=1)

    async def test_original_prompts_and_train_partition_are_required(self):
        controller = await self.create()
        data = self.solver()
        data.pop("raw_prompt")
        with self.assertRaisesRegex(ValueError, "raw_prompt"):
            await controller.prepare_update(data, 1)
        data = self.solver()
        tools = tu.get(data, "tools_kwargs")
        tools[0]["task"]["metadata"]["dataset_phase"] = "dev"
        tu.assign_non_tensor_data(data, "tools_kwargs", tools)
        with self.assertRaisesRegex(ValueError, "optimization-train"):
            await controller.prepare_update(data, 1)

    async def test_pure_preflight_identity_allocates_no_route_or_gateway(self):
        with patch.object(
            controller_module, "create_policy_routes", side_effect=AssertionError("No routes in preflight")
        ):
            identity = controller_module.preflight_controller_identity(self.config, native_runtime=self.runtime)
        self.assertEqual(identity.template_version, self.runtime.template_version)
        self.assertEqual(len(identity.frozen_version), 64)
        self.assertEqual(self.gateways, [])

    async def test_one_invalid_candidate_censors_entire_cohort_after_one_verified_optimizer(self):
        self.curate_bad_content[1] = "This is not JSON."
        controller = await self.create()
        data = self.solver()
        prepared = await self.update(controller, data, 1)
        self.assertEqual(self.actor.clock, 1)
        self.assertEqual(len(self.actor.update_calls), 1)
        self.assertIsNone(controller.state["prepared_update"])
        self.assertEqual(controller.state["optimizer_step"], 1)
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(controller.ledger.sample_ready(100), [])
        self.assertEqual(controller.store.get_snapshot().skills, ())
        self.assertEqual(len(list((controller.replay.root / "records").glob("*.json"))), 1)
        dispositions = list(controller.state["curate_job_dispositions"].values())
        self.assertEqual({row["status"] for row in dispositions}, {"semantic_rejected", "censored"})
        rejected = next(row for row in dispositions if row["status"] == "semantic_rejected")
        self.assertFalse(rejected["training_label_created"])
        self.assertNotIn("skill_ref", rejected)
        self.assertNotIn("retained_record_id", rejected)
        self.assertTrue(all(row["status"] == "censored" for row in controller.state["cohorts"].values()))
        report = controller.state["post_update_reports"][prepared.update_id]
        self.assertEqual(
            (report["new_curate_windows"], report["curate_semantic_rejections"], report["curate_retained_actions"]),
            (0, 1, 1),
        )
        repeat = await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertTrue(repeat["already_completed"])
        self.assertEqual(len(self.curate_calls), 2)
        following = await controller.prepare_update(self.solver(("future-b", "future-c")), 2)
        self.assertFalse((following.data["role_id"] == 1).any())
        self.assertEqual(self.probe_calls, [])

    async def test_invalid_field_type_is_nonfatal_but_has_no_fake_replay_label(self):
        self.curate_bad_fields[0] = {"title": []}
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        self.assertEqual(self.actor.clock, 1)
        self.assertEqual(len(self.curate_calls), 1)
        self.assertEqual(controller.state["curate_metrics"]["curate_semantic_rejections"], 1)
        self.assertEqual(list((controller.replay.root / "records").glob("*.json")), [])

    async def test_proved_tool_invocation_censors_cohort_without_sampling_its_other_member(self):
        self.curate_tool_indices.add(0)
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        self.assertEqual(self.actor.clock, 1)
        self.assertEqual(len(self.curate_calls), 1)
        self.assertEqual(controller.state["pending"], {})
        self.assertEqual(controller.state["curate_metrics"]["curate_retained_actions"], 0)
        self.assertIsNone(controller.state["prepared_update"])

    async def test_oversize_whole_source_is_censored_before_sampling_against_total_capacity(self):
        controller = await self.create()
        data = self.solver()
        files = list((self.root / "episodes").glob("*/*/initial-state.json"))
        for path in files:
            path.write_text(json.dumps({"observation": "Long observed feedback. " * 3000}))
        originals = {str(path): path.read_bytes() for path in files}
        await self.update(controller, data, 1)
        self.assertEqual(self.curate_calls, [])
        self.assertEqual(controller.state["pending"], {})
        disposition = next(
            row for row in controller.state["curate_job_dispositions"].values() if row["status"] == "source_censored"
        )
        policy = disposition["input_policy"]
        self.assertEqual(policy["gateway_total_capacity"], 36864)
        self.assertEqual(policy["host_prompt_budget"], 36864 - 4096 - 2048)
        self.assertGreater(policy["host_prompt_tokens"], policy["host_prompt_budget"])
        self.assertFalse(policy["source_truncated"])
        self.assertFalse(disposition["sampled"])
        self.assertEqual(originals, {str(path): path.read_bytes() for path in files})

    async def test_actual_version_failure_is_fatal_after_update_and_preserves_postprocessing_cut(self):
        controller = await self.create()
        native = controller._collect_curate

        async def broken(**kwargs):
            kwargs["gateway_manager"].client.optimizer_step += 1
            return await native(**kwargs)

        controller._collect_curate = broken
        data = self.solver()
        prepared = await controller.prepare_update(data, 1)
        self.actor.update_actor(prepared.data)
        with self.assertRaisesRegex(curate.CurateProtocolError, "weight versions"):
            await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertEqual(controller.state["optimizer_step"], 1)
        self.assertEqual(controller.state["prepared_update"]["status"], "optimizer_applied")
        self.assertNotIn(prepared.update_id, controller.state.get("post_update_reports", {}))
        self.assertEqual(controller.state.get("curate_metrics", {}), {})

    async def test_replay_io_failure_is_not_mislabeled_as_semantic_and_retry_does_not_step_again(self):
        controller = await self.create()
        data = self.solver()
        prepared = await controller.prepare_update(data, 1)
        self.actor.update_actor(prepared.data)
        with patch.object(controller.replay, "append", side_effect=OSError("durable append failed")):
            with self.assertRaises(OSError):
                await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertEqual(controller.state["prepared_update"]["status"], "optimizer_applied")
        self.assertEqual(controller.state.get("curate_job_dispositions", {}), {})
        await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertEqual(self.actor.clock, 1)
        self.assertEqual(len(self.actor.update_calls), 1)
        self.assertEqual(len(controller.state["pending"]), 2)

    async def test_unverified_optimizer_receipt_keeps_existing_replay_leases_uncertain(self):
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        data = self.solver(("future-b", "future-c"))
        prepared = await controller.prepare_update(data, 2)
        self.actor.update_actor(prepared.data)
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.after_update_and_sync(data, 2, prepared=prepared, optimizer_step=3)
        for ack in prepared.acknowledgments:
            self.assertEqual(controller.ledger.training_status(ack["settlement_id"])["state"], "leased")


if __name__ == "__main__":
    unittest.main()
