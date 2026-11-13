"""CPU end-to-end controller contracts with native/model boundaries replaced."""

from __future__ import annotations

import asyncio
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
# Replace only external framework classes; no native services are started.
try:
    importlib.import_module("uni_agent")
except ModuleNotFoundError:
    entry, framework = types.ModuleType("uni_agent.framework.entry"), types.ModuleType("uni_agent.framework.framework")
    entry.AgentFrameworkRolloutAdapter, framework.GatewayAgentFramework = object, object
    sys.modules.update({entry.__name__: entry, framework.__name__: framework})
    try:
        importlib.import_module(PACKAGE + ".framework")
    finally:
        sys.modules.pop(entry.__name__)
        sys.modules.pop(framework.__name__)
controller_module = importlib.import_module(PACKAGE + ".controller")
curate = importlib.import_module(PACKAGE + ".curate")
probes = importlib.import_module(PACKAGE + ".probes")
context = importlib.import_module(PACKAGE + ".context")
store_module = importlib.import_module(PACKAGE + ".store")
data_module = importlib.import_module(PACKAGE + ".alfworld.data")
batcher = importlib.import_module(PACKAGE + ".replay_batch")


class FakeAccountPool:
    """Replace OS lease allocation only; the real branch proof verifier still runs."""

    def __init__(self):
        self.leases, self.finished = {}, {}

    def acquire(self, *, run_root, scope_id, kind):
        identity = f"{len(self.leases) + 1:032x}"
        ref = {
            "lease_id": identity, "token": "CPU-lease-fence-" + identity,
            "uid": 1001, "gid": 1001, "name": "CPU-pair-account",
            "owner": {"run_root": str(Path(run_root).resolve()), "scope_id": scope_id, "kind": kind},
        }
        self.leases[identity] = copy.deepcopy(ref)
        return ref

    @staticmethod
    def public(ref):
        return {key: copy.deepcopy(ref[key]) for key in ("lease_id", "uid", "gid", "name", "owner")}

    def finish(self, ref, *, abandoned=False):
        if self.leases[ref["lease_id"]] != ref:
            raise RuntimeError("CPU lease fence differs")
        self.finished[ref["lease_id"]] = bool(abandoned)



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


class ControllerFixture(unittest.IsolatedAsyncioTestCase):
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
        self.unknown_backend = False
        self.account_pool = FakeAccountPool()
        self.cf_input_attack = None
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
            def render(messages):
                text = "".join("<" + item["role"] + ">" + item["content"] + "</turn>" for item in messages)
                return context.RenderedPrompt(text, tuple(text.encode()))

            built = context.build_skill_context(
                fields["raw_prompt"],
                store=store,
                snapshot_id=metadata["skill_snapshot_id"],
                task_family=metadata["task_type"],
                render_prompt=render,
                encode_text=lambda text: list(text.encode()),
                max_prompt_tokens=100000,
                skill_token_budget=10000,
                top_k=1 if loaded and self.cf_input_attack == "displaced" else 6,
            )
            (output / "agent-messages.json").write_text(json.dumps(built.messages))
            (output / "skill-context.json").write_text(json.dumps(built.manifest))
            prefix = "Native fixed prefix\n"
            if self.cf_input_attack == "home":
                prefix += "Skill root=/tmp/codex-home-" + ("a" if loaded else "b") * 32 + "/skills/.system\n"
            if self.cf_input_attack == "path":
                prefix += "Native cwd=/tmp/uncontrolled-" + str(loaded) + "\n"
            native_prompt = prefix + render(built.messages).text
            group, variant = metadata["credit_probe_group"], metadata["credit_probe_variant"]
            workdir = "/tmp/alfworld-skill-episodes/credit_" + group
            home = "/tmp/codex-home-" + hashlib.sha256(session.encode()).hexdigest()[:32]
            cleanup = {"status": "cleaned", "sources": [
                {"path": path, "removed": True} for path in (workdir, home)
            ]}
            # This replaces the native/backend boundary, not the real CF
            # verifier. A normal completed session must still attest its exact
            # private attempt and every owned backend RPC having drained.
            binding = {
                "binding_id": "binding_" + hashlib.sha256(session.encode()).hexdigest()[:16],
                "binding_sha256": hashlib.sha256(session.encode()).hexdigest(),
                "attempt_id": "attempt_" + session,
                "gateway_session_id": session,
                "environment_session_id": "environment_" + hashlib.sha256(session.encode()).hexdigest()[:16],
            }
            receipt = {
                **binding,
                "schema_version": 1,
                "status": "unconfirmed" if self.unknown_backend else "drained",
                "fenced": False,
                "active_requests": int(self.unknown_backend),
                "backend_stop_confirmed": not self.unknown_backend,
                "expected_backend_terminal_stop": False,
            }
            return {
                "session_id": session,
                "finished": True,
                "reward": reward,
                "accuracy": reward,
                "policy_attestation": arguments["frozen_attestation"],
                "extra_info": {
                    "eval_completed": True,
                    "agent_ok": True,
                    "error": None,
                    "episode_output_dir": str(output),
                    "attempt_id": binding["attempt_id"],
                    "environment_session_id": binding["environment_session_id"],
                    "backend_binding": binding,
                    "backend_terminal": receipt,
                    "backend_stop_confirmed": not self.unknown_backend,
                    "account_isolation": {
                        **self.account_pool.public(metadata["host_account_lease"]),
                        "policy_version": "fake-provider-v1",
                        "status": "attempt_closed", "backend_drained": not self.unknown_backend,
                        "uid_drained": True,
                    },
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
                        "native_files": cleanup,
                        "host_uid": 0,
                        "agent_uid": 1001,
                        "agent_gid": 1001,
                    },
                    "skill_context": built.manifest,
                },
                "trajectories": [
                    {
                        "extra_fields": {},
                        "prompt_ids": list(native_prompt.encode()),
                        "response_ids": [20],
                        "response_mask": [1],
                    }
                ],
            }

        return await probes.collect_counterfactual_probe(
            **kwargs,
            run_branch=branch,
            decode_prompt_ids=lambda ids: bytes(ids).decode(),
            _account_pool=self.account_pool,
        )

    async def update(self, controller, data, step):
        prepared = await controller.prepare_update(data, step)
        self.actor.update_actor(prepared.data)
        await controller.after_update_and_sync(data, step, prepared=prepared, optimizer_step=self.actor.clock)
        return prepared



class ControllerContractTest(ControllerFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.controllers = []
        self.addAsyncCleanup(self.stop_all)

    async def stop_all(self):
        for controller in self.controllers:
            await asyncio.wait_for(controller.stop_background(), timeout=3)

    async def create(self):
        controller = await super().create()
        self.controllers.append(controller)
        return controller

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
        self.assertEqual(self.actor.update_calls, [])

    async def test_preflight_identity_allocates_no_route_or_gateway(self):
        with patch.object(controller_module, "create_policy_routes", side_effect=AssertionError("No preflight routes")):
            identity = controller_module.preflight_controller_identity(self.config, native_runtime=self.runtime)
        self.assertEqual(identity.template_version, self.runtime.template_version)
        self.assertEqual(len(identity.frozen_version), 64)
        self.assertEqual(self.gateways, [])

    async def test_claimed_clock_cannot_acknowledge_an_optimizer_that_did_not_run(self):
        controller = await self.create()
        data = self.solver()
        prepared = await controller.prepare_update(data, 1)
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.after_update_and_sync(data, 1, prepared=prepared, optimizer_step=1)
        self.assertEqual(controller.state["optimizer_step"], 0)
        self.assertEqual(self.curate_calls, [])
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            await controller.prepare_update(data, 1)

    async def test_controller_payload_preserves_pending_and_requires_its_protocol_and_clock(self):
        controller = await self.create()
        await self.update(controller, self.solver(), 1)
        await controller.leave_learner()
        async def generated():
            while controller.source_queue.metrics()["sources_generated"] != 2:
                await asyncio.sleep(0.005)
        await asyncio.wait_for(generated(), timeout=3)
        await controller.pause_background_writes()
        state = controller.export_state()
        controller.restore_state(state, expected_optimizer_step=1)
        self.assertEqual(len(controller.state["pending"]), 4)
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            controller.restore_state(state, expected_optimizer_step=0)
        altered = copy.deepcopy(state)
        altered["protocol_fingerprint"] = "wrong"
        with self.assertRaises(controller_module.ControllerRecoveryRequired):
            controller.restore_state(altered, expected_optimizer_step=1)


if __name__ == "__main__":
    unittest.main()
