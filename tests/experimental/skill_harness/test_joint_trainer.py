"""CPU orchestration tests with real controller, joint worker, loss and optimizer.

Only service/engine boundaries are replaced. No Ray initialization, GPU model,
process group, gateway server or ALFWorld environment is started.
"""

# ruff: noqa: E402 - pin runtime provenance before importing its optional dependencies.

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

# Only the external framework base classes are absent in the CPU environment.
# The tested controller, joint worker, optimizer and loss below remain real.
try:
    importlib.import_module("uni_agent")
except ModuleNotFoundError:
    entry, framework = ModuleType("uni_agent.framework.entry"), ModuleType("uni_agent.framework.framework")
    entry.AgentFrameworkRolloutAdapter, framework.GatewayAgentFramework = object, object
    sys.modules.update({entry.__name__: entry, framework.__name__: framework})
    try:
        importlib.import_module("verl.experimental.skill_harness.framework")
    finally:
        sys.modules.pop(entry.__name__)
        sys.modules.pop(framework.__name__)

import ray
import torch
from omegaconf import OmegaConf

from verl.experimental.skill_harness.alfworld.data import stable_game_id
from verl.experimental.skill_harness.controller import JointSettings, SkillJointController
from verl.experimental.skill_harness.credit import CreditLedger
from verl.experimental.skill_harness.joint_checkpoint import JointCheckpointCoordinator, JointCheckpointError
from verl.experimental.skill_harness.joint_worker import JointEMATrainingWorker
from verl.experimental.skill_harness.replay import CurateReplay
from verl.experimental.skill_harness.replay_batch import _nested, _position_ids
from verl.experimental.skill_harness.source_queue import SourceQueue
from verl.experimental.skill_harness.store import SkillStore
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.engine_workers import TrainingWorker
from verl.workers.utils.losses import ppo_loss

MODULE_PATH = Path(os.environ.get(
    "SKILL_JOINT_TRAINER_PATH",
    Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/joint_trainer.py",
))
SPEC = importlib.util.spec_from_file_location("verl.experimental.skill_harness.joint_trainer", MODULE_PATH)
joint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = joint
SPEC.loader.exec_module(joint)


def config(root):
    return OmegaConf.create({
        "skill_harness": {"artifact_root": str(root / "artifacts"),
                          "store_root": str(root / "skills"), "evaluation": {"final": False},
                          "ema": {"enabled": True, "decay": 0.95, "update_every": 1, "buffer_policy": "copy"},
                          "curate_loss": {"coefficient": 0.1}, "joint": {}},
        "trainer": {"v1": {"trainer_mode": "sync"}, "max_actor_ckpt_to_keep": None,
                    "remove_previous_ckpt_in_save": False, "del_local_ckpt_after_load": False,
                    "resume_mode": "disable", "default_local_dir": str(root / "checkpoints"),
                    "default_hdfs_dir": None, "n_gpus_per_node": 1, "nnodes": 1},
        "actor_rollout_ref": {
            "model": {"path": str(root / "model"), "lora": {"rank": 0}, "use_fused_kernels": False},
            "actor": {"ppo_epochs": 1, "checkpoint": {"async_save": False}, "use_kl_loss": False,
                      "calculate_entropy": False, "entropy_coeff": 0., "data_loader_seed": 3, "shuffle": False},
            "rollout": {"temperature": 1., "calculate_log_probs": False,
                        "custom": {"agent_framework": {"gateway_count": 1}}},
        },
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "rollout_correction": None},
        "critic": {"enable": False},
        "reward": {"reward_model": {"enable": False, "enable_resource_pool": False,
                                    "n_gpus_per_node": 0, "nnodes": 1}},
    })


def solver_data():
    path, digest = "json_2.1.1/train/pick-task/trial-1/game.tw-pddl", "a" * 64
    metadata = {"game_id": stable_game_id(path, digest), "game_sha256": digest,
                "relative_game_path": path, "catalog_digest": "b" * 64, "split": "train",
                "dataset_phase": "train", "scene_id": "scene-1", "task_type": "pick_and_place_simple"}
    prompts, responses = [torch.tensor([10, 11]), torch.tensor([12])], [torch.tensor([20, 21]), torch.tensor([22])]
    ids = [torch.cat((p, r)) for p, r in zip(prompts, responses, strict=False)]
    return tu.get_tensordict({
        "prompts": _nested(prompts), "responses": _nested(responses), "input_ids": _nested(ids),
        "position_ids": _nested([_position_ids(x.numel(), 4) for x in ids]),
        "response_mask": _nested([torch.tensor([1, 1]), torch.tensor([1])]),
        "loss_mask": _nested([torch.tensor([1, 1]), torch.tensor([1])]),
        "old_log_probs": _nested([torch.tensor([-1., -1.]), torch.tensor([-1.])]),
        "advantages": _nested([torch.tensor([1., 1.]), torch.tensor([-1.])]),
        "uid": ["source-a", "source-b"], "session_id": ["sample-a", "sample-b"],
        "tools_kwargs": [{"task": {"name": "alfworld_skill_training", "metadata": metadata}}] * 2,
        "extra_fields": [{}, {}], "raw_prompt": [[{"role": "user", "content": "Solve the observed task"}]] * 2,
    })


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_probability = torch.nn.Parameter(torch.tensor(-1.0))


class CPUEngine:
    def __init__(self):
        self.module = Policy()
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.05)
        self.steps = 0
        self.optimizer.register_step_post_hook(self.count_step)

    def count_step(self, *_args):
        self.steps += 1

    def get_data_parallel_group(self):
        return None

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def is_mp_src_rank_with_outputs(self):
        return True

    def train_mode(self, **_kwargs):
        return nullcontext()

    def train_batch(self, data, loss_function):
        self.optimizer.zero_grad()
        total = 0.
        for index in range(len(data)):
            micro = tu.index_select_tensor_dict(data, torch.tensor([index]))
            length = micro["responses"][0].numel()
            output = {"log_probs": self.module.log_probability.expand(1, length)}
            loss, _ = loss_function(model_output=output, data=micro, dp_group=None)
            loss.backward()
            total += float(loss.detach())
        gradient = float(self.module.log_probability.grad.abs())
        self.optimizer.step()
        return tu.get_tensordict({}, {"metrics": {
            "actor/pg_loss": total, "skill/joint_loss": total, "grad_norm": gradient, "mfu": 0.01,
        }})


def cpu_adapter(worker, data):
    return worker.engine.train_batch(data, worker.loss_fn)


class ActorService:
    world_size = 1

    def __init__(self, events):
        self.events, self.updates, self.initializations = events, 0, 0
        self.loads, self.ema_loads, self.model_saves = 0, 0, 0
        self.worker = object.__new__(JointEMATrainingWorker)
        self.worker._joint_control_lock = threading.RLock()
        self.worker._joint_update_active = False
        self.worker._rank = 0
        self.worker.profiler = SimpleNamespace(step=lambda: None)
        self.worker.engine = CPUEngine()
        self.worker.set_loss_fn(partial(ppo_loss, config=ActorConfig(
            strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1,
        )))
        self.fail_update = False

    def get_online_layout_metadata(self, buffer_policy="copy"):
        from verl.experimental.skill_harness.ema import describe_local_layout

        module = self.worker.engine.module
        return [{"worker_rank": 0, "world_size": 1,
                 **describe_local_layout(module.named_parameters(), module.named_buffers(),
                                         buffer_policy=buffer_policy)}]

    def configure_joint_loss(self, **settings):
        self.events.append("configure_loss")
        return [self.worker.configure_joint_loss(**settings)]

    def initialize_ema(self, **settings):
        self.events.append("initialize_ema")
        self.initializations += 1
        return [self.worker.initialize_ema(**settings)]

    def get_ema_metadata(self):
        return [self.worker._ready().metadata()]

    def update_actor(self, data):
        self.events.append("actor_update")
        self.updates += 1
        if self.fail_update:
            raise RuntimeError("worker RPC failed before optimizer")
        return self.worker.train_mini_batch(data)

    def save_ema(self, directory):
        return [self.worker.save_ema(directory)]

    def save_checkpoint(self, local_path, _remote_path, _global_step, max_ckpt_to_keep=None):
        self.model_saves += 1
        self.events.append("core_model_save")
        if max_ckpt_to_keep is not None:
            raise AssertionError("joint checkpoints cannot prune native actor files early")
        root = Path(local_path)
        root.mkdir(parents=True)
        torch.save(self.worker.engine.module.state_dict(), root / "model_world_size_1_rank_0.pt")
        torch.save(self.worker.engine.optimizer.state_dict(), root / "optim_world_size_1_rank_0.pt")
        torch.save({"rng": torch.get_rng_state()}, root / "extra_state_world_size_1_rank_0.pt")
        return [None]

    def load_checkpoint(self, local_path, **_kwargs):
        self.loads += 1
        self.events.append("load_online")
        root = Path(local_path)
        self.worker.engine.module.load_state_dict(torch.load(root / "model_world_size_1_rank_0.pt", weights_only=True))
        self.worker.engine.optimizer.load_state_dict(
            torch.load(root / "optim_world_size_1_rank_0.pt", weights_only=True)
        )
        return [None]

    def load_ema(self, directory, **kwargs):
        self.ema_loads += 1
        self.events.append("load_ema")
        return [self.worker.load_ema(directory, **kwargs)]


class GatewayService:
    async def shutdown(self):
        return None


class CuratorRouteService:
    def __init__(self, config, events):
        self.config, self.events, self.optimizer_step = config, events, 0

    async def sync_ema(self, step):
        self.optimizer_step = step
        self.events.append("curator_sync")

    def get_client(self):
        return object()


class JointTrainerTests(unittest.TestCase):
    def test_background_loop_progresses_between_synchronous_learner_calls(self):
        driver = object.__new__(joint.SkillJointSyncTrainer)
        driver._joint_loop = asyncio.new_event_loop()
        driver._joint_loop_ready = threading.Event()
        driver._joint_thread = None
        completed = threading.Event()

        async def start_background():
            async def work():
                await asyncio.sleep(0.02)
                completed.set()
            return SimpleNamespace(task=asyncio.create_task(work()))

        try:
            task = driver._await(start_background()).task
            # No second driver call pumps the event loop. This models the
            # learner spending time in a synchronous solver rollout.
            self.assertTrue(completed.wait(2), "Background work stalled outside _await")
            self.assertIsNone(driver._await(task))
            self.assertTrue(task.done())
            self.assertIsNone(task.result())
        finally:
            if driver._joint_loop.is_running():
                driver._joint_loop.call_soon_threadsafe(driver._joint_loop.stop)
                driver._joint_thread.join(2)
            driver._joint_loop.close()

    def setUp(self):
        self.assertFalse(ray.is_initialized())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.events = Path(temporary.name), []
        self.trainer = self.make_trainer(self.root)
        self.trainer.on_init_end()
        self.controller = self.make_controller(self.trainer)
        self.trainer._joint_controller = self.controller
        self.source = solver_data()
        self.batch = SimpleNamespace(keys=["source-a_sample-a_0", "source-b_sample-b_0"],
                                     partition_id="train", extra_info={})

    def make_trainer(self, root):
        trainer = object.__new__(joint.SkillJointSyncTrainer)
        trainer.config, trainer.artifact_root = config(root), root / "artifacts"
        trainer.artifact_root.mkdir(parents=True, exist_ok=True)
        trainer._joint_lock, trainer._joint_loop = threading.RLock(), asyncio.new_event_loop()
        self.addCleanup(trainer._joint_loop.close)
        trainer._joint_controller, trainer._joint_restored = None, None
        trainer._joint_coordinator = JointCheckpointCoordinator()
        trainer.actor_rollout_wg = ActorService(self.events)
        identity = SimpleNamespace(frozen_version="frozen:1", protocol_fingerprint="p" * 64,
                                   tokenizer_version="tok:v1", template_version="template:v1",
                                   runtime=SimpleNamespace(template_version="template:v1"))
        identity.manifest = lambda: {"frozen_version": identity.frozen_version,
                                     "protocol_fingerprint": identity.protocol_fingerprint}
        trainer._joint_identity = identity
        trainer._joint_compatibility = {
            "world_size": 1,
            "layout_fingerprint": joint.ordered_layout_fingerprint(
                trainer.actor_rollout_wg.get_online_layout_metadata(), 1,
            ),
            "versions": {"model": "model:v1",
                         "tokenizer": "tok:v1", "template": "template:v1"},
            "evaluator_fingerprint": identity.frozen_version, "protocol_fingerprint": identity.protocol_fingerprint,
        }
        trainer.global_steps, trainer.total_training_steps = 0, 2
        trainer.use_critic, trainer.trainer_mode = False, "sync"
        trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step: self.events.append("solver_sync"))
        trainer.checkpoint_callback = joint.SkillJointCheckpointCallback(config=trainer.config)
        trainer.train_dataloader = SimpleNamespace(load_state_dict=Mock(), state_dict=lambda: {"position": 2})
        return trainer

    def make_controller(self, trainer):
        controller = object.__new__(SkillJointController)
        controller.trainer, controller.config, controller.actor = trainer, trainer.config, trainer.actor_rollout_wg
        controller.settings = JointSettings(solver_gpus=1, available_gpus=3, curator_gpus=1, credit_gpus=1,
                                             candidates_per_source=1, probes_per_step=0, replay_batch_size=0)
        controller.artifact_root = trainer.artifact_root
        controller.state_path = trainer.artifact_root / "controller/state.json"
        controller.store = SkillStore(trainer.config.skill_harness.store_root)
        controller.replay = CurateReplay(trainer.artifact_root / "curate_replay")
        controller.ledger = CreditLedger(trainer.artifact_root / "credit_ledger", controller.replay)
        controller.frozen_version, controller.protocol_fingerprint = "frozen:1", "p" * 64
        controller.loading = {"tokenizer_version": "tok:v1"}
        controller.runtime = SimpleNamespace(template_version="template:v1")
        controller.dp_size, controller.pad_token_id, controller._busy = 1, 0, False
        controller._writes_allowed = asyncio.Event()
        controller._writes_allowed.set()
        controller._background_pump = None
        controller._background_tasks = {}
        controller._background_stopping = False
        controller._ema_publication_task = None
        controller._learner_active = True
        controller.ema_gateway, controller.frozen_gateway = GatewayService(), GatewayService()
        controller.routes = {"curator": CuratorRouteService(trainer.config, self.events)}

        async def gateway_factory(**_kwargs):
            self.events.append("gateway_refresh")
            return GatewayService()

        controller._gateway_factory = gateway_factory
        controller.state = {"schema_version": 1, "optimizer_step": 0, "protocol_fingerprint": "p" * 64,
                            "pending": {}, "cohorts": {}, "prepared_update": None,
                            "applied_update_ids": {}, "collections": {}, "last_global_step": 0, "probe_jobs": {}}
        controller.source_queue = SourceQueue(controller.state.setdefault("source_queue", {}),
                                              persist=controller.save_state,
                                              max_backlog=controller.settings.max_source_backlog)
        prepare, after = controller.prepare_update, controller.after_update_and_sync

        async def observed_prepare(*args, **kwargs):
            self.events.append("prepare")
            return await prepare(*args, **kwargs)

        async def observed_after(*args, **kwargs):
            self.events.append("after")
            return await after(*args, **kwargs)

        controller.prepare_update, controller.after_update_and_sync = observed_prepare, observed_after
        return controller

    def update(self):
        self.trainer.global_steps = 1
        fake_tq = SimpleNamespace(kv_clear=Mock(), kv_batch_get=Mock(
            side_effect=lambda **kwargs: self.source.select(*kwargs["select_fields"]).clone()
        ), kv_describe=Mock(side_effect=lambda **kwargs: {
            key: {"fields": list(self.source.keys()), "unfinalized": [], "links": {}} for key in kwargs["keys"]
        }))

        def gather(output, value, *_args, **_kwargs):
            output[0] = value

        metrics = {}
        with (patch.object(joint, "tq", fake_tq), patch.object(TrainingWorker, "train_batch", cpu_adapter),
              patch.object(torch.distributed, "get_world_size", return_value=1),
              patch.object(torch.distributed, "all_gather_object", side_effect=gather)):
            result = self.trainer._update_actor(self.batch, metrics)
        fake_tq.kv_clear.assert_not_called()
        return result, metrics, fake_tq

    def save_native_then_callback(self):
        checkpoint = self.root / "checkpoints/global_step_1"
        with patch.object(torch.distributed, "get_world_size", return_value=1):
            self.trainer._save_checkpoint()
        return checkpoint

    def test_one_actual_optimizer_call_then_real_controller_ack_before_source_cleanup(self):
        result, metrics, fake_tq = self.update()
        self.assertIs(result, self.batch)
        self.assertEqual(self.trainer.actor_rollout_wg.updates, 1)
        self.assertEqual(self.trainer.actor_rollout_wg.model_saves, 0)
        self.assertEqual(self.trainer.actor_rollout_wg.worker.engine.steps, 1)
        self.assertEqual(self.trainer.actor_rollout_wg.worker.parameter_ema.num_updates, 1)
        self.assertEqual(self.controller.state["optimizer_step"], 1)
        self.assertIsNone(self.controller.state["prepared_update"])
        self.assertEqual(len(self.controller.state["applied_update_ids"]), 1)
        order = [self.events.index(name) for name in (
            "prepare", "actor_update", "after", "curator_sync", "gateway_refresh",
        )]
        self.assertEqual(order, sorted(order))
        self.assertIn("actor/pg_loss", metrics)
        self.assertIn("skill/joint_loss", metrics)
        self.assertIn("perf/mfu/actor", metrics)
        self.assertFalse(any(name.startswith("actor/actor/") for name in metrics))
        self.assertIn("raw_prompt", fake_tq.kv_batch_get.call_args.kwargs["select_fields"])
        self.assertEqual(self.source["advantages"][0].tolist(), [1., 1.])

    def test_failed_worker_does_not_ack_clear_or_retry_optimizer(self):
        self.trainer.actor_rollout_wg.fail_update = True
        with self.assertRaisesRegex(RuntimeError, "worker RPC"):
            self.update()
        self.assertEqual(self.trainer.actor_rollout_wg.updates, 1)
        self.assertEqual(self.trainer.actor_rollout_wg.worker.engine.steps, 0)
        self.assertEqual(self.controller.state["optimizer_step"], 0)
        self.assertIsNotNone(self.controller.state["prepared_update"])
        self.assertNotIn("after", self.events)

    def test_fresh_store_loss_and_ema_precede_parent_online_sync(self):
        self.assertTrue(Path(self.trainer.config.skill_harness.store_root).is_dir())
        self.assertEqual(self.events[:3], ["configure_loss", "initialize_ema", "solver_sync"])
        self.assertEqual(self.trainer.actor_rollout_wg.initializations, 1)

    def test_controller_is_created_once_before_training(self):
        self.trainer._joint_controller = None
        self.trainer.global_steps = 1

        async def create_service(trainer, **kwargs):
            self.events.append("controller_ready")
            self.assertIs(trainer, self.trainer)
            self.assertIs(kwargs["native_runtime"], self.trainer._joint_identity.runtime)
            return self.controller

        with patch.object(SkillJointController, "create", side_effect=create_service):
            self.trainer.on_train_begin()
        self.assertIs(self.trainer._joint_controller, self.controller)
        with self.assertRaisesRegex(RuntimeError, "twice"):
            self.trainer.on_train_begin()

    def test_post_save_callback_binds_completed_core_save_without_model_or_optimizer_step(self):
        self.update()
        checkpoint = self.save_native_then_callback()
        self.assertEqual(self.trainer.actor_rollout_wg.model_saves, 1)
        self.assertTrue((checkpoint / "joint_manifest.json").is_file())
        self.assertEqual(self.trainer.actor_rollout_wg.worker.engine.steps, 1)
        self.assertEqual(self.trainer.actor_rollout_wg.updates, 1)
        self.assertEqual(self.trainer._last_joint_checkpoint.payload["spec"]["optimizer_step"], 1)
        with self.assertRaisesRegex(JointCheckpointError, "asynchronous"):
            joint.SkillJointCheckpointCallback().on_save(self.trainer, 1, str(checkpoint), async_save=True)

    def test_complete_resume_restores_dataloader_online_and_ema_before_init_sync(self):
        self.update()
        checkpoint = self.save_native_then_callback()
        resumed = self.make_trainer(self.root / "resume-run")
        resumed.config.trainer.resume_mode, resumed.config.trainer.resume_from_path = "resume_path", str(checkpoint)
        with patch.object(torch.distributed, "get_world_size", return_value=1):
            resumed._load_checkpoint()
        resumed.on_init_end()
        self.assertEqual(resumed.global_steps, 1)
        self.assertEqual(resumed.actor_rollout_wg.loads, 1)
        self.assertEqual(resumed.actor_rollout_wg.ema_loads, 1)
        self.assertEqual(resumed.actor_rollout_wg.initializations, 0)
        self.assertEqual(resumed.actor_rollout_wg.worker.engine.steps, 0)
        resumed.train_dataloader.load_state_dict.assert_called_once_with({"position": 2})
        self.assertEqual(resumed._joint_restored.controller_state["optimizer_step"], 1)
        self.assertEqual(Path(resumed.config.skill_harness.store_root), resumed._joint_restored.skill_store.root)

    def test_old_core_pointer_never_adopts_a_nonjoint_checkpoint(self):
        trainer = self.make_trainer(self.root / "old-core")
        trainer.config.trainer.resume_mode = "auto"
        checkpoint_root = Path(trainer.config.trainer.default_local_dir)
        checkpoint_root.mkdir()
        (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("99")
        with self.assertRaisesRegex(JointCheckpointError, "COMPLETE"):
            trainer._load_checkpoint()
        self.assertEqual(trainer.actor_rollout_wg.loads, 0)
        self.assertEqual(trainer.actor_rollout_wg.initializations, 0)

    def test_resume_does_not_trust_saved_versions_over_current_runtime(self):
        self.update()
        checkpoint = self.save_native_then_callback()
        resumed = self.make_trainer(self.root / "incompatible-run")
        resumed.config.trainer.resume_mode, resumed.config.trainer.resume_from_path = "resume_path", str(checkpoint)
        resumed._joint_compatibility["versions"]["tokenizer"] = "different-current-tokenizer"
        with self.assertRaisesRegex(JointCheckpointError, "mismatch"):
            resumed._load_checkpoint()
        self.assertEqual(resumed.actor_rollout_wg.loads, 0)
        self.assertEqual(resumed.actor_rollout_wg.ema_loads, 0)
        self.assertEqual(resumed.actor_rollout_wg.initializations, 0)
        resumed.train_dataloader.load_state_dict.assert_not_called()

    def test_config_rejects_async_rotation_and_multiple_epochs(self):
        cfg = config(self.root)
        joint.SkillJointSyncTrainer._validate_joint_config(cfg)
        for path, value in (("actor_rollout_ref.actor.ppo_epochs", 2),
                            ("actor_rollout_ref.actor.checkpoint.async_save", True),
                            ("trainer.max_actor_ckpt_to_keep", 1), ("skill_harness.ema.update_every", 2)):
            modified = copy.deepcopy(cfg)
            OmegaConf.update(modified, path, value)
            with self.subTest(path=path), self.assertRaises(ValueError):
                joint.SkillJointSyncTrainer._validate_joint_config(modified)

    def test_peak_metrics_and_throughput_count_extra_reserved_routes(self):
        metrics = joint.normalize_actor_metrics({"metrics": {
            "perf/max_memory_allocated_gb": [1., 3.], "minimum_latency": [2., 4.], "actor/pg_loss": [2., 4.],
        }})
        self.assertEqual(metrics["perf/max_memory_allocated_gb"], 3.)
        self.assertEqual(metrics["actor/minimum_latency"], 2.)
        self.assertEqual(metrics["actor/pg_loss"], 3.)
        self.trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
        self.assertEqual(self.trainer._get_n_gpus_for_throughput(), 3)
        with self.assertRaises(FloatingPointError):
            joint.normalize_actor_metrics({"metrics": {
                "actor/pg_loss": joint.Metric(value=float("nan"), aggregation="mean"),
            }})

    def test_actual_actor_role_mapping_is_replaced_through_parent_resource_setup(self):
        with patch.object(ray, "remote", side_effect=lambda cls: cls):
            self.trainer._init_resource_pool_mgr()
        roles = [role for role in (Role.Actor, Role.ActorRollout, Role.ActorRolloutRef)
                 if role in self.trainer.role_worker_mapping]
        self.assertEqual(len(roles), 1)
        self.assertIs(self.trainer.role_worker_mapping[roles[0]], joint.JointEMAActorRolloutRefWorker)
        self.assertFalse(ray.is_initialized())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
