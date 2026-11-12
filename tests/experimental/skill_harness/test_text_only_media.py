"""Reject media before text field selection; CPU fixtures are not training evidence."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(os.environ.get("SKILL_TEXT_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
DIRECTORY = ROOT / "verl/experimental/skill_harness"
FRAMEWORK_PATH = Path(os.environ.get("SKILL_TEXT_FRAMEWORK_PATH", DIRECTORY / "framework.py"))
TRAINER_PATH = Path(os.environ.get("SKILL_TEXT_TRAINER_PATH", DIRECTORY / "joint_trainer.py"))
PARENT_PATH = Path(
    os.environ.get(
        "SKILL_UNI_FRAMEWORK_PATH", "/root/nas_chaohuang/benchmarks/alfworld/uni-agent/uni_agent/framework/framework.py"
    )
)


def load_framework(parent):
    package = "_text_only_media_cpu"
    module = types.ModuleType(package)
    module.__path__ = [str(DIRECTORY)]
    sys.modules[package] = module
    entry = types.ModuleType("uni_agent.framework.entry")
    entry.AgentFrameworkRolloutAdapter = object
    upstream = types.ModuleType("uni_agent.framework.framework")
    upstream.GatewayAgentFramework = parent
    specification = importlib.util.spec_from_file_location(package + ".framework", FRAMEWORK_PATH)
    result = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, {"uni_agent.framework.entry": entry, "uni_agent.framework.framework": upstream}):
        specification.loader.exec_module(result)
    return result


class ParentBoundary:
    def _trajectory_to_tq_field_and_tag(self, **_):
        self.calls += 1
        return self.native_field, self.native_tag


class TextOnlyBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.module = load_framework(ParentBoundary)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.artifact_root = Path(self.temporary.name)

    def test_none_and_empty_text_carriers_are_accepted(self):
        for value in (None, {}, [], (), [None, {}, []], {"images": [], "videos": None}):
            with self.subTest(value=value):
                self.module.require_text_only_media(value, source="CPU text fixture")

    def test_unknown_media_bytes_paths_and_objects_fail_closed(self):
        for value in (object(), b"PNG fixture", "data:image/png;base64,fixture", {"images": [object()]}):
            with self.subTest(type=type(value).__name__), self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
                self.module.require_text_only_media(value, source="CPU media fixture")

    def serializer(self, *, media=None, task="alfworld_skill_training", returned=None):
        instance = self.module.SkillAwareAgentFramework()
        instance._skill_loading = {"enabled": False}
        instance._text_protocol_artifact_root = self.artifact_root
        instance.calls = 0
        instance.native_field = returned if returned is not None else {"multi_modal_inputs": {}, "opaque": object()}
        instance.native_tag = {"global_steps": 7}
        arguments = {
            "trajectory": SimpleNamespace(multi_modal_data=media),
            "sample_fields": {"tools_kwargs": {"task": {"name": task}}},
            "session_index": 0,
            "global_steps": 7,
            "uid": "CPU fixture",
        }
        return instance, arguments

    def test_raw_media_rejected_before_parent_and_returned_media_before_publish(self):
        instance, arguments = self.serializer(media={"images": [object()]})
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            instance._trajectory_to_tq_field_and_tag(**arguments)
        self.assertEqual(instance.calls, 0)
        instance, arguments = self.serializer(returned={"multi_modal_inputs": {"pixel_values": object()}})
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            instance._trajectory_to_tq_field_and_tag(**arguments)
        self.assertEqual(instance.calls, 1)

    def test_text_return_objects_are_unchanged_and_other_benchmarks_are_not_disabled(self):
        instance, arguments = self.serializer()
        field, tag = instance._trajectory_to_tq_field_and_tag(**arguments)
        self.assertIs(field, instance.native_field)
        self.assertIs(tag, instance.native_tag)
        instance, arguments = self.serializer(
            task="another_multimodal_task",
            media={"images": [object()]},
            returned={"multi_modal_inputs": {"pixel_values": object()}},
        )
        self.assertIs(instance._trajectory_to_tq_field_and_tag(**arguments)[0], instance.native_field)

    def queue(self, *, payload=None, unfinalized=False):
        payload = [{"pixel_values": object()}] if payload is None else payload
        description = {
            "text": {"fields": ["input_ids"], "unfinalized": []},
            "media": {
                "fields": ["input_ids", "multi_modal_inputs"],
                "unfinalized": ["multi_modal_inputs"] if unfinalized else [],
            },
        }
        queue = SimpleNamespace(
            kv_describe=Mock(return_value=description), kv_batch_get=Mock(return_value={"multi_modal_inputs": payload})
        )
        return queue, SimpleNamespace(keys=["text", "media"], partition_id="CPU", extra_info={})

    def test_sparse_optional_media_column_is_checked_on_its_actual_key_subset(self):
        queue, batch = self.queue()
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            self.module.guard_text_only_tq_batch(queue, batch)
        self.assertEqual(
            queue.kv_batch_get.call_args.kwargs,
            {"keys": ["media"], "partition_id": "CPU", "select_fields": ["multi_modal_inputs"]},
        )
        queue, batch = self.queue(payload=[{}, None])
        self.module.guard_text_only_tq_batch(queue, batch)

    def test_unfinalized_missing_or_unreadable_media_is_not_silently_ignored(self):
        queue, batch = self.queue(unfinalized=True)
        with self.assertRaisesRegex(ValueError, "unfinalized"):
            self.module.guard_text_only_tq_batch(queue, batch)
        queue.kv_batch_get.assert_not_called()
        queue, batch = self.queue()
        queue.kv_batch_get.return_value = {}
        with self.assertRaisesRegex(ValueError, "not returned"):
            self.module.guard_text_only_tq_batch(queue, batch)
        queue.kv_batch_get.side_effect = OSError("CPU storage failure")
        with self.assertRaises(OSError):
            self.module.guard_text_only_tq_batch(queue, batch)

    def materializer(self, queue):
        tree = ast.parse(TRAINER_PATH.read_text())
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkillJointSyncTrainer"
        )
        node = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_materialize_solver"
        )
        scope = {
            "tq": queue,
            "guard_text_only_tq_batch": self.module.guard_text_only_tq_batch,
            "TEXT_ONLY_MEDIA_FIELDS": self.module.TEXT_ONLY_MEDIA_FIELDS,
            "require_text_only_media": self.module.require_text_only_media,
            "required_solver_fields": lambda: ("input_ids", "old_log_probs"),
            "tu": SimpleNamespace(
                assign_non_tensor=lambda *_args, **_kwargs: None,
                assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value),
            ),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER_PATH), "exec"), scope)
        actor = SimpleNamespace(
            use_kl_loss=False, calculate_entropy=False, entropy_coeff=0, data_loader_seed=0, shuffle=False
        )
        holder = SimpleNamespace(
            config=SimpleNamespace(
                skill_harness=SimpleNamespace(artifact_root=str(self.artifact_root)),
                actor_rollout_ref=SimpleNamespace(
                    actor=actor, rollout=SimpleNamespace(calculate_log_probs=False), model={}
                ),
                algorithm={},
            )
        )
        return scope["_materialize_solver"], holder

    def actor_update(self):
        tree = ast.parse(TRAINER_PATH.read_text())
        cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SkillJointSyncTrainer"
        )
        node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_update_actor")
        scope = {
            "assert_no_nontext_incidents": self.module.assert_no_nontext_incidents,
            # Filesystem incidents have their own actual-host-guard regression.
            "assert_no_isolation_incidents": Mock(),
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER_PATH), "exec"), scope)
        holder = SimpleNamespace(
            _joint_lock=nullcontext(),
            config=SimpleNamespace(skill_harness=SimpleNamespace(artifact_root=str(self.artifact_root))),
            _joint_controller=SimpleNamespace(prepare_update=Mock()),
            _materialize_solver=Mock(return_value={"input_ids": [101, 102]}),
            global_steps=1,
            _await=lambda value: value,
            actor_rollout_wg=SimpleNamespace(update_actor=Mock()),
        )
        return scope["_update_actor"], holder

    def test_rejected_media_case_stays_run_fatal_after_runner_drops_it_from_tq(self):
        instance, arguments = self.serializer(media={"images": [object()]})
        untrusted = self.artifact_root / "model-chosen-root"
        arguments["sample_fields"]["tools_kwargs"]["task"]["metadata"] = {"run_dir": str(untrusted)}
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            instance._trajectory_to_tq_field_and_tag(**arguments)
        incidents = list((self.artifact_root / "protocol_incidents/unsupported_media").glob("*.json"))
        self.assertEqual(len(incidents), 1)
        self.assertFalse(untrusted.exists())
        # Simulate the official runner dropping that case. The surviving TQ
        # batch contains only text, so the media-column check alone passes.
        queue = SimpleNamespace(kv_describe=Mock(return_value={"text": {"fields": ["input_ids"], "unfinalized": []}}))
        batch = SimpleNamespace(keys=["text"], partition_id="CPU", extra_info={})
        self.module.guard_text_only_tq_batch(queue, batch)
        function, holder = self.actor_update()
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            function(holder, batch, {})
        holder.actor_rollout_wg.update_actor.assert_not_called()
        holder._joint_controller.prepare_update.assert_not_called()

    def test_incident_during_prepare_is_rechecked_immediately_before_optimizer_rpc(self):
        function, holder = self.actor_update()

        def prepare(*_):
            self.module.record_nontext_incident(self.artifact_root, source="CPU concurrent protocol fixture")
            return SimpleNamespace(data="must not update")

        holder._joint_controller.prepare_update.side_effect = prepare
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            function(holder, SimpleNamespace(), {})
        holder.actor_rollout_wg.update_actor.assert_not_called()

    def test_incomplete_incident_write_is_still_sticky_not_a_text_batch(self):
        with patch.object(self.module.os, "fsync", side_effect=OSError("CPU durability failure")):
            with self.assertRaises(OSError):
                self.module.record_nontext_incident(self.artifact_root, source="CPU interrupted incident")
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            self.module.assert_no_nontext_incidents(self.artifact_root)

    def test_actual_materializer_rejects_before_text_drop_and_omits_only_proved_empty_carriers(self):
        queue, batch = self.queue()
        function, holder = self.materializer(queue)
        with self.assertRaises(self.module.NonTextAlfworldTrajectoryError):
            function(holder, batch)
        self.assertEqual(queue.kv_batch_get.call_count, 1)  # No text-only fetch or optimizer reached.
        queue, batch = self.queue(payload=[{}])
        batch.extra_info = None
        native_ids, native_mu = [101, 102, 201], [-0.25]
        native = {"input_ids": native_ids, "old_log_probs": native_mu, "multi_modal_inputs": {}}
        queue.kv_batch_get.side_effect = lambda **kwargs: (
            {"multi_modal_inputs": [{}]} if kwargs["select_fields"] == ["multi_modal_inputs"] else native
        )
        function, holder = self.materializer(queue)
        output = function(holder, batch)
        self.assertIs(output["input_ids"], native_ids)
        self.assertIs(output["old_log_probs"], native_mu)
        self.assertNotIn("multi_modal_inputs", output)


class ActualTensorAndSerializerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from tensordict import TensorDict
            from tensordict.tensorclass import NonTensorData, NonTensorStack
        except ImportError:
            raise unittest.SkipTest("Run this class in the existing CPU Torch/TensorDict environment") from None
        cls.torch, cls.NonTensorData, cls.NonTensorStack = torch, NonTensorData, NonTensorStack
        cls.TensorDict = TensorDict

    def test_real_tensors_and_tq_non_tensor_carriers_do_not_confuse_black_images_with_empty_text(self):
        module = load_framework(ParentBoundary)
        for value in (
            self.torch.empty(0),
            self.NonTensorData({}),
            self.NonTensorStack.from_list([self.NonTensorData({}), self.NonTensorData(None)]),
            {"mm_token_type_ids": self.torch.zeros((1, 3), dtype=self.torch.long)},
        ):
            module.require_text_only_media(value, source="real CPU empty carrier")
        empty_rows = self.NonTensorStack.from_list([self.NonTensorData({}), self.NonTensorData(None)])
        actual_td = self.TensorDict({"multi_modal_inputs": empty_rows}, batch_size=[2])
        module.require_text_only_media(actual_td["multi_modal_inputs"], source="actual TensorDict field access")
        for value in (
            {"pixel_values": self.torch.zeros(1, 3, 32, 32)},
            {"image_grid_thw": self.torch.tensor([[1, 2, 2]])},
            {"mm_token_type_ids": self.torch.tensor([[0, 1, 0]])},
            self.NonTensorStack.from_list(
                [self.NonTensorData({}), self.NonTensorData({"pixel_values": self.torch.zeros(1, 3, 32, 32)})]
            ),
        ):
            with self.assertRaises(module.NonTextAlfworldTrajectoryError):
                module.require_text_only_media(value, source="real CPU media fixture")

    def test_actual_official_serializer_preserves_text_fields_and_exposes_pixels_to_the_return_guard(self):
        if not PARENT_PATH.is_file():
            self.skipTest("Supply the real UniAgent framework source for the official serializer check")
        tree = ast.parse(PARENT_PATH.read_text())
        parent = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GatewayAgentFramework"
        )
        method = next(
            node
            for node in parent.body
            if isinstance(node, ast.FunctionDef) and node.name == "_trajectory_to_tq_field_and_tag"
        )
        payload = {"value": {}}
        scope = {
            "torch": self.torch,
            "compute_multi_modal_inputs": lambda *_args: payload["value"],
            "compute_position_id_with_mask": lambda mask: self.torch.clip(mask.cumsum(-1) - 1, min=0),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(PARENT_PATH), "exec"), scope)
        actual_parent = type(
            "ActualOfficialSerializer",
            (),
            {
                "_trajectory_to_tq_field_and_tag": scope[method.name],
                "_resolve_processor": lambda *_: None,
                "_extra_tags": lambda *_: {},
            },
        )
        module = load_framework(actual_parent)
        trajectory = SimpleNamespace(
            prompt_ids=[101, 102],
            response_ids=[201, 202],
            response_mask=[1, 0],
            response_logprobs=[-0.25, 0.0],
            finished=True,
            extra_fields={},
            multi_modal_data=None,
            routed_experts=None,
            reward_score=1.0,
            reward_metrics={"acc": 1.0},
            num_turns=1,
        )
        kwargs = {
            "trajectory": trajectory,
            "sample_fields": {"tools_kwargs": {"task": {"name": "alfworld_skill_training"}}},
            "session_index": 0,
            "global_steps": 7,
            "uid": "real-serializer-CPU-fixture",
        }
        original, guarded = actual_parent(), module.SkillAwareAgentFramework()
        original._mask_unfinished_episode = guarded._mask_unfinished_episode = False
        guarded._skill_loading = {"enabled": False}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        guarded._text_protocol_artifact_root = Path(temporary.name)
        before, before_tags = original._trajectory_to_tq_field_and_tag(**kwargs)
        after, after_tags = guarded._trajectory_to_tq_field_and_tag(**kwargs)
        self.assertEqual(before_tags, after_tags)
        for key in before:
            if isinstance(before[key], self.torch.Tensor):
                self.assertTrue(self.torch.equal(before[key], after[key]), key)
            else:
                self.assertEqual(before[key], after[key])
        pixels = self.torch.zeros((1, 3, 32, 32))  # A CPU interface fixture, never model/training evidence.
        payload["value"] = {"pixel_values": pixels}
        official_field, _ = original._trajectory_to_tq_field_and_tag(**kwargs)
        self.assertIs(official_field["multi_modal_inputs"]["pixel_values"], pixels)
        with self.assertRaises(module.NonTextAlfworldTrajectoryError):
            guarded._trajectory_to_tq_field_and_tag(**kwargs)


if __name__ == "__main__":
    unittest.main()
