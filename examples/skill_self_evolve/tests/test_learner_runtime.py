"""CPU ordering/config checks; the separate Linux proof uses real torchdata."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import test_curate_generation as contract

PATH = Path(__file__).resolve().parents[1] / "scripts/learner_runtime.py"
SPEC = importlib.util.spec_from_file_location("_learner_runtime_cpu", PATH)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class Frame(list):
    _fingerprint = "unchanged-filtered-frame"
    column_names = ["prompt"]

    def __init__(self):
        super().__init__([{"prompt": [{"role": "user", "content": "Original text task"}]} for _ in range(3)])

    def __getitem__(self, key):
        return [row[key] for row in self] if isinstance(key, str) else super().__getitem__(key)


class DatasetBase:
    def __init__(self):
        self.serialize_dataset = False
        self.dataframe = Frame()
        self.processor = types.SimpleNamespace(label="original-processor")
        self.prompt_key, self.image_key, self.video_key, self.audio_key = "prompt", "images", "videos", "audios"

    def __len__(self):
        return 3


Dataset = type("RLHFDataset", (DatasetBase,), {"__module__": "verl.utils.dataset.rl_dataset"})


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = types.SimpleNamespace(
            data=types.SimpleNamespace(dataloader_num_workers=4),
            skill_harness=types.SimpleNamespace(artifact_root=temporary.name,
                                               joint=types.SimpleNamespace(credit_concurrency=8)),
        )
        self.method, self.calls = "fork", []

        def set_method(method, force):
            self.method = method
            self.calls.append((method, force))

        context = types.SimpleNamespace(get_start_method=lambda **kwargs: self.method, set_start_method=set_method)
        self.mp_patch = patch.object(runtime, "multiprocessing", context)
        self.mp_patch.start()
        self.addCleanup(self.mp_patch.stop)
        namespace = types.ModuleType("skill_harness_examples")
        namespace.__path__ = []
        namespace.curate_generation = contract.generation
        modules = patch.dict(sys.modules, {"skill_harness_examples": namespace})
        modules.start()
        self.addCleanup(modules.stop)
        owner = patch.object(contract.generation, "process_identity", return_value=contract.OWNER)
        owner.start()
        self.addCleanup(owner.stop)

    def trainer(self):
        ref = runtime.prepare_learner(self.config)

        def loader():
            return types.SimpleNamespace(dataset=Dataset(), sampler=object(), num_workers=4,
                                         multiprocessing_context=None, _iterator=None, next_iter_state={"cursor": 320})

        train, val = loader(), loader()
        val.dataset.processor = train.dataset.processor
        return types.SimpleNamespace(config=self.config, _learner_runtime_ref=ref, train_dataloader_it=None,
                                     train_dataloader=train, val_dataloader=val, processor=train.dataset.processor)

    def test_actual_default_observation_records_force_before_trainer_construction(self):
        ref = runtime.prepare_learner(self.config)
        value = contract.generation.read_ref(ref)
        self.assertEqual(self.calls, [("spawn", True)])
        self.assertEqual(value["previous_default_start_method"], "fork")
        self.assertEqual(value["observed_default_start_method"], "spawn")
        self.assertEqual(value["observed_before"], "trainer_construct")
        self.assertEqual(value["configured_credit_concurrency"], 8)
        self.assertEqual(value["physical_credit_probe_concurrency"], 2)

    def test_explicit_new_provider_records_cap_eight_without_changing_historical_default(self):
        ref = runtime.prepare_learner(self.config, physical_probe_concurrency=8)
        self.assertEqual(contract.generation.read_ref(ref)["physical_credit_probe_concurrency"], 8)
        self.assertEqual(runtime.PROBE_CONCURRENCY, 2)

    def test_physical_limit_outside_configured_ceiling_rejects_before_process_change(self):
        for value in (0, 9, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "physical probe"):
                runtime.prepare_learner(self.config, physical_probe_concurrency=value)
        self.assertEqual(self.calls, [])

    def test_transport_preserves_config_sampler_and_queued_checkpoint_state(self):
        trainer = self.trainer()
        config = copy.deepcopy(self.config)
        sampler, state, frame = (trainer.train_dataloader.sampler, trainer.train_dataloader.next_iter_state,
                                 trainer.train_dataloader.dataset.dataframe)
        result = runtime.prepare_loaders(trainer)
        self.assertEqual(self.config, config)
        self.assertIs(trainer.train_dataloader.sampler, sampler)
        self.assertIs(trainer.train_dataloader.next_iter_state, state)
        self.assertIs(trainer.train_dataloader.dataset.dataframe, frame)
        self.assertEqual(result["default_start_method"], "spawn")
        for name in ("train", "val"):
            self.assertFalse(result[name]["iterator_started"])
            self.assertFalse(result[name]["serialize_dataset_before"])
            self.assertTrue(result[name]["serialize_dataset_after"])
            self.assertTrue(result[name]["filtered_dataframe_unchanged"])
            self.assertIsNone(result[name]["processor_after_class"])
            self.assertTrue(result[name]["trainer_processor_unchanged"])

    def test_configured_ceiling_is_not_silently_rewritten(self):
        self.config.skill_harness.joint.credit_concurrency = 2
        with self.assertRaisesRegex(ValueError, "ceiling8"):
            runtime.prepare_learner(self.config)
        self.assertEqual(self.config.skill_harness.joint.credit_concurrency, 2)
        self.assertEqual(self.calls, [])

    def test_worker_count_zero_is_not_used_as_migration(self):
        self.config.data.dataloader_num_workers = 0
        with self.assertRaisesRegex(ValueError, "workers4"):
            runtime.prepare_learner(self.config)
        self.assertEqual(self.config.data.dataloader_num_workers, 0)

    def test_started_training_iterator_rejects_before_flag_change(self):
        trainer = self.trainer()
        trainer.train_dataloader_it = object()
        with self.assertRaisesRegex(ValueError, "unstarted training iterator"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)

    def test_started_val_iterator_rejects_before_either_dataset_changes(self):
        trainer = self.trainer()
        trainer.val_dataloader._iterator = object()
        with self.assertRaisesRegex(ValueError, "unstarted workers"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)
        self.assertFalse(trainer.val_dataloader.dataset.serialize_dataset)

    def test_explicit_fork_override_rejects(self):
        trainer = self.trainer()
        trainer.train_dataloader.multiprocessing_context = types.SimpleNamespace(get_start_method=lambda: "fork")
        with self.assertRaisesRegex(ValueError, "fork override"):
            runtime.prepare_loaders(trainer)

    def test_changed_process_default_is_observed_and_rejected(self):
        trainer = self.trainer()
        self.method = "fork"
        with self.assertRaisesRegex(ValueError, "actual process"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)

    def test_unknown_dataset_transport_is_not_assumed_pickleable(self):
        trainer = self.trainer()
        trainer.val_dataloader.dataset = object()
        with self.assertRaisesRegex(TypeError, "supported dataframe"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)

    def test_media_payload_rejects_before_either_dataset_changes(self):
        trainer = self.trainer()
        trainer.val_dataloader.dataset.dataframe.column_names = ["prompt", "images"]
        for row in trainer.val_dataloader.dataset.dataframe:
            row["images"] = ["media-path"]
        with self.assertRaisesRegex(ValueError, "text-only data"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)

    def test_custom_dataset_class_does_not_silently_detach_processor(self):
        trainer = self.trainer()
        trainer.val_dataloader.dataset = DatasetBase()
        with self.assertRaisesRegex(TypeError, "exact default"):
            runtime.prepare_loaders(trainer)
        self.assertFalse(trainer.train_dataloader.dataset.serialize_dataset)

    def test_multimodal_message_content_rejects(self):
        trainer = self.trainer()
        trainer.val_dataloader.dataset.dataframe[0]["prompt"][0]["content"] = [{"type": "image"}]
        with self.assertRaisesRegex(ValueError, "plain text"):
            runtime.prepare_loaders(trainer)


if __name__ == "__main__":
    unittest.main()
