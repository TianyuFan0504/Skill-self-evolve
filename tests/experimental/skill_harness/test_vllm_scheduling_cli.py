"""CPU-only checks of the real pinned vLLM CLI and backend queue setting.

Run with the base server interpreter. No engine configuration, model, CUDA
context, Ray cluster, learner or inference request is created.
"""

# ruff: noqa: E402 - establish offline CPU import provenance first.

import os
import unittest
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")

from verl.experimental.skill_harness.bootstrap import configure_runtime

configure_runtime(register_tasks=False)

import ray
import torch
import vllm
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config


class VllmSchedulingCliTest(unittest.TestCase):
    def test_actual_verl_false_flag_reaches_actual_engine_args(self):
        self.assertEqual(vllm.__version__, "0.24.0")
        flags = build_cli_args_from_config({"async_scheduling": False})
        self.assertTrue(any(flag.startswith("--no-async") for flag in flags))
        parser = FlexibleArgumentParser()
        AsyncEngineArgs.add_cli_args(parser)
        namespace = parser.parse_args(["--model", "/fixture/not-loaded", *flags])
        arguments = AsyncEngineArgs.from_cli_args(namespace)
        self.assertIs(arguments.async_scheduling, False)
        self.assertEqual(arguments.model, "/fixture/not-loaded")
        self.assertFalse(ray.is_initialized())
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_tp1_pp1_backend_queue_changes_from_two_batches_to_one(self):
        configuration = SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
            scheduler_config=SimpleNamespace(async_scheduling=True),
            use_v2_model_runner=False,
        )
        actual_property = VllmConfig.max_concurrent_batches.fget
        self.assertEqual(actual_property(configuration), 2)
        configuration.scheduler_config.async_scheduling = False
        self.assertEqual(actual_property(configuration), 1)
        self.assertFalse(ray.is_initialized())
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
