"""Regression tests for the structured Hydra runtime-environment failure."""

import importlib.util
import unittest
from pathlib import Path

from omegaconf import OmegaConf

PATH = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/bootstrap.py"
SPEC = importlib.util.spec_from_file_location("skill_runtime_bootstrap", PATH)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class RuntimeEnvironmentMergeTest(unittest.TestCase):
    def test_struct_empty_runtime_accepts_worker_setup_and_environment(self):
        config = OmegaConf.create({"ray_kwargs": {"ray_init": {"runtime_env": {}}}})
        OmegaConf.set_struct(config, True)
        bootstrap.apply_runtime_environment(
            config, {"env_vars": {"LOCAL_ONLY": "1"}, "worker_process_setup_hook": "package.worker"}
        )
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.env_vars.LOCAL_ONLY, "1")
        self.assertEqual(config.ray_kwargs.ray_init.runtime_env.worker_process_setup_hook, "package.worker")

    def test_inherited_environment_is_preserved_without_struct_mutation(self):
        config = OmegaConf.create({"ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"OLD": "x"}}}}})
        OmegaConf.set_struct(config, True)
        bootstrap.apply_runtime_environment(config, {"env_vars": {"NEW": "y"}})
        self.assertEqual(dict(config.ray_kwargs.ray_init.runtime_env.env_vars), {"OLD": "x", "NEW": "y"})

    def test_remote_package_resolution_is_rejected(self):
        config = OmegaConf.create({"ray_kwargs": {"ray_init": {"runtime_env": {"pip": ["external-package"]}}}})
        with self.assertRaises(ValueError):
            bootstrap.apply_runtime_environment(config, {})


if __name__ == "__main__":
    unittest.main()
