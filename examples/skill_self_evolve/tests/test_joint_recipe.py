"""Public joint recipe delivery checks; no training, engine or GPU jobs run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "examples/skill_self_evolve"


class JointRecipeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        data, model = self.root / "data", self.root / "model"
        data.mkdir()
        model.mkdir()
        for filename in ("train.parquet", "dev.parquet", "split_manifest.json"):
            (data / filename).write_text("input fixture\n")
        task, runtime, service = [self.root / name for name in ("task.json", "runtime.json", "service.json")]
        task.write_text(json.dumps([{"name": "alfworld_skill_training"}]))
        runtime.write_text(json.dumps({"account_pool": {"factory": "fixture:Pool", "kwargs": {}}}))
        service.write_text("{}\n")
        self.environment = {
            "TRAINING_PYTHON": sys.executable,
            "MODEL_PATH": str(model), "MODEL_NAME": "fixture-model",
            "SKILL_RUN_DIR": str(self.root / "run"),
            "SKILL_DATA_DIR": str(data), "ALFWORLD_DATA_DIR": str(data),
            "ALFWORLD_TASK_CONFIG": str(task), "ALFWORLD_RUNTIME_CONFIG": str(runtime),
            "ALFWORLD_SERVER_CONFIG": str(service),
            "ALFWORLD_ENV_SERVER": str(RECIPE / "scripts/parallel_environment_server.py"),
            "CURATOR_USER": "fixture-curator",
            "CURATOR_PUBLIC_ROOT": str(self.root / "public"),
            "CURATOR_WORK_ROOT": str(self.root / "curator"),
            "CUDA_VISIBLE_DEVICES": "",
        }
        self.hook = self.root / "hook"
        self.hook.mkdir()
        (self.hook / "sitecustomize.py").write_text('''
import json, os, sys
from pathlib import Path

def guard(event, args):
    if event == "os.exec" and os.environ.get("CAPTURE_LAUNCH"):
        Path(os.environ["CAPTURE_LAUNCH"]).write_text(json.dumps(list(args[1])))
        raise SystemExit(0)
    forbidden_import = event == "import" and (
        args[0] == "verl.experimental.skill_harness.train"
        or args[0].split(".")[0] in {"alfworld", "textworld"}
    )
    if event in {"os.exec", "subprocess.Popen"} or forbidden_import:
        Path(os.environ["FORBIDDEN_ACTIVITY"]).write_text(event + ":" + str(args[0]))
        raise RuntimeError("Configuration checks must not start a training or environment runtime")

sys.addaudithook(guard)
''')
        # These packages are availability fixtures only. No SDK behavior is
        # replaced or claimed as verified by a configuration-only check.
        for name in ("uni_agent", "transfer_queue"):
            package = self.hook / name
            package.mkdir()
            (package / "__init__.py").write_text("")
        self.forbidden = self.root / "forbidden-activity"
        self.cli_environment = {
            **os.environ, **self.environment,
            "PYTHONPATH": os.pathsep.join((str(self.hook), str(ROOT))),
            "FORBIDDEN_ACTIVITY": str(self.forbidden),
        }

    def compose(self, overrides=()):
        with (
            patch.dict(os.environ, self.environment),
            initialize_config_dir(config_dir=str(RECIPE / "config"), version_base=None),
        ):
            config = compose(config_name="alfworld_joint", overrides=list(overrides))
            OmegaConf.resolve(config)
        return config

    def test_joint_configuration_resolves_actual_official_defaults_and_delayed_routes(self):
        config = self.compose(("trainer.total_training_steps=9",))
        self.assertEqual(config.algorithm.adv_estimator, "grpo")
        self.assertEqual(config.actor_rollout_ref.actor.policy_loss.loss_mode, "vanilla")
        self.assertEqual(config.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertEqual(config.trainer.total_training_steps, 9)
        self.assertEqual(config.trainer.v1.trainer_mode, "sync")
        self.assertTrue(config.trainer.v1.sampler.sync_refill_failed_groups)
        self.assertEqual(config.trainer.v1.sampler.custom_sampler.path,
                         "pkg://verl.experimental.skill_harness.group_sampler")
        self.assertEqual(config.trainer.v1.sampler.custom_sampler.name, "CompleteGroupReplayBuffer")
        self.assertTrue(config.skill_harness.ema.enabled)
        self.assertEqual(config.skill_harness.ema.update_every, 1)
        self.assertGreater(config.skill_harness.curate_loss.coefficient, 0)
        joint = config.skill_harness.joint
        self.assertGreater(joint.curator_gpus, 0)
        self.assertGreater(joint.credit_gpus, 0)
        self.assertGreaterEqual(joint.available_gpus, joint.solver_gpus + joint.curator_gpus + joint.credit_gpus)
        self.assertGreater(joint.probes_per_step, 0)
        self.assertGreater(joint.min_delay_steps, 0)
        self.assertGreater(joint.required_exposures, 0)
        self.assertGreater(joint.replay_batch_size, 0)
        self.assertEqual(joint.frozen_model_path, self.environment["MODEL_PATH"])
        self.assertEqual(config.skill_harness.joint.curator_unix_user, self.environment["CURATOR_USER"])
        self.assertNotIn("mode", config.skill_harness)
        self.assertNotIn("initial", config.skill_harness.evaluation)
        self.assertEqual(
            config.actor_rollout_ref.rollout.agent.agent_loop_manager_class,
            "verl.experimental.skill_harness.framework.SkillAwareRolloutAdapter",
        )

    def test_actual_preflight_cli_resolves_inputs_without_starting_runtime(self):
        command = [sys.executable, str(RECIPE / "scripts/preflight.py"), "trainer.total_training_steps=9"]
        result = subprocess.run(command, cwd=ROOT, env=self.cli_environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no training was started", result.stdout)
        self.assertFalse(self.forbidden.exists())
        self.assertFalse((self.root / "run").exists())
        (self.root / "data/train.parquet").unlink()
        result = subprocess.run(command, cwd=ROOT, env=self.cli_environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing training data", result.stderr)
        self.assertFalse(self.forbidden.exists())

    def test_actual_launch_chain_preserves_interpreter_and_hydra_arguments(self):
        captured = self.root / "launch.json"
        environment = {**self.cli_environment, "CAPTURE_LAUNCH": str(captured)}
        overrides = ["trainer.total_training_steps=9", "trainer.experiment_name=literal run"]
        result = subprocess.run(
            ["bash", str(RECIPE / "run.sh"), *overrides], cwd=ROOT,
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(captured.read_text()), [
            sys.executable, "-m", "verl.experimental.skill_harness.train",
            f"--config-path={RECIPE / 'config'}", "--config-name=alfworld_joint", *overrides,
        ])
        self.assertFalse(self.forbidden.exists())
        self.assertEqual(sorted(path.name for path in RECIPE.glob("run*.sh")), ["run.sh"])
        self.assertEqual(sorted(path.name for path in (RECIPE / "config").glob("*.yaml")), ["alfworld_joint.yaml"])
        self.assertEqual(list((RECIPE / "scripts").glob("train*.py")), [])
        self.assertFalse((ROOT / "skill_harness_examples/__init__.py").exists())

    def test_optional_linux_provider_imports_from_recipe_checkout_without_startup(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "from examples.skill_self_evolve.runtime.linux_accounts import LinuxAccountPool; "
                "print(LinuxAccountPool.__module__)"
            )], cwd=ROOT, env=self.cli_environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "examples.skill_self_evolve.runtime.linux_accounts")
        self.assertFalse(self.forbidden.exists())


if __name__ == "__main__":
    unittest.main()
