"""Execute the shell recipe with a recorder; no training or engine is imported."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RECIPE = Path(__file__).resolve().parents[1]
RUN_SCRIPT = RECIPE / "run.sh"

# The public experiment knobs must become explicit Hydra overrides even when a
# deployment environment supplies different values later in the launch process.
ENVIRONMENT_KEYS = {
    "MODEL_PATH": "actor_rollout_ref.model.path",
    "TRAIN_BATCH_SIZE": "data.train_batch_size",
    "PPO_MINI_BATCH_SIZE": "actor_rollout_ref.actor.ppo_mini_batch_size",
    "PPO_MICRO_BATCH_SIZE_PER_GPU": "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
    "ACTOR_LOGPROB_MICRO_BATCH_SIZE_PER_GPU": "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
    "REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU": "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
    "MAX_PROMPT_LENGTH": "data.max_prompt_length",
    "MAX_RESPONSE_LENGTH": "data.max_response_length",
    "MAX_MODEL_LENGTH": "actor_rollout_ref.rollout.max_model_len",
    "LEARNING_RATE": "actor_rollout_ref.actor.optim.lr",
    "LR_WARMUP_STEPS": "actor_rollout_ref.actor.optim.lr_warmup_steps",
    "KL_LOSS_COEF": "actor_rollout_ref.actor.kl_loss_coef",
    "ENTROPY_COEFF": "actor_rollout_ref.actor.entropy_coeff",
    "ROLLOUT_N": "actor_rollout_ref.rollout.n",
    "ROLLOUT_TP": "actor_rollout_ref.rollout.tensor_model_parallel_size",
    "GPU_MEMORY_UTILIZATION": "actor_rollout_ref.rollout.gpu_memory_utilization",
    "ROLLOUT_CONCURRENCY": (
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions"
    ),
    "MAX_NUM_SEQS": "actor_rollout_ref.rollout.max_num_seqs",
    "TRAINING_GPUS": "trainer.n_gpus_per_node",
    "CURATOR_GPUS": "skill_harness.joint.curator_gpus",
    "CREDIT_GPUS": "skill_harness.joint.credit_gpus",
    "PHYSICAL_GPU_COUNT": "skill_harness.joint.available_gpus",
    "EMA_DECAY": "skill_harness.ema.decay",
    "CURATE_LOSS_COEFFICIENT": "skill_harness.curate_loss.coefficient",
    "CURATOR_CONCURRENCY": "skill_harness.joint.curator_concurrency",
    "CREDIT_CONCURRENCY": "skill_harness.joint.credit_concurrency",
    "PROBES_PER_STEP": "skill_harness.joint.probes_per_step",
    "REQUIRED_EXPOSURES": "skill_harness.joint.required_exposures",
    "MIN_DELAY_STEPS": "skill_harness.joint.min_delay_steps",
    "REPLAY_BATCH_SIZE": "skill_harness.joint.replay_batch_size",
    "MAX_REPLAY_LAG_STEPS": "skill_harness.joint.max_replay_lag_steps",
    "TOTAL_TRAINING_STEPS": "trainer.total_training_steps",
    "TOTAL_EPOCHS": "trainer.total_epochs",
    "SAVE_FREQUENCY": "trainer.save_freq",
    "DEV_EVAL_FREQUENCY": "trainer.test_freq",
    "PROJECT_NAME": "trainer.project_name",
    "EXPERIMENT_NAME": "trainer.experiment_name",
    "LOGGERS": "trainer.logger",
}


class RunRecipeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="shell recipe ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture = self.root / "captured arguments.json"
        self.environment_capture = self.root / "captured environment.json"
        recorder = self.root / "record arguments.py"
        recorder.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURE_LAUNCH']).write_text(json.dumps(sys.argv[1:]))\n"
            "keys = ('MAX_PROMPT_LENGTH', 'SAVE_FREQUENCY', 'TOTAL_TRAINING_STEPS', 'CUDA_VISIBLE_DEVICES')\n"
            "captured_environment = {key: os.environ.get(key) for key in keys}\n"
            "Path(os.environ['CAPTURE_ENVIRONMENT']).write_text(json.dumps(captured_environment))\n"
            "raise SystemExit(int(os.environ.get('RECORDER_EXIT_CODE', '0')))\n"
        )
        # The bootstrap path contains spaces; its target uses only the standard
        # library and records launch.py's arguments without executing that file.
        bootstrap = self.root / "bootstrap python"
        bootstrap.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(recorder))} "$@"\n')
        bootstrap.chmod(0o700)
        self.environment = {
            key: value for key, value in os.environ.items() if key not in {*ENVIRONMENT_KEYS, "DEPLOYMENT", "RUN_DIR"}
        }
        self.environment.update(
            {
                "TRAINING_PYTHON": str(bootstrap),
                "CAPTURE_LAUNCH": str(self.capture),
                "CAPTURE_ENVIRONMENT": str(self.environment_capture),
            }
        )

    def run_recipe(self, *arguments, environment=None, script=RUN_SCRIPT):
        result = subprocess.run(
            ["bash", str(script), *arguments],
            cwd=self.root,
            env={**self.environment, **(environment or {})},
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertTrue(self.capture.is_file(), result.stderr)
        captured = json.loads(self.capture.read_text())
        self.assertEqual(captured[0], str(script.parent / "scripts/launch.py"))
        return result, captured[1:]

    def test_unset_knobs_preserve_smoke_and_user_arguments_without_recipe_defaults(self):
        arguments = ("--smoke", "trainer.total_training_steps=11", "trainer.experiment_name=explicit run")
        result, captured = self.run_recipe(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured, list(arguments))

    def test_every_explicit_environment_knob_becomes_one_recipe_override_before_cli(self):
        environment = {key: str(index + 1) for index, key in enumerate(ENVIRONMENT_KEYS)}
        environment.update(
            MODEL_PATH="/models/a model",
            PROJECT_NAME="a project",
            EXPERIMENT_NAME="an experiment",
            LOGGERS='["console","tensorboard"]',
        )
        cli = ("--smoke", "trainer.total_training_steps=97", "trainer.experiment_name=CLI wins")
        result, captured = self.run_recipe(*cli, environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured[-len(cli) :], list(cli))
        self.assertCountEqual(
            captured[: -len(cli)],
            [f"--recipe-override={hydra_key}={environment[name]}" for name, hydra_key in ENVIRONMENT_KEYS.items()],
        )

    def test_empty_environment_knobs_inherit_deployment_and_smoke_defaults(self):
        environment = {**dict.fromkeys(ENVIRONMENT_KEYS, ""), "CUDA_VISIBLE_DEVICES": ""}
        result, captured = self.run_recipe("--smoke", environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured, ["--smoke"])
        self.assertEqual(
            json.loads(self.environment_capture.read_text()),
            {
                "MAX_PROMPT_LENGTH": None,
                "SAVE_FREQUENCY": None,
                "TOTAL_TRAINING_STEPS": None,
                "CUDA_VISIBLE_DEVICES": "",
            },
        )

    def test_editing_a_recipe_knob_creates_an_explicit_override(self):
        script = self.root / "edited recipe.sh"
        source = RUN_SCRIPT.read_text()
        original = "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE-}"
        self.assertEqual(source.count(original), 1)
        script.write_text(source.replace(original, "TRAIN_BATCH_SIZE=23", 1))
        result, captured = self.run_recipe("--smoke", script=script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(captured, ["--recipe-override=data.train_batch_size=23", "--smoke"])

    def test_optional_deployment_paths_precede_explicit_cli_flags(self):
        environment = {"DEPLOYMENT": "/profiles/environment profile.json", "RUN_DIR": "/runs/environment run"}
        cli = ("--deployment", "/profiles/explicit profile.json", "--run-dir", "/runs/explicit run", "--dry-run")
        result, captured = self.run_recipe(*cli, environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            captured,
            ["--deployment", environment["DEPLOYMENT"], "--run-dir", environment["RUN_DIR"], *cli],
        )

    def test_shell_metacharacters_and_spaces_remain_literal_arguments(self):
        marker = self.root / "unexpected execution"
        substitution = f"$(touch {shlex.quote(str(marker))})"
        model = f"/models/model with spaces {substitution}"
        experiment = f"trainer.experiment_name=literal {substitution}; `false`"
        result, captured = self.run_recipe(
            experiment,
            "trainer.logger=[console, tensorboard]",
            environment={"MODEL_PATH": model},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            captured,
            [
                f"--recipe-override=actor_rollout_ref.model.path={model}",
                experiment,
                "trainer.logger=[console, tensorboard]",
            ],
        )
        self.assertFalse(marker.exists())

    def test_bootstrap_failure_exit_status_is_preserved(self):
        result, captured = self.run_recipe("--help", environment={"RECORDER_EXIT_CODE": "37"})
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual(captured, ["--help"])


if __name__ == "__main__":
    unittest.main()
