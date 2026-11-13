#!/usr/bin/env bash
# ALFWorld joint solver/curator recipe: GRPO, FSDP2, async vLLM, one node.
#
# Usage: bash examples/skill_self_evolve/run.sh --deployment /path/deployment.json \
#          --run-dir /path/new-run [--smoke] [Hydra overrides...]
# Add --dry-run to inspect the command plan without preparing or starting a run.
#
# Empty knobs inherit the joint YAML, deployment, and optional smoke preset.
# Set a knob here or export it to deliberately override those settings.
# Priority: YAML < deployment overrides < smoke < recipe knobs < CLI overrides.
# The comments below show inherited YAML defaults; YAML owns the defaults.
#
# The default allocation is 4 solver + 2 EMA curator + 2 frozen credit GPUs.
# PHYSICAL_GPU_COUNT is the full visible budget (8), not the solver count.
# Keep all allocated GPUs visible; the joint controller partitions the budget.
# Model, native agent, accounts, installed interpreters, and public assets come
# from the deployment JSON. This script does not prepare a Python environment.
set -euo pipefail

# ---- User-adjustable recipe knobs: empty means inherit ----
DEPLOYMENT=${DEPLOYMENT-}       # Optional deployment JSON; CLI wins.
RUN_DIR=${RUN_DIR-}             # Optional fresh output directory; CLI wins.
MODEL_PATH=${MODEL_PATH-}       # Deployment's local model weights.

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE-}                 # 16
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE-}             # 16
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU-} # 1
ACTOR_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_LOGPROB_MICRO_BATCH_SIZE_PER_GPU-} # 1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU-} # 1
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH-}               # 4096
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH-}           # 32768
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH-}                 # 36864

LEARNING_RATE=${LEARNING_RATE-}       # 1e-6
LR_WARMUP_STEPS=${LR_WARMUP_STEPS-}   # 2
KL_LOSS_COEF=${KL_LOSS_COEF-}         # 0.001
ENTROPY_COEFF=${ENTROPY_COEFF-}       # 0.0

ROLLOUT_N=${ROLLOUT_N-}                       # 4
ROLLOUT_TP=${ROLLOUT_TP-}                     # 1
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION-} # 0.55
ROLLOUT_CONCURRENCY=${ROLLOUT_CONCURRENCY-}   # 64 native sessions
MAX_NUM_SEQS=${MAX_NUM_SEQS-}                 # 32 vLLM sequences

TRAINING_GPUS=${TRAINING_GPUS-}           # 4 solver GPUs
CURATOR_GPUS=${CURATOR_GPUS-}             # 2 EMA curator GPUs
CREDIT_GPUS=${CREDIT_GPUS-}               # 2 frozen credit GPUs
PHYSICAL_GPU_COUNT=${PHYSICAL_GPU_COUNT-} # 8 total visible GPUs
EMA_DECAY=${EMA_DECAY-}                   # 0.95
CURATE_LOSS_COEFFICIENT=${CURATE_LOSS_COEFFICIENT-} # 0.1
CURATOR_CONCURRENCY=${CURATOR_CONCURRENCY-} # 8
CREDIT_CONCURRENCY=${CREDIT_CONCURRENCY-}   # 8
PROBES_PER_STEP=${PROBES_PER_STEP-}         # 32
REQUIRED_EXPOSURES=${REQUIRED_EXPOSURES-}   # 2
MIN_DELAY_STEPS=${MIN_DELAY_STEPS-}         # 2
REPLAY_BATCH_SIZE=${REPLAY_BATCH_SIZE-}     # 32
MAX_REPLAY_LAG_STEPS=${MAX_REPLAY_LAG_STEPS-} # 64

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS-} # 64 joint optimizer updates
TOTAL_EPOCHS=${TOTAL_EPOCHS-}               # 4; step limit also applies
SAVE_FREQUENCY=${SAVE_FREQUENCY-}           # 10 updates
DEV_EVAL_FREQUENCY=${DEV_EVAL_FREQUENCY-}   # 4 updates
PROJECT_NAME=${PROJECT_NAME-}             # alfworld_skill_self_evolve
EXPERIMENT_NAME=${EXPERIMENT_NAME-}         # alfworld_joint
LOGGERS=${LOGGERS-}                       # '["console","tensorboard"]'

# Recipe knobs travel as explicit Hydra arguments, not environment defaults.
# In particular, an exported empty value must not suppress an oc.env default.
export -n MODEL_PATH TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU \
    ACTOR_LOGPROB_MICRO_BATCH_SIZE_PER_GPU REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LENGTH LEARNING_RATE LR_WARMUP_STEPS \
    KL_LOSS_COEF ENTROPY_COEFF ROLLOUT_N ROLLOUT_TP GPU_MEMORY_UTILIZATION \
    ROLLOUT_CONCURRENCY MAX_NUM_SEQS TRAINING_GPUS CURATOR_GPUS CREDIT_GPUS \
    PHYSICAL_GPU_COUNT EMA_DECAY CURATE_LOSS_COEFFICIENT CURATOR_CONCURRENCY \
    CREDIT_CONCURRENCY PROBES_PER_STEP REQUIRED_EXPOSURES MIN_DELAY_STEPS REPLAY_BATCH_SIZE \
    MAX_REPLAY_LAG_STEPS TOTAL_TRAINING_STEPS TOTAL_EPOCHS SAVE_FREQUENCY DEV_EVAL_FREQUENCY \
    PROJECT_NAME EXPERIMENT_NAME LOGGERS

# ---- Parameter groups; only nonempty values become recipe overrides ----
DATA=(
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
)

MODEL=(
    "actor_rollout_ref.model.path=${MODEL_PATH}"
)

ACTOR=(
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    "actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
    "actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
)

ROLLOUT=(
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ACTOR_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH}"
    "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${ROLLOUT_CONCURRENCY}"
)

REF=(
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
)

JOINT=(
    "skill_harness.joint.available_gpus=${PHYSICAL_GPU_COUNT}"
    "skill_harness.joint.curator_gpus=${CURATOR_GPUS}"
    "skill_harness.joint.credit_gpus=${CREDIT_GPUS}"
    "skill_harness.ema.decay=${EMA_DECAY}"
    "skill_harness.curate_loss.coefficient=${CURATE_LOSS_COEFFICIENT}"
    "skill_harness.joint.curator_concurrency=${CURATOR_CONCURRENCY}"
    "skill_harness.joint.credit_concurrency=${CREDIT_CONCURRENCY}"
    "skill_harness.joint.probes_per_step=${PROBES_PER_STEP}"
    "skill_harness.joint.required_exposures=${REQUIRED_EXPOSURES}"
    "skill_harness.joint.min_delay_steps=${MIN_DELAY_STEPS}"
    "skill_harness.joint.replay_batch_size=${REPLAY_BATCH_SIZE}"
    "skill_harness.joint.max_replay_lag_steps=${MAX_REPLAY_LAG_STEPS}"
)

TRAINER=(
    "trainer.n_gpus_per_node=${TRAINING_GPUS}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.save_freq=${SAVE_FREQUENCY}"
    "trainer.test_freq=${DEV_EVAL_FREQUENCY}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.logger=${LOGGERS}"
)

# ---- Launch one supervisor; it owns preparation, environment, and training ----
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# This bootstrap needs only Python's standard library. The deployment selects
# the installed interpreter for data preparation, preflight, and training.
COMMAND=("${TRAINING_PYTHON:-python3}" "$recipe_dir/scripts/launch.py")
if [[ -n "$DEPLOYMENT" ]]; then
    COMMAND+=(--deployment "$DEPLOYMENT")
fi
if [[ -n "$RUN_DIR" ]]; then
    COMMAND+=(--run-dir "$RUN_DIR")
fi
for setting in "${DATA[@]}" "${MODEL[@]}" "${ACTOR[@]}" \
    "${ROLLOUT[@]}" "${REF[@]}" "${JOINT[@]}" "${TRAINER[@]}"; do
    if [[ -n "${setting#*=}" ]]; then
        COMMAND+=("--recipe-override=$setting")
    fi
done
exec "${COMMAND[@]}" "$@"
