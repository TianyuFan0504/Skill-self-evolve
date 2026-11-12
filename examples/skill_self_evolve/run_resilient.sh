#!/usr/bin/env bash
# One-command joint training using the existing offline launcher infrastructure.
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(cd -- "$recipe_dir/../.." && pwd)
export VERL_SOURCE_ROOT="${VERL_SOURCE_ROOT:-$source_root}"
export ALFWORLD_BENCHMARK_ROOT="${ALFWORLD_BENCHMARK_ROOT:-/root/nas_chaohuang/benchmarks/alfworld}"
export UNI_AGENT_SOURCE_ROOT="${UNI_AGENT_SOURCE_ROOT:-$ALFWORLD_BENCHMARK_ROOT/uni-agent}"
export PYTHONPATH="$VERL_SOURCE_ROOT:$UNI_AGENT_SOURCE_ROOT"
exec "${TRAINING_PYTHON:-/opt/conda/bin/python}" "$recipe_dir/scripts/launch_resilient.py" "$@"
