#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${TRAINING_PYTHON:-python}" "$recipe_dir/scripts/launch.py" "$@"
