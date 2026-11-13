#!/usr/bin/env bash
set -euo pipefail

recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# The bootstrap uses only the standard library. The deployment explicitly
# selects the installed interpreter used by preparation and training.
exec "${TRAINING_PYTHON:-python3}" "$recipe_dir/scripts/launch.py" "$@"
