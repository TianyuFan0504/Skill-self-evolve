#!/usr/bin/env bash
# Keep the upstream Uni-Agent CodexAgent unchanged while controlling instructions.
set -euo pipefail

native_bin="${CODEX_NATIVE_BIN:-/usr/local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex}"
instructions_file="${ALFWORLD_SOLVER_INSTRUCTIONS_FILE:-/opt/alfworld-tools/skill_self_evolve/solver_system.md}"

if [[ ! -x "$native_bin" || ! -r "$instructions_file" ]]; then
  printf '%s\n' "Native Codex or its ALFWorld instruction file is unavailable." >&2
  exit 78
fi

override=$(/usr/bin/python3 -c 'import json,sys; print("model_instructions_file=" + json.dumps(sys.argv[1]))' "$instructions_file")
exec "$native_bin" -c "$override" -c allow_login_shell=false --enable skip_host_skill_discovery -c skills.include_instructions=false "$@"
