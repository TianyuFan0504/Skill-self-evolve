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
# ALFWorld exposes a text MDP through alfworld-step. Keep native terminal/file
# operations, but do not advertise unrelated agents, media, or external tools.
# This tool boundary does not replace per-attempt Unix-account isolation.
disabled_features=(
  multi_agent multi_agent_v2 apps plugins hooks
  browser_use browser_use_external computer_use image_generation view_image
  skill_search workspace_dependencies sleep_tool goals
)
native_policy=(
  -c "$override"
  -c allow_login_shell=false
  --enable skip_host_skill_discovery
  --enable shell_tool --enable unified_exec --enable unified_exec_tty
  -c skills.include_instructions=false
  -c 'web_search="disabled"'
  -c tools.experimental_request_user_input.enabled=false
)
for feature in "${disabled_features[@]}"; do
  native_policy+=(--disable "$feature")
done
exec "$native_bin" "${native_policy[@]}" "$@"
