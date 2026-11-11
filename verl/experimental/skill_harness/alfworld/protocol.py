"""Dependency-free identities and public-state boundaries for ALFWorld tasks."""

from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def new_attempt_directory(run_dir: Path, game_id: str) -> tuple[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", game_id):
        raise ValueError("game_id must be a safe single path component")
    parent = run_dir.resolve() / "episodes" / game_id
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for _ in range(8):
        attempt = uuid.uuid4().hex
        output = parent / ("attempt_" + attempt)
        try:
            output.mkdir(mode=0o700)
            return attempt, output
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate a unique ALFWorld episode directory")


def validate_loopback_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("ALFWorld service must be a plain loopback HTTP endpoint")
    return value.rstrip("/")


def validate_game_reference(metadata: dict[str, Any], row: dict[str, Any], digest: str) -> None:
    for key in ("game_id", "relative_game_path", "game_sha256", "split"):
        if metadata.get(key) != row.get(key):
            raise ValueError(f"ALFWorld task identity mismatch: {key}")
    if metadata.get("catalog_digest") != digest:
        raise ValueError("ALFWorld task catalog digest mismatch")


def authoritative_reward(engine: dict[str, Any]) -> float:
    won = engine.get("won")
    if type(won) is not bool:
        raise ValueError("The environment must return a boolean won outcome")
    reward = float(won)
    if "reward" in engine:
        returned = float(engine["reward"])
        if not math.isfinite(returned) or returned != reward:
            raise ValueError("Environment reward disagrees with its won outcome")
    return reward


def public_state(state: dict[str, Any], *, steps: int, max_steps: int, won: bool, done: bool) -> dict[str, Any]:
    """Explicit allowlist: expert plans, filenames and internal state stay private."""
    observation = state.get("feedback", state.get("observation", ""))
    raw_commands = state.get("admissible_commands", [])
    if not isinstance(observation, str) or not isinstance(raw_commands, (list, tuple)):
        raise ValueError("ALFWorld public observations and command lists must be text-only")
    if not all(isinstance(command, str) for command in raw_commands):
        raise ValueError("ALFWorld admissible commands must contain strings only")
    commands = sorted(set(raw_commands) - {"help"})
    for command in ("look", "inventory"):
        if command not in commands:
            commands.append(command)
    return {
        "observation": observation,
        "admissible_commands": commands,
        "steps": int(steps),
        "max_steps": int(max_steps),
        "won": bool(won),
        "done": bool(done),
    }


def agent_messages(
    prompt: list[dict[str, Any]], initial: dict[str, Any], *, instruction: str | None = None
) -> list[dict[str, str]]:
    """Honor native Codex's (system?, user) initial-message contract."""
    if __package__:
        from .data import FALLBACK_SOLVER_PROMPT
    else:
        from data import FALLBACK_SOLVER_PROMPT

    system = None
    user = None
    for message in prompt:
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user") or not isinstance(content, str):
            raise ValueError("ALFWorld Codex prompts require text-only system/user messages")
        if role == "system":
            if system is not None or user is not None:
                raise ValueError("At most one system message may precede the user message")
            system = content
        elif user is not None:
            raise ValueError("At most one initial user message is supported by native Codex")
        else:
            user = content
    if instruction is not None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("metadata.instruction must be non-empty text")
        user = instruction
    safe = public_state(
        initial,
        steps=int(initial.get("steps", 0)),
        max_steps=int(initial.get("max_steps", 50)),
        won=bool(initial.get("won", False)),
        done=bool(initial.get("done", False)),
    )
    content = (user or FALLBACK_SOLVER_PROMPT) + "\n\nInitial game state:\n" + json.dumps(safe, indent=2)
    messages = [{"role": "system", "content": system}] if system is not None else []
    return messages + [{"role": "user", "content": content}]
