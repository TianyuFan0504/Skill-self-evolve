"""Append-only, finite-valued experiment telemetry without model dependencies."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        return json_value(value.item())
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    raise TypeError(f"Unsupported telemetry value: {type(value).__name__}")


def append_metrics(
    root: str | Path,
    *,
    step: int,
    phase: str,
    metrics: Mapping[str, Any],
    stage: str | None = None,
) -> None:
    """Persist one complete row; metrics are evidence, not acceptance decisions."""
    if phase not in {"train", "dev", "test_seen", "test_unseen", "system"}:
        raise ValueError(f"Unknown metric phase: {phase}")
    if stage is not None and (stage not in {"initial", "final"} or not phase.startswith("test_")):
        raise ValueError("Evaluation stage is only valid for initial/final official test metrics")
    path = Path(root) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "step": int(step),
        "phase": phase,
        "timestamp": time.time(),
        "metrics": json_value(metrics),
    }
    if stage is not None:
        row["evaluation_stage"] = stage
    encoded = json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def episode_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Deduplicate Gateway prefix trajectories by real prompt/session identity.

    A native agent episode may yield multiple token trajectories. Averaging those
    rows directly would weight prefix-splitting episodes more heavily.
    """
    episodes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("padding"):
            continue
        key = (str(row["uid"]), str(row["session_id"]))
        reward = float(row["reward"])
        accuracy = float(row["accuracy"])
        if not math.isfinite(reward) or accuracy not in (0.0, 1.0):
            raise ValueError("ALFWorld telemetry requires finite reward and binary engine accuracy")
        candidate = {"reward": reward, "accuracy": accuracy, "error": bool(row.get("error"))}
        if key in episodes and episodes[key] != candidate:
            raise ValueError(f"Conflicting terminal outcomes for episode {key}")
        episodes[key] = candidate
    if not episodes:
        return {"train/episodes": 0.0}
    n = len(episodes)
    return {
        "train/episodes": float(n),
        "train/reward": sum(row["reward"] for row in episodes.values()) / n,
        "train/accuracy": sum(row["accuracy"] for row in episodes.values()) / n,
        "train/infrastructure_error_rate": sum(row["error"] for row in episodes.values()) / n,
    }
