"""Pure-stdlib ALFWorld identities, catalogs, and train-only development splits."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

TASK_NAME = "alfworld_skill_training"
OFFICIAL_SPLITS = ("train", "valid_seen", "valid_unseen")
TASK_TYPES = frozenset(
    {
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    }
)
FALLBACK_SOLVER_PROMPT = """Complete the ALFWorld household goal in the initial observation.
Use only alfworld-step 'exact environment action' to interact with the game.
Choose one action from admissible_commands at a time. All environment actions,
including look and inventory, count toward the episode's action limit.
Use alfworld-step --observe to read the current state without changing it.
Continue until done=true; stop immediately when won=true or the action limit is
reached. Report the environment outcome accurately. Do not inspect datasets,
expert plans, other episode directories, model files, or unrelated services.
Do not reset the game, install software, or modify the game client.
"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_relative_game_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or len(path.parts) != 5
        or path.parts[0] != "json_2.1.1"
        or path.parts[1] not in OFFICIAL_SPLITS
        or path.name != "game.tw-pddl"
        or str(path) != value
    ):
        raise ValueError(f"Invalid ALFWorld relative game path: {value!r}")
    return value


def stable_game_id(relative_game_path: str, game_sha256: str) -> str:
    """Identity is independent of scan order, catalog size, and mount location."""
    validate_relative_game_path(relative_game_path)
    if not re.fullmatch(r"[0-9a-f]{64}", game_sha256):
        raise ValueError("game_sha256 must contain 64 lowercase hexadecimal characters")
    split = PurePosixPath(relative_game_path).parts[1]
    digest = hashlib.sha256((relative_game_path + "\0" + game_sha256).encode()).hexdigest()
    return f"{split}_{digest[:24]}"


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    split: str
    task_type: str
    scene_id: str
    relative_game_path: str
    game_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GameRecord:
        record = cls(**{key: str(value[key]) for key in cls.__dataclass_fields__})
        relative = validate_relative_game_path(record.relative_game_path)
        if record.split != PurePosixPath(relative).parts[1]:
            raise ValueError("Catalog split disagrees with the source path")
        if record.task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported ALFWorld task type: {record.task_type}")
        if record.game_id != stable_game_id(relative, record.game_sha256):
            raise ValueError("Catalog game_id does not match its immutable identity")
        if not record.scene_id:
            raise ValueError("A scene_id is required for grouped development splits")
        return record


def _scene_id(trajectory: dict[str, Any], task_directory: str) -> str:
    scene = trajectory.get("scene") or {}
    if scene.get("floor_plan"):
        return str(scene["floor_plan"])
    number = scene.get("scene_num", task_directory.rsplit("-", 1)[-1])
    return f"FloorPlan{number}"


def build_catalog(benchmark_root: Path) -> list[GameRecord]:
    """Read existing official games; no task generation or environment execution.

    Match AlfredTWEnv's six supported task types and known-solvable filtering.
    Expert plans are never copied into a TaskConfig or agent prompt.
    """
    data_root = benchmark_root.resolve() / "data"
    records = []
    for split in OFFICIAL_SPLITS:
        split_root = data_root / "json_2.1.1" / split
        for path in sorted(split_root.glob("*/*/game.tw-pddl")):
            if data_root not in path.resolve().parents:
                raise ValueError(f"Game path escapes the benchmark data root: {path}")
            game = json.loads(path.read_text())
            if game.get("solvable") is not True:
                continue
            trajectory_path = path.parent / "traj_data.json"
            if not trajectory_path.is_file() or not (path.parent / "initial_state.pddl").is_file():
                raise ValueError(f"Missing companion data for {path}")
            trajectory = json.loads(trajectory_path.read_text())
            task_type = str(trajectory.get("task_type", ""))
            if task_type not in TASK_TYPES:
                continue
            relative = path.relative_to(data_root).as_posix()
            digest = sha256_file(path)
            records.append(
                GameRecord(
                    stable_game_id(relative, digest),
                    split,
                    task_type,
                    _scene_id(trajectory, path.parent.parent.name),
                    relative,
                    digest,
                )
            )
    if not records:
        raise ValueError(f"No eligible ALFWorld games under {data_root}")
    return records


def catalog_digest(records: list[GameRecord]) -> str:
    payload = [r.to_dict() for r in sorted(records, key=lambda r: r.relative_game_path)]
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def write_catalog(records: list[GameRecord], path: Path) -> str:
    digest = catalog_digest(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "catalog_digest": digest, "games": [r.to_dict() for r in records]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return digest


def load_catalog(path: Path) -> tuple[list[GameRecord], str]:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported ALFWorld catalog schema")
    records = [GameRecord.from_dict(row) for row in value["games"]]
    if len({r.game_id for r in records}) != len(records):
        raise ValueError("Duplicate game identities in catalog")
    if len({r.relative_game_path for r in records}) != len(records):
        raise ValueError("Duplicate game paths in catalog")
    digest = catalog_digest(records)
    if digest != value.get("catalog_digest"):
        raise ValueError("ALFWorld catalog digest mismatch")
    return records, digest


def partition_records(
    records: list[GameRecord], *, dev_fraction: float = 0.1, seed: int = 42
) -> dict[str, list[GameRecord]]:
    """Keep official test splits intact; hold out whole scenes from train only."""
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be in [0, 1)")
    train = [r for r in records if r.split == "train"]
    if not train:
        raise ValueError("The official training split is empty")
    scenes = sorted(
        {r.scene_id for r in train},
        key=lambda scene: hashlib.sha256(f"{seed}\0{scene}".encode()).hexdigest(),
    )
    if dev_fraction and len(scenes) < 2:
        raise ValueError("A scene-group development split needs at least two training scenes")
    count = min(len(scenes) - 1, max(1, round(len(scenes) * dev_fraction))) if dev_fraction else 0
    dev_scenes = set(scenes[:count])
    result = {
        "train": [r for r in train if r.scene_id not in dev_scenes],
        "dev": [r for r in train if r.scene_id in dev_scenes],
        "test_seen": [r for r in records if r.split == "valid_seen"],
        "test_unseen": [r for r in records if r.split == "valid_unseen"],
    }
    for rows in result.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}\0{r.game_id}".encode()).hexdigest())
    return result


def task_sample(
    record: GameRecord,
    *,
    phase: str,
    digest: str,
    run_dir: Path,
    prompt: str = FALLBACK_SOLVER_PROMPT,
    index: int = 0,
) -> dict[str, Any]:
    """Serialize a native Uni-Agent TaskConfig beneath tools_kwargs['task']."""
    if phase not in ("train", "dev", "test_seen", "test_unseen"):
        raise ValueError(f"Unknown dataset phase: {phase}")
    expected = {"train": "train", "dev": "train", "test_seen": "valid_seen", "test_unseen": "valid_unseen"}
    if record.split != expected[phase]:
        raise ValueError("A TaskConfig cannot move an official evaluation game into training")
    metadata = {
        **record.to_dict(),
        "stable_game_id": record.game_id,
        "catalog_digest": digest,
        "dataset_phase": phase,
        "run_dir": str(run_dir.resolve()),
    }
    return {
        "data_source": "alfworld",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": "environment_won"},
        "extra_info": {
            "index": index,
            "split": phase,
            "game_id": record.game_id,
            "tools_kwargs": {"task": {"name": TASK_NAME, "metadata": metadata}},
        },
    }


def partition_summary(partitions: dict[str, list[GameRecord]]) -> dict[str, Any]:
    return {
        phase: {
            "games": len(rows),
            "scenes": len({r.scene_id for r in rows}),
            "task_types": dict(sorted(Counter(r.task_type for r in rows).items())),
        }
        for phase, rows in partitions.items()
    }
