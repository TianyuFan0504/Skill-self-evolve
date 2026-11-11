"""Prepare native Uni-Agent ALFWorld parquet files from existing official games.

This CLI never downloads data, generates tasks, or starts an environment.  It can
also be invoked by file path to avoid importing verl's training dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .data import (
        FALLBACK_SOLVER_PROMPT,
        build_catalog,
        partition_records,
        partition_summary,
        task_sample,
        write_catalog,
    )
else:
    from data import (
        FALLBACK_SOLVER_PROMPT,
        build_catalog,
        partition_records,
        partition_summary,
        task_sample,
        write_catalog,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args()
    for name in ("max_train_samples", "max_dev_samples", "max_test_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Parquet output requires the training environment's pyarrow package") from exc

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "split_manifest.json").exists():
        raise SystemExit("The output already contains a prepared dataset; choose a new output directory")
    prompt = args.prompt_file.read_text() if args.prompt_file else FALLBACK_SOLVER_PROMPT
    if not prompt.strip():
        parser.error("The solver task prompt cannot be empty")
    records = build_catalog(args.benchmark_root)
    digest = write_catalog(records, output / "catalog.json")
    complete = partition_records(records, dev_fraction=args.dev_fraction, seed=args.seed)
    selected = {}
    limits = {
        "train": args.max_train_samples,
        "dev": args.max_dev_samples,
        "test_seen": args.max_test_samples,
        "test_unseen": args.max_test_samples,
    }
    files = {}
    for phase, all_rows in complete.items():
        rows = all_rows[: limits[phase]] if limits[phase] is not None else all_rows
        selected[phase] = rows
        values = [
            task_sample(record, phase=phase, digest=digest, run_dir=args.run_dir, prompt=prompt, index=i)
            for i, record in enumerate(rows)
        ]
        if values:
            table = pa.Table.from_pylist(values)
        else:
            example = task_sample(records[0], phase="train", digest=digest, run_dir=args.run_dir, prompt=prompt)
            table = pa.Table.from_pylist([example]).slice(0, 0)
        path = output / (phase + ".parquet")
        pq.write_table(table, path)
        files[phase] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    manifest = {
        "schema_version": 1,
        "benchmark_root": str(args.benchmark_root.resolve()),
        "run_dir": str(args.run_dir.resolve()),
        "catalog_digest": digest,
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "development_grouping": "whole training scenes, deterministic seeded hash order",
        "task_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "available": partition_summary(complete),
        "selected": partition_summary(selected),
        "phase_game_ids": {phase: [row.game_id for row in rows] for phase, rows in selected.items()},
        "parquet_files": files,
        "no_synthetic_tasks": True,
        "evaluation_games_used_for_training": False,
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"catalog_digest": digest, "selected": manifest["selected"], "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
