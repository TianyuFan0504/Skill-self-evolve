#!/usr/bin/env python3
"""Export recorded train/dev metrics without altering or smoothing observations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

TRAIN_REWARD_KEYS = ("train/reward", "critic/rewards/mean", "critic/score/mean", "reward/mean")
TRAIN_ACCURACY_KEYS = ("train/accuracy", "train/acc", "reward_extra_info/acc", "train/accuracy_mean")


def read_points(path: Path) -> list[dict[str, object]]:
    points = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics", record.get("data", record))
        step = record.get("step", record.get("global_step", metrics.get("training/global_step")))
        if step is None:
            continue
        for key, value in metrics.items():
            if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
                points.append(
                    {
                        "step": int(step),
                        "metric": key,
                        "value": float(value),
                        "phase": record.get("phase", "unspecified"),
                        "source_line": number,
                    }
                )
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    points = read_points(root / "metrics.jsonl")
    if not points:
        raise SystemExit("No numeric observations were found in metrics.jsonl")
    with (root / "metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("step", "phase", "metric", "value", "source_line"))
        writer.writeheader()
        writer.writerows(points)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {p["metric"] for p in points}
    reward = next((name for name in TRAIN_REWARD_KEYS if name in names), None)
    train_accuracy = next((name for name in TRAIN_ACCURACY_KEYS if name in names), None)
    dev_accuracy = sorted(
        name for name in names if str(name).startswith("val-core/") and str(name).endswith("/acc/mean@1")
    )
    if not dev_accuracy:
        dev_accuracy = sorted(
            name
            for name in names
            if str(name).startswith("val/")
            and any(token in str(name).lower() for token in ("accuracy", "success_rate", "test_score"))
        )
    panels = [
        ("Training reward", [reward] if reward else []),
        ("Accuracy (recorded train/dev observations)", ([train_accuracy] if train_accuracy else []) + dev_accuracy),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for axis, (title, keys) in zip(axes, panels, strict=True):
        for key in keys:
            values = [p for p in points if p["metric"] == key]
            desired_phase = "dev" if str(key).startswith("val") else "train"
            phase_values = [p for p in values if p["phase"] == desired_phase]
            if phase_values:
                values = phase_values
            axis.plot([p["step"] for p in values], [p["value"] for p in values], marker="o", label=key)
        axis.set_title(title)
        axis.set_xlabel("Optimizer/training step reported by the trainer")
        axis.set_ylabel("Observed score")
        axis.grid(alpha=0.2)
        if keys:
            axis.legend(fontsize="small")
        else:
            axis.text(0.5, 0.5, "No observations for this panel", ha="center", va="center", transform=axis.transAxes)
    fig.savefig(root / "reward_accuracy.png", dpi=160)
    fig.savefig(root / "reward_accuracy.svg")
    plt.close(fig)
    print(
        json.dumps(
            {
                "numeric_observations": len(points),
                "reward_metric": reward,
                "train_accuracy_metric": train_accuracy,
                "dev_accuracy_metrics": dev_accuracy,
                "csv": str(root / "metrics.csv"),
                "plot": str(root / "reward_accuracy.png"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
