#!/usr/bin/env python3
"""Forward the joint recipe and Hydra overrides to the training module."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "verl.experimental.skill_harness.train",
            f"--config-path={config_dir}",
            "--config-name=alfworld_joint",
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
