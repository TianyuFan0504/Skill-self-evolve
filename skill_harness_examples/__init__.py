"""Dedicated import namespace for the project's examples entrypoints."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "examples/skill_self_evolve/scripts")]
