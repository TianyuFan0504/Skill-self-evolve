"""Load native task modules with only their optional external interfaces stubbed."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "verl/experimental/skill_harness/alfworld"
PACKAGE = "_native_runtime_unit"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]
sys.modules.setdefault(PACKAGE, package)
accounts = importlib.import_module(PACKAGE + ".accounts")
native_files = importlib.import_module(PACKAGE + ".native_files")
budget = importlib.import_module(PACKAGE + ".budget")


class Value:
    def __init__(self, **values):
        self.__dict__.update(values)


class LocalSandbox:
    async def write_file(self, path, content):
        Path(path).write_text(content)


def load_task():
    stubs = {}
    for name in (
        "uni_agent.sandbox.base",
        "uni_agent.sandbox.local",
        "uni_agent.tasks.base",
        "uni_agent.tasks.registry",
        "aiohttp",
    ):
        stubs[name] = types.ModuleType(name)
    stubs["uni_agent.sandbox.base"].ExecResult = Value
    stubs["uni_agent.sandbox.local"].LocalSandbox = LocalSandbox
    stubs["uni_agent.tasks.base"].Task = object
    stubs["uni_agent.tasks.base"].TaskConfig = Value
    stubs["uni_agent.tasks.base"].TaskResult = Value
    stubs["uni_agent.tasks.registry"].register_task = lambda name: lambda cls: cls
    stubs["aiohttp"].ClientTimeout = Value
    with patch.dict(sys.modules, stubs):
        return importlib.import_module(PACKAGE + ".task")


task = load_task()
SPEC = importlib.util.spec_from_file_location(
    "_optional_linux_unit", ROOT / "examples/skill_self_evolve/runtime/linux_accounts.py"
)
linux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linux)
