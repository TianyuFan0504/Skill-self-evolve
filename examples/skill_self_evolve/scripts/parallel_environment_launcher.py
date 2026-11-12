"""Launch or attach to the explicit four-process ALFWorld service profile."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

POLICY = "sticky-process-environments-durable-io-v1"
PROTOCOL = "alfworld-step-request-id-v1"


def compatible_health(settings, health, catalog):
    return (
        set(settings.get("allowed_splits", [])) == {"train"}
        and settings.get("environment_policy") == POLICY
        and settings.get("action_request_protocol") == PROTOCOL
        and settings.get("environment_processes") == 4
        and settings.get("io_workers") == 4
        and health.get("ready") is True
        and health.get("catalog_digest") == catalog
        and health.get("environment_policy") == POLICY
        and health.get("action_request_protocol") == PROTOCOL
        and health.get("environment_processes") == 4
        and health.get("io_concurrency") == 4
    )


def read_health(settings):
    with urllib.request.urlopen(settings["url"].rstrip("/") + "/health", timeout=5) as response:
        return json.load(response)


def start_environment(run, benchmark, source):
    """Never replace a service or its config; recovery owns any transition."""
    run, benchmark, source = Path(run), Path(benchmark), Path(source)
    config = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
    catalog = json.loads((run / "data/split_manifest.json").read_text())["catalog_digest"]
    if config.exists():
        settings = json.loads(config.read_text())
        health = read_health(settings)
        if not compatible_health(settings, health, catalog):
            raise RuntimeError("Reconcile the existing environment before selecting the four-process profile")
        pid = int(settings["pid"])
        os.kill(pid, 0)
        return pid, False
    command = [
        os.environ.get("ALFWORLD_ENV_PYTHON", str(benchmark / "env/bin/python")),
        str(source / "examples/skill_self_evolve/scripts/parallel_environment_server.py"),
        "--benchmark-root", str(benchmark), "--run-root", str(run), "--config-path", str(config),
        "--catalog-path", str(run / "data/catalog.json"), "--allowed-splits", "train",
        "--environment-processes", "4", "--io-workers", "4", "--port", "0",
    ]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    with (run / "environment-server.log").open("a") as output:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True, env=environment)
    deadline = time.monotonic() + 60
    try:
        (run / "environment-server.pid").write_text(str(child.pid) + "\n")
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("The parallel environment exited; inspect environment-server.log")
            if config.exists():
                try:
                    settings = json.loads(config.read_text())
                    health = read_health(settings)
                except (OSError, ValueError):
                    time.sleep(.1)
                    continue
                if compatible_health(settings, health, catalog) and settings.get("pid") == child.pid:
                    return child.pid, True
                raise RuntimeError("The new environment published an incompatible profile")
            time.sleep(.1)
        raise TimeoutError("The parallel environment did not become ready within 60 seconds")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        raise
