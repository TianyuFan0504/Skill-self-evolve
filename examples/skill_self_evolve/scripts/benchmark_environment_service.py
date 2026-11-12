#!/usr/bin/env python3
"""Bounded real ALFWorld/NAS service comparison; never creates training data.

Runs the frozen single-thread service and the proposed process/I/O extension
on disposable loopback ports. Eight warm-up sessions precede a mixed wave of
32 new creations and ongoing look actions. No UID, model or backend is used.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
from aiohttp import web

if __package__:
    from . import parallel_environment_server as proposed
else:
    import parallel_environment_server as proposed

core = proposed.core
_BASE_TIMING = contextvars.ContextVar("baseline_request_timing", default=None)


class TimedEngineExecutor(ThreadPoolExecutor):
    def submit(self, function, /, *args, **kwargs):
        timing, submitted = _BASE_TIMING.get(), time.monotonic_ns()

        def execute():
            entered = time.monotonic_ns()
            try:
                return function(*args, **kwargs)
            finally:
                if timing is not None:
                    timing["engine_queue_ms"] = timing.get("engine_queue_ms", 0) + (entered - submitted) / 1e6
                    timing["engine_execute_ms"] = (
                        timing.get("engine_execute_ms", 0) + (time.monotonic_ns() - entered) / 1e6
                    )

        return super().submit(execute)


class InstrumentedBaseline(core.GameService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = ThreadPoolExecutor(max_workers=1, thread_name_prefix="baseline-http-timing")
        self.telemetry_pending, self.telemetry_errors = set(), []
        self.timing_path = self.run_root / "diagnostics/environment-http-timing.jsonl"
        self.timing_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    _record_timing = proposed.ParallelGameService._record_timing

    @web.middleware
    async def timing_middleware(self, request, handler):
        async def tracked(actual_request):
            token = _BASE_TIMING.set(actual_request["environment_timing"])
            try:
                return await handler(actual_request)
            finally:
                _BASE_TIMING.reset(token)

        return await proposed.ParallelGameService.timing_middleware(self, request, tracked)

    def _budget_event(self, *args, **kwargs):
        started = time.monotonic_ns()
        try:
            return super()._budget_event(*args, **kwargs)
        finally:
            timing = _BASE_TIMING.get()
            if timing is not None:
                timing["io_persist_ms"] = timing.get("io_persist_ms", 0) + (time.monotonic_ns() - started) / 1e6

    async def close_all(self, app):
        await super().close_all(app)
        if self.telemetry_pending:
            await asyncio.gather(*tuple(self.telemetry_pending))
        self.telemetry.shutdown(wait=True)


def baseline_application(service):
    app = web.Application(middlewares=[service.timing_middleware])
    app.router.add_get("/health", service.health)
    app.router.add_post("/admin/create", service.create)
    app.router.add_post("/admin/close/{session}", service.close)
    app.router.add_post("/admin/budget/{session}/start", service.start_budget)
    app.router.add_get("/admin/budget/{session}", service.observe_budget)
    app.router.add_get("/session/{session}/observe", service.observe)
    app.router.add_post("/session/{session}/step", service.step)
    app.on_cleanup.append(service.close_all)
    return app


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        name: ordered[round((len(ordered) - 1) * fraction)]
        for name, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("max", 1))
    }


def summarize(rows, health, wall_seconds):
    result = {
        "wall_seconds": wall_seconds,
        "http_requests": len(rows),
        "http_errors": sum(row["status"] != 200 for row in rows),
        "phases": {},
    }
    fields = [
        "client_ms",
        "session_queue_ms",
        "engine_queue_ms",
        "engine_execute_ms",
        "io_queue_ms",
        "io_persist_ms",
        "response_ready_ms",
        "response_send_ms",
    ]
    for phase in sorted({row["phase"] for row in rows}):
        selected = [row for row in rows if row["phase"] == phase]
        result["phases"][phase] = {
            "count": len(selected),
            "errors": sum(row["status"] != 200 for row in selected),
            **{field: quantiles([row[field] for row in selected if field in row]) for field in fields},
        }
    result["health"] = {
        "count": len(health),
        "errors": sum(row["status"] != 200 for row in health),
        "client_ms": quantiles([row["client_ms"] for row in health]),
        "event_loop_lag_ms": quantiles([row["event_loop_lag_ms"] for row in health]),
    }
    return result


async def one_mode(args, records, mode):
    run = args.output_root / mode
    run.mkdir(mode=0o700)
    loop = asyncio.get_running_loop()
    executor = None
    if mode == "baseline":
        executor = TimedEngineExecutor(max_workers=1)
        loop.set_default_executor(executor)
        service = InstrumentedBaseline(records, args.benchmark_root, run, allowed_splits=["train"])
        app = baseline_application(service)
    else:
        service = proposed.ParallelGameService(
            records, args.benchmark_root, run, allowed_splits=["train"], process_count=4, io_workers=4
        )
        await service.engines.start()
        app = proposed.application(service)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    handle = {
        "kind": "disposable_cpu_environment_benchmark",
        "pid": os.getpid(),
        "mode": mode,
        "port": port,
        "run_root": str(run),
        "worker_pids": getattr(getattr(service, "engines", None), "pids", []),
        "no_uid_model_backend_training_data": True,
    }
    proposed.core.write_private_json(run / "handle.json", handle)
    rows, health, sessions = [], [], []
    headers = {"X-Admin-Token": service.admin_token}
    stop = asyncio.Event()
    started = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as client:

            async def call(method, path, *, phase, payload=None, admin=False):
                entered = time.monotonic()
                item = {"phase": phase, "path": path, "status": None}
                try:
                    async with client.request(
                        method, base + path, json=payload, headers=headers if admin else None
                    ) as response:
                        item["status"] = response.status
                        body = await response.json()
                        item.update(json.loads(response.headers.get("X-Environment-Timing", "{}")))
                        item["http_request_id"] = response.headers.get("X-Environment-HTTP-ID")
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    item["status"], item["error"] = 0, type(error).__name__
                    body = None
                item["client_ms"] = (time.monotonic() - entered) * 1000
                rows.append(item)
                if item["status"] != 200:
                    raise RuntimeError("Benchmark request failed: " + json.dumps(item))
                return body

            async def create(index, phase):
                row = records[index % len(records)]
                attempt = uuid.uuid4().hex
                output = run / "episodes" / row.game_id / ("attempt_" + attempt)
                value = await call(
                    "POST",
                    "/admin/create",
                    phase=phase,
                    payload={**row.to_dict(), "catalog_digest": service.catalog_digest, "output_dir": str(output)},
                    admin=True,
                )
                session = value["session_id"]
                sessions.append(session)
                # Valid protocol metadata for this diagnostic only. No real
                # account is allocated and these rows never enter training.
                registration = {
                    "schema_version": 1,
                    "budget_seconds": 900,
                    "binding": {
                        "binding_id": "a" * 64,
                        "binding_sha256": "b" * 64,
                        "attempt_id": attempt,
                        "gateway_session_id": "service-cpu-benchmark",
                        "environment_session_id": session,
                    },
                    "account": {
                        "lease_id": "c" * 32,
                        "uid": 20000,
                        "gid": 20000,
                        "name": "benchmark-metadata-only",
                        "owner": {"kind": "ordinary", "run_root": str(run), "scope_id": attempt},
                    },
                }
                await call(
                    "POST",
                    "/admin/budget/" + session + "/start",
                    phase=phase + "_budget",
                    payload=registration,
                    admin=True,
                )
                return session

            async def actions(session, count, phase):
                for _ in range(count):
                    await call(
                        "POST",
                        "/session/" + session + "/step",
                        phase=phase,
                        payload={"action": "look", "request_id": uuid.uuid4().hex},
                    )

            async def health_loop():
                while not stop.is_set():
                    expected = time.monotonic() + 0.05
                    await asyncio.sleep(0.05)
                    entered = time.monotonic()
                    item = {"event_loop_lag_ms": max(0, entered - expected) * 1000}
                    try:
                        async with client.get(base + "/health", timeout=aiohttp.ClientTimeout(total=1)) as response:
                            await response.read()
                            item["status"] = response.status
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        item["status"] = 0
                    item["client_ms"] = (time.monotonic() - entered) * 1000
                    health.append(item)

            monitor = asyncio.create_task(health_loop())
            try:
                warm = await asyncio.gather(*(create(index, "warm_create") for index in range(args.warm_sessions)))
                await asyncio.gather(*(actions(session, 2, "warm_action") for session in warm[:4]))
                wave = asyncio.gather(*(create(index, "mixed_create") for index in range(args.create_wave)))
                await asyncio.sleep(0.02)
                await asyncio.gather(
                    wave, *(actions(session, args.actions_per_session, "mixed_action") for session in warm[:4])
                )
                for session in list(sessions):
                    await call("POST", "/admin/close/" + session, phase="close", admin=True)
            finally:
                stop.set()
                await monitor
    finally:
        await runner.cleanup()
        if executor is not None:
            executor.shutdown(wait=True)
    wall = time.monotonic() - started
    server_timings = {
        row["http_request_id"]: row
        for line in service.timing_path.read_text().splitlines()
        if (row := json.loads(line)).get("http_request_id")
    }
    for row in rows:
        observed = server_timings.get(row.get("http_request_id"), {})
        row.update({key: value for key, value in observed.items() if key.endswith("_ms")})
    result = {
        "schema_version": 1,
        "mode": mode,
        "kind": "diagnostic_not_training_or_eval",
        "no_uid_model_backend_created": True,
        "original_games": [row.game_id for row in records],
        "warm_sessions": args.warm_sessions,
        "create_wave": args.create_wave,
        "active_sessions_during_creates": 4,
        "actions_per_active_session": args.actions_per_session,
        "frozen_core": str(proposed.CORE_DIR / "server.py"),
        "baseline_io_measurement_scope": "synchronous budget-event append/flush/fsync; not all legacy file writes",
        "parallel_io_measurement_scope": "all awaited durable I/O operations",
        **summarize(rows, health, wall),
    }
    proposed.core.write_private_json(run / "summary.json", result)
    proposed.core.write_private_json(run / "raw-client-measurements.json", {"requests": rows, "health": health})
    print(
        json.dumps(
            {"mode": mode, "summary": str(run / "summary.json"), "wall_seconds": wall, "errors": result["http_errors"]}
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warm-sessions", type=int, default=8)
    parser.add_argument("--create-wave", type=int, default=32)
    parser.add_argument("--actions-per-session", type=int, default=8)
    args = parser.parse_args()
    if args.warm_sessions < 4 or min(args.create_wave, args.actions_per_session) < 1:
        parser.error("At least four warm sessions and positive workload counts are required")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(mode=0o700)
    records, _ = core.load_catalog(args.catalog_path)
    chosen = {}
    for row in records:
        if row.split == "train":
            chosen.setdefault(row.task_type, row)
        if len(chosen) == 4:
            break
    if len(chosen) != 4:
        raise RuntimeError("Four original training task families are required")
    results = [asyncio.run(one_mode(args, list(chosen.values()), mode)) for mode in ("baseline", "parallel4")]
    core.write_private_json(
        args.output_root / "comparison.json",
        {
            "schema_version": 1,
            "kind": "real_alfworld_service_cpu_comparison",
            "diagnostic_only": True,
            "results": results,
            "interpretation_limit": "Bounded loopback service workload; not whole-rollout speedup or heldout accuracy",
        },
    )


if __name__ == "__main__":
    main()
