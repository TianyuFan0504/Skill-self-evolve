#!/usr/bin/env python3
"""ALFWorld HTTP service with sticky process shards and asynchronous durable I/O.

This examples-only entry preserves the core budget protocol and public state
allowlist. Each TextWorld process handles one engine call at a time; session
locks also order durable journals. No TextWorld object crosses a process.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
import signal
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

CORE_DIR = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/alfworld"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
_spec = importlib.util.spec_from_file_location("_frozen_alfworld_server", CORE_DIR / "server.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)

POLICY = "sticky-process-environments-durable-io-v1"
_ENGINES = {}


def _state_projection(state):
    return {
        "feedback": state.get("feedback", state.get("observation", "")),
        "admissible_commands": list(state.get("admissible_commands", [])),
        "won": bool(state.get("won", False)),
    }


def _engine_operation(operation, payload):
    """Runs in exactly one dedicated spawned process per shard."""
    entered = time.monotonic_ns()
    session_id = payload.get("session_id")
    if operation == "ping":
        result = {"pid": os.getpid()}
    elif operation == "create":
        import textworld
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env = textworld.start(
            payload["game_path"], request_infos=infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]
        )
        try:
            env.seed(42)
            state = env.reset()
        except BaseException:
            env.close()
            raise
        _ENGINES[session_id] = env
        result = {"state": _state_projection(state)}
    elif operation == "step":
        # Admission happens after this shard's actual queue, before engine entry.
        entered = time.monotonic_ns()
        if entered >= payload["deadline_ns"]:
            result = {"admitted": False}
        else:
            state, score, done = _ENGINES[session_id].step(payload["action"])
            result = {
                "admitted": True,
                "state": _state_projection(state),
                "engine_score": float(score),
                "done": bool(done),
            }
    elif operation == "close":
        _ENGINES.pop(session_id).close()
        result = {}
    else:
        raise ValueError("Unknown engine operation")
    return {**result, "worker_pid": os.getpid(), "entered_ns": entered, "completed_ns": time.monotonic_ns()}


def _append_durable(path, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("Short append to an environment journal")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _start_budget_files(output, registration):
    core.write_private_json(output / "budget-registration.json", registration, immutable=True)
    with (output / "budget-events.jsonl").open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_output(output, game_path, expected_hash):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (output / "initial-state.json").exists():
        raise ValueError("Episode output already contains an environment attempt")
    if core.sha256_file(game_path) != expected_hash:
        raise ValueError("The game file changed after catalog preparation")


class StickyEnginePool:
    def __init__(self, count=4, *, operation=_engine_operation):
        if count < 1:
            raise ValueError("Environment process count must be positive")
        self.executors = [
            ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) for _ in range(count)
        ]
        self.operation, self.cursor = operation, 0
        self.pids = []

    async def start(self):
        results = await asyncio.gather(*(self.call(index, "ping", {}) for index in range(len(self.executors))))
        self.pids = [result["worker_pid"] for result in results]
        if len(set(self.pids)) != len(self.executors):
            raise RuntimeError("Environment shards do not have independent process identities")

    def assign(self):
        shard = self.cursor % len(self.executors)
        self.cursor += 1
        return shard

    async def call(self, shard, operation, payload, timing=None):
        submitted = time.monotonic_ns()
        result = await asyncio.get_running_loop().run_in_executor(
            self.executors[shard], self.operation, operation, payload
        )
        if timing is not None:
            timing["engine_queue_ms"] = timing.get("engine_queue_ms", 0) + (result["entered_ns"] - submitted) / 1e6
            timing["engine_execute_ms"] = (
                timing.get("engine_execute_ms", 0) + (result["completed_ns"] - result["entered_ns"]) / 1e6
            )
            timing["engine_shard"], timing["engine_pid"] = shard, result["worker_pid"]
        return result

    def shutdown(self):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=False)


class ParallelGameService(core.GameService):
    def __init__(self, *args, process_count=4, io_workers=4, engine_pool=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.engines = engine_pool or StickyEnginePool(process_count)
        self.io_workers = int(io_workers)
        source = Path(__file__).resolve()
        self.extension_identity = {
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "policy": POLICY,
        }
        self.io = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="alfworld-durable-io")
        self.telemetry = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alfworld-http-timing")
        self.operations, self.telemetry_pending, self.telemetry_errors = {}, set(), []
        self.timing_path = self.run_root / "diagnostics/environment-http-timing.jsonl"
        self.timing_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.outputs = set()

    async def _io(self, function, *args, timing=None, **kwargs):
        submitted = time.monotonic_ns()

        def execute():
            entered = time.monotonic_ns()
            result = function(*args, **kwargs)
            return result, entered, time.monotonic_ns()

        result, entered, completed = await asyncio.get_running_loop().run_in_executor(self.io, execute)
        if timing is not None:
            timing["io_queue_ms"] = timing.get("io_queue_ms", 0) + (entered - submitted) / 1e6
            timing["io_persist_ms"] = timing.get("io_persist_ms", 0) + (completed - entered) / 1e6
        return result

    async def _locked(self, session, timing):
        started = time.monotonic_ns()
        await session["lock"].acquire()
        timing["session_queue_ms"] = (time.monotonic_ns() - started) / 1e6

    def _record_timing(self, value):
        future = asyncio.get_running_loop().run_in_executor(self.telemetry, _append_durable, self.timing_path, value)
        self.telemetry_pending.add(future)

        def finished(done):
            self.telemetry_pending.discard(done)
            try:
                done.result()
            except BaseException as error:
                self.telemetry_errors.append(repr(error))

        future.add_done_callback(finished)

    @web.middleware
    async def timing_middleware(self, request, handler):
        started = time.monotonic_ns()
        timing = request["environment_timing"] = {
            "received_ns": started,
            "method": request.method,
            "path": request.path,
            "http_request_id": uuid.uuid4().hex,
        }
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = web.json_response({"error": error.text}, status=error.status)
        ready = time.monotonic_ns()
        timing["response_ready_ms"] = (ready - started) / 1e6
        response.headers["X-Environment-Service"] = POLICY
        response.headers["X-Environment-HTTP-ID"] = timing["http_request_id"]
        if "request_id" in timing:
            response.headers["X-Request-ID"] = timing["request_id"]
        response.headers["X-Environment-Timing"] = json.dumps(
            {key: value for key, value in timing.items() if key.endswith("_ms")}, separators=(",", ":")
        )
        try:
            await response.prepare(request)
            await response.write_eof()
        finally:
            finished = time.monotonic_ns()
            timing.update(
                response_send_ms=(finished - ready) / 1e6,
                total_ms=(finished - started) / 1e6,
                response_status=response.status,
            )
            self._record_timing(dict(timing))
        return response

    async def health(self, _request):
        return web.json_response(
            {
                "ready": bool(self.engines.pids),
                "catalog_digest": self.catalog_digest,
                "max_steps": self.max_steps,
                "budget_policy_version": core.BUDGET_POLICY_VERSION,
                "budget_seconds": self.budget_seconds,
                "environment_policy": POLICY,
                "environment_processes": len(self.engines.executors),
                "environment_worker_pids": self.engines.pids,
                "io_concurrency": self.io_workers,
                "environment_extension": self.extension_identity,
                "timing_writes_pending": len(self.telemetry_pending),
                "timing_write_errors": len(self.telemetry_errors),
                "action_request_ids": True,
                "action_request_protocol": "alfworld-step-request-id-v1",
                "server_instance_id": self.server_instance_id,
            }
        )

    async def create(self, request):
        self.authorized(request)
        timing = request["environment_timing"]
        payload = await request.json()
        row = self.games.get(payload.get("game_id"))
        if row is None:
            raise web.HTTPNotFound(text="Unknown ALFWorld game identity")
        if row.split not in self.allowed_splits:
            raise web.HTTPForbidden(text="This service does not allow the requested official split")
        try:
            core.validate_game_reference(payload, row.to_dict(), self.catalog_digest)
        except ValueError as error:
            raise web.HTTPConflict(text=str(error)) from error
        output = Path(payload["output_dir"]).resolve()
        game_path = (self.data_root / row.relative_game_path).resolve()
        if self.run_root not in output.parents or self.data_root not in game_path.parents:
            raise web.HTTPBadRequest(text="Episode/game path is outside its configured root")
        if output in self.outputs:
            raise web.HTTPConflict(text="Episode output is already assigned to an environment attempt")
        self.outputs.add(output)
        try:
            await self._io(_prepare_output, output, game_path, row.game_sha256, timing=timing)
        except ValueError as error:
            raise web.HTTPConflict(text=str(error)) from error
        session_id, shard = uuid.uuid4().hex, self.engines.assign()
        result = await self.engines.call(
            shard, "create", {"session_id": session_id, "game_path": str(game_path)}, timing
        )
        session = {
            "id": session_id,
            "game": row,
            "output": output,
            "shard": shard,
            "state": result["state"],
            "steps": 0,
            "won": bool(result["state"].get("won", False)),
            "done": False,
            "invalid_actions": 0,
            "started": time.time(),
            "lock": asyncio.Lock(),
            "budget": None,
        }
        self.sessions[session_id] = session
        initial = self.state(session)
        await self._io(core.write_private_json, output / "initial-state.json", initial, timing=timing, immutable=True)
        return web.json_response({"session_id": session_id, "state": initial, "catalog_digest": self.catalog_digest})

    async def observe(self, request):
        session, timing = self._session(request), request["environment_timing"]
        await self._locked(session, timing)
        try:
            return web.json_response(self.state(session))
        finally:
            session["lock"].release()

    async def start_budget(self, request):
        self.authorized(request)
        session, timing = self._session(request), request["environment_timing"]
        payload = await request.json()
        await self._locked(session, timing)
        try:
            existing = session["budget"]
            registration = core.make_registration(
                payload,
                session_id=session["id"],
                run_root=self.run_root,
                output=session["output"],
                server_instance_id=self.server_instance_id,
                started_ns=existing["registration"]["started_ns"] if existing else self.clock_ns(),
                budget_seconds=self.budget_seconds,
            )
            if existing:
                if registration != existing["registration"]:
                    raise core.BudgetOutcomeUnknown("A registered deadline cannot be replaced or extended")
            else:
                if session["steps"]:
                    raise core.BudgetOutcomeUnknown("The budget must start before any engine action")
                await self._io(_start_budget_files, session["output"], registration, timing=timing)
                session["budget"] = {
                    "registration": registration,
                    "sequence": 0,
                    "admitted": 0,
                    "rejected": 0,
                    "last_admitted": None,
                    "last_completed": None,
                    "errors": 0,
                    "ordered_snapshot": None,
                }
            return web.json_response(registration)
        except core.BudgetOutcomeUnknown as error:
            raise web.HTTPConflict(text=str(error)) from error
        finally:
            session["lock"].release()

    async def observe_budget(self, request):
        self.authorized(request)
        session, timing = self._session(request), request["environment_timing"]
        await self._locked(session, timing)
        try:
            previous = (session["budget"] or {}).get("ordered_snapshot")
            if previous is None:
                previous = await self._io(self._budget_snapshot, session, timing=timing)
                if previous["deadline_reached"]:
                    await self._io(
                        core.write_private_json,
                        session["output"] / "budget-ordered-state.json",
                        previous,
                        immutable=True,
                        timing=timing,
                    )
                    session["budget"]["ordered_snapshot"] = previous
            return web.json_response(previous)
        finally:
            session["lock"].release()

    def _request_identity(self, request, payload):
        body, header = payload.get("request_id"), request.headers.get("X-Request-ID")
        if body is not None and header is not None and body != header:
            raise web.HTTPConflict(text="Body/header action request identities differ")
        value = body if body is not None else header
        if value is None:
            # Legacy single-attempt clients remain usable, but cannot recover
            # a timed-out response by retrying with an invented action identity.
            value = uuid.uuid4().hex
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
            raise web.HTTPBadRequest(text="Action request_id must be UUID hex32")
        return value

    async def step(self, request):
        payload = await request.json()
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip() or len(action) > 512 or "\n" in action:
            raise web.HTTPBadRequest(text="One non-empty single-line environment action is required")
        action, request_id = action.strip(), self._request_identity(request, payload)
        session_id, timing = request.match_info["session"], request["environment_timing"]
        timing["request_id"] = request_id
        key = session_id, request_id
        record = self.operations.get(key)
        if record is None:
            session = self._session(request)
            record = {"action": action, "request_id": request_id}
            self.operations[key] = record
            record["task"] = asyncio.create_task(self._run_action(session, record, timing))
        elif record["action"] != action:
            raise web.HTTPConflict(text="Action request_id was already used for a different action")
        result = await asyncio.shield(record["task"])
        return web.json_response(result["result"], status=result["response_status"])

    async def query_request(self, request):
        key = request.match_info["session"], request.match_info["request_id"]
        record = self.operations.get(key)
        if record is None:
            raise web.HTTPNotFound(text="Unknown action request identity")
        if not record["task"].done():
            return web.json_response({"request_id": key[1], "status": "pending"}, status=202)
        result = record["task"].result()
        return web.json_response({"request_id": key[1], "status": "completed", **result})

    async def _run_action(self, session, record, timing):
        request_id, action = record["request_id"], record["action"]
        await self._locked(session, timing)
        try:
            try:
                result = await self._execute_action(session, action, timing)
                response = {"response_status": 200, "result": result}
            except web.HTTPException as error:
                response = {"response_status": error.status, "result": {"error": error.text}}
            except Exception as error:
                # Unknown engine state must never be re-executed by a retry.
                # Keep the exception private; the public result only identifies
                # an infrastructure failure, never a zero-reward task outcome.
                await self._io(
                    core.write_private_json,
                    session["output"] / ("action-error-" + request_id + ".json"),
                    {"error_type": type(error).__name__, "message": str(error)},
                    timing=timing,
                )
                response = {"response_status": 500, "result": {"error": "Environment action failed; do not re-execute"}}
            await self._io(
                core.write_private_json,
                session["output"] / ("action-result-" + request_id + ".json"),
                {
                    "schema_version": 1,
                    "session_id": session["id"],
                    "request_id": request_id,
                    "action": action,
                    **response,
                },
                immutable=True,
                timing=timing,
            )
            return response
        finally:
            session["lock"].release()

    async def _execute_action(self, session, action, timing):
        if session["done"]:
            return self.state(session)
        budget = session["budget"]
        if budget is None:
            raise web.HTTPConflict(text="The host must register the episode budget before actions")
        checked = self.clock_ns()
        if checked >= budget["registration"]["deadline_ns"]:
            await self._io(
                self._budget_event, session, kind="deadline_rejected", checked_ns=checked, action=action, timing=timing
            )
            raise web.HTTPConflict(text="The declared episode budget has expired; no action was applied")
        before = self.state(session)
        invalid = action not in before["admissible_commands"]
        try:
            result = await self.engines.call(
                session["shard"],
                "step",
                {"session_id": session["id"], "action": action, "deadline_ns": budget["registration"]["deadline_ns"]},
                timing,
            )
        except Exception:
            await self._io(
                self._budget_event, session, kind="engine_error", checked_ns=checked, action=action, timing=timing
            )
            raise
        if not result["admitted"]:
            await self._io(
                self._budget_event,
                session,
                kind="deadline_rejected",
                checked_ns=result["entered_ns"],
                action=action,
                timing=timing,
            )
            raise web.HTTPConflict(text="The declared episode budget expired before engine admission")
        session["state"], session["steps"] = result["state"], session["steps"] + 1
        session["invalid_actions"] += int(invalid)
        session["won"] = bool(result["state"].get("won", False))
        session["done"] = bool(result["done"] or session["won"] or session["steps"] >= self.max_steps)
        after = self.state(session)
        await self._io(
            self._budget_event,
            session,
            kind="admitted",
            checked_ns=result["entered_ns"],
            completed_ns=result["completed_ns"],
            action=action,
            timing=timing,
        )
        await self._io(
            _append_durable,
            session["output"] / "environment-trace.jsonl",
            {
                "step": session["steps"],
                "action": action,
                "action_in_exposed_commands": not invalid,
                "before": before,
                "after": after,
                "engine_score": result["engine_score"],
                "time": time.time(),
            },
            timing=timing,
        )
        return after

    async def _finish(self, session, *, termination_reason="agent_finished", timing=None):
        row = session["game"]
        result = {
            **row.to_dict(),
            "catalog_digest": self.catalog_digest,
            "reward": float(session["won"]),
            "won": session["won"],
            "done": session["done"],
            "steps": session["steps"],
            "step_limit": session["steps"] >= self.max_steps and not session["won"],
            "max_steps": self.max_steps,
            "invalid_actions": session["invalid_actions"],
            "wall_seconds": time.time() - session["started"],
            "termination_reason": termination_reason,
        }
        if session["budget"] is not None:
            result["episode_budget"] = await self._io(
                self._budget_snapshot, session, closed=True, close_reason=termination_reason, timing=timing
            )
            await self._io(
                core.write_private_json,
                session["output"] / "budget-environment.json",
                result["episode_budget"],
                timing=timing,
            )
        await self._io(core.write_private_json, session["output"] / "engine-result.json", result, timing=timing)
        await self.engines.call(session["shard"], "close", {"session_id": session["id"]}, timing)
        self.sessions.pop(session["id"], None)
        return result

    async def close(self, request):
        self.authorized(request)
        session, timing = self._session(request), request["environment_timing"]
        await self._locked(session, timing)
        try:
            return web.json_response(await self._finish(session, timing=timing))
        finally:
            session["lock"].release()

    async def close_all(self, _app):
        errors = []
        try:
            pending = [record["task"] for record in self.operations.values() if not record["task"].done()]
            if pending:
                results = await asyncio.gather(*pending, return_exceptions=True)
                errors.extend(value for value in results if isinstance(value, BaseException))
            for session in list(self.sessions.values()):
                try:
                    async with session["lock"]:
                        await self._finish(session, termination_reason="server_shutdown")
                except BaseException as error:
                    errors.append(error)
            if self.telemetry_pending:
                results = await asyncio.gather(*tuple(self.telemetry_pending), return_exceptions=True)
                errors.extend(value for value in results if isinstance(value, BaseException))
        finally:
            # One failed close must not skip the other owned worlds or leave
            # process/I/O pools running. Failure remains explicit and is never
            # converted into a successfully closed world or a task reward.
            for cleanup in (
                self.engines.shutdown,
                lambda: self.io.shutdown(wait=True),
                lambda: self.telemetry.shutdown(wait=True),
            ):
                try:
                    cleanup()
                except BaseException as error:
                    errors.append(error)
        if self.telemetry_errors:
            errors.append(RuntimeError("HTTP timing journals failed: " + "; ".join(self.telemetry_errors[:3])))
        if errors:
            raise RuntimeError("Environment cleanup failed: " + "; ".join(repr(error) for error in errors)) from errors[
                0
            ]


def application(service):
    app = web.Application(middlewares=[service.timing_middleware])
    app.router.add_get("/health", service.health)
    app.router.add_post("/admin/create", service.create)
    app.router.add_post("/admin/close/{session}", service.close)
    app.router.add_post("/admin/budget/{session}/start", service.start_budget)
    app.router.add_get("/admin/budget/{session}", service.observe_budget)
    app.router.add_get("/session/{session}/observe", service.observe)
    app.router.add_post("/session/{session}/step", service.step)
    app.router.add_get("/session/{session}/requests/{request_id}", service.query_request)
    app.on_cleanup.append(service.close_all)
    return app


async def serve(args):
    records, _ = (
        core.load_catalog(args.catalog_path) if args.catalog_path else (core.build_catalog(args.benchmark_root), None)
    )
    service = ParallelGameService(
        records,
        args.benchmark_root,
        args.run_root,
        max_steps=args.max_steps,
        allowed_splits=args.allowed_splits,
        budget_seconds=args.budget_seconds,
        process_count=args.environment_processes,
        io_workers=args.io_workers,
    )
    await service.engines.start()
    runner = web.AppRunner(application(service))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", args.port)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    settings = {
        "schema_version": 1,
        "url": f"http://127.0.0.1:{port}",
        "admin_token": service.admin_token,
        "catalog_digest": service.catalog_digest,
        "allowed_splits": list(args.allowed_splits),
        "max_steps": args.max_steps,
        "budget_policy_version": core.BUDGET_POLICY_VERSION,
        "budget_seconds": args.budget_seconds,
        "pid": os.getpid(),
        "environment_policy": POLICY,
        "environment_processes": args.environment_processes,
        "io_workers": args.io_workers,
        "environment_worker_pids": service.engines.pids,
        "io_concurrency": args.io_workers,
        "environment_extension": service.extension_identity,
        "action_request_ids": True,
        "action_request_protocol": "alfworld-step-request-id-v1",
        "server_instance_id": service.server_instance_id,
    }
    args.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(args.config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        await runner.cleanup()
        raise RuntimeError("Service config already exists; choose a new path") from None
    with os.fdopen(descriptor, "w") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stop.set)
    print(
        json.dumps(
            {
                "event": "ALFWORLD_PARALLEL_SERVICE_READY",
                "port": port,
                "environment_processes": args.environment_processes,
                "worker_pids": service.engines.pids,
            }
        ),
        flush=True,
    )
    try:
        await stop.wait()
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--allowed-splits", nargs="+", choices=core.OFFICIAL_SPLITS, default=list(core.OFFICIAL_SPLITS))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--environment-processes", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if min(args.max_steps, args.budget_seconds, args.environment_processes, args.io_workers) < 1:
        parser.error("Step, budget, process and I/O counts must be positive")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
