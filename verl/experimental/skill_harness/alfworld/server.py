"""Authoritative ALFWorld service with a serialized TextWorld engine executor.

Launch this file using the ALFWorld environment Python. File-path execution does
not import verl, Torch, or Ray. Agent sessions only receive public observations;
expert plans, dataset paths and the administration capability stay server-side.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import web

if __package__:
    from .budget import BUDGET_POLICY_VERSION, BudgetOutcomeUnknown, digest, make_registration, write_private_json
    from .data import OFFICIAL_SPLITS, build_catalog, catalog_digest, load_catalog, sha256_file
    from .protocol import public_state, validate_game_reference
else:
    from budget import BUDGET_POLICY_VERSION, BudgetOutcomeUnknown, digest, make_registration, write_private_json
    from data import OFFICIAL_SPLITS, build_catalog, catalog_digest, load_catalog, sha256_file
    from protocol import public_state, validate_game_reference


class GameService:
    def __init__(
        self,
        records,
        benchmark_root,
        run_root,
        *,
        max_steps=50,
        allowed_splits=OFFICIAL_SPLITS,
        budget_seconds=900,
        clock_ns=None,
    ):
        self.games = {record.game_id: record for record in records}
        self.catalog_digest = catalog_digest(records)
        self.data_root = Path(benchmark_root).resolve() / "data"
        self.run_root = Path(run_root).resolve()
        self.max_steps = int(max_steps)
        self.allowed_splits = frozenset(allowed_splits)
        self.admin_token = secrets.token_hex(32)
        self.sessions = {}
        self.budget_seconds = int(budget_seconds)
        if self.budget_seconds <= 0:
            raise ValueError("The declared episode budget must be positive")
        self.clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
        self.server_instance_id = uuid.uuid4().hex

    def authorized(self, request):
        if not secrets.compare_digest(request.headers.get("X-Admin-Token", ""), self.admin_token):
            raise web.HTTPForbidden()

    def state(self, session):
        return public_state(
            session["state"],
            steps=session["steps"],
            max_steps=self.max_steps,
            won=session["won"],
            done=session["done"],
        )

    async def create(self, request):
        self.authorized(request)
        payload = await request.json()
        row = self.games.get(payload.get("game_id"))
        if row is None:
            raise web.HTTPNotFound(text="Unknown ALFWorld game identity")
        if row.split not in self.allowed_splits:
            raise web.HTTPForbidden(text="This service does not allow the requested official split")
        try:
            validate_game_reference(payload, row.to_dict(), self.catalog_digest)
        except ValueError as exc:
            raise web.HTTPConflict(text=str(exc)) from exc
        output = Path(payload["output_dir"]).resolve()
        if self.run_root not in output.parents:
            raise web.HTTPBadRequest(text="Episode output must belong to the configured run root")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (output / "initial-state.json").exists():
            raise web.HTTPConflict(text="Episode output already contains an environment attempt")
        game_path = (self.data_root / row.relative_game_path).resolve()
        if self.data_root not in game_path.parents or sha256_file(game_path) != row.game_sha256:
            raise web.HTTPConflict(text="The game file changed after catalog preparation")

        def initialize():
            import textworld
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

            infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
            env = textworld.start(
                str(game_path), request_infos=infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]
            )
            env.seed(42)
            return env, env.reset()

        env, state = await asyncio.to_thread(initialize)
        session_id = uuid.uuid4().hex
        session = {
            "id": session_id,
            "game": row,
            "output": output,
            "env": env,
            "state": state,
            "steps": 0,
            "won": bool(state.get("won", False)),
            "done": False,
            "invalid_actions": 0,
            "started": time.time(),
            "lock": asyncio.Lock(),
            "budget": None,
        }
        self.sessions[session_id] = session
        initial = self.state(session)
        (output / "initial-state.json").write_text(json.dumps(initial, indent=2) + "\n")
        return web.json_response({"session_id": session_id, "state": initial, "catalog_digest": self.catalog_digest})

    def _session(self, request):
        session = self.sessions.get(request.match_info["session"])
        if session is None:
            raise web.HTTPNotFound()
        return session

    async def observe(self, request):
        session = self._session(request)
        async with session["lock"]:
            return web.json_response(self.state(session))

    async def health(self, _request):
        return web.json_response(
            {
                "ready": True,
                "catalog_digest": self.catalog_digest,
                "max_steps": self.max_steps,
                "budget_policy_version": BUDGET_POLICY_VERSION,
                "budget_seconds": self.budget_seconds,
            }
        )

    async def start_budget(self, request):
        self.authorized(request)
        session = self._session(request)
        payload = await request.json()
        async with session["lock"]:
            try:
                existing = session["budget"]
                registration = make_registration(
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
                        raise BudgetOutcomeUnknown("A registered deadline cannot be replaced or extended")
                    return web.json_response(registration)
                if session["steps"]:
                    raise BudgetOutcomeUnknown("The budget must start before any engine action")
                write_private_json(session["output"] / "budget-registration.json", registration, immutable=True)
                with (session["output"] / "budget-events.jsonl").open("x") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.flush()
                    os.fsync(stream.fileno())
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
            except BudgetOutcomeUnknown as error:
                raise web.HTTPConflict(text=str(error)) from error

    def _budget_snapshot(self, session, *, closed=False, close_reason=None):
        budget = session["budget"]
        if budget is None:
            raise web.HTTPConflict(text="No host episode budget has been registered")
        now = self.clock_ns()
        state = {"won": session["won"], "done": session["done"], "steps": session["steps"], "max_steps": self.max_steps}
        return {
            **budget["registration"],
            "now_ns": now,
            "server_closed": closed,
            "closed_ns": now if closed else None,
            "close_reason": close_reason,
            "deadline_reached": now >= budget["registration"]["deadline_ns"],
            "world_state": state,
            "world_state_sha256": digest(state),
            "admitted_actions": budget["admitted"],
            "rejected_after_deadline": budget["rejected"],
            "last_admitted_ns": budget["last_admitted"],
            "last_completed_ns": budget["last_completed"],
            "last_sequence": budget["sequence"],
            "action_error_count": budget["errors"],
            "events_sha256": sha256_file(session["output"] / "budget-events.jsonl"),
        }

    async def observe_budget(self, request):
        self.authorized(request)
        session = self._session(request)
        async with session["lock"]:
            previous = (session["budget"] or {}).get("ordered_snapshot")
            if previous is not None:
                return web.json_response(previous)
            snapshot = self._budget_snapshot(session)
            if snapshot["deadline_reached"]:
                # The host fence binds this exact ordered view. A retry must
                # not replace its hash with a later clock or rejection count.
                write_private_json(session["output"] / "budget-ordered-state.json", snapshot, immutable=True)
                session["budget"]["ordered_snapshot"] = snapshot
            return web.json_response(snapshot)

    def _budget_event(self, session, *, kind, checked_ns, action, completed_ns=None):
        budget = session["budget"]
        budget["sequence"] += 1
        if kind == "admitted":
            budget["admitted"] += 1
            budget["last_admitted"], budget["last_completed"] = checked_ns, completed_ns
        elif kind == "deadline_rejected":
            budget["rejected"] += 1
        else:
            budget["errors"] += 1
        entry = {
            "schema_version": 1,
            "registration_sha256": budget["registration"]["registration_sha256"],
            "sequence": budget["sequence"],
            "kind": kind,
            "checked_ns": checked_ns,
            "completed_ns": completed_ns,
            "step": session["steps"],
            "action_sha256": hashlib.sha256(action.encode()).hexdigest(),
            "won_after": session["won"],
            "done_after": session["done"],
        }
        with (session["output"] / "budget-events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    async def step(self, request):
        session = self._session(request)
        payload = await request.json()
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip() or len(action) > 512 or "\n" in action:
            raise web.HTTPBadRequest(text="One non-empty single-line environment action is required")
        action = action.strip()
        async with session["lock"]:
            if session["done"]:
                return web.json_response(self.state(session))
            budget = session["budget"]
            if budget is None:
                raise web.HTTPConflict(text="The host must register the episode budget before actions")
            checked = self.clock_ns()
            if checked >= budget["registration"]["deadline_ns"]:
                self._budget_event(session, kind="deadline_rejected", checked_ns=checked, action=action)
                raise web.HTTPConflict(text="The declared episode budget has expired; no action was applied")
            before = self.state(session)
            invalid = action not in before["admissible_commands"]

            def execute():
                # The shared engine executor may have a queue. Admission occurs
                # at actual engine entry, not before waiting for another game.
                entered = self.clock_ns()
                if entered >= budget["registration"]["deadline_ns"]:
                    return entered, None, None
                result = session["env"].step(action)
                return entered, self.clock_ns(), result

            execution = asyncio.create_task(asyncio.to_thread(execute))
            cancelled = False
            while True:
                try:
                    entered, completed, result = await asyncio.shield(execution)
                    break
                except asyncio.CancelledError:
                    cancelled = True
                    if execution.cancelled():
                        self._budget_event(session, kind="engine_error", checked_ns=checked, action=action)
                        raise
                    # Cancellation cannot release the session lock while a
                    # real engine thread can still change this game's state.
                    # Resolve it first and keep the cancellation as unknown.
                except Exception:
                    self._budget_event(session, kind="engine_error", checked_ns=checked, action=action)
                    raise
            if result is None:
                self._budget_event(session, kind="deadline_rejected", checked_ns=entered, action=action)
                if cancelled:
                    self._budget_event(session, kind="handler_cancelled", checked_ns=entered, action=action)
                    raise asyncio.CancelledError
                raise web.HTTPConflict(text="The declared episode budget expired before engine admission")
            state, engine_score, done = result
            session["state"] = state
            session["steps"] += 1
            session["invalid_actions"] += int(invalid)
            session["won"] = bool(state.get("won", False))
            session["done"] = bool(done or session["won"] or session["steps"] >= self.max_steps)
            after = self.state(session)
            self._budget_event(session, kind="admitted", checked_ns=entered, completed_ns=completed, action=action)
            entry = {
                "step": session["steps"],
                "action": action,
                "action_in_exposed_commands": not invalid,
                "before": before,
                "after": after,
                "engine_score": float(engine_score),
                "time": time.time(),
            }
            with (session["output"] / "environment-trace.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            if cancelled:
                self._budget_event(session, kind="handler_cancelled", checked_ns=completed, action=action)
                raise asyncio.CancelledError
            return web.json_response(after)

    async def _finish(self, session, *, termination_reason="agent_finished"):
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
            result["episode_budget"] = self._budget_snapshot(session, closed=True, close_reason=termination_reason)
            write_private_json(session["output"] / "budget-environment.json", result["episode_budget"])
        (session["output"] / "engine-result.json").write_text(json.dumps(result, indent=2) + "\n")
        await asyncio.to_thread(session["env"].close)
        self.sessions.pop(session["id"], None)
        return result

    async def close(self, request):
        self.authorized(request)
        session = self._session(request)
        async with session["lock"]:
            return web.json_response(await self._finish(session))

    async def close_all(self, _app):
        for session in list(self.sessions.values()):
            async with session["lock"]:
                await self._finish(session, termination_reason="server_shutdown")


async def serve(args):
    # TextWorld's shared TatSu parser is not thread-safe. Every engine operation
    # must use this one executor thread, even when HTTP/LLM sessions overlap.
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    loop.set_default_executor(executor)
    if args.catalog_path:
        records, _ = load_catalog(args.catalog_path)
    else:
        records = build_catalog(args.benchmark_root)
    service = GameService(
        records,
        args.benchmark_root,
        args.run_root,
        max_steps=args.max_steps,
        allowed_splits=args.allowed_splits,
        budget_seconds=args.budget_seconds,
    )
    app = web.Application()
    app.router.add_get("/health", service.health)
    app.router.add_post("/admin/create", service.create)
    app.router.add_post("/admin/close/{session}", service.close)
    app.router.add_post("/admin/budget/{session}/start", service.start_budget)
    app.router.add_get("/admin/budget/{session}", service.observe_budget)
    app.router.add_get("/session/{session}/observe", service.observe)
    app.router.add_post("/session/{session}/step", service.step)
    app.on_cleanup.append(service.close_all)
    runner = web.AppRunner(app)
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
        "budget_policy_version": BUDGET_POLICY_VERSION,
        "budget_seconds": args.budget_seconds,
        "pid": os.getpid(),
    }
    args.config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        fd = os.open(args.config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        await runner.cleanup()
        raise RuntimeError("Service config already exists; choose a new path") from None
    with os.fdopen(fd, "w") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    print(
        json.dumps({"event": "ALFWORLD_SERVICE_READY", "port": port, "catalog_digest": service.catalog_digest}),
        flush=True,
    )
    try:
        await stop.wait()
    finally:
        await runner.cleanup()
        executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--allowed-splits", nargs="+", choices=OFFICIAL_SPLITS, default=list(OFFICIAL_SPLITS))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--budget-seconds", type=int, default=900)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.budget_seconds < 1:
        parser.error("--budget-seconds must be positive")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
