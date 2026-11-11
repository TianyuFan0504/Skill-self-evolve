"""Authoritative ALFWorld service with a serialized TextWorld engine executor.

Launch this file using the ALFWorld environment Python. File-path execution does
not import verl, Torch, or Ray. Agent sessions only receive public observations;
expert plans, dataset paths and the administration capability stay server-side.
"""

from __future__ import annotations

import argparse
import asyncio
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
    from .data import OFFICIAL_SPLITS, build_catalog, catalog_digest, load_catalog, sha256_file
    from .protocol import public_state, validate_game_reference
else:
    from data import OFFICIAL_SPLITS, build_catalog, catalog_digest, load_catalog, sha256_file
    from protocol import public_state, validate_game_reference


class GameService:
    def __init__(self, records, benchmark_root, run_root, *, max_steps=50, allowed_splits=OFFICIAL_SPLITS):
        self.games = {record.game_id: record for record in records}
        self.catalog_digest = catalog_digest(records)
        self.data_root = Path(benchmark_root).resolve() / "data"
        self.run_root = Path(run_root).resolve()
        self.max_steps = int(max_steps)
        self.allowed_splits = frozenset(allowed_splits)
        self.admin_token = secrets.token_hex(32)
        self.sessions = {}

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
            {"ready": True, "catalog_digest": self.catalog_digest, "max_steps": self.max_steps}
        )

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
            before = self.state(session)
            invalid = action not in before["admissible_commands"]
            state, engine_score, done = await asyncio.to_thread(session["env"].step, action)
            session["state"] = state
            session["steps"] += 1
            session["invalid_actions"] += int(invalid)
            session["won"] = bool(state.get("won", False))
            session["done"] = bool(done or session["won"] or session["steps"] >= self.max_steps)
            after = self.state(session)
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
            return web.json_response(after)

    async def _finish(self, session, *, termination_reason="agent_finished"):
        row = session["game"]
        result = {
            **row.to_dict(),
            "catalog_digest": self.catalog_digest,
            "reward": float(session["won"]),
            "won": session["won"],
            "steps": session["steps"],
            "step_limit": session["steps"] >= self.max_steps and not session["won"],
            "max_steps": self.max_steps,
            "invalid_actions": session["invalid_actions"],
            "wall_seconds": time.time() - session["started"],
            "termination_reason": termination_reason,
        }
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
    )
    app = web.Application()
    app.router.add_get("/health", service.health)
    app.router.add_post("/admin/create", service.create)
    app.router.add_post("/admin/close/{session}", service.close)
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
        json.dumps(
            {"event": "ALFWORLD_SERVICE_READY", "port": port, "catalog_digest": service.catalog_digest}
        ),
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
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
