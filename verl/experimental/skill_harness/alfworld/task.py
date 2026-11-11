"""Native Uni-Agent Codex Task with authoritative ALFWorld engine scoring."""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import signal
import time
import uuid
from pathlib import Path

import aiohttp
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.local import LocalSandbox
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .data import TASK_NAME
from .protocol import (
    agent_messages,
    authoritative_reward,
    new_attempt_directory,
    validate_game_reference,
    validate_loopback_url,
)


class AlfworldUserSandbox(LocalSandbox):
    """Use the existing non-root benchmark account and retain partial Codex logs."""

    CAPTURE_DRAIN_TIMEOUT_SECONDS = 10.0
    TERMINATION_TIMEOUT_SECONDS = 10.0

    def __init__(self, session_url: str, output: Path):
        self.session_url = session_url
        self.output = output
        self.account = pwd.getpwnam("alfworld_eval")

    async def write_file(self, path, content):
        await super().write_file(path, content)
        os.chown(path, self.account.pw_uid, self.account.pw_gid)
        os.chmod(path, 0o600)

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
        process_env = {
            "PATH": "/opt/alfworld-tools:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "ALFWORLD_SESSION_URL": self.session_url,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
        for name in (
            "ALFWORLD_CODEX_PACKAGE_ROOT",
            "CODEX_NATIVE_BIN",
            "ALFWORLD_SOLVER_INSTRUCTIONS_FILE",
        ):
            if os.environ.get(name):
                target = "CODEX_MANAGED_PACKAGE_ROOT" if name == "ALFWORLD_CODEX_PACKAGE_ROOT" else name
                process_env[target] = os.environ[name]
        # Uni-Agent supplies the isolated CODEX_HOME and local gateway bindings.
        # The environment service administration capability must remain host-only.
        for name, value in (env or {}).items():
            if name not in ("ALFWORLD_SERVER_CONFIG", "ALFWORLD_ADMIN_TOKEN"):
                process_env[name] = value
        proc = await asyncio.create_subprocess_exec(
            "/usr/sbin/runuser",
            "-u",
            "alfworld_eval",
            "--",
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=process_env,
            start_new_session=True,
        )

        started = time.time()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        termination_signals = []

        async def capture(reader, channel, filename):
            with (self.output / filename).open("ab") as stream:
                while True:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    buffers[channel].extend(chunk)
                    stream.write(chunk)
                    stream.flush()

        pumps = [
            asyncio.create_task(capture(proc.stdout, "stdout", "codex-events.jsonl")),
            asyncio.create_task(capture(proc.stderr, "stderr", "codex-stderr.log")),
        ]

        async def wait_for_parent_exit():
            # asyncio Process.wait may wait for inherited pipes as well as the
            # direct child. Its returncode is set by the child watcher at exit;
            # observe that fact independently, then bound pipe draining below.
            while proc.returncode is None:
                await asyncio.sleep(0.01)

        def signal_group(value):
            try:
                os.killpg(proc.pid, value)
                termination_signals.append(signal.Signals(value).name)
            except ProcessLookupError:
                pass

        async def terminate():
            signal_group(signal.SIGTERM)
            try:
                await asyncio.wait_for(wait_for_parent_exit(), timeout=self.TERMINATION_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                signal_group(signal.SIGKILL)
                await wait_for_parent_exit()
            # The parent may already have exited while a descendant retains
            # the pipes. Do not leave an owned background process alive merely
            # because waiting for the direct child returned immediately.
            await asyncio.sleep(0.01)
            if not proc.stdout.at_eof() or not proc.stderr.at_eof():
                signal_group(signal.SIGKILL)

        timed_out = False
        cancelled = False
        drain_timed_out = False
        try:
            await asyncio.wait_for(wait_for_parent_exit(), timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await terminate()
        except asyncio.CancelledError:
            cancelled = True
            await terminate()
        draining = asyncio.gather(*pumps)
        try:
            await asyncio.wait_for(draining, timeout=self.CAPTURE_DRAIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # A descendant holding an inherited pipe cannot stall the runner.
            drain_timed_out = True
            await terminate()
        except asyncio.CancelledError:
            cancelled = True
            await terminate()
        # Cancellation may finish the gathering future after wait_for raises.
        # Retrieve its result so no orphaned CancelledError reaches the loop.
        await asyncio.gather(draining, return_exceptions=True)
        diagnostics = {
            "schema_version": 1,
            "pid": proc.pid,
            "started_at_epoch": started,
            "finished_at_epoch": time.time(),
            "native_exit_code": proc.returncode,
            "returned_exit_code": None if cancelled else -1 if timed_out else proc.returncode,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "drain_timed_out": drain_timed_out,
            "stdout_bytes": len(buffers["stdout"]),
            "stderr_bytes": len(buffers["stderr"]),
            "termination_signals": termination_signals,
        }
        # Capture failures are host diagnostics, never invented native stderr.
        with (self.output / "native-capture-diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(diagnostics, sort_keys=True) + "\n")
        if cancelled:
            raise asyncio.CancelledError
        return ExecResult(
            exit_code=-1 if timed_out else proc.returncode,
            stdout=bytes(buffers["stdout"]).decode(errors="replace"),
            stderr=bytes(buffers["stderr"]).decode(errors="replace"),
        )


@register_task(TASK_NAME)
class AlfworldSkillTrainingTask(Task):
    config_model = TaskConfig

    async def run(self):
        settings_path = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
        settings = json.loads(settings_path.read_text())
        base = validate_loopback_url(settings["url"])
        metadata = self.config.metadata
        if metadata.get("catalog_digest") != settings.get("catalog_digest"):
            raise ValueError("Task data and ALFWorld service catalogs differ")
        if metadata.get("split") not in settings.get("allowed_splits", []):
            raise ValueError("The environment service excludes this official task split")
        account = pwd.getpwnam("alfworld_eval")
        attempt_id, output = new_attempt_directory(Path(metadata["run_dir"]), metadata["game_id"])
        started = time.time()
        identity = {
            key: metadata[key]
            for key in (
                "game_id",
                "stable_game_id",
                "split",
                "dataset_phase",
                "task_type",
                "scene_id",
                "relative_game_path",
                "game_sha256",
                "catalog_digest",
            )
        }
        for key in (
            "sample_index",
            "role",
            "chain_id",
            "skill_snapshot_id",
            "skill_context_version",
            "skill_context_group_id",
            "parameter_version",
            "global_step",
        ):
            if key in metadata:
                identity[key] = metadata[key]
        manifest = {**identity, "attempt_id": attempt_id, "started_at_epoch": started}
        (output / "attempt.json").write_text(json.dumps(manifest, indent=2) + "\n")
        headers = {"X-Admin-Token": settings["admin_token"]}
        payload = {
            **{
                key: metadata[key]
                for key in ("game_id", "relative_game_path", "game_sha256", "split", "catalog_digest")
            },
            "output_dir": str(output),
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as client:
            async with client.post(base + "/admin/create", headers=headers, json=payload) as response:
                response.raise_for_status()
                created = await response.json()
            session_id = created["session_id"]
            workdir = None
            agent_info = {}
            error = None
            skill_context = None
            try:
                if created.get("catalog_digest") != metadata["catalog_digest"]:
                    raise ValueError("The environment response has a different catalog digest")
                work_root = Path("/tmp/alfworld-skill-episodes")
                work_root.mkdir(mode=0o711, exist_ok=True)
                if work_root.is_symlink() or work_root.stat().st_uid != os.geteuid():
                    raise ValueError("The shared episode parent must be a host-owned directory")
                os.chmod(work_root, 0o711)
                workdir = work_root / uuid.uuid4().hex
                workdir.mkdir(mode=0o700)
                os.chown(workdir, account.pw_uid, account.pw_gid)
                messages = agent_messages(
                    self.config.prompt, created["state"], instruction=metadata.get("instruction")
                )
                if metadata.get("skill_context_options", {}).get("enabled") is True:
                    from ..context import enrich_task_messages

                    enriched = enrich_task_messages(messages, metadata)
                    messages, skill_context = enriched.messages, enriched.manifest
                    (output / "skill-context.json").write_text(json.dumps(skill_context, indent=2) + "\n")
                (output / "agent-messages.json").write_text(json.dumps(messages, indent=2) + "\n")
                (output / "agent-prompt.txt").write_text("\n\n".join(m["content"] for m in messages) + "\n")
                sandbox = AlfworldUserSandbox(base + "/session/" + session_id, output)
                agent_result = await self.build_agent().run(sandbox=sandbox, messages=messages, workdir=str(workdir))
                agent_info = agent_result.info or {}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    async with client.post(base + "/admin/close/" + session_id, headers=headers) as response:
                        response.raise_for_status()
                        engine = await response.json()
                finally:
                    # Completed workspaces must not become memory for later runs.
                    if workdir is not None:
                        os.chown(workdir, 0, 0)
                        os.chmod(workdir, 0o700)
            validate_game_reference(engine, metadata, metadata["catalog_digest"])
            reward = authoritative_reward(engine)
            result = {
                **manifest,
                **engine,
                "reward": reward,
                "accuracy": reward,
                "harness": "Uni-Agent native CodexAgent",
                "agent_ok": bool(agent_info.get("ok")),
                "agent_error": error or agent_info.get("error"),
                "agent_exit_code": agent_info.get("exit_code"),
                "agent_exit_status": agent_info.get("exit_status"),
                "wall_seconds_total": time.time() - started,
                "workdir": str(workdir) if workdir is not None else None,
                **({"skill_context": skill_context} if skill_context is not None else {}),
            }
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("ALFWORLD_EPISODE_RESULT " + json.dumps(result), flush=True)
            return TaskResult(
                reward=reward,
                accuracy=reward,
                finished=True,
                extra_info={
                    **identity,
                    "attempt_id": attempt_id,
                    "episode_output_dir": str(output),
                    "resolved": engine["won"],
                    "steps": engine["steps"],
                    "invalid_actions": engine["invalid_actions"],
                    "agent_ok": result["agent_ok"],
                    "error": result["agent_error"],
                    "eval_completed": error is None,
                    "eval_execution_time": result["wall_seconds_total"],
                    **({"skill_context": skill_context} if skill_context is not None else {}),
                },
            )
