"""Native Uni-Agent Codex Task with authoritative ALFWorld engine scoring."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import tarfile
import time
import uuid
from pathlib import Path

import aiohttp
from uni_agent.sandbox.base import ExecResult
from uni_agent.sandbox.local import LocalSandbox
from uni_agent.tasks.base import Task, TaskConfig, TaskResult
from uni_agent.tasks.registry import register_task

from .accounts import (
    POLICY_VERSION,
    AccountIsolationError,
    AccountPool,
    decode_launcher_proof,
    launch_arguments,
    record_isolation_incident,
)
from .data import TASK_NAME
from .protocol import (
    agent_messages,
    authoritative_reward,
    new_attempt_directory,
    validate_game_reference,
    validate_loopback_url,
)
from .terminal_guard import _validated_state, monitor_terminal_process


async def _register_native_budget(
    client, *, base, session_id, headers, binding, account, run_root, output, timeout, budget_seconds
):
    """Host-only authenticated registration immediately before native Popen."""
    from .budget import BINDING_KEYS, validate_budget_registration

    if type(budget_seconds) is not int or budget_seconds <= 0 or isinstance(timeout, bool) or timeout != budget_seconds:
        raise AccountIsolationError("The native timeout and declared server budget must agree")
    payload = {
        "schema_version": 1,
        "budget_seconds": budget_seconds,
        "binding": {key: getattr(binding, key) for key in BINDING_KEYS},
        "account": account,
    }
    async with client.post(base + "/admin/budget/" + session_id + "/start", headers=headers, json=payload) as response:
        response.raise_for_status()
        registration = await response.json()
    return validate_budget_registration(
        registration,
        payload=payload,
        session_id=session_id,
        run_root=run_root,
        output=output,
        budget_seconds=budget_seconds,
    )


async def _observe_native_budget(client, *, base, session_id, headers):
    """Read the private server-ordered world/deadline view, never a model message."""
    async with client.get(base + "/admin/budget/" + session_id, headers=headers) as response:
        response.raise_for_status()
        return await response.json()


class PairedWorkspace:
    """Serialize one host-authorized pair and archive/reset its identical cwd.

    The marker and lock live outside the agent-readable directory. An active or
    failed marker is never silently recovered: host reconciliation is required.
    Only this group's directory and its freshly provisioned Codex homes are
    archived; symlinks inside archives are preserved without following targets.
    """

    def __init__(self, *, group, variant, work_root, output, agent_uid, agent_gid):
        if not isinstance(group, str) or not re.fullmatch(r"[0-9a-f]{64}", group):
            raise ValueError("credit_probe_group must be a 64-character lowercase SHA256")
        if variant not in ("with_skill", "without_skill"):
            raise ValueError("credit_probe_variant must identify one assigned paired branch")
        self.group, self.variant = group, variant
        self.work_root, self.output = Path(work_root), Path(output)
        if (
            not self.work_root.is_absolute()
            or self.work_root.is_symlink()
            or not self.work_root.is_dir()
            or self.work_root.stat().st_uid != os.geteuid()
            or self.output.resolve().is_relative_to(self.work_root.resolve())
        ):
            raise ValueError(
                "Paired cwd requires an absolute host-owned parent and an external private archive directory"
            )
        self.workdir = self.work_root / ("credit_" + group)
        self.agent_uid, self.agent_gid = agent_uid, agent_gid
        self.host_uid, self.host_gid = os.geteuid(), os.getegid()
        self.lock = None
        self.active = False
        self.provenance = {
            "schema_version": 1,
            "kind": "paired_empty_workdir",
            "credit_probe_group": group,
            "variant": variant,
            "workdir": str(self.workdir),
            "host_uid": self.host_uid,
            "agent_uid": agent_uid,
            "agent_gid": agent_gid,
            "empty_before_run": False,
            "cleanup_completed": False,
            "archives": [],
        }

    @staticmethod
    def _json(path, value):
        stage = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with stage.open("x", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(value, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(stage, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            stage.unlink(missing_ok=True)

    def _release(self):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            os.close(self.lock)
            self.lock = None

    def acquire(self):
        marker_written = False
        try:
            control = self.work_root / ".credit-control"
            control.mkdir(mode=0o700, exist_ok=True)
            if control.is_symlink() or not control.is_dir() or control.stat().st_uid != self.host_uid:
                raise ValueError("Paired workspace controls must be a private host-owned directory")
            os.chmod(control, 0o700)
            lock_path = control / (self.group + ".lock")
            self.lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            info = os.fstat(self.lock)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != self.host_uid or info.st_nlink != 1:
                raise ValueError("Paired workspace lock has an unexpected owner or file identity")
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.marker_path = control / (self.group + ".json")
            expected = {
                "schema_version": 1,
                "group": self.group,
                "host_uid": self.host_uid,
                "agent_uid": self.agent_uid,
                "agent_gid": self.agent_gid,
                "workdir": str(self.workdir),
            }
            exists = self.marker_path.exists() or self.marker_path.is_symlink()
            if exists:
                if self.marker_path.is_symlink() or self.marker_path.stat().st_uid != self.host_uid:
                    raise ValueError("Paired workspace ownership marker has an unexpected UID")
                marker = json.loads(self.marker_path.read_text())
                if any(marker.get(key) != value for key, value in expected.items()):
                    raise ValueError("Paired workspace ownership marker differs from this host/agent UID")
                if marker.get("status") != "idle":
                    raise RuntimeError("An active or failed paired workspace requires explicit reconciliation")
            else:
                marker = {}
            self.used_codex_homes = list(marker.get("used_codex_homes", []))
            if self.workdir.exists() or self.workdir.is_symlink():
                if not exists or self.workdir.is_symlink() or not self.workdir.is_dir():
                    raise ValueError("Refusing an unowned or redirected paired workspace")
                if self.workdir.stat().st_uid not in (self.host_uid, self.agent_uid) or any(self.workdir.iterdir()):
                    raise ValueError("A paired branch must start with an owned, completely empty workspace")
            self.marker = expected
            self._json(
                self.marker_path,
                {
                    **expected,
                    "status": "active",
                    "variant": self.variant,
                    "used_codex_homes": self.used_codex_homes,
                },
            )
            marker_written = True
            self.workdir.mkdir(mode=0o700, exist_ok=True)
            os.chmod(self.workdir, 0o700)
            os.chown(self.workdir, self.agent_uid, self.agent_gid)
            self.provenance["empty_before_run"] = True
            self.active = True
            self._json(self.output / "paired-workdir.json", self.provenance)
            return self.workdir
        except BaseException:
            try:
                if marker_written:
                    self._json(self.marker_path, {**self.marker, "status": "failed", "variant": self.variant})
            finally:
                self._release()
            raise

    def _archive_remove(self, path, archive_name, *, codex_home=False):
        if path.is_symlink() or not path.is_dir() or path.stat().st_uid not in (self.host_uid, self.agent_uid):
            raise ValueError("Refusing to archive an unexpected workspace/home owner or symlink")
        os.chown(path, self.host_uid, self.host_gid)
        os.chmod(path, 0o700)
        archive = self.output / archive_name
        with tarfile.open(archive, "x", dereference=False) as target:
            target.add(path, arcname="codex-home" if codex_home else "workspace", recursive=True)
        os.chmod(archive, 0o600)
        hasher = hashlib.sha256()
        with archive.open("rb") as source:
            while chunk := source.read(1 << 20):
                hasher.update(chunk)
            os.fsync(source.fileno())
        record = {"source": str(path), "path": str(archive), "sha256": hasher.hexdigest(), "removed": False}
        self.provenance["archives"].append(record)
        shutil.rmtree(path)
        record["removed"] = True

    def retain(self, reason):
        """Unlock host control without reading/removing a possibly live writer's files."""
        if not self.active:
            return
        try:
            self.provenance.update(cleanup_completed=False, cleanup_error=str(reason), sources_retained=True)
            self._json(self.output / "paired-workdir.json", self.provenance)
            self._json(self.marker_path, {**self.marker, "status": "failed", "variant": self.variant})
        finally:
            self.active = False
            self._release()

    def finish(self, codex_homes=(), *, model_catalogs=(), account_lease_id=None):
        if not self.active:
            raise RuntimeError("A paired workspace must be acquired before cleanup")
        try:
            homes = list(dict.fromkeys(str(path) for path in codex_homes))
            self.provenance["codex_homes"] = homes
            if len(homes) != 1:
                raise ValueError("A native paired branch must own exactly one separately provisioned Codex home")
            if any(home in self.used_codex_homes for home in homes):
                raise ValueError("Paired branches must use distinct native Codex homes")
            for value in homes:
                path = Path(value)
                if path.parent != Path("/tmp") or not re.fullmatch(r"codex-home-[0-9a-f]{32}", path.name):
                    raise ValueError("Refusing a Codex home outside the native per-episode home namespace")
                if path.is_symlink() or not path.is_dir() or path.stat().st_uid not in (self.host_uid, self.agent_uid):
                    raise ValueError("The native Codex home has an unexpected owner or file identity")
            from .native_files import archive_native_files

            native = archive_native_files(
                self.output,
                codex_homes=homes,
                model_catalogs=model_catalogs,
                workspace=self.workdir,
                workspace_root=self.work_root,
                paired_group=self.group,
                agent_uid=self.agent_uid,
                account_lease_id=account_lease_id,
            )
            self.provenance["archives"] = [
                {
                    "source": row["path"],
                    "path": native["archive_path"],
                    "sha256": native["archive_sha256"],
                    "removed": row["removed"],
                }
                for row in native["sources"]
                if row["kind"] in ("workspace", "codex_home")
            ]
            self.provenance.update(cleanup_completed=True, codex_homes=homes, native_files=native)
            self._json(self.output / "paired-workdir.json", self.provenance)
            self._json(
                self.marker_path,
                {
                    **self.marker,
                    "status": "idle",
                    "last_variant": self.variant,
                    "used_codex_homes": [*self.used_codex_homes, *homes],
                },
            )
            return json.loads(json.dumps(self.provenance))
        except BaseException as exc:
            self.provenance["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            self._json(self.output / "paired-workdir.json", self.provenance)
            self._json(self.marker_path, {**self.marker, "status": "failed", "variant": self.variant})
            raise
        finally:
            self.active = False
            self._release()


class AlfworldUserSandbox(LocalSandbox):
    """Apply one exclusive host UID/kernel boundary and retain partial Codex logs."""

    CAPTURE_DRAIN_TIMEOUT_SECONDS = 10.0
    TERMINATION_TIMEOUT_SECONDS = 10.0
    TERMINAL_GRACE_SECONDS = 5.0
    TERMINAL_POLL_SECONDS = 0.25
    TERMINAL_OBSERVATION_TIMEOUT_SECONDS = 2.0

    def __init__(self, session_url: str, output: Path, *, account, account_pool, account_lease, attempt_id):
        self.session_url = session_url
        self.output = output
        self.account = account
        self.account_pool, self.account_lease, self.attempt_id = account_pool, account_lease, attempt_id
        self.native_started = False
        self.launcher_proofs = []
        self.private_tmp_identity = None
        self.native_codex_homes = []
        self.native_model_catalogs = []
        self.terminal_stop_evidence = []
        self.host_stop_evidence = []
        self.backend_binding = None
        self.budget_registration = None
        self.register_native_budget = None
        self.observe_native_budget = None

    def _persist_backend_stop(self, requested_stop_kind, proof):
        evidence = {
            "requested_stop_kind": requested_stop_kind,
            "effective_stop_kind": proof["stop_kind"],
            "binding_id": self.backend_binding.binding_id,
            "binding_sha256": self.backend_binding.binding_sha256,
            "attempt_id": self.backend_binding.attempt_id,
            "recorded_at": time.time(),
            "backend_stop_confirmed": None,
            "model_eos_invented": False,
        }
        self.host_stop_evidence.append(evidence)
        with (self.output / "backend-host-stop-evidence.jsonl").open("a") as stream:
            stream.write(json.dumps(evidence, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return evidence

    def _request_backend_stop(self, stop_kind, *, reason=None):
        from .backend_terminal import fence_host_stop

        if self.backend_binding is None:
            raise AccountIsolationError("Native stop has no trusted backend attempt binding")
        proof = fence_host_stop(self.backend_binding, stop_kind, reason=reason or "native_cli_" + stop_kind)
        return self._persist_backend_stop(stop_kind, proof)

    async def _budget_deadline_stop(self):
        from .backend_terminal import fence_episode
        from .budget import digest, make_budget_stop_reason

        registration = self.budget_registration
        if not registration or not callable(self.observe_native_budget):
            raise AccountIsolationError("A budget stop has no trusted registered server deadline")
        ordered = await self.observe_native_budget()
        state = _validated_state(ordered.get("world_state"))
        if (
            not all(ordered.get(key) == value for key, value in registration.items())
            or ordered.get("server_closed") is not False
            or ordered.get("deadline_reached") is not True
            or type(ordered.get("now_ns")) is not int
            or ordered["now_ns"] < registration["deadline_ns"]
            or state is None
            or ordered.get("world_state_sha256") != digest(ordered["world_state"])
        ):
            raise AccountIsolationError("The private ordered budget snapshot is unconfirmed")
        if state[0]["won"] or state[0]["done"]:
            # The session lock ordered this completed pre-deadline action before
            # the deadline view. Preserve a real win/MDP terminal before stopping.
            proof = fence_episode(self.backend_binding, ordered["world_state"])
            return self._persist_backend_stop("budget_timeout", proof)
        return self._request_backend_stop("budget_timeout", reason=make_budget_stop_reason(ordered))

    async def _observe_terminal_state(self):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.TERMINAL_OBSERVATION_TIMEOUT_SECONDS)
        ) as client:
            async with client.get(self.session_url + "/observe") as response:
                response.raise_for_status()
                return await response.json()

    async def write_file(self, path, content):
        try:
            if not re.fullmatch(r"/tmp/codex-model-catalog-[0-9a-f]{32}\.json", str(path)):
                raise AccountIsolationError("Native host writes require the exact catalog UUID namespace")
            if str(path) not in self.native_model_catalogs:
                self.native_model_catalogs.append(str(path))
            if Path(path).exists() or Path(path).is_symlink():
                raise AccountIsolationError("A native catalog must have a fresh file identity")
            await super().write_file(path, content)
            # Root owns immutable catalog contents; only this private GID may read.
            os.chown(path, os.geteuid(), self.account.pw_gid)
            os.chmod(path, 0o440)
            self.account_pool.register_resource(self.account_lease, self.attempt_id, path, "model_catalog")
        except Exception as error:
            record_isolation_incident(
                self.account_lease["owner"]["run_root"],
                error,
                lease_id=self.account_lease["lease_id"],
                attempt_id=self.attempt_id,
            )
            raise AccountIsolationError("Native immutable catalog preparation failed") from error

    async def _exec(self, argv, *, timeout=None, workdir=None, env=None):
        try:
            homes = []
            if argv[:2] == ["bash", "-c"]:
                import re
                import shlex

                for token in shlex.split(argv[-1]):
                    if re.fullmatch(r"CODEX_HOME=/tmp/codex-home-[0-9a-f]{32}", token):
                        home = token.split("=", 1)[1]
                        if Path(home).exists() or Path(home).is_symlink():
                            raise AccountIsolationError("Native Codex must start with a fresh per-episode home")
                        if not hasattr(self, "native_codex_homes"):
                            self.native_codex_homes = []
                        self.native_codex_homes.append(home)
                        Path(home).mkdir(mode=0o700)
                        os.chown(home, self.account.pw_uid, self.account.pw_gid)
                        self.account_pool.register_resource(self.account_lease, self.attempt_id, home, "codex_home")
                        homes.append(home)
            if len(homes) != 1 or not workdir:
                raise AccountIsolationError("A native launch requires one fresh host-recorded HOME and controlled cwd")
        except Exception as error:
            record_isolation_incident(
                self.account_lease["owner"]["run_root"],
                error,
                lease_id=self.account_lease["lease_id"],
                attempt_id=self.attempt_id,
            )
            raise AccountIsolationError("Native launch resource preparation failed") from error
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
            if name not in (
                "ALFWORLD_SERVER_CONFIG",
                "ALFWORLD_ADMIN_TOKEN",
                "BASH_ENV",
                "ENV",
            ) and not name.startswith(("LD_", "PYTHON")):
                process_env[name] = value
        ready_read, ready_write = os.pipe()
        try:
            child, controlled_env, self.private_tmp_identity = launch_arguments(
                self.account,
                argv,
                home=homes[0],
                cwd=workdir,
                catalogs=self.native_model_catalogs,
                ready_fd=ready_write,
                expected_tmp=self.private_tmp_identity,
            )
            process_env.update(controlled_env)
            if not callable(self.register_native_budget):
                raise AccountIsolationError("The native launcher requires a host budget registration callback")
            self.budget_registration = await self.register_native_budget(timeout)
            if (
                not isinstance(self.budget_registration, dict)
                or type(self.budget_registration.get("deadline_ns")) is not int
                or time.monotonic_ns() >= self.budget_registration["deadline_ns"]
            ):
                raise AccountIsolationError("The registered budget expired or is invalid before native launch")
            native_started_ns = time.monotonic_ns()
            proc = await asyncio.create_subprocess_exec(
                *child,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=process_env,
                start_new_session=True,
                close_fds=True,
                pass_fds=(ready_write,),
            )
            self.native_started = True
        except BaseException as error:
            os.close(ready_read)
            record_isolation_incident(
                self.account_lease["owner"]["run_root"],
                error,
                lease_id=self.account_lease["lease_id"],
                attempt_id=self.attempt_id,
            )
            raise AccountIsolationError("Native isolation launcher failed before execution") from error
        finally:
            os.close(ready_write)

        started = time.time()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        termination_signals = []
        termination_lock = asyncio.Lock()
        termination_finished = False
        isolation_error = None

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
            nonlocal termination_finished
            async with termination_lock:
                if termination_finished:
                    return
                signal_group(signal.SIGTERM)
                try:
                    await asyncio.wait_for(wait_for_parent_exit(), timeout=self.TERMINATION_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    signal_group(signal.SIGKILL)
                    await wait_for_parent_exit()
                await asyncio.sleep(0.01)
                if not proc.stdout.at_eof() or not proc.stderr.at_eof():
                    signal_group(signal.SIGKILL)
                termination_finished = True

        async def stop_owned(stop_kind):
            nonlocal isolation_error
            try:
                if stop_kind == "budget_timeout":
                    await self._budget_deadline_stop()
                else:
                    self._request_backend_stop(stop_kind)  # Durable fence BEFORE the first CLI signal.
            except Exception as error:
                isolation_error = error
                try:
                    self._request_backend_stop("host_error", reason="budget_or_stop_proof_unconfirmed")
                except Exception as fence_error:
                    error.add_note("Host-error fallback fence failed: " + str(fence_error))
                record_isolation_incident(
                    self.account_lease["owner"]["run_root"],
                    error,
                    lease_id=self.account_lease["lease_id"],
                    attempt_id=self.attempt_id,
                )
            finally:
                await terminate()  # Fence persistence failure is fatal; still stop this owned CLI.

        def preserve_terminal_evidence(evidence):
            value = evidence.to_dict()
            value["backend_stop_confirmed"] = None  # CLI ownership does not attest backend GPU cancellation.
            if not hasattr(self, "terminal_stop_evidence"):
                self.terminal_stop_evidence = []
            self.terminal_stop_evidence.append(value)
            with (self.output / "terminal-stop-evidence.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
                stream.flush()

        monitor = asyncio.create_task(
            monitor_terminal_process(
                self._observe_terminal_state,
                lambda: proc.returncode is None,
                terminate,
                process_id=proc.pid,
                grace_seconds=self.TERMINAL_GRACE_SECONDS,
                poll_seconds=self.TERMINAL_POLL_SECONDS,
                observation_timeout_seconds=self.TERMINAL_OBSERVATION_TIMEOUT_SECONDS,
                on_evidence=preserve_terminal_evidence,
            )
        )

        timed_out = False
        cancelled = False
        drain_timed_out = False
        try:
            raw = await asyncio.wait_for(asyncio.to_thread(os.read, ready_read, 65536), timeout=10)
            proof = decode_launcher_proof(raw, uid=self.account.pw_uid, gid=self.account.pw_gid)
            self.account_pool.mark_launcher(self.account_lease, self.attempt_id, proof)
            self.launcher_proofs.append(proof)
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                cancelled = True
                await stop_owned("task_cancel")
            else:
                isolation_error = error
                record_isolation_incident(
                    self.account_lease["owner"]["run_root"],
                    error,
                    lease_id=self.account_lease["lease_id"],
                    attempt_id=self.attempt_id,
                )
                await stop_owned("host_error")
        finally:
            os.close(ready_read)
        try:
            remaining = max(0.0, (self.budget_registration["deadline_ns"] - time.monotonic_ns()) / 1_000_000_000)
            await asyncio.wait_for(wait_for_parent_exit(), timeout=remaining)
        except asyncio.TimeoutError:
            timed_out = True
            await stop_owned("budget_timeout")
        except asyncio.CancelledError:
            cancelled = True
            await stop_owned("task_cancel")
        draining = asyncio.gather(*pumps)
        try:
            await asyncio.wait_for(draining, timeout=self.CAPTURE_DRAIN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # A descendant holding an inherited pipe cannot stall the runner.
            drain_timed_out = True
            await stop_owned("host_error")
        except asyncio.CancelledError:
            cancelled = True
            await stop_owned("task_cancel")
        except Exception as error:
            isolation_error = error
            await stop_owned("host_error")
        # Cancellation may finish the gathering future after wait_for raises.
        # Retrieve its result so no orphaned CancelledError reaches the loop.
        await asyncio.gather(draining, return_exceptions=True)
        if cancelled or timed_out:
            monitor.cancel()
        monitor_result = (await asyncio.gather(monitor, return_exceptions=True))[0]
        terminal_monitor_error = (
            type(monitor_result).__name__
            if isinstance(monitor_result, BaseException) and not isinstance(monitor_result, asyncio.CancelledError)
            else None
        )
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
            "terminal_monitor_error": terminal_monitor_error,
            "backend_stop_confirmed": None,
            "host_stop_evidence": list(getattr(self, "host_stop_evidence", [])),
            "declared_budget_seconds": self.budget_registration.get("budget_seconds"),
            "budget_registration_sha256": self.budget_registration.get("registration_sha256"),
            "budget_started_ns": self.budget_registration.get("started_ns"),
            "budget_deadline_ns": self.budget_registration["deadline_ns"],
            "native_started_ns": native_started_ns,
        }
        # Capture failures are host diagnostics, never invented native stderr.
        with (self.output / "native-capture-diagnostics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(diagnostics, sort_keys=True) + "\n")
        if cancelled:
            raise asyncio.CancelledError
        if isolation_error is not None:
            raise AccountIsolationError(
                "Native kernel isolation was not attested; learner must stop"
            ) from isolation_error
        return ExecResult(
            exit_code=-1 if timed_out else proc.returncode,
            stdout=bytes(buffers["stdout"]).decode(errors="replace"),
            stderr=bytes(buffers["stderr"]).decode(errors="replace"),
        )


@register_task(TASK_NAME)
class AlfworldSkillTrainingTask(Task):
    config_model = TaskConfig

    @staticmethod
    def _trusted_run_root():
        # These values come from the trusted launcher/Ray runtime environment,
        # never task messages or model-generated metadata.
        declared = os.environ.get("SKILL_RUN_DIR")
        if declared:
            root = Path(declared)
        else:
            settings = Path(os.environ.get("ALFWORLD_SERVER_CONFIG", ""))
            if settings.name != "environment.json" or settings.parent.name != "runtime":
                raise AccountIsolationError("A trusted host SKILL_RUN_DIR is required for Task failure evidence")
            root = settings.parent.parent
        if not root.is_absolute() or root.is_symlink():
            raise AccountIsolationError("The trusted host run root must be an absolute non-symlink directory")
        return root.resolve(strict=True)

    async def run(self):
        from .budget import BudgetOutcomeUnknown

        trusted_root = self._trusted_run_root()
        try:
            if Path(self.config.metadata["run_dir"]).resolve() != trusted_root:
                raise AccountIsolationError("Task metadata run_dir differs from the trusted host run root")
            return await self._run_episode()
        except asyncio.CancelledError:
            raise
        except BudgetOutcomeUnknown as budget_error:
            if (
                self.config.metadata.get("role") == "counterfactual_probe"
                and getattr(budget_error, "safely_closed_budget_censor", False) is True
            ):
                raise  # Only a host-attested, safely closed CF unknown may be censored per case.
            record_isolation_incident(trusted_root, "Unobserved solver budget outcome: " + str(budget_error))
            raise AccountIsolationError(
                "An unobserved budget outcome forbids partial GRPO-group training"
            ) from budget_error
        except Exception as error:
            # The official adapter may filter a failed Task. A durable host
            # barrier must survive that filtering and forbid learner updates.
            record_isolation_incident(trusted_root, f"Task host exception: {type(error).__name__}: {error}")
            raise

    async def _run_episode(self):
        settings_path = Path(os.environ["ALFWORLD_SERVER_CONFIG"])
        settings = json.loads(settings_path.read_text())
        base = validate_loopback_url(settings["url"])
        metadata = self.config.metadata
        if metadata.get("catalog_digest") != settings.get("catalog_digest"):
            raise ValueError("Task data and ALFWorld service catalogs differ")
        if metadata.get("split") not in settings.get("allowed_splits", []):
            raise ValueError("The environment service excludes this official task split")
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
            "rollout_uid",
            "session_index",
            "evaluation_audit_id",
            "evaluation_stage",
            "role",
            "chain_id",
            "skill_snapshot_id",
            "skill_context_version",
            "skill_context_group_id",
            "parameter_version",
            "global_step",
            "credit_probe_group",
            "credit_probe_variant",
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
            native_finished = None
            error = None
            skill_context = None
            paired_workspace = None
            workspace_provenance = None
            native_files = None
            sandbox = None
            backend_binding = backend_receipt = None
            backend_error = None
            gateway_session_id = None
            account = account_pool = account_lease = None
            account_summary = None
            isolation_error = None
            agent_started = False
            uid_drained = False

            def request_host_stop(stop_kind, reason):
                nonlocal isolation_error
                if backend_binding is None:
                    return
                try:
                    from .backend_terminal import fence_host_stop

                    fence_host_stop(backend_binding, stop_kind, reason=reason)
                except Exception as stop_error:
                    isolation_error = AccountIsolationError("Backend host-stop fence failed: " + str(stop_error))
                    record_isolation_incident(
                        metadata["run_dir"],
                        isolation_error,
                        lease_id=account_lease.get("lease_id") if account_lease else None,
                        attempt_id=attempt_id,
                    )
                    raise

            try:
                if created.get("catalog_digest") != metadata["catalog_digest"]:
                    raise ValueError("The environment response has a different catalog digest")
                account_pool = AccountPool()
                if metadata.get("role") == "counterfactual_probe":
                    candidate_lease = metadata.get("host_account_lease")
                    if (
                        not isinstance(candidate_lease, dict)
                        or not all(
                            key in candidate_lease for key in ("lease_id", "token", "uid", "gid", "name", "owner")
                        )
                        or candidate_lease.get("owner")
                        != {
                            "run_root": str(Path(metadata["run_dir"]).resolve()),
                            "scope_id": metadata.get("credit_probe_group"),
                            "kind": "pair",
                        }
                    ):
                        raise AccountIsolationError("A paired Task requires its coordinator's exclusive UID lease")
                    variant = metadata.get("credit_probe_variant")
                    if variant not in ("with_skill", "without_skill"):
                        raise AccountIsolationError("A paired Task requires an explicit valid variant")
                    account_lease = candidate_lease
                else:
                    account_lease = account_pool.acquire(
                        run_root=metadata["run_dir"],
                        scope_id=attempt_id,
                        kind="ordinary",
                    )
                    variant = "ordinary"
                account_pool.begin_attempt(account_lease, attempt_id=attempt_id, output=output, variant=variant)
                account = account_pool.account(account_lease)
                PairedWorkspace._json(
                    output / "account-lease.json",
                    {
                        "policy_version": POLICY_VERSION,
                        **account_pool.public(account_lease),
                        "attempt_id": attempt_id,
                    },
                )
                work_root = Path("/tmp/alfworld-skill-episodes")
                work_root.mkdir(mode=0o711, exist_ok=True)
                if work_root.is_symlink() or work_root.stat().st_uid != os.geteuid():
                    raise ValueError("The shared episode parent must be a host-owned directory")
                os.chmod(work_root, 0o711)
                controlled = any(
                    metadata.get(key) is not None for key in ("credit_probe_group", "credit_probe_variant")
                )
                if metadata.get("role") == "counterfactual_probe":
                    paired_workspace = PairedWorkspace(
                        group=metadata.get("credit_probe_group"),
                        variant=metadata.get("credit_probe_variant"),
                        work_root=work_root,
                        output=output,
                        agent_uid=account.pw_uid,
                        agent_gid=account.pw_gid,
                    )
                    workdir = paired_workspace.acquire()
                elif controlled:
                    raise ValueError("Controlled probe workspace metadata requires the trusted counterfactual role")
                else:
                    workdir = work_root / uuid.uuid4().hex
                    workdir.mkdir(mode=0o700)
                    os.chown(workdir, account.pw_uid, account.pw_gid)
                account_pool.register_resource(account_lease, attempt_id, workdir, "workspace")
                messages = agent_messages(self.config.prompt, created["state"], instruction=metadata.get("instruction"))
                if metadata.get("skill_context_options", {}).get("enabled") is True:
                    from ..context import enrich_task_messages

                    enriched = enrich_task_messages(messages, metadata)
                    messages, skill_context = enriched.messages, enriched.manifest
                    (output / "skill-context.json").write_text(json.dumps(skill_context, indent=2) + "\n")
                (output / "agent-messages.json").write_text(json.dumps(messages, indent=2) + "\n")
                (output / "agent-prompt.txt").write_text("\n\n".join(m["content"] for m in messages) + "\n")
                sandbox = AlfworldUserSandbox(
                    base + "/session/" + session_id,
                    output,
                    account=account,
                    account_pool=account_pool,
                    account_lease=account_lease,
                    attempt_id=attempt_id,
                )
                agent = self.build_agent()
                from urllib.parse import urlsplit

                from .backend_terminal import bind_episode

                gateway_url = agent.config.model.base_url
                parts = urlsplit(gateway_url).path.split("/")
                gateway_session_id = parts[parts.index("sessions") + 1]
                backend_binding = bind_episode(
                    Path(metadata["run_dir"]) / "terminal_bindings",
                    gateway_base_url=gateway_url,
                    environment_session_url=base + "/session/" + session_id,
                    attempt_id=attempt_id,
                )
                sandbox.backend_binding = backend_binding

                async def register_native_budget(timeout):
                    return await _register_native_budget(
                        client,
                        base=base,
                        session_id=session_id,
                        headers=headers,
                        binding=backend_binding,
                        account=account_pool.public(account_lease),
                        run_root=Path(metadata["run_dir"]),
                        output=output,
                        timeout=timeout,
                        budget_seconds=settings.get("budget_seconds", 900),
                    )

                async def observe_native_budget():
                    return await _observe_native_budget(client, base=base, session_id=session_id, headers=headers)

                sandbox.register_native_budget = register_native_budget
                sandbox.observe_native_budget = observe_native_budget
                agent_started = True
                agent_result = await agent.run(sandbox=sandbox, messages=messages, workdir=str(workdir))
                agent_info = agent_result.info or {}
                native_finished = getattr(agent_result, "finished", None)
            except asyncio.CancelledError as cancelled_error:
                try:
                    request_host_stop("task_cancel", "task_cancelled")
                except Exception as stop_error:
                    cancelled_error.add_note("Backend stop fence failed: " + str(stop_error))
                raise
            except AccountIsolationError as exc:
                isolation_error = exc
                error = f"{type(exc).__name__}: {exc}"
                try:
                    request_host_stop("host_error", type(exc).__name__)
                except Exception as stop_error:
                    exc.add_note("Backend stop fence failed: " + str(stop_error))
                record_isolation_incident(
                    metadata["run_dir"],
                    error,
                    lease_id=account_lease.get("lease_id") if account_lease else None,
                    attempt_id=attempt_id,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                phase = "execution" if agent_started else "boundary setup"
                isolation_error = AccountIsolationError("Host attempt " + phase + " failed: " + error)
                try:
                    request_host_stop("host_error", type(exc).__name__)
                except Exception as stop_error:
                    exc.add_note("Backend stop fence failed: " + str(stop_error))
                record_isolation_incident(
                    metadata["run_dir"],
                    isolation_error,
                    lease_id=account_lease.get("lease_id") if account_lease else None,
                    attempt_id=attempt_id,
                )
            finally:
                try:
                    try:
                        if account_pool is not None and account_lease is not None:
                            try:
                                drained = await asyncio.to_thread(account_pool.drain, account_lease)
                                uid_drained = (
                                    isinstance(drained, dict)
                                    and drained.get("drained") is True
                                    and drained.get("uid") == account_lease["uid"]
                                    and drained.get("remaining_pids") == []
                                )
                                if not uid_drained:
                                    raise AccountIsolationError("The UID drain returned no positive closure proof")
                            except Exception as exc:
                                isolation_error = AccountIsolationError("Native UID process drain failed: " + str(exc))
                        if backend_binding is not None:
                            from .backend_terminal import wait_for_terminal_drain

                            try:
                                backend_receipt = await wait_for_terminal_drain(backend_binding, timeout_seconds=15)
                                (output / "backend-terminal-drain.json").write_text(
                                    json.dumps(backend_receipt, sort_keys=True) + "\n"
                                )
                            except Exception as exc:
                                backend_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        async with client.post(base + "/admin/close/" + session_id, headers=headers) as response:
                            response.raise_for_status()
                            engine = await response.json()
                finally:
                    bound = backend_binding
                    raw_drained = bool(
                        isinstance(backend_receipt, dict)
                        and bound is not None
                        and backend_receipt.get("backend_stop_confirmed") is True
                        and backend_receipt.get("active_requests") == 0
                        and backend_receipt.get("attempt_id") == attempt_id
                        and backend_receipt.get("gateway_session_id") == gateway_session_id
                        and backend_receipt.get("environment_session_id") == session_id
                        and backend_receipt.get("binding_sha256")
                        == (
                            bound.get("binding_sha256")
                            if isinstance(bound, dict)
                            else getattr(bound, "binding_sha256", None)
                        )
                    )
                    backend_safe = raw_drained or not agent_started
                    if account_lease is not None and not backend_safe:
                        isolation_error = AccountIsolationError("Native backend drain is unconfirmed; retain sources")
                    sources_safe = uid_drained and backend_safe
                    # Completed workspaces must not become memory for later runs.
                    normal_workspace = None
                    if paired_workspace is not None and paired_workspace.active:
                        try:
                            if not sources_safe:
                                paired_workspace.retain(
                                    "Process/backend drain is unconfirmed; retain every native source"
                                )
                                workspace_provenance = paired_workspace.provenance
                            else:
                                workspace_provenance = paired_workspace.finish(
                                    getattr(sandbox, "native_codex_homes", ()) if sandbox is not None else (),
                                    model_catalogs=getattr(sandbox, "native_model_catalogs", ()) if sandbox else (),
                                    account_lease_id=account_lease["lease_id"],
                                )
                                native_files = workspace_provenance["native_files"]
                        except Exception as exc:
                            error = (error + "; " if error else "") + f"Paired workspace cleanup failed: {exc}"
                            workspace_provenance = paired_workspace.provenance
                            isolation_error = AccountIsolationError(
                                "Paired native resource archive failed: " + str(exc)
                            )
                    elif paired_workspace is None and workdir is not None:
                        if workdir.is_dir() and not workdir.is_symlink():
                            normal_workspace = workdir
                        else:
                            error = (error + "; " if error else "") + "Native workspace missing during cleanup"
                    try:
                        from .native_files import archive_native_files

                        if account is not None and paired_workspace is None and sources_safe:
                            native_files = archive_native_files(
                                output,
                                codex_homes=getattr(sandbox, "native_codex_homes", ()) if sandbox else (),
                                model_catalogs=getattr(sandbox, "native_model_catalogs", ()) if sandbox else (),
                                workspace=normal_workspace,
                                agent_uid=account.pw_uid,
                                account_lease_id=account_lease["lease_id"],
                            )
                    except Exception as exc:
                        # Preserve the primary execution/ENOSPC error and every
                        # unarchived source; cleanup never invents a task reward.
                        error = (error + "; " if error else "") + f"Native file archival failed: {exc}"
                        isolation_error = AccountIsolationError("Native file archival failed: " + str(exc))
                    if account_pool is not None and account_lease is not None:
                        try:
                            if isolation_error is not None:
                                account_pool.quarantine(account_lease, isolation_error, attempt_id=attempt_id)
                            else:
                                if native_files and native_files.get("status") == "archived_and_removed":
                                    account_summary = account_pool.close_attempt(
                                        account_lease,
                                        attempt_id,
                                        receipt_path=output / "native-files.json",
                                        backend_drained=backend_safe,
                                        native_started=bool(sandbox is not None and sandbox.native_started),
                                    )
                                else:
                                    account_pool.close_empty_attempt(
                                        account_lease,
                                        attempt_id,
                                        backend_drained=backend_safe,
                                    )
                        except Exception as exc:
                            isolation_error = AccountIsolationError("Native account closure failed: " + str(exc))
                            record_isolation_incident(
                                metadata["run_dir"],
                                isolation_error,
                                lease_id=account_lease["lease_id"],
                                attempt_id=attempt_id,
                            )
            validate_game_reference(engine, metadata, metadata["catalog_digest"])
            reward = authoritative_reward(engine)
            terminal_stop = (getattr(sandbox, "terminal_stop_evidence", []) or [None])[-1]
            expected_terminal_stop = False
            private_done = engine["won"] or engine.get("step_limit") is True
            trace = output / "environment-trace.jsonl"
            if not private_done and trace.is_file():
                entries = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
                private_done = bool(entries and entries[-1].get("after", {}).get("done") is True)
            if terminal_stop:
                observed = _validated_state(terminal_stop.get("terminal_state"))
                expected_terminal_stop = bool(
                    observed
                    and (observed[0]["done"] or observed[0]["won"])
                    and private_done
                    and terminal_stop.get("status") == "terminal_stop"
                    and terminal_stop.get("termination_requested") is True
                    and terminal_stop.get("termination_completed") is True
                    and terminal_stop.get("alive_after_termination") is False
                    and terminal_stop.get("terminal_state_hash") == observed[1]
                    and observed[0]["won"] == engine["won"]
                    and all(observed[0].get(key) == engine.get(key) for key in ("steps", "max_steps"))
                )

            def bound_field(key):
                if isinstance(backend_binding, dict):
                    return backend_binding.get(key)
                return getattr(backend_binding, key, None)

            backend_confirmed = None
            expected_backend_stop = False
            if isinstance(backend_receipt, dict):
                bound = bool(
                    backend_receipt.get("schema_version") == 1
                    and backend_receipt.get("attempt_id") == attempt_id
                    and backend_receipt.get("gateway_session_id") == gateway_session_id
                    and backend_receipt.get("environment_session_id") == session_id
                    and bound_field("binding_id") is not None
                    and bound_field("binding_sha256") is not None
                    and all(backend_receipt.get(key) == bound_field(key) for key in ("binding_id", "binding_sha256"))
                    and backend_receipt.get("active_requests") == 0
                    and (
                        backend_receipt.get("fenced") is True
                        or (
                            backend_receipt.get("status") == "drained"
                            and agent_info.get("ok") is True
                            and agent_info.get("error") is None
                            and agent_info.get("exit_code") == 0
                            and not expected_terminal_stop
                        )
                    )
                )
                if (
                    bound
                    and backend_receipt.get("backend_stop_confirmed") is True
                    and backend_receipt.get("status")
                    in ("drained", "terminal_refused", "terminal_aborted", "host_stopped")
                ):
                    backend_confirmed = True
                    observed = _validated_state(backend_receipt.get("terminal_state"))
                    expected_backend_stop = bool(
                        observed
                        and private_done
                        and (observed[0]["done"] or observed[0]["won"])
                        and backend_receipt.get("expected_backend_terminal_stop") is True
                        and backend_receipt.get("fenced") is True
                        and backend_receipt.get("status") in ("terminal_refused", "terminal_aborted")
                        and backend_receipt.get("terminal_state_hash") == observed[1]
                        and observed[0]["won"] == engine["won"]
                        and all(observed[0].get(key) == engine.get(key) for key in ("steps", "max_steps"))
                    )
            host_stop_kind = (backend_receipt or {}).get("stop_kind")
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
                "native_finished": native_finished,
                "wall_seconds_total": time.time() - started,
                "workdir": str(workdir) if workdir is not None else None,
                "terminal_stop": terminal_stop,
                "expected_host_terminal_stop": expected_terminal_stop,
                "backend_terminal": backend_receipt,
                "backend_terminal_error": backend_error,
                "backend_stop_confirmed": backend_confirmed,
                "expected_backend_terminal_stop": expected_backend_stop,
                "backend_binding": {
                    key: bound_field(key)
                    for key in (
                        "binding_id",
                        "binding_sha256",
                        "attempt_id",
                        "gateway_session_id",
                        "environment_session_id",
                    )
                }
                if backend_binding is not None
                else None,
                "environment_session_id": session_id,
                "host_stop_kind": host_stop_kind,
                "host_stop_requested": (backend_receipt or {}).get("host_stop_requested", False),
                "budget_exhausted": host_stop_kind == "budget_timeout",
                "task_cancelled": host_stop_kind == "task_cancel",
                "budget_reward_eligible": False,
                "account_isolation": account_summary,
                "account_isolation_error": str(isolation_error) if isolation_error is not None else None,
                **({"skill_context": skill_context} if skill_context is not None else {}),
                **({"paired_workdir": workspace_provenance} if workspace_provenance is not None else {}),
                **({"native_files": native_files} if native_files is not None else {}),
            }
            budget_proof = None
            budget_unknown = None
            if result["budget_exhausted"] and isolation_error is None:
                from .budget import BudgetOutcomeUnknown, verify_budget_failure

                try:
                    budget_proof = verify_budget_failure(task=result, engine=engine, output_dir=output)
                    result.update(budget_reward_eligible=True, budget_failure_proof=budget_proof)
                except BudgetOutcomeUnknown as budget_error:
                    budget_unknown = budget_error
                    result["budget_outcome_unknown"] = str(budget_error)
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("ALFWORLD_EPISODE_RESULT " + json.dumps(result), flush=True)
            if isolation_error is not None:
                raise isolation_error
            if budget_unknown is not None:
                safely_closed_cf = bool(
                    metadata.get("role") == "counterfactual_probe"
                    and account_summary
                    and account_summary.get("status") == "attempt_closed"
                    and account_summary.get("backend_drained") is True
                    and account_summary.get("uid_drained") is True
                    and backend_confirmed is True
                    and not backend_error
                    and native_files
                    and native_files.get("status") == "archived_and_removed"
                )
                if safely_closed_cf:
                    budget_unknown.safely_closed_budget_censor = True
                raise budget_unknown  # No TaskResult/zero is emitted; public run enforces caller-specific censor/fatal.
            return TaskResult(
                reward=reward,
                accuracy=reward,
                # This is the scored Task lifecycle, not the raw Native CLI
                # finish signal. A proven environment terminal or validated
                # wall-budget failure closes the Task even when the host stopped
                # Native. Unknown budgets raised above and never reach this path.
                finished=True,
                extra_info={
                    **identity,
                    "attempt_id": attempt_id,
                    "episode_output_dir": str(output),
                    "resolved": engine["won"],
                    "steps": engine["steps"],
                    "invalid_actions": engine["invalid_actions"],
                    "agent_ok": result["agent_ok"],
                    "agent_exit_code": result["agent_exit_code"],
                    "native_finished": native_finished,
                    "error": result["agent_error"] or backend_error,
                    "eval_completed": error is None,
                    "eval_execution_time": result["wall_seconds_total"],
                    "terminal_stop": terminal_stop,
                    "expected_host_terminal_stop": expected_terminal_stop,
                    "backend_terminal": backend_receipt,
                    "backend_terminal_error": backend_error,
                    "backend_stop_confirmed": backend_confirmed,
                    "expected_backend_terminal_stop": expected_backend_stop,
                    "backend_binding": result["backend_binding"],
                    "environment_session_id": session_id,
                    "account_isolation": account_summary,
                    "host_stop_kind": host_stop_kind,
                    "host_stop_requested": (backend_receipt or {}).get("host_stop_requested", False),
                    "budget_exhausted": host_stop_kind == "budget_timeout",
                    "task_cancelled": host_stop_kind == "task_cancel",
                    "budget_reward_eligible": result["budget_reward_eligible"],
                    **({"budget_failure_proof": budget_proof} if budget_proof is not None else {}),
                    **(
                        {"budget_outcome_unknown": result["budget_outcome_unknown"]}
                        if "budget_outcome_unknown" in result
                        else {}
                    ),
                    **({"skill_context": skill_context} if skill_context is not None else {}),
                    **({"paired_workdir": workspace_provenance} if workspace_provenance is not None else {}),
                    **({"native_files": native_files} if native_files is not None else {}),
                },
            )
