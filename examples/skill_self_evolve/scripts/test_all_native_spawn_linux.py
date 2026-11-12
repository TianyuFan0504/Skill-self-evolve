#!/usr/bin/env python3
"""CPU-only three-stdio helper checks plus one optional owned runuser probe.

No model, gateway, ALFWorld world or pool account is created. The optional
production-shape check uses an existing idle curator account without changing
the account or draining any other process belonging to it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pwd
import shlex
import shutil
import signal
import sys
import tempfile
import threading
import time
import types
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
namespace = types.ModuleType("skill_harness_examples")
namespace.__path__ = [str(HERE)]
sys.modules[namespace.__name__] = namespace
from skill_harness_examples import all_native_spawn as native  # noqa: E402

CHILD = r"""
import json, os, sys, time
mode = sys.argv[1]
if mode == 'sleep':
    print('ready', flush=True)
    time.sleep(30)
elif mode == 'large':
    os.write(1, b'x' * 400000)
    os.write(2, b'y' * 400000)
else:
    fds = []
    for fd in range(3, 1024):
        try: os.fstat(fd)
        except OSError: continue
        fds.append(fd)
    print(json.dumps({'pid':os.getpid(),'sid':os.getsid(0),'cwd':os.getcwd(),
                      'uid':os.getuid(),'euid':os.geteuid(),'gid':os.getgid(),
                      'fds':fds,'stdin_eof':os.read(0,1)==b'','env':dict(os.environ)}))
    print('stderr-sentinel', file=sys.stderr)
"""
READY_CHILD = "import os; os.write(3,b'R'); os.close(3); print('solver-ready')"


class CPULoop(native.AllNativeSelectorLoop):
    async def subprocess_exec(self, protocol_factory, program, *args, **options):
        if program == sys.executable and args[:2] == ("-c", READY_CHILD):
            return await self._spawn_native(protocol_factory, [program, *args], cwd=options["cwd"],
                                            env=options["env"], ready_fd=options["pass_fds"][0])
        if ((program == sys.executable and args[:2] == ("-c", CHILD))
                or (program == "/bin/sh" and args == ("-c", "kill -PIPE $$"))):
            return await self._spawn_native(protocol_factory, [program, *args], cwd=options["cwd"],
                                            env=options["env"], ready_fd=-1)
        return await super().subprocess_exec(protocol_factory, program, *args, **options)


class CPUPolicy(native.AllNativePolicy):
    def new_event_loop(self):
        return CPULoop(self.spawner, self.curator_profiles)


def reaped(pid):
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    raise AssertionError("Owned child was not reaped")


async def launch(mode, cwd):
    return await asyncio.create_subprocess_exec(
        sys.executable, "-c", CHILD, mode, cwd=cwd,
        env={"ONLY": "exact", "LC_ALL": "C.UTF-8", "PYTHONCOERCECLOCALE": "0"},
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )


async def generic_checks(cwd):
    leak = os.open(os.devnull, os.O_RDONLY)
    os.set_inheritable(leak, True)
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        process = await launch("inspect", cwd)
        stdout, stderr = await process.communicate()
        observed = json.loads(stdout)
        assert observed["sid"] == observed["pid"] == process.pid
        assert observed["cwd"] == cwd and observed["fds"] == [] and observed["stdin_eof"]
        assert observed["env"] == {"ONLY": "exact", "LC_ALL": "C.UTF-8", "PYTHONCOERCECLOCALE": "0"}
        assert stderr == b"stderr-sentinel\n" and process.returncode == 0 and reaped(process.pid)
        process = await launch("large", cwd)
        assert await process.communicate() == (b"x" * 400000, b"y" * 400000)
        assert process.returncode == 0 and reaped(process.pid)
        process = await launch("sleep", cwd)
        assert await process.stdout.readline() == b"ready\n"
        waiter = asyncio.create_task(process.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        process.kill()
        assert await process.wait() == -signal.SIGKILL
        await process.communicate()
        assert reaped(process.pid)
        for argv, directory in ((["/missing-all-native-test"], cwd), (["/bin/true"], cwd + "/missing")):
            try:
                await asyncio.get_running_loop()._spawn_native(
                    asyncio.SubprocessProtocol, argv, cwd=directory, env={}, ready_fd=-1
                )
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Invalid curator spawn was accepted")
        processes = await asyncio.gather(*(launch("inspect", cwd) for _ in range(12)))
        await asyncio.gather(*(process.communicate() for process in processes))
        assert all(process.returncode == 0 and reaped(process.pid) for process in processes)
        default_signal = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", "kill -PIPE $$", cwd=cwd, env={},
        )
        assert await default_signal.wait() == -signal.SIGPIPE
        await default_signal.communicate()
        assert reaped(default_signal.pid)
        ordinary = await asyncio.create_subprocess_exec("/bin/true")
        assert await ordinary.wait() == 0
        await asyncio.sleep(0)
        assert not asyncio.get_running_loop().native_transports
        assert len(os.listdir("/proc/self/fd")) == baseline
        return {"status": "passed", "scope": "three-stdio C helper and unchanged solver transport/pidfd reaper",
                "checks": ["closefrom3", "exact env", "cwd", "setsid", "stdin", "streams/backpressure",
                           "wait cancellation", "pidfd kill", "exit/reap", "error cleanup", "12 concurrent",
                           "restored SIGPIPE", "unrelated command fallback", "no FD leak"]}
    finally:
        os.close(leak)


async def concurrent_fork_eof(cwd):
    """A concurrent fork cannot inherit transient native stdout/stderr writers."""
    loop = asyncio.get_running_loop()
    original = loop.native_spawner
    spawned, attempting, release, returned, exited = [threading.Event() for _ in range(5)]
    evidence = {"fork_was_excluded": False, "fork_error": None}

    class CoordinatedSpawner:
        def __getattr__(self, name):
            return getattr(original, name)

        def __call__(self, *args):
            pid = original(*args)
            spawned.set()
            if not release.wait(3):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise RuntimeError("Concurrent fork fixture was not released")
            return pid

    def fork_worker():
        try:
            assert spawned.wait(3)
            attempting.set()
            pid = os.fork()  # Same public Python boundary as a fork-context DataLoader.
            if pid == 0:
                time.sleep(2)
                os._exit(0)
            returned.set()
            os.waitpid(pid, 0)
            exited.set()
        except BaseException as error:
            evidence["fork_error"] = type(error).__name__
            release.set()

    def release_worker():
        try:
            assert attempting.wait(3)
            evidence["fork_was_excluded"] = not returned.wait(.03)
        finally:
            release.set()

    loop.native_spawner = CoordinatedSpawner()
    forker = threading.Thread(target=fork_worker)
    controller = threading.Thread(target=release_worker)
    forker.start()
    controller.start()
    try:
        process = await launch("inspect", cwd)
        stdout, stderr = await asyncio.wait_for(process.communicate(), 1)
        assert json.loads(stdout)["fds"] == [] and stderr == b"stderr-sentinel\n"
        assert process.returncode == 0 and reaped(process.pid)
        assert evidence["fork_was_excluded"] and not exited.is_set()
        return {"status": "passed", "fork_was_excluded": True,
                "native_stdio_eof_before_fork_child_exit": True,
                "scope": "Python os.fork concurrent with real three-stdio C spawn; no model/UID allocation"}
    finally:
        release.set()
        loop.native_spawner = original
        controller.join(timeout=4)
        forker.join(timeout=4)
        assert not forker.is_alive() and not controller.is_alive() and evidence["fork_error"] is None


async def solver_ready_fd(cwd):
    """The unchanged four-FD helper still delivers the solver's fd3 proof pipe."""
    read, write = os.pipe()
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", READY_CHILD, cwd=cwd, env={"LC_ALL": "C.UTF-8"},
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, close_fds=True, pass_fds=(write,),
        )
        os.close(write)
        write = None
        assert await asyncio.wait_for(asyncio.to_thread(os.read, read, 1), 1) == b"R"
        assert await process.communicate() == (b"solver-ready\n", b"")
        assert process.returncode == 0 and reaped(process.pid)
        return {"status": "passed", "ready_fd3_delivered": True,
                "scope": "Original solver helper through all-role parent transport; no account or kernel-proof claim"}
    finally:
        os.close(read)
        if write is not None:
            os.close(write)


def uid_processes(uid):
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            rows = (path / "status").read_text().splitlines()
            values = next(row for row in rows if row.startswith("Uid:")).split()[1:]
            if uid in [int(value) for value in values]:
                found.append(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


async def production_probe(policy, user):
    account = pwd.getpwnam(user)
    if account.pw_uid == 0 or uid_processes(account.pw_uid):
        return {"status": "not_run", "reason": "Requires an existing idle non-root curator account"}
    root = Path(tempfile.mkdtemp(prefix="curator-spawn-probe-", dir="/tmp"))
    root.chmod(0o711)
    cwd = root / (uuid.uuid4().hex + uuid.uuid4().hex)
    home = cwd / ".home"
    for path in (cwd, home):
        path.mkdir(mode=0o700)
        os.chown(path, account.pw_uid, account.pw_gid)
    package = Path("/usr/local/lib/node_modules/@openai/codex").resolve(strict=True)
    runtime = types.SimpleNamespace(unix_user=user, uid=account.pw_uid, gid=account.pw_gid,
                                    work_root=str(root), package_root=str(package))
    profile = policy.register_curator(runtime)
    environment = {**native._ENVIRONMENT, "HOME": str(home), "CURATOR_CODEX_HOME": str(home),
                   "CODEX_MANAGED_PACKAGE_ROOT": str(package)}
    script = "/usr/bin/python3 -c " + shlex.quote(CHILD) + " inspect"
    process = None
    complete = False
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/sbin/runuser", "-u", user, "--", "/bin/bash", "-c", script,
            cwd=cwd, env=environment, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        observed = json.loads(stdout)
        assert observed["uid"] == observed["euid"] == account.pw_uid and observed["gid"] == account.pw_gid
        assert observed["cwd"] == str(cwd) and observed["sid"] == process.pid
        assert observed["fds"] == [] and observed["stdin_eof"]
        assert observed["env"]["CURATOR_CODEX_HOME"] == str(home)
        assert observed["env"]["CODEX_MANAGED_PACKAGE_ROOT"] == str(package)
        assert "ALFWORLD_SESSION_URL" not in observed["env"]
        assert stderr == b"stderr-sentinel\n" and process.returncode == 0 and reaped(process.pid)
        assert not Path("/proc", str(observed["pid"])).exists()
        complete = True
        return {"status": "passed", "scope": "one public matcher/runuser production-shape CPU launch",
                "profile": vars(profile), "child": {key: observed[key] for key in (
                    "pid", "sid", "cwd", "uid", "euid", "gid", "fds", "stdin_eof")},
                "process_pid": process.pid, "returncode": process.returncode,
                "no_model_gateway_world_or_pool_account": True,
                "runuser_environment_scope": "Input env passed unchanged; runuser/PAM may add its ordinary user fields"}
    finally:
        if complete:
            shutil.rmtree(root)
        elif process is not None and process.returncode is None:
            process.kill()  # Exact owned parent via its transport pidfd; no UID-wide drain.
            await process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-curator-user")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit("Choose a new result path")
    helpers = (HERE / "native_posix_spawn_helper.so", HERE / "curator_posix_spawn_helper.so")
    cpu_policy = CPUPolicy(*helpers)
    asyncio.set_event_loop_policy(cpu_policy)
    with tempfile.TemporaryDirectory(prefix="all-native-generic-") as cwd:
        generic = asyncio.run(generic_checks(cwd))
        fork_eof = asyncio.run(concurrent_fork_eof(cwd))
        ready_fd = asyncio.run(solver_ready_fd(cwd))
    production = None
    actual = native.AllNativePolicy(*helpers)
    asyncio.set_event_loop_policy(actual)
    if args.production_curator_user:
        production = asyncio.run(production_probe(actual, args.production_curator_user))
    result = {"schema_version": 1, "generic": generic, "concurrent_fork_eof": fork_eof,
              "solver_ready_fd": ready_fd, "production": production,
              "identity": native.identity(actual),
              "source_hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in (
                  "all_native_spawn.py", "curator_posix_spawn.c", "test_all_native_spawn_linux.py")}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({"result": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "generic": generic["status"],
                      "production": None if production is None else production["status"]}))


if __name__ == "__main__":
    main()
