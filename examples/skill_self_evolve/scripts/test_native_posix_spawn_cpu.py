#!/usr/bin/env python3
"""CPU-only staging test; never invokes accounts.py, setpriv or an ALFWorld task.

Build adjacent native_posix_spawn_helper.so with cc -O2 -shared -fPIC -Wall -Wextra
-Werror native_posix_spawn.c -o native_posix_spawn_helper.so, then run this script with
the existing core and Uni-Agent source roots on PYTHONPATH. No Ray/GPU starts.
"""

import asyncio
import hashlib
import json
import os
import platform
import signal
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
namespace = types.ModuleType("skill_harness_examples")
namespace.__path__ = [str(HERE)]
sys.modules[namespace.__name__] = namespace
from skill_harness_examples import native_posix_spawn as native  # noqa: E402

CHILD = r"""
import json, os, signal, sys, time
mode = sys.argv[1]
os.write(3, b'R')
os.close(3)
if mode == 'sleep':
    time.sleep(60)
elif mode == 'large':
    os.write(1, b'x' * 400000)
    os.write(2, b'y' * 400000)
else:
    inherited = []
    for fd in range(3, 1024):
        try: os.fstat(fd)
        except OSError: continue
        inherited.append(fd)
    print(json.dumps({'cwd': os.getcwd(), 'env': dict(os.environ),
                      'sid': os.getsid(0), 'pid': os.getpid(), 'fds': inherited,
                      'stdin': os.read(0, 1).hex(),
                      'sigpipe': signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN,
                      'blocked': sorted(signal.pthread_sigmask(signal.SIG_BLOCK, []))}))
    print('stderr-sentinel', file=sys.stderr)
    sys.exit(7)
"""


class CPULoop(native.NativeSpawnSelectorLoop):
    """Test-only routing seam; production matcher remains unchanged."""

    async def subprocess_exec(self, protocol_factory, program, *args, **options):
        if ((program == sys.executable and args[:2] == ("-c", CHILD))
                or (program == "/bin/sh" and args == ("-c", "kill -PIPE $$"))):
            ready = options.pop("pass_fds")[0]
            return await self._spawn_native(protocol_factory, [program, *args],
                                            cwd=options["cwd"], env=options["env"], ready_fd=ready)
        return await super().subprocess_exec(protocol_factory, program, *args, **options)


class CPUPolicy(native.NativeSpawnPolicy):
    def new_event_loop(self):
        return CPULoop(self.spawner)


def matcher_test():
    argv = ["/usr/bin/setpriv", "--reuid=20001", "--regid=20001", "--clear-groups",
            "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs",
            "/usr/bin/python3", native.ACCOUNTS, "isolate-exec", "--uid", "20001", "--gid", "20001",
            "--ready-fd", "91", "--read-roots", "[]", "--write-roots", "[]", "--", "/bin/true"]
    options = dict(stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                   stderr=asyncio.subprocess.PIPE, cwd="/tmp", env={"ONLY": "exact"},
                   start_new_session=True, close_fds=True, pass_fds=(91,))
    rewritten, ready = native._native_arguments(argv[0], argv[1:], options)
    assert ready == 91 and rewritten[16] == "3" and argv[16] == "91"
    assert rewritten[:16] + rewritten[17:] == argv[:16] + argv[17:]
    assert native._native_arguments("/bin/true", (), {}) is None
    for key, value in (("pass_fds", (91, 92)), ("close_fds", False), ("cwd", "relative")):
        try:
            native._native_arguments(argv[0], argv[1:], {**options, key: value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid {key}")


async def launch(mode, cwd):
    ready_read, ready_write = os.pipe()
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", CHILD, mode, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=cwd, env={"ONLY": "exact", "LC_ALL": "C.UTF-8", "PYTHONCOERCECLOCALE": "0"},
            start_new_session=True, close_fds=True, pass_fds=(ready_write,))
        os.close(ready_write)
        ready_write = None
        assert await asyncio.to_thread(os.read, ready_read, 1) == b"R"
        return process
    finally:
        os.close(ready_read)
        if ready_write is not None:
            os.close(ready_write)


def assert_reaped(pid):
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    raise AssertionError(f"unreaped child {pid}")


async def checks(cwd):
    leak = os.open(os.devnull, os.O_RDONLY)
    os.set_inheritable(leak, True)
    baseline = len(os.listdir("/proc/self/fd"))
    try:
        for argv, directory in ((["/definitely-missing-native-cpu"], cwd),
                                (["/bin/true"], cwd + "/missing")):
            try:
                await asyncio.get_running_loop()._spawn_native(
                    asyncio.SubprocessProtocol, argv, cwd=directory, env={}, ready_fd=leak)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("accepted invalid spawn")
            assert len(os.listdir("/proc/self/fd")) == baseline
        print("CPU spawn-error paths passed", flush=True)
        process = await launch("inspect", cwd)
        stdout, stderr = await process.communicate()
        data = json.loads(stdout)
        assert data["cwd"] == cwd and data["env"] == {
            "ONLY": "exact", "LC_ALL": "C.UTF-8", "PYTHONCOERCECLOCALE": "0"}
        assert data["sid"] == data["pid"] == process.pid
        assert data["fds"] == [] and data["stdin"] == "" and data["blocked"] == []
        assert stderr == b"stderr-sentinel\n" and process.returncode == 7
        assert_reaped(process.pid)
        process = await launch("large", cwd)
        assert await process.communicate() == (b"x" * 400000, b"y" * 400000)
        assert process.returncode == 0
        process = await launch("sleep", cwd)
        try:
            await asyncio.wait_for(process.wait(), 0.02)
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("wait did not time out")
        process.terminate()
        assert await process.wait() == -signal.SIGTERM
        await process.communicate()
        assert_reaped(process.pid)
        process = await launch("sleep", cwd)
        waiter = asyncio.create_task(process.wait())
        await asyncio.sleep(0)
        waiter.cancel()
        try:
            await waiter
        except asyncio.CancelledError:
            pass
        process.kill()
        assert await process.wait() == -signal.SIGKILL
        await process.communicate()
        assert_reaped(process.pid)
        processes = await asyncio.gather(*(launch("inspect", cwd) for _ in range(24)))
        await asyncio.gather(*(process.communicate() for process in processes))
        for process in processes:
            assert process.returncode == 7
            assert_reaped(process.pid)
        # Cancel during pipe attachment after spawn; direct-child reaper survives.
        ready_read, ready_write = os.pipe()
        try:
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                sys.executable, "-c", CHILD, "sleep", cwd=cwd, env={"LC_ALL": "C.UTF-8"},
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, pass_fds=(ready_write,)))
            await asyncio.sleep(0)
            transports = tuple(asyncio.get_running_loop().native_transports)
            assert len(transports) == 1
            spawning.cancel()
            try:
                await spawning
            except asyncio.CancelledError:
                pass
            for transport in transports:
                assert await transport._wait() == -signal.SIGKILL
                assert_reaped(transport.pid)
        finally:
            os.close(ready_read)
            os.close(ready_write)
        # A shell preserves inherited ignored SIGPIPE; helper must restore it.
        default_signal = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", "kill -PIPE $$", cwd=cwd, env={}, pass_fds=(leak,))
        assert await default_signal.wait() == -signal.SIGPIPE
        await default_signal.communicate()
        # Unrelated subprocess uses the standard loop implementation.
        ordinary = await asyncio.create_subprocess_exec("/bin/true")
        assert await ordinary.wait() == 0
        await asyncio.sleep(0)
        assert not asyncio.get_running_loop().native_transports
        assert len(os.listdir("/proc/self/fd")) == baseline
    finally:
        os.close(leak)


def main():
    print("CPU matcher start", flush=True)
    matcher_test()
    policy = native.install_spawn_policy()
    assert native.install_spawn_policy() is policy
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    # Actual SDK materialization, without running the runner or starting Ray.
    print("CPU actual-SDK import start", flush=True)
    from uni_agent.framework.framework import _materialize_runner
    runner = _materialize_runner("skill_harness_examples.train_spawn_safe.SpawnSafeRunner",
                                 {"task_config_path": "cpu-only"})
    assert runner.runner_kwargs == {"task_config_path": "cpu-only"}
    assert isinstance(asyncio.get_event_loop_policy(), native.NativeSpawnPolicy)
    print("CPU actual-SDK materialization passed", flush=True)

    async def probe():
        assert type(asyncio.get_running_loop()).__module__ == "skill_harness_examples.native_posix_spawn"
    asyncio.run(probe())
    asyncio.set_event_loop_policy(CPUPolicy(HERE / "native_posix_spawn_helper.so"))
    with tempfile.TemporaryDirectory(prefix="native-spawn-cpu-") as directory:
        asyncio.run(checks(directory))
    identity = native.spawn_identity()
    assert identity["helper"]["loaded_sha256"] == identity["helper"]["sha256"]
    import ray
    assert not ray.is_initialized()
    names = ["native_posix_spawn.py", "native_posix_spawn.c", "train_spawn_safe.py", __file__]
    result = {"status": "passed", "exit_code": 0,
              "checks": ["actual SDK constructor installs policy before asyncio.run", "strict native matcher",
                         "missing executable error", "invalid cwd error", "ready fd3 byte",
                         "inheritable FD exclusion", "cwd", "new session", "exact environment", "DEVNULL stdin",
                         "400KB stdout/stderr and backpressure", "wait/returncode", "SIGTERM", "SIGKILL",
                         "wait timeout", "wait cancellation", "spawn attachment cancellation",
                         "SIGPIPE restoration", "24 concurrent launches", "no zombies", "no FD leaks",
                         "completed transport removal", "unrelated subprocess fallback", "idempotent policy"],
              "python": {"executable": sys.executable, "version": sys.version,
                         "build": platform.python_build(), "compiler": platform.python_compiler()},
              "actual_sdk": {"fqn": "skill_harness_examples.train_spawn_safe.SpawnSafeRunner",
                             "materializer_source": _materialize_runner.__code__.co_filename,
                             "materializer_line": _materialize_runner.__code__.co_firstlineno,
                             "default_policy_before_constructor": True, "native_policy_after_constructor": True,
                             "benign_asyncio_run_loop_module": "skill_harness_examples.native_posix_spawn"},
              "not_invoked": ["UID isolate-exec", "account pool", "models", "GPU work", "evaluation", "ray.init"],
              "ray_initialized": ray.is_initialized(), "trusted_command_matcher_execution": "shape validation only",
              "sources": {Path(name).name: {"path": str(HERE / Path(name).name),
                                           "sha256": hashlib.sha256((HERE / Path(name).name).read_bytes()).hexdigest()}
                          for name in names}, "identity": identity}
    receipt = HERE / "native_posix_spawn_cpu_result.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    receipt.chmod(0o600)
    print(json.dumps({"status": result["status"], "receipt": str(receipt),
                      "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(), "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
