"""Fork-free native launch extension; requires the separately reviewed glibc helper.

Only the frozen native setpriv/isolate-exec launch is intercepted. The public
asyncio policy, event-loop and transport interfaces supply the integration.
The `_wait` method is also supplied for asyncio.subprocess.Process.wait.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import signal
import threading
from pathlib import Path

POLICY_ID = "native-posix-spawn-v1"
ACCOUNTS = "/workspace/verl-0.9.1/verl/experimental/skill_harness/alfworld/accounts.py"


def _helper_path(path=None):
    return Path(path or Path(__file__).with_name("native_posix_spawn_helper.so")).resolve(strict=True)


def spawn_identity(helper_path=None):
    paths = {"python": Path(__file__).resolve(), "c": Path(__file__).with_suffix(".c").resolve(),
             "helper": _helper_path(helper_path)}
    identity = {
        "policy": POLICY_ID,
        **{name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
           for name, path in paths.items()},
    }
    current = asyncio.get_event_loop_policy()
    if isinstance(current, NativeSpawnPolicy) and current.helper_path == str(paths["helper"]):
        if current.spawner.sha256 != identity["helper"]["sha256"]:
            raise RuntimeError("Native spawn helper changed after loading")
        identity["helper"]["loaded_sha256"] = current.spawner.sha256
    return identity


class _Spawner:
    def __init__(self, path):
        self.sha256 = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(self.sha256).hexdigest()
        self.library = ctypes.CDLL(str(path))
        self.spawn = self.library.native_posix_spawn
        strings = ctypes.POINTER(ctypes.c_char_p)
        self.spawn.argtypes = [ctypes.c_char_p, strings, strings, ctypes.c_char_p,
                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.spawn.restype = ctypes.c_int
        self._pidfd_open = self.library.native_pidfd_open
        self._pidfd_open.argtypes, self._pidfd_open.restype = [ctypes.c_int], ctypes.c_int
        self._pidfd_signal = self.library.native_pidfd_signal
        self._pidfd_signal.argtypes, self._pidfd_signal.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int

    def pidfd_open(self, pid):
        fd = self._pidfd_open(pid)
        if fd < 0:
            raise OSError(-fd, os.strerror(-fd))
        return fd

    def pidfd_signal(self, fd, number):
        error = self._pidfd_signal(fd, number)
        if error:
            raise OSError(error, os.strerror(error))

    def __call__(self, argv, env, cwd, sources):
        def vector(values):
            encoded = [os.fsencode(value) for value in values]
            if any(b"\0" in value for value in encoded):
                raise ValueError("NUL in native spawn arguments")
            return (ctypes.c_char_p * (len(encoded) + 1))(*encoded, None)
        if (any(not isinstance(key, str) or not key or "=" in key or "\0" in key
                or not isinstance(value, str) or "\0" in value for key, value in env.items())
                or b"\0" in os.fsencode(cwd)):
            raise ValueError("Invalid native spawn environment or cwd")
        pid = ctypes.c_int()
        args, environment = vector(argv), vector(f"{key}={value}" for key, value in env.items())
        error = self.spawn(os.fsencode(argv[0]), args, environment, os.fsencode(cwd),
                           (ctypes.c_int * 4)(*sources), ctypes.byref(pid))
        if error:
            raise OSError(error, os.strerror(error), argv[0])
        return pid.value


class _Pipe(asyncio.Protocol):
    def __init__(self, owner, fd):
        self.owner, self.fd = owner, fd

    def data_received(self, data):
        self.owner._emit("pipe_data_received", self.fd, data)

    def connection_lost(self, error):
        self.owner._open.discard(self.fd)
        self.owner._emit("pipe_connection_lost", self.fd, error)
        self.owner._finish()


class NativeSpawnTransport(asyncio.SubprocessTransport):
    """Read-only native stdio transport with an independent direct-child reaper."""

    def __init__(self, loop, protocol, pid, readers):
        super().__init__({"pid": pid, "native_spawn_policy": POLICY_ID})
        self.loop, self.protocol, self.pid = loop, protocol, pid
        self.returncode, self.pipes, self._readers = None, {}, readers
        self._reap_error = None
        self._open, self._waiters, self._pending = set(readers), [], []
        self._ready = self._closed = self._lost = False
        self._reaped = threading.Event()
        self._signal_lock = threading.Lock()
        self._pidfd = loop.native_spawner.pidfd_open(pid)
        try:
            threading.Thread(target=self._reap, name=f"native-reap-{pid}", daemon=True).start()
        except BaseException:
            os.close(self._pidfd)
            raise

    async def connect(self):
        try:
            for fd, reader in self._readers.items():
                pipe, _ = await self.loop.connect_read_pipe(lambda fd=fd: _Pipe(self, fd), reader)
                self.pipes[fd] = pipe
            self.protocol.connection_made(self)
            self._ready = True
            for method, args in self._pending:
                getattr(self.protocol, method)(*args)
            self._pending.clear()
            self._finish()
        except BaseException:
            self.close()
            for reader in self._readers.values():
                reader.close()
            self.loop.native_transports.discard(self)
            # Reaping does not depend on this loop or cancellation state.
            raise

    def _emit(self, method, *args):
        if self._ready:
            getattr(self.protocol, method)(*args)
        else:
            self._pending.append((method, args))

    def _reap(self):
        try:
            while True:
                try:
                    _, status = os.waitpid(self.pid, 0)
                    break
                except InterruptedError:
                    continue
            callback, result = self._exited, os.waitstatus_to_exitcode(status)
        except BaseException as error:
            callback, result = self._reap_failed, error
        with self._signal_lock:
            self._reaped.set()
            os.close(self._pidfd)
        try:
            self.loop.call_soon_threadsafe(callback, result)
        except RuntimeError:  # Loop already closed; waitpid still reaped the child.
            if callback == self._exited:
                self.returncode = result
            else:
                self._reap_error = result

    def _reap_failed(self, error):
        self._reap_error = error
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_exception(error)
        self._waiters.clear()
        self.close()
        self.loop.call_exception_handler({"message": "Native child reaping failed", "exception": error,
                                          "transport": self})
        self._finish()

    def _exited(self, code):
        self.returncode = code
        self._emit("process_exited")
        for waiter in self._waiters:
            if not waiter.done():
                waiter.set_result(code)
        self._waiters.clear()
        self._finish()

    def _finish(self):
        if self._ready and (self.returncode is not None or self._reap_error) and not self._open and not self._lost:
            self._lost = True
            self.protocol.connection_lost(self._reap_error)
            self.loop.native_transports.discard(self)

    async def _wait(self):
        if self._reap_error:
            raise self._reap_error
        if self.returncode is not None:
            return self.returncode
        waiter = self.loop.create_future()
        self._waiters.append(waiter)
        return await waiter

    def get_pid(self):
        return self.pid

    def get_returncode(self):
        return self.returncode

    def get_pipe_transport(self, fd):
        return self.pipes.get(fd)

    def get_protocol(self):
        return self.protocol

    def set_protocol(self, protocol):
        self.protocol = protocol

    def is_closing(self):
        return self._closed

    def send_signal(self, number):
        with self._signal_lock:
            if not self._reaped.is_set():
                try:
                    self.loop.native_spawner.pidfd_signal(self._pidfd, number)
                except ProcessLookupError:
                    pass

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    def close(self):
        if not self._closed:
            self._closed = True
            self.kill()
            for pipe in self.pipes.values():
                pipe.close()


def _native_arguments(program, args, options):
    """Return rewritten argv, or None for an unrelated subprocess."""
    if program != "/usr/bin/setpriv" or "isolate-exec" not in args:
        return None
    argv = [program, *args]
    flags = ["--clear-groups", "--bounding-set=-all", "--inh-caps=-all",
             "--ambient-caps=-all", "--no-new-privs", "/usr/bin/python3", ACCOUNTS, "isolate-exec"]
    if (len(argv) < 23 or any(not isinstance(arg, str) for arg in argv)
            or argv[3:11] != flags or argv[11] != "--uid" or argv[13] != "--gid"
            or argv[15] != "--ready-fd" or argv[17] != "--read-roots"
            or argv[19] != "--write-roots" or argv[21] != "--"
            or not argv[12].isdigit() or not argv[14].isdigit() or not argv[16].isdigit()
            or not 20000 <= int(argv[12]) <= 20127 or argv[14] != argv[12]
            or argv[1] != "--reuid=" + argv[12] or argv[2] != "--regid=" + argv[14]):
        raise ValueError("Unrecognized native isolation launch")
    ready = int(argv[16])
    required = {"stdin": asyncio.subprocess.DEVNULL, "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE, "start_new_session": True, "close_fds": True,
                "pass_fds": (ready,)}
    if (set(options) != set(required) | {"cwd", "env"}
            or any(options.get(key) != value for key, value in required.items())
            or ready < 3 or not isinstance(options["env"], dict)
            or not options["cwd"] or not os.path.isabs(options["cwd"])):
        raise ValueError("Unsupported native subprocess options")
    argv[16] = "3"
    return argv, ready


class NativeSpawnSelectorLoop(asyncio.SelectorEventLoop):
    def __init__(self, spawner):
        super().__init__()
        self.native_spawner = spawner
        self.native_transports = set()

    async def _spawn_native(self, protocol_factory, argv, *, cwd, env, ready_fd):
        """Internal CPU-test seam; the public override validates native commands."""
        owned, readers, pid, transport = [], {}, None, None
        try:
            protocol = protocol_factory()
            null = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
            owned.append(null)
            sources = [null]
            for fd in (1, 2):
                read, write = os.pipe()
                owned.extend((read, write))
                sources.append(write)
                readers[fd] = os.fdopen(read, "rb", buffering=0)
                owned.remove(read)
            pid = self.native_spawner(argv, env, cwd, [*sources, ready_fd])
            transport = NativeSpawnTransport(self, protocol, pid, readers)
            self.native_transports.add(transport)
            await transport.connect()
            return transport, transport.protocol
        except BaseException:
            if pid is not None and transport is None:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            for reader in readers.values():
                reader.close()
            raise
        finally:
            for fd in owned:
                os.close(fd)

    async def subprocess_exec(self, protocol_factory, program, *args, **kwargs):
        native = _native_arguments(program, args, kwargs)
        if native is None:
            return await super().subprocess_exec(protocol_factory, program, *args, **kwargs)
        argv, ready = native
        return await self._spawn_native(protocol_factory, argv, cwd=kwargs["cwd"],
                                        env=kwargs["env"], ready_fd=ready)

    def close(self):
        for transport in tuple(self.native_transports):
            transport.close()
        super().close()


class NativeSpawnPolicy(asyncio.DefaultEventLoopPolicy):
    def __init__(self, helper_path):
        super().__init__()
        self.helper_path = str(_helper_path(helper_path))
        self.spawner = _Spawner(self.helper_path)

    def new_event_loop(self):
        return NativeSpawnSelectorLoop(self.spawner)


def install_spawn_policy(helper_path=None):
    """Install once per Ray worker, before its asyncio.run/new_event_loop call."""
    path, current = str(_helper_path(helper_path)), asyncio.get_event_loop_policy()
    if isinstance(current, NativeSpawnPolicy) and current.helper_path == path:
        return current
    policy = NativeSpawnPolicy(path)
    asyncio.set_event_loop_policy(policy)
    return policy
