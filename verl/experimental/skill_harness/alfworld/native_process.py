"""Launch a native session without forking a multithreaded Python worker.

The small isolated Python bootstrap closes inherited descriptors before replacing
itself with the unchanged native launcher. Exit notification belongs to this
module; pipe EOF is deliberately independent of the direct child's exit.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time


def _require_linux():
    if sys.platform != "linux" or not hasattr(os, "posix_spawn"):
        raise RuntimeError("Native process launch requires Linux posix_spawn support")


def _close_fds(descriptors):
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError as error:
            if error.errno != errno.EBADF:
                raise


def _pipe():
    descriptors = list(os.pipe2(os.O_CLOEXEC))
    try:
        for index, descriptor in enumerate(descriptors):
            if descriptor < 3:
                replacement = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
                os.close(descriptor)
                descriptors[index] = replacement
        return tuple(descriptors)
    except BaseException:
        _close_fds(descriptors)
        raise


async def _complete_cleanup(task):
    """Finish owned cleanup even when the caller receives further cancellation."""
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _consume_acknowledgement(task):
    if not task.cancelled():
        task.exception()


async def _read_acknowledgement(reader):
    acknowledgement = asyncio.create_task(reader.readexactly(1))
    try:
        # wait_for on Python 3.10/3.11 can swallow owner cancellation when the
        # read finishes in the same event-loop turn. wait preserves that owner.
        done, _ = await asyncio.wait((acknowledgement,), timeout=10)
        if not done:
            raise asyncio.TimeoutError()
        return acknowledgement.result()
    finally:
        if not acknowledgement.done():
            acknowledgement.cancel()
        acknowledgement.add_done_callback(_consume_acknowledgement)


def _kill_owned_process(pid, *, group_owned=False):
    # The bootstrap might not have reached setsid yet. Its direct PID is ours
    # from spawn onward; its process group becomes ours before acknowledgement.
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # No await/reap occurs between killing the exact child and checking its
    # final group. This also catches setsid racing with the termination signal.
    if not group_owned:
        try:
            group_owned = os.getpgid(pid) == pid
        except ProcessLookupError:
            pass
    if group_owned:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def _reap_failed_construction(pid):
    _kill_owned_process(pid)
    loop = asyncio.get_running_loop()
    exited = loop.create_future()

    def poll():
        try:
            child, _ = os.waitpid(pid, os.WNOHANG)
            if child:
                exited.set_result(None)
            else:
                loop.call_later(0.01, poll)
        except BaseException as error:
            exited.set_exception(error)

    poll()
    await _complete_cleanup(exited)


class _ReadPipeTransport(asyncio.ReadTransport):
    """Own a pipe FD and feed a StreamReader without asynchronous attachment."""

    def __init__(self, descriptor, reader, loop):
        super().__init__()
        self._descriptor, self._reader, self._loop = descriptor, reader, loop
        self._reading = False
        try:
            os.set_blocking(descriptor, False)
            reader.set_transport(self)
            self.resume_reading()
        except BaseException:
            # The spawning caller still owns this FD until attachment returns.
            loop.remove_reader(descriptor)
            self._descriptor = None
            raise

    def _read_ready(self):
        if self._descriptor is None:
            return
        try:
            data = os.read(self._descriptor, 65536)
        except BlockingIOError:
            return
        except OSError as error:
            self._reader.set_exception(error)
            self.close()
            return
        if data:
            self._reader.feed_data(data)
        else:
            self.close()

    def is_closing(self):
        return self._descriptor is None

    def is_reading(self):
        return self._reading

    def pause_reading(self):
        if self._reading:
            self._loop.remove_reader(self._descriptor)
            self._reading = False

    def resume_reading(self):
        if self._descriptor is not None and not self._reading:
            self._loop.add_reader(self._descriptor, self._read_ready)
            self._reading = True

    def close(self):
        if self._descriptor is not None:
            self.pause_reading()
            descriptor, self._descriptor = self._descriptor, None
            os.close(descriptor)
            self._reader.feed_eof()


async def _attach_reader(descriptor):
    # No await inside allocation/registration: the caller retains FD ownership
    # until the complete transport has been returned.
    reader = asyncio.StreamReader()
    return reader, _ReadPipeTransport(descriptor, reader, asyncio.get_running_loop())


class NativeProcess:
    """The subset of asyncio Process used by the native ALFWorld launcher."""

    def __init__(self, pid, loop):
        self.pid = pid
        self.stdout = self.stderr = None
        self.returncode = None
        self._loop = loop
        self._exit = loop.create_future()
        self._transports = []
        self._session_ready = False
        self._poll = loop.call_soon(self._poll_exit)

    def _record_exit(self, status):
        self.returncode = os.waitstatus_to_exitcode(status)
        if self._poll is not None:
            self._poll.cancel()
            self._poll = None
        if not self._exit.done():
            self._exit.set_result(self.returncode)

    def _poll_exit(self):
        self._poll = None
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self._record_exit(status)
            else:
                self._poll = self._loop.call_later(0.01, self._poll_exit)
        except BaseException as error:
            if not self._exit.done():
                self._exit.set_exception(error)

    async def wait(self):
        # wait_for(process.wait(), ...) must never cancel our sole exit proof.
        return await asyncio.shield(self._exit)

    def send_signal(self, number):
        if self.returncode is None:
            os.kill(self.pid, number)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)

    async def _abort(self):
        try:
            if self.returncode is None:
                _kill_owned_process(self.pid, group_owned=self._session_ready)
            elif self._session_ready:
                try:
                    os.killpg(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await _complete_cleanup(self._exit)
        finally:
            for transport in self._transports:
                transport.close()


def _wait_owned_pid(pid, timeout):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        child, status = os.waitpid(pid, os.WNOHANG)
        if child:
            return os.waitstatus_to_exitcode(status)
        if deadline is not None and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("owned native process", timeout)
        time.sleep(0.01)


def _spawn_bootstrap(argv, *, env, cwd, stdout_fd, stderr_fd, pass_fds=(), close_fds=()):
    """The shared descriptor/signal/session boundary for both process APIs."""
    _require_linux()
    arguments = [os.fspath(value) for value in argv]
    if not arguments or any(not isinstance(value, str) for value in arguments):
        raise ValueError("Supply a nonempty native argument vector of strings")
    inherited = tuple(sorted(set(pass_fds)))
    if any(type(descriptor) is not int or descriptor < 3 for descriptor in inherited):
        raise ValueError("Pass descriptors must be integers greater than two")
    for descriptor in inherited:
        os.fstat(descriptor)
    directory = os.getcwd() if cwd is None else os.path.abspath(os.fspath(cwd))
    executable = os.path.abspath(sys.executable)
    helper = os.path.abspath(__file__)
    bootstrap = [executable, "-I", "-S", helper, "--exec", directory, json.dumps(inherited), *arguments]
    owned = set()
    pid = None
    try:
        stdin = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
        owned.add(stdin)
        if stdin < 3:
            replacement = fcntl.fcntl(stdin, fcntl.F_DUPFD_CLOEXEC, 3)
            owned.add(replacement)
            os.close(stdin)
            owned.remove(stdin)
            stdin = replacement
        ready_read, ready_write = _pipe()
        owned.update((ready_read, ready_write))
        bootstrap.insert(7, str(ready_write))
        actions = [
            (os.POSIX_SPAWN_DUP2, stdin, 0),
            (os.POSIX_SPAWN_DUP2, stdout_fd, 1),
            (os.POSIX_SPAWN_DUP2, stderr_fd, 2),
            (os.POSIX_SPAWN_DUP2, ready_write, ready_write),
            *((os.POSIX_SPAWN_DUP2, descriptor, descriptor) for descriptor in inherited),
            *(
                (os.POSIX_SPAWN_CLOSE, descriptor)
                for descriptor in sorted(owned | set(close_fds) | {stdout_fd, stderr_fd})
                if descriptor > 2 and descriptor != ready_write and descriptor not in inherited
            ),
        ]
        defaults = tuple(getattr(signal, name) for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ") if hasattr(signal, name))
        pid = os.posix_spawn(
            executable,
            bootstrap,
            dict(os.environ if env is None else env),
            file_actions=actions,
            setsigmask=(),
            setsigdef=defaults,
        )
        _close_fds((stdin, ready_write))
        owned.difference_update((stdin, ready_write, ready_read))
        return pid, ready_read
    except BaseException as error:
        if pid is not None:
            try:
                _kill_owned_process(pid)
                _wait_owned_pid(pid, 5)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("Native bootstrap cleanup failed: " + repr(cleanup_error))
        raise
    finally:
        _close_fds(owned)


class OwnedServiceProcess:
    """Synchronous, exact-child ownership for the evaluation environment service."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None
        self._session_ready = False

    def poll(self):
        if self.returncode is None:
            child, status = os.waitpid(self.pid, os.WNOHANG)
            if child:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = _wait_owned_pid(self.pid, timeout)
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            try:
                os.kill(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def kill(self):
        if self.returncode is None:
            _kill_owned_process(self.pid, group_owned=self._session_ready)


def spawn_owned_service(argv, *, stdout, env=None, cwd=None):
    """Launch an owned service with merged file output, without a Python fork."""
    pid = ready_read = process = None
    try:
        descriptor = stdout if isinstance(stdout, int) else stdout.fileno()
        pid, ready_read = _spawn_bootstrap(
            argv,
            env=env,
            cwd=cwd,
            stdout_fd=descriptor,
            stderr_fd=descriptor,
        )
        process = OwnedServiceProcess(pid)
        readable, _, _ = select.select((ready_read,), (), (), 10)
        if not readable:
            raise TimeoutError("Native bootstrap did not acknowledge its session")
        if os.read(ready_read, 1) != b"\x01":
            raise RuntimeError("Native bootstrap exited before session acknowledgement")
        process._session_ready = True
        return process
    except BaseException as error:
        if pid is not None:
            try:
                if process is None:
                    _kill_owned_process(pid)
                    _wait_owned_pid(pid, 5)
                elif process.returncode is None:
                    process.kill()
                    process.wait(timeout=5)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("Owned service cleanup failed: " + repr(cleanup_error))
        raise
    finally:
        if ready_read is not None:
            _close_fds((ready_read,))


async def spawn_native_process(argv, *, env, cwd, pass_fds=()):
    """Spawn an owned native session with async output and direct-child exit proof."""
    _require_linux()
    owned = set()
    process = None
    pid = None
    try:
        stdout_read, stdout_write = _pipe()
        owned.update((stdout_read, stdout_write))
        stderr_read, stderr_write = _pipe()
        owned.update((stderr_read, stderr_write))
        pid, ready_read = _spawn_bootstrap(
            argv,
            env=env,
            cwd=cwd,
            stdout_fd=stdout_write,
            stderr_fd=stderr_write,
            pass_fds=pass_fds,
            close_fds=owned,
        )
        owned.add(ready_read)
        process = NativeProcess(pid, asyncio.get_running_loop())
        _close_fds((stdout_write, stderr_write))
        owned.difference_update((stdout_write, stderr_write))
        process.stdout, transport = await _attach_reader(stdout_read)
        owned.remove(stdout_read)
        process._transports.append(transport)
        process.stderr, transport = await _attach_reader(stderr_read)
        owned.remove(stderr_read)
        process._transports.append(transport)
        ready, transport = await _attach_reader(ready_read)
        owned.remove(ready_read)
        process._transports.append(transport)
        try:
            acknowledgement = await _read_acknowledgement(ready)
            if acknowledgement != b"\x01":
                raise RuntimeError("Native bootstrap acknowledgement is invalid")
            process._session_ready = True
        except asyncio.IncompleteReadError as error:
            raise RuntimeError("Native bootstrap exited before session acknowledgement") from error
        finally:
            transport.close()
        return process
    except BaseException as error:
        if pid is not None:
            try:
                if process is not None:
                    await process._abort()
                else:
                    await _reap_failed_construction(pid)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note("Native launch cleanup failed: " + repr(cleanup_error))
        raise
    finally:
        _close_fds(owned)


def _exec_bootstrap(arguments):
    directory, encoded_descriptors, encoded_ready, *command = arguments
    ready = int(encoded_ready)
    keep = {0, 1, 2, ready, *json.loads(encoded_descriptors)}
    os.setsid()
    # CLOEXEC is not sufficient: the host may own inheritable descriptors that
    # the native UID must never receive. Enumerate before changing cwd/identity.
    for name in os.listdir("/proc/self/fd"):
        descriptor = int(name)
        if descriptor not in keep:
            _close_fds((descriptor,))
    os.chdir(directory)
    # CPython startup ignores SIGPIPE again, so restore the native exec contract
    # here as well as in posix_spawn's signal attributes.
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    os.write(ready, b"\x01")
    os.close(ready)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    if len(sys.argv) < 6 or sys.argv[1] != "--exec":
        raise SystemExit("This module is an internal native exec bootstrap")
    _exec_bootstrap(sys.argv[2:])
