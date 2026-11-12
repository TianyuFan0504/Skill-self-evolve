"""Fork-free launches for solver ready-pipe and registered curator stdio shapes.

The original solver helper/transport is reused unchanged. The separate curator
helper closes every non-stdio descriptor and executes the original runuser
argv/environment, preserving its existing UID-switch behavior.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import pwd
import re
import signal
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from skill_harness_examples import native_posix_spawn as solver

POLICY = "solver-and-registered-curator-posix-spawn-v1"
_FORK_GUARD = threading.Lock()
_FORK_REGISTRATION = threading.Lock()
_FORK_HOOK_INSTALLED = False
_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
}


def _install_fork_hooks():
    """Exclude Python os.fork while transient native pipe writers exist.

    Install before constructing the trainer/DataLoader. No lock is held over
    an await, and libc posix_spawn does not run Python's at-fork callbacks.
    This intentionally does not interpose on arbitrary C-library forks.
    """
    global _FORK_HOOK_INSTALLED
    if _FORK_HOOK_INSTALLED:
        return
    with _FORK_REGISTRATION:
        if not _FORK_HOOK_INSTALLED:
            os.register_at_fork(before=_FORK_GUARD.acquire, after_in_parent=_FORK_GUARD.release,
                                after_in_child=_FORK_GUARD.release)
            _FORK_HOOK_INSTALLED = True


@dataclass(frozen=True)
class CuratorProfile:
    user: str
    uid: int
    gid: int
    work_root: str
    package_root: str

    @classmethod
    def from_runtime(cls, runtime):
        account = pwd.getpwnam(runtime.unix_user)
        if (account.pw_uid == 0 or account.pw_uid != runtime.uid or account.pw_gid != runtime.gid
                or not isinstance(runtime.unix_user, str)):
            raise ValueError("Curator runtime account identity changed")
        try:
            if pwd.getpwnam("alfworld_eval").pw_uid == account.pw_uid:
                raise ValueError("Curator cannot reuse the solver account")
        except KeyError:
            pass
        root = Path(runtime.work_root)
        if root.is_symlink() or not root.is_absolute() or root.resolve() == Path("/tmp").resolve():
            raise ValueError("Curator requires its explicit non-symlink work root")
        root = root.resolve(strict=True)
        info = root.stat()
        if (not root.is_relative_to(Path("/tmp").resolve()) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & 0o022):
            raise ValueError("Curator work root must be private host-provisioned state")
        package = Path(runtime.package_root).resolve(strict=True)
        if not package.is_dir():
            raise ValueError("Curator package root is unavailable")
        return cls(account.pw_name, account.pw_uid, account.pw_gid, str(root), str(package))


def curator_arguments(program, args, options, profiles):
    """Match only a registered CuratorSandbox runuser transport, never shell text."""
    if program != "/usr/sbin/runuser":
        return None
    environment = options.get("env")
    looks_curator = isinstance(environment, dict) and "CURATOR_CODEX_HOME" in environment
    if not looks_curator:
        return None
    if (len(args) != 6 or args[0] != "-u" or args[2:5] != ("--", "/bin/bash", "-c")
            or any(not isinstance(value, str) or "\0" in value for value in args)):
        raise ValueError("Unrecognized curator runuser launch")
    profile = profiles.get(args[1])
    if profile is None:
        raise ValueError("Curator account has not been registered with this spawn policy")
    account = pwd.getpwnam(profile.user)
    if (account.pw_uid, account.pw_gid) != (profile.uid, profile.gid) or account.pw_uid == 0:
        raise ValueError("Registered curator account changed")
    required = {"stdin": asyncio.subprocess.DEVNULL, "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE, "start_new_session": True}
    if (not set(required) | {"env", "cwd"} <= set(options)
            or set(options) - set(required) - {"env", "cwd", "close_fds", "pass_fds"}
            or any(options[key] != value for key, value in required.items())
            or options.get("close_fds", True) is not True or options.get("pass_fds", ()) != ()):
        raise ValueError("Unsupported curator subprocess options")
    cwd = Path(options["cwd"])
    if (not cwd.is_absolute() or cwd.parent != Path(profile.work_root)
            or re.fullmatch(r"[0-9a-f]{64}", cwd.name) is None or cwd.is_symlink()):
        raise ValueError("Curator cwd differs from its registered collection scope")
    home = cwd / ".home"
    for path in (cwd, home):
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or (info.st_uid, info.st_gid) != (profile.uid, profile.gid)
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("Curator workspace/home ownership or permissions changed")
    expected = {**_ENVIRONMENT, "HOME": str(home), "CURATOR_CODEX_HOME": str(home),
                "CODEX_MANAGED_PACKAGE_ROOT": profile.package_root}
    if environment != expected:
        raise ValueError("Curator environment differs from its clean registered profile")
    return [program, *args]


class AllNativeSpawner:
    def __init__(self, solver_helper, curator_helper):
        self.solver = solver._Spawner(solver_helper)
        self.curator_path = Path(curator_helper).resolve(strict=True)
        self.curator_sha256 = hashlib.sha256(self.curator_path.read_bytes()).hexdigest()
        self.library = ctypes.CDLL(str(self.curator_path))
        self.curator = self.library.curator_posix_spawn
        strings = ctypes.POINTER(ctypes.c_char_p)
        self.curator.argtypes = [ctypes.c_char_p, strings, strings, ctypes.c_char_p,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.curator.restype = ctypes.c_int
        self.sha256 = self.solver.sha256  # Existing solver transport provenance remains a component identity.

    def pidfd_open(self, pid):
        return self.solver.pidfd_open(pid)

    def pidfd_signal(self, descriptor, number):
        return self.solver.pidfd_signal(descriptor, number)

    def __call__(self, argv, environment, cwd, sources):
        if len(sources) != 4 or sources[3] != -1:
            return self.solver(argv, environment, cwd, sources)

        def vector(values):
            encoded = [os.fsencode(value) for value in values]
            if any(b"\0" in value for value in encoded):
                raise ValueError("NUL in curator spawn arguments")
            return (ctypes.c_char_p * (len(encoded) + 1))(*encoded, None)

        if any(not isinstance(key, str) or not key or "=" in key or "\0" in key
               or not isinstance(value, str) or "\0" in value for key, value in environment.items()):
            raise ValueError("Invalid curator spawn environment")
        directory = os.fsencode(cwd)
        if b"\0" in directory:
            raise ValueError("Invalid curator spawn cwd")
        pid = ctypes.c_int()
        error = self.curator(os.fsencode(argv[0]), vector(argv),
                             vector(f"{key}={value}" for key, value in environment.items()), directory,
                             (ctypes.c_int * 3)(*sources[:3]), ctypes.byref(pid))
        if error:
            raise OSError(error, os.strerror(error), argv[0])
        return pid.value


class AllNativeSelectorLoop(solver.NativeSpawnSelectorLoop):
    def __init__(self, spawner, profiles):
        super().__init__(spawner)
        self.curator_profiles = profiles

    async def _spawn_native(self, protocol_factory, argv, *, cwd, env, ready_fd):
        # A DataLoader os.fork must not inherit transient stdout/stderr writers
        # or C helper duplicates. Reader attachment can yield only AFTER those
        # parent writers have closed and the fork guard has been released.
        readers, pid, transport = {}, None, None
        protocol = protocol_factory()
        try:
            with _FORK_GUARD:
                owned = []
                try:
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
                    transport = solver.NativeSpawnTransport(self, protocol, pid, readers)
                    self.native_transports.add(transport)
                finally:
                    for fd in owned:
                        os.close(fd)
            await transport.connect()
            return transport, protocol
        except BaseException:
            if pid is not None and transport is None:
                os.kill(pid, signal.SIGKILL)  # Still this unreaped direct child; its numeric PID cannot be reused.
                os.waitpid(pid, 0)
            for reader in readers.values():
                reader.close()
            raise

    async def subprocess_exec(self, protocol_factory, program, *args, **options):
        argv = curator_arguments(program, args, options, self.curator_profiles)
        if argv is not None:
            return await self._spawn_native(protocol_factory, argv, cwd=options["cwd"],
                                            env=options["env"], ready_fd=-1)
        return await super().subprocess_exec(protocol_factory, program, *args, **options)


class AllNativePolicy(solver.NativeSpawnPolicy):
    def __init__(self, solver_helper=None, curator_helper=None):
        # Keep the existing solver identity and pidfd implementation unchanged.
        asyncio.DefaultEventLoopPolicy.__init__(self)
        self.helper_path = str(solver._helper_path(solver_helper))
        self.curator_helper_path = str(Path(curator_helper or Path(__file__).with_name(
            "curator_posix_spawn_helper.so")).resolve(strict=True))
        self.spawner = AllNativeSpawner(self.helper_path, self.curator_helper_path)
        self.curator_profiles = {}
        _install_fork_hooks()

    def register_curator(self, runtime):
        profile = CuratorProfile.from_runtime(runtime)
        old = self.curator_profiles.get(profile.user)
        if old is not None and old != profile:
            raise ValueError("A curator account cannot be rebound to another runtime profile")
        self.curator_profiles[profile.user] = profile
        return profile

    def new_event_loop(self):
        return AllNativeSelectorLoop(self.spawner, self.curator_profiles)


def install_policy(solver_helper=None, curator_helper=None):
    current = asyncio.get_event_loop_policy()
    if isinstance(current, AllNativePolicy):
        return current
    policy = AllNativePolicy(solver_helper, curator_helper)
    asyncio.set_event_loop_policy(policy)
    return policy


def identity(policy):
    files = {"python": Path(__file__).resolve(), "c": Path(__file__).with_name("curator_posix_spawn.c"),
             "helper": Path(policy.curator_helper_path)}
    current = {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for key, path in files.items()}
    if current["helper"]["sha256"] != policy.spawner.curator_sha256:
        raise RuntimeError("Curator spawn helper changed after loading")
    return {"policy": POLICY, "files": current, "loaded_curator_helper_sha256": policy.spawner.curator_sha256,
            "solver_component": solver.spawn_identity(policy.helper_path),
            "loop_class": AllNativeSelectorLoop.__module__ + "." + AllNativeSelectorLoop.__qualname__,
            "parent_fd_fork_guard": "Python os.register_at_fork exclusion; parent stdio writers close before await",
            "curator_profiles": [vars(profile) for profile in policy.curator_profiles.values()]}
