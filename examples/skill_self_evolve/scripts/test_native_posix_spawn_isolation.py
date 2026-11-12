#!/usr/bin/env python3
"""One CPU-only production isolation handshake with an explicitly owned UID.

Uses the provisioned pool's normal acquire/archive/close lifecycle.  It never
creates a model, ALFWorld world or backend session.  Failure quarantines this
test's exact lease; it never resets the pool or retries with another account.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import types
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
namespace = types.ModuleType("skill_harness_examples")
namespace.__path__ = [str(HERE)]
sys.modules[namespace.__name__] = namespace

from skill_harness_examples import native_posix_spawn as native  # noqa: E402
from verl.experimental.skill_harness.alfworld.accounts import (  # noqa: E402
    WORK_ROOT,
    AccountPool,
    decode_launcher_proof,
    launch_arguments,
)
from verl.experimental.skill_harness.alfworld.native_files import archive_native_files  # noqa: E402

CHILD = r"""
import json, os
fds = []
for fd in range(3, 1024):
    try: os.fstat(fd)
    except OSError: continue
    fds.append(fd)
with open('probe-sentinel.txt', 'w') as stream: stream.write('owned CPU probe\n')
print(json.dumps({'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
                  'groups': os.getgroups(), 'cwd': os.getcwd(),
                  'pid': os.getpid(), 'sid': os.getsid(0), 'fds': fds,
                  'stdin_eof': os.read(0, 1) == b''}))
"""


def tree_identity(root):
    if not root.is_dir():
        raise RuntimeError("The protected historical UID quarantine is missing")
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


async def handshake(run):
    pool = AccountPool()
    protected = pool.root / "quarantined-uids/20071"
    protected_before = tree_identity(protected)
    available_before = pool.available()
    scope, attempt = uuid.uuid4().hex, uuid.uuid4().hex
    output = run / "diagnostics" / ("account_preflight_" + scope)
    output.mkdir(parents=True, mode=0o700)
    lease = pool.acquire(run_root=run, scope_id=scope)
    reader = writer = None
    process = None
    released = False
    try:
        assert lease["uid"] != 20071
        pool.begin_attempt(lease, attempt_id=attempt, output=output)
        account = pool.account(lease)
        if WORK_ROOT.is_symlink() or not WORK_ROOT.is_dir() or WORK_ROOT.stat().st_uid != os.geteuid():
            raise RuntimeError("The provisioned work parent differs from the trusted host parent")
        cwd, home = WORK_ROOT / uuid.uuid4().hex, Path("/tmp") / ("codex-home-" + uuid.uuid4().hex)
        for path, kind in ((cwd, "workspace"), (home, "codex_home")):
            path.mkdir(mode=0o700)
            os.chown(path, account.pw_uid, account.pw_gid)
            pool.register_resource(lease, attempt, path, kind)
        reader, writer = os.pipe()
        argv, controlled, _ = launch_arguments(
            account, ["/usr/bin/python3", "-c", CHILD], home=home, cwd=cwd, ready_fd=writer,
        )
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **controlled},
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, pass_fds=(writer,), close_fds=True,
            start_new_session=True,
        )
        os.close(writer)
        writer = None
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
        proof = decode_launcher_proof(os.read(reader, 65536), uid=account.pw_uid, gid=account.pw_gid)
        pool.mark_launcher(lease, attempt, proof)
        if process.returncode != 0:
            raise RuntimeError("Production isolation child failed: " + stderr.decode(errors="replace")[:500])
        observed = json.loads(stdout)
        assert observed["uid"] == observed["euid"] == observed["gid"] == account.pw_uid
        assert observed["groups"] == [] and observed["cwd"] == str(cwd)
        assert observed["pid"] == observed["sid"] == process.pid
        assert observed["fds"] == [] and observed["stdin_eof"]
        assert not stderr
        try:
            os.waitpid(process.pid, os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            raise RuntimeError("Production child was not reaped")
        pool.drain(lease)
        assert not pool.processes(account.pw_uid)
        (output / "production-handshake.json").write_text(json.dumps({
            "kernel_proof": proof, "child_observed": observed,
            "returncode": process.returncode,
        }, indent=2) + "\n")
        archive = archive_native_files(
            output, codex_homes=[home], workspace=cwd, agent_uid=account.pw_uid,
            account_lease_id=lease["lease_id"],
        )
        closure = pool.close_attempt(
            lease, attempt, receipt_path=output / "native-files.json",
            backend_drained=True, native_started=True,
        )
        released = True
        state = json.loads((pool.root / "state.json").read_text())
        lease_state = json.loads((pool.root / "leases" / (lease["lease_id"] + ".json")).read_text())
        assert lease_state["status"] == "released" and state["slots"][str(account.pw_uid)] is None
        assert tree_identity(protected) == protected_before
        report = {
            "schema_version": 1, "status": "passed", "observed_at_ns": time.time_ns(),
            "scope": "one actual production matcher/setpriv/isolate-exec CPU handshake",
            "model_or_gpu_or_backend_session_created": False,
            "allocated_uid": account.pw_uid, "available_before": available_before,
            "available_after": pool.available(), "kernel_proof": proof,
            "child_observed": observed, "returncode": process.returncode,
            "native_spawn_identity": native.spawn_identity(),
            "closure": {key: closure[key] for key in ("status", "backend_drained", "uid_drained")},
            "backend_drained_scope": "No backend session was created by this CPU probe",
            "released_own_slot": True, "protected_uid20071_quarantine_unchanged": True,
            "archive_sha256": archive["archive_sha256"], "output_dir": str(output),
        }
        path = run / "production-isolation-result.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "passed", "receipt": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "allocated_uid": account.pw_uid, "released_own_slot": True}))
    except BaseException as error:
        if not released:
            try:
                pool.drain(lease)
                if process is not None and process.returncode is None:
                    await asyncio.wait_for(process.communicate(), 5)
            except BaseException as cleanup_error:
                error.add_note("Owned cleanup also failed: " + str(cleanup_error))
            finally:
                try:
                    pool.quarantine(lease, error, attempt_id=attempt)
                except BaseException as quarantine_error:
                    error.add_note("Owned quarantine also failed: " + str(quarantine_error))
        raise
    finally:
        for fd in (reader, writer):
            if fd is not None:
                os.close(fd)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Supply a fresh private staging run directory")
    run = Path(sys.argv[1]).resolve()
    run.mkdir(mode=0o700)
    native.install_spawn_policy()
    asyncio.run(handshake(run))
