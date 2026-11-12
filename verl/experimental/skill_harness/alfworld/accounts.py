"""Host-owned UID leases and an inherited Landlock filesystem boundary.

No accounts are created on import or by a Task. The explicit CLI can print a
provisioning plan and initialize the private lease registry after an operator
has provisioned and validated the fixed roster. Failed leases are quarantined;
PID disappearance and elapsed time never make a lease reusable.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import fcntl
import grp
import hashlib
import json
import os
import platform
import pwd
import re
import signal
import stat
import sys
import tarfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

POLICY_VERSION = "alfworld-uid-landlock-v1"
SECCOMP_GUARD = "abi1-truncate-x86_64-v1"
POOL_ROOT = Path("/var/lib/alfworld-skill-harness/account_pool_v2")
UID_BASE = 20000
POOL_SIZE = 128
ACCOUNT_PREFIX = "alfworld_ep2_"
ACCOUNT_COMMENT = "ALFWorld private attempt account v1"
INCIDENT_DIRECTORY = "diagnostics/account_isolation_incidents"
DATA_INCIDENT_DIRECTORY = "diagnostics/local_archive_data_incidents"
WORK_ROOT = Path("/tmp/alfworld-skill-episodes")
_NAME = re.compile(r"[0-9a-f]{32}\Z")
_GROUP = re.compile(r"[0-9a-f]{64}\Z")


class AccountIsolationError(RuntimeError):
    """A host isolation/lifecycle failure; never an agent outcome reward."""


class LocalArchiveDataError(RuntimeError):
    """A drained attempt's raw bundle is unavailable; never an RL outcome."""

    def __init__(self, reason, disposition=None):
        super().__init__(str(reason))
        self.disposition = disposition or {}


def is_local_archive_failure(error):
    """Recognize storage/readback failures, not owner/path/source races."""
    native = Path(__file__).with_name("native_files.py").resolve()
    frames = []
    current = error.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        if Path(code.co_filename).resolve() == native:
            frames.append(code.co_name)
        current = current.tb_next
    if "verify_native_archive" in frames:
        return isinstance(error, ValueError | OSError | EOFError | tarfile.TarError)
    return (
        "archive_native_files" in frames
        and isinstance(error, OSError)
        and error.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS, errno.EFBIG)
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _atomic(path, value):
    path = Path(path)
    stage = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with stage.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        stage.unlink(missing_ok=True)


def _private(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise AccountIsolationError("The lease/incident directory is not private and host-owned")
    return path


def record_isolation_incident(run_root, reason, *, lease_id=None, attempt_id=None):
    path = _private(Path(run_root) / INCIDENT_DIRECTORY) / (uuid.uuid4().hex + ".json")
    _atomic(
        path,
        {
            "schema_version": 1,
            "policy_version": POLICY_VERSION,
            "time": time.time(),
            "reason": str(reason),
            "lease_id": lease_id,
            "attempt_id": attempt_id,
        },
    )
    return path


def assert_no_isolation_incidents(run_root):
    path = Path(run_root) / INCIDENT_DIRECTORY
    if path.exists() or path.is_symlink():
        raise AccountIsolationError("An unresolved host isolation incident forbids an optimizer update: " + str(path))


def local_archive_failed_groups(run_root, *, pool_root=POOL_ROOT):
    """Read settled local archive failures before sampling any sibling rows."""
    failed = set()
    run_root = str(Path(run_root).resolve())
    for path in (Path(pool_root) / "quarantined-uids").glob("*/local-archive.json"):
        value = json.loads(path.read_text())
        if value.get("account", {}).get("owner", {}).get("run_root") != run_root:
            continue
        identity = value.get("identity", {})
        if identity.get("dataset_phase") != "train" or identity.get("role") not in (None, "solver"):
            continue
        if (
            value.get("kind") != "local_archive_data_invalid"
            or value.get("training_data_valid") is not False
            or value.get("uid_drained") is not True
            or value.get("backend_drained") is not True
            or not isinstance(identity.get("rollout_uid"), str)
            or not identity["rollout_uid"]
        ):
            raise AccountIsolationError("The local archive failure registry has an invalid prompt identity")
        failed.add(identity["rollout_uid"])
    return failed


def expected_roster(*, base=UID_BASE, size=POOL_SIZE):
    return [
        {"name": ACCOUNT_PREFIX + f"{index:03d}", "uid": base + index, "gid": base + index} for index in range(size)
    ]


def validate_system_roster(roster=None):
    """Read system identity; never alter an existing user/group or password."""
    roster = expected_roster() if roster is None else roster
    users, groups = pwd.getpwall(), grp.getgrall()
    shadow = {}
    try:
        for line in Path("/etc/shadow").read_text().splitlines():
            fields = line.split(":")
            if len(fields) > 1:
                shadow[fields[0]] = fields[1]
    except OSError as error:
        raise AccountIsolationError("The host cannot validate locked pool passwords") from error
    for item in roster:
        matches = [user for user in users if user.pw_name == item["name"] or user.pw_uid == item["uid"]]
        matches_group = [group for group in groups if group.gr_name == item["name"] or group.gr_gid == item["gid"]]
        if len(matches) != 1 or len(matches_group) != 1:
            raise AccountIsolationError("Provision the exact private UID roster before launch: " + item["name"])
        user, group = matches[0], matches_group[0]
        if (
            user.pw_name != item["name"]
            or user.pw_uid != item["uid"]
            or user.pw_gid != item["gid"]
            or user.pw_dir != "/nonexistent"
            or user.pw_shell not in ("/usr/sbin/nologin", "/sbin/nologin")
            or user.pw_gecos != ACCOUNT_COMMENT
            or group.gr_name != item["name"]
            or group.gr_gid != item["gid"]
            or group.gr_mem
            or not shadow.get(item["name"], "").startswith(("!", "*"))
            or any(item["name"] in candidate.gr_mem for candidate in groups)
        ):
            raise AccountIsolationError("An existing account/group has incompatible attributes: " + item["name"])
    return roster


def provisioning_plan():
    """Print commands for absent identities only; conflicts fail closed."""
    users, groups = pwd.getpwall(), grp.getgrall()
    commands = []
    for item in expected_roster():
        names = [user for user in users if user.pw_name == item["name"] or user.pw_uid == item["uid"]]
        gids = [group for group in groups if group.gr_name == item["name"] or group.gr_gid == item["gid"]]
        if names or gids:
            if not names or not gids:
                raise AccountIsolationError("Partial or conflicting pre-existing pool identity")
            validate_system_roster([item])
            continue
        commands += [
            ["/usr/sbin/groupadd", "--gid", str(item["gid"]), item["name"]],
            [
                "/usr/sbin/useradd",
                "--uid",
                str(item["uid"]),
                "--gid",
                str(item["gid"]),
                "--no-create-home",
                "--no-log-init",
                "--home-dir",
                "/nonexistent",
                "--shell",
                "/usr/sbin/nologin",
                "--comment",
                ACCOUNT_COMMENT,
                "--password",
                "!",
                item["name"],
            ],
        ]
    return commands


def uid_processes(uid):
    """UID ownership is sufficient only while that UID has an exclusive lease."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            text = (proc / "status").read_text()
            ids = next(line.split()[1:] for line in text.splitlines() if line.startswith("Uid:"))
            if str(uid) in ids:
                found.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, StopIteration) as error:
            raise AccountIsolationError("Cannot prove the UID has no remaining processes") from error
    return sorted(found)


class AccountPool:
    """Persistent exclusive leases; paired callers retain one UID across variants."""

    def __init__(self, root=POOL_ROOT, *, roster=None, processes=None, validate=None, signal_process=None):
        self.root = Path(root)
        self.roster = expected_roster() if roster is None else roster
        self.processes = uid_processes if processes is None else processes
        self.validate = validate_system_roster if validate is None else validate
        self.signal_process = _signal_uid_process if signal_process is None else signal_process

    def initialize(self):
        self.validate(self.roster)
        _private(self.root)
        _private(self.root / "leases")
        with self._lock(initializing=True):
            path = self.root / "state.json"
            if not path.exists():
                _atomic(
                    path,
                    {
                        "schema_version": 1,
                        "policy_version": POLICY_VERSION,
                        "roster": self.roster,
                        "cursor": 0,
                        "slots": {str(x["uid"]): None for x in self.roster},
                        "owners": {},
                    },
                )
            self._state()

    @contextlib.contextmanager
    def _lock(self, *, initializing=False):
        if not initializing and not (self.root / "state.json").is_file():
            raise AccountIsolationError("The private account pool has not been initialized")
        _private(self.root)
        with (self.root / ".lock").open("a+b") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _state(self):
        value = json.loads((self.root / "state.json").read_text())
        if value.get("policy_version") != POLICY_VERSION or value.get("roster") != self.roster:
            raise AccountIsolationError("The pool policy or system roster differs")
        return value

    def _lease(self, ref, *, allow_released=False):
        if not _NAME.fullmatch(str(ref.get("lease_id", ""))):
            raise AccountIsolationError("Invalid lease identity")
        value = json.loads((self.root / "leases" / (ref["lease_id"] + ".json")).read_text())
        if any(value.get(key) != ref.get(key) for key in ("lease_id", "token", "uid", "gid", "name", "owner")):
            raise AccountIsolationError("Lease fencing or owner identity differs")
        if self._blocked(value["uid"]) and not allow_released:
            marker = self.root / "quarantined-uids" / str(value["uid"]) / "local-archive.json"
            if marker.is_file():
                disposition = json.loads(marker.read_text())
                if (
                    disposition.get("kind") == "local_archive_data_invalid"
                    and disposition.get("account", {}).get("lease_id") == ref["lease_id"]
                    and disposition.get("uid_drained") is True
                    and disposition.get("backend_drained") is True
                ):
                    raise LocalArchiveDataError(disposition["reason"], disposition)
            raise AccountIsolationError("The UID has a sticky quarantine; no active operation is allowed")
        if value["status"] != "leased" and not (allow_released and value["status"] == "released"):
            raise AccountIsolationError("Lease is not active; explicit host reconciliation is required")
        if value["status"] == "leased" and self._state()["slots"].get(str(value["uid"])) != value["lease_id"]:
            raise AccountIsolationError("UID slot is not exclusively owned by this lease")
        return value

    @staticmethod
    def public(ref):
        return {key: ref[key] for key in ("lease_id", "uid", "gid", "name", "owner")}

    def _save(self, lease):
        _atomic(self.root / "leases" / (lease["lease_id"] + ".json"), lease)

    def _blocked(self, uid):
        path = self.root / "quarantined-uids" / str(uid)
        return path.exists() or path.is_symlink()

    def available(self):
        self.validate(self.roster)
        with self._lock():
            state = self._state()
            return sum(
                state["slots"][str(item["uid"])] is None
                and not self._blocked(item["uid"])
                and not self.processes(item["uid"])
                for item in self.roster
            )

    def acquire(self, *, run_root, scope_id, kind="ordinary"):
        if kind not in ("ordinary", "pair") or (kind == "pair" and not _GROUP.fullmatch(scope_id)):
            raise AccountIsolationError("Unknown lease kind or paired group identity")
        self.validate(self.roster)
        run_root = str(Path(run_root).resolve())
        owner = {"run_root": run_root, "scope_id": str(scope_id), "kind": kind}
        owner_key = hashlib.sha256(_canonical(owner)).hexdigest()
        with self._lock():
            state = self._state()
            if owner_key in state["owners"]:
                raise AccountIsolationError("A logical scope already has a retained lease; no automatic reattachment")
            for offset in range(len(self.roster)):
                index = (state["cursor"] + offset) % len(self.roster)
                account = self.roster[index]
                if (
                    state["slots"][str(account["uid"])] is not None
                    or self._blocked(account["uid"])
                    or self.processes(account["uid"])
                ):
                    continue
                value = {
                    **account,
                    "schema_version": 1,
                    "policy_version": POLICY_VERSION,
                    "lease_id": uuid.uuid4().hex,
                    "token": uuid.uuid4().hex + uuid.uuid4().hex,
                    "owner": owner,
                    "owner_key": owner_key,
                    "status": "leased",
                    "created_at": time.time(),
                    "active_attempt": None,
                    "attempts": {},
                }
                self._save(value)  # Ownership is durable before any native file/process can exist.
                state["slots"][str(account["uid"])] = value["lease_id"]
                state["owners"][owner_key] = value["lease_id"]
                state["cursor"] = (index + 1) % len(self.roster)
                _atomic(self.root / "state.json", state)
                return {key: value[key] for key in ("lease_id", "token", "uid", "gid", "name", "owner")}
        raise AccountIsolationError("No clean exclusive UID is available; shared-user fallback is forbidden")

    def begin_attempt(self, ref, *, attempt_id, output, variant="ordinary"):
        with self._lock():
            lease = self._lease(ref)
            if (
                not _NAME.fullmatch(attempt_id)
                or lease["active_attempt"] is not None
                or attempt_id in lease["attempts"]
                or variant in {x["variant"] for x in lease["attempts"].values()}
                or (lease["owner"]["kind"] == "pair" and variant not in ("with_skill", "without_skill"))
                or (lease["owner"]["kind"] == "ordinary" and variant != "ordinary")
            ):
                raise AccountIsolationError("Lease already owns an active/repeated or invalid attempt")
            output = Path(output).resolve()
            if not output.is_relative_to(Path(lease["owner"]["run_root"])):
                raise AccountIsolationError("Attempt output escaped its run")
            lease["active_attempt"] = attempt_id
            lease["attempts"][attempt_id] = {
                "variant": variant,
                "output": str(output),
                "status": "active",
                "resources": [],
                "launcher_proofs": [],
                "started_at": time.time(),
            }
            self._save(lease)

    def account(self, ref):
        with self._lock():
            self._lease(ref)
        selected = [{key: ref[key] for key in ("uid", "gid", "name")}]
        self.validate(selected)
        return SimpleNamespace(pw_name=ref["name"], pw_uid=ref["uid"], pw_gid=ref["gid"], pw_dir="/nonexistent")

    def register_resource(self, ref, attempt_id, path, kind):
        path = Path(path)
        allowed = (
            (
                kind == "codex_home"
                and path.parent == Path("/tmp")
                and re.fullmatch(r"codex-home-[0-9a-f]{32}", path.name)
            )
            or (
                kind == "model_catalog"
                and path.parent == Path("/tmp")
                and re.fullmatch(r"codex-model-catalog-[0-9a-f]{32}\.json", path.name)
            )
            or (
                kind == "workspace"
                and path.parent == WORK_ROOT
                and (re.fullmatch(r"[0-9a-f]{32}", path.name) or path.name == "credit_" + ref["owner"]["scope_id"])
            )
        )
        if not allowed or path.is_symlink():
            raise AccountIsolationError("Native resource is outside its exact host-assigned namespace")
        info = path.lstat()
        expected_uid = os.geteuid() if kind == "model_catalog" else ref["uid"]
        expected_mode = 0o440 if kind == "model_catalog" else 0o700
        if info.st_uid != expected_uid or info.st_gid != ref["gid"] or stat.S_IMODE(info.st_mode) != expected_mode:
            raise AccountIsolationError("Native resource has an unexpected owner/group/mode")
        with self._lock():
            lease = self._lease(ref)
            if lease["active_attempt"] != attempt_id:
                raise AccountIsolationError("Resource caller does not own the active attempt")
            resources = lease["attempts"][attempt_id]["resources"]
            row = {
                "path": str(path),
                "kind": kind,
                "device": info.st_dev,
                "inode": info.st_ino,
                "uid": info.st_uid,
                "gid": info.st_gid,
            }
            if row not in resources:
                if any(x["path"] == str(path) for x in resources):
                    raise AccountIsolationError("A native resource path was replaced")
                resources.append(row)
                self._save(lease)

    def mark_launcher(self, ref, attempt_id, proof):
        if (
            proof.get("policy_version") != POLICY_VERSION
            or proof.get("uid") != ref["uid"]
            or proof.get("gid") != ref["gid"]
            or proof.get("groups") != []
            or proof.get("no_new_privs") != 1
            or proof.get("capabilities_zero") is not True
            or proof.get("resuid") != [ref["uid"]] * 3
            or proof.get("resgid") != [ref["gid"]] * 3
            or proof.get("landlock_abi", 0) < 1
            or proof.get("landlock_applied") is not True
            or proof.get("seccomp_guard") != SECCOMP_GUARD
        ):
            raise AccountIsolationError("The actual launcher did not establish the required kernel boundary")
        with self._lock():
            lease = self._lease(ref)
            if lease["active_attempt"] != attempt_id:
                raise AccountIsolationError("Launcher caller does not own the attempt")
            lease["attempts"][attempt_id]["launcher_proofs"].append(proof)
            self._save(lease)

    def drain(self, ref, *, timeout=5.0):
        with self._lock():
            self._lease(ref)
        deadline = time.monotonic() + timeout
        for number in (signal.SIGTERM, signal.SIGKILL):
            for pid in self.processes(ref["uid"]):
                if pid == os.getpid():
                    raise AccountIsolationError("A native UID lease must never identify the host process")
                # Keep fencing validation and pidfd signal in one short lease
                # lock: a paused old drain must never signal a re-leased UID.
                with self._lock():
                    self._lease(ref)
                    self.signal_process(pid, ref["uid"], number)
            until = min(deadline, time.monotonic() + (1.0 if number == signal.SIGTERM else timeout))
            while self.processes(ref["uid"]) and time.monotonic() < until:
                # Reap only already-terminated children belonging to this host.
                for pid in self.processes(ref["uid"]):
                    with contextlib.suppress(ChildProcessError, ProcessLookupError):
                        os.waitpid(pid, os.WNOHANG)
                time.sleep(0.02)
            if not self.processes(ref["uid"]):
                return {"uid": ref["uid"], "remaining_pids": [], "drained": True}
        raise AccountIsolationError("Native UID processes remain; quarantine rather than reuse")

    def quarantine(self, ref, reason, *, attempt_id=None):
        # Directory existence is the sticky barrier even if detail/state writes fail.
        _private(Path(ref["owner"]["run_root"]) / INCIDENT_DIRECTORY)
        _private(self.root / "quarantined-uids" / str(ref["uid"]))
        record_isolation_incident(ref["owner"]["run_root"], reason, lease_id=ref["lease_id"], attempt_id=attempt_id)
        with self._lock():
            lease = self._lease(ref, allow_released=True)
            lease.update(status="quarantined", quarantine_reason=str(reason), quarantined_at=time.time())
            self._save(lease)

    def quarantine_local_archive(self, ref, reason, *, attempt_id, backend_receipt, backend_binding, metadata=None):
        """Hold only this drained UID; incomplete raw bundles are not RL labels.

        The caller classifies a known archive storage/readback error. This
        method rechecks the exclusive owner and actual process/backend drain.
        Existing global incidents and older quarantine markers are never reset.
        """
        keys = ("binding_id", "binding_sha256", "attempt_id", "gateway_session_id", "environment_session_id")
        bound = {
            key: backend_binding.get(key) if isinstance(backend_binding, dict) else getattr(backend_binding, key, None)
            for key in keys
        }
        if (
            not isinstance(backend_receipt, dict)
            or backend_receipt.get("backend_stop_confirmed") is not True
            or backend_receipt.get("active_requests") != 0
            or backend_receipt.get("backend_error_types", [])
            or bound["attempt_id"] != attempt_id
            or any(not isinstance(bound[k], str) or not bound[k] or backend_receipt.get(k) != bound[k] for k in keys)
        ):
            raise AccountIsolationError("A local archive disposition requires this exact confirmed backend drain")
        with self._lock():
            lease = self._lease(ref)
            if lease["active_attempt"] != attempt_id or self.processes(ref["uid"]):
                raise AccountIsolationError("A local archive disposition requires the exclusive drained UID")
            attempt = lease["attempts"][attempt_id]
            disposition = {
                "schema_version": 1,
                "kind": "local_archive_data_invalid",
                "policy_version": POLICY_VERSION,
                "account": self.public(ref),
                "attempt_id": attempt_id,
                "output_dir": attempt["output"],
                "reason": str(reason),
                "time": time.time(),
                "backend_binding": bound,
                "backend_receipt_sha256": hashlib.sha256(_canonical(backend_receipt)).hexdigest(),
                "uid_drained": True,
                "backend_drained": True,
                "training_data_valid": False,
                "resources_retained_for_reconciliation": list(attempt["resources"]),
                "identity": {
                    k: metadata[k]
                    for k in ("rollout_uid", "session_index", "game_id", "dataset_phase", "role", "global_step")
                    if metadata is not None and k in metadata
                },
            }
            marker = _private(self.root / "quarantined-uids" / str(ref["uid"])) / "local-archive.json"
            _atomic(marker, disposition)  # Local persistent control state, even if NAS archival is unavailable.
            lease.update(status="quarantined", quarantine_kind="local_archive", quarantine_reason=str(reason))
            self._save(lease)
        try:
            report = _private(Path(ref["owner"]["run_root"]) / DATA_INCIDENT_DIRECTORY) / (attempt_id + ".json")
            _atomic(report, disposition)
            disposition["report_path"] = str(report)
        except OSError as error:
            # Whole-group failure accounting uses the Task error, not this
            # optional NAS copy. The durable per-UID marker still forbids reuse.
            disposition["report_copy_error"] = type(error).__name__
        disposition["uid_marker_path"] = str(marker)
        return disposition

    def close_empty_attempt(self, ref, attempt_id, *, backend_drained):
        """No program/resource was started; still persist an explicit closure."""
        with self._lock():
            lease = self._lease(ref)
            attempt = lease["attempts"][attempt_id]
            if (
                not backend_drained
                or self.processes(ref["uid"])
                or attempt["resources"]
                or attempt["launcher_proofs"]
                or lease["active_attempt"] != attempt_id
            ):
                raise AccountIsolationError("An empty-attempt abandonment has unclosed resources")
            attempt.update(
                status="closed",
                closed_at=time.time(),
                no_native_resources_created=True,
                backend_drained=True,
                uid_drained=True,
            )
            lease["active_attempt"] = None
            self._save(lease)
        if ref["owner"]["kind"] == "ordinary":
            self.finish(ref)

    def close_attempt(self, ref, attempt_id, *, receipt_path, backend_drained, native_started):
        try:
            if not backend_drained or self.processes(ref["uid"]):
                raise AccountIsolationError("Attempt process/backend drain is not confirmed")
            receipt_path = Path(receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("status") != "archived_and_removed"
                or receipt.get("account_lease_id") != ref["lease_id"]
                or receipt.get("agent_uid") != ref["uid"]
            ):
                raise AccountIsolationError("A complete archive receipt is not bound to this UID lease")
            if (
                _sha(receipt["archive_path"]) != receipt["archive_sha256"]
                or _sha(receipt["source_manifest_path"]) != receipt["source_manifest_sha256"]
            ):
                raise AccountIsolationError("Retained native files changed after readback")
            with self._lock():
                lease = self._lease(ref)
                attempt = lease["attempts"][attempt_id]
                if lease["active_attempt"] != attempt_id or (native_started and not attempt["launcher_proofs"]):
                    raise AccountIsolationError("Attempt identity or kernel launcher attestation is missing")
                removed = {x["path"]: x for x in receipt["sources"]}
                for resource in attempt["resources"]:
                    row = removed.get(resource["path"], {})
                    if (
                        row.get("removed") is not True
                        or os.path.lexists(resource["path"])
                        or row.get("device") != resource["device"]
                        or row.get("inode") != resource["inode"]
                        or row.get("original_uid") != resource["uid"]
                        or row.get("original_gid") != resource["gid"]
                    ):
                        raise AccountIsolationError("Not every exact owned resource was durably archived and removed")
                attempt.update(
                    status="closed",
                    closed_at=time.time(),
                    archive_receipt=str(receipt_path),
                    archive_receipt_sha256=_sha(receipt_path),
                    backend_drained=True,
                    uid_drained=True,
                )
                lease["active_attempt"] = None
                self._save(lease)
            if ref["owner"]["kind"] == "ordinary":
                self.finish(ref)
            return {
                "policy_version": POLICY_VERSION,
                **self.public(ref),
                "status": "attempt_closed",
                "archive_receipt": str(receipt_path),
                "backend_drained": True,
                "uid_drained": True,
            }
        except LocalArchiveDataError:
            raise
        except Exception as error:
            with contextlib.suppress(Exception):
                self.quarantine(ref, error, attempt_id=attempt_id)
            if isinstance(error, AccountIsolationError):
                raise
            raise AccountIsolationError("Native lease closure failed; original resources/evidence retained") from error

    def finish(self, ref, *, abandoned=False):
        """Pair finally may abandon an unstarted second variant, never its data."""
        try:
            if self.processes(ref["uid"]):
                self.drain(ref)
            with self._lock():
                try:
                    lease = self._lease(ref)
                    attempts = lease["attempts"]
                    variants = {x["variant"] for x in attempts.values()}
                    if (
                        lease["active_attempt"] is not None
                        or any(x["status"] != "closed" for x in attempts.values())
                        or (
                            lease["owner"]["kind"] == "pair"
                            and not abandoned
                            and variants != {"with_skill", "without_skill"}
                        )
                    ):
                        raise AccountIsolationError("Cannot release an unfinished logical scope")
                    if self.processes(ref["uid"]):
                        raise AccountIsolationError("UID process appeared before release")
                    lease.update(status="released", released_at=time.time(), abandoned=bool(abandoned))
                    self._save(lease)
                    state = self._state()
                    if state["slots"][str(ref["uid"])] != ref["lease_id"]:
                        raise AccountIsolationError("UID fencing changed before release")
                    state["slots"][str(ref["uid"])] = None
                    state["owners"].pop(lease["owner_key"], None)
                    _atomic(self.root / "state.json", state)
                except LocalArchiveDataError:
                    raise
                except Exception:
                    _private(Path(ref["owner"]["run_root"]) / INCIDENT_DIRECTORY)
                    _private(self.root / "quarantined-uids" / str(ref["uid"]))
                    raise
        except LocalArchiveDataError:
            raise
        except Exception as error:
            with contextlib.suppress(Exception):
                self.quarantine(ref, error)
            if isinstance(error, AccountIsolationError):
                raise
            raise AccountIsolationError("UID release failed; explicit reconciliation required") from error

    def historical_pair(self, public, *, run_root, group):
        if not _NAME.fullmatch(str(public.get("lease_id", ""))):
            raise AccountIsolationError("Invalid historical paired lease identity")
        value = json.loads((self.root / "leases" / (public["lease_id"] + ".json")).read_text())
        if (
            value["status"] != "released"
            or value["owner"] != {"run_root": str(Path(run_root).resolve()), "scope_id": group, "kind": "pair"}
            or any(value[k] != public[k] for k in ("uid", "gid", "name", "owner"))
            or value.get("abandoned") is True
            or {x["variant"] for x in value["attempts"].values()} != {"with_skill", "without_skill"}
        ):
            raise AccountIsolationError("The cached pair has no complete historical UID lease")
        return value


def _native_pidfd_call(name, syscall_number, argtypes, values):
    """Use the same Linux pidfd mechanism when CPython omits its wrappers."""
    if sys.platform != "linux":
        raise AccountIsolationError("Linux pidfd process ownership is required for native cleanup")
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, name, None)
    if function is not None:
        function.argtypes, function.restype = list(argtypes), ctypes.c_int
        result = function(*values)
    else:
        # These numbers are verified against the production x86_64 headers.
        # Do not assume another architecture shares this syscall table.
        if platform.machine() != "x86_64":
            raise AccountIsolationError("This architecture has no verified pidfd syscall compatibility path")
        libc.syscall.restype = ctypes.c_long
        result = libc.syscall(
            ctypes.c_long(syscall_number), *(kind(value) for kind, value in zip(argtypes, values, strict=True))
        )
    if result < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return result


def _open_pidfd(pid):
    function = getattr(os, "pidfd_open", None)
    if callable(function):
        return function(pid, 0)
    return _native_pidfd_call("pidfd_open", 434, (ctypes.c_int, ctypes.c_uint), (pid, 0))


def _send_pidfd_signal(descriptor, number):
    function = getattr(signal, "pidfd_send_signal", None)
    if callable(function):
        return function(descriptor, number, None, 0)
    return _native_pidfd_call(
        "pidfd_send_signal",
        424,
        (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint),
        (descriptor, number, None, 0),
    )


def _signal_uid_process(pid, uid, number):
    """Signal the opened process identity, not a reusable numeric PID."""
    try:
        descriptor = _open_pidfd(pid)
    except ProcessLookupError:
        return
    try:
        try:
            text = Path("/proc", str(pid), "status").read_text()
            ids = next(line.split()[1:] for line in text.splitlines() if line.startswith("Uid:"))
        except (FileNotFoundError, ProcessLookupError):
            return
        if str(uid) not in ids:
            raise AccountIsolationError("A process UID changed before its pidfd signal")
        with contextlib.suppress(ProcessLookupError):
            _send_pidfd_signal(descriptor, number)
    finally:
        os.close(descriptor)


def landlock_abi():
    if sys.platform != "linux":
        raise AccountIsolationError("The UID filesystem boundary requires Linux Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1))
    if result < 1:
        raise AccountIsolationError("Landlock ABI1 or newer is required; no unconfined fallback")
    return int(result)


def _landlock(read_roots, write_roots):
    """ABI1 path-beneath allowlist, inherited across fork/exec and non-relaxable."""
    abi = landlock_abi()
    libc = ctypes.CDLL(None, use_errno=True)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    def call(number, *arguments):
        result = libc.syscall(number, *arguments)
        if result < 0:
            raise AccountIsolationError("Landlock operation failed: errno=" + str(ctypes.get_errno()))
        return result

    # ABI1 handles basic reads/writes/creation. It does not claim network or
    # metadata-stat isolation; distinct UID/DAC and no capabilities remain required.
    rules = Ruleset((1 << 13) - 1)
    descriptor = call(444, ctypes.byref(rules), ctypes.sizeof(rules), 0)
    try:
        for writable, roots in ((False, read_roots), (True, write_roots)):
            for raw in dict.fromkeys(roots):
                path = Path(raw).resolve(strict=True)
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    directory = stat.S_ISDIR(os.fstat(fd).st_mode)
                    access = (1 << 0) | (1 << 2) | ((1 << 3) if directory else 0)
                    if writable:
                        access |= 1 << 1
                        if directory:
                            access |= sum(1 << bit for bit in (4, 5, 7, 8, 9, 10, 12))
                    rule = Beneath(access, fd)
                    call(445, descriptor, 1, ctypes.byref(rule), 0)
                finally:
                    os.close(fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise AccountIsolationError("Cannot establish no_new_privs")
        call(446, descriptor, 0)
    finally:
        os.close(descriptor)
    return abi


def _seccomp_program():
    """Classic BPF for the narrow ABI1 truncate hole, x86_64 only.

    Landlock still decides path access. A read-only open with O_TRUNC and the
    path-based truncate syscall must not evade WRITE_FILE. openat2's pointer
    flags and io_uring operations cannot be inspected by classic BPF: ENOSYS
    makes supported applications take their normal openat/synchronous fallback.
    """
    # Tuples are sock_filter(code, jt, jf, k). Labels keep jump offsets auditable.
    program, labels, jumps = [], {}, []

    def emit(code, k, yes=0, no=0):
        program.append([code, yes, no, k])

    def label(name):
        labels[name] = len(program)

    def equal(value, yes, no):
        jumps.append((len(program), yes, no))
        emit(0x15, value)

    def greater_equal(value, yes, no):
        jumps.append((len(program), yes, no))
        emit(0x35, value)

    emit(0x20, 4)  # seccomp_data.arch
    equal(0xC000003E, "number", "deny")
    label("number")
    emit(0x20, 0)  # seccomp_data.nr
    greater_equal(0x40000000, "nosys", "truncate")  # Reject x32 ABI aliases.
    label("truncate")
    equal(76, "deny", "openat2")
    label("openat2")
    equal(437, "nosys", "uring_setup")
    label("uring_setup")
    equal(425, "nosys", "uring_enter")
    label("uring_enter")
    equal(426, "nosys", "uring_register")
    label("uring_register")
    equal(427, "nosys", "open")
    label("open")
    equal(2, "open_flags", "openat")
    label("openat")
    equal(257, "openat_flags", "allow")
    label("open_flags")
    emit(0x20, 24)  # args[1] low flags word
    jumps.append((len(program), "check_flags", None))
    emit(0x05, 0)  # JA
    label("openat_flags")
    emit(0x20, 32)  # args[2] low flags word
    label("check_flags")
    emit(0x54, os.O_ACCMODE | os.O_TRUNC)  # AND K
    equal(os.O_TRUNC, "deny", "ioctl_mode")  # read-only access mode is zero.
    label("ioctl_mode")
    equal(os.O_TRUNC | os.O_ACCMODE, "deny", "allow")
    label("allow")
    emit(0x06, 0x7FFF0000)  # SECCOMP_RET_ALLOW
    label("deny")
    emit(0x06, 0x00050000 | errno.EPERM)
    label("nosys")
    emit(0x06, 0x00050000 | errno.ENOSYS)
    for index, yes, no in jumps:
        if program[index][0] == 0x05:
            program[index][3] = labels[yes] - index - 1
        else:
            program[index][1] = labels[yes] - index - 1
            program[index][2] = labels[no] - index - 1
    return [tuple(row) for row in program]


def _seccomp_truncate_guard():
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise AccountIsolationError("The validated ABI1 truncate guard requires Linux x86_64")

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    rows = _seccomp_program()
    filters = (Filter * len(rows))(*(Filter(*row) for row in rows))
    program = Program(len(rows), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        raise AccountIsolationError("Cannot establish the ABI1 truncate syscall guard: " + str(ctypes.get_errno()))
    return SECCOMP_GUARD


def _credentials():
    libc = ctypes.CDLL(None, use_errno=True)

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

    header, values = Header(0x20080522, 0), (Data * 2)()
    if libc.capget(ctypes.byref(header), values) != 0:
        raise AccountIsolationError("Cannot inspect actual native capabilities")
    effective_zero = all(not (x.effective or x.permitted or x.inheritable) for x in values)
    bounding_zero = True
    for capability in range(64):
        result = libc.prctl(23, capability, 0, 0, 0)
        if result > 0:
            bounding_zero = False
        elif result < 0 and ctypes.get_errno() != errno.EINVAL:
            raise AccountIsolationError("Cannot inspect native capability bounding set")
    return {
        "uid": os.getuid(),
        "euid": os.geteuid(),
        "gid": os.getgid(),
        "groups": os.getgroups(),
        "resuid": list(os.getresuid()),
        "resgid": list(os.getresgid()),
        "no_new_privs": libc.prctl(39, 0, 0, 0, 0),
        "capabilities_zero": effective_zero and bounding_zero,
    }


def launch_arguments(account, argv, *, home, cwd, catalogs=(), ready_fd, expected_tmp=None):
    """Root plans exact paths; the child attests the kernel boundary before exec."""
    cwd, home = Path(cwd), Path(home)
    temporary = cwd / ".native-tmp"
    if not temporary.exists() and not temporary.is_symlink():
        temporary.mkdir(mode=0o700)
        os.chown(temporary, account.pw_uid, account.pw_gid)
    info = temporary.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or temporary.is_symlink()
        or info.st_uid != account.pw_uid
        or info.st_gid != account.pw_gid
        or stat.S_IMODE(info.st_mode) != 0o700
        or (expected_tmp is not None and (info.st_dev, info.st_ino) != tuple(expected_tmp))
    ):
        raise AccountIsolationError("The same-attempt private temporary directory changed")
    system = [
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc/ssl",
        "/etc/ld.so.cache",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/etc/localtime",
        "/dev/random",
        "/dev/urandom",
        "/opt/alfworld-tools/alfworld-step",
    ]
    for variable in ("CODEX_NATIVE_BIN", "ALFWORLD_CODEX_PACKAGE_ROOT"):
        if os.environ.get(variable):
            system.append(os.environ[variable])
    instructions = os.environ.get("ALFWORLD_SOLVER_INSTRUCTIONS_FILE")
    if instructions:
        system += [instructions, str(Path(instructions).with_name("codex-entrypoint"))]
    read = [str(Path(x).resolve()) for x in [*system, *map(str, catalogs)] if Path(x).exists()]
    write = [str(home), str(cwd), "/dev/null", "/dev/zero", "/dev/pts"]
    child = [
        "/usr/bin/setpriv",
        "--reuid=" + str(account.pw_uid),
        "--regid=" + str(account.pw_gid),
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "/usr/bin/python3",
        str(Path(__file__).resolve()),
        "isolate-exec",
        "--uid",
        str(account.pw_uid),
        "--gid",
        str(account.pw_gid),
        "--ready-fd",
        str(ready_fd),
        "--read-roots",
        json.dumps(read),
        "--write-roots",
        json.dumps(write),
        "--",
        *argv,
    ]
    return (
        child,
        {
            "HOME": "/nonexistent",
            "USER": account.pw_name,
            "LOGNAME": account.pw_name,
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
        },
        (info.st_dev, info.st_ino),
    )


def decode_launcher_proof(data, *, uid, gid):
    value = json.loads(data)
    if (
        value.get("policy_version") != POLICY_VERSION
        or value.get("uid") != uid
        or value.get("euid") != uid
        or value.get("gid") != gid
        or value.get("groups") != []
        or value.get("resuid") != [uid] * 3
        or value.get("resgid") != [gid] * 3
        or value.get("no_new_privs") != 1
        or value.get("capabilities_zero") is not True
        or value.get("landlock_applied") is not True
        or value.get("landlock_abi", 0) < 1
        or value.get("seccomp_guard") != SECCOMP_GUARD
    ):
        raise AccountIsolationError("Native launcher readiness attestation is absent or invalid")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("provision-plan")
    commands.add_parser("initialize")
    commands.add_parser("check")
    child = commands.add_parser("isolate-exec")
    for name in ("uid", "gid", "ready-fd"):
        child.add_argument("--" + name, type=int, required=True)
    child.add_argument("--read-roots", required=True)
    child.add_argument("--write-roots", required=True)
    child.add_argument("program", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "provision-plan":
        print(json.dumps({"commands": provisioning_plan(), "automatic_execution": False}, indent=2))
    elif args.command == "initialize":
        AccountPool().initialize()
        print(json.dumps({"status": "initialized", "available": AccountPool().available()}))
    elif args.command == "check":
        print(
            json.dumps(
                {
                    "policy_version": POLICY_VERSION,
                    "landlock_abi": landlock_abi(),
                    "available": AccountPool().available(),
                    "pool_size": POOL_SIZE,
                }
            )
        )
    else:
        proof = _credentials()
        if (
            proof["uid"] != args.uid
            or proof["euid"] != args.uid
            or proof["gid"] != args.gid
            or proof["resuid"] != [args.uid] * 3
            or proof["resgid"] != [args.gid] * 3
            or proof["groups"]
            or proof["no_new_privs"] != 1
            or not proof["capabilities_zero"]
        ):
            raise AccountIsolationError("setpriv did not drop identity/groups/capabilities as required")
        proof.update(
            policy_version=POLICY_VERSION,
            landlock_abi=_landlock(json.loads(args.read_roots), json.loads(args.write_roots)),
            landlock_applied=True,
            seccomp_guard=_seccomp_truncate_guard(),
        )
        os.write(args.ready_fd, _canonical(proof) + b"\n")
        os.close(args.ready_fd)  # Generated code never inherits the readiness channel.
        program = args.program[1:] if args.program[:1] == ["--"] else args.program
        if not program:
            raise AccountIsolationError("Native launcher has no program")
        os.execvpe(program[0], program, os.environ)


if __name__ == "__main__":
    main()
