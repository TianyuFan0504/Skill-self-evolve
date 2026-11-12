"""Explicit, process-local selection of a separately provisioned UID pool.

Import has no side effects. Only the explicit ``initialize`` CLI initializes a
new registry; user/group creation is always left to the operator. The frozen
AccountPool implementation, lease fencing and native launcher are unchanged.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import importlib
import json
import os
import pwd
import re
import stat
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

POLICY = "operator-provisioned-account-pool-epoch-v1"
WORKER_SETUP_HOOK = "skill_harness_examples.account_pool_epoch.setup_worker"
MANIFEST_ENV = "ALFWORLD_ACCOUNT_POOL_MANIFEST"
MANIFEST_SHA_ENV = "ALFWORLD_ACCOUNT_POOL_MANIFEST_SHA256"
EPOCH_POOL_ROOT = Path("/var/lib/alfworld-skill-harness/account_pool_epoch3")
OLD_POOL_ROOT = Path("/var/lib/alfworld-skill-harness/account_pool_v2")
ISOLATION_POLICY = "alfworld-uid-landlock-v1"
CAPACITY = 128
MINIMUM_AVAILABLE = 40  # 32 primary sessions plus eight paired probe UIDs.
BINDING_MODULES = [
    "verl.experimental.skill_harness.alfworld.task.AccountPool",
    "verl.experimental.skill_harness.probes.AccountPool",
    "verl.experimental.skill_harness.group_sampler.local_archive_failed_groups",
]
_active = None
_original_archive = None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def roster():
    return [{"name": f"alfworld_ep3_{index:03d}", "uid": 30000 + index, "gid": 30000 + index}
            for index in range(CAPACITY)]


def fenced_pool():
    return {"root": str(OLD_POOL_ROOT), "uid_min": 20000, "uid_max": 20127,
            "protected_uid": 20071, "operation": "preserve-without-reset-release-remap"}


def provider_ref():
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _publish(path, value):
    """Publish private immutable operational evidence with no curator imports."""
    path = Path(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = path.parent.lstat()
    if (path.parent != path.parent.resolve() or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022):
        raise ValueError("Account epoch evidence needs a host-owned directory without public writes")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pending = path.with_name(".pending-" + uuid.uuid4().hex)
    try:
        with pending.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, path)
        except FileExistsError:
            _private(path)
            if path.read_bytes() != data:
                raise ValueError("An account epoch observation cannot be rewritten") from None
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        pending.unlink(missing_ok=True)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def process_identity():
    text = (Path("/proc") / str(os.getpid()) / "stat").read_text()
    return {"pid": os.getpid(), "start_ticks": int(text[text.rfind(")") + 2:].split()[19])}


def _accounts():
    return importlib.import_module("verl.experimental.skill_harness.alfworld.accounts")


def _validate_primary_groups(users):
    selected = {item["gid"]: item for item in roster()}
    for user in users:
        item = selected.get(user.pw_gid)
        if item is not None and (user.pw_name, user.pw_uid) != (item["name"], item["uid"]):
            raise ValueError("An unrelated existing user shares a proposed fresh pool primary GID")


def _private(path, *, directory=False):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("Pool epoch paths must be absolute and contain no symlinks")
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("Pool epoch files/directories must be private and host-owned")
    return path


def read_manifest(*, environ=None):
    """Validate the operator's immutable file/hash; never create or initialize."""
    env = os.environ if environ is None else environ
    path = _private(Path(env[MANIFEST_ENV]))
    _private(path.parent, directory=True)
    expected_hash = env[MANIFEST_SHA_ENV]
    data = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash) or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("The operator pool manifest SHA256 differs")
    value = json.loads(data)
    expected = {
        "schema_version": 1, "kind": "operator_provisioned_account_pool_epoch", "policy": POLICY,
        "epoch": 3, "pool_root": str(EPOCH_POOL_ROOT), "roster": roster(),
        "roster_digest": digest(roster()), "isolation_policy": ISOLATION_POLICY,
        "fenced_pool": fenced_pool(),
    }
    if (set(value) != set(expected) | {"created_at_ns"}
            or any(value.get(key) != item for key, item in expected.items())
            or type(value["created_at_ns"]) is not int or value["created_at_ns"] <= 0):
        raise ValueError("The pool epoch manifest is not the exact nonoverlapping epoch3 policy")
    return value, {"path": str(path), "sha256": expected_hash}


def readonly_preflight(*, environ=None, minimum_available=MINIMUM_AVAILABLE):
    """Read roster/process/registry state without a lease, lock write or reset.

    This is a point-in-time capacity check. Frozen AccountPool.acquire performs
    the authoritative locked process/slot/quarantine checks for every lease.
    """
    if type(minimum_available) is not int or not 0 <= minimum_available <= CAPACITY:
        raise ValueError("Invalid pool epoch capacity threshold")
    value, ref = read_manifest(environ=environ)
    accounts = _accounts()
    accounts.validate_system_roster(value["roster"])
    _validate_primary_groups(pwd.getpwall())
    root = _private(Path(value["pool_root"]), directory=True)
    _private(root / "leases", directory=True)
    state_path = _private(root / "state.json")
    state = json.loads(state_path.read_bytes())
    slots = state.get("slots", {})
    if (state.get("schema_version") != 1 or state.get("policy_version") != ISOLATION_POLICY
            or state.get("roster") != value["roster"]
            or set(slots) != {str(item["uid"]) for item in value["roster"]}
            or not isinstance(state.get("owners"), dict)):
        raise ValueError("The preprovisioned epoch registry policy/roster/slots differ")
    available, leased, quarantined, live = [], [], [], []
    for item in value["roster"]:
        uid = item["uid"]
        quarantine = root / "quarantined-uids" / str(uid)
        if quarantine.exists() or quarantine.is_symlink():
            quarantined.append(uid)
        elif slots[str(uid)] is not None:
            leased.append(uid)
        elif accounts.uid_processes(uid):
            live.append(uid)
        else:
            available.append(uid)
    if len(available) < minimum_available:
        raise ValueError(f"Fresh epoch has {len(available)} available UIDs; requires {minimum_available}")
    return {"manifest_ref": ref, "pool_root": str(root), "roster_digest": value["roster_digest"],
            "capacity": CAPACITY, "available": len(available), "leased": leased,
            "quarantined": quarantined, "live_without_lease": live,
            "minimum_available": minimum_available, "read_only": True}


def provisioning_plan():
    """Return argv for missing accounts after read-only name/UID/GID checks."""
    accounts = _accounts()
    users, groups = pwd.getpwall(), grp.getgrall()
    _validate_primary_groups(users)
    commands = []
    for item in roster():
        users_found = [x for x in users if x.pw_name == item["name"] or x.pw_uid == item["uid"]]
        groups_found = [x for x in groups if x.gr_name == item["name"] or x.gr_gid == item["gid"]]
        if users_found or groups_found:
            accounts.validate_system_roster([item])
            continue
        commands.extend([
            ["/usr/sbin/groupadd", "--gid", str(item["gid"]), item["name"]],
            ["/usr/sbin/useradd", "--uid", str(item["uid"]), "--gid", str(item["gid"]),
             "--no-create-home", "--no-log-init", "--home-dir", "/nonexistent", "--shell",
             "/usr/sbin/nologin", "--comment", accounts.ACCOUNT_COMMENT, "--password", "!", item["name"]],
        ])
    return {"policy": POLICY, "pool_root": str(EPOCH_POOL_ROOT), "roster": roster(),
            "fenced_pool": fenced_pool(), "commands": commands,
            "execution": "Operator executes reviewed argv; this API never executes commands"}


def initialize_epoch(manifest_path):
    """Explicit operator action; only initialize a completely absent epoch root.

    Never edits accounts or any old root. Refuse even a partial existing epoch:
    a retry needs explicit host review, rather than silent registry recreation.
    """
    accounts = _accounts()
    accounts.validate_system_roster(roster())
    _validate_primary_groups(pwd.getpwall())
    if any(accounts.uid_processes(item["uid"]) for item in roster()):
        raise ValueError("Fresh epoch identities already own live processes")
    if EPOCH_POOL_ROOT.exists() or EPOCH_POOL_ROOT.is_symlink():
        raise ValueError("Fresh epoch root already exists; initialization cannot reset it")
    path = Path(manifest_path)
    if not path.is_absolute() or path != path.resolve() or path.exists() or path.is_symlink():
        raise ValueError("A new absolute manifest path without symlinks is required")
    _private(path.parent, directory=True)
    _private(EPOCH_POOL_ROOT.parent, directory=True)
    # Exclusive mkdir closes the absent-root race between independent operator
    # initializers. A partial root is retained for review if later work fails.
    EPOCH_POOL_ROOT.mkdir(mode=0o700)
    accounts.AccountPool(EPOCH_POOL_ROOT, roster=roster()).initialize()
    value = {"schema_version": 1, "kind": "operator_provisioned_account_pool_epoch", "policy": POLICY,
             "epoch": 3, "pool_root": str(EPOCH_POOL_ROOT), "roster": roster(),
             "roster_digest": digest(roster()), "isolation_policy": ISOLATION_POLICY,
             "fenced_pool": fenced_pool(), "created_at_ns": time.time_ns()}
    return _publish(path, value)


def identity():
    value, ref = read_manifest()
    return {"policy": POLICY, "provider_ref": provider_ref(), "manifest_ref": ref,
            "pool_root": value["pool_root"], "roster_digest": value["roster_digest"],
            "worker_setup_hook": WORKER_SETUP_HOOK, "expected_capacity": CAPACITY}


def activation_directory(run_root=None):
    root = Path(os.environ["SKILL_RUN_DIR"] if run_root is None else run_root)
    if not root.is_absolute() or root != root.resolve():
        raise ValueError("Pool epoch requires the canonical absolute trusted run root")
    return root / "runtime/account_pool_epoch/activations"


def _observe(role):
    value = identity()
    owner = process_identity()
    path = activation_directory() / f"{owner['pid']}-{owner['start_ticks']}.{role}.json"
    return _publish(path, {
        "schema_version": 1, "kind": "activated_account_pool_epoch", "policy": POLICY,
        "owner": owner, "activation_role": role, **{key: value[key] for key in (
            "provider_ref", "manifest_ref", "pool_root", "roster_digest", "worker_setup_hook")},
        "binding_modules": BINDING_MODULES, "preserved_core_account_policy": ISOLATION_POLICY,
        "native_spawn_account_pool": value,
    })


def _pool(role):
    if _active is None or _active != identity():
        raise ValueError("Pool epoch binding is absent or its operator manifest/provider changed")
    _observe(role)
    return _accounts().AccountPool(EPOCH_POOL_ROOT, roster=roster())


def task_account_pool():
    return _pool("task_account_pool")


class ProbeAccountPool:
    """Fresh live leases plus the frozen read-only retired-pair proof reader."""

    def __init__(self, fresh):
        if fresh.root != EPOCH_POOL_ROOT or fresh.roster != roster():
            raise ValueError("The probe façade requires the exact fresh live pool")
        self._fresh = fresh
        self.root, self.roster = fresh.root, fresh.roster

    @staticmethod
    def _epoch(public):
        accounts = _accounts()
        if isinstance(public, dict) and type(public.get("uid")) is int and type(public.get("gid")) is int:
            uid = public["uid"]
            for base, prefix, label in ((30000, "alfworld_ep3_", "fresh"), (20000, "alfworld_ep2_", "retired")):
                if (base <= uid < base + CAPACITY and public["gid"] == uid
                        and public.get("name") == prefix + f"{uid - base:03d}"):
                    return label
        raise accounts.AccountIsolationError("Probe account identity is outside the exact epoch rosters")

    def acquire(self, **kwargs):
        return self._fresh.acquire(**kwargs)

    def public(self, ref):
        self._require_fresh(ref)
        return self._fresh.public(ref)

    def _require_fresh(self, ref):
        if self._epoch(ref) != "fresh":
            raise _accounts().AccountIsolationError("A retired account is available only for read-only pair history")

    def finish(self, ref, *, abandoned=False):
        self._require_fresh(ref)
        return self._fresh.finish(ref, abandoned=abandoned)

    def historical_pair(self, public, *, run_root, group):
        if self._epoch(public) == "fresh":
            return self._fresh.historical_pair(public, run_root=run_root, group=group)
        accounts = _accounts()
        if not re.fullmatch(r"[0-9a-f]{32}", str(public.get("lease_id", ""))):
            raise accounts.AccountIsolationError("Invalid historical paired lease identity")
        _private(OLD_POOL_ROOT, directory=True)
        _private(OLD_POOL_ROOT / "leases", directory=True)
        _private(OLD_POOL_ROOT / "leases" / (public["lease_id"] + ".json"))
        # The frozen reader dereferences only root/leases and verifies this
        # exact released, complete, nonabandoned pair. No old live pool object
        # or old mutation API is exposed, and private lease tokens are not logged.
        return accounts.AccountPool.historical_pair(
            SimpleNamespace(root=OLD_POOL_ROOT), public, run_root=run_root, group=group,
        )


def probe_account_pool():
    return ProbeAccountPool(_pool("probe_account_pool"))


def preflight_account_pool():
    """Selected pool for the explicit native preflight; no Task-use claim."""
    return _pool("driver")


def archive_failed_groups(run_root):
    if _active is None or _active != identity() or _original_archive is None:
        raise ValueError("Pool epoch archive reader binding is absent or changed")
    # Historical settled data failures still exclude their groups; the new
    # epoch adds its own dispositions without changing unknown-artifact rules.
    return (_original_archive(run_root, pool_root=OLD_POOL_ROOT)
            | _original_archive(run_root, pool_root=EPOCH_POOL_ROOT))


def install_process(*, role):
    """Bind three owned-process import aliases to the public frozen constructor.

    No core constants, class methods, NativeCuratorRuntime identity, registry or
    global bootstrap functions change. Every process must run the official hook.
    """
    global _active, _original_archive
    if role not in ("driver", "ray_worker", "learner"):
        raise ValueError("Unknown pool epoch process role")
    selected = identity()
    if _active is not None and _active != selected:
        raise ValueError("A process cannot switch account pool epochs")
    # Validate the entire read-only host state before changing any alias. Worker
    # creation may occur after other workers lease accounts, so require no spare
    # minimum here; the driver performs the >=40 capacity gate before launch.
    readonly_preflight(minimum_available=0)
    accounts = _accounts()
    task = importlib.import_module("verl.experimental.skill_harness.alfworld.task")
    probes = importlib.import_module("verl.experimental.skill_harness.probes")
    sampler = importlib.import_module("verl.experimental.skill_harness.group_sampler")
    targets = ((task, "AccountPool", accounts.AccountPool, task_account_pool),
               (probes, "AccountPool", accounts.AccountPool, probe_account_pool),
               (sampler, "local_archive_failed_groups", accounts.local_archive_failed_groups, archive_failed_groups))
    for module, name, original, replacement in targets:
        if getattr(module, name) not in (original, replacement):
            raise ValueError("Another provider already changed the account pool import alias")
    _original_archive = accounts.local_archive_failed_groups
    for module, name, _, replacement in targets:
        setattr(module, name, replacement)
    _active = selected
    from skill_harness_examples import native_posix_spawn

    native_posix_spawn.register_account_pool_epoch(selected, roster())
    return _observe(role)


def setup_worker():
    """Official Ray no-argument setup hook, executed in each owned worker."""
    from verl.experimental.skill_harness.bootstrap import configure_runtime

    configure_runtime(register_tasks=True)
    install_process(role="ray_worker")


def prepare_driver(config):
    from verl.experimental.skill_harness.bootstrap import configure_runtime

    configure_runtime(register_tasks=True)
    run_root = str(Path(config.skill_harness.artifact_root).resolve())
    if os.environ.get("SKILL_RUN_DIR") != run_root:
        raise ValueError("Pool epoch run root differs from the trusted SKILL_RUN_DIR")
    readonly_preflight()
    return install_process(role="driver")


def prepare_runtime_environment(config):
    """Call AFTER the base bootstrap merge, preserving all other runtime fields."""
    from omegaconf import OmegaConf

    current = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env", default={}) or {}
    plain = OmegaConf.to_container(current, resolve=True) if OmegaConf.is_config(current) else dict(current)
    env = dict(plain.get("env_vars", {}))
    value = identity()
    env.update({MANIFEST_ENV: value["manifest_ref"]["path"], MANIFEST_SHA_ENV: value["manifest_ref"]["sha256"],
                "SKILL_RUN_DIR": str(Path(config.skill_harness.artifact_root).resolve())})
    plain.update(env_vars=env, worker_process_setup_hook=WORKER_SETUP_HOOK)
    OmegaConf.update(config, "ray_kwargs.ray_init.runtime_env", plain, merge=False, force_add=True)
    return value


def require_worker_activation(config):
    """Reject a lost/overwritten hook before the actual learner is constructed."""
    from omegaconf import OmegaConf

    value = identity()
    runtime = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env", default={}) or {}
    env = runtime.get("env_vars", {})
    if (_active != value or runtime.get("worker_process_setup_hook") != WORKER_SETUP_HOOK
            or env.get(MANIFEST_ENV) != value["manifest_ref"]["path"]
            or env.get(MANIFEST_SHA_ENV) != value["manifest_ref"]["sha256"]
            or env.get("SKILL_RUN_DIR") != str(Path(config.skill_harness.artifact_root).resolve())):
        raise ValueError("The actual learner is missing the selected account-pool worker setup hook")
    owner = process_identity()
    path = activation_directory() / f"{owner['pid']}-{owner['start_ticks']}.ray_worker.json"
    _private(path)
    observation = json.loads(path.read_bytes())
    if (observation.get("owner") != owner or observation.get("activation_role") != "ray_worker"
            or observation.get("native_spawn_account_pool") != value):
        raise ValueError("The actual learner has no matching completed Ray worker activation")
    return value


def inspect_worker_bindings():
    """Model-free Ray acceptance body: construct pools, never acquire a UID.

    Execute with the production runtime_env/setup hook after provisioning.
    The hook must already have run; this function never installs a missing hook.
    """
    selected = identity()
    if _active != selected:
        raise ValueError("Ray acceptance worker did not execute the pool epoch hook")
    task = importlib.import_module("verl.experimental.skill_harness.alfworld.task")
    probes = importlib.import_module("verl.experimental.skill_harness.probes")
    if task.AccountPool is not task_account_pool or probes.AccountPool is not probe_account_pool:
        raise ValueError("Ray acceptance worker lost its actual Task/probe pool bindings")
    primary, paired = task.AccountPool(), probes.AccountPool()
    if primary.root != paired.root or primary.roster != paired.roster:
        raise ValueError("Ray primary and paired probe account pools differ")
    return {"account_pool_epoch_extension": selected, "owner": process_identity(),
            "task_pool_root": str(primary.root), "probe_pool_root": str(paired.root),
            "task_pool_class": type(primary).__module__ + "." + type(primary).__qualname__,
            "probe_pool_class": type(paired).__module__ + "." + type(paired).__qualname__,
            "roster_digest": digest(primary.roster), "lease_acquired": False,
            "activation_refs": activation_refs()}


def activation_refs():
    """Only already-published observations; never predict future worker activation."""
    result = []
    for path in sorted(activation_directory().glob("*.json")):
        pattern = r"[0-9]+-[0-9]+\.(driver|ray_worker|learner|task_account_pool|probe_account_pool)\.json"
        if re.fullmatch(pattern, path.name):
            _private(path)
            result.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def publish_inference_expected(run_root, *, baseline_extension):
    """Record common-pool inference intent; this helper performs no curation."""
    return _publish(Path(run_root) / "runtime/account_pool_epoch.expected.json", {
        "schema_version": 1, "kind": "account_pool_epoch_inference_expected",
        "account_pool_epoch_extension": identity(), "activation_directory": str(activation_directory(run_root)),
        "baseline_extension": baseline_extension,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "initialize"))
    parser.add_argument("--manifest", help="New private operator manifest path; initialize only")
    args = parser.parse_args()
    if args.command == "plan":
        result = provisioning_plan()
    elif args.command == "preflight":
        result = readonly_preflight()
    else:
        if not args.manifest:
            parser.error("initialize requires --manifest")
        result = initialize_epoch(args.manifest)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
