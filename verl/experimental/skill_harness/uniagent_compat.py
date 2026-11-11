"""Explicit offline installer/provenance for one narrowly pinned Uni-Agent fix.

Run this file by path to avoid importing verl/Torch/Ray. Verification never edits
the external dependency. Only --apply writes, after the owner has stopped all
workers using that checkout; existing interpreters must not be reused afterward.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import stat
import sys
import tempfile
from pathlib import Path

PATCH_ID = "responses-effective-no-tools-v1"
ADAPTER_RELATIVE_PATH = "uni_agent/gateway/adapters/openai_response.py"
PRISTINE_SHA256 = "a71653b73a19ff4e501871f6eaf458a06bd00f7cc00070093c6997287635ee1f"
PATCHED_SHA256 = "573a7c74221189e377d2b2db631ac3b2fc1802453e642284173fc85ad42138fa"
EDITS = (
    (
        b"    if parallel_tool_calls is False:\n"
        b'        raise MalformedRequestError("parallel_tool_calls=false is not supported")\n\n',
        b"",
    ),
    (
        b"    converted_tools, _, _ = _convert_tools(_collect_tool_defs(payload), drop_deferred=True)\n"
        b"    chat_payload: dict[str, Any] = {\n",
        b"    converted_tools, _, _ = _convert_tools(_collect_tool_defs(payload), drop_deferred=True)\n"
        b'    effective_tools = None if tool_choice == "none" else converted_tools or None\n'
        b"    if parallel_tool_calls is False and effective_tools:\n"
        b"        raise MalformedRequestError(\n"
        b'            "parallel_tool_calls=false is not supported when effective tools are enabled"\n'
        b"        )\n"
        b"    chat_payload: dict[str, Any] = {\n",
    ),
    (
        b'        "tools": None if tool_choice == "none" else converted_tools or None,\n',
        b'        "tools": effective_tools,\n',
    ),
)


class UniAgentCompatibilityError(RuntimeError):
    """The external adapter is missing, unknown, or not explicitly patched."""


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def actual_source_root():
    """Prefer the already imported dependency, not a desired environment path."""
    module = sys.modules.get("uni_agent")
    if module is not None and getattr(module, "__file__", None):
        return Path(module.__file__).resolve().parent.parent
    return Path(
        os.environ.get(
            "UNI_AGENT_SOURCE_ROOT",
            "/root/nas_chaohuang/benchmarks/alfworld/uni-agent",
        )
    ).resolve()


def adapter_provenance(source_root=None):
    root = Path(source_root).resolve() if source_root is not None else actual_source_root()
    adapter = root / ADAPTER_RELATIVE_PATH
    digest = _sha256(adapter.read_bytes()) if adapter.is_file() else None
    status = (
        "verified_patched"
        if digest == PATCHED_SHA256
        else "known_pristine"
        if digest == PRISTINE_SHA256
        else "missing"
        if digest is None
        else "unknown"
    )
    return {
        "patch_id": PATCH_ID,
        "source_root": str(root),
        "adapter_path": str(adapter),
        "adapter_sha256": digest,
        "status": status,
        "expected_pristine_sha256": PRISTINE_SHA256,
        "expected_patched_sha256": PATCHED_SHA256,
    }


def offline_apply_command(source_root):
    return shlex.join(
        [sys.executable, str(Path(__file__).resolve()), "--uni-agent-root", str(Path(source_root).resolve()), "--apply"]
    )


def require_patched_adapter(source_root=None):
    provenance = adapter_provenance(source_root)
    if provenance["status"] != "verified_patched":
        raise UniAgentCompatibilityError(
            "Uni-Agent Responses compatibility verification failed: "
            + provenance["status"]
            + " (actual SHA256="
            + str(provenance["adapter_sha256"])
            + "). "
            "Stop/reconcile all workers using this checkout before the explicit offline command: "
            + offline_apply_command(provenance["source_root"])
            + ". Unknown source hashes are never patched. No automatic dependency edit was performed."
        )
    return provenance


def actual_adapter_sha256():
    """Checkpoint identity is always measured, even for non-task CPU imports."""
    value = adapter_provenance()
    if value["status"] not in {"known_pristine", "verified_patched"}:
        raise UniAgentCompatibilityError("Cannot bind a checkpoint to a missing or unknown Uni-Agent adapter")
    return value["adapter_sha256"]


def _patched_bytes(original):
    value = original
    for before, after in EDITS:
        if value.count(before) != 1:
            raise UniAgentCompatibilityError("Pinned patch context is absent or ambiguous")
        value = value.replace(before, after, 1)
    if _sha256(value) != PATCHED_SHA256:
        raise UniAgentCompatibilityError("Pinned replacement does not produce the approved adapter hash")
    return value


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def apply_patch(source_root):
    """Explicitly patch only the exact known pristine file, with durable backup."""
    import fcntl

    root = Path(source_root).resolve()
    adapter = root / ADAPTER_RELATIVE_PATH
    if adapter.is_symlink() or not adapter.is_file():
        raise UniAgentCompatibilityError("The adapter must be an existing regular non-symlink source file")
    descriptor = os.open(adapter.parent / ".skill-harness-compat.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        original, information = adapter.read_bytes(), adapter.stat()
        digest = _sha256(original)
        backup = adapter.with_name(adapter.name + ".before-" + PATCH_ID + ".bak")
        if digest == PATCHED_SHA256:
            return {
                **require_patched_adapter(root),
                "action": "already_applied",
                "backup": str(backup) if backup.exists() else None,
                "restart_workers_required": True,
            }
        if digest != PRISTINE_SHA256:
            raise UniAgentCompatibilityError("Unknown adapter SHA256; refusing to alter " + str(adapter))
        updated = _patched_bytes(original)
        if backup.exists():
            if backup.is_symlink() or _sha256(backup.read_bytes()) != PRISTINE_SHA256:
                raise UniAgentCompatibilityError("An existing compatibility backup has unknown contents")
        else:
            backup_descriptor = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(backup_descriptor, "wb") as output:
                output.write(original)
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(adapter.parent)
        temporary_descriptor, temporary = tempfile.mkstemp(prefix=".adapter-compat-", dir=adapter.parent)
        try:
            with os.fdopen(temporary_descriptor, "wb") as output:
                os.fchmod(output.fileno(), stat.S_IMODE(information.st_mode))
                actual = os.fstat(output.fileno())
                if (actual.st_uid, actual.st_gid) != (information.st_uid, information.st_gid):
                    os.fchown(output.fileno(), information.st_uid, information.st_gid)
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            if adapter.is_symlink() or adapter.read_bytes() != original:
                raise UniAgentCompatibilityError("Adapter changed concurrently before atomic publication")
            os.replace(temporary, adapter)
            _fsync_directory(adapter.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
        return {
            **require_patched_adapter(root),
            "action": "applied",
            "backup": str(backup),
            "restart_workers_required": True,
        }
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uni-agent-root", type=Path, default=actual_source_root())
    parser.add_argument(
        "--apply", action="store_true", help="Explicitly edit the stopped checkout; never use on live workers"
    )
    args = parser.parse_args(argv)
    try:
        result = apply_patch(args.uni_agent_root) if args.apply else require_patched_adapter(args.uni_agent_root)
    except UniAgentCompatibilityError as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
