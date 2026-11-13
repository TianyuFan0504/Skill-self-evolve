"""Owned native resource cleanup and optional diagnostic archives.

Callers must stop and confirm the UID/backend drain before calling cleanup.
Archival is a separate diagnostic operation and is never a score prerequisite.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import uuid
from pathlib import Path


def _resources(
    *,
    codex_homes=(),
    model_catalogs=(),
    workspace=None,
    workspace_root=None,
    curator_workdir=None,
    curator_root=None,
    curator_collection_id=None,
    paired_group=None,
):
    rows = []
    for kind, values in (("codex_home", codex_homes), ("model_catalog", model_catalogs)):
        expression = r"codex-home-[0-9a-f]{32}" if kind == "codex_home" else r"codex-model-catalog-[0-9a-f]{32}\.json"
        for value in dict.fromkeys(map(str, values)):
            path = Path(value)
            if path.parent != Path("/tmp") or not re.fullmatch(expression, path.name):
                raise ValueError("Native resource escaped its host-assigned temporary namespace")
            rows.append({"kind": kind, "path": path})
    if workspace is not None:
        path = Path(workspace)
        paired = (
            isinstance(paired_group, str)
            and re.fullmatch(r"[0-9a-f]{64}", paired_group)
            and path.name == "credit_" + paired_group
        )
        if (
            workspace_root is None
            or path.parent != Path(workspace_root)
            or not (re.fullmatch(r"[0-9a-f]{32}", path.name) or paired)
        ):
            raise ValueError("Workspace cleanup requires its explicit host-assigned parent")
        rows.append({"kind": "workspace", "path": path})
    if curator_workdir is not None:
        path, root = Path(curator_workdir), Path(curator_root)
        if (
            not isinstance(curator_collection_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", curator_collection_id)
            or not root.is_absolute()
            or root.is_symlink()
            or root == Path("/tmp")
            or path != root / curator_collection_id
        ):
            raise ValueError("Curator cleanup requires its exact assigned workspace")
        rows.append({"kind": "curator_workspace", "path": path})
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate native resource ownership")
    return rows


def _identities(rows, agent_uid):
    for row in rows:
        path = row["path"]
        info = path.lstat()
        expected = stat.S_ISREG(info.st_mode) if row["kind"] == "model_catalog" else stat.S_ISDIR(info.st_mode)
        if path.is_symlink() or not expected or info.st_uid not in (os.geteuid(), agent_uid):
            raise ValueError("Native resource has an unexpected owner, type or symlink identity")
        row.update(
            device=info.st_dev, inode=info.st_ino, original_uid=info.st_uid, original_gid=info.st_gid, removed=False
        )
    return rows


def _write_receipt(path, receipt):
    stage = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with stage.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def cleanup_native_files(output, *, agent_uid, account_lease_id=None, **resources):
    """Remove only recorded resources and return their ownership/closure receipt.

    Exact root identities are validated before removal. The process/backend owner
    must already be drained, so there can be no live agent writers. No archive,
    source-content hash or source-version inspection participates in cleanup.
    """
    rows = _identities(_resources(**resources), agent_uid)
    receipt = {
        "schema_version": 1,
        "status": "cleaning",
        "agent_uid": agent_uid,
        "account_lease_id": account_lease_id,
        "sources": [{**row, "path": str(row["path"])} for row in rows],
    }
    path = Path(output) / "native-cleanup.json"
    _write_receipt(path, receipt)
    for row, recorded in zip(rows, receipt["sources"], strict=True):
        info = row["path"].lstat()
        if (info.st_dev, info.st_ino, info.st_uid, info.st_gid) != (
            row["device"],
            row["inode"],
            row["original_uid"],
            row["original_gid"],
        ):
            raise ValueError("Native resource identity changed before cleanup")
        if row["kind"] == "model_catalog":
            row["path"].unlink()
        else:
            shutil.rmtree(row["path"])
        recorded["removed"] = True
        _write_receipt(path, receipt)
    receipt["status"] = "cleaned"
    _write_receipt(path, receipt)
    return receipt


def archive_native_files(output, *, agent_uid, account_lease_id=None, **resources):
    """Optionally copy native resources into a diagnostic tar without removing them.

    A caller may record an archive error and still perform normal resource
    cleanup. Contexts, token records, engine outcomes and native events live in
    the episode output and do not depend on this optional HOME/workspace copy.
    """
    rows = _identities(_resources(**resources), agent_uid)
    archive = Path(output) / "native-files.tar.gz"
    with tarfile.open(archive, "x:gz", dereference=False) as target:
        os.chmod(archive, 0o600)
        for index, row in enumerate(rows):
            target.add(row["path"], arcname=f"{row['kind']}-{index}", recursive=True)
    return {
        "schema_version": 1,
        "status": "archived",
        "archive_path": str(archive),
        "agent_uid": agent_uid,
        "account_lease_id": account_lease_id,
        "sources": [{**row, "path": str(row["path"])} for row in rows],
    }
