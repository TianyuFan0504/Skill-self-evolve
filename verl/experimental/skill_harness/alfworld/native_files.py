"""Archive explicitly owned native-Codex files before releasing local storage."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
from pathlib import Path


def verify_native_archive(archive, contents, manifest):
    """Verify every member in physical order without seeking through gzip.

    Hardlinks may reference only a previously verified regular file or hardlink.
    No source path is opened, changed or removed by this verification helper.
    """
    seen, verified_files = set(), {}
    manifest_name = "native-source-manifest.json"
    with gzip.open(archive, "rb") as decoded:
        with tarfile.open(archive, "r|", fileobj=decoded) as check:
            for entry in check:
                name = entry.name
                if name in seen or (name not in contents and name != manifest_name):
                    raise ValueError("Archive entry inventory differs from the original native files")
                seen.add(name)
                if name == manifest_name:
                    if not entry.isfile() or entry.size != len(manifest):
                        raise ValueError("Archive source manifest differs from the original native files")
                    with check.extractfile(entry) as source:
                        if source.read() != manifest:
                            raise ValueError("Archive source manifest differs from the original native files")
                    continue
                expected = contents[name]
                if expected["kind"] == "file":
                    if entry.isfile():
                        hasher, size = hashlib.sha256(), 0
                        with check.extractfile(entry) as source:
                            while chunk := source.read(1 << 20):
                                hasher.update(chunk)
                                size += len(chunk)
                        actual = {"size": size, "sha256": hasher.hexdigest()}
                    elif entry.islnk() and entry.size == 0 and entry.linkname in verified_files:
                        actual = verified_files[entry.linkname]
                    else:
                        raise ValueError("Archive file or prior hardlink target differs; preserving sources")
                    valid = actual["size"] == expected["size"] and actual["sha256"] == expected["sha256"]
                    if valid:
                        verified_files[name] = actual
                elif expected["kind"] == "symlink":
                    valid = entry.issym() and entry.linkname == expected["target"]
                else:
                    valid = expected["kind"] == "directory" and entry.isdir()
                if not valid:
                    raise ValueError("Archive content differs from the original native files; preserving sources")
        # Consume through gzip EOF so its CRC/trailer is checked too. Tar itself
        # stops at its zero blocks and may leave padding buffered in the stream.
        while decoded.read(1 << 20):
            pass
    if seen != set(contents) | {manifest_name}:
        raise ValueError("Archive entry inventory differs from the original native files")
    return {"verified_entries": len(seen), "verified_files": len(verified_files)}


def archive_native_files(
    output,
    *,
    codex_homes=(),
    model_catalogs=(),
    workspace=None,
    agent_uid,
    workspace_root=Path("/tmp/alfworld-skill-episodes"),
    curator_workdir=None,
    curator_root=None,
    curator_collection_id=None,
    paired_group=None,
    account_lease_id=None,
):
    """Keep raw files in one private durable archive; delete only after fsync.

    The caller must first stop/drain its native process and backend. Paths come
    from trusted Sandbox command/file hooks, never generated model text. A
    failed archive leaves every local source intact for host reconciliation.
    This function does not inspect other episodes or glob the shared /tmp.
    """
    output = Path(output)
    host_uid, host_gid = os.geteuid(), os.getegid()
    rows = []
    for kind, paths in (("codex_home", codex_homes), ("model_catalog", model_catalogs)):
        for value in dict.fromkeys(map(str, paths)):
            path = Path(value)
            expression = (
                r"codex-home-[0-9a-f]{32}" if kind == "codex_home" else r"codex-model-catalog-[0-9a-f]{32}\.json"
            )
            if path.parent != Path("/tmp") or not re.fullmatch(expression, path.name):
                raise ValueError("Native temporary path escaped the recorded Codex UUID namespace")
            rows.append({"kind": kind, "path": path})
    if workspace is not None:
        path = Path(workspace)
        paired = (
            isinstance(paired_group, str)
            and re.fullmatch(r"[0-9a-f]{64}", paired_group)
            and path.name == "credit_" + paired_group
        )
        if path.parent != Path(workspace_root) or not (re.fullmatch(r"[0-9a-f]{32}", path.name) or paired):
            raise ValueError("Normal workspace escaped its host-owned episode parent")
        rows.append({"kind": "workspace", "path": path})
    if curator_workdir is not None:
        path, root = Path(curator_workdir), Path(curator_root)
        if (
            not isinstance(curator_collection_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", curator_collection_id)
            or not root.is_absolute()
            or root.is_symlink()
            or root.resolve() == Path("/tmp").resolve()
            or not root.resolve().is_relative_to(Path("/tmp").resolve())
            or path != root / curator_collection_id
        ):
            raise ValueError("Curator cleanup requires the trusted runtime root and exact collection ID")
        rows.append({"kind": "curator_workspace", "path": path})
    if not rows:
        return {"schema_version": 1, "status": "no_native_files_created", "sources": []}
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate native cleanup ownership")
    contents, source_stats = {}, {}

    def signature(path):
        value = path.lstat()
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    for index, row in enumerate(rows):
        path = row["path"]
        identity = path.lstat()  # Missing files are evidence gaps, never silently successful cleanup.
        expected = stat.S_ISREG(identity.st_mode) if row["kind"] == "model_catalog" else stat.S_ISDIR(identity.st_mode)
        if path.is_symlink() or not expected or identity.st_uid not in (host_uid, agent_uid):
            raise ValueError("Recorded native file has an unexpected owner, type or symlink identity")
        row.update(
            device=identity.st_dev,
            inode=identity.st_ino,
            original_uid=identity.st_uid,
            original_gid=identity.st_gid,
            original_mode=stat.S_IMODE(identity.st_mode),
            removed=False,
        )
        # The native process is finished. Quarantine its root before taking
        # the stability snapshot; these deliberate permission changes are not
        # source-content changes during archival.
        os.chown(path, host_uid, host_gid)
        os.chmod(path, 0o600 if row["kind"] == "model_catalog" else 0o700)
        paths = [path, *path.rglob("*")] if path.is_dir() else [path]
        prefix = f"{row['kind']}-{index}"
        for item in paths:
            name = prefix + ("/" + item.relative_to(path).as_posix() if item != path else "")
            before = signature(item)
            mode = before[2]
            if stat.S_ISREG(mode):
                hasher = hashlib.sha256()
                with item.open("rb") as source:
                    while chunk := source.read(1 << 20):
                        hasher.update(chunk)
                contents[name] = {"kind": "file", "size": before[5], "sha256": hasher.hexdigest()}
            elif stat.S_ISDIR(mode):
                contents[name] = {"kind": "directory"}
            elif stat.S_ISLNK(mode):
                contents[name] = {"kind": "symlink", "target": os.readlink(item)}
            else:
                raise ValueError("A native resource contains an unsupported live special file; preserving sources")
            if signature(item) != before:
                raise ValueError("Native source changed while hashing; preserving every source")
            source_stats[item] = before
    archive = output / "native-files.tar.gz"
    # Raw streams/configs/captures are outside the HOME archive and must also
    # reach durable NAS storage before releasing local state.
    for path in output.iterdir():
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    manifest = json.dumps(contents, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = output / "native-files-manifest.json"
    with manifest_path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(manifest)
        stream.flush()
        os.fsync(stream.fileno())
    with tarfile.open(archive, "x:gz", compresslevel=1, dereference=False) as target:
        archive.chmod(0o600)
        for index, row in enumerate(rows):
            target.add(row["path"], arcname=f"{row['kind']}-{index}", recursive=True)
        entry = tarfile.TarInfo("native-source-manifest.json")
        entry.size, entry.mode = len(manifest), 0o600
        target.addfile(entry, io.BytesIO(manifest))
    verify_native_archive(archive, contents, manifest)

    def verify_source_stability():
        if any(signature(path) != expected for path, expected in source_stats.items()):
            raise ValueError("Native source changed during archival; preserving every source")

    verify_source_stability()
    hasher = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            hasher.update(chunk)
        os.fsync(stream.fileno())
    receipt = {
        "schema_version": 1,
        "status": "archived_before_removal",
        "archive_path": str(archive),
        "archive_sha256": hasher.hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "source_manifest_path": str(manifest_path),
        "source_entry_count": len(contents),
        "source_stats_stable": True,
        "sources": [{**row, "path": str(row["path"])} for row in rows],
        "agent_uid": agent_uid,
        "account_lease_id": account_lease_id,
    }
    receipt_path = output / "native-files.json"

    def save_receipt():
        stage = receipt_path.with_suffix(".json.tmp")
        with stage.open("w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stage, receipt_path)
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    save_receipt()  # Publish durable archive provenance before any destructive step.
    verify_source_stability()  # Check all sources before deleting the first one.
    for index, row in enumerate(rows):
        identity = row["path"].lstat()
        if (identity.st_dev, identity.st_ino) != (row["device"], row["inode"]) or row["path"].is_symlink():
            raise ValueError("Recorded native file identity changed after archival; refusing removal")
        if row["kind"] == "model_catalog":
            row["path"].unlink()
        else:
            shutil.rmtree(row["path"])
        receipt["sources"][index]["removed"] = True
        save_receipt()
    receipt["status"] = "archived_and_removed"
    save_receipt()
    return receipt
