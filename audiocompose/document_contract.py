"""Dependency-free reader for the canonical SSMDDoc v1 workspace contract."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

DOCUMENT_ID = "document-0001"
DOCUMENT_SOURCE_FILENAME = "document.ssmd.md"
DOCUMENT_UTTERPLAN_FILENAME = "document.utterplan.toml"
DOCUMENT_RENDERPLAN_FILENAME = "document.renderplan.toml"
PROFILE_SNAPSHOT_FILENAME = "renderprofile.toml"
RENDER_MANIFEST_FILENAME = "render-manifest.toml"
COMPOSE_REPORT_FILENAME = "compose-report.toml"


@dataclass(frozen=True, slots=True)
class DocumentWorkspaceIndex:
    """Validated identity and canonical document paths for one SSMDDoc workspace."""

    root: Path
    manifest_path: Path
    document_id: str
    document_path: Path


class DocumentWorkspaceContractError(ValueError):
    """An input does not satisfy the SSMDDoc v1 filesystem contract."""


def _fail(label: str, message: str) -> DocumentWorkspaceContractError:
    return DocumentWorkspaceContractError(f"{label}: {message}")


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    current = Path(expanded.anchor) if expanded.is_absolute() else Path.cwd()
    for part in expanded.parts:
        if part in {expanded.anchor, "", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _fail(label, f"path is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise _fail(label, f"path contains a symlink: {current}")
    return Path(os.path.abspath(current))


def _existing_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute_without_symlinks(path, label=label)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise _fail(label, f"directory is unavailable: {absolute}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _fail(label, f"not a directory: {absolute}")
    return absolute.resolve(strict=True)


def _contained_regular_file(root: Path, path: Path, *, label: str) -> Path:
    absolute = _absolute_without_symlinks(path, label=label)
    try:
        resolved = absolute.resolve(strict=True)
        info = resolved.lstat()
    except OSError as exc:
        raise _fail(label, f"file is unavailable: {absolute}: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise _fail(label, f"file escapes its workspace: {absolute}")
    if not stat.S_ISREG(info.st_mode):
        raise _fail(label, f"not a regular file: {absolute}")
    return resolved


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise _fail("workspace.invalid", f"cannot read SSMDDoc manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise _fail("workspace.invalid", "manifest root must be a JSON object")
    if (
        manifest.get("format") != "ssmdconvert.document"
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
    ):
        raise _fail("workspace.invalid", "manifest is not a supported SSMDDoc v1 document")
    # The current producer contract requires SSMD 0.9; tolerate other producer
    # metadata, but do not silently accept a different document format version.
    if manifest.get("ssmd_version") != "0.9":
        raise _fail("workspace.invalid", "manifest ssmd_version must be '0.9'")
    document = manifest.get("document")
    if not isinstance(document, dict):
        raise _fail("workspace.invalid", "manifest document must be an object")
    document_id = document.get("id")
    document_path = document.get("path")
    if document_id != DOCUMENT_ID:
        raise _fail("workspace.invalid", f"manifest document id must be {DOCUMENT_ID!r}")
    if document_path != DOCUMENT_SOURCE_FILENAME:
        raise _fail(
            "workspace.invalid",
            f"manifest document path must be {DOCUMENT_SOURCE_FILENAME!r}",
        )
    if (
        not isinstance(document_path, str)
        or "\\" in document_path
        or "\x00" in document_path
        or PurePosixPath(document_path).is_absolute()
        or PureWindowsPath(document_path).drive
        or any(part in {"", ".", ".."} for part in document_path.split("/"))
    ):
        raise _fail("workspace.invalid", "manifest document path must be a safe relative path")
    return manifest


def read_document_workspace(document: str | Path) -> DocumentWorkspaceIndex:
    """Read and validate a workspace root or its canonical document file.

    ZIP bundles are intentionally unsupported; AudioCompose consumes the
    directory workspace produced by the current SSMDDoc contract.
    """
    supplied = Path(document).expanduser()
    if supplied.suffix.lower() == ".zip":
        raise _fail("workspace.invalid", "ZIP input is not supported; provide an SSMDDoc directory")
    if supplied.name == DOCUMENT_SOURCE_FILENAME:
        root_input = supplied.parent
        document_input = supplied
    else:
        root_input = supplied
        document_input = None

    root = _existing_directory(root_input, label="SSMDDoc workspace")
    manifest_path = _contained_regular_file(
        root,
        root / "manifest.json",
        label="SSMDDoc manifest",
    )
    manifest = _load_manifest(manifest_path)
    canonical_path = _contained_regular_file(
        root,
        root / manifest["document"]["path"],
        label="canonical SSMDDoc document",
    )
    if document_input is not None:
        supplied_document = _contained_regular_file(
            root,
            document_input,
            label="canonical SSMDDoc input",
        )
        if supplied_document != canonical_path:
            raise _fail("workspace.invalid", "input is not the canonical SSMDDoc document")
    return DocumentWorkspaceIndex(
        root=root,
        manifest_path=manifest_path,
        document_id=manifest["document"]["id"],
        document_path=canonical_path,
    )


__all__ = [
    "COMPOSE_REPORT_FILENAME",
    "DOCUMENT_ID",
    "DOCUMENT_RENDERPLAN_FILENAME",
    "DOCUMENT_SOURCE_FILENAME",
    "DOCUMENT_UTTERPLAN_FILENAME",
    "DocumentWorkspaceContractError",
    "DocumentWorkspaceIndex",
    "PROFILE_SNAPSHOT_FILENAME",
    "RENDER_MANIFEST_FILENAME",
    "read_document_workspace",
]
