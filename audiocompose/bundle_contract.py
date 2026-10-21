"""Shared identity and input contract for SSMD bundle workspaces."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import AudioComposeError

BundleKind = Literal["book", "document"]


class BundleComposeError(AudioComposeError):
    """First-level SSMD bundle error with a stable CLI error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class BundleIdentity:
    root: Path
    kind: BundleKind
    manifest_path: Path


def load_bundle_identity(bundle: str | Path) -> BundleIdentity:
    """Safely load an SSMD bundle identity; only a real workspace root is valid."""
    path = Path(bundle).expanduser()
    if path.name.endswith(".ssmd.md") or path.name == "chapters":
        raise BundleComposeError(
            "bundle.invalid", "invalid SSMD bundle: input must be a .ssmd-bundle workspace root"
        )
    if not path.name.endswith(".ssmd-bundle"):
        raise BundleComposeError(
            "bundle.invalid", "invalid SSMD bundle: workspace root must end in .ssmd-bundle"
        )
    current = Path(path.anchor) if path.is_absolute() else Path.cwd()
    for part in path.parts:
        if part in {path.anchor, "", "."}:
            continue
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise BundleComposeError("bundle.invalid", f"invalid SSMD bundle: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise BundleComposeError(
                "bundle.invalid", f"invalid SSMD bundle: symlink in workspace path: {current}"
            )
    root = Path(os.path.abspath(path))
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise BundleComposeError(
            "bundle.invalid", "invalid SSMD bundle: workspace root is not a directory"
        )
    manifest_path = root / "manifest.json"
    try:
        if not stat.S_ISREG(manifest_path.lstat().st_mode):
            raise OSError("manifest.json is not a regular file")
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleComposeError(
            "bundle.invalid", f"invalid SSMD bundle: cannot read manifest.json: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise BundleComposeError(
            "bundle.invalid", "invalid SSMD bundle: manifest must be a JSON object"
        )
    formats: dict[str, BundleKind] = {
        "ssmdconvert.book": "book",
        "ssmdconvert.document": "document",
    }
    format_name = manifest.get("format")
    if not isinstance(format_name, str) or format_name not in formats:
        raise BundleComposeError(
            "bundle.unsupported_format",
            f"unsupported SSMD bundle format: {format_name!r}",
        )
    kind = formats[format_name]
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise BundleComposeError(
            "bundle.unsupported_format", "unsupported SSMD bundle format: schema_version must be 1"
        )
    return BundleIdentity(root=root, kind=kind, manifest_path=manifest_path)
