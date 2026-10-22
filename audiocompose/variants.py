"""Read-only discovery of SSMD bundle variants and their pipeline status."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = importlib.import_module("tomli")

from renderplan import (
    load_manifest,
    load_plan,
    load_profile,
    load_renderplan_set,
    validate_renderplan_set_profile,
    validate_set_chapter_plan,
)

from .bundle_contract import BundleComposeError, BundleIdentity, BundleKind, load_bundle_identity
from .document_contract import (
    COMPOSE_REPORT_FILENAME,
    DOCUMENT_ID,
    DOCUMENT_RENDERPLAN_FILENAME,
    PROFILE_SNAPSHOT_FILENAME,
    RENDER_MANIFEST_FILENAME,
    DocumentWorkspaceContractError,
    read_document_workspace,
)
from .integrations.voicerender import (
    _validate_render_diagnostics,
    _validate_voicerender_producer,
)

PlanStatus = Literal["valid", "invalid"]
RenderStatus = Literal["not_rendered", "partial", "complete", "failed", "unknown"]
ComposeStatus = Literal["not_composed", "partial", "complete", "failed", "stale", "unknown"]
VariantState = Literal[
    "invalid",
    "not_rendered",
    "render_partial",
    "render_failed",
    "ready",
    "compose_partial",
    "compose_failed",
    "stale",
    "complete",
    "unknown",
]
UnitKind = Literal["chapters", "documents"]

MAX_METADATA_BYTES = 2 * 1024 * 1024
SET_FILENAME = "renderplan-set.toml"
DIAGNOSTICS_FILENAME = "render-diagnostics.json"
BOOK_REPORT_FORMAT = "audiocompose-book-report"
DOCUMENT_REPORT_FORMAT = "audiocompose-document-report"


class _VariantInspectionError(ValueError):
    """An unsafe or contradictory artifact in one variant."""


@dataclass(frozen=True, slots=True)
class BundleVariant:
    name: str
    profile_id: str | None
    plan_status: PlanStatus
    render_status: RenderStatus
    compose_status: ComposeStatus
    state: VariantState
    unit_kind: UnitKind
    total_units: int
    rendered_units: int
    composed_units: int
    planned_speech: int
    available_speech: int
    ready_to_compose: bool
    validation_error: str | None = None
    render_error: str | None = None
    compose_error: str | None = None
    voicerender_report_path: Path | None = None
    compose_report_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "profile_id": self.profile_id,
            "plan_status": self.plan_status,
            "render_status": self.render_status,
            "compose_status": self.compose_status,
            "state": self.state,
            "unit_kind": self.unit_kind,
            "total_units": self.total_units,
            "rendered_units": self.rendered_units,
            "composed_units": self.composed_units,
            "planned_speech": self.planned_speech,
            "available_speech": self.available_speech,
            "ready_to_compose": self.ready_to_compose,
            "validation_error": self.validation_error,
            "render_error": self.render_error,
            "compose_error": self.compose_error,
            "voicerender_report_path": (
                str(self.voicerender_report_path)
                if self.voicerender_report_path is not None
                else None
            ),
            "compose_report_path": (
                str(self.compose_report_path) if self.compose_report_path is not None else None
            ),
        }
        result["completed_chapters" if self.unit_kind == "chapters" else "completed_documents"] = (
            self.composed_units
        )
        result["total_chapters" if self.unit_kind == "chapters" else "total_documents"] = (
            self.total_units
        )
        return result


@dataclass(frozen=True, slots=True)
class BundleVariantsResult:
    bundle: Path
    kind: BundleKind
    renderplan_root: Path
    voicerender_root: Path
    output_root: Path
    variants: tuple[BundleVariant, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "audiocompose-bundle-variants",
            "version": 1,
            "bundle": str(self.bundle),
            "kind": self.kind,
            "renderplan_root": str(self.renderplan_root),
            "voicerender_root": str(self.voicerender_root),
            "output_root": str(self.output_root),
            "count": len(self.variants),
            "variants": [variant.to_dict() for variant in self.variants],
        }


@dataclass(frozen=True, slots=True)
class _PlanUnit:
    id: str
    plan: Any
    plan_path: Path
    plan_id: str
    source_plan_id: str


@dataclass(frozen=True, slots=True)
class _ValidatedPlan:
    profile_id: str
    units: tuple[_PlanUnit, ...]
    unit_kind: UnitKind
    planned_speech: int


@dataclass(frozen=True, slots=True)
class _RenderInspection:
    status: RenderStatus
    rendered_units: int
    available_speech: int
    ready: bool
    report_path: Path | None
    error: str | None
    identities: dict[str, dict[str, str | None]]


@dataclass(frozen=True, slots=True)
class _ComposeInspection:
    status: ComposeStatus
    composed_units: int
    report_path: Path | None
    error: str | None


def _abs(path: str | Path) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlinks(path: Path, *, allow_missing: bool, label: str) -> Path:
    """Normalize lexically while checking every supplied path component with lstat."""
    expanded = Path(path).expanduser()
    current = Path(expanded.anchor) if expanded.is_absolute() else Path.cwd()
    missing = False
    for part in expanded.parts:
        if part in {expanded.anchor, "", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        if missing:
            continue
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            if not allow_missing:
                raise _VariantInspectionError(f"{label} is unavailable: {current}") from exc
            missing = True
            continue
        except OSError as exc:
            raise _VariantInspectionError(f"{label} is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise _VariantInspectionError(f"{label} contains a symlink: {current}")
    return _abs(expanded)


def _stage_root(bundle: Path, override: str | Path | None, name: str, label: str) -> Path:
    raw = bundle / name if override is None else Path(override).expanduser()
    try:
        root = _reject_symlinks(raw, allow_missing=True, label=label)
    except _VariantInspectionError as exc:
        raise BundleComposeError("bundle.invalid", str(exc)) from exc
    try:
        info = root.lstat()
    except FileNotFoundError:
        return root
    except OSError as exc:
        raise BundleComposeError("bundle.invalid", f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise BundleComposeError("bundle.invalid", f"{label} is not a directory: {root}")
    return root


def _contained_file(root: Path, path: Path, *, label: str, max_bytes: int | None = None) -> Path:
    root_abs = _abs(root)
    path_abs = _abs(path)
    if not path_abs.is_relative_to(root_abs):
        raise _VariantInspectionError(f"{label} escapes its stage root: {path}")
    try:
        root_info = root_abs.lstat()
    except OSError as exc:
        raise _VariantInspectionError(f"{label} root is unavailable: {exc}") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise _VariantInspectionError(f"{label} root is not a directory: {root_abs}")
    relative_parts = path_abs.relative_to(root_abs).parts
    current = root_abs
    for index, part in enumerate(relative_parts):
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise _VariantInspectionError(f"{label} is unavailable: {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise _VariantInspectionError(f"{label} must not be a symlink: {current}")
        final = index == len(relative_parts) - 1
        if final:
            if not stat.S_ISREG(info.st_mode):
                raise _VariantInspectionError(f"{label} must be a regular file: {current}")
            if max_bytes is not None and info.st_size > max_bytes:
                raise _VariantInspectionError(f"{label} exceeds the {max_bytes}-byte limit")
        elif not stat.S_ISDIR(info.st_mode):
            raise _VariantInspectionError(f"{label} path component is not a directory: {current}")
    resolved = path_abs.resolve(strict=True)
    if not resolved.is_relative_to(root_abs):
        raise _VariantInspectionError(f"{label} escapes its stage root: {path}")
    return resolved


def _load_toml(path: Path, *, root: Path, label: str) -> dict[str, Any]:
    safe_path = _contained_file(root, path, label=label, max_bytes=MAX_METADATA_BYTES)
    try:
        payload = tomllib.loads(safe_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _VariantInspectionError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _VariantInspectionError(f"{label} must contain a TOML table")
    return payload


def _safe_relative_file(root: Path, base: Path, reference: Any, *, label: str) -> Path:
    if (
        not isinstance(reference, str)
        or not reference
        or "\\" in reference
        or "\x00" in reference
        or PurePosixPath(reference).is_absolute()
        or PureWindowsPath(reference).drive
        or any(part in {"", ".", ".."} for part in reference.split("/"))
    ):
        raise _VariantInspectionError(f"{label} must be a safe relative POSIX path")
    return _contained_file(
        root,
        base.joinpath(*PurePosixPath(reference).parts),
        label=label,
        max_bytes=MAX_METADATA_BYTES,
    )


def _validate_book_candidate(candidate: Path, renderplan_root: Path) -> _ValidatedPlan:
    set_path = _contained_file(
        renderplan_root,
        candidate / SET_FILENAME,
        label="RenderPlan set",
        max_bytes=MAX_METADATA_BYTES,
    )
    try:
        renderplan_set = load_renderplan_set(set_path)
    except Exception as exc:
        raise _VariantInspectionError(f"cannot load RenderPlan set: {exc}") from exc
    expected_name = f"{renderplan_set.variant}--{renderplan_set.profile_fingerprint}"
    if candidate.name != expected_name:
        raise _VariantInspectionError(
            "variant directory name does not match the RenderPlan set label and profile fingerprint"
        )
    snapshot = _safe_relative_file(
        renderplan_root, candidate, renderplan_set.snapshot, label="RenderProfile snapshot"
    )
    try:
        validate_renderplan_set_profile(
            renderplan_set, snapshot, expected_profile_id=renderplan_set.profile_id
        )
    except Exception as exc:
        raise _VariantInspectionError(f"invalid RenderProfile snapshot: {exc}") from exc
    units: list[_PlanUnit] = []
    for chapter in renderplan_set.chapters:
        plan_path = _safe_relative_file(
            renderplan_root, candidate, chapter.renderplan, label=f"RenderPlan for {chapter.id}"
        )
        try:
            plan = load_plan(plan_path)
            validate_set_chapter_plan(renderplan_set, chapter, plan)
        except Exception as exc:
            raise _VariantInspectionError(f"invalid RenderPlan for {chapter.id}: {exc}") from exc
        units.append(
            _PlanUnit(
                id=chapter.id,
                plan=plan,
                plan_path=plan_path,
                plan_id=plan.plan_id or plan.computed_id(),
                source_plan_id=chapter.source_plan_id,
            )
        )
    return _ValidatedPlan(
        profile_id=renderplan_set.profile_id,
        units=tuple(units),
        unit_kind="chapters",
        planned_speech=sum(len(unit.plan.speech) for unit in units),
    )


def _validate_document_candidate(candidate: Path) -> _ValidatedPlan:
    profile_path = _contained_file(
        _abs(candidate.parent),
        candidate / PROFILE_SNAPSHOT_FILENAME,
        label="RenderProfile snapshot",
        max_bytes=MAX_METADATA_BYTES,
    )
    plan_path = _contained_file(
        _abs(candidate.parent),
        candidate / DOCUMENT_RENDERPLAN_FILENAME,
        label="document RenderPlan",
        max_bytes=MAX_METADATA_BYTES,
    )
    try:
        profile = load_profile(profile_path)
        plan = load_plan(plan_path)
    except Exception as exc:
        raise _VariantInspectionError(
            f"cannot load document RenderProfile or RenderPlan: {exc}"
        ) from exc
    profile_id = profile.profile_id or profile.computed_id()
    if not isinstance(profile_id, str) or not profile_id.startswith("sha256:"):
        raise _VariantInspectionError("RenderProfile has no valid profile identity")
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    if not candidate.name.endswith(f"--{fingerprint}") or candidate.name == f"--{fingerprint}":
        raise _VariantInspectionError(
            f"variant name does not end with the RenderProfile fingerprint {fingerprint!r}"
        )
    if getattr(plan, "render_profile_id", None) != profile_id:
        raise _VariantInspectionError(
            "document RenderPlan profile identity does not match the RenderProfile snapshot"
        )
    unit = _PlanUnit(
        id=DOCUMENT_ID,
        plan=plan,
        plan_path=plan_path,
        plan_id=plan.plan_id or plan.computed_id(),
        source_plan_id=getattr(plan, "source_plan_id", ""),
    )
    return _ValidatedPlan(
        profile_id=profile_id,
        units=(unit,),
        unit_kind="documents",
        planned_speech=len(plan.speech),
    )


def _enumerate_candidates(root: Path) -> tuple[Path, ...]:
    try:
        info = root.lstat()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise BundleComposeError(
            "bundle.invalid", f"cannot enumerate RenderPlan root: {exc}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise BundleComposeError("bundle.invalid", f"RenderPlan root is not a directory: {root}")
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise BundleComposeError(
            "bundle.invalid", f"cannot enumerate RenderPlan root: {exc}"
        ) from exc
    candidates: list[Path] = []
    for child in children:
        try:
            child_info = child.lstat()
        except OSError as exc:
            raise BundleComposeError(
                "bundle.invalid", f"cannot inspect RenderPlan root: {exc}"
            ) from exc
        if stat.S_ISLNK(child_info.st_mode) or stat.S_ISDIR(child_info.st_mode):
            candidates.append(child)
    return tuple(candidates)


def _read_json(path: Path, *, root: Path, label: str) -> tuple[dict[str, Any], str]:
    safe_path = _contained_file(root, path, label=label, max_bytes=MAX_METADATA_BYTES)
    try:
        raw = safe_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _VariantInspectionError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _VariantInspectionError(f"{label} must contain a JSON object")
    return payload, "sha256:" + hashlib.sha256(raw).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _inspect_voice_report(
    voice_variant: Path, voicerender_root: Path
) -> tuple[Path | None, bool, bool, str | None]:
    report_path = voice_variant / "voicerender-report.toml"
    try:
        report = _load_toml(report_path, root=voicerender_root, label="VoiceRender report")
    except _VariantInspectionError as exc:
        if not report_path.exists() and not report_path.is_symlink():
            return None, False, False, None
        return report_path, False, False, str(exc)
    except OSError:
        return report_path, False, False, "VoiceRender report is unavailable"
    status = report.get("status")
    if status is not None and not isinstance(status, str):
        return report_path, False, False, "VoiceRender report status must be a string"
    failed_count = report.get("failed")
    if failed_count is not None and (
        isinstance(failed_count, bool) or not isinstance(failed_count, int) or failed_count < 0
    ):
        return report_path, False, False, "VoiceRender report failed count is invalid"
    valid_statuses = {"failed", "error", "partial", "complete", "completed", "success", "succeeded"}
    if status is not None and status not in valid_statuses:
        return report_path, False, False, f"VoiceRender report status is invalid: {status!r}"
    complete = report.get("complete")
    if complete is not None and not isinstance(complete, bool):
        return report_path, False, False, "VoiceRender report complete field is invalid"
    if status is None and failed_count is None and complete is None:
        return report_path, False, False, "VoiceRender report has no recognized status"
    failed = status in {"failed", "error"} or (failed_count is not None and failed_count > 0)
    claimed_complete = (
        status in {"complete", "completed", "success", "succeeded"} or complete is True
    )
    return report_path, failed, claimed_complete, None


def _safe_unit_directory(root: Path, variant: str, unit_id: str) -> Path | None:
    variant_dir = root / variant
    try:
        variant_info = variant_dir.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _VariantInspectionError(f"VoiceRender variant is unavailable: {exc}") from exc
    if stat.S_ISLNK(variant_info.st_mode) or not stat.S_ISDIR(variant_info.st_mode):
        raise _VariantInspectionError("VoiceRender variant is not a safe directory")
    unit_dir = variant_dir / unit_id
    try:
        unit_info = unit_dir.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _VariantInspectionError(f"VoiceRender unit directory is unavailable: {exc}") from exc
    if stat.S_ISLNK(unit_info.st_mode) or not stat.S_ISDIR(unit_info.st_mode):
        raise _VariantInspectionError(f"VoiceRender unit {unit_id} is not a safe directory")
    return unit_dir


def _manifest_speech_paths(manifest: Any, render_dir: Path, voice_root: Path) -> None:
    seen: set[tuple[int, str]] = set()
    for artifact in manifest.speech:
        occurrence = (artifact.segment_number, artifact.segment_id)
        if occurrence in seen:
            raise _VariantInspectionError("RenderManifest contains duplicate speech occurrences")
        seen.add(occurrence)
        wav = getattr(artifact, "wav", None)
        if (
            not isinstance(wav, str)
            or not wav
            or "\\" in wav
            or "\x00" in wav
            or PurePosixPath(wav).is_absolute()
            or PureWindowsPath(wav).drive
            or any(part in {"", ".", ".."} for part in wav.split("/"))
        ):
            raise _VariantInspectionError("RenderManifest WAV path is not a safe relative path")
        _contained_file(
            voice_root,
            render_dir.joinpath(*PurePosixPath(wav).parts),
            label="VoiceRender WAV",
        )


def _inspect_render(variant: str, plan: _ValidatedPlan, voice_root: Path) -> _RenderInspection:
    voice_variant = voice_root / variant
    report_path: Path | None = None
    failed_report = False
    claimed_complete = False
    report_error: str | None = None
    if voice_variant.exists() or voice_variant.is_symlink():
        if voice_variant.is_symlink() or not voice_variant.is_dir():
            return _RenderInspection(
                "unknown", 0, 0, False, None, "VoiceRender variant is unsafe", {}
            )
        report_path, failed_report, claimed_complete, report_error = _inspect_voice_report(
            voice_variant, voice_root
        )
        if report_error is not None:
            return _RenderInspection("unknown", 0, 0, False, report_path, report_error, {})
    identities: dict[str, dict[str, str | None]] = {}
    rendered_units = 0
    complete_units = 0
    available_speech = 0
    diagnostics_ready = True
    observed_any = False
    try:
        for unit in plan.units:
            render_dir = _safe_unit_directory(voice_root, variant, unit.id)
            if render_dir is None:
                continue
            manifest_path = render_dir / RENDER_MANIFEST_FILENAME
            if not manifest_path.exists() and not manifest_path.is_symlink():
                continue
            observed_any = True
            manifest_safe = _contained_file(
                voice_root, manifest_path, label="RenderManifest", max_bytes=MAX_METADATA_BYTES
            )
            # Bound metadata before handing it to the shared public contract loader.
            manifest_raw = manifest_safe.read_bytes()
            manifest = load_manifest(manifest_safe)
            if manifest.manifest_id is not None and manifest.manifest_id != manifest.computed_id():
                raise _VariantInspectionError(f"RenderManifest identity is invalid for {unit.id}")
            manifest.validate_against(unit.plan)
            _validate_voicerender_producer(manifest)
            _manifest_speech_paths(manifest, render_dir, voice_root)
            rendered_units += 1
            available_speech += len(manifest.speech)
            manifest_id = manifest.manifest_id or manifest.computed_id()
            manifest_sha = "sha256:" + hashlib.sha256(manifest_raw).hexdigest()
            diagnostics_path = render_dir / DIAGNOSTICS_FILENAME
            diagnostics_hash: str | None = None
            if diagnostics_path.exists() or diagnostics_path.is_symlink():
                diagnostics, diagnostics_hash = _read_json(
                    diagnostics_path, root=voice_root, label="VoiceRender diagnostics"
                )
                try:
                    _validate_render_diagnostics(unit.plan, diagnostics, diagnostics_hash)
                except Exception as exc:
                    raise _VariantInspectionError(
                        f"invalid VoiceRender diagnostics for {unit.id}: {exc}"
                    ) from exc
            else:
                diagnostics_ready = False
            identities[unit.id] = {
                "render_plan_id": unit.plan_id,
                "source_plan_id": unit.source_plan_id,
                "render_profile_id": plan.profile_id,
                "manifest_id": manifest_id,
                "manifest_sha256": manifest_sha,
                "render_diagnostics_sha256": diagnostics_hash,
            }
            if manifest.complete is True:
                complete_units += 1
    except Exception as exc:
        return _RenderInspection(
            "unknown", rendered_units, available_speech, False, report_path, str(exc), identities
        )
    if failed_report:
        return _RenderInspection(
            "failed",
            rendered_units,
            available_speech,
            False,
            report_path,
            "VoiceRender report records a failed run",
            identities,
        )
    if claimed_complete and (
        not observed_any or rendered_units != len(plan.units) or complete_units != len(plan.units)
    ):
        return _RenderInspection(
            "unknown",
            rendered_units,
            available_speech,
            False,
            report_path,
            "VoiceRender report claims completion but complete manifests are missing",
            identities,
        )
    if not observed_any:
        return _RenderInspection("not_rendered", 0, 0, False, report_path, None, identities)
    if rendered_units == 0:
        return _RenderInspection("not_rendered", 0, 0, False, report_path, None, identities)
    if rendered_units == len(plan.units) and complete_units == len(plan.units):
        return _RenderInspection(
            "complete",
            rendered_units,
            available_speech,
            diagnostics_ready,
            report_path,
            None,
            identities,
        )
    return _RenderInspection(
        "partial", rendered_units, available_speech, False, report_path, None, identities
    )


def _safe_output_variant(root: Path, variant: str) -> Path | None:
    directory = root / variant
    try:
        info = directory.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _VariantInspectionError(f"AudioCompose output variant is unavailable: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise _VariantInspectionError("AudioCompose output variant is not a safe directory")
    return directory


def _output_file(root: Path, variant_dir: Path, unit_id: str, output_name: Any) -> Path:
    expected = f"{unit_id}.wav"
    if output_name != expected:
        raise _VariantInspectionError(
            f"composition report output for {unit_id} is not {expected!r}"
        )
    return _contained_file(root, variant_dir / expected, label=f"AudioCompose output for {unit_id}")


def _report_common(
    report: dict[str, Any], *, expected_format: str, variant: str, profile_id: str
) -> None:
    if (
        report.get("format") != expected_format
        or type(report.get("version")) is not int
        or report.get("version") != 1
        or report.get("variant") != variant
        or report.get("render_profile_id") != profile_id
    ):
        raise _VariantInspectionError("AudioCompose report identity or format is invalid")


def _inspect_book_compose(
    variant: str, plan: _ValidatedPlan, render: _RenderInspection, output_root: Path
) -> _ComposeInspection:
    report_path = output_root / variant / "compose-report.toml"
    try:
        variant_dir = _safe_output_variant(output_root, variant)
        if variant_dir is None:
            return _ComposeInspection("not_composed", 0, None, None)
        if not report_path.exists() and not report_path.is_symlink():
            any_output = any(
                (variant_dir / f"{unit.id}.wav").exists()
                or (variant_dir / f"{unit.id}.wav").is_symlink()
                for unit in plan.units
            )
            if any_output:
                return _ComposeInspection(
                    "unknown", 0, None, "AudioCompose output exists without an ownership report"
                )
            return _ComposeInspection("not_composed", 0, None, None)
        report = _load_toml(report_path, root=output_root, label="AudioCompose book report")
        _report_common(
            report, expected_format=BOOK_REPORT_FORMAT, variant=variant, profile_id=plan.profile_id
        )
        items = report.get("item")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise _VariantInspectionError("AudioCompose book report items are invalid")
        failed = report.get("failed", 0)
        if isinstance(failed, bool) or not isinstance(failed, int) or failed < 0:
            raise _VariantInspectionError("AudioCompose book report failed count is invalid")
        if (
            failed > 0
            or report.get("progress_error") is not None
            or any(item.get("status") == "failed" for item in items)
        ):
            return _ComposeInspection(
                "failed", 0, report_path, "AudioCompose report records failed work"
            )
        by_id: dict[str, dict[str, Any]] = {}
        for item in items:
            item_id = item.get("id")
            if not isinstance(item_id, str) or item_id in by_id:
                raise _VariantInspectionError(
                    "AudioCompose book report has invalid or duplicate item ids"
                )
            by_id[item_id] = item
        expected_ids = {unit.id for unit in plan.units}
        if not set(by_id).issubset(expected_ids):
            raise _VariantInspectionError("AudioCompose book report contains an unknown unit")
        if any(
            item.get("status") not in {"written", "reused", "failed", "skipped"}
            for item in by_id.values()
        ):
            raise _VariantInspectionError(
                "AudioCompose book report contains an unsupported item status"
            )
        completed = 0
        stale_reasons: list[str] = []
        for unit in plan.units:
            item = by_id.get(unit.id)
            if item is None:
                continue
            if item.get("status") not in {"written", "reused"}:
                continue
            _output_file(output_root, variant_dir, unit.id, item.get("output"))
            current = render.identities.get(unit.id)
            if current is None:
                stale_reasons.append(f"{unit.id} no longer has a valid VoiceRender manifest")
                continue
            checks = {
                "render_plan_id": current["render_plan_id"],
                "source_plan_id": current["source_plan_id"],
                "render_profile_id": current["render_profile_id"],
                "manifest_id": current["manifest_id"],
                "manifest_sha256": current["manifest_sha256"],
                "render_diagnostics_sha256": current["render_diagnostics_sha256"],
            }
            if any(item.get(key) != value for key, value in checks.items()):
                stale_reasons.append(f"{unit.id} upstream identity changed")
                continue
            completed += 1
        if stale_reasons:
            return _ComposeInspection("stale", completed, report_path, "; ".join(stale_reasons))
        if completed == len(plan.units) and len(by_id) == len(expected_ids):
            return _ComposeInspection("complete", completed, report_path, None)
        return _ComposeInspection("partial", completed, report_path, None)
    except Exception as exc:
        return _ComposeInspection(
            "unknown",
            0,
            report_path if report_path.exists() or report_path.is_symlink() else None,
            str(exc),
        )


def _inspect_document_compose(
    variant: str, plan: _ValidatedPlan, render: _RenderInspection, output_root: Path
) -> _ComposeInspection:
    report_path = output_root / variant / COMPOSE_REPORT_FILENAME
    unit = plan.units[0]
    try:
        variant_dir = _safe_output_variant(output_root, variant)
        if variant_dir is None:
            return _ComposeInspection("not_composed", 0, None, None)
        expected_output = variant_dir / f"{unit.id}.wav"
        if not report_path.exists() and not report_path.is_symlink():
            if expected_output.exists() or expected_output.is_symlink():
                return _ComposeInspection(
                    "unknown", 0, None, "AudioCompose output exists without an ownership report"
                )
            return _ComposeInspection("not_composed", 0, None, None)
        report = _load_toml(report_path, root=output_root, label="AudioCompose document report")
        if (
            report.get("format") != DOCUMENT_REPORT_FORMAT
            or type(report.get("version")) is not int
            or report.get("version") != 1
            or report.get("document_id") != unit.id
            or report.get("variant") != variant
            or report.get("render_profile_id") != plan.profile_id
        ):
            raise _VariantInspectionError(
                "AudioCompose document report identity or format is invalid"
            )
        status = report.get("status")
        if status not in {"written", "reused"}:
            raise _VariantInspectionError("AudioCompose document report status is invalid")
        _output_file(output_root, variant_dir, unit.id, report.get("output"))
        current = render.identities.get(unit.id)
        if current is None:
            return _ComposeInspection(
                "stale", 0, report_path, "document no longer has a valid VoiceRender manifest"
            )
        checks = {
            "source_plan_id": current["source_plan_id"],
            "render_profile_id": current["render_profile_id"],
            "render_plan_id": current["render_plan_id"],
            "render_manifest_id": current["manifest_id"],
            "render_manifest_sha256": current["manifest_sha256"],
            "render_diagnostics_sha256": current["render_diagnostics_sha256"],
        }
        if any(report.get(key) != value for key, value in checks.items()):
            return _ComposeInspection("stale", 0, report_path, "document upstream identity changed")
        return _ComposeInspection("complete", 1, report_path, None)
    except Exception as exc:
        return _ComposeInspection(
            "unknown",
            0,
            report_path if report_path.exists() or report_path.is_symlink() else None,
            str(exc),
        )


def _derive_state(
    plan_status: PlanStatus, render_status: RenderStatus, compose_status: ComposeStatus
) -> VariantState:
    if plan_status == "invalid":
        return "invalid"
    if render_status == "failed":
        return "render_failed"
    if render_status == "unknown":
        return "unknown"
    if render_status == "not_rendered":
        return "not_rendered"
    if render_status == "partial":
        return "render_partial"
    if compose_status == "stale":
        return "stale"
    if compose_status == "failed":
        return "compose_failed"
    if compose_status == "unknown":
        return "unknown"
    if compose_status == "partial":
        return "compose_partial"
    if compose_status == "complete":
        return "complete"
    return "ready"


def _invalid_variant(name: str, unit_kind: UnitKind, error: str) -> BundleVariant:
    return BundleVariant(
        name=name,
        profile_id=None,
        plan_status="invalid",
        render_status="not_rendered",
        compose_status="not_composed",
        state="invalid",
        unit_kind=unit_kind,
        total_units=0,
        rendered_units=0,
        composed_units=0,
        planned_speech=0,
        available_speech=0,
        ready_to_compose=False,
        validation_error=error,
    )


def _inspect_candidate(
    candidate: Path,
    identity: BundleIdentity,
    renderplan_root: Path,
    voicerender_root: Path,
    output_root: Path,
) -> BundleVariant:
    unit_kind: UnitKind = "chapters" if identity.kind == "book" else "documents"
    try:
        candidate_info = candidate.lstat()
        if stat.S_ISLNK(candidate_info.st_mode) or not stat.S_ISDIR(candidate_info.st_mode):
            raise _VariantInspectionError("variant candidate is not a real directory")
        candidate_real = candidate.resolve(strict=True)
        if not candidate_real.is_relative_to(renderplan_root):
            raise _VariantInspectionError("variant candidate escapes the RenderPlan root")
        plan = (
            _validate_book_candidate(candidate, renderplan_root)
            if identity.kind == "book"
            else _validate_document_candidate(candidate)
        )
    except Exception as exc:
        return _invalid_variant(candidate.name, unit_kind, str(exc))
    render = _inspect_render(candidate.name, plan, voicerender_root)
    if identity.kind == "book":
        composition = _inspect_book_compose(candidate.name, plan, render, output_root)
    else:
        composition = _inspect_document_compose(candidate.name, plan, render, output_root)
    state = _derive_state("valid", render.status, composition.status)
    return BundleVariant(
        name=candidate.name,
        profile_id=plan.profile_id,
        plan_status="valid",
        render_status=render.status,
        compose_status=composition.status,
        state=state,
        unit_kind=plan.unit_kind,
        total_units=len(plan.units),
        rendered_units=render.rendered_units,
        composed_units=composition.composed_units,
        planned_speech=plan.planned_speech,
        available_speech=render.available_speech,
        ready_to_compose=render.status == "complete" and render.ready,
        render_error=render.error,
        compose_error=composition.error,
        voicerender_report_path=render.report_path,
        compose_report_path=composition.report_path,
    )


def list_bundle_variants(
    bundle: str | Path,
    *,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> BundleVariantsResult:
    """List RenderPlan variants and inspect their downstream artifact status without writes."""
    identity = load_bundle_identity(bundle)
    plan_root = _stage_root(identity.root, renderplan_root, "renderplan", "RenderPlan root")
    if identity.kind == "document":
        try:
            read_document_workspace(identity.root)
        except DocumentWorkspaceContractError as exc:
            raise BundleComposeError("bundle.invalid", str(exc)) from exc
    voice_root = _stage_root(identity.root, voicerender_root, "voicerender", "VoiceRender root")
    audio_root = _stage_root(identity.root, output_root, "audiocompose", "AudioCompose output root")
    candidates = _enumerate_candidates(plan_root)
    variants = tuple(
        _inspect_candidate(candidate, identity, plan_root, voice_root, audio_root)
        for candidate in candidates
    )
    return BundleVariantsResult(
        bundle=identity.root,
        kind=identity.kind,
        renderplan_root=plan_root,
        voicerender_root=voice_root,
        output_root=audio_root,
        variants=variants,
    )


__all__ = ["BundleVariant", "BundleVariantsResult", "list_bundle_variants"]
