"""Book-level orchestration for composing RenderPlan-set variants.

This module deliberately depends only on the shared ``renderplan`` contracts;
chapter handoff validation and artifact verification remain in the existing
RenderPlan materializer.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal

try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = importlib.import_module("tomli")

from renderplan import (
    RenderPlanSet,
    RenderPlanSetChapter,
    load_plan,
    load_renderplan_set,
    validate_renderplan_set_profile,
    validate_set_chapter_plan,
)

from ._workspace_policy import build_workspace_output_policy
from .errors import BookComposeError, HandoffValidationError, IntegrationError
from .integrations.renderplan_metadata import (
    RENDER_DIAGNOSTICS_FILENAME,
    RENDERPLAN_EMPHASIS,
    VOICERENDER_DEFERRED_CAPABILITIES,
    VOICERENDER_DIAGNOSTICS_SCHEMA,
    VOICERENDER_DIAGNOSTICS_SHA256,
    VOICERENDER_EMPHASIS_POLICY,
)
from .model import OutputPolicy
from .progress import CompositionProgress

SET_FILENAME = "renderplan-set.toml"
MANIFEST_FILENAME = "render-manifest.toml"
_CHAPTER_NUMBER = re.compile(r"[1-9][0-9]*\Z")
_BOOK_REPORT_FORMAT = "audiocompose" + "-book-report"
_BOOK_RESULT_FORMAT = "audiocompose" + "-book-result"


ChapterComposeStatus = Literal["written", "reused", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class BookComposeRequest:
    """Normalized immutable inputs and resolved identities for one book run."""

    workspace: Path
    variant: str
    renderplan_variant_dir: Path
    voicerender_variant_dir: Path
    output_variant_dir: Path
    chapters: tuple[RenderPlanSetChapter, ...]
    chapter_selector: str
    profile_snapshot_path: Path
    pause_policy: Any = None
    pause_policy_provenance: Literal["renderprofile", "explicit"] = "renderprofile"
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    force: bool = False
    fail_fast: bool = False
    chapter_paths: tuple[ChapterPaths, ...] = ()

    def __post_init__(self) -> None:
        _validate_variant_name(self.variant)
        for name in (
            "workspace",
            "renderplan_variant_dir",
            "voicerender_variant_dir",
            "output_variant_dir",
            "profile_snapshot_path",
        ):
            path = Path(getattr(self, name)).expanduser().resolve(strict=False)
            object.__setattr__(self, name, path)
        if not self.chapter_selector.strip():
            raise BookComposeError(
                "selection.invalid", "chapter_selector must not be empty", variant=self.variant
            )
        object.__setattr__(self, "chapters", tuple(self.chapters))
        object.__setattr__(self, "chapter_paths", tuple(self.chapter_paths))
        if not isinstance(self.output_policy, OutputPolicy):
            raise TypeError("output_policy must be an AudioCompose OutputPolicy")


@dataclass(frozen=True, slots=True)
class BookComposeOptions:
    """Optional output/loudness overrides for :func:`compose_book`.

    ``None`` inherits the corresponding :class:`OutputPolicy` default.
    """

    sample_rate: int | None = None
    channels: int | None = None
    clip_policy: Literal["clamp", "warn", "error"] | None = None
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = None
    peak_policy: Literal["reduce_gain", "error"] | None = None
    collect_metrics: bool | None = None


@dataclass(frozen=True, slots=True)
class ChapterComposeOutcome:
    """Typed outcome for one selected chapter."""

    id: str
    source_number: int
    render_plan_id: str
    status: ChapterComposeStatus
    output: Path | None = None
    sha256: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    index: int | None = None
    source_plan_id: str | None = None
    render_profile_id: str | None = None
    manifest_id: str | None = None
    manifest_sha256: str | None = None
    render_diagnostics_sha256: str | None = None
    voicerender_diagnostics_schema: str | None = None
    voicerender_emphasis_policy: str | None = None
    voicerender_deferred_capabilities: tuple[str, ...] | None = None
    pause_policy_id: str | None = None
    output_policy_id: str | None = None
    emphasis_warning: str | None = None
    audiojob: str | None = None

    utterplan: Path | None = None
    render_manifest: Path | None = None
    render_diagnostics: Path | None = None
    job_path: Path | None = None
    job_id: str | None = None
    job_schema_version: int | None = None
    audiocompose_version: str | None = None

    @property
    def output_path(self) -> Path | None:
        """Return the final chapter WAV path (legacy alias: ``output``)."""
        return self.output

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy CLI/report-item shape, omitting absent optional fields."""
        result: dict[str, Any] = {
            "id": self.id,
            "source_number": self.source_number,
            "render_plan_id": self.render_plan_id,
            "status": self.status,
        }
        for name in (
            "index",
            "source_plan_id",
            "render_profile_id",
            "manifest_id",
            "sha256",
            "manifest_sha256",
            "render_diagnostics_sha256",
            "voicerender_diagnostics_schema",
            "voicerender_emphasis_policy",
            "pause_policy_id",
            "sample_rate",
            "channels",
            "frames",
            "duration_seconds",
            "error_code",
            "error_message",
            "output_policy_id",
            "emphasis_warning",
            "audiojob",
            "job_schema_version",
            "audiocompose_version",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.voicerender_deferred_capabilities is not None:
            result["voicerender_deferred_capabilities"] = list(
                self.voicerender_deferred_capabilities
            )
        if self.output is not None:
            result["output"] = self.output.name
        for name in ("utterplan", "render_manifest", "render_diagnostics", "job_path"):
            value = getattr(self, name)
            if value is not None:
                result[name] = str(value)
        if self.job_id is not None:
            result["job_id"] = self.job_id
        if self.output is not None:
            result["output_path"] = self.output.name
        return result


@dataclass(frozen=True, slots=True)
class BookComposeProgress:
    """Book-level progress with stable chapter identity and an optional legacy outcome."""

    completed_chapters: int
    requested_chapters: int
    kind: str = "chapter_compose_completed"
    chapter_id: str | None = None
    source_number: int | None = None
    variant: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    outcome: ChapterComposeOutcome | None = None
    composition: CompositionProgress | None = None
    fraction: float | None = None

    def __post_init__(self) -> None:
        details = MappingProxyType(dict(self.details))
        object.__setattr__(self, "details", details)
        if self.outcome is not None:
            if self.chapter_id is None:
                object.__setattr__(self, "chapter_id", self.outcome.id)
            if self.source_number is None:
                object.__setattr__(self, "source_number", self.outcome.source_number)
            if self.kind == "chapter_compose_completed":
                kind = {
                    "failed": "chapter_failed",
                    "skipped": "chapter_skipped",
                    "reused": "chapter_reused",
                }.get(self.outcome.status, self.kind)
                object.__setattr__(self, "kind", kind)


BookComposeProgressCallback = Callable[[BookComposeProgress], None]


def _notify_progress(
    callback: BookComposeProgressCallback | None, progress: BookComposeProgress
) -> None:
    if callback is None:
        return
    if progress.fraction is None:
        requested = progress.requested_chapters
        completed = progress.completed_chapters
        if requested <= 0:
            fraction = 1.0 if progress.kind == "book_completed" else 0.0
        elif progress.outcome is not None or progress.kind == "book_completed":
            fraction = completed / requested
        else:
            chapter_fraction = (
                progress.composition.fraction
                if progress.composition is not None and progress.composition.fraction is not None
                else 0.0
            )
            fraction = (completed + chapter_fraction) / requested
        progress = replace(progress, fraction=min(1.0, max(0.0, fraction)))
    try:
        callback(progress)
    except Exception as exc:
        raise BookComposeError(
            "progress.callback_failed",
            f"progress callback failed during {progress.kind}: {exc}",
        ) from exc


@dataclass(frozen=True, slots=True)
class BookComposeResult:
    """Typed summary and per-chapter outcomes for a book composition."""

    request: BookComposeRequest
    workspace: Path
    variant: str
    output_variant_dir: Path
    report_path: Path
    render_profile_id: str
    outcomes: tuple[ChapterComposeOutcome, ...]
    requested: int
    written: int
    reused: int
    failed: int
    skipped: int
    renderplan_variant_dir: Path
    voicerender_variant_dir: Path
    selector: str = "all"
    emphasis_policy: Literal["segment_fallback", "strict"] = "segment_fallback"
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    report_error: Mapping[str, str] | None = None

    @property
    def exit_code(self) -> int:
        """Return the CLI-compatible status code for composition/report failures."""
        return 2 if self.failed or self.report_error is not None else 0

    @property
    def succeeded(self) -> int:
        """Count chapters with a complete output, whether written now or reused."""
        return self.written + self.reused

    @property
    def success(self) -> bool:
        """Whether all requested chapters completed without failure."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible result shape used by the CLI."""
        return {
            "format": _BOOK_RESULT_FORMAT,
            "request": {
                "workspace": str(self.request.workspace),
                "variant": self.request.variant,
                "renderplan_variant_dir": str(self.request.renderplan_variant_dir),
                "voicerender_variant_dir": str(self.request.voicerender_variant_dir),
                "output_variant_dir": str(self.request.output_variant_dir),
                "chapter_selector": self.request.chapter_selector,
                "profile_snapshot_path": str(self.request.profile_snapshot_path),
                "pause_policy_id": getattr(self.request.pause_policy, "policy_id", None),
                "pause_policy_provenance": self.request.pause_policy_provenance,
                "pause_policy": (
                    self.request.pause_policy.to_dict()
                    if callable(getattr(self.request.pause_policy, "to_dict", None))
                    else None
                ),
                "chapters": [chapter.id for chapter in self.request.chapters],
                "chapter_paths": [
                    {
                        "chapter_id": paths.chapter.id,
                        "source_number": paths.chapter.source_number,
                        "utterplan": str(paths.utterplan_path),
                        "renderplan": str(paths.plan_path),
                        "render_manifest": str(paths.manifest_path),
                        "render_diagnostics": str(paths.diagnostics_path),
                        "output": str(paths.output_path),
                    }
                    for paths in self.request.chapter_paths
                ],
                "force": self.request.force,
                "fail_fast": self.request.fail_fast,
                "output_policy": _output_policy_dict(self.request.output_policy),
            },
            "variant": self.variant,
            "emphasis_policy": self.emphasis_policy,
            "pause_policy_id": getattr(self.request.pause_policy, "policy_id", None),
            "pause_policy_provenance": self.request.pause_policy_provenance,
            "diagnostics": [dict(item) for item in self.diagnostics],
            "workspace": str(self.workspace),
            "renderplan_root": str(self.renderplan_variant_dir),
            "voicerender_root": str(self.voicerender_variant_dir),
            "output_root": str(self.output_variant_dir),
            "selector": self.selector,
            "requested": self.requested,
            "succeeded": self.succeeded,
            "success": self.success,
            "written": self.written,
            "reused": self.reused,
            "failed": self.failed,
            "skipped": self.skipped,
            "report": str(self.report_path),
            "report_path": str(self.report_path),
            "items": [outcome.to_dict() for outcome in self.outcomes],
            "report_error": dict(self.report_error) if self.report_error is not None else None,
        }


def default_book_output_policy(
    options: BookComposeOptions | None = None,
) -> OutputPolicy:
    """Build a policy from CLI-compatible defaults and optional public overrides."""
    return build_workspace_output_policy(options)


@dataclass(frozen=True, slots=True)
class ChapterPaths:
    utterplan_path: Path
    chapter: RenderPlanSetChapter
    plan_path: Path
    render_dir: Path
    manifest_path: Path
    diagnostics_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class BookContext:
    workspace: Path
    utterplan_root: Path
    variant: str
    renderplan_root: Path
    renderplan_variant_dir: Path
    voicerender_root: Path
    voicerender_variant_dir: Path
    output_root: Path
    output_variant_dir: Path
    set_path: Path
    profile_snapshot_path: Path
    report_path: Path
    renderplan_set: RenderPlanSet
    selector: str
    chapters: tuple[RenderPlanSetChapter, ...]
    explicit_media_root: Path | None = None
    save_audiojobs_dir: Path | None = None
    resolved_chapter_paths: tuple[ChapterPaths, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedRenderHandoff:
    chapter: RenderPlanSetChapter
    plan: Any
    manifest_id: str
    render_dir: Path
    job: Any
    diagnostics_sha256: str


def _path_without_symlinks(path: Path, *, label: str) -> Path:
    """Return an absolute path after rejecting symlinks along the lexical walk."""
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
            raise BookComposeError("workspace.invalid", f"{label} is unavailable: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise BookComposeError("workspace.invalid", f"{label} contains a symlink: {current}")
    return Path(os.path.abspath(current))


def _existing_directory(path: Path, *, label: str, code: str = "workspace.invalid") -> Path:
    absolute = _path_without_symlinks(path, label=label)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise BookComposeError(code, f"{label} is unavailable: {absolute}: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise BookComposeError(code, f"{label} is not a directory: {absolute}")
    return absolute.resolve(strict=True)


def _regular_file(path: Path, *, label: str, code: str) -> Path:
    absolute = _path_without_symlinks(path, label=label)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise BookComposeError(code, f"{label} is unavailable: {absolute}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise BookComposeError(code, f"{label} is not a regular file: {absolute}")
    return absolute.resolve(strict=True)


def _relative_posix_parts(reference: str, *, label: str, code: str) -> tuple[str, ...]:
    if (
        not isinstance(reference, str)
        or not reference
        or "\\" in reference
        or "\x00" in reference
        or PurePosixPath(reference).is_absolute()
        or PureWindowsPath(reference).drive
        or any(part in {"", "."} for part in reference.split("/"))
    ):
        raise BookComposeError(code, f"{label} must be a normalized relative POSIX path")
    return PurePosixPath(reference).parts


def _owned_regular_file(
    base: Path,
    owner_root: Path,
    reference: str,
    *,
    label: str,
    code: str,
    require_exists: bool = True,
) -> Path:
    parts = _relative_posix_parts(reference, label=label, code=code)
    candidate = _path_without_symlinks(base.joinpath(*parts), label=label)
    try:
        resolved = candidate.resolve(strict=require_exists)
    except (OSError, RuntimeError) as exc:
        raise BookComposeError(code, f"{label} is unavailable: {candidate}: {exc}") from exc
    if not resolved.is_relative_to(owner_root):
        raise BookComposeError(code, f"{label} escapes its owning stage root: {reference}")
    if not require_exists and resolved.exists():
        return _regular_file(resolved, label=label, code=code)
    return _regular_file(resolved, label=label, code=code) if require_exists else resolved


def _resolve_workspace(book: str | Path) -> Path:
    path = Path(book).expanduser()
    try:
        if path.name == "chapters":
            _existing_directory(path, label="chapters directory")
            path = path.parent
        root = _existing_directory(path, label="workspace")
        manifest_path = _regular_file(
            root / "manifest.json", label="SSMD bundle manifest", code="workspace.invalid"
        )
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BookComposeError(
                "workspace.invalid", f"cannot read SSMD bundle manifest {manifest_path}: {exc}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("format") != "ssmdconvert.book"
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
        ):
            raise BookComposeError(
                "workspace.invalid", "workspace manifest is not a supported .ssmd-bundle manifest"
            )
        return root
    except BookComposeError as exc:
        raise BookComposeError("workspace.invalid", str(exc)) from exc


def _validate_variant_name(variant: str) -> None:
    if (
        not isinstance(variant, str)
        or not variant
        or variant in {".", ".."}
        or "/" in variant
        or "\\" in variant
        or "\x00" in variant
        or PureWindowsPath(variant).drive
    ):
        raise BookComposeError(
            "variant.invalid",
            "--variant must be a single directory basename",
            variant=variant,
        )


def parse_chapter_selector(
    selector: str | None,
    chapters: tuple[RenderPlanSetChapter, ...],
) -> tuple[RenderPlanSetChapter, ...]:
    """Select source chapter numbers while preserving authoritative set order."""
    value = "all" if selector is None else selector.strip()
    if value == "all":
        return chapters
    if not value:
        raise BookComposeError(
            "selection.invalid", "--chapters must be 'all' or a number/range selector"
        )

    available = {chapter.source_number for chapter in chapters}
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise BookComposeError(
                "selection.invalid", f"invalid empty chapter selector in {value!r}"
            )
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not all(_CHAPTER_NUMBER.fullmatch(part) for part in bounds):
                raise BookComposeError("selection.invalid", f"invalid chapter range: {token!r}")
            first, last = (int(part) for part in bounds)
            if first > last:
                raise BookComposeError(
                    "selection.invalid", f"chapter range start exceeds end: {token!r}"
                )
            present = sorted(number for number in available if first <= number <= last)
            if len(present) != last - first + 1:
                missing = first
                for number in present:
                    if number != missing:
                        break
                    missing += 1
                raise BookComposeError(
                    "selection.invalid", f"unknown selected chapter number: {missing}"
                )
            duplicates = selected.intersection(present)
            if duplicates:
                number = min(duplicates)
                raise BookComposeError(
                    "selection.invalid", f"chapter {number} was requested more than once"
                )
            selected.update(present)
        elif _CHAPTER_NUMBER.fullmatch(token):
            number = int(token)
            if number not in available:
                raise BookComposeError(
                    "selection.invalid", f"unknown selected chapter number: {number}"
                )
            if number in selected:
                raise BookComposeError(
                    "selection.invalid", f"chapter {number} was requested more than once"
                )
            selected.add(number)
        else:
            raise BookComposeError("selection.invalid", f"invalid chapter selector: {token!r}")
    return tuple(chapter for chapter in chapters if chapter.source_number in selected)


def _variant_directory(root: Path, variant: str, *, label: str) -> Path:
    directory = root / variant
    try:
        return _existing_directory(directory, label=label, code="variant.not_found")
    except BookComposeError as exc:
        raise BookComposeError("variant.not_found", str(exc)) from exc


def resolve_book(
    book: str | Path,
    variant: str,
    *,
    chapters: str | None = None,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
    media_root: str | Path | None = None,
    save_audiojobs: str | Path | None = None,
) -> BookContext:
    """Resolve a book and validate its shared RenderPlan-set prerequisite."""
    _validate_variant_name(variant)
    workspace = _resolve_workspace(book)
    plan_root_arg = (
        Path(renderplan_root).expanduser()
        if renderplan_root is not None
        else workspace / "renderplan"
    )
    voice_root_arg = (
        Path(voicerender_root).expanduser()
        if voicerender_root is not None
        else workspace / "voicerender"
    )
    output_root_arg = (
        Path(output_root).expanduser() if output_root is not None else workspace / "audiocompose"
    )

    plan_root = _existing_directory(plan_root_arg, label="RenderPlan root")
    voice_root = _existing_directory(voice_root_arg, label="VoiceRender root")
    output_root_path = _path_without_symlinks(output_root_arg, label="output root")
    plan_variant_dir = _variant_directory(plan_root, variant, label="RenderPlan variant")
    voice_variant_dir = _variant_directory(voice_root, variant, label="VoiceRender variant")

    try:
        set_path = _regular_file(
            plan_variant_dir / SET_FILENAME,
            label="RenderPlan set index",
            code="index.invalid",
        )
    except BookComposeError as exc:
        raise BookComposeError("index.invalid", str(exc)) from exc
    try:
        renderplan_set = load_renderplan_set(set_path)
    except Exception as exc:
        raise BookComposeError("index.invalid", f"cannot load RenderPlan set: {exc}") from exc
    if variant != f"{renderplan_set.variant}--{renderplan_set.profile_fingerprint}":
        raise BookComposeError(
            "variant.identity_mismatch",
            "variant directory name does not match the RenderPlan set label and profile fingerprint",
        )

    try:
        utterplan_parts = _relative_posix_parts(
            renderplan_set.utterplan_dir, label="UtterPlan root", code="index.invalid"
        )
        utterplan_root = _existing_directory(
            plan_variant_dir.joinpath(*utterplan_parts),
            label="UtterPlan root",
            code="index.invalid",
        )
    except BookComposeError as exc:
        raise BookComposeError("index.invalid", str(exc), variant=variant) from exc
    if not utterplan_root.is_relative_to(workspace):
        raise BookComposeError(
            "index.invalid", "UtterPlan root escapes the .ssmd-bundle workspace", variant=variant
        )

    try:
        snapshot_path = _regular_file(
            plan_variant_dir / renderplan_set.snapshot,
            label="RenderProfile snapshot",
            code="profile.identity_mismatch",
        )
    except BookComposeError as exc:
        raise BookComposeError("profile.identity_mismatch", str(exc)) from exc
    try:
        validate_renderplan_set_profile(
            renderplan_set, snapshot_path, expected_profile_id=renderplan_set.profile_id
        )
    except Exception as exc:
        raise BookComposeError("profile.identity_mismatch", str(exc)) from exc

    selected = parse_chapter_selector(chapters, renderplan_set.chapters)
    explicit_media = Path(media_root).expanduser().absolute() if media_root is not None else None
    jobs_dir = (
        Path(save_audiojobs).expanduser().absolute()
        if save_audiojobs is not None
        else output_root_path / variant
    )
    context = BookContext(
        workspace=workspace,
        utterplan_root=utterplan_root,
        variant=variant,
        renderplan_root=plan_root,
        renderplan_variant_dir=plan_variant_dir,
        voicerender_root=voice_root,
        voicerender_variant_dir=voice_variant_dir,
        output_root=output_root_path,
        output_variant_dir=output_root_path / variant,
        set_path=set_path,
        report_path=output_root_path / variant / "compose-report.toml",
        profile_snapshot_path=snapshot_path,
        renderplan_set=renderplan_set,
        selector="all" if chapters is None else chapters,
        chapters=selected,
        explicit_media_root=explicit_media,
        save_audiojobs_dir=jobs_dir,
    )
    resolved_paths = tuple(paths_for_chapter(context, chapter) for chapter in selected)
    return replace(context, resolved_chapter_paths=resolved_paths)


def paths_for_chapter(context: BookContext, chapter: RenderPlanSetChapter) -> ChapterPaths:
    """Derive stage-owned chapter paths from one validated RenderPlan-set entry."""
    utterplan_path = _owned_regular_file(
        context.renderplan_variant_dir,
        context.utterplan_root,
        chapter.utterplan,
        label="UtterPlan source",
        code="source_plan.missing",
        require_exists=False,
    )
    plan_path = _owned_regular_file(
        context.renderplan_variant_dir,
        context.renderplan_variant_dir,
        chapter.renderplan,
        label="RenderPlan",
        code="plan.missing",
        require_exists=False,
    )
    render_dir = _path_without_symlinks(
        context.voicerender_variant_dir / chapter.id, label="VoiceRender chapter directory"
    )
    output_path = _path_without_symlinks(
        context.output_variant_dir / f"{chapter.id}.wav", label="AudioCompose chapter output"
    )
    if not output_path.is_relative_to(context.output_variant_dir):
        raise BookComposeError(
            "output.write_failed",
            "chapter output escapes the AudioCompose variant root",
            chapter_id=chapter.id,
            source_number=chapter.source_number,
            variant=context.variant,
        )
    return ChapterPaths(
        utterplan_path=utterplan_path,
        chapter=chapter,
        plan_path=plan_path,
        render_dir=render_dir,
        manifest_path=render_dir / MANIFEST_FILENAME,
        diagnostics_path=render_dir / RENDER_DIAGNOSTICS_FILENAME,
        output_path=output_path,
    )


def load_chapter_plan(context: BookContext, paths: ChapterPaths) -> Any:
    """Load both source and render plans, checking their shared identities."""
    details = {"chapter_id": paths.chapter.id, "source_number": paths.chapter.source_number}
    plan_path = _owned_regular_file(
        context.renderplan_variant_dir,
        context.renderplan_variant_dir,
        paths.chapter.renderplan,
        label="RenderPlan",
        code="plan.missing",
    )
    try:
        plan = load_plan(plan_path)
    except Exception as exc:
        raise BookComposeError(
            "plan.missing",
            f"cannot load RenderPlan {plan_path}: {exc}",
            variant=context.variant,
            details=details,
        ) from exc
    try:
        validate_set_chapter_plan(context.renderplan_set, paths.chapter, plan)
    except Exception as exc:
        raise BookComposeError(
            "plan.identity_mismatch",
            f"RenderPlan identity does not match RenderPlan set for {paths.chapter.id}: {exc}",
            variant=context.variant,
            details=details,
        ) from exc

    from .integrations.voicerender import load_source_plan

    source_path = _owned_regular_file(
        context.renderplan_variant_dir,
        context.utterplan_root,
        paths.chapter.utterplan,
        label="UtterPlan source",
        code="source_plan.missing",
    )
    try:
        source_plan = load_source_plan(source_path)
    except HandoffValidationError as exc:
        raise BookComposeError(
            "handoff.invalid",
            f"cannot load UtterPlan source for {paths.chapter.id}: {exc}",
            variant=context.variant,
            details=details,
        ) from exc
    except IntegrationError as exc:
        raise BookComposeError(
            "dependency.api_incompatible",
            str(exc),
            variant=context.variant,
            details={**details, "extra": "voicerender"},
        ) from exc
    if (
        source_plan.plan_id != plan.source_plan_id
        or source_plan.plan_id != paths.chapter.source_plan_id
    ):
        raise BookComposeError(
            "handoff.invalid",
            f"UtterPlan source identity does not match the RenderPlan handoff for {paths.chapter.id}",
            variant=context.variant,
            details=details,
        )
    return plan


def _output_policy_dict(policy: Any) -> dict[str, Any]:
    loudness = policy.loudness
    return {
        "sample_rate": policy.sample_rate,
        "channels": policy.channels,
        "clip_policy": policy.clip_policy,
        "target_lufs": loudness.target_lufs,
        "true_peak_ceiling_dbtp": loudness.true_peak_ceiling_dbtp,
        "peak_policy": loudness.peak_policy,
    }


def _policy_id(policy_dict: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(policy_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _emphasis_diagnostics(job: Any) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in job.items:
        report = item.metadata.get(RENDERPLAN_EMPHASIS)
        if not isinstance(report, dict) or report.get("scope") != "segment_fallback":
            continue
        ranges = report.get("partial_ranges", [])
        segment_number = item.metadata.get("renderplan.segment_number")
        diagnostics.append(
            {
                "code": "emphasis.segment_fallback",
                "severity": "warning",
                "segment_number": segment_number,
                "character_ranges": ranges,
                "gain_db": report.get("gain_db"),
                "message": (
                    f"speech {segment_number}: applied +{report.get('gain_db')} dB to the entire "
                    f"segment for character range(s) {ranges}; no trusted word-level sample alignment is available"
                ),
            }
        )
    return diagnostics


def _report_relative_path(path: Path, report_path: Path, workspace: Path, *, label: str) -> str:
    absolute = Path(os.path.abspath(path))
    workspace_absolute = Path(os.path.abspath(workspace))
    if not absolute.is_relative_to(workspace_absolute):
        raise BookComposeError(
            "workspace.invalid", f"{label} escapes the .ssmd-bundle workspace: {absolute}"
        )
    relative = Path(os.path.relpath(absolute, report_path.parent)).as_posix()
    report_relative = Path(os.path.abspath(report_path.parent / relative))
    if report_relative != absolute or not report_relative.is_relative_to(workspace_absolute):
        raise BookComposeError("workspace.invalid", f"{label} is not safely report-relative")
    return relative


def _toml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"Unsupported TOML scalar value: {type(value).__name__}")


def _report_toml(report: dict[str, Any]) -> str:
    lines = [
        f'format = "{_BOOK_REPORT_FORMAT}"',
        "version = 1",
        f"emphasis_policy = {_toml_scalar(report.get('emphasis_policy', 'segment_fallback'))}",
        f"variant = {_toml_scalar(report['variant'])}",
        f"render_profile_id = {_toml_scalar(report['render_profile_id'])}",
        f"pause_policy_id = {_toml_scalar(report.get('pause_policy_id', ''))}",
        f"pause_policy_provenance = {_toml_scalar(report.get('pause_policy_provenance', 'renderprofile'))}",
        f"audiocompose_version = {_toml_scalar(report.get('audiocompose_version', ''))}",
        f"workspace = {_toml_scalar(report.get('workspace', '.'))}",
        f"audiojobs_root = {_toml_scalar(report.get('audiojobs_root', '.'))}",
        f"output_root = {_toml_scalar(report.get('output_root', '.'))}",
        f"requested = {report['requested']}",
        f"written = {report['written']}",
        f"reused = {report['reused']}",
        f"failed = {report['failed']}",
        f"skipped = {report['skipped']}",
    ]
    progress_error = report.get("progress_error")
    if isinstance(progress_error, dict):
        lines.extend(
            (
                f"progress_error_code = {_toml_scalar(progress_error.get('code', ''))}",
                f"progress_error_message = {_toml_scalar(progress_error.get('message', ''))}",
            )
        )
    lines.extend(("", "[output]"))
    for key, value in report["output"].items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for entry in report["item"]:
        lines.extend(("", "[[item]]"))
        for key, value in entry.items():
            if value is not None:
                lines.append(f"{key} = {_toml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _read_prior_items(
    path: Path,
    variant: str,
    profile_id: str,
    pause_policy_id: str,
    pause_policy_provenance: str,
    audiocompose_version: str,
) -> dict[str, dict[str, Any]]:
    try:
        absolute = _regular_file(path, label="composition report", code="output.write_failed")
        with absolute.open("rb") as stream:
            report = tomllib.load(stream)
        if (
            report.get("format") != _BOOK_REPORT_FORMAT
            or report.get("version") != 1
            or report.get("variant") != variant
            or report.get("render_profile_id") != profile_id
            or report.get("pause_policy_id") != pause_policy_id
            or report.get("pause_policy_provenance") != pause_policy_provenance
            or report.get("audiocompose_version") != audiocompose_version
            or not isinstance(report.get("item"), list)
        ):
            return {}
        return {
            item["id"]: item
            for item in report["item"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    except (BookComposeError, OSError, ValueError, TypeError):
        return {}


def _verified_existing_output(
    path: Path,
    prior: dict[str, Any] | None,
    *,
    variant: str,
    chapter_id: str,
    source_number: int,
    source_plan_id: str,
    render_profile_id: str,
    plan_id: str,
    manifest_id: str,
    manifest_sha256: str,
    diagnostics_sha256: str,
    pause_policy_id: str,
    job_id: str,
    job_schema_version: int,
    audiocompose_version: str,
    prior_job_valid: bool,
    policy_id: str,
    expected_output: str,
    expected_sample_rate: int,
    expected_channels: int,
) -> tuple[bool, dict[str, Any] | None]:
    from .wav import sha256_file, wav_info

    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return False, None
        digest = "sha256:" + sha256_file(path)
        geometry = wav_info(path)
    except (OSError, ValueError):
        return False, None
    if not isinstance(prior, dict):
        return False, {
            "sha256": digest,
            "sample_rate": geometry.sample_rate,
            "frames": geometry.frames,
        }
    matches = (
        prior_job_valid
        and prior.get("variant") == variant
        and prior.get("id") == chapter_id
        and prior.get("source_number") == source_number
        and prior.get("source_plan_id") == source_plan_id
        and prior.get("render_profile_id") == render_profile_id
        and prior.get("render_plan_id") == plan_id
        and prior.get("manifest_id") == manifest_id
        and prior.get("manifest_sha256") == manifest_sha256
        and prior.get("render_diagnostics_sha256") == diagnostics_sha256
        and prior.get("pause_policy_id") == pause_policy_id
        and prior.get("job_id") == job_id
        and prior.get("job_schema_version") == job_schema_version
        and prior.get("audiocompose_version") == audiocompose_version
        and prior.get("status") in {"written", "reused"}
        and prior.get("output") == expected_output
        and prior.get("output_policy_id") == policy_id
        and geometry.sample_rate == expected_sample_rate
        and geometry.channels == expected_channels
        and prior.get("sha256") == digest
        and prior.get("sample_rate") == geometry.sample_rate
        and prior.get("channels") == geometry.channels
        and prior.get("frames") == geometry.frames
    )
    return matches, {
        "sha256": digest,
        "sample_rate": geometry.sample_rate,
        "channels": geometry.channels,
        "frames": geometry.frames,
    }


def _ensure_output_directory(path: Path) -> None:
    absolute = _path_without_symlinks(path, label="output directory")
    try:
        absolute.mkdir(parents=True, exist_ok=True)
        _existing_directory(absolute, label="output directory", code="output.write_failed")
    except (OSError, BookComposeError) as exc:
        raise BookComposeError(
            "output.write_failed", f"cannot prepare output directory {absolute}: {exc}"
        ) from exc


def _chapter_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, BookComposeError):
        return exc.code, str(exc)
    from .errors import HandoffValidationError, MediaResolutionError, PausePolicyError

    if isinstance(exc, PausePolicyError):
        code = (
            "pause_policy.unresolved"
            if "no duration configured" in str(exc)
            else "pause_policy.invalid"
        )
        return code, str(exc)
    if isinstance(exc, MediaResolutionError):
        return "media.invalid", str(exc)
    if isinstance(exc, HandoffValidationError):
        message = str(exc)
        if "incomplete" in message.lower() or "must be complete" in message.lower():
            return "manifest.incomplete", message
        return "handoff.invalid", message
    if isinstance(exc, OSError):
        return "output.write_failed", str(exc)
    return "compose.failed", str(exc)


def _compose_resolved_book(
    context: BookContext,
    output_policy: OutputPolicy,
    *,
    force: bool = False,
    pause_policy: Any = None,
    pause_policy_provenance: Literal["renderprofile", "explicit"] = "renderprofile",
    default_role: str = "",
    fail_fast: bool = False,
    emphasis_policy: Literal["segment_fallback", "strict"] = "segment_fallback",
    on_progress: BookComposeProgressCallback | None = None,
) -> dict[str, Any]:
    """Materialize and compose selected chapters, recording a provenance report."""
    from ._version import __version__
    from .composer import Composer
    from .integrations.voicerender import materialize_voicerender_job
    from .job import load_job
    from .wav import atomic_write_wav, sha256_file

    policy_dict = _output_policy_dict(output_policy)
    base_policy_id = _policy_id({"output": policy_dict, "pause_policy_id": pause_policy.policy_id})
    if emphasis_policy not in ("segment_fallback", "strict"):
        raise BookComposeError(
            "emphasis.invalid", "emphasis_policy must be 'segment_fallback' or 'strict'"
        )
    prior_items = _read_prior_items(
        context.report_path,
        context.variant,
        context.renderplan_set.profile_id,
        pause_policy.policy_id,
        pause_policy_provenance,
        __version__,
    )
    item_by_id = dict(prior_items)
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failed = written = reused = skipped = 0
    progress_error: BookComposeError | None = None
    try:
        _notify_progress(
            on_progress,
            BookComposeProgress(
                completed_chapters=0,
                requested_chapters=len(context.chapters),
                kind="book_started",
                variant=context.variant,
                details={"selector": context.selector},
            ),
        )
    except BookComposeError as exc:
        progress_error = exc
    stop = progress_error is not None

    def compose_and_write_chapter(job, chapter, output_path):
        _notify_progress(
            on_progress,
            BookComposeProgress(
                completed_chapters=len(results),
                requested_chapters=len(context.chapters),
                kind="chapter_compose_started",
                chapter_id=chapter.id,
                source_number=chapter.source_number,
                variant=context.variant,
            ),
        )
        _ensure_output_directory(context.output_variant_dir)

        def forward_composition(event: CompositionProgress) -> None:
            _notify_progress(
                on_progress,
                BookComposeProgress(
                    completed_chapters=len(results),
                    requested_chapters=len(context.chapters),
                    kind="chapter_compose_progress",
                    chapter_id=chapter.id,
                    source_number=chapter.source_number,
                    variant=context.variant,
                    composition=event,
                ),
            )

        if on_progress is None:
            composition = Composer().compose(job)
        else:
            composition = Composer().compose(job, on_progress=forward_composition)
        atomic_write_wav(
            output_path,
            composition.audio,
            composition.sample_rate,
            clip_policy=job.output.clip_policy,
        )
        return composition

    for index, chapter in enumerate(context.chapters, start=1):
        paths = context.resolved_chapter_paths[index - 1]
        entry: dict[str, Any] = {
            "variant": context.variant,
            "index": index,
            "id": chapter.id,
            "source_number": chapter.source_number,
            "render_plan_id": chapter.render_plan_id,
            "source_plan_id": chapter.source_plan_id,
            "render_profile_id": context.renderplan_set.profile_id,
            "utterplan": _report_relative_path(
                paths.utterplan_path,
                context.report_path,
                context.workspace,
                label="UtterPlan report path",
            ),
            "render_manifest": _report_relative_path(
                paths.manifest_path,
                context.report_path,
                context.workspace,
                label="RenderManifest report path",
            ),
            "render_diagnostics": _report_relative_path(
                paths.diagnostics_path,
                context.report_path,
                context.workspace,
                label="VoiceRender diagnostics report path",
            ),
            "output": paths.output_path.name,
            "status": "failed",
        }
        if stop:
            if progress_error is not None and not results:
                entry.update(
                    status="failed",
                    error_code=progress_error.code,
                    error_message=str(progress_error),
                )
                failed += 1
            else:
                entry.update(
                    status="skipped",
                    error_code="progress.callback_failed" if progress_error else "fail_fast",
                    error_message=(
                        str(progress_error)
                        if progress_error is not None
                        else "skipped after an earlier chapter failed"
                    ),
                )
                skipped += 1
            results.append(entry)
            item_by_id[chapter.id] = entry
            if progress_error is None:
                try:
                    _notify_progress(
                        on_progress,
                        BookComposeProgress(
                            completed_chapters=len(results),
                            requested_chapters=len(context.chapters),
                            outcome=_chapter_outcome(entry, context),
                        ),
                    )
                except BookComposeError as exc:
                    progress_error = exc
                    stop = True
            continue
        try:
            _notify_progress(
                on_progress,
                BookComposeProgress(
                    completed_chapters=len(results),
                    requested_chapters=len(context.chapters),
                    kind="chapter_started",
                    chapter_id=chapter.id,
                    source_number=chapter.source_number,
                    variant=context.variant,
                ),
            )
            plan = load_chapter_plan(context, paths)
            try:
                render_dir = _existing_directory(
                    paths.render_dir,
                    label="VoiceRender chapter directory",
                    code="render_dir.missing",
                )
            except BookComposeError as exc:
                raise BookComposeError("render_dir.missing", str(exc)) from exc
            try:
                manifest_path = _regular_file(
                    paths.manifest_path, label="RenderManifest", code="manifest.missing"
                )
            except BookComposeError as exc:
                raise BookComposeError("manifest.missing", str(exc)) from exc
            try:
                _regular_file(
                    paths.diagnostics_path,
                    label="VoiceRender diagnostics",
                    code="diagnostics.missing",
                )
            except BookComposeError as exc:
                raise BookComposeError("diagnostics.missing", str(exc)) from exc
            media_root = context.explicit_media_root
            job = materialize_voicerender_job(
                paths.utterplan_path,
                plan,
                manifest_path,
                render_dir=render_dir,
                output_policy=output_policy,
                pause_policy=pause_policy,
                pause_policy_provenance=pause_policy_provenance,
                default_role=default_role,
                media_root=media_root,
                emphasis_policy=emphasis_policy,
            )
            handoff = VerifiedRenderHandoff(
                chapter=chapter,
                plan=plan,
                manifest_id=str(job.source["render_manifest.manifest_id"]),
                render_dir=render_dir,
                job=job,
                diagnostics_sha256=str(job.source[VOICERENDER_DIAGNOSTICS_SHA256]),
            )
            job = handoff.job
            emphasis_reports = [
                item.metadata[RENDERPLAN_EMPHASIS]
                for item in job.items
                if RENDERPLAN_EMPHASIS in item.metadata
            ]
            chapter_policy_id = (
                _policy_id(
                    {
                        "output": policy_dict,
                        "pause_policy_id": pause_policy.policy_id,
                        "emphasis": emphasis_reports,
                    }
                )
                if emphasis_reports
                else base_policy_id
            )
            entry["output_policy_id"] = chapter_policy_id
            chapter_diagnostics = _emphasis_diagnostics(job)
            for diagnostic in chapter_diagnostics:
                diagnostic["chapter_id"] = chapter.id
            diagnostics.extend(chapter_diagnostics)
            if chapter_diagnostics:
                entry["emphasis_warning"] = "; ".join(
                    diagnostic["message"] for diagnostic in chapter_diagnostics
                )
            manifest_id = handoff.manifest_id

            manifest_sha256 = "sha256:" + sha256_file(manifest_path)
            entry.update(
                manifest_id=manifest_id,
                manifest_sha256=manifest_sha256,
                render_diagnostics_sha256=handoff.diagnostics_sha256,
                voicerender_diagnostics_schema=job.source[VOICERENDER_DIAGNOSTICS_SCHEMA],
                voicerender_emphasis_policy=job.source[VOICERENDER_EMPHASIS_POLICY],
                voicerender_deferred_capabilities=list(
                    job.source[VOICERENDER_DEFERRED_CAPABILITIES]
                ),
                pause_policy_id=pause_policy.policy_id,
                audiocompose_version=__version__,
            )
            prior = prior_items.get(chapter.id)
            if context.save_audiojobs_dir is None:
                raise BookComposeError("job.write_failed", "an AudioJob bundle path is required")
            try:
                job_bundle = _path_without_symlinks(
                    context.save_audiojobs_dir / f"{chapter.id}.audiojob",
                    label="AudioCompose AudioJob bundle",
                )
            except BookComposeError as exc:
                raise BookComposeError("job.write_failed", str(exc)) from exc
            if job_bundle.is_symlink():
                raise BookComposeError(
                    "job.write_failed", f"refusing symlink AudioJob bundle: {job_bundle}"
                )
            prior_job_valid = False
            if not force and isinstance(prior, dict) and job_bundle.exists():
                try:
                    prior_job = load_job(job_bundle)
                    prior_source = prior_job.source
                    prior_job_valid = (
                        prior_job.job_id == prior.get("job_id")
                        and prior_job.schema_version == prior.get("job_schema_version")
                        and prior.get("audiojob") == job_bundle.name
                        and prior_source.get("renderplan.plan_id")
                        == (plan.plan_id or plan.computed_id())
                        and prior_source.get("render_manifest.manifest_id") == manifest_id
                        and prior_source.get(VOICERENDER_DIAGNOSTICS_SCHEMA)
                        == job.source.get(VOICERENDER_DIAGNOSTICS_SCHEMA)
                        and prior_source.get(VOICERENDER_DIAGNOSTICS_SHA256)
                        == job.source.get(VOICERENDER_DIAGNOSTICS_SHA256)
                        and prior_source.get(VOICERENDER_EMPHASIS_POLICY)
                        == job.source.get(VOICERENDER_EMPHASIS_POLICY)
                        and prior_source.get(VOICERENDER_DEFERRED_CAPABILITIES)
                        == job.source.get(VOICERENDER_DEFERRED_CAPABILITIES)
                        and prior.get("render_diagnostics_sha256") == handoff.diagnostics_sha256
                        and prior_source.get("audiocompose.pause_policy_id")
                        == pause_policy.policy_id
                        and prior_source.get("audiocompose.pause_policy_provenance")
                        == pause_policy_provenance
                    )
                except Exception:
                    prior_job_valid = False
            try:
                _ensure_output_directory(job_bundle.parent)
                job.save(job_bundle)
                persisted_job = load_job(job_bundle)
            except Exception as exc:
                raise BookComposeError(
                    "job.write_failed", f"cannot save or verify AudioJob bundle {job_bundle}: {exc}"
                ) from exc
            job_id = persisted_job.job_id
            job_schema_version = persisted_job.schema_version
            if not isinstance(job_id, str) or not job_id:
                raise BookComposeError(
                    "job.write_failed", f"persisted AudioJob has no stable identity: {job_bundle}"
                )
            entry.update(
                audiojob=job_bundle.name,
                job=job_bundle.name,
                job_id=job_id,
                job_schema_version=job_schema_version,
            )
            _notify_progress(
                on_progress,
                BookComposeProgress(
                    completed_chapters=len(results),
                    requested_chapters=len(context.chapters),
                    kind="chapter_materialized",
                    chapter_id=chapter.id,
                    source_number=chapter.source_number,
                    variant=context.variant,
                    details={"job_id": job_id, "manifest_id": manifest_id},
                ),
            )
            output_exists = paths.output_path.exists() or paths.output_path.is_symlink()
            if paths.output_path.is_symlink():
                raise BookComposeError(
                    "output.write_failed", f"refusing symlink output: {paths.output_path}"
                )
            if output_exists:
                matches, geometry = _verified_existing_output(
                    paths.output_path,
                    prior,
                    variant=context.variant,
                    chapter_id=chapter.id,
                    source_number=chapter.source_number,
                    source_plan_id=chapter.source_plan_id,
                    render_profile_id=context.renderplan_set.profile_id,
                    plan_id=plan.plan_id or plan.computed_id(),
                    manifest_id=manifest_id,
                    manifest_sha256=manifest_sha256,
                    diagnostics_sha256=handoff.diagnostics_sha256,
                    pause_policy_id=pause_policy.policy_id,
                    job_id=job_id,
                    job_schema_version=job_schema_version,
                    audiocompose_version=__version__,
                    prior_job_valid=prior_job_valid,
                    policy_id=chapter_policy_id,
                    expected_output=paths.output_path.name,
                    expected_sample_rate=policy_dict["sample_rate"],
                    expected_channels=policy_dict["channels"],
                )
                if matches and not force:
                    entry.update(geometry or {})
                    entry["status"] = "reused"
                    reused += 1
                elif not force:
                    raise BookComposeError(
                        "output.exists",
                        f"existing chapter output has no matching verified provenance: {paths.output_path}; use --force",
                    )
                else:
                    compose_and_write_chapter(job, chapter, paths.output_path)
                    entry.update(_wav_result(paths.output_path))
                    entry["status"] = "written"
                    written += 1
            else:
                compose_and_write_chapter(job, chapter, paths.output_path)
                entry.update(_wav_result(paths.output_path))
                entry["status"] = "written"
                written += 1
        except Exception as exc:
            code, message = _chapter_error(exc)
            if entry.get("status") == "written":
                written -= 1
            elif entry.get("status") == "reused":
                reused -= 1
            entry.update(status="failed", error_code=code, error_message=message)
            failed += 1
            stop = fail_fast or code == "progress.callback_failed"
        results.append(entry)
        item_by_id[chapter.id] = entry

        try:
            _notify_progress(
                on_progress,
                BookComposeProgress(
                    completed_chapters=len(results),
                    requested_chapters=len(context.chapters),
                    outcome=_chapter_outcome(entry, context),
                ),
            )
        except BookComposeError as exc:
            progress_error = exc
            stop = True
    try:
        _notify_progress(
            on_progress,
            BookComposeProgress(
                completed_chapters=len(results),
                requested_chapters=len(context.chapters),
                kind="book_completed",
                variant=context.variant,
                details={
                    "written": written,
                    "reused": reused,
                    "failed": failed,
                    "skipped": skipped,
                },
            ),
        )
    except BookComposeError as exc:
        progress_error = exc
    # Reports retain records for unselected chapters, ordered by the authoritative set.
    ordered_entries = [
        item_by_id[chapter.id]
        for chapter in context.renderplan_set.chapters
        if chapter.id in item_by_id
    ]
    report = {
        "variant": context.variant,
        "emphasis_policy": emphasis_policy,
        "render_profile_id": context.renderplan_set.profile_id,
        "pause_policy_id": pause_policy.policy_id,
        "pause_policy_provenance": pause_policy_provenance,
        "audiocompose_version": __version__,
        "workspace": _report_relative_path(
            context.workspace, context.report_path, context.workspace, label="workspace report path"
        ),
        "audiojobs_root": (
            Path(os.path.relpath(context.save_audiojobs_dir, context.report_path.parent)).as_posix()
            if context.save_audiojobs_dir is not None
            else "."
        ),
        "output_root": ".",
        "requested": len(context.chapters),
        "written": written,
        "reused": reused,
        "failed": failed,
        "skipped": skipped,
        "output": policy_dict,
        "item": ordered_entries,
    }
    if progress_error is not None:
        report["progress_error"] = {
            "code": progress_error.code,
            "message": str(progress_error),
        }
    report_error = (
        {"code": progress_error.code, "message": str(progress_error)}
        if progress_error is not None
        else None
    )
    try:
        _write_report(context.report_path, report)
    except Exception as exc:
        report_error = {"code": "output.write_failed", "message": str(exc)}
    return {
        "format": _BOOK_RESULT_FORMAT,
        "variant": context.variant,
        "emphasis_policy": emphasis_policy,
        "pause_policy_id": pause_policy.policy_id,
        "pause_policy_provenance": pause_policy_provenance,
        "diagnostics": diagnostics,
        "workspace": str(context.workspace),
        "renderplan_root": str(context.renderplan_variant_dir),
        "voicerender_root": str(context.voicerender_variant_dir),
        "output_root": str(context.output_variant_dir),
        "selector": context.selector,
        "requested": len(context.chapters),
        "written": written,
        "reused": reused,
        "failed": failed,
        "skipped": skipped,
        "report": str(context.report_path),
        "items": results,
        "report_error": report_error,
    }


def _chapter_outcome(entry: Mapping[str, Any], context: BookContext) -> ChapterComposeOutcome:
    output_name = entry.get("output")
    output = context.output_variant_dir / output_name if isinstance(output_name, str) else None
    utterplan_name = entry.get("utterplan")
    utterplan = Path(utterplan_name) if isinstance(utterplan_name, str) else None
    manifest_name = entry.get("render_manifest")
    render_manifest = Path(manifest_name) if isinstance(manifest_name, str) else None
    diagnostics_name = entry.get("render_diagnostics")
    render_diagnostics = Path(diagnostics_name) if isinstance(diagnostics_name, str) else None
    job_name = entry.get("job")
    job_path = Path(job_name) if isinstance(job_name, str) else None
    return ChapterComposeOutcome(
        id=entry["id"],
        source_number=entry["source_number"],
        render_plan_id=entry["render_plan_id"],
        status=entry["status"],
        output=output,
        sha256=entry.get("sha256"),
        sample_rate=entry.get("sample_rate"),
        channels=entry.get("channels"),
        frames=entry.get("frames"),
        duration_seconds=entry.get("duration_seconds"),
        error_code=entry.get("error_code"),
        error_message=entry.get("error_message"),
        index=entry.get("index"),
        source_plan_id=entry.get("source_plan_id"),
        render_profile_id=entry.get("render_profile_id"),
        manifest_id=entry.get("manifest_id"),
        manifest_sha256=entry.get("manifest_sha256"),
        render_diagnostics_sha256=entry.get("render_diagnostics_sha256"),
        voicerender_diagnostics_schema=entry.get("voicerender_diagnostics_schema"),
        voicerender_emphasis_policy=entry.get("voicerender_emphasis_policy"),
        voicerender_deferred_capabilities=(
            tuple(entry["voicerender_deferred_capabilities"])
            if isinstance(entry.get("voicerender_deferred_capabilities"), list)
            else None
        ),
        pause_policy_id=entry.get("pause_policy_id"),
        output_policy_id=entry.get("output_policy_id"),
        emphasis_warning=entry.get("emphasis_warning"),
        audiojob=entry.get("audiojob"),
        utterplan=utterplan,
        render_manifest=render_manifest,
        render_diagnostics=render_diagnostics,
        job_path=job_path,
        job_id=entry.get("job_id"),
        job_schema_version=entry.get("job_schema_version"),
        audiocompose_version=entry.get("audiocompose_version"),
    )


def _book_result_from_dict(
    context: BookContext, result: Mapping[str, Any], request: BookComposeRequest
) -> BookComposeResult:
    return BookComposeResult(
        request=request,
        workspace=context.workspace,
        variant=context.variant,
        output_variant_dir=context.output_variant_dir,
        report_path=context.report_path,
        render_profile_id=context.renderplan_set.profile_id,
        outcomes=tuple(_chapter_outcome(entry, context) for entry in result["items"]),
        requested=result["requested"],
        written=result["written"],
        reused=result["reused"],
        failed=result["failed"],
        skipped=result["skipped"],
        renderplan_variant_dir=context.renderplan_variant_dir,
        voicerender_variant_dir=context.voicerender_variant_dir,
        selector=context.selector,
        emphasis_policy=result["emphasis_policy"],
        diagnostics=tuple(result["diagnostics"]),
        report_error=result["report_error"],
    )


def compose_book(
    book: str | Path,
    *,
    variant: str,
    chapters: str = "all",
    pause_policy: Any = None,
    output_policy: OutputPolicy | None = None,
    options: BookComposeOptions | None = None,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
    media_root: str | Path | None = None,
    save_audiojobs: str | Path | None = None,
    force: bool = False,
    fail_fast: bool = False,
    emphasis_policy: Literal["segment_fallback", "strict"] = "segment_fallback",
    on_progress: BookComposeProgressCallback | None = None,
) -> BookComposeResult:
    """Resolve and compose a book variant through the stable public service.

    ``output_policy`` accepts a fully constructed policy. Alternatively,
    ``options`` supplies individual output/loudness overrides using
    :func:`default_book_output_policy`; passing both is an error.
    """
    if output_policy is not None and options is not None:
        raise ValueError("pass either output_policy or options, not both")
    if output_policy is not None and not isinstance(output_policy, OutputPolicy):
        raise TypeError("output_policy must be an AudioCompose OutputPolicy")
    policy = output_policy or default_book_output_policy(options)
    context = resolve_book(
        book,
        variant,
        chapters=chapters,
        renderplan_root=renderplan_root,
        voicerender_root=voicerender_root,
        output_root=output_root,
        media_root=media_root,
        save_audiojobs=save_audiojobs,
    )
    from renderplan import load_profile

    from .errors import PausePolicyError
    from .integrations.voicerender import PausePolicy, load_pause_policy

    try:
        profile = load_profile(context.profile_snapshot_path)
    except Exception as exc:
        raise BookComposeError(
            "profile.invalid",
            f"cannot load RenderProfile snapshot {context.profile_snapshot_path}: {exc}",
            variant=context.variant,
        ) from exc
    pause_policy_provenance: Literal["renderprofile", "explicit"] = (
        "renderprofile" if pause_policy is None else "explicit"
    )
    try:
        resolved_pause_policy = (
            PausePolicy.from_render_profile(profile)
            if pause_policy is None
            else load_pause_policy(pause_policy)
        )
    except PausePolicyError as exc:
        raise BookComposeError("pause_policy.invalid", str(exc), variant=context.variant) from exc
    request = BookComposeRequest(
        workspace=context.workspace,
        variant=context.variant,
        renderplan_variant_dir=context.renderplan_variant_dir,
        voicerender_variant_dir=context.voicerender_variant_dir,
        output_variant_dir=context.output_variant_dir,
        chapters=context.chapters,
        chapter_paths=context.resolved_chapter_paths,
        chapter_selector=context.selector,
        profile_snapshot_path=context.profile_snapshot_path,
        pause_policy=resolved_pause_policy,
        pause_policy_provenance=pause_policy_provenance,
        output_policy=policy,
        force=force,
        fail_fast=fail_fast,
    )
    result = _compose_resolved_book(
        context,
        policy,
        force=force,
        pause_policy=resolved_pause_policy,
        pause_policy_provenance=pause_policy_provenance,
        default_role=profile.default_role,
        fail_fast=fail_fast,
        emphasis_policy=emphasis_policy,
        on_progress=on_progress,
    )
    return _book_result_from_dict(context, result, request)


def _wav_result(path: Path) -> dict[str, Any]:
    from .wav import sha256_file, wav_info

    info = wav_info(path)
    return {
        "sha256": "sha256:" + sha256_file(path),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": info.frames / info.sample_rate,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    import tempfile

    _ensure_output_directory(path.parent)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_report_toml(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise BookComposeError(
            "output.write_failed", f"cannot write composition report {path}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "BookComposeError",
    "BookComposeOptions",
    "BookComposeRequest",
    "BookComposeProgress",
    "BookComposeProgressCallback",
    "BookComposeResult",
    "ChapterComposeOutcome",
    "ChapterComposeStatus",
    "compose_book",
    "default_book_output_policy",
    "BookContext",
    "ChapterPaths",
    "VerifiedRenderHandoff",
    "load_chapter_plan",
    "parse_chapter_selector",
    "paths_for_chapter",
    "resolve_book",
]
