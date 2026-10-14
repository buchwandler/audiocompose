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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

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

from .errors import AudioComposeError

SET_FILENAME = "renderplan-set.toml"
MANIFEST_FILENAME = "render-manifest.toml"
_CHAPTER_NUMBER = re.compile(r"[1-9][0-9]*\Z")


class BookComposeError(AudioComposeError):
    """A book orchestration failure with a stable machine-readable category."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ChapterPaths:
    chapter: RenderPlanSetChapter
    plan_path: Path
    render_dir: Path
    manifest_path: Path
    output_path: Path


@dataclass(frozen=True, slots=True)
class BookContext:
    workspace: Path
    variant: str
    renderplan_root: Path
    renderplan_variant_dir: Path
    voicerender_root: Path
    voicerender_variant_dir: Path
    output_root: Path
    output_variant_dir: Path
    set_path: Path
    report_path: Path
    renderplan_set: RenderPlanSet
    selector: str
    chapters: tuple[RenderPlanSetChapter, ...]
    explicit_media_root: Path | None = None
    save_audiojobs_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class VerifiedRenderHandoff:
    chapter: RenderPlanSetChapter
    plan: Any
    manifest_id: str
    render_dir: Path
    job: Any


def _path_without_symlinks(path: Path, *, label: str) -> Path:
    """Return an absolute path after rejecting every existing symlink component."""
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BookComposeError("workspace.invalid", f"{label} is unavailable: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise BookComposeError("workspace.invalid", f"{label} contains a symlink: {current}")
    return absolute


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


def _resolve_workspace(book: str | Path) -> Path:
    path = Path(book).expanduser()
    try:
        if path.name == "chapters":
            _existing_directory(path, label="chapters directory")
            path = path.parent
        return _existing_directory(path, label="workspace")
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
        raise BookComposeError("variant.not_found", "--variant must be a single directory basename")


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
            "index.invalid", "--chapters must be 'all' or a number/range selector"
        )

    available = {chapter.source_number for chapter in chapters}
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise BookComposeError("index.invalid", f"invalid empty chapter selector in {value!r}")
        if "-" in token:
            bounds = token.split("-")
            if len(bounds) != 2 or not all(_CHAPTER_NUMBER.fullmatch(part) for part in bounds):
                raise BookComposeError("index.invalid", f"invalid chapter range: {token!r}")
            first, last = (int(part) for part in bounds)
            if first > last:
                raise BookComposeError(
                    "index.invalid", f"chapter range start exceeds end: {token!r}"
                )
            present = sorted(number for number in available if first <= number <= last)
            if len(present) != last - first + 1:
                missing = first
                for number in present:
                    if number != missing:
                        break
                    missing += 1
                raise BookComposeError(
                    "index.invalid", f"unknown selected chapter number: {missing}"
                )
            selected.update(present)
        elif _CHAPTER_NUMBER.fullmatch(token):
            number = int(token)
            if number not in available:
                raise BookComposeError(
                    "index.invalid", f"unknown selected chapter number: {number}"
                )
            selected.add(number)
        else:
            raise BookComposeError("index.invalid", f"invalid chapter selector: {token!r}")
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
    jobs_dir = Path(save_audiojobs).expanduser().absolute() if save_audiojobs is not None else None
    return BookContext(
        workspace=workspace,
        variant=variant,
        renderplan_root=plan_root,
        renderplan_variant_dir=plan_variant_dir,
        voicerender_root=voice_root,
        voicerender_variant_dir=voice_variant_dir,
        output_root=output_root_path,
        output_variant_dir=output_root_path / variant,
        set_path=set_path,
        report_path=output_root_path / variant / "compose-report.toml",
        renderplan_set=renderplan_set,
        selector="all" if chapters is None else chapters,
        chapters=selected,
        explicit_media_root=explicit_media,
        save_audiojobs_dir=jobs_dir,
    )


def paths_for_chapter(context: BookContext, chapter: RenderPlanSetChapter) -> ChapterPaths:
    """Derive safe chapter input and output paths from a validated set entry."""
    relative = PurePosixPath(chapter.renderplan)
    plan_path = context.renderplan_variant_dir.joinpath(*relative.parts)
    try:
        absolute_plan = _path_without_symlinks(plan_path, label="RenderPlan path")
    except BookComposeError as exc:
        raise BookComposeError("plan.missing", str(exc)) from exc
    try:
        plan_info = absolute_plan.lstat()
    except OSError as exc:
        raise BookComposeError(
            "plan.missing", f"RenderPlan is unavailable: {absolute_plan}: {exc}"
        ) from exc
    if not stat.S_ISREG(plan_info.st_mode) or not absolute_plan.resolve().is_relative_to(
        context.renderplan_variant_dir
    ):
        raise BookComposeError(
            "plan.missing", f"RenderPlan is not a regular file in the variant: {absolute_plan}"
        )

    render_dir = context.voicerender_variant_dir / chapter.id
    render_manifest_path = render_dir / MANIFEST_FILENAME
    output_path = context.output_variant_dir / f"{chapter.id}.wav"
    return ChapterPaths(
        chapter, absolute_plan.resolve(strict=True), render_dir, render_manifest_path, output_path
    )


def load_chapter_plan(context: BookContext, paths: ChapterPaths) -> Any:
    """Load a chapter RenderPlan and verify all identities declared by the set."""
    try:
        plan = load_plan(paths.plan_path)
    except Exception as exc:
        raise BookComposeError(
            "plan.missing", f"cannot load RenderPlan {paths.plan_path}: {exc}"
        ) from exc
    try:
        validate_set_chapter_plan(context.renderplan_set, paths.chapter, plan)
    except Exception as exc:
        raise BookComposeError(
            "plan.identity_mismatch",
            f"RenderPlan identity does not match RenderPlan set for {paths.chapter.id}: {exc}",
        ) from exc
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


def _toml_scalar(value: str | int | float | bool) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)


def _report_toml(report: dict[str, Any]) -> str:
    lines = [
        'format = "audiocompose-book-report"',
        "version = 1",
        f"variant = {_toml_scalar(report['variant'])}",
        f"render_profile_id = {_toml_scalar(report['render_profile_id'])}",
        f"requested = {report['requested']}",
        f"written = {report['written']}",
        f"reused = {report['reused']}",
        f"failed = {report['failed']}",
        f"skipped = {report['skipped']}",
        "",
        "[output]",
    ]
    for key, value in report["output"].items():
        if value is not None:
            lines.append(f"{key} = {_toml_scalar(value)}")
    for entry in report["item"]:
        lines.extend(("", "[[item]]"))
        for key, value in entry.items():
            if value is not None:
                lines.append(f"{key} = {_toml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _read_prior_items(path: Path, variant: str, profile_id: str) -> dict[str, dict[str, Any]]:
    try:
        absolute = _regular_file(path, label="composition report", code="output.write_failed")
        with absolute.open("rb") as stream:
            report = tomllib.load(stream)
        if (
            report.get("format") != "audiocompose-book-report"
            or report.get("version") != 1
            or report.get("variant") != variant
            or report.get("render_profile_id") != profile_id
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
    plan_id: str,
    manifest_id: str,
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
        prior.get("status") in {"written", "reused"}
        and prior.get("output") == expected_output
        and prior.get("render_plan_id") == plan_id
        and prior.get("manifest_id") == manifest_id
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
    from .errors import HandoffValidationError, MediaResolutionError

    if isinstance(exc, MediaResolutionError):
        return "media.invalid", str(exc)
    if isinstance(exc, HandoffValidationError):
        message = str(exc)
        if "incomplete" in message.lower():
            return "manifest.incomplete", message
        return "handoff.invalid", message
    if isinstance(exc, OSError):
        return "output.write_failed", str(exc)
    return "compose.failed", str(exc)


def compose_book(
    context: BookContext,
    output_policy: Any,
    *,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Materialize and compose selected chapters, recording a provenance report."""
    from .composer import Composer
    from .integrations.renderplan import load_and_materialize_renderplan_job
    from .wav import atomic_write_wav

    policy_dict = _output_policy_dict(output_policy)
    policy_id = _policy_id(policy_dict)
    prior_items = _read_prior_items(
        context.report_path, context.variant, context.renderplan_set.profile_id
    )
    item_by_id = dict(prior_items)
    results: list[dict[str, Any]] = []
    failed = written = reused = skipped = 0
    stop = False

    for index, chapter in enumerate(context.chapters, start=1):
        entry: dict[str, Any] = {
            "index": index,
            "id": chapter.id,
            "source_number": chapter.source_number,
            "render_plan_id": chapter.render_plan_id,
            "source_plan_id": chapter.source_plan_id,
            "render_profile_id": context.renderplan_set.profile_id,
            "status": "failed",
        }
        if stop:
            entry.update(
                status="skipped",
                error_code="fail_fast",
                error_message="skipped after an earlier chapter failed",
            )
            skipped += 1
            results.append(entry)
            item_by_id[chapter.id] = entry
            continue
        try:
            paths = paths_for_chapter(context, chapter)
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
            media_root = context.explicit_media_root
            job = load_and_materialize_renderplan_job(
                paths.plan_path,
                manifest_path,
                render_dir=render_dir,
                output_policy=output_policy,
                media_root=media_root,
            )
            handoff = VerifiedRenderHandoff(
                chapter=chapter,
                plan=plan,
                manifest_id=str(job.source["render_manifest.manifest_id"]),
                render_dir=render_dir,
                job=job,
            )
            job = handoff.job
            manifest_id = handoff.manifest_id
            entry["manifest_id"] = manifest_id
            entry["output"] = paths.output_path.name
            prior = prior_items.get(chapter.id)
            if paths.output_path.exists() or paths.output_path.is_symlink():
                matches, geometry = _verified_existing_output(
                    paths.output_path,
                    prior,
                    plan_id=plan.plan_id or plan.computed_id(),
                    manifest_id=manifest_id,
                    policy_id=policy_id,
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
                    if paths.output_path.is_symlink():
                        raise BookComposeError(
                            "output.write_failed", f"refusing symlink output: {paths.output_path}"
                        )
                    _ensure_output_directory(context.output_variant_dir)
                    composition = Composer().compose(job)
                    atomic_write_wav(
                        paths.output_path,
                        composition.audio,
                        composition.sample_rate,
                        clip_policy=job.output.clip_policy,
                    )
                    entry.update(_wav_result(paths.output_path))
                    entry["status"] = "written"
                    written += 1
            else:
                _ensure_output_directory(context.output_variant_dir)
                composition = Composer().compose(job)
                atomic_write_wav(
                    paths.output_path,
                    composition.audio,
                    composition.sample_rate,
                    clip_policy=job.output.clip_policy,
                )
                entry.update(_wav_result(paths.output_path))
                entry["status"] = "written"
                written += 1
            entry["output_policy_id"] = policy_id
            if context.save_audiojobs_dir is not None:
                if entry["status"] in {"written", "reused"}:
                    job.save(context.save_audiojobs_dir / f"{chapter.id}.audiojob")
                    entry["audiojob"] = f"{chapter.id}.audiojob"
        except Exception as exc:
            code, message = _chapter_error(exc)
            if entry.get("status") == "written":
                written -= 1
            elif entry.get("status") == "reused":
                reused -= 1
            entry.update(status="failed", error_code=code, error_message=message)
            failed += 1
            stop = fail_fast
        results.append(entry)
        item_by_id[chapter.id] = entry

    # Reports retain records for unselected chapters, ordered by the authoritative set.
    ordered_entries = [
        item_by_id[chapter.id]
        for chapter in context.renderplan_set.chapters
        if chapter.id in item_by_id
    ]
    report = {
        "variant": context.variant,
        "render_profile_id": context.renderplan_set.profile_id,
        "requested": len(context.chapters),
        "written": written,
        "reused": reused,
        "failed": failed,
        "skipped": skipped,
        "output": policy_dict,
        "item": ordered_entries,
    }
    report_error = None
    try:
        _write_report(context.report_path, report)
    except Exception as exc:
        report_error = {"code": "output.write_failed", "message": str(exc)}
    return {
        "format": "audiocompose-book-result",
        "variant": context.variant,
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
    "BookContext",
    "ChapterPaths",
    "VerifiedRenderHandoff",
    "load_chapter_plan",
    "parse_chapter_selector",
    "paths_for_chapter",
    "resolve_book",
]
