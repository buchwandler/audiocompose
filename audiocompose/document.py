"""Workspace-level orchestration for one rendered document bundle."""

from __future__ import annotations

import importlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

from .document_contract import (
    COMPOSE_REPORT_FILENAME,
    DOCUMENT_ID,
    DOCUMENT_RENDERPLAN_FILENAME,
    DOCUMENT_UTTERPLAN_FILENAME,
    PROFILE_SNAPSHOT_FILENAME,
    RENDER_MANIFEST_FILENAME,
    DocumentWorkspaceContractError,
    read_document_workspace,
)
from .errors import DocumentComposeError, HandoffValidationError, IntegrationError
from .integrations.renderplan_metadata import RENDER_DIAGNOSTICS_FILENAME
from .progress import CompositionProgressCallback

_DOCUMENT_REPORT_FORMAT = "audiocompose" + "-document-report"
_DOCUMENT_RESULT_FORMAT = "audiocompose" + "-document-result"


def _error(
    code: str,
    message: str,
    *,
    document_id: str | None = DOCUMENT_ID,
    variant: str | None = None,
) -> DocumentComposeError:
    return DocumentComposeError(code, message, document_id=document_id, variant=variant)


def _safe_path(path: Path, *, label: str, variant: str) -> Path:
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
            raise _error(
                "workspace.invalid", f"{label} is unavailable: {exc}", variant=variant
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise _error(
                "workspace.invalid", f"{label} contains a symlink: {current}", variant=variant
            )
    return Path(os.path.abspath(current))


def _existing_directory(
    path: Path,
    *,
    label: str,
    variant: str,
    code: str = "variant.not_found",
) -> Path:
    absolute = _safe_path(path, label=label, variant=variant)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise _error(code, f"{label} is unavailable: {absolute}: {exc}", variant=variant) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise _error(code, f"{label} is not a directory: {absolute}", variant=variant)
    return absolute.resolve(strict=True)


def _regular_file(path: Path, *, label: str, code: str, variant: str) -> Path:
    absolute = _safe_path(path, label=label, variant=variant)
    try:
        info = absolute.lstat()
    except OSError as exc:
        raise _error(code, f"{label} is unavailable: {absolute}: {exc}", variant=variant) from exc
    if not stat.S_ISREG(info.st_mode):
        raise _error(code, f"{label} is not a regular file: {absolute}", variant=variant)
    return absolute.resolve(strict=True)


def _variant_name(variant: str) -> None:
    if (
        not isinstance(variant, str)
        or not variant
        or variant in {".", ".."}
        or "/" in variant
        or "\\" in variant
        or "\x00" in variant
        or PureWindowsPath(variant).drive
    ):
        raise _error(
            "variant.invalid", "--variant must be a single directory basename", variant=variant
        )


@dataclass(frozen=True, slots=True)
class DocumentContext:
    """Resolved and identity-checked paths for a document composition."""

    workspace: Path
    document_id: str
    variant: str
    utterplan_path: Path
    renderplan_root: Path
    renderplan_variant_dir: Path
    profile_snapshot_path: Path
    profile: Any
    render_profile_id: str
    renderplan_path: Path
    plan: Any
    source_plan_id: str
    voicerender_root: Path
    voicerender_variant_dir: Path
    render_dir: Path
    manifest_path: Path
    diagnostics_path: Path
    output_root: Path
    output_variant_dir: Path
    output_path: Path
    audiojob_path: Path
    report_path: Path
    media_root: Path


def resolve_document(
    document: str | Path,
    variant: str,
    *,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
    media_root: str | Path | None = None,
    save_audiojob: str | Path | None = None,
) -> DocumentContext:
    """Resolve direct stage roots and validate the document/profile/plan identity chain."""
    _variant_name(variant)
    try:
        index = read_document_workspace(document)
    except DocumentWorkspaceContractError as exc:
        raise _error("workspace.invalid", str(exc), variant=variant) from exc

    workspace = index.root
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
    plan_root = _existing_directory(
        plan_root_arg, label="RenderPlan root", variant=variant, code="plan.missing"
    )
    voice_root = _existing_directory(
        voice_root_arg, label="VoiceRender root", variant=variant, code="handoff.invalid"
    )
    output_root_path = _safe_path(
        output_root_arg, label="AudioCompose output root", variant=variant
    )

    plan_variant_dir = _existing_directory(
        plan_root / variant, label="RenderPlan variant", variant=variant
    )
    voice_variant_dir = _existing_directory(
        voice_root / variant, label="VoiceRender variant", variant=variant
    )
    profile_path = _regular_file(
        plan_variant_dir / PROFILE_SNAPSHOT_FILENAME,
        label="RenderProfile snapshot",
        code="profile.invalid",
        variant=variant,
    )
    renderplan_path = _regular_file(
        plan_variant_dir / DOCUMENT_RENDERPLAN_FILENAME,
        label="document RenderPlan",
        code="plan.missing",
        variant=variant,
    )
    utterplan_path = _regular_file(
        workspace / "utterplan" / DOCUMENT_UTTERPLAN_FILENAME,
        label="UtterPlan source plan",
        code="source_plan.missing",
        variant=variant,
    )

    try:
        renderplan = importlib.import_module("renderplan")
        profile = renderplan.load_profile(profile_path)
        profile_id = profile.profile_id or profile.computed_id()
    except Exception as exc:
        raise _error(
            "profile.invalid", f"cannot load RenderProfile: {exc}", variant=variant
        ) from exc
    if not isinstance(profile_id, str) or not profile_id.startswith("sha256:"):
        raise _error(
            "profile.invalid", "RenderProfile has no valid profile identity", variant=variant
        )
    try:
        plan = renderplan.load_plan(renderplan_path)
    except Exception as exc:
        raise _error(
            "plan.missing", f"cannot load document RenderPlan: {exc}", variant=variant
        ) from exc
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    if not variant.endswith(f"--{fingerprint}"):
        raise _error(
            "profile.identity_mismatch",
            f"variant name does not end with the RenderProfile fingerprint {fingerprint!r}",
            variant=variant,
        )
    if getattr(plan, "render_profile_id", None) != profile_id:
        raise _error(
            "plan.identity_mismatch",
            "RenderPlan profile identity does not match the selected RenderProfile snapshot",
            variant=variant,
        )

    try:
        from .integrations.voicerender import load_source_plan

        source_plan = load_source_plan(utterplan_path)
    except HandoffValidationError as exc:
        raise _error("handoff.invalid", str(exc), variant=variant) from exc
    except IntegrationError as exc:
        raise _error(
            "dependency.api_incompatible",
            str(exc),
            variant=variant,
        ) from exc
    if getattr(source_plan, "plan_id", None) != getattr(plan, "source_plan_id", None):
        raise _error(
            "plan.identity_mismatch",
            "UtterPlan source identity does not match RenderPlan source_plan_id",
            variant=variant,
        )

    render_dir = _existing_directory(
        voice_variant_dir / index.document_id,
        label="VoiceRender document directory",
        variant=variant,
    )
    manifest_path = _regular_file(
        render_dir / RENDER_MANIFEST_FILENAME,
        label="VoiceRender RenderManifest",
        code="manifest.missing",
        variant=variant,
    )
    diagnostics_path = _regular_file(
        render_dir / RENDER_DIAGNOSTICS_FILENAME,
        label="VoiceRender diagnostics",
        code="diagnostics.missing",
        variant=variant,
    )

    output_variant_dir = output_root_path / variant
    output_path = _safe_path(
        output_variant_dir / f"{index.document_id}.wav",
        label="AudioCompose output",
        variant=variant,
    )
    if not output_path.is_relative_to(output_variant_dir):
        raise _error(
            "output.write_failed", "output escapes AudioCompose variant directory", variant=variant
        )
    default_audiojob = output_variant_dir / f"{index.document_id}.audiojob"
    audiojob_path = _safe_path(
        Path(save_audiojob).expanduser() if save_audiojob is not None else default_audiojob,
        label="AudioJob path",
        variant=variant,
    )
    report_path = _safe_path(
        output_variant_dir / COMPOSE_REPORT_FILENAME, label="compose report", variant=variant
    )
    chosen_media_root = (
        Path(media_root).expanduser().absolute() if media_root is not None else plan_variant_dir
    )
    return DocumentContext(
        workspace=workspace,
        document_id=index.document_id,
        variant=variant,
        utterplan_path=utterplan_path,
        renderplan_root=plan_root,
        renderplan_variant_dir=plan_variant_dir,
        profile_snapshot_path=profile_path,
        profile=profile,
        render_profile_id=profile_id,
        renderplan_path=renderplan_path,
        plan=plan,
        source_plan_id=source_plan.plan_id,
        voicerender_root=voice_root,
        voicerender_variant_dir=voice_variant_dir,
        render_dir=render_dir,
        manifest_path=manifest_path,
        diagnostics_path=diagnostics_path,
        output_root=output_root_path,
        output_variant_dir=output_variant_dir,
        output_path=output_path,
        audiojob_path=audiojob_path,
        report_path=report_path,
        media_root=chosen_media_root,
    )


__all__ = ["DocumentContext", "resolve_document"]


@dataclass(frozen=True, slots=True)
class DocumentComposeOptions:
    """Optional output and loudness overrides for :func:`compose_document`."""

    sample_rate: int | None = None
    channels: int | None = None
    clip_policy: Any | None = None
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = None
    peak_policy: Any | None = None
    collect_metrics: bool | None = None


@dataclass(frozen=True, slots=True)
class DocumentComposeRequest:
    """Immutable normalized request and verified workspace identities."""

    context: DocumentContext
    output_policy: Any
    pause_policy: Any
    pause_policy_provenance: Literal["renderprofile", "explicit"]
    force: bool = False
    emphasis_policy: Literal["segment_fallback", "strict"] = "segment_fallback"

    @property
    def workspace(self) -> Path:
        return self.context.workspace

    @property
    def document_id(self) -> str:
        return self.context.document_id

    @property
    def variant(self) -> str:
        return self.context.variant

    @property
    def utterplan_path(self) -> Path:
        return self.context.utterplan_path

    @property
    def renderplan_variant_dir(self) -> Path:
        return self.context.renderplan_variant_dir

    @property
    def profile_snapshot_path(self) -> Path:
        return self.context.profile_snapshot_path

    @property
    def renderplan_path(self) -> Path:
        return self.context.renderplan_path

    @property
    def voicerender_variant_dir(self) -> Path:
        return self.context.voicerender_variant_dir

    @property
    def render_dir(self) -> Path:
        return self.context.render_dir

    @property
    def manifest_path(self) -> Path:
        return self.context.manifest_path

    @property
    def diagnostics_path(self) -> Path:
        return self.context.diagnostics_path

    @property
    def output_variant_dir(self) -> Path:
        return self.context.output_variant_dir

    @property
    def output_path(self) -> Path:
        return self.context.output_path

    @property
    def audiojob_path(self) -> Path:
        return self.context.audiojob_path

    @property
    def report_path(self) -> Path:
        return self.context.report_path

    def __post_init__(self) -> None:
        if self.pause_policy_provenance not in {"renderprofile", "explicit"}:
            raise ValueError("pause_policy_provenance must be 'renderprofile' or 'explicit'")
        if self.emphasis_policy not in {"segment_fallback", "strict"}:
            raise ValueError("emphasis_policy must be 'segment_fallback' or 'strict'")


@dataclass(frozen=True, slots=True)
class DocumentComposeResult:
    """Typed result for one successful document composition or reuse."""

    request: DocumentComposeRequest
    render_plan_id: str
    manifest_id: str
    output: Path
    audiojob: Path
    report_path: Path
    status: str
    sha256: str
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    diagnostics: tuple[dict[str, Any], ...] = ()

    @property
    def workspace(self) -> Path:
        return self.request.context.workspace

    @property
    def document_id(self) -> str:
        return self.request.context.document_id

    @property
    def variant(self) -> str:
        return self.request.context.variant

    @property
    def render_profile_id(self) -> str:
        return self.request.context.render_profile_id

    @property
    def source_plan_id(self) -> str:
        return self.request.context.source_plan_id

    @property
    def exit_code(self) -> int:
        return 0

    @property
    def success(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        context = self.request.context
        return {
            "format": _DOCUMENT_RESULT_FORMAT,
            "workspace": str(self.workspace),
            "document_id": self.document_id,
            "variant": self.variant,
            "render_profile_id": self.render_profile_id,
            "source_plan_id": self.source_plan_id,
            "render_plan_id": self.render_plan_id,
            "manifest_id": self.manifest_id,
            "renderplan_root": str(context.renderplan_variant_dir),
            "voicerender_root": str(context.voicerender_variant_dir),
            "output_root": str(context.output_variant_dir),
            "output": str(self.output),
            "audiojob": str(self.audiojob),
            "report": str(self.report_path),
            "status": self.status,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration_seconds": self.duration_seconds,
            "sha256": self.sha256,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


def default_document_output_policy(
    options: DocumentComposeOptions | None = None,
) -> Any:
    """Build the document policy using the same defaults as book composition."""
    from ._workspace_policy import build_workspace_output_policy

    return build_workspace_output_policy(options)


def _document_policy_dict(policy: Any) -> dict[str, Any]:
    loudness = policy.loudness
    return {
        "sample_rate": policy.sample_rate,
        "channels": policy.channels,
        "clip_policy": policy.clip_policy,
        "target_lufs": loudness.target_lufs,
        "true_peak_ceiling_dbtp": loudness.true_peak_ceiling_dbtp,
        "peak_policy": loudness.peak_policy,
        "collect_metrics": loudness.collect_metrics,
    }


def _document_policy_id(value: Any) -> str:
    import hashlib
    import json

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _document_toml_scalar(value: Any) -> str:
    import json

    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"unsupported report scalar {type(value).__name__}")


def _document_report_text(report: dict[str, Any]) -> str:
    lines = [
        f'format = "{_DOCUMENT_REPORT_FORMAT}"',
        "version = 1",
    ]
    for key in (
        "document_id",
        "variant",
        "render_profile_id",
        "source_plan_id",
        "render_plan_id",
        "render_manifest_id",
        "render_manifest_sha256",
        "render_diagnostics_sha256",
        "pause_policy_id",
        "pause_policy_provenance",
        "output_policy_id",
        "audiojob_id",
        "audiojob_schema_version",
        "audiocompose_version",
        "status",
        "workspace",
        "utterplan",
        "renderplan",
        "render_manifest",
        "render_diagnostics",
        "audiojob",
        "output",
    ):
        lines.append(f"{key} = {_document_toml_scalar(report[key])}")
    lines.extend(("", "[output_geometry]"))
    for key in ("sha256", "sample_rate", "channels", "frames", "duration_seconds"):
        lines.append(f"{key} = {_document_toml_scalar(report[key])}")
    return "\n".join(lines) + "\n"


def _write_document_report(path: Path, report: dict[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".compose-report-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_document_report_text(report))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _report_relative(path: Path, report_path: Path) -> str:
    import os

    absolute = Path(os.path.abspath(path))
    relative = Path(os.path.relpath(absolute, report_path.parent)).as_posix()
    if Path(os.path.abspath(report_path.parent / relative)) != absolute:
        raise _error("workspace.invalid", f"cannot safely reference report path {path}")
    return relative


def _read_document_report(path: Path) -> dict[str, Any] | None:
    import importlib
    import stat

    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        try:
            tomllib = importlib.import_module("tomllib")
        except ModuleNotFoundError:
            tomllib = importlib.import_module("tomli")
        with path.open("rb") as stream:
            report = tomllib.load(stream)
    except (ImportError, OSError, ValueError, TypeError):
        return None
    if report.get("format") != _DOCUMENT_REPORT_FORMAT or report.get("version") != 1:
        return None
    return report


def _document_wav_data(path: Path) -> dict[str, Any] | None:
    from .wav import sha256_file, wav_info

    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        geometry = wav_info(path)
        return {
            "sha256": "sha256:" + sha256_file(path),
            "sample_rate": geometry.sample_rate,
            "channels": geometry.channels,
            "frames": geometry.frames,
            "duration_seconds": geometry.frames / geometry.sample_rate,
        }
    except (OSError, ValueError, ZeroDivisionError):
        return None


def _document_job_is_prior(path: Path, report: dict[str, Any], job_id: str, schema: int) -> bool:
    from .job import load_job

    try:
        persisted = load_job(path)
    except Exception:
        return False
    return (
        persisted.job_id == job_id
        and persisted.schema_version == schema
        and report.get("audiojob_id") == job_id
        and report.get("audiojob_schema_version") == schema
    )


def compose_document(
    document: str | Path,
    *,
    variant: str,
    pause_policy: Any = None,
    output_policy: Any = None,
    options: DocumentComposeOptions | None = None,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
    media_root: str | Path | None = None,
    save_audiojob: str | Path | None = None,
    force: bool = False,
    emphasis_policy: Literal["segment_fallback", "strict"] = "segment_fallback",
    on_progress: CompositionProgressCallback | None = None,
) -> DocumentComposeResult:
    """Verify and compose one rendered SSMD document bundle using its selected variant."""
    from .composer import Composer
    from .errors import MediaResolutionError, PausePolicyError
    from .integrations.voicerender import (
        PausePolicy,
        load_pause_policy,
        materialize_voicerender_job,
    )
    from .job import load_job
    from .model import OutputPolicy
    from .wav import atomic_write_wav

    if output_policy is not None and options is not None:
        raise ValueError("pass either output_policy or options, not both")
    if output_policy is not None and not isinstance(output_policy, OutputPolicy):
        raise TypeError("output_policy must be an AudioCompose OutputPolicy")
    if emphasis_policy not in {"segment_fallback", "strict"}:
        raise _error(
            "handoff.invalid",
            "emphasis_policy must be 'segment_fallback' or 'strict'",
            variant=variant,
        )
    context = resolve_document(
        document,
        variant,
        renderplan_root=renderplan_root,
        voicerender_root=voicerender_root,
        output_root=output_root,
        media_root=media_root,
        save_audiojob=save_audiojob,
    )
    policy = output_policy or default_document_output_policy(options)
    pause_provenance: Literal["renderprofile", "explicit"]
    try:
        if pause_policy is None:
            selected_pause_policy = PausePolicy.from_render_profile(context.profile)
            pause_provenance = "renderprofile"
        else:
            selected_pause_policy = load_pause_policy(pause_policy)
            pause_provenance = "explicit"
    except PausePolicyError as exc:
        raise _error("pause_policy.invalid", str(exc), variant=variant) from exc
    request = DocumentComposeRequest(
        context=context,
        output_policy=policy,
        pause_policy=selected_pause_policy,
        pause_policy_provenance=pause_provenance,
        force=force,
        emphasis_policy=emphasis_policy,
    )

    try:
        job = materialize_voicerender_job(
            context.utterplan_path,
            context.plan,
            context.manifest_path,
            render_dir=context.render_dir,
            output_policy=policy,
            pause_policy=selected_pause_policy,
            pause_policy_provenance=pause_provenance,
            default_role=getattr(context.profile, "default_role", ""),
            media_root=context.media_root,
            emphasis_policy=emphasis_policy,
        )
    except PausePolicyError as exc:
        raise _error("pause_policy.invalid", str(exc), variant=variant) from exc
    except MediaResolutionError as exc:
        raise _error("handoff.invalid", str(exc), variant=variant) from exc
    except HandoffValidationError as exc:
        raise _error("handoff.invalid", str(exc), variant=variant) from exc
    except IntegrationError as exc:
        raise _error("dependency.api_incompatible", str(exc), variant=variant) from exc

    from ._version import __version__ as audiocompose_version
    from .book import _emphasis_diagnostics
    from .integrations.renderplan_metadata import VOICERENDER_DIAGNOSTICS_SHA256
    from .wav import sha256_file

    manifest_sha256 = "sha256:" + sha256_file(context.manifest_path)
    diagnostics_sha256 = str(job.source[VOICERENDER_DIAGNOSTICS_SHA256])
    manifest_id = str(job.source["render_manifest.manifest_id"])
    policy_dict = _document_policy_dict(policy)
    policy_id = _document_policy_id(policy_dict)

    try:
        context.audiojob_path.parent.mkdir(parents=True, exist_ok=True)
        if context.audiojob_path.is_symlink():
            raise OSError(f"refusing symlink AudioJob bundle: {context.audiojob_path}")
        job.save(context.audiojob_path)
        persisted_job = load_job(context.audiojob_path)
    except Exception as exc:
        raise _error(
            "output.write_failed", f"cannot save or verify AudioJob: {exc}", variant=variant
        ) from exc
    job_id = persisted_job.job_id
    schema_version = persisted_job.schema_version
    if not isinstance(job_id, str) or not job_id:
        raise _error(
            "output.write_failed", "persisted AudioJob has no stable identity", variant=variant
        )

    prior = _read_document_report(context.report_path)
    prior_output = _document_wav_data(context.output_path)
    current_identity = {
        "document_id": context.document_id,
        "variant": context.variant,
        "source_plan_id": context.source_plan_id,
        "render_profile_id": context.render_profile_id,
        "render_plan_id": context.plan.plan_id or context.plan.computed_id(),
        "render_manifest_id": manifest_id,
        "render_manifest_sha256": manifest_sha256,
        "render_diagnostics_sha256": diagnostics_sha256,
        "pause_policy_id": selected_pause_policy.policy_id,
        "pause_policy_provenance": pause_provenance,
        "output_policy_id": policy_id,
        "audiojob_id": job_id,
        "audiojob_schema_version": schema_version,
        "audiocompose_version": audiocompose_version,
        "output": context.output_path.name,
    }
    prior_geometry = prior.get("output_geometry") if prior is not None else None
    prior_matches = (
        prior is not None
        and prior_output is not None
        and isinstance(prior_geometry, dict)
        and all(prior.get(key) == value for key, value in current_identity.items())
        and all(prior_geometry.get(key) == value for key, value in prior_output.items())
        and prior_output["sample_rate"] == policy.sample_rate
        and prior_output["channels"] == policy.channels
        and _document_job_is_prior(context.audiojob_path, prior, job_id, schema_version)
    )
    output_exists = context.output_path.exists() or context.output_path.is_symlink()
    if context.output_path.is_symlink():
        raise _error(
            "output.write_failed",
            f"refusing symlink output: {context.output_path}",
            variant=variant,
        )
    if output_exists and not force:
        if not prior_matches:
            raise _error(
                "output.exists",
                f"existing document output has no matching verified provenance: {context.output_path}; use --force",
                variant=variant,
            )
        status = "reused"
        output_data = prior_output
    else:
        try:
            context.output_variant_dir.mkdir(parents=True, exist_ok=True)
            if context.output_variant_dir.is_symlink():
                raise OSError(f"refusing symlink output directory: {context.output_variant_dir}")
            composition = Composer().compose(job, on_progress=on_progress)
            atomic_write_wav(
                context.output_path,
                composition.audio,
                composition.sample_rate,
                clip_policy=job.output.clip_policy,
            )
        except Exception as exc:
            raise _error(
                "output.write_failed", f"cannot write document WAV: {exc}", variant=variant
            ) from exc
        output_data = _document_wav_data(context.output_path)
        if output_data is None:
            raise _error(
                "output.write_failed", "written document WAV failed verification", variant=variant
            )
        status = "written"

    assert output_data is not None
    diagnostics = tuple(_emphasis_diagnostics(job))
    report = {
        **current_identity,
        "status": status,
        "workspace": _report_relative(context.workspace, context.report_path),
        "utterplan": _report_relative(context.utterplan_path, context.report_path),
        "renderplan": _report_relative(context.renderplan_path, context.report_path),
        "render_manifest": _report_relative(context.manifest_path, context.report_path),
        "render_diagnostics": _report_relative(context.diagnostics_path, context.report_path),
        "audiojob": _report_relative(context.audiojob_path, context.report_path),
        **output_data,
    }
    try:
        _write_document_report(context.report_path, report)
    except Exception as exc:
        raise _error(
            "output.write_failed", f"cannot write compose report: {exc}", variant=variant
        ) from exc

    return DocumentComposeResult(
        request=request,
        render_plan_id=current_identity["render_plan_id"],
        manifest_id=manifest_id,
        output=context.output_path,
        audiojob=context.audiojob_path,
        report_path=context.report_path,
        status=status,
        sha256=output_data["sha256"],
        sample_rate=output_data["sample_rate"],
        channels=output_data["channels"],
        frames=output_data["frames"],
        duration_seconds=output_data["duration_seconds"],
        diagnostics=diagnostics,
    )


__all__ = [
    "DocumentComposeOptions",
    "DocumentComposeRequest",
    "DocumentComposeResult",
    "DocumentContext",
    "compose_document",
    "default_document_output_policy",
    "resolve_document",
]
