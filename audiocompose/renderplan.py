"""Public single-document composition API for shared RenderPlan handoffs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

from .composer import Composer
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .integrations.renderplan import load_and_materialize_renderplan_job
from .model import AudioJob, CompositionResult, OutputPolicy
from .wav import atomic_write_wav

EmphasisPolicy = Literal["segment_fallback", "strict"]


@dataclass(frozen=True, slots=True)
class RenderPlanComposeRequest:
    """Paths and policies for composing one RenderPlan handoff."""

    plan: Path
    render_dir: Path
    output: Path
    manifest: Path | None = None
    media_root: Path | None = None
    output_policy: OutputPolicy | None = None
    save_audiojob: Path | None = None
    emphasis_policy: EmphasisPolicy = "segment_fallback"

    def __post_init__(self) -> None:
        for name in ("plan", "render_dir", "output"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser())
        for name in ("manifest", "media_root", "save_audiojob"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())


@dataclass(frozen=True, slots=True)
class RenderPlanComposeResult:
    """Typed outcome and provenance for a single RenderPlan composition."""

    output: Path
    plan_id: str
    manifest_id: str
    sample_rate: int
    frames: int
    duration_seconds: float
    timeline: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[CompositionDiagnostic, ...]
    audiojob: Path | None = None


def _timeline(job: AudioJob, composition: CompositionResult) -> tuple[Mapping[str, Any], ...]:
    entries: list[Mapping[str, Any]] = []
    for item, composed in zip(job.items, composition.items, strict=True):
        metadata = dict(item.metadata)
        kind = str(metadata.get("kind", composed.kind))
        entry: dict[str, Any] = {
            "id": item.id,
            "kind": kind,
            "start_sample": composed.start_sample,
            "end_sample": composed.end_sample,
            "duration_seconds": composed.duration_samples / composition.sample_rate,
        }
        for key in (
            "renderplan.segment_number",
            "renderplan.segment_id",
            "renderplan.source",
            "renderplan.pause_id",
            "renderplan.pause_reason",
            "renderplan.pause_source",
            "renderplan.audio_id",
            "renderplan.audio.src",
            "renderplan.audio.options",
            "render_manifest.wav",
            "render_manifest.wav_sha256",
            "renderplan.emphasis",
        ):
            if key in metadata:
                entry[key] = metadata[key]
        entries.append(entry)
    return tuple(entries)


def _emphasis_diagnostics(job: AudioJob) -> tuple[CompositionDiagnostic, ...]:
    diagnostics: list[CompositionDiagnostic] = []
    for item in job.items:
        report = item.metadata.get("renderplan.emphasis")
        if not isinstance(report, dict) or report.get("scope") != "segment_fallback":
            continue
        segment_number = item.metadata.get("renderplan.segment_number")
        ranges = report.get("partial_ranges", [])
        gain_db = report.get("gain_db")
        diagnostics.append(
            CompositionDiagnostic(
                code="emphasis.segment_fallback",
                severity=DiagnosticSeverity.WARNING,
                item_id=item.id,
                context={
                    "segment_number": segment_number,
                    "character_ranges": ranges,
                    "gain_db": gain_db,
                },
                message=(
                    f"speech {segment_number}: applied +{gain_db} dB to the entire segment "
                    f"for character range(s) {ranges}; no trusted word-level sample alignment is available"
                ),
            )
        )
    return tuple(diagnostics)


@overload
def compose_renderplan(plan: RenderPlanComposeRequest) -> RenderPlanComposeResult: ...


@overload
def compose_renderplan(
    plan: str | Path,
    *,
    render_dir: str | Path,
    output: str | Path,
    manifest: str | Path | None = None,
    media_root: str | Path | None = None,
    output_policy: OutputPolicy | None = None,
    save_audiojob: str | Path | None = None,
    emphasis_policy: EmphasisPolicy = "segment_fallback",
) -> RenderPlanComposeResult: ...


def compose_renderplan(
    plan: str | Path | RenderPlanComposeRequest,
    *,
    render_dir: str | Path | None = None,
    output: str | Path | None = None,
    manifest: str | Path | None = None,
    media_root: str | Path | None = None,
    output_policy: OutputPolicy | None = None,
    save_audiojob: str | Path | None = None,
    emphasis_policy: EmphasisPolicy | None = None,
) -> RenderPlanComposeResult:
    """Verify and compose a RenderPlan handoff into one atomically written WAV.

    When ``plan`` is a path and ``media_root`` is omitted, relative external
    AudioItems are resolved from the plan file's parent directory. The manifest
    defaults to ``render-manifest.toml`` inside ``render_dir``.
    """
    if isinstance(plan, RenderPlanComposeRequest):
        if any(
            value is not None
            for value in (
                render_dir,
                output,
                manifest,
                media_root,
                output_policy,
                save_audiojob,
                emphasis_policy,
            )
        ):
            raise TypeError("Pass either a RenderPlanComposeRequest or path arguments, not both")
        request = plan
    else:
        if render_dir is None or output is None:
            raise TypeError("render_dir and output are required when plan is a path")
        request = RenderPlanComposeRequest(
            plan=Path(plan),
            render_dir=Path(render_dir),
            output=Path(output),
            manifest=Path(manifest) if manifest is not None else None,
            media_root=Path(media_root) if media_root is not None else None,
            output_policy=output_policy,
            save_audiojob=Path(save_audiojob) if save_audiojob is not None else None,
            emphasis_policy=(
                emphasis_policy if emphasis_policy is not None else "segment_fallback"
            ),
        )

    resolved_media_root = request.media_root
    if resolved_media_root is None:
        resolved_media_root = request.plan.absolute().parent

    job = load_and_materialize_renderplan_job(
        request.plan,
        request.manifest,
        render_dir=request.render_dir,
        output_policy=request.output_policy,
        media_root=resolved_media_root,
        emphasis_policy=request.emphasis_policy,
    )
    audiojob_path = job.save(request.save_audiojob) if request.save_audiojob is not None else None
    composition = Composer().compose(job)
    output_path = atomic_write_wav(
        request.output,
        composition.audio,
        composition.sample_rate,
        clip_policy=job.output.clip_policy,
    )
    return RenderPlanComposeResult(
        output=output_path,
        plan_id=str(job.source["renderplan.plan_id"]),
        manifest_id=str(job.source["render_manifest.manifest_id"]),
        sample_rate=composition.sample_rate,
        frames=len(composition.audio),
        duration_seconds=composition.duration_seconds,
        timeline=_timeline(job, composition),
        diagnostics=_emphasis_diagnostics(job),
        audiojob=Path(audiojob_path) if audiojob_path is not None else None,
    )


__all__ = [
    "RenderPlanComposeRequest",
    "RenderPlanComposeResult",
    "compose_renderplan",
]
