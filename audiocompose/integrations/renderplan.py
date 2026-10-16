"""Materialize verified shared RenderPlan contracts as generic AudioCompose jobs."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from renderplan import AudioItem, PauseItem, RenderManifest, RenderPlan, SpeechItem

from .._version import __version__
from ..errors import HandoffValidationError, MediaResolutionError
from ..model import AudioClip, AudioJob, OutputPolicy, Silence
from ..operations import Gain
from ..sources import AudioFileSource
from ..wav import read_wav, sha256_file, wav_info
from .media import (
    LocalWavResolver,
    MediaResolver,
    ResolvedMedia,
    apply_renderplan_media_options,
    renderplan_media_request,
)
from .renderplan_metadata import (
    AUDIOCOMPOSE_MATERIALIZER,
    AUDIOCOMPOSE_MEDIA_IDENTITY,
    AUDIOCOMPOSE_MEDIA_METADATA,
    MANIFEST_CHANNELS,
    MANIFEST_FRAMES,
    MANIFEST_ID,
    MANIFEST_SAMPLE_RATE,
    MANIFEST_WAV,
    MANIFEST_WAV_SHA256,
    RENDERPLAN_AUDIO_ID,
    RENDERPLAN_AUDIO_OPTIONS,
    RENDERPLAN_AUDIO_SOURCE,
    RENDERPLAN_EMPHASIS,
    RENDERPLAN_PAUSE_ID,
    RENDERPLAN_PAUSE_REASON,
    RENDERPLAN_PAUSE_SOURCE,
    RENDERPLAN_PLAN_ID,
    RENDERPLAN_PROFILE_ID,
    RENDERPLAN_SEGMENT_ID,
    RENDERPLAN_SEGMENT_NUMBER,
    RENDERPLAN_SOURCE,
    RENDERPLAN_SOURCE_PLAN_ID,
    VOICERENDER_DEFERRED_CAPABILITIES,
    VOICERENDER_DIAGNOSTICS_SCHEMA,
    VOICERENDER_DIAGNOSTICS_SHA256,
    VOICERENDER_EMPHASIS_POLICY,
    VOICERENDER_REALIZED_CONTROLS,
)

MANIFEST_FILENAME = "render-manifest.toml"


@dataclass(frozen=True, slots=True)
class _ValidatedControlOwnership:
    """Private provenance proof for controls already realized by a trusted producer."""

    producer: str
    plan_id: str
    realized_controls: frozenset[str]
    diagnostics_sha256: str
    diagnostics_schema: str
    emphasis_policy: str
    deferred_capabilities: tuple[str, ...]


def _validate_contract(plan: RenderPlan, manifest: RenderManifest) -> str:
    if not isinstance(plan, RenderPlan):
        raise HandoffValidationError("plan must be a renderplan.RenderPlan")
    if not isinstance(manifest, RenderManifest):
        raise HandoffValidationError("manifest must be a renderplan.RenderManifest")

    plan_id = plan.computed_id()
    if plan.plan_id is not None and plan.plan_id != plan_id:
        raise HandoffValidationError("RenderPlan plan_id does not match its content")
    manifest_id = manifest.computed_id()
    if manifest.manifest_id is not None and manifest.manifest_id != manifest_id:
        raise HandoffValidationError("RenderManifest manifest_id does not match its content")
    if manifest.complete is not True:
        raise HandoffValidationError(
            "RenderManifest is incomplete; final composition requires complete=true"
        )
    try:
        manifest.validate_against(plan)
    except Exception as exc:
        raise HandoffValidationError(f"RenderManifest does not match RenderPlan: {exc}") from exc
    return plan_id


def _safe_directory(path: str | Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    try:
        absolute = Path(os.path.abspath(raw))
        current = Path(absolute.anchor)
        for part in absolute.parts[1:]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise HandoffValidationError(f"{label} contains a symlink: {current}")
            if not stat.S_ISDIR(info.st_mode):
                raise HandoffValidationError(f"{label} is not a directory: {current}")
        resolved = absolute.resolve(strict=True)
    except HandoffValidationError:
        raise
    except OSError as exc:
        raise HandoffValidationError(f"{label} is unavailable: {exc}") from exc
    if not resolved.is_dir():
        raise HandoffValidationError(f"{label} is not a directory: {resolved}")
    return resolved


def _resolve_render_wav(render_root: Path, wav: str) -> Path:
    if not isinstance(wav, str) or not wav or "\\" in wav or "\x00" in wav:
        raise HandoffValidationError("RenderManifest WAV name must be a safe relative POSIX path")
    relative = PurePosixPath(wav)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise HandoffValidationError("RenderManifest WAV path must not escape render_dir")
    candidate = render_root
    try:
        for part in relative.parts:
            candidate = candidate / part
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise HandoffValidationError(f"Render WAV path contains a symlink: {wav}")
        resolved = candidate.resolve(strict=True)
        info = candidate.stat()
    except HandoffValidationError:
        raise
    except OSError as exc:
        raise HandoffValidationError(f"Render WAV is unavailable: {wav}: {exc}") from exc
    if not resolved.is_relative_to(render_root) or not stat.S_ISREG(info.st_mode):
        raise HandoffValidationError(f"Render WAV is not a regular file beneath render_dir: {wav}")
    return resolved


def _verify_speech_artifacts(
    plan: RenderPlan, manifest: RenderManifest, render_root: Path
) -> dict[int, tuple[Any, Path, str]]:
    artifacts = {artifact.segment_number: artifact for artifact in manifest.speech}
    if len(artifacts) != len(manifest.speech):
        raise HandoffValidationError("RenderManifest contains duplicate speech segment numbers")
    speeches = {item.number: item for item in plan.flow if isinstance(item, SpeechItem)}
    if set(artifacts) != set(speeches):
        missing = sorted(set(speeches).difference(artifacts))
        extra = sorted(set(artifacts).difference(speeches))
        raise HandoffValidationError(
            f"RenderManifest speech occurrences differ from RenderPlan (missing={missing}, extra={extra})"
        )

    verified: dict[int, tuple[Any, Path, str]] = {}

    for number, item in speeches.items():
        artifact = artifacts[number]
        expected_segment_id = item.segment_id or item.computed_id()
        if artifact.segment_id != expected_segment_id:
            raise HandoffValidationError(
                f"speech {number}: manifest segment_id does not match RenderPlan"
            )
        if artifact.wav != item.wav:
            raise HandoffValidationError(
                f"speech {number}: manifest WAV name does not match RenderPlan"
            )
        path = _resolve_render_wav(render_root, artifact.wav)
        try:
            actual_sha256 = sha256_file(path)
            if artifact.sha256 != f"sha256:{actual_sha256}":
                raise HandoffValidationError(f"speech {number}: WAV SHA-256 mismatch")
            info = wav_info(path)
            audio, sample_rate = read_wav(path, expected_channels=None)
        except HandoffValidationError:
            raise
        except Exception as exc:
            raise HandoffValidationError(f"speech {number}: invalid supported WAV: {exc}") from exc
        if info.channels != artifact.channels:
            raise HandoffValidationError(
                f"speech {number}: WAV channel mismatch (manifest {artifact.channels}, file {info.channels})"
            )
        if info.sample_rate != artifact.sample_rate or sample_rate != artifact.sample_rate:
            raise HandoffValidationError(
                f"speech {number}: WAV sample-rate mismatch "
                f"(manifest {artifact.sample_rate}, file {info.sample_rate})"
            )
        if info.frames != artifact.frames or audio.size != artifact.frames:
            raise HandoffValidationError(
                f"speech {number}: WAV frame mismatch (manifest {artifact.frames}, file {info.frames})"
            )
        if info.channels != 1:
            raise HandoffValidationError(
                f"speech {number}: AudioCompose supports mono WAV artifacts only"
            )
        verified[number] = (artifact, path, actual_sha256)
    return verified


def _source_metadata(item: SpeechItem) -> list[dict[str, Any]]:
    return [span.to_dict() for span in item.source]


_EMPHASIS_GAIN_DB = {"reduced": 0.75, "moderate": 1.5, "strong": 2.5}


def _speech_emphasis(
    item: SpeechItem,
    policy: Literal["segment_fallback", "strict"],
    *,
    control_ownership: _ValidatedControlOwnership | None = None,
) -> tuple[Gain | None, dict[str, Any] | None]:
    """Validate controls and return the bounded AudioCompose emphasis effect."""
    if policy not in ("segment_fallback", "strict"):
        raise HandoffValidationError("emphasis_policy must be 'segment_fallback' or 'strict'")

    controls = item.controls
    if not isinstance(controls, (tuple, list)):
        raise HandoffValidationError(
            f"speech {item.number} control range [unknown]: controls must be a sequence"
        )
    if not controls:
        return None, None

    normalized: list[dict[str, Any]] = []
    for control in controls:
        start = getattr(control, "start", None)
        end = getattr(control, "end", None)
        range_label = f"[{start!r}, {end!r})"
        context = f"speech {item.number} control range {range_label}"
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(item.text)
        ):
            raise HandoffValidationError(
                f"{context}: expected character offsets satisfying 0 <= start < end <= {len(item.text)}"
            )
        name = getattr(control, "name", None)
        if name != "emphasis":
            if control_ownership is not None and name in control_ownership.realized_controls:
                continue
            raise HandoffValidationError(
                f"{context}: unsupported speech control {name!r}; AudioCompose supports 'emphasis' only"
            )
        params = getattr(control, "params", None)
        if not isinstance(params, Mapping):
            raise HandoffValidationError(f"{context}: emphasis params must be a mapping")
        unknown = sorted(set(params).difference({"level"}))
        if unknown:
            raise HandoffValidationError(
                f"{context}: unsupported emphasis parameter(s): {', '.join(map(str, unknown))}"
            )
        level = params.get("level", "moderate")
        if not isinstance(level, str) or level not in _EMPHASIS_GAIN_DB:
            raise HandoffValidationError(
                f"{context}: unsupported emphasis level {level!r}; choose reduced, moderate, or strong"
            )
        normalized.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "params": dict(params),
                "level": level,
            }
        )

    if not normalized:
        return None, None
    partial = [
        control
        for control in normalized
        if (control["start"], control["end"]) != (0, len(item.text))
    ]
    if partial and policy == "strict":
        ranges = ", ".join(f"[{c['start']}, {c['end']})" for c in partial)
        raise HandoffValidationError(
            f"speech {item.number} emphasis range(s) {ranges}: partial-span emphasis has no trusted sample alignment; "
            "use segment_fallback or split the speech upstream"
        )

    selected = max(normalized, key=lambda control: _EMPHASIS_GAIN_DB[control["level"]])
    scope = "segment_fallback" if partial else "full_span"
    gain_db = _EMPHASIS_GAIN_DB[selected["level"]]
    report = {
        "policy_version": 1,
        "policy": policy,
        "scope": scope,
        "approximation": bool(partial),
        "controls": normalized,
        "partial_ranges": [[control["start"], control["end"]] for control in partial],
        "selected_level": selected["level"],
        "gain_db": gain_db,
    }
    return Gain(gain_db), report


def _materialize_speech(
    item: SpeechItem,
    plan_id: str,
    manifest: RenderManifest,
    verified: dict[int, tuple[Any, Path, str]],
    emphasis_policy: Literal["segment_fallback", "strict"],
    control_ownership: _ValidatedControlOwnership | None = None,
) -> AudioClip:
    artifact, path, digest = verified[item.number]
    segment_id = item.segment_id or item.computed_id()
    gain, emphasis_report = _speech_emphasis(
        item, emphasis_policy, control_ownership=control_ownership
    )
    metadata: dict[str, Any] = {
        "kind": "speech",
        RENDERPLAN_PLAN_ID: plan_id,
        RENDERPLAN_SEGMENT_NUMBER: item.number,
        RENDERPLAN_SEGMENT_ID: segment_id,
        RENDERPLAN_SOURCE: _source_metadata(item),
        MANIFEST_ID: manifest.manifest_id or manifest.computed_id(),
        MANIFEST_WAV_SHA256: artifact.sha256,
        MANIFEST_WAV: artifact.wav,
        MANIFEST_SAMPLE_RATE: artifact.sample_rate,
        MANIFEST_CHANNELS: artifact.channels,
        MANIFEST_FRAMES: artifact.frames,
    }
    if control_ownership is not None:
        realized = [
            {"name": control.name, "start": control.start, "end": control.end}
            for control in item.controls
            if control.name in control_ownership.realized_controls
        ]
        if realized:
            metadata["renderplan.upstream_realized_controls"] = realized
    operations = (gain,) if gain is not None else ()
    if emphasis_report is not None:
        metadata[RENDERPLAN_EMPHASIS] = emphasis_report
    return AudioClip(
        id=f"speech:{item.number:06d}",
        source=AudioFileSource(
            path=path,
            expected_sha256=digest,
            sample_rate=artifact.sample_rate,
            channels=1,
            frames=artifact.frames,
        ),
        operations=operations,
        metadata=metadata,
    )


def _materialize_pause(item: PauseItem, plan_id: str, flow_index: int) -> Silence:
    pause_id = item.pause_id or item.computed_id()
    metadata: dict[str, Any] = {
        "kind": "pause",
        RENDERPLAN_PLAN_ID: plan_id,
        RENDERPLAN_PAUSE_ID: pause_id,
        RENDERPLAN_PAUSE_REASON: item.reason,
    }
    if item.source:
        metadata[RENDERPLAN_PAUSE_SOURCE] = [span.to_dict() for span in item.source]
    return Silence(
        id=f"pause:{flow_index + 1:06d}:{pause_id}",
        seconds=item.duration_ms / 1000.0,
        metadata=metadata,
    )


def _resolve_audio_items(
    flow: tuple[Any, ...],
    *,
    media_root: Path | None,
    media_resolver: MediaResolver | None,
) -> dict[int, tuple[Any, Any, ResolvedMedia]]:
    resolver = media_resolver if media_resolver is not None else LocalWavResolver()
    resolved_items: dict[int, tuple[Any, Any, ResolvedMedia]] = {}
    for flow_index, item in enumerate(flow):
        if not isinstance(item, AudioItem):
            continue
        audio_id = item.audio_id or item.computed_id()
        locator = f"renderplan.audio:{audio_id}"
        request = renderplan_media_request(
            item.src,
            item.options,
            source_locator=locator,
            media_root=media_root,
        )
        try:
            resolved = resolver.resolve(request)
        except MediaResolutionError:
            raise
        except Exception as exc:
            raise MediaResolutionError(f"{locator}: media resolver failed: {exc}") from exc
        if not isinstance(resolved, ResolvedMedia):
            raise MediaResolutionError("MediaResolver must return ResolvedMedia")
        try:
            resolved = apply_renderplan_media_options(resolved, request)
        except MediaResolutionError:
            raise
        except Exception as exc:
            raise MediaResolutionError(f"{locator}: cannot apply media options: {exc}") from exc
        resolved_items[flow_index] = (item, request, resolved)
    return resolved_items


def _materialize_audio(
    item: AudioItem, flow_index: int, plan_id: str, record: tuple[Any, Any, ResolvedMedia]
) -> AudioClip:
    _, _, resolved = record
    audio_id = item.audio_id or item.computed_id()
    metadata = {
        "kind": "external_audio",
        RENDERPLAN_PLAN_ID: plan_id,
        RENDERPLAN_AUDIO_ID: audio_id,
        RENDERPLAN_SOURCE: [span.to_dict() for span in item.source],
        RENDERPLAN_AUDIO_SOURCE: item.src,
        RENDERPLAN_AUDIO_OPTIONS: dict(item.options),
        AUDIOCOMPOSE_MEDIA_METADATA: resolved.metadata,
        AUDIOCOMPOSE_MEDIA_IDENTITY: resolved.identity,
    }
    return AudioClip(
        id=f"media:{flow_index + 1:06d}",
        source=resolved.source,
        operations=resolved.operations,
        metadata=metadata,
    )


def materialize_renderplan_job(
    plan: RenderPlan,
    manifest: RenderManifest,
    *,
    render_dir: str | Path,
    output_policy: OutputPolicy | None = None,
    media_resolver: MediaResolver | None = None,
    media_root: str | Path | None = None,
    emphasis_policy: Literal["segment_fallback", "strict"] = "strict",
    _control_ownership: _ValidatedControlOwnership | None = None,
) -> AudioJob:
    """Verify a complete RenderPlan handoff and return a generic AudioJob.

    Plan and manifest identity, speech artifacts, and all external media are
    validated before AudioJob construction. Relative external paths use only the
    explicit ``media_root`` supplied for this in-memory plan.
    """
    if emphasis_policy not in ("segment_fallback", "strict"):
        raise HandoffValidationError("emphasis_policy must be 'segment_fallback' or 'strict'")
    plan_id = _validate_contract(plan, manifest)
    if _control_ownership is not None:
        if not isinstance(_control_ownership, _ValidatedControlOwnership):
            raise HandoffValidationError(
                "control ownership must be validated by a supported producer bridge"
            )
        if _control_ownership.producer != "voicerender" or _control_ownership.plan_id != plan_id:
            raise HandoffValidationError(
                "validated control ownership does not match this RenderPlan handoff"
            )
    render_root = _safe_directory(render_dir, label="render_dir")
    verified = _verify_speech_artifacts(plan, manifest, render_root)
    if output_policy is not None and not isinstance(output_policy, OutputPolicy):
        raise HandoffValidationError("output_policy must be an AudioCompose OutputPolicy")
    resolved_media_root = (
        _safe_directory(media_root, label="media_root") if media_root is not None else None
    )
    resolved_audio = _resolve_audio_items(
        plan.flow,
        media_root=resolved_media_root,
        media_resolver=media_resolver,
    )

    items: list[AudioClip | Silence] = []
    for flow_index, item in enumerate(plan.flow):
        if isinstance(item, SpeechItem):
            items.append(
                _materialize_speech(
                    item, plan_id, manifest, verified, emphasis_policy, _control_ownership
                )
            )
        elif isinstance(item, PauseItem):
            items.append(_materialize_pause(item, plan_id, flow_index))
        elif isinstance(item, AudioItem):
            items.append(_materialize_audio(item, flow_index, plan_id, resolved_audio[flow_index]))
        else:  # RenderPlan validation should make this unreachable.
            raise HandoffValidationError(f"Unsupported RenderPlan flow item: {type(item).__name__}")

    source_metadata: dict[str, Any] = {
        RENDERPLAN_PLAN_ID: plan_id,
        RENDERPLAN_SOURCE_PLAN_ID: plan.source_plan_id,
        RENDERPLAN_PROFILE_ID: plan.render_profile_id,
        MANIFEST_ID: manifest.manifest_id or manifest.computed_id(),
        AUDIOCOMPOSE_MATERIALIZER: "renderplan.v1",
    }
    if _control_ownership is not None:
        source_metadata.update(
            {
                VOICERENDER_DIAGNOSTICS_SCHEMA: _control_ownership.diagnostics_schema,
                VOICERENDER_DIAGNOSTICS_SHA256: _control_ownership.diagnostics_sha256,
                VOICERENDER_EMPHASIS_POLICY: _control_ownership.emphasis_policy,
                VOICERENDER_DEFERRED_CAPABILITIES: list(_control_ownership.deferred_capabilities),
                VOICERENDER_REALIZED_CONTROLS: sorted(_control_ownership.realized_controls),
            }
        )
    job = AudioJob(
        items=tuple(items),
        output=output_policy if output_policy is not None else OutputPolicy(),
        producer={"name": "audiocompose.renderplan-materializer", "version": __version__},
        source=source_metadata,
    )
    try:
        job.validate(verify_sources=True)
    except Exception as exc:
        raise HandoffValidationError(f"Materialized AudioJob failed validation: {exc}") from exc
    return job


def load_and_materialize_renderplan_job(
    plan_path: str | Path,
    manifest_path: str | Path | None = None,
    *,
    render_dir: str | Path,
    output_policy: OutputPolicy | None = None,
    media_resolver: MediaResolver | None = None,
    media_root: str | Path | None = None,
    emphasis_policy: Literal["segment_fallback", "strict"] = "strict",
) -> AudioJob:
    """Load shared-contract TOML files and materialize them as an AudioJob."""
    from renderplan import load_manifest, load_plan

    plan_location = Path(plan_path).expanduser()
    manifest_location = (
        Path(manifest_path).expanduser()
        if manifest_path is not None
        else Path(render_dir).expanduser() / MANIFEST_FILENAME
    )
    if manifest_location.is_symlink():
        raise HandoffValidationError(f"RenderManifest must not be a symlink: {manifest_location}")
    try:
        plan = load_plan(plan_location)
        manifest = load_manifest(manifest_location)
    except Exception as exc:
        raise HandoffValidationError(f"Cannot load RenderPlan handoff: {exc}") from exc
    return materialize_renderplan_job(
        plan,
        manifest,
        render_dir=render_dir,
        output_policy=output_policy,
        media_resolver=media_resolver,
        media_root=media_root,
        emphasis_policy=emphasis_policy,
    )


__all__ = [
    "MANIFEST_FILENAME",
    "load_and_materialize_renderplan_job",
    "materialize_renderplan_job",
]
