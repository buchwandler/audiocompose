"""Materialize verified shared RenderPlan contracts as generic AudioCompose jobs."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from renderplan import AudioItem, PauseItem, RenderManifest, RenderPlan, SpeechItem

from .._version import __version__
from ..errors import HandoffValidationError, MediaResolutionError
from ..model import AudioClip, AudioJob, OutputPolicy, Silence
from ..sources import AudioFileSource
from ..wav import read_wav, sha256_file, wav_info
from .media import (
    LocalWavResolver,
    MediaResolver,
    ResolvedMedia,
    apply_renderplan_media_options,
    renderplan_media_request,
)

MANIFEST_FILENAME = "render-manifest.toml"


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


def _materialize_speech(
    item: SpeechItem,
    plan_id: str,
    manifest: RenderManifest,
    verified: dict[int, tuple[Any, Path, str]],
) -> AudioClip:
    artifact, path, digest = verified[item.number]
    segment_id = item.segment_id or item.computed_id()
    metadata: dict[str, Any] = {
        "kind": "speech",
        "renderplan.plan_id": plan_id,
        "renderplan.segment_number": item.number,
        "renderplan.segment_id": segment_id,
        "renderplan.source": _source_metadata(item),
        "render_manifest.manifest_id": manifest.manifest_id or manifest.computed_id(),
        "render_manifest.wav_sha256": artifact.sha256,
        "render_manifest.wav": artifact.wav,
        "render_manifest.sample_rate": artifact.sample_rate,
        "render_manifest.channels": artifact.channels,
        "render_manifest.frames": artifact.frames,
    }
    return AudioClip(
        id=f"speech:{item.number:06d}",
        source=AudioFileSource(
            path=path,
            expected_sha256=digest,
            sample_rate=artifact.sample_rate,
            channels=1,
            frames=artifact.frames,
        ),
        metadata=metadata,
    )


def _materialize_pause(item: PauseItem, plan_id: str, flow_index: int) -> Silence:
    pause_id = item.pause_id or item.computed_id()
    metadata: dict[str, Any] = {
        "kind": "pause",
        "renderplan.plan_id": plan_id,
        "renderplan.pause_id": pause_id,
        "renderplan.pause_reason": item.reason,
    }
    if item.source:
        metadata["renderplan.pause_source"] = [span.to_dict() for span in item.source]
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
        "renderplan.plan_id": plan_id,
        "renderplan.audio_id": audio_id,
        "renderplan.source": [span.to_dict() for span in item.source],
        "renderplan.audio.src": item.src,
        "renderplan.audio.options": dict(item.options),
        "audiocompose.media.metadata": resolved.metadata,
        "audiocompose.media.identity": resolved.identity,
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
) -> AudioJob:
    """Verify a complete RenderPlan handoff and return a generic AudioJob.

    Plan and manifest identity, speech artifacts, and all external media are
    validated before AudioJob construction. Relative external paths use only the
    explicit ``media_root`` supplied for this in-memory plan.
    """
    plan_id = _validate_contract(plan, manifest)
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
            items.append(_materialize_speech(item, plan_id, manifest, verified))
        elif isinstance(item, PauseItem):
            items.append(_materialize_pause(item, plan_id, flow_index))
        elif isinstance(item, AudioItem):
            items.append(_materialize_audio(item, flow_index, plan_id, resolved_audio[flow_index]))
        else:  # RenderPlan validation should make this unreachable.
            raise HandoffValidationError(f"Unsupported RenderPlan flow item: {type(item).__name__}")

    job = AudioJob(
        items=tuple(items),
        output=output_policy if output_policy is not None else OutputPolicy(),
        producer={"name": "audiocompose.renderplan-materializer", "version": __version__},
        source={
            "renderplan.plan_id": plan_id,
            "renderplan.source_plan_id": plan.source_plan_id,
            "renderplan.render_profile_id": plan.render_profile_id,
            "render_manifest.manifest_id": manifest.manifest_id or manifest.computed_id(),
            "audiocompose.materializer": "renderplan.v1",
        },
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
    )


__all__ = [
    "MANIFEST_FILENAME",
    "load_and_materialize_renderplan_job",
    "materialize_renderplan_job",
]
