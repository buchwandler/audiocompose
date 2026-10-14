"""Safe external-media resolution for shared-contract audio integrations."""

from __future__ import annotations

import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..errors import MediaResolutionError
from ..operations import AudioOperation, Gain, Operation, Tempo
from ..sources import AudioBufferSource, AudioFileSource, AudioSource
from ..wav import read_wav, sha256_file, wav_info


@dataclass(frozen=True, slots=True)
class MediaRequest:
    """A normalized external-media request passed to a resolver."""

    src: str
    source_locator: str
    media_root: Path | None = None
    clip_begin_ms: float | None = None
    clip_end_ms: float | None = None
    speed: float | None = None
    gain_db: float | None = None


@dataclass(frozen=True, slots=True)
class ResolvedMedia:
    """Generic AudioCompose source, operations, and opaque resolver metadata."""

    source: AudioSource
    operations: tuple[Operation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, (AudioBufferSource, AudioFileSource)):
            raise MediaResolutionError("ResolvedMedia.source must be an AudioCompose audio source")
        try:
            operations = tuple(self.operations)
        except TypeError as exc:
            raise MediaResolutionError("ResolvedMedia.operations must be a sequence") from exc
        if any(not isinstance(item, AudioOperation) for item in operations):
            raise MediaResolutionError("ResolvedMedia.operations contains an unsupported operation")
        if not isinstance(self.metadata, dict) or not isinstance(self.identity, dict):
            raise MediaResolutionError("ResolvedMedia metadata and identity must be JSON objects")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "identity", dict(self.identity))


class MediaResolver(Protocol):
    """Resolve a media request into generic AudioCompose audio values."""

    def resolve(self, request: MediaRequest) -> ResolvedMedia:
        """Resolve media or raise ``MediaResolutionError`` with a useful message."""
        ...


def _safe_media_root(path: str | Path, *, locator: str) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise MediaResolutionError(f"{locator}: media root contains a symlink")
            if not stat.S_ISDIR(info.st_mode):
                raise MediaResolutionError(f"{locator}: media root is not a directory")
        root = absolute.resolve(strict=True)
    except MediaResolutionError:
        raise
    except OSError as exc:
        raise MediaResolutionError(f"{locator}: media root is unavailable: {exc}") from exc
    if not root.is_dir():
        raise MediaResolutionError(f"{locator}: media root is not a directory")
    return root


class LocalWavResolver:
    """Resolve plain relative WAV paths beneath the request's explicit media root."""

    def resolve(self, request: MediaRequest) -> ResolvedMedia:
        src = request.src
        if not isinstance(src, str) or not src.strip():
            raise MediaResolutionError(f"{request.source_locator}: audio.src must be non-empty")
        parsed = urlsplit(src)
        if parsed.scheme:
            raise MediaResolutionError(
                f"{request.source_locator}: no resolver for URI scheme {parsed.scheme!r} in {src!r}"
            )
        if "\\" in src or "\x00" in src:
            raise MediaResolutionError(
                f"{request.source_locator}: audio.src is not a safe relative path"
            )
        relative = PurePosixPath(src)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise MediaResolutionError(
                f"{request.source_locator}: audio.src must remain beneath the media root"
            )
        if Path(src).suffix.lower() != ".wav":
            raise MediaResolutionError(
                f"{request.source_locator}: local media supports WAV files only, got {src!r}"
            )
        if request.media_root is None:
            raise MediaResolutionError(
                f"{request.source_locator}: relative media requires an explicit media_root"
            )
        root = _safe_media_root(request.media_root, locator=request.source_locator)
        candidate = root
        try:
            for part in relative.parts:
                candidate = candidate / part
                info = candidate.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise MediaResolutionError(
                        f"{request.source_locator}: local media path contains a symlink"
                    )
            path = candidate.resolve(strict=True)
            file_info = candidate.stat()
        except MediaResolutionError:
            raise
        except OSError as exc:
            raise MediaResolutionError(
                f"{request.source_locator}: cannot resolve local WAV {src!r}: {exc}"
            ) from exc
        if not path.is_relative_to(root) or not stat.S_ISREG(file_info.st_mode):
            raise MediaResolutionError(
                f"{request.source_locator}: local media is not a regular file beneath the media root"
            )
        try:
            audio, sample_rate = read_wav(path, expected_channels=1)
            wav_geometry = wav_info(path)
            digest = sha256_file(path)
        except Exception as exc:
            raise MediaResolutionError(
                f"{request.source_locator}: invalid local WAV {path}: {exc}"
            ) from exc
        if not audio.size or wav_geometry.channels != 1:
            raise MediaResolutionError(
                f"{request.source_locator}: local media WAV must be non-empty and mono"
            )
        if wav_geometry.frames != audio.size or wav_geometry.sample_rate != sample_rate:
            raise MediaResolutionError(
                f"{request.source_locator}: local media WAV geometry is inconsistent"
            )
        return ResolvedMedia(
            source=AudioFileSource(
                path=path,
                expected_sha256=digest,
                sample_rate=sample_rate,
                channels=1,
                frames=int(audio.size),
            )
        )


def _finite_option(
    options: Mapping[str, Any],
    name: str,
    *,
    source_locator: str,
    minimum: float | None = None,
    strict: bool = False,
) -> float | None:
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MediaResolutionError(f"{source_locator}: {name} must be a finite number")
    result = float(value)
    if minimum is not None and (result <= minimum if strict else result < minimum):
        relation = ">" if strict else ">="
        raise MediaResolutionError(f"{source_locator}: {name} must be {relation} {minimum}")
    return result


def renderplan_media_request(
    src: str,
    options: Mapping[str, Any],
    *,
    source_locator: str,
    media_root: str | Path | None,
) -> MediaRequest:
    """Validate normalized RenderPlan media options and make a resolver request."""
    if not isinstance(options, Mapping):
        raise MediaResolutionError(f"{source_locator}: AudioItem.options must be a mapping")
    allowed = {"clip_begin_ms", "clip_end_ms", "speed", "gain_db"}
    unknown = set(options).difference(allowed)
    if unknown:
        raise MediaResolutionError(
            f"{source_locator}: unsupported AudioItem option(s): {', '.join(sorted(map(str, unknown)))}"
        )
    clip_begin_ms = _finite_option(
        options, "clip_begin_ms", source_locator=source_locator, minimum=0
    )
    clip_end_ms = _finite_option(options, "clip_end_ms", source_locator=source_locator, minimum=0)
    speed = _finite_option(options, "speed", source_locator=source_locator, minimum=0, strict=True)
    gain_db = _finite_option(options, "gain_db", source_locator=source_locator)
    if clip_end_ms is not None and clip_end_ms <= (clip_begin_ms or 0.0):
        raise MediaResolutionError(f"{source_locator}: clip_end_ms must be after clip_begin_ms")
    return MediaRequest(
        src=src,
        source_locator=source_locator,
        media_root=Path(media_root).expanduser() if media_root is not None else None,
        clip_begin_ms=clip_begin_ms,
        clip_end_ms=clip_end_ms,
        speed=speed,
        gain_db=gain_db,
    )


def numeric_media_operations(request: MediaRequest) -> tuple[Operation, ...]:
    """Lower normalized playback speed to the existing native Tempo operation."""
    if request.speed is None:
        return ()
    if isinstance(request.speed, bool) or not isinstance(request.speed, (int, float)):
        raise MediaResolutionError(
            f"{request.source_locator}: speed must be a numeric playback factor"
        )
    speed = float(request.speed)
    if not math.isfinite(speed) or speed <= 0:
        raise MediaResolutionError(f"{request.source_locator}: speed must be finite and > 0")
    return (Tempo(speed),)


def apply_renderplan_media_options(resolved: ResolvedMedia, request: MediaRequest) -> ResolvedMedia:
    """Apply normalized RenderPlan trim, speed, and gain controls."""
    operations = list(numeric_media_operations(request))
    if request.gain_db is not None:
        if (
            isinstance(request.gain_db, bool)
            or not isinstance(request.gain_db, (int, float))
            or not math.isfinite(request.gain_db)
        ):
            raise MediaResolutionError(f"{request.source_locator}: gain_db must be a finite number")
        operations.append(Gain(float(request.gain_db)))

    source = resolved.source
    if request.clip_begin_ms is not None or request.clip_end_ms is not None:
        try:
            audio, sample_rate = source.load()
        except Exception as exc:
            raise MediaResolutionError(
                f"{request.source_locator}: cannot read media for safe trimming: {exc}"
            ) from exc
        begin_ms = request.clip_begin_ms or 0.0
        start = round(begin_ms * sample_rate / 1000.0)
        end = (
            round(request.clip_end_ms * sample_rate / 1000.0)
            if request.clip_end_ms is not None
            else len(audio)
        )
        if start < 0 or end > len(audio) or start >= end:
            raise MediaResolutionError(
                f"{request.source_locator}: clip range is outside the media or contains no samples"
            )
        source = AudioBufferSource(audio[start:end], sample_rate)

    return ResolvedMedia(
        source=source,
        operations=(*resolved.operations, *operations),
        metadata=resolved.metadata,
        identity=resolved.identity,
    )


__all__ = [
    "LocalWavResolver",
    "MediaRequest",
    "MediaResolver",
    "ResolvedMedia",
    "apply_renderplan_media_options",
    "numeric_media_operations",
    "renderplan_media_request",
]
