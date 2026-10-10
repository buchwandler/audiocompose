"""Media-resolution protocol and safe local-WAV resolver for integrations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any, Protocol
from urllib.parse import urlsplit

from ..errors import MediaResolutionError
from ..operations import AudioOperation, Gain, Tempo
from ..sources import AudioBufferSource, AudioFileSource, AudioSource
from ..wav import read_wav, sha256_file, wav_info

_NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_DECIBELS = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))dB$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MediaRequest:
    """One opaque UtterPlan external-audio request passed to a resolver."""

    src: str
    plan_path: Path
    source_locator: str
    clip_begin: str | None = None
    clip_end: str | None = None
    speed: str | float | int | None = None
    repeat_duration: str | None = None
    repeat_count: float | None = None
    sound_level: str | None = None
    description: str | None = None
    alt_text: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedMedia:
    """Generic AudioCompose source, operations, and opaque resolver metadata."""

    source: AudioSource
    operations: tuple[AudioOperation, ...] = ()
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
    """Resolve an opaque producer media request into generic audio values."""

    def resolve(self, request: MediaRequest) -> ResolvedMedia:
        """Resolve media or raise ``MediaResolutionError`` with a useful message."""
        ...


class LocalWavResolver:
    """Resolve plain relative WAV paths beneath the UtterPlan directory."""

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
        relative = PurePath(src)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise MediaResolutionError(
                f"{request.source_locator}: audio.src must remain beneath the UtterPlan directory"
            )
        if Path(src).suffix.lower() != ".wav":
            raise MediaResolutionError(
                f"{request.source_locator}: local media supports WAV files only, got {src!r}"
            )
        plan_path = Path(request.plan_path)
        if plan_path.is_symlink() or not plan_path.is_file():
            raise MediaResolutionError(
                f"{request.source_locator}: UtterPlan file is not a regular file"
            )
        root = plan_path.parent.resolve(strict=True)
        candidate = root
        try:
            for part in relative.parts:
                candidate = candidate / part
                candidate.lstat()
                if candidate.is_symlink():
                    raise MediaResolutionError(
                        f"{request.source_locator}: local media path contains a symlink"
                    )
            path = candidate.resolve(strict=True)
        except OSError as exc:
            raise MediaResolutionError(
                f"{request.source_locator}: cannot resolve local WAV {src!r}: {exc}"
            ) from exc
        if not path.is_relative_to(root) or not path.is_file():
            raise MediaResolutionError(
                f"{request.source_locator}: local media is not a regular file beneath the plan directory"
            )
        if request.clip_begin is not None or request.clip_end is not None:
            raise MediaResolutionError(
                f"{request.source_locator}: clip_begin/clip_end are unsupported by AudioJob v2"
            )
        if request.repeat_duration is not None or request.repeat_count is not None:
            raise MediaResolutionError(
                f"{request.source_locator}: repeat_duration/repeat_count are unsupported by AudioJob v2"
            )
        try:
            audio, sample_rate = read_wav(path, expected_channels=1)
            info = wav_info(path)
            digest = sha256_file(path)
        except Exception as exc:
            raise MediaResolutionError(
                f"{request.source_locator}: invalid local WAV {path}: {exc}"
            ) from exc
        if not audio.size or info.channels != 1:
            raise MediaResolutionError(
                f"{request.source_locator}: local WAV must be non-empty and mono"
            )
        if info.frames != audio.size or info.sample_rate != sample_rate:
            raise MediaResolutionError(
                f"{request.source_locator}: local WAV geometry is inconsistent"
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


def numeric_media_operations(request: MediaRequest) -> tuple[AudioOperation, ...]:
    """Lower only unambiguous numeric speed/dB controls to native operations."""
    operations: list[AudioOperation] = []
    if request.speed is not None:
        if isinstance(request.speed, bool) or not isinstance(request.speed, (str, int, float)):
            raise MediaResolutionError(
                f"{request.source_locator}: audio.speed must be a numeric playback factor"
            )
        if isinstance(request.speed, str):
            if _NUMERIC.fullmatch(request.speed) is None:
                raise MediaResolutionError(
                    f"{request.source_locator}: audio.speed must be a numeric playback factor"
                )
            speed = float(request.speed)
        else:
            speed = float(request.speed)
        if not math.isfinite(speed) or speed <= 0:
            raise MediaResolutionError(
                f"{request.source_locator}: audio.speed must be finite and > 0"
            )
        operations.append(Tempo(speed))
    if request.sound_level is not None:
        if not isinstance(request.sound_level, str):
            raise MediaResolutionError(
                f"{request.source_locator}: audio.sound_level must be an explicit dB value"
            )
        match = _DECIBELS.fullmatch(request.sound_level)
        if match is None:
            raise MediaResolutionError(
                f"{request.source_locator}: audio.sound_level must use an explicit numeric dB value"
            )
        gain = float(match.group(1))
        if not math.isfinite(gain):
            raise MediaResolutionError(
                f"{request.source_locator}: audio.sound_level must be finite"
            )
        operations.append(Gain(gain))
    return tuple(operations)


__all__ = ["MediaRequest", "ResolvedMedia", "MediaResolver", "LocalWavResolver"]
