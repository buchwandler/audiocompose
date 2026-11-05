from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AudioValidationError
from .wav import _as_finite_mono_float32, read_wav


@dataclass(frozen=True, slots=True)
class AudioBufferSource:
    audio: np.ndarray
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        values = _as_finite_mono_float32(self.audio, name="buffer audio", copy=True)
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise AudioValidationError("sample_rate must be a positive integer")
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels != 1
        ):
            raise AudioValidationError("AudioCompose supports mono sources only")
        values.setflags(write=False)
        object.__setattr__(self, "audio", values)

    def load(self) -> tuple[np.ndarray, int]:
        return self.audio.copy(), self.sample_rate


@dataclass(frozen=True, slots=True)
class RenderedAudioBuffer:
    """Generic in-memory speech audio for RenderPlan composition."""

    audio: np.ndarray
    sample_rate: int
    word_timings: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = AudioBufferSource(self.audio, self.sample_rate)
        if not isinstance(self.word_timings, (tuple, list)) or any(
            not isinstance(timing, Mapping) for timing in self.word_timings
        ):
            raise AudioValidationError("word_timings must be a sequence of mappings")
        if not isinstance(self.metadata, Mapping):
            raise AudioValidationError("metadata must be a mapping")
        object.__setattr__(self, "audio", source.audio)
        object.__setattr__(self, "word_timings", tuple(dict(item) for item in self.word_timings))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class AudioFileSource:
    path: str | Path
    expected_sha256: str | None = None
    sample_rate: int | None = None
    channels: int = 1
    frames: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)) or not str(self.path):
            raise AudioValidationError("path must be a non-empty string or path")
        if self.expected_sha256 is not None and (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise AudioValidationError(
                "expected_sha256 must be 64 lowercase hexadecimal characters"
            )
        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise AudioValidationError("sample_rate must be a positive integer or None")
        if self.frames is not None and (
            isinstance(self.frames, bool) or not isinstance(self.frames, int) or self.frames < 0
        ):
            raise AudioValidationError("frames must be a non-negative integer or None")
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels != 1
        ):
            raise AudioValidationError("AudioCompose supports mono sources only")
        object.__setattr__(self, "path", str(self.path))

    def load(self) -> tuple[np.ndarray, int]:
        audio, rate = read_wav(self.path, expected_channels=self.channels)
        if self.sample_rate is not None and rate != self.sample_rate:
            raise AudioValidationError(
                f"source sample rate mismatch for {self.path}: expected {self.sample_rate}, got {rate}"
            )
        if self.frames is not None and audio.size != self.frames:
            raise AudioValidationError(
                f"source frame count mismatch for {self.path}: expected {self.frames}, got {audio.size}"
            )
        if self.expected_sha256:
            from .wav import sha256_file

            try:
                actual = sha256_file(self.path)
            except OSError as exc:
                raise AudioValidationError(
                    f"cannot verify source hash for {self.path}: {exc}"
                ) from exc
            if actual != self.expected_sha256:
                raise AudioValidationError(f"source hash mismatch for {self.path}")
        return audio, rate


AudioSource = AudioBufferSource | AudioFileSource
