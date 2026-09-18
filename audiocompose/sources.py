from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import AudioValidationError
from .wav import read_wav


@dataclass(frozen=True, slots=True)
class AudioBufferSource:
    audio: np.ndarray
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        values = np.asarray(self.audio, dtype=np.float32)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise AudioValidationError("buffer audio must be a finite one-dimensional waveform")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise AudioValidationError("sample_rate must be a positive integer")
        if self.channels != 1:
            raise AudioValidationError("audiocompose v1 supports mono sources only")
        object.__setattr__(self, "audio", np.ascontiguousarray(values))

    def load(self) -> tuple[np.ndarray, int]:
        return self.audio.copy(), self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioFileSource:
    path: str | Path
    expected_sha256: str | None = None
    sample_rate: int | None = None
    channels: int = 1
    frames: int | None = None

    def __post_init__(self) -> None:
        if self.channels != 1:
            raise AudioValidationError("audiocompose v1 supports mono sources only")
        object.__setattr__(self, "path", str(self.path))

    def load(self) -> tuple[np.ndarray, int]:
        audio, rate = read_wav(self.path, expected_channels=self.channels)
        if self.sample_rate is not None and rate != self.sample_rate:
            raise AudioValidationError(f"source sample rate mismatch for {self.path}: expected {self.sample_rate}, got {rate}")
        if self.frames is not None and audio.size != self.frames:
            raise AudioValidationError(f"source frame count mismatch for {self.path}: expected {self.frames}, got {audio.size}")
        if self.expected_sha256:
            from .wav import sha256_file
            actual = sha256_file(self.path)
            if actual != self.expected_sha256:
                raise AudioValidationError(f"source hash mismatch for {self.path}")
        return audio, rate


AudioSource = AudioBufferSource | AudioFileSource
