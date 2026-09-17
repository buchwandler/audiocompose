from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .alignment import AudioTextSpan
from .diagnostics import RenderDiagnostic
from .errors import FragmentValidationError
from .prosody import ProsodyAxis


def _waveform(value: np.ndarray) -> np.ndarray:
    audio = np.asarray(value, dtype=np.float32)
    if audio.ndim != 1:
        raise FragmentValidationError(
            f"audio must be one-dimensional, got shape {audio.shape}"
        )
    if not np.all(np.isfinite(audio)):
        raise FragmentValidationError("audio contains non-finite samples")
    return np.ascontiguousarray(audio)


@dataclass(frozen=True, slots=True)
class AudioFragment:
    """One backend-rendered TTS segment."""

    segment_id: str
    audio: np.ndarray
    sample_rate: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    realized_prosody: frozenset[ProsodyAxis] = frozenset()
    alignment: tuple[AudioTextSpan, ...] = ()
    diagnostics: tuple[RenderDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise FragmentValidationError("segment_id must not be empty")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise FragmentValidationError("sample_rate must be an integer")
        if self.sample_rate <= 0:
            raise FragmentValidationError("sample_rate must be > 0")
        allowed = {"rate", "pitch", "volume"}
        realized = frozenset(self.realized_prosody)
        invalid = sorted(set(realized) - allowed)
        if invalid:
            raise FragmentValidationError(
                "realized_prosody contains unsupported axes: " + ", ".join(invalid)
            )
        object.__setattr__(self, "realized_prosody", realized)
        object.__setattr__(self, "audio", _waveform(self.audio))
        object.__setattr__(self, "alignment", tuple(self.alignment))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True, slots=True)
class RenderedSegment:
    segment_id: str
    block_start_sample: int
    audio_start_sample: int
    audio_end_sample: int
    block_end_sample: int
    pause_before_samples: int
    pause_after_samples: int

    @property
    def audio_samples(self) -> int:
        return self.audio_end_sample - self.audio_start_sample

    @property
    def block_samples(self) -> int:
        return self.block_end_sample - self.block_start_sample


@dataclass(frozen=True, slots=True)
class RenderedUnit:
    unit_id: str
    index: int
    start_sample: int
    end_sample: int
    segment_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderedMarker:
    marker_id: str
    name: str
    char_offset: int
    sample_offset: int | None
    timing: Literal["resolved", "unresolved"]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    plan_id: str
    segments: tuple[RenderedSegment, ...]
    units: tuple[RenderedUnit, ...]
    markers: tuple[RenderedMarker, ...]
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[RenderDiagnostic, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    audio_in_range: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "audio", _waveform(self.audio))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate
