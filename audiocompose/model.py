from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .alignment import AudioAnchor, AudioSpan, ComposedMarker, ComposedSpan
from .loudness import LoudnessPolicy
from .sources import AudioSource
from .wav import ClipPolicy


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    sample_rate: int = 24000
    channels: int = 1
    loudness: LoudnessPolicy = field(default_factory=LoudnessPolicy)
    clip_policy: ClipPolicy = "clamp"

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if self.channels != 1:
            raise ValueError("audiocompose v1 supports mono output only")
        if self.clip_policy not in {"clamp", "warn", "error"}:
            raise ValueError(f"unknown clip policy: {self.clip_policy!r}")


@dataclass(frozen=True, slots=True)
class AudioClip:
    id: str
    source: AudioSource
    operations: tuple[Any, ...] = ()
    anchors: tuple[AudioAnchor, ...] = ()
    spans: tuple[AudioSpan, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("clip id must not be empty")
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "spans", tuple(self.spans))


@dataclass(frozen=True, slots=True)
class Silence:
    id: str
    seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math
        if not self.id:
            raise ValueError("silence id must not be empty")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("silence seconds must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class AudioJob:
    items: tuple[AudioClip | Silence, ...]
    output: OutputPolicy = field(default_factory=OutputPolicy)
    producer: Mapping[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        ids: set[str] = set()
        for item in self.items:
            if item.id in ids:
                raise ValueError(f"duplicate AudioJob item id: {item.id!r}")
            ids.add(item.id)

    def validate(self, *, base_dir: str | None = None) -> None:
        from .job import validate_job
        validate_job(self, base_dir=base_dir)

    def save(self, path: str) -> str:
        from .job import save_job
        return str(save_job(self, path))

    @classmethod
    def load(cls, path: str) -> AudioJob:
        from .job import load_job
        return load_job(path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, base_dir: str = ".") -> AudioJob:
        from .job import job_from_dict
        return job_from_dict(payload, base_dir=base_dir)

    def to_dict(self, *, base_dir: str | None = None) -> dict[str, Any]:
        from .job import job_to_dict
        return job_to_dict(self, base_dir=base_dir)


@dataclass(frozen=True, slots=True)
class ComposedItem:
    item_id: str
    kind: str
    start_sample: int
    end_sample: int
    source_sample_rate: int | None = None

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class CompositionResult:
    audio: Any
    sample_rate: int
    items: tuple[ComposedItem, ...] = ()
    markers: tuple[ComposedMarker, ...] = ()
    spans: tuple[ComposedSpan, ...] = ()
    diagnostics: tuple[Any, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def waveform(self) -> Any:
        return self.audio
