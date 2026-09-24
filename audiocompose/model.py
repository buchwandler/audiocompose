from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._json_value import snapshot_json_object
from .alignment import AudioAnchor, AudioSpan, ComposedMarker, ComposedSpan
from .diagnostics import CompositionDiagnostic
from .errors import AudioValidationError
from .loudness import LoudnessPolicy
from .operations import Operation
from .sources import AudioSource
from .wav import ClipPolicy

if TYPE_CHECKING:
    from .loudness import LoudnessResult


@dataclass(frozen=True, slots=True)
class OutputPolicy:
    sample_rate: int = 24000
    channels: int = 1
    loudness: LoudnessPolicy = field(default_factory=LoudnessPolicy)
    clip_policy: ClipPolicy = "clamp"

    def __post_init__(self) -> None:
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
            raise AudioValidationError("AudioCompose supports mono output only")
        if self.clip_policy not in ("clamp", "warn", "error"):
            raise AudioValidationError(f"unknown clip policy: {self.clip_policy!r}")


@dataclass(frozen=True, slots=True)
class AudioClip:
    id: str
    source: AudioSource
    operations: tuple[Operation, ...] = ()
    anchors: tuple[AudioAnchor, ...] = ()
    spans: tuple[AudioSpan, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise AudioValidationError("clip id must not be empty")
        operations = tuple(self.operations)
        if any(not isinstance(operation, Operation) for operation in operations):
            raise AudioValidationError(f"clip {self.id!r} contains an unsupported operation")
        anchors = tuple(self.anchors)
        seen: set[str] = set()
        for anchor in anchors:
            if not isinstance(anchor, AudioAnchor):
                raise AudioValidationError(f"clip {self.id!r} contains an invalid anchor")
            if anchor.id in seen:
                raise AudioValidationError(f"duplicate anchor id {anchor.id!r} in clip {self.id!r}")
            seen.add(anchor.id)
        spans = tuple(self.spans)
        if any(not isinstance(span, AudioSpan) for span in spans):
            raise AudioValidationError(f"clip {self.id!r} contains an invalid span")
        metadata = snapshot_json_object(self.metadata, f"clip {self.id!r} metadata")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "spans", spans)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class Silence:
    id: str
    seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        if not isinstance(self.id, str) or not self.id:
            raise AudioValidationError("silence id must not be empty")
        if not isinstance(self.seconds, (int, float)) or isinstance(self.seconds, bool):
            raise AudioValidationError("silence seconds must be a finite number >= 0")
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise AudioValidationError("silence seconds must be a finite number >= 0")
        object.__setattr__(
            self, "metadata", snapshot_json_object(self.metadata, f"silence {self.id!r} metadata")
        )


@dataclass(frozen=True, slots=True)
class AudioJob:
    items: tuple[AudioClip | Silence, ...]
    output: OutputPolicy = field(default_factory=OutputPolicy)
    producer: Mapping[str, Any] = field(default_factory=dict)
    job_id: str | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    schema_version: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version not in {1, 2}
        ):
            raise AudioValidationError("unsupported AudioJob schema version")
        items = tuple(self.items)
        ids: set[str] = set()
        for item in items:
            if not isinstance(item, (AudioClip, Silence)):
                raise AudioValidationError("AudioJob items must be AudioClip or Silence")
            if item.id in ids:
                raise AudioValidationError(f"duplicate AudioJob item id: {item.id!r}")
            ids.add(item.id)
        producer = snapshot_json_object(self.producer, "producer metadata")
        source = snapshot_json_object(self.source, "source metadata")
        if self.job_id is not None and (
            not isinstance(self.job_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.job_id) is None
        ):
            raise AudioValidationError(
                "job_id must use the canonical sha256:<64 lowercase hex> format"
            )
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "source", source)

    def validate(self, *, base_dir: str | None = None, verify_sources: bool = True) -> None:
        from .job import validate_job

        validate_job(self, base_dir=base_dir, verify_sources=verify_sources)

    def save(self, path: str | Path) -> str:
        """Save to a bundle directory and return its ``audiojob.json`` path."""
        from .job import save_job

        return str(save_job(self, path))

    @classmethod
    def load(cls, path: str, *, verify_sources: bool = True) -> AudioJob:
        """Load a manifest, optionally deferring source reads until use.

        With ``verify_sources=False``, manifest structure is still validated,
        while source integrity and sample geometry are checked when sources
        are loaded by composition or inspection.
        """
        from .job import load_job

        return load_job(path, verify_sources=verify_sources)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        base_dir: str = ".",
        verify_sources: bool = True,
    ) -> AudioJob:
        from .job import job_from_dict

        return job_from_dict(payload, base_dir=base_dir, verify_sources=verify_sources)

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
    audio: np.ndarray
    sample_rate: int
    items: tuple[ComposedItem, ...] = ()
    markers: tuple[ComposedMarker, ...] = ()
    spans: tuple[ComposedSpan, ...] = ()
    diagnostics: tuple[CompositionDiagnostic, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    loudness: LoudnessResult | None = None

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / self.sample_rate

    @property
    def waveform(self) -> np.ndarray:
        return self.audio
