from __future__ import annotations

from dataclasses import dataclass

from .errors import AudioValidationError


@dataclass(frozen=True, slots=True)
class AudioAnchor:
    id: str
    sample_offset: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise AudioValidationError("anchor id must not be empty")
        if (
            isinstance(self.sample_offset, bool)
            or not isinstance(self.sample_offset, int)
            or self.sample_offset < 0
        ):
            raise AudioValidationError("sample_offset must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AudioSpan:
    source_start: int
    source_end: int
    sample_start: int
    sample_end: int

    def __post_init__(self) -> None:
        values = (
            ("source_start", self.source_start),
            ("source_end", self.source_end),
            ("sample_start", self.sample_start),
            ("sample_end", self.sample_end),
        )
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AudioValidationError(f"{name} must be a non-negative integer")
        if self.source_end < self.source_start:
            raise AudioValidationError("source_end must be >= source_start")
        if self.sample_end < self.sample_start:
            raise AudioValidationError("sample_end must be >= sample_start")


@dataclass(frozen=True, slots=True)
class ComposedSpan:
    item_id: str
    source_start: int
    source_end: int
    sample_start: int
    sample_end: int


@dataclass(frozen=True, slots=True)
class ComposedMarker:
    id: str
    sample_offset: int
    name: str | None = None
    item_id: str | None = None


Marker = AudioAnchor
