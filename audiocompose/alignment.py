from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioAnchor:
    id: str
    sample_offset: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("anchor id must not be empty")
        if isinstance(self.sample_offset, bool) or not isinstance(self.sample_offset, int) or self.sample_offset < 0:
            raise ValueError("sample_offset must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AudioSpan:
    source_start: int
    source_end: int
    sample_start: int
    sample_end: int


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
