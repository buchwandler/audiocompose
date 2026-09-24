from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._json_value import snapshot_json_object
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
        if self.name is not None and not isinstance(self.name, str):
            raise AudioValidationError("anchor name must be a string or None")


@dataclass(frozen=True, slots=True)
class AudioSpan:
    source_start: int
    source_end: int
    sample_start: int
    sample_end: int
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise AudioValidationError("span id must be a non-empty string or None")
        object.__setattr__(self, "metadata", snapshot_json_object(self.metadata, "span metadata"))


@dataclass(frozen=True, slots=True)
class ComposedSpan:
    item_id: str
    source_start: int
    source_end: int
    sample_start: int
    sample_end: int
    id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

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
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise AudioValidationError("span id must be a non-empty string or None")
        object.__setattr__(
            self, "metadata", snapshot_json_object(self.metadata, "composed span metadata")
        )


@dataclass(frozen=True, slots=True)
class ComposedMarker:
    id: str
    sample_offset: int
    name: str | None = None
    item_id: str | None = None


Marker = AudioAnchor
