from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AlignmentKind = Literal["word", "phoneme"]


@dataclass(frozen=True, slots=True)
class AudioTextSpan:
    """Map a spoken-text range to a rendered audio range."""

    spoken_start: int
    spoken_end: int
    sample_start: int
    sample_end: int
    kind: AlignmentKind

    def __post_init__(self) -> None:
        if self.spoken_start < 0 or self.spoken_end < self.spoken_start:
            raise ValueError("spoken span must be ordered and non-negative")
        if self.sample_start < 0 or self.sample_end < self.sample_start:
            raise ValueError("audio span must be ordered and non-negative")

    @property
    def spoken_length(self) -> int:
        return self.spoken_end - self.spoken_start

    @property
    def sample_length(self) -> int:
        return self.sample_end - self.sample_start
