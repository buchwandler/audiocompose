from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CompositionProgress:
    """A producer-neutral observation of composition progress."""

    kind: str
    completed_items: int
    total_items: int
    item_index: int | None = None
    item_id: str | None = None
    item_kind: str | None = None
    item_metadata: Mapping[str, Any] = field(default_factory=dict)
    operation_index: int | None = None
    operation_count: int | None = None
    operation: Mapping[str, Any] | None = None
    source_sample_rate: int | None = None
    target_sample_rate: int | None = None
    input_frames: int | None = None
    output_frames: int | None = None
    completed_audio_seconds: float | None = None
    total_audio_seconds: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


CompositionProgressCallback = Callable[[CompositionProgress], None]


__all__ = ["CompositionProgress", "CompositionProgressCallback"]
