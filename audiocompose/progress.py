from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ProgressEventKind = Literal[
    "compose_started",
    "item_started",
    "item_completed",
    "source_load_started",
    "source_load_completed",
    "operation_started",
    "operation_completed",
    "resample_started",
    "resample_completed",
    "assembly_started",
    "assembly_completed",
    "loudness_started",
    "loudness_completed",
    "compose_completed",
]
ProgressItemKind = Literal["clip", "silence"]


@dataclass(frozen=True, slots=True)
class CompositionProgress:
    """A producer-neutral observation of composition progress."""

    kind: ProgressEventKind
    completed_items: int
    total_items: int
    item_index: int | None = None
    item_id: str | None = None
    item_kind: ProgressItemKind | None = None
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

__all__ = [
    "CompositionProgress",
    "CompositionProgressCallback",
    "ProgressEventKind",
    "ProgressItemKind",
]
