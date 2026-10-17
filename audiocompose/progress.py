from __future__ import annotations

import math
import time
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
class CompositionEstimate:
    """Deterministic output-geometry and execution-work estimate for a job.

    Work units are an internal planning metric, not a wall-clock prediction or
    part of AudioJob identity. The model may evolve between minor releases.
    """

    sample_rate: int
    item_output_frames: tuple[int | None, ...]
    total_output_frames: int | None
    total_audio_seconds: float | None
    total_work_units: float | None
    unknown_items: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompositionProgress:
    """A producer-neutral runtime observation of composition progress.

    Work fields describe an approximate frame-pass model, not audio semantics or
    wall-clock time. ``fraction`` is absent when total work is unknown; work
    units are never serialized into AudioJob identity or provenance."""

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

    completed_work_units: float | None = None
    total_work_units: float | None = None
    fraction: float | None = None


@dataclass(frozen=True, slots=True)
class ProgressTiming:
    """Live elapsed-time and ETA view derived from composition work progress."""

    fraction: float | None
    percent: float | None
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    estimated_total_seconds: float | None
    status: Literal["indeterminate", "warming_up", "estimated", "complete"]


class ProgressEstimator:
    """Estimate wall-clock duration from observed frame-work throughput.

    The estimator is process-local and never persists machine performance. A
    caller may supply its own ``initial_work_rate`` as work units per second.
    """

    _EWMA_ALPHA = 0.3
    _MAX_ETA_SECONDS = 365 * 24 * 60 * 60

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        initial_work_rate: float | None = None,
    ) -> None:
        if initial_work_rate is not None and (
            isinstance(initial_work_rate, bool)
            or not isinstance(initial_work_rate, (int, float))
            or not math.isfinite(initial_work_rate)
            or initial_work_rate <= 0
        ):
            raise ValueError("initial_work_rate must be a positive finite number or None")
        self._clock = clock
        now = float(clock())
        self._started_at = now if math.isfinite(now) else 0.0
        self._last_now = self._started_at
        self._last_sample_time: float | None = None
        self._last_sample_work: float | None = None
        self._work_rate = float(initial_work_rate) if initial_work_rate is not None else None
        self._fraction: float | None = None
        self._completed_work_units: float | None = None

    def update(self, event: CompositionProgress) -> ProgressTiming:
        now = float(self._clock())
        if not math.isfinite(now) or now < self._last_now:
            now = self._last_now
        self._last_now = now
        elapsed = max(0.0, now - self._started_at)

        fraction = event.fraction
        if fraction is None and event.total_work_units is not None and event.total_work_units > 0:
            if event.completed_work_units is not None:
                fraction = event.completed_work_units / event.total_work_units
        if fraction is not None and math.isfinite(fraction):
            fraction = min(1.0, max(0.0, fraction))
            if self._fraction is not None:
                fraction = max(self._fraction, fraction)
            self._fraction = fraction
        else:
            fraction = None
        percent = fraction * 100.0 if fraction is not None else None

        total_work = event.total_work_units
        completed_work = event.completed_work_units
        if completed_work is not None and math.isfinite(completed_work):
            if self._completed_work_units is not None:
                completed_work = max(self._completed_work_units, completed_work)
            self._completed_work_units = completed_work
        elif self._completed_work_units is not None:
            completed_work = self._completed_work_units
        if (
            total_work is not None
            and math.isfinite(total_work)
            and total_work > 0
            and completed_work is not None
            and math.isfinite(completed_work)
        ):
            if self._last_sample_time is None:
                self._last_sample_time = now
                self._last_sample_work = completed_work
            else:
                delta_time = now - self._last_sample_time
                delta_work = completed_work - (self._last_sample_work or 0.0)
                if delta_time > 0:
                    if delta_work > 0:
                        instant_rate = delta_work / delta_time
                        if math.isfinite(instant_rate) and instant_rate > 0:
                            if self._work_rate is None:
                                self._work_rate = instant_rate
                            else:
                                self._work_rate = (
                                    self._EWMA_ALPHA * instant_rate
                                    + (1.0 - self._EWMA_ALPHA) * self._work_rate
                                )
                    self._last_sample_time = now
                    self._last_sample_work = completed_work

        complete = event.kind == "compose_completed" or fraction == 1.0
        remaining: float | None = None
        estimated_total: float | None = None
        if complete:
            remaining = 0.0
            estimated_total = elapsed
            status: Literal["indeterminate", "warming_up", "estimated", "complete"] = "complete"
        elif total_work is None or not math.isfinite(total_work):
            status = "indeterminate"
        else:
            rate = self._work_rate
            if rate is None or not math.isfinite(rate) or rate <= 0:
                status = "warming_up"
            else:
                completed = completed_work if completed_work is not None else 0.0
                remaining_work = max(0.0, total_work - completed)
                candidate = remaining_work / rate
                candidate_total = elapsed + candidate
                if (
                    math.isfinite(candidate)
                    and math.isfinite(candidate_total)
                    and candidate <= self._MAX_ETA_SECONDS
                    and candidate_total <= self._MAX_ETA_SECONDS
                ):
                    remaining = candidate
                    estimated_total = candidate_total
                    status = "estimated"
                else:
                    status = "warming_up"

        return ProgressTiming(
            fraction=fraction,
            percent=percent,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
            estimated_total_seconds=estimated_total,
            status=status,
        )


CompositionProgressCallback = Callable[[CompositionProgress], None]

__all__ = [
    "CompositionProgress",
    "CompositionProgressCallback",
    "ProgressEventKind",
    "CompositionEstimate",
    "ProgressEstimator",
    "ProgressTiming",
    "ProgressItemKind",
]
