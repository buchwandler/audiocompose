from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ._plan import Plan
from .diagnostics import RenderDiagnostic
from .errors import AssemblyError
from .model import (
    AudioFragment,
    RenderedMarker,
    RenderedSegment,
    RenderedUnit,
    RenderResult,
)
from .protocols import FragmentProcessor


def samples_for_duration(seconds: float, sample_rate: int) -> int:
    """Convert a non-negative duration to the nearest whole sample."""

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and >= 0")
    return round(sample_rate * seconds)


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    """Return float32 silence using utterrender's canonical duration rounding."""

    return np.zeros(samples_for_duration(seconds, sample_rate), dtype=np.float32)


def _index_fragments(fragments: Iterable[AudioFragment]) -> dict[str, AudioFragment]:
    indexed: dict[str, AudioFragment] = {}
    for fragment in fragments:
        if fragment.segment_id in indexed:
            raise AssemblyError(f"duplicate fragment for segment {fragment.segment_id!r}")
        indexed[fragment.segment_id] = fragment
    return indexed


def _apply_processors(
    fragment: AudioFragment,
    segment: Any,
    processors: Sequence[FragmentProcessor],
) -> AudioFragment:
    current = fragment
    for processor in processors:
        result = processor(current, segment=segment)
        if not isinstance(result, AudioFragment):
            raise AssemblyError(
                f"fragment processor {processor!r} returned {type(result).__name__}, "
                "expected AudioFragment"
            )
        if result.segment_id != fragment.segment_id:
            raise AssemblyError(
                "fragment processors must preserve segment_id: "
                f"{fragment.segment_id!r} became {result.segment_id!r}"
            )
        current = result
    return current


def assemble(
    plan: Plan,
    fragments: Iterable[AudioFragment],
    *,
    processors: Sequence[FragmentProcessor] = (),
    sample_rate: int | None = None,
    validate_plan: bool = True,
) -> RenderResult:
    """Assemble model-rendered fragments according to a :class:`utterplan.Plan`.

    Each plan segment must have exactly one fragment, including segments that
    intentionally render to zero samples. Resolved pauses are inserted before
    and after the corresponding segment. Marker offsets are resolved only when
    their spoken position maps unambiguously to a known segment boundary.
    """

    if validate_plan:
        plan.validate()

    fragment_by_id = _index_fragments(fragments)
    expected_ids = tuple(segment.id for segment in plan.segments)
    expected_set = set(expected_ids)
    missing = [segment_id for segment_id in expected_ids if segment_id not in fragment_by_id]
    extra = sorted(set(fragment_by_id) - expected_set)
    if missing:
        raise AssemblyError(f"missing fragment(s): {', '.join(missing)}")
    if extra:
        raise AssemblyError(f"fragment(s) not present in plan: {', '.join(extra)}")

    processed: dict[str, AudioFragment] = {}
    for segment in plan.segments:
        processed[segment.id] = _apply_processors(
            fragment_by_id[segment.id], segment, processors
        )

    inferred_rates = {fragment.sample_rate for fragment in processed.values()}
    if sample_rate is None:
        if len(inferred_rates) == 1:
            sample_rate = next(iter(inferred_rates))
        elif not inferred_rates:
            raise AssemblyError("sample_rate is required when assembling an empty plan")
        else:
            raise AssemblyError(
                "all fragments must share one sample rate; got "
                + ", ".join(str(rate) for rate in sorted(inferred_rates))
            )
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AssemblyError("sample_rate must be a positive integer")
    mismatched = [
        fragment.segment_id
        for fragment in processed.values()
        if fragment.sample_rate != sample_rate
    ]
    if mismatched:
        raise AssemblyError(
            f"fragment sample rate differs from output rate {sample_rate}: "
            + ", ".join(mismatched)
        )

    parts: list[np.ndarray] = []
    rendered_segments: list[RenderedSegment] = []
    position_offsets: dict[int, set[int]] = defaultdict(set)
    diagnostics: list[RenderDiagnostic] = []
    cursor = 0

    for segment in plan.segments:
        fragment = processed[segment.id]
        before = samples_for_duration(segment.pause_before.seconds, sample_rate)
        after = samples_for_duration(segment.pause_after.seconds, sample_rate)
        block_start = cursor

        if before:
            parts.append(np.zeros(before, dtype=np.float32))
            cursor += before

        audio_start = cursor
        position_offsets[segment.spoken_start].add(audio_start)
        diagnostics.extend(fragment.diagnostics)
        for alignment in fragment.alignment:
            if alignment.spoken_length == 0:
                continue
            for spoken_position in range(alignment.spoken_start, alignment.spoken_end + 1):
                ratio = (spoken_position - alignment.spoken_start) / alignment.spoken_length
                sample_offset = alignment.sample_start + round(ratio * alignment.sample_length)
                position_offsets[spoken_position].add(audio_start + sample_offset)
        if fragment.audio.size:
            parts.append(fragment.audio)
            cursor += int(fragment.audio.size)
        audio_end = cursor
        position_offsets[segment.spoken_end].add(audio_end)

        if after:
            parts.append(np.zeros(after, dtype=np.float32))
            cursor += after

        rendered_segments.append(
            RenderedSegment(
                segment_id=segment.id,
                block_start_sample=block_start,
                audio_start_sample=audio_start,
                audio_end_sample=audio_end,
                block_end_sample=cursor,
                pause_before_samples=before,
                pause_after_samples=after,
            )
        )

    audio = (
        np.concatenate(parts).astype(np.float32, copy=False)
        if parts
        else np.zeros(0, dtype=np.float32)
    )

    span_by_id = {span.segment_id: span for span in rendered_segments}
    rendered_units: list[RenderedUnit] = []
    for unit in plan.units:
        if unit.segment_ids:
            first = span_by_id[unit.segment_ids[0]]
            last = span_by_id[unit.segment_ids[-1]]
            start_sample = first.block_start_sample
            end_sample = last.block_end_sample
        else:
            start_sample = end_sample = 0
        rendered_units.append(
            RenderedUnit(
                unit_id=unit.id,
                index=unit.index,
                start_sample=start_sample,
                end_sample=end_sample,
                segment_ids=tuple(unit.segment_ids),
            )
        )

    rendered_markers: list[RenderedMarker] = []
    for marker in plan.markers:
        candidates = position_offsets.get(marker.spoken_position, set())
        if len(candidates) == 1:
            rendered_markers.append(
                RenderedMarker(
                    marker_id=marker.id,
                    name=marker.name,
                    char_offset=marker.spoken_position,
                    sample_offset=next(iter(candidates)),
                    timing="resolved",
                )
            )
        elif len(candidates) > 1:
            rendered_markers.append(
                RenderedMarker(
                    marker_id=marker.id,
                    name=marker.name,
                    char_offset=marker.spoken_position,
                    sample_offset=None,
                    timing="unresolved",
                    reason="spoken position maps to multiple audio boundaries",
                )
            )
        else:
            rendered_markers.append(
                RenderedMarker(
                    marker_id=marker.id,
                    name=marker.name,
                    char_offset=marker.spoken_position,
                    sample_offset=None,
                    timing="unresolved",
                    reason="marker is inside a fragment without backend timing",
                )
            )

    metadata = {
        "utterplan_producer": dict(plan.producer),
        "utterplan_schema_version": plan.schema_version,
    }
    return RenderResult(
        audio=audio,
        sample_rate=sample_rate,
        plan_id=plan.plan_id,
        segments=tuple(rendered_segments),
        units=tuple(rendered_units),
        markers=tuple(rendered_markers),
        diagnostics=tuple(diagnostics),
        audio_in_range=bool(np.all(np.abs(audio) <= 1.0)),
        warnings=tuple(plan.warnings),
        metadata=metadata,
    )
