from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence

from audiosig import AudioSignalError, speech_effects_output_frames

from .errors import AudioValidationError

RatePoints = Sequence[tuple[float, float]]


def output_seconds_for_source_seconds(
    source_seconds: float,
    rate_points: RatePoints,
) -> float:
    """Invert an output-time linear rate curve for source time."""
    if not math.isfinite(source_seconds) or source_seconds < 0:
        raise AudioValidationError("source time must be finite and >= 0")
    if not rate_points:
        return source_seconds

    times = [point[0] for point in rate_points]
    rates = [point[1] for point in rate_points]
    source_knots = [0.0]
    for index in range(len(times) - 1):
        duration = times[index + 1] - times[index]
        average_rate = 0.5 * rates[index] + 0.5 * rates[index + 1]
        source_knots.append(source_knots[-1] + duration * average_rate)

    index = bisect_right(source_knots, source_seconds) - 1
    if index >= len(times) - 1:
        output_seconds = times[-1] + (source_seconds - source_knots[-1]) / rates[-1]
    else:
        duration = times[index + 1] - times[index]
        source_delta = source_seconds - source_knots[index]
        scale = max(rates[index], rates[index + 1])
        normalized_start = rates[index] / scale
        normalized_end = rates[index + 1] / scale
        normalized_delta = (source_delta / scale) / duration
        discriminant = (
            normalized_start * normalized_start
            + 2.0 * (normalized_end - normalized_start) * normalized_delta
        )
        if discriminant < 0.0 or not math.isfinite(discriminant):
            raise AudioValidationError("rate map inverse exceeds finite timing range")
        denominator = normalized_start + math.sqrt(discriminant)
        fraction = 2.0 * normalized_delta / denominator
        output_seconds = times[index] + duration * fraction

    if not math.isfinite(output_seconds):
        raise AudioValidationError("rate map inverse exceeds finite timing range")
    return output_seconds


def map_source_frame(
    source_offset: int,
    sample_rate: int,
    rate_points: RatePoints,
) -> int:
    output_seconds = output_seconds_for_source_seconds(
        source_offset / sample_rate,
        rate_points,
    )
    return round(output_seconds * sample_rate)


def output_frames_for_input_frames(
    input_frames: int,
    sample_rate: int,
    rate_points: RatePoints,
) -> int:
    try:
        return speech_effects_output_frames(
            input_frames,
            sample_rate=sample_rate,
            rate_points=rate_points,
        )
    except AudioSignalError as exc:
        raise AudioValidationError(str(exc)) from exc
