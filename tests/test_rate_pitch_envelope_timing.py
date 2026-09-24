import numpy as np
import pytest
from audiosig import apply_speech_effects_envelope

from audiocompose import (
    AutomationPoint,
    FadeIn,
    FadeOut,
    Gain,
    PitchShift,
    RatePitchEnvelope,
    Tempo,
)
from audiocompose._automation import output_seconds_for_source_seconds


def test_constant_rate_maps_length_and_source_offsets() -> None:
    operation = RatePitchEnvelope(rate=(AutomationPoint(0, 0.5),))

    assert operation.output_length(1000, 1000) == 2000
    assert operation.map_offset_at_rate(250, 1000, 1000) == 500


def test_linear_rate_ramp_maps_source_time_analytically() -> None:
    points = ((0.0, 1.0), (1.0, 2.0))
    operation = RatePitchEnvelope(rate=tuple(AutomationPoint(*point) for point in points))

    assert output_seconds_for_source_seconds(1.5, points) == pytest.approx(1.0)
    assert operation.output_length(2500, 1000) == 1500
    assert operation.map_offset_at_rate(1500, 2500, 1000) == 1000


def test_rate_is_held_after_final_point() -> None:
    points = ((0.0, 1.0), (1.0, 2.0))

    assert output_seconds_for_source_seconds(3.5, points) == pytest.approx(2.0)


def test_multi_point_rate_mapping_accumulates_segment_source_time() -> None:
    points = ((0.0, 1.0), (1.0, 2.0), (2.0, 1.0))

    assert output_seconds_for_source_seconds(1.5, points) == pytest.approx(1.0)
    assert output_seconds_for_source_seconds(3.0, points) == pytest.approx(2.0)
    assert output_seconds_for_source_seconds(3.5, points) == pytest.approx(2.5)


def test_pitch_only_envelope_preserves_length_and_coordinates() -> None:
    operation = RatePitchEnvelope(pitch_semitones=(AutomationPoint(0, 0), AutomationPoint(0.3, 2)))

    assert operation.output_length(1234, 24000) == 1234
    assert operation.map_offset_at_rate(456, 1234, 24000) == 456


@pytest.mark.parametrize(
    ("input_frames", "rate_points", "pitch_points"),
    [
        (24000, ((0.0, 0.5),), ()),
        (24000, ((0.0, 1.15),), ()),
        (24000, ((0.0, 1.0), (0.45, 0.85)), ()),
        (24000, ((0.0, 0.85), (0.45, 1.0)), ()),
        (24000, ((0.0, 1.0), (0.1, 0.95), (0.25, 0.85)), ()),
        (24000, (), ((0.0, 0.0), (0.3, 2.0))),
        (24000, ((0.0, 1.0), (0.45, 0.85)), ((0.0, 0.0), (0.3, 2.0))),
        (8, ((0.0, 1.0), (0.45, 0.85)), ((0.0, 0.0), (0.3, 2.0))),
        (48000, ((0.0, 1.0), (0.45, 0.85)), ()),
    ],
)
def test_predicted_output_length_matches_audiosig(
    input_frames: int,
    rate_points: tuple[tuple[float, float], ...],
    pitch_points: tuple[tuple[float, float], ...],
) -> None:
    operation = RatePitchEnvelope(
        rate=tuple(AutomationPoint(seconds, factor) for seconds, factor in rate_points),
        pitch_semitones=tuple(
            AutomationPoint(seconds, semitones) for seconds, semitones in pitch_points
        ),
    )
    source = np.zeros(input_frames, dtype=np.float32)
    actual = apply_speech_effects_envelope(
        source,
        sample_rate=24000,
        rate_points=rate_points,
        pitch_points=pitch_points,
        time_base="output",
        interpolation="linear",
    )

    assert operation.output_length(input_frames, 24000) == len(actual)


def test_generic_timing_hooks_preserve_existing_operations() -> None:
    for operation in (Gain(1.0), PitchShift(1.0), FadeIn(0.1), FadeOut(0.1)):
        assert operation.output_length(1001, 24000) == 1001
        assert operation.map_offset_at_rate(250, 1001, 24000) == 250

    tempo = Tempo(0.85)
    assert tempo.output_length(1001, 24000) == round(1001 / 0.85)
    assert tempo.map_offset_at_rate(250, 1001, 24000) == round(250 / 0.85)
