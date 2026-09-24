import math

import pytest

from audiocompose import (
    AudioValidationError,
    AutomationPoint,
    RatePitchEnvelope,
    operation_from_dict,
)


def test_automation_point_requires_finite_value_and_nonnegative_time() -> None:
    assert AutomationPoint(0, 1).seconds == 0.0
    assert AutomationPoint(0, 1).value == 1.0
    for seconds, value in [(-0.1, 1), (math.nan, 1), (0, math.inf)]:
        with pytest.raises(AudioValidationError):
            AutomationPoint(seconds, value)


def test_rate_pitch_envelope_validates_curves_and_parameters() -> None:
    with pytest.raises(AudioValidationError, match="must not be empty"):
        RatePitchEnvelope()
    with pytest.raises(AudioValidationError, match="start at 0"):
        RatePitchEnvelope(rate=(AutomationPoint(0.1, 1.0),))
    with pytest.raises(AudioValidationError, match="strictly increasing"):
        RatePitchEnvelope(rate=(AutomationPoint(0, 1.0), AutomationPoint(0, 0.9)))
    with pytest.raises(AudioValidationError, match="strictly increasing"):
        RatePitchEnvelope(
            pitch_semitones=(
                AutomationPoint(0, 0),
                AutomationPoint(0.4, 1),
                AutomationPoint(0.3, 2),
            )
        )
    with pytest.raises(AudioValidationError, match="rate values must be > 0"):
        RatePitchEnvelope(rate=(AutomationPoint(0, 0),))
    with pytest.raises(AudioValidationError, match="interpolation"):
        RatePitchEnvelope(pitch_semitones=(AutomationPoint(0, 1),), interpolation="cubic")
    with pytest.raises(AudioValidationError, match="time base"):
        RatePitchEnvelope(pitch_semitones=(AutomationPoint(0, 1),), time_base="source")


def test_transition_builds_combined_rate_and_pitch_curves() -> None:
    operation = RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.85,
        rate_seconds=0.450,
        from_semitones=0.0,
        to_semitones=2.0,
        pitch_seconds=0.300,
    )

    assert operation.rate == (
        AutomationPoint(0.0, 1.0),
        AutomationPoint(0.450, 0.85),
    )
    assert operation.pitch_semitones == (
        AutomationPoint(0.0, 0.0),
        AutomationPoint(0.300, 2.0),
    )


def test_transition_uses_target_for_zero_duration() -> None:
    operation = RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.85,
        from_semitones=0.0,
        to_semitones=2.0,
    )

    assert operation.rate == (AutomationPoint(0.0, 0.85),)
    assert operation.pitch_semitones == (AutomationPoint(0.0, 2.0),)


def test_transition_canonicalizes_unchanged_dimensions() -> None:
    operation = RatePitchEnvelope.transition(
        from_rate=0.85,
        to_rate=0.85,
        rate_seconds=0.4,
        from_semitones=2.0,
        to_semitones=2.0,
        pitch_seconds=0.3,
    )
    assert operation.rate == (AutomationPoint(0.0, 0.85),)
    assert operation.pitch_semitones == (AutomationPoint(0.0, 2.0),)

    with pytest.raises(AudioValidationError, match="must not be empty"):
        RatePitchEnvelope.transition()


def test_envelope_serialization_roundtrips_through_operation_parser() -> None:
    operation = RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.85,
        rate_seconds=0.45,
        from_semitones=0.0,
        to_semitones=2.0,
        pitch_seconds=0.3,
    )
    serialized = operation.to_dict()

    assert serialized == {
        "type": "rate_pitch_envelope",
        "time_base": "output",
        "interpolation": "linear",
        "rate": [
            {"seconds": 0.0, "factor": 1.0},
            {"seconds": 0.45, "factor": 0.85},
        ],
        "pitch": [
            {"seconds": 0.0, "semitones": 0.0},
            {"seconds": 0.3, "semitones": 2.0},
        ],
    }
    assert operation_from_dict(serialized) == operation


def test_operation_parser_rejects_invalid_envelopes() -> None:
    with pytest.raises(AudioValidationError, match="invalid 'rate_pitch_envelope'"):
        operation_from_dict(
            {
                "type": "rate_pitch_envelope",
                "rate": [{"seconds": 0, "factor": 0}],
                "pitch": [],
            }
        )
