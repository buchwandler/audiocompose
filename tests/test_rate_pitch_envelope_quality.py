from __future__ import annotations

import numpy as np
from audiosig import apply_speech_effects_envelope

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
    RatePitchEnvelope,
    resample_audio,
)


def _output(sample_rate: int) -> OutputPolicy:
    return OutputPolicy(
        sample_rate=sample_rate,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
    )


def _voiced_signal(sample_rate: int, duration: float = 0.7) -> np.ndarray:
    time = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    fundamental = 170.0 + 12.0 * np.sin(2.0 * np.pi * 1.7 * time)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / sample_rate
    signal = 0.22 * np.sin(phase) + 0.10 * np.sin(2.0 * phase) + 0.04 * np.sin(3.0 * phase)
    signal *= 0.5 + 0.5 * np.sin(np.pi * time / duration) ** 2
    return signal.astype(np.float32)


def _transition() -> RatePitchEnvelope:
    return RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.85,
        rate_seconds=0.45,
        from_semitones=0.0,
        to_semitones=2.0,
        pitch_seconds=0.3,
    )


def test_public_composer_renders_envelope_with_stable_identity_and_resampled_spans() -> None:
    source_rate = 16_000
    output_rate = 24_000
    source = _voiced_signal(source_rate)
    operation = _transition()
    metadata = {"readio.segment_id": "seg-000481", "custom": {"opaque": True}}
    clip = AudioClip(
        "seg-000481",
        AudioBufferSource(source, source_rate),
        operations=(operation,),
        anchors=(AudioAnchor("before", 3200), AudioAnchor("after", 9600)),
        spans=(AudioSpan(10, 20, 3000, 10000, id="span-481", metadata={"word": 1}),),
        metadata=metadata,
    )
    events = []
    job = AudioJob((clip,), producer={"name": "test"}, output=_output(output_rate))

    result = Composer(sample_rate=output_rate).compose(job, on_progress=events.append)

    direct = apply_speech_effects_envelope(
        source,
        sample_rate=source_rate,
        rate_points=tuple((point.seconds, point.value) for point in operation.rate),
        pitch_points=tuple((point.seconds, point.value) for point in operation.pitch_semitones),
        time_base="output",
        interpolation="linear",
    )
    expected = resample_audio(direct, source_rate, output_rate)

    assert len(direct) == operation.output_length(len(source), source_rate)
    assert result.sample_rate == output_rate
    assert np.array_equal(result.audio, expected)
    assert result.audio.dtype == np.float32
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) > 0.0
    assert result.items[0].item_id == "seg-000481"
    assert (result.items[0].start_sample, result.items[0].end_sample) == (
        0,
        len(expected),
    )
    assert [marker.id for marker in result.markers] == ["before", "after"]
    assert [marker.sample_offset for marker in result.markers] == [
        round(
            operation.map_offset_at_rate(3200, len(source), source_rate) * output_rate / source_rate
        ),
        round(
            operation.map_offset_at_rate(9600, len(source), source_rate) * output_rate / source_rate
        ),
    ]
    span = result.spans[0]
    assert span.id == "span-481"
    assert span.metadata == {"word": 1}
    assert (span.sample_start, span.sample_end) == (
        round(
            operation.map_offset_at_rate(3000, len(source), source_rate) * output_rate / source_rate
        ),
        round(
            operation.map_offset_at_rate(10000, len(source), source_rate)
            * output_rate
            / source_rate
        ),
    )
    item_events = [event for event in events if event.item_id == "seg-000481"]
    assert item_events
    assert all(event.item_metadata == metadata for event in item_events)


def test_public_composer_preserves_silence_with_rate_and_pitch_envelopes() -> None:
    sample_rate = 24_000
    source = np.zeros(round(sample_rate * 0.2), dtype=np.float32)
    operation = _transition()
    clip = AudioClip(
        "silent-segment",
        AudioBufferSource(source, sample_rate),
        operations=(operation,),
    )

    result = Composer(sample_rate=sample_rate).compose(
        AudioJob((clip,), output=_output(sample_rate))
    )

    assert len(result.audio) == operation.output_length(len(source), sample_rate)
    assert np.isfinite(result.audio).all()
    assert np.count_nonzero(result.audio) == 0


def test_short_clip_renders_only_the_available_transition_portion() -> None:
    sample_rate = 24_000
    source = _voiced_signal(sample_rate, duration=0.12)
    operation = _transition()
    result = Composer(sample_rate=sample_rate).compose(
        AudioJob(
            (
                AudioClip(
                    "short-segment",
                    AudioBufferSource(source, sample_rate),
                    operations=(operation,),
                ),
            ),
            output=_output(sample_rate),
        )
    )

    assert operation.rate[-1].seconds == 0.45
    assert operation.pitch_semitones[-1].seconds == 0.3
    assert len(result.audio) == operation.output_length(len(source), sample_rate)
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) > 0.0


def test_envelope_knots_do_not_create_composition_boundary_jumps() -> None:
    sample_rate = 24_000
    source = _voiced_signal(sample_rate)
    result = Composer(sample_rate=sample_rate).compose(
        AudioJob(
            (
                AudioClip(
                    "boundary-check",
                    AudioBufferSource(source, sample_rate),
                    operations=(_transition(),),
                ),
            ),
            output=_output(sample_rate),
        )
    )

    for point in (0.3, 0.45):
        boundary = round(point * sample_rate)
        local = result.audio[max(0, boundary - 400) : boundary + 400]
        typical_jump = float(np.percentile(np.abs(np.diff(local)), 95.0))
        boundary_jump = abs(float(result.audio[boundary] - result.audio[boundary - 1]))
        assert boundary_jump <= 8.0 * typical_jump
