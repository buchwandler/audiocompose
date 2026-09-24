from __future__ import annotations

import numpy as np

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    AutomationPoint,
    Composer,
    FadeIn,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    RatePitchEnvelope,
    Silence,
    Tempo,
)


def _output(sample_rate: int) -> OutputPolicy:
    return OutputPolicy(
        sample_rate=sample_rate,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
    )


def test_tempo_maps_markers_and_spans() -> None:
    clip = AudioClip(
        "clip",
        AudioBufferSource(np.zeros(100), 100),
        operations=(Tempo(2.0),),
        anchors=(AudioAnchor("middle", 50),),
        spans=(AudioSpan(10, 20, 25, 75, id="span"),),
    )
    result = Composer(sample_rate=100).compose(AudioJob((clip,), output=_output(100)))

    assert len(result.audio) == 50
    assert result.markers[0].sample_offset == 25
    assert (result.spans[0].sample_start, result.spans[0].sample_end) == (12, 38)
    assert result.spans[0].id == "span"


def test_fused_tempos_keep_sequential_marker_and_span_mapping() -> None:
    operations = (Tempo(0.85), PitchShift(-3.0), Tempo(1.2))
    clip = AudioClip(
        "clip",
        AudioBufferSource(np.zeros(1_000, dtype=np.float32), 100),
        operations=operations,
        anchors=(AudioAnchor("middle", 450),),
        spans=(AudioSpan(100, 200, 200, 800, id="fused"),),
    )
    result = Composer(sample_rate=100).compose(AudioJob((clip,), output=_output(100)))

    assert len(result.audio) == round(1_000 / (0.85 * 1.2))
    assert result.markers[0].sample_offset == round(round(450 / 0.85) / 1.2)
    assert (result.spans[0].sample_start, result.spans[0].sample_end) == (
        round(round(200 / 0.85) / 1.2),
        round(round(800 / 0.85) / 1.2),
    )


def test_duration_preserving_operations_keep_coordinates_stable() -> None:
    clip = AudioClip(
        "clip",
        AudioBufferSource(np.zeros(100), 100),
        operations=(Gain(3.0), FadeIn(0.1), PitchShift(1.0)),
        anchors=(AudioAnchor("middle", 50),),
        spans=(AudioSpan(10, 20, 25, 75),),
    )
    result = Composer(sample_rate=100).compose(AudioJob((clip,), output=_output(100)))

    assert len(result.audio) == 100
    assert result.markers[0].sample_offset == 50
    assert (result.spans[0].sample_start, result.spans[0].sample_end) == (25, 75)


def test_resampling_maps_coordinates_consistently() -> None:
    clip = AudioClip(
        "clip",
        AudioBufferSource(np.zeros(100), 100),
        anchors=(AudioAnchor("middle", 50),),
        spans=(AudioSpan(10, 20, 25, 75, id="resampled"),),
    )
    result = Composer(sample_rate=200).compose(AudioJob((clip,), output=_output(200)))

    assert len(result.audio) == 200
    assert result.markers[0].sample_offset == 100
    assert (result.spans[0].sample_start, result.spans[0].sample_end) == (50, 150)


def test_rate_envelope_maps_anchors_and_spans_before_resampling(monkeypatch) -> None:
    def render_at_predicted_length(audio, sample_rate, operation):
        return np.zeros(operation.output_length(len(audio), sample_rate), dtype=np.float32)

    monkeypatch.setattr("audiocompose.composer.apply_operation", render_at_predicted_length)
    operation = RatePitchEnvelope(rate=(AutomationPoint(0, 1.0), AutomationPoint(1.0, 2.0)))
    clip = AudioClip(
        "curve",
        AudioBufferSource(np.zeros(3000, dtype=np.float32), 1000),
        operations=(operation,),
        anchors=(
            AudioAnchor("before_knot", 500),
            AudioAnchor("at_knot", 1500),
            AudioAnchor("after_knot", 2500),
        ),
        spans=(AudioSpan(10, 20, 500, 2500, id="curve-span"),),
    )

    result = Composer(sample_rate=2000).compose(AudioJob((clip,), output=_output(1000)))

    assert len(result.audio) == 3500
    assert [marker.sample_offset for marker in result.markers] == [828, 2000, 3000]
    assert (result.spans[0].sample_start, result.spans[0].sample_end) == (828, 3000)
    assert result.spans[0].id == "curve-span"
    assert (result.items[0].start_sample, result.items[0].end_sample) == (0, 3500)


def test_item_offsets_are_added_once() -> None:
    first = AudioClip(
        "first",
        AudioBufferSource(np.zeros(10), 10),
        anchors=(AudioAnchor("end", 10),),
    )
    second = AudioClip(
        "second",
        AudioBufferSource(np.zeros(10), 10),
        anchors=(AudioAnchor("start", 0),),
    )
    result = Composer(sample_rate=10).compose(
        AudioJob((first, Silence("pause", 1.0), second), output=_output(10))
    )

    assert [marker.sample_offset for marker in result.markers] == [10, 20]
    assert [(item.start_sample, item.end_sample) for item in result.items] == [
        (0, 10),
        (10, 20),
        (20, 30),
    ]
