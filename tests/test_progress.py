from __future__ import annotations

import numpy as np
import pytest

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioFileSource,
    AudioJob,
    AutomationPoint,
    Composer,
    FadeIn,
    FadeOut,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    RatePitchEnvelope,
    Silence,
    Tempo,
)
from audiocompose.wav import write_wav


def make_job(*items: AudioClip | Silence, sample_rate: int = 100) -> AudioJob:
    return AudioJob(
        items,
        output=OutputPolicy(
            sample_rate=sample_rate,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )


def test_no_callback_preserves_composition_output() -> None:
    job = make_job(
        AudioClip(
            "clip",
            AudioBufferSource(np.linspace(-0.5, 0.5, 100, dtype=np.float32), 100),
            (Gain(2.0), Tempo(1.0)),
            metadata={"segment_id": "seg-0042"},
        ),
        Silence("pause", 0.1),
    )

    plain = Composer().compose(job)
    events = []
    observed = Composer().compose(job, on_progress=events.append)

    assert plain.sample_rate == observed.sample_rate
    assert np.array_equal(plain.audio, observed.audio)
    assert plain.items == observed.items
    assert plain.markers == observed.markers
    assert plain.spans == observed.spans
    assert events


def test_lifecycle_sequence_for_clip_without_operations() -> None:
    events = []
    Composer().compose(
        make_job(AudioClip("clip", AudioBufferSource(np.ones(10, dtype=np.float32), 100))),
        on_progress=events.append,
    )

    assert [event.kind for event in events] == [
        "compose_started",
        "item_started",
        "source_load_started",
        "source_load_completed",
        "item_completed",
        "assembly_started",
        "assembly_completed",
        "loudness_started",
        "loudness_completed",
        "compose_completed",
    ]
    assert events[0].item_index is None
    assert events[1].item_index == 0
    assert events[1].completed_items == 0
    assert events[4].completed_items == 1
    assert events[-1].completed_items == 1


def test_operation_events_preserve_order_and_identity() -> None:
    operations = (Gain(2.0), Tempo(1.0), PitchShift(0.0), FadeIn(0.1), FadeOut(0.1))
    events = []
    job = make_job(
        AudioClip("clip", AudioBufferSource(np.ones(100, dtype=np.float32), 100), operations),
    )
    Composer().compose(job, on_progress=events.append)

    started = [event for event in events if event.kind == "operation_started"]
    completed = [event for event in events if event.kind == "operation_completed"]
    assert [event.operation for event in started] == [
        operation.to_dict() for operation in operations
    ]
    assert [event.operation_index for event in started] == list(range(len(operations)))
    assert [event.operation for event in completed] == [
        operation.to_dict() for operation in operations
    ]


def test_resampling_events_contain_both_rates_and_frame_counts() -> None:
    events = []
    Composer().compose(
        make_job(
            AudioClip("clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4)),
            sample_rate=8,
        ),
        on_progress=events.append,
    )

    resampling = [event for event in events if event.kind.startswith("resample_")]
    assert [event.kind for event in resampling] == ["resample_started", "resample_completed"]
    assert resampling[0].source_sample_rate == 4
    assert resampling[0].target_sample_rate == 8
    assert resampling[0].input_frames == 4
    assert resampling[1].input_frames == 4
    assert resampling[1].output_frames == 8


def test_matching_rates_do_not_emit_resampling_events() -> None:
    events = []
    Composer().compose(
        make_job(
            AudioClip("clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4)), sample_rate=4
        ),
        on_progress=events.append,
    )

    assert not any(event.kind.startswith("resample_") for event in events)


def test_silence_has_item_lifecycle_without_source_or_operation_events() -> None:
    events = []
    Composer().compose(make_job(Silence("pause", 0.5)), on_progress=events.append)

    kinds = [event.kind for event in events]
    assert kinds[:3] == ["compose_started", "item_started", "item_completed"]
    assert not any(kind.startswith("source_") for kind in kinds)
    assert not any(kind.startswith("operation_") for kind in kinds)
    assert events[1].item_kind == "silence"
    assert events[1].details == {"seconds": 0.5}


def test_progress_durations_include_transformed_audio_and_silence() -> None:
    events = []
    job = make_job(
        AudioClip(
            "clip",
            AudioBufferSource(np.ones(4000, dtype=np.float32), 4000),
            (Tempo(2.0),),
        ),
        Silence("pause", 0.125),
        sample_rate=4000,
    )

    result = Composer(sample_rate=4000).compose(job, on_progress=events.append)

    assert len(result.audio) == 2500
    assert events[0].completed_audio_seconds == 0.0
    assert events[0].total_audio_seconds == pytest.approx(0.625)
    completed = [event for event in events if event.kind == "item_completed"]
    assert completed[0].completed_audio_seconds == pytest.approx(0.5)
    assert completed[0].total_audio_seconds == pytest.approx(0.625)
    assert completed[1].completed_audio_seconds == pytest.approx(0.625)
    assert events[-1].completed_audio_seconds == pytest.approx(0.625)


def test_progress_duration_matches_fused_tempo_rounding() -> None:
    sample_rate = 24_000
    events = []
    job = make_job(
        AudioClip(
            "clip",
            AudioBufferSource(np.ones(10_010, dtype=np.float32), sample_rate),
            (Tempo(0.85), Tempo(0.85)),
        ),
        sample_rate=sample_rate,
    )

    result = Composer(sample_rate=sample_rate).compose(job, on_progress=events.append)

    assert len(result.audio) == round(10_010 / (0.85 * 0.85))
    assert events[0].total_audio_seconds == pytest.approx(len(result.audio) / sample_rate)
    assert events[-1].completed_audio_seconds == pytest.approx(len(result.audio) / sample_rate)


def test_progress_keeps_unknown_total_unknown_but_reports_completed_duration(tmp_path) -> None:
    path = tmp_path / "source.wav"
    write_wav(path, np.full(10, 0.2, dtype=np.float32), 10)
    events = []
    job = make_job(AudioClip("clip", AudioFileSource(path)), sample_rate=10)

    Composer(sample_rate=10).compose(job, on_progress=events.append)

    assert events[0].total_audio_seconds is None
    completed = next(event for event in events if event.kind == "item_completed")
    assert completed.completed_audio_seconds == pytest.approx(1.0)
    assert completed.total_audio_seconds is None


def test_metadata_is_forwarded_without_special_cases() -> None:
    events = []
    metadata = {"producer": "test", "segment_id": "seg-0042"}
    Composer().compose(
        make_job(
            AudioClip(
                "clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4), metadata=metadata
            ),
        ),
        on_progress=events.append,
    )

    item_events = [event for event in events if event.item_index == 0]
    assert item_events
    assert all(event.item_metadata == metadata for event in item_events)


def test_wav_helpers_forward_progress_callback(tmp_path) -> None:
    job = make_job(AudioClip("clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4)))
    direct_events = []
    Composer().to_wav(job, tmp_path / "direct.wav", on_progress=direct_events.append)
    assert direct_events[-1].kind == "compose_completed"

    manifest = job.save(tmp_path / "job.audiojob")
    manifest_events = []
    Composer().compose_to_wav(
        manifest,
        tmp_path / "manifest.wav",
        on_progress=manifest_events.append,
    )
    assert manifest_events[-1].kind == "compose_completed"


def test_compose_to_wav_reads_each_persisted_source_once(tmp_path, monkeypatch) -> None:
    job = make_job(AudioClip("clip", AudioBufferSource(np.ones(100, dtype=np.float32), 100)))
    manifest = job.save(tmp_path / "bundle.audiojob")
    original_load = AudioFileSource.load
    load_count = 0

    def counted_load(source):
        nonlocal load_count
        load_count += 1
        return original_load(source)

    monkeypatch.setattr(AudioFileSource, "load", counted_load)
    Composer().compose_to_wav(manifest, tmp_path / "output.wav")

    assert load_count == 1


def test_callback_exception_propagates() -> None:
    job = make_job(AudioClip("clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4)))

    def fail(event) -> None:
        raise RuntimeError(event.kind)

    with pytest.raises(RuntimeError, match="compose_started"):
        Composer().compose(job, on_progress=fail)


def test_composer_does_not_print_progress(capsys) -> None:
    job = make_job(AudioClip("clip", AudioBufferSource(np.ones(4, dtype=np.float32), 4)))
    Composer().compose(job)
    Composer().compose(job, on_progress=lambda event: None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_fused_operation_progress_preserves_logical_events_and_group_frames() -> None:
    operations = (Tempo(0.85), PitchShift(-3.0), Gain(-2.0))
    events = []
    source = np.ones(24_000, dtype=np.float32)
    job = make_job(
        AudioClip("clip", AudioBufferSource(source, 24_000), operations),
        sample_rate=24_000,
    )

    Composer(sample_rate=24_000).compose(job, on_progress=events.append)

    operation_events = [event for event in events if event.kind.startswith("operation_")]
    assert [event.kind for event in operation_events] == [
        "operation_started",
        "operation_started",
        "operation_completed",
        "operation_completed",
        "operation_started",
        "operation_completed",
    ]
    started = [event for event in operation_events if event.kind == "operation_started"]
    completed = [event for event in operation_events if event.kind == "operation_completed"]
    assert [event.operation_index for event in started] == [0, 1, 2]
    assert [event.operation for event in started] == [
        operation.to_dict() for operation in operations
    ]
    assert [event.operation for event in completed] == [
        operation.to_dict() for operation in operations
    ]

    fused_events = operation_events[:4]
    assert all(
        event.details == {"processing_group_start": 0, "processing_group_size": 2}
        for event in fused_events
    )
    assert [event.input_frames for event in fused_events] == [24_000] * 4
    assert [event.output_frames for event in fused_events[2:]] == [round(24_000 / 0.85)] * 2


def test_rate_pitch_envelope_delegates_once_and_reports_operation_frames(monkeypatch) -> None:
    calls = []

    def fake_time_stretch(audio, factor, *, sample_rate, method):
        assert factor == 0.5
        assert sample_rate == 24
        assert method == "wsola"
        return np.repeat(audio, 2)

    def fake_envelope(audio, **kwargs):
        calls.append((len(audio), kwargs))
        return np.repeat(audio, 2)

    monkeypatch.setattr("audiocompose.operations.time_stretch", fake_time_stretch)
    monkeypatch.setattr("audiocompose.operations.apply_speech_effects_envelope", fake_envelope)
    envelope = RatePitchEnvelope(rate=(AutomationPoint(0, 0.5),))
    operations = (Tempo(0.5), envelope)
    events = []
    source = np.ones(24, dtype=np.float32)
    job = make_job(
        AudioClip("seg-000481", AudioBufferSource(source, 24), operations),
        sample_rate=24,
    )

    result = Composer(sample_rate=24).compose(job, on_progress=events.append)

    assert len(result.audio) == 96
    assert calls == [
        (
            48,
            {
                "sample_rate": 24,
                "rate_points": ((0.0, 0.5),),
                "pitch_points": (),
                "time_base": "output",
                "interpolation": "linear",
                "method": "wsola",
            },
        )
    ]
    started = [
        event
        for event in events
        if event.kind == "operation_started" and event.operation["type"] == "rate_pitch_envelope"
    ]
    completed = [
        event
        for event in events
        if event.kind == "operation_completed" and event.operation["type"] == "rate_pitch_envelope"
    ]
    assert len(started) == len(completed) == 1
    assert started[0].item_id == completed[0].item_id == "seg-000481"
    assert started[0].operation_index == completed[0].operation_index == 1
    assert started[0].operation_count == completed[0].operation_count == 2
    assert started[0].input_frames == completed[0].input_frames == 48
    assert completed[0].output_frames == 96
