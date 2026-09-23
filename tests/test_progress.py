from __future__ import annotations

import numpy as np
import pytest

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    Composer,
    FadeIn,
    FadeOut,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    Silence,
    Tempo,
)


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
