from __future__ import annotations

import numpy as np
import pytest

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
    Silence,
)
from audiocompose.errors import CompositionError
from audiocompose.loudness import apply_complete_output_loudness


def _tone(sample_rate: int = 1000, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 100 * time)).astype(np.float32)


def test_loudness_result_exposes_before_after_and_gain_diagnostics() -> None:
    result = apply_complete_output_loudness(
        _tone(),
        1000,
        LoudnessPolicy(target_lufs=-20.0, true_peak_ceiling_dbtp=-1.0),
    )

    assert result.before.integrated_lufs is not None
    assert result.after.integrated_lufs is not None
    assert result.requested_gain_db != 0.0
    assert result.measured_lufs == result.before.integrated_lufs
    assert result.true_peak_dbtp == result.after.true_peak_dbtp


def test_silence_reports_non_normalizable_loudness() -> None:
    result = apply_complete_output_loudness(
        np.zeros(1000, dtype=np.float32),
        1000,
        LoudnessPolicy(target_lufs=-20.0),
    )

    assert result.before.integrated_lufs is None
    assert result.after.integrated_lufs is None
    assert result.warning is not None
    assert not result.target_reached


def test_integrated_loudness_requires_a_complete_400ms_block() -> None:
    policy = LoudnessPolicy(target_lufs=-20.0, true_peak_ceiling_dbtp=None)
    short = apply_complete_output_loudness(_tone(seconds=0.399), 1000, policy)
    boundary = apply_complete_output_loudness(_tone(seconds=0.4), 1000, policy)

    assert short.before.integrated_lufs is None
    assert short.before.sample_peak_dbfs is not None
    assert short.before.true_peak_dbtp is not None
    assert short.requested_gain_db == 0.0
    assert boundary.before.integrated_lufs is not None


def test_peak_ceiling_limitation_is_reported() -> None:
    result = apply_complete_output_loudness(
        np.full(1000, 0.5, dtype=np.float32),
        1000,
        LoudnessPolicy(target_lufs=0.0, true_peak_ceiling_dbtp=-12.0),
    )

    assert result.requested_gain_db > result.applied_gain_db
    assert result.warning == "true-peak ceiling limited the requested loudness target"
    assert not result.target_reached


def test_peak_policy_error_rejects_unreachable_target() -> None:
    with pytest.raises(CompositionError, match="true-peak ceiling"):
        apply_complete_output_loudness(
            np.full(1000, 0.5, dtype=np.float32),
            1000,
            LoudnessPolicy(target_lufs=0.0, true_peak_ceiling_dbtp=-12.0, peak_policy="error"),
        )


def _output_policy(enabled: bool) -> OutputPolicy:
    return OutputPolicy(
        sample_rate=1000,
        loudness=LoudnessPolicy(
            target_lufs=-20.0 if enabled else None,
            true_peak_ceiling_dbtp=None,
        ),
    )


def test_composition_exposes_loudness_and_keeps_coordinates_stable() -> None:
    audio = _tone()
    base = AudioJob(
        (AudioClip("clip", AudioBufferSource(audio, 1000)), Silence("pause", 0.1)),
        output=_output_policy(False),
    )
    mastered = AudioJob(
        base.items,
        output=_output_policy(True),
    )

    without = Composer(sample_rate=1000).compose(base)
    with_loudness = Composer(sample_rate=1000).compose(mastered)

    assert with_loudness.loudness is not None
    assert without.items == with_loudness.items
    assert without.markers == with_loudness.markers
    assert without.spans == with_loudness.spans


def test_composition_reports_silence_diagnostic() -> None:
    result = Composer(sample_rate=1000).compose(
        AudioJob((Silence("pause", 1.0),), output=_output_policy(True))
    )
    assert any(d.code == "LOUDNESS_UNMEASURABLE" for d in result.diagnostics)


def test_compose_uses_one_analysis_and_emits_phase_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    from audiocompose import loudness as loudness_module

    original = loudness_module.analyze_loudness
    calls: list[dict[str, object]] = []

    def record_analysis(*args: object, **kwargs: object):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(loudness_module, "analyze_loudness", record_analysis)
    events = []
    job = AudioJob(
        (AudioClip("voice", AudioBufferSource(_tone(), 1000)),),
        output=_output_policy(True),
    )

    result = Composer(sample_rate=1000).compose(job, on_progress=events.append)

    assert len(calls) == 1
    assert result.loudness is not None
    completed = next(event for event in events if event.kind == "loudness_completed")
    assert completed.details["integrated_lufs_before"] is not None
    assert completed.details["integrated_lufs_after"] is not None
    assert completed.details["requested_gain_db"] == result.loudness.requested_gain_db
    assert completed.details["analysis_seconds"] >= 0.0
    assert completed.details["gain_seconds"] >= 0.0
    assert completed.details["post_gain_metrics_seconds"] >= 0.0


def test_off_policy_without_diagnostics_is_a_true_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    from audiocompose import loudness as loudness_module

    def fail_analysis(*args: object, **kwargs: object) -> None:
        pytest.fail("no-op mastering unexpectedly analyzed audio")

    monkeypatch.setattr(loudness_module, "analyze_loudness", fail_analysis)
    audio = _tone()
    result = apply_complete_output_loudness(
        audio, 1000, LoudnessPolicy(None, None, collect_metrics=False)
    )

    assert result.audio is audio
    assert result.before == result.after
    assert result.before.integrated_lufs is None
    assert result.before.true_peak_dbtp is None
    assert result.applied_gain_db == 0.0
    assert result.analysis_seconds == 0.0


def test_selective_metrics_cover_only_policy_requirements() -> None:
    audio = _tone()
    target_only = apply_complete_output_loudness(
        audio,
        1000,
        LoudnessPolicy(target_lufs=-20.0, true_peak_ceiling_dbtp=None, collect_metrics=False),
    )
    peak_only = apply_complete_output_loudness(
        audio,
        1000,
        LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=-1.0, collect_metrics=False),
    )

    assert target_only.before.integrated_lufs is not None
    assert target_only.before.sample_peak_dbfs is None
    assert target_only.before.true_peak_dbtp is None
    assert peak_only.before.integrated_lufs is None
    assert peak_only.before.sample_peak_dbfs is None
    assert peak_only.before.true_peak_dbtp is not None


def test_collect_metrics_false_is_opt_in_in_job_policy_serialization() -> None:
    from audiocompose.job import _policy_to_dict

    default = _policy_to_dict(OutputPolicy())
    off = _policy_to_dict(
        OutputPolicy(
            loudness=LoudnessPolicy(
                target_lufs=None, true_peak_ceiling_dbtp=None, collect_metrics=False
            )
        )
    )

    assert "collect_metrics" not in default["loudness"]
    assert off["loudness"]["collect_metrics"] is False
