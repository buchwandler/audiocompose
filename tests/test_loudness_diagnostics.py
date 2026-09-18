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
    apply_complete_output_loudness,
)
from audiocompose.errors import CompositionError


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
