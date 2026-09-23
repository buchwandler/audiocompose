from __future__ import annotations

import numpy as np
import pytest
from _quality_helpers import speech_like_signal

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    Tempo,
)


@pytest.mark.quality
@pytest.mark.parametrize(
    ("operations", "tempo_factor"),
    [
        ((Tempo(0.85),), 0.85),
        ((PitchShift(4.0),), 1.0),
        ((Tempo(0.85), PitchShift(-3.0)), 0.85),
    ],
)
def test_public_composer_processes_speech_like_dsp_and_mixed_rates(
    operations: tuple[Tempo | PitchShift, ...],
    tempo_factor: float,
) -> None:
    source_rate = 24_000
    output_rate = 16_000
    source = speech_like_signal(source_rate)
    job = AudioJob(
        (
            AudioClip(
                "quality",
                AudioBufferSource(source, source_rate),
                operations,
            ),
        ),
        output=OutputPolicy(
            sample_rate=output_rate,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )

    result = Composer(sample_rate=output_rate).compose(job)
    operation_frames = round(len(source) / tempo_factor)

    assert result.sample_rate == output_rate
    assert len(result.audio) == round(operation_frames * output_rate / source_rate)
    assert result.audio.dtype == np.float32
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) > 0.0


@pytest.mark.quality
def test_rate_only_speech_like_composition_has_exact_duration() -> None:
    sample_rate = 24_000
    source = speech_like_signal(sample_rate)
    job = AudioJob(
        (AudioClip("rate-only", AudioBufferSource(source, sample_rate), (Tempo(0.8),)),),
        output=OutputPolicy(
            sample_rate=sample_rate,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )

    result = Composer(sample_rate=sample_rate).compose(job)

    assert len(result.audio) == round(len(source) / 0.8)
    assert np.isfinite(result.audio).all()
