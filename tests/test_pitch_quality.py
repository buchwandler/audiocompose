from __future__ import annotations

import numpy as np
import pytest
from _quality_helpers import dominant_frequency, speech_like_signal

from audiocompose import PitchShift
from audiocompose.operations import apply_operation


@pytest.mark.quality
@pytest.mark.parametrize("semitones", [-4.0, -2.0, 2.0, 4.0])
def test_pitch_shift_hits_target_and_preserves_duration(semitones: float) -> None:
    sample_rate = 24_000
    frequency = 220.0
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * frequency * time).astype(np.float32)

    result = apply_operation(source, sample_rate, PitchShift(semitones))
    expected_frequency = frequency * 2.0 ** (semitones / 12.0)

    assert len(result) == len(source)
    assert dominant_frequency(result, sample_rate) == pytest.approx(expected_frequency, abs=20.0)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


@pytest.mark.quality
def test_pitch_shift_preserves_silence_and_processes_speech_like_input() -> None:
    sample_rate = 24_000
    silence = np.zeros(sample_rate // 4, dtype=np.float32)
    speech = speech_like_signal(sample_rate)

    silent_result = apply_operation(silence, sample_rate, PitchShift(4.0))
    speech_result = apply_operation(speech, sample_rate, PitchShift(-4.0))

    assert len(silent_result) == len(silence)
    assert np.array_equal(silent_result, np.zeros_like(silent_result))
    assert len(speech_result) == len(speech)
    assert np.isfinite(speech_result).all()
