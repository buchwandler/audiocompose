from __future__ import annotations

import numpy as np
import pytest
from _quality_helpers import dominant_frequency, speech_like_signal

from audiocompose import Tempo
from audiocompose.operations import apply_operation


@pytest.mark.quality
@pytest.mark.parametrize("factor", [0.8, 1.2, 1.4])
def test_tempo_preserves_pitch_and_has_exact_duration(factor: float) -> None:
    sample_rate = 24_000
    frequency = 220.0
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * frequency * time).astype(np.float32)

    result = apply_operation(source, sample_rate, Tempo(factor))

    assert len(result) == round(len(source) / factor)
    assert dominant_frequency(result, sample_rate) == pytest.approx(frequency, abs=15.0)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


@pytest.mark.quality
def test_tempo_preserves_silence_and_processes_speech_like_input() -> None:
    sample_rate = 24_000
    silence = np.zeros(sample_rate // 4, dtype=np.float32)
    speech = speech_like_signal(sample_rate)

    silent_result = apply_operation(silence, sample_rate, Tempo(1.2))
    speech_result = apply_operation(speech, sample_rate, Tempo(0.8))

    assert len(silent_result) == round(len(silence) / 1.2)
    assert np.array_equal(silent_result, np.zeros_like(silent_result))
    assert len(speech_result) == round(len(speech) / 0.8)
    assert np.isfinite(speech_result).all()
