from __future__ import annotations

import numpy as np
import pytest
from _quality_helpers import dominant_frequency
from audiosig import apply_speech_effects as audiosig_apply_speech_effects

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    Composer,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    Tempo,
)
from audiocompose import operations as operations_module


def _compose(source: np.ndarray, operations: tuple, sample_rate: int = 24_000):
    job = AudioJob(
        (AudioClip("quality", AudioBufferSource(source, sample_rate), operations),),
        output=OutputPolicy(
            sample_rate=sample_rate,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )
    return Composer(sample_rate=sample_rate).compose(job)


@pytest.mark.quality
def test_contiguous_temporal_operations_use_one_combined_dsp_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 24_000
    frequency = 220.0
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = np.sin(2.0 * np.pi * frequency * time).astype(np.float32)
    calls = []

    def spy(audio: np.ndarray, **kwargs) -> np.ndarray:
        calls.append(kwargs)
        return audiosig_apply_speech_effects(audio, **kwargs)

    monkeypatch.setattr(operations_module, "apply_speech_effects", spy)
    result = _compose(source, (Tempo(0.85), PitchShift(-3.0), Tempo(1.2)))

    assert len(calls) == 1
    assert calls[0]["rate"] == pytest.approx(0.85 * 1.2)
    assert calls[0]["semitones"] == -3.0
    assert calls[0]["method"] == "wsola"
    assert len(result.audio) == round(len(source) / (0.85 * 1.2))
    assert dominant_frequency(result.audio, sample_rate) == pytest.approx(
        frequency * 2.0 ** (-3.0 / 12.0), abs=20.0
    )
    assert np.isfinite(result.audio).all()


@pytest.mark.quality
def test_gain_is_a_boundary_between_temporal_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    sample_rate = 24_000
    source = np.ones(sample_rate, dtype=np.float32)
    calls = []

    def spy(audio: np.ndarray, **kwargs) -> np.ndarray:
        calls.append(kwargs)
        return audiosig_apply_speech_effects(audio, **kwargs)

    monkeypatch.setattr(operations_module, "apply_speech_effects", spy)
    result = _compose(source, (Tempo(0.85), Gain(-3.0), PitchShift(2.0)))

    assert calls == []
    assert len(result.audio) == round(len(source) / 0.85)
    assert np.isfinite(result.audio).all()
