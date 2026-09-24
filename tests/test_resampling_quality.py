from __future__ import annotations

import numpy as np
import pytest
from _quality_helpers import dominant_frequency, rms, tone_amplitude
from audiosig import InvalidParameterError

import audiocompose.resampling as resampling
from audiocompose import AudioValidationError
from audiocompose.resampling import resample_audio


@pytest.mark.quality
@pytest.mark.parametrize("tone", [9_000.0, 10_000.0, 11_000.0])
def test_downsampling_rejects_above_nyquist_tones(tone: float) -> None:
    source_rate = 24_000
    target_rate = 16_000
    time = np.arange(source_rate, dtype=np.float64) / source_rate
    source = np.sin(2.0 * np.pi * tone * time).astype(np.float32)

    result = resample_audio(source, source_rate, target_rate)

    assert rms(result[256:-256]) < 0.01


@pytest.mark.quality
@pytest.mark.parametrize(
    ("source_rate", "target_rate", "tone"),
    [
        (24_000, 16_000, 1_000.0),
        (24_000, 16_000, 6_000.0),
        (16_000, 24_000, 1_000.0),
        (16_000, 24_000, 7_000.0),
    ],
)
def test_resampling_preserves_passband(
    source_rate: int,
    target_rate: int,
    tone: float,
) -> None:
    time = np.arange(source_rate, dtype=np.float64) / source_rate
    source = (0.8 * np.sin(2.0 * np.pi * tone * time)).astype(np.float32)

    result = resample_audio(source, source_rate, target_rate)

    assert len(result) == round(len(source) * target_rate / source_rate)
    assert dominant_frequency(result, target_rate) == pytest.approx(tone, abs=3.0)
    assert tone_amplitude(result, target_rate, tone) == pytest.approx(0.8, rel=0.12, abs=0.04)


@pytest.mark.quality
def test_resampling_preserves_dc_and_silence() -> None:
    source = np.full(24_000, 0.25, dtype=np.float32)

    dc_result = resample_audio(source, 24_000, 16_000)
    silence_result = resample_audio(np.zeros_like(source), 24_000, 16_000)

    np.testing.assert_allclose(dc_result[256:-256], 0.25, atol=1e-5)
    assert np.array_equal(silence_result, np.zeros(16_000, dtype=np.float32))


@pytest.mark.quality
@pytest.mark.parametrize(
    ("source_rate", "target_rate", "sample_count"),
    [(24_000, 16_000, 24_000), (16_000, 24_000, 16_000), (96_000, 8_000, 1)],
)
def test_resampling_uses_rounded_output_length(
    source_rate: int,
    target_rate: int,
    sample_count: int,
) -> None:
    source = np.ones(sample_count, dtype=np.float32)

    result = resample_audio(source, source_rate, target_rate)

    assert len(result) == round(sample_count * target_rate / source_rate)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


@pytest.mark.quality
def test_same_rate_and_empty_inputs_return_contiguous_float32() -> None:
    source = np.arange(12, dtype=np.float64)[::2]

    same_rate = resample_audio(source, 24_000, 24_000)
    empty = resample_audio(np.zeros(0, dtype=np.float32), 24_000, 16_000)

    assert np.array_equal(same_rate, source.astype(np.float32))
    assert same_rate.flags.c_contiguous
    assert same_rate.dtype == np.float32
    assert empty.size == 0
    assert empty.dtype == np.float32


@pytest.mark.quality
def test_invalid_input_keeps_audio_validation_error_contract() -> None:
    with pytest.raises(AudioValidationError, match="sample rates must be positive"):
        resample_audio(np.ones(4, dtype=np.float32), 0, 16_000)
    with pytest.raises(AudioValidationError, match="one-dimensional"):
        resample_audio(np.ones((2, 4), dtype=np.float32), 24_000, 16_000)


@pytest.mark.quality
def test_audiosig_parameter_error_is_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise InvalidParameterError("invalid resampling parameters")

    monkeypatch.setattr(resampling, "resample", fail)

    with pytest.raises(AudioValidationError, match="invalid resampling parameters"):
        resample_audio(np.ones(8, dtype=np.float32), 8_000, 16_000)
