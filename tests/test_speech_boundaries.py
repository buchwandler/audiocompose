from __future__ import annotations

import numpy as np
import pytest

from audiocompose.errors import AudioValidationError
from audiocompose.speech_boundaries import (
    SPEECH_BOUNDARY_POLICY_ID,
    SpeechBoundaryConditioning,
    condition_speech_tail,
)


def test_trims_confirmed_terminal_silence_and_preserves_leading_samples() -> None:
    sample_rate = 1_000
    audio = np.concatenate(
        (
            np.zeros(200, dtype=np.float32),
            np.full(300, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )

    conditioned, result = condition_speech_tail(audio, sample_rate=sample_rate)

    assert len(conditioned) == 500
    np.testing.assert_array_equal(conditioned, audio[:500])
    assert result == SpeechBoundaryConditioning(
        policy=SPEECH_BOUNDARY_POLICY_ID,
        input_frames=900,
        output_frames=500,
        removed_frames=400,
        removed_seconds=0.4,
        trim_end_sample=500,
    )
    assert conditioned.flags.c_contiguous


def test_no_confirmed_activity_preserves_silence_and_uncertain_audio() -> None:
    sample_rate = 1_000
    for audio in (
        np.zeros(1_000, dtype=np.float32),
        np.full(1_000, 1e-4, dtype=np.float32),
    ):
        conditioned, result = condition_speech_tail(audio, sample_rate=sample_rate)

        np.testing.assert_array_equal(conditioned, audio)
        assert result.output_frames == len(audio)
        assert result.removed_frames == 0
        assert result.trim_end_sample == len(audio)
        assert conditioned.flags.c_contiguous


def test_audio_active_through_eof_is_not_trimmed() -> None:
    audio = np.full(1_000, 0.5, dtype=np.float32)

    conditioned, result = condition_speech_tail(audio, sample_rate=1_000)

    np.testing.assert_array_equal(conditioned, audio)
    assert result.removed_frames == 0
    assert result.trim_end_sample == len(audio)


def test_short_resumed_speech_after_internal_pause_is_preserved() -> None:
    sample_rate = 1_000
    audio = np.concatenate(
        (
            np.full(200, 0.5, dtype=np.float32),
            np.zeros(150, dtype=np.float32),
            np.full(60, 0.25, dtype=np.float32),
            np.zeros(300, dtype=np.float32),
        )
    )

    conditioned, result = condition_speech_tail(audio, sample_rate=sample_rate)

    assert result.trim_end_sample == 410
    np.testing.assert_array_equal(conditioned, audio[:410])
    assert result.removed_frames == 300


def test_noncontiguous_input_returns_contiguous_buffer() -> None:
    audio = np.zeros(2_000, dtype=np.float32)[::2]
    assert not audio.flags.c_contiguous

    conditioned, result = condition_speech_tail(audio, sample_rate=1_000)

    assert conditioned.flags.c_contiguous
    np.testing.assert_array_equal(conditioned, audio)
    assert result.input_frames == len(audio)


@pytest.mark.parametrize("sample_rate", [0, -1, True])
def test_rejects_invalid_sample_rate(sample_rate: int) -> None:
    with pytest.raises(AudioValidationError, match="sample_rate must be a positive integer"):
        condition_speech_tail(np.ones(10, dtype=np.float32), sample_rate=sample_rate)


@pytest.mark.parametrize(
    "audio",
    [
        np.ones((2, 10), dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([], dtype=np.float32),
    ],
)
def test_rejects_invalid_speech_audio(audio: np.ndarray) -> None:
    with pytest.raises(AudioValidationError):
        condition_speech_tail(audio, sample_rate=1_000)
