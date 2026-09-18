"""Tests for span coordinate-domain separation.

Source-text offsets (source_start, source_end) are producer-defined and must
be independent of audio sample length. Sample offsets (sample_start, sample_end)
are clip-local and must be validated against the clip's audio length.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioSpan,
    AudioValidationError,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
)
from audiocompose.job import validate_job
from audiocompose.model import AudioJob


# 10.1 Valid large source offsets
def test_valid_large_source_offsets_with_valid_sample_range() -> None:
    """Source character offsets much larger than audio sample count must be valid."""
    audio = np.zeros(100, dtype=np.float32)
    clip = AudioClip(
        "clip",
        AudioBufferSource(audio, 24000),
        spans=(
            AudioSpan(
                source_start=10_000,
                source_end=10_020,
                sample_start=10,
                sample_end=90,
            ),
        ),
    )
    job = AudioJob((clip,))
    validate_job(job)
    result = Composer(sample_rate=24000).compose(job)
    assert len(result.audio) > 0


# 10.2 Sample overrun remains invalid
def test_sample_overrun_rejected() -> None:
    """sample_end exceeding clip audio length must be rejected."""
    audio = np.zeros(100, dtype=np.float32)
    clip = AudioClip(
        "clip",
        AudioBufferSource(audio, 24000),
        spans=(
            AudioSpan(
                source_start=0,
                source_end=5,
                sample_start=20,
                sample_end=101,
            ),
        ),
    )
    job = AudioJob((clip,))
    with pytest.raises(AudioValidationError, match=r"span\[0\].*exceeds source length.*clip"):
        validate_job(job)


# 10.3 Negative sample coordinate
def test_negative_sample_start_rejected() -> None:
    """sample_start < 0 must be rejected by model validation."""
    with pytest.raises(AudioValidationError, match="sample_start must be a non-negative integer"):
        AudioSpan(source_start=0, source_end=5, sample_start=-1, sample_end=10)


def test_negative_sample_end_rejected() -> None:
    """sample_end < 0 must be rejected by model validation."""
    with pytest.raises(AudioValidationError, match="sample_end must be a non-negative integer"):
        AudioSpan(source_start=0, source_end=5, sample_start=0, sample_end=-1)


# 10.4 Reversed sample range
def test_reversed_sample_range_rejected() -> None:
    """sample_end < sample_start must be rejected."""
    with pytest.raises(AudioValidationError, match="sample_end must be >= sample_start"):
        AudioSpan(source_start=0, source_end=5, sample_start=10, sample_end=5)


# 10.5 Reversed source range
def test_reversed_source_range_rejected() -> None:
    """source_end < source_start must be rejected."""
    with pytest.raises(AudioValidationError, match="source_end must be >= source_start"):
        AudioSpan(source_start=10, source_end=5, sample_start=0, sample_end=5)


# 10.6 Save/load/replay
def test_save_load_replay_preserves_large_source_offsets(tmp_path: Path) -> None:
    """Save/load/replay must preserve large source offsets and valid sample ranges."""
    audio = np.zeros(100, dtype=np.float32)
    clip = AudioClip(
        "clip",
        AudioBufferSource(audio, 24000),
        spans=(
            AudioSpan(
                source_start=10_000,
                source_end=10_020,
                sample_start=10,
                sample_end=90,
            ),
        ),
    )
    job = AudioJob(
        (clip,),
        output=OutputPolicy(
            sample_rate=24000,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )

    saved = Path(job.save(tmp_path / "test.audiojob"))
    loaded = AudioJob.load(saved)
    result = Composer(sample_rate=24000).compose(loaded)
    assert len(result.audio) > 0

    assert loaded.items[0].spans[0].source_start == 10_000
    assert loaded.items[0].spans[0].source_end == 10_020
    assert loaded.items[0].spans[0].sample_start == 10
    assert loaded.items[0].spans[0].sample_end == 90


# Additional: anchor validation
def test_anchor_negative_offset_rejected() -> None:
    """AudioAnchor with negative sample_offset must be rejected."""
    with pytest.raises(AudioValidationError, match="sample_offset must be a non-negative integer"):
        AudioAnchor("anchor", -1)


def test_anchor_exceeds_source_length_rejected() -> None:
    """AudioAnchor with sample_offset > len(audio) must be rejected by validate_job."""
    audio = np.zeros(100, dtype=np.float32)
    clip = AudioClip(
        "clip",
        AudioBufferSource(audio, 24000),
        anchors=(AudioAnchor("late", 101),),
    )
    job = AudioJob((clip,))
    with pytest.raises(AudioValidationError, match=r"anchor.*exceeds source length"):
        validate_job(job)


def test_negative_source_start_rejected() -> None:
    """source_start < 0 must be rejected."""
    with pytest.raises(AudioValidationError, match="source_start must be a non-negative integer"):
        AudioSpan(source_start=-1, source_end=5, sample_start=0, sample_end=5)
