"""AudioCompose policy for removing confirmed synthesized-speech tail silence."""

from __future__ import annotations

from dataclasses import dataclass

import audiosig
import numpy as np

from .errors import AudioValidationError
from .wav import _as_finite_mono_float32

SPEECH_BOUNDARY_POLICY_ID = "audiocompose.trailing-silence.v1"


@dataclass(frozen=True, slots=True)
class SpeechBoundaryConditioning:
    policy: str
    input_frames: int
    output_frames: int
    removed_frames: int
    removed_seconds: float
    trim_end_sample: int


def condition_speech_tail(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, SpeechBoundaryConditioning]:
    """Remove confirmed terminal silence from a synthesized speech buffer."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    source = _as_finite_mono_float32(audio, name="speech audio")
    if source.size == 0:
        raise AudioValidationError("speech audio must contain at least one frame")

    conditioned = audiosig.trim_trail(source, sample_rate=sample_rate, trail_seconds=0.0)
    conditioned = np.ascontiguousarray(conditioned, dtype=np.float32)
    input_frames = int(source.size)
    output_frames = int(conditioned.size)
    removed_frames = input_frames - output_frames
    return conditioned, SpeechBoundaryConditioning(
        policy=SPEECH_BOUNDARY_POLICY_ID,
        input_frames=input_frames,
        output_frames=output_frames,
        removed_frames=removed_frames,
        removed_seconds=removed_frames / sample_rate,
        trim_end_sample=output_frames,
    )
