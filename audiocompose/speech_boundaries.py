"""AudioCompose policy for conditioning synthesized-speech boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import audiosig
import numpy as np

from .errors import AudioValidationError
from .wav import _as_finite_mono_float32

SPEECH_BOUNDARY_POLICY_ID = "audiocompose.trailing-silence.v2"

_SPEECH_TRIM_KEEP_SECONDS = 0.150
_SPEECH_TRIM_FRAME_SECONDS = 0.005
_SPEECH_TRIM_TAIL_REFERENCE_SECONDS = 0.080
_SPEECH_TRIM_MIN_SILENCE_SECONDS = 0.150
_SPEECH_TRIM_FLOOR_MARGIN_DB = 4.0
_SPEECH_TRIM_MIN_SEPARATION_DB = 18.0
_SPEECH_TRIM_TRANSITION_MARGIN_DB = 6.0

_PAUSE_CREDIT_MIN_SILENCE_SECONDS = 0.020
_PAUSE_CREDIT_MAX_SECONDS = 0.150


@dataclass(frozen=True, slots=True)
class SpeechBoundaryConditioning:
    policy: str
    input_frames: int
    output_frames: int
    removed_frames: int
    removed_seconds: float
    trim_end_sample: int
    trim_confirmed: bool
    trim_silence_start_sample: int | None
    tail_estimate_confirmed: bool
    tail_estimate_start_sample: int | None
    estimated_tail_frames: int
    estimated_tail_seconds: float


def condition_speech_tail(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, SpeechBoundaryConditioning]:
    """Trim confirmed terminal padding and estimate the retained speech tail."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    source = _as_finite_mono_float32(audio, name="speech audio")
    if source.size == 0:
        raise AudioValidationError("speech audio must contain at least one frame")

    conditioned, trim_analysis = audiosig.trim_terminal_silence(
        source,
        sample_rate=sample_rate,
        keep_seconds=_SPEECH_TRIM_KEEP_SECONDS,
        frame_seconds=_SPEECH_TRIM_FRAME_SECONDS,
        tail_reference_seconds=_SPEECH_TRIM_TAIL_REFERENCE_SECONDS,
        min_silence_seconds=_SPEECH_TRIM_MIN_SILENCE_SECONDS,
        floor_margin_db=_SPEECH_TRIM_FLOOR_MARGIN_DB,
        min_separation_db=_SPEECH_TRIM_MIN_SEPARATION_DB,
        transition_margin_db=_SPEECH_TRIM_TRANSITION_MARGIN_DB,
    )
    conditioned = np.ascontiguousarray(conditioned, dtype=np.float32)
    credit_analysis = audiosig.analyze_terminal_silence(
        source,
        sample_rate=sample_rate,
        frame_seconds=_SPEECH_TRIM_FRAME_SECONDS,
        tail_reference_seconds=_SPEECH_TRIM_TAIL_REFERENCE_SECONDS,
        min_silence_seconds=_PAUSE_CREDIT_MIN_SILENCE_SECONDS,
        floor_margin_db=_SPEECH_TRIM_FLOOR_MARGIN_DB,
        min_separation_db=_SPEECH_TRIM_MIN_SEPARATION_DB,
        transition_margin_db=_SPEECH_TRIM_TRANSITION_MARGIN_DB,
    )

    if credit_analysis.confirmed and credit_analysis.silence_start_sample is not None:
        estimated_tail_frames = max(
            0,
            trim_analysis.trim_end_sample - credit_analysis.silence_start_sample,
        )
    else:
        estimated_tail_frames = 0
    estimated_tail_frames = min(
        estimated_tail_frames,
        round(_PAUSE_CREDIT_MAX_SECONDS * sample_rate),
        trim_analysis.trim_end_sample,
    )

    input_frames = int(source.size)
    output_frames = int(conditioned.size)
    removed_frames = input_frames - output_frames
    return conditioned, SpeechBoundaryConditioning(
        policy=SPEECH_BOUNDARY_POLICY_ID,
        input_frames=input_frames,
        output_frames=output_frames,
        removed_frames=removed_frames,
        removed_seconds=removed_frames / sample_rate,
        trim_end_sample=trim_analysis.trim_end_sample,
        trim_confirmed=trim_analysis.confirmed,
        trim_silence_start_sample=trim_analysis.silence_start_sample,
        tail_estimate_confirmed=credit_analysis.confirmed,
        tail_estimate_start_sample=credit_analysis.silence_start_sample,
        estimated_tail_frames=estimated_tail_frames,
        estimated_tail_seconds=estimated_tail_frames / sample_rate,
    )
