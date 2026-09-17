from __future__ import annotations

import warnings
import wave
from pathlib import Path
from typing import Literal

import numpy as np

ClipPolicy = Literal["clamp", "warn", "error"]


def prepare_output(audio: np.ndarray, *, clip_policy: ClipPolicy = "clamp") -> np.ndarray:
    """Apply the explicit output range policy before PCM encoding."""

    samples = np.asarray(audio, dtype=np.float32)
    over_range = bool(np.any((samples < -1.0) | (samples > 1.0)))
    if over_range and clip_policy == "error":
        raise ValueError("audio contains samples outside the PCM range [-1, 1]")
    if over_range and clip_policy == "warn":
        warnings.warn("audio was clipped to the PCM range [-1, 1]", RuntimeWarning, stacklevel=2)
    return np.clip(samples, -1.0, 1.0)


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    clip_policy: ClipPolicy = "clamp",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = prepare_output(audio, clip_policy=clip_policy)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination
