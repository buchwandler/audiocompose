from __future__ import annotations

import math

import numpy as np

from .errors import AudioValidationError


def samples_for_duration(seconds: float, sample_rate: int) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    if not math.isfinite(seconds) or seconds < 0:
        raise AudioValidationError("seconds must be finite and >= 0")
    return round(seconds * sample_rate)


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(samples_for_duration(seconds, sample_rate), dtype=np.float32)
