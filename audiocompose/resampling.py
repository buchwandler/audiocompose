from __future__ import annotations

import numpy as np

from .errors import AudioValidationError


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise AudioValidationError("sample rates must be positive")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1:
        raise AudioValidationError("audio must be one-dimensional")
    if source_rate == target_rate or values.size == 0:
        return np.ascontiguousarray(values, dtype=np.float32)
    output_size = round(values.size * target_rate / source_rate)
    if output_size == 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 1:
        return np.full(output_size, values[0], dtype=np.float32)
    old = np.linspace(0.0, 1.0, values.size, endpoint=True)
    new = np.linspace(0.0, 1.0, output_size, endpoint=True)
    return np.interp(new, old, values).astype(np.float32)
