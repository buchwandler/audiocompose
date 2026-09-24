from __future__ import annotations

import numpy as np
from audiosig import AudioSignalError, resample

from .errors import AudioValidationError
from .wav import _as_finite_mono_float32


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise AudioValidationError("sample rates must be positive")
    values = _as_finite_mono_float32(audio)
    if source_rate == target_rate or values.size == 0:
        return np.ascontiguousarray(values, dtype=np.float32)
    output_size = round(values.size * target_rate / source_rate)
    if output_size == 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 1:
        return np.full(output_size, values[0], dtype=np.float32)
    try:
        result = resample(
            values,
            source_rate=source_rate,
            target_rate=target_rate,
            filter_width=32,
            rolloff=0.945,
            length_mode="round",
        )
    except AudioSignalError as exc:
        raise AudioValidationError(str(exc)) from exc
    return _as_finite_mono_float32(result, name="resampling output")
