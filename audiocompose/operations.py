from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import AudioValidationError, CompositionError
from .resampling import resample_audio


class AudioOperation:
    type: str

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def map_offset(self, offset: int, length: int) -> int:
        del length
        return offset


@dataclass(frozen=True, slots=True)
class Gain(AudioOperation):
    db: float
    type: str = "gain"

    def __post_init__(self) -> None:
        if not math.isfinite(self.db):
            raise ValueError("gain db must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "db": self.db}


@dataclass(frozen=True, slots=True)
class PitchShift(AudioOperation):
    semitones: float
    type: str = "pitch"

    def __post_init__(self) -> None:
        if not math.isfinite(self.semitones):
            raise ValueError("pitch semitones must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "semitones": self.semitones}


@dataclass(frozen=True, slots=True)
class Tempo(AudioOperation):
    factor: float
    type: str = "tempo"

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("tempo factor must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "factor": self.factor}

    def map_offset(self, offset: int, length: int) -> int:
        del length
        return round(offset / self.factor)


@dataclass(frozen=True, slots=True)
class FadeIn(AudioOperation):
    seconds: float
    type: str = "fade_in"

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("fade-in seconds must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "seconds": self.seconds}


@dataclass(frozen=True, slots=True)
class FadeOut(AudioOperation):
    seconds: float
    type: str = "fade_out"

    def __post_init__(self) -> None:
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("fade-out seconds must be finite and >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "seconds": self.seconds}


Operation = Gain | PitchShift | Tempo | FadeIn | FadeOut


def operation_from_dict(value: dict[str, Any]) -> Operation:
    try:
        kind = value["type"]
    except (KeyError, TypeError) as exc:
        raise AudioValidationError("operation must contain a type") from exc
    try:
        if kind == "gain":
            return Gain(float(value["db"]))
        if kind == "pitch":
            return PitchShift(float(value["semitones"]))
        if kind == "tempo":
            return Tempo(float(value["factor"]))
        if kind == "fade_in":
            return FadeIn(float(value["seconds"]))
        if kind == "fade_out":
            return FadeOut(float(value["seconds"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioValidationError(f"invalid {kind!r} operation parameters") from exc
    raise AudioValidationError(f"unsupported operation: {kind!r}")


def _phase_vocoder(audio: np.ndarray, rate: float) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if values.size < 4 or rate == 1.0:
        return values.copy()
    n_fft = min(1024, 2 ** int(math.floor(math.log2(values.size))))
    n_fft = max(16, n_fft)
    hop = n_fft // 4
    window = np.hanning(n_fft).astype(np.float64)
    padded = np.pad(values.astype(np.float64), (n_fft // 2, n_fft // 2 + n_fft), mode="constant")
    starts = range(0, len(padded) - n_fft + 1, hop)
    spectra = np.stack([np.fft.rfft(padded[start : start + n_fft] * window) for start in starts])
    time_steps = np.arange(0.0, max(1.0, len(spectra) - 1.0), rate)
    expected = 2.0 * np.pi * hop * np.arange(spectra.shape[1]) / n_fft
    phase = np.angle(spectra[0])
    output = np.zeros((len(time_steps) + 1) * hop + n_fft, dtype=np.float64)
    normalization = np.zeros_like(output)
    for frame_index, step in enumerate(time_steps):
        left = min(int(step), len(spectra) - 2)
        fraction = step - left
        magnitude = (1.0 - fraction) * np.abs(spectra[left]) + fraction * np.abs(spectra[left + 1])
        phase_delta = np.angle(spectra[left + 1]) - np.angle(spectra[left]) - expected
        phase_delta = np.angle(np.exp(1j * phase_delta))
        phase += expected + phase_delta
        frame = np.fft.irfft(magnitude * np.exp(1j * phase), n_fft)
        start = frame_index * hop
        output[start : start + n_fft] += frame * window
        normalization[start : start + n_fft] += window * window
    valid = normalization > 1e-12
    output[valid] /= normalization[valid]
    target = max(1, round(values.size / rate))
    result = output[n_fft // 2 : n_fft // 2 + target]
    if len(result) < target:
        result = np.pad(result, (0, target - len(result)))
    return result.astype(np.float32, copy=False)


def _pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    if audio.size < 4 or semitones == 0:
        return np.asarray(audio, dtype=np.float32).copy()
    ratio = 2.0 ** (semitones / 12.0)
    stretched = _phase_vocoder(audio, 1.0 / ratio)
    shifted = resample_audio(stretched, sample_rate, max(1, round(sample_rate / ratio)))
    target = len(audio)
    if len(shifted) > target:
        shifted = shifted[:target]
    elif len(shifted) < target:
        shifted = np.pad(shifted, (0, target - len(shifted)))
    return shifted.astype(np.float32, copy=False)


def apply_operation(audio: np.ndarray, sample_rate: int, operation: Operation) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if isinstance(operation, Gain):
        return (values * (10.0 ** (operation.db / 20.0))).astype(np.float32)
    if isinstance(operation, Tempo):
        return _phase_vocoder(values, operation.factor)
    if isinstance(operation, PitchShift):
        return _pitch_shift(values, sample_rate, operation.semitones)
    if isinstance(operation, (FadeIn, FadeOut)):
        result = values.copy()
        count = min(result.size, round(operation.seconds * sample_rate))
        if count:
            ramp = np.linspace(0.0, 1.0, count, endpoint=True, dtype=np.float32)
            if isinstance(operation, FadeIn):
                result[:count] *= ramp
            else:
                result[-count:] *= ramp[::-1]
        return result
    raise CompositionError(f"unsupported operation: {operation!r}")
