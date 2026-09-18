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


def apply_operation(audio: np.ndarray, sample_rate: int, operation: Operation) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if isinstance(operation, Gain):
        return (values * (10.0 ** (operation.db / 20.0))).astype(np.float32)
    if isinstance(operation, Tempo):
        return resample_audio(values, sample_rate, max(1, round(sample_rate / operation.factor)))
    if isinstance(operation, PitchShift):
        if values.size < 2 or operation.semitones == 0:
            return values.copy()
        ratio = 2.0 ** (-operation.semitones / 12.0)
        shifted = resample_audio(values, sample_rate, max(1, round(sample_rate * ratio)))
        return resample_audio(shifted, max(1, round(sample_rate * ratio)), sample_rate)
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
