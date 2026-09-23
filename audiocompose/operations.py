from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from audiosig import apply_speech_effects, pitch_shift, time_stretch

from .errors import AudioValidationError, CompositionError


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


def apply_temporal_group(
    audio: np.ndarray,
    sample_rate: int,
    operations: Sequence[Tempo | PitchShift],
) -> np.ndarray:
    rate = math.prod(operation.factor for operation in operations if isinstance(operation, Tempo))
    semitones = math.fsum(
        operation.semitones for operation in operations if isinstance(operation, PitchShift)
    )
    result = apply_speech_effects(
        np.asarray(audio, dtype=np.float32),
        sample_rate=sample_rate,
        rate=rate,
        semitones=semitones,
        method="wsola",
    )
    return np.ascontiguousarray(result, dtype=np.float32)


def apply_operation(audio: np.ndarray, sample_rate: int, operation: Operation) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if isinstance(operation, Gain):
        return (values * (10.0 ** (operation.db / 20.0))).astype(np.float32)
    if isinstance(operation, Tempo):
        return np.ascontiguousarray(
            time_stretch(
                values,
                operation.factor,
                sample_rate=sample_rate,
                method="wsola",
            ),
            dtype=np.float32,
        )
    if isinstance(operation, PitchShift):
        return np.ascontiguousarray(
            pitch_shift(
                values,
                sample_rate=sample_rate,
                semitones=operation.semitones,
                method="wsola",
            ),
            dtype=np.float32,
        )
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
