from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from audiosig import (
    InvalidParameterError,
    apply_speech_effects,
    apply_speech_effects_envelope,
    pitch_shift,
    time_stretch,
)

from ._automation import map_source_frame, output_frames_for_input_frames
from .errors import AudioValidationError, CompositionError


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AudioValidationError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class AutomationPoint:
    seconds: float
    value: float

    def __post_init__(self) -> None:
        seconds = _finite_float(self.seconds, "automation seconds")
        value = _finite_float(self.value, "automation value")
        if seconds < 0:
            raise AudioValidationError("automation seconds must be >= 0")
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "value", value)


class AudioOperation:
    type: str

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def map_offset(self, offset: int, length: int) -> int:
        del length
        return offset

    def map_offset_at_rate(
        self,
        offset: int,
        length: int,
        sample_rate: int,
    ) -> int:
        del sample_rate
        return self.map_offset(offset, length)

    def output_length(self, length: int, sample_rate: int) -> int:
        del sample_rate
        return length


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

    def output_length(self, length: int, sample_rate: int) -> int:
        del sample_rate
        return round(length / self.factor)


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


@dataclass(frozen=True, slots=True)
class RatePitchEnvelope(AudioOperation):
    rate: tuple[AutomationPoint, ...] = ()
    pitch_semitones: tuple[AutomationPoint, ...] = ()
    interpolation: str = "linear"
    time_base: str = "output"
    type: str = "rate_pitch_envelope"

    def __post_init__(self) -> None:
        try:
            rate = tuple(self.rate)
            pitch_semitones = tuple(self.pitch_semitones)
        except TypeError as exc:
            raise AudioValidationError("envelope curves must be sequences") from exc
        self._validate_curve(rate, "rate", positive=True)
        self._validate_curve(pitch_semitones, "pitch_semitones")
        if not rate and not pitch_semitones:
            raise AudioValidationError("rate/pitch envelope must not be empty")
        if self.interpolation != "linear":
            raise AudioValidationError(
                f"unsupported envelope interpolation: {self.interpolation!r}"
            )
        if self.time_base != "output":
            raise AudioValidationError(f"unsupported envelope time base: {self.time_base!r}")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "pitch_semitones", pitch_semitones)

    @staticmethod
    def _validate_curve(
        points: tuple[AutomationPoint, ...],
        name: str,
        *,
        positive: bool = False,
    ) -> None:
        if any(not isinstance(point, AutomationPoint) for point in points):
            raise AudioValidationError(f"{name} curve must contain AutomationPoint values")
        if points and points[0].seconds != 0.0:
            raise AudioValidationError(f"{name} curve must start at 0 seconds")
        if any(
            current.seconds <= previous.seconds
            for previous, current in zip(points[:-1], points[1:], strict=True)
        ):
            raise AudioValidationError(f"{name} point times must be strictly increasing")
        if positive and any(point.value <= 0 for point in points):
            raise AudioValidationError("rate values must be > 0")

    @classmethod
    def transition(
        cls,
        *,
        from_rate: float = 1.0,
        to_rate: float = 1.0,
        rate_seconds: float = 0.0,
        from_semitones: float = 0.0,
        to_semitones: float = 0.0,
        pitch_seconds: float = 0.0,
    ) -> RatePitchEnvelope:
        from_rate = _finite_float(from_rate, "from_rate")
        to_rate = _finite_float(to_rate, "to_rate")
        rate_seconds = _finite_float(rate_seconds, "rate_seconds")
        from_semitones = _finite_float(from_semitones, "from_semitones")
        to_semitones = _finite_float(to_semitones, "to_semitones")
        pitch_seconds = _finite_float(pitch_seconds, "pitch_seconds")
        if from_rate <= 0 or to_rate <= 0:
            raise AudioValidationError("rate values must be > 0")
        if rate_seconds < 0 or pitch_seconds < 0:
            raise AudioValidationError("transition durations must be >= 0")

        rate: tuple[AutomationPoint, ...] = ()
        if from_rate != to_rate:
            rate = (
                (AutomationPoint(0.0, to_rate),)
                if rate_seconds == 0
                else (
                    AutomationPoint(0.0, from_rate),
                    AutomationPoint(rate_seconds, to_rate),
                )
            )
        elif to_rate != 1.0:
            rate = (AutomationPoint(0.0, to_rate),)

        pitch_semitones: tuple[AutomationPoint, ...] = ()
        if from_semitones != to_semitones:
            pitch_semitones = (
                (AutomationPoint(0.0, to_semitones),)
                if pitch_seconds == 0
                else (
                    AutomationPoint(0.0, from_semitones),
                    AutomationPoint(pitch_seconds, to_semitones),
                )
            )
        elif to_semitones != 0.0:
            pitch_semitones = (AutomationPoint(0.0, to_semitones),)

        return cls(rate=rate, pitch_semitones=pitch_semitones)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "time_base": self.time_base,
            "interpolation": self.interpolation,
            "rate": [{"seconds": point.seconds, "factor": point.value} for point in self.rate],
            "pitch": [
                {"seconds": point.seconds, "semitones": point.value}
                for point in self.pitch_semitones
            ],
        }

    def map_offset_at_rate(
        self,
        offset: int,
        length: int,
        sample_rate: int,
    ) -> int:
        del length
        if not self.rate:
            return offset
        rate_points = tuple((point.seconds, point.value) for point in self.rate)
        return map_source_frame(offset, sample_rate, rate_points)

    def output_length(self, length: int, sample_rate: int) -> int:
        rate_points = tuple((point.seconds, point.value) for point in self.rate)
        return output_frames_for_input_frames(length, sample_rate, rate_points)


Operation = Gain | PitchShift | Tempo | FadeIn | FadeOut | RatePitchEnvelope


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
        if kind == "rate_pitch_envelope":
            rate = tuple(
                AutomationPoint(point["seconds"], point["factor"]) for point in value["rate"]
            )
            pitch_semitones = tuple(
                AutomationPoint(point["seconds"], point["semitones"]) for point in value["pitch"]
            )
            return RatePitchEnvelope(
                rate=rate,
                pitch_semitones=pitch_semitones,
                interpolation=value["interpolation"],
                time_base=value["time_base"],
            )
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
    if isinstance(operation, RatePitchEnvelope):
        rate_points = tuple((point.seconds, point.value) for point in operation.rate)
        pitch_points = tuple((point.seconds, point.value) for point in operation.pitch_semitones)
        try:
            result = apply_speech_effects_envelope(
                values,
                sample_rate=sample_rate,
                rate_points=rate_points,
                pitch_points=pitch_points,
                time_base=operation.time_base,
                interpolation=operation.interpolation,
                method="wsola",
            )
        except InvalidParameterError as exc:
            raise AudioValidationError(str(exc)) from exc
        return np.ascontiguousarray(result, dtype=np.float32)
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
