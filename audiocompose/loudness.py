from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .errors import CompositionError

PeakPolicy = Literal["reduce_gain", "error"]


@dataclass(frozen=True, slots=True)
class LoudnessPolicy:
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"

    def __post_init__(self) -> None:
        if self.peak_policy not in {"reduce_gain", "error"}:
            raise ValueError(f"unknown peak policy: {self.peak_policy!r}")
        for name in ("target_lufs", "true_peak_ceiling_dbtp"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be a finite real number or None")


@dataclass(frozen=True, slots=True)
class LoudnessResult:
    audio: np.ndarray
    applied_gain_db: float
    measured_lufs: float | None
    target_reached: bool
    true_peak_dbtp: float | None
    warning: str | None = None


def _measure(audio: np.ndarray) -> tuple[float, float]:
    values = np.asarray(audio, dtype=np.float64)
    if not values.size:
        return -math.inf, -math.inf
    rms = float(np.sqrt(np.mean(values * values)))
    lufs = 20.0 * math.log10(rms) if rms > 0 else -math.inf
    peak = float(np.max(np.abs(values)))
    peak_db = 20.0 * math.log10(peak) if peak > 0 else -math.inf
    return lufs, peak_db


def apply_complete_output_loudness(audio: np.ndarray, sample_rate: int, policy: LoudnessPolicy) -> LoudnessResult:
    del sample_rate
    values = np.asarray(audio, dtype=np.float32)
    if policy.target_lufs is None or values.size == 0:
        return LoudnessResult(values, 0.0, None, False, None)
    measured, peak = _measure(values)
    if not math.isfinite(measured):
        return LoudnessResult(values, 0.0, None, False, peak, "digital silence has no integrated LUFS; output unchanged")
    requested = policy.target_lufs - measured
    safe = math.inf
    if policy.true_peak_ceiling_dbtp is not None and math.isfinite(peak):
        safe = policy.true_peak_ceiling_dbtp - peak
    if policy.peak_policy == "error" and requested > safe:
        raise CompositionError(f"complete-output loudness target exceeds true-peak ceiling: requested gain={requested:.3f} dB, safe gain={safe:.3f} dB")
    applied = requested if requested <= 0 else min(requested, safe)
    normalized = (values * (10.0 ** (applied / 20.0))).astype(np.float32)
    return LoudnessResult(normalized, applied, measured, math.isclose(applied, requested, abs_tol=1e-9), peak)
