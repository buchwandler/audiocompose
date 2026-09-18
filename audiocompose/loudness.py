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
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite real number or None")


@dataclass(frozen=True, slots=True)
class LoudnessResult:
    audio: np.ndarray
    applied_gain_db: float
    measured_lufs: float | None
    target_reached: bool
    true_peak_dbtp: float | None
    warning: str | None = None


def _biquad(
    values: np.ndarray, coefficients: tuple[float, float, float, float, float]
) -> np.ndarray:
    b0, b1, b2, a1, a2 = coefficients
    output = np.empty_like(values, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    for index, value in enumerate(values):
        current = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = current
        x2, x1 = x1, value
        y2, y1 = y1, current
    return output


def _high_shelf(sample_rate: int) -> tuple[float, float, float, float, float]:
    frequency = 1681.974450955533
    gain = 4.0
    q = 0.7071752369554196
    k = math.tan(math.pi * frequency / sample_rate)
    vh = 10.0 ** (gain / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    return (
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    )


def _rlb_high_pass(sample_rate: int) -> tuple[float, float, float, float, float]:
    frequency = 38.13547087602444
    q = 0.5003270373253953
    k = math.tan(math.pi * frequency / sample_rate)
    a0 = 1.0 + k / q + k * k
    return (
        1.0 / a0,
        -2.0 / a0,
        1.0 / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    )


def _k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return _biquad(
        _biquad(audio.astype(np.float64), _high_shelf(sample_rate)), _rlb_high_pass(sample_rate)
    )


def _integrated_loudness(audio: np.ndarray, sample_rate: int) -> float:
    if not audio.size:
        return -math.inf
    block_size = max(1, round(0.4 * sample_rate))
    hop = max(1, round(0.1 * sample_rate))
    if len(audio) <= block_size:
        blocks = [audio]
    else:
        blocks = [
            audio[start : start + block_size]
            for start in range(0, len(audio) - block_size + 1, hop)
        ]
    powers = np.asarray([float(np.mean(block * block)) for block in blocks], dtype=np.float64)
    finite = powers > 1e-15
    if not np.any(finite):
        return -math.inf
    absolute = -0.691 + 10.0 * np.log10(np.maximum(powers, 1e-15))
    gated = powers[absolute > -70.0]
    if not gated.size:
        return -math.inf
    relative_gate = -0.691 + 10.0 * math.log10(float(np.mean(gated))) - 10.0
    gated = gated[(-0.691 + 10.0 * np.log10(np.maximum(gated, 1e-15))) > relative_gate]
    if not gated.size:
        return -math.inf
    return -0.691 + 10.0 * math.log10(float(np.mean(gated)))


def _true_peak(audio: np.ndarray) -> float:
    if not audio.size:
        return -math.inf
    oversample = 4
    if len(audio) < 2:
        peak = float(np.max(np.abs(audio)))
    else:
        spectrum = np.fft.rfft(audio.astype(np.float64))
        peak = float(oversample * np.max(np.abs(np.fft.irfft(spectrum, n=len(audio) * oversample))))
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


def _measure(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    values = np.asarray(audio, dtype=np.float64)
    if not values.size:
        return -math.inf, -math.inf
    return _integrated_loudness(_k_weight(values, sample_rate), sample_rate), _true_peak(values)


def apply_complete_output_loudness(
    audio: np.ndarray,
    sample_rate: int,
    policy: LoudnessPolicy,
) -> LoudnessResult:
    values = np.asarray(audio, dtype=np.float32)
    if not values.size:
        return LoudnessResult(values, 0.0, None, False, None)

    measured, peak = _measure(values, sample_rate)
    requested = (
        0.0
        if policy.target_lufs is None or not math.isfinite(measured)
        else policy.target_lufs - measured
    )
    safe = math.inf
    if policy.true_peak_ceiling_dbtp is not None and math.isfinite(peak):
        safe = policy.true_peak_ceiling_dbtp - peak
    if policy.peak_policy == "error" and requested > safe:
        raise CompositionError(
            "complete-output loudness target exceeds true-peak ceiling: "
            f"requested gain={requested:.3f} dB, safe gain={safe:.3f} dB"
        )
    applied = min(requested, safe)
    normalized = (values * (10.0 ** (applied / 20.0))).astype(np.float32)
    post_loudness, post_peak = _measure(normalized, sample_rate)
    target_reached = policy.target_lufs is not None and math.isclose(
        post_loudness, policy.target_lufs, abs_tol=0.1
    )
    warning = None
    if policy.target_lufs is not None and not target_reached:
        warning = "true-peak ceiling limited the requested loudness target"
    return LoudnessResult(normalized, applied, post_loudness, target_reached, post_peak, warning)
