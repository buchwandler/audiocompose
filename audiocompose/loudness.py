from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from audiosig import (
    AudioSignalError,
    measure_loudness,
    sample_peak_dbfs,
    true_peak_dbtp,
)

from .errors import AudioValidationError, CompositionError
from .wav import _as_finite_mono_float32

PeakPolicy = Literal["reduce_gain", "error"]


@dataclass(frozen=True, slots=True)
class LoudnessPolicy:
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"

    def __post_init__(self) -> None:
        if self.peak_policy not in ("reduce_gain", "error"):
            raise AudioValidationError(f"unknown peak policy: {self.peak_policy!r}")
        for name in ("target_lufs", "true_peak_ceiling_dbtp"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise AudioValidationError(f"{name} must be a finite real number or None")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class LoudnessMetrics:
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    sample_peak_dbfs: float | None = None


@dataclass(frozen=True, slots=True)
class LoudnessResult:
    audio: np.ndarray
    before: LoudnessMetrics
    after: LoudnessMetrics
    target_lufs: float | None
    true_peak_ceiling_dbtp: float | None
    requested_gain_db: float
    applied_gain_db: float
    target_reached: bool
    peak_policy: PeakPolicy
    warning: str | None = None

    @property
    def measured_lufs(self) -> float | None:
        """Compatibility alias for the pre-normalization loudness measurement."""
        return self.before.integrated_lufs

    @property
    def true_peak_dbtp(self) -> float | None:
        """Compatibility alias for the post-normalization true peak."""
        return self.after.true_peak_dbtp


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _measure(audio: np.ndarray, sample_rate: int) -> LoudnessMetrics:
    values = _as_finite_mono_float32(audio)
    if not values.size:
        return LoudnessMetrics(None, None, None)
    try:
        if round(0.1 * sample_rate) == 0:
            integrated_lufs = None
            sample_peak = sample_peak_dbfs(values)
            true_peak = true_peak_dbtp(values, sample_rate=sample_rate)
        else:
            measured = measure_loudness(values, sample_rate=sample_rate)
            integrated_lufs = measured.integrated_lufs
            sample_peak = measured.sample_peak_dbfs
            true_peak = measured.true_peak_dbtp
    except AudioSignalError as exc:
        raise AudioValidationError(f"AudioSig loudness measurement failed: {exc}") from exc
    return LoudnessMetrics(
        _finite_or_none(integrated_lufs) if integrated_lufs is not None else None,
        _finite_or_none(true_peak),
        _finite_or_none(sample_peak),
    )


def apply_complete_output_loudness(
    audio: np.ndarray,
    sample_rate: int,
    policy: LoudnessPolicy,
) -> LoudnessResult:
    """Measure and normalize the complete output waveform.

    AudioSig reports no integrated loudness below one complete 400 ms block.
    Such output is represented by ``None`` and receives no LUFS normalization.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    values = _as_finite_mono_float32(audio)
    before = _measure(values, sample_rate)
    requested = (
        0.0
        if policy.target_lufs is None or before.integrated_lufs is None
        else policy.target_lufs - before.integrated_lufs
    )
    safe = math.inf
    if policy.true_peak_ceiling_dbtp is not None and before.true_peak_dbtp is not None:
        safe = policy.true_peak_ceiling_dbtp - before.true_peak_dbtp
    if policy.peak_policy == "error" and requested > safe:
        raise CompositionError(
            "complete-output loudness target exceeds true-peak ceiling: "
            f"requested gain={requested:.3f} dB, safe gain={safe:.3f} dB"
        )
    applied = min(requested, safe)
    normalized = (values * (10.0 ** (applied / 20.0))).astype(np.float32)
    after = _measure(normalized, sample_rate)
    target_reached = (
        policy.target_lufs is not None
        and after.integrated_lufs is not None
        and math.isclose(after.integrated_lufs, policy.target_lufs, abs_tol=0.1)
    )
    warning = None
    if before.integrated_lufs is None:
        warning = "input has no meaningful integrated loudness, such as digital silence"
    elif policy.target_lufs is not None and not target_reached:
        warning = "true-peak ceiling limited the requested loudness target"
    return LoudnessResult(
        audio=normalized,
        before=before,
        after=after,
        target_lufs=policy.target_lufs,
        true_peak_ceiling_dbtp=policy.true_peak_ceiling_dbtp,
        requested_gain_db=requested,
        applied_gain_db=applied,
        target_reached=target_reached,
        peak_policy=policy.peak_policy,
        warning=warning,
    )
