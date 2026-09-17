from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .errors import AssemblyError

PeakPolicy = Literal["reduce_gain", "error"]


@dataclass(frozen=True, slots=True)
class LoudnessPolicy:
    """Complete-output loudness and true-peak policy."""

    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"

    def __post_init__(self) -> None:
        if self.peak_policy not in {"reduce_gain", "error"}:
            raise ValueError(f"unknown peak policy: {self.peak_policy!r}")
        for name in ("target_lufs", "true_peak_ceiling_dbtp"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
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


def apply_complete_output_loudness(
    audio: np.ndarray,
    sample_rate: int,
    policy: LoudnessPolicy,
) -> LoudnessResult:
    """Apply one gain to a complete waveform after assembly."""

    values = np.asarray(audio, dtype=np.float32)
    if policy.target_lufs is None or values.size == 0:
        return LoudnessResult(values, 0.0, None, False, None)
    try:
        from audiosig import apply_gain_db, measure_loudness
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise AssemblyError(
            "complete-output loudness requires the optional 'audiosig' dependency; "
            "install utterrender[prosody]"
        ) from exc

    metrics = measure_loudness(values, sample_rate=sample_rate)
    if not math.isfinite(metrics.integrated_lufs):
        return LoudnessResult(
            values,
            0.0,
            None,
            False,
            metrics.true_peak_dbtp,
            "digital silence has no integrated LUFS; output unchanged",
        )

    requested_gain_db = policy.target_lufs - metrics.integrated_lufs
    safe_gain_db = math.inf
    if policy.true_peak_ceiling_dbtp is not None and math.isfinite(metrics.true_peak_dbtp):
        safe_gain_db = policy.true_peak_ceiling_dbtp - metrics.true_peak_dbtp
    if policy.peak_policy == "error" and requested_gain_db > safe_gain_db:
        raise AssemblyError(
            "complete-output loudness target exceeds true-peak ceiling: "
            f"requested gain={requested_gain_db:.3f} dB, safe gain={safe_gain_db:.3f} dB"
        )
    applied_gain_db = requested_gain_db if requested_gain_db <= 0 else min(requested_gain_db, safe_gain_db)
    normalized = np.asarray(apply_gain_db(values, applied_gain_db), dtype=np.float32)
    return LoudnessResult(
        normalized,
        applied_gain_db,
        metrics.integrated_lufs,
        math.isclose(applied_gain_db, requested_gain_db, abs_tol=1e-9),
        metrics.true_peak_dbtp,
    )
