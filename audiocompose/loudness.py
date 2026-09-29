from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from audiosig import AudioSignalError, analyze_loudness
from audiosig import LoudnessAnalysis as AudioSigAnalysis

from .errors import AudioValidationError, CompositionError
from .wav import _as_finite_mono_float32

PeakPolicy = Literal["reduce_gain", "error"]


@dataclass(frozen=True, slots=True)
class LoudnessPolicy:
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = -1.0
    peak_policy: PeakPolicy = "reduce_gain"
    collect_metrics: bool = True

    def __post_init__(self) -> None:
        if self.peak_policy not in ("reduce_gain", "error"):
            raise AudioValidationError(f"unknown peak policy: {self.peak_policy!r}")
        if not isinstance(self.collect_metrics, bool):
            raise AudioValidationError("collect_metrics must be a bool")
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
    analysis_seconds: float = 0.0
    gain_seconds: float = 0.0
    post_gain_metrics_seconds: float = 0.0

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


def _result_metrics(
    integrated_lufs: float | None, sample_peak: float | None, true_peak: float | None
) -> LoudnessMetrics:
    return LoudnessMetrics(
        integrated_lufs=(_finite_or_none(integrated_lufs) if integrated_lufs is not None else None),
        true_peak_dbtp=_finite_or_none(true_peak) if true_peak is not None else None,
        sample_peak_dbfs=_finite_or_none(sample_peak) if sample_peak is not None else None,
    )


def apply_complete_output_loudness(
    audio: np.ndarray,
    sample_rate: int,
    policy: LoudnessPolicy,
) -> LoudnessResult:
    """Measure and normalize the complete output waveform with one analysis.

    AudioSig reports no integrated loudness below one complete 400 ms block.
    Such output is represented by ``None`` and receives no LUFS normalization.
    Set ``collect_metrics=False`` with no target or ceiling for a true no-op.
    """
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    values = _as_finite_mono_float32(audio)
    no_metrics = LoudnessMetrics(None, None, None)
    if (
        not policy.collect_metrics
        and policy.target_lufs is None
        and policy.true_peak_ceiling_dbtp is None
    ):
        return LoudnessResult(
            audio=values,
            before=no_metrics,
            after=no_metrics,
            target_lufs=None,
            true_peak_ceiling_dbtp=None,
            requested_gain_db=0.0,
            applied_gain_db=0.0,
            target_reached=False,
            peak_policy=policy.peak_policy,
        )

    analysis: AudioSigAnalysis | None = None
    before = no_metrics
    analysis_seconds = 0.0
    if values.size:
        integrated = (policy.collect_metrics or policy.target_lufs is not None) and round(
            0.1 * sample_rate
        ) > 0
        sample_peak = policy.collect_metrics
        true_peak = policy.collect_metrics or policy.true_peak_ceiling_dbtp is not None
        started = time.perf_counter()
        try:
            analysis = analyze_loudness(
                values,
                sample_rate=sample_rate,
                integrated=integrated,
                sample_peak=sample_peak,
                true_peak=true_peak,
            )
        except AudioSignalError as exc:
            raise AudioValidationError(f"AudioSig loudness measurement failed: {exc}") from exc
        analysis_seconds = time.perf_counter() - started
        before = _result_metrics(
            analysis.integrated_lufs, analysis.sample_peak_dbfs, analysis.true_peak_dbtp
        )

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
    gain_started = time.perf_counter()
    normalized = (
        values if applied == 0.0 else (values * (10.0 ** (applied / 20.0))).astype(np.float32)
    )
    gain_seconds = time.perf_counter() - gain_started

    metrics_started = time.perf_counter()
    if analysis is None:
        after = before
    else:
        after_metrics = analysis.metrics_after_gain(applied)
        after = _result_metrics(
            after_metrics.integrated_lufs,
            after_metrics.sample_peak_dbfs,
            after_metrics.true_peak_dbtp,
        )
    post_gain_metrics_seconds = time.perf_counter() - metrics_started

    target_reached = (
        policy.target_lufs is not None
        and after.integrated_lufs is not None
        and math.isclose(after.integrated_lufs, policy.target_lufs, abs_tol=0.1)
    )
    warning = None
    if (
        policy.collect_metrics or policy.target_lufs is not None
    ) and before.integrated_lufs is None:
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
        analysis_seconds=analysis_seconds,
        gain_seconds=gain_seconds,
        post_gain_metrics_seconds=post_gain_metrics_seconds,
    )
