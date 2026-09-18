from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    frame_ms: float = 20.0
    hop_ms: float = 5.0
    active_threshold_dbfs: float = -40.0
    release_threshold_dbfs: float = -45.0
    min_active_ms: float = 30.0
    min_gap_ms: float = 60.0

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.release_threshold_dbfs > self.active_threshold_dbfs:
            raise ValueError("release threshold must not exceed active threshold")
        if self.min_active_ms < 0 or self.min_gap_ms < 0:
            raise ValueError("minimum durations must be non-negative")


@dataclass(frozen=True, slots=True)
class ActivityRegion:
    start_sample: int
    end_sample: int
    rms_dbfs: float
    peak_dbfs: float

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class AcousticGap:
    start_sample: int
    end_sample: int

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class ActivityReport:
    sample_rate: int
    frames: int
    activity: tuple[ActivityRegion, ...]
    gaps: tuple[AcousticGap, ...]
    leading_gap: AcousticGap | None
    trailing_gap: AcousticGap | None

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate

    def seconds(self, samples: int) -> float:
        return samples / self.sample_rate


def _dbfs(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return -math.inf, -math.inf
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(values)))
    return (
        20.0 * math.log10(rms) if rms > 0 else -math.inf,
        20.0 * math.log10(peak) if peak > 0 else -math.inf,
    )


def _raw_regions(
    values: np.ndarray,
    sample_rate: int,
    config: ActivityConfig,
) -> list[tuple[int, int]]:
    frame_size = max(1, round(config.frame_ms * sample_rate / 1000.0))
    hop_size = max(1, round(config.hop_ms * sample_rate / 1000.0))
    active = False
    regions: list[tuple[int, int]] = []
    start = 0
    for frame_start in range(0, len(values), hop_size):
        frame = values[frame_start : frame_start + frame_size]
        rms_dbfs, _ = _dbfs(frame)
        if not active and rms_dbfs >= config.active_threshold_dbfs:
            active = True
            start = frame_start
        elif active and rms_dbfs < config.release_threshold_dbfs:
            end = min(len(values), frame_start + frame_size)
            regions.append((start, end))
            active = False
    if active:
        regions.append((start, len(values)))
    minimum = round(config.min_active_ms * sample_rate / 1000.0)
    return [region for region in regions if region[1] - region[0] >= minimum]


def _merge_regions(regions: list[tuple[int, int]], minimum_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in regions:
        if merged and start - merged[-1][1] < minimum_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def analyze_activity(
    audio: np.ndarray,
    sample_rate: int,
    config: ActivityConfig | None = None,
) -> ActivityReport:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if config is None:
        config = ActivityConfig()
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("audio must be a finite one-dimensional waveform")
    raw = _raw_regions(values, sample_rate, config)
    minimum_gap = round(config.min_gap_ms * sample_rate / 1000.0)
    merged = _merge_regions(raw, minimum_gap)
    activity = tuple(ActivityRegion(start, end, *_dbfs(values[start:end])) for start, end in merged)
    gaps: list[AcousticGap] = []
    if activity:
        for left, right in zip(activity, activity[1:], strict=False):
            gap = AcousticGap(left.end_sample, right.start_sample)
            if gap.duration_samples >= minimum_gap:
                gaps.append(gap)
        leading = (
            AcousticGap(0, activity[0].start_sample)
            if activity[0].start_sample >= minimum_gap
            else None
        )
        trailing = (
            AcousticGap(activity[-1].end_sample, len(values))
            if len(values) - activity[-1].end_sample >= minimum_gap
            else None
        )
    else:
        whole = AcousticGap(0, len(values))
        leading = whole if whole.duration_samples >= minimum_gap else None
        trailing = None
    if activity:
        gaps_with_edges = ([leading] if leading else []) + gaps + ([trailing] if trailing else [])
    else:
        gaps_with_edges = [leading] if leading else []
    return ActivityReport(
        sample_rate, len(values), activity, tuple(gaps_with_edges), leading, trailing
    )


def measure_gap_near(
    report: ActivityReport,
    sample_offset: int,
    *,
    search_window_ms: float = 500.0,
) -> AcousticGap | None:
    window = round(search_window_ms * report.sample_rate / 1000.0)
    candidates = [
        gap
        for gap in report.gaps
        if gap.end_sample >= sample_offset - window and gap.start_sample <= sample_offset + window
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda gap: (
            0
            if gap.start_sample <= sample_offset <= gap.end_sample
            else min(abs(sample_offset - gap.start_sample), abs(sample_offset - gap.end_sample))
        ),
    )
