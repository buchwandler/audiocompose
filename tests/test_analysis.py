from __future__ import annotations

import numpy as np

from audiocompose import ActivityConfig, analyze_activity, measure_gap_near


def test_activity_analysis_reports_internal_gap_in_sample_coordinates() -> None:
    sample_rate = 1_000
    audio = np.zeros(1_000, dtype=np.float32)
    audio[200:300] = 0.5
    audio[500:800] = 0.5
    report = analyze_activity(
        audio,
        sample_rate,
        ActivityConfig(
            frame_ms=20,
            hop_ms=5,
            active_threshold_dbfs=-20,
            release_threshold_dbfs=-25,
            min_active_ms=30,
            min_gap_ms=100,
        ),
    )

    assert len(report.activity) == 2
    assert report.leading_gap is not None
    assert report.trailing_gap is not None
    gap = measure_gap_near(report, 400, search_window_ms=150)
    assert gap is not None
    assert abs(gap.start_sample - 300) <= 25
    assert abs(gap.end_sample - 500) <= 25
    assert abs(gap.duration_samples - 200) <= 50
    assert abs(report.seconds(gap.duration_samples) - 0.2) <= 0.05
