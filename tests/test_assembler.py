from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import utterplan

Plan = getattr(utterplan, "UtterPlan", getattr(utterplan, "TTSPlan"))

from utterrender import AudioFragment, AssemblyError, assemble, samples_for_duration


FIXTURE = Path(__file__).parent / "fixtures" / "markers.utterplan.json"


def load_plan() -> Plan:
    return Plan.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_samples_for_duration_uses_nearest_sample() -> None:
    assert samples_for_duration(0.0015, 1000) == 2
    assert samples_for_duration(0.6, 10) == 6


def test_assemble_inserts_resolved_pause_and_resolves_boundary_marker() -> None:
    plan = load_plan()
    result = assemble(
        plan,
        [
            AudioFragment("seg-000000", np.ones(4, dtype=np.float32), 10),
            AudioFragment("seg-000001", np.full(5, 0.5, dtype=np.float32), 10),
        ],
    )

    # 4 samples + 0.6 s at 10 Hz + 5 samples.
    assert result.audio.size == 15
    np.testing.assert_array_equal(result.audio[:4], np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(result.audio[4:10], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(result.audio[10:], np.full(5, 0.5, dtype=np.float32))

    assert result.segments[0].audio_start_sample == 0
    assert result.segments[0].audio_end_sample == 4
    assert result.segments[0].pause_after_samples == 6
    assert result.segments[1].audio_start_sample == 10
    assert result.markers[0].timing == "resolved"
    assert result.markers[0].sample_offset == 4


def test_processor_can_modify_audio_without_knowing_assembly() -> None:
    plan = load_plan()

    def halve(fragment: AudioFragment, *, segment: object) -> AudioFragment:
        return AudioFragment(
            segment_id=fragment.segment_id,
            audio=fragment.audio * 0.5,
            sample_rate=fragment.sample_rate,
            metadata=fragment.metadata,
        )

    result = assemble(
        plan,
        [
            AudioFragment("seg-000000", np.ones(2, dtype=np.float32), 10),
            AudioFragment("seg-000001", np.ones(3, dtype=np.float32), 10),
        ],
        processors=[halve],
    )
    assert result.audio[0] == pytest.approx(0.5)
    assert result.audio[-1] == pytest.approx(0.5)


def test_missing_fragment_is_rejected() -> None:
    plan = load_plan()
    with pytest.raises(AssemblyError, match="missing fragment"):
        assemble(
            plan,
            [AudioFragment("seg-000000", np.ones(2, dtype=np.float32), 10)],
        )


def test_mixed_sample_rates_are_rejected() -> None:
    plan = load_plan()
    with pytest.raises(AssemblyError, match="one sample rate"):
        assemble(
            plan,
            [
                AudioFragment("seg-000000", np.ones(2, dtype=np.float32), 10),
                AudioFragment("seg-000001", np.ones(3, dtype=np.float32), 12),
            ],
        )
