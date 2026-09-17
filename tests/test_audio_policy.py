from __future__ import annotations

import numpy as np
import pytest

from utterrender import AudioFragment, VoiceCalibration, VoiceCalibrationRegistry
from utterrender.effects import apply_emphasis
from utterrender.loudness import LoudnessPolicy, apply_complete_output_loudness
from utterrender.wav import prepare_output


def test_model_qualified_calibration_does_not_match_other_model() -> None:
    registry = VoiceCalibrationRegistry()
    registry.set_record(VoiceCalibration("voice", 3.0, model_id="model-a"))

    assert registry.gain_db("voice", model_id="model-a") == pytest.approx(3.0)
    assert registry.gain_db("voice", model_id="model-b") == 0.0


def test_emphasis_policy_can_approximate_with_gain() -> None:
    fragment = AudioFragment("seg", np.ones(2, dtype=np.float32), 100)
    result = apply_emphasis(fragment, 6.020599913279624, policy="gain")

    assert result.audio[0] == pytest.approx(2.0)
    assert result.alignment == fragment.alignment


def test_output_range_policy_is_explicit() -> None:
    samples = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(prepare_output(samples), [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="PCM range"):
        prepare_output(samples, clip_policy="error")


def test_silent_complete_output_is_left_unchanged() -> None:
    result = apply_complete_output_loudness(
        np.zeros(20, dtype=np.float32),
        1000,
        LoudnessPolicy(target_lufs=-18.0),
    )

    assert result.applied_gain_db == 0.0
    assert result.warning is not None
    assert not result.target_reached
