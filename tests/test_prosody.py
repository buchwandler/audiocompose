from __future__ import annotations

import numpy as np
import pytest
from utterplan import PlanSegment, ProsodyDirective, SegmentDirectives

from utterrender import (
    AudioFragment,
    AudioSigProsodyProcessor,
    FragmentValidationError,
    parse_pitch,
    parse_rate,
    parse_volume,
    resolve_prosody,
)


def segment_with_prosody(**values: str) -> PlanSegment:
    return PlanSegment(
        id="seg",
        text="Hello",
        spoken_start=0,
        spoken_end=5,
        language="en-us",
        directives=SegmentDirectives(prosody=ProsodyDirective(**values)),
    )


def test_named_prosody_values_match_existing_kokoro_semantics() -> None:
    assert parse_rate("slow") == pytest.approx(0.75)
    assert parse_rate("fast") == pytest.approx(1.25)
    assert parse_pitch("low") == pytest.approx(-2.0)
    assert parse_volume("loud") == pytest.approx(6.0)


def test_relative_prosody_values_are_normalized() -> None:
    assert parse_rate("+20%") == pytest.approx(1.2)
    assert parse_rate("80%") == pytest.approx(0.8)
    assert parse_pitch("+2st") == pytest.approx(2.0)
    assert parse_volume("+6dB") == pytest.approx(6.0)


def test_resolve_prosody_tracks_explicit_axes() -> None:
    resolved = resolve_prosody(ProsodyDirective(rate="slow", volume="medium"))
    assert resolved.rate == pytest.approx(0.75)
    assert resolved.gain_db == pytest.approx(0.0)
    assert resolved.requested == frozenset({"rate", "volume"})


def test_fragment_rejects_unknown_realized_axis() -> None:
    with pytest.raises(FragmentValidationError, match="unsupported axes"):
        AudioFragment(
            "seg",
            np.ones(2, dtype=np.float32),
            100,
            realized_prosody=frozenset({"tempo"}),  # type: ignore[arg-type]
        )


def test_audiosig_processor_applies_slow_after_inference() -> None:
    segment = segment_with_prosody(rate="slow")
    fragment = AudioFragment("seg", np.ones(1000, dtype=np.float32) * 0.1, 1000)

    result = AudioSigProsodyProcessor(method="wsola")(fragment, segment=segment)

    # 0.75x speech rate means a waveform around 1 / 0.75 times as long.
    assert result.audio.size == round(1000 / 0.75)
    assert result.realized_prosody == frozenset({"rate"})
    assert result.metadata["utterrender.prosody"]["rate"] == pytest.approx(0.75)


def test_native_rate_is_not_applied_twice() -> None:
    segment = segment_with_prosody(rate="slow", volume="loud")
    fragment = AudioFragment(
        "seg",
        np.ones(1000, dtype=np.float32) * 0.1,
        1000,
        realized_prosody=frozenset({"rate"}),
    )

    result = AudioSigProsodyProcessor(method="wsola")(fragment, segment=segment)

    assert result.audio.size == 1000
    assert result.realized_prosody == frozenset({"rate", "volume"})
    assert result.metadata["utterrender.prosody"]["rate"] == pytest.approx(1.0)
    assert result.metadata["utterrender.prosody"]["gain_db"] == pytest.approx(6.0)
