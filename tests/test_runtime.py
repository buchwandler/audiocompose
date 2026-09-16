from __future__ import annotations

import numpy as np
from dataclasses import replace
import utterplan
from utterplan import PlannerConfig, ProsodyDirective, SegmentDirectives, VoiceDirective

Planner = getattr(utterplan, "UtterPlanner", getattr(utterplan, "TTSPlanner"))
from utterplan.hashing import semantic_hash, unit_hash_payload

from utterrender import AudioFragment, RenderCapabilities, Renderer, VoiceInfo
from utterrender.plugins.base import RenderRequest


class _HashUnit:
    def __init__(self, segments, marker_ids, marker_values):
        self.segments = segments
        self.marker_ids = marker_ids
        self.marker_values = marker_values


def rehash(plan):
    segment_by_id = {segment.id: segment for segment in plan.segments}
    marker_by_id = {marker.id: marker for marker in plan.markers}
    units = []
    for unit in plan.units:
        segments = [segment_by_id[item] for item in unit.segment_ids]
        markers = tuple(marker_by_id[item] for item in unit.marker_ids)
        digest = semantic_hash(unit_hash_payload(_HashUnit(segments, unit.marker_ids, markers)))
        units.append(replace(unit, content_hash=digest))
    return replace(plan, units=tuple(units)).with_identity()


class FakePlugin:
    def __init__(self, plugin_id: str, voices: tuple[VoiceInfo, ...], value: float, rate: int):
        self.id = plugin_id
        self._voices = voices
        self.value = value
        self.rate = rate
        self.requests: list[RenderRequest] = []

    def voices(self, *, language: str | None = None):
        if language is None:
            return self._voices
        return tuple(v for v in self._voices if v.supports_language(language))

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities:
        return RenderCapabilities()

    def ensure_voice(self, voice: VoiceInfo) -> None:
        return None

    def render(self, request: RenderRequest) -> AudioFragment:
        self.requests.append(request)
        return AudioFragment(
            segment_id=request.segment.id,
            audio=np.full(100, self.value, dtype=np.float32),
            sample_rate=self.rate,
            metadata={"voice": request.voice.id},
        )

    def close(self) -> None:
        return None


def test_mixed_plugins_route_logical_voices_and_resample() -> None:
    plan = Planner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Hello there.\n\nGeneral Kenobi.")
    plan = rehash(replace(
        plan,
        segments=(
            replace(plan.segments[0], directives=SegmentDirectives(voice=VoiceDirective("narrator"))),
            replace(plan.segments[1], directives=SegmentDirectives(voice=VoiceDirective("quote"))),
        ),
    ))
    a = FakePlugin(
        "a",
        (VoiceInfo("a:narrator", "a", "narrator", ("en-us",)),),
        0.25,
        24000,
    )
    b = FakePlugin(
        "b",
        (VoiceInfo("b:quote", "b", "quote", ("en-us",)),),
        -0.25,
        12000,
    )
    renderer = Renderer(
        plugins=(a, b),
        bindings={"narrator": "a:narrator", "quote": "b:quote"},
        sample_rate=24000,
        apply_prosody=False,
    )
    result = renderer.render(plan)

    assert len(a.requests) == 1
    assert len(b.requests) == 1
    assert {request.voice.id for request in (*a.requests, *b.requests)} == {
        "a:narrator",
        "b:quote",
    }
    # 100 samples at 12 kHz become 200 samples at 24 kHz.
    spans = {span.segment_id: span.audio_samples for span in result.segments}
    assert sorted(spans.values()) == [100, 200]
    assert result.sample_rate == 24000


def test_direct_namespaced_voice_reference_needs_no_binding() -> None:
    # A plan can still opt into a concrete voice explicitly if desired.
    plan = Planner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Hello.")
    plan = rehash(replace(
        plan,
        segments=(
            replace(plan.segments[0], directives=SegmentDirectives(voice=VoiceDirective("fake:one"))),
        ),
    ))
    plugin = FakePlugin(
        "fake",
        (VoiceInfo("fake:one", "fake", "one", ("en-us",)),),
        0.1,
        24000,
    )
    result = Renderer(plugins=(plugin,), apply_prosody=False).render(plan)
    assert result.segments[0].audio_samples == 100
    assert plugin.requests[0].voice.id == "fake:one"


def test_default_voice_is_used_without_plan_voice() -> None:
    plan = Planner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Hello.")
    plugin = FakePlugin(
        "fake",
        (VoiceInfo("fake:one", "fake", "one", ("en-us",)),),
        0.1,
        24000,
    )
    Renderer(plugins=(plugin,), default_voice="fake:one", apply_prosody=False).render(plan)
    assert plugin.requests[0].voice.id == "fake:one"


def test_ssmd_slow_reaches_plugin_as_resolved_semantics() -> None:
    plan = Planner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Slow words.")
    plan = rehash(replace(
        plan,
        segments=(
            replace(
                plan.segments[0],
                directives=SegmentDirectives(prosody=ProsodyDirective(rate="slow")),
            ),
        ),
    ))
    plugin = FakePlugin(
        "fake",
        (VoiceInfo("fake:one", "fake", "one", ("en-us",)),),
        0.1,
        24000,
    )
    Renderer(plugins=(plugin,), default_voice="fake:one", apply_prosody=False).render(plan)
    assert plugin.requests[0].prosody.rate == 0.75
    assert "rate" in plugin.requests[0].prosody.requested


def test_voice_calibration_is_applied_before_assembly() -> None:
    plan = Planner(
        PlannerConfig(language="en-us", document_format="plain", text_preparation="identity")
    ).plan("Hello.")
    plugin = FakePlugin(
        "fake",
        (VoiceInfo("fake:one", "fake", "one", ("en-us",)),),
        0.1,
        24000,
    )
    result = Renderer(
        plugins=(plugin,),
        default_voice="fake:one",
        calibration={"fake:one": 6.020599913279624},
        apply_prosody=False,
    ).render(plan)
    # +6.0206 dB = 2x amplitude.
    span = result.segments[0]
    assert np.allclose(result.audio[span.audio_start_sample:span.audio_end_sample], 0.2)
