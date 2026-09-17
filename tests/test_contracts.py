from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from utterplan import UtterancePlan

from utterrender import (
    AssetProgressEvent,
    AudioFragment,
    AudioTextSpan,
    ModelInfo,
    RenderCapabilities,
    RenderDiagnostic,
    RenderRequest,
    SegmentRenderContext,
    VoiceInfo,
)
from utterrender.prosody import ResolvedProsody

FIXTURE = Path(__file__).parent / "fixtures" / "markers.utterplan.json"


def test_segment_context_preserves_plan_records() -> None:
    plan = UtterancePlan.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    context = SegmentRenderContext.from_plan(plan, plan.segments[0])

    assert context.plan is plan
    assert context.segment.text == "One."
    assert context.tokens[0].text == "One."
    assert context.annotations[0].id == "annotation-000000"
    assert context.unit_id == "unit-0000"


def test_render_request_exposes_directive_metadata() -> None:
    plan = UtterancePlan.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    context = SegmentRenderContext.from_plan(plan, plan.segments[0])
    voice = VoiceInfo("piper:v", "piper", "v", ("en-us",), speaker_id_map={"narrator": 2})
    request = RenderRequest(
        segment=context.segment,
        voice=voice,
        prosody=ResolvedProsody(),
        context=context,
        speaker="narrator",
        pronunciation={"alphabet": "ipa", "value": "wʌn"},
        emphasis={"mode": "gain"},
        audio_directive={"alt_text": "one"},
        streaming=True,
    )

    assert request.plan_context is context
    assert request.speaker == "narrator"
    assert request.pronunciation["alphabet"] == "ipa"
    assert request.streaming


def test_model_voice_alignment_and_runtime_diagnostics_are_typed() -> None:
    model = ModelInfo(
        id="kokoro:github/v1/fp32/af_heart",
        plugin="kokoro",
        model_id="v1",
        source="github",
        quality="fp32",
        languages=("en-us",),
        sample_rate=24000,
    )
    voice = VoiceInfo(
        "kokoro:github/v1/fp32/af_heart",
        "kokoro",
        "af_heart",
        ("en-us",),
        model=model,
    )
    alignment = AudioTextSpan(0, 3, 0, 120, "word")
    diagnostic = RenderDiagnostic("test.warning", "warning", "test")
    fragment = AudioFragment(
        "seg",
        np.zeros(120, dtype=np.float32),
        24000,
        alignment=(alignment,),
        diagnostics=(diagnostic,),
    )
    event = AssetProgressEvent("kokoro", "download", voice.id, 1, 2)

    assert voice.model_identity == model.id
    assert fragment.alignment == (alignment,)
    assert fragment.diagnostics == (diagnostic,)
    assert event.completed == 1


def test_capabilities_report_native_axes_and_alignment() -> None:
    capabilities = RenderCapabilities(native_rate=True, word_timing=True)

    assert capabilities.native_prosody_axes == frozenset({"rate"})
    assert capabilities.supports_alignment("word")
    assert not capabilities.supports_alignment("phoneme")
