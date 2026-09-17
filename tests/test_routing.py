from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from utterplan import SegmentDirectives, UtterancePlan, VoiceDirective

from utterrender import VoiceBindings, VoiceInfo, VoiceRoutingError
from utterrender.plugins.registry import PluginRegistry
from utterrender.routing import VoiceRouter

FIXTURE = Path(__file__).parent / "fixtures" / "markers.utterplan.json"


class RoutingPlugin:
    def __init__(self, plugin_id: str, voices: tuple[VoiceInfo, ...]) -> None:
        self.id = plugin_id
        self._voices = voices

    def voices(self, *, language: str | None = None, model: str | None = None):
        del model
        if language is None:
            return self._voices
        return tuple(voice for voice in self._voices if voice.supports_language(language))

    def ensure_voice(self, voice: VoiceInfo, *, progress=None) -> None:
        del voice, progress

    def close(self) -> None:
        return None


def plan_segment(**directives):
    plan = UtterancePlan.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
    segment = plan.segments[0]
    return replace(segment, directives=SegmentDirectives(**directives))


def test_unbound_logical_voice_does_not_use_default() -> None:
    voice = VoiceInfo("fake:default", "fake", "default", ("en-us",))
    router = VoiceRouter(
        PluginRegistry((RoutingPlugin("fake", (voice,)),)),
        VoiceBindings(),
        default_voice=voice.id,
    )

    with pytest.raises(VoiceRoutingError, match="not bound"):
        router.resolve(plan_segment(voice=VoiceDirective("narrator")))


def test_unbound_logical_voice_can_explicitly_use_default_policy() -> None:
    voice = VoiceInfo("fake:default", "fake", "default", ("en-us",))
    router = VoiceRouter(
        PluginRegistry((RoutingPlugin("fake", (voice,)),)),
        VoiceBindings(),
        default_voice=voice.id,
        unbound_voice_policy="default",
    )

    assert router.resolve(plan_segment(voice=VoiceDirective("narrator"))) == voice


def test_preferred_plugin_controls_automatic_selection() -> None:
    first = VoiceInfo("a:first", "a", "first", ("en-us",))
    preferred = VoiceInfo("b:preferred", "b", "preferred", ("en-us",))
    router = VoiceRouter(
        PluginRegistry(
            (
                RoutingPlugin("a", (first,)),
                RoutingPlugin("b", (preferred,)),
            )
        ),
        VoiceBindings(),
        preferred_plugins=("b",),
    )

    assert router.resolve(plan_segment()) == preferred


def test_explicit_voice_and_language_mismatch_are_validated() -> None:
    voice = VoiceInfo("fake:one", "fake", "one", ("de-de",))
    router = VoiceRouter(PluginRegistry((RoutingPlugin("fake", (voice,)),)), VoiceBindings())

    with pytest.raises(VoiceRoutingError, match="does not advertise"):
        router.resolve(plan_segment(voice=VoiceDirective("fake:one")))


def test_speaker_names_and_ids_resolve() -> None:
    voice = VoiceInfo(
        "fake:multi",
        "fake",
        "multi",
        ("en-us",),
        speakers=("alice", "bob"),
        speaker_id_map={"alice": 0, "bob": 1},
    )

    assert voice.resolve_speaker("alice") == 0
    assert voice.resolve_speaker(1) == 1
    with pytest.raises(ValueError, match="unknown speaker"):
        voice.resolve_speaker("unknown")
