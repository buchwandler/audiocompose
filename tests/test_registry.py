from __future__ import annotations

from utterrender import VoiceBindings, VoiceInfo


def test_voice_language_matching_handles_region_forms() -> None:
    voice = VoiceInfo("x:v", "x", "v", ("de-de",))
    assert voice.supports_language("de-DE")
    assert voice.supports_language("de")
    assert not voice.supports_language("en-us")


def test_voice_bindings() -> None:
    bindings = VoiceBindings({"narrator": "kokoro:af_heart"})
    assert bindings.resolve("narrator") == "kokoro:af_heart"
    bindings.bind("quote", "piper:en_US-lessac-medium")
    assert bindings.to_dict()["quote"].startswith("piper:")
