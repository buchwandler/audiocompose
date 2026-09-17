from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from utterplan import PlanSegment, SegmentDirectives

from utterrender import RenderRequest, VoiceInfo
from utterrender.plugins.piper import PiperPlugin, _LoadedPiperVoice
from utterrender.prosody import ResolvedProsody


class FakeSession:
    def __init__(self) -> None:
        self.arguments = None

    def run(self, outputs, arguments):
        del outputs
        self.arguments = arguments
        return [np.array([[0.25, 0.5]], dtype=np.float32)]

    def close(self) -> None:
        return None


class FakeFrontend:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, phonemes):
        values = tuple(phonemes)
        self.encoded.append(" ".join(values))
        return SimpleNamespace(ids=(4, 5))

    def phonemize_prepared(self, text: str, *, annotations=None):
        del text, annotations
        return SimpleNamespace(sentences=(SimpleNamespace(ids=(1, 2)),), warnings=())

    def close(self) -> None:
        return None


def request(*, pronunciation=None, speaker="bob") -> RenderRequest:
    segment = PlanSegment(
        id="seg",
        text="Prepared text.",
        spoken_start=0,
        spoken_end=14,
        language="en-us",
        directives=SegmentDirectives(),
    )
    return RenderRequest(
        segment=segment,
        voice=VoiceInfo(
            "piper:test",
            "piper",
            "test",
            ("en-us",),
            speakers=("alice", "bob"),
            speaker_id_map={"alice": 0, "bob": 1},
        ),
        prosody=ResolvedProsody(rate=0.5, requested=frozenset({"rate"})),
        pronunciation=pronunciation,
        speaker=speaker,
    )


def test_piper_consumes_prepared_text_and_uses_native_rate_without_normalizing() -> None:
    session = FakeSession()
    frontend = FakeFrontend()
    config = SimpleNamespace(
        sample_rate=22050,
        num_speakers=2,
        speaker_id_map={"alice": 0, "bob": 1},
        default_speaker_id=0,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
    )
    plugin = PiperPlugin()
    plugin._voices["piper:test"] = _LoadedPiperVoice(session, config, frontend)

    fragment = plugin.render(request())

    assert fragment.audio.tolist() == [0.25, 0.5]
    assert fragment.realized_prosody == frozenset({"rate"})
    assert session.arguments["sid"].tolist() == [1]
    assert session.arguments["scales"][1] == 2.0
    assert frontend.encoded == []


def test_piper_pronunciation_override_bypasses_text_phonemization() -> None:
    session = FakeSession()
    frontend = FakeFrontend()
    config = SimpleNamespace(
        sample_rate=22050,
        num_speakers=1,
        speaker_id_map={},
        default_speaker_id=None,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
    )
    plugin = PiperPlugin()
    plugin._voices["piper:test"] = _LoadedPiperVoice(session, config, frontend)

    plugin.render(request(pronunciation={"phoneme_ids": [7, 8]}, speaker=None))

    assert session.arguments["input"].tolist() == [[7, 8]]
    assert frontend.encoded == []
