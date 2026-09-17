from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from utterplan import PlanSegment, SegmentDirectives

from utterrender.plugins.base import RenderRequest
from utterrender.plugins.kokoro import KokoroPlugin, _LoadedKokoro
from utterrender.prosody import ResolvedProsody
from utterrender.voices import VoiceInfo


class FakeSession:
    def __init__(self) -> None:
        self.arguments = None
        self.closed = False

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="ref_s", type="tensor(float)"),
            SimpleNamespace(name="speed", type="tensor(float)"),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="waveform"), SimpleNamespace(name="pred_dur")]

    def run(self, _outputs, arguments):
        self.arguments = arguments
        return [np.asarray([[0.25, 0.5]], dtype=np.float32), np.asarray([1, 2])]

    def close(self) -> None:
        self.closed = True


class FakeG2P:
    def __init__(self) -> None:
        self.calls = []
        self.OverrideSpan = SimpleNamespace

    def phonemize(self, text, language, **kwargs):
        self.calls.append((text, language, kwargs))
        return SimpleNamespace(token_ids=[4, 5], phonemes="ab")


def make_request(**kwargs) -> RenderRequest:
    return RenderRequest(
        segment=PlanSegment(
            id="segment-1",
            text="Prepared text.",
            spoken_start=0,
            spoken_end=14,
            language="en-us",
            directives=SegmentDirectives(),
        ),
        voice=VoiceInfo(
            "kokoro:test/af_heart",
            "kokoro",
            "af_heart",
            ("en-us",),
            model_id="test",
            sample_rate=24000,
        ),
        prosody=ResolvedProsody(rate=0.75, requested=frozenset({"rate"})),
        **kwargs,
    )


def test_kokoro_consumes_prepared_segment_once_and_uses_native_speed() -> None:
    session = FakeSession()
    g2p = FakeG2P()
    plugin = KokoroPlugin(
        voice_loader=lambda _voice: _LoadedKokoro(
            session, {"af_heart": np.zeros((512, 256), dtype=np.float32)}, 24000
        )
    )
    plugin._module = lambda: g2p

    fragment = plugin.render(make_request())

    assert fragment.audio.tolist() == [0.25, 0.5]
    assert fragment.realized_prosody == frozenset({"rate"})
    assert g2p.calls[0][0:2] == ("Prepared text.", "en-us")
    assert session.arguments["input_ids"].tolist() == [[0, 4, 5, 0]]
    assert session.arguments["speed"].tolist() == [0.75]
    assert fragment.metadata["pred_dur"].tolist() == [1, 2]


def test_kokoro_model_profile_discovery_does_not_open_a_session() -> None:
    plugin = KokoroPlugin(
        model_profiles={
            "github/v1/fp32": {
                "languages": ("en-us",),
                "voices": ("af_heart",),
                "quality": "fp32",
            }
        }
    )

    models = tuple(plugin.models(language="en-us"))
    voices = tuple(plugin.voices(language="en-us"))

    assert models[0].id == "kokoro:github/v1/fp32"
    assert voices[0].id == "kokoro:github/v1/fp32/af_heart"
    assert voices[0].installed is False


def test_kokoro_close_releases_cached_session() -> None:
    session = FakeSession()
    plugin = KokoroPlugin(
        voice_loader=lambda _voice: _LoadedKokoro(
            session, {"af_heart": np.zeros((1, 256), dtype=np.float32)}, 24000
        )
    )

    plugin.ensure_voice(make_request().voice)
    plugin.close()

    assert session.closed
