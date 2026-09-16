from __future__ import annotations

from typing import Any

import numpy as np

from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..voices import VoiceInfo
from .base import RenderRequest


class PiperPlugin:
    """Compatibility bridge to the current ``pipersynth`` package.

    It deliberately talks to ``PiperVoice`` directly so utterrender remains the owner
    of Plan pauses, routing, prosody fallback, and final assembly.
    """

    id = "piper"

    def __init__(self, *, cache_dir: str | None = None, offline: bool | None = None) -> None:
        self.cache_dir = cache_dir
        self.offline = offline
        self._voices: dict[str, Any] = {}

    def _module(self) -> Any:
        import pipersynth

        return pipersynth

    def voices(self, *, language: str | None = None):
        mod = self._module()
        for item in mod.list_voices(language=language, cache_dir=self.cache_dir, offline=self.offline):
            speakers = tuple(getattr(item, "speaker_id_map", {}).keys())
            yield VoiceInfo(
                id=f"piper:{item.id}",
                plugin=self.id,
                voice_id=item.id,
                languages=(item.language_code,),
                quality=item.quality,
                installed=None,
                downloadable=True,
                speakers=speakers,
                metadata={"name": item.name, "aliases": tuple(item.aliases)},
            )

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities:
        return RenderCapabilities(native_rate=True, multi_speaker=bool(voice.speakers))

    def ensure_voice(self, voice: VoiceInfo) -> None:
        self._get_voice(voice.voice_id)

    def _get_voice(self, voice_id: str) -> Any:
        cached = self._voices.get(voice_id)
        if cached is not None:
            return cached
        mod = self._module()
        loaded = mod.PiperVoice.from_pretrained(
            voice_id,
            cache_dir=self.cache_dir,
            offline=self.offline,
        )
        self._voices[voice_id] = loaded
        return loaded

    def render(self, request: RenderRequest) -> AudioFragment:
        mod = self._module()
        voice = self._get_voice(request.voice.voice_id)
        # Piper's length_scale is inverse speech speed: larger = slower.
        length_scale = 1.0 / request.prosody.rate if "rate" in request.prosody.requested else None
        speaker = request.options.get("speaker_id")
        config = mod.SynthesisConfig(
            speaker_id=speaker,
            length_scale=length_scale,
            normalize_audio=True,
            volume=1.0,
        )
        audio = np.asarray(
            voice.synthesize_array(request.segment.text, config, sentence_silence=0.0),
            dtype=np.float32,
        )
        realized = frozenset({"rate"}) if "rate" in request.prosody.requested else frozenset()
        return AudioFragment(
            segment_id=request.segment.id,
            audio=audio,
            sample_rate=int(voice.config.sample_rate),
            realized_prosody=realized,
            metadata={"plugin": self.id, "voice": request.voice.id},
        )

    def close(self) -> None:
        for voice in self._voices.values():
            close = getattr(voice, "close", None)
            if close is not None:
                close()
        self._voices.clear()
