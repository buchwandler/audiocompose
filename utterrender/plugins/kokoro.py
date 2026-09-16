from __future__ import annotations

from typing import Any

import numpy as np

from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..voices import VoiceInfo
from .base import RenderRequest


class KokoroPlugin:
    """Compatibility bridge to the current ``pykokoro`` runtime.

    Short-sentence optimization, model/voice asset handling and Kokoro-specific ONNX
    behavior stay inside PyKokoro for this migration MVP. The public utterrender API does
    not expose PyKokoro objects.
    """

    id = "kokoro"

    def __init__(
        self,
        *,
        offline: bool = False,
        provider: str = "auto",
        short_sentence_config: Any | None = None,
    ) -> None:
        self.offline = offline
        self.provider = provider
        self.short_sentence_config = short_sentence_config
        self._pipelines: dict[tuple[str, str], Any] = {}

    def _module(self) -> Any:
        import pykokoro

        return pykokoro

    def voices(self, *, language: str | None = None):
        mod = self._module()
        inventory = mod.discover_models(offline=self.offline)
        seen: set[str] = set()
        for model in inventory.models:
            detail_by_name = {item.name: item for item in getattr(model, "voice_details", ())}
            for native_name in model.voices:
                if native_name in seen:
                    continue
                detail = detail_by_name.get(native_name)
                languages = (detail.locale,) if detail is not None else tuple(model.languages)
                if language is not None:
                    probe = VoiceInfo("", self.id, native_name, languages)
                    if not probe.supports_language(language):
                        continue
                seen.add(native_name)
                yield VoiceInfo(
                    id=f"kokoro:{native_name}",
                    plugin=self.id,
                    voice_id=native_name,
                    model_id=model.model_id,
                    languages=languages,
                    sample_rate=model.sample_rate,
                    installed=None,
                    downloadable=True,
                    quality=model.qualities[0] if model.qualities else None,
                    metadata={"source": model.source, "status": model.status},
                )

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities:
        # The compatibility bridge leaves lexical prosody for utterrender/audiosig.
        return RenderCapabilities(word_timing=True)

    def ensure_voice(self, voice: VoiceInfo) -> None:
        # Asset download/session creation is lazy in the underlying pipeline.
        self._get_pipeline(voice, voice.languages[0] if voice.languages else "en-us")

    def _get_pipeline(self, voice: VoiceInfo, language: str) -> Any:
        key = (voice.voice_id, language)
        cached = self._pipelines.get(key)
        if cached is not None:
            return cached
        mod = self._module()
        config = mod.PipelineConfig(
            voice=voice.voice_id,
            provider=self.provider,
            generation=mod.GenerationConfig(lang=language),
            retain_segment_audio=False,
            short_sentence_config=self.short_sentence_config,
        )
        pipe = mod.build_pipeline(config=config)
        self._pipelines[key] = pipe
        return pipe

    def render(self, request: RenderRequest) -> AudioFragment:
        pipe = self._get_pipeline(request.voice, request.segment.language)
        # Compatibility limitation: PyKokoro currently receives the prepared segment
        # as plain text and therefore performs its own local frontend preparation.
        result = pipe.run(request.segment.text)
        return AudioFragment(
            segment_id=request.segment.id,
            audio=np.asarray(result.audio, dtype=np.float32),
            sample_rate=int(result.sample_rate),
            realized_prosody=frozenset(),
            metadata={
                "plugin": self.id,
                "voice": request.voice.id,
                "compatibility_bridge": True,
            },
        )

    def close(self) -> None:
        for pipeline in self._pipelines.values():
            close = getattr(pipeline, "close", None)
            if close is not None:
                close()
        self._pipelines.clear()
