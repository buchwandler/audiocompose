from __future__ import annotations

from dataclasses import dataclass

from ._plan import PlanSegment

from .errors import VoiceRoutingError
from .plugins.registry import PluginRegistry
from .voices import VoiceBindings, VoiceInfo


@dataclass(slots=True)
class VoiceRouter:
    registry: PluginRegistry
    bindings: VoiceBindings
    default_voice: str | None = None
    auto_language: bool = True

    def resolve(self, segment: PlanSegment) -> VoiceInfo:
        logical = segment.directives.voice.reference if segment.directives.voice else None
        bound = self.bindings.resolve(logical)
        candidate = bound or (logical if logical and ":" in logical else None) or self.default_voice
        if candidate is not None:
            voice = self.registry.voice(candidate)
            if segment.language and not voice.supports_language(segment.language):
                raise VoiceRoutingError(
                    f"voice {voice.id!r} does not advertise language {segment.language!r}"
                )
            return voice
        if self.auto_language:
            voices = self.registry.voices(language=segment.language)
            if voices:
                return voices[0]
        label = f"logical voice {logical!r}" if logical else "segment"
        raise VoiceRoutingError(
            f"no concrete runtime voice is bound for {label} ({segment.language})"
        )
