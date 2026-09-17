from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._plan import PlanSegment
from .errors import VoiceRoutingError
from .plugins.registry import PluginRegistry
from .voices import VoiceBindings, VoiceInfo

UnboundVoicePolicy = Literal["error", "default"]


@dataclass(slots=True)
class VoiceRouter:
    registry: PluginRegistry
    bindings: VoiceBindings
    default_voice: str | None = None
    auto_language: bool = True
    unbound_voice_policy: UnboundVoicePolicy = "error"
    preferred_plugins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.unbound_voice_policy not in {"error", "default"}:
            raise ValueError(f"unknown unbound voice policy: {self.unbound_voice_policy!r}")

    def _select_auto(self, language: str | None) -> VoiceInfo | None:
        voices = self.registry.voices(language=language)
        if not voices:
            return None
        priority = {plugin: index for index, plugin in enumerate(self.preferred_plugins)}
        return min(
            voices,
            key=lambda voice: (
                priority.get(voice.plugin, len(priority)),
                voice.model_identity or "",
                voice.id,
            ),
        )

    def resolve(self, segment: PlanSegment) -> VoiceInfo:
        logical = segment.directives.voice.reference if segment.directives.voice else None
        bound = self.bindings.resolve(logical)
        candidate = bound or (logical if logical and ":" in logical else None)
        if candidate is None:
            if logical is not None and self.unbound_voice_policy == "error":
                raise VoiceRoutingError(
                    f"logical voice {logical!r} is not bound to a concrete runtime voice"
                )
            candidate = self.default_voice

        if candidate is not None:
            voice = self.registry.voice(candidate)
            if voice is None:
                raise VoiceRoutingError(f"unknown concrete runtime voice {candidate!r}")
            if segment.language and not voice.supports_language(segment.language):
                raise VoiceRoutingError(
                    f"voice {voice.id!r} does not advertise language {segment.language!r}"
                )
            return voice
        if logical is None and self.auto_language:
            selected = self._select_auto(segment.language)
            if selected is not None:
                return selected
        label = f"logical voice {logical!r}" if logical else "segment"
        raise VoiceRoutingError(
            f"no concrete runtime voice is bound for {label} ({segment.language})"
        )
