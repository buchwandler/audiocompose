from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """Engine-independent description of one selectable runtime voice."""

    id: str
    plugin: str
    voice_id: str
    languages: tuple[str, ...]
    model_id: str | None = None
    sample_rate: int | None = None
    installed: bool | None = None
    downloadable: bool = True
    quality: str | None = None
    speakers: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def supports_language(self, language: str) -> bool:
        requested = language.lower().replace("_", "-")
        for candidate in self.languages:
            value = candidate.lower().replace("_", "-")
            if requested == value or requested.startswith(value + "-") or value.startswith(requested + "-"):
                return True
        return False


class VoiceBindings:
    """Map logical utterplan voice references to concrete runtime voices."""

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self._values = dict(values or {})

    def bind(self, logical: str, voice: str) -> None:
        if not logical or not voice:
            raise ValueError("logical and voice must be non-empty strings")
        self._values[logical] = voice

    def resolve(self, logical: str | None) -> str | None:
        return self._values.get(logical) if logical is not None else None

    def to_dict(self) -> dict[str, str]:
        return dict(self._values)
