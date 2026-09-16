from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .._plan import PlanSegment

from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..prosody import ResolvedProsody
from ..voices import VoiceInfo


@dataclass(frozen=True, slots=True)
class RenderRequest:
    segment: PlanSegment
    voice: VoiceInfo
    prosody: ResolvedProsody
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RenderPlugin(Protocol):
    @property
    def id(self) -> str: ...

    def voices(self, *, language: str | None = None) -> Iterable[VoiceInfo]: ...

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities: ...

    def ensure_voice(self, voice: VoiceInfo) -> None: ...

    def render(self, request: RenderRequest) -> AudioFragment: ...

    def close(self) -> None: ...
