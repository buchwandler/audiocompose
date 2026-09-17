from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .._plan import Plan, PlanSegment
from ..assets import AssetProgressCallback
from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..models import ModelInfo
from ..prosody import ResolvedProsody
from ..voices import VoiceInfo


@dataclass(frozen=True, slots=True)
class SegmentRenderContext:
    """Immutable plan context passed to a backend for one segment."""

    plan: Plan
    segment: PlanSegment
    tokens: tuple[Any, ...] = ()
    annotations: tuple[Any, ...] = ()
    unit_id: str | None = None

    @classmethod
    def from_plan(cls, plan: Plan, segment: PlanSegment) -> SegmentRenderContext:
        token_indices = tuple(getattr(segment, "token_indices", ()))
        tokens = tuple(plan.tokens[index] for index in token_indices)
        annotation_ids = set(getattr(segment, "annotation_ids", ()))
        annotations = tuple(
            annotation
            for annotation in plan.annotations
            if getattr(annotation, "id", None) in annotation_ids
        )
        unit_id = next(
            (unit.id for unit in plan.units if segment.id in unit.segment_ids),
            None,
        )
        return cls(plan, segment, tokens, annotations, unit_id)


@dataclass(frozen=True, slots=True)
class RenderRequest:
    """Backend request containing prepared text and renderer-relevant context."""

    segment: PlanSegment
    voice: VoiceInfo
    prosody: ResolvedProsody
    options: Mapping[str, Any] = field(default_factory=dict)
    context: SegmentRenderContext | None = None
    speaker: str | int | None = None
    pronunciation: Any | None = None
    emphasis: Any | None = None
    audio_directive: Any | None = None
    streaming: bool = False

    @property
    def plan_context(self) -> SegmentRenderContext:
        if self.context is None:
            raise ValueError("RenderRequest requires SegmentRenderContext")
        return self.context


@runtime_checkable
class RenderPlugin(Protocol):
    @property
    def id(self) -> str: ...

    def models(self, *, language: str | None = None) -> Iterable[ModelInfo]: ...

    def voices(
        self,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> Iterable[VoiceInfo]: ...

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities: ...

    def ensure_voice(
        self,
        voice: VoiceInfo,
        *,
        progress: AssetProgressCallback | None = None,
    ) -> None: ...

    def render(self, request: RenderRequest) -> AudioFragment: ...

    def close(self) -> None: ...
