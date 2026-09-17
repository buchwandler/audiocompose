from __future__ import annotations

from typing import Protocol

from ._plan import PlanSegment
from .model import AudioFragment


class FragmentProcessor(Protocol):
    """Optional post-model transformation applied before plan assembly."""

    def __call__(
        self,
        fragment: AudioFragment,
        *,
        segment: PlanSegment,
    ) -> AudioFragment: ...
