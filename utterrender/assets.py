from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssetProgressEvent:
    """Progress notification emitted while a plugin provisions an asset."""

    plugin: str
    operation: str
    asset_id: str
    completed: int | None = None
    total: int | None = None
    message: str | None = None


AssetProgressCallback = Callable[[AssetProgressEvent], None]
