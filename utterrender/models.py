from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Stable metadata and identity for one backend model variant."""

    id: str
    plugin: str
    model_id: str
    source: str | None = None
    quality: str | None = None
    languages: tuple[str, ...] = ()
    sample_rate: int | None = None
    installed: bool | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.plugin or not self.model_id:
            raise ValueError("model id, plugin, and model_id must be non-empty")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
