from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

DiagnosticSeverity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class RenderDiagnostic:
    """Structured runtime diagnostic kept separate from plan diagnostics."""

    code: str
    severity: DiagnosticSeverity
    message: str
    plugin: str | None = None
    segment_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
