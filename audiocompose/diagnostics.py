from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CompositionDiagnostic:
    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    item_id: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.item_id}] " if self.item_id else ""
        return f"{self.severity.value}: {prefix}{self.message}"
