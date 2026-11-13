from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AudioComposeError(Exception):
    """Base class for audiocompose errors."""


class BookComposeError(AudioComposeError):
    """Book orchestration error with stable downstream-facing attributes."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        chapter_id: str | None = None,
        source_number: int | None = None,
        variant: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.chapter_id = chapter_id
        self.source_number = source_number
        self.variant = variant
        self.details = dict(details or {})


class DocumentComposeError(AudioComposeError):
    """Document orchestration error with stable downstream-facing attributes."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        document_id: str | None = None,
        variant: str | None = None,
        details: Mapping[str, Any] | None = None,
        recovery_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.document_id = document_id
        self.variant = variant
        self.details = dict(details or {})
        self.recovery_action = recovery_action


class AudioValidationError(AudioComposeError, ValueError):
    """An AudioJob or audio source is invalid."""


class CompositionError(AudioComposeError):
    """Audio composition failed."""


class IntegrationError(AudioComposeError):
    """An optional producer integration failed."""


class HandoffValidationError(IntegrationError):
    """An upstream handoff or its source artifacts are invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "handoff.invalid",
        details: Mapping[str, Any] | None = None,
        recovery_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})
        self.recovery_action = recovery_action


class PausePolicyError(IntegrationError, ValueError):
    """A pause policy or symbolic pause intent is invalid or unresolved."""


class MediaResolutionError(IntegrationError):
    """An external audio source cannot be resolved safely."""


class CompositionDiagnosticError(CompositionError):
    """A composition diagnostic is fatal under the selected policy."""
