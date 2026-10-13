class AudioComposeError(Exception):
    """Base class for audiocompose errors."""


class AudioValidationError(AudioComposeError, ValueError):
    """An AudioJob or audio source is invalid."""


class CompositionError(AudioComposeError):
    """Audio composition failed."""


class IntegrationError(AudioComposeError):
    """An optional producer integration failed."""


class HandoffValidationError(IntegrationError):
    """An upstream handoff or its source artifacts are invalid."""


class MediaResolutionError(IntegrationError):
    """An external media source cannot be resolved safely."""


class CompositionDiagnosticError(CompositionError):
    """A composition diagnostic is fatal under the selected policy."""
