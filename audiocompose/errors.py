class AudioComposeError(Exception):
    """Base class for audiocompose errors."""


class AudioValidationError(AudioComposeError, ValueError):
    """An AudioJob or audio source is invalid."""


class CompositionError(AudioComposeError):
    """Audio composition failed."""


class CompositionDiagnosticError(CompositionError):
    """A composition diagnostic is fatal under the selected policy."""
