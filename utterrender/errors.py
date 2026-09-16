class UtterRenderError(Exception):
    """Base class for utterrender errors."""


class FragmentValidationError(UtterRenderError, ValueError):
    pass


class AssemblyError(UtterRenderError):
    pass


class RenderPluginError(UtterRenderError):
    pass


class VoiceRoutingError(UtterRenderError):
    pass
