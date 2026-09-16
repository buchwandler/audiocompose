from .assembler import assemble, samples_for_duration, silence
from .calibration import VoiceCalibration, VoiceCalibrationRegistry
from .capabilities import RenderCapabilities
from .effects import AudioSigProsodyProcessor
from .errors import (
    AssemblyError,
    FragmentValidationError,
    RenderPluginError,
    UtterRenderError,
    VoiceRoutingError,
)
from .model import AudioFragment, RenderedMarker, RenderedSegment, RenderedUnit, RenderResult
from .plugins import KokoroPlugin, PiperPlugin, PluginRegistry, RenderPlugin, RenderRequest
from .prosody import ResolvedProsody, parse_pitch, parse_rate, parse_volume, resolve_prosody
from .runtime import Renderer, TTS
from .voices import VoiceBindings, VoiceInfo
from .wav import write_wav

__version__ = "0.1.0"

__all__ = [
    "AssemblyError",
    "AudioFragment",
    "AudioSigProsodyProcessor",
    "FragmentValidationError",
    "KokoroPlugin",
    "PiperPlugin",
    "PluginRegistry",
    "RenderCapabilities",
    "RenderedMarker",
    "RenderedSegment",
    "RenderedUnit",
    "RenderPlugin",
    "RenderPluginError",
    "RenderRequest",
    "RenderResult",
    "Renderer",
    "ResolvedProsody",
    "TTS",
    "UtterRenderError",
    "VoiceBindings",
    "VoiceCalibration",
    "VoiceCalibrationRegistry",
    "VoiceInfo",
    "VoiceRoutingError",
    "assemble",
    "parse_pitch",
    "parse_rate",
    "parse_volume",
    "resolve_prosody",
    "samples_for_duration",
    "silence",
    "write_wav",
]
