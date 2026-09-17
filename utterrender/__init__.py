from .alignment import AlignmentKind, AudioTextSpan
from .assembler import assemble, samples_for_duration, silence
from .assets import AssetProgressCallback, AssetProgressEvent
from .calibration import VoiceCalibration, VoiceCalibrationRegistry
from .capabilities import RenderCapabilities
from .diagnostics import DiagnosticSeverity, RenderDiagnostic
from .effects import AudioSigProsodyProcessor, apply_emphasis
from .errors import (
    AssemblyError,
    FragmentValidationError,
    RenderPluginError,
    UtterRenderError,
    VoiceRoutingError,
)
from .loudness import LoudnessPolicy, LoudnessResult, apply_complete_output_loudness
from .model import AudioFragment, RenderedMarker, RenderedSegment, RenderedUnit, RenderResult
from .models import ModelInfo
from .plugins import (
    KokoroPlugin,
    PiperPlugin,
    PluginAvailability,
    PluginRegistry,
    RenderPlugin,
    RenderRequest,
    SegmentRenderContext,
)
from .prosody import ResolvedProsody, parse_pitch, parse_rate, parse_volume, resolve_prosody
from .runtime import TTS, Renderer
from .voices import VoiceBindings, VoiceInfo
from .wav import prepare_output, write_wav

__version__ = "0.1.0"

__all__ = [
    "AlignmentKind",
    "AssemblyError",
    "AssetProgressCallback",
    "AssetProgressEvent",
    "AudioFragment",
    "AudioSigProsodyProcessor",
    "apply_emphasis",
    "AudioTextSpan",
    "DiagnosticSeverity",
    "LoudnessPolicy",
    "LoudnessResult",
    "FragmentValidationError",
    "PluginAvailability",
    "KokoroPlugin",
    "ModelInfo",
    "PiperPlugin",
    "PluginRegistry",
    "PluginStatus",
    "RenderCapabilities",
    "RenderDiagnostic",
    "RenderedMarker",
    "RenderedSegment",
    "RenderedUnit",
    "RenderPlugin",
    "RenderPluginError",
    "RenderRequest",
    "RenderResult",
    "Renderer",
    "ResolvedProsody",
    "SegmentRenderContext",
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
    "apply_complete_output_loudness",
    "samples_for_duration",
    "silence",
    "prepare_output",
    "write_wav",
]
