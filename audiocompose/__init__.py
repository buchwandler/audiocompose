from ._version import __version__
from .alignment import AudioAnchor, AudioSpan, ComposedMarker, ComposedSpan
from .book import (
    BookComposeError,
    BookComposeOptions,
    BookComposeProgress,
    BookComposeProgressCallback,
    BookComposeRequest,
    BookComposeResult,
    ChapterComposeOutcome,
    ChapterComposeStatus,
    compose_book,
    default_book_output_policy,
    resolve_book,
)
from .composer import Composer
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .errors import AudioComposeError, AudioValidationError, CompositionError
from .job import load_job, save_job
from .loudness import LoudnessPolicy, LoudnessResult
from .model import AudioClip, AudioJob, ComposedItem, CompositionResult, OutputPolicy, Silence
from .operations import (
    AutomationPoint,
    FadeIn,
    FadeOut,
    Gain,
    Operation,
    PitchShift,
    RatePitchEnvelope,
    Tempo,
)
from .progress import CompositionProgress, CompositionProgressCallback
from .renderplan import (
    RenderPlanComposeRequest,
    RenderPlanComposeResult,
    compose_renderplan,
)
from .schema import audiojob_schema
from .sources import AudioBufferSource, AudioFileSource

__all__ = [
    "__version__",
    "AudioAnchor",
    "AudioBufferSource",
    "AudioClip",
    "AudioComposeError",
    "BookComposeError",
    "BookComposeRequest",
    "BookComposeOptions",
    "BookComposeProgress",
    "BookComposeProgressCallback",
    "BookComposeResult",
    "ChapterComposeOutcome",
    "ChapterComposeStatus",
    "AudioFileSource",
    "AudioJob",
    "AudioSpan",
    "AudioValidationError",
    "AutomationPoint",
    "CompositionDiagnostic",
    "CompositionError",
    "CompositionProgress",
    "CompositionProgressCallback",
    "CompositionResult",
    "ComposedItem",
    "ComposedMarker",
    "ComposedSpan",
    "Composer",
    "compose_book",
    "compose_renderplan",
    "default_book_output_policy",
    "resolve_book",
    "DiagnosticSeverity",
    "FadeIn",
    "FadeOut",
    "Gain",
    "LoudnessPolicy",
    "LoudnessResult",
    "Operation",
    "OutputPolicy",
    "PitchShift",
    "RatePitchEnvelope",
    "RenderPlanComposeRequest",
    "RenderPlanComposeResult",
    "Silence",
    "Tempo",
    "audiojob_schema",
    "load_job",
    "save_job",
]
