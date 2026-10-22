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
from .bundle import BundleComposeOptions, BundleComposeResult, compose_bundle
from .bundle_batch import (
    BatchAttemptStatus,
    BatchVariantStatus,
    BundleBatchComposeResult,
    BundleBatchComposeSummary,
    BundleVariantComposeOutcome,
    compose_bundle_batch,
)
from .bundle_contract import BundleIdentity, load_bundle_identity
from .composer import Composer, estimate_composition
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .document import (
    DocumentComposeOptions,
    DocumentComposeRequest,
    DocumentComposeResult,
    compose_document,
    default_document_output_policy,
    resolve_document,
)
from .errors import (
    AudioComposeError,
    AudioValidationError,
    CompositionError,
    DocumentComposeError,
)
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
from .progress import (
    CompositionEstimate,
    CompositionProgress,
    CompositionProgressCallback,
    ProgressEstimator,
    ProgressTiming,
)
from .renderplan import (
    RenderPlanComposeRequest,
    RenderPlanComposeResult,
    compose_renderplan,
)
from .schema import audiojob_schema
from .sources import AudioBufferSource, AudioFileSource
from .variants import BundleVariant, BundleVariantsResult, list_bundle_variants

__all__ = [
    "__version__",
    "BundleComposeOptions",
    "BundleComposeResult",
    "BatchAttemptStatus",
    "BatchVariantStatus",
    "BundleBatchComposeResult",
    "BundleBatchComposeSummary",
    "BundleVariantComposeOutcome",
    "compose_bundle_batch",
    "BundleIdentity",
    "compose_bundle",
    "load_bundle_identity",
    "BundleVariant",
    "BundleVariantsResult",
    "list_bundle_variants",
    "AudioAnchor",
    "AudioBufferSource",
    "AudioClip",
    "AudioComposeError",
    "BookComposeError",
    "DocumentComposeError",
    "DocumentComposeOptions",
    "DocumentComposeRequest",
    "DocumentComposeResult",
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
    "ProgressEstimator",
    "ProgressTiming",
    "CompositionResult",
    "ComposedItem",
    "ComposedMarker",
    "ComposedSpan",
    "Composer",
    "CompositionEstimate",
    "compose_book",
    "compose_document",
    "compose_renderplan",
    "estimate_composition",
    "default_book_output_policy",
    "default_document_output_policy",
    "resolve_book",
    "resolve_document",
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
