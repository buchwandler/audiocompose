from ._version import __version__
from .alignment import AudioAnchor, AudioSpan, ComposedMarker, ComposedSpan, Marker
from .analysis import (
    AcousticGap,
    ActivityConfig,
    ActivityRegion,
    ActivityReport,
    analyze_activity,
    measure_gap_near,
)
from .composer import Composer
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .errors import AudioComposeError, AudioValidationError, CompositionError
from .job import load_job, save_job, validate_job
from .loudness import LoudnessPolicy, LoudnessResult, apply_complete_output_loudness
from .model import AudioClip, AudioJob, ComposedItem, CompositionResult, OutputPolicy, Silence
from .operations import (
    AudioOperation,
    FadeIn,
    FadeOut,
    Gain,
    PitchShift,
    Tempo,
    apply_operation,
    operation_from_dict,
)
from .resampling import resample_audio
from .sources import AudioBufferSource, AudioFileSource, AudioSource
from .timeline import samples_for_duration, silence
from .wav import prepare_output, read_wav, sha256_file, wav_info, write_intermediate_wav, write_wav

__all__ = [
    "ActivityConfig",
    "ActivityRegion",
    "ActivityReport",
    "AcousticGap",
    "analyze_activity",
    "measure_gap_near",
    "__version__",
    "AudioAnchor",
    "AudioBufferSource",
    "AudioClip",
    "AudioComposeError",
    "AudioFileSource",
    "AudioJob",
    "AudioOperation",
    "AudioSource",
    "AudioSpan",
    "CompositionDiagnostic",
    "CompositionError",
    "CompositionResult",
    "ComposedItem",
    "ComposedMarker",
    "ComposedSpan",
    "Composer",
    "DiagnosticSeverity",
    "FadeIn",
    "FadeOut",
    "Gain",
    "LoudnessPolicy",
    "LoudnessResult",
    "Marker",
    "OutputPolicy",
    "PitchShift",
    "Silence",
    "Tempo",
    "apply_complete_output_loudness",
    "apply_operation",
    "load_job",
    "operation_from_dict",
    "prepare_output",
    "read_wav",
    "resample_audio",
    "samples_for_duration",
    "save_job",
    "sha256_file",
    "silence",
    "validate_job",
    "wav_info",
    "write_intermediate_wav",
    "write_wav",
    "AudioValidationError",
]
