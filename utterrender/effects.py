from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._plan import PlanSegment

from .errors import AssemblyError
from .model import AudioFragment
from .prosody import ProsodyAxis, resolve_prosody

ProsodyMethod = Literal["phase_vocoder", "wsola", "esola", "td_psola"]


@dataclass(frozen=True, slots=True)
class AudioSigProsodyProcessor:
    """Apply unresolved plan prosody through ``audiosig`` after inference.

    Backends advertise natively realized axes on ``AudioFragment.realized_prosody``.
    Only the remaining requested axes are transformed, preventing e.g. Piper's
    native rate control from being applied a second time after inference.
    """

    method: ProsodyMethod = "wsola"
    clip: bool = False
    n_fft: int = 2048
    hop_length: int | None = None
    filter_width: int = 32
    rolloff: float = 0.945

    def __call__(self, fragment: AudioFragment, *, segment: PlanSegment) -> AudioFragment:
        prosody = resolve_prosody(segment.directives.prosody)
        pending = prosody.requested - fragment.realized_prosody
        if not pending:
            return fragment

        rate = prosody.rate if "rate" in pending else 1.0
        semitones = prosody.semitones if "pitch" in pending else 0.0
        gain_db = prosody.gain_db if "volume" in pending else 0.0

        try:
            from audiosig import apply_speech_effects
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AssemblyError(
                "AudioSigProsodyProcessor requires the optional 'audiosig' dependency; "
                "install utterrender[prosody]"
            ) from exc

        audio = apply_speech_effects(
            fragment.audio,
            sample_rate=fragment.sample_rate,
            rate=rate,
            semitones=semitones,
            gain_db=gain_db,
            clip=self.clip,
            method=self.method,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            filter_width=self.filter_width,
            rolloff=self.rolloff,
        )
        realized: frozenset[ProsodyAxis] = fragment.realized_prosody | pending
        metadata = dict(fragment.metadata)
        metadata["utterrender.prosody"] = {
            "rate": rate,
            "semitones": semitones,
            "gain_db": gain_db,
            "method": self.method,
            "realized_axes": sorted(pending),
        }
        return AudioFragment(
            segment_id=fragment.segment_id,
            audio=audio,
            sample_rate=fragment.sample_rate,
            realized_prosody=realized,
            metadata=metadata,
        )
