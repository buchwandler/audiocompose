from __future__ import annotations

import numpy as np

from .model import AudioFragment


def resample_fragment(fragment: AudioFragment, sample_rate: int) -> AudioFragment:
    if fragment.sample_rate == sample_rate:
        return fragment
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    try:
        from audiosig import resample

        audio = resample(fragment.audio, fragment.sample_rate, sample_rate)
    except (ImportError, AttributeError, TypeError):
        # Lightweight deterministic MVP fallback. Production builds should prefer
        # audiosig's band-limited resampler.
        if fragment.audio.size == 0:
            audio = np.zeros(0, dtype=np.float32)
        else:
            output_size = round(fragment.audio.size * sample_rate / fragment.sample_rate)
            if output_size <= 1:
                audio = np.asarray(fragment.audio[:1], dtype=np.float32)
            else:
                old = np.linspace(0.0, 1.0, fragment.audio.size, endpoint=True)
                new = np.linspace(0.0, 1.0, output_size, endpoint=True)
                audio = np.interp(new, old, fragment.audio).astype(np.float32)
    metadata = dict(fragment.metadata)
    metadata["utterrender.resampled_from"] = fragment.sample_rate
    return AudioFragment(
        segment_id=fragment.segment_id,
        audio=np.asarray(audio, dtype=np.float32),
        sample_rate=sample_rate,
        metadata=metadata,
        realized_prosody=fragment.realized_prosody,
    )
