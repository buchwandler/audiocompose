from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .model import AudioFragment


@dataclass(frozen=True, slots=True)
class VoiceCalibration:
    """Static reviewed correction for one concrete runtime voice."""

    voice_id: str
    gain_db: float = 0.0


class VoiceCalibrationRegistry:
    def __init__(self, values: Mapping[str, float] | None = None) -> None:
        self._gain_db = {str(key): float(value) for key, value in dict(values or {}).items()}

    def set(self, voice_id: str, gain_db: float) -> None:
        self._gain_db[voice_id] = float(gain_db)

    def gain_db(self, voice_id: str) -> float:
        return self._gain_db.get(voice_id, 0.0)

    def apply(self, fragment: AudioFragment, voice_id: str) -> AudioFragment:
        gain_db = self.gain_db(voice_id)
        if gain_db == 0.0:
            return fragment
        multiplier = 10.0 ** (gain_db / 20.0)
        audio = (fragment.audio.astype(np.float32, copy=False) * multiplier).astype(
            np.float32, copy=False
        )
        metadata = dict(fragment.metadata)
        metadata["utterrender.voice_calibration_db"] = gain_db
        return AudioFragment(
            segment_id=fragment.segment_id,
            audio=audio,
            sample_rate=fragment.sample_rate,
            metadata=metadata,
            realized_prosody=fragment.realized_prosody,
        )
