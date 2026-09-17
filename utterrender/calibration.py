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
    model_id: str | None = None
    source: str | None = None
    quality: str | None = None
    measured_lufs: float | None = None
    reference_lufs: float | None = None
    samples: int | None = None
    method: str | None = None
    corpus_version: str | None = None

    @property
    def key(self) -> tuple[str | None, str | None, str | None, str]:
        return self.source, self.model_id, self.quality, self.voice_id


class VoiceCalibrationRegistry:
    def __init__(self, values: Mapping[str, float | VoiceCalibration] | None = None) -> None:
        self._records: dict[str, VoiceCalibration] = {}
        for key, value in dict(values or {}).items():
            self._records[str(key)] = (
                value if isinstance(value, VoiceCalibration) else VoiceCalibration(str(key), float(value))
            )

    def set(self, voice_id: str, gain_db: float) -> None:
        self._records[voice_id] = VoiceCalibration(voice_id, float(gain_db))

    def set_record(self, calibration: VoiceCalibration) -> None:
        self._records[calibration.voice_id] = calibration

    def get(self, voice_id: str, *, model_id: str | None = None) -> VoiceCalibration | None:
        record = self._records.get(voice_id)
        if record is None:
            return None
        if model_id is not None and record.model_id not in {None, model_id}:
            return None
        return record

    def gain_db(self, voice_id: str, *, model_id: str | None = None) -> float:
        record = self.get(voice_id, model_id=model_id)
        return record.gain_db if record is not None else 0.0

    def apply(
        self,
        fragment: AudioFragment,
        voice_id: str,
        *,
        model_id: str | None = None,
    ) -> AudioFragment:
        record = self.get(voice_id, model_id=model_id)
        gain_db = record.gain_db if record is not None else 0.0
        if gain_db == 0.0:
            return fragment
        multiplier = 10.0 ** (gain_db / 20.0)
        audio = (fragment.audio.astype(np.float32, copy=False) * multiplier).astype(
            np.float32, copy=False
        )
        metadata = dict(fragment.metadata)
        metadata["utterrender.voice_calibration_db"] = gain_db
        if record is not None:
            metadata["utterrender.voice_calibration_key"] = record.key
        return AudioFragment(
            segment_id=fragment.segment_id,
            audio=audio,
            sample_rate=fragment.sample_rate,
            metadata=metadata,
            realized_prosody=fragment.realized_prosody,
            alignment=fragment.alignment,
            diagnostics=fragment.diagnostics,
        )
