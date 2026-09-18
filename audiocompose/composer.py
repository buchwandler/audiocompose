from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import ComposedMarker, ComposedSpan
from .errors import CompositionError
from .loudness import apply_complete_output_loudness
from .model import AudioJob, ComposedItem, CompositionResult, Silence
from .operations import Tempo, apply_operation
from .resampling import resample_audio
from .wav import write_wav


def _seconds_samples(seconds: float, rate: int) -> int:
    return round(seconds * rate)


@dataclass(frozen=True, slots=True)
class Composer:
    sample_rate: int | None = None

    def compose(self, job: AudioJob) -> CompositionResult:
        job.validate()
        rate = self.sample_rate or job.output.sample_rate
        if rate <= 0:
            raise CompositionError("sample_rate must be positive")
        parts: list[np.ndarray] = []
        composed: list[ComposedItem] = []
        markers: list[ComposedMarker] = []
        spans: list[ComposedSpan] = []
        cursor = 0
        for item in job.items:
            start = cursor
            if isinstance(item, Silence):
                audio = np.zeros(_seconds_samples(item.seconds, rate), dtype=np.float32)
                source_rate = None
            else:
                audio, source_rate = item.source.load()
                original_length = len(audio)
                for operation in item.operations:
                    audio = apply_operation(audio, source_rate, operation)
                for anchor in item.anchors:
                    offset = anchor.sample_offset
                    length = original_length
                    for operation in item.operations:
                        offset = operation.map_offset(offset, length)
                        if isinstance(operation, Tempo):
                            length = round(length / operation.factor)
                    offset = round(offset * rate / source_rate)
                    markers.append(ComposedMarker(anchor.id, cursor + offset, anchor.name, item.id))
                for span in item.spans:
                    span_start = span.sample_start
                    span_end = span.sample_end
                    length = original_length
                    for operation in item.operations:
                        span_start = operation.map_offset(span_start, length)
                        span_end = operation.map_offset(span_end, length)
                        if isinstance(operation, Tempo):
                            length = round(length / operation.factor)
                    spans.append(
                        ComposedSpan(
                            item.id,
                            span.source_start,
                            span.source_end,
                            cursor + round(span_start * rate / source_rate),
                            cursor + round(span_end * rate / source_rate),
                        )
                    )
                audio = resample_audio(audio, source_rate, rate)
            parts.append(np.asarray(audio, dtype=np.float32))
            cursor += len(audio)
            composed.append(
                ComposedItem(
                    item.id,
                    "silence" if isinstance(item, Silence) else "clip",
                    start,
                    cursor,
                    source_rate,
                )
            )
        waveform = (
            np.concatenate(parts).astype(np.float32, copy=False)
            if parts
            else np.zeros(0, dtype=np.float32)
        )
        loudness = apply_complete_output_loudness(waveform, rate, job.output.loudness)
        waveform = loudness.audio
        return CompositionResult(
            audio=waveform,
            sample_rate=rate,
            items=tuple(composed),
            markers=tuple(markers),
            spans=tuple(spans),
            diagnostics=(),
            provenance={
                "job_id": job.job_id,
                "producer": dict(job.producer),
                "applied_loudness_gain_db": loudness.applied_gain_db,
            },
        )

    def to_wav(self, job: AudioJob, path: str | Path) -> Path:
        result = self.compose(job)
        return write_wav(path, result.audio, result.sample_rate, clip_policy=job.output.clip_policy)

    def compose_to_wav(self, manifest: str | Path, path: str | Path) -> Path:
        return self.to_wav(AudioJob.load(str(manifest)), path)
