from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import ComposedMarker, ComposedSpan
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .errors import AudioValidationError, CompositionError
from .job import validate_clip_geometry
from .loudness import LoudnessResult, apply_complete_output_loudness
from .model import AudioClip, AudioJob, ComposedItem, CompositionResult, Silence
from .operations import PitchShift, Tempo, apply_operation, apply_temporal_group
from .progress import CompositionProgress, CompositionProgressCallback, ProgressItemKind
from .resampling import resample_audio
from .sources import AudioBufferSource
from .wav import prepare_output, write_wav


def _seconds_samples(seconds: float, rate: int) -> int:
    return round(seconds * rate)


def _predicted_output_frames(item: AudioClip | Silence, output_rate: int) -> int | None:
    if isinstance(item, Silence):
        return _seconds_samples(item.seconds, output_rate)
    source = item.source
    frames: int | None
    source_rate: int | None
    if isinstance(source, AudioBufferSource):
        frames = len(source.audio)
        source_rate = source.sample_rate
    else:
        frames = source.frames
        source_rate = source.sample_rate
    if frames is None or source_rate is None:
        return None
    operation_index = 0
    while operation_index < len(item.operations):
        group_start = operation_index
        group_end = group_start + 1
        operation = item.operations[group_start]
        if isinstance(operation, (Tempo, PitchShift)):
            while group_end < len(item.operations) and isinstance(
                item.operations[group_end], (Tempo, PitchShift)
            ):
                group_end += 1
        group = item.operations[group_start:group_end]
        if len(group) > 1:
            rate = math.prod(
                group_operation.factor
                for group_operation in group
                if isinstance(group_operation, Tempo)
            )
            frames = round(frames / rate)
        else:
            frames = operation.output_length(frames, source_rate)
        operation_index = group_end
    if source_rate != output_rate:
        frames = round(frames * output_rate / source_rate)
    return frames


def _emit(callback: CompositionProgressCallback | None, event: CompositionProgress) -> None:
    if callback is not None:
        callback(event)


def _loudness_summary(result: LoudnessResult) -> dict[str, object]:
    return {
        "before": {
            "integrated_lufs": result.before.integrated_lufs,
            "true_peak_dbtp": result.before.true_peak_dbtp,
            "sample_peak_dbfs": result.before.sample_peak_dbfs,
        },
        "after": {
            "integrated_lufs": result.after.integrated_lufs,
            "true_peak_dbtp": result.after.true_peak_dbtp,
            "sample_peak_dbfs": result.after.sample_peak_dbfs,
        },
        "target_lufs": result.target_lufs,
        "true_peak_ceiling_dbtp": result.true_peak_ceiling_dbtp,
        "requested_gain_db": result.requested_gain_db,
        "applied_gain_db": result.applied_gain_db,
        "target_reached": result.target_reached,
        "peak_policy": result.peak_policy,
        "warning": result.warning,
    }


@dataclass(frozen=True, slots=True)
class Composer:
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise AudioValidationError("sample_rate must be a positive integer or None")

    def compose(
        self,
        job: AudioJob,
        *,
        on_progress: CompositionProgressCallback | None = None,
    ) -> CompositionResult:
        job.validate(verify_sources=False)
        rate = self.sample_rate if self.sample_rate is not None else job.output.sample_rate
        total_items = len(job.items)
        item_output_frames = [_predicted_output_frames(item, rate) for item in job.items]
        total_output_frames = (
            sum(frames for frames in item_output_frames if frames is not None)
            if all(frames is not None for frames in item_output_frames)
            else None
        )
        total_audio_seconds = (
            total_output_frames / rate if total_output_frames is not None else None
        )
        completed_audio_seconds = 0.0
        metadata_kind_counts: dict[str, int] = {}
        for item in job.items:
            metadata_kind = item.metadata.get("kind")
            if isinstance(metadata_kind, str):
                metadata_kind_counts[metadata_kind] = metadata_kind_counts.get(metadata_kind, 0) + 1
        _emit(
            on_progress,
            CompositionProgress(
                kind="compose_started",
                completed_items=0,
                total_items=total_items,
                target_sample_rate=rate,
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
                details={
                    "clip_items": sum(isinstance(item, AudioClip) for item in job.items),
                    "silence_items": sum(isinstance(item, Silence) for item in job.items),
                    "metadata_kinds": metadata_kind_counts,
                },
            ),
        )
        parts: list[np.ndarray] = []
        composed: list[ComposedItem] = []
        markers: list[ComposedMarker] = []
        spans: list[ComposedSpan] = []
        cursor = 0
        for index, item in enumerate(job.items):
            item_kind: ProgressItemKind = "silence" if isinstance(item, Silence) else "clip"
            _emit(
                on_progress,
                CompositionProgress(
                    kind="item_started",
                    completed_items=index,
                    total_items=total_items,
                    item_index=index,
                    item_id=item.id,
                    item_kind=item_kind,
                    item_metadata=item.metadata,
                    target_sample_rate=rate,
                    completed_audio_seconds=completed_audio_seconds,
                    total_audio_seconds=total_audio_seconds,
                    details={"seconds": item.seconds} if isinstance(item, Silence) else {},
                ),
            )
            start = cursor
            if isinstance(item, Silence):
                audio = np.zeros(_seconds_samples(item.seconds, rate), dtype=np.float32)
                source_rate = None
            else:
                _emit(
                    on_progress,
                    CompositionProgress(
                        kind="source_load_started",
                        completed_items=index,
                        total_items=total_items,
                        item_index=index,
                        item_id=item.id,
                        item_kind="clip",
                        item_metadata=item.metadata,
                        target_sample_rate=rate,
                    ),
                )
                audio, source_rate = item.source.load()
                validate_clip_geometry(item, audio)
                _emit(
                    on_progress,
                    CompositionProgress(
                        kind="source_load_completed",
                        completed_items=index,
                        total_items=total_items,
                        item_index=index,
                        item_id=item.id,
                        item_kind="clip",
                        item_metadata=item.metadata,
                        source_sample_rate=source_rate,
                        target_sample_rate=rate,
                        input_frames=len(audio),
                        output_frames=len(audio),
                    ),
                )
                original_length = len(audio)
                operation_count = len(item.operations)
                operation_index = 0
                while operation_index < operation_count:
                    group_start = operation_index
                    group_end = group_start + 1
                    operation = item.operations[group_start]
                    if isinstance(operation, (Tempo, PitchShift)):
                        while group_end < operation_count and isinstance(
                            item.operations[group_end], (Tempo, PitchShift)
                        ):
                            group_end += 1
                    group = item.operations[group_start:group_end]
                    fused = len(group) > 1
                    group_input_frames = len(audio)
                    group_details = (
                        {
                            "processing_group_start": group_start,
                            "processing_group_size": len(group),
                        }
                        if fused
                        else {}
                    )
                    for group_offset, group_operation in enumerate(group):
                        _emit(
                            on_progress,
                            CompositionProgress(
                                kind="operation_started",
                                completed_items=index,
                                total_items=total_items,
                                item_index=index,
                                item_id=item.id,
                                item_kind="clip",
                                item_metadata=item.metadata,
                                operation_index=group_start + group_offset,
                                operation_count=operation_count,
                                operation=dict(group_operation.to_dict()),
                                source_sample_rate=source_rate,
                                target_sample_rate=rate,
                                input_frames=(group_input_frames if fused else len(audio)),
                                details=group_details,
                            ),
                        )
                    if fused:
                        temporal_group = tuple(
                            group_operation
                            for group_operation in group
                            if isinstance(group_operation, (Tempo, PitchShift))
                        )
                        audio = apply_temporal_group(audio, source_rate, temporal_group)
                    else:
                        audio = apply_operation(audio, source_rate, operation)
                    for group_offset, group_operation in enumerate(group):
                        _emit(
                            on_progress,
                            CompositionProgress(
                                kind="operation_completed",
                                completed_items=index,
                                total_items=total_items,
                                item_index=index,
                                item_id=item.id,
                                item_kind="clip",
                                item_metadata=item.metadata,
                                operation_index=group_start + group_offset,
                                operation_count=operation_count,
                                operation=dict(group_operation.to_dict()),
                                source_sample_rate=source_rate,
                                target_sample_rate=rate,
                                input_frames=group_input_frames,
                                output_frames=len(audio),
                                details=group_details,
                            ),
                        )
                    operation_index = group_end
                for anchor in item.anchors:
                    offset = anchor.sample_offset
                    length = original_length
                    for operation in item.operations:
                        offset = operation.map_offset_at_rate(offset, length, source_rate)
                        length = operation.output_length(length, source_rate)
                    offset = round(offset * rate / source_rate)
                    markers.append(ComposedMarker(anchor.id, cursor + offset, anchor.name, item.id))
                for span in item.spans:
                    span_start = span.sample_start
                    span_end = span.sample_end
                    length = original_length
                    for operation in item.operations:
                        span_start = operation.map_offset_at_rate(span_start, length, source_rate)
                        span_end = operation.map_offset_at_rate(span_end, length, source_rate)
                        length = operation.output_length(length, source_rate)
                    spans.append(
                        ComposedSpan(
                            item.id,
                            span.source_start,
                            span.source_end,
                            cursor + round(span_start * rate / source_rate),
                            cursor + round(span_end * rate / source_rate),
                            id=span.id,
                            metadata=span.metadata,
                        )
                    )
                if source_rate != rate:
                    resample_input_frames = len(audio)
                    _emit(
                        on_progress,
                        CompositionProgress(
                            kind="resample_started",
                            completed_items=index,
                            total_items=total_items,
                            item_index=index,
                            item_id=item.id,
                            item_kind="clip",
                            item_metadata=item.metadata,
                            source_sample_rate=source_rate,
                            target_sample_rate=rate,
                            input_frames=resample_input_frames,
                        ),
                    )
                    audio = resample_audio(audio, source_rate, rate)
                    _emit(
                        on_progress,
                        CompositionProgress(
                            kind="resample_completed",
                            completed_items=index,
                            total_items=total_items,
                            item_index=index,
                            item_id=item.id,
                            item_kind="clip",
                            item_metadata=item.metadata,
                            source_sample_rate=source_rate,
                            target_sample_rate=rate,
                            input_frames=resample_input_frames,
                            output_frames=len(audio),
                        ),
                    )
            parts.append(np.asarray(audio, dtype=np.float32))
            cursor += len(audio)
            completed_audio_seconds = cursor / rate
            _emit(
                on_progress,
                CompositionProgress(
                    kind="item_completed",
                    completed_items=index + 1,
                    total_items=total_items,
                    item_index=index,
                    item_id=item.id,
                    item_kind=item_kind,
                    item_metadata=item.metadata,
                    source_sample_rate=source_rate,
                    target_sample_rate=rate,
                    output_frames=len(audio),
                    completed_audio_seconds=completed_audio_seconds,
                    total_audio_seconds=total_audio_seconds,
                    details={"seconds": item.seconds} if isinstance(item, Silence) else {},
                ),
            )
            composed.append(
                ComposedItem(
                    item.id,
                    item_kind,
                    start,
                    cursor,
                    source_rate,
                )
            )
        _emit(
            on_progress,
            CompositionProgress(
                kind="assembly_started",
                completed_items=total_items,
                total_items=total_items,
                target_sample_rate=rate,
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
            ),
        )
        waveform = (
            np.concatenate(parts).astype(np.float32, copy=False)
            if parts
            else np.zeros(0, dtype=np.float32)
        )
        _emit(
            on_progress,
            CompositionProgress(
                kind="assembly_completed",
                completed_items=total_items,
                total_items=total_items,
                target_sample_rate=rate,
                output_frames=len(waveform),
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
            ),
        )
        loudness_policy = job.output.loudness
        _emit(
            on_progress,
            CompositionProgress(
                kind="loudness_started",
                completed_items=total_items,
                total_items=total_items,
                target_sample_rate=rate,
                input_frames=len(waveform),
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
                details={
                    "target_lufs": loudness_policy.target_lufs,
                    "true_peak_ceiling_dbtp": loudness_policy.true_peak_ceiling_dbtp,
                    "peak_policy": loudness_policy.peak_policy,
                    "frames": len(waveform),
                    "sample_rate": rate,
                },
            ),
        )
        loudness = apply_complete_output_loudness(waveform, rate, loudness_policy)
        waveform = loudness.audio
        _emit(
            on_progress,
            CompositionProgress(
                kind="loudness_completed",
                completed_items=total_items,
                total_items=total_items,
                target_sample_rate=rate,
                input_frames=len(loudness.audio),
                output_frames=len(waveform),
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
                details={
                    "measured_lufs_before": loudness.before.integrated_lufs,
                    "measured_lufs_after": loudness.after.integrated_lufs,
                    "true_peak_dbtp_after": loudness.after.true_peak_dbtp,
                    "applied_gain_db": loudness.applied_gain_db,
                },
            ),
        )
        try:
            waveform = prepare_output(waveform, clip_policy=job.output.clip_policy)
        except AudioValidationError as exc:
            if job.output.clip_policy != "error":
                raise
            raise CompositionError(str(exc)) from exc
        diagnostics: list[CompositionDiagnostic] = []
        if loudness.warning:
            if loudness.before.integrated_lufs is None:
                diagnostics.append(
                    CompositionDiagnostic(
                        code="LOUDNESS_UNMEASURABLE",
                        message=loudness.warning,
                        severity=DiagnosticSeverity.WARNING,
                        context={"sample_count": len(waveform)},
                    )
                )
            else:
                diagnostics.append(
                    CompositionDiagnostic(
                        code="LOUDNESS_TARGET_LIMITED",
                        message=loudness.warning,
                        severity=DiagnosticSeverity.WARNING,
                        context={
                            "requested_gain_db": loudness.requested_gain_db,
                            "applied_gain_db": loudness.applied_gain_db,
                            "true_peak_ceiling_dbtp": loudness.true_peak_ceiling_dbtp,
                        },
                    )
                )
        if job.output.clip_policy == "warn" and np.any(np.abs(waveform) > 1.0):
            diagnostics.append(
                CompositionDiagnostic(
                    code="CLIPPING_POSSIBLE",
                    message="final waveform contains samples outside the PCM range",
                    severity=DiagnosticSeverity.WARNING,
                    context={"sample_count": len(waveform)},
                )
            )
        result = CompositionResult(
            audio=waveform,
            sample_rate=rate,
            items=tuple(composed),
            markers=tuple(markers),
            spans=tuple(spans),
            diagnostics=tuple(diagnostics),
            provenance={
                "job_id": job.job_id,
                "producer": dict(job.producer),
                "applied_loudness_gain_db": loudness.applied_gain_db,
                "loudness": _loudness_summary(loudness),
            },
            loudness=loudness,
        )
        _emit(
            on_progress,
            CompositionProgress(
                kind="compose_completed",
                completed_items=total_items,
                total_items=total_items,
                target_sample_rate=rate,
                output_frames=len(waveform),
                completed_audio_seconds=completed_audio_seconds,
                total_audio_seconds=total_audio_seconds,
            ),
        )
        return result

    def to_wav(
        self,
        job: AudioJob,
        path: str | Path,
        *,
        on_progress: CompositionProgressCallback | None = None,
    ) -> Path:
        result = self.compose(job, on_progress=on_progress)
        return write_wav(path, result.audio, result.sample_rate, clip_policy=job.output.clip_policy)

    def compose_to_wav(
        self,
        manifest: str | Path,
        path: str | Path,
        *,
        on_progress: CompositionProgressCallback | None = None,
    ) -> Path:
        return self.to_wav(
            AudioJob.load(str(manifest), verify_sources=False),
            path,
            on_progress=on_progress,
        )
