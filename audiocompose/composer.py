from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from audiosig import LoudnessProgress

from .alignment import ComposedMarker, ComposedSpan
from .diagnostics import CompositionDiagnostic, DiagnosticSeverity
from .errors import AudioValidationError, CompositionError
from .job import validate_clip_geometry
from .loudness import LoudnessPolicy, LoudnessResult, apply_complete_output_loudness
from .model import AudioClip, AudioJob, ComposedItem, CompositionResult, Silence
from .operations import PitchShift, Tempo, apply_operation, apply_temporal_group
from .progress import (
    CompositionEstimate,
    CompositionProgress,
    CompositionProgressCallback,
    ProgressItemKind,
)
from .resampling import resample_audio
from .sources import AudioBufferSource, AudioFileSource
from .wav import _prepare_output, _write_prepared_wav


def _seconds_samples(seconds: float, rate: int) -> int:
    return round(seconds * rate)


def _item_output_frames_and_work(
    item: AudioClip | Silence, output_rate: int
) -> tuple[int | None, float | None]:
    if isinstance(item, Silence):
        frames = _seconds_samples(item.seconds, output_rate)
        return frames, float(frames)

    source = item.source
    if isinstance(source, AudioBufferSource):
        source_frames: int | None = len(source.audio)
        source_rate: int | None = source.sample_rate
        source_load_passes = 1
    else:
        source_frames = source.frames
        source_rate = source.sample_rate
        source_load_passes = 2 if source.expected_sha256 is not None else 1
    if source_frames is None or source_rate is None:
        return None, None
    frames = source_frames

    work_units = float(frames * source_load_passes)
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
        work_units += frames
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
        work_units += frames
        frames = round(frames * output_rate / source_rate)

    # Item availability is a small pass in addition to the source/DSP work.
    work_units += frames
    return frames, work_units


def _predicted_output_frames(item: AudioClip | Silence, output_rate: int) -> int | None:
    return _item_output_frames_and_work(item, output_rate)[0]


def _loudness_work_phases(policy: LoudnessPolicy, sample_rate: int) -> tuple[str, ...]:
    integrated = (policy.collect_metrics or policy.target_lufs is not None) and round(
        0.1 * sample_rate
    ) > 0
    sample_peak = policy.collect_metrics
    true_peak = policy.collect_metrics or policy.true_peak_ceiling_dbtp is not None
    phases: list[str] = []
    if integrated:
        phases.append("integrated_loudness")
    if true_peak:
        # AudioSig measures sample peak in this same streaming pass.
        phases.append("true_peak")
    elif sample_peak:
        phases.append("sample_peak")
    return tuple(phases)


def estimate_composition(
    job: AudioJob,
    *,
    sample_rate: int | None = None,
) -> CompositionEstimate:
    """Estimate deterministic output geometry and execution work without processing audio.

    ``total_work_units`` is an approximate frame-pass metric, not a prediction
    of wall-clock duration. If any source geometry is unknown, output totals and
    the total work estimate remain unknown.
    """
    rate = job.output.sample_rate if sample_rate is None else sample_rate
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer or None")

    geometry = tuple(_item_output_frames_and_work(item, rate) for item in job.items)
    item_output_frames = tuple(frames for frames, _ in geometry)
    unknown_items = tuple(
        index for index, frames in enumerate(item_output_frames) if frames is None
    )
    total_output_frames = (
        sum(frames for frames in item_output_frames if frames is not None)
        if not unknown_items
        else None
    )
    total_audio_seconds = total_output_frames / rate if total_output_frames is not None else None
    total_work_units: float | None = None
    if not unknown_items and total_output_frames is not None:
        total_work_units = sum(work or 0.0 for _, work in geometry)
        # Placement is fused into item processing; loudness and final clipping remain extra passes.
        total_work_units += total_output_frames * len(
            _loudness_work_phases(job.output.loudness, rate)
        )
        total_work_units += total_output_frames

    return CompositionEstimate(
        sample_rate=rate,
        item_output_frames=item_output_frames,
        total_output_frames=total_output_frames,
        total_audio_seconds=total_audio_seconds,
        total_work_units=total_work_units,
        unknown_items=unknown_items,
    )


class _WorkProgressCallback:
    def __init__(
        self,
        callback: CompositionProgressCallback,
        job: AudioJob,
        total_work_units: float | None,
        sample_rate: int,
    ) -> None:
        self._callback = callback
        self._job = job
        self._total_work_units = total_work_units
        self._sample_rate = sample_rate
        self._completed_work_units = 0.0
        self._fraction = 0.0
        self._loudness_start_work_units: float | None = None
        self._loudness_phases: tuple[str, ...] = ()
        self._loudness_frames = 0

    def __call__(self, event: CompositionProgress) -> None:
        completed = 0.0
        if event.kind == "source_load_completed" and event.input_frames is not None:
            item = self._job.items[event.item_index] if event.item_index is not None else None
            passes = (
                2
                if isinstance(item, AudioClip)
                and isinstance(item.source, AudioFileSource)
                and item.source.expected_sha256 is not None
                else 1
            )
            completed = event.input_frames * passes
        elif event.kind == "operation_completed" and event.input_frames is not None:
            group_size = event.details.get("processing_group_size")
            group_start = event.details.get("processing_group_start")
            is_last_group_event = (
                isinstance(group_size, int)
                and isinstance(group_start, int)
                and event.operation_index == group_start + group_size - 1
            )
            if not isinstance(group_size, int) or is_last_group_event:
                completed = event.input_frames
        elif event.kind == "resample_completed" and event.input_frames is not None:
            completed = event.input_frames
        elif event.kind == "item_completed" and event.output_frames is not None:
            completed = event.output_frames
        elif event.kind == "loudness_started":
            self._loudness_start_work_units = self._completed_work_units
            self._loudness_phases = _loudness_work_phases(
                self._job.output.loudness, self._sample_rate
            )
            self._loudness_frames = event.input_frames or 0
        elif event.kind == "loudness_progress":
            phase = event.details.get("phase")
            phase_fraction = event.details.get("phase_fraction")
            if (
                self._loudness_start_work_units is not None
                and self._loudness_frames > 0
                and phase in self._loudness_phases
                and isinstance(phase_fraction, (int, float))
                and math.isfinite(phase_fraction)
            ):
                phase_index = self._loudness_phases.index(phase)
                bounded_fraction = min(1.0, max(0.0, float(phase_fraction)))
                stage_work = self._loudness_frames * (phase_index + bounded_fraction)
                self._completed_work_units = max(
                    self._completed_work_units,
                    self._loudness_start_work_units + stage_work,
                )
        elif event.kind == "loudness_completed":
            if self._loudness_start_work_units is not None:
                stage_work = self._loudness_frames * len(self._loudness_phases)
                self._completed_work_units = max(
                    self._completed_work_units,
                    self._loudness_start_work_units + stage_work,
                )
            self._loudness_start_work_units = None
            self._loudness_phases = ()
            self._loudness_frames = 0
        elif event.kind == "compose_completed" and event.output_frames is not None:
            completed = event.output_frames

        self._completed_work_units += float(completed)
        fraction: float | None = None
        if self._total_work_units is not None:
            if event.kind == "compose_completed":
                fraction = 1.0
            elif self._total_work_units == 0:
                fraction = 0.0
            else:
                fraction = min(1.0, self._completed_work_units / self._total_work_units)
                fraction = max(self._fraction, fraction)
            self._fraction = fraction
        self._callback(
            replace(
                event,
                completed_work_units=self._completed_work_units,
                total_work_units=self._total_work_units,
                fraction=fraction,
            )
        )


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
        composition_started = time.perf_counter()
        job.validate(verify_sources=False)
        rate = self.sample_rate if self.sample_rate is not None else job.output.sample_rate
        total_items = len(job.items)
        estimate = estimate_composition(job, sample_rate=rate)
        total_audio_seconds = estimate.total_audio_seconds
        if on_progress is not None:
            on_progress = _WorkProgressCallback(on_progress, job, estimate.total_work_units, rate)
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
        waveform = (
            np.empty(estimate.total_output_frames, dtype=np.float32)
            if estimate.total_output_frames is not None
            else None
        )
        parts: list[np.ndarray] | None = [] if waveform is None else None
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
            audio = np.asarray(audio, dtype=np.float32)
            next_cursor = cursor + len(audio)
            if waveform is None:
                assert parts is not None
                parts.append(audio)
            else:
                if next_cursor > len(waveform):
                    raise CompositionError("composed audio exceeded its estimated frame count")
                waveform[cursor:next_cursor] = audio
            cursor = next_cursor
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
            del audio
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
        if waveform is None:
            if parts:
                waveform = np.concatenate(parts).astype(np.float32, copy=False)
                parts.clear()
            else:
                waveform = np.zeros(0, dtype=np.float32)
        if cursor != len(waveform):
            raise CompositionError(
                f"composed output has {cursor} frames but allocated geometry requires {len(waveform)}"
            )
        assert waveform is not None
        loudness_input_frames = len(waveform)
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
                    "collect_metrics": loudness_policy.collect_metrics,
                    "frames": len(waveform),
                    "sample_rate": rate,
                },
            ),
        )

        def emit_loudness_progress(progress: LoudnessProgress) -> None:
            _emit(
                on_progress,
                CompositionProgress(
                    kind="loudness_progress",
                    completed_items=total_items,
                    total_items=total_items,
                    target_sample_rate=rate,
                    input_frames=loudness_input_frames,
                    completed_audio_seconds=completed_audio_seconds,
                    total_audio_seconds=total_audio_seconds,
                    details={
                        "phase": progress.phase,
                        "completed_frames": progress.completed_frames,
                        "total_frames": progress.total_frames,
                        "phase_fraction": progress.phase_fraction,
                    },
                ),
            )

        loudness = apply_complete_output_loudness(
            waveform,
            rate,
            loudness_policy,
            on_progress=emit_loudness_progress if on_progress is not None else None,
        )
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
                    "integrated_lufs_before": loudness.before.integrated_lufs,
                    "integrated_lufs_after": loudness.after.integrated_lufs,
                    "sample_peak_dbfs_before": loudness.before.sample_peak_dbfs,
                    "sample_peak_dbfs_after": loudness.after.sample_peak_dbfs,
                    "true_peak_dbtp_before": loudness.before.true_peak_dbtp,
                    "true_peak_dbtp_after": loudness.after.true_peak_dbtp,
                    "requested_gain_db": loudness.requested_gain_db,
                    "applied_gain_db": loudness.applied_gain_db,
                    "target_reached": loudness.target_reached,
                    "warning": loudness.warning,
                    "analysis_seconds": loudness.analysis_seconds,
                    "gain_seconds": loudness.gain_seconds,
                    "post_gain_metrics_seconds": loudness.post_gain_metrics_seconds,
                },
            ),
        )
        output_preparation_started = time.perf_counter()
        try:
            waveform, output_out_of_range = _prepare_output(
                waveform, clip_policy=job.output.clip_policy
            )
        except AudioValidationError as exc:
            if job.output.clip_policy != "error":
                raise
            raise CompositionError(str(exc)) from exc
        output_preparation_seconds = time.perf_counter() - output_preparation_started
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
        if job.output.clip_policy == "warn" and output_out_of_range:
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
            runtime={
                "composition_seconds": time.perf_counter() - composition_started,
                "output_preparation_seconds": output_preparation_seconds,
                "loudness_analysis_seconds": loudness.analysis_seconds,
                "loudness_gain_seconds": loudness.gain_seconds,
                "loudness_post_gain_metrics_seconds": loudness.post_gain_metrics_seconds,
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
        return _write_prepared_wav(
            path, result.audio, result.sample_rate, clip_policy=job.output.clip_policy
        )

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
