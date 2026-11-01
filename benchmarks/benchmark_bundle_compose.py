from __future__ import annotations

import argparse
import gc
import importlib
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

audiocompose = importlib.import_module("audiocompose")
job_module = importlib.import_module("audiocompose.job")
source_module = importlib.import_module("audiocompose.sources")
wav_module = importlib.import_module("audiocompose.wav")

AudioClip = audiocompose.AudioClip
AudioJob = audiocompose.AudioJob
Composer = audiocompose.Composer
Gain = audiocompose.Gain
LoudnessPolicy = audiocompose.LoudnessPolicy
OutputPolicy = audiocompose.OutputPolicy
Silence = audiocompose.Silence
AudioFileSource = source_module.AudioFileSource
_save_job_bundle = job_module._save_job_bundle
_atomic_write_prepared_wav = wav_module._atomic_write_prepared_wav
sha256_file = wav_module.sha256_file
write_wav = wav_module.write_wav

SAMPLE_RATE = 24_000
POLICIES = (
    ("A", True, None, -1.0),
    ("B", False, None, -1.0),
    ("C", False, None, None),
    ("D", False, -18.0, -1.0),
)


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _build_job(
    source_path: Path,
    source_hash: str,
    frames: int,
    clips: int,
    pause_seconds: float,
    policy: LoudnessPolicy,
) -> AudioJob:
    source = AudioFileSource(
        source_path,
        expected_sha256=source_hash,
        sample_rate=SAMPLE_RATE,
        channels=1,
        frames=frames,
    )
    gain_indices = {0, clips // 2}
    items = []
    for index in range(clips):
        operations = (Gain(-1.0),) if index in gain_indices else ()
        items.append(AudioClip(f"speech-{index:04d}", source, operations=operations))
        items.append(Silence(f"pause-{index:04d}", pause_seconds))
    return AudioJob(
        tuple(items),
        output=OutputPolicy(sample_rate=SAMPLE_RATE, loudness=policy),
        producer={"benchmark": "synthetic-long-mono-chapter"},
    )


def _analysis_selection(policy: LoudnessPolicy) -> dict[str, bool]:
    return {
        "integrated_loudness": policy.collect_metrics or policy.target_lufs is not None,
        "sample_peak": policy.collect_metrics,
        "true_peak": policy.collect_metrics or policy.true_peak_ceiling_dbtp is not None,
        "calls_audiosig": (
            policy.collect_metrics
            or policy.target_lufs is not None
            or policy.true_peak_ceiling_dbtp is not None
        ),
    }


def _run_policy(
    label: str,
    policy: LoudnessPolicy,
    source_path: Path,
    source_hash: str,
    source_frames: int,
    clips: int,
    pause_seconds: float,
    work_root: Path,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    materialization_started = time.perf_counter()
    job = _build_job(source_path, source_hash, source_frames, clips, pause_seconds, policy)
    materialization_seconds = time.perf_counter() - materialization_started

    load_hash_seconds = {"save": 0.0, "composition": 0.0}
    load_hash_calls = {"save": 0, "composition": 0}
    phase = "save"
    original_load = AudioFileSource.load

    original_job_id = job_module._job_id
    identity_seconds = [0.0]

    def timed_job_id(serialized) -> str:
        started = time.perf_counter()
        try:
            return original_job_id(serialized)
        finally:
            identity_seconds[0] += time.perf_counter() - started

    def timed_load(source: AudioFileSource) -> tuple[np.ndarray, int]:
        started = time.perf_counter()
        try:
            return original_load(source)
        finally:
            load_hash_seconds[phase] += time.perf_counter() - started
            load_hash_calls[phase] += 1

    AudioFileSource.load = timed_load  # type: ignore[method-assign]
    job_module._job_id = timed_job_id
    try:
        bundle = work_root / f"audiojob-{label}"
        save_started = time.perf_counter()
        saved_bundle = _save_job_bundle(job, bundle)
        audiojob_save_seconds = time.perf_counter() - save_started
        saved_job_id = saved_bundle.job_id

        phase = "composition"
        item_starts: dict[int, float] = {}
        item_processing_seconds = 0.0
        assembly_started: float | None = None
        assembly_seconds = 0.0

        def on_progress(event) -> None:
            nonlocal item_processing_seconds, assembly_started, assembly_seconds
            now = time.perf_counter()
            if event.kind == "item_started" and event.item_index is not None:
                item_starts[event.item_index] = now
            elif event.kind == "item_completed" and event.item_index is not None:
                started = item_starts.pop(event.item_index, None)
                if started is not None:
                    item_processing_seconds += now - started
            elif event.kind == "assembly_started":
                assembly_started = now
            elif event.kind == "assembly_completed" and assembly_started is not None:
                assembly_seconds = now - assembly_started
                assembly_started = None

        composition_started = time.perf_counter()
        composition = Composer().compose(job, on_progress=on_progress)
        composition_seconds = time.perf_counter() - composition_started

        output_path = work_root / f"output-{label}.wav"
        write_started = time.perf_counter()
        _atomic_write_prepared_wav(
            output_path,
            composition.audio,
            composition.sample_rate,
            clip_policy=job.output.clip_policy,
        )
        wav_write_seconds = time.perf_counter() - write_started
        final_hash_started = time.perf_counter()
        output_hash = sha256_file(output_path)
        final_hash_seconds = time.perf_counter() - final_hash_started
    finally:
        AudioFileSource.load = original_load  # type: ignore[method-assign]
        job_module._job_id = original_job_id

    loudness = composition.loudness
    result = {
        "policy": {
            "label": label,
            "collect_metrics": policy.collect_metrics,
            "target_lufs": policy.target_lufs,
            "true_peak_ceiling_dbtp": policy.true_peak_ceiling_dbtp,
            "selected_analysis": _analysis_selection(policy),
        },
        "job_id": saved_job_id,
        "clip_count": clips,
        "silence_count": clips,
        "item_count": len(job.items),
        "output_frames": len(composition.audio),
        "output_duration_seconds": composition.duration_seconds,
        "output_sha256": output_hash,
        "stages": {
            "render_manifest_verification_seconds": None,
            "materialization_seconds": materialization_seconds,
            "audiojob_save_seconds": audiojob_save_seconds,
            "audiojob_identity_seconds": identity_seconds[0],
            "audiojob_source_load_and_hash_seconds": load_hash_seconds["save"],
            "audiojob_source_load_and_hash_calls": load_hash_calls["save"],
            "composition_seconds": composition_seconds,
            "composition_source_load_and_hash_seconds": load_hash_seconds["composition"],
            "composition_source_load_and_hash_calls": load_hash_calls["composition"],
            "item_processing_and_fused_placement_seconds": item_processing_seconds,
            "assembly_seconds": assembly_seconds,
            "loudness_analysis_seconds": loudness.analysis_seconds if loudness else 0.0,
            "loudness_gain_seconds": loudness.gain_seconds if loudness else 0.0,
            "loudness_post_gain_metrics_seconds": (
                loudness.post_gain_metrics_seconds if loudness else 0.0
            ),
            "output_preparation_seconds": composition.runtime.get(
                "output_preparation_seconds", 0.0
            ),
            "wav_write_seconds": wav_write_seconds,
            "final_hash_seconds": final_hash_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
    }
    del composition, job
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run a short fixture suitable for validation"
    )
    parser.add_argument("--clips", type=int, help="speech clip count (default: 400)")
    parser.add_argument(
        "--speech-seconds", type=float, help="duration of each synthetic speech clip"
    )
    parser.add_argument("--pause-seconds", type=float, help="duration of each pause")
    args = parser.parse_args()

    clips = args.clips if args.clips is not None else (8 if args.smoke else 400)
    speech_seconds = (
        args.speech_seconds if args.speech_seconds is not None else (0.1 if args.smoke else 2.6)
    )
    pause_seconds = (
        args.pause_seconds if args.pause_seconds is not None else (0.04 if args.smoke else 0.5)
    )
    if clips <= 0 or speech_seconds <= 0 or pause_seconds < 0:
        parser.error(
            "clips and speech duration must be positive; pause duration cannot be negative"
        )

    source_frames = round(speech_seconds * SAMPLE_RATE)
    if source_frames <= 0:
        parser.error("speech duration is shorter than one sample")

    with tempfile.TemporaryDirectory(prefix="audiocompose-benchmark-") as temporary:
        work_root = Path(temporary)
        source_path = work_root / "synthetic-speech.wav"
        samples = (
            0.15
            * np.sin(2.0 * np.pi * 440.0 * np.arange(source_frames, dtype=np.float32) / SAMPLE_RATE)
        ).astype(np.float32)
        write_wav(source_path, samples, SAMPLE_RATE)
        source_hash = sha256_file(source_path)

        policy_results = []
        for label, collect_metrics, target_lufs, ceiling in POLICIES:
            policy = LoudnessPolicy(
                collect_metrics=collect_metrics,
                target_lufs=target_lufs,
                true_peak_ceiling_dbtp=ceiling,
            )
            policy_results.append(
                _run_policy(
                    label,
                    policy,
                    source_path,
                    source_hash,
                    source_frames,
                    clips,
                    pause_seconds,
                    work_root,
                )
            )

    report = {
        "format": "audiocompose-bundle-compose-benchmark",
        "version": 1,
        "sample_rate": SAMPLE_RATE,
        "speech_clips": clips,
        "silence_items": clips,
        "speech_seconds_each": speech_seconds,
        "pause_seconds_each": pause_seconds,
        "estimated_chapter_seconds": clips * (speech_seconds + pause_seconds),
        "synthetic_fixture": True,
        "render_manifest_verification_scope": "not measured: fixture starts from synthetic AudioJob inputs",
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_scope": "whole benchmark process across all policy variants",
        "policies": policy_results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
