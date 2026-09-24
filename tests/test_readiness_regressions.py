from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioFileSource,
    AudioJob,
    AudioSpan,
    AudioValidationError,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    Tempo,
)
from audiocompose.loudness import apply_complete_output_loudness
from audiocompose.wav import wav_info, write_wav


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    windowed = audio * np.hanning(len(audio))
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(frequencies[int(np.argmax(spectrum))])


def _tone(frequency: float = 440.0, sample_rate: int = 24_000, seconds: float = 1.0) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


def test_pitch_shift_changes_frequency_and_preserves_duration() -> None:
    sample_rate = 24_000
    source = _tone(sample_rate=sample_rate)
    result = Composer(sample_rate=sample_rate).compose(
        AudioJob(
            (AudioClip("tone", AudioBufferSource(source, sample_rate), (PitchShift(12.0),)),),
            output=OutputPolicy(
                sample_rate=sample_rate,
                loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
            ),
        )
    )

    assert len(result.audio) == len(source)
    assert _dominant_frequency(result.audio, sample_rate) == pytest.approx(880.0, abs=20.0)


def test_tempo_preserves_frequency_and_changes_duration() -> None:
    sample_rate = 24_000
    source = _tone(sample_rate=sample_rate)
    result = Composer(sample_rate=sample_rate).compose(
        AudioJob(
            (AudioClip("tone", AudioBufferSource(source, sample_rate), (Tempo(2.0),)),),
            output=OutputPolicy(
                sample_rate=sample_rate,
                loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
            ),
        )
    )

    assert len(result.audio) == pytest.approx(len(source) / 2, abs=2)
    assert _dominant_frequency(result.audio, sample_rate) == pytest.approx(440.0, abs=20.0)


def test_peak_ceiling_applies_without_target_and_for_negative_requested_gain() -> None:
    audio = np.full(4_800, 0.5, dtype=np.float32)
    peak_only = apply_complete_output_loudness(
        audio,
        24_000,
        LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=-12.0),
    )
    assert np.max(np.abs(peak_only.audio)) <= 10.0 ** (-12.0 / 20.0) + 1e-6

    limited = apply_complete_output_loudness(
        audio,
        24_000,
        LoudnessPolicy(target_lufs=-7.0, true_peak_ceiling_dbtp=-12.0),
    )
    assert np.max(np.abs(limited.audio)) <= 10.0 ** (-12.0 / 20.0) + 1e-6


def test_save_rejects_clip_id_traversal_before_writing_outside_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.audiojob"
    outside = tmp_path / "escape.wav"
    job = AudioJob((AudioClip("../../escape", AudioBufferSource(np.zeros(8), 8)),))

    job.save(bundle)

    assert not outside.exists()


def test_save_uses_unique_parts_for_duplicate_source_basenames(tmp_path: Path) -> None:
    first = tmp_path / "first" / "same.wav"
    second = tmp_path / "second" / "same.wav"
    first.parent.mkdir()
    second.parent.mkdir()
    write_wav(first, np.zeros(8), 8)
    write_wav(second, np.ones(8) * 0.25, 8)
    job = AudioJob(
        (
            AudioClip("one", AudioFileSource(first)),
            AudioClip("two", AudioFileSource(second)),
        )
    )

    manifest = Path(job.save(tmp_path / "bundle.audiojob"))
    parts = sorted(manifest.parent.joinpath("parts").glob("*.wav"))
    assert len(parts) == 2
    assert len({part.name for part in parts}) == 2


def test_file_sources_are_canonical_pcm32_parts(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    write_wav(source, np.linspace(-0.5, 0.5, 16), 8)

    manifest = Path(
        AudioJob((AudioClip("one", AudioFileSource(source)),)).save(tmp_path / "bundle.audiojob")
    )
    part = next(manifest.parent.joinpath("parts").glob("*.wav"))
    assert wav_info(part).sample_width == 4


def test_persisted_source_integrity_is_checked_when_loaded(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    write_wav(source, np.linspace(-0.5, 0.5, 16), 8)
    manifest = Path(
        AudioJob((AudioClip("one", AudioFileSource(source)),)).save(tmp_path / "bundle.audiojob")
    )
    part = next(manifest.parent.joinpath("parts").glob("*.wav"))
    write_wav(part, np.zeros(16, dtype=np.float32), 8)

    with pytest.raises(AudioValidationError, match="source hash mismatch"):
        AudioJob.load(manifest)

    deferred = AudioJob.load(manifest, verify_sources=False)
    with pytest.raises(AudioValidationError, match="source hash mismatch"):
        Composer().compose(deferred)


def test_job_id_is_verified_on_load(tmp_path: Path) -> None:
    manifest = Path(
        AudioJob((AudioClip("one", AudioBufferSource(np.zeros(8), 8)),)).save(
            tmp_path / "bundle.audiojob"
        )
    )
    payload = json.loads(manifest.read_text())
    payload["job_id"] = "sha256:" + "0" * 64
    manifest.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(AudioValidationError, match="job_id"):
        AudioJob.load(manifest)


def test_malformed_manifest_uses_audio_validation_error(tmp_path: Path) -> None:
    manifest = tmp_path / "audiojob.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "audiojob",
                "schema_version": 1,
                "items": [{"kind": "silence", "id": "pause", "seconds": "not-a-number"}],
            }
        )
    )

    with pytest.raises(AudioValidationError):
        AudioJob.load(manifest)


def test_model_rejects_invalid_spans_duplicate_anchors_and_out_of_range_coordinates() -> None:
    with pytest.raises(AudioValidationError):
        AudioSpan(3, 2, 0, 1)
    with pytest.raises(AudioValidationError):
        AudioClip(
            "clip",
            AudioBufferSource(np.zeros(8), 8),
            anchors=(AudioAnchor("same", 0), AudioAnchor("same", 1)),
        )
    job = AudioJob(
        (AudioClip("clip", AudioBufferSource(np.zeros(8), 8), anchors=(AudioAnchor("late", 9),)),)
    )
    with pytest.raises(AudioValidationError, match="exceeds source length"):
        job.validate()
