from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioValidationError,
    Composer,
    CompositionError,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
)
from audiocompose.cli import main
from audiocompose.wav import prepare_output, read_wav


def _output_policy(clip_policy: str = "clamp") -> OutputPolicy:
    return OutputPolicy(
        sample_rate=24_000,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        clip_policy=clip_policy,  # type: ignore[arg-type]
    )


def _job(audio: np.ndarray, *, clip_policy: str = "clamp") -> AudioJob:
    return AudioJob(
        (
            AudioClip(
                "clip",
                AudioBufferSource(audio, 24_000),
                operations=(Gain(6.0),),
            ),
        ),
        output=_output_policy(clip_policy),
    )


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.asarray(0.5, dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float32),
        np.asarray([-np.inf], dtype=np.float32),
    ],
)
def test_prepare_output_rejects_non_mono_or_nonfinite(audio: np.ndarray) -> None:
    with pytest.raises(AudioValidationError, match="finite one-dimensional waveform"):
        prepare_output(audio)


def test_prepare_output_allows_empty_and_returns_contiguous_float32() -> None:
    output = prepare_output(np.asarray([], dtype=np.float64))

    assert output.dtype == np.float32
    assert output.ndim == 1
    assert output.flags.c_contiguous
    assert len(output) == 0


def test_prepare_output_policies_and_caller_storage() -> None:
    audio = np.asarray([0.25, 1.5, -1.5], dtype=np.float32)
    original = audio.copy()

    clamped = prepare_output(audio, clip_policy="clamp")
    warned = prepare_output(audio, clip_policy="warn")

    np.testing.assert_array_equal(clamped, [0.25, 1.0, -1.0])
    np.testing.assert_array_equal(warned, original)
    np.testing.assert_array_equal(audio, original)
    with pytest.raises(AudioValidationError, match="outside the PCM range"):
        prepare_output(audio, clip_policy="error")
    with pytest.raises(AudioValidationError, match="unknown clip policy"):
        prepare_output(audio, clip_policy="ignore")  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", [0, -1, True, 24_000.0])
def test_composer_rejects_invalid_sample_rate_at_construction(sample_rate: object) -> None:
    with pytest.raises(AudioValidationError, match="sample_rate"):
        Composer(sample_rate=sample_rate)  # type: ignore[arg-type]


def test_composer_clip_policies_agree_with_bundle_wav_output(tmp_path: Path) -> None:
    for clip_policy in ("clamp", "warn", "error"):
        job = _job(np.asarray([0.1, 0.75, -0.75], dtype=np.float32), clip_policy=clip_policy)
        manifest = Path(job.save(str(tmp_path / f"{clip_policy}.audiojob")))
        composer = Composer()

        if clip_policy == "error":
            with pytest.raises(CompositionError, match="outside the PCM range"):
                composer.compose(job)
            with pytest.raises(CompositionError, match="outside the PCM range"):
                composer.compose_to_wav(manifest, tmp_path / "error.wav")
            assert not (tmp_path / "error.wav").exists()
            continue

        result = composer.compose(job)
        output_path = tmp_path / f"{clip_policy}.wav"
        composer.compose_to_wav(manifest, output_path)
        written, sample_rate = read_wav(output_path)
        expected = np.asarray([0.1, 0.75, -0.75], dtype=np.float32) * (10.0 ** (6.0 / 20.0))

        assert sample_rate == 24_000
        if clip_policy == "clamp":
            np.testing.assert_allclose(result.audio, np.clip(expected, -1.0, 1.0), atol=1e-6)
        else:
            np.testing.assert_allclose(result.audio, expected, atol=1e-6)
            assert any(diagnostic.code == "CLIPPING_POSSIBLE" for diagnostic in result.diagnostics)
        np.testing.assert_allclose(written, np.clip(expected, -1.0, 1.0), atol=1 / 32767)


def test_validate_cli_reports_persisted_schema_version(tmp_path: Path, capsys) -> None:
    for schema_version in (1, 2):
        job = AudioJob(
            (AudioClip("clip", AudioBufferSource(np.zeros(8, dtype=np.float32), 8)),),
            schema_version=schema_version,
        )
        manifest = Path(job.save(str(tmp_path / f"v{schema_version}.audiojob")))

        assert main(["validate", str(manifest), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["schema_version"] == schema_version


def test_arbitrary_clip_metadata_does_not_change_composition_semantics() -> None:
    def compose(metadata: dict[str, str]):
        events = []
        job = AudioJob(
            (
                AudioClip(
                    "clip",
                    AudioBufferSource(np.ones(240, dtype=np.float32) * 0.1, 24_000),
                    metadata=metadata,
                ),
            ),
            output=_output_policy(),
        )
        result = Composer().compose(job, on_progress=events.append)
        return result, events

    ordinary, ordinary_events = compose({"producer": "example"})
    misleading, misleading_events = compose({"kind": "silence", "voice": "arbitrary"})

    np.testing.assert_array_equal(ordinary.audio, misleading.audio)
    assert ordinary.items == misleading.items
    assert ordinary_events[0].total_audio_seconds == misleading_events[0].total_audio_seconds
    assert ordinary_events[0].total_audio_seconds == pytest.approx(0.01)
    ordinary_item = next(event for event in ordinary_events if event.kind == "item_started")
    misleading_item = next(event for event in misleading_events if event.kind == "item_started")
    assert ordinary_item.item_kind == misleading_item.item_kind == "clip"
    assert ordinary_item.item_metadata != misleading_item.item_metadata
