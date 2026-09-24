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
    AudioValidationError,
    Composer,
    Gain,
    Silence,
    Tempo,
)
from audiocompose.job import _job_id
from audiocompose.wav import write_wav


def test_composes_buffers_silence_and_mixed_rates() -> None:
    job = AudioJob(
        (
            AudioClip("a", AudioBufferSource(np.ones(24000, dtype=np.float32), 24000)),
            Silence("pause", 0.5),
            AudioClip("b", AudioBufferSource(np.ones(22050, dtype=np.float32), 22050)),
        )
    )

    result = Composer(sample_rate=24000).compose(job)

    assert result.audio.dtype == np.float32
    assert result.sample_rate == 24000
    assert len(result.audio) == 60000
    assert result.items[1].start_sample == 24000
    assert np.all(result.audio[24000:36000] == 0)


def test_operations_are_applied_in_declared_order() -> None:
    audio = np.full(100, 0.5, dtype=np.float32)
    job = AudioJob((AudioClip("clip", AudioBufferSource(audio, 100), (Gain(6.0), Gain(-6.0))),))

    result = Composer(sample_rate=100).compose(job)

    np.testing.assert_allclose(result.audio, audio, rtol=1e-6, atol=1e-6)


def test_tempo_updates_duration_and_marker_coordinates() -> None:
    audio = np.arange(100, dtype=np.float32)
    job = AudioJob(
        (
            AudioClip(
                "clip", AudioBufferSource(audio, 100), (Tempo(2.0),), (AudioAnchor("middle", 50),)
            ),
        )
    )

    result = Composer(sample_rate=100).compose(job)

    assert len(result.audio) == 50
    assert result.markers[0].sample_offset == 25


def test_bundle_roundtrip(tmp_path) -> None:
    job = AudioJob(
        (
            AudioClip("one", AudioBufferSource(np.linspace(-0.5, 0.5, 100), 100), (Gain(-3),)),
            Silence("pause", 0.1),
        )
    )

    manifest = job.save(tmp_path / "simple.audiojob")
    loaded = AudioJob.load(manifest)
    result = Composer().compose(loaded)

    assert (
        json.loads((tmp_path / "simple.audiojob" / "audiojob.json").read_text())["format"]
        == "audiojob"
    )
    assert len(result.audio) == 26400
    assert isinstance(loaded.items[0].source, AudioFileSource)


def test_bundle_rejects_path_traversal(tmp_path) -> None:
    manifest = Path(
        AudioJob((AudioClip("x", AudioBufferSource(np.zeros(8, dtype=np.float32), 8)),)).save(
            tmp_path / "job.audiojob"
        )
    )
    payload = json.loads(manifest.read_text())
    payload["items"][0]["source"]["path"] = "../outside.wav"
    payload["job_id"] = _job_id(payload)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(AudioValidationError, match="relative"):
        AudioJob.load(manifest)


def test_wav_output(tmp_path) -> None:
    output = tmp_path / "out.wav"
    write_wav(output, np.zeros(10, dtype=np.float32), 8000)
    assert output.exists()


def test_checked_in_fixture() -> None:
    job = AudioJob.load("tests/fixtures/simple.audiojob/audiojob.json")
    result = Composer().compose(job)
    assert len(result.audio) == 720
    assert result.markers[0].sample_offset == 120
