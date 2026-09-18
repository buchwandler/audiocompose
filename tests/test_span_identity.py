from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    AudioValidationError,
    Composer,
    write_wav,
)


def test_span_identity_and_metadata_survive_composition() -> None:
    span = AudioSpan(120, 125, 2, 6, id="word:17", metadata={"kind": "word", "text": "hello"})
    result = Composer(sample_rate=8).compose(
        AudioJob((AudioClip("clip", AudioBufferSource(np.zeros(8), 8), spans=(span,)),))
    )

    assert result.spans[0].id == "word:17"
    assert result.spans[0].metadata == {"kind": "word", "text": "hello"}
    assert result.spans[0].source_start == 120


def test_old_positional_span_constructor_remains_valid() -> None:
    assert AudioSpan(1, 2, 3, 4).metadata == {}


def test_source_coordinates_are_not_limited_by_audio_length() -> None:
    job = AudioJob(
        (
            AudioClip(
                "clip",
                AudioBufferSource(np.zeros(4), 4),
                spans=(AudioSpan(1000, 1005, 1, 3),),
            ),
        )
    )

    job.validate()


def test_sample_coordinates_are_limited_by_audio_length() -> None:
    job = AudioJob(
        (AudioClip("clip", AudioBufferSource(np.zeros(4), 4), spans=(AudioSpan(0, 5, 1, 5),)),)
    )

    with pytest.raises(AudioValidationError, match="sample range"):
        job.validate()


def test_span_metadata_must_be_json_safe() -> None:
    job = AudioJob(
        (
            AudioClip(
                "clip",
                AudioBufferSource(np.zeros(4), 4),
                spans=(AudioSpan(0, 1, 0, 1, metadata={"bad": object()}),),
            ),
        )
    )

    with pytest.raises(AudioValidationError, match=r"span\[0\]\.metadata"):
        job.validate()


def test_span_metadata_roundtrips_and_changes_job_identity(tmp_path: Path) -> None:
    source = AudioBufferSource(np.linspace(-0.2, 0.2, 8), 8)
    first_job = AudioJob(
        (AudioClip("clip", source, spans=(AudioSpan(100, 105, 1, 6, id="x", metadata={"v": 1}),)),)
    )
    first_manifest = Path(first_job.save(tmp_path / "first.audiojob"))
    first_payload = json.loads(first_manifest.read_text())
    loaded = AudioJob.load(first_manifest)

    assert loaded.items[0].spans[0].id == "x"
    assert loaded.items[0].spans[0].metadata == {"v": 1}

    second_job = AudioJob(
        (AudioClip("clip", source, spans=(AudioSpan(100, 105, 1, 6, id="x", metadata={"v": 2}),)),)
    )
    second_payload = json.loads(Path(second_job.save(tmp_path / "second.audiojob")).read_text())
    assert first_payload["job_id"] != second_payload["job_id"]


def test_old_manifest_without_optional_span_fields_loads(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_wav(source, np.zeros(4), 4)
    manifest = Path(
        AudioJob((AudioClip("clip", AudioBufferSource(np.zeros(4), 4), spans=(AudioSpan(0, 1, 0, 1),)),)).save(
            tmp_path / "bundle.audiojob"
        )
    )
    payload = json.loads(manifest.read_text())
    payload["items"][0]["spans"][0].pop("id", None)
    payload["items"][0]["spans"][0].pop("metadata", None)
    payload["job_id"] = "sha256:" + "0" * 64
    payload.pop("job_id")
    manifest.write_text(json.dumps(payload, indent=2) + "\n")

    loaded = AudioJob.load(manifest)
    assert loaded.items[0].spans[0].id is None
    assert loaded.items[0].spans[0].metadata == {}
