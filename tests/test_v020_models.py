from __future__ import annotations

from typing import Any, ClassVar

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
    FadeIn,
    FadeOut,
    Gain,
    LoudnessPolicy,
    OutputPolicy,
    PitchShift,
    Silence,
    Tempo,
)
from audiocompose.alignment import ComposedSpan
from audiocompose.operations import AudioOperation, operation_from_dict


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        (Gain, True),
        (PitchShift, True),
        (Tempo, True),
        (FadeIn, True),
        (FadeOut, True),
        (Gain, "1.0"),
        (PitchShift, float("nan")),
        (Tempo, float("inf")),
        (FadeIn, -1),
        (FadeOut, -1),
    ],
)
def test_builtin_operation_models_reject_invalid_numeric_values(
    operation: type, value: Any
) -> None:
    with pytest.raises(AudioValidationError):
        operation(value)


def test_builtin_operation_discriminator_is_not_a_constructor_argument() -> None:
    gain = Gain(1)

    assert gain.db == 1.0
    assert gain.type == "gain"
    with pytest.raises(TypeError):
        Gain(1.0, "bogus")  # type: ignore[call-arg]


def test_operation_parser_does_not_coerce_numeric_strings_or_booleans() -> None:
    for value in ("1.0", True):
        with pytest.raises(AudioValidationError):
            operation_from_dict({"type": "gain", "db": value})


class _CustomOperation(AudioOperation):
    type: ClassVar[str] = "custom"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type}


def test_audio_clip_rejects_operations_outside_the_closed_union() -> None:
    source = AudioBufferSource(np.zeros(8, dtype=np.float32), 8)

    with pytest.raises(AudioValidationError, match="unsupported operation"):
        AudioClip("clip", source, operations=(_CustomOperation(),))


def test_audio_buffer_source_snapshots_and_protects_its_input() -> None:
    samples = np.arange(8, dtype=np.float32)
    source = AudioBufferSource(samples, 24_000)
    samples[:] = -1

    loaded, sample_rate = source.load()
    np.testing.assert_array_equal(loaded, np.arange(8, dtype=np.float32))
    assert sample_rate == 24_000
    assert loaded.flags.writeable
    with pytest.raises(ValueError):
        source.audio[0] = -1


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (lambda value: AudioBufferSource(np.zeros(1), value), True),
        (lambda value: AudioBufferSource(np.zeros(1), 1, channels=value), True),
        (lambda value: OutputPolicy(sample_rate=value), True),
        (lambda value: OutputPolicy(channels=value), True),
        (lambda value: LoudnessPolicy(target_lufs=value), True),
    ],
)
def test_public_numeric_models_reject_booleans(factory, value: Any) -> None:
    with pytest.raises(AudioValidationError):
        factory(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"path": ""},
        {"path": "source.wav", "expected_sha256": "bad"},
        {"path": "source.wav", "expected_sha256": "A" * 64},
        {"path": "source.wav", "sample_rate": True},
        {"path": "source.wav", "sample_rate": 0},
        {"path": "source.wav", "frames": True},
        {"path": "source.wav", "frames": -1},
        {"path": "source.wav", "channels": True},
        {"path": "source.wav", "channels": 2},
    ],
)
def test_audio_file_source_validates_fields_at_construction(kwargs: dict[str, Any]) -> None:
    with pytest.raises(AudioValidationError):
        AudioFileSource(**kwargs)


def test_audio_file_source_accepts_valid_optional_metadata() -> None:
    source = AudioFileSource(
        "source.wav",
        expected_sha256="a" * 64,
        sample_rate=24_000,
        channels=1,
        frames=0,
    )

    assert source.path == "source.wav"


def test_model_metadata_is_deeply_detached_from_callers() -> None:
    clip_metadata = {"nested": {"values": [1]}}
    producer = {"producer": {"tags": ["original"]}}
    source_metadata = {"source": {"labels": ["source"]}}
    span_metadata = {"span": {"values": [2]}}
    silence_metadata = {"pause": {"labels": ["quiet"]}}
    clip = AudioClip(
        "clip",
        AudioBufferSource(np.zeros(8, dtype=np.float32), 8),
        metadata=clip_metadata,
    )
    span = AudioSpan(0, 1, 0, 1, metadata=span_metadata)
    composed_span = ComposedSpan("clip", 0, 1, 0, 1, metadata=span_metadata)
    silence = Silence("pause", 0.1, metadata=silence_metadata)
    job = AudioJob((clip, silence), producer=producer, source=source_metadata)

    clip_metadata["nested"]["values"][0] = 9
    producer["producer"]["tags"][0] = "changed"
    source_metadata["source"]["labels"][0] = "changed"
    span_metadata["span"]["values"][0] = 9
    silence_metadata["pause"]["labels"][0] = "changed"

    assert clip.metadata["nested"]["values"] == [1]
    assert job.producer["producer"]["tags"] == ["original"]
    assert job.source["source"]["labels"] == ["source"]
    assert span.metadata["span"]["values"] == [2]
    assert composed_span.metadata["span"]["values"] == [2]
    assert job.items[1].metadata["pause"]["labels"] == ["quiet"]


@pytest.mark.parametrize(
    "metadata",
    [{1: "non-string key"}, {"nested": float("nan")}, {"value": object()}],
)
def test_metadata_rejects_values_outside_json(metadata: dict[Any, Any]) -> None:
    with pytest.raises(AudioValidationError):
        AudioClip("clip", AudioBufferSource(np.zeros(8, dtype=np.float32), 8), metadata=metadata)


@pytest.mark.parametrize("job_id", ["", "sha256:" + "A" * 64, "sha256:1234", "other:" + "a" * 64])
def test_audio_job_rejects_noncanonical_identity(job_id: str) -> None:
    with pytest.raises(AudioValidationError, match="canonical sha256"):
        AudioJob((), job_id=job_id)


def test_audio_job_accepts_canonical_identity_syntax() -> None:
    job = AudioJob((), job_id="sha256:" + "a" * 64)

    assert job.job_id == "sha256:" + "a" * 64


def test_anchor_and_span_numeric_booleans_remain_invalid() -> None:
    with pytest.raises(AudioValidationError):
        AudioAnchor("anchor", True)
    with pytest.raises(AudioValidationError):
        AudioSpan(True, 1, 0, 1)
