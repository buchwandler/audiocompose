from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    AudioValidationError,
    Gain,
    RatePitchEnvelope,
    Silence,
)
from audiocompose.job import _job_id


def _saved_payload(tmp_path: Path, schema_version: int) -> tuple[Path, dict[str, Any]]:
    job = AudioJob(
        (
            AudioClip(
                "clip",
                AudioBufferSource(np.linspace(-0.1, 0.1, 16, dtype=np.float32), 16),
                operations=(Gain(-3.0),),
                anchors=(AudioAnchor("middle", 8, "middle point"),),
                spans=(AudioSpan(0, 8, 0, 8, id="span", metadata={"label": "word"}),),
                metadata={"producer": {"name": "test"}},
            ),
            Silence("pause", 0.01),
        ),
        producer={"name": "fixture", "tags": ["strict"]},
        source={"origin": {"revision": 1}},
        schema_version=schema_version,
    )
    manifest = Path(job.save(tmp_path / f"v{schema_version}.audiojob"))
    return manifest, json.loads(manifest.read_text())


def _schema(version: int) -> Draft202012Validator:
    schema = json.loads(Path(f"spec/audiojob-v{version}.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _refresh_job_id(payload: dict[str, Any]) -> None:
    payload["job_id"] = _job_id(payload)


def _unknown_field_mutations() -> list[tuple[str, Callable[[dict[str, Any]], None]]]:
    return [
        ("root", lambda payload: payload.__setitem__("unexpected", 1)),
        ("output", lambda payload: payload["output"].__setitem__("unexpected", 1)),
        (
            "loudness",
            lambda payload: payload["output"]["loudness"].__setitem__("unexpected", 1),
        ),
        ("item", lambda payload: payload["items"][0].__setitem__("unexpected", 1)),
        (
            "source",
            lambda payload: payload["items"][0]["source"].__setitem__("unexpected", 1),
        ),
        (
            "anchor",
            lambda payload: payload["items"][0]["anchors"][0].__setitem__("unexpected", 1),
        ),
        (
            "span",
            lambda payload: payload["items"][0]["spans"][0].__setitem__("unexpected", 1),
        ),
        (
            "operation",
            lambda payload: payload["items"][0]["operations"][0].__setitem__("unexpected", 1),
        ),
    ]


@pytest.mark.parametrize("version", [1, 2])
def test_saved_manifests_validate_and_roundtrip_canonically(tmp_path: Path, version: int) -> None:
    manifest, payload = _saved_payload(tmp_path, version)
    validator = _schema(version)
    validator.validate(payload)

    loaded = AudioJob.load(manifest)
    assert loaded.schema_version == version
    next_manifest = Path(loaded.save(tmp_path / f"roundtrip-v{version}.audiojob"))
    next_payload = json.loads(next_manifest.read_text())
    validator.validate(next_payload)
    assert next_payload == payload


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "name,mutate",
    _unknown_field_mutations(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_unknown_fields_fail_schema_and_runtime(
    tmp_path: Path,
    version: int,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest, original = _saved_payload(tmp_path, version)
    payload = copy.deepcopy(original)
    mutate(payload)
    _refresh_job_id(payload)

    assert not _schema(version).is_valid(payload), name
    with pytest.raises(AudioValidationError, match="unknown field"):
        AudioJob.from_dict(payload, base_dir=str(manifest.parent))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["output"].__setitem__("sample_rate", True),
        lambda payload: payload["output"].__setitem__("sample_rate", "24000"),
        lambda payload: payload["items"][0]["anchors"][0].__setitem__("sample_offset", "2"),
        lambda payload: payload["items"][0]["anchors"][0].__setitem__("sample_offset", 2.5),
        lambda payload: payload["items"][0]["spans"][0].__setitem__("sample_start", True),
        lambda payload: payload["items"][0]["operations"][0].__setitem__("db", "-3"),
        lambda payload: payload["items"][1].__setitem__("seconds", "0.01"),
        lambda payload: payload["items"][0]["source"].__setitem__("path", 7),
    ],
)
def test_wrong_types_fail_schema_and_runtime(
    tmp_path: Path,
    version: int,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest, original = _saved_payload(tmp_path, version)
    payload = copy.deepcopy(original)
    mutate(payload)
    _refresh_job_id(payload)

    assert not _schema(version).is_valid(payload)
    with pytest.raises(AudioValidationError):
        AudioJob.from_dict(payload, base_dir=str(manifest.parent))


@pytest.mark.parametrize("version", [1, 2])
def test_persisted_loading_requires_canonical_job_identity(tmp_path: Path, version: int) -> None:
    manifest, original = _saved_payload(tmp_path, version)
    validator = _schema(version)

    missing = copy.deepcopy(original)
    del missing["job_id"]
    assert not validator.is_valid(missing)
    with pytest.raises(AudioValidationError, match="missing required field.*job_id"):
        AudioJob.from_dict(missing, base_dir=str(manifest.parent))

    malformed = copy.deepcopy(original)
    malformed["job_id"] = "sha256:" + "A" * 64
    assert not validator.is_valid(malformed)
    with pytest.raises(AudioValidationError, match="canonical sha256"):
        AudioJob.from_dict(malformed, base_dir=str(manifest.parent))

    mismatched = copy.deepcopy(original)
    mismatched["job_id"] = "sha256:" + "0" * 64
    assert validator.is_valid(mismatched)
    with pytest.raises(AudioValidationError, match="job_id does not match"):
        AudioJob.from_dict(mismatched, base_dir=str(manifest.parent))


def test_v1_schema_uses_type_specific_operation_shapes(tmp_path: Path) -> None:
    manifest, original = _saved_payload(tmp_path, 1)
    payload = copy.deepcopy(original)
    operation = payload["items"][0]["operations"][0]
    operation["factor"] = 2.0
    _refresh_job_id(payload)

    assert not _schema(1).is_valid(payload)
    with pytest.raises(AudioValidationError, match="unknown field"):
        AudioJob.from_dict(payload, base_dir=str(manifest.parent))


@pytest.mark.parametrize(("curve", "index"), [("rate", 0), ("pitch", 0)])
def test_v2_schema_and_parser_reject_unknown_curve_point_fields(
    tmp_path: Path, curve: str, index: int
) -> None:
    manifest, payload = _saved_payload(tmp_path, 2)
    payload["items"][0]["operations"][0] = RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.9,
        rate_seconds=0.5,
        from_semitones=0.0,
        to_semitones=1.0,
        pitch_seconds=0.25,
    ).to_dict()
    payload["items"][0]["operations"][0][curve][index]["future"] = True
    _refresh_job_id(payload)

    assert not _schema(2).is_valid(payload)
    with pytest.raises(AudioValidationError, match="unknown field"):
        AudioJob.from_dict(payload, base_dir=str(manifest.parent))
