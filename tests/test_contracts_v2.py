from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioValidationError,
    RatePitchEnvelope,
)


def make_job(operation: RatePitchEnvelope, *, schema_version: int = 2) -> AudioJob:
    source = AudioBufferSource(np.zeros(24000, dtype=np.float32), 24000)
    clip = AudioClip("seg-000481", source, operations=(operation,))
    return AudioJob((clip,), schema_version=schema_version)


def save_payload(job: AudioJob, path: Path) -> tuple[Path, dict[str, object]]:
    manifest = Path(job.save(path))
    return manifest, json.loads(manifest.read_text())


def envelope() -> RatePitchEnvelope:
    return RatePitchEnvelope.transition(
        from_rate=1.0,
        to_rate=0.85,
        rate_seconds=0.45,
        from_semitones=0.0,
        to_semitones=2.0,
        pitch_seconds=0.3,
    )


def test_new_jobs_default_to_audiojob_v2() -> None:
    assert AudioJob(()).schema_version == 2


def test_v2_schema_validates_envelopes_and_rejects_malformed_points(
    tmp_path: Path,
) -> None:
    schema_path = Path("spec/audiojob-v2.schema.json")
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    manifest, payload = save_payload(make_job(envelope()), tmp_path / "v2.audiojob")

    validator.validate(payload)
    assert payload["schema_version"] == 2
    operation = payload["items"][0]["operations"][0]
    assert operation["type"] == "rate_pitch_envelope"

    malformed = copy.deepcopy(payload)
    malformed["items"][0]["operations"][0]["rate"][0]["factor"] = 0
    assert not validator.is_valid(malformed)

    malformed = copy.deepcopy(payload)
    malformed["items"][0]["operations"][0]["pitch"][0].pop("semitones")
    assert not validator.is_valid(malformed)

    malformed = copy.deepcopy(payload)
    operation = malformed["items"][0]["operations"][0]
    operation["rate"] = []
    operation["pitch"] = []
    assert not validator.is_valid(malformed)

    v1_schema = json.loads(Path("spec/audiojob-v1.schema.json").read_text())
    Draft202012Validator.check_schema(v1_schema)
    assert not Draft202012Validator(v1_schema).is_valid(payload)
    assert manifest.exists()


def test_v2_save_load_save_preserves_schema_and_identity(tmp_path: Path) -> None:
    first, first_payload = save_payload(make_job(envelope()), tmp_path / "first.audiojob")
    original_id = first_payload["job_id"]
    loaded = AudioJob.load(first)
    assert loaded.schema_version == 2
    assert loaded.items[0].operations[0] == envelope()

    second, second_payload = save_payload(loaded, tmp_path / "second.audiojob")
    assert second_payload["schema_version"] == 2
    assert second_payload["job_id"] == original_id


def test_v2_job_identity_includes_each_envelope_field(tmp_path: Path) -> None:
    manifest, payload = save_payload(make_job(envelope()), tmp_path / "identity.audiojob")
    mutations = (
        lambda operation: operation["rate"][0].__setitem__("seconds", 0.01),
        lambda operation: operation["rate"][1].__setitem__("factor", 0.86),
        lambda operation: operation["pitch"][1].__setitem__("seconds", 0.31),
        lambda operation: operation["pitch"][1].__setitem__("semitones", 2.1),
        lambda operation: operation.__setitem__("interpolation", "cubic"),
        lambda operation: operation.__setitem__("time_base", "source"),
    )

    for mutate in mutations:
        changed = copy.deepcopy(payload)
        mutate(changed["items"][0]["operations"][0])
        with pytest.raises(AudioValidationError, match="job_id"):
            AudioJob.from_dict(changed, base_dir=str(manifest.parent))


def test_v1_jobs_reject_envelopes_on_save_and_load(tmp_path: Path) -> None:
    with pytest.raises(AudioValidationError, match="schema v1"):
        make_job(envelope(), schema_version=1).validate()

    _, payload = save_payload(make_job(envelope()), tmp_path / "v2.audiojob")
    payload["schema_version"] = 1
    payload.pop("job_id")
    with pytest.raises(AudioValidationError, match="schema v1"):
        AudioJob.from_dict(payload, base_dir=str(tmp_path / "v2.audiojob"))


def test_v2_parser_rejects_nonincreasing_curve_points(tmp_path: Path) -> None:
    manifest, payload = save_payload(make_job(envelope()), tmp_path / "duplicate.audiojob")
    payload.pop("job_id")
    payload["items"][0]["operations"][0]["rate"][1]["seconds"] = 0.0

    with pytest.raises(AudioValidationError, match="rate_pitch_envelope"):
        AudioJob.from_dict(payload, base_dir=str(manifest.parent))


def test_v2_parser_requires_explicit_envelope_fields(tmp_path: Path) -> None:
    manifest, payload = save_payload(make_job(envelope()), tmp_path / "missing-fields.audiojob")
    for field in ("interpolation", "time_base"):
        malformed = copy.deepcopy(payload)
        malformed.pop("job_id")
        del malformed["items"][0]["operations"][0][field]
        with pytest.raises(AudioValidationError, match="rate_pitch_envelope"):
            AudioJob.from_dict(malformed, base_dir=str(manifest.parent))
