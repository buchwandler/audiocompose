from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audiocompose import (
    AudioBufferSource,
    AudioClip,
    AudioJob,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
)


def test_audiojob_schema_document_is_valid_and_requires_identity() -> None:
    schema = json.loads(Path("spec/audiojob-v1.schema.json").read_text())
    assert schema["properties"]["format"]["const"] == "audiojob"
    assert "job_id" in schema["required"]


def test_save_load_save_preserves_canonical_identity(tmp_path: Path) -> None:
    job = AudioJob(
        (AudioClip("one", AudioBufferSource(np.linspace(-0.25, 0.25, 32), 32)),),
        output=OutputPolicy(
            sample_rate=32,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
    )
    first = Path(job.save(tmp_path / "first.audiojob"))
    payload = json.loads(first.read_text())
    original_id = payload["job_id"]
    first.write_text(json.dumps(payload, separators=(",", ":")))
    loaded = AudioJob.load(first)
    assert loaded.job_id == original_id

    second = Path(loaded.save(tmp_path / "second.audiojob"))
    second_payload = json.loads(second.read_text())
    assert second_payload["job_id"] == original_id
    np.testing.assert_allclose(
        Composer(sample_rate=32).compose(loaded).audio,
        Composer(sample_rate=32).compose(AudioJob.load(second)).audio,
    )
