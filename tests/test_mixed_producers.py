from __future__ import annotations

from pathlib import Path

import numpy as np

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    AudioSpan,
    Composer,
    LoudnessPolicy,
    OutputPolicy,
    Silence,
)


def test_mixed_producer_job_is_composed_and_roundtrips(tmp_path: Path) -> None:
    job = AudioJob(
        (
            AudioClip(
                "engine-a.clip",
                AudioBufferSource(np.full(8, 0.1, dtype=np.float32), 8),
                anchors=(AudioAnchor("a-end", 8, "engine A end"),),
                spans=(AudioSpan(100, 105, 2, 6, id="engine_a.word", metadata={"engine": "a"}),),
                metadata={"engine_a.kind": "speech"},
            ),
            Silence("shared.pause", 0.5, metadata={"engine_b.kind": "pause"}),
            AudioClip(
                "engine-b.clip",
                AudioBufferSource(np.full(4, 0.2, dtype=np.float32), 4),
                anchors=(AudioAnchor("b-start", 0),),
                spans=(AudioSpan(200, 205, 1, 3, id="engine_b.word", metadata={"engine": "b"}),),
                metadata={"engine_b.kind": "speech"},
            ),
        ),
        output=OutputPolicy(
            sample_rate=8,
            loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
        ),
        producer={"engine_a.version": "1", "engine_b.version": "2"},
    )

    result = Composer().compose(job)
    manifest = job.save(tmp_path / "mixed.audiojob")
    loaded = AudioJob.load(manifest)
    loaded_result = Composer().compose(loaded)

    np.testing.assert_allclose(result.audio, loaded_result.audio)
    assert result.sample_rate == 8
    assert result.items == loaded_result.items
    assert result.markers == loaded_result.markers
    assert result.spans == loaded_result.spans
    assert result.loudness is not None
    assert loaded.items[0].metadata == {"engine_a.kind": "speech"}
    assert loaded.items[2].spans[0].metadata == {"engine": "b"}
