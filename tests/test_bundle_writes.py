from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import audiocompose.job as job_module
from audiocompose import AudioBufferSource, AudioClip, AudioJob


def _job(clip_count: int, *, value: float = 0.25) -> AudioJob:
    return AudioJob(
        tuple(
            AudioClip(
                f"clip-{index}",
                AudioBufferSource(np.full(32, value, dtype=np.float32), 32),
            )
            for index in range(clip_count)
        )
    )


def test_repeated_bundle_saves_replace_the_parts_set_deterministically(tmp_path: Path) -> None:
    bundle = tmp_path / "job.audiojob"
    unrelated = bundle / "notes.txt"
    first_manifest = Path(_job(3).save(bundle))
    unrelated.write_text("keep me", encoding="utf-8")

    second_manifest = Path(_job(1, value=0.5).save(bundle))
    second_bytes = second_manifest.read_bytes()
    second_part_bytes = (bundle / "parts" / "000001.wav").read_bytes()
    assert sorted(path.name for path in (bundle / "parts").iterdir()) == ["000001.wav"]
    assert unrelated.read_text(encoding="utf-8") == "keep me"

    repeated_manifest = Path(_job(1, value=0.5).save(bundle))

    assert first_manifest == second_manifest == repeated_manifest
    assert repeated_manifest.read_bytes() == second_bytes
    assert (bundle / "parts" / "000001.wav").read_bytes() == second_part_bytes
    assert sorted(path.name for path in (bundle / "parts").iterdir()) == ["000001.wav"]
    assert not list(bundle.glob(".audiojob-stage-*"))


def test_bundle_write_rolls_back_parts_if_atomic_manifest_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "job.audiojob"
    manifest = Path(_job(2).save(bundle))
    original_manifest = manifest.read_bytes()
    original_parts = {path.name: path.read_bytes() for path in (bundle / "parts").iterdir()}
    real_replace = os.replace

    def fail_manifest_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]):
        if Path(destination) == manifest:
            raise OSError("manifest replacement failed")
        return real_replace(source, destination)

    monkeypatch.setattr(job_module.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="manifest replacement failed"):
        _job(1, value=0.75).save(bundle)

    assert manifest.read_bytes() == original_manifest
    assert {path.name: path.read_bytes() for path in (bundle / "parts").iterdir()} == original_parts
    assert not list(bundle.glob(".audiojob-stage-*"))
