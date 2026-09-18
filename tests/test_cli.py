from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audiocompose import AudioAnchor, AudioBufferSource, AudioClip, AudioJob, write_wav
from audiocompose.cli import main


def test_cli_json_validate_inspect_timeline_and_report(tmp_path: Path, capsys) -> None:
    job = AudioJob(
        (
            AudioClip(
                "clip", AudioBufferSource(np.zeros(100), 100), anchors=(AudioAnchor("mark", 50),)
            ),
        )
    )
    manifest = Path(job.save(tmp_path / "job.audiojob"))

    assert main(["validate", str(manifest), "--json"]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["valid"] is True

    assert main(["inspect", str(manifest), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["clips"] == 1

    assert main(["timeline", str(manifest), "--json"]) == 0
    timeline = json.loads(capsys.readouterr().out)
    assert timeline["markers"][0]["id"] == "mark"

    report_path = tmp_path / "report.html"
    assert main(["report", str(manifest), "-o", str(report_path)]) == 0
    capsys.readouterr()
    assert "<svg" in report_path.read_text()


def test_cli_analyze_wav_json_and_malformed_json_error(tmp_path: Path, capsys) -> None:
    audio_path = tmp_path / "input.wav"
    audio = np.zeros(1_000, dtype=np.float32)
    audio[200:800] = 0.5
    write_wav(audio_path, audio, 1_000)

    assert main(["analyze", str(audio_path), "--json"]) == 0
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["sample_rate"] == 1_000

    malformed = tmp_path / "bad.json"
    malformed.write_text("{not json")
    assert main(["validate", str(malformed), "--json"]) == 2
    assert "traceback" not in capsys.readouterr().err.lower()
