from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audiocompose import (
    AudioAnchor,
    AudioBufferSource,
    AudioClip,
    AudioJob,
    BookComposeProgress,
    Composer,
    CompositionProgress,
)
from audiocompose._cli_progress import CliProgressRenderer
from audiocompose.cli import main
from audiocompose.wav import write_wav


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


def test_cli_analyze_near_marker_reuses_composition(tmp_path, capsys, monkeypatch) -> None:
    job = AudioJob(
        (
            AudioClip(
                "clip",
                AudioBufferSource(np.zeros(100, dtype=np.float32), 100),
                anchors=(AudioAnchor("mark", 50),),
            ),
        )
    )
    manifest = job.save(tmp_path / "job.audiojob")
    original_compose = Composer.compose
    compose_count = 0

    def counted_compose(self, job, *, on_progress=None):
        nonlocal compose_count
        compose_count += 1
        return original_compose(self, job, on_progress=on_progress)

    monkeypatch.setattr(Composer, "compose", counted_compose)

    assert main(["analyze", manifest, "--near-marker", "mark", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["near_marker"] == "mark"
    assert compose_count == 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TTYStream(io.StringIO):
    def isatty(self) -> bool:
        return True


def _composition_event(
    kind: str,
    completed_work: float,
    *,
    completed_items: int = 0,
) -> CompositionProgress:
    return CompositionProgress(
        kind=kind,
        completed_items=completed_items,
        total_items=1,
        item_id="clip-1",
        completed_work_units=completed_work,
        total_work_units=100.0,
        fraction=completed_work / 100.0,
    )


def test_progress_renderer_tty_detection_eta_and_monotonic_output() -> None:
    clock = FakeClock()
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", stream=stream, clock=clock)
    assert renderer.enabled

    renderer.on_composition_progress(_composition_event("compose_started", 0.0))
    clock.advance(2.0)
    renderer.on_composition_progress(_composition_event("item_completed", 20.0, completed_items=1))
    draw_count = stream.getvalue().count("\r")
    renderer.on_composition_progress(_composition_event("item_completed", 25.0, completed_items=1))
    assert stream.getvalue().count("\r") == draw_count
    renderer.on_composition_progress(
        _composition_event("compose_completed", 100.0, completed_items=1)
    )
    renderer.finish(success=True)

    output = stream.getvalue()
    assert "ETA 00:08" in output
    assert "100.0%" in output
    assert "\r" in output
    assert output.endswith("\n")

    json_tty = CliProgressRenderer(mode="auto", json_mode=True, stream=TTYStream())
    assert not json_tty.enabled
    assert not CliProgressRenderer(mode="auto", stream=io.StringIO()).enabled


def test_generic_compose_progress_modes_and_json_stdout(tmp_path: Path, capsys) -> None:
    job = AudioJob((AudioClip("clip", AudioBufferSource(np.zeros(100, dtype=np.float32), 100)),))
    manifest = Path(job.save(tmp_path / "job.audiojob"))

    output = tmp_path / "forced.wav"
    assert main(["compose", str(manifest), str(output), "--json", "--progress", "always"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["composed"] is True
    assert result["output"] == str(output)
    forced_stderr = captured.err
    assert "composing" in forced_stderr
    assert "writing output" in forced_stderr
    assert "verifying output" in forced_stderr
    assert "elapsed" in forced_stderr
    assert "\r" not in forced_stderr

    auto_output = tmp_path / "auto.wav"
    assert main(["compose", str(manifest), str(auto_output), "--json"]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert captured.err == ""

    never_output = tmp_path / "never.wav"
    assert main(["compose", str(manifest), str(never_output), "--progress", "never"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_renderplan_cli_progress_stays_on_stderr_with_json(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    import audiocompose

    output = tmp_path / "renderplan.wav"

    def fake_compose(plan, **kwargs):
        callback = kwargs["on_progress"]
        callback(_composition_event("compose_started", 0.0))
        callback(_composition_event("compose_completed", 100.0, completed_items=1))
        write_wav(output, np.zeros(4, dtype=np.float32), 8)
        return SimpleNamespace(
            output=output,
            plan_id="plan-id",
            manifest_id="manifest-id",
            sample_rate=8,
            frames=4,
            duration_seconds=0.5,
            timeline=(),
            diagnostics=(),
            audiojob=None,
        )

    monkeypatch.setattr(audiocompose, "compose_renderplan", fake_compose)
    assert (
        main(
            [
                "compose-renderplan",
                "plan.toml",
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--json",
                "--progress",
                "always",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["output"] == str(output)
    progress = captured.err
    assert "preparing handoff" in progress
    assert "composing RenderPlan" in progress
    assert "writing output" in progress
    assert "verifying output" in progress


def test_book_cli_renders_nested_progress_to_stderr(monkeypatch, tmp_path: Path, capsys) -> None:
    import audiocompose.book as book_module

    def fake_compose(book, *, on_progress, **kwargs):
        on_progress(BookComposeProgress(0, 1, kind="book_started", fraction=0.0))
        on_progress(
            BookComposeProgress(
                0, 1, kind="chapter_started", chapter_id="chapter-0001", fraction=0.0
            )
        )
        on_progress(
            BookComposeProgress(
                0, 1, kind="chapter_compose_started", chapter_id="chapter-0001", fraction=0.0
            )
        )
        on_progress(
            BookComposeProgress(
                0,
                1,
                kind="chapter_compose_progress",
                chapter_id="chapter-0001",
                composition=_composition_event("compose_started", 0.0),
                fraction=0.0,
            )
        )
        on_progress(
            BookComposeProgress(
                0,
                1,
                kind="chapter_compose_progress",
                chapter_id="chapter-0001",
                composition=_composition_event("compose_completed", 100.0, completed_items=1),
                fraction=1.0,
            )
        )
        on_progress(BookComposeProgress(1, 1, kind="book_completed", fraction=1.0))
        return SimpleNamespace(exit_code=0, to_dict=lambda: {"composed": True})

    monkeypatch.setattr(book_module, "compose_book", fake_compose)
    bundle = tmp_path / "fixture.ssmd-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1})
    )
    assert (
        main(
            [
                "bundle",
                "compose",
                str(bundle),
                "--variant",
                "fixture",
                "--json",
                "--progress",
                "always",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "format": "audiocompose-bundle-result",
        "version": 1,
        "kind": "book",
        "result": {"composed": True},
    }
    progress = captured.err
    assert "chapter 1/1" in progress
    assert "preparing handoff" in progress or "starting composition" in progress
    assert "chapter 100.0%" in progress
    assert "finalizing report" in progress
    assert "complete" in progress


def test_compose_output_failure_does_not_finish_progress_successfully(
    tmp_path: Path, capsys
) -> None:
    job = AudioJob((AudioClip("clip", AudioBufferSource(np.zeros(100, dtype=np.float32), 100)),))
    manifest = Path(job.save(tmp_path / "job.audiojob"))

    assert main(["compose", str(manifest), str(tmp_path), "--progress", "always"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed" in captured.err
    assert not any("100.0%" in line and "complete" in line for line in captured.err.splitlines())
