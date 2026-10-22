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


def test_bundle_variants_help_and_empty_json_from_dot(tmp_path: Path, monkeypatch, capsys) -> None:
    import pytest

    from audiocompose.cli import _parser

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["bundle", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "compose" in help_text
    assert "variants" in help_text

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["bundle", "variants", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--renderplan-root", "--voicerender-root", "--output-root", "--json"):
        assert option in help_text

    bundle = tmp_path / "empty.ssmd-bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}), encoding="utf-8"
    )
    monkeypatch.chdir(bundle)
    assert main(["bundle", "variants", ".", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["format"] == "audiocompose-bundle-variants"
    assert payload["version"] == 1
    assert payload["count"] == 0
    assert captured.err == ""


def test_bundle_variants_human_table_and_custom_root_dispatch(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import audiocompose.variants as variants_module
    from audiocompose import BundleVariant, BundleVariantsResult

    root = tmp_path / "story.ssmd-bundle"
    plans = tmp_path / "plans"
    voices = tmp_path / "voices"
    outputs = tmp_path / "outputs"
    variant = BundleVariant(
        name="narration--abcdef123456",
        profile_id="sha256:" + "a" * 64,
        plan_status="valid",
        render_status="complete",
        compose_status="not_composed",
        state="ready",
        unit_kind="chapters",
        total_units=3,
        rendered_units=3,
        composed_units=0,
        planned_speech=12,
        available_speech=12,
        ready_to_compose=True,
    )
    result = BundleVariantsResult(root, "book", plans, voices, outputs, (variant,))
    captured_args = {}

    def fake_list(bundle, **kwargs):
        captured_args.update(bundle=bundle, **kwargs)
        return result

    monkeypatch.setattr(variants_module, "list_bundle_variants", fake_list)
    assert (
        main(
            [
                "bundle",
                "variants",
                str(root),
                "--renderplan-root",
                str(plans),
                "--voicerender-root",
                str(voices),
                "--output-root",
                str(outputs),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "VARIANT" in output
    assert "narration--abcdef123456" in output
    assert "complete" in output and "ready" in output
    assert "render 3/3 chapters" in output
    assert captured_args == {
        "bundle": str(root),
        "renderplan_root": str(plans),
        "voicerender_root": str(voices),
        "output_root": str(outputs),
    }


def test_bundle_variants_json_root_error_is_deterministic_on_stdout(tmp_path: Path, capsys) -> None:
    invalid = tmp_path / "not-a-bundle"
    invalid.mkdir()

    assert main(["bundle", "variants", str(invalid), "--json"]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["format"] == "audiocompose-bundle-variants"
    assert payload["version"] == 1
    assert payload["error_code"] == "bundle.invalid"
    assert payload["exit_code"] == 2
    assert "error" in payload
    assert captured.err == ""


def test_bundle_variants_invalid_candidate_does_not_fail_discovery(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "broken-candidate.ssmd-bundle"
    (bundle / "renderplan" / "broken--123456789012").mkdir(parents=True)
    (bundle / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}), encoding="utf-8"
    )

    assert main(["bundle", "variants", str(bundle), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["variants"][0]["plan_status"] == "invalid"
    assert payload["variants"][0]["state"] == "invalid"
