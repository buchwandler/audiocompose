from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
from audiocompose.progress import ProgressEventKind
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
    kind: ProgressEventKind,
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


def test_progress_renderer_keeps_primary_fields_on_narrow_tty(monkeypatch) -> None:
    import audiocompose._cli_progress as cli_progress

    monkeypatch.setattr(
        cli_progress.shutil,
        "get_terminal_size",
        lambda *, fallback: os.terminal_size((50, 24)),
    )
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", stream=stream, clock=FakeClock())
    renderer.set_context("1/1 eval2--pt-engaging-v1--b8a526d23fd")
    renderer.on_composition_progress(_composition_event("compose_started", 30.0))

    frame = stream.getvalue().rsplit("\r", 1)[-1].strip()
    assert "[1/1]" in frame
    assert "30.0%" in frame
    assert "composing" in frame
    assert "00:00" in frame


def test_book_progress_keeps_overall_fields_on_narrow_tty(monkeypatch) -> None:
    import audiocompose._cli_progress as cli_progress

    monkeypatch.setattr(
        cli_progress.shutil,
        "get_terminal_size",
        lambda *, fallback: os.terminal_size((60, 24)),
    )
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", command="book", stream=stream, clock=FakeClock())
    renderer.set_context("1/1 eval2--pt-engaging-v1--b8a526d23fd")
    renderer.on_book_progress(
        BookComposeProgress(
            0,
            2,
            kind="chapter_compose_progress",
            chapter_id="chapter-0001",
            composition=_composition_event("item_completed", 42.6),
            fraction=0.213,
        )
    )

    frame = stream.getvalue().rsplit("\r", 1)[-1].strip()
    assert "[1/1]" in frame
    assert "21.3%" in frame
    assert "composing" in frame
    assert "00:00" in frame


def test_progress_renderer_preserves_variant_identity_on_wide_tty(monkeypatch) -> None:
    import audiocompose._cli_progress as cli_progress

    monkeypatch.setattr(
        cli_progress.shutil,
        "get_terminal_size",
        lambda *, fallback: os.terminal_size((160, 24)),
    )
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", stream=stream, clock=FakeClock())
    renderer.set_context("1/1 eval2--pt-engaging-v1--b8a526d23fd")
    renderer.on_composition_progress(_composition_event("compose_started", 30.0))

    frame = stream.getvalue().rsplit("\r", 1)[-1].strip()
    assert "[1/1]" in frame
    assert "30.0%" in frame
    assert "composing" in frame
    assert "eval2--pt-engaging-v1--b8a526d23fd" in frame


def test_progress_heartbeat_refreshes_elapsed_without_advancing_work() -> None:
    clock = FakeClock()
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", stream=stream, clock=clock, heartbeat_interval=None)
    renderer.on_composition_progress(_composition_event("compose_started", 25.0))
    timing = renderer._composition_timing
    estimator = renderer._estimator

    clock.advance(65.0)
    renderer._heartbeat_redraw()
    frame = stream.getvalue().rsplit("\r", 1)[-1]
    assert "25.0%" in frame
    assert "01:05" in frame
    assert renderer._composition is not None
    assert renderer._composition.fraction == 0.25
    assert renderer._composition_timing is timing
    assert renderer._estimator is estimator

    renderer.finish(success=True)
    finished_output = stream.getvalue()
    clock.advance(10.0)
    renderer._heartbeat_redraw()
    assert stream.getvalue() == finished_output
    assert renderer._heartbeat_stop.is_set()


def test_progress_heartbeat_thread_stops_on_finish() -> None:
    renderer = CliProgressRenderer(mode="auto", stream=TTYStream(), clock=FakeClock())
    renderer.on_composition_progress(_composition_event("compose_started", 0.0))
    thread = renderer._heartbeat_thread
    assert thread is not None and thread.is_alive()

    renderer.finish(success=True)

    assert renderer._heartbeat_stop.is_set()
    assert not thread.is_alive()


def test_progress_heartbeat_is_disabled_for_non_tty() -> None:
    clock = FakeClock()
    stream = io.StringIO()
    renderer = CliProgressRenderer(
        mode="always", stream=stream, clock=clock, heartbeat_interval=0.01
    )
    renderer.on_composition_progress(_composition_event("compose_started", 10.0))
    output = stream.getvalue()

    clock.advance(10.0)
    renderer._heartbeat_redraw()

    assert stream.getvalue() == output
    assert renderer._heartbeat_thread is None


def test_progress_renderer_shows_eta_when_narrow_terminal_has_room(monkeypatch) -> None:
    import audiocompose._cli_progress as cli_progress

    monkeypatch.setattr(
        cli_progress.shutil,
        "get_terminal_size",
        lambda *, fallback: os.terminal_size((60, 24)),
    )
    clock = FakeClock()
    stream = TTYStream()
    renderer = CliProgressRenderer(mode="auto", stream=stream, clock=clock, heartbeat_interval=None)
    renderer.set_context("1/1 eval2--pt-engaging-v1--b8a526d23fd")
    renderer.on_composition_progress(_composition_event("compose_started", 0.0))
    clock.advance(2.0)
    renderer.on_composition_progress(_composition_event("item_completed", 20.0, completed_items=1))

    frame = stream.getvalue().rsplit("\r", 1)[-1]
    assert "[1/1]" in frame
    assert "20.0%" in frame
    assert "ETA 00:08" in frame


def test_progress_renderer_keeps_unknown_work_indeterminate() -> None:
    stream = TTYStream()
    renderer = CliProgressRenderer(
        mode="auto", stream=stream, clock=FakeClock(), heartbeat_interval=None
    )
    renderer.on_composition_progress(
        CompositionProgress(kind="compose_started", completed_items=0, total_items=0)
    )

    frame = stream.getvalue().rsplit("\r", 1)[-1]
    assert "composing" in frame
    assert "00:00" in frame
    assert "%" not in frame


def test_book_progress_retains_chapter_context_when_wide(monkeypatch) -> None:
    import audiocompose._cli_progress as cli_progress

    monkeypatch.setattr(
        cli_progress.shutil,
        "get_terminal_size",
        lambda *, fallback: os.terminal_size((160, 24)),
    )
    stream = TTYStream()
    renderer = CliProgressRenderer(
        mode="auto", command="book", stream=stream, clock=FakeClock(), heartbeat_interval=None
    )
    renderer.set_context("1/1 eval2--pt-engaging-v1--b8a526d23fd")
    renderer.on_book_progress(
        BookComposeProgress(
            0,
            10,
            kind="chapter_compose_progress",
            chapter_id="chapter-0001",
            composition=_composition_event("item_completed", 73.8, completed_items=684),
            fraction=0.374,
        )
    )

    frame = stream.getvalue().rsplit("\r", 1)[-1]
    assert "[1/1]" in frame
    assert "37.4%" in frame
    assert "ch 1/10" in frame
    assert "ch 73.8%" in frame
    assert "eval2--pt-engaging-v1--b8a526d23fd" in frame


def test_variant_label_middle_elision_preserves_prefix_and_fingerprint() -> None:
    label = "eval2--pt-engaging-v1--b8a526d23fd"

    elided = CliProgressRenderer._elide_variant(label, 28)

    assert elided is not None
    assert len(elided) == 28
    assert elided.startswith("eval2--pt-eng")
    assert elided.endswith("b8a526d23fd")


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
        render_warning="VoiceRender summary report ignored after chapter validation",
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
    assert "render warning: VoiceRender summary report ignored" in output
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


def test_bundle_compose_optional_variant_routes_batch_and_preserves_single_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from types import SimpleNamespace

    import audiocompose.bundle as bundle_module
    import audiocompose.bundle_batch as batch_module
    from audiocompose.bundle import BundleComposeResult
    from audiocompose.bundle_batch import BundleBatchComposeResult, BundleBatchComposeSummary

    root = tmp_path / "story.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}), encoding="utf-8"
    )
    summary = BundleBatchComposeSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    batch_result = BundleBatchComposeResult(
        bundle=root,
        kind="book",
        renderplan_root=root / "renderplan",
        voicerender_root=root / "voicerender",
        output_root=root / "audiocompose",
        force=False,
        variants=(),
        summary=summary,
        exit_code=0,
    )
    batch = Mock(return_value=batch_result)
    single_result = BundleComposeResult(
        kind="book",
        value=SimpleNamespace(exit_code=0, to_dict=lambda: {"single": True}),
    )
    single = Mock(return_value=single_result)
    monkeypatch.setattr(batch_module, "compose_bundle_batch", batch)
    monkeypatch.setattr(bundle_module, "compose_bundle", single)

    assert main(["bundle", "compose", str(root), "--json", "--progress", "always"]) == 0
    batch_output = capsys.readouterr()
    assert json.loads(batch_output.out)["format"] == "audiocompose-bundle-batch-result"
    assert batch_output.out.count("{") >= 1
    assert batch_output.err
    batch.assert_called_once()
    single.assert_not_called()
    assert batch.call_args.kwargs["on_progress"] is not None

    assert (
        main(
            ["bundle", "compose", str(root), "--variant", "target", "--json", "--progress", "never"]
        )
        == 0
    )
    single_output = capsys.readouterr()
    assert json.loads(single_output.out)["format"] == "audiocompose-bundle-result"
    assert single.call_args.kwargs["variant"] == "target"
    batch.assert_called_once()


def test_bundle_compose_batch_root_error_is_json_on_stdout(tmp_path: Path, capsys) -> None:
    invalid = tmp_path / "not-a-bundle"
    invalid.mkdir()

    assert main(["bundle", "compose", str(invalid), "--json"]) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["format"] == "audiocompose-bundle-batch-result"
    assert payload["error_code"] == "bundle.invalid"
    assert payload["exit_code"] == 2
    assert captured.err == ""


def test_batch_cli_reports_invalid_selector_even_when_global_variant_is_partial(
    tmp_path: Path, capsys
) -> None:
    from _bundle_fixtures import two_chapter_book

    workspace, _ = two_chapter_book(tmp_path)

    assert main(["bundle", "compose", str(workspace), "--chapters", "999", "--json"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["error_code"] == "selection.invalid"
    assert "unknown selected chapter number: 999" in payload["error_message"]
    assert payload["summary"]["scheduled"] == 0


def test_bundle_compose_batch_human_no_variant_message(tmp_path: Path, capsys) -> None:
    root = tmp_path / "empty.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}), encoding="utf-8"
    )

    assert main(["bundle", "compose", str(root)]) == 0

    output = capsys.readouterr().out
    assert "mode:          batch" in output
    assert "discovered:" in output
    assert "scheduled:" in output
    assert "No variants found; nothing to do." in output


def test_bundle_compose_batch_human_reports_all_blocker_details(capsys) -> None:
    from types import SimpleNamespace

    from audiocompose.cli import _print_bundle_batch_result

    def outcome(
        name: str,
        status: str,
        *,
        initial_state: str,
        initial_render_status: str,
        initial_compose_status: str,
        ready: bool,
        error_message: str | None = None,
        validation_error: str | None = None,
        render_error: str | None = None,
        compose_error: str | None = None,
        render_warning: str | None = None,
    ):
        return SimpleNamespace(
            name=name,
            status=status,
            initial_state=initial_state,
            initial_render_status=initial_render_status,
            initial_compose_status=initial_compose_status,
            ready_to_compose=ready,
            selected=False,
            attempt_status="failed" if error_message else None,
            final_state=None,
            final_render_status=None,
            final_compose_status=None,
            error_message=error_message,
            validation_error=validation_error,
            render_error=render_error,
            render_warning=render_warning,
            compose_error=compose_error,
            result=None,
        )

    result = SimpleNamespace(
        bundle="story.ssmd-bundle",
        kind="book",
        renderplan_root=None,
        voicerender_root=None,
        output_root=None,
        error_message=None,
        summary=SimpleNamespace(
            discovered=4,
            compose_ready=0,
            scheduled=0,
            attempted=1,
            skipped_complete=0,
            skipped_not_ready=2,
            invalid=1,
            completed=0,
            partial=0,
            failed=1,
            not_attempted=0,
        ),
        variants=(
            outcome(
                "render-blocked",
                "skipped_not_ready",
                initial_state="unknown",
                initial_render_status="unknown",
                initial_compose_status="partial",
                ready=False,
                render_error="bad diagnostics",
                render_warning="optional report was ignored",
            ),
            outcome(
                "compose-blocked",
                "skipped_not_ready",
                initial_state="render_partial",
                initial_render_status="partial",
                initial_compose_status="partial",
                ready=False,
                compose_error="stale composition state",
            ),
            outcome(
                "invalid",
                "invalid",
                initial_state="invalid",
                initial_render_status="complete",
                initial_compose_status="not_composed",
                ready=False,
                validation_error="invalid manifest",
            ),
            outcome(
                "attempt-failed",
                "failed",
                initial_state="ready",
                initial_render_status="complete",
                initial_compose_status="not_composed",
                ready=True,
                error_message="chapter composition failed",
            ),
        ),
    )

    _print_bundle_batch_result(result)
    output = capsys.readouterr().out

    for diagnostic in (
        "render warning: optional report was ignored",
        "render error: bad diagnostics",
        "compose error: stale composition state",
        "validation error: invalid manifest",
        "error: chapter composition failed",
        "render: unknown",
        "compose: partial",
        "ready: no",
        "One or more variants are blocked by render-handoff validation errors",
    ):
        assert diagnostic in output
    assert "No variants are currently ready to compose; nothing to do." not in output


def test_bundle_compose_batch_json_preserves_blocker_diagnostic_fields(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import audiocompose.bundle_batch as batch_module
    from audiocompose.bundle_batch import (
        BundleBatchComposeResult,
        BundleBatchComposeSummary,
        BundleVariantComposeOutcome,
    )

    root = tmp_path / "story.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"format": "ssmdconvert.book", "schema_version": 1}), encoding="utf-8"
    )
    outcome = BundleVariantComposeOutcome(
        name="broken",
        initial_state="unknown",
        initial_render_status="unknown",
        initial_compose_status="partial",
        ready_to_compose=False,
        plan_status="valid",
        selected=False,
        status="skipped_not_ready",
        render_error="bad diagnostics",
        compose_error="stale composition state",
    )
    result = BundleBatchComposeResult(
        bundle=root,
        kind="book",
        renderplan_root=root / "renderplan",
        voicerender_root=root / "voicerender",
        output_root=root / "audiocompose",
        force=False,
        variants=(outcome,),
        summary=BundleBatchComposeSummary(1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        exit_code=1,
    )
    monkeypatch.setattr(batch_module, "compose_bundle_batch", Mock(return_value=result))

    assert main(["bundle", "compose", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["variants"][0]["render_error"] == "bad diagnostics"
    assert payload["variants"][0]["compose_error"] == "stale composition state"
    assert captured.err == ""


def test_batch_progress_adapter_labels_variants_and_resets_estimators() -> None:
    from audiocompose.cli import _BundleProgressAdapter

    clock = FakeClock()
    stream = io.StringIO()
    renderer = CliProgressRenderer(mode="always", stream=stream, clock=clock)
    adapter = _BundleProgressAdapter(renderer, "document")

    adapter.set_context("1/2 profile-a")
    adapter(_composition_event("compose_started", 0.0))
    clock.advance(2.0)
    adapter(_composition_event("item_completed", 20.0, completed_items=1))
    assert renderer._composition_timing is not None

    adapter.set_context("2/2 profile-b")
    assert renderer._composition is None
    assert renderer._composition_timing is None
    assert renderer._phase is None
    adapter(_composition_event("compose_started", 0.0))

    output = stream.getvalue()
    assert "[1/2 profile-a]" in output
    assert "[2/2 profile-b]" in output
    assert "elapsed 00:00" in output


def test_batch_progress_adapter_routes_book_events_to_book_renderer() -> None:
    from audiocompose.cli import _BundleProgressAdapter

    renderer = Mock()
    adapter = _BundleProgressAdapter(renderer, "book")
    event = object()

    adapter.set_context("1/1 variant-a")
    adapter(event)

    renderer.set_context.assert_called_once_with("1/1 variant-a")
    renderer.on_book_progress.assert_called_once_with(event)
    renderer.on_composition_progress.assert_not_called()
