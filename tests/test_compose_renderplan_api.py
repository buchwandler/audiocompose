from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from renderplan import (
    AudioItem,
    RenderManifest,
    RenderPlan,
    SourceSpan,
    SpeechArtifact,
    SpeechControl,
    SpeechItem,
    save_manifest,
    save_plan,
)

import audiocompose
from audiocompose import (
    AudioJob,
    CompositionDiagnostic,
    DiagnosticSeverity,
    LoudnessPolicy,
    OutputPolicy,
)
from audiocompose.cli import main
from audiocompose.errors import HandoffValidationError
from audiocompose.wav import read_wav, wav_info, write_wav


def _id(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _speech(*, text: str = "Hello", controls=()) -> SpeechItem:
    return SpeechItem(
        number=1,
        text=text,
        language="en-US",
        role="narrator",
        binding={"engine": "fixture"},
        controls=tuple(controls),
        source=(SourceSpan("source.txt", 12, 12 + len(text)),),
    ).with_identity()


def _plan(*flow) -> RenderPlan:
    return RenderPlan(
        source_plan_id=_id("source"),
        render_profile_id=_id("profile"),
        flow=tuple(flow),
    ).with_identity()


def _manifest(plan: RenderPlan, artifacts=()) -> RenderManifest:
    return RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=tuple(artifacts),
        complete=True,
    ).with_identity()


def _rendered(render_dir: Path, speech: SpeechItem) -> SpeechArtifact:
    render_dir.mkdir(parents=True, exist_ok=True)
    path = render_dir / speech.wav
    write_wav(path, np.full(32, 0.125, dtype=np.float32), 16_000)
    return SpeechArtifact(
        segment_number=speech.number,
        segment_id=speech.segment_id or speech.computed_id(),
        wav=speech.wav,
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=32,
        sample_rate=16_000,
        channels=1,
    )


def _save_handoff(
    root: Path,
    plan: RenderPlan,
    manifest: RenderManifest,
    *,
    render_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    actual_render_dir = render_dir or root / "renders"
    actual_render_dir.mkdir(parents=True, exist_ok=True)
    plan_path = root / "chapter.renderplan.toml"
    actual_manifest_path = manifest_path or actual_render_dir / "render-manifest.toml"
    if actual_manifest_path.parent != actual_render_dir:
        actual_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_plan(plan, plan_path)
    save_manifest(manifest, actual_manifest_path)
    return plan_path, actual_render_dir


def test_public_api_composes_with_implicit_manifest_and_returns_typed_result(tmp_path):
    render_dir = tmp_path / "renders"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)
    output = tmp_path / "chapter.wav"

    result = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=output,
        output_policy=OutputPolicy(sample_rate=16_000),
    )

    assert isinstance(result, audiocompose.RenderPlanComposeResult)
    assert result.output == output
    assert result.plan_id == plan.plan_id
    assert result.manifest_id == _manifest(plan, [artifact]).manifest_id
    assert result.sample_rate == 16_000
    assert result.frames == 32
    assert result.duration_seconds == pytest.approx(32 / 16_000)
    assert result.timeline[0]["renderplan.segment_id"] == speech.segment_id
    assert result.diagnostics == ()
    assert result.audiojob is None
    assert wav_info(output).frames == result.frames


def test_public_api_accepts_explicit_manifest_path(tmp_path):
    render_dir = tmp_path / "renders"
    explicit_manifest = tmp_path / "handoff" / "final.toml"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(
        tmp_path,
        plan,
        _manifest(plan, [artifact]),
        render_dir=render_dir,
        manifest_path=explicit_manifest,
    )

    result = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=tmp_path / "explicit.wav",
        manifest=explicit_manifest,
    )

    assert result.manifest_id == _manifest(plan, [artifact]).manifest_id
    assert result.frames > 0


def test_public_api_rejects_missing_render_wav_and_wrong_plan_identity(tmp_path):
    render_dir = tmp_path / "renders"
    speech = _speech()
    plan = _plan(speech)
    artifact = SpeechArtifact(
        segment_number=speech.number,
        segment_id=speech.segment_id or speech.computed_id(),
        wav=speech.wav,
        sha256=_id("missing"),
        frames=32,
        sample_rate=16_000,
        channels=1,
    )
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)
    with pytest.raises(HandoffValidationError, match="WAV is unavailable"):
        audiocompose.compose_renderplan(
            plan_path, render_dir=render_dir, output=tmp_path / "missing.wav"
        )

    _rendered(render_dir, speech)
    wrong_manifest = _manifest(_plan(_speech(text="Other")), [artifact])
    wrong_path = tmp_path / "wrong-manifest.toml"
    save_manifest(wrong_manifest, wrong_path)
    with pytest.raises(HandoffValidationError, match="does not match RenderPlan"):
        audiocompose.compose_renderplan(
            plan_path,
            render_dir=render_dir,
            output=tmp_path / "wrong.wav",
            manifest=wrong_path,
        )
    assert not (tmp_path / "wrong.wav").exists()


def test_default_media_root_is_plan_parent(tmp_path):
    plan_root = tmp_path / "document"
    render_dir = plan_root / "renders"
    render_dir.mkdir(parents=True)
    write_wav(plan_root / "music.wav", np.full(32, 0.2, dtype=np.float32), 16_000)
    plan = _plan(
        AudioItem(src="music.wav", source=(SourceSpan("source.txt", 0, 5),)).with_identity()
    )
    plan_path, _ = _save_handoff(plan_root, plan, _manifest(plan), render_dir=render_dir)

    result = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=plan_root / "default-root.wav",
        output_policy=OutputPolicy(sample_rate=16_000),
    )

    assert result.frames == 32
    assert np.max(np.abs(read_wav(result.output)[0])) > 0


def test_explicit_media_root_and_output_policy_are_forwarded(tmp_path):
    plan_root = tmp_path / "document"
    media_root = tmp_path / "assets"
    render_dir = plan_root / "renders"
    media_root.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    write_wav(media_root / "music.wav", np.full(32, 0.2, dtype=np.float32), 16_000)
    plan = _plan(
        AudioItem(src="music.wav", source=(SourceSpan("source.txt", 0, 5),)).with_identity()
    )
    plan_path, _ = _save_handoff(plan_root, plan, _manifest(plan), render_dir=render_dir)
    policy = OutputPolicy(
        sample_rate=8_000,
        clip_policy="error",
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
    )

    result = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=plan_root / "explicit-root.wav",
        media_root=media_root,
        output_policy=policy,
    )

    assert result.sample_rate == 8_000
    assert result.frames == 16
    assert wav_info(result.output).sample_rate == 8_000


def test_emphasis_fallback_returns_diagnostic_and_strict_fails_without_output(tmp_path):
    render_dir = tmp_path / "renders"
    speech = _speech(controls=(SpeechControl(1, 3, "emphasis", {}),))
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)

    fallback = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=tmp_path / "fallback.wav",
    )
    assert len(fallback.diagnostics) == 1
    diagnostic = fallback.diagnostics[0]
    assert isinstance(diagnostic, CompositionDiagnostic)
    assert diagnostic.code == "emphasis.segment_fallback"
    assert diagnostic.severity is DiagnosticSeverity.WARNING
    assert diagnostic.context["segment_number"] == 1
    assert diagnostic.context["character_ranges"] == [[1, 3]]
    assert fallback.timeline[0]["renderplan.emphasis"]["scope"] == "segment_fallback"

    strict_output = tmp_path / "strict.wav"
    with pytest.raises(HandoffValidationError, match="split the speech upstream"):
        audiocompose.compose_renderplan(
            plan_path,
            render_dir=render_dir,
            output=strict_output,
            emphasis_policy="strict",
        )
    assert not strict_output.exists()


def test_save_audiojob_is_optional_and_persists_materialized_job(tmp_path):
    render_dir = tmp_path / "renders"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)

    no_save = audiocompose.compose_renderplan(
        plan_path, render_dir=render_dir, output=tmp_path / "no-save.wav"
    )
    assert no_save.audiojob is None
    assert not (tmp_path / "unexpected.audiojob").exists()

    saved_path = tmp_path / "job.audiojob"
    saved = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=tmp_path / "saved.wav",
        save_audiojob=saved_path,
    )
    assert saved.audiojob == saved_path / "audiojob.json"
    restored = AudioJob.load(saved_path, verify_sources=True)
    assert restored.source["renderplan.plan_id"] == plan.plan_id


def test_api_matches_cli_json_essentials(tmp_path, capsys):
    render_dir = tmp_path / "renders"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)
    api_result = audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=tmp_path / "api.wav",
        output_policy=OutputPolicy(sample_rate=16_000),
    )

    cli_output = tmp_path / "cli.wav"
    assert (
        main(
            [
                "compose-renderplan",
                str(plan_path),
                "--render-dir",
                str(render_dir),
                "--output",
                str(cli_output),
                "--sample-rate",
                "16000",
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["composed"] is True
    assert response["output"] == str(cli_output)
    assert response["plan_id"] == api_result.plan_id
    assert response["manifest_id"] == api_result.manifest_id
    assert response["sample_rate"] == api_result.sample_rate
    assert response["frames"] == api_result.frames
    assert response["duration_seconds"] == api_result.duration_seconds
    assert response["timeline"] == [dict(item) for item in api_result.timeline]
    assert response["emphasis_policy"] == "segment_fallback"
    assert response["diagnostics"] == []


def test_package_root_exports_api_and_request_object_is_supported(tmp_path):
    assert audiocompose.RenderPlanComposeRequest.__module__ == "audiocompose.renderplan"
    assert "compose_renderplan" in audiocompose.__all__
    assert "RenderPlanComposeRequest" in audiocompose.__all__
    assert "RenderPlanComposeResult" in audiocompose.__all__

    render_dir = tmp_path / "renders"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)
    request = audiocompose.RenderPlanComposeRequest(
        plan=plan_path,
        render_dir=render_dir,
        output=tmp_path / "request.wav",
        output_policy=OutputPolicy(sample_rate=16_000),
    )
    result = audiocompose.compose_renderplan(request)
    assert result.output == request.output
    assert result.frames == 32


def test_cli_handler_calls_public_compose_function(monkeypatch, tmp_path, capsys):
    diagnostic = CompositionDiagnostic(
        code="emphasis.segment_fallback",
        message="fallback applied",
        severity=DiagnosticSeverity.WARNING,
        context={"segment_number": 2, "character_ranges": [[1, 3]], "gain_db": 1.5},
    )
    api_result = audiocompose.RenderPlanComposeResult(
        output=tmp_path / "created.wav",
        plan_id=_id("plan"),
        manifest_id=_id("manifest"),
        sample_rate=24_000,
        frames=10,
        duration_seconds=10 / 24_000,
        timeline=({"id": "speech:000001", "kind": "speech"},),
        diagnostics=(diagnostic,),
    )
    calls = []

    def fake_compose(plan, **kwargs):
        calls.append((plan, kwargs))
        return api_result

    monkeypatch.setattr(audiocompose, "compose_renderplan", fake_compose)
    assert (
        main(
            [
                "compose-renderplan",
                "chapter.renderplan.toml",
                "--render-dir",
                str(tmp_path / "renders"),
                "--output",
                str(tmp_path / "cli.wav"),
                "--json",
            ]
        )
        == 0
    )

    response = json.loads(capsys.readouterr().out)
    assert len(calls) == 1
    assert calls[0][0] == "chapter.renderplan.toml"
    assert calls[0][1]["render_dir"] == str(tmp_path / "renders")
    assert calls[0][1]["output"] == str(tmp_path / "cli.wav")
    assert isinstance(calls[0][1]["output_policy"], OutputPolicy)
    assert response["plan_id"] == api_result.plan_id
    assert response["diagnostics"] == [
        {
            "code": "emphasis.segment_fallback",
            "severity": "warning",
            "segment_number": 2,
            "character_ranges": [[1, 3]],
            "gain_db": 1.5,
            "message": "fallback applied",
        }
    ]


def test_public_api_forwards_core_progress_and_preserves_output(tmp_path) -> None:
    render_dir = tmp_path / "renders"
    speech = _speech()
    artifact = _rendered(render_dir, speech)
    plan = _plan(speech)
    plan_path, _ = _save_handoff(tmp_path, plan, _manifest(plan, [artifact]), render_dir=render_dir)
    observed_output = tmp_path / "observed.wav"
    plain_output = tmp_path / "plain.wav"
    events = []

    audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=observed_output,
        output_policy=OutputPolicy(sample_rate=16_000),
        on_progress=events.append,
    )
    audiocompose.compose_renderplan(
        plan_path,
        render_dir=render_dir,
        output=plain_output,
        output_policy=OutputPolicy(sample_rate=16_000),
    )

    assert events[0].kind == "compose_started"
    assert events[-1].kind == "compose_completed"
    observed_audio, observed_rate = read_wav(observed_output)
    plain_audio, plain_rate = read_wav(plain_output)
    assert observed_rate == plain_rate
    assert np.array_equal(observed_audio, plain_audio)

    request_output = tmp_path / "request-failed.wav"
    request = audiocompose.RenderPlanComposeRequest(
        plan=plan_path,
        render_dir=render_dir,
        output=request_output,
        output_policy=OutputPolicy(sample_rate=16_000),
    )

    def fail(event):
        raise RuntimeError(event.kind)

    with pytest.raises(RuntimeError, match="compose_started"):
        audiocompose.compose_renderplan(request, on_progress=fail)
    assert not request_output.exists()
