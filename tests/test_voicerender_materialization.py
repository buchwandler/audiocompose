"""Compatibility tests for released UtterPlan/RenderPlan/VoiceRender packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from audiocompose.errors import HandoffValidationError, IntegrationError, PausePolicyError
from audiocompose.integrations import voicerender as voicerender_integration
from audiocompose.integrations.renderplan_metadata import (
    RENDER_DIAGNOSTICS_FILENAME,
    VOICERENDER_DEFERRED_CAPABILITIES,
    VOICERENDER_DIAGNOSTICS_SCHEMA,
    VOICERENDER_DIAGNOSTICS_SHA256,
    VOICERENDER_EMPHASIS_POLICY,
)
from audiocompose.integrations.voicerender import (
    EMPHASIS_COMPOSITION_CAPABILITY,
    materialize_voicerender_job,
)
from audiocompose.wav import write_wav

utterplan = pytest.importorskip("utterplan")
renderplan = pytest.importorskip("renderplan")
pytest.importorskip("voicerender")


def _book_handoff(
    tmp_path: Path,
    *,
    source_language: str = "en-US",
    controls=(),
    diagnostics_overrides: dict | None = None,
    producer_name: str = "voicerender",
    producer_version: str | None = "0.2.0",
    manifest_complete: bool = True,
):
    workspace = tmp_path / "book.ssmd-bundle"
    utterplan_dir = workspace / "utterplan"
    chapter_id = "chapter-0001"
    utterplan_dir.mkdir(parents=True, exist_ok=True)

    profile = renderplan.RenderProfile(
        roles={
            "narrator": renderplan.RoleProfile(language="en-US", binding={"renderer": "fixture"})
        },
        default_role="narrator",
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    variant_plan_dir = workspace / "renderplan" / variant
    variant_voice_dir = workspace / "voicerender" / variant
    chapter_render_dir = variant_voice_dir / chapter_id
    variant_plan_dir.mkdir(parents=True, exist_ok=True)
    chapter_render_dir.mkdir(parents=True, exist_ok=True)
    renderplan.save_profile(profile, variant_plan_dir / "renderprofile.toml")

    source = utterplan.FlowPlan(language=source_language, flow=())
    source_path = utterplan_dir / f"{chapter_id}.utterplan.toml"
    source.save(source_path)

    speech = renderplan.SpeechItem(
        number=1,
        text="released API smoke",
        language="en-US",
        role="narrator",
        binding={"renderer": "fixture"},
        controls=tuple(controls),
        source=(renderplan.SourceSpan("chapter-0001.md", 0, 19),),
    ).with_identity()
    plan = renderplan.RenderPlan(
        source_plan_id=source.plan_id,
        render_profile_id=profile_id,
        flow=(speech,),
    ).with_identity()
    plan_path = variant_plan_dir / f"{chapter_id}.renderplan.toml"
    renderplan.save_plan(plan, plan_path)

    wav_path = chapter_render_dir / speech.wav
    write_wav(wav_path, np.zeros(64, dtype=np.float32), 16_000)
    artifact = renderplan.SpeechArtifact(
        segment_number=speech.number,
        segment_id=speech.segment_id or speech.computed_id(),
        wav=speech.wav,
        sha256="sha256:" + hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        frames=64,
        sample_rate=16_000,
        channels=1,
    )
    manifest = renderplan.RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=(artifact,),
        complete=manifest_complete,
        producer=renderplan.Producer(name=producer_name, version=producer_version),
    ).with_identity()
    manifest_path = chapter_render_dir / "render-manifest.toml"
    renderplan.save_manifest(manifest, manifest_path)
    emphasis_count = sum(control.name == "emphasis" for control in speech.controls)
    diagnostics = {
        "schema": "voicerender.render-diagnostics.v1",
        "render_plan_id": plan.plan_id or plan.computed_id(),
        "emphasis_policy": "defer-to-compose" if emphasis_count else "reject",
        "pending_composition": bool(emphasis_count),
        "deferred_controls": (
            [
                {
                    "segment_number": speech.number,
                    "segment_id": speech.segment_id or speech.computed_id(),
                    "capability": EMPHASIS_COMPOSITION_CAPABILITY,
                    "count": emphasis_count,
                }
            ]
            if emphasis_count
            else []
        ),
        "warnings": [],
    }
    if diagnostics_overrides:
        diagnostics.update(diagnostics_overrides)
    (chapter_render_dir / "render-diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )

    chapter = renderplan.RenderPlanSetChapter(
        id=chapter_id,
        source_number=1,
        source=f"../../chapters/{chapter_id}.md",
        utterplan=f"../../utterplan/{chapter_id}.utterplan.toml",
        source_plan_id=source.plan_id,
        renderplan=plan_path.name,
        render_plan_id=plan.plan_id or plan.computed_id(),
    )
    renderplan_set = renderplan.RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="0.2.0",
        manifest="../../book.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=(chapter,),
    )
    set_path = variant_plan_dir / "renderplan-set.toml"
    renderplan.save_renderplan_set(set_path, renderplan_set)
    return workspace, set_path, source, plan, manifest, chapter_render_dir


def test_released_public_book_handoff_materializes_audiojob(tmp_path):
    _, set_path, source, plan, manifest, render_dir = _book_handoff(tmp_path)
    renderplan_set = renderplan.load_renderplan_set(set_path)
    chapter = renderplan_set.chapters[0]
    source_path = (set_path.parent / chapter.utterplan).resolve()
    render_plan_path = set_path.parent / chapter.renderplan
    manifest_path = render_dir / "render-manifest.toml"

    job = materialize_voicerender_job(
        source_path,
        render_plan_path,
        manifest_path,
        render_dir=render_dir,
    )

    assert job.source["renderplan.source_plan_id"] == source.plan_id
    assert job.source["renderplan.plan_id"] == (plan.plan_id or plan.computed_id())
    assert job.source["render_manifest.manifest_id"] == manifest.manifest_id
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    assert job.source[VOICERENDER_DIAGNOSTICS_SCHEMA] == "voicerender.render-diagnostics.v1"
    assert (
        job.source[VOICERENDER_DIAGNOSTICS_SHA256]
        == "sha256:" + hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
    )
    assert job.source[VOICERENDER_EMPHASIS_POLICY] == "reject"
    assert job.source[VOICERENDER_DEFERRED_CAPABILITIES] == []
    assert len(job.items) == 1
    assert job.items[0].operations == ()
    job.validate(verify_sources=True)


def test_diagnostics_digest_participates_in_audiojob_identity(tmp_path):
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path)
    first = materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    diagnostics_path.write_text(
        diagnostics_path.read_text(encoding="utf-8") + "  ", encoding="utf-8"
    )
    second = materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)

    assert (
        first.source[VOICERENDER_DIAGNOSTICS_SHA256]
        != second.source[VOICERENDER_DIAGNOSTICS_SHA256]
    )
    first_bundle = tmp_path / "first.audiojob"
    second_bundle = tmp_path / "second.audiojob"
    first.save(first_bundle)
    second.save(second_bundle)
    first_id = json.loads((first_bundle / "audiojob.json").read_text(encoding="utf-8"))["job_id"]
    second_id = json.loads((second_bundle / "audiojob.json").read_text(encoding="utf-8"))["job_id"]
    assert first_id != second_id


def test_source_plan_identity_mismatch_fails_before_materialization(tmp_path):
    _, set_path, _, plan, manifest, render_dir = _book_handoff(tmp_path)
    wrong = utterplan.FlowPlan(language="fr-FR", flow=())
    wrong_path = tmp_path / "wrong.utterplan.toml"
    wrong.save(wrong_path)

    assert wrong.plan_id != plan.source_plan_id
    with pytest.raises(HandoffValidationError, match="source identity"):
        materialize_voicerender_job(
            wrong_path,
            set_path.parent / renderplan.load_renderplan_set(set_path).chapters[0].renderplan,
            render_dir / "render-manifest.toml",
            render_dir=render_dir,
        )


def test_real_package_root_contracts_are_used():
    assert callable(utterplan.FlowPlan.load)
    assert callable(renderplan.load_plan)
    assert callable(renderplan.load_manifest)
    assert callable(renderplan.validate_manifest_handoff)


def test_pause_policy_profile_derivation_and_strict_explicit_loader(tmp_path):
    from audiocompose.integrations.voicerender import PausePolicy, load_pause_policy

    profile = renderplan.RenderProfile(
        roles={
            "narrator": renderplan.RoleProfile(language="en-US", binding={"renderer": "fixture"})
        },
        default_role="narrator",
        pauses=renderplan.PausePolicy(mode="manual", durations_ms={"strong": 450}),
    )
    profile_policy = PausePolicy.from_render_profile(profile)
    assert profile_policy.resolve("strong", role="narrator", where="test") == pytest.approx(0.45)

    explicit = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {"strong": "410ms"},
        "role": {"narrator": {"strong": "425ms"}},
    }
    path = tmp_path / "pauses.toml"
    path.write_text(
        'schema = "audiocompose.pause-policy.v1"\nmerge = "max"\n'
        '[default]\nstrong = "410ms"\n[role.narrator]\nstrong = "425ms"\n',
        encoding="utf-8",
    )
    loaded = load_pause_policy(path)
    assert loaded.policy_id == PausePolicy.from_mapping(explicit).policy_id
    assert loaded.resolve("strong", role="narrator", where="test") == pytest.approx(0.425)
    assert loaded.resolve(
        utterplan.PauseIntent("timed", "25ms"), role="narrator", where="test"
    ) == pytest.approx(0.025)

    with pytest.raises(PausePolicyError, match="unknown pause class"):
        PausePolicy.from_mapping(
            {"schema": "audiocompose.pause-policy.v1", "merge": "max", "default": {"unknown": "1s"}}
        )
    with pytest.raises(PausePolicyError, match="no duration configured"):
        PausePolicy.from_mapping(
            {"schema": "audiocompose.pause-policy.v1", "merge": "max", "default": {}}
        ).resolve("strong", role="narrator", where="test")

    symlink = tmp_path / "pauses-symlink.toml"
    symlink.symlink_to(path)
    with pytest.raises(PausePolicyError, match="must not be a symlink"):
        PausePolicy.load(symlink)


def test_voicerender_v020_handoff_accepts_native_controls_and_applies_deferred_emphasis(tmp_path):
    from audiocompose import Gain

    controls = (
        renderplan.SpeechControl(0, 18, "emphasis", {"level": "moderate"}),
        renderplan.SpeechControl(0, 18, "pronunciation", {"mode": "fixture"}),
        renderplan.SpeechControl(0, 18, "prosody", {"rate": "slow"}),
    )
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path, controls=controls)

    job = materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)

    clip = job.items[0]
    assert clip.operations == (Gain(1.5),)
    assert clip.metadata["renderplan.emphasis"]["selected_level"] == "moderate"
    assert clip.metadata["renderplan.upstream_realized_controls"] == [
        {"name": "pronunciation", "start": 0, "end": 18},
        {"name": "prosody", "start": 0, "end": 18},
    ]
    assert job.source[VOICERENDER_DIAGNOSTICS_SCHEMA] == "voicerender.render-diagnostics.v1"
    assert job.source[VOICERENDER_EMPHASIS_POLICY] == "defer-to-compose"
    assert job.source[VOICERENDER_DEFERRED_CAPABILITIES] == [EMPHASIS_COMPOSITION_CAPABILITY]
    job.validate(verify_sources=True)


def test_voicerender_handoff_rejects_missing_diagnostics(tmp_path):
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path)
    (render_dir / "render-diagnostics.json").unlink()

    with pytest.raises(HandoffValidationError, match="Cannot read VoiceRender diagnostics"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


@pytest.mark.parametrize(
    ("diagnostics_overrides", "message"),
    [
        (
            {"schema": "voicerender.render-diagnostics.v2"},
            "Unsupported VoiceRender diagnostics schema",
        ),
        ({"render_plan_id": "sha256:" + "0" * 64}, "diagnostics are stale"),
        ({"pending_composition": True}, "pending_composition contradicts"),
        ({"emphasis_policy": []}, "Unsupported VoiceRender emphasis_policy"),
    ],
)
def test_voicerender_handoff_rejects_invalid_diagnostics(tmp_path, diagnostics_overrides, message):
    _, _, source, plan, manifest, render_dir = _book_handoff(
        tmp_path, diagnostics_overrides=diagnostics_overrides
    )
    with pytest.raises(HandoffValidationError, match=message):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_voicerender_handoff_rejects_unknown_deferred_capability(tmp_path):
    controls = (renderplan.SpeechControl(0, 18, "emphasis", {}),)
    overrides = {
        "deferred_controls": [
            {
                "segment_number": 1,
                "segment_id": "wrong-occurrence",
                "capability": "audiocompose.future-control.v1",
                "count": 1,
            }
        ]
    }
    _, _, source, plan, manifest, render_dir = _book_handoff(
        tmp_path, controls=controls, diagnostics_overrides=overrides
    )
    with pytest.raises(HandoffValidationError, match="not supported by AudioCompose"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_voicerender_handoff_rejects_missing_deferred_emphasis_occurrence(tmp_path):
    controls = (renderplan.SpeechControl(0, 18, "emphasis", {}),)
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path, controls=controls)
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["deferred_controls"] = []
    diagnostics["pending_composition"] = False
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(HandoffValidationError, match="omit deferred emphasis occurrence"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_voicerender_handoff_rejects_deferred_emphasis_count_mismatch(tmp_path):
    controls = (renderplan.SpeechControl(0, 18, "emphasis", {}),)
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path, controls=controls)
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["deferred_controls"][0]["count"] = 2
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(HandoffValidationError, match="reported 2 deferred emphasis controls"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_voicerender_handoff_rejects_wrong_deferred_segment_id(tmp_path):
    controls = (renderplan.SpeechControl(0, 18, "emphasis", {}),)
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path, controls=controls)
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["deferred_controls"][0]["segment_id"] = "sha256:" + "0" * 64
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    with pytest.raises(HandoffValidationError, match="unknown speech occurrence"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


@pytest.mark.parametrize(
    ("producer_name", "producer_version", "message"),
    [
        ("another-producer", "0.2.0", "produced by 'voicerender'"),
        ("voicerender", "0.1.1", "incompatible; AudioCompose v0.3 supports 0.2.x"),
    ],
)
def test_voicerender_handoff_rejects_incompatible_manifest_producer(
    tmp_path, producer_name, producer_version, message
):
    _, _, source, plan, manifest, render_dir = _book_handoff(
        tmp_path, producer_name=producer_name, producer_version=producer_version
    )
    with pytest.raises(HandoffValidationError, match=message):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_voicerender_handoff_rejects_incomplete_manifest(tmp_path):
    _, _, source, plan, manifest, render_dir = _book_handoff(tmp_path, manifest_complete=False)
    with pytest.raises(HandoffValidationError, match="must be complete"):
        materialize_voicerender_job(source, plan, manifest, render_dir=render_dir)


def test_missing_utterplan_error_names_the_requirement(monkeypatch):
    original_import = voicerender_integration.importlib.import_module

    def import_without_utterplan(name, package=None):
        if name == "utterplan":
            raise ImportError("simulated missing dependency")
        return original_import(name, package=package)

    monkeypatch.setattr(
        voicerender_integration.importlib, "import_module", import_without_utterplan
    )
    with pytest.raises(
        IntegrationError,
        match="UtterPlan is required for VoiceRender book handoff materialization",
    ):
        voicerender_integration._flow_plan_type()
