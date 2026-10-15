"""Compatibility tests for released UtterPlan/RenderPlan/VoiceRender packages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from audiocompose.errors import HandoffValidationError, PausePolicyError
from audiocompose.integrations.voicerender import materialize_voicerender_job
from audiocompose.wav import write_wav

utterplan = pytest.importorskip("utterplan")
renderplan = pytest.importorskip("renderplan")
pytest.importorskip("voicerender")


def _book_handoff(tmp_path: Path, *, source_language: str = "en-US"):
    workspace = tmp_path / "book.ssmdbook"
    utterplan_dir = workspace / "utterplan"
    chapter_id = "chapter-0001"
    utterplan_dir.mkdir(parents=True, exist_ok=True)

    profile = renderplan.RenderProfile(
        roles={"narrator": renderplan.RoleProfile(language="en-US", binding={"renderer": "fixture"})},
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
        render_plan_id=plan.plan_id or plan.computed_id(), speech=(artifact,), complete=True
    ).with_identity()
    manifest_path = chapter_render_dir / "render-manifest.toml"
    renderplan.save_manifest(manifest, manifest_path)

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
        producer_version="0.1.1",
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
    assert len(job.items) == 1
    job.validate(verify_sources=True)


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
        roles={"narrator": renderplan.RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
        pauses=renderplan.PausePolicy(mode="manual", durations_ms={"strong": 450}),
    )
    profile_policy = PausePolicy.from_render_profile(profile)
    assert profile_policy.resolve("strong", voice="narrator", where="test") == pytest.approx(0.45)

    explicit = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {"strong": "410ms"},
        "voice": {"narrator": {"strong": "425ms"}},
    }
    path = tmp_path / "pauses.toml"
    path.write_text(
        'schema = "audiocompose.pause-policy.v1"\nmerge = "max"\n'
        '[default]\nstrong = "410ms"\n[voice.narrator]\nstrong = "425ms"\n',
        encoding="utf-8",
    )
    loaded = load_pause_policy(path)
    assert loaded.policy_id == PausePolicy.from_mapping(explicit).policy_id
    assert loaded.resolve("strong", voice="narrator", where="test") == pytest.approx(0.425)
    assert loaded.resolve(utterplan.PauseIntent("timed", "25ms"), voice="narrator", where="test") == pytest.approx(0.025)

    with pytest.raises(PausePolicyError, match="unknown pause class"):
        PausePolicy.from_mapping(
            {"schema": "audiocompose.pause-policy.v1", "merge": "max", "default": {"unknown": "1s"}}
        )
    with pytest.raises(PausePolicyError, match="no duration configured"):
        PausePolicy.from_mapping(
            {"schema": "audiocompose.pause-policy.v1", "merge": "max", "default": {}}
        ).resolve("strong", voice="narrator", where="test")

    symlink = tmp_path / "pauses-symlink.toml"
    symlink.symlink_to(path)
    with pytest.raises(PausePolicyError, match="must not be a symlink"):
        PausePolicy.load(symlink)
