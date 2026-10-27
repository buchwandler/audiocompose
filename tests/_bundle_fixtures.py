from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from renderplan import (
    PauseItem,
    Producer,
    RenderManifest,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    SourceSpan,
    SpeechArtifact,
    SpeechItem,
    save_manifest,
    save_plan,
    save_profile,
    save_renderplan_set,
)

from audiocompose.wav import write_wav


def two_chapter_book(tmp_path: Path) -> tuple[Path, str]:
    """Build a valid two-chapter book with only source chapter 5 rendered."""
    utterplan = pytest.importorskip("utterplan")
    workspace = tmp_path / "selection.ssmd-bundle"
    workspace.mkdir()
    chapters_dir = workspace / "chapters"
    utterplan_dir = workspace / "utterplan"
    renderplan_root = workspace / "renderplan"
    voicerender_root = workspace / "voicerender"
    for path in (chapters_dir, utterplan_dir, renderplan_root, voicerender_root):
        path.mkdir()

    profile = RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    label = "narration"
    variant = f"{label}--{profile_id.removeprefix('sha256:')[:12]}"
    plan_dir = renderplan_root / variant
    plan_dir.mkdir()
    save_profile(profile, plan_dir / "renderprofile.toml")

    plan_set_chapters = []
    manifest_chapters = []
    for source_number in (5, 6):
        chapter_id = f"chapter-{source_number:04d}"
        source_path = chapters_dir / f"{chapter_id}.md"
        source_path.write_text(f"chapter {source_number}\n", encoding="utf-8")
        source_plan = utterplan.FlowPlan(
            language="en-US",
            flow=(
                utterplan.FlowUnit(
                    segments=(
                        utterplan.FlowSegment(text=f"chapter {source_number}", language="en-US"),
                    )
                ),
            ),
        )
        source_plan_path = utterplan_dir / f"{chapter_id}.toml"
        source_plan.save(source_plan_path)
        speech = SpeechItem(
            number=1,
            text=f"chapter {source_number}",
            language="en-US",
            role="narrator",
            binding={"renderer": "fixture"},
            source=(SourceSpan(f"{chapter_id}.txt", 0, 10),),
        ).with_identity()
        plan = RenderPlan(
            source_plan_id=source_plan.plan_id,
            render_profile_id=profile_id,
            flow=(
                PauseItem(duration_ms=10, reason="leading").with_identity(),
                speech,
                PauseItem(duration_ms=10, reason="trailing").with_identity(),
            ),
        ).with_identity()
        plan_name = f"{chapter_id}.renderplan.toml"
        save_plan(plan, plan_dir / plan_name)
        plan_set_chapters.append(
            RenderPlanSetChapter(
                id=chapter_id,
                source_number=source_number,
                source=f"../../chapters/{chapter_id}.md",
                utterplan=f"../../utterplan/{chapter_id}.toml",
                source_plan_id=source_plan.plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
        manifest_chapters.append(
            {
                "id": chapter_id,
                "source_number": source_number,
                "title": f"Chapter {source_number}",
                "path": f"chapters/{chapter_id}.md",
            }
        )
        if source_number == 5:
            render_dir = voicerender_root / variant / chapter_id
            render_dir.mkdir(parents=True)
            wav_path = render_dir / speech.wav
            write_wav(wav_path, np.zeros(32, dtype=np.float32), 16_000)
            artifact = SpeechArtifact(
                segment_number=speech.number,
                segment_id=speech.segment_id or speech.computed_id(),
                wav=speech.wav,
                sha256="sha256:" + hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                frames=32,
                sample_rate=16_000,
                channels=1,
            )
            render_manifest = RenderManifest(
                render_plan_id=plan.plan_id or plan.computed_id(),
                speech=(artifact,),
                complete=True,
                producer=Producer(name="voicerender", version="0.2.0"),
            ).with_identity()
            save_manifest(render_manifest, render_dir / "render-manifest.toml")
            diagnostics = {
                "schema": "voicerender.render-diagnostics.v1",
                "render_plan_id": plan.plan_id or plan.computed_id(),
                "emphasis_policy": "reject",
                "pending_composition": False,
                "deferred_controls": [],
                "warnings": [],
            }
            (render_dir / "render-diagnostics.json").write_text(
                json.dumps(diagnostics), encoding="utf-8"
            )

    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.book",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "chapters": manifest_chapters,
            }
        ),
        encoding="utf-8",
    )
    renderplan_set = RenderPlanSet(
        variant=label,
        profile_fingerprint=profile_id.removeprefix("sha256:")[:12],
        producer_version="0.2.0",
        manifest="../../manifest.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=tuple(plan_set_chapters),
    )
    save_renderplan_set(plan_dir / "renderplan-set.toml", renderplan_set)
    return workspace, variant
