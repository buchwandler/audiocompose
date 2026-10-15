"""Installed-wheel smoke for the released RenderPlan handoff contract."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import renderplan
from renderplan import (
    RenderManifest,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    SourceSpan,
    SpeechArtifact,
    SpeechItem,
    load_manifest,
    load_plan,
    load_renderplan_set,
    save_manifest,
    save_plan,
    save_profile,
    save_renderplan_set,
)

from audiocompose import BookComposeResult, compose_book
from audiocompose.integrations.renderplan import materialize_renderplan_job
from audiocompose.wav import write_wav

assert Path(sys.prefix).resolve() in Path(__import__("audiocompose").__file__).resolve().parents
assert tuple(map(int, version("renderplan").split(".")[:3])) >= (0, 1, 1)
assert hasattr(renderplan, "validate_set_chapter_plan")

root = Path(tempfile.mkdtemp(prefix="audiocompose-renderplan-smoke-"))
profile = RenderProfile(
    roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
    default_role="narrator",
).with_identity()
profile_id = profile.profile_id or profile.computed_id()
fingerprint = profile_id.removeprefix("sha256:")[:12]
variant = f"smoke--{fingerprint}"
variant_dir = root / "renderplan" / variant
render_dir = root / "voicerender" / variant / "chapter-0001"
variant_dir.mkdir(parents=True)
render_dir.mkdir(parents=True)
save_profile(profile, variant_dir / "renderprofile.toml")
speech = SpeechItem(
    number=1,
    text="smoke",
    language="en-US",
    role="narrator",
    binding={"renderer": "fixture"},
    source=(SourceSpan("chapter.md", 0, 5),),
).with_identity()
plan = RenderPlan(
    source_plan_id="sha256:" + "1" * 64,
    render_profile_id=profile_id,
    flow=(speech,),
).with_identity()
plan_path = variant_dir / "chapter.renderplan.toml"
save_plan(plan, plan_path)
render_wav = render_dir / speech.wav
write_wav(render_wav, np.zeros(64, dtype=np.float32), 16_000)
artifact = SpeechArtifact(
    segment_number=speech.number,
    segment_id=speech.segment_id or speech.computed_id(),
    wav=speech.wav,
    sha256="sha256:" + hashlib.sha256(render_wav.read_bytes()).hexdigest(),
    frames=64,
    sample_rate=16_000,
    channels=1,
)
manifest = RenderManifest(
    render_plan_id=plan.plan_id or plan.computed_id(), speech=(artifact,), complete=True
).with_identity()
manifest_path = render_dir / "render-manifest.toml"
save_manifest(manifest, manifest_path)
chapter = RenderPlanSetChapter(
    id="chapter-0001",
    source_number=1,
    source="chapter.md",
    utterplan="chapter.utterplan.toml",
    source_plan_id=plan.source_plan_id,
    renderplan=plan_path.name,
    render_plan_id=plan.plan_id or plan.computed_id(),
)
renderplan_set = RenderPlanSet(
    variant="smoke",
    profile_fingerprint=fingerprint,
    producer_version="0.1.1",
    manifest="book.json",
    utterplan_dir="utterplan",
    profile_id=profile_id,
    snapshot="renderprofile.toml",
    source_profile="profile.toml",
    chapters=(chapter,),
)
set_path = variant_dir / "renderplan-set.toml"
save_renderplan_set(set_path, renderplan_set)
loaded_set = load_renderplan_set(set_path)
loaded_plan = load_plan(plan_path)
loaded_manifest = load_manifest(manifest_path)
job = materialize_renderplan_job(loaded_plan, loaded_manifest, render_dir=render_dir)
assert loaded_set.chapters[0].render_plan_id == job.source["renderplan.plan_id"]
assert job.source["renderplan.source_plan_id"] == loaded_plan.source_plan_id
assert job.source["renderplan.render_profile_id"] == profile_id
assert job.source["render_manifest.manifest_id"] == loaded_manifest.manifest_id
job.validate(verify_sources=True)
book_result = compose_book(root, variant=variant)
assert isinstance(book_result, BookComposeResult)
assert book_result.requested == book_result.written == 1
assert book_result.exit_code == 0
assert book_result.report_path.is_file()
assert (book_result.output_variant_dir / "chapter-0001.wav").is_file()
print("installed RenderPlan set/plan/manifest -> AudioJob smoke passed")
