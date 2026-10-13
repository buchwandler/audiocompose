from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from renderplan import (
    AudioItem,
    PauseItem,
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

from audiocompose.book import (
    BookComposeError,
    load_chapter_plan,
    parse_chapter_selector,
    paths_for_chapter,
    resolve_book,
)
from audiocompose.wav import write_wav


def _profile() -> RenderProfile:
    return RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
    ).with_identity()


def _fixture_book(tmp_path: Path, source_numbers=(7, 6, 5, 8)):
    workspace = tmp_path / "book"
    profile = _profile()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    plan_variant = workspace / "renderplan" / variant
    voice_variant = workspace / "voicerender" / variant
    plan_variant.mkdir(parents=True)
    voice_variant.mkdir(parents=True)
    (workspace / "chapters").mkdir()
    save_profile(profile, plan_variant / "renderprofile.toml")

    chapters = []
    for source_number in source_numbers:
        chapter_id = f"chapter-{source_number:04d}"
        speech = SpeechItem(
            number=1,
            text=f"chapter {source_number}",
            language="en-US",
            role="narrator",
            binding={"renderer": "fixture"},
            source=(SourceSpan(f"{chapter_id}.txt", 0, 10),),
        ).with_identity()
        plan = RenderPlan(
            source_plan_id=f"sha256:{source_number:064d}",
            render_profile_id=profile_id,
            flow=(
                PauseItem(duration_ms=125, reason="leading").with_identity(),
                speech,
                PauseItem(duration_ms=250, reason="trailing").with_identity(),
            ),
        ).with_identity()
        plan_name = f"{chapter_id}.renderplan.toml"
        save_plan(plan, plan_variant / plan_name)
        render_dir = voice_variant / chapter_id
        render_dir.mkdir()
        wav_path = render_dir / speech.wav
        write_wav(wav_path, np.full(32, source_number / 100.0, dtype=np.float32), 16_000)
        artifact = SpeechArtifact(
            segment_number=speech.number,
            segment_id=speech.segment_id or speech.computed_id(),
            wav=speech.wav,
            sha256="sha256:" + hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            frames=32,
            sample_rate=16_000,
            channels=1,
        )
        manifest = RenderManifest(
            render_plan_id=plan.plan_id or plan.computed_id(),
            speech=(artifact,),
            complete=True,
        ).with_identity()
        save_manifest(manifest, render_dir / "render-manifest.toml")
        chapters.append(
            RenderPlanSetChapter(
                id=chapter_id,
                source_number=source_number,
                source=f"../../chapters/{chapter_id}.md",
                utterplan=f"../../utterplan/{chapter_id}.toml",
                source_plan_id=plan.source_plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
    index = RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="0.1.0",
        manifest="../../book.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=tuple(chapters),
    )
    save_renderplan_set(plan_variant / "renderplan-set.toml", index)
    return workspace, variant, plan_variant, tuple(chapters)


def test_chapter_selector_preserves_renderplan_set_order(tmp_path):
    _, variant, _, chapters = _fixture_book(tmp_path)
    assert [item.source_number for item in parse_chapter_selector("5-7", chapters)] == [7, 6, 5]
    assert [item.source_number for item in parse_chapter_selector("5,8", chapters)] == [5, 8]
    assert [item.source_number for item in parse_chapter_selector("all", chapters)] == [7, 6, 5, 8]
    assert variant.startswith("narration--")


def test_unknown_chapter_selector_fails_before_composition(tmp_path):
    _, _, _, chapters = _fixture_book(tmp_path)
    with pytest.raises(BookComposeError, match="unknown selected chapter") as exc:
        parse_chapter_selector("5,9", chapters)
    assert exc.value.code == "index.invalid"
    with pytest.raises(BookComposeError, match="unknown selected chapter"):
        parse_chapter_selector("1-999999999", chapters)


def test_resolve_book_validates_set_profile_and_chapter_plan_identity(tmp_path):
    workspace, variant, _, chapters = _fixture_book(tmp_path)
    context = resolve_book(workspace / "chapters", variant, chapters="5")
    assert [item.source_number for item in context.chapters] == [5]
    assert context.workspace == workspace
    assert context.output_variant_dir == workspace / "audiocompose" / variant

    selected = next(chapter for chapter in chapters if chapter.source_number == 5)
    paths = paths_for_chapter(context, selected)
    plan = load_chapter_plan(context, paths)
    assert plan.plan_id == selected.render_plan_id


def test_renderplan_set_chapter_identity_mismatch_is_rejected(tmp_path):
    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    context = resolve_book(workspace, variant)
    wrong = RenderPlanSetChapter(
        id=chapters[0].id,
        source_number=chapters[0].source_number,
        source=chapters[0].source,
        utterplan=chapters[0].utterplan,
        source_plan_id=chapters[0].source_plan_id,
        renderplan=chapters[0].renderplan,
        render_plan_id="sha256:" + "0" * 64,
    )
    with pytest.raises(BookComposeError, match="identity does not match") as exc:
        load_chapter_plan(context, paths_for_chapter(context, wrong))
    assert exc.value.code == "plan.identity_mismatch"


def test_compose_book_cli_writes_report_and_reuses_verified_output(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    command = ["compose-book", str(workspace), "--variant", variant, "--chapters", "5", "--json"]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["written"] == 1 and first["reused"] == 0
    assert first["items"][0]["status"] == "written"
    assert first["items"][0]["frames"] == 9_048
    output = workspace / "audiocompose" / variant / "chapter-0005.wav"
    assert output.is_file()
    assert Path(first["report"]).is_file()

    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["written"] == 0 and second["reused"] == 1
    assert second["items"][0]["status"] == "reused"


def test_compose_book_uses_set_order_for_full_book_ranges_and_commas(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, _ = _fixture_book(tmp_path)
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 0
    full = json.loads(capsys.readouterr().out)
    assert [item["source_number"] for item in full["items"]] == [7, 6, 5, 8]
    assert all(item["status"] == "written" for item in full["items"])

    assert (
        main(["compose-book", str(workspace), "--variant", variant, "--chapters", "5-7", "--json"])
        == 0
    )
    ranged = json.loads(capsys.readouterr().out)
    assert [item["source_number"] for item in ranged["items"]] == [7, 6, 5]
    assert all(item["status"] == "reused" for item in ranged["items"])

    assert (
        main(["compose-book", str(workspace), "--variant", variant, "--chapters", "5,8", "--json"])
        == 0
    )
    comma = json.loads(capsys.readouterr().out)
    assert [item["source_number"] for item in comma["items"]] == [5, 8]


def test_unknown_selected_chapter_fails_before_output_is_created(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    assert (
        main(["compose-book", str(workspace), "--variant", variant, "--chapters", "9", "--json"])
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["errors"][0]["code"] == "index.invalid"
    assert not (workspace / "audiocompose" / variant).exists()


def test_cli_root_overrides_redirect_outputs(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    output_root = tmp_path / "custom-audiocompose"
    assert (
        main(
            [
                "compose-book",
                str(workspace / "chapters"),
                "--variant",
                variant,
                "--renderplan-root",
                str(workspace / "renderplan"),
                "--voicerender-root",
                str(workspace / "voicerender"),
                "--output-root",
                str(output_root),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["written"] == 1
    assert Path(result["report"]).is_file()
    assert (output_root / variant / "chapter-0005.wav").is_file()


def test_variant_label_fingerprint_and_profile_snapshot_must_match(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, plan_variant, _ = _fixture_book(tmp_path, source_numbers=(5,))
    wrong_variant = f"other--{variant.split('--', 1)[1]}"
    shutil.copytree(plan_variant, plan_variant.parent / wrong_variant)
    voice_variant = workspace / "voicerender" / variant
    shutil.copytree(voice_variant, workspace / "voicerender" / wrong_variant)
    assert main(["compose-book", str(workspace), "--variant", wrong_variant, "--json"]) == 2
    mismatch = json.loads(capsys.readouterr().err)
    assert mismatch["errors"][0]["code"] == "variant.identity_mismatch"

    different_profile = RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "changed"})},
        default_role="narrator",
    ).with_identity()
    save_profile(different_profile, plan_variant / "renderprofile.toml")
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    mismatch = json.loads(capsys.readouterr().err)
    assert mismatch["errors"][0]["code"] == "profile.identity_mismatch"


def test_missing_renderplan_and_set_plan_identity_mismatch_fail_cleanly(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, plan_variant, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    plan_path = plan_variant / chapters[0].renderplan
    plan_path.unlink()
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    missing = json.loads(capsys.readouterr().out)
    assert missing["items"][0]["error_code"] == "plan.missing"

    workspace, variant, plan_variant, chapters = _fixture_book(
        tmp_path / "identity", source_numbers=(5,)
    )
    set_path = plan_variant / "renderplan-set.toml"
    renderplan_set = load_renderplan_set(set_path)
    wrong = replace(renderplan_set.chapters[0], render_plan_id="sha256:" + "0" * 64)
    save_renderplan_set(set_path, replace(renderplan_set, chapters=(wrong,)))
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["items"][0]["error_code"] == "plan.identity_mismatch"


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("incomplete", "manifest.incomplete"),
        ("missing-speech", "handoff.invalid"),
        ("sha-mismatch", "handoff.invalid"),
        ("geometry-mismatch", "handoff.invalid"),
        ("wav-symlink", "handoff.invalid"),
        ("manifest-symlink", "manifest.missing"),
    ],
)
def test_strict_handoff_validation_rejects_bad_manifest_and_artifacts(
    tmp_path, capsys, mode, expected_code
):
    import json

    from audiocompose.cli import main

    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    render_dir = workspace / "voicerender" / variant / chapters[0].id
    manifest_path = render_dir / "render-manifest.toml"
    manifest = load_manifest(manifest_path)
    if mode == "incomplete":
        changed = replace(manifest, complete=False, manifest_id=None).with_identity()
        save_manifest(changed, manifest_path)
    elif mode == "missing-speech":
        changed = replace(manifest, speech=(), manifest_id=None).with_identity()
        save_manifest(changed, manifest_path)
    elif mode == "sha-mismatch":
        artifact = replace(manifest.speech[0], sha256="sha256:" + "0" * 64)
        save_manifest(
            replace(manifest, speech=(artifact,), manifest_id=None).with_identity(), manifest_path
        )
    elif mode == "geometry-mismatch":
        artifact = replace(manifest.speech[0], frames=manifest.speech[0].frames + 1)
        save_manifest(
            replace(manifest, speech=(artifact,), manifest_id=None).with_identity(), manifest_path
        )
    elif mode == "wav-symlink":
        speech_wav = render_dir / manifest.speech[0].wav
        outside = tmp_path / "outside.wav"
        outside.write_bytes(speech_wav.read_bytes())
        speech_wav.unlink()
        speech_wav.symlink_to(outside)
    elif mode == "manifest-symlink":
        outside = tmp_path / "render-manifest-copy.toml"
        outside.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink()
        manifest_path.symlink_to(outside)

    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["items"][0]["error_code"] == expected_code


def test_missing_voicerender_chapter_directory_is_reported(tmp_path, capsys):
    import json
    import shutil

    from audiocompose.cli import main

    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    shutil.rmtree(workspace / "voicerender" / variant / chapters[0].id)
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["items"][0]["error_code"] == "render_dir.missing"


def test_unknown_existing_output_requires_force(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    command = ["compose-book", str(workspace), "--variant", variant, "--json"]
    assert main(command) == 0
    capsys.readouterr()
    report = workspace / "audiocompose" / variant / "compose-report.toml"
    report.unlink()

    assert main(command) == 2
    unknown = json.loads(capsys.readouterr().out)
    assert unknown["items"][0]["error_code"] == "output.exists"
    assert main(command + ["--force"]) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced["items"][0]["status"] == "written"


def test_unsafe_set_chapter_id_cannot_escape_output_root(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, plan_variant, _ = _fixture_book(tmp_path, source_numbers=(5,))
    set_path = plan_variant / "renderplan-set.toml"
    renderplan_set = load_renderplan_set(set_path)
    unsafe = replace(renderplan_set.chapters[0], id="../escape")
    save_renderplan_set(set_path, replace(renderplan_set, chapters=(unsafe,)))
    assert main(["compose-book", str(workspace), "--variant", variant, "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["items"][0]["error_code"] == "index.invalid"
    assert not (workspace / "audiocompose" / "escape.wav").exists()


def test_compose_book_uses_policy_and_saves_optional_audiojobs(tmp_path, capsys):
    import json

    from audiocompose import AudioJob
    from audiocompose.cli import main
    from audiocompose.wav import wav_info

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    jobs_root = tmp_path / "audiojobs"
    assert (
        main(
            [
                "compose-book",
                str(workspace),
                "--variant",
                variant,
                "--sample-rate",
                "16000",
                "--save-audiojobs",
                str(jobs_root),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    output = workspace / "audiocompose" / variant / "chapter-0005.wav"
    assert wav_info(output).sample_rate == 16_000
    audiojob = jobs_root / "chapter-0005.audiojob" / "audiojob.json"
    assert audiojob.is_file()
    assert AudioJob.load(audiojob).output.sample_rate == 16_000
    assert result["items"][0]["audiojob"] == "chapter-0005.audiojob"


def test_manifest_change_invalidates_reuse_and_force_recomposes(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    command = ["compose-book", str(workspace), "--variant", variant, "--json"]
    assert main(command) == 0
    capsys.readouterr()
    plan_path = workspace / "renderplan" / variant / chapters[0].renderplan
    plan = load_plan(plan_path)
    speech = next(item for item in plan.flow if isinstance(item, SpeechItem))
    render_dir = workspace / "voicerender" / variant / chapters[0].id
    wav_path = render_dir / speech.wav
    write_wav(wav_path, np.full(32, -0.25, dtype=np.float32), 16_000)
    artifact = SpeechArtifact(
        segment_number=speech.number,
        segment_id=speech.segment_id or speech.computed_id(),
        wav=speech.wav,
        sha256="sha256:" + hashlib.sha256(wav_path.read_bytes()).hexdigest(),
        frames=32,
        sample_rate=16_000,
        channels=1,
    )
    replacement = RenderManifest(
        render_plan_id=chapters[0].render_plan_id,
        speech=(artifact,),
        complete=True,
    ).with_identity()
    save_manifest(replacement, render_dir / "render-manifest.toml")

    assert main(command) == 2
    stale = json.loads(capsys.readouterr().out)
    assert stale["items"][0]["error_code"] == "output.exists"
    assert main(command + ["--force"]) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced["items"][0]["status"] == "written"
    assert main(command + ["--force"]) == 0
    forced_again = json.loads(capsys.readouterr().out)
    assert forced_again["items"][0]["status"] == "written"


def test_batch_continues_by_default_and_fail_fast_skips_later_chapters(tmp_path, capsys):
    import json

    from audiocompose.cli import main

    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5, 6, 7))
    missing_manifest = workspace / "voicerender" / variant / chapters[1].id / "render-manifest.toml"
    missing_manifest.unlink()
    command = ["compose-book", str(workspace), "--variant", variant, "--json"]
    assert main(command) == 2
    continued = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in continued["items"]] == ["written", "failed", "written"]
    assert continued["items"][1]["error_code"] == "manifest.missing"
    report_text = (workspace / "audiocompose" / variant / "compose-report.toml").read_text()
    assert 'error_code = "manifest.missing"' in report_text

    assert main(command + ["--fail-fast"]) == 2
    fast = json.loads(capsys.readouterr().out)
    assert [item["status"] for item in fast["items"]] == ["reused", "failed", "skipped"]


def test_atomic_write_wav_replaces_only_complete_files(tmp_path, monkeypatch):
    import audiocompose.wav as wav_module
    from audiocompose.wav import atomic_write_wav, read_wav

    output = tmp_path / "chapter.wav"
    write_wav(output, np.full(4, 0.1, dtype=np.float32), 8_000)
    atomic_write_wav(output, np.full(8, 0.25, dtype=np.float32), 16_000)
    replaced_bytes = output.read_bytes()
    audio, rate = read_wav(output)
    assert rate == 16_000 and len(audio) == 8

    def interrupted_write(path, audio, sample_rate, *, clip_policy="clamp"):
        Path(path).write_bytes(b"partial wav")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(wav_module, "write_wav", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_wav(output, np.ones(20, dtype=np.float32), 16_000)
    assert output.read_bytes() == replaced_bytes
    assert not list(tmp_path.glob(".chapter.wav.tmp-*.wav"))


def test_compose_report_write_is_atomic(tmp_path, monkeypatch):
    import audiocompose.book as book_module

    report_path = tmp_path / "compose-report.toml"
    report = {
        "variant": "narration--000000000000",
        "render_profile_id": "sha256:" + "0" * 64,
        "requested": 0,
        "written": 0,
        "reused": 0,
        "failed": 0,
        "skipped": 0,
        "output": {"sample_rate": 24_000, "channels": 1, "clip_policy": "clamp"},
        "item": [],
    }
    book_module._write_report(report_path, report)
    original = report_path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(book_module.os, "replace", fail_replace)
    with pytest.raises(BookComposeError, match="cannot write composition report"):
        book_module._write_report(report_path, report)
    assert report_path.read_bytes() == original
    assert not list(tmp_path.glob(".compose-report.toml.*.tmp"))


def test_external_audio_and_output_policy_match_direct_compose_renderplan(tmp_path, capsys):
    from audiocompose.cli import main

    workspace, variant, plan_variant, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    chapter = chapters[0]
    old_plan = load_plan(plan_variant / chapter.renderplan)
    audio_item = AudioItem(
        src="music.wav",
        source=(SourceSpan("chapter-0005.txt", 0, 5),),
        options={"gain_db": -3.0},
    ).with_identity()
    plan = RenderPlan(
        source_plan_id=old_plan.source_plan_id,
        render_profile_id=old_plan.render_profile_id,
        flow=(audio_item,),
    ).with_identity()
    save_plan(plan, plan_variant / chapter.renderplan)
    set_path = plan_variant / "renderplan-set.toml"
    renderplan_set = load_renderplan_set(set_path)
    updated_chapter = replace(chapter, render_plan_id=plan.plan_id or plan.computed_id())
    save_renderplan_set(set_path, replace(renderplan_set, chapters=(updated_chapter,)))
    render_dir = workspace / "voicerender" / variant / chapter.id
    save_manifest(
        RenderManifest(
            render_plan_id=plan.plan_id or plan.computed_id(), speech=(), complete=True
        ).with_identity(),
        render_dir / "render-manifest.toml",
    )
    media_root = tmp_path / "media"
    media_root.mkdir()
    write_wav(media_root / "music.wav", np.linspace(-0.4, 0.4, 320, dtype=np.float32), 16_000)

    common = [
        "--sample-rate",
        "16000",
        "--clip-policy",
        "clamp",
        "--target-lufs",
        "-23",
        "--true-peak-ceiling-dbtp",
        "-1",
        "--peak-policy",
        "reduce_gain",
    ]
    assert (
        main(
            [
                "compose-book",
                str(workspace),
                "--variant",
                variant,
                "--chapters",
                "5",
                "--media-root",
                str(media_root),
                *common,
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    direct_output = tmp_path / "direct.wav"
    assert (
        main(
            [
                "compose-renderplan",
                str(plan_variant / chapter.renderplan),
                "--render-dir",
                str(render_dir),
                "--media-root",
                str(media_root),
                "--output",
                str(direct_output),
                *common,
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    book_output = workspace / "audiocompose" / variant / f"{chapter.id}.wav"
    assert book_output.read_bytes() == direct_output.read_bytes()


def test_compose_book_runs_with_producer_packages_blocked(tmp_path):
    import subprocess
    import sys

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))
    script = """
import importlib.abc
import sys
class BlockProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'voicerender' or fullname.startswith('voicerender.'):
            raise ModuleNotFoundError(fullname)
        if fullname == 'utterplan' or fullname.startswith('utterplan.'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockProducerImports())
from audiocompose.cli import main
assert main(['compose-book', sys.argv[1], '--variant', sys.argv[2], '--json']) == 0
assert not any(name == 'voicerender' or name.startswith('voicerender.') for name in sys.modules)
assert not any(name == 'utterplan' or name.startswith('utterplan.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(workspace), variant],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
