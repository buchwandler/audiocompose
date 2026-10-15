from __future__ import annotations

import hashlib
import json
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
    SpeechControl,
    SpeechItem,
    load_manifest,
    load_plan,
    load_renderplan_set,
    save_manifest,
    save_plan,
    save_profile,
    save_renderplan_set,
)
from renderplan import PausePolicy as RenderPausePolicy

from audiocompose.book import (
    BookComposeError,
    load_chapter_plan,
    parse_chapter_selector,
    paths_for_chapter,
    resolve_book,
)
from audiocompose.wav import write_wav


def _profile(pauses: RenderPausePolicy | None = None) -> RenderProfile:
    return RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
        pauses=pauses if pauses is not None else RenderPausePolicy(),
    ).with_identity()

def _fixture_book(
    tmp_path: Path,
    source_numbers=(7, 6, 5, 8),
    *,
    emphasis_controls=(),
    profile_pauses: RenderPausePolicy | None = None,
    source_pause: str | None = None,
):
    utterplan = pytest.importorskip("utterplan")
    workspace = tmp_path / "book"
    profile = _profile(profile_pauses)
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"narration--{fingerprint}"
    plan_variant = workspace / "renderplan" / variant
    voice_variant = workspace / "voicerender" / variant
    utterplan_dir = workspace / "utterplan"
    chapters_dir = workspace / "chapters"
    plan_variant.mkdir(parents=True)
    voice_variant.mkdir(parents=True)
    utterplan_dir.mkdir()
    chapters_dir.mkdir()
    save_profile(profile, plan_variant / "renderprofile.toml")

    chapters = []
    book_chapters = []
    for source_number in source_numbers:
        chapter_id = f"chapter-{source_number:04d}"
        chapter_source = chapters_dir / f"{chapter_id}.md"
        chapter_source.write_text(f"chapter {source_number}\n", encoding="utf-8")
        if source_pause is None:
            source_plan = utterplan.FlowPlan(language="en-US", flow=())
        else:
            source_plan = utterplan.FlowPlan(
                language="en-US",
                flow=(
                    utterplan.FlowUnit(
                        segments=(
                            utterplan.FlowSegment(
                                text=f"chapter {source_number}",
                                language="en-US",
                                pause_before=utterplan.PauseIntent(source_pause),
                            ),
                        ),
                    ),
                ),
            )
        source_path = utterplan_dir / f"{chapter_id}.toml"
        source_plan.save(source_path)
        speech = SpeechItem(
            number=1,
            text=f"chapter {source_number}",
            language="en-US",
            role="narrator",
            binding={"renderer": "fixture"},
            controls=tuple(emphasis_controls),
            source=(SourceSpan(f"{chapter_id}.txt", 0, 10),),
        ).with_identity()
        plan = RenderPlan(
            source_plan_id=source_plan.plan_id,
            render_profile_id=profile_id,
            flow=(
                PauseItem(duration_ms=125, reason=source_pause or "leading").with_identity(),
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
                source_plan_id=source_plan.plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
        book_chapters.append(
            {
                "id": chapter_id,
                "source_number": source_number,
                "title": f"Chapter {source_number}",
                "path": f"chapters/{chapter_id}.md",
            }
        )
    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.book",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "chapters": book_chapters,
            }
        ),
        encoding="utf-8",
    )
    index = RenderPlanSet(
        variant="narration",
        profile_fingerprint=fingerprint,
        producer_version="0.1.0",
        manifest="../../manifest.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=tuple(chapters),
    )
    save_renderplan_set(plan_variant / "renderplan-set.toml", index)
    return workspace, variant, plan_variant, tuple(chapters)

def test_renderprofile_pause_policy_is_derived_recorded_and_persisted(tmp_path):
    from audiocompose import Silence, compose_book
    from audiocompose.job import load_job

    workspace, variant, _, _ = _fixture_book(
        tmp_path,
        source_numbers=(5,),
        profile_pauses=RenderPausePolicy(mode="manual", durations_ms={"strong": 400}),
        source_pause="strong",
    )
    result = compose_book(workspace, variant=variant)
    item = result.outcomes[0]
    assert item.status == "written"
    assert result.to_dict()["pause_policy_provenance"] == "renderprofile"
    assert result.to_dict()["pause_policy_id"].startswith("sha256:")
    assert item.audiojob == "chapter-0005.audiojob"

    job = load_job(workspace / "audiocompose" / variant / item.audiojob)
    assert job.source["audiocompose.pause_policy_id"] == result.to_dict()["pause_policy_id"]
    assert job.source["audiocompose.pause_policy_provenance"] == "renderprofile"
    pauses = [part for part in job.items if isinstance(part, Silence)]
    assert pauses[0].seconds == pytest.approx(0.4)


def test_explicit_pause_policy_overrides_profile_and_rejects_unresolved_intents(tmp_path):
    from audiocompose import Silence, compose_book
    from audiocompose.job import load_job

    workspace, variant, _, _ = _fixture_book(
        tmp_path / "explicit", source_numbers=(5,), source_pause="strong"
    )
    explicit = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {"strong": "410ms"},
    }
    result = compose_book(workspace, variant=variant, pause_policy=explicit)
    assert result.to_dict()["pause_policy_provenance"] == "explicit"
    job = load_job(workspace / "audiocompose" / variant / result.outcomes[0].audiojob)
    pauses = [part for part in job.items if isinstance(part, Silence)]
    assert pauses[0].seconds == pytest.approx(0.41)

    unresolved_workspace, unresolved_variant, _, _ = _fixture_book(
        tmp_path / "unresolved", source_numbers=(5,), source_pause="strong"
    )
    unresolved = compose_book(unresolved_workspace, variant=unresolved_variant)
    assert unresolved.outcomes[0].status == "failed"
    assert unresolved.outcomes[0].error_code == "pause_policy.unresolved"
    assert not (unresolved_workspace / "audiocompose" / unresolved_variant / "chapter-0005.wav").exists()

def test_compose_book_cli_pause_override_and_human_report_summary(tmp_path, capsys):
    from audiocompose.cli import main
    from audiocompose.job import load_job

    workspace, variant, _, _ = _fixture_book(
        tmp_path, source_numbers=(5,), source_pause="strong"
    )
    policy_path = tmp_path / "narration.pause.toml"
    policy_path.write_text(
        'schema = "audiocompose.pause-policy.v1"\n'
        'merge = "max"\n\n'
        '[default]\n'
        'strong = "410ms"\n',
        encoding="utf-8",
    )
    command = [
        "compose-book",
        str(workspace),
        "--variant",
        variant,
        "--pauses",
        str(policy_path),
        "--json",
    ]
    assert main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pause_policy_provenance"] == "explicit"
    assert result["items"][0]["pause_policy_id"].startswith("sha256:")
    job = load_job(workspace / "audiocompose" / variant / "chapter-0005.audiojob")
    assert job.source["audiocompose.pause_policy_provenance"] == "explicit"

    assert main(command[:-1]) == 0
    human = capsys.readouterr().out
    assert "AudioCompose summary" in human
    assert f"report:      {workspace / 'audiocompose' / variant / 'compose-report.toml'}" in human



def test_invalid_selector_ranges_and_duplicates_are_rejected(tmp_path):
    _, _, _, chapters = _fixture_book(tmp_path)
    for selector in ("", "7-5", "5,5", "5-7,6", "5,,7"):
        with pytest.raises(BookComposeError) as exc:
            parse_chapter_selector(selector, chapters)
        assert exc.value.code == "selection.invalid"


def test_book_request_and_error_models_are_public_and_immutable(tmp_path):
    from dataclasses import FrozenInstanceError

    from audiocompose import BookComposeRequest
    from audiocompose.errors import BookComposeError as PublicBookComposeError

    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    context = resolve_book(workspace, variant)
    request = BookComposeRequest(
        workspace=context.workspace,
        variant=variant,
        renderplan_variant_dir=context.renderplan_variant_dir,
        voicerender_variant_dir=context.voicerender_variant_dir,
        output_variant_dir=context.output_variant_dir,
        chapters=context.chapters,
        chapter_selector="5",
        profile_snapshot_path=context.profile_snapshot_path,
    )
    assert request.workspace == workspace.resolve()
    assert request.chapters == chapters
    assert request.output_policy.sample_rate > 0
    with pytest.raises(FrozenInstanceError):
        attribute = "variant"
        setattr(request, attribute, "other")

    error = PublicBookComposeError(
        "handoff.invalid",
        "source identity mismatch",
        chapter_id="chapter-0005",
        source_number=5,
        variant=variant,
        details={"upstream": "identity"},
    )
    assert error.code == "handoff.invalid"
    assert error.chapter_id == "chapter-0005"
    assert error.source_number == 5
    assert error.variant == variant
    assert error.details == {"upstream": "identity"}


def test_native_workspace_and_variant_names_fail_closed(tmp_path):
    workspace, variant, _, _ = _fixture_book(tmp_path)
    with pytest.raises(BookComposeError) as exc:
        resolve_book(workspace, "../escape")
    assert exc.value.code == "variant.invalid"
    assert exc.value.variant == "../escape"

    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps({"format": "other", "schema_version": 1}), encoding="utf-8")
    with pytest.raises(BookComposeError) as exc:
        resolve_book(workspace, variant)
    assert exc.value.code == "workspace.invalid"


def test_utterplan_symlink_escape_is_rejected(tmp_path):
    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    context = resolve_book(workspace, variant)
    chapter = chapters[0]
    source_path = workspace / "utterplan" / f"{chapter.id}.toml"
    outside = tmp_path / "outside.utterplan.toml"
    outside.write_bytes(source_path.read_bytes())
    source_path.unlink()
    source_path.symlink_to(outside)

    with pytest.raises(BookComposeError) as exc:
        paths_for_chapter(context, chapter)
    assert exc.value.code == "workspace.invalid"


def test_utterplan_identity_mismatch_is_rejected_before_composition(tmp_path):
    utterplan = pytest.importorskip("utterplan")
    workspace, variant, _, chapters = _fixture_book(tmp_path, source_numbers=(5,))
    context = resolve_book(workspace, variant)
    source_path = workspace / "utterplan" / f"{chapters[0].id}.toml"
    utterplan.FlowPlan(language="fr-FR", flow=()).save(source_path)

    with pytest.raises(BookComposeError) as exc:
        load_chapter_plan(context, paths_for_chapter(context, chapters[0]))
    assert exc.value.code == "handoff.invalid"


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
    assert exc.value.code == "selection.invalid"
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
    from utterplan import FlowPlan

    source_plan_id = FlowPlan.load(workspace / "utterplan" / "chapter-0005.toml").plan_id
    assert first["items"][0]["source_plan_id"] == source_plan_id
    assert first["items"][0]["render_profile_id"]
    assert first["items"][0]["manifest_id"]
    assert first["items"][0]["sha256"].startswith("sha256:")
    assert first["items"][0]["duration_seconds"] > 0
    output = workspace / "audiocompose" / variant / "chapter-0005.wav"
    assert output.is_file()
    assert Path(first["report"]).is_file()

    assert main(command) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["written"] == 0 and second["reused"] == 1
    assert second["items"][0]["status"] == "reused"


def test_public_book_api_and_cli_have_matching_outputs_and_reuse(tmp_path, capsys):
    import json

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    from audiocompose import BookComposeResult, compose_book
    from audiocompose.cli import main

    api_workspace, api_variant, _, _ = _fixture_book(tmp_path / "api", source_numbers=(7, 6, 5, 8))
    cli_workspace, cli_variant, _, _ = _fixture_book(tmp_path / "cli", source_numbers=(7, 6, 5, 8))
    progress = []
    api_result = compose_book(
        api_workspace, variant=api_variant, chapters="5,8", on_progress=progress.append
    )
    assert isinstance(api_result, BookComposeResult)
    assert [item.source_number for item in api_result.outcomes] == [5, 8]
    assert [item.kind for item in progress] == [
        "book_started",
        "chapter_started",
        "chapter_materialized",
        "chapter_compose_started",
        "chapter_compose_completed",
        "chapter_started",
        "chapter_materialized",
        "chapter_compose_started",
        "chapter_compose_completed",
        "book_completed",
    ]
    assert [item.completed_chapters for item in progress if item.outcome is not None] == [1, 2]
    assert api_result.exit_code == 0
    assert not (api_workspace / ".readio").exists()

    cli_command = [
        "compose-book",
        str(cli_workspace),
        "--variant",
        cli_variant,
        "--chapters",
        "5,8",
        "--json",
    ]
    assert main(cli_command) == 0
    assert not (cli_workspace / ".readio").exists()
    cli_result = json.loads(capsys.readouterr().out)
    api_items = api_result.to_dict()["items"]
    assert api_items == cli_result["items"]
    assert [item["id"] for item in api_items] == ["chapter-0005", "chapter-0008"]
    assert [item["sha256"] for item in api_items] == [
        item["sha256"] for item in cli_result["items"]
    ]
    assert [item["output_policy_id"] for item in api_items] == [
        item["output_policy_id"] for item in cli_result["items"]
    ]
    with Path(api_result.report_path).open("rb") as stream:
        api_report = tomllib.load(stream)
    with Path(cli_result["report"]).open("rb") as stream:
        cli_report = tomllib.load(stream)
    assert api_report["item"] == cli_report["item"]
    assert api_report["workspace"] == "../.."
    assert api_report["audiojobs_root"] == "."
    for key in ("utterplan", "render_manifest"):
        assert not Path(api_report["item"][0][key]).is_absolute()
        assert (Path(api_result.report_path).parent / api_report["item"][0][key]).is_file()
    assert api_report["item"][0]["job"] == "chapter-0005.audiojob"

    api_reused = compose_book(api_workspace, variant=api_variant, chapters="5,8")
    assert main(cli_command) == 0
    cli_reused = json.loads(capsys.readouterr().out)
    assert [item.status for item in api_reused.outcomes] == ["reused", "reused"]
    assert api_reused.to_dict()["items"] == cli_reused["items"]


def test_public_book_api_and_cli_report_matching_chapter_errors(tmp_path, capsys):
    import json

    from audiocompose import compose_book
    from audiocompose.cli import main

    api_workspace, api_variant, _, chapters = _fixture_book(tmp_path / "api", source_numbers=(5,))
    cli_workspace, cli_variant, _, cli_chapters = _fixture_book(
        tmp_path / "cli", source_numbers=(5,)
    )
    (api_workspace / "voicerender" / api_variant / chapters[0].id / "render-manifest.toml").unlink()
    (
        cli_workspace / "voicerender" / cli_variant / cli_chapters[0].id / "render-manifest.toml"
    ).unlink()

    api_result = compose_book(api_workspace, variant=api_variant)
    cli_command = ["compose-book", str(cli_workspace), "--variant", cli_variant, "--json"]
    assert main(cli_command) == api_result.exit_code == 2
    cli_result = json.loads(capsys.readouterr().out)
    assert api_result.outcomes[0].status == "failed"
    assert api_result.outcomes[0].error_code == "manifest.missing"
    api_item = api_result.to_dict()["items"][0]
    cli_item = cli_result["items"][0]
    assert {
        key: api_item[key]
        for key in ("id", "source_number", "render_plan_id", "status", "error_code")
    } == {
        key: cli_item[key]
        for key in ("id", "source_number", "render_plan_id", "status", "error_code")
    }


def test_installed_package_metadata_rejects_renderplan_010():
    from importlib.metadata import requires

    from packaging.requirements import Requirement
    from packaging.version import Version

    requirements = [Requirement(value) for value in requires("audiocompose") or ()]
    renderplan = next(
        requirement for requirement in requirements if requirement.name == "renderplan"
    )
    assert Version("0.1.0") not in renderplan.specifier
    assert Version("0.1.1") in renderplan.specifier


def test_compose_book_emphasis_default_warning_and_strict_failure(tmp_path, capsys):
    import json

    from audiocompose import AudioClip, AudioJob, Gain
    from audiocompose.cli import main
    from audiocompose.wav import read_wav

    controls = (SpeechControl(1, 3, "emphasis", {}),)
    workspace, variant, _, _ = _fixture_book(
        tmp_path / "fallback", source_numbers=(5,), emphasis_controls=controls
    )
    jobs_root = tmp_path / "jobs"
    assert (
        main(
            [
                "compose-book",
                str(workspace),
                "--variant",
                variant,
                "--save-audiojobs",
                str(jobs_root),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["emphasis_policy"] == "segment_fallback"
    assert result["diagnostics"][0]["code"] == "emphasis.segment_fallback"
    assert "entire segment" in result["items"][0]["emphasis_warning"]
    job = AudioJob.load(jobs_root / "chapter-0005.audiojob")
    speech_clip = job.items[1]
    assert isinstance(speech_clip, AudioClip)
    assert speech_clip.operations == (Gain(1.5),)
    report = Path(result["report"]).read_text(encoding="utf-8")
    assert "emphasis_warning" in report and "segment_fallback" in report
    chapter_audio, _ = read_wav(workspace / "audiocompose" / variant / "chapter-0005.wav")
    assert np.max(np.abs(chapter_audio)) > 0

    strict_workspace, strict_variant, _, _ = _fixture_book(
        tmp_path / "strict", source_numbers=(5,), emphasis_controls=controls
    )
    assert (
        main(
            [
                "compose-book",
                str(strict_workspace),
                "--variant",
                strict_variant,
                "--emphasis-policy",
                "strict",
                "--json",
            ]
        )
        == 2
    )
    strict_result = json.loads(capsys.readouterr().out)
    assert strict_result["items"][0]["status"] == "failed"
    assert "split the speech upstream" in strict_result["items"][0]["error_message"]
    assert not (strict_workspace / "audiocompose" / strict_variant / "chapter-0005.wav").exists()


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
    assert error["errors"][0]["code"] == "selection.invalid"
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


def test_unsafe_set_chapter_id_is_rejected_by_shared_set_loader(tmp_path):
    from renderplan import RenderPlanSetError

    workspace, variant, plan_variant, _ = _fixture_book(tmp_path, source_numbers=(5,))
    set_path = plan_variant / "renderplan-set.toml"
    renderplan_set = load_renderplan_set(set_path)
    chapter_id = renderplan_set.chapters[0].id
    raw = set_path.read_text(encoding="utf-8")
    raw = raw.replace(f'id = "{chapter_id}"', 'id = "../escape"', 1)
    set_path.write_text(raw, encoding="utf-8")

    with pytest.raises(RenderPlanSetError, match="safe single basename"):
        load_renderplan_set(set_path)
    assert not (workspace / "audiocompose" / variant / "escape.wav").exists()


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


def test_reuse_requires_current_policy_and_valid_audiojob_identity(tmp_path):
    from audiocompose import compose_book

    workspace, variant, _, _ = _fixture_book(
        tmp_path, source_numbers=(5,), source_pause="strong"
    )
    first_policy = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {"strong": "410ms"},
    }
    changed_policy = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {"strong": "420ms"},
    }

    first = compose_book(workspace, variant=variant, pause_policy=first_policy)
    assert first.outcomes[0].status == "written"
    assert compose_book(workspace, variant=variant, pause_policy=first_policy).outcomes[0].status == "reused"

    stale_policy = compose_book(workspace, variant=variant, pause_policy=changed_policy)
    assert stale_policy.outcomes[0].error_code == "output.exists"
    forced = compose_book(workspace, variant=variant, pause_policy=changed_policy, force=True)
    assert forced.outcomes[0].status == "written"

    job_json = workspace / "audiocompose" / variant / "chapter-0005.audiojob" / "audiojob.json"
    job_json.write_text("not a valid AudioJob bundle", encoding="utf-8")
    tampered_job = compose_book(workspace, variant=variant, pause_policy=changed_policy)
    assert tampered_job.outcomes[0].error_code == "output.exists"
    compose_book(workspace, variant=variant, pause_policy=changed_policy, force=True)

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    report_path = workspace / "audiocompose" / variant / "compose-report.toml"
    with report_path.open("rb") as stream:
        report = tomllib.load(stream)
    old_version = report["item"][0]["audiocompose_version"]
    report_text = report_path.read_text(encoding="utf-8")
    report_text = report_text.replace(
        f'audiocompose_version = "{old_version}"',
        'audiocompose_version = "0.0.0-stale"',
        1,
    )
    report_path.write_text(report_text, encoding="utf-8")
    stale_version = compose_book(workspace, variant=variant, pause_policy=changed_policy)
    assert stale_version.outcomes[0].error_code == "output.exists"


def test_progress_callback_failure_writes_report_without_final_wav(tmp_path):
    from audiocompose import compose_book

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5, 6))

    def fail_after_materialization(progress):
        if progress.kind == "chapter_materialized":
            raise RuntimeError("observer unavailable")

    result = compose_book(workspace, variant=variant, on_progress=fail_after_materialization)
    assert [item.status for item in result.outcomes] == ["failed", "skipped"]
    assert result.outcomes[0].error_code == "progress.callback_failed"
    assert result.exit_code == 2
    assert result.report_path.is_file()
    assert not (result.output_variant_dir / "chapter-0005.wav").exists()
    assert not (result.output_variant_dir / "chapter-0006.wav").exists()
    report_text = result.report_path.read_text(encoding="utf-8")
    assert 'error_code = "progress.callback_failed"' in report_text


def test_composer_failure_never_publishes_a_chapter_wav(tmp_path, monkeypatch):
    import audiocompose.composer as composer_module
    from audiocompose import compose_book

    workspace, variant, _, _ = _fixture_book(tmp_path, source_numbers=(5,))

    def fail_composition(self, job):
        raise RuntimeError("simulated composer failure")

    monkeypatch.setattr(composer_module.Composer, "compose", fail_composition)
    result = compose_book(workspace, variant=variant)
    assert result.outcomes[0].status == "failed"
    assert result.outcomes[0].error_code == "compose.failed"
    assert not (result.output_variant_dir / "chapter-0005.wav").exists()
    assert result.report_path.is_file()


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


def test_compose_book_requires_optional_producer_extra_but_import_is_lazy(tmp_path):
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
from audiocompose import AudioJob, Composer
from audiocompose.cli import main
assert main(['compose-book', sys.argv[1], '--variant', sys.argv[2], '--json']) == 2
assert not any(name == 'voicerender' or name.startswith('voicerender.') for name in sys.modules)
assert not any(name == 'utterplan' or name.startswith('utterplan.') for name in sys.modules)
assert AudioJob and Composer
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(workspace), variant],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "dependency.api_incompatible" in completed.stdout
