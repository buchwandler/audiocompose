from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from audiocompose.document_contract import (
    DOCUMENT_ID,
    DOCUMENT_SOURCE_FILENAME,
    DocumentWorkspaceContractError,
    read_document_workspace,
)


def _document_workspace(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / DOCUMENT_SOURCE_FILENAME).write_text("# Sample\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.document",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "document": {"id": DOCUMENT_ID, "path": DOCUMENT_SOURCE_FILENAME},
            }
        ),
        encoding="utf-8",
    )
    return root


def _document_render_workspace(
    root: Path,
    *,
    profile_pause_ms: int | None = None,
    include_pause: bool = False,
    emphasis_controls=(),
) -> tuple[Path, str]:
    utterplan = pytest.importorskip("utterplan")
    import hashlib

    import numpy as np
    from renderplan import (
        PauseItem,
        Producer,
        RenderManifest,
        RenderPlan,
        RenderProfile,
        RoleProfile,
        SourceSpan,
        SpeechArtifact,
        SpeechItem,
        save_manifest,
        save_plan,
        save_profile,
    )
    from renderplan import PausePolicy as RenderPausePolicy

    from audiocompose.integrations.voicerender import EMPHASIS_COMPOSITION_CAPABILITY
    from audiocompose.wav import write_wav

    workspace = _document_workspace(root)
    utterplan_dir = workspace / "utterplan"
    utterplan_dir.mkdir()
    profile = RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
        pauses=(
            RenderPausePolicy(mode="manual", durations_ms={"strong": profile_pause_ms})
            if profile_pause_ms is not None
            else RenderPausePolicy()
        ),
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"printer-cast--{fingerprint}"
    renderplan_variant = workspace / "renderplan" / variant
    voicerender_variant = workspace / "voicerender" / variant
    renderplan_variant.mkdir(parents=True)
    voicerender_variant.mkdir(parents=True)
    save_profile(profile, renderplan_variant / "renderprofile.toml")

    source_plan = utterplan.FlowPlan(language="en-US", flow=())
    source_path = utterplan_dir / "document.utterplan.toml"
    source_plan.save(source_path)
    speech = SpeechItem(
        number=1,
        text="Sample",
        language="en-US",
        role="narrator",
        binding={"renderer": "fixture"},
        controls=tuple(emphasis_controls),
        source=(SourceSpan("document", 0, 6),),
    ).with_identity()
    flow = (
        (PauseItem(duration_ms=125, reason="strong").with_identity(),) if include_pause else ()
    ) + (speech,)
    plan = RenderPlan(
        source_plan_id=source_plan.plan_id,
        render_profile_id=profile_id,
        flow=flow,
    ).with_identity()
    save_plan(plan, renderplan_variant / "document.renderplan.toml")

    render_dir = voicerender_variant / DOCUMENT_ID
    render_dir.mkdir()
    speech_wav = render_dir / speech.wav
    write_wav(speech_wav, np.full(160, 0.1, dtype=np.float32), 16_000)
    artifact = SpeechArtifact(
        segment_number=1,
        segment_id=speech.segment_id or speech.computed_id(),
        wav=speech.wav,
        sha256="sha256:" + hashlib.sha256(speech_wav.read_bytes()).hexdigest(),
        frames=160,
        sample_rate=16_000,
        channels=1,
    )
    manifest = RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=(artifact,),
        complete=True,
        producer=Producer(name="voicerender", version="0.2.4"),
    ).with_identity()
    save_manifest(manifest, render_dir / "render-manifest.toml")
    deferred = []
    for item in plan.flow:
        if not isinstance(item, SpeechItem):
            continue
        count = sum(control.name == "emphasis" for control in item.controls)
        if count:
            deferred.append(
                {
                    "segment_number": item.number,
                    "segment_id": item.segment_id or item.computed_id(),
                    "capability": EMPHASIS_COMPOSITION_CAPABILITY,
                    "count": count,
                }
            )
    diagnostics = {
        "schema": "voicerender.render-diagnostics.v1",
        "render_plan_id": plan.plan_id or plan.computed_id(),
        "emphasis_policy": "defer-to-compose" if deferred else "reject",
        "pending_composition": bool(deferred),
        "deferred_controls": deferred,
        "warnings": [],
    }
    (render_dir / "render-diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    return workspace, variant


def test_document_workspace_accepts_root_and_canonical_file(tmp_path: Path) -> None:
    workspace = _document_workspace(tmp_path / "story.ssmd-bundle")

    from_root = read_document_workspace(workspace)
    from_file = read_document_workspace(workspace / DOCUMENT_SOURCE_FILENAME)

    assert from_root == from_file
    assert from_root.root == workspace.resolve()
    assert from_root.manifest_path == workspace.resolve() / "manifest.json"
    assert from_root.document_id == DOCUMENT_ID
    assert from_root.document_path == workspace.resolve() / DOCUMENT_SOURCE_FILENAME


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format", "ssmdconvert.book"),
        ("schema_version", 2),
        ("schema_version", True),
        ("ssmd_version", "0.8"),
        ("document.id", "document-0002"),
        ("document.path", "../outside.ssmd.md"),
        ("document.path", "/tmp/document.ssmd.md"),
        ("document.path", "C:\\document.ssmd.md"),
        ("document.path", "nested/document.ssmd.md"),
    ],
)
def test_document_workspace_rejects_invalid_manifest_identity(
    tmp_path: Path, field: str, value: object
) -> None:
    workspace = _document_workspace(tmp_path / "story.ssmd-bundle")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if field.startswith("document."):
        manifest["document"][field.removeprefix("document.")] = value
    else:
        manifest[field] = value
    (workspace / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DocumentWorkspaceContractError):
        read_document_workspace(workspace)


def test_document_workspace_rejects_zip_and_missing_document(tmp_path: Path) -> None:
    archive = tmp_path / "story.ssmd-bundle.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("manifest.json", "{}")
    with pytest.raises(DocumentWorkspaceContractError, match="ZIP input is not supported"):
        read_document_workspace(archive)

    workspace = _document_workspace(tmp_path / "missing.ssmd-bundle")
    (workspace / DOCUMENT_SOURCE_FILENAME).unlink()
    with pytest.raises(DocumentWorkspaceContractError, match="canonical SSMD document source"):
        read_document_workspace(workspace)


def test_document_workspace_rejects_symlinked_and_escaping_files(tmp_path: Path) -> None:
    workspace = _document_workspace(tmp_path / "story.ssmd-bundle")
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_bytes((workspace / "manifest.json").read_bytes())
    (workspace / "manifest.json").unlink()
    try:
        (workspace / "manifest.json").symlink_to(outside_manifest)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(DocumentWorkspaceContractError, match="symlink"):
        read_document_workspace(workspace)

    (workspace / "manifest.json").unlink()
    (workspace / "manifest.json").write_bytes(outside_manifest.read_bytes())
    outside_document = tmp_path / "outside.ssmd.md"
    outside_document.write_text("# outside\n", encoding="utf-8")
    (workspace / DOCUMENT_SOURCE_FILENAME).unlink()
    (workspace / DOCUMENT_SOURCE_FILENAME).symlink_to(outside_document)
    with pytest.raises(DocumentWorkspaceContractError, match="symlink"):
        read_document_workspace(workspace)


def test_document_workspace_rejects_duplicate_json_fields_and_wrong_file_name(
    tmp_path: Path,
) -> None:
    workspace = _document_workspace(tmp_path / "story.ssmd-bundle")
    (workspace / "manifest.json").write_text(
        '{"format":"ssmdconvert.document","format":"other"}', encoding="utf-8"
    )
    with pytest.raises(DocumentWorkspaceContractError, match="duplicate JSON field"):
        read_document_workspace(workspace)

    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.document",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "document": {"id": DOCUMENT_ID, "path": DOCUMENT_SOURCE_FILENAME},
            }
        ),
        encoding="utf-8",
    )
    other_file = workspace / "other.ssmd.md"
    other_file.write_text("# Other\n", encoding="utf-8")
    with pytest.raises(DocumentWorkspaceContractError):
        read_document_workspace(other_file)


def test_resolve_document_uses_direct_stage_roots_and_canonical_paths(tmp_path: Path) -> None:
    from audiocompose.document import resolve_document

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    context = resolve_document(workspace / DOCUMENT_SOURCE_FILENAME, variant)

    assert context.workspace == workspace.resolve()
    assert context.document_id == DOCUMENT_ID
    assert context.utterplan_path == workspace / "utterplan" / "document.utterplan.toml"
    assert context.renderplan_variant_dir == workspace / "renderplan" / variant
    assert context.profile_snapshot_path == context.renderplan_variant_dir / "renderprofile.toml"
    assert context.renderplan_path == context.renderplan_variant_dir / "document.renderplan.toml"
    assert context.voicerender_variant_dir == workspace / "voicerender" / variant
    assert context.render_dir == context.voicerender_variant_dir / DOCUMENT_ID
    assert context.output_variant_dir == workspace / "audiocompose" / variant
    assert context.output_path == context.output_variant_dir / f"{DOCUMENT_ID}.wav"
    assert context.audiojob_path == context.output_variant_dir / f"{DOCUMENT_ID}.audiojob"
    assert context.report_path == context.output_variant_dir / "compose-report.toml"
    assert context.media_root == context.renderplan_variant_dir


def test_resolve_document_rejects_unsafe_variant_and_profile_fingerprint(
    tmp_path: Path,
) -> None:
    from audiocompose.document import resolve_document
    from audiocompose.errors import DocumentComposeError

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    with pytest.raises(DocumentComposeError) as exc_info:
        resolve_document(workspace, "../escape")
    assert exc_info.value.code == "variant.invalid"

    wrong_variant = "wrong--000000000000"
    shutil.copytree(
        workspace / "renderplan" / variant,
        workspace / "renderplan" / wrong_variant,
    )
    shutil.copytree(
        workspace / "voicerender" / variant,
        workspace / "voicerender" / wrong_variant,
    )
    with pytest.raises(DocumentComposeError) as exc_info:
        resolve_document(workspace, wrong_variant)
    assert exc_info.value.code == "profile.identity_mismatch"


def test_compose_document_writes_job_report_and_reuses_verified_output(tmp_path: Path) -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    from audiocompose.document import compose_document
    from audiocompose.errors import DocumentComposeError
    from audiocompose.job import load_job

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    events = []
    first = compose_document(workspace, variant=variant, on_progress=events.append)
    assert events
    assert events[0].kind == "compose_started"
    assert events[-1].kind == "compose_completed"
    assert events[0].completed_items == 0
    assert events[0].total_items == events[-1].completed_items == events[-1].total_items == 1
    item_events = [event for event in events if event.item_index is not None]
    assert item_events
    assert all(event.item_index == 0 and event.total_items == 1 for event in item_events)
    assert [event.completed_items for event in events] == sorted(
        event.completed_items for event in events
    )

    assert first.status == "written"
    assert first.success and first.exit_code == 0
    assert first.output == workspace / "audiocompose" / variant / f"{DOCUMENT_ID}.wav"
    assert first.audiojob == workspace / "audiocompose" / variant / f"{DOCUMENT_ID}.audiojob"
    assert first.report_path.is_file()
    assert first.output.is_file()
    assert load_job(first.audiojob).job_id
    assert first.to_dict()["format"] == "audiocompose" + "-document-result"
    with first.report_path.open("rb") as stream:
        report = tomllib.load(stream)
    assert report["format"] == "audiocompose" + "-document-report"
    assert report["document_id"] == DOCUMENT_ID
    assert report["pause_policy_provenance"] == "renderprofile"
    assert report["output_geometry"]["sha256"] == first.sha256

    output_before_reuse = first.output.stat()
    reuse_events = []
    reused = compose_document(workspace, variant=variant, on_progress=reuse_events.append)
    assert reused.status == "reused"
    output_after_reuse = first.output.stat()
    assert reuse_events == []
    assert reused.sha256 == first.sha256
    assert output_after_reuse.st_size == output_before_reuse.st_size
    assert output_after_reuse.st_mtime_ns == output_before_reuse.st_mtime_ns

    diagnostics_path = workspace / "voicerender" / variant / DOCUMENT_ID / "render-diagnostics.json"
    diagnostics_path.write_text(
        diagnostics_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(DocumentComposeError) as exc_info:
        compose_document(workspace, variant=variant)
    assert exc_info.value.code == "output.exists"

    forced = compose_document(workspace, variant=variant, force=True)
    assert forced.status == "written"


def test_compose_document_applies_profile_and_explicit_pause_policies(tmp_path: Path) -> None:
    from audiocompose import Silence
    from audiocompose.document import compose_document
    from audiocompose.job import load_job

    profile_workspace, profile_variant = _document_render_workspace(
        tmp_path / "profile.ssmd-bundle", profile_pause_ms=400, include_pause=True
    )
    profile_result = compose_document(profile_workspace, variant=profile_variant)
    profile_job = load_job(profile_result.audiojob)
    assert profile_result.request.pause_policy_provenance == "renderprofile"
    assert next(
        item for item in profile_job.items if isinstance(item, Silence)
    ).seconds == pytest.approx(0.4)

    explicit_workspace, explicit_variant = _document_render_workspace(
        tmp_path / "explicit.ssmd-bundle", include_pause=True
    )
    explicit = {
        "schema": "audiocompose.pause-policy.v1",
        "merge": "max",
        "default": {},
        "role": {"narrator": {"strong": "410ms"}},
    }
    explicit_result = compose_document(
        explicit_workspace, variant=explicit_variant, pause_policy=explicit
    )
    explicit_job = load_job(explicit_result.audiojob)
    assert explicit_result.request.pause_policy_provenance == "explicit"
    assert next(
        item for item in explicit_job.items if isinstance(item, Silence)
    ).seconds == pytest.approx(0.41)


def test_compose_document_supports_root_overrides_output_options_and_audiojob_path(
    tmp_path: Path,
) -> None:
    from audiocompose.document import DocumentComposeOptions, compose_document

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    custom_renderplan = tmp_path / "custom-renderplan"
    custom_voicerender = tmp_path / "custom-voicerender"
    shutil.copytree(workspace / "renderplan", custom_renderplan)
    shutil.copytree(workspace / "voicerender", custom_voicerender)
    custom_output = tmp_path / "custom-audiocompose"
    custom_audiojob = tmp_path / "jobs" / f"{DOCUMENT_ID}.audiojob"

    result = compose_document(
        workspace,
        variant=variant,
        renderplan_root=custom_renderplan,
        voicerender_root=custom_voicerender,
        output_root=custom_output,
        save_audiojob=custom_audiojob,
        options=DocumentComposeOptions(sample_rate=8_000, clip_policy="warn"),
    )

    assert result.status == "written"
    assert result.sample_rate == 8_000
    assert result.output == custom_output / variant / f"{DOCUMENT_ID}.wav"
    assert result.audiojob == custom_audiojob
    assert result.report_path == custom_output / variant / "compose-report.toml"
    assert result.to_dict()["renderplan_root"] == str(custom_renderplan / variant)
    assert result.to_dict()["voicerender_root"] == str(custom_voicerender / variant)


def test_compose_document_rejects_incomplete_manifest(tmp_path: Path) -> None:
    from dataclasses import replace

    from renderplan import load_manifest, save_manifest

    from audiocompose.document import compose_document
    from audiocompose.errors import DocumentComposeError

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    manifest_path = workspace / "voicerender" / variant / DOCUMENT_ID / "render-manifest.toml"
    manifest = load_manifest(manifest_path)
    save_manifest(
        replace(manifest, complete=False, manifest_id=None).with_identity(), manifest_path
    )

    with pytest.raises(DocumentComposeError) as exc_info:
        compose_document(workspace, variant=variant)
    assert exc_info.value.code == "handoff.invalid"


def test_resolve_document_rejects_renderplan_and_source_identity_mismatches(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from renderplan import load_plan, save_plan

    from audiocompose.document import resolve_document
    from audiocompose.errors import DocumentComposeError

    plan_workspace, plan_variant = _document_render_workspace(tmp_path / "plan.ssmd-bundle")
    plan_path = plan_workspace / "renderplan" / plan_variant / "document.renderplan.toml"
    plan = load_plan(plan_path)
    save_plan(
        replace(plan, render_profile_id="sha256:" + "0" * 64, plan_id=None).with_identity(),
        plan_path,
    )
    with pytest.raises(DocumentComposeError) as exc_info:
        resolve_document(plan_workspace, plan_variant)
    assert exc_info.value.code == "plan.identity_mismatch"

    source_workspace, source_variant = _document_render_workspace(tmp_path / "source.ssmd-bundle")
    utterplan = pytest.importorskip("utterplan")
    utterplan.FlowPlan(language="fr-FR", flow=()).save(
        source_workspace / "utterplan" / "document.utterplan.toml"
    )
    with pytest.raises(DocumentComposeError) as exc_info:
        resolve_document(source_workspace, source_variant)
    assert exc_info.value.code == "plan.identity_mismatch"


def test_compose_document_emphasis_fallback_and_strict_ownership(tmp_path: Path) -> None:
    from renderplan import SpeechControl

    from audiocompose.document import compose_document
    from audiocompose.errors import DocumentComposeError

    controls = (SpeechControl(1, 3, "emphasis", {}),)
    fallback_workspace, fallback_variant = _document_render_workspace(
        tmp_path / "fallback.ssmd-bundle", emphasis_controls=controls
    )
    fallback = compose_document(fallback_workspace, variant=fallback_variant)
    assert fallback.diagnostics[0]["code"] == "emphasis.segment_fallback"

    strict_workspace, strict_variant = _document_render_workspace(
        tmp_path / "strict.ssmd-bundle", emphasis_controls=controls
    )
    with pytest.raises(DocumentComposeError) as exc_info:
        compose_document(strict_workspace, variant=strict_variant, emphasis_policy="strict")
    assert exc_info.value.code == "handoff.invalid"
    assert "split the speech upstream" in str(exc_info.value)
    assert not (strict_workspace / "audiocompose" / strict_variant / f"{DOCUMENT_ID}.wav").exists()


def test_document_package_root_exports_stable_api() -> None:
    from audiocompose import (
        DocumentComposeError,
        DocumentComposeOptions,
        DocumentComposeRequest,
        DocumentComposeResult,
        compose_document,
        default_document_output_policy,
        resolve_document,
    )

    assert issubclass(DocumentComposeError, Exception)
    assert DocumentComposeOptions().sample_rate is None
    assert callable(compose_document)
    assert callable(default_document_output_policy)
    assert callable(resolve_document)
    assert DocumentComposeRequest is not None
    assert DocumentComposeResult is not None


def test_bundle_compose_document_cli_json_human_overrides_and_errors(
    tmp_path: Path, capsys
) -> None:
    from audiocompose.cli import main

    workspace, variant = _document_render_workspace(tmp_path / "story.ssmd-bundle")
    custom_renderplan = tmp_path / "custom-renderplan"
    custom_voicerender = tmp_path / "custom-voicerender"
    shutil.copytree(workspace / "renderplan", custom_renderplan)
    shutil.copytree(workspace / "voicerender", custom_voicerender)
    custom_output = tmp_path / "custom-audiocompose"
    audiojobs_root = tmp_path / "custom-jobs"
    command = [
        "bundle",
        "compose",
        str(workspace),
        "--variant",
        variant,
        "--renderplan-root",
        str(custom_renderplan),
        "--voicerender-root",
        str(custom_voicerender),
        "--output-root",
        str(custom_output),
        "--save-audiojobs",
        str(audiojobs_root),
        "--sample-rate",
        "8000",
        "--json",
        "--progress",
        "always",
    ]
    assert main(command) == 0
    captured = capsys.readouterr()
    wrapper = json.loads(captured.out)
    assert wrapper["format"] == "audiocompose-bundle-result"
    assert wrapper["kind"] == "document"
    result = wrapper["result"]
    assert "preparing bundle" in captured.err
    assert "composing" in captured.err and "complete" in captured.err
    assert result["format"] == "audiocompose" + "-document-result"
    assert result["status"] == "written"
    assert result["sample_rate"] == 8_000
    assert result["output"] == str(custom_output / variant / f"{DOCUMENT_ID}.wav")
    assert result["audiojob"] == str(audiojobs_root / f"{DOCUMENT_ID}.audiojob")
    assert (audiojobs_root / f"{DOCUMENT_ID}.audiojob").is_dir()
    assert result["renderplan_root"] == str(custom_renderplan / variant)
    assert result["voicerender_root"] == str(custom_voicerender / variant)

    human_command = list(command)
    human_command[2] = str(workspace)
    human_command.remove("--json")
    human_command.remove("--output-root")
    human_command.remove(str(custom_output))
    assert main(human_command) == 0
    human_capture = capsys.readouterr()
    assert "AudioCompose bundle compose" in human_capture.out
    assert "kind:      document" in human_capture.out
    assert "selected:  1" in human_capture.out
    assert "complete" in human_capture.err

    assert main(["bundle", "compose", str(workspace), "--variant", "../escape", "--json"]) == 2
    failure = capsys.readouterr()
    assert failure.out == ""
    assert json.loads(failure.err)["valid"] is False
    assert "traceback" not in failure.err.lower()
