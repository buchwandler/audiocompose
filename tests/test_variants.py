from __future__ import annotations

import json
from pathlib import Path

import pytest
from _bundle_fixtures import two_chapter_book
from renderplan import (
    PauseItem,
    Producer,
    RenderManifest,
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    save_manifest,
    save_plan,
    save_profile,
    save_renderplan_set,
)

from audiocompose.variants import (
    BundleCompositionInspection,
    BundleVariantsResult,
    _derive_state,
    _inspect_bundle_variants,
    inspect_bundle_composition,
    list_bundle_variants,
)


def _bundle(root: Path, kind: str = "book") -> Path:
    root.mkdir(parents=True)
    if kind == "document":
        (root / "document.ssmd.md").write_text("# Test\n", encoding="utf-8")
        manifest = {
            "format": "ssmdconvert.document",
            "schema_version": 1,
            "ssmd_version": "0.9",
            "document": {"id": "document-0001", "path": "document.ssmd.md"},
        }
    else:
        manifest = {"format": "ssmdconvert.book", "schema_version": 1}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _profile() -> RenderProfile:
    return RenderProfile(
        roles={"narrator": RoleProfile(language="en-US", binding={"renderer": "fixture"})},
        default_role="narrator",
    ).with_identity()


def _book_variant(root: Path, label: str = "narration") -> tuple[str, str, RenderPlan]:
    profile = _profile()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"{label}--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    plan = RenderPlan(
        source_plan_id="sha256:" + "1" * 64,
        render_profile_id=profile_id,
        flow=(PauseItem(duration_ms=0, reason="leading").with_identity(),),
    ).with_identity()
    save_plan(plan, variant_dir / "chapter-0001.renderplan.toml")
    chapter = RenderPlanSetChapter(
        id="chapter-0001",
        source_number=1,
        source="../../chapters/chapter-0001.md",
        utterplan="../../utterplan/chapter-0001.toml",
        source_plan_id=plan.source_plan_id,
        renderplan="chapter-0001.renderplan.toml",
        render_plan_id=plan.plan_id or plan.computed_id(),
    )
    renderplan_set = RenderPlanSet(
        variant=label,
        profile_fingerprint=fingerprint,
        producer_version="0.2.0",
        manifest="../../manifest.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="../../narration.renderprofile.toml",
        chapters=(chapter,),
    )
    save_renderplan_set(variant_dir / "renderplan-set.toml", renderplan_set)
    return variant, profile_id, plan


def _document_variant(root: Path, label: str = "narration") -> tuple[str, str, RenderPlan]:
    profile = _profile()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"{label}--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")
    plan = RenderPlan(
        source_plan_id="sha256:" + "2" * 64,
        render_profile_id=profile_id,
        flow=(PauseItem(duration_ms=0, reason="leading").with_identity(),),
    ).with_identity()
    save_plan(plan, variant_dir / "document.renderplan.toml")
    return variant, profile_id, plan


def _write_render(root: Path, variant: str, unit_id: str, plan: RenderPlan, *, complete=True):
    render_dir = root / "voicerender" / variant / unit_id
    render_dir.mkdir(parents=True)
    manifest = RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=(),
        complete=complete,
        producer=Producer(name="voicerender", version="0.2.12"),
    ).with_identity()
    save_manifest(manifest, render_dir / "render-manifest.toml")
    diagnostics = {
        "schema": "voicerender.render-diagnostics.v1",
        "audio_artifact_policy": "voicerender.preserve-silence.v3",
        "render_plan_id": plan.plan_id or plan.computed_id(),
        "emphasis_policy": "reject",
        "pending_composition": False,
        "deferred_controls": [],
    }
    (render_dir / "render-diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    return manifest, render_dir


def _write_book_compose_report(
    root: Path,
    variant: str,
    profile_id: str,
    plan: RenderPlan,
    manifest: RenderManifest,
    manifest_path: Path,
    diagnostics_path: Path,
):
    output_dir = root / "audiocompose" / variant
    output_dir.mkdir(parents=True)
    (output_dir / "chapter-0001.wav").write_bytes(b"owned output placeholder")
    import hashlib

    manifest_digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    diagnostics_digest = "sha256:" + hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
    (output_dir / "compose-report.toml").write_text(
        "\n".join(
            [
                'format = "audiocompose-book-report"',
                "version = 1",
                f'variant = "{variant}"',
                f'render_profile_id = "{profile_id}"',
                "requested = 1",
                "written = 1",
                "reused = 0",
                "failed = 0",
                "skipped = 0",
                "",
                "[[item]]",
                'id = "chapter-0001"',
                f'source_plan_id = "{plan.source_plan_id}"',
                f'render_profile_id = "{profile_id}"',
                f'render_plan_id = "{plan.plan_id or plan.computed_id()}"',
                f'manifest_id = "{manifest.manifest_id or manifest.computed_id()}"',
                f'manifest_sha256 = "{manifest_digest}"',
                f'render_diagnostics_sha256 = "{diagnostics_digest}"',
                'output = "chapter-0001.wav"',
                'status = "written"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_dir / "compose-report.toml"


def _write_document_compose_report(
    root: Path,
    variant: str,
    profile_id: str,
    plan: RenderPlan,
    manifest: RenderManifest,
    manifest_path: Path,
    diagnostics_path: Path,
):
    import hashlib

    output_dir = root / "audiocompose" / variant
    output_dir.mkdir(parents=True)
    (output_dir / "document-0001.wav").write_bytes(b"owned output placeholder")
    values = {
        "format": "audiocompose-document-report",
        "version": 1,
        "document_id": "document-0001",
        "variant": variant,
        "render_profile_id": profile_id,
        "source_plan_id": plan.source_plan_id,
        "render_plan_id": plan.plan_id or plan.computed_id(),
        "render_manifest_id": manifest.manifest_id or manifest.computed_id(),
        "render_manifest_sha256": "sha256:"
        + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "render_diagnostics_sha256": "sha256:"
        + hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        "status": "written",
        "output": "document-0001.wav",
    }
    lines = [
        f"{key} = {json.dumps(value)}" if isinstance(value, str) else f"{key} = {value}"
        for key, value in values.items()
    ]
    report_path = output_dir / "compose-report.toml"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_failed_book_compose_report(
    root: Path,
    variant: str,
    profile_id: str,
    plan: RenderPlan,
    manifest_path: Path,
    diagnostics_path: Path,
    *,
    error_code: str,
    error_message: str,
    failure_stage: str | None = None,
    recovery_action: str | None = None,
    include_identity: bool = True,
) -> Path:
    import hashlib

    output_dir = root / "audiocompose" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    top = [
        'format = "audiocompose-book-report"',
        "version = 1",
        f"variant = {json.dumps(variant)}",
        f"render_profile_id = {json.dumps(profile_id)}",
        "requested = 1",
        "written = 0",
        "reused = 0",
        "failed = 1",
        "skipped = 0",
    ]
    item = [
        'id = "chapter-0001"',
        "source_number = 1",
        f"render_plan_id = {json.dumps(plan.plan_id or plan.computed_id())}",
        'status = "failed"',
        f"error_code = {json.dumps(error_code)}",
        f"error_message = {json.dumps(error_message)}",
    ]
    if failure_stage is not None:
        item.append(f"failure_stage = {json.dumps(failure_stage)}")
    if recovery_action is not None:
        item.append(f"recovery_action = {json.dumps(recovery_action)}")
    if include_identity:
        item.extend(
            (
                "manifest_sha256 = "
                + json.dumps("sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()),
                "render_diagnostics_sha256 = "
                + json.dumps("sha256:" + hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()),
            )
        )
    report_path = output_dir / "compose-report.toml"
    report_path.write_text(chr(10).join(top + ["", "[[item]]", *item]) + chr(10), encoding="utf-8")
    return report_path


LEGACY_VOICERENDER_POLICY_ERROR = (
    "VoiceRender artifacts use an unsupported audio artifact policy; "
    "re-render with VoiceRender >=0.2.11 before composition"
)


def _write_multi_chapter_failure_report(
    root: Path,
    variant: str,
    profile_id: str,
    unit_ids: tuple[str, ...],
    *,
    error_code: str = "handoff.invalid",
    error_message: str = LEGACY_VOICERENDER_POLICY_ERROR,
) -> Path:
    output_dir = root / "audiocompose" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        'format = "audiocompose-book-report"',
        "version = 1",
        f"variant = {json.dumps(variant)}",
        f"render_profile_id = {json.dumps(profile_id)}",
        'audiocompose_version = "0.3.11.dev0+fixture"',
        'emphasis_policy = "segment_fallback"',
        'pause_policy_provenance = "renderprofile"',
        f"requested = {len(unit_ids)}",
        "written = 0",
        "reused = 0",
        f"failed = {len(unit_ids)}",
        "skipped = 0",
    ]
    for unit_id in unit_ids:
        lines.extend(
            [
                "",
                "[[item]]",
                f"id = {json.dumps(unit_id)}",
                f"source_number = {int(unit_id[-4:])}",
                'status = "failed"',
                f"error_code = {json.dumps(error_code)}",
                f"error_message = {json.dumps(error_message)}",
            ]
        )
    report_path = output_dir / "compose-report.toml"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _legacy_multi_chapter_bundle(tmp_path: Path):
    from dataclasses import replace

    root = _bundle(tmp_path / "historical.ssmd-bundle")
    profile = _profile()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    label = "narrator--pt-engaging-v1"
    variant = f"{label}--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    variant_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")

    chapters = []
    artifacts = {}
    for source_number in range(5, 18):
        unit_id = f"chapter-{source_number:04d}"
        plan = RenderPlan(
            source_plan_id=f"sha256:{source_number:064x}",
            render_profile_id=profile_id,
            flow=(PauseItem(duration_ms=0, reason="leading").with_identity(),),
        ).with_identity()
        plan_name = f"{unit_id}.renderplan.toml"
        save_plan(plan, variant_dir / plan_name)
        chapters.append(
            RenderPlanSetChapter(
                id=unit_id,
                source_number=source_number,
                source=f"../../chapters/{unit_id}.md",
                utterplan=f"../../utterplan/{unit_id}.toml",
                source_plan_id=plan.source_plan_id,
                renderplan=plan_name,
                render_plan_id=plan.plan_id or plan.computed_id(),
            )
        )
        manifest, render_dir = _write_render(root, variant, unit_id, plan)
        manifest_path = render_dir / "render-manifest.toml"
        diagnostics_path = render_dir / "render-diagnostics.json"
        historical_manifest = replace(
            manifest,
            producer=Producer(name="voicerender", version="0.2.11"),
            manifest_id=None,
        ).with_identity()
        save_manifest(historical_manifest, manifest_path)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["audio_artifact_policy"] = "voicerender.raw-synthesis.v2"
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        artifacts[unit_id] = (manifest, manifest_path, diagnostics_path)

    save_renderplan_set(
        variant_dir / "renderplan-set.toml",
        RenderPlanSet(
            variant=label,
            profile_fingerprint=fingerprint,
            producer_version="0.2.0",
            manifest="../../manifest.json",
            utterplan_dir="../../utterplan",
            profile_id=profile_id,
            snapshot="renderprofile.toml",
            source_profile="../../narration.renderprofile.toml",
            chapters=tuple(chapters),
        ),
    )
    unit_ids = tuple(f"chapter-{source_number:04d}" for source_number in range(5, 18))
    report_path = _write_multi_chapter_failure_report(root, variant, profile_id, unit_ids)
    return root, variant, profile_id, unit_ids, artifacts, report_path


def _repair_multi_chapter_renders(artifacts) -> None:
    for manifest, manifest_path, diagnostics_path in artifacts.values():
        save_manifest(manifest, manifest_path)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["audio_artifact_policy"] = "voicerender.preserve-silence.v3"
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")


def _bundle_snapshot(root: Path):
    return {
        path.relative_to(root): (
            path.lstat().st_mode,
            path.lstat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def test_selection_scoped_inspection_keeps_global_variant_listing_whole(tmp_path: Path) -> None:
    workspace, _ = two_chapter_book(tmp_path)

    scoped = _inspect_bundle_variants(workspace, chapter_selector="5").variants[0]
    global_variant = list_bundle_variants(workspace).variants[0]

    assert scoped.total_units == 1
    assert scoped.rendered_units == 1
    assert scoped.render_status == "complete"
    assert scoped.ready_to_compose
    assert global_variant.total_units == 2
    assert global_variant.rendered_units == 1
    assert global_variant.render_status == "partial"
    assert global_variant.state == "render_partial"


def test_book_and_document_missing_renderplan_roots_are_empty(tmp_path: Path) -> None:
    book = _bundle(tmp_path / "book.ssmd-bundle")
    document = _bundle(tmp_path / "document.ssmd-bundle", "document")

    book_result = list_bundle_variants(book)
    document_result = list_bundle_variants(document)

    assert book_result.variants == ()
    assert document_result.variants == ()
    assert book_result.to_dict()["count"] == 0
    assert document_result.kind == "document"


def test_list_bundle_variants_accepts_dot_from_inside_bundle(tmp_path: Path, monkeypatch) -> None:
    root = _bundle(tmp_path / "inside.ssmd-bundle")
    monkeypatch.chdir(root)

    result = list_bundle_variants(".")

    assert result.bundle == root.resolve()
    assert result.variants == ()


def test_invalid_document_bundle_is_a_bundle_level_error(tmp_path: Path) -> None:
    from audiocompose.bundle_contract import BundleComposeError

    root = _bundle(tmp_path / "invalid-document.ssmd-bundle", "document")
    (root / "document.ssmd.md").unlink()

    with pytest.raises(BundleComposeError) as exc:
        list_bundle_variants(root)
    assert exc.value.code == "bundle.invalid"


def test_valid_book_variant_without_render_is_not_rendered_and_read_only(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, _ = _book_variant(root)
    before = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    result = list_bundle_variants(root)

    discovered = result.variants[0]
    after = {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    assert discovered.name == variant
    assert discovered.profile_id == profile_id
    assert discovered.plan_status == "valid"
    assert discovered.render_status == "not_rendered"
    assert discovered.compose_status == "not_composed"
    assert discovered.state == "not_rendered"
    assert not discovered.ready_to_compose
    assert before == after
    assert set(result.to_dict()) == {
        "format",
        "version",
        "bundle",
        "kind",
        "renderplan_root",
        "voicerender_root",
        "output_root",
        "count",
        "variants",
    }


def test_valid_document_variant_is_discovered_without_voicerender(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "document.ssmd-bundle", "document")
    variant, profile_id, _ = _document_variant(root)

    result = list_bundle_variants(root)
    discovered = result.variants[0]

    assert discovered.name == variant
    assert discovered.profile_id == profile_id
    assert discovered.unit_kind == "documents"
    assert discovered.total_units == 1
    assert discovered.to_dict()["total_documents"] == 1
    assert discovered.state == "not_rendered"


def test_variant_discovery_sorts_candidates_ignores_files_and_isolates_bad_variant(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    good_name, _, _ = _book_variant(root, "good")
    (root / "renderplan" / "unrelated.txt").write_text("ignored", encoding="utf-8")
    (root / "renderplan" / "broken--123456789012").mkdir()
    another_name, _, _ = _book_variant(root, "another")

    result = list_bundle_variants(root)

    assert [item.name for item in result.variants] == sorted(
        [good_name, another_name, "broken--123456789012"]
    )
    assert (
        next(item for item in result.variants if item.name == "broken--123456789012").plan_status
        == "invalid"
    )
    assert any(item.name == good_name and item.plan_status == "valid" for item in result.variants)


def test_symlink_variant_candidate_is_reported_invalid(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    target = root / "external"
    target.mkdir()
    candidate = root / "renderplan" / "linked--123456789012"
    candidate.parent.mkdir()
    try:
        candidate.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    result = list_bundle_variants(root)

    assert result.variants[0].plan_status == "invalid"
    assert "directory" in (result.variants[0].validation_error or "")


def test_render_status_complete_requires_diagnostics_for_readiness(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, plan = _book_variant(root)
    _write_render(root, variant, "chapter-0001", plan)

    result = list_bundle_variants(root).variants[0]

    assert result.render_status == "complete"
    assert result.state == "ready"
    assert result.ready_to_compose
    assert result.rendered_units == 1


def test_complete_manifest_without_diagnostics_is_not_ready(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, plan = _book_variant(root)
    _, render_dir = _write_render(root, variant, "chapter-0001", plan)
    (render_dir / "render-diagnostics.json").unlink()

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.render_status == "complete"
    assert not discovered.ready_to_compose


def test_oversized_report_does_not_bypass_invalid_diagnostics(tmp_path: Path) -> None:
    from audiocompose.variants import MAX_METADATA_BYTES

    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, plan = _book_variant(root)
    _, render_dir = _write_render(root, variant, "chapter-0001", plan)
    (render_dir / "render-diagnostics.json").write_text("not json", encoding="utf-8")
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.write_bytes(b"x" * (MAX_METADATA_BYTES + 1))

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.render_status == "unknown"
    assert not discovered.ready_to_compose
    assert "VoiceRender diagnostics" in (discovered.render_error or "")
    assert discovered.render_warning is not None


def test_oversized_optional_report_does_not_block_complete_handoffs(tmp_path: Path) -> None:
    from audiocompose.variants import MAX_METADATA_BYTES

    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, plan = _book_variant(root)
    _write_render(root, variant, "chapter-0001", plan)
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.write_bytes(b"x" * (MAX_METADATA_BYTES + 1))

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.render_status == "complete"
    assert discovered.ready_to_compose
    assert discovered.render_error is None
    assert discovered.render_warning is not None
    assert (
        "ignored because all selected RenderManifests and diagnostics" in discovered.render_warning
    )
    assert discovered.to_dict()["render_warning"] == discovered.render_warning


def test_oversized_optional_report_does_not_infer_complete_from_partial_handoffs(
    tmp_path: Path,
) -> None:
    from audiocompose.variants import MAX_METADATA_BYTES

    root, variant = two_chapter_book(tmp_path)
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.write_bytes(b"x" * (MAX_METADATA_BYTES + 1))

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.render_status == "partial"
    assert discovered.rendered_units == 1
    assert not discovered.ready_to_compose


def test_incomplete_manifest_is_partial_and_failed_report_is_reported(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, plan = _book_variant(root)
    _write_render(root, variant, "chapter-0001", plan, complete=False)

    partial = list_bundle_variants(root).variants[0]
    assert partial.render_status == "partial"
    assert partial.state == "render_partial"

    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('status = "failed"\n', encoding="utf-8")
    failed = list_bundle_variants(root).variants[0]
    assert failed.render_status == "failed"
    assert failed.state == "render_failed"


def test_compose_stale_diagnostic_distinguishes_uninspected_handoff(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    manifest, render_dir = _write_render(root, variant, "chapter-0001", plan)
    _write_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        manifest,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
    )

    voice_variant = root / "voicerender" / variant
    voice_variant.rename(voice_variant.with_name(f"{variant}-original"))
    external = tmp_path / "external-voice-root"
    external.mkdir()
    try:
        voice_variant.symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.compose_status == "stale"
    assert "chapter-0001 VoiceRender handoff was not inspected" in (discovered.compose_error or "")
    assert "no longer has a valid VoiceRender manifest" not in (discovered.compose_error or "")


def test_malformed_render_manifest_is_unknown_without_hiding_sibling(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    broken_name, _, _ = _book_variant(root, "broken")
    good_name, _, good_plan = _book_variant(root, "good")
    broken_render = root / "voicerender" / broken_name / "chapter-0001"
    broken_render.mkdir(parents=True)
    (broken_render / "render-manifest.toml").write_text("not toml = [", encoding="utf-8")
    _write_render(root, good_name, "chapter-0001", good_plan)

    result = list_bundle_variants(root)

    assert (
        next(item for item in result.variants if item.name == broken_name).render_status
        == "unknown"
    )
    assert (
        next(item for item in result.variants if item.name == good_name).render_status == "complete"
    )


def test_compose_report_current_output_is_complete_and_upstream_change_is_stale(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    manifest, render_dir = _write_render(root, variant, "chapter-0001", plan)
    report_path = _write_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        manifest,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
    )

    completed = list_bundle_variants(root).variants[0]
    assert completed.compose_status == "complete"
    assert completed.state == "complete"
    assert completed.composed_units == 1
    assert completed.compose_report_path == report_path

    # The report remains well-formed, but provenance changes make it stale.
    content = report_path.read_text(encoding="utf-8").replace(
        manifest.manifest_id or manifest.computed_id(), "sha256:" + "0" * 64
    )
    report_path.write_text(content, encoding="utf-8")
    stale = list_bundle_variants(root).variants[0]
    assert stale.compose_status == "stale"
    assert stale.state == "stale"


def test_failed_handoff_becomes_recomposable_after_real_voicerender_repair(tmp_path: Path) -> None:
    from dataclasses import replace

    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    manifest, render_dir = _write_render(root, variant, "chapter-0001", plan)
    manifest_path = render_dir / "render-manifest.toml"
    diagnostics_path = render_dir / "render-diagnostics.json"

    old_manifest = replace(
        manifest,
        producer=Producer(name="voicerender", version="0.2.11"),
        manifest_id=None,
    ).with_identity()
    save_manifest(old_manifest, manifest_path)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["audio_artifact_policy"] = "voicerender.raw-synthesis.v2"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    report_path = _write_failed_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        manifest_path,
        diagnostics_path,
        error_code="handoff.voicerender_rerender_required",
        error_message="rerender required",
        failure_stage="handoff",
        recovery_action="rerender",
    )

    incompatible = list_bundle_variants(root).variants[0]
    assert incompatible.render_status == "incompatible"
    assert incompatible.compose_status == "failed"
    assert incompatible.state == "render_incompatible"
    assert not incompatible.ready_to_compose
    assert incompatible.recovery_action == "rerender"
    assert incompatible.automatic_convergence_safe
    assert incompatible.to_dict()["recovery_action"] == "rerender"
    assert incompatible.required_audio_artifact_policy == "voicerender.preserve-silence.v3"
    assert incompatible.required_voicerender == ">=0.2.12,<0.3"
    assert incompatible.to_dict()["required_audio_artifact_policy"] == (
        "voicerender.preserve-silence.v3"
    )
    assert incompatible.to_dict()["required_voicerender"] == ">=0.2.12,<0.3"

    repaired_manifest = replace(
        old_manifest,
        producer=Producer(name="voicerender", version="0.2.12"),
        manifest_id=None,
    ).with_identity()
    save_manifest(repaired_manifest, manifest_path)
    diagnostics["audio_artifact_policy"] = "voicerender.preserve-silence.v3"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    repaired = list_bundle_variants(root).variants[0]
    assert repaired.render_status == "complete"
    assert repaired.compose_status == "stale"
    assert repaired.state == "stale"
    assert repaired.ready_to_compose
    assert repaired.recovery_action == "recompose"
    assert repaired.automatic_convergence_safe
    assert repaired.compose_report_path == report_path
    assert "handoff.voicerender_rerender_required" in report_path.read_text(encoding="utf-8")


def test_exact_legacy_handoff_failure_recovers_after_current_handoff(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    _, render_dir = _write_render(root, variant, "chapter-0001", plan)
    report_path = _write_failed_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
        error_code="handoff.invalid",
        error_message=(
            "VoiceRender artifacts use an unsupported audio artifact policy; "
            "re-render with VoiceRender >=0.2.11 before composition"
        ),
        include_identity=False,
    )

    recovered = list_bundle_variants(root).variants[0]
    assert recovered.render_status == "complete"
    assert recovered.compose_status == "stale"
    assert recovered.ready_to_compose
    assert recovered.recovery_action == "recompose"
    assert recovered.automatic_convergence_safe
    assert recovered.compose_report_path == report_path


def test_public_composition_inspector_exposes_legacy_multichapter_recovery(
    tmp_path: Path,
) -> None:
    root, variant, _, unit_ids, artifacts, report_path = _legacy_multi_chapter_bundle(tmp_path)
    historical_report = report_path.read_text(encoding="utf-8")
    assert 'variant = "' + variant + '"' in historical_report
    assert 'audiocompose_version = "0.3.11.dev0+fixture"' in historical_report
    assert 'emphasis_policy = "segment_fallback"' in historical_report
    assert 'pause_policy_provenance = "renderprofile"' in historical_report
    assert "requested = 13" in historical_report
    assert "written = 0" in historical_report
    assert "reused = 0" in historical_report
    assert "failed = 13" in historical_report
    assert "skipped = 0" in historical_report
    assert 'id = "chapter-0005"' in historical_report
    assert 'id = "chapter-0017"' in historical_report
    legacy_defaults = BundleCompositionInspection(variant, "error", ())
    assert legacy_defaults.recovery_action is None
    assert not legacy_defaults.automatic_convergence_safe

    before = _bundle_snapshot(root)
    rich = list_bundle_variants(root).variants[0]
    assert _bundle_snapshot(root) == before
    rerender = inspect_bundle_composition(root, variant=variant, chapters="5-17")
    assert _bundle_snapshot(root) == before
    assert rich.recovery_action == rerender.recovery_action == "rerender"
    assert rich.automatic_convergence_safe == rerender.automatic_convergence_safe is True
    assert rerender.state == "error"
    assert rerender.selected == unit_ids
    assert rerender.force_required is False
    assert rerender.replacement_safe is False
    assert rerender.required_audio_artifact_policy == "voicerender.preserve-silence.v3"
    assert rerender.required_voicerender == ">=0.2.12,<0.3"
    assert "VoiceRender" in (rerender.reason or "")
    assert "audio artifact policy" in (rerender.reason or "")
    assert rerender.recovery_reason == rich.recovery_reason
    serialized = rerender.to_dict()
    assert serialized["version"] == 1
    assert serialized["recovery_action"] == "rerender"
    assert serialized["recovery_reason"] == rerender.recovery_reason
    assert serialized["automatic_convergence_safe"] is True
    assert serialized["required_audio_artifact_policy"] == "voicerender.preserve-silence.v3"
    assert serialized["required_voicerender"] == ">=0.2.12,<0.3"
    assert "failed selected work" not in (rerender.reason or "")

    original_report = report_path.read_bytes()
    _repair_multi_chapter_renders(artifacts)
    assert report_path.read_bytes() == original_report
    before = _bundle_snapshot(root)
    rich = list_bundle_variants(root).variants[0]
    assert _bundle_snapshot(root) == before
    repaired = inspect_bundle_composition(root, variant=variant, chapters="5-17")
    assert _bundle_snapshot(root) == before
    assert rich.recovery_action == repaired.recovery_action == "recompose"
    assert rich.automatic_convergence_safe == repaired.automatic_convergence_safe is True
    assert repaired.state == "stale"
    assert repaired.selected == repaired.stale == unit_ids
    assert repaired.force_required
    assert repaired.replacement_safe


def test_public_composition_inspector_blocks_unrelated_multichapter_failures(
    tmp_path: Path,
) -> None:
    root, variant, profile_id, unit_ids, artifacts, _ = _legacy_multi_chapter_bundle(tmp_path)
    _repair_multi_chapter_renders(artifacts)
    unrelated = (
        ("compose.failed", "renderer composition failed"),
        ("handoff.invalid", "an unrelated handoff validation error"),
    )
    for error_code, error_message in unrelated:
        _write_multi_chapter_failure_report(
            root,
            variant,
            profile_id,
            unit_ids,
            error_code=error_code,
            error_message=error_message,
        )
        before = _bundle_snapshot(root)
        rich = list_bundle_variants(root).variants[0]
        assert _bundle_snapshot(root) == before
        result = inspect_bundle_composition(root, variant=variant, chapters="5-17")
        assert _bundle_snapshot(root) == before
        assert result.state == "error"
        assert result.recovery_action is None
        assert not result.automatic_convergence_safe
        assert rich.recovery_action is None
        assert not rich.automatic_convergence_safe
        assert "chapter-0005" in (result.reason or "")
        assert "recognized recoverable handoff failure" in (result.reason or "")


@pytest.mark.parametrize(
    ("error_code", "error_message"),
    [
        ("compose.failed", "renderer composition failed"),
        ("handoff.invalid", "an unrelated handoff validation error"),
    ],
)
def test_unrelated_failed_compose_reports_remain_fail_closed(
    tmp_path: Path, error_code: str, error_message: str
) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    _, render_dir = _write_render(root, variant, "chapter-0001", plan)
    _write_failed_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
        error_code=error_code,
        error_message=error_message,
        failure_stage="handoff" if error_code == "handoff.invalid" else None,
    )

    failed = list_bundle_variants(root).variants[0]
    assert failed.render_status == "complete"
    assert failed.compose_status == "failed"
    assert failed.state == "compose_failed"
    assert failed.recovery_action is None
    assert not failed.automatic_convergence_safe


def test_missing_output_for_owned_compose_report_is_unknown(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)
    _write_render(root, variant, "chapter-0001", plan)
    output_dir = root / "audiocompose" / variant
    output_dir.mkdir(parents=True)
    (output_dir / "compose-report.toml").write_text(
        f'format = "audiocompose-book-report"\nversion = 1\nvariant = "{variant}"\n'
        f'render_profile_id = "{profile_id}"\nrequested = 1\nwritten = 1\nreused = 0\nfailed = 0\nskipped = 0\n'
        '\n[[item]]\nid = "chapter-0001"\nstatus = "written"\noutput = "chapter-0001.wav"\n',
        encoding="utf-8",
    )

    result = list_bundle_variants(root).variants[0]

    assert result.compose_status == "unknown"
    assert result.state == "unknown"


def test_custom_stage_roots_and_root_errors(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, _ = _book_variant(root)
    custom_plans = tmp_path / "custom-plans"
    custom_plans.mkdir()
    source = root / "renderplan" / variant
    import shutil

    shutil.copytree(source, custom_plans / variant)
    result = list_bundle_variants(root, renderplan_root=custom_plans)
    assert result.renderplan_root == custom_plans.resolve()
    assert result.variants[0].name == variant

    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("file", encoding="utf-8")
    with pytest.raises(Exception, match="not a directory"):
        list_bundle_variants(root, renderplan_root=bad_root)


def test_overall_state_precedence_matrix() -> None:
    assert _derive_state("invalid", "not_rendered", "not_composed") == "invalid"
    assert _derive_state("valid", "not_rendered", "not_composed") == "not_rendered"
    assert _derive_state("valid", "partial", "not_composed") == "render_partial"
    assert _derive_state("valid", "failed", "not_composed") == "render_failed"
    assert _derive_state("valid", "complete", "not_composed") == "ready"
    assert _derive_state("valid", "complete", "partial") == "compose_partial"
    assert _derive_state("valid", "complete", "failed") == "compose_failed"
    assert _derive_state("valid", "complete", "stale") == "stale"
    assert _derive_state("valid", "complete", "complete") == "complete"
    assert _derive_state("valid", "unknown", "not_composed") == "unknown"


def test_variants_result_is_typed_and_immutable(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    result = list_bundle_variants(root)

    assert isinstance(result, BundleVariantsResult)
    with pytest.raises((AttributeError, TypeError)):
        result.__setattr__("bundle", tmp_path)


def test_document_compose_report_is_complete_and_detects_staleness(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "document.ssmd-bundle", "document")
    variant, profile_id, plan = _document_variant(root)
    manifest, render_dir = _write_render(root, variant, "document-0001", plan)
    report_path = _write_document_compose_report(
        root,
        variant,
        profile_id,
        plan,
        manifest,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
    )

    completed = list_bundle_variants(root).variants[0]
    assert completed.compose_status == "complete"
    assert completed.state == "complete"
    assert completed.composed_units == 1
    assert completed.compose_report_path == report_path

    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            plan.plan_id or plan.computed_id(), "sha256:" + "0" * 64
        ),
        encoding="utf-8",
    )
    stale = list_bundle_variants(root).variants[0]
    assert stale.compose_status == "stale"
    assert stale.state == "stale"


def test_complete_voicerender_report_without_manifests_is_unknown(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, _ = _book_variant(root)
    report = root / "voicerender" / variant / "voicerender-report.toml"
    report.parent.mkdir(parents=True)
    report.write_text('status = "complete"\n', encoding="utf-8")

    discovered = list_bundle_variants(root).variants[0]

    assert discovered.render_status == "unknown"
    assert discovered.state == "unknown"


def test_bundle_composition_inspector_reports_selected_states_without_writes(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, profile_id, plan = _book_variant(root)

    def snapshot():
        return {
            path.relative_to(root): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")
        }

    before = snapshot()
    missing = inspect_bundle_composition(root, variant=variant)
    assert missing.state == "not_started"
    assert missing.selected == ("chapter-0001",)
    assert missing.missing == ("chapter-0001",)
    assert snapshot() == before

    manifest, render_dir = _write_render(root, variant, "chapter-0001", plan)
    report_path = _write_book_compose_report(
        root,
        variant,
        profile_id,
        plan,
        manifest,
        render_dir / "render-manifest.toml",
        render_dir / "render-diagnostics.json",
    )
    before = snapshot()
    complete = inspect_bundle_composition(root, variant=variant)
    assert complete.state == "complete"
    assert complete.selected == complete.complete == ("chapter-0001",)
    assert complete.missing == complete.stale == ()
    assert snapshot() == before

    report = report_path.read_text(encoding="utf-8").replace(
        manifest.manifest_id or manifest.computed_id(), "sha256:" + "0" * 64
    )
    report_path.write_text(report, encoding="utf-8")
    before = snapshot()
    stale = inspect_bundle_composition(root, variant=variant, chapters="1")
    assert stale.state == "stale"
    assert stale.selected == stale.stale == ("chapter-0001",)
    assert stale.force_required
    assert stale.replacement_safe
    assert "upstream identity changed" in (stale.reason or "")
    assert snapshot() == before


def test_bundle_composition_inspector_does_not_trust_unowned_output(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "book.ssmd-bundle")
    variant, _, _ = _book_variant(root)
    output = root / "audiocompose" / variant
    output.mkdir(parents=True)
    (output / "chapter-0001.wav").write_bytes(b"unowned")

    result = inspect_bundle_composition(root, variant=variant)

    assert result.state == "error"
    assert result.selected == ("chapter-0001",)
    assert result.complete == ()
    assert "without an ownership report" in (result.reason or "")
