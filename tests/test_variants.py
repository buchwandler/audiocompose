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
    BundleVariantsResult,
    _derive_state,
    _inspect_bundle_variants,
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
        producer=Producer(name="voicerender", version="0.2.0"),
    ).with_identity()
    save_manifest(manifest, render_dir / "render-manifest.toml")
    diagnostics = {
        "schema": "voicerender.render-diagnostics.v1",
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
