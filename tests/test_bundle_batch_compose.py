from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import audiocompose.bundle_batch as bundle_batch
from audiocompose.bundle import BundleComposeResult
from audiocompose.bundle_batch import compose_bundle_batch
from audiocompose.variants import BundleVariant, BundleVariantsResult


def _variant(
    name: str,
    state: str = "ready",
    *,
    ready: bool = True,
    plan_status: str | None = None,
    compose_status: str | None = None,
    validation_error: str | None = None,
) -> BundleVariant:
    inferred_plan = "invalid" if state == "invalid" else "valid"
    inferred_compose = {
        "ready": "not_composed",
        "compose_partial": "partial",
        "compose_failed": "failed",
        "stale": "stale",
        "unknown": "unknown",
        "complete": "complete",
    }.get(state, "not_composed")
    inferred_render = {
        "not_rendered": "not_rendered",
        "render_partial": "partial",
        "render_failed": "failed",
    }.get(state, "complete")
    return BundleVariant(
        name=name,
        profile_id="profile-1",
        plan_status=plan_status or inferred_plan,
        render_status=inferred_render,
        compose_status=compose_status or inferred_compose,
        state=state,
        unit_kind="chapters",
        total_units=1,
        rendered_units=1 if ready else 0,
        composed_units=1 if state == "complete" else 0,
        planned_speech=1,
        available_speech=1 if ready else 0,
        ready_to_compose=ready,
        validation_error=validation_error,
        render_error=None,
        compose_error=None,
    )


def _listing(tmp_path: Path, variants=(), *, kind: str = "book") -> BundleVariantsResult:
    root = tmp_path / "story.ssmd-bundle"
    return BundleVariantsResult(
        bundle=root,
        kind=kind,
        renderplan_root=root / "renderplan",
        voicerender_root=root / "voicerender",
        output_root=root / "audiocompose",
        variants=tuple(variants),
    )


def _result(exit_code: int = 0, *, value=None, kind: str = "book") -> BundleComposeResult:
    if value is None:
        value = SimpleNamespace(exit_code=exit_code, to_dict=lambda: {"exit_code": exit_code})
    return BundleComposeResult(kind=kind, value=value)


def _install_listing(monkeypatch, *listings):
    listing = Mock(side_effect=listings)
    monkeypatch.setattr(bundle_batch, "list_bundle_variants", listing)
    return listing


def test_default_selection_uses_validity_readiness_and_open_composition_state(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _listing(
        tmp_path,
        (
            _variant("invalid", "invalid", ready=True, validation_error="bad plan"),
            _variant("not-rendered", "not_rendered", ready=False),
            _variant("render-partial", "render_partial", ready=False),
            _variant("render-failed", "render_failed", ready=False),
            _variant("ready"),
            _variant("compose-partial", "compose_partial"),
            _variant("compose-failed", "compose_failed"),
            _variant("stale", "stale"),
            _variant("unknown", "unknown"),
            _variant("complete", "complete"),
        ),
    )
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result())
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    result = compose_bundle_batch(initial.bundle)

    assert [call.kwargs["variant"] for call in compose.call_args_list] == [
        "ready",
        "compose-partial",
        "compose-failed",
        "stale",
        "unknown",
    ]
    by_name = {variant.name: variant for variant in result.variants}
    assert by_name["invalid"].status == "invalid"
    assert by_name["invalid"].validation_error == "bad plan"
    assert by_name["not-rendered"].status == "skipped_not_ready"
    assert by_name["complete"].status == "skipped_complete"
    assert result.summary.discovered == 10
    assert result.summary.scheduled == 5
    assert result.summary.attempted == 5
    assert result.exit_code == 1


def test_force_selects_complete_variants_and_forwards_force_without_selecting_blocked(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _listing(
        tmp_path,
        (
            _variant("ready"),
            _variant("complete", "complete"),
            _variant("invalid", "invalid", ready=True),
            _variant("upstream", "render_partial", ready=False),
        ),
    )
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result())
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    result = compose_bundle_batch(initial.bundle, force=True)

    assert [call.kwargs["variant"] for call in compose.call_args_list] == ["ready", "complete"]
    assert all(call.kwargs["options"].force is True for call in compose.call_args_list)
    assert result.summary.invalid == 1
    assert result.summary.skipped_not_ready == 1
    assert result.exit_code == 1


def test_forwards_all_options_and_namespaces_audiojobs(tmp_path: Path, monkeypatch) -> None:
    initial = _listing(tmp_path, (_variant("profile-a"),))
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result())
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)
    callback = Mock()

    compose_bundle_batch(
        initial.bundle,
        chapters="2-3",
        pauses="pause.toml",
        renderplan_root="plans",
        voicerender_root="voice",
        output_root="audio",
        media_root="media",
        emphasis_policy="strict",
        sample_rate=48_000,
        clip_policy="warn",
        target_lufs=-18.0,
        true_peak_ceiling_dbtp=-1.0,
        peak_policy="reduce_gain",
        save_audiojobs=tmp_path / "jobs",
        force=True,
        fail_fast=True,
        on_progress=callback,
    )

    call = compose.call_args
    options = call.kwargs["options"]
    assert call.kwargs["on_progress"] is callback
    assert options.chapters == "2-3"
    assert options.pauses == "pause.toml"
    assert options.renderplan_root == "plans"
    assert options.voicerender_root == "voice"
    assert options.output_root == "audio"
    assert options.media_root == "media"
    assert options.emphasis_policy == "strict"
    assert options.sample_rate == 48_000
    assert options.clip_policy == "warn"
    assert options.target_lufs == -18.0
    assert options.true_peak_ceiling_dbtp == -1.0
    assert options.peak_policy == "reduce_gain"
    assert options.save_audiojobs == tmp_path / "jobs" / "profile-a"
    assert options.force is True
    assert options.fail_fast is True


@pytest.mark.parametrize("kind", ["book", "document"])
def test_save_audiojobs_namespace_applies_to_both_bundle_kinds(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    initial = _listing(tmp_path, (_variant("variant-a"),), kind=kind)
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result(kind=kind))
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    compose_bundle_batch(initial.bundle, save_audiojobs=tmp_path / "saved", chapters="all")

    assert compose.call_args.kwargs["options"].save_audiojobs == tmp_path / "saved" / "variant-a"


def test_noop_cases_do_not_call_single_variant_composer(tmp_path: Path, monkeypatch) -> None:
    for variants in (
        (),
        (_variant("complete", "complete"),),
        (_variant("waiting", "render_partial", ready=False),),
    ):
        initial = _listing(tmp_path, variants)
        _install_listing(monkeypatch, initial, initial)
        compose = Mock()
        monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

        result = compose_bundle_batch(initial.bundle)

        compose.assert_not_called()
        assert result.exit_code == 0


def test_invalid_variant_is_reported_while_valid_sibling_runs(tmp_path: Path, monkeypatch) -> None:
    initial = _listing(
        tmp_path,
        (
            _variant("bad", "invalid", validation_error="invalid candidate"),
            _variant("ready"),
        ),
    )
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result())
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    result = compose_bundle_batch(initial.bundle)

    compose.assert_called_once()
    assert compose.call_args.kwargs["variant"] == "ready"
    assert result.variants[0].validation_error == "invalid candidate"
    assert result.summary.invalid == 1
    assert result.exit_code == 1


@pytest.mark.parametrize("fail_fast", [False, True])
def test_failure_isolation_and_fail_fast_leave_unattempted_selected_variants(
    tmp_path: Path, monkeypatch, fail_fast: bool
) -> None:
    initial = _listing(tmp_path, tuple(_variant(name) for name in ("a", "b", "c")))
    _install_listing(monkeypatch, initial, initial)
    compose = Mock(side_effect=[RuntimeError("first failed"), _result(), _result()])
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    result = compose_bundle_batch(initial.bundle, fail_fast=fail_fast)

    assert compose.call_count == (1 if fail_fast else 3)
    assert result.variants[0].status == "failed"
    assert result.variants[0].error_message == "first failed"
    assert result.exit_code == 1
    if fail_fast:
        for outcome in result.variants[1:]:
            assert outcome.selected is True
            assert outcome.status == "not_attempted"
            assert outcome.attempt_status is None
        assert result.summary.not_attempted == 2
    else:
        assert [outcome.attempt_status for outcome in result.variants] == [
            "failed",
            "complete",
            "complete",
        ]


def test_result_error_prefers_report_then_failed_chapter_details(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _listing(tmp_path, (_variant("a"),))
    _install_listing(monkeypatch, initial, initial)
    chapter = SimpleNamespace(
        status="failed", error_code="chapter.failed", error_message="bad chapter"
    )
    value = SimpleNamespace(exit_code=2, report_error=None, outcomes=(chapter,), to_dict=lambda: {})
    monkeypatch.setattr(bundle_batch, "compose_bundle", Mock(return_value=_result(2, value=value)))

    result = compose_bundle_batch(initial.bundle)

    outcome = result.variants[0]
    assert outcome.attempt_status == "failed"
    assert outcome.error_code == "chapter.failed"
    assert outcome.error_message == "bad chapter"

    _install_listing(monkeypatch, initial, initial)
    value.report_error = {"code": "report.failed", "message": "report could not be saved"}
    monkeypatch.setattr(bundle_batch, "compose_bundle", Mock(return_value=_result(2, value=value)))
    report_result = compose_bundle_batch(initial.bundle)
    assert report_result.variants[0].error_code == "report.failed"
    assert report_result.variants[0].error_message == "report could not be saved"


def test_final_reinspection_derives_complete_and_partial_states(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _listing(tmp_path, (_variant("complete-me"), _variant("subset")))
    final = _listing(
        tmp_path,
        (_variant("complete-me", "complete"), _variant("subset", "compose_partial")),
    )
    _install_listing(monkeypatch, initial, final)
    monkeypatch.setattr(bundle_batch, "compose_bundle", Mock(return_value=_result()))

    result = compose_bundle_batch(initial.bundle)

    outcomes = {outcome.name: outcome for outcome in result.variants}
    assert outcomes["complete-me"].status == "complete"
    assert outcomes["complete-me"].final_state == "complete"
    assert outcomes["subset"].attempt_status == "complete"
    assert outcomes["subset"].status == "partial"
    assert outcomes["subset"].final_state == "compose_partial"
    assert result.summary.completed == 1
    assert result.summary.partial == 1


def test_final_inspection_failure_preserves_attempt_and_uses_batch_exit_code_two(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _listing(tmp_path, (_variant("a"),))
    _install_listing(monkeypatch, initial, RuntimeError("inspection failed"))
    monkeypatch.setattr(bundle_batch, "compose_bundle", Mock(return_value=_result()))

    result = compose_bundle_batch(initial.bundle)

    assert result.exit_code == 2
    assert result.error_code == "bundle.invalid"
    assert result.error_message == "inspection failed"
    assert result.variants[0].attempt_status == "complete"
    assert result.variants[0].result is not None
    assert result.summary.attempted == 1


def test_document_chapter_subset_is_a_batch_contract_error(tmp_path: Path, monkeypatch) -> None:
    initial = _listing(tmp_path, (_variant("a"),), kind="document")
    listing = _install_listing(monkeypatch, initial)
    compose = Mock()
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)

    result = compose_bundle_batch(initial.bundle, chapters="2")

    listing.assert_called_once()
    compose.assert_not_called()
    assert result.exit_code == 2
    assert result.error_code == "bundle.chapters_invalid"


def test_stale_output_never_auto_forces_overwrite(tmp_path: Path, monkeypatch) -> None:
    initial = _listing(tmp_path, (_variant("stale", "stale"),))
    _install_listing(monkeypatch, initial, initial)

    class OutputExistsError(Exception):
        code = "output.exists"

    compose = Mock(side_effect=OutputExistsError("use --force to replace unverified output"))
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)
    result = compose_bundle_batch(initial.bundle)

    assert compose.call_args.kwargs["options"].force is False
    assert result.variants[0].error_code == "output.exists"
    assert "use --force" in result.variants[0].error_message

    _install_listing(monkeypatch, initial, initial)
    compose = Mock(return_value=_result())
    monkeypatch.setattr(bundle_batch, "compose_bundle", compose)
    compose_bundle_batch(initial.bundle, force=True)
    assert compose.call_args.kwargs["options"].force is True


def test_batch_json_dict_has_stable_envelope_and_invalid_root_is_exit_two(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        bundle_batch,
        "list_bundle_variants",
        Mock(side_effect=ValueError("not a bundle")),
    )

    result = compose_bundle_batch(tmp_path / "bad.ssmd-bundle")

    payload = result.to_dict()
    assert payload["format"] == "audiocompose-bundle-batch-result"
    assert payload["version"] == 1
    assert payload["kind"] is None
    assert payload["error_code"] == "bundle.invalid"
    assert payload["exit_code"] == 2


def test_empty_bundle_discovery_does_not_create_stage_directories(tmp_path: Path) -> None:
    root = tmp_path / "empty.ssmd-bundle"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"format":"ssmdconvert.book","schema_version":1}', encoding="utf-8"
    )

    result = compose_bundle_batch(root)

    assert result.exit_code == 0
    assert result.summary.discovered == 0
    assert not (root / "renderplan").exists()
    assert not (root / "voicerender").exists()
    assert not (root / "audiocompose").exists()


def test_batch_sets_and_clears_variant_progress_context(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import call

    initial = _listing(tmp_path, (_variant("a"), _variant("b")))
    _install_listing(monkeypatch, initial, initial)
    monkeypatch.setattr(bundle_batch, "compose_bundle", Mock(return_value=_result()))
    callback = Mock()

    compose_bundle_batch(initial.bundle, on_progress=callback)

    assert callback.set_context.call_args_list == [call("1/2 a"), call("2/2 b"), call(None)]
