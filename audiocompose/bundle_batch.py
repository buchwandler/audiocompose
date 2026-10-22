"""State-aware orchestration for composing ready SSMD bundle variants."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .bundle import (
    BundleComposeOptions,
    BundleComposeResult,
    ClipPolicy,
    EmphasisPolicy,
    PeakPolicy,
    compose_bundle,
)
from .bundle_contract import BundleKind
from .variants import (
    BundleVariant,
    ComposeStatus,
    PlanStatus,
    RenderStatus,
    VariantState,
    list_bundle_variants,
)

BatchVariantStatus = Literal[
    "complete",
    "partial",
    "failed",
    "skipped_complete",
    "skipped_not_ready",
    "invalid",
    "not_attempted",
]
BatchAttemptStatus = Literal["complete", "failed"]


@dataclass(frozen=True, slots=True)
class BundleVariantComposeOutcome:
    """Initial, attempted, and final state for one discovered variant."""

    name: str
    initial_state: VariantState
    initial_render_status: RenderStatus
    initial_compose_status: ComposeStatus
    ready_to_compose: bool
    plan_status: PlanStatus
    selected: bool
    status: BatchVariantStatus
    attempt_status: BatchAttemptStatus | None = None
    final_state: VariantState | None = None
    final_render_status: RenderStatus | None = None
    final_compose_status: ComposeStatus | None = None
    exit_code: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    validation_error: str | None = None
    render_error: str | None = None
    compose_error: str | None = None
    result: BundleComposeResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_state": self.initial_state,
            "initial_render_status": self.initial_render_status,
            "initial_compose_status": self.initial_compose_status,
            "ready_to_compose": self.ready_to_compose,
            "plan_status": self.plan_status,
            "selected": self.selected,
            "status": self.status,
            "attempt_status": self.attempt_status,
            "final_state": self.final_state,
            "final_render_status": self.final_render_status,
            "final_compose_status": self.final_compose_status,
            "exit_code": self.exit_code,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "validation_error": self.validation_error,
            "render_error": self.render_error,
            "compose_error": self.compose_error,
            "result": self.result.to_dict() if self.result is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BundleBatchComposeSummary:
    """Aggregate work counts for a batch composition run."""

    discovered: int
    compose_ready: int
    scheduled: int
    attempted: int
    completed: int
    partial: int
    failed: int
    skipped_complete: int
    skipped_not_ready: int
    invalid: int
    not_attempted: int

    def to_dict(self) -> dict[str, int]:
        return {
            "discovered": self.discovered,
            "compose_ready": self.compose_ready,
            "scheduled": self.scheduled,
            "attempted": self.attempted,
            "completed": self.completed,
            "partial": self.partial,
            "failed": self.failed,
            "skipped_complete": self.skipped_complete,
            "skipped_not_ready": self.skipped_not_ready,
            "invalid": self.invalid,
            "not_attempted": self.not_attempted,
        }


@dataclass(frozen=True, slots=True)
class BundleBatchComposeResult:
    """Typed result for batch composition, including batch-level failures."""

    bundle: Path
    kind: BundleKind | None
    renderplan_root: Path | None
    voicerender_root: Path | None
    output_root: Path | None
    force: bool
    variants: tuple[BundleVariantComposeOutcome, ...]
    summary: BundleBatchComposeSummary
    exit_code: int
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "audiocompose-bundle-batch-result",
            "version": 1,
            "kind": self.kind,
            "bundle": str(self.bundle),
            "renderplan_root": (
                str(self.renderplan_root) if self.renderplan_root is not None else None
            ),
            "voicerender_root": (
                str(self.voicerender_root) if self.voicerender_root is not None else None
            ),
            "output_root": str(self.output_root) if self.output_root is not None else None,
            "force": self.force,
            "summary": self.summary.to_dict(),
            "variants": [variant.to_dict() for variant in self.variants],
            "error_code": self.error_code,
            "error_message": self.error_message,
            "exit_code": self.exit_code,
        }


def _empty_summary() -> BundleBatchComposeSummary:
    return BundleBatchComposeSummary(
        discovered=0,
        compose_ready=0,
        scheduled=0,
        attempted=0,
        completed=0,
        partial=0,
        failed=0,
        skipped_complete=0,
        skipped_not_ready=0,
        invalid=0,
        not_attempted=0,
    )


def _summarize(
    variants: tuple[BundleVariantComposeOutcome, ...],
    *,
    discovered: int,
    compose_ready: int,
    scheduled: int,
) -> BundleBatchComposeSummary:
    return BundleBatchComposeSummary(
        discovered=discovered,
        compose_ready=compose_ready,
        scheduled=scheduled,
        attempted=sum(variant.attempt_status is not None for variant in variants),
        completed=sum(variant.status == "complete" for variant in variants),
        partial=sum(variant.status == "partial" for variant in variants),
        failed=sum(variant.status == "failed" for variant in variants),
        skipped_complete=sum(variant.status == "skipped_complete" for variant in variants),
        skipped_not_ready=sum(variant.status == "skipped_not_ready" for variant in variants),
        invalid=sum(variant.status == "invalid" for variant in variants),
        not_attempted=sum(variant.status == "not_attempted" for variant in variants),
    )


def _initial_outcome(variant: BundleVariant, *, force: bool) -> BundleVariantComposeOutcome:
    invalid = variant.plan_status == "invalid" or variant.state == "invalid"
    selected = (
        not invalid and variant.ready_to_compose and (force or variant.compose_status != "complete")
    )
    if invalid:
        status: BatchVariantStatus = "invalid"
    elif not variant.ready_to_compose:
        status = "skipped_not_ready"
    elif variant.compose_status == "complete" and not force:
        status = "skipped_complete"
    else:
        status = "not_attempted"
    return BundleVariantComposeOutcome(
        name=variant.name,
        initial_state=variant.state,
        initial_render_status=variant.render_status,
        initial_compose_status=variant.compose_status,
        ready_to_compose=variant.ready_to_compose,
        plan_status=variant.plan_status,
        selected=selected,
        status=status,
        validation_error=variant.validation_error,
        render_error=variant.render_error,
        compose_error=variant.compose_error,
    )


def _exception_details(exc: Exception) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    return (
        code if isinstance(code, str) and code else "compose.error",
        str(exc) or type(exc).__name__,
    )


def _result_error_details(result: BundleComposeResult) -> tuple[str, str]:
    value = result.value
    report_error = getattr(value, "report_error", None)
    if isinstance(report_error, Mapping):
        code = report_error.get("code")
        message = report_error.get("message")
        return (
            code if isinstance(code, str) and code else "compose.failed",
            message if isinstance(message, str) and message else "composition report failed",
        )
    for outcome in getattr(value, "outcomes", ()):
        if getattr(outcome, "status", None) == "failed":
            code = getattr(outcome, "error_code", None)
            message = getattr(outcome, "error_message", None)
            return (
                code if isinstance(code, str) and code else "compose.failed",
                message if isinstance(message, str) and message else "chapter composition failed",
            )
    return "compose.failed", f"composer exited with status {result.exit_code}."


def _options_for_variant(
    *,
    variant: str,
    chapters: str,
    pauses: str | Path | None,
    renderplan_root: str | Path | None,
    voicerender_root: str | Path | None,
    output_root: str | Path | None,
    media_root: str | Path | None,
    emphasis_policy: EmphasisPolicy,
    sample_rate: int | None,
    clip_policy: ClipPolicy | None,
    target_lufs: float | None,
    true_peak_ceiling_dbtp: float | None,
    peak_policy: PeakPolicy | None,
    save_audiojobs: str | Path | None,
    force: bool,
    fail_fast: bool,
) -> BundleComposeOptions:
    variant_jobs = (
        Path(save_audiojobs).expanduser() / variant if save_audiojobs is not None else None
    )
    return BundleComposeOptions(
        chapters=chapters,
        pauses=pauses,
        renderplan_root=renderplan_root,
        voicerender_root=voicerender_root,
        output_root=output_root,
        media_root=media_root,
        emphasis_policy=emphasis_policy,
        sample_rate=sample_rate,
        clip_policy=clip_policy,
        target_lufs=target_lufs,
        true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
        peak_policy=peak_policy,
        save_audiojobs=variant_jobs,
        force=force,
        fail_fast=fail_fast,
    )


def compose_bundle_batch(
    bundle: str | Path,
    *,
    chapters: str = "all",
    pauses: str | Path | None = None,
    renderplan_root: str | Path | None = None,
    voicerender_root: str | Path | None = None,
    output_root: str | Path | None = None,
    media_root: str | Path | None = None,
    emphasis_policy: EmphasisPolicy = "segment_fallback",
    sample_rate: int | None = None,
    clip_policy: ClipPolicy | None = None,
    target_lufs: float | None = None,
    true_peak_ceiling_dbtp: float | None = None,
    peak_policy: PeakPolicy | None = None,
    save_audiojobs: str | Path | None = None,
    force: bool = False,
    fail_fast: bool = False,
    on_progress: Callable[[Any], None] | None = None,
) -> BundleBatchComposeResult:
    """Compose valid, compose-ready variants with outstanding composition work."""
    bundle_path = Path(bundle).expanduser()
    try:
        listing = list_bundle_variants(
            bundle,
            renderplan_root=renderplan_root,
            voicerender_root=voicerender_root,
            output_root=output_root,
        )
    except Exception as exc:
        code, message = _exception_details(exc)
        if code == "compose.error":
            code = "bundle.invalid"
        return BundleBatchComposeResult(
            bundle=bundle_path,
            kind=None,
            renderplan_root=None,
            voicerender_root=None,
            output_root=None,
            force=force,
            variants=(),
            summary=_empty_summary(),
            exit_code=2,
            error_code=code,
            error_message=message,
        )

    if listing.kind == "document" and chapters.strip().casefold() != "all":
        return BundleBatchComposeResult(
            bundle=listing.bundle,
            kind=listing.kind,
            renderplan_root=listing.renderplan_root,
            voicerender_root=listing.voicerender_root,
            output_root=listing.output_root,
            force=force,
            variants=(),
            summary=_empty_summary(),
            exit_code=2,
            error_code="bundle.chapters_invalid",
            error_message="--chapters must be 'all' for a document bundle",
        )

    outcomes = [_initial_outcome(variant, force=force) for variant in listing.variants]
    scheduled_names = [outcome.name for outcome in outcomes if outcome.selected]
    compose_ready = sum(
        variant.plan_status == "valid" and variant.ready_to_compose for variant in listing.variants
    )
    failed = False
    outcome_index = {outcome.name: index for index, outcome in enumerate(outcomes)}

    for index, name in enumerate(scheduled_names, start=1):
        if fail_fast and failed:
            break
        context_setter = getattr(on_progress, "set_context", None)
        if callable(context_setter):
            context_setter(f"{index}/{len(scheduled_names)} {name}")
        options = _options_for_variant(
            variant=name,
            chapters=chapters,
            pauses=pauses,
            renderplan_root=renderplan_root,
            voicerender_root=voicerender_root,
            output_root=output_root,
            media_root=media_root,
            emphasis_policy=emphasis_policy,
            sample_rate=sample_rate,
            clip_policy=clip_policy,
            target_lufs=target_lufs,
            true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
            peak_policy=peak_policy,
            save_audiojobs=save_audiojobs,
            force=force,
            fail_fast=fail_fast,
        )
        position = outcome_index[name]
        try:
            result = compose_bundle(
                listing.bundle,
                variant=name,
                options=options,
                on_progress=on_progress,
            )
        except Exception as exc:
            code, message = _exception_details(exc)
            outcomes[position] = replace(
                outcomes[position],
                status="failed",
                attempt_status="failed",
                exit_code=1,
                error_code=code,
                error_message=message,
            )
            failed = True
            continue

        if result.exit_code != 0:
            code, message = _result_error_details(result)
            outcomes[position] = replace(
                outcomes[position],
                status="failed",
                attempt_status="failed",
                exit_code=result.exit_code,
                error_code=code,
                error_message=message,
                result=result,
            )
            failed = True
        else:
            outcomes[position] = replace(
                outcomes[position],
                status="partial",
                attempt_status="complete",
                exit_code=result.exit_code,
                result=result,
            )

    context_setter = getattr(on_progress, "set_context", None)
    if callable(context_setter):
        context_setter(None)

    try:
        final_listing = list_bundle_variants(
            bundle,
            renderplan_root=renderplan_root,
            voicerender_root=voicerender_root,
            output_root=output_root,
        )
    except Exception as exc:
        code, message = _exception_details(exc)
        summary = _summarize(
            tuple(outcomes),
            discovered=len(listing.variants),
            compose_ready=compose_ready,
            scheduled=len(scheduled_names),
        )
        return BundleBatchComposeResult(
            bundle=listing.bundle,
            kind=listing.kind,
            renderplan_root=listing.renderplan_root,
            voicerender_root=listing.voicerender_root,
            output_root=listing.output_root,
            force=force,
            variants=tuple(outcomes),
            summary=summary,
            exit_code=2,
            error_code=code if code != "compose.error" else "bundle.invalid",
            error_message=message,
        )

    final_variants = {variant.name: variant for variant in final_listing.variants}
    for position, outcome in enumerate(outcomes):
        final = final_variants.get(outcome.name)
        if final is None:
            continue
        status = outcome.status
        if outcome.attempt_status == "complete":
            status = "complete" if final.state == "complete" else "partial"
        outcomes[position] = replace(
            outcome,
            status=status,
            final_state=final.state,
            final_render_status=final.render_status,
            final_compose_status=final.compose_status,
            validation_error=final.validation_error,
            render_error=final.render_error,
            compose_error=final.compose_error,
        )

    final_outcomes = tuple(outcomes)
    summary = _summarize(
        final_outcomes,
        discovered=len(listing.variants),
        compose_ready=compose_ready,
        scheduled=len(scheduled_names),
    )
    exit_code = 1 if summary.failed or summary.invalid or summary.not_attempted else 0
    return BundleBatchComposeResult(
        bundle=listing.bundle,
        kind=listing.kind,
        renderplan_root=listing.renderplan_root,
        voicerender_root=listing.voicerender_root,
        output_root=listing.output_root,
        force=force,
        variants=final_outcomes,
        summary=summary,
        exit_code=exit_code,
    )


__all__ = [
    "BatchAttemptStatus",
    "BatchVariantStatus",
    "BundleBatchComposeResult",
    "BundleBatchComposeSummary",
    "BundleVariantComposeOutcome",
    "compose_bundle_batch",
]
