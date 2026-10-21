"""Neutral public facade for composing either supported SSMD bundle kind."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .bundle_contract import BundleComposeError, BundleKind, load_bundle_identity

ClipPolicy = Literal["clamp", "warn", "error"]
PeakPolicy = Literal["reduce_gain", "error"]
EmphasisPolicy = Literal["segment_fallback", "strict"]


def _translate_stage_error(exc: Exception) -> BundleComposeError | None:
    code = getattr(exc, "code", "")
    message = str(exc)
    lowered = message.lower()
    if code == "variant.not_found":
        return BundleComposeError("bundle.variant_unknown", f"unknown bundle variant: {message}")
    if "renderplan root" in lowered:
        return BundleComposeError(
            "bundle.renderplan_root_invalid", f"bundle RenderPlan root is invalid: {message}"
        )
    if "voicerender root" in lowered:
        return BundleComposeError(
            "bundle.voicerender_root_invalid", f"bundle VoiceRender root is invalid: {message}"
        )
    return None


@dataclass(frozen=True, slots=True)
class BundleComposeOptions:
    """Options shared by book and document bundle composition."""

    chapters: str = "all"
    pauses: str | Path | None = None
    renderplan_root: str | Path | None = None
    voicerender_root: str | Path | None = None
    output_root: str | Path | None = None
    media_root: str | Path | None = None
    emphasis_policy: EmphasisPolicy = "segment_fallback"
    sample_rate: int | None = None
    clip_policy: ClipPolicy | None = None
    target_lufs: float | None = None
    true_peak_ceiling_dbtp: float | None = None
    peak_policy: PeakPolicy | None = None
    save_audiojobs: str | Path | None = None
    force: bool = False
    fail_fast: bool = False


@dataclass(frozen=True, slots=True)
class BundleComposeResult:
    """Neutral wrapper around the unchanged kind-specific composition result."""

    kind: BundleKind
    value: Any

    @property
    def exit_code(self) -> int:
        return self.value.exit_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "audiocompose-bundle-result",
            "version": 1,
            "kind": self.kind,
            "result": self.value.to_dict(),
        }


def compose_bundle(
    bundle: str | Path,
    *,
    variant: str,
    options: BundleComposeOptions | None = None,
    on_progress: Callable[[Any], None] | None = None,
) -> BundleComposeResult:
    """Dispatch a root workspace to its existing composition implementation."""
    identity = load_bundle_identity(bundle)
    selected = options or BundleComposeOptions()
    value: Any
    if identity.kind == "book":
        from .book import BookComposeError, BookComposeOptions, compose_book

        try:
            value = compose_book(
                identity.root,
                variant=variant,
                chapters=selected.chapters,
                pause_policy=selected.pauses,
                options=BookComposeOptions(
                    sample_rate=selected.sample_rate,
                    clip_policy=selected.clip_policy,
                    target_lufs=selected.target_lufs,
                    true_peak_ceiling_dbtp=selected.true_peak_ceiling_dbtp,
                    peak_policy=selected.peak_policy,
                ),
                renderplan_root=selected.renderplan_root,
                voicerender_root=selected.voicerender_root,
                output_root=selected.output_root,
                media_root=selected.media_root,
                save_audiojobs=selected.save_audiojobs,
                force=selected.force,
                fail_fast=selected.fail_fast,
                emphasis_policy=selected.emphasis_policy,
                on_progress=on_progress,
            )
        except BookComposeError as exc:
            translated = _translate_stage_error(exc)
            if translated is not None:
                raise translated from exc
            raise
    else:
        if selected.chapters != "all":
            from .errors import AudioComposeError

            raise AudioComposeError("document bundles support only --chapters all")
        from .document import DocumentComposeError, DocumentComposeOptions, compose_document
        from .document_contract import DOCUMENT_ID

        save_audiojob = (
            Path(selected.save_audiojobs).expanduser() / f"{DOCUMENT_ID}.audiojob"
            if selected.save_audiojobs is not None
            else None
        )
        try:
            value = compose_document(
                identity.root,
                variant=variant,
                pause_policy=selected.pauses,
                options=DocumentComposeOptions(
                    sample_rate=selected.sample_rate,
                    clip_policy=selected.clip_policy,
                    target_lufs=selected.target_lufs,
                    true_peak_ceiling_dbtp=selected.true_peak_ceiling_dbtp,
                    peak_policy=selected.peak_policy,
                ),
                renderplan_root=selected.renderplan_root,
                voicerender_root=selected.voicerender_root,
                output_root=selected.output_root,
                media_root=selected.media_root,
                save_audiojob=save_audiojob,
                force=selected.force,
                emphasis_policy=selected.emphasis_policy,
                on_progress=on_progress,
            )
        except DocumentComposeError as exc:
            translated = _translate_stage_error(exc)
            if translated is not None:
                raise translated from exc
            raise
    return BundleComposeResult(kind=identity.kind, value=value)
