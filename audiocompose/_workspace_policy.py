"""Shared construction rules for workspace-level output policy overrides."""

from __future__ import annotations

from typing import Any

from .loudness import LoudnessPolicy
from .model import OutputPolicy


def build_workspace_output_policy(options: Any | None = None) -> OutputPolicy:
    """Map optional CLI-compatible workspace overrides onto ``OutputPolicy``."""
    defaults = OutputPolicy()
    settings = options
    loudness_defaults = defaults.loudness
    return OutputPolicy(
        sample_rate=(
            defaults.sample_rate
            if settings is None or settings.sample_rate is None
            else settings.sample_rate
        ),
        channels=(
            defaults.channels
            if settings is None or settings.channels is None
            else settings.channels
        ),
        clip_policy=(
            defaults.clip_policy
            if settings is None or settings.clip_policy is None
            else settings.clip_policy
        ),
        loudness=LoudnessPolicy(
            target_lufs=None if settings is None else settings.target_lufs,
            true_peak_ceiling_dbtp=(
                loudness_defaults.true_peak_ceiling_dbtp
                if settings is None or settings.true_peak_ceiling_dbtp is None
                else settings.true_peak_ceiling_dbtp
            ),
            peak_policy=(
                loudness_defaults.peak_policy
                if settings is None or settings.peak_policy is None
                else settings.peak_policy
            ),
            collect_metrics=(
                loudness_defaults.collect_metrics
                if settings is None or settings.collect_metrics is None
                else settings.collect_metrics
            ),
        ),
    )
