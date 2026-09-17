"""Canonical import boundary for the public :mod:`utterplan` API.

This module keeps internal renderer names short while targeting the documented
current class names directly.
"""
from __future__ import annotations

from utterplan import (
    PlannerConfig,
    PlanSegment,
    ProsodyDirective,
    UtterancePlan,
    UtterancePlanner,
)

Plan = UtterancePlan
Planner = UtterancePlanner

__all__ = [
    "Plan",
    "Planner",
    "PlannerConfig",
    "PlanSegment",
    "ProsodyDirective",
]
