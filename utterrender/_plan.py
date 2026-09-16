"""Small compatibility boundary around the renamed :mod:`utterplan` package.

The MVP intentionally supports either the new ``UtterPlan``/``UtterPlanner`` naming
or the legacy class names if the project rename kept those public symbols.
"""
from __future__ import annotations

import utterplan as _utterplan

Plan = getattr(_utterplan, "UtterPlan", getattr(_utterplan, "TTSPlan", None))
Planner = getattr(_utterplan, "UtterPlanner", getattr(_utterplan, "TTSPlanner", None))
PlannerConfig = getattr(_utterplan, "PlannerConfig")
PlanSegment = getattr(_utterplan, "PlanSegment")
ProsodyDirective = getattr(_utterplan, "ProsodyDirective")

if Plan is None or Planner is None:  # pragma: no cover - import contract guard
    raise ImportError(
        "utterplan must export UtterPlan/UtterPlanner or legacy TTSPlan/TTSPlanner"
    )

__all__ = [
    "Plan",
    "Planner",
    "PlannerConfig",
    "PlanSegment",
    "ProsodyDirective",
]
