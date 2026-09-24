from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .errors import AudioValidationError


def audiojob_schema(version: int) -> dict[str, Any]:
    """Load a packaged AudioJob JSON Schema by persisted schema version."""
    if type(version) is not int or version not in (1, 2):
        raise AudioValidationError("AudioJob schema version must be 1 or 2")
    resource = (
        files("audiocompose").joinpath("schemas").joinpath(f"audiojob-v{version}.schema.json")
    )
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioValidationError(f"cannot read AudioJob schema v{version}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AudioValidationError(f"AudioJob schema v{version} must contain a JSON object")
    return payload
