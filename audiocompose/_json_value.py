from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .errors import AudioValidationError


def _snapshot(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AudioValidationError(f"{name} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        snapshot: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AudioValidationError(f"{name} object keys must be strings")
            snapshot[key] = _snapshot(item, name)
        return snapshot
    if isinstance(value, (list, tuple)):
        return [_snapshot(item, name) for item in value]
    raise AudioValidationError(f"{name} must contain only JSON values")


def snapshot_json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioValidationError(f"{name} must be an object")
    snapshot = _snapshot(value, name)
    return snapshot
