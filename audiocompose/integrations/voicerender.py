"""Public UtterPlan/RenderPlan bridge for rendered chapter materialization.

VoiceRender produces artifacts against the shared RenderPlan contracts. This
adapter loads the upstream source plan through UtterPlan's public API, checks its
identity against the RenderPlan, and delegates artifact validation and AudioJob
construction to AudioCompose's existing RenderPlan materializer.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from ..errors import HandoffValidationError, IntegrationError, PausePolicyError
from ..model import OutputPolicy, Silence
from .renderplan import (
    MediaResolver,
    _ValidatedControlOwnership,
    materialize_renderplan_job,
)
from .renderplan_metadata import RENDER_DIAGNOSTICS_FILENAME

EMPHASIS_COMPOSITION_CAPABILITY = "audiocompose.emphasis.gain.v1"
_VOICERENDER_DIAGNOSTICS_SCHEMA = "voicerender.render-diagnostics.v1"
_VOICERENDER_EMPHASIS_POLICIES = frozenset({"reject", "defer-to-compose"})


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_render_diagnostics(path: Path) -> tuple[dict[str, Any], str]:
    location = Path(path).expanduser()
    try:
        info = location.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise HandoffValidationError(
                f"VoiceRender diagnostics must not be a symlink: {location}"
            )
        if not stat.S_ISREG(info.st_mode):
            raise HandoffValidationError(
                f"VoiceRender diagnostics must be a regular file: {location}"
            )
        raw = location.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except HandoffValidationError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise HandoffValidationError(
            f"Cannot read VoiceRender diagnostics {location}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise HandoffValidationError("VoiceRender diagnostics JSON root must be an object")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return payload, digest


def _validate_voicerender_producer(manifest: Any) -> None:
    producer = getattr(manifest, "producer", None)
    if getattr(producer, "name", None) != "voicerender":
        raise HandoffValidationError(
            "VoiceRender diagnostics can only be trusted with a RenderManifest produced by 'voicerender'"
        )
    version = getattr(producer, "version", None)
    if version is None:
        return
    if not isinstance(version, str):
        raise HandoffValidationError("RenderManifest VoiceRender producer version must be a string")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-][0-9A-Za-z.+-]+)?", version)
    if match is None or (int(match.group(1)), int(match.group(2))) != (0, 2):
        raise HandoffValidationError(
            f"VoiceRender producer version {version!r} is incompatible; AudioCompose v0.3 supports 0.2.x"
        )


def _validate_render_diagnostics(
    plan: Any, diagnostics: Mapping[str, Any], diagnostics_sha256: str
) -> _ValidatedControlOwnership:
    if diagnostics.get("schema") != _VOICERENDER_DIAGNOSTICS_SCHEMA:
        raise HandoffValidationError(
            f"Unsupported VoiceRender diagnostics schema {diagnostics.get('schema')!r}"
        )
    plan_id = plan.plan_id or plan.computed_id()
    if diagnostics.get("render_plan_id") != plan_id:
        raise HandoffValidationError(
            f"VoiceRender diagnostics refer to RenderPlan {diagnostics.get('render_plan_id')!r}, "
            f"but the selected plan is {plan_id!r}; diagnostics are stale"
        )
    emphasis_policy = diagnostics.get("emphasis_policy")
    if (
        not isinstance(emphasis_policy, str)
        or emphasis_policy not in _VOICERENDER_EMPHASIS_POLICIES
    ):
        raise HandoffValidationError(
            f"Unsupported VoiceRender emphasis_policy {emphasis_policy!r} in diagnostics"
        )
    pending = diagnostics.get("pending_composition")
    if not isinstance(pending, bool):
        raise HandoffValidationError(
            "VoiceRender diagnostics pending_composition must be a boolean"
        )
    deferred = diagnostics.get("deferred_controls")
    if not isinstance(deferred, list):
        raise HandoffValidationError("VoiceRender diagnostics deferred_controls must be a list")

    renderplan = importlib.import_module("renderplan")
    speeches: dict[tuple[int, str], Any] = {}
    emphasis_counts: dict[tuple[int, str], int] = {}
    realized_controls: set[str] = set()
    for item in plan.flow:
        if not isinstance(item, renderplan.SpeechItem):
            continue
        segment_id = item.segment_id or item.computed_id()
        occurrence = (item.number, segment_id)
        if occurrence in speeches:
            raise HandoffValidationError(f"RenderPlan repeats speech occurrence {occurrence!r}")
        speeches[occurrence] = item
        controls = item.controls
        if not isinstance(controls, (tuple, list)):
            raise HandoffValidationError(
                f"speech {item.number} control range [unknown]: controls must be a sequence"
            )
        for control in controls:
            name = getattr(control, "name", None)
            if name == "emphasis":
                emphasis_counts[occurrence] = emphasis_counts.get(occurrence, 0) + 1
            elif isinstance(name, str) and name:
                realized_controls.add(name)

    observed_counts: dict[tuple[int, str], int] = {}
    for entry in deferred:
        if not isinstance(entry, dict):
            raise HandoffValidationError("Each VoiceRender deferred control must be an object")
        number = entry.get("segment_number")
        segment_id = entry.get("segment_id")
        capability = entry.get("capability")
        count = entry.get("count")
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise HandoffValidationError(
                "VoiceRender deferred segment_number must be a positive integer"
            )
        if not isinstance(segment_id, str) or not segment_id:
            raise HandoffValidationError(
                "VoiceRender deferred segment_id must be a non-empty string"
            )
        if not isinstance(capability, str) or capability != EMPHASIS_COMPOSITION_CAPABILITY:
            raise HandoffValidationError(
                f"VoiceRender deferred capability {capability!r} is not supported by AudioCompose v0.3.0"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise HandoffValidationError("VoiceRender deferred count must be a positive integer")
        occurrence = (number, segment_id)
        if occurrence not in speeches:
            raise HandoffValidationError(
                f"VoiceRender deferred control refers to unknown speech occurrence {occurrence!r}"
            )
        if occurrence in observed_counts:
            raise HandoffValidationError(
                f"VoiceRender diagnostics repeat deferred speech occurrence {occurrence!r}"
            )
        expected_count = emphasis_counts.get(occurrence, 0)
        if count != expected_count:
            raise HandoffValidationError(
                f"VoiceRender reported {count} deferred emphasis controls for speech occurrence "
                f"{occurrence!r}, but the RenderPlan contains {expected_count}"
            )
        observed_counts[occurrence] = count

    if pending is not bool(deferred):
        raise HandoffValidationError(
            "VoiceRender diagnostics pending_composition contradicts deferred_controls"
        )
    if deferred and emphasis_policy != "defer-to-compose":
        raise HandoffValidationError(
            "VoiceRender diagnostics report deferred controls without defer-to-compose policy"
        )
    if emphasis_counts and emphasis_policy != "defer-to-compose":
        raise HandoffValidationError(
            "RenderPlan emphasis controls require VoiceRender emphasis_policy='defer-to-compose'"
        )
    if emphasis_policy == "defer-to-compose" and observed_counts != emphasis_counts:
        missing = sorted(set(emphasis_counts).difference(observed_counts))
        raise HandoffValidationError(
            f"VoiceRender diagnostics omit deferred emphasis occurrence(s) {missing!r}"
        )

    return _ValidatedControlOwnership(
        producer="voicerender",
        plan_id=plan_id,
        realized_controls=frozenset(realized_controls),
        diagnostics_sha256=diagnostics_sha256,
        diagnostics_schema=_VOICERENDER_DIAGNOSTICS_SCHEMA,
        emphasis_policy=emphasis_policy,
        deferred_capabilities=tuple(sorted({entry["capability"] for entry in deferred})),
    )


def _flow_plan_type() -> Any:
    """Return the public UtterPlan FlowPlan type or explain the missing extra."""
    try:
        return importlib.import_module("utterplan").FlowPlan
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
        raise IntegrationError(
            "UtterPlan is required for VoiceRender book handoff materialization. "
            "Install audiocompose[voicerender]."
        ) from exc


def load_source_plan(path: str | Path) -> Any:
    """Load a schema-v5+ UtterPlan through its exported public FlowPlan API."""
    flow_plan_type = _flow_plan_type()
    location = Path(path).expanduser()
    if location.is_symlink():
        raise HandoffValidationError(f"UtterPlan source must not be a symlink: {location}")
    try:
        return flow_plan_type.load(location)
    except Exception as exc:
        raise HandoffValidationError(f"Cannot load UtterPlan source {location}: {exc}") from exc


PAUSE_POLICY_SCHEMA = "audiocompose.pause-policy.v1"
PAUSE_CLASSES = frozenset(
    {
        "none",
        "x-weak",
        "weak",
        "medium",
        "strong",
        "x-strong",
        "clause",
        "sentence",
        "paragraph",
        "parenthetical",
        "voice_change",
    }
)
_DURATION = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(ms|s)$")
_POLICY_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


def _duration_seconds(value: Any, where: str) -> float:
    if not isinstance(value, str):
        raise PausePolicyError(f"{where} must be a duration string such as '320ms'")
    match = _DURATION.fullmatch(value)
    if match is None:
        raise PausePolicyError(f"{where} has invalid duration {value!r}; use ms or s")
    amount = float(value[: -len(match.group(1))])
    seconds = amount / 1000.0 if match.group(1) == "ms" else amount
    if not math.isfinite(seconds) or seconds < 0:
        raise PausePolicyError(f"{where} must be a finite duration >= 0")
    return seconds


def _normalized_pause_policy(
    default: Mapping[str, float], roles: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    return {
        "schema": PAUSE_POLICY_SCHEMA,
        "merge": "max",
        "default": dict(sorted(default.items())),
        "role": {name: dict(sorted(table.items())) for name, table in sorted(roles.items())},
    }


def _pause_policy_id(document: Mapping[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_pause_table(values: Mapping[str, Any], where: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise PausePolicyError(f"{where} must be a pause-class table")
    unknown = set(values).difference(PAUSE_CLASSES)
    if unknown:
        raise PausePolicyError(f"{where} has unknown pause class(es): {sorted(unknown)}")
    normalized: dict[str, float] = {}
    for name, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise PausePolicyError(f"{where}.{name} must be finite seconds >= 0")
        normalized[name] = float(value)
    if normalized.get("none", 0.0) != 0.0:
        raise PausePolicyError(f"{where}.none must resolve to zero")
    return normalized


@dataclass(frozen=True, slots=True)
class PausePolicy:
    """Strict AudioCompose-owned mapping from symbolic pauses to durations."""

    default: Mapping[str, float]
    roles: Mapping[str, Mapping[str, float]]
    policy_id: str
    merge: str = "max"

    def __post_init__(self) -> None:
        if self.merge != "max":
            raise PausePolicyError("PausePolicy merge must be 'max'")
        default = _validate_pause_table(self.default, "default")
        if not isinstance(self.roles, Mapping):
            raise PausePolicyError("role must be a table of RenderPlan-role overrides")
        roles: dict[str, Mapping[str, float]] = {}
        for role, values in self.roles.items():
            if not isinstance(role, str) or not role.strip():
                raise PausePolicyError("role override names must be non-empty strings")
            roles[role] = MappingProxyType(_validate_pause_table(values, f"role.{role}"))
        if not isinstance(self.policy_id, str) or _POLICY_ID.fullmatch(self.policy_id) is None:
            raise PausePolicyError("PausePolicy policy_id must be a canonical SHA-256 identity")
        expected = _pause_policy_id(_normalized_pause_policy(default, roles))
        if self.policy_id != expected:
            raise PausePolicyError("PausePolicy policy_id does not match its normalized contents")
        object.__setattr__(self, "default", MappingProxyType(default))
        object.__setattr__(self, "roles", MappingProxyType(roles))

    @classmethod
    def _create(
        cls, default: Mapping[str, float], roles: Mapping[str, Mapping[str, float]]
    ) -> PausePolicy:
        normalized = _normalized_pause_policy(default, roles)
        return cls(default, roles, _pause_policy_id(normalized))

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> PausePolicy:
        if not isinstance(document, Mapping):
            raise PausePolicyError("Pause policy must be a mapping")
        unknown = set(document).difference({"schema", "merge", "default", "role"})
        if unknown:
            raise PausePolicyError(f"Pause policy has unknown root field(s): {sorted(unknown)}")
        if document.get("schema") != PAUSE_POLICY_SCHEMA:
            raise PausePolicyError(f"Expected pause policy schema {PAUSE_POLICY_SCHEMA!r}")
        if document.get("merge") != "max":
            raise PausePolicyError("Pause policy merge must be 'max'")
        raw_default = document.get("default")
        if not isinstance(raw_default, Mapping):
            raise PausePolicyError("default must be a pause-class table")
        default = {
            name: _duration_seconds(value, f"default.{name}") for name, value in raw_default.items()
        }
        raw_roles = document.get("role", {})
        if not isinstance(raw_roles, Mapping):
            raise PausePolicyError("role must be a table of RenderPlan-role overrides")
        roles: dict[str, dict[str, float]] = {}
        for role, values in raw_roles.items():
            if not isinstance(role, str) or not role.strip():
                raise PausePolicyError("role override names must be non-empty strings")
            if not isinstance(values, Mapping):
                raise PausePolicyError(f"role.{role} must be a pause-class table")
            roles[role] = {
                name: _duration_seconds(value, f"role.{role}.{name}")
                for name, value in values.items()
            }
        return cls._create(default, roles)

    @classmethod
    def from_render_profile(cls, profile: Any) -> PausePolicy:
        pauses = getattr(profile, "pauses", None)
        mode = getattr(pauses, "mode", None)
        durations_ms = getattr(pauses, "durations_ms", None)
        if mode not in {"manual", "none"} or not isinstance(durations_ms, Mapping):
            raise PausePolicyError("RenderProfile snapshot has an unsupported pause configuration")
        unknown = set(durations_ms).difference(PAUSE_CLASSES)
        if unknown:
            raise PausePolicyError(
                f"RenderProfile contains pause classes that AudioCompose cannot resolve: {sorted(unknown)}"
            )
        if mode == "none":
            default = {name: 0.0 for name in PAUSE_CLASSES}
        else:
            default = {}
        for name, duration in durations_ms.items():
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                raise PausePolicyError(
                    f"RenderProfile pauses.{name} must be an integer millisecond duration"
                )
            default[name] = duration / 1000.0
        return cls._create(default, {})

    @classmethod
    def load(cls, path: str | Path) -> PausePolicy:
        location = Path(path).expanduser()
        if location.is_symlink():
            raise PausePolicyError(f"Pause policy must not be a symlink: {location}")
        try:
            try:
                tomllib = importlib.import_module("tomllib")
            except ImportError:  # pragma: no cover - Python 3.10
                tomllib = importlib.import_module("tomli")
            document = tomllib.loads(location.read_text(encoding="utf-8"))
        except PausePolicyError:
            raise
        except (ImportError, OSError, UnicodeError, ValueError) as exc:
            raise PausePolicyError(f"Cannot read pause policy {location}: {exc}") from exc
        return cls.from_mapping(document)

    def to_dict(self) -> dict[str, Any]:
        return {
            **_normalized_pause_policy(self.default, self.roles),
            "policy_id": self.policy_id,
        }

    def resolve(self, intent: Any, *, role: str, where: str) -> float:
        if intent is None:
            return 0.0
        if callable(getattr(intent, "to_dict", None)):
            intent = intent.to_dict()
        if isinstance(intent, Mapping):
            if set(intent) != {"type", "time"} or intent.get("type") != "timed":
                raise PausePolicyError(f"{where}: timed pause must contain exactly type and time")
            return _duration_seconds(intent["time"], f"{where}.time")
        if not isinstance(intent, str):
            raise PausePolicyError(f"{where}: invalid pause intent {intent!r}")
        if intent not in PAUSE_CLASSES:
            raise PausePolicyError(f"{where}: unknown symbolic pause {intent!r}")
        if intent == "none":
            return 0.0
        duration = self.roles.get(role, {}).get(intent, self.default.get(intent))
        if duration is None:
            raise PausePolicyError(
                f"{where}: no duration configured for pause {intent!r} (RenderPlan role {role!r})"
            )
        return float(duration)


def load_pause_policy(value: PausePolicy | str | Path | Mapping[str, Any]) -> PausePolicy:
    if isinstance(value, PausePolicy):
        return value
    if isinstance(value, (str, Path)):
        return PausePolicy.load(value)
    if isinstance(value, Mapping):
        return PausePolicy.from_mapping(value)
    raise PausePolicyError("pause_policy must be a policy path, mapping, or PausePolicy")


def _validate_source_pauses(source: Any, policy: PausePolicy) -> set[str]:
    """Validate pause syntax without interpreting logical voices as RenderPlan roles.

    Role-specific duration selection happens only from RenderPlan SpeechItem.role
    in ``_apply_pause_policy``; source directives are not policy-table keys.
    """
    resolved: set[str] = set()
    for unit_index, unit in enumerate(getattr(source, "flow", ())):
        for segment_index, segment in enumerate(getattr(unit, "segments", ())):
            for side in ("pause_before", "pause_after"):
                intent = getattr(segment, side, None)
                if intent is None:
                    continue
                value = intent.to_dict() if callable(getattr(intent, "to_dict", None)) else intent
                where = f"flow[{unit_index}].segments[{segment_index}].{side}"
                if isinstance(value, str):
                    if value not in PAUSE_CLASSES:
                        raise PausePolicyError(f"{where}: unknown symbolic pause {value!r}")
                    if (
                        value != "none"
                        and value not in policy.default
                        and not any(value in table for table in policy.roles.values())
                    ):
                        raise PausePolicyError(
                            f"{where}: no duration configured for pause {value!r} in the default or role tables"
                        )
                    resolved.add(value)
                else:
                    policy.resolve(value, role="", where=where)
                    if isinstance(value, Mapping) and isinstance(value.get("type"), str):
                        resolved.add(value["type"])
    return resolved


def _render_pause_role(flow: tuple[Any, ...], index: int, default_role: str) -> str:
    renderplan = importlib.import_module("renderplan")
    for candidate in (*range(index - 1, -1, -1), *range(index + 1, len(flow))):
        item = flow[candidate]
        if isinstance(item, renderplan.SpeechItem):
            role = item.role
            if isinstance(role, str) and role:
                return role
    return default_role


def _apply_pause_policy(
    job: Any,
    plan: Any,
    policy: PausePolicy,
    *,
    source_intents: set[str],
    provenance: Literal["renderprofile", "explicit"],
    default_role: str,
) -> Any:
    renderplan = importlib.import_module("renderplan")
    flow = tuple(plan.flow)
    if len(flow) != len(job.items):
        raise HandoffValidationError(
            "materialized AudioJob items do not match RenderPlan flow order"
        )
    items: list[Any] = []
    for index, (plan_item, job_item) in enumerate(zip(flow, job.items, strict=True)):
        if not isinstance(plan_item, renderplan.PauseItem) or not isinstance(job_item, Silence):
            items.append(job_item)
            continue
        reason = plan_item.reason
        seconds = job_item.seconds
        if reason in PAUSE_CLASSES and reason != "timed":
            use_policy = provenance == "explicit" or reason in source_intents
            use_policy = (
                use_policy
                or reason in policy.default
                or any(reason in values for values in policy.roles.values())
            )
            if use_policy:
                role = _render_pause_role(flow, index, default_role)
                seconds = policy.resolve(
                    reason,
                    role=role,
                    where=f"RenderPlan pause {plan_item.pause_id or index}",
                )
        if seconds != job_item.seconds:
            from dataclasses import replace as dataclass_replace

            items.append(dataclass_replace(job_item, seconds=seconds))
        else:
            items.append(job_item)
    source = dict(job.source)
    source["audiocompose.pause_policy_id"] = policy.policy_id
    source["audiocompose.pause_policy_provenance"] = provenance
    from dataclasses import replace as dataclass_replace

    result = dataclass_replace(job, items=tuple(items), job_id=None, source=source)
    result.validate(verify_sources=True)
    return result


def materialize_voicerender_job(
    source_plan: str | Path | Any,
    render_plan: str | Path | Any,
    render_manifest: str | Path | Any,
    *,
    render_dir: str | Path,
    output_policy: OutputPolicy | None = None,
    pause_policy: PausePolicy | str | Path | Mapping[str, Any] | None = None,
    pause_policy_provenance: Literal["renderprofile", "explicit"] = "explicit",
    default_role: str = "",
    media_resolver: MediaResolver | None = None,
    media_root: str | Path | None = None,
    emphasis_policy: Literal["segment_fallback", "strict"] = "strict",
) -> Any:
    """Validate the UtterPlan → RenderPlan → manifest handoff and materialize it.

    UtterPlan and RenderPlan artifacts are loaded only through their documented
    package-root APIs. RenderPlan owns verification of manifest identity,
    completeness, artifact membership, and external audio. VoiceRender modules
    are not probed for private or nonexistent loader/selection helpers.
    """
    flow_plan_type = _flow_plan_type()
    try:
        renderplan = importlib.import_module("renderplan")
        render_plan_type = renderplan.RenderPlan
        manifest_type = renderplan.RenderManifest
        load_plan = renderplan.load_plan
        load_manifest = renderplan.load_manifest
        validate_manifest_handoff = renderplan.validate_manifest_handoff
    except (ImportError, AttributeError) as exc:
        raise IntegrationError(
            "The installed RenderPlan package does not expose the required public handoff API."
        ) from exc

    source = load_source_plan(source_plan) if isinstance(source_plan, (str, Path)) else source_plan
    if not isinstance(source, flow_plan_type):
        raise HandoffValidationError("source_plan must be a FlowPlan or an UtterPlan path")
    if pause_policy_provenance not in {"renderprofile", "explicit"}:
        raise PausePolicyError("pause_policy_provenance must be 'renderprofile' or 'explicit'")
    policy = load_pause_policy(pause_policy) if pause_policy is not None else None
    source_intents = _validate_source_pauses(source, policy) if policy is not None else set()

    try:
        plan = load_plan(render_plan) if isinstance(render_plan, (str, Path)) else render_plan
        manifest = (
            load_manifest(render_manifest)
            if isinstance(render_manifest, (str, Path))
            else render_manifest
        )
    except Exception as exc:
        raise HandoffValidationError(f"Cannot load RenderPlan handoff artifacts: {exc}") from exc

    if not isinstance(plan, render_plan_type):
        raise HandoffValidationError("render_plan must be a RenderPlan or a RenderPlan path")
    if not isinstance(manifest, manifest_type):
        raise HandoffValidationError(
            "render_manifest must be a RenderManifest or a RenderManifest path"
        )
    if source.plan_id != plan.source_plan_id:
        raise HandoffValidationError(
            "UtterPlan source identity does not match RenderPlan source_plan_id"
        )
    try:
        validate_manifest_handoff(plan, manifest, require_complete=True)
    except Exception as exc:
        raise HandoffValidationError(f"RenderManifest handoff is invalid: {exc}") from exc

    _validate_voicerender_producer(manifest)
    diagnostics, diagnostics_sha256 = _load_render_diagnostics(
        Path(render_dir).expanduser() / RENDER_DIAGNOSTICS_FILENAME
    )
    control_ownership = _validate_render_diagnostics(plan, diagnostics, diagnostics_sha256)
    job = materialize_renderplan_job(
        plan,
        manifest,
        render_dir=render_dir,
        output_policy=output_policy,
        media_resolver=media_resolver,
        media_root=media_root,
        emphasis_policy=emphasis_policy,
        _control_ownership=control_ownership,
    )
    if policy is None:
        return job
    return _apply_pause_policy(
        job,
        plan,
        policy,
        source_intents=source_intents,
        provenance=pause_policy_provenance,
        default_role=default_role,
    )


__all__ = ["PausePolicy", "load_pause_policy", "load_source_plan", "materialize_voicerender_job"]
