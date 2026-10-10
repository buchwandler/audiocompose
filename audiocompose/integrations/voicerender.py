"""Materialize verified UtterPlan v5 / VoiceRender artifacts as AudioJobs.

This adapter is optional and imports VoiceRender only when invoked. The
AudioCompose core models and Composer do not depend on producer packages.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .._version import __version__
from ..alignment import AudioAnchor, AudioSpan
from ..errors import (
    HandoffValidationError,
    IntegrationError,
    MediaResolutionError,
    PausePolicyError,
)
from ..model import AudioClip, AudioJob, OutputPolicy, Silence
from ..operations import (
    AutomationPoint,
    FadeIn,
    FadeOut,
    Gain,
    Operation,
    PitchShift,
    RatePitchEnvelope,
    Tempo,
)
from ..sources import AudioFileSource
from .media import (
    LocalWavResolver,
    MediaRequest,
    MediaResolver,
    ResolvedMedia,
    numeric_media_operations,
)

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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _policy_id(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(document)).hexdigest()


def _duration(value: Any, where: str) -> float:
    if not isinstance(value, str):
        raise PausePolicyError(f"{where} must be a non-negative duration such as '320ms'")
    match = _DURATION.fullmatch(value)
    if match is None:
        raise PausePolicyError(f"{where} has invalid duration {value!r}; use ms or s")
    amount = float(value[: -len(match.group(1))])
    seconds = amount / 1000.0 if match.group(1) == "ms" else amount
    if not math.isfinite(seconds) or seconds < 0:
        raise PausePolicyError(f"{where} must be a finite duration >= 0")
    return seconds


def _parse_pause_table(value: Any, where: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise PausePolicyError(f"{where} must be a TOML table or mapping")
    unknown = set(value).difference(PAUSE_CLASSES)
    if unknown:
        raise PausePolicyError(f"{where} has unknown pause class(es): {sorted(unknown)}")
    parsed = {name: _duration(item, f"{where}.{name}") for name, item in value.items()}
    if parsed.get("none", 0.0) != 0:
        raise PausePolicyError(f"{where}.none must resolve to zero")
    return parsed


def _validate_resolved_table(value: Mapping[str, Any], where: str) -> None:
    if not isinstance(value, Mapping):
        raise PausePolicyError(f"{where} must be a pause-class mapping")
    unknown = set(value).difference(PAUSE_CLASSES)
    if unknown:
        raise PausePolicyError(f"{where} has unknown pause class(es): {sorted(unknown)}")
    for name, seconds in value.items():
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise PausePolicyError(f"{where}.{name} must be finite seconds >= 0")
    if value.get("none", 0) != 0:
        raise PausePolicyError(f"{where}.none must resolve to zero")


def _normalized_policy(
    default: Mapping[str, float], voices: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    return {
        "schema": PAUSE_POLICY_SCHEMA,
        "merge": "max",
        "default": dict(sorted(default.items())),
        "voice": {role: dict(sorted(table.items())) for role, table in sorted(voices.items())},
    }


@dataclass(frozen=True, slots=True)
class PausePolicy:
    """Strict symbolic-to-numeric pause policy owned by AudioCompose."""

    default: Mapping[str, float]
    voices: Mapping[str, Mapping[str, float]]
    policy_id: str
    merge: str = "max"

    def __post_init__(self) -> None:
        if self.merge != "max":
            raise PausePolicyError("PausePolicy merge must be 'max'")
        if not isinstance(self.default, Mapping) or not isinstance(self.voices, Mapping):
            raise PausePolicyError("PausePolicy defaults and voice overrides must be mappings")
        if (
            not isinstance(self.policy_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.policy_id) is None
        ):
            raise PausePolicyError("PausePolicy policy_id must be a canonical SHA-256 identity")
        _validate_resolved_table(self.default, "default")
        for voice, values in self.voices.items():
            if not isinstance(voice, str) or not voice.strip():
                raise PausePolicyError("voice override names must be non-empty strings")
            _validate_resolved_table(values, f"voice.{voice}")

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> PausePolicy:
        if not isinstance(document, Mapping):
            raise PausePolicyError("Pause policy must be a mapping")
        unknown = set(document).difference({"schema", "merge", "default", "voice"})
        if unknown:
            raise PausePolicyError(f"Pause policy has unknown root field(s): {sorted(unknown)}")
        if document.get("schema") != PAUSE_POLICY_SCHEMA:
            raise PausePolicyError(f"Expected pause policy schema {PAUSE_POLICY_SCHEMA!r}")
        if document.get("merge") != "max":
            raise PausePolicyError("Pause policy merge must be 'max'")
        default = _parse_pause_table(document.get("default"), "default")
        raw_voices = document.get("voice", {})
        if not isinstance(raw_voices, Mapping):
            raise PausePolicyError("voice must be a table of logical-voice overrides")
        voices: dict[str, dict[str, float]] = {}
        for role, table in raw_voices.items():
            if not isinstance(role, str) or not role.strip():
                raise PausePolicyError("voice override names must be non-empty strings")
            voices[role] = _parse_pause_table(table, f"voice.{role}")
        normalized = _normalized_policy(default, voices)
        return cls(default, voices, _policy_id(normalized))

    @classmethod
    def load(cls, path: str | Path) -> PausePolicy:
        location = Path(path).expanduser()
        if location.is_symlink():
            raise PausePolicyError(f"Pause policy must not be a symlink: {location}")
        try:
            try:
                tomllib = importlib.import_module("tomllib")
            except ImportError:  # Python 3.10; supplied by the optional VoiceRender extra.
                tomllib = importlib.import_module("tomli")
            document = tomllib.loads(location.read_text(encoding="utf-8"))
        except (ImportError, OSError, UnicodeError, ValueError) as exc:
            raise PausePolicyError(f"Cannot read pause policy {location}: {exc}") from exc
        return cls.from_mapping(document)

    def resolve(self, intent: Any, *, voice: str, where: str) -> float:
        if intent is None:
            return 0.0
        if isinstance(intent, str):
            if intent not in PAUSE_CLASSES:
                raise PausePolicyError(f"{where}: unknown symbolic pause {intent!r}")
            if intent == "none":
                return 0.0
            value = self.voices.get(voice, {}).get(intent, self.default.get(intent))
            if value is None:
                raise PausePolicyError(
                    f"{where}: no duration configured for pause {intent!r} "
                    f"(logical voice {voice!r})"
                )
            return float(value)
        if isinstance(intent, Mapping):
            if set(intent) != {"type", "time"} or intent.get("type") != "timed":
                raise PausePolicyError(f"{where}: timed pause must contain exactly type and time")
            return _duration(intent["time"], f"{where}.time")
        raise PausePolicyError(f"{where}: invalid pause intent {intent!r}")


def load_pause_policy(value: PausePolicy | str | Path | Mapping[str, Any]) -> PausePolicy:
    if isinstance(value, PausePolicy):
        return value
    if isinstance(value, (str, Path)):
        return PausePolicy.load(value)
    if isinstance(value, Mapping):
        return PausePolicy.from_mapping(value)
    raise PausePolicyError("pause_policy must be a policy path, mapping, or PausePolicy")


def _require_voicerender() -> Any:
    try:
        voicerender = importlib.import_module("voicerender")
    except ImportError as exc:  # pragma: no cover - depends on optional installation.
        raise IntegrationError(
            "VoiceRender materialization requires the optional dependency; "
            "install audiocompose[voicerender]"
        ) from exc
    required = (
        "load_source_plan",
        "verify_render_manifest",
        "validate_manifest_source",
        "select_segments",
        "RenderSelection",
        "UtterPlanV5",
    )
    missing = [name for name in required if not hasattr(voicerender, name)]
    if missing:
        raise IntegrationError(
            "Installed VoiceRender does not provide the required public handoff API "
            f"({', '.join(missing)}); install a compatible voicerender>=0.1,<0.2"
        )
    return voicerender


def _manifest_selection(plan: Any, selection_data: Mapping[str, Any], vr: Any) -> Any:
    kind = selection_data.get("kind")
    if kind == "all":
        chosen = vr.select_segments(plan)
    elif kind == "take":
        chosen = vr.select_segments(plan, take=selection_data.get("take"))
    elif kind == "locators":
        chosen = vr.select_segments(plan, select=selection_data.get("locators"))
    else:
        raise HandoffValidationError(f"Unsupported manifest selection kind: {kind!r}")
    if chosen.as_dict() != dict(selection_data):
        raise HandoffValidationError("Render manifest selection does not match the source plan")
    return chosen


def _requested_selection(plan: Any, requested: Any, authoritative: Any, vr: Any) -> Any:
    if requested is None:
        return authoritative
    if isinstance(requested, vr.RenderSelection):
        locators = [segment.source.locator for segment in requested.segments]
        chosen = vr.select_segments(plan, select=locators)
        if chosen.as_dict() != requested.as_dict():
            raise HandoffValidationError("Explicit RenderSelection is not canonical for this plan")
    elif type(requested) is int:
        chosen = vr.select_segments(plan, take=requested)
    elif isinstance(requested, str):
        chosen = vr.select_segments(plan, select=requested)
    elif isinstance(requested, Iterable) and not isinstance(requested, Mapping):
        chosen = vr.select_segments(plan, select=requested)
    else:
        raise HandoffValidationError(
            "selection must be a source-locator selection, take count, or RenderSelection"
        )
    if chosen.as_dict() != authoritative.as_dict():
        raise HandoffValidationError(
            "Explicit selection differs from the render manifest selection"
        )
    return chosen


def _safe_manifest_artifact(manifest_path: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path:
        raise HandoffValidationError(
            "Render manifest audio path must be a safe relative POSIX path"
        )
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise HandoffValidationError("Render manifest audio path must not escape its bundle")
    if manifest_path.is_symlink():
        raise HandoffValidationError("Render manifest must not be a symlink")
    root = manifest_path.parent.resolve(strict=True)
    candidate = root
    try:
        for part in pure.parts:
            candidate = candidate / part
            candidate.lstat()
            if candidate.is_symlink():
                raise HandoffValidationError(
                    f"Render artifact path contains a symlink: {relative_path}"
                )
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise HandoffValidationError(
            f"Render artifact is unavailable: {relative_path}: {exc}"
        ) from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HandoffValidationError(
            f"Render artifact is not a regular file in its bundle: {relative_path}"
        )
    return resolved


def _validate_source_operation(operation: Mapping[str, Any], segment: Any) -> str:
    source = operation.get("source")
    if not isinstance(source, Mapping):
        raise HandoffValidationError("Render source operation is missing its source record")
    expected = {
        "locator": segment.source.locator,
        "flow_index": segment.source.flow_index,
        "segment_index": segment.source.segment_index,
        "flow_hash": segment.source.flow_hash,
        "semantic_segment_hash": segment.semantic_segment_hash,
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise HandoffValidationError(
                f"{segment.source.locator}: render source operation {field} does not match the plan"
            )
    kind = operation.get("kind")
    if kind not in {"speech", "external_audio"}:
        raise HandoffValidationError(
            f"{segment.source.locator}: unsupported source operation kind {kind!r}"
        )
    return kind


def _operation_from_record(record: Mapping[str, Any], *, locator: str) -> Operation:
    if set(record) != {"name", "value"}:
        raise HandoffValidationError(f"{locator}: composition directive has an invalid shape")
    name, value = record["name"], record["value"]
    if not isinstance(name, str) or not isinstance(value, Mapping):
        raise HandoffValidationError(
            f"{locator}: composition directive must be a named numeric object"
        )
    constructors: dict[str, tuple[str, type[Any]]] = {
        "tempo": ("factor", Tempo),
        "pitch_shift": ("semitones", PitchShift),
        "gain": ("db", Gain),
        "fade_in": ("seconds", FadeIn),
        "fade_out": ("seconds", FadeOut),
    }
    if name in constructors:
        field, constructor = constructors[name]
        if set(value) != {field}:
            raise HandoffValidationError(f"{locator}: malformed numeric {name!r} directive")
        try:
            return constructor(value[field])
        except (TypeError, ValueError) as exc:
            raise HandoffValidationError(
                f"{locator}: invalid numeric {name!r} directive: {exc}"
            ) from exc
    if name == "rate_pitch_envelope":
        if set(value) != {"rate", "pitch", "interpolation", "time_base"}:
            raise HandoffValidationError(f"{locator}: malformed rate_pitch_envelope directive")
        if not isinstance(value["rate"], list) or not isinstance(value["pitch"], list):
            raise HandoffValidationError(f"{locator}: envelope curves must be arrays")
        try:
            rate = tuple(
                AutomationPoint(point["seconds"], point["factor"])
                for point in value["rate"]
                if isinstance(point, Mapping) and set(point) == {"seconds", "factor"}
            )
            pitch = tuple(
                AutomationPoint(point["seconds"], point["semitones"])
                for point in value["pitch"]
                if isinstance(point, Mapping) and set(point) == {"seconds", "semitones"}
            )
            if len(rate) != len(value["rate"]) or len(pitch) != len(value["pitch"]):
                raise ValueError("envelope point fields are invalid")
            return RatePitchEnvelope(
                rate=rate,
                pitch_semitones=pitch,
                interpolation=value["interpolation"],
                time_base=value["time_base"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HandoffValidationError(f"{locator}: invalid numeric envelope: {exc}") from exc
    raise HandoffValidationError(
        f"{locator}: unknown composition directive {name!r}; refusing to drop semantics"
    )


def _lowering_operations(segment: Any, entry: Mapping[str, Any]) -> tuple[Operation, ...]:
    lowering = entry.get("semantic_lowering")
    if not isinstance(lowering, Mapping):
        raise HandoffValidationError(f"{segment.source.locator}: semantic_lowering is missing")
    native = lowering.get("native_directives")
    composition = lowering.get("composition_directives")
    if not isinstance(native, list) or not isinstance(composition, list):
        raise HandoffValidationError(
            f"{segment.source.locator}: semantic lowering lists are invalid"
        )
    native_names: list[str] = []
    for item in native:
        if not isinstance(item, Mapping) or set(item) != {"name", "value"}:
            raise HandoffValidationError(
                f"{segment.source.locator}: invalid native directive record"
            )
        name = item["name"]
        if not isinstance(name, str) or name in native_names:
            raise HandoffValidationError(
                f"{segment.source.locator}: duplicate/invalid native directive"
            )
        native_names.append(name)
        if item["value"] != segment.directives.get(name):
            raise HandoffValidationError(
                f"{segment.source.locator}: native directive {name!r} differs from plan"
            )
    composition_names: list[str] = []
    for item in composition:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise HandoffValidationError(
                f"{segment.source.locator}: invalid composition directive record"
            )
        if item["name"] in composition_names:
            raise HandoffValidationError(
                f"{segment.source.locator}: duplicate composition directive"
            )
        composition_names.append(item["name"])
    if set(native_names).intersection(composition_names):
        raise HandoffValidationError(f"{segment.source.locator}: directive lowered twice")
    expected_native = set(segment.directives).difference({"voice", "prosody"})
    if set(native_names) != expected_native:
        raise HandoffValidationError(
            f"{segment.source.locator}: native directive set differs from the source plan"
        )
    required_composition: set[str] = set()
    prosody = segment.directives.get("prosody")
    if isinstance(prosody, Mapping):
        for field, lowered_name in (
            ("rate", "tempo"),
            ("pitch", "pitch_shift"),
            ("volume", "gain"),
        ):
            if prosody.get(field) is not None:
                required_composition.add(lowered_name)
    if not required_composition.issubset(composition_names):
        raise HandoffValidationError(
            f"{segment.source.locator}: manifest omitted numeric lowering for source prosody"
        )
    return tuple(
        _operation_from_record(item, locator=segment.source.locator) for item in composition
    )


def _span_values(segment: Any, timing: Mapping[str, Any]) -> tuple[AudioSpan, ...]:
    raw_spans = timing.get("spans")
    if not isinstance(raw_spans, list):
        raise HandoffValidationError(f"{segment.source.locator}: timing spans must be an array")
    spans: list[AudioSpan] = []
    cursor = 0
    for index, raw in enumerate(raw_spans):
        if not isinstance(raw, Mapping):
            raise HandoffValidationError(
                f"{segment.source.locator}: timing span {index} is not an object"
            )
        start, end = raw.get("start_frame"), raw.get("end_frame")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start:
            raise HandoffValidationError(
                f"{segment.source.locator}: timing span {index} has invalid frames"
            )
        text = raw.get("text")
        if "text_start" in raw or "text_end" in raw:
            text_start, text_end = raw.get("text_start"), raw.get("text_end")
            if (
                type(text_start) is not int
                or type(text_end) is not int
                or text_start < 0
                or text_end < text_start
                or text_end > len(segment.text)
            ):
                raise HandoffValidationError(
                    f"{segment.source.locator}: timing span {index} has invalid text offsets"
                )
            if text is not None and segment.text[text_start:text_end] != text:
                raise HandoffValidationError(
                    f"{segment.source.locator}: timing span {index} text does not match offsets"
                )
        elif isinstance(text, str):
            found = segment.text.find(text, cursor)
            if found < 0:
                text_start = text_end = 0
            else:
                text_start, text_end = found, found + len(text)
                cursor = text_end
        else:
            text_start = text_end = 0
        metadata: dict[str, Any] = {"timing_provenance": "provided"}
        if isinstance(text, str):
            metadata["text"] = text
        if isinstance(raw.get("kind"), str):
            metadata["kind"] = raw["kind"]
        spans.append(
            AudioSpan(
                source_start=text_start,
                source_end=text_end,
                sample_start=start,
                sample_end=end,
                id=f"{segment.source.locator}:span:{index}",
                metadata=metadata,
            )
        )
    return tuple(spans)


def _json_intent(intent: Any) -> Any:
    if isinstance(intent, Mapping):
        return {str(key): _json_intent(value) for key, value in intent.items()}
    if isinstance(intent, (list, tuple)):
        return [_json_intent(value) for value in intent]
    return intent


def _pause_boundaries(
    selected: Any,
    entries_by_locator: Mapping[str, Mapping[str, Any]],
    policy: PausePolicy,
    plan_id: str,
) -> dict[str, dict[str, Any]]:
    boundaries: dict[str, dict[str, Any]] = {}
    for left, right in zip(selected.segments, selected.segments[1:], strict=False):
        left_entry = entries_by_locator.get(left.source.locator, {})
        right_entry = entries_by_locator.get(right.source.locator, {})
        left_voice = left_entry.get("logical_voice") or left.logical_voice or ""
        right_voice = right_entry.get("logical_voice") or right.logical_voice or ""
        left_seconds = policy.resolve(
            left.pause_after,
            voice=left_voice,
            where=f"plan_id {plan_id} {left.source.locator} pause_after",
        )
        right_seconds = policy.resolve(
            right.pause_before,
            voice=right_voice,
            where=f"plan_id {plan_id} {right.source.locator} pause_before",
        )
        seconds = max(left_seconds, right_seconds)
        if not math.isfinite(seconds) or seconds < 0:
            raise PausePolicyError(
                f"{left.source.locator}->{right.source.locator}: invalid resolved pause"
            )
        authored: list[dict[str, Any]] = []
        seen_intents: set[bytes] = set()
        for locator, side, intent in (
            (left.source.locator, "pause_after", left.pause_after),
            (right.source.locator, "pause_before", right.pause_before),
        ):
            if intent is None:
                continue
            identity = _canonical_json(_json_intent(intent))
            if identity in seen_intents:
                continue
            seen_intents.add(identity)
            authored.append({"id": f"{locator}:{side}", "intent": _json_intent(intent)})
        if seconds > 0:
            boundaries[left.source.locator] = {
                "right": right,
                "seconds": seconds,
                "authored": authored,
            }
    return boundaries


def _selection_as_dict(selection: Any) -> dict[str, Any]:
    value = selection.as_dict()
    if not isinstance(value, Mapping):
        raise HandoffValidationError("VoiceRender selection did not produce a mapping")
    return dict(value)


def materialize_voicerender_job(
    plan: Any,
    render_manifest: str | Path,
    pause_policy: PausePolicy | str | Path | Mapping[str, Any],
    *,
    output_policy: OutputPolicy | None = None,
    selection: Any = None,
    media_resolver: MediaResolver | None = None,
) -> AudioJob:
    """Validate a VoiceRender handoff and return an ordinary AudioCompose job.

    If ``selection`` is supplied, it must exactly repeat the render manifest's
    source selection. External audio is resolved in source order; callers can
    provide a ``MediaResolver`` for non-filesystem URIs.
    """
    vr = _require_voicerender()
    try:
        plan_location: Path | None
        if isinstance(plan, (str, Path)):
            plan_location = Path(plan).expanduser().resolve(strict=True)
            parsed = vr.load_source_plan(plan_location)
        else:
            parsed = plan
            raw_plan_path = getattr(parsed, "path", None)
            plan_location = (
                Path(raw_plan_path).expanduser().resolve(strict=True)
                if isinstance(raw_plan_path, (str, Path))
                else None
            )
        if not isinstance(parsed, vr.UtterPlanV5):
            raise HandoffValidationError("plan must be an UtterPlan v5 value or path")
        manifest_path = Path(render_manifest).expanduser().absolute()
        document = vr.verify_render_manifest(manifest_path)
        vr.validate_manifest_source(parsed, document)
        if document.get("complete") is not True:
            raise HandoffValidationError("Render manifest is incomplete")
        policy = load_pause_policy(pause_policy)
        selection_data = document.get("selection")
        if not isinstance(selection_data, Mapping):
            raise HandoffValidationError("Render manifest selection must be an object")
        authoritative = _manifest_selection(parsed, selection_data, vr)
        chosen = _requested_selection(parsed, selection, authoritative, vr)
        source_operations = document.get("source_operations")
        if not isinstance(source_operations, list) or len(source_operations) != len(
            chosen.segments
        ):
            raise HandoffValidationError(
                "Render manifest lacks complete selected source operations"
            )
        raw_entries = document.get("segments", [])
        if not isinstance(raw_entries, list):
            raise HandoffValidationError("Render manifest segments must be an array")
        by_locator: dict[str, list[Mapping[str, Any]]] = {}
        for entry in raw_entries:
            source_ref = entry.get("source", {}) if isinstance(entry, Mapping) else {}
            locator = source_ref.get("locator") if isinstance(source_ref, Mapping) else None
            if isinstance(locator, str):
                by_locator.setdefault(locator, []).append(entry)

        positions = {segment.source.locator: index for index, segment in enumerate(parsed.segments)}
        selected_positions = [
            positions.get(segment.source.locator, -1) for segment in chosen.segments
        ]
        if not selected_positions or selected_positions != list(
            range(selected_positions[0], selected_positions[0] + len(selected_positions))
        ):
            raise HandoffValidationError(
                "AudioJob composition requires a contiguous source selection"
            )

        records: dict[str, dict[str, Any]] = {}
        entries_by_locator: dict[str, Mapping[str, Any]] = {}
        resolver = media_resolver if media_resolver is not None else LocalWavResolver()
        for segment, operation in zip(chosen.segments, source_operations, strict=True):
            locator = segment.source.locator
            kind = _validate_source_operation(operation, segment)
            matches = by_locator.get(locator, [])
            if kind == "speech":
                if len(matches) != 1:
                    raise HandoffValidationError(
                        f"{locator}: expected exactly one render manifest entry, found {len(matches)}"
                    )
                entry = matches[0]
                if entry.get("status") not in {"rendered", "reused"} or not document.get(
                    "complete"
                ):
                    raise HandoffValidationError(
                        f"{locator}: speech render is incomplete (status={entry.get('status')!r})"
                    )
                for field, expected, actual in (
                    ("language", segment.language, entry.get("language")),
                    ("logical_voice", segment.logical_voice, entry.get("logical_voice")),
                ):
                    if expected is not None and expected != actual:
                        raise HandoffValidationError(
                            f"{locator}: manifest {field} differs from the plan"
                        )
                audio = entry.get("audio")
                if not isinstance(audio, Mapping):
                    raise HandoffValidationError(f"{locator}: rendered WAV metadata is missing")
                path = _safe_manifest_artifact(manifest_path, audio.get("path"))
                entries_by_locator[locator] = entry
                records[locator] = {"kind": kind, "entry": entry, "path": path}
                continue

            if matches:
                raise HandoffValidationError(
                    f"{locator}: external audio must not have a speech render entry"
                )
            directive = operation.get("audio_directive")
            plan_directive = segment.directives.get("audio")
            if not isinstance(directive, Mapping) or not isinstance(plan_directive, Mapping):
                raise HandoffValidationError(f"{locator}: external audio directive is missing")
            if dict(directive) != dict(plan_directive):
                raise HandoffValidationError(
                    f"{locator}: manifest audio directive differs from the source plan"
                )
            if plan_location is None:
                raise MediaResolutionError(
                    f"{locator}: external media resolution requires a plan file path"
                )
            request = MediaRequest(
                src=directive["src"],
                plan_path=plan_location,
                source_locator=locator,
                clip_begin=directive.get("clip_begin"),
                clip_end=directive.get("clip_end"),
                speed=directive.get("speed"),
                repeat_duration=directive.get("repeat_duration"),
                repeat_count=directive.get("repeat_count"),
                sound_level=directive.get("sound_level"),
                description=directive.get("description"),
                alt_text=directive.get("alt_text"),
            )
            if request.clip_begin is not None or request.clip_end is not None:
                raise MediaResolutionError(
                    f"{locator}: clip_begin/clip_end are unsupported by AudioJob v2"
                )
            if request.repeat_duration is not None or request.repeat_count is not None:
                raise MediaResolutionError(
                    f"{locator}: repeat_duration/repeat_count are unsupported by AudioJob v2"
                )
            numeric_operations = numeric_media_operations(request)
            try:
                resolved = resolver.resolve(request)
            except MediaResolutionError:
                raise
            except Exception as exc:
                raise MediaResolutionError(f"{locator}: media resolver failed: {exc}") from exc
            if not isinstance(resolved, ResolvedMedia):
                raise MediaResolutionError(f"{locator}: MediaResolver must return ResolvedMedia")
            media_operations = (*resolved.operations, *numeric_operations)
            records[locator] = {
                "kind": kind,
                "request": request,
                "media": ResolvedMedia(
                    source=resolved.source,
                    operations=media_operations,
                    metadata=resolved.metadata,
                    identity=resolved.identity,
                ),
            }

        boundaries = _pause_boundaries(chosen, entries_by_locator, policy, parsed.plan_id)
        items: list[AudioClip | Silence] = []
        for segment in chosen.segments:
            locator = segment.source.locator
            record = records[locator]
            if record["kind"] == "speech":
                entry = record["entry"]
                audio = entry["audio"]
                digest = audio.get("sha256")
                if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                    raise HandoffValidationError(f"{locator}: manifest WAV hash is invalid")
                operations = _lowering_operations(segment, entry)
                timing = entry.get("timing")
                if not isinstance(timing, Mapping):
                    raise HandoffValidationError(f"{locator}: render timing metadata is missing")
                spans = _span_values(segment, timing)
                anchors = tuple(
                    AudioAnchor(id=f"{locator}:marker:{index}", sample_offset=0, name=name)
                    for index, name in enumerate(segment.markers)
                )
                metadata = {
                    "kind": "speech",
                    "utterplan.plan_id": parsed.plan_id,
                    "utterplan.locator": locator,
                    "voicerender.render_manifest_id": document["manifest_id"],
                    "voicerender.acoustic_cache_key": entry["acoustic_cache_key"],
                    "voicerender.logical_voice": entry["logical_voice"],
                    "voicerender.timing_provenance": timing["provenance"],
                    "voicerender.semantic_segment_hash": segment.semantic_segment_hash,
                }
                source_item = AudioFileSource(
                    path=record["path"],
                    expected_sha256=digest,
                    sample_rate=audio["sample_rate"],
                    channels=audio["channels"],
                    frames=audio["frames"],
                )
                items.append(
                    AudioClip(
                        f"speech:{locator}", source_item, operations, anchors, spans, metadata
                    )
                )
            else:
                request = record["request"]
                resolved = record["media"]
                anchors = tuple(
                    AudioAnchor(id=f"{locator}:marker:{index}", sample_offset=0, name=name)
                    for index, name in enumerate(segment.markers)
                )
                metadata = {
                    "kind": "external_audio",
                    "utterplan.plan_id": parsed.plan_id,
                    "utterplan.locator": locator,
                    "utterplan.audio.src": request.src,
                    "utterplan.audio.alt_text": request.alt_text,
                    "utterplan.audio.description": request.description,
                    "audiocompose.media.metadata": resolved.metadata,
                    "audiocompose.media.identity": resolved.identity,
                }
                items.append(
                    AudioClip(
                        f"media:{locator}",
                        resolved.source,
                        resolved.operations,
                        anchors,
                        metadata=metadata,
                    )
                )
            boundary = boundaries.get(locator)
            if boundary is not None:
                right = boundary["right"]
                items.append(
                    Silence(
                        id=f"pause:{locator}->{right.source.locator}",
                        seconds=boundary["seconds"],
                        metadata={
                            "kind": "pause",
                            "utterplan.plan_id": parsed.plan_id,
                            "utterplan.left_locator": locator,
                            "utterplan.right_locator": right.source.locator,
                            "audiocompose.pause_policy_id": policy.policy_id,
                            "audiocompose.pause_events": boundary["authored"],
                        },
                    )
                )

        source = {
            "utterplan.plan_id": parsed.plan_id,
            "utterplan.plan_file_sha256": "sha256:" + parsed.file_sha256,
            "voicerender.render_manifest_id": document["manifest_id"],
            "voicerender.producer_version": document["producer"]["version"],
            "audiocompose.pause_policy_id": policy.policy_id,
            "voicerender.selection": _selection_as_dict(chosen),
            "audiocompose.materializer": "voicerender.v1",
        }
        job = AudioJob(
            items=tuple(items),
            output=output_policy if output_policy is not None else OutputPolicy(),
            producer={"name": "audiocompose.voicerender-materializer", "version": __version__},
            source=source,
        )
        job.validate(verify_sources=True)
        return job
    except IntegrationError:
        raise
    except Exception as exc:
        raise HandoffValidationError(f"VoiceRender handoff validation failed: {exc}") from exc


__all__ = [
    "PausePolicy",
    "MediaRequest",
    "ResolvedMedia",
    "MediaResolver",
    "load_pause_policy",
    "materialize_voicerender_job",
]
