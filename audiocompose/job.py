from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import AudioAnchor, AudioSpan
from .errors import AudioValidationError
from .loudness import LoudnessPolicy
from .model import AudioClip, AudioJob, OutputPolicy, Silence
from .operations import AudioOperation, RatePitchEnvelope, operation_from_dict
from .sources import AudioBufferSource, AudioFileSource
from .wav import sha256_file, wav_info, write_intermediate_wav

AUDIOJOB_FORMAT = "audiojob"
AUDIOJOB_SCHEMA_VERSION = 2
AUDIOJOB_SUPPORTED_SCHEMA_VERSIONS = (1, AUDIOJOB_SCHEMA_VERSION)


def _exact_object(
    value: Any,
    path: str,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AudioValidationError(f"{path} must be an object")
    allowed = set(required) | set(optional)
    missing = set(required) - value.keys()
    unknown = [key for key in value if key not in allowed]
    if missing:
        fields = ", ".join(sorted(missing))
        raise AudioValidationError(f"{path} is missing required field(s): {fields}")
    if unknown:
        fields = ", ".join(sorted(map(str, unknown)))
        raise AudioValidationError(f"{path} has unknown field(s): {fields}")
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AudioValidationError(f"{path} must contain only finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise AudioValidationError(f"{path} contains a non-string JSON object key: {key!r}")
            _validate_json_value(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_value(nested, f"{path}[{index}]")
        return
    raise AudioValidationError(
        f"{path} contains unsupported non-JSON value of type {type(value).__name__}"
    )


def _safe_relative(path: str) -> str:
    if not path or "\\" in path:
        raise AudioValidationError(
            f"source path must be relative and contained in bundle: {path!r}"
        )
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AudioValidationError(
            f"source path must be relative and contained in bundle: {path!r}"
        )
    return candidate.as_posix()


def _relative_source_path(path: Path, base_dir: str | Path) -> str:
    resolved_path = path.resolve()
    resolved_base = Path(base_dir).resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError as exc:
        raise AudioValidationError(
            f"source path must be relative and contained in bundle: {resolved_path!r}"
        ) from exc
    return _safe_relative(relative.as_posix())


def _policy_to_dict(policy: OutputPolicy) -> dict[str, Any]:
    loudness = policy.loudness
    return {
        "sample_rate": policy.sample_rate,
        "channels": policy.channels,
        "loudness": {
            "target_lufs": loudness.target_lufs,
            "true_peak_ceiling_dbtp": loudness.true_peak_ceiling_dbtp,
            "peak_policy": loudness.peak_policy,
        },
        "clip_policy": policy.clip_policy,
    }


def _source_to_dict(
    source: AudioBufferSource | AudioFileSource, base_dir: str | None
) -> dict[str, Any]:
    if isinstance(source, AudioBufferSource):
        raise AudioValidationError(
            "buffer sources must be saved into a bundle before serialization"
        )
    path = Path(source.path)
    if base_dir is not None:
        relative = _relative_source_path(path, base_dir)
    else:
        relative = _safe_relative(str(path))
    info = wav_info(path)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "frames": info.frames,
    }


def job_to_dict(
    job: AudioJob, *, base_dir: str | None = None, verify_sources: bool = True
) -> dict[str, Any]:
    job.validate(base_dir=base_dir, verify_sources=verify_sources)
    items: list[dict[str, Any]] = []
    for item in job.items:
        if isinstance(item, Silence):
            value: dict[str, Any] = {
                "kind": "silence",
                "id": item.id,
                "seconds": item.seconds,
            }
            if item.metadata:
                value["metadata"] = dict(item.metadata)
            items.append(value)
            continue

        source = _source_to_dict(item.source, base_dir)
        clip: dict[str, Any] = {
            "kind": "clip",
            "id": item.id,
            "source": source,
            "operations": [operation.to_dict() for operation in item.operations],
        }
        if item.anchors:
            clip["anchors"] = [
                {
                    "id": anchor.id,
                    "sample_offset": anchor.sample_offset,
                    **({"name": anchor.name} if anchor.name else {}),
                }
                for anchor in item.anchors
            ]
        if item.spans:
            clip["spans"] = [
                {
                    "source_start": span.source_start,
                    "source_end": span.source_end,
                    "sample_start": span.sample_start,
                    "sample_end": span.sample_end,
                    **({"id": span.id} if span.id is not None else {}),
                    **({"metadata": dict(span.metadata)} if span.metadata else {}),
                }
                for span in item.spans
            ]
        if item.metadata:
            clip["metadata"] = dict(item.metadata)
        items.append(clip)

    payload: dict[str, Any] = {
        "format": AUDIOJOB_FORMAT,
        "schema_version": job.schema_version,
        "producer": dict(job.producer),
        "items": items,
        "output": _policy_to_dict(job.output),
    }
    if job.job_id:
        payload["job_id"] = job.job_id
    if job.source:
        payload["source"] = dict(job.source)
    return payload


def _canonical_payload(payload: dict[str, Any]) -> bytes:
    content = {key: value for key, value in payload.items() if key != "job_id"}
    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AudioValidationError(f"manifest is not canonical JSON: {exc}") from exc


def _job_id(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_payload(payload)).hexdigest()


def _source_from_dict(source: Any, base_dir: Path) -> AudioFileSource:
    source = _exact_object(
        source,
        "source",
        required=("path", "sha256", "sample_rate", "channels", "frames"),
    )
    path = source["path"]
    if not isinstance(path, str):
        raise AudioValidationError("source.path must be a string")
    path = _safe_relative(path)
    root = base_dir.resolve()
    full_path = (root / path).resolve()
    if full_path == root or root not in full_path.parents:
        raise AudioValidationError(f"source path escapes bundle: {path!r}")
    return AudioFileSource(
        full_path,
        expected_sha256=source["sha256"],
        sample_rate=source["sample_rate"],
        channels=source["channels"],
        frames=source["frames"],
    )


def job_from_dict(
    payload: dict[str, Any],
    *,
    base_dir: str | Path,
    verify_sources: bool = True,
) -> AudioJob:
    payload = _exact_object(
        payload,
        "manifest",
        required=("format", "schema_version", "producer", "items", "output", "job_id"),
        optional=("source",),
    )
    schema_version = payload["schema_version"]
    if (
        payload["format"] != AUDIOJOB_FORMAT
        or type(schema_version) is not int
        or schema_version not in AUDIOJOB_SUPPORTED_SCHEMA_VERSIONS
    ):
        raise AudioValidationError("unsupported AudioJob format or schema version")

    declared_job_id = payload["job_id"]
    if (
        not isinstance(declared_job_id, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", declared_job_id) is None
    ):
        raise AudioValidationError("job_id must use the canonical sha256:<64 lowercase hex> format")
    if declared_job_id != _job_id(payload):
        raise AudioValidationError("job_id does not match the canonical manifest contents")

    output_data = _exact_object(
        payload["output"],
        "output",
        required=("sample_rate", "channels", "loudness", "clip_policy"),
    )
    loudness_data = _exact_object(
        output_data["loudness"],
        "output.loudness",
        required=("target_lufs", "true_peak_ceiling_dbtp", "peak_policy"),
    )
    try:
        output = OutputPolicy(
            sample_rate=output_data["sample_rate"],
            channels=output_data["channels"],
            clip_policy=output_data["clip_policy"],
            loudness=LoudnessPolicy(
                target_lufs=loudness_data["target_lufs"],
                true_peak_ceiling_dbtp=loudness_data["true_peak_ceiling_dbtp"],
                peak_policy=loudness_data["peak_policy"],
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"output: {exc}") from exc

    raw_items = payload["items"]
    if not isinstance(raw_items, list):
        raise AudioValidationError("items must be an array")
    items: list[AudioClip | Silence] = []
    root = Path(base_dir)
    for index, raw in enumerate(raw_items):
        path = f"items[{index}]"
        if not isinstance(raw, dict):
            raise AudioValidationError(f"{path} must be an object")
        kind = raw.get("kind")
        try:
            if kind == "silence":
                raw = _exact_object(
                    raw,
                    path,
                    required=("kind", "id", "seconds"),
                    optional=("metadata",),
                )
                items.append(Silence(raw["id"], raw["seconds"], raw.get("metadata", {})))
            elif kind == "clip":
                raw = _exact_object(
                    raw,
                    path,
                    required=("kind", "id", "source", "operations"),
                    optional=("anchors", "spans", "metadata"),
                )
                anchors_data = raw.get("anchors", [])
                spans_data = raw.get("spans", [])
                operations_data = raw["operations"]
                if not isinstance(anchors_data, list):
                    raise AudioValidationError("anchors must be an array")
                if not isinstance(spans_data, list):
                    raise AudioValidationError("spans must be an array")
                if not isinstance(operations_data, list):
                    raise AudioValidationError("operations must be an array")
                anchors: list[AudioAnchor] = []
                for anchor_index, anchor in enumerate(anchors_data):
                    anchor_path = f"{path}.anchors[{anchor_index}]"
                    anchor = _exact_object(
                        anchor,
                        anchor_path,
                        required=("id", "sample_offset"),
                        optional=("name",),
                    )
                    anchors.append(
                        AudioAnchor(anchor["id"], anchor["sample_offset"], anchor.get("name"))
                    )
                spans: list[AudioSpan] = []
                for span_index, span in enumerate(spans_data):
                    span_path = f"{path}.spans[{span_index}]"
                    span = _exact_object(
                        span,
                        span_path,
                        required=(
                            "source_start",
                            "source_end",
                            "sample_start",
                            "sample_end",
                        ),
                        optional=("id", "metadata"),
                    )
                    spans.append(
                        AudioSpan(
                            span["source_start"],
                            span["source_end"],
                            span["sample_start"],
                            span["sample_end"],
                            span.get("id"),
                            span.get("metadata", {}),
                        )
                    )
                operations = tuple(operation_from_dict(operation) for operation in operations_data)
                items.append(
                    AudioClip(
                        raw["id"],
                        _source_from_dict(raw["source"], root),
                        operations,
                        tuple(anchors),
                        tuple(spans),
                        raw.get("metadata", {}),
                    )
                )
            else:
                raise AudioValidationError(f"unsupported AudioJob item kind: {kind!r}")
        except AudioValidationError as exc:
            raise AudioValidationError(f"{path}: {exc}") from exc
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AudioValidationError(f"{path}: malformed item: {exc}") from exc

    try:
        job = AudioJob(
            tuple(items),
            output,
            producer=payload["producer"],
            job_id=declared_job_id,
            source=payload.get("source", {}),
            schema_version=schema_version,
        )
        job.validate(base_dir=str(root), verify_sources=verify_sources)
    except AudioValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"manifest validation failed: {exc}") from exc
    return job


def validate_clip_geometry(item: AudioClip, audio: np.ndarray) -> None:
    for anchor in item.anchors:
        if anchor.sample_offset > len(audio):
            raise AudioValidationError(
                f"anchor {anchor.id!r} exceeds source length for {item.id!r}"
            )
    for span_index, span in enumerate(item.spans):
        if span.sample_end > len(audio):
            raise AudioValidationError(
                f"span[{span_index}] sample range exceeds source length for {item.id!r}"
            )


def validate_job(
    job: AudioJob, *, base_dir: str | None = None, verify_sources: bool = True
) -> None:
    _validate_json_value(job.producer, "producer")
    _validate_json_value(job.source, "source")
    if job.output.channels != 1:
        raise AudioValidationError("only mono output is supported")
    for index, item in enumerate(job.items):
        _validate_json_value(item.metadata, f"items[{index}] ({item.id!r}) metadata")
        if not isinstance(item, AudioClip):
            continue
        if not isinstance(item.source, (AudioBufferSource, AudioFileSource)):
            raise AudioValidationError(f"unsupported source on clip {item.id!r}")
        for operation in item.operations:
            if not isinstance(operation, AudioOperation):
                raise AudioValidationError(f"unsupported operation on clip {item.id!r}")
            if job.schema_version == 1 and isinstance(operation, RatePitchEnvelope):
                raise AudioValidationError(
                    "rate_pitch_envelope is not supported in AudioJob schema v1"
                )
        for span_index, span in enumerate(item.spans):
            _validate_json_value(span.metadata, f"clip {item.id!r} span[{span_index}].metadata")
        if isinstance(item.source, AudioFileSource) and base_dir is not None:
            _relative_source_path(Path(item.source.path), base_dir)
        if verify_sources:
            audio, _ = item.source.load()
            validate_clip_geometry(item, audio)


def save_job(job: AudioJob, path: str | Path) -> Path:
    """Save an AudioJob bundle directory and return its manifest path."""
    job.validate(verify_sources=False)
    bundle = Path(path).resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    parts = bundle / "parts"
    if parts.is_symlink():
        raise AudioValidationError("bundle parts directory must not be a symlink")
    if parts.exists() and not parts.is_dir():
        raise AudioValidationError("bundle parts path must be a directory")
    manifest = bundle / "audiojob.json"

    with tempfile.TemporaryDirectory(prefix=".audiojob-stage-", dir=bundle) as temp_dir:
        stage = Path(temp_dir)
        staged_parts = stage / "parts"
        staged_parts.mkdir()
        saved_items: list[AudioClip | Silence] = []
        for index, item in enumerate(job.items, start=1):
            if isinstance(item, Silence):
                saved_items.append(item)
                continue

            output = staged_parts / f"{index:06d}.wav"
            audio, sample_rate = item.source.load()
            validate_clip_geometry(item, audio)
            write_intermediate_wav(output, audio, sample_rate)
            info = wav_info(output)
            saved_items.append(
                AudioClip(
                    item.id,
                    AudioFileSource(
                        output,
                        sha256_file(output),
                        info.sample_rate,
                        info.channels,
                        info.frames,
                    ),
                    item.operations,
                    item.anchors,
                    item.spans,
                    item.metadata,
                )
            )

        saved_job = AudioJob(
            tuple(saved_items),
            job.output,
            job.producer,
            None,
            job.source,
            job.schema_version,
        )
        serialized = job_to_dict(saved_job, base_dir=str(stage), verify_sources=False)
        serialized["job_id"] = _job_id(serialized)
        staged_manifest = stage / "audiojob.json"
        staged_manifest.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        previous_parts = stage / "previous-parts"
        moved_previous_parts = False
        installed_parts = False
        try:
            if parts.exists():
                os.replace(parts, previous_parts)
                moved_previous_parts = True
            os.replace(staged_parts, parts)
            installed_parts = True
            os.replace(staged_manifest, manifest)
        except BaseException:
            if installed_parts:
                os.replace(parts, staged_parts)
            if moved_previous_parts:
                os.replace(previous_parts, parts)
            raise

    return manifest


def load_job(path: str | Path, *, verify_sources: bool = True) -> AudioJob:
    manifest = Path(path)
    if manifest.is_dir():
        manifest /= "audiojob.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioValidationError(f"cannot read AudioJob manifest {manifest}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AudioValidationError("AudioJob manifest root must be an object")
    try:
        return job_from_dict(payload, base_dir=manifest.parent, verify_sources=verify_sources)
    except AudioValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"invalid AudioJob manifest: {exc}") from exc
