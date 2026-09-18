from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .alignment import AudioAnchor, AudioSpan
from .errors import AudioValidationError
from .loudness import LoudnessPolicy
from .model import AudioClip, AudioJob, OutputPolicy, Silence
from .operations import AudioOperation, operation_from_dict
from .sources import AudioBufferSource, AudioFileSource
from .wav import sha256_file, wav_info, write_intermediate_wav

AUDIOJOB_FORMAT = "audiojob"
AUDIOJOB_SCHEMA_VERSION = 1


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


def job_to_dict(job: AudioJob, *, base_dir: str | None = None) -> dict[str, Any]:
    job.validate(base_dir=base_dir)
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
        "schema_version": AUDIOJOB_SCHEMA_VERSION,
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


def _source_from_dict(source: dict[str, Any], base_dir: Path) -> AudioFileSource:
    path = _safe_relative(str(source.get("path", "")))
    root = base_dir.resolve()
    full_path = (root / path).resolve()
    if full_path == root or root not in full_path.parents:
        raise AudioValidationError(f"source path escapes bundle: {path!r}")
    return AudioFileSource(
        full_path,
        expected_sha256=source.get("sha256"),
        sample_rate=source.get("sample_rate"),
        channels=source.get("channels", 1),
        frames=source.get("frames"),
    )


def job_from_dict(payload: dict[str, Any], *, base_dir: str | Path) -> AudioJob:
    if not isinstance(payload, dict):
        raise AudioValidationError("manifest root must be an object")
    if (
        payload.get("format") != AUDIOJOB_FORMAT
        or payload.get("schema_version") != AUDIOJOB_SCHEMA_VERSION
    ):
        raise AudioValidationError("unsupported AudioJob format or schema version")

    declared_job_id = payload.get("job_id")
    if declared_job_id is not None and declared_job_id != _job_id(payload):
        raise AudioValidationError("job_id does not match the canonical manifest contents")

    output_data = payload.get("output", {})
    if not isinstance(output_data, dict):
        raise AudioValidationError("output must be an object")
    loudness_data = output_data.get("loudness", {})
    if not isinstance(loudness_data, dict):
        raise AudioValidationError("output.loudness must be an object")
    try:
        output = OutputPolicy(
            sample_rate=output_data.get("sample_rate", 24000),
            channels=output_data.get("channels", 1),
            clip_policy=output_data.get("clip_policy", "clamp"),
            loudness=LoudnessPolicy(
                target_lufs=loudness_data.get("target_lufs"),
                true_peak_ceiling_dbtp=loudness_data.get("true_peak_ceiling_dbtp", -1.0),
                peak_policy=loudness_data.get("peak_policy", "reduce_gain"),
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"output: {exc}") from exc

    raw_items = payload.get("items", [])
    if not isinstance(raw_items, list):
        raise AudioValidationError("items must be an array")
    items: list[AudioClip | Silence] = []
    root = Path(base_dir)
    for index, raw in enumerate(raw_items):
        path = f"items[{index}]"
        try:
            if not isinstance(raw, dict):
                raise AudioValidationError("must be an object")
            kind = raw.get("kind")
            if kind == "silence":
                items.append(Silence(raw["id"], float(raw["seconds"]), raw.get("metadata", {})))
            elif kind == "clip":
                anchors_data = raw.get("anchors", [])
                spans_data = raw.get("spans", [])
                operations_data = raw.get("operations", [])
                if not isinstance(anchors_data, list):
                    raise AudioValidationError("anchors must be an array")
                if not isinstance(spans_data, list):
                    raise AudioValidationError("spans must be an array")
                if not isinstance(operations_data, list):
                    raise AudioValidationError("operations must be an array")
                anchors = tuple(
                    AudioAnchor(a["id"], int(a["sample_offset"]), a.get("name"))
                    for a in anchors_data
                )
                spans = tuple(
                    AudioSpan(
                        int(s["source_start"]),
                        int(s["source_end"]),
                        int(s["sample_start"]),
                        int(s["sample_end"]),
                        s.get("id"),
                        s.get("metadata", {}),
                    )
                    for s in spans_data
                )
                operations = tuple(operation_from_dict(op) for op in operations_data)
                items.append(
                    AudioClip(
                        raw["id"],
                        _source_from_dict(raw["source"], root),
                        operations,
                        anchors,
                        spans,
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
            payload.get("producer", {}),
            declared_job_id or _job_id(payload),
            payload.get("source", {}),
        )
        job.validate(base_dir=str(root))
    except AudioValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"manifest validation failed: {exc}") from exc
    return job


def validate_job(job: AudioJob, *, base_dir: str | None = None) -> None:
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
        audio, _ = item.source.load()
        for anchor in item.anchors:
            if anchor.sample_offset > len(audio):
                raise AudioValidationError(
                    f"anchor {anchor.id!r} exceeds source length for {item.id!r}"
                )
        for span_index, span in enumerate(item.spans):
            _validate_json_value(span.metadata, f"clip {item.id!r} span[{span_index}].metadata")
            if span.sample_end > len(audio):
                raise AudioValidationError(
                    f"span[{span_index}] sample range exceeds source length for {item.id!r}"
                )
        if isinstance(item.source, AudioFileSource) and base_dir is not None:
            path = Path(item.source.path)
            _relative_source_path(path, base_dir)


def save_job(job: AudioJob, path: str | Path) -> Path:
    job.validate()
    destination = Path(path)
    bundle = destination if destination.suffix != ".json" else destination.parent
    bundle.mkdir(parents=True, exist_ok=True)
    parts = (bundle / "parts").resolve()
    parts.mkdir(parents=True, exist_ok=True)

    saved_items: list[AudioClip | Silence] = []
    for index, item in enumerate(job.items, start=1):
        if isinstance(item, Silence):
            saved_items.append(item)
            continue

        output = parts / f"{index:06d}.wav"
        resolved_output = output.resolve()
        if resolved_output == parts or parts not in resolved_output.parents:
            raise AudioValidationError(f"bundle part path escapes parts directory: {output}")
        audio, sample_rate = item.source.load()
        write_intermediate_wav(resolved_output, audio, sample_rate)
        info = wav_info(resolved_output)
        saved_items.append(
            AudioClip(
                item.id,
                AudioFileSource(
                    resolved_output,
                    sha256_file(resolved_output),
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

    saved_job = AudioJob(tuple(saved_items), job.output, job.producer, None, job.source)
    serialized = job_to_dict(saved_job, base_dir=str(bundle))
    serialized["job_id"] = _job_id(serialized)
    manifest = bundle / "audiojob.json"
    manifest.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_job(path: str | Path) -> AudioJob:
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
        return job_from_dict(payload, base_dir=manifest.parent)
    except AudioValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioValidationError(f"invalid AudioJob manifest: {exc}") from exc
