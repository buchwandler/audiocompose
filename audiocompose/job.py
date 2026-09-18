from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .alignment import AudioAnchor, AudioSpan
from .errors import AudioValidationError
from .loudness import LoudnessPolicy
from .model import AudioClip, AudioJob, OutputPolicy, Silence
from .operations import operation_from_dict
from .sources import AudioBufferSource, AudioFileSource
from .wav import sha256_file, wav_info, write_intermediate_wav


def _safe_relative(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AudioValidationError(f"source path must be relative and contained in bundle: {path!r}")
    return candidate.as_posix()


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


def _source_to_dict(source: AudioBufferSource | AudioFileSource, base_dir: str | None) -> dict[str, Any]:
    if isinstance(source, AudioBufferSource):
        raise AudioValidationError("buffer sources must be saved into a bundle before serialization")
    path = Path(source.path)
    relative = _safe_relative(os.path.relpath(path, base_dir)) if base_dir and path.is_absolute() else _safe_relative(str(path))
    info = wav_info(path)
    result: dict[str, Any] = {"path": relative, "sha256": sha256_file(path), "sample_rate": info.sample_rate, "channels": info.channels, "frames": info.frames}
    return result


def job_to_dict(job: AudioJob, *, base_dir: str | None = None) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in job.items:
        if isinstance(item, Silence):
            items.append({"kind": "silence", "id": item.id, "seconds": item.seconds, **({"metadata": dict(item.metadata)} if item.metadata else {})})
            continue
        source = _source_to_dict(item.source, base_dir)
        clip: dict[str, Any] = {"kind": "clip", "id": item.id, "source": source, "operations": [operation.to_dict() for operation in item.operations]}
        if item.anchors:
            clip["anchors"] = [{"id": anchor.id, "sample_offset": anchor.sample_offset, **({"name": anchor.name} if anchor.name else {})} for anchor in item.anchors]
        if item.spans:
            clip["spans"] = [{"source_start": span.source_start, "source_end": span.source_end, "sample_start": span.sample_start, "sample_end": span.sample_end} for span in item.spans]
        if item.metadata:
            clip["metadata"] = dict(item.metadata)
        items.append(clip)
    payload: dict[str, Any] = {"format": "audiojob", "schema_version": 1, "producer": dict(job.producer), "items": items, "output": _policy_to_dict(job.output)}
    if job.job_id:
        payload["job_id"] = job.job_id
    if job.source:
        payload["source"] = dict(job.source)
    return payload


def _job_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_from_dict(source: dict[str, Any], base_dir: Path) -> AudioFileSource:
    path = _safe_relative(str(source.get("path", "")))
    full_path = (base_dir / path).resolve()
    if base_dir.resolve() not in full_path.parents:
        raise AudioValidationError(f"source path escapes bundle: {path!r}")
    return AudioFileSource(full_path, expected_sha256=source.get("sha256"), sample_rate=source.get("sample_rate"), channels=source.get("channels", 1), frames=source.get("frames"))


def job_from_dict(payload: dict[str, Any], *, base_dir: str | Path) -> AudioJob:
    if payload.get("format") != "audiojob" or payload.get("schema_version") != 1:
        raise AudioValidationError("unsupported AudioJob format or schema version")
    output_data = payload.get("output", {})
    loudness_data = output_data.get("loudness", {})
    output = OutputPolicy(sample_rate=output_data.get("sample_rate", 24000), channels=output_data.get("channels", 1), clip_policy=output_data.get("clip_policy", "clamp"), loudness=LoudnessPolicy(target_lufs=loudness_data.get("target_lufs"), true_peak_ceiling_dbtp=loudness_data.get("true_peak_ceiling_dbtp", -1.0), peak_policy=loudness_data.get("peak_policy", "reduce_gain")))
    items: list[AudioClip | Silence] = []
    root = Path(base_dir)
    for raw in payload.get("items", []):
        kind = raw.get("kind")
        if kind == "silence":
            items.append(Silence(raw["id"], float(raw["seconds"]), raw.get("metadata", {})))
        elif kind == "clip":
            anchors = tuple(AudioAnchor(a["id"], int(a["sample_offset"]), a.get("name")) for a in raw.get("anchors", []))
            spans = tuple(AudioSpan(int(s["source_start"]), int(s["source_end"]), int(s["sample_start"]), int(s["sample_end"])) for s in raw.get("spans", []))
            items.append(AudioClip(raw["id"], _source_from_dict(raw["source"], root), tuple(operation_from_dict(op) for op in raw.get("operations", [])), anchors, spans, raw.get("metadata", {})))
        else:
            raise AudioValidationError(f"unsupported AudioJob item kind: {kind!r}")
    job = AudioJob(tuple(items), output, payload.get("producer", {}), payload.get("job_id") or _job_id(payload), payload.get("source", {}))
    job.validate(base_dir=str(root))
    return job


def validate_job(job: AudioJob, *, base_dir: str | None = None) -> None:
    if job.output.channels != 1:
        raise AudioValidationError("only mono output is supported")
    for item in job.items:
        if isinstance(item, AudioClip):
            for operation in item.operations:
                if not hasattr(operation, "to_dict"):
                    raise AudioValidationError(f"unsupported operation on clip {item.id!r}")
            for anchor in item.anchors:
                audio, _ = item.source.load() if base_dir is not None else ((item.source.audio, item.source.sample_rate) if isinstance(item.source, AudioBufferSource) else (None, None))
                if audio is not None and anchor.sample_offset > len(audio):
                    raise AudioValidationError(f"anchor {anchor.id!r} exceeds source length for {item.id!r}")
            if isinstance(item.source, AudioFileSource):
                path = Path(item.source.path)
                if base_dir is not None and not path.is_absolute():
                    _safe_relative(str(path))
                item.source.load()


def save_job(job: AudioJob, path: str | Path) -> Path:
    destination = Path(path)
    bundle = destination if destination.suffix != ".json" else destination.parent
    bundle.mkdir(parents=True, exist_ok=True)
    parts = bundle / "parts"
    parts.mkdir(exist_ok=True)
    saved_items: list[AudioClip | Silence] = []
    for item in job.items:
        if isinstance(item, Silence):
            saved_items.append(item)
            continue
        if isinstance(item.source, AudioBufferSource):
            output = parts / f"{item.id}.wav"
            write_intermediate_wav(output, item.source.audio, item.source.sample_rate)
        else:
            source = Path(item.source.path)
            output = parts / source.name
            if source.resolve() != output.resolve():
                shutil.copyfile(source, output)
        info = wav_info(output)
        saved_items.append(AudioClip(item.id, AudioFileSource(output, sha256_file(output), info.sample_rate, info.channels, info.frames), item.operations, item.anchors, item.spans, item.metadata))
    serialized = job_to_dict(AudioJob(tuple(saved_items), job.output, job.producer, job.job_id, job.source), base_dir=str(bundle))
    serialized.setdefault("job_id", _job_id(serialized))
    manifest = bundle / "audiojob.json"
    manifest.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_job(path: str | Path) -> AudioJob:
    manifest = Path(path)
    if manifest.is_dir():
        manifest /= "audiojob.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioValidationError(f"cannot read AudioJob manifest {manifest}: {exc}") from exc
    return job_from_dict(payload, base_dir=manifest.parent)
