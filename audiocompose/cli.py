from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .analysis import ActivityConfig, ActivityReport, analyze_activity, measure_gap_near
from .composer import Composer
from .errors import AudioComposeError
from .job import validate_clip_geometry
from .loudness import LoudnessPolicy
from .model import AudioClip, AudioJob, CompositionResult, OutputPolicy, Silence
from .wav import read_wav, write_wav


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="write machine-readable JSON")


def _add_analysis_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=5.0)
    parser.add_argument("--active-threshold-dbfs", type=float, default=-40.0)
    parser.add_argument("--release-threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--min-active-ms", type=float, default=30.0)
    parser.add_argument("--min-gap-ms", type=float, default=60.0)
    parser.add_argument("--near-marker")
    parser.add_argument("--window-ms", type=float, default=500.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiocompose", description="Compose declarative AudioJob bundles"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("manifest")
    _add_json(validate)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("manifest")
    _add_json(inspect)

    compose = subparsers.add_parser("compose")
    compose.add_argument("manifest")
    compose.add_argument("output")

    timeline = subparsers.add_parser("timeline")
    timeline.add_argument("manifest")
    _add_json(timeline)

    compose_renderplan = subparsers.add_parser(
        "compose-renderplan", help="verify and directly compose a RenderPlan and RenderManifest"
    )
    compose_renderplan.add_argument("plan", help="shared RenderPlan TOML file")
    compose_renderplan.add_argument(
        "--manifest", help="RenderManifest TOML (defaults inside render-dir)"
    )
    compose_renderplan.add_argument(
        "--render-dir", required=True, help="directory containing speech WAVs"
    )
    compose_renderplan.add_argument(
        "--media-root", help="base directory for relative external AudioItems"
    )
    compose_renderplan.add_argument("--output", required=True, help="final output WAV path")
    compose_renderplan.add_argument(
        "--save-audiojob", help="optional directory for saving the intermediate AudioJob"
    )
    compose_renderplan.add_argument("--sample-rate", type=int)
    compose_renderplan.add_argument("--clip-policy", choices=("clamp", "warn", "error"))
    compose_renderplan.add_argument("--target-lufs", type=float)
    compose_renderplan.add_argument("--true-peak-ceiling-dbtp", type=float)
    compose_renderplan.add_argument("--peak-policy", choices=("reduce_gain", "error"))
    compose_renderplan.add_argument(
        "--explain", action="store_true", help="print ordered RenderPlan provenance"
    )
    _add_json(compose_renderplan)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("input")
    _add_analysis_options(analyze)
    _add_json(analyze)

    report = subparsers.add_parser("report")
    report.add_argument("input")
    report.add_argument("-o", "--output", required=True)
    _add_analysis_options(report)

    return parser


def _analysis_config(args: argparse.Namespace) -> ActivityConfig:
    return ActivityConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        active_threshold_dbfs=args.active_threshold_dbfs,
        release_threshold_dbfs=args.release_threshold_dbfs,
        min_active_ms=args.min_active_ms,
        min_gap_ms=args.min_gap_ms,
    )


def _report_dict(report: ActivityReport) -> dict[str, Any]:
    def gap_dict(gap: Any) -> dict[str, Any]:
        return {
            "start_sample": gap.start_sample,
            "end_sample": gap.end_sample,
            "duration_samples": gap.duration_samples,
            "duration_seconds": report.seconds(gap.duration_samples),
        }

    return {
        "sample_rate": report.sample_rate,
        "frames": report.frames,
        "duration_seconds": report.duration_seconds,
        "activity": [
            {
                "start_sample": region.start_sample,
                "end_sample": region.end_sample,
                "duration_samples": region.duration_samples,
                "rms_dbfs": region.rms_dbfs,
                "peak_dbfs": region.peak_dbfs,
            }
            for region in report.activity
        ],
        "gaps": [gap_dict(gap) for gap in report.gaps],
        "leading_gap": gap_dict(report.leading_gap) if report.leading_gap else None,
        "trailing_gap": gap_dict(report.trailing_gap) if report.trailing_gap else None,
    }


def _load_input(
    path: str,
) -> tuple[Any, int, AudioJob | None, CompositionResult | None]:
    if Path(path).suffix.lower() == ".wav":
        audio, sample_rate = read_wav(path)
        return audio, sample_rate, None, None
    job = AudioJob.load(path, verify_sources=False)
    result = Composer().compose(job)
    return result.audio, result.sample_rate, job, result


def _inspect(job: AudioJob) -> dict[str, Any]:
    clips = [item for item in job.items if isinstance(item, AudioClip)]
    silence = [item for item in job.items if isinstance(item, Silence)]
    loaded = [(item, *item.source.load()) for item in clips]
    for item, audio, _ in loaded:
        validate_clip_geometry(item, audio)
    rates = sorted({rate for _, _, rate in loaded})
    operations = sorted({operation.type for item in clips for operation in item.operations})
    raw_duration = sum(len(audio) / rate for _, audio, rate in loaded) + sum(
        item.seconds for item in silence
    )
    return {
        "job_id": job.job_id,
        "producer": dict(job.producer),
        "clips": len(clips),
        "silence": len(silence),
        "source_sample_rates": rates,
        "operations": operations,
        "expected_output_rate": job.output.sample_rate,
        "total_raw_duration": raw_duration,
        "loudness": {
            "target_lufs": job.output.loudness.target_lufs,
            "true_peak_ceiling_dbtp": job.output.loudness.true_peak_ceiling_dbtp,
        },
    }


def _timeline(job: AudioJob) -> dict[str, Any]:
    result = Composer().compose(job)
    return {
        "sample_rate": result.sample_rate,
        "items": [
            {
                "id": item.item_id,
                "kind": item.kind,
                "start_sample": item.start_sample,
                "end_sample": item.end_sample,
            }
            for item in result.items
        ],
        "markers": [
            {
                "id": marker.id,
                "item_id": marker.item_id,
                "sample_offset": marker.sample_offset,
                "name": marker.name,
            }
            for marker in result.markers
        ],
        "spans": [
            {
                "item_id": span.item_id,
                "source_start": span.source_start,
                "source_end": span.source_end,
                "sample_start": span.sample_start,
                "sample_end": span.sample_end,
            }
            for span in result.spans
        ],
    }


def _output_policy_from_args(args: argparse.Namespace) -> OutputPolicy:
    defaults = OutputPolicy()
    loudness = LoudnessPolicy(
        target_lufs=args.target_lufs,
        true_peak_ceiling_dbtp=(
            defaults.loudness.true_peak_ceiling_dbtp
            if args.true_peak_ceiling_dbtp is None
            else args.true_peak_ceiling_dbtp
        ),
        peak_policy=args.peak_policy or defaults.loudness.peak_policy,
        collect_metrics=defaults.loudness.collect_metrics,
    )
    return OutputPolicy(
        sample_rate=defaults.sample_rate if args.sample_rate is None else args.sample_rate,
        clip_policy=args.clip_policy or defaults.clip_policy,
        loudness=loudness,
    )


def _renderplan_timeline(job: AudioJob, composition: CompositionResult) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for item, composed in zip(job.items, composition.items, strict=True):
        metadata = dict(item.metadata)
        kind = str(metadata.get("kind", composed.kind))
        entry: dict[str, Any] = {
            "id": item.id,
            "kind": kind,
            "start_sample": composed.start_sample,
            "end_sample": composed.end_sample,
            "duration_seconds": composed.duration_samples / composition.sample_rate,
        }
        for key in (
            "renderplan.segment_number",
            "renderplan.segment_id",
            "renderplan.source",
            "renderplan.pause_id",
            "renderplan.pause_reason",
            "renderplan.pause_source",
            "renderplan.audio_id",
            "renderplan.audio.src",
            "renderplan.audio.options",
            "render_manifest.wav",
            "render_manifest.wav_sha256",
        ):
            if key in metadata:
                entry[key] = metadata[key]
        timeline.append(entry)
    return timeline


def _print_renderplan_explanation(
    plan_id: str, manifest_id: str, timeline: list[dict[str, Any]]
) -> None:
    print(f"RenderPlan: {plan_id}")
    print(f"RenderManifest: {manifest_id}")
    for index, item in enumerate(timeline, start=1):
        kind = item["kind"]
        duration = item["duration_seconds"]
        label = item.get("render_manifest.wav", item.get("renderplan.audio.src", ""))
        print(f"{index:06d} {kind:<12} {duration:.3f}s  {label}")
        if "renderplan.segment_id" in item:
            print(f"       segment: {item['renderplan.segment_id']}")
        if "renderplan.source" in item:
            spans = item["renderplan.source"]
            rendered = ", ".join(
                f"{span['locator']}[{span['start']}:{span['end']}]" for span in spans
            )
            print(f"       source:  {rendered}")
        if "renderplan.pause_reason" in item:
            print(f"       reason:  {item['renderplan.pause_reason']}")
        if "renderplan.audio.options" in item and item["renderplan.audio.options"]:
            print(f"       options: {item['renderplan.audio.options']}")


def _svg_report(audio: Any, report: ActivityReport, job: AudioJob | None) -> str:
    width = 1000
    height = 260
    scale = width / max(1, len(audio))
    lines: list[str] = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    for gap in report.gaps:
        x = gap.start_sample * scale
        w = max(1.0, gap.duration_samples * scale)
        lines.append(f'<rect x="{x:.2f}" y="20" width="{w:.2f}" height="180" fill="#fee2e2"/>')
    if len(audio):
        step = max(1, len(audio) // width)
        points = []
        for index in range(0, len(audio), step):
            values = audio[index : index + step]
            points.append(f"{index * scale:.2f},{130 - float(max(abs(values))) * 100:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="#1d4ed8" stroke-width="1"/>'
        )
    for region in report.activity:
        x = region.start_sample * scale
        w = max(1.0, region.duration_samples * scale)
        lines.append(f'<rect x="{x:.2f}" y="205" width="{w:.2f}" height="20" fill="#86efac"/>')
    lines.append("</svg>")
    payload = json.dumps(_report_dict(report), indent=2, sort_keys=True)
    title = "AudioCompose activity report"
    if job is not None:
        title += f" ({job.job_id})"
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>'
        + html.escape(title)
        + "</title></head><body>"
        + f"<h1>{html.escape(title)}</h1>"
        + "<p>Green regions are detected audio activity. Red regions are acoustic gaps.</p>"
        + "<section>"
        + "".join(lines)
        + "</section><pre>"
        + html.escape(payload)
        + "</pre></body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compose-renderplan":
            from .integrations.renderplan import load_and_materialize_renderplan_job

            plan_path = Path(args.plan).expanduser()
            media_root = (
                Path(args.media_root).expanduser()
                if args.media_root is not None
                else plan_path.absolute().parent
            )
            job = load_and_materialize_renderplan_job(
                args.plan,
                args.manifest,
                render_dir=args.render_dir,
                output_policy=_output_policy_from_args(args),
                media_root=media_root,
            )
            audiojob_path = job.save(args.save_audiojob) if args.save_audiojob else None
            composition = Composer().compose(job)
            output_path = write_wav(
                args.output,
                composition.audio,
                composition.sample_rate,
                clip_policy=job.output.clip_policy,
            )
            timeline = _renderplan_timeline(job, composition)
            result = {
                "composed": True,
                "output": str(output_path),
                "plan_id": job.source["renderplan.plan_id"],
                "manifest_id": job.source["render_manifest.manifest_id"],
                "sample_rate": composition.sample_rate,
                "frames": len(composition.audio),
                "duration_seconds": composition.duration_seconds,
                "timeline": timeline,
            }
            if audiojob_path is not None:
                result["audiojob"] = str(audiojob_path)
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"wrote {output_path} ({composition.duration_seconds:.3f}s)")
                if args.explain:
                    _print_renderplan_explanation(
                        result["plan_id"], result["manifest_id"], timeline
                    )
        elif args.command == "validate":
            validated_job = AudioJob.load(args.manifest)
            result = {
                "valid": True,
                "format": "audiojob",
                "schema_version": validated_job.schema_version,
                "job_id": validated_job.job_id,
            }
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"valid AudioJob: {args.manifest}")
        elif args.command == "inspect":
            result = _inspect(AudioJob.load(args.manifest, verify_sources=False))
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "compose":
            Composer().to_wav(AudioJob.load(args.manifest, verify_sources=False), args.output)
            print(f"wrote {args.output}")
        elif args.command == "timeline":
            result = _timeline(AudioJob.load(args.manifest, verify_sources=False))
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command in {"analyze", "report"}:
            audio, sample_rate, job, composition = _load_input(args.input)
            report = analyze_activity(audio, sample_rate, _analysis_config(args))
            if args.command == "analyze":
                result = _report_dict(report)
                if args.near_marker:
                    if job is None or composition is None:
                        raise AudioComposeError("--near-marker requires an AudioJob input")
                    marker = next(
                        (marker for marker in composition.markers if marker.id == args.near_marker),
                        None,
                    )
                    if marker is None:
                        raise AudioComposeError(f"unknown marker: {args.near_marker}")
                    gap = measure_gap_near(
                        report, marker.sample_offset, search_window_ms=args.window_ms
                    )
                    result["near_marker"] = args.near_marker
                    result["near_gap"] = (
                        _report_dict(
                            ActivityReport(
                                report.sample_rate,
                                report.frames,
                                (),
                                (gap,) if gap else (),
                                gap,
                                gap,
                            )
                        )["leading_gap"]
                        if gap
                        else None
                    )
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                Path(args.output).write_text(_svg_report(audio, report, job), encoding="utf-8")
                print(f"wrote {args.output}")
        return 0
    except (AudioComposeError, ValueError, OSError) as exc:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {"valid": False, "errors": [{"code": "validation_error", "message": str(exc)}]},
                    indent=2,
                ),
                file=sys.stderr,
            )
        else:
            print(f"audiocompose: {exc}", file=sys.stderr)
        return 2
