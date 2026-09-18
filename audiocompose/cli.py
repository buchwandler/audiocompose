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
from .model import AudioClip, AudioJob, Silence
from .wav import read_wav


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


def _load_input(path: str) -> tuple[Any, int, AudioJob | None]:
    if Path(path).suffix.lower() == ".wav":
        audio, sample_rate = read_wav(path)
        return audio, sample_rate, None
    job = AudioJob.load(path)
    result = Composer().compose(job)
    return result.audio, result.sample_rate, job


def _inspect(job: AudioJob) -> dict[str, Any]:
    clips = [item for item in job.items if isinstance(item, AudioClip)]
    silence = [item for item in job.items if isinstance(item, Silence)]
    loaded = [(item, *item.source.load()) for item in clips]
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
        if args.command == "validate":
            validated_job = AudioJob.load(args.manifest)
            result = {
                "valid": True,
                "format": "audiojob",
                "schema_version": 1,
                "job_id": validated_job.job_id,
            }
            if args.json:
                print(json.dumps(result, indent=2, sort_keys=True))
            else:
                print(f"valid AudioJob: {args.manifest}")
        elif args.command == "inspect":
            result = _inspect(AudioJob.load(args.manifest))
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "compose":
            Composer().to_wav(AudioJob.load(args.manifest), args.output)
            print(f"wrote {args.output}")
        elif args.command == "timeline":
            result = _timeline(AudioJob.load(args.manifest))
            print(json.dumps(result, indent=2, sort_keys=True))
        elif args.command in {"analyze", "report"}:
            audio, sample_rate, job = _load_input(args.input)
            report = analyze_activity(audio, sample_rate, _analysis_config(args))
            if args.command == "analyze":
                result = _report_dict(report)
                if args.near_marker:
                    if job is None:
                        raise AudioComposeError("--near-marker requires an AudioJob input")
                    timeline = Composer().compose(job)
                    marker = next(
                        (marker for marker in timeline.markers if marker.id == args.near_marker),
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
