from __future__ import annotations

import argparse
import json
import sys

from .composer import Composer
from .errors import AudioComposeError
from .model import AudioClip, AudioJob, Silence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audiocompose", description="Compose declarative AudioJob bundles")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "inspect"):
        subparsers.add_parser(name).add_argument("manifest")
    compose = subparsers.add_parser("compose")
    compose.add_argument("manifest")
    compose.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        job = AudioJob.load(args.manifest)
        if args.command == "validate":
            print(f"valid AudioJob: {args.manifest}")
        elif args.command == "inspect":
            clips = [item for item in job.items if isinstance(item, AudioClip)]
            silence = [item for item in job.items if isinstance(item, Silence)]
            rates = sorted({item.source.load()[1] for item in clips})
            operations = sorted({operation.type for item in clips for operation in item.operations})
            raw_duration = sum(len(item.source.load()[0]) / item.source.load()[1] for item in clips) + sum(item.seconds for item in silence)
            print(json.dumps({"job_id": job.job_id, "producer": dict(job.producer), "clips": len(clips), "silence": len(silence), "source_sample_rates": rates, "operations": operations, "expected_output_rate": job.output.sample_rate, "total_raw_duration": raw_duration, "loudness": {"target_lufs": job.output.loudness.target_lufs, "true_peak_ceiling_dbtp": job.output.loudness.true_peak_ceiling_dbtp}}, indent=2, sort_keys=True))
        else:
            Composer().to_wav(job, args.output)
            print(f"wrote {args.output}")
        return 0
    except AudioComposeError as exc:
        print(f"audiocompose: {exc}", file=sys.stderr)
        return 2
