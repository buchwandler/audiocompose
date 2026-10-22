[![PyPI - Version](https://img.shields.io/pypi/v/audiocompose)](https://pypi.org/project/audiocompose/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiocompose)
![PyPI - Downloads](https://img.shields.io/pypi/dm/audiocompose)
[![codecov](https://codecov.io/gh/buchwandler/audiocompose/graph/badge.svg?token=viiusddeDL)](https://codecov.io/gh/buchwandler/audiocompose)

# audiocompose

`audiocompose` declaratively processes and composes existing audio fragments into finalized audio. The **AudioCompose core** is independent of text-to-speech engines, voices, models, G2P, and semantic speech plans. Optional integration modules may validate upstream artifacts and materialize them as ordinary producer-neutral `AudioJob` values; the core composer and persisted job schemas remain producer-neutral.

## AudioJob boundary

Producers normally create an `AudioJob` containing concrete clips, numeric audio operations, explicit silence, anchors, and provenance. An optional adapter may also materialize a job from a verified upstream handoff. In either case, `Composer` receives only the generic job, executes operations in order, resamples items to the output rate, assembles the timeline, applies complete-output loudness policy, and writes the final WAV.

```python
import numpy as np
from audiocompose import AudioBufferSource, AudioClip, AudioJob, AudioSpan, Composer, Gain, Silence
job = AudioJob((
    AudioClip("intro", AudioBufferSource(np.zeros(24000, dtype=np.float32), 24000), (Gain(-3),)),
    Silence("pause", 0.5),
))
result = Composer(sample_rate=24000).compose(job)
Composer().to_wav(job, "final.wav")
```

## DSP behavior and reproducibility

AudioCompose delegates band-limited resampling, static WSOLA time/pitch processing, and `RatePitchEnvelope` DSP to AudioSig. The envelope API requires `audiosig>=0.1.6,<0.2`. Contiguous static `Tempo` and `PitchShift` operations remain fused unless a `Gain` or fade separates them. Each envelope is one complete AudioSig call. Numeric operation semantics and timeline mapping are the compatibility contract; exact PCM samples may change between AudioCompose or AudioSig versions.

## Composition estimates and progress

`estimate_composition(job, sample_rate=None)` returns a deterministic `CompositionEstimate` before processing audio. It reports per-item and total output frames, duration, and an approximate frame-pass `total_work_units`; `unknown_items` contains zero-based item indexes whose source geometry is unavailable. If any item geometry is unknown, total frames, duration, and work remain unknown rather than being guessed. Work units account for source loading/integrity verification, grouped operations, resampling, item output, assembly, optional complete-output loudness, and final output preparation. They are a planning metric—not seconds or a promise of CPU time—and the model may evolve.

`Composer.compose()` accepts an optional synchronous `on_progress` callback. It receives typed `CompositionProgress` events for item loading, operations, resampling, assembly, complete-output loudness, and completion. When the estimate is known, events include monotonic `completed_work_units`, `total_work_units`, and `fraction`; otherwise the fraction and total stay `None`. Events carry generic item metadata and never print, alter the AudioJob, or affect composition identity.

```python
import sys

from audiocompose import (
    Composer,
    ProgressEstimator,
    estimate_composition,
)

estimate = estimate_composition(job)
print(estimate.total_output_frames, estimate.total_work_units)

timing = ProgressEstimator()  # one estimator per composition

def on_progress(event):
    view = timing.update(event)
    if view.estimated_remaining_seconds is not None:
        print(f"ETA {view.estimated_remaining_seconds:.1f}s", file=sys.stderr)

Composer().compose(job, on_progress=on_progress)
```

`ProgressTiming` reports elapsed time, fraction/percent when known, and optional estimated total/remaining seconds. `ProgressEstimator` uses a monotonic clock and observed work throughput; without an initial rate it warms up before showing an ETA. Callers may seed `initial_work_rate` in work units per second when they have a relevant prior measurement. Unknown totals stay indeterminate. Its status is `indeterminate`, `warming_up`, `estimated`, or `complete`. Timing is process-local and is never saved in AudioJob identity, provenance, schemas, results, or reports. Callback exceptions propagate to the caller. The callback is also available through `to_wav()` and `compose_to_wav()`.

## Mixed producers

Different producers can place clips and opaque timing metadata in one job without AudioCompose knowing their engine names:

```python
job = AudioJob((
    AudioClip(
        "engine-a",
        AudioBufferSource(np.zeros(1200, dtype=np.float32), 24000),
        spans=(AudioSpan(100, 105, 100, 900, id="engine_a.word"),),
        metadata={"engine_a.kind": "speech"},
    ),
    Silence("pause", 0.05),
    AudioClip(
        "engine-b",
        AudioBufferSource(np.zeros(800, dtype=np.float32), 16000),
        metadata={"engine_b.kind": "speech"},
    ),
))
result = Composer(sample_rate=24000).compose(job)
```

The producer resolves speech behavior before creating the job. AudioCompose handles completed audio, ordering, silence, resampling, markers, spans, loudness, diagnostics, and WAV output.

## File bundles

```python
job.save("chapter.audiojob")
loaded = AudioJob.load("chapter.audiojob")
Composer().compose_to_wav("chapter.audiojob", "chapter.wav")
```

Bundles contain deterministic `audiojob.json` plus writer-owned `parts/000001.wav` fragments. Saves stage parts, replace the owned `parts/` set, then atomically replace the manifest last; unrelated bundle files are preserved. Every saved source is canonical mono PCM32 WAV at its native sample rate. Paths cannot escape the bundle, duplicate source basenames cannot collide, and SHA-256, WAV metadata, and the canonical manifest `job_id` are checked when sources are loaded.

## CLI

```text
audiocompose validate chapter.audiojob
audiocompose inspect chapter.audiojob
audiocompose compose chapter.audiojob chapter.wav
```

`compose`, `compose-renderplan`, and `bundle compose` accept `--progress {auto,always,never}`. Progress is rendered to stderr and never contaminates JSON stdout.

```bash
audiocompose compose chapter.audiojob chapter.wav --progress always
audiocompose compose-renderplan chapter.renderplan.toml --render-dir render/chapter \
  --output chapter.wav --progress always
audiocompose bundle compose BOOK.ssmd-bundle --variant narration--<fingerprint> --json --progress always
```

The display includes elapsed time and a live ETA after enough work-rate samples exist. Numeric progress is omitted for unknown work geometry; named stages such as handoff preparation, composition, WAV writing/verification, and book chapter slots remain visible. A full composition may reach 100% before its output file is finished, so the renderer explicitly transitions to writing/verification and only reports command completion after success.

## Direct RenderPlan composition

AudioCompose consumes the shared `renderplan` contract directly; it does not create plans, run preflight, synthesize speech, or interpret producer-specific intent. The `renderplan>=0.1.2,<0.2` package is a runtime dependency. The adapter validates the plan and complete manifest correspondence, then verifies every speech WAV's occurrence, segment ID, safe path, SHA-256, and WAV geometry before creating a generic `AudioJob`. Speech WAVs must be mono. Unsafe paths, symlinks, incomplete manifests, and mismatched or invalid artifacts fail closed.

Compose a plan directly without first saving an AudioJob:

```bash
audiocompose compose-renderplan chapter.renderplan.toml \
  --render-dir render/chapter \
  --output chapter.wav \
  --sample-rate 24000 \
  --target-lufs -16 \
  --true-peak-ceiling-dbtp -1 \
  --explain
```

`--manifest` defaults to `<render-dir>/render-manifest.toml`. The CLI defaults `--media-root` to the plan file's directory; relative external `AudioItem` WAV paths must resolve beneath that root. The built-in Python local-WAV resolver requires an explicit `media_root` for relative paths. Local media is mono WAV; custom URI schemes require a Python `MediaResolver`. Only `clip_begin_ms`, `clip_end_ms`, `speed`, and `gain_db` are supported; unknown options and unsafe paths are rejected. Speed and gain become existing `Tempo` and `Gain` operations, and trims are checked against the resolved WAV.

RenderPlan `emphasis` controls are realized after synthesis as bounded relative gain: default/`moderate` +1.5 dB, `reduced` +0.75 dB, and `strong` +2.5 dB. The integrated commands, `compose-renderplan` and `bundle compose`, default to `--emphasis-policy segment_fallback`; this boosts the entire speech segment for partial character ranges and reports that word-level alignment was unavailable. Use `--emphasis-policy strict` to reject partial emphasis when exact alignment is required (the error recommends upstream splitting/alignment). Full-span controls are applied at clip scope. Overlapping controls choose the strongest level, never sum. These are relative pre-normalization gains; final loudness/peak processing may reduce the audible difference, and emphasis does not promise native prosody. The library materializer defaults to strict; generic AudioJob composition does not interpret speech controls.
The adapter follows `RenderPlan.flow` in order. It materializes each pause using its declared millisecond duration, including zero-duration pauses; there is no symbolic pause policy or inferred timing. Source spans are retained as opaque provenance metadata, not fabricated sample-aligned `AudioSpan` values. The Composer's normal seconds-to-samples behavior and output policy still apply.

### Public single-document Python API

The package-root `compose_renderplan` function verifies the handoff, materializes and composes one document, atomically writes its WAV, and returns a typed `RenderPlanComposeResult`. Other Python orchestration clients can use this supported API without importing `audiocompose.integrations`:

```python
import audiocompose

events = []
result = audiocompose.compose_renderplan(
    "chapter.renderplan.toml",
    render_dir="render/chapter",
    output="chapter.wav",
    output_policy=audiocompose.OutputPolicy(
        sample_rate=24000,
        clip_policy="warn",
    ),
    on_progress=events.append,
)
print(result.output, result.plan_id, result.duration_seconds)
```

If `manifest` is omitted, the API looks for `render-manifest.toml` inside `render_dir`. If `media_root` is omitted, relative external `AudioItem` WAV paths resolve beneath the plan file's parent directory. `output_policy` accepts the existing `OutputPolicy` (including sample rate, clip policy, target LUFS, true-peak ceiling, and peak policy); output remains mono-only. The default emphasis policy is `segment_fallback`; choose `strict` to reject partial-span emphasis. `result.timeline` and `result.diagnostics` provide typed provenance and warnings. Set `save_audiojob="chapter.audiojob"` to persist the materialized intermediate job; otherwise no AudioJob artifact is created.

The optional synchronous `on_progress` callback receives the same producer-neutral `CompositionProgress` events as `Composer`; its work/fraction fields and timing remain runtime-only.

Low-level materialization is also available for callers that need an AudioJob without producing a final WAV:

```python
from audiocompose.integrations.renderplan import load_and_materialize_renderplan_job

job = load_and_materialize_renderplan_job(
    "chapter.renderplan.toml",
    render_dir="render/chapter",
    media_root="media",
)
```

The manifest path is optional and uses the same default as the CLI. The result is an ordinary, inspectable `AudioJob`; use `--save-audiojob` on the direct CLI or `job.save("chapter.audiojob")` to persist it, then continue with the existing generic `validate`, `inspect`, `timeline`, and `compose` workflows.

## Bundle composition for rendered variants

AudioCompose consumes completed VoiceRender artifacts; it does not synthesize speech. The bundle commands accept the editable `.ssmd-bundle` workspace root and dispatch by `manifest.json` format. Discover planned variants before composing:

```bash
audiocompose bundle variants project.ssmd-bundle
audiocompose bundle variants project.ssmd-bundle --json
```

`bundle variants` is read-only: it lists direct RenderPlan variant directories and reports plan validity, VoiceRender availability, AudioCompose composition status, and whether the handoff is ready. It supports book and document bundles, isolates malformed candidates, accepts `--renderplan-root`, `--voicerender-root`, and `--output-root` overrides, and does not render, compose, create output directories, or require a TTS engine.

The overall state is `not_rendered` when no VoiceRender output is present, `render_partial` when rendering is incomplete, `ready` when the render handoff is complete and there is no current composition, `complete` when AudioCompose output is current, and `stale` when composition provenance no longer matches upstream artifacts. `invalid`, `render_failed`, `compose_partial`, `compose_failed`, and `unknown` identify invalid or contradictory stage results. Check `ready_to_compose` in JSON when a complete manifest may still be missing required diagnostics.

From inside a bundle, `.` is a valid spelling for the canonical workspace root:

```bash
cd project.ssmd-bundle
audiocompose bundle variants .
audiocompose bundle compose .
audiocompose bundle compose . --force
audiocompose bundle compose . --variant narration--<fingerprint>
```

Explicit paths also work:

```bash
audiocompose bundle compose project.ssmd-bundle --variant narration--<fingerprint>
```

A book manifest composes selected chapters (`--chapters all|N|N-M|...`); a document manifest composes its single document and accepts only `--chapters all`. Omit `--variant` to compose every valid, compose-ready variant with open composition work. Variants without a complete VoiceRender handoff are skipped, complete variants are skipped by default, and invalid candidates are reported. Batch `--force` includes complete compose-ready variants and forwards overwrite permission to the existing composer; stale or unverified outputs still obey the single-variant provenance checks. `--fail-fast` also stops later selected variants after a failure. In batch mode an explicit `--save-audiojobs DIR` is namespaced beneath `DIR/<variant>` to prevent collisions. Target one variant with `--variant NAME`. ZIP archives and inner chapter/document paths are not command inputs; the bundle manifest remains authoritative.

Defaults are `renderplan/<variant>`, `voicerender/<variant>`, and `audiocompose/<variant>` beneath the workspace. Book workspaces save one AudioJob per selected chapter below the `--save-audiojobs` root; document workspaces save their single AudioJob below that root. Batch output and status are re-inspected from persisted artifacts after composition; the batch layer only selects and orchestrates calls to the existing `compose_bundle()` facade, so kind-specific reports, verification, and reuse policies remain unchanged.

The public `compose_bundle(bundle, *, variant, options=...)` facade and `BundleComposeOptions` expose single-variant manifest-based dispatch. `BundleComposeResult` wraps the kind-specific result and exposes `kind`, `value`, `exit_code`, and neutral JSON via `to_dict()`. `audiocompose.list_bundle_variants(bundle)` returns typed `BundleVariantsResult` and `BundleVariant` values; the read-only listing is the state source for the public `compose_bundle_batch(bundle, ...)` API. Batch results expose per-variant initial/final state, attempt status, errors, and aggregate counts.

```python
from audiocompose import compose_bundle_batch

result = compose_bundle_batch("story.ssmd-bundle")
for variant in result.variants:
    print(variant.name, variant.status)
raise SystemExit(result.exit_code)
```

## Debugging and analysis

The CLI also exposes generic, TTS-neutral diagnostics:

```text
audiocompose --version
audiocompose validate JOB --json
audiocompose inspect JOB --json
audiocompose timeline JOB --json
audiocompose analyze INPUT --json
audiocompose report INPUT -o report.html
```

`analyze` detects waveform activity and acoustic gaps using sample coordinates. It does not classify speech, infer semantic pauses, or interpret model timing tensors. `report` creates a self-contained HTML/SVG view of the waveform, activity regions, gaps, and machine-readable measurements.

## Schema and provenance

The canonical packaged AudioJob schemas are in `audiocompose/schemas/`; the repository-level `spec/` files mirror those resources and are checked for byte equality. `audiojob_schema(1)` and `audiojob_schema(2)` load the corresponding JSON Schema for external validation. New jobs use v2; a loaded v1 job retains v1 when saved, and `RatePitchEnvelope` requires v2. Package versioning is SCM-derived and independent from the persisted AudioJob schema version. Stable upstream segment IDs should be used as generic `AudioClip.id` values where a one-to-one mapping exists; producer metadata remains opaque.

The package root intentionally exposes common composition models, built-in operations, composition results, diagnostics, progress, errors, and schema loading. Specialized analysis, WAV, resampling, and low-level DSP helpers remain available from their modules, for example `audiocompose.analysis`, `audiocompose.wav`, `audiocompose.resampling`, and `audiocompose.operations`.
See [the v0.2 migration guide](docs/migration-v0.2.md) for breaking import, persistence, clipping, loudness, and progress changes.

`AudioSpan` and `ComposedSpan` carry optional producer-defined IDs and JSON-safe metadata. AudioCompose preserves these opaque values while mapping sample coordinates through operations and resampling; it never interprets the metadata.

Composition completes in this order: load clips, apply operations, map anchors and spans, resample, concatenate clips and explicit silence, then measure and apply complete-output loudness once. Loudness changes waveform amplitude only, so item ranges, markers, and spans remain stable.

For producer frame parity, quantize a duration to producer frames before constructing a job: `frames = int(seconds * producer_rate)` and `seconds = frames / producer_rate`. AudioCompose keeps its public rounded seconds-to-samples rule and does not add engine-specific silence types.

`CompositionResult.loudness` exposes typed before/after metrics, requested and applied gain, target and ceiling, and warnings. `CompositionResult.diagnostics` contains generic codes and numeric context for loudness limitations and other composition warnings.
The core does not add engine-specific branches or interpret upstream semantics. The RenderPlan adapter is an explicit boundary module: it verifies the shared plan, manifest, and speech artifacts, then lowers the ordered flow into a generic job without changing Composer or AudioJob schemas.

## Supported operations

AudioJob v1 supports `Gain`, `PitchShift`, `Tempo`, `FadeIn`, and `FadeOut`. AudioJob v2 adds `AutomationPoint` and `RatePitchEnvelope`. Envelope point times are seconds on the operation output timeline; rate values are playback factors, pitch values are semitone offsets, interpolation is linear, and the last value is held. An omitted rate or pitch curve means identity for that dimension.

```python
from audiocompose import RatePitchEnvelope

operation = RatePitchEnvelope.transition(
    from_rate=1.0,
    to_rate=0.85,
    rate_seconds=0.450,
    from_semitones=0.0,
    to_semitones=2.0,
    pitch_seconds=0.300,
)
```

The envelope describes numeric DSP only. The producer decides whether a transition is appropriate; AudioCompose does not compare voices or interpret producer metadata. Semantic values such as `slow`, `loud`, voices, and phonemes remain producer-layer concepts.

## Development

```text
python -m build
audiocompose --version
python -m pytest -q
ruff check .
mypy audiocompose
```
