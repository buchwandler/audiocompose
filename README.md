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

`compose`, `compose-renderplan`, `compose-book`, and `compose-document` accept `--progress {auto,always,never}`. The default `auto` renders only when stderr is a TTY and `--json` is not requested. `always` forces progress to stderr (including alongside JSON); `never` disables it. TTY output refreshes a throttled live line, while forced non-TTY output uses newline-delimited updates. Progress never contaminates JSON stdout.

```bash
audiocompose compose chapter.audiojob chapter.wav --progress always
audiocompose compose-renderplan chapter.renderplan.toml --render-dir render/chapter \
  --output chapter.wav --progress always
audiocompose compose-book BOOK --variant narration--<fingerprint> --json --progress always
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

RenderPlan `emphasis` controls are realized after synthesis as bounded relative gain: default/`moderate` +1.5 dB, `reduced` +0.75 dB, and `strong` +2.5 dB. Both integrated commands, `compose-renderplan` and `compose-book`, default to `--emphasis-policy segment_fallback`; this boosts the entire speech segment for partial character ranges and reports that word-level alignment was unavailable. Use `--emphasis-policy strict` to reject partial emphasis when exact alignment is required (the error recommends upstream splitting/alignment). Full-span controls are applied at clip scope. Overlapping controls choose the strongest level, never sum. These are relative pre-normalization gains; final loudness/peak processing may reduce the audible difference, and emphasis does not promise native prosody. The library materializer defaults to strict; generic AudioJob composition does not interpret speech controls.
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

## Book-level composition for rendered variants

AudioCompose consumes completed VoiceRender artifacts; it does not synthesize speech. If the RenderPlan contains supported `emphasis`, explicitly ask VoiceRender to defer that capability at both preflight and render, then compose the completed variant:

```bash
# If preflight is a separate step, also pass --emphasis-policy defer-to-compose there.
voicerender render-book BOOK \
  --variant narration--<fingerprint> \
  --emphasis-policy defer-to-compose

audiocompose compose-book BOOK \
  --variant narration--<fingerprint>
```

The book handoff requires a complete RenderManifest and VoiceRender v0.2 diagnostics. AudioCompose validates producer identity, plan correspondence, diagnostics provenance, and declared deferred controls; it owns only the declared `audiocompose.emphasis.gain.v1` capability. VoiceRender-native controls remain producer-owned, and unknown deferred capabilities fail closed. The optional `audiocompose[voicerender]` extra requires UtterPlan `>=0.6.4,<0.7` and VoiceRender `>=0.2.4,<0.3`; these floors are required for the SSMDDoc direct-workspace contract.
The minimal Python counterpart is in [examples/compose_rendered_book.py](examples/compose_rendered_book.py); it assumes rendered artifacts already exist and never synthesizes speech.

`BOOK` may be the workspace root or its canonical `chapters/` directory. The required `--variant` is the exact `<profile-label>--<profile-fingerprint>` directory present under both `renderplan/` and `voicerender/`. AudioCompose loads `renderplan/<variant>/renderplan-set.toml` through `renderplan`; this set is authoritative for selected chapter IDs and order. A number selector, inclusive range, comma-separated selectors, or `all` may be passed with `--chapters` (default `all`):

```bash
audiocompose compose-book . --variant narration--525856299523 --chapters 5
audiocompose compose-book . --variant narration--525856299523 --chapters 5-7
audiocompose compose-book . --variant narration--525856299523 --chapters 5,8,10-12
```

Use the RenderProfile snapshot's pause values by default. An explicit strict AudioCompose policy can override it:

```bash
audiocompose compose-book . --variant narration--525856299523 \\
  --pauses narration.pause.toml
```

Pause-policy `role` overrides are keyed by the RenderPlan `SpeechItem.role`, not UtterPlan's logical voice reference. Source-pause validation checks known classes and table availability without treating logical voice references as role keys; actual durations resolve against the RenderPlan `SpeechItem.role` selected for each pause, failing closed if that role/default lacks a value.

By default, one final WAV per selected chapter and an atomic report are written under `audiocompose/<variant>/`:

```text
audiocompose/<variant>/
├── compose-report.toml
├── chapter-0005.audiojob/
├── chapter-0005.wav
└── chapter-0006.wav
```

Optional `--renderplan-root`, `--voicerender-root`, and `--output-root` override the corresponding book roots. `--media-root` is forwarded to every RenderPlan materialization; when omitted, relative external `AudioItem` WAV paths use the individual plan's directory. Output mastering options (`--sample-rate`, `--clip-policy`, `--target-lufs`, `--true-peak-ceiling-dbtp`, `--peak-policy`) match the direct command. Reproducible AudioJob bundles are always saved beside the chapter WAVs by default; `--save-audiojobs DIR` overrides their root. Supply `--pauses POLICY.toml` to override the default policy derived from the selected RenderProfile snapshot.

Normal production composition requires a **complete** chapter RenderManifest. Partial VoiceRender output is useful for QC, but `compose-book` rejects a selected chapter whose manifest is incomplete; finish rendering before composing. The specialist book facade verifies the chapter handoff and persists its reproducible AudioJob, then invokes the ordinary producer-neutral `Composer`. For books, symbolic UtterPlan pause intents resolve through a strict AudioCompose policy derived from the selected RenderProfile snapshot by default or supplied with `--pauses`; direct `compose-renderplan` continues to use literal RenderPlan pause durations. The base install does not require UtterPlan or VoiceRender; install `audiocompose[voicerender]` for optional book and SSMDDoc integrations.

Reuse is conservative: the report, persisted AudioJob, and WAV must match the current variant, source chapter/plan, RenderManifest identity and file hash, pause/output policies, AudioJob ID/schema, AudioCompose version, output hash, and WAV geometry. Symlinked or stale artifacts are never reused; an existing output without matching provenance fails with `output.exists`, while `--force` recomposes and atomically replaces selected WAVs. The batch continues after independent chapter failures by default; `--fail-fast` marks later selections skipped. Ordered progress callbacks receive book and chapter events; callback exceptions become `progress.callback_failed`. `--json` writes only the machine-readable result to stdout, while the human summary includes the report path.

### Public Python service

Python orchestrators can compose the same exact variant without constructing an internal `BookContext` or reimplementing CLI defaults:

```python
from audiocompose import BookComposeOptions, compose_book

events = []
result = compose_book(
    "BOOK",
    variant="narration--525856299523",
    chapters="5-7",  # source chapter numbers, not positional indexes
    options=BookComposeOptions(sample_rate=24000, target_lufs=-16.0),
    on_progress=events.append,
)
for event in events:
    if event.composition is not None:
        print(
            event.kind,
            event.chapter_id,
            event.fraction,
            event.composition.kind,
            event.composition.fraction,
        )

assert result.exit_code == 0
for outcome in result.outcomes:
    print(outcome.id, outcome.status, outcome.output, outcome.sha256)
```

`compose_book(book, *, variant, chapters="all", ...)` returns a `BookComposeResult` with workspace/variant roots, report path, profile and pause-policy identities, typed outcomes, and requested/written/reused/failed/skipped counts. `exit_code` is `2` if a chapter fails or the report/callback fails. Each `ChapterComposeOutcome` has a status of `written`, `reused`, `failed`, or `skipped`, an output `Path`, provenance, and stable error fields when applicable. The progress callback receives ordered `BookComposeProgress` events (`book_started`, chapter start/materialization/composition/result, and `book_completed`). Each event exposes a uniform chapter-slot `fraction` (the current chapter's core fraction fills its slot, not a duration-weighted forecast) and nested `composition` events while composing; nested work/timing stay runtime-only. A callback failure is recorded as `progress.callback_failed`.

Pass either a complete AudioCompose `output_policy` or `BookComposeOptions`, not both. Options expose sample rate, channels, clip policy, target loudness, true-peak ceiling, peak policy, and metric collection; omitted values inherit `OutputPolicy()` defaults, including mono-only output. `default_book_output_policy(options)` is available when a caller needs the same defaults independently. `BookComposeResult.to_dict()` returns the JSON-compatible shape used by the CLI. The CLI delegates composition to this service; it does not implement a second book pipeline.

`compose-report.toml` is the version-1 filesystem hand-off (`format = "audiocompose-book-report"`, `version = 1`). Its root records the variant, profile and pause-policy identities/provenance, AudioCompose version, relative workspace/AudioJob roots, counts, and output policy. Each item includes source/order and handoff identity, report-relative UtterPlan/RenderManifest paths, output basename, status, and any available errors. Successful items also record RenderManifest ID and file hash, pause/output-policy IDs, AudioJob ID/schema and bundle name, WAV SHA-256, sample rate, channels, and frames; newly written outputs include duration. Paths stay relative to their owning book/report roots where practical. Unselected prior item records remain in RenderPlan-set order when a subset is composed.

## SSMDDoc document-workspace composition

`audiocompose compose-document` composes one completed SSMDDoc from its canonical workspace. Pass either the workspace directory or its canonical `document.ssmd.md`; AudioCompose reads the SSMDDoc v1 manifest and resolves the direct stage roots:

```text
WORKSPACE/
├── manifest.json
├── document.ssmd.md
├── utterplan/document.utterplan.toml
├── renderplan/<variant>/
│   ├── renderprofile.toml
│   └── document.renderplan.toml
├── voicerender/<variant>/document-0001/
│   ├── render-manifest.toml
│   ├── render-diagnostics.json
│   └── 000001.wav
└── audiocompose/<variant>/
    ├── document-0001.audiojob/
    ├── document-0001.wav
    └── compose-report.toml
```

The selected variant is the exact `<variant>` directory under each producer root. AudioCompose verifies the RenderProfile fingerprint and RenderPlan/source identities, then requires a complete VoiceRender manifest, matching speech artifacts, and trusted diagnostics before composing. It does not create plans, run preflight, or synthesize speech.

```bash
audiocompose compose-document . \\
  --variant narration--<fingerprint> \\
  --sample-rate 24000 \\
  --progress always \\
  --json
```

The flat CLI also accepts `--progress`, `--pauses`, `--renderplan-root`, `--voicerender-root`, `--output-root`, `--media-root`, `--emphasis-policy`, `--clip-policy`, `--target-lufs`, `--true-peak-ceiling-dbtp`, `--peak-policy`, `--save-audiojob`, and `--force`. Output is written beneath `audiocompose/<variant>/` by default: one `document-0001.wav`, its reproducible `document-0001.audiojob/`, and an atomic `compose-report.toml`. Existing WAVs are reused only when the report, AudioJob, current handoff identities, output policy, content hash, and WAV geometry all match; stale or unverified files fail closed unless `--force` is supplied.

The package-root API exposes the same service:

```python
from audiocompose import (
    CompositionProgress,
    CompositionProgressCallback,
    DocumentComposeOptions,
    compose_document,
)

events: list[CompositionProgress] = []
progress_callback: CompositionProgressCallback = events.append
result = compose_document(
    ".",
    variant="narration--525856299523",
    options=DocumentComposeOptions(sample_rate=24_000),
    on_progress=progress_callback,
)
print(result.status, result.output, result.report_path)
```

`on_progress` uses the existing `CompositionProgressCallback` contract and reports actual Composer work. When a verified output is reused, no DSP runs and no composition events are emitted; AudioCompose does not synthesize progress for reuse. Progress is observational only: it does not change artifact identity, output roots, or output/report behavior. The `renderplan_root`, `voicerender_root`, `output_root`, and media-root overrides work as before.

UtterPlan owns semantic planning and VoiceRender owns synthesis and producer-native controls. AudioCompose follows RenderPlan flow order and role semantics, derives symbolic pause durations from the RenderProfile snapshot by default (or accepts a strict explicit pause policy), and realizes only the explicitly deferred emphasis capability it owns. All completed audio is lowered to the ordinary producer-neutral `AudioJob` and passed through the existing `Composer`; no document-specific DSP path is introduced.

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
See the [v0.3 migration guide](docs/migration-v0.3.md) for RenderPlan and optional VoiceRender compatibility, diagnostics, and deferred emphasis behavior.

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
