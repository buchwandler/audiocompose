[![PyPI - Version](https://img.shields.io/pypi/v/audiocompose)](https://pypi.org/project/audiocompose/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiocompose)
![PyPI - Downloads](https://img.shields.io/pypi/dm/audiocompose)
[![codecov](https://codecov.io/gh/buchwandler/audiocompose/graph/badge.svg?token=viiusddeDL)](https://codecov.io/gh/buchwandler/audiocompose)

# audiocompose

`audiocompose` declaratively processes and composes existing audio fragments into finalized audio. It is intentionally independent of text-to-speech engines, voices, models, G2P, and semantic speech plans.

## AudioJob boundary

A synthesis producer such as PyKokoro or PiperSynth creates an `AudioJob` containing concrete clips, numeric audio operations, explicit silence, anchors, and provenance. `Composer` loads those items, executes operations in order, resamples them to the output rate, assembles the timeline, applies complete-output loudness policy, and writes the final WAV.

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

AudioCompose delegates band-limited resampling, static WSOLA time/pitch processing, and `RatePitchEnvelope` DSP to AudioSig. The envelope API requires `audiosig>=0.1.5,<0.2`. Contiguous static `Tempo` and `PitchShift` operations remain fused unless a `Gain` or fade separates them. Each envelope is one complete AudioSig call. Numeric operation semantics and timeline mapping are the compatibility contract; exact PCM samples may change between AudioCompose or AudioSig versions.

## Composition progress

`Composer.compose()` accepts an optional synchronous `on_progress` callback. It receives typed `CompositionProgress` events for item loading, operations, resampling, assembly, complete-output loudness, and completion. Events carry generic item metadata and never print, alter the AudioJob, or affect composition identity.

```python
events = []
Composer().compose(job, on_progress=events.append)
```

The callback is also available through `to_wav()` and `compose_to_wav()`. Callback exceptions propagate to the caller so producer code can detect programming errors.

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
This checkout contains no PyKokoro or PiperSynth producer source. Producer adaptation and natural-pause calibration remain follow-up integration work. AudioCompose deliberately does not add engine-specific branches or interpret UtterPlan, ONNX, Kokoro, or Piper semantics.

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
