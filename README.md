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
loaded = AudioJob.load("chapter.audiojob/audiojob.json")
Composer().compose_to_wav("chapter.audiojob/audiojob.json", "chapter.wav")
```

Bundles contain deterministic `audiojob.json` plus writer-owned `parts/000001.wav` fragments. Every saved source is canonical mono PCM32 WAV at its native sample rate. Paths cannot escape the bundle, duplicate source basenames cannot collide, and SHA-256, WAV metadata, and the canonical manifest `job_id` are checked at load time.

## CLI

```text
audiocompose validate chapter.audiojob/audiojob.json
audiocompose inspect chapter.audiojob/audiojob.json
audiocompose compose chapter.audiojob/audiojob.json chapter.wav
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

AudioJob schema v1 is documented in `spec/audiojob-v1.schema.json`. Package versioning is SCM-derived and independent from the persisted AudioJob schema version. Stable upstream segment IDs should be used as generic `AudioClip.id` values where a one-to-one mapping exists; producer metadata remains opaque.

`AudioSpan` and `ComposedSpan` carry optional producer-defined IDs and JSON-safe metadata. AudioCompose preserves these opaque values while mapping sample coordinates through operations and resampling; it never interprets the metadata.

Composition completes in this order: load clips, apply operations, map anchors and spans, resample, concatenate clips and explicit silence, then measure and apply complete-output loudness once. Loudness changes waveform amplitude only, so item ranges, markers, and spans remain stable.

For producer frame parity, quantize a duration to producer frames before constructing a job: `frames = int(seconds * producer_rate)` and `seconds = frames / producer_rate`. AudioCompose keeps its public rounded seconds-to-samples rule and does not add engine-specific silence types.

`CompositionResult.loudness` exposes typed before/after metrics, requested and applied gain, target and ceiling, and warnings. `CompositionResult.diagnostics` contains generic codes and numeric context for loudness limitations and other composition warnings.
This checkout contains no PyKokoro or PiperSynth producer source. Producer adaptation and natural-pause calibration remain follow-up integration work. AudioCompose deliberately does not add engine-specific branches or interpret UtterPlan, ONNX, Kokoro, or Piper semantics.

## Supported operations

Version 1 supports `Gain`, `PitchShift`, `Tempo`, `FadeIn`, and `FadeOut`. Operation values are numeric and are applied exactly in manifest order. Semantic values such as `slow`, `loud`, voices, and phonemes belong to the producer layer, not this package.

## Development

```text
python -m build
audiocompose --version
python -m pytest -q
ruff check .
mypy audiocompose
```
