# Migrating to AudioCompose v0.2

AudioCompose v0.2 tightens public Python and persisted-job contracts. The Python package version is independent from the AudioJob wire version: new jobs still use schema v2, loaded v1 jobs remain v1 when saved, and there is no schema v3.

## Public imports

The package root is limited to common composition models, built-in operations, results, diagnostics, progress, errors, persistence helpers, and `audiojob_schema`. Low-level and specialized helpers remain available from their owning modules:

| Previous root import                                                                                        | Import from               |
| ----------------------------------------------------------------------------------------------------------- | ------------------------- |
| `AudioOperation`, `apply_operation`, `operation_from_dict`                                                  | `audiocompose.operations` |
| `AudioSource`                                                                                               | `audiocompose.sources`    |
| `AcousticGap`, `ActivityConfig`, `ActivityRegion`, `ActivityReport`, `analyze_activity`, `measure_gap_near` | `audiocompose.analysis`   |
| `LoudnessMetrics`, `apply_complete_output_loudness`                                                         | `audiocompose.loudness`   |
| `resample_audio`                                                                                            | `audiocompose.resampling` |
| `samples_for_duration`, `silence`                                                                           | `audiocompose.timeline`   |
| `prepare_output`, `read_wav`, `sha256_file`, `wav_info`, `write_intermediate_wav`, `write_wav`              | `audiocompose.wav`        |
| `validate_job`                                                                                              | `audiocompose.job`        |
| `Marker`                                                                                                    | `audiocompose.alignment`  |

For example:

```python
from audiocompose import AudioJob, Composer, Operation
from audiocompose.resampling import resample_audio
from audiocompose.wav import read_wav, write_wav
```

Use `audiojob_schema(1)` or `audiojob_schema(2)` from `audiocompose` to load the packaged JSON Schema resources.

## Strict AudioJob loading

Persisted manifests and `AudioJob.from_dict()` now follow the declared schema exactly. Unknown fields, numeric strings, booleans in numeric fields, malformed operation objects, missing `job_id` values, non-canonical identities, and identity mismatches are rejected instead of being coerced, ignored, or repaired. Validate producer-created dictionaries before saving them, and include the canonical `job_id` when constructing persisted data.

Schema v1 and v2 are packaged under `audiocompose/schemas/`. The repository `spec/` files are mirrors. Package version 0.2 does not change either wire format.

## Bundle saving and source verification

`AudioJob.save(path)` treats `path` as a bundle directory and returns the path to its `audiojob.json` manifest. It writes the referenced PCM32 WAV files under `parts/`. Repeated saves replace the writer-owned `parts/` set deterministically, preserve unrelated bundle files, and replace the manifest last. Pass the bundle directory to `AudioJob.load()` or composition helpers. Source hashes and WAV metadata are checked by default; `verify_sources=False` defers source reads until inspection or composition loads them.

## Final output clipping

`CompositionResult.audio` is finalized after operations, resampling, concatenation, complete-output loudness, and the output clip policy:

- `clamp` returns samples limited to `[-1, 1]`.
- `error` raises `CompositionError` during composition when samples exceed that range.
- `warn` preserves out-of-range float samples and adds a clipping diagnostic. WAV serialization safely clips those samples to the PCM range.

Consumers should not expect `compose()` to return an unclipped intermediate when the selected policy is `clamp`.

## Model inputs and metadata

Numeric model and operation fields reject booleans and invalid or non-finite values. Built-in operation discriminators are fixed by their classes; they are not caller-supplied constructor values. Models snapshot caller-owned audio arrays and JSON metadata at construction. Producer and source metadata remain opaque and cannot select silence, DSP, timing, or progress behavior.

## Loudness and progress

Loudness measurement now follows AudioSig. Audio shorter than one complete 400 ms analysis block has no reportable integrated LUFS value, represented as `None`; LUFS normalization is not applied to such output. Peak metrics remain available where measurable.

`CompositionProgress.total_audio_seconds` and `completed_audio_seconds` describe predicted output-duration progress, including explicit silence and operation timing changes, rather than raw source-processing duration.

## WAV inputs

WAV reading supports signed 16-bit and 32-bit PCM. The default reader and AudioFileSource require mono input; multichannel support is not part of this release contract. AudioJob bundle fragments are canonical mono PCM32 WAV files at their source sample rate.
