# audiocompose architecture

## Boundary

`audiocompose` is the generic audio composition layer. It accepts a producer-neutral `AudioJob`, not an UtterPlan or speech document. Synthesis ends before composition begins.

```text
UtterPlan -> PyKokoro / PiperSynth -> AudioJob -> Composer -> final.wav
```

For mixed-engine use, any producer can emit the same generic job shape:

```text
engine A -> AudioJob --\
                       -> Composer -> CompositionResult
engine B -> AudioJob --/
```

The producer resolves semantic and engine behavior before composition. AudioCompose handles completed audio only and treats producer metadata and span identity as opaque.

PyKokoro and PiperSynth own G2P, engine configuration, voice and model selection, inference, engine-native controls, calibration, and timing extraction. They translate unresolved semantic intent into concrete audio and numeric audio-domain operations. Audiocompose has no knowledge of TTS, Kokoro, Piper, voices, phonemes, ONNX, models, or UtterPlan semantics.

## AudioJob

An AudioJob is an ordered collection of `AudioClip` and `Silence` items. A clip references either an in-memory NumPy `AudioBufferSource` or a file-backed `AudioFileSource`, followed by ordered v1 operations. Clip anchors and spans are generic timing metadata. Output policy contains the final sample rate, channel count, clipping behavior, and complete-output loudness policy.

The normative schema is `spec/audiojob-v1.schema.json`. The bundle identity is `sha256:` plus the SHA-256 of canonical UTF-8 JSON with sorted keys and compact separators, excluding `job_id` itself.

## Generic analysis boundary

`audiocompose.analysis` reports acoustic activity and gaps using sample coordinates. The analyzer is intentionally not a voice activity detector and does not interpret semantic pauses or producer timing. Producer code may supply a generic anchor, then request gap measurement near that coordinate.
Serialized bundles use schema version 1 and contain `audiojob.json` and relative PCM32 WAV fragments. Bundle loading validates paths, hashes, WAV metadata, supported operations, and output policy before composition.

## Composition

`Composer` loads each source, applies its operations in manifest order, resamples it to the requested output rate, places it after preceding items, and resolves anchors in final waveform coordinates. Silence is an explicit timeline item. `CompositionResult` contains float32 waveform data, sample rate, item spans, composed marker offsets, diagnostics, and provenance.


The complete-output ordering is normative: load clips, apply clip operations, map anchors and spans through those operations, resample clips, concatenate clips and explicit silence, then measure and apply loudness once. Loudness is amplitude-only and never changes item ranges, marker offsets, or composed span coordinates.

`AudioSpan` and `ComposedSpan` support optional IDs and JSON-safe metadata. The fields are producer-neutral and are preserved through AudioJob serialization and composition. `CompositionResult.loudness` provides typed before and after metrics, requested and applied gain, policy, target, ceiling, and warnings. `CompositionResult.diagnostics` contains generic diagnostic codes and numeric context.
Timing transformations are explicit. Tempo operations map offsets proportionally. Resampling changes sample coordinates but not time. Pitch processing preserves duration in the v1 implementation. Final loudness and true-peak/clipping policy are applied once to the complete waveform.

Silence keeps the public seconds-to-samples rule based on rounded `seconds * output_rate`. A producer that requires exact frame parity should quantize first with `frames = int(seconds * producer_rate)` and pass `frames / producer_rate` as the job duration. No engine-specific silence type is part of AudioCompose.

## Ownership matrix

| Concern                                                                        | Owner                                  |
| ------------------------------------------------------------------------------ | -------------------------------------- |
| SSMD, language analysis, segmentation, semantic pauses, prosody intent         | UtterPlan                              |
| G2P, voice/model choice, model assets, inference, native controls, calibration | PyKokoro / PiperSynth / engine package |
| AudioJob format and bundle validation                                          | audiocompose                           |
| WAV loading, audio operations, resampling, timeline assembly                   | audiocompose                           |
| Silence insertion and marker finalization                                      | audiocompose                           |
| Complete-output LUFS, true peak, clipping, and final WAV                       | audiocompose                           |

## Non-TTS guarantee

The package's required dependency is NumPy. Tempo and PitchShift use deterministic NumPy phase-vocoder processing in the current implementation; the optional `dsp` extra remains available for future quality backends. The core package does not import `utterplan`, PyKokoro, PiperSynth, Kokoro or Piper G2P packages, or ONNX Runtime. The canonical tests use deterministic synthetic audio and require no TTS models.
