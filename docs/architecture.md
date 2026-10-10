# audiocompose architecture

## Boundary

AudioCompose core is the generic audio composition layer. Its `Composer`, `AudioJob`, and schemas accept producer-neutral values; synthesis or upstream artifact interpretation ends at the job boundary. Optional adapters may materialize generic jobs from validated producer artifacts without adding producer-specific semantics to core composition.

```text
any producer -> AudioJob ----------------\
                                       -> Composer -> final.wav
UtterPlan + VoiceRender artifacts -> optional adapter -> AudioJob --/
```

For mixed-engine use, any producer can emit the same generic job shape:

```text
engine A -> AudioJob --\
                       -> Composer -> CompositionResult
engine B -> AudioJob --/
```

The producer resolves semantic and engine behavior before composition. The AudioCompose core handles completed generic audio only and treats producer metadata and span identity as opaque.

PyKokoro and PiperSynth own G2P, engine configuration, voice and model selection, inference, engine-native controls, calibration, and timing extraction. They translate unresolved semantic intent into concrete audio and numeric audio-domain operations. Core AudioCompose modules do not import TTS engines, UtterPlan readers, model runtimes, or engine semantics.

## Optional VoiceRender materialization

`audiocompose.integrations.voicerender` is an explicit adapter boundary enabled by the `audiocompose[voicerender]` extra. It uses VoiceRender's public handoff APIs to verify UtterPlan v5, render-manifest binding, selection, and speech artifacts, then produces an ordinary v2 `AudioJob`. The base package, `Composer`, and persisted schema do not depend on VoiceRender.

The standalone `audiocompose materialize PLAN --renders MANIFEST --pauses POLICY --output JOB` command resolves output preferences into the existing `OutputPolicy`, persists the job, and leaves validation, inspection, and composition to the normal CLI commands. `PausePolicy` maps symbolic intents to durations at source boundaries. Relative local WAV media is resolved safely beneath the plan directory; opaque URIs need a custom `MediaResolver`. Numeric speed and explicit dB gain are supported, while clip-region and repeat controls fail closed because AudioJob v2 has no generic source-region model.

## AudioJob

An AudioJob is an ordered collection of `AudioClip` and `Silence` items. A clip references either an in-memory NumPy `AudioBufferSource` or a file-backed `AudioFileSource`, followed by ordered numeric operations allowed by its schema version. V1 contains the existing static operations; v2 adds the generic time-varying `RatePitchEnvelope`. Clip anchors and spans are generic timing metadata. Output policy contains the final sample rate, channel count, clipping behavior, and complete-output loudness policy.

The canonical packaged schemas are `audiocompose/schemas/audiojob-v1.schema.json` and `audiocompose/schemas/audiojob-v2.schema.json`; repository-level `spec/` copies are verified mirrors. `audiojob_schema(version)` loads a packaged schema. New jobs default to v2. Loading and saving a v1 job preserves v1, and v1 rejects the v2-only envelope operation. The bundle identity is `sha256:` plus the SHA-256 of canonical UTF-8 JSON with sorted keys and compact separators, excluding `job_id` itself.

## Generic analysis boundary

`audiocompose.analysis` reports acoustic activity and gaps using sample coordinates. The analyzer is intentionally not a voice activity detector and does not interpret semantic pauses or producer timing. Producer code may supply a generic anchor, then request gap measurement near that coordinate.
New serialized bundles use schema version 2 and contain `audiojob.json` plus relative PCM32 WAV fragments. Bundle directories are the save target; writes stage deterministic parts, replace the writer-owned `parts/` directory, then atomically replace `audiojob.json` last. Unrelated bundle files are preserved. Loaded v1 bundles retain schema version 1 when saved. `AudioJob.load()` validates source integrity by default; explicit `verify_sources=False` defers file reads, while composition and inspection verify each source as it is loaded and check source-coordinate geometry against that same waveform.

## Composition

`Composer` loads each source, applies its operations in manifest order, resamples it to the requested output rate, places it after preceding items, and resolves anchors in final waveform coordinates. Silence is an explicit timeline item. `CompositionResult` contains float32 waveform data, sample rate, item spans, composed marker offsets, diagnostics, and provenance.

The complete-output ordering is normative: load clips, apply clip operations, map anchors and spans through those operations, resample clips, concatenate clips and explicit silence, then measure and apply loudness once. Loudness is amplitude-only and never changes item ranges, marker offsets, or composed span coordinates.

## Composition progress

`Composer` exposes optional synchronous `CompositionProgress` events as a runtime observation boundary. The events identify generic items, operations, source and target sample rates, frame counts, assembly, complete-output loudness, and completion. Item metadata is forwarded opaquely so a producer can map events to its own identifiers without adding producer-specific concepts to audiocompose.

Progress delivery is not part of AudioJob serialization, job identity, result provenance, or audio semantics. With no callback, the composition path remains unchanged. Callback exceptions propagate to the caller, while terminal stream failures are a responsibility of the consuming UI layer.
`AudioSpan` and `ComposedSpan` support optional IDs and JSON-safe metadata. The fields are producer-neutral and are preserved through AudioJob serialization and composition. `CompositionResult.loudness` provides typed before and after metrics, requested and applied gain, policy, target, ceiling, and warnings. `CompositionResult.diagnostics` contains generic diagnostic codes and numeric context.
Timing transformations use generic operation hooks. Static `Tempo` maps offsets proportionally; `RatePitchEnvelope` maps source coordinates by inverting the integrated output-time rate curve, with the final rate held after its last point. Pitch-only envelopes preserve duration and coordinates. Operation mappings happen before the single source-to-output resampling map. Final loudness and true-peak/clipping policy are applied once to the complete waveform.

## DSP behavior and reproducibility

AudioSig owns band-limited resampling, time-scale modification, static pitch-shift DSP, and the public rate/pitch-envelope DSP API (`audiosig>=0.1.6,<0.2`). AudioCompose owns the numeric operation model, deterministic timeline mapping, and orchestration. Contiguous `Tempo` and `PitchShift` operations are fused only when no other operation intervenes; a `RatePitchEnvelope` is one complete operation and a grouping boundary. Marker and span mapping is based on declared operation semantics, not the DSP implementation.

AudioCompose represents numeric control trajectories only. The producer decides whether to create an envelope, including eligibility decisions based on voice, segment, language, or engine. AudioCompose does not inspect those semantics or metadata.

For a fused group, Composer emits each logical `operation_started` event in declaration order before the combined AudioSig call, then emits the matching `operation_completed` events in declaration order. Event `details` identifies the group start and size; frame counts describe the fused group's input and output boundaries.

Exact PCM can change between AudioCompose or AudioSig versions. Numeric operation semantics and timeline mapping are the compatibility contract.

Silence keeps the public seconds-to-samples rule based on rounded `seconds * output_rate`. A producer that requires exact frame parity should quantize first with `frames = int(seconds * producer_rate)` and pass `frames / producer_rate` as the job duration. No engine-specific silence type is part of AudioCompose.

## Ownership matrix

| Concern                                                                        | Owner                                      |
| ------------------------------------------------------------------------------ | ------------------------------------------ |
| Concern                                                                        | Owner                                      |
| ------------------------------------------------------------------------------ | ------------------------------------------ |
| SSMD, language analysis, segmentation, semantic pauses, prosody intent         | UtterPlan                                  |
| G2P, voice/model choice, model assets, inference, native controls, calibration | PyKokoro / PiperSynth / engine package     |
| UtterPlan v5 parsing and render-manifest handoff verification                  | VoiceRender public handoff API             |
| VoiceRender artifact to generic AudioJob materialization                       | optional AudioCompose VoiceRender adapter  |
| Symbolic pause-to-duration composition policy                                  | AudioCompose integration                   |
| AudioJob format and bundle validation                                          | audiocompose core                          |
| WAV loading, operation model/orchestration, timeline assembly                  | audiocompose core                          |
| Numeric rate/pitch envelopes and timing maps                                   | audiocompose core                          |
| Transition eligibility and voice/segment semantics                             | producer                                   |
| Band-limited resampling, time-scale modification, pitch-shift DSP              | AudioSig                                   |
| Silence insertion and marker finalization                                      | audiocompose core                          |
| Complete-output LUFS, true peak, clipping, and final WAV                       | audiocompose core                          |

## Non-TTS guarantee

The base AudioCompose install requires NumPy and AudioSig. Importing `audiocompose`, constructing `AudioJob`, and using `Composer` do not import or require VoiceRender, Readio, PyKokoro, PiperSynth, Kokoro or Piper G2P packages, or ONNX Runtime. Only the explicitly invoked VoiceRender adapter depends on the optional `audiocompose[voicerender]` extra. AudioSig supplies band-limited resampling and WSOLA time/pitch DSP; AudioCompose core owns the generic operation model and composition orchestration. Canonical tests use deterministic synthetic audio and require no TTS models.
