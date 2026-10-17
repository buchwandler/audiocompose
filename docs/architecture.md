# audiocompose architecture

## Boundary

AudioCompose core is the generic audio composition layer. Its `Composer`, `AudioJob`, and schemas accept producer-neutral values. Optional adapters verify upstream handoffs and translate completed audio plus its provenance into ordinary jobs; synthesis, planning, and semantic interpretation remain outside the core.

The RenderPlan adapter is an explicit consumer boundary. It takes values from the shared `renderplan` contract and separately located media; it does not generate plans, run preflight, or invoke speech synthesis.

```text
any producer -> AudioJob --------------------------------\
                                                         -> Composer -> final.wav
RenderPlan + complete RenderManifest -> adapter -> AudioJob --/
                                  + speech WAVs + media
```

For mixed-engine use, any producer can emit the same generic job shape:

```text
engine A -> AudioJob --\
                       -> Composer -> CompositionResult
engine B -> AudioJob --/
```

The producer resolves semantic and engine behavior before composition. The AudioCompose core handles completed generic audio only and treats producer metadata and span identity as opaque.

PyKokoro and PiperSynth own G2P, engine configuration, voice and model selection, inference, engine-native controls, calibration, and timing extraction. They translate unresolved semantic intent into concrete audio and numeric audio-domain operations. Core AudioCompose modules do not import TTS engines, producer-specific planners, model runtimes, or engine semantics.

## Direct RenderPlan consumption

AudioCompose's `renderplan>=0.1.2,<0.2` runtime dependency supplies the shared contract types and TOML loaders. `audiocompose.integrations.renderplan` validates plan and manifest identity, requires `complete=true`, and checks exact manifest correspondence. Before constructing an `AudioJob`, it verifies each speech occurrence, segment ID, manifest WAV name, safe containment under the render directory, regular-file and symlink safety, SHA-256, and WAV sample-rate/channel/frame geometry. Speech artifacts must be mono.

The adapter iterates `RenderPlan.flow` literally. Pauses become `Silence` items at the declared `duration_ms / 1000`, including zero-duration pauses; AudioCompose does not infer pause meaning or apply a symbolic pause policy. Shared source spans are retained only as opaque provenance metadata, not fabricated `AudioSpan` alignment. The resulting job uses the same persistence, `OutputPolicy`, Composer, and output paths as any other AudioJob.

External `AudioItem` media is resolved separately from the speech manifest. The built-in local-WAV resolver requires an explicit `media_root` for relative paths and rejects traversal, symlinks, non-WAV files, and invalid or non-mono WAVs. `compose-renderplan` defaults the media root to the plan directory. The built-in resolver accepts local WAV paths; a Python `MediaResolver` can handle explicit URI schemes. The adapter fail-closes on unknown options and supports validated trim ranges, playback `speed` as `Tempo`, and `gain_db` as `Gain`.

Speech `emphasis` is interpreted only at this RenderPlan materialization boundary. AudioCompose validates the `level` parameter and turns the selected bounded gain into a real `Gain` operation on the speech clip; policy version, source controls/ranges, policy, gain, and application scope are provenance metadata. Without trusted character-to-sample alignment, a full-span control is exact at clip scope; partial controls either fail in `strict` mode or boost the complete segment under the integrated `segment_fallback` default, with a visible warning and original character ranges. Overlaps use the strongest level, not cumulative gain. The generic AudioJob model remains speech-control agnostic, and downstream output-wide loudness/peak processing may reduce the relative boost.

The direct CLI composes without first persisting an AudioJob:

```text
audiocompose compose-renderplan PLAN --manifest MANIFEST \
  --render-dir DIR --output WAV [--media-root DIR] [--explain]
```

The manifest defaults to `<render-dir>/render-manifest.toml`. The command accepts the existing output-policy controls, writes the final WAV, and reports a RenderPlan-derived timeline; `--explain` prints ordered provenance. Use `--save-audiojob` for an optional intermediate bundle. The generic `validate`, `inspect`, `compose`, `timeline`, and save/load workflows are unchanged.

## Native book composition

`audiocompose.book.compose_book` is the specialist orchestration facade for a completed `.ssmdbook` RenderPlan-set variant. It resolves the set-owned RenderProfile snapshot and chapter order, verifies each selected UtterPlan/RenderPlan/VoiceRender handoff through the public integration boundary, materializes the chapter as an ordinary AudioJob, and calls the same generic `Composer`. It does not change AudioJob, Composer, or DSP semantics.

The dependency boundary remains asymmetric: `renderplan` is the shared base contract dependency; UtterPlan and VoiceRender are optional and are imported only when book materialization runs. Generic package-root imports and generic AudioJob composition do not require the `[voicerender]` extra.

```text
.ssmdbook + RenderPlan-set variant
          |                         RenderProfile snapshot
          |                                  |
          +---- UtterPlan + RenderManifest --+--> book pause policy
                              |
                   optional book adapter
                              |
                reproducible AudioJob bundle
                              |
                    generic Composer
                              |
             atomic chapter WAV + TOML report
```

### VoiceRender v0.2 control ownership

The book bridge is a producer-specific trust boundary; the generic RenderPlan materializer stays strict. A complete VoiceRender v0.2 manifest and its `render-diagnostics.json` sidecar are required. AudioCompose verifies producer/version, RenderPlan identity, diagnostic contents, and control occurrence/count correspondence before using the evidence. The diagnostics digest and the accepted ownership decision participate in AudioJob identity and book-output reuse.

```text
RenderPlan preserves intent.
VoiceRender realizes producer-native controls.
VoiceRender diagnostics declare remaining composition capabilities.
AudioCompose realizes only declared capabilities it explicitly owns.
Generic RenderPlan composition does not infer producer ownership.
```

For v0.3, the only accepted deferred capability is `audiocompose.emphasis.gain.v1`; unknown capabilities fail closed. Do not use this diagnostics trust path to relax direct generic `materialize_renderplan_job` validation. Pause-policy `role` overrides are keyed by RenderPlan `SpeechItem.role`. Source-pause validation checks known classes and table availability without treating logical voice references as policy keys; actual durations resolve against RenderPlan role and fail closed if no role/default value exists.

For book composition, symbolic source pause intents resolve from the selected RenderProfile snapshot by default. A strict `audiocompose.pause-policy.v1` TOML override is accepted explicitly; unresolved intents fail closed. The policy identity and provenance are stored in AudioJob sources and the report. This policy conversion belongs to the book integration and is not applied by direct `compose-renderplan`, whose RenderPlan pauses remain literal durations.

AudioJobs are persisted for every selected chapter before composition. Existing WAVs are reused only when the current variant, source/plan, RenderManifest identity and content hash, pause/output policy, persisted AudioJob ID/schema, AudioCompose version, and WAV hash/geometry all match the atomic report. Symlinked or unverified outputs are not reused; `--force` recomposes. Chapter failures are isolated by default, `--fail-fast` skips later selections, and ordered book/chapter progress callbacks report callback failures as `progress.callback_failed`. `compose-report.toml` is atomically published with per-chapter outcomes and report-relative stage paths.

The CLI surface is `audiocompose compose-book BOOK --variant VARIANT`, with source-number selection, `--pauses`, output-policy controls, `--force`, `--fail-fast`, `--save-audiojobs`, and machine-clean `--json` output. The CLI calls `compose_book`; it does not implement another composition path.

## AudioJob

An AudioJob is an ordered collection of `AudioClip` and `Silence` items. A clip references either an in-memory NumPy `AudioBufferSource` or a file-backed `AudioFileSource`, followed by ordered numeric operations allowed by its schema version. V1 contains the existing static operations; v2 adds the generic time-varying `RatePitchEnvelope`. Clip anchors and spans are generic timing metadata. Output policy contains the final sample rate, channel count, clipping behavior, and complete-output loudness policy.

The canonical packaged schemas are `audiocompose/schemas/audiojob-v1.schema.json` and `audiocompose/schemas/audiojob-v2.schema.json`; repository-level `spec/` copies are verified mirrors. `audiojob_schema(version)` loads a packaged schema. New jobs default to v2. Loading and saving a v1 job preserves v1, and v1 rejects the v2-only envelope operation. The bundle identity is `sha256:` plus the SHA-256 of canonical UTF-8 JSON with sorted keys and compact separators, excluding `job_id` itself.

## Generic analysis boundary

`audiocompose.analysis` reports acoustic activity and gaps using sample coordinates. The analyzer is intentionally not a voice activity detector and does not interpret semantic pauses or producer timing. Producer code may supply a generic anchor, then request gap measurement near that coordinate.
New serialized bundles use schema version 2 and contain `audiojob.json` plus relative PCM32 WAV fragments. Bundle directories are the save target; writes stage deterministic parts, replace the writer-owned `parts/` directory, then atomically replace `audiojob.json` last. Unrelated bundle files are preserved. Loaded v1 bundles retain schema version 1 when saved. `AudioJob.load()` validates source integrity by default; explicit `verify_sources=False` defers file reads, while composition and inspection verify each source as it is loaded and check source-coordinate geometry against that same waveform.

## Composition

`Composer` loads each source, applies its operations in manifest order, resamples it to the requested output rate, places it after preceding items, and resolves anchors in final waveform coordinates. Silence is an explicit timeline item. `CompositionResult` contains float32 waveform data, sample rate, item spans, composed marker offsets, diagnostics, and provenance.

The complete-output ordering is normative: load clips, apply clip operations, map anchors and spans through those operations, resample clips, concatenate clips and explicit silence, then measure and apply loudness once. Loudness is amplitude-only and never changes item ranges, marker offsets, or composed span coordinates.

## Composition estimates and progress

`estimate_composition(job, sample_rate=None)` is a deterministic preflight over the generic AudioJob shape. It predicts per-item/output frame geometry and approximate frame-pass work, including source loading/integrity verification, operation groups, resampling, assembly, optional complete-output loudness, and final output preparation. Missing source geometry remains unknown: output totals and total work are `None`, and events have no numeric fraction. Work units are neither wall-clock estimates nor persisted values; their model may evolve.

`Composer` exposes optional synchronous `CompositionProgress` events as a runtime observation boundary. Events identify generic items, operations, source and target sample rates, frame counts, assembly, loudness, completion, and (when known) monotonic work/fraction. Item metadata is forwarded opaquely so a producer can map events to its own identifiers without adding producer-specific concepts to AudioCompose. `ProgressEstimator` is an opt-in caller-side view using monotonic time and observed work throughput; it can report elapsed time and bounded ETA after it has enough evidence, or accept a caller-supplied initial work rate. It has no hardware-specific constants and retains no measurements after the process.

Progress delivery and timing are not part of AudioJob serialization, job identity, `CompositionResult`, provenance, reports, or audio semantics. With no callback, the composition path and DSP results remain unchanged. Core callbacks do not print; their exceptions propagate to the caller. Terminal stream handling belongs to the consuming UI layer.

### CLI progress boundary

The three rendering commands (`compose`, `compose-renderplan`, and `compose-book`) share the private standard-library CLI renderer and accept `--progress auto|always|never`. `auto` is the default and renders only to a TTY stderr when `--json` is absent; `always` forces stderr updates, including with JSON, and `never` disables them. TTY refreshes are throttled and non-TTY forced output is newline-delimited. JSON stdout remains machine-clean, and no terminal dependency is introduced into Composer or the API layers. Unknown work totals show elapsed time and named lifecycle phases without invented percentages or ETAs; command completion is shown only after successful output handling.

The package-root `compose_renderplan` API forwards its optional `on_progress` callback to Composer. The book facade forwards those core events inside `BookComposeProgress.composition`; its `fraction` is a uniform chapter-slot measure, not a duration-weighted forecast. Book callback failures retain the existing `progress.callback_failed` behavior. Progress, fractions, and timing remain runtime-only across both integrations.

`AudioSpan` and `ComposedSpan` support optional IDs and JSON-safe metadata. The fields are producer-neutral and are preserved through AudioJob serialization and composition. `CompositionResult.loudness` provides typed before and after metrics, requested and applied gain, policy, target, ceiling, and warnings. `CompositionResult.diagnostics` contains generic diagnostic codes and numeric context.
Timing transformations use generic operation hooks. Static `Tempo` maps offsets proportionally; `RatePitchEnvelope` maps source coordinates by inverting the integrated output-time rate curve, with the final rate held after its last point. Pitch-only envelopes preserve duration and coordinates. Operation mappings happen before the single source-to-output resampling map. Final loudness and true-peak/clipping policy are applied once to the complete waveform.

## DSP behavior and reproducibility

AudioSig owns band-limited resampling, time-scale modification, static pitch-shift DSP, and the public rate/pitch-envelope DSP API (`audiosig>=0.1.6,<0.2`). AudioCompose owns the numeric operation model, deterministic timeline mapping, and orchestration. Contiguous `Tempo` and `PitchShift` operations are fused only when no other operation intervenes; a `RatePitchEnvelope` is one complete operation and a grouping boundary. Marker and span mapping is based on declared operation semantics, not the DSP implementation.

AudioCompose represents numeric control trajectories only. The producer decides whether to create an envelope, including eligibility decisions based on voice, segment, language, or engine. AudioCompose does not inspect those semantics or metadata.

For a fused group, Composer emits each logical `operation_started` event in declaration order before the combined AudioSig call, then emits the matching `operation_completed` events in declaration order. Event `details` identifies the group start and size; frame counts describe the fused group's input and output boundaries.

Exact PCM can change between AudioCompose or AudioSig versions. Numeric operation semantics and timeline mapping are the compatibility contract.

Silence keeps the public seconds-to-samples rule based on rounded `seconds * output_rate`. A producer that requires exact frame parity should quantize first with `frames = int(seconds * producer_rate)` and pass `frames / producer_rate` as the job duration. No engine-specific silence type is part of AudioCompose.

## Ownership matrix

| Concern                                                             | Owner                             |
| ------------------------------------------------------------------- | --------------------------------- |
| Language analysis, segmentation, semantic intent, speech planning   | Upstream planner / producer       |
| Voice/model choice, inference, calibration, native controls         | Speech producer / engine          |
| RenderPlan and RenderManifest contracts and TOML loading            | RenderPlan package                |
| Speech artifact production                                          | Upstream renderer                 |
| Manifest correspondence and speech artifact integrity               | AudioCompose RenderPlan adapter   |
| RenderPlan flow ordering and provenance lowering to AudioJob values | AudioCompose RenderPlan adapter   |
| Media-root policy, media resolver contract, and option lowering     | AudioCompose integration boundary |
| Native book orchestration and profile-derived pause policy          | AudioCompose optional book facade |
| AudioJob format, schema, persistence, and bundle validation         | AudioCompose core                 |
| Numeric operation model and timing maps                             | AudioCompose core                 |
| Resampling, time-scale modification, and pitch-shift DSP            | AudioSig                          |
| AudioJob order, silence placement, markers, and timeline assembly   | AudioCompose core                 |
| Complete-output loudness, clipping policy, and final WAV            | AudioCompose core                 |

## Non-TTS guarantee

The base AudioCompose install requires NumPy, AudioSig, and the shared RenderPlan contract package. Importing `audiocompose`, constructing a generic `AudioJob`, and using `Composer` do not import or require TTS engines, producer-specific planners, model runtimes, or inference frameworks. The RenderPlan adapter is a consumer-only boundary; it verifies completed artifacts and resolves audio into generic job values. AudioSig supplies band-limited resampling and WSOLA time/pitch DSP, while AudioCompose core owns the generic operation model and composition orchestration. Canonical tests use deterministic synthetic audio and require no TTS models.
