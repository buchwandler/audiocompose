# Architecture MVP

## Public boundary

`utterplan` compiles semantic speech intent. `utterrender` executes an utterance plan.

`utterplan` remains independent of ONNX, concrete voices, model caches and waveforms.
`utterrender` owns runtime concerns: plugin discovery, concrete voice selection, assets,
synthesis, shared DSP, sample-rate normalization, timeline assembly and WAV output.

## Runtime sequence

1. Accept a validated utterance plan from `utterplan`.
2. Resolve each logical voice reference with `VoiceBindings`.
3. Select the plugin from the concrete namespaced voice (`kokoro:*`, `piper:*`).
4. Resolve lexical prosody into backend-independent numeric semantics.
5. Let the plugin realize engine-native controls and return an `AudioFragment`.
6. Apply per-voice calibration.
7. Normalize fragment sample rates.
8. Apply unresolved prosody through AudioSig.
9. Insert plan pauses, build segment/unit spans and resolve safe marker offsets.
10. Return `RenderResult` or write WAV.

## Direct backend ownership

The built-in plugins are direct implementations. Kokoro owns Kokoro G2P, voice styles, short-sentence
workarounds, model timing extraction, and ONNX sessions. Piper owns Piper G2P, config and speaker
resolution, native length scale, and ONNX sessions. Neither plugin invokes a planner or parser.

Kokoro short-sentence optimization remains a Kokoro plugin implementation detail. It is not moved into
`utterplan` or generalized into the shared renderer.

## Canonical plan API

The runtime imports the current public `utterplan` API directly: `UtterancePlan`, `UtterancePlanner`,
`PlannerConfig`, `PlanSegment` and `ProsodyDirective`. Legacy `UtterPlan` and `UtterPlanner` names are
not required.
