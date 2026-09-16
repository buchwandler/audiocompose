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

## Migration rule

The 0.1 MVP uses bridges to existing PyKokoro/PiperSynth implementations. Moving their
code physically into `utterrender.plugins.kokoro` and `utterrender.plugins.piper` must not
change the `RenderPlugin`, `VoiceInfo`, `RenderRequest`, `AudioFragment`, `Renderer` or
`TTS` public contracts.

Kokoro short-sentence optimization remains a Kokoro plugin implementation detail. The
runtime may expose common policy knobs such as `quality="balanced"`, but must not pretend
that model-specific algorithms are portable between plugins.

## Plan rename compatibility

The runtime imports the `utterplan` package through a small internal compatibility layer.
It prefers `UtterPlan`/`UtterPlanner` when present and accepts legacy
`TTSPlan`/`TTSPlanner` names if the package rename retained them.
