# Changelog

## 0.1.0

- Initial architectural MVP for `utterrender`.
- Defines `utterplan -> utterrender -> WAV` as the public two-layer workflow.
- Adds plugin protocol, registry, voice discovery, logical voice bindings, language-aware routing, and plugin capabilities.
- Adds built-in migration bridges for PiperSynth and PyKokoro.
- Adds high-level `TTS` facade for `text -> utterplan -> utterrender -> WAV`.
- Adds central sample-rate normalization, WAV writing, per-voice static gain calibration, and shared final assembly.
- Adds normalized prosody resolution and AudioSig post-processing while tracking axes already realized natively by a plugin.
- Keeps Kokoro short-sentence optimization inside the Kokoro plugin boundary.
- Supports either `UtterPlan`/`UtterPlanner` or legacy `TTSPlan`/`TTSPlanner` symbols exported by the renamed `utterplan` package.
