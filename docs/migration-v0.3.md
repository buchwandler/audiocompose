# Migrating to planned AudioCompose v0.3.0

> **Release status:** v0.3.0 remains planned and unreleased. This guide records the intended compatibility contract; it does not imply a release date.

AudioCompose v0.3.0 adds book-level composition for already-rendered variants and updates the optional VoiceRender bridge. The core `AudioJob`, `Composer`, and generic composition semantics remain producer-neutral.

## Dependency floors

- The base installation requires `renderplan>=0.1.2,<0.2`.
- The optional `audiocompose[voicerender]` extra requires `utterplan>=0.6.3,<0.7` and `voicerender>=0.2.0,<0.3`.
- VoiceRender 0.1.x is not supported by the v0.3 optional integration. Install the extra with `pip install 'audiocompose[voicerender]'` and use VoiceRender 0.2.x.

## VoiceRender book handoff

Book composition requires, for every selected chapter, a complete VoiceRender `RenderManifest` and the VoiceRender v0.2 `render-diagnostics.json` sidecar. AudioCompose verifies the manifest producer/version, plan identity, diagnostics schema and provenance, deferred-capability declarations, and their correspondence to RenderPlan control occurrences. Missing, stale, malformed, or contradictory diagnostics fail closed. The diagnostics digest is recorded in AudioJob provenance, the book report, and output-reuse identity; changing diagnostics invalidates prior reuse.

When the RenderPlan contains `emphasis` that AudioCompose should apply after synthesis, opt in at VoiceRender preflight and render:

```bash
# Repeat the same option on a separately run preflight command.
voicerender render-book BOOK \
  --variant narration--<fingerprint> \
  --emphasis-policy defer-to-compose

audiocompose compose-book BOOK \
  --variant narration--<fingerprint>
```

AudioCompose v0.3.0 recognizes only the deferred capability `audiocompose.emphasis.gain.v1`. VoiceRender owns controls it natively realizes; AudioCompose applies only the deferred capability declared by validated diagnostics. Unknown or inconsistent deferred capabilities are rejected. A complete manifest remains mandatory even when no control is deferred.

The VoiceRender producer option `--emphasis-policy defer-to-compose` is distinct from AudioCompose's downstream `--emphasis-policy segment_fallback|strict`. The latter controls how AudioCompose applies deferred emphasis without word-level alignment; `strict` rejects partial-range fallback, while `segment_fallback` applies the selected level to the complete speech segment. It does not grant ownership of any other producer control.

## Generic RenderPlan materialization is still strict

Direct generic `materialize_renderplan_job` and `compose-renderplan` do not infer VoiceRender ownership from manifest metadata and do not consume the VoiceRender diagnostics trust path. Unsupported producer-native controls remain errors. The generic adapter's own documented RenderPlan behavior is unchanged, and generic AudioJob composition does not interpret speech semantics.

## Pause-policy override keys

Book pause-policy overrides use the root `role` table, keyed by the resulting RenderPlan `SpeechItem.role`. They are not keyed directly by UtterPlan logical voice references. Source-pause validation checks known classes and table availability without treating logical voice references as role keys; actual durations resolve against the RenderPlan role selected for each pause and fail closed if that role/default lacks a value. Pre-release policy files using a `voice` root must be changed to `role` before using the v0.3 book integration.

## No AudioJob schema migration

The book bridge stores diagnostics provenance in the existing generic source and metadata mappings. AudioCompose v0.3.0 does not require an AudioJob schema-version change. Existing generic jobs can continue through `validate`, `inspect`, and `compose` unchanged.

See [the README](https://github.com/buchwandler/audiocompose#book-level-composition-for-rendered-variants) for the complete producer-to-composer example and [the architecture guide](architecture.md#voicerender-v02-control-ownership) for the ownership boundary.
