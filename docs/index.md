# AudioCompose documentation

AudioCompose composes completed audio through producer-neutral `AudioJob` values. The repository README describes generic AudioJob workflows and direct RenderPlan composition.

## Book-level RenderPlan composition

The `audiocompose compose-book BOOK --variant VARIANT` command uses the RenderPlan-set's authoritative chapter order, resolves source pause intents from its RenderProfile snapshot (or an explicit `--pauses POLICY.toml`), saves inspectable AudioJobs, composes through generic `Composer`, atomically writes chapter WAVs, and records exact reuse provenance in `compose-report.toml`. UtterPlan and VoiceRender are optional; install `audiocompose[voicerender]` for this integration.

```bash
audiocompose compose-book . --variant narration--525856299523 --chapters 5-7
```

Production composition requires each selected VoiceRender `render-manifest.toml` to be complete; incomplete QC renders are rejected. The [repository README](https://github.com/buchwandler/audiocompose#book-level-composition-for-rendered-variants) documents output roots, pause policy, CLI controls, reuse, and failure behavior.

```{toctree}
:maxdepth: 2
:caption: Documentation

architecture
migration-v0.2
migration-v0.3
changelog
```
