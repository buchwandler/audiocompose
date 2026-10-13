# AudioCompose documentation

AudioCompose composes completed audio through producer-neutral `AudioJob` values. The repository README describes generic AudioJob workflows and direct RenderPlan composition.

## Book-level RenderPlan composition

The `audiocompose compose-book BOOK --variant VARIANT` command composes selected chapters in the order declared by `renderplan-set.toml`. It reuses AudioCompose's strict chapter materializer and `Composer`, writes one atomic WAV per chapter, and records batch provenance in `compose-report.toml`.

```bash
audiocompose compose-book . --variant narration--525856299523 --chapters 5-7
```

Production composition requires each selected VoiceRender `render-manifest.toml` to be complete; incomplete QC renders are rejected. The repository README's book-composition guide describes the output layout, options, reuse policy, and failure behavior.

```{toctree}
:maxdepth: 2
:caption: Documentation

architecture
migration-v0.2
changelog
```
