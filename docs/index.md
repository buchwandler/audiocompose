# AudioCompose documentation

AudioCompose composes completed audio through producer-neutral `AudioJob` values. The repository README describes generic AudioJob workflows and direct RenderPlan composition.

## Single-document RenderPlan composition

`audiocompose.compose_renderplan(...)` is the public package-root API for composing one completed RenderPlan and RenderManifest into a WAV. It validates the handoff and speech artifacts, materializes the generic AudioJob, composes the document, and returns a typed `RenderPlanComposeResult` with output metadata, timeline, and emphasis diagnostics. Callers such as Readio do not need to import `audiocompose.integrations`.

```python
import audiocompose

result = audiocompose.compose_renderplan(
    "chapter.renderplan.toml",
    render_dir="render/chapter",
    output="chapter.wav",
    output_policy=audiocompose.OutputPolicy(sample_rate=24000),
)
```

The manifest defaults to `render-manifest.toml` inside `render_dir`; when `media_root` is omitted, relative external media resolves beneath the plan file's parent. `OutputPolicy` carries the existing sample-rate, clipping, and loudness/peak controls, with mono-only output. Emphasis defaults to `segment_fallback`; `strict` rejects partial-span emphasis. Set `save_audiojob` only when the intermediate job should be persisted. See the [direct RenderPlan composition guide](https://github.com/buchwandler/audiocompose#direct-renderplan-composition) for the CLI workflow and full behavior.

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
