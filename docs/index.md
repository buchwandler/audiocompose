# AudioCompose reference

AudioCompose composes completed audio through producer-neutral `AudioJob` values. The repository README describes generic AudioJob workflows and direct RenderPlan composition.

## Single-document RenderPlan composition

`audiocompose.compose_renderplan(...)` is the public package-root API for composing one completed RenderPlan and RenderManifest into a WAV. It validates the handoff and speech artifacts, materializes the generic AudioJob, composes the document, and returns a typed `RenderPlanComposeResult` with output metadata, timeline, and emphasis diagnostics. Other Python orchestration clients do not need to import `audiocompose.integrations`.

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

## Bundle composition and variant discovery

`audiocompose bundle variants NAME.ssmd-bundle` is a read-only discovery command for both supported SSMD bundle kinds. It lists direct RenderPlan candidates and their plan, VoiceRender, and AudioCompose states, including readiness and stale composition provenance. Invalid candidates are reported without hiding valid siblings; missing stage roots are treated as empty. JSON output is available with `--json`, and `--renderplan-root`, `--voicerender-root`, and `--output-root` select custom stage roots. Discovery does not render or compose, modify the workspace, or require a TTS engine.

```bash
audiocompose bundle variants project.ssmd-bundle
audiocompose bundle variants project.ssmd-bundle --json

# The canonical bundle root may also be addressed from inside the workspace.
cd project.ssmd-bundle
audiocompose bundle variants .
audiocompose bundle compose . --variant narration--<fingerprint>
```

The overall states include `not_rendered`, `render_partial`, `ready`, `complete`, and `stale`, plus error states `invalid`, `render_failed`, `compose_partial`, `compose_failed`, and `unknown`. The JSON `ready_to_compose` field also accounts for required VoiceRender diagnostics.

`audiocompose bundle compose NAME.ssmd-bundle --variant VARIANT` composes either supported SSMD bundle kind. AudioCompose reads `manifest.json` and dispatches on its `format`: a book bundle composes selected chapters, while a document bundle composes its single document (and accepts only `--chapters all`). Explicit paths and `.` from inside the canonical `.ssmd-bundle` are accepted; inner source paths and ZIP files are not. Existing book/document report contracts remain kind-specific. See the unified [bundle composition guide](https://github.com/buchwandler/audiocompose#bundle-composition-for-rendered-variants).

```bash
audiocompose bundle compose project.ssmd-bundle --variant narration--<fingerprint> --json
```

Production composition requires complete VoiceRender handoffs. Default RenderPlan, VoiceRender, and AudioCompose roots are resolved beneath the bundle root. The package-root `audiocompose.list_bundle_variants(bundle)` API returns typed `BundleVariantsResult` and `BundleVariant` values.

## Architecture

```{toctree}
:maxdepth: 2
:caption: Documentation

architecture
migration-v0.2
changelog
```
