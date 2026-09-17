# utterrender

Architectural MVP for a two-layer public speech synthesis stack:

```text
text / SSMD
    |
    v
 utterplan              semantic, deterministic, portable
    |
    v
 utterrender            runtime / voice routing / plugins / DSP / assembly
    |
    +--> Kokoro plugin
    +--> Piper plugin
    +--> third-party plugins
    |
    v
   WAV
```

The goal is that application developers use one runtime interface while `utterplan` stays
engine-independent. A plan stores language, logical voice references, pauses, prosody,
markers and prepared speech. `utterrender` owns installed engines, concrete voices, model
assets, routing, synthesis, runtime DSP and final audio assembly.

## Status

This release provides direct prepared-segment Piper and Kokoro backends. The backend packages own
G2P, model assets, ONNX sessions, voice data, native inference controls, and lifecycle.

## One interface

```python
from utterrender import TTS

with TTS(default_voice="kokoro:af_heart") as tts:
    tts.to_wav(
        "Hello world.",
        "hello.wav",
        language="en-us",
    )
```

For SSMD:

```python
from utterrender import TTS

text = '[This part is slow.]{rate="slow"}'
with TTS(default_voice="kokoro:af_heart") as tts:
    tts.to_wav(text, "slow.wav", language="en-us", input_format="ssmd")
```

## Logical voices and mixed engines

A plan should contain logical intent such as `narrator` or `quote`, not engine-specific
model paths. Bind those logical voices at runtime:

```python
from utterrender import Renderer

renderer = Renderer(
    bindings={
        "narrator": "kokoro:af_heart",
        "quote": "piper:en_US-lessac-medium",
    },
    sample_rate=24000,
)
result = renderer.render(plan)
```

Different segments may therefore be rendered by different plugins and normalized onto
one output timeline.

## Voice discovery

```python
with Renderer() as renderer:
    for voice in renderer.voices(language="de-de"):
        print(voice.id, voice.languages, voice.quality)
```

Voice IDs are namespaced (`kokoro:...`, `piper:...`) while `utterplan` voice references
remain logical and portable.

## Prosody ownership

`utterplan` stores semantic prosody intent. A plugin may realize an axis natively and marks
that on `AudioFragment.realized_prosody`. `utterrender` applies only unresolved axes through
`AudioSigProsodyProcessor`, preventing double application.

Example: Piper can realize `rate` with `length_scale`, while pitch/volume can remain
post-audio effects.

## Short sentences

Kokoro's short-sentence algorithm stays plugin-specific:

```text
utterrender request
  -> Kokoro plugin
     -> context phrase / retry / duration alignment / cut
     -> AudioFragment
  -> shared utterrender processing
```

The runtime can expose a common quality/policy option later, but the algorithm itself is
not generalized because it depends on Kokoro model behavior.

## Plugin contract

Plugins expose:

- voice discovery
- normalized capabilities
- asset/voice readiness
- segment rendering
- lifecycle cleanup

Third-party packages can register entry points under `utterrender.plugins`.

## Installation during migration

```bash
pip install "utterrender[kokoro,prosody]"
pip install "utterrender[piper,prosody]"
pip install "utterrender[all]"
```

The backend extras install direct G2P and ONNX Runtime dependencies. `utterplan` remains the
planning package, and `utterrender` remains the runtime and audio package.

## CLI

```bash
utterrender voices --language en-us
utterrender say "Hello" hello.wav --language en-us --voice kokoro:af_heart
utterrender render chapter.utterplan.json chapter.wav --voice piper:en_US-lessac-medium
```

## Voice asset provisioning and calibration

The runtime owns readiness and progress reporting. Discovery is metadata-only and does not open an
ONNX session. Configure local model and voice assets on the plugin, then call:

```python
renderer.ensure_voice("piper:en_US-lessac-medium")
```

Static reviewed per-voice level corrections can be centralized:

```python
renderer = Renderer(
    default_voice="kokoro:af_heart",
    calibration={
        "kokoro:af_heart": -1.2,
        "piper:en_US-lessac-medium": +2.4,
    },
)
```

The configured calibration gain is applied before final assembly. Optional loudness policy is applied
to the complete output, with true-peak and clipping policy handled by `utterrender`.
