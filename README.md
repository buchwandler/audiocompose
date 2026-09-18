# audiocompose

`audiocompose` declaratively processes and composes existing audio fragments into finalized audio. It is intentionally independent of text-to-speech engines, voices, models, G2P, and semantic speech plans.

## AudioJob boundary

A synthesis producer such as PyKokoro or PiperSynth creates an `AudioJob` containing concrete clips, numeric audio operations, explicit silence, anchors, and provenance. `Composer` loads those items, executes operations in order, resamples them to the output rate, assembles the timeline, applies complete-output loudness policy, and writes the final WAV.

```python
import numpy as np
from audiocompose import AudioBufferSource, AudioClip, AudioJob, Composer, Gain, Silence

job = AudioJob((
    AudioClip("intro", AudioBufferSource(np.zeros(24000, dtype=np.float32), 24000), (Gain(-3),)),
    Silence("pause", 0.5),
))
result = Composer(sample_rate=24000).compose(job)
Composer().to_wav(job, "final.wav")
```

## File bundles

```python
job.save("chapter.audiojob")
loaded = AudioJob.load("chapter.audiojob/audiojob.json")
Composer().compose_to_wav("chapter.audiojob/audiojob.json", "chapter.wav")
```

Bundles contain deterministic `audiojob.json` plus relative `parts/*.wav` files. Sources are mono PCM WAV files. Paths cannot escape the bundle, and optional SHA-256 and WAV metadata are checked at load time.

## CLI

```text
audiocompose validate chapter.audiojob/audiojob.json
audiocompose inspect chapter.audiojob/audiojob.json
audiocompose compose chapter.audiojob/audiojob.json chapter.wav
```

## Supported operations

Version 1 supports `Gain`, `PitchShift`, `Tempo`, `FadeIn`, and `FadeOut`. Operation values are numeric and are applied exactly in manifest order. Semantic values such as `slow`, `loud`, voices, and phonemes belong to the producer layer, not this package.

## Development

```text
python -m pytest -q
ruff check .
mypy audiocompose
```
