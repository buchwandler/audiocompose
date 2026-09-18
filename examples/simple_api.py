import numpy as np

from audiocompose import AudioBufferSource, AudioClip, AudioJob, Composer, Silence

job = AudioJob((
    AudioClip("speech-part", AudioBufferSource(np.zeros(24000, dtype=np.float32), 24000)),
    Silence("pause", 0.25),
))
Composer(sample_rate=24000).to_wav(job, "example.wav")
