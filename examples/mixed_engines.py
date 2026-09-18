import numpy as np

from audiocompose import AudioBufferSource, AudioClip, AudioJob, Composer, Gain, Silence

# Producers may contribute clips from different engines. Composer only sees audio.
job = AudioJob(
    (
        AudioClip(
            "producer-a", AudioBufferSource(np.zeros(24000, dtype=np.float32), 24000), (Gain(-1),)
        ),
        Silence("pause", 0.1),
        AudioClip("producer-b", AudioBufferSource(np.zeros(22050, dtype=np.float32), 22050)),
    )
)
Composer(sample_rate=24000).to_wav(job, "mixed.wav")
