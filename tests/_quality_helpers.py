from __future__ import annotations

import numpy as np


def rms(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def tone_amplitude(signal: np.ndarray, sample_rate: int, frequency: float) -> float:
    values = np.asarray(signal, dtype=np.float64)
    time = np.arange(values.size, dtype=np.float64) / sample_rate
    scale = 2.0 / max(1, values.size)
    cosine = np.cos(2.0 * np.pi * frequency * time)
    sine = np.sin(2.0 * np.pi * frequency * time)
    return float(scale * np.hypot(np.dot(values, cosine), np.dot(values, sine)))


def dominant_frequency(signal: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(signal, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(values * np.hanning(values.size)))
    frequencies = np.fft.rfftfreq(values.size, 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def speech_like_signal(sample_rate: int, duration: float = 0.5) -> np.ndarray:
    sample_count = round(sample_rate * duration)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    fundamental = 125.0 + 18.0 * np.sin(2.0 * np.pi * 2.1 * time)
    phase = 2.0 * np.pi * np.cumsum(fundamental) / sample_rate
    signal = np.zeros(sample_count, dtype=np.float64)

    for harmonic in range(1, 17):
        frequency = harmonic * fundamental
        formants = (
            np.exp(-0.5 * ((frequency - 700.0) / 260.0) ** 2)
            + 0.8 * np.exp(-0.5 * ((frequency - 1_800.0) / 420.0) ** 2)
            + 0.35 * np.exp(-0.5 * ((frequency - 3_000.0) / 650.0) ** 2)
        )
        signal += formants / harmonic**0.8 * np.sin(harmonic * phase)

    envelope = 0.35 + 0.65 * np.sin(np.pi * np.clip(time / duration, 0.0, 1.0)) ** 2
    signal *= envelope
    signal += np.random.default_rng(412).normal(0.0, 0.002, sample_count)
    signal += 0.25 * np.exp(-0.5 * ((time - 0.16) / 0.0015) ** 2)
    signal += 0.12 * np.exp(-0.5 * ((time - 0.43) / 0.002) ** 2)
    signal[(time < 0.02) | (time >= duration - 0.02)] = 0.0
    return signal.astype(np.float32)
