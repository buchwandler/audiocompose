from __future__ import annotations

import hashlib
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .errors import AudioValidationError

ClipPolicy = Literal["clamp", "warn", "error"]


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate: int
    channels: int
    frames: int
    sample_width: int


def prepare_output(audio: np.ndarray, *, clip_policy: ClipPolicy = "clamp") -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise AudioValidationError("audio must be a finite one-dimensional waveform")
    over_range = bool(np.any((samples < -1.0) | (samples > 1.0)))
    if over_range and clip_policy == "error":
        raise AudioValidationError("audio contains samples outside the PCM range [-1, 1]")
    if over_range and clip_policy == "warn":
        warnings.warn("audio was clipped to the PCM range [-1, 1]", RuntimeWarning, stacklevel=2)
    return np.clip(samples, -1.0, 1.0)


def _pcm_to_float(raw: bytes, width: int) -> np.ndarray:
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise AudioValidationError(f"unsupported PCM sample width: {width} bytes")


def read_wav(path: str | Path, *, expected_channels: int | None = 1) -> tuple[np.ndarray, int]:
    source = Path(path)
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise AudioValidationError(f"invalid WAV file {source}: {exc}") from exc
    if channels <= 0 or rate <= 0:
        raise AudioValidationError(f"invalid WAV metadata in {source}")
    if expected_channels is not None and channels != expected_channels:
        raise AudioValidationError(f"expected {expected_channels} channel(s), got {channels} in {source}")
    audio = _pcm_to_float(frames, width)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), rate


def wav_info(path: str | Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as handle:
            return WavInfo(handle.getframerate(), handle.getnchannels(), handle.getnframes(), handle.getsampwidth())
    except (OSError, wave.Error) as exc:
        raise AudioValidationError(f"invalid WAV file {path}: {exc}") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_pcm(path: str | Path, audio: np.ndarray, sample_rate: int, width: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = prepare_output(audio)
    if width == 2:
        pcm = np.round(clipped * 32767.0).astype("<i2")
    elif width == 4:
        pcm = np.round(clipped * 2147483647.0).astype("<i4")
    else:
        raise ValueError("sample width must be 2 or 4")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination


def write_intermediate_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write the bundle representation: mono PCM32 at the source rate."""
    return _write_pcm(path, audio, sample_rate, 4)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, *, clip_policy: ClipPolicy = "clamp") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = prepare_output(audio, clip_policy=clip_policy)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination
