from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderCapabilities:
    """Normalized capabilities advertised by a render plugin."""

    native_rate: bool = False
    native_pitch: bool = False
    native_volume: bool = False
    word_timing: bool = False
    phoneme_timing: bool = False
    multi_speaker: bool = False
    streaming: bool = False
