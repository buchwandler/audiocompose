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

    @property
    def native_prosody_axes(self) -> frozenset[str]:
        return frozenset(
            axis
            for axis, supported in (
                ("rate", self.native_rate),
                ("pitch", self.native_pitch),
                ("volume", self.native_volume),
            )
            if supported
        )

    def supports_alignment(self, kind: str) -> bool:
        if kind == "word":
            return self.word_timing
        if kind == "phoneme":
            return self.phoneme_timing
        return False
