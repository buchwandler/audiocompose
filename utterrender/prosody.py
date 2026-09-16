from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from ._plan import ProsodyDirective

ProsodyAxis = Literal["rate", "pitch", "volume"]

RATE_ABSOLUTE_MAP: dict[str, float] = {
    "x-slow": 0.5,
    "slow": 0.75,
    "medium": 1.0,
    "fast": 1.25,
    "x-fast": 1.5,
    "default": 1.0,
}

PITCH_ABSOLUTE_MAP: dict[str, float] = {
    "x-low": -4.0,
    "low": -2.0,
    "medium": 0.0,
    "high": 2.0,
    "x-high": 4.0,
    "default": 0.0,
}

VOLUME_ABSOLUTE_MAP: dict[str, float] = {
    "silent": -math.inf,
    "x-soft": -12.0,
    "soft": -6.0,
    "medium": 0.0,
    "loud": 6.0,
    "x-loud": 12.0,
    "default": 0.0,
}


@dataclass(frozen=True, slots=True)
class ResolvedProsody:
    """Numeric prosody semantics independent of any synthesis backend.

    ``requested`` records which axes were explicitly present in the plan. Neutral
    values such as ``rate="medium"`` are therefore still distinguishable from an
    absent directive.
    """

    rate: float = 1.0
    semitones: float = 0.0
    gain_db: float = 0.0
    requested: frozenset[ProsodyAxis] = frozenset()

    @property
    def is_neutral(self) -> bool:
        return (
            math.isclose(self.rate, 1.0)
            and math.isclose(self.semitones, 0.0)
            and math.isclose(self.gain_db, 0.0)
        )


def parse_rate(value: str) -> float:
    """Resolve an SSMD-style rate value to a speed multiplier."""

    text = value.strip().lower()
    if text in RATE_ABSOLUTE_MAP:
        return RATE_ABSOLUTE_MAP[text]

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", text)
    if match:
        number = float(match.group(1))
        rate = number / 100.0 if not text.startswith(("+", "-")) else 1.0 + number / 100.0
        if rate <= 0.0:
            raise ValueError(f"rate must resolve to > 0, got {value!r}")
        return rate

    # utterplan/pipersynth already accept plain numeric values, so keep that
    # compatibility in the shared resolver.
    try:
        rate = float(text)
    except ValueError as exc:
        raise ValueError(f"invalid prosody rate: {value!r}") from exc
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"rate must resolve to a finite value > 0, got {value!r}")
    return rate


def parse_pitch(value: str) -> float:
    """Resolve an SSMD-style pitch value to semitones."""

    text = value.strip().lower()
    if text in PITCH_ABSOLUTE_MAP:
        return PITCH_ABSOLUTE_MAP[text]

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*st", text)
    if match:
        return float(match.group(1))

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", text)
    if match:
        number = float(match.group(1))
        if not text.startswith(("+", "-")):
            number -= 100.0
        ratio = 1.0 + number / 100.0
        if ratio <= 0.0:
            raise ValueError(f"pitch percentage must resolve to > 0, got {value!r}")
        return 12.0 * math.log2(ratio)

    raise ValueError(f"invalid prosody pitch: {value!r}")


def parse_volume(value: str) -> float:
    """Resolve an SSMD-style volume value to a gain in dB."""

    text = value.strip().lower()
    if text in VOLUME_ABSOLUTE_MAP:
        return VOLUME_ABSOLUTE_MAP[text]

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*db", text)
    if match:
        return float(match.group(1))

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)%", text)
    if match:
        number = float(match.group(1))
        if not text.startswith(("+", "-")):
            number -= 100.0
        multiplier = 1.0 + number / 100.0
        if multiplier <= 0.0:
            return -math.inf
        return 20.0 * math.log10(multiplier)

    # A plain numeric volume is intentionally not interpreted as dB. PiperSynth
    # currently treats plain numeric volume as a linear multiplier, while SSMD
    # volume semantics are dB/labels/percent. Ambiguous input should remain a
    # backend decision rather than silently changing meaning here.
    raise ValueError(f"invalid prosody volume: {value!r}")


def resolve_prosody(value: ProsodyDirective | None) -> ResolvedProsody:
    """Convert current utterplan lexical prosody fields into numeric semantics.

    This is a compatibility adapter for utterplan 0.1, whose ``ProsodyDirective``
    stores strings. Once utterplan emits normalized numeric semantics directly,
    callers can construct :class:`ResolvedProsody` without this adapter.
    """

    if value is None:
        return ResolvedProsody()

    requested: set[ProsodyAxis] = set()
    rate = 1.0
    semitones = 0.0
    gain_db = 0.0

    if value.rate is not None:
        requested.add("rate")
        rate = parse_rate(value.rate)
    if value.pitch is not None:
        requested.add("pitch")
        semitones = parse_pitch(value.pitch)
    if value.volume is not None:
        requested.add("volume")
        gain_db = parse_volume(value.volume)

    return ResolvedProsody(
        rate=rate,
        semitones=semitones,
        gain_db=gain_db,
        requested=frozenset(requested),
    )
