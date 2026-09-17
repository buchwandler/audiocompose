from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ShortSentencePhraseSet:
    language: str
    neutral: tuple[str, ...] = ()
    declarative: tuple[str, ...] = ()
    question: tuple[str, ...] = ()
    exclamation: tuple[str, ...] = ()
    ellipsis: tuple[str, ...] = ()
    fragment: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.language.strip():
            raise ValueError("phrase catalog language must not be empty")
        for values in self.categories():
            if any(not value.strip() for value in values):
                raise ValueError("phrase catalog templates must not be empty")
            if any(value.count("{segment}") != 1 for value in values):
                raise ValueError("phrase templates must contain exactly one '{segment}'")

    def categories(self) -> tuple[tuple[str, ...], ...]:
        return (self.neutral, self.declarative, self.question, self.exclamation, self.ellipsis, self.fragment)


ENGLISH_SHORT_SENTENCE_PHRASES = ShortSentencePhraseSet(
    "en",
    neutral=("The conversation stopped, {segment}, before someone answered.",),
    declarative=("The conversation stopped after one last reply: {segment}",),
    question=("The question was asked plainly: {segment}",),
    exclamation=("The speaker called out: {segment}",),
    ellipsis=("The thought trailed off with: {segment}",),
    fragment=("The short message read: {segment}",),
)


@dataclass(frozen=True, slots=True)
class WrapResolveMode:
    kind: Literal["wrap"] = "wrap"
    phoneme_pretext: str = "—"


@dataclass(frozen=True, slots=True)
class PhraseResolveMode:
    kind: Literal["phrase"] = "phrase"
    phrase_selection: Literal["auto", "neutral", "end"] = "auto"
    neutral_phrase: str = ENGLISH_SHORT_SENTENCE_PHRASES.neutral[0]
    end_phrase: str = ENGLISH_SHORT_SENTENCE_PHRASES.declarative[0]
    cutter: Literal["linear", "timestamp-adaptive"] = "linear"

    def __post_init__(self) -> None:
        if self.neutral_phrase.count("{segment}") != 1 or self.end_phrase.count("{segment}") != 1:
            raise ValueError("phrase templates must contain exactly one '{segment}'")


@dataclass(frozen=True, slots=True)
class RandomizedPhraseResolveMode:
    kind: Literal["randomized-phrase"] = "randomized-phrase"
    phrases: tuple[str, ...] = (ENGLISH_SHORT_SENTENCE_PHRASES.neutral[0],)
    end_phrases: tuple[str, ...] = (ENGLISH_SHORT_SENTENCE_PHRASES.declarative[0],)
    cutter: Literal["linear", "timestamp-adaptive"] = "linear"

    def __post_init__(self) -> None:
        if not self.phrases and not self.end_phrases:
            raise ValueError("randomized phrase mode needs at least one template")
        if any(value.count("{segment}") != 1 for value in (*self.phrases, *self.end_phrases)):
            raise ValueError("phrase templates must contain exactly one '{segment}'")


@dataclass(frozen=True, slots=True)
class ShortSentenceConfig:
    min_phoneme_length: int = 30
    phoneme_pretext: str = "—"
    enabled: bool = True
    resolve_mode: str | Literal[False] = "randomized-phrase"
    wrap: WrapResolveMode = field(default_factory=WrapResolveMode)
    phrase: PhraseResolveMode = field(default_factory=PhraseResolveMode)
    randomized_phrase: RandomizedPhraseResolveMode = field(default_factory=RandomizedPhraseResolveMode)
    phrase_fallback_tries: int = 1
    seed: int | None = None
    phrase_catalog: dict[str, ShortSentencePhraseSet] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.min_phoneme_length, bool) or self.min_phoneme_length < 0:
            raise ValueError("min_phoneme_length must be a non-negative integer")
        if not isinstance(self.phrase_fallback_tries, int) or self.phrase_fallback_tries < 0:
            raise ValueError("phrase_fallback_tries must be a non-negative integer")
        if self.resolve_mode is not False and self.resolve_mode not in {"wrap", "phrase", "randomized-phrase"}:
            raise ValueError(f"unsupported short sentence mode {self.resolve_mode!r}")
        if self.phrase_catalog is not None:
            for key, value in self.phrase_catalog.items():
                if key.lower().replace("_", "-") != value.language.lower().replace("_", "-"):
                    raise ValueError("phrase catalog key must match its language")

    def mode_for(self, phoneme_length: int) -> str | Literal[False]:
        if not self.enabled or phoneme_length >= self.min_phoneme_length:
            return False
        return self.resolve_mode


@dataclass(frozen=True, slots=True)
class ShortSentenceResult:
    text: str
    mode: str
    target_start: float = 0.0
    target_end: float = 1.0
    fallback: bool = False


def _terminal_category(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return "question"
    if stripped.endswith("!"):
        return "exclamation"
    if stripped.endswith(("…", "...")):
        return "ellipsis"
    if len(stripped.split()) <= 3:
        return "fragment"
    return "declarative"


def prepare_short_sentence(
    text: str,
    phoneme_length: int,
    config: ShortSentenceConfig,
    *,
    language: str = "en-us",
    seed: int | None = None,
) -> ShortSentenceResult:
    mode = config.mode_for(phoneme_length)
    if mode is False:
        return ShortSentenceResult(text, "none")
    if mode == "wrap":
        return ShortSentenceResult(text, "wrap")
    catalog = config.phrase_catalog or {"en": ENGLISH_SHORT_SENTENCE_PHRASES}
    lang = language.lower().replace("_", "-")
    phrase_set = catalog.get(lang, catalog.get(lang.split("-", 1)[0]))
    if phrase_set is None:
        return ShortSentenceResult(text, "wrap", fallback=True)
    category = _terminal_category(text)
    templates = getattr(phrase_set, category) or phrase_set.declarative or phrase_set.neutral
    if not templates:
        return ShortSentenceResult(text, "wrap", fallback=True)
    rng = random.Random(config.seed if seed is None else seed)
    template = rng.choice(tuple(templates))
    start = template.index("{segment}")
    end = start + len(text)
    return ShortSentenceResult(template.replace("{segment}", text), mode, start / len(template), end / len(template))


def cut_linear(audio, result: ShortSentenceResult):
    size = len(audio)
    start = max(0, min(size, round(size * result.target_start)))
    end = max(start, min(size, round(size * result.target_end)))
    if end <= start:
        raise ValueError("short sentence cutter produced an empty target")
    return audio[start:end]


__all__ = [
    "ENGLISH_SHORT_SENTENCE_PHRASES",
    "PhraseResolveMode",
    "RandomizedPhraseResolveMode",
    "ShortSentenceConfig",
    "ShortSentencePhraseSet",
    "ShortSentenceResult",
    "WrapResolveMode",
    "cut_linear",
    "prepare_short_sentence",
]
