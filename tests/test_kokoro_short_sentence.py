from __future__ import annotations

import numpy as np
import pytest

from utterrender.plugins.kokoro_short_sentence import (
    PhraseResolveMode,
    ShortSentenceConfig,
    ShortSentencePhraseSet,
    ShortSentenceResult,
    WrapResolveMode,
    cut_linear,
    prepare_short_sentence,
)


def test_short_sentence_threshold_and_disabled_modes() -> None:
    config = ShortSentenceConfig(min_phoneme_length=3, resolve_mode="wrap")

    assert config.mode_for(3) is False
    assert config.mode_for(2) == "wrap"
    assert ShortSentenceConfig(enabled=False).mode_for(1) is False


def test_short_sentence_phrase_selection_is_seeded_and_localized() -> None:
    catalog = {
        "de": ShortSentencePhraseSet("de", fragment=("Notiz: {segment}",)),
    }
    config = ShortSentenceConfig(
        resolve_mode="phrase",
        phrase=PhraseResolveMode(neutral_phrase="unused: {segment}"),
        phrase_catalog=catalog,
        seed=7,
    )

    result = prepare_short_sentence("Hallo.", 2, config, language="de-de")

    assert result.mode == "phrase"
    assert result.text == "Notiz: Hallo."
    assert result.target_start < result.target_end
    assert prepare_short_sentence("Hallo.", 2, config, language="de-de").text == result.text


def test_short_sentence_cutter_rejects_empty_geometry() -> None:
    with pytest.raises(ValueError, match="empty target"):
        cut_linear(np.ones(4), ShortSentenceResult("x", "phrase", 0.5, 0.5))


def test_invalid_phrase_template_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PhraseResolveMode(neutral_phrase="missing segment")


def test_wrap_mode_is_model_free() -> None:
    config = ShortSentenceConfig(resolve_mode="wrap", wrap=WrapResolveMode(phoneme_pretext="…"))
    result = prepare_short_sentence("Hi!", 1, config)

    assert result.mode == "wrap"
    assert result.text == "Hi!"
