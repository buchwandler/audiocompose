from __future__ import annotations

import numpy as np
import pytest

from utterrender import AudioFragment, FragmentValidationError


def test_fragment_normalizes_to_float32_1d() -> None:
    fragment = AudioFragment("seg", np.array([0, 1], dtype=np.int16), 24000)
    assert fragment.audio.dtype == np.float32
    assert fragment.audio.ndim == 1


def test_fragment_rejects_nonfinite_audio() -> None:
    with pytest.raises(FragmentValidationError, match="non-finite"):
        AudioFragment("seg", np.array([np.nan], dtype=np.float32), 24000)
