from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import get_args

import pytest
from jsonschema import Draft202012Validator

import audiocompose
from audiocompose import AudioValidationError, Operation, audiojob_schema


def test_root_api_is_bounded_and_exposes_the_closed_operation_union() -> None:
    expected = {
        "__version__",
        "AudioAnchor",
        "AudioBufferSource",
        "AudioClip",
        "AudioComposeError",
        "AudioFileSource",
        "AudioJob",
        "AudioSpan",
        "AudioValidationError",
        "AutomationPoint",
        "CompositionDiagnostic",
        "CompositionError",
        "CompositionProgress",
        "CompositionProgressCallback",
        "CompositionResult",
        "ComposedItem",
        "ComposedMarker",
        "ComposedSpan",
        "Composer",
        "DiagnosticSeverity",
        "FadeIn",
        "FadeOut",
        "Gain",
        "LoudnessPolicy",
        "LoudnessResult",
        "Operation",
        "OutputPolicy",
        "PitchShift",
        "RatePitchEnvelope",
        "Silence",
        "Tempo",
        "audiojob_schema",
        "load_job",
        "save_job",
    }
    assert set(audiocompose.__all__) == expected
    assert get_args(Operation) == (
        audiocompose.Gain,
        audiocompose.PitchShift,
        audiocompose.Tempo,
        audiocompose.FadeIn,
        audiocompose.FadeOut,
        audiocompose.RatePitchEnvelope,
    )


@pytest.mark.parametrize("version", [1, 2])
def test_schema_resources_are_packaged_and_match_repository_mirrors(version: int) -> None:
    filename = f"audiojob-v{version}.schema.json"
    resource = files("audiocompose").joinpath("schemas", filename)
    packaged_text = resource.read_text(encoding="utf-8")
    repository_text = Path("spec", filename).read_text(encoding="utf-8")
    schema = audiojob_schema(version)

    assert packaged_text == repository_text
    Draft202012Validator.check_schema(schema)


def test_typing_marker_is_a_package_resource() -> None:
    assert files("audiocompose").joinpath("py.typed").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize("version", [True, 0, 3, "2"])
def test_schema_loader_rejects_unknown_versions(version: object) -> None:
    with pytest.raises(AudioValidationError, match="schema version"):
        audiojob_schema(version)  # type: ignore[arg-type]
