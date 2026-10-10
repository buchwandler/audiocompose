from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

voicerender = pytest.importorskip("voicerender")
from voicerender.engines.api_probe import EngineApiProbe  # noqa: E402
from voicerender.engines.base import (  # noqa: E402
    EngineCapabilities,
    EngineSelection,
    RenderedSpeech,
    RequestMeasure,
    SpeechWordTiming,
)
from voicerender.engines.registry import _ADAPTERS  # noqa: E402
from voicerender.inputs.utterplan_v5 import parse_utterplan_v5  # noqa: E402
from voicerender.manifest import MANIFEST_FILENAME  # noqa: E402
from voicerender.models import VoiceBinding, VoiceProfile  # noqa: E402
from voicerender.render_v5 import render_utterplan_v5  # noqa: E402

from audiocompose import AudioClip, Silence  # noqa: E402
from audiocompose.errors import HandoffValidationError, PausePolicyError  # noqa: E402
from audiocompose.integrations.voicerender import (  # noqa: E402
    PAUSE_POLICY_SCHEMA,
    PausePolicy,
    materialize_voicerender_job,
)


def _hash(value) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _plan(
    texts, *, pause_before=None, pause_after=None, markers=None, directives=None, file_hash="a" * 64
):
    pause_before = pause_before or {}
    pause_after = pause_after or {}
    markers = markers or {}
    directives = directives or {}
    segments = []
    semantic_segments = []
    for index, text in enumerate(texts):
        segment = {"text": text, "language": "en-US"}
        projection = {
            "text": text,
            "language": "en-US",
            "directives": {},
            "tokens": [],
            "markers": list(markers.get(index, ())),
        }
        if index in pause_before:
            segment["pause_before"] = pause_before[index]
            projection["pause_before"] = pause_before[index]
        if index in pause_after:
            segment["pause_after"] = pause_after[index]
            projection["pause_after"] = pause_after[index]
        if index in markers:
            segment["marker"] = list(markers[index])
        if index in directives:
            segment.update(directives[index])
            normalized = {}
            for name, value in directives[index].items():
                if name == "prosody":
                    normalized[name] = {key: value.get(key) for key in ("rate", "pitch", "volume")}
                elif name == "audio":
                    normalized[name] = {
                        key: value.get(key)
                        for key in (
                            "src",
                            "alt_text",
                            "clip_begin",
                            "clip_end",
                            "speed",
                            "repeat_duration",
                            "repeat_count",
                            "sound_level",
                        )
                    }
                    if value.get("description") is not None:
                        normalized[name]["description"] = value["description"]
                else:
                    normalized[name] = value
            projection["directives"] = normalized
        segments.append(segment)
        semantic_segments.append(projection)
    flow = [{"hash": _hash({"segments": semantic_segments}), "segment": segments}]
    data = {
        "format": "utterplan",
        "schema_version": 5,
        "language": "en-US",
        "unit": "paragraph",
        "hash_schema": "utterplan-flow-v1",
        "flow": flow,
    }
    data["plan_id"] = _hash(
        {
            "format": "utterplan",
            "schema_version": 5,
            "language": "en-US",
            "unit": "paragraph",
            "hash_schema": "utterplan-flow-v1",
            "document_semantics": {},
            "flow": [{"segments": semantic_segments}],
        }
    )
    return parse_utterplan_v5(data, file_sha256=file_hash)


class _Session:
    def __init__(self, calls):
        self.calls = calls

    def measure(self, request):
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=len(request.text),
            unit="unknown",
            source="test-fake",
        )

    def synthesize(self, request):
        self.calls.append(request)
        return RenderedSpeech(
            id=request.id,
            audio=np.full(128, 0.1, dtype=np.float32),
            sample_rate=16_000,
            word_timings=(
                SpeechWordTiming(
                    text=request.text,
                    char_start=0,
                    char_end=len(request.text),
                    start_sample=10,
                    end_sample=40,
                ),
            ),
        )


class _Adapter:
    id = "audiocompose-test"
    package_name = "audiocompose-test"

    def __init__(self, composition_semantics=()):
        self.calls = []
        self.composition_semantics = frozenset(composition_semantics)

    def capabilities(self):
        return EngineCapabilities(
            id=self.id,
            supports_live=True,
            supports_timestamps=True,
            composition_semantics=self.composition_semantics,
        )

    def version(self):
        return "1.0"

    def probe_api(self):
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            distribution_version="1.0",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def resolve(self, request):
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or "fixture-model",
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(self, selection):
        assert selection.engine == self.id
        return ()

    def canonical_synthesis_identity(self, selection):
        return {
            "engine": self.id,
            "target_id": selection.target_id,
            "voice": selection.voice,
            "speaker": selection.speaker,
            "options": dict(selection.options),
        }

    @contextmanager
    def open(self, selection):
        del selection
        yield _Session(self.calls)


@pytest.fixture
def rendered_bundle(tmp_path, monkeypatch):
    def render(plan, composition_semantics=(), select=None):
        adapter = _Adapter(composition_semantics)
        monkeypatch.setitem(_ADAPTERS, adapter.id, adapter)
        profile = VoiceProfile(
            name="narrator",
            roles={
                "narrator": VoiceBinding(engine=adapter.id, target_id="fixture-model"),
            },
            defaults={"role": "narrator"},
        )
        bundle = tmp_path / "renders"
        render_utterplan_v5(
            plan,
            profile=profile,
            output_dir=bundle,
            cache_dir=tmp_path / "cache",
            select=select,
        )
        return bundle / MANIFEST_FILENAME

    return render


def _policy(**default):
    return {
        "schema": PAUSE_POLICY_SCHEMA,
        "merge": "max",
        "default": default,
    }


def test_pause_policy_parsing_exact_timed_and_voice_override():
    policy = PausePolicy.from_mapping(
        {
            "schema": PAUSE_POLICY_SCHEMA,
            "merge": "max",
            "default": {"sentence": "250ms"},
            "voice": {"narrator": {"sentence": "320ms"}},
        }
    )
    assert policy.resolve("sentence", voice="narrator", where="test") == pytest.approx(0.32)
    assert policy.resolve(
        {"type": "timed", "time": "125ms"}, voice="narrator", where="test"
    ) == pytest.approx(0.125)
    with pytest.raises(PausePolicyError, match="unknown pause class"):
        PausePolicy.from_mapping(_policy(unknown="10ms"))


def test_materialize_verified_speech_pause_spans_markers_and_identity(tmp_path, rendered_bundle):
    plan = _plan(
        ["Hello", "world"],
        pause_after={0: "sentence"},
        pause_before={1: "sentence"},
        markers={0: ("opening",)},
    )
    manifest = rendered_bundle(plan)
    job = materialize_voicerender_job(plan, manifest, _policy(sentence="300ms"))

    assert [type(item) for item in job.items] == [AudioClip, Silence, AudioClip]
    pause = job.items[1]
    assert isinstance(pause, Silence)
    assert pause.seconds == pytest.approx(0.3)
    assert len(pause.metadata["audiocompose.pause_events"]) == 1
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.id == "speech:f000000.s000"
    assert clip.anchors[0].name == "opening"
    assert clip.spans[0].source_start == 0
    assert clip.spans[0].source_end == len("Hello")
    assert clip.spans[0].sample_start == 10
    assert clip.metadata["voicerender.render_manifest_id"]
    job.validate(verify_sources=True)

    first = Path(job.save(tmp_path / "first.audiojob"))
    restored = type(job).load(str(first))
    second = Path(restored.save(tmp_path / "second.audiojob"))
    assert type(job).load(str(second)).job_id == restored.job_id


def test_materialize_rejects_plan_mismatch_and_corrupt_speech(rendered_bundle):
    plan = _plan(["Integrity."])
    manifest = rendered_bundle(plan)
    other_plan = _plan(["Integrity."], file_hash="b" * 64)
    with pytest.raises(HandoffValidationError, match="source|plan"):
        materialize_voicerender_job(other_plan, manifest, _policy())

    document = voicerender.verify_render_manifest(manifest)
    audio_path = manifest.parent / document["segments"][0]["audio"]["path"]
    audio_path.write_bytes(b"corrupt WAV")
    with pytest.raises(HandoffValidationError, match="manifest|WAV|audio"):
        materialize_voicerender_job(plan, manifest, _policy())


def test_numeric_composition_directives_lower_to_audiocompose_operations(rendered_bundle):
    from audiocompose import Tempo

    plan = _plan(["Slowly."], directives={0: {"prosody": {"rate": "slow"}}})
    manifest = rendered_bundle(plan)
    job = materialize_voicerender_job(plan, manifest, _policy())
    assert isinstance(job.items[0], AudioClip)
    assert len(job.items[0].operations) == 1
    assert isinstance(job.items[0].operations[0], Tempo)
    assert job.items[0].operations[0].factor == pytest.approx(0.8)


def _with_plan_path(plan, tmp_path):
    from dataclasses import replace

    path = tmp_path / "fixture.utterplan.toml"
    path.write_text("in-memory UtterPlan fixture\n", encoding="utf-8")
    return replace(plan, path=path)


def test_mixed_speech_local_wav_speech_preserves_source_order_and_numeric_controls(
    tmp_path, rendered_bundle
):
    from audiocompose import Gain, Tempo
    from audiocompose.sources import AudioFileSource
    from audiocompose.wav import write_wav

    plan = _plan(
        ["Before.", "", "After."],
        pause_after={0: "sentence"},
        pause_before={2: "sentence"},
        directives={
            1: {
                "audio": {
                    "src": "music.wav",
                    "alt_text": "music",
                    "speed": 1.25,
                    "sound_level": "-3dB",
                }
            }
        },
    )
    plan = _with_plan_path(plan, tmp_path)
    write_wav(tmp_path / "music.wav", np.full(160, 0.1, dtype=np.float32), 16_000)
    manifest = rendered_bundle(plan)

    job = materialize_voicerender_job(
        plan,
        manifest,
        _policy(sentence="100ms"),
    )

    assert [item.id for item in job.items] == [
        "speech:f000000.s000",
        "pause:f000000.s000->f000000.s001",
        "media:f000000.s001",
        "pause:f000000.s001->f000000.s002",
        "speech:f000000.s002",
    ]
    media = job.items[2]
    assert isinstance(media, AudioClip)
    assert isinstance(media.source, AudioFileSource)
    assert media.metadata["utterplan.audio.alt_text"] == "music"
    assert isinstance(media.operations[0], Tempo)
    assert media.operations[0].factor == pytest.approx(1.25)
    assert isinstance(media.operations[1], Gain)
    assert media.operations[1].db == pytest.approx(-3.0)
    job.validate(verify_sources=True)


def test_all_media_uses_custom_resolver_and_rejects_unknown_scheme_by_default(
    tmp_path, rendered_bundle
):
    from audiocompose import AudioBufferSource
    from audiocompose.errors import MediaResolutionError
    from audiocompose.integrations.media import ResolvedMedia

    plan = _with_plan_path(
        _plan([""], directives={0: {"audio": {"src": "sfx:impact.knock?seed=42"}}}),
        tmp_path,
    )
    manifest = rendered_bundle(plan)
    with pytest.raises(MediaResolutionError, match="no resolver for URI scheme"):
        materialize_voicerender_job(plan, manifest, _policy())

    class Resolver:
        request = None

        def resolve(self, request):
            self.request = request
            return ResolvedMedia(
                AudioBufferSource(np.full(32, 0.2, dtype=np.float32), 16_000),
                metadata={"provider": "fixture"},
                identity={"asset": "impact.knock"},
            )

    resolver = Resolver()
    job = materialize_voicerender_job(plan, manifest, _policy(), media_resolver=resolver)
    assert len(job.items) == 1
    media = job.items[0]
    assert isinstance(media, AudioClip)
    assert media.id == "media:f000000.s000"
    assert resolver.request.src == "sfx:impact.knock?seed=42"
    assert media.metadata["audiocompose.media.identity"] == {"asset": "impact.knock"}
    job.validate(verify_sources=True)


def test_unsupported_external_audio_controls_fail_closed(tmp_path, rendered_bundle):
    from audiocompose.errors import MediaResolutionError
    from audiocompose.wav import write_wav

    plan = _with_plan_path(
        _plan([""], directives={0: {"audio": {"src": "music.wav", "clip_begin": "1s"}}}),
        tmp_path,
    )
    write_wav(tmp_path / "music.wav", np.zeros(16, dtype=np.float32), 16_000)
    manifest = rendered_bundle(plan)

    with pytest.raises(MediaResolutionError, match="clip_begin/clip_end are unsupported"):
        materialize_voicerender_job(plan, manifest, _policy())


def test_noncontiguous_selected_source_flow_is_rejected(rendered_bundle):
    plan = _plan(["First.", "Skipped.", "Third."])
    manifest = rendered_bundle(plan, select=["f000000.s000", "f000000.s002"])

    with pytest.raises(HandoffValidationError, match="contiguous source selection"):
        materialize_voicerender_job(plan, manifest, _policy())


def test_explicit_selection_must_exactly_match_manifest_selection(rendered_bundle):
    plan = _plan(["First.", "Second."])
    manifest = rendered_bundle(plan)

    with pytest.raises(HandoffValidationError, match="Explicit selection differs"):
        materialize_voicerender_job(plan, manifest, _policy(), selection=1)


def _write_cli_source_plan(path: Path, text: str):
    plan = _plan([text])
    segment = plan.segments[0]
    path.write_text(
        "\n".join(
            (
                'format = "utterplan"',
                "schema_version = 5",
                f"plan_id = {json.dumps(plan.plan_id)}",
                'language = "en-US"',
                'unit = "paragraph"',
                'hash_schema = "utterplan-flow-v1"',
                "",
                "[[flow]]",
                f"hash = {json.dumps(segment.source.flow_hash)}",
                "",
                "[[flow.segment]]",
                f"text = {json.dumps(text, ensure_ascii=False)}",
                'language = "en-US"',
                "",
            )
        ),
        encoding="utf-8",
    )
    loaded = voicerender.load_source_plan(path)
    assert loaded.plan_id == plan.plan_id
    return loaded


def test_cli_manual_materialize_validate_inspect_compose_without_readio(
    tmp_path, rendered_bundle, capsys
):
    from audiocompose.cli import main
    from audiocompose.wav import read_wav

    plan_path = tmp_path / "chapter.utterplan.toml"
    plan = _write_cli_source_plan(plan_path, "CLI workflow.")
    manifest = rendered_bundle(plan)
    pause_path = tmp_path / "pause.toml"
    pause_path.write_text(
        'schema = "audiocompose.pause-policy.v1"\nmerge = "max"\n[default]\nsentence = "250ms"\n',
        encoding="utf-8",
    )
    bundle = tmp_path / "chapter.audiojob"

    assert (
        main(
            [
                "materialize",
                str(plan_path),
                "--renders",
                str(manifest),
                "--pauses",
                str(pause_path),
                "--output",
                str(bundle),
                "--sample-rate",
                "16000",
                "--clip-policy",
                "warn",
                "--target-lufs",
                "-16",
                "--true-peak-ceiling-dbtp",
                "-1.5",
                "--peak-policy",
                "error",
                "--json",
            ]
        )
        == 0
    )
    materialized = json.loads(capsys.readouterr().out)
    assert materialized["materialized"] is True
    assert materialized["job_id"].startswith("sha256:")
    manifest_path = bundle / "audiojob.json"

    assert main(["validate", str(manifest_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert main(["inspect", str(manifest_path), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["expected_output_rate"] == 16_000
    assert inspected["loudness"] == {
        "target_lufs": -16.0,
        "true_peak_ceiling_dbtp": -1.5,
    }

    output_wav = tmp_path / "chapter.wav"
    assert main(["compose", str(manifest_path), str(output_wav)]) == 0
    capsys.readouterr()
    audio, sample_rate = read_wav(output_wav)
    assert sample_rate == 16_000
    assert audio.size > 0


def test_optional_extra_metadata_and_base_import_boundary(tmp_path):
    import subprocess
    import sys

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    project = Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies = tomllib.loads(project)["project"]["optional-dependencies"]
    assert dependencies["voicerender"] == ["voicerender>=0.1,<0.2"]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import audiocompose; "
            "assert not any(name == 'voicerender' or name.startswith('voicerender.') "
            "for name in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_local_media_resolver_rejects_plan_directory_traversal(tmp_path, rendered_bundle):
    from audiocompose.errors import MediaResolutionError

    plan = _with_plan_path(
        _plan([""], directives={0: {"audio": {"src": "../escape.wav"}}}),
        tmp_path,
    )
    manifest = rendered_bundle(plan)

    with pytest.raises(MediaResolutionError, match="remain beneath the UtterPlan directory"):
        materialize_voicerender_job(plan, manifest, _policy())


def test_malformed_local_wav_is_reported_as_media_resolution_error(tmp_path, rendered_bundle):
    from audiocompose.errors import MediaResolutionError

    plan = _with_plan_path(
        _plan([""], directives={0: {"audio": {"src": "broken.wav"}}}),
        tmp_path,
    )
    (tmp_path / "broken.wav").write_bytes(b"not a wav")
    manifest = rendered_bundle(plan)

    with pytest.raises(MediaResolutionError, match="invalid local WAV"):
        materialize_voicerender_job(plan, manifest, _policy())
