from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from renderplan import (
    PauseItem,
    RenderManifest,
    RenderPlan,
    SourceSpan,
    SpeechArtifact,
    SpeechControl,
    SpeechItem,
    save_manifest,
    save_plan,
)

from audiocompose import AudioClip, AudioJob, AudioSpan, Gain, OutputPolicy, Silence
from audiocompose.errors import HandoffValidationError
from audiocompose.integrations.renderplan import (
    load_and_materialize_renderplan_job,
    materialize_renderplan_job,
)
from audiocompose.wav import write_wav


def _id(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _speech(
    number: int = 1,
    *,
    text: str = "Hello",
    controls=(),
) -> SpeechItem:
    return SpeechItem(
        number=number,
        text=text,
        language="en-US",
        role="narrator",
        binding={"engine": "fixture"},
        controls=tuple(controls),
        source=(SourceSpan("source.txt", 12, 12 + len(text)),),
    ).with_identity()


def _plan(*flow) -> RenderPlan:
    return RenderPlan(
        source_plan_id=_id("source"),
        render_profile_id=_id("profile"),
        flow=tuple(flow),
    ).with_identity()


def _manifest(plan: RenderPlan, artifacts=(), *, complete: bool = True) -> RenderManifest:
    return RenderManifest(
        render_plan_id=plan.plan_id or plan.computed_id(),
        speech=tuple(artifacts),
        complete=complete,
    ).with_identity()


def _artifact(item: SpeechItem, path: Path, *, frames=32, sample_rate=16_000, channels=1):
    return SpeechArtifact(
        segment_number=item.number,
        segment_id=item.segment_id or item.computed_id(),
        wav=item.wav,
        sha256="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=frames,
        sample_rate=sample_rate,
        channels=channels,
    )


def _rendered(tmp_path: Path, item: SpeechItem, *, frames=32, sample_rate=16_000):
    path = tmp_path / item.wav
    write_wav(path, np.full(frames, 0.125, dtype=np.float32), sample_rate)
    return path, _artifact(item, path, frames=frames, sample_rate=sample_rate)


def test_matching_contract_materializes_verified_speech_and_provenance(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    # The item identity is plan-independent, so the artifact remains exact.
    manifest = _manifest(plan, [artifact])

    job = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, output_policy=OutputPolicy(sample_rate=16_000)
    )

    assert len(job.items) == 1
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.id == "speech:000001"
    assert clip.spans == ()
    assert clip.metadata["renderplan.plan_id"] == plan.plan_id
    assert clip.metadata["renderplan.segment_number"] == 1
    assert clip.metadata["renderplan.segment_id"] == speech.segment_id
    assert clip.metadata["renderplan.source"] == [speech.source[0].to_dict()]
    assert clip.metadata["render_manifest.manifest_id"] == manifest.manifest_id
    assert clip.metadata["render_manifest.wav_sha256"] == artifact.sha256
    assert job.source["renderplan.plan_id"] == plan.plan_id
    assert job.output.sample_rate == 16_000
    job.validate(verify_sources=True)
    assert clip.operations == ()


@pytest.mark.parametrize(
    ("level", "expected_db"),
    [(None, 1.5), ("moderate", 1.5), ("reduced", 0.75), ("strong", 2.5)],
)
def test_full_span_emphasis_maps_gain_and_roundtrips_audiojob(tmp_path, level, expected_db):
    params = {} if level is None else {"level": level}
    speech = _speech(controls=(SpeechControl(0, 5, "emphasis", params),))
    wav_path, artifact = _rendered(tmp_path, speech)
    original_bytes = wav_path.read_bytes()
    plan = _plan(speech)
    job = materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)

    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.operations == (Gain(expected_db),)
    assert clip.metadata["renderplan.emphasis"] == {
        "policy_version": 1,
        "policy": "strict",
        "scope": "full_span",
        "approximation": False,
        "controls": [
            {
                "name": "emphasis",
                "start": 0,
                "end": 5,
                "params": params,
                "level": level or "moderate",
            }
        ],
        "partial_ranges": [],
        "selected_level": level or "moderate",
        "gain_db": expected_db,
    }
    assert clip.metadata["renderplan.segment_id"] == speech.segment_id
    assert clip.metadata["render_manifest.wav_sha256"] == artifact.sha256
    assert wav_path.read_bytes() == original_bytes

    saved = job.save(tmp_path / f"emphasis-{level or 'default'}.audiojob")
    loaded = AudioJob.load(saved)
    loaded_clip = loaded.items[0]
    assert isinstance(loaded_clip, AudioClip)
    assert loaded.schema_version == 2
    assert loaded_clip.operations == (Gain(expected_db),)
    assert loaded_clip.metadata["renderplan.emphasis"] == clip.metadata["renderplan.emphasis"]
    assert loaded.job_id


def test_partial_emphasis_uses_explicit_fallback_and_strict_rejects(tmp_path):
    from audiocompose.operations import apply_operation

    speech = _speech(controls=(SpeechControl(1, 3, "emphasis", {}),))
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])

    job = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, emphasis_policy="segment_fallback"
    )
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    report = clip.metadata["renderplan.emphasis"]
    assert clip.operations == (Gain(1.5),)
    assert report["scope"] == "segment_fallback"
    assert report["approximation"] is True
    assert report["controls"][0]["start"] == 1
    assert report["controls"][0]["end"] == 3
    boosted = apply_operation(np.full(32, 0.125, dtype=np.float32), 16_000, clip.operations[0])
    assert np.sqrt(np.mean(boosted**2)) / 0.125 == pytest.approx(10 ** (1.5 / 20))

    with pytest.raises(HandoffValidationError, match=r"speech 1 emphasis range.*segment_fallback"):
        materialize_renderplan_job(plan, manifest, render_dir=tmp_path, emphasis_policy="strict")


def test_overlapping_emphasis_uses_strongest_gain_once(tmp_path):
    speech = _speech(
        controls=(
            SpeechControl(0, 5, "emphasis", {"level": "moderate"}),
            SpeechControl(1, 4, "emphasis", {"level": "strong"}),
        )
    )
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    job = materialize_renderplan_job(
        plan,
        _manifest(plan, [artifact]),
        render_dir=tmp_path,
        emphasis_policy="segment_fallback",
    )
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.operations == (Gain(2.5),)
    assert clip.metadata["renderplan.emphasis"]["selected_level"] == "strong"


@pytest.mark.parametrize(
    ("control", "message"),
    [
        (SpeechControl(1, 3, "emphasis", {"unknown": 1}), "unsupported emphasis parameter"),
        (SpeechControl(1, 3, "emphasis", {"level": "mild"}), "unsupported emphasis level"),
        (SpeechControl(1, 3, "prosody", {"level": "moderate"}), "unsupported speech control"),
        (SpeechControl(1, 3, "emphasis", {"level": True}), "unsupported emphasis level"),
    ],
)
def test_invalid_emphasis_controls_fail_with_segment_range(tmp_path, control, message):
    speech = _speech(controls=(control,))
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    with pytest.raises(
        HandoffValidationError, match=rf"speech 1 control range \[1, 3\).*{message}"
    ):
        materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)


def test_invalid_mutated_control_range_is_rejected():
    from audiocompose.integrations.renderplan import _speech_emphasis

    speech = _speech()
    control = SpeechControl(0, 1, "emphasis", {})
    object.__setattr__(control, "end", len(speech.text) + 1)
    object.__setattr__(speech, "controls", (control,))
    with pytest.raises(
        HandoffValidationError, match="speech 1 control range.*expected character offsets"
    ):
        _speech_emphasis(speech, "segment_fallback")


def test_emphasis_policy_and_metadata_affect_audiojob_identity(tmp_path):
    speech = _speech(controls=(SpeechControl(0, 5, "emphasis", {}),))
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])
    strict = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, emphasis_policy="strict"
    )
    fallback = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, emphasis_policy="segment_fallback"
    )
    strict_id = AudioJob.load(strict.save(tmp_path / "strict.audiojob")).job_id
    fallback_id = AudioJob.load(fallback.save(tmp_path / "fallback.audiojob")).job_id
    assert strict_id != fallback_id


def test_emphasis_preserves_peak_limits_and_clip_policy(tmp_path):
    from audiocompose import Composer, LoudnessPolicy
    from audiocompose.errors import CompositionError

    speech = _speech(controls=(SpeechControl(0, 5, "emphasis", {"level": "strong"}),))
    wav_path = tmp_path / speech.wav
    write_wav(wav_path, np.full(32, 0.9, dtype=np.float32), 16_000)
    artifact = _artifact(speech, wav_path)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])
    no_limit = OutputPolicy(
        sample_rate=16_000,
        clip_policy="error",
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=None),
    )
    job = materialize_renderplan_job(plan, manifest, render_dir=tmp_path, output_policy=no_limit)
    assert job.output.clip_policy == "error"
    emphasized_clip = job.items[0]
    assert isinstance(emphasized_clip, AudioClip)
    assert emphasized_clip.operations == (Gain(2.5),)
    with pytest.raises(CompositionError, match="outside the PCM range"):
        Composer().compose(job)

    limited_policy = OutputPolicy(
        sample_rate=16_000,
        loudness=LoudnessPolicy(target_lufs=None, true_peak_ceiling_dbtp=-1.0),
    )
    limited_job = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, output_policy=limited_policy
    )
    composition = Composer().compose(limited_job)
    assert np.max(np.abs(composition.audio)) <= 10 ** (-1.0 / 20) + 1e-6
    assert composition.loudness is not None
    assert composition.loudness.applied_gain_db < 0


def test_unemphasized_job_identity_and_operations_ignore_emphasis_policy(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])
    strict = materialize_renderplan_job(plan, manifest, render_dir=tmp_path)
    fallback = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, emphasis_policy="segment_fallback"
    )
    strict_clip = strict.items[0]
    fallback_clip = fallback.items[0]
    assert isinstance(strict_clip, AudioClip)
    assert isinstance(fallback_clip, AudioClip)
    assert strict_clip.operations == fallback_clip.operations == ()
    assert "renderplan.emphasis" not in strict_clip.metadata
    assert "renderplan.emphasis" not in fallback_clip.metadata
    strict_id = AudioJob.load(strict.save(tmp_path / "plain-strict.audiojob")).job_id
    fallback_id = AudioJob.load(fallback.save(tmp_path / "plain-fallback.audiojob")).job_id
    assert strict_id == fallback_id


def test_multiple_full_span_emphasis_uses_strongest_gain_once(tmp_path):
    speech = _speech(
        controls=(
            SpeechControl(0, 5, "emphasis", {"level": "moderate"}),
            SpeechControl(0, 5, "emphasis", {"level": "strong"}),
        )
    )
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    job = materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.operations == (Gain(2.5),)
    assert clip.metadata["renderplan.emphasis"]["scope"] == "full_span"


def test_exact_flow_pauses_include_leading_and_zero_duration_events(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    leading = PauseItem(duration_ms=500, reason="sentence").with_identity()
    trailing = PauseItem(duration_ms=0, reason="capacity").with_identity()
    plan = _plan(leading, speech, trailing)
    manifest = _manifest(plan, [artifact])

    job = materialize_renderplan_job(plan, manifest, render_dir=tmp_path)

    assert [item.id for item in job.items] == [
        f"pause:000001:{leading.pause_id}",
        "speech:000001",
        f"pause:000003:{trailing.pause_id}",
    ]
    first, _, last = job.items
    assert isinstance(first, Silence) and first.seconds == 0.5
    assert first.metadata["renderplan.pause_reason"] == "sentence"
    assert isinstance(last, Silence) and last.seconds == 0
    assert last.metadata["renderplan.pause_reason"] == "capacity"
    assert first.metadata["renderplan.pause_id"] == leading.pause_id
    job.validate(verify_sources=True)


def test_manifest_must_be_complete_and_belong_to_plan(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    with pytest.raises(HandoffValidationError, match="incomplete"):
        materialize_renderplan_job(
            plan, _manifest(plan, [artifact], complete=False), render_dir=tmp_path
        )

    wrong_plan = _manifest(_plan(_speech(text="Other")), [artifact])
    with pytest.raises(HandoffValidationError, match="match RenderPlan"):
        materialize_renderplan_job(plan, wrong_plan, render_dir=tmp_path)


def test_missing_and_extra_artifacts_are_rejected(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    with pytest.raises(HandoffValidationError, match="missing"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path)

    extra = replace(artifact, segment_number=2, segment_id=_id("extra"), wav="000002.wav")
    with pytest.raises(HandoffValidationError, match="not in plan"):
        materialize_renderplan_job(plan, _manifest(plan, [artifact, extra]), render_dir=tmp_path)


def test_speech_artifact_hash_and_geometry_must_match(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)

    bad_hash = replace(artifact, sha256=_id("different"))
    with pytest.raises(HandoffValidationError, match="SHA-256"):
        materialize_renderplan_job(plan, _manifest(plan, [bad_hash]), render_dir=tmp_path)

    bad_frames = replace(artifact, frames=artifact.frames + 1)
    with pytest.raises(HandoffValidationError, match="frame mismatch"):
        materialize_renderplan_job(plan, _manifest(plan, [bad_frames]), render_dir=tmp_path)

    bad_rate = replace(artifact, sample_rate=artifact.sample_rate + 1)
    with pytest.raises(HandoffValidationError, match="sample-rate mismatch"):
        materialize_renderplan_job(plan, _manifest(plan, [bad_rate]), render_dir=tmp_path)

    bad_channels = replace(artifact, channels=2)
    with pytest.raises(HandoffValidationError, match="channel mismatch"):
        materialize_renderplan_job(plan, _manifest(plan, [bad_channels]), render_dir=tmp_path)


def test_segment_identity_mismatch_and_unsafe_wav_path_fail_closed(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    wrong_segment = replace(artifact, segment_id=_id("wrong-segment"))
    with pytest.raises(HandoffValidationError, match="match RenderPlan"):
        materialize_renderplan_job(plan, _manifest(plan, [wrong_segment]), render_dir=tmp_path)

    object.__setattr__(artifact, "wav", "../escape.wav")
    with pytest.raises(HandoffValidationError, match="WAV name|escape"):
        materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)


def test_symlink_escape_is_rejected(tmp_path):
    speech = _speech()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.wav"
    write_wav(outside, np.zeros(32, dtype=np.float32), 16_000)
    (tmp_path / speech.wav).symlink_to(outside)
    artifact = _artifact(speech, outside)
    plan = _plan(speech)

    with pytest.raises(HandoffValidationError, match="symlink"):
        materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)


def test_path_loader_defaults_manifest_inside_render_dir(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    plan_path = tmp_path / "chapter.renderplan.toml"
    manifest_path = tmp_path / "render-manifest.toml"
    save_plan(plan, plan_path)
    save_manifest(_manifest(plan, [artifact]), manifest_path)
    serialized = plan_path.read_text(encoding="utf-8")
    assert "schema_version = 2" in serialized
    assert "[bindings.narrator]" in serialized

    job = load_and_materialize_renderplan_job(plan_path, render_dir=tmp_path)

    assert len(job.items) == 1
    assert isinstance(job.items[0], AudioClip)


def test_parenthetical_semantic_end_uses_explicit_pause_only(tmp_path):
    speech = replace(_speech(text="aside"), semantic_end="parenthetical").with_identity()
    pause = PauseItem(duration_ms=180, reason="parenthetical").with_identity()
    plan = _plan(speech, pause)
    _, artifact = _rendered(tmp_path, speech)

    job = materialize_renderplan_job(
        plan,
        _manifest(plan, [artifact]),
        render_dir=tmp_path,
    )

    assert speech.semantic_end == "parenthetical"
    assert len(job.items) == 2
    assert isinstance(job.items[0], AudioClip)
    assert isinstance(job.items[1], Silence)
    assert job.items[1].seconds == pytest.approx(0.180)


def test_speech_metadata_does_not_create_sample_aligned_spans(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    job = materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)

    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.spans == ()
    assert not any(isinstance(span, AudioSpan) for span in clip.spans)


def _audio(src="music.wav", *, options=None):
    from renderplan import AudioItem

    return AudioItem(
        src=src,
        source=(SourceSpan("source.txt", 0, 5),),
        options=options or {},
    ).with_identity()


def test_verified_speech_is_not_reloaded_but_external_file_sources_are_validated(
    tmp_path: Path, monkeypatch
) -> None:
    from audiocompose.integrations.media import ResolvedMedia
    from audiocompose.sources import AudioFileSource
    from audiocompose.wav import write_wav

    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    media_file = tmp_path / "external.wav"
    write_wav(media_file, np.full(24, 0.25, dtype=np.float32), 1000)
    media_hash = hashlib.sha256(media_file.read_bytes()).hexdigest()
    source = AudioFileSource(media_file, media_hash, sample_rate=1000, channels=1, frames=24)

    class Resolver:
        def __init__(self, resolved_source):
            self.resolved_source = resolved_source

        def resolve(self, request):
            return ResolvedMedia(self.resolved_source)

    original_load = AudioFileSource.load
    loaded_paths = []

    def counted_load(self):
        loaded_paths.append(Path(self.path))
        return original_load(self)

    monkeypatch.setattr(AudioFileSource, "load", counted_load)
    plan = _plan(speech, _audio("fixture:external"))
    job = materialize_renderplan_job(
        plan,
        _manifest(plan, [artifact]),
        render_dir=tmp_path,
        media_resolver=Resolver(source),
    )
    assert len(job.items) == 2
    assert loaded_paths == [media_file]

    loaded_paths.clear()
    untrusted_source = AudioFileSource(
        media_file, "0" * 64, sample_rate=1000, channels=1, frames=24
    )
    with pytest.raises(HandoffValidationError, match="source hash mismatch"):
        materialize_renderplan_job(
            plan,
            _manifest(plan, [artifact]),
            render_dir=tmp_path,
            media_resolver=Resolver(untrusted_source),
        )
    assert loaded_paths == [media_file]


def test_external_audio_uses_explicit_media_root_and_supported_operations(tmp_path):
    from audiocompose import Gain, Tempo
    from audiocompose.sources import AudioBufferSource
    from audiocompose.wav import write_wav

    media_root = tmp_path / "media"
    media_root.mkdir()
    media_file = media_root / "music.wav"
    write_wav(media_file, np.full(100, 0.25, dtype=np.float32), 1000)
    external = _audio(
        options={
            "clip_begin_ms": 20,
            "clip_end_ms": 50,
            "speed": 1.25,
            "gain_db": -3.0,
        }
    )
    plan = _plan(external)

    job = materialize_renderplan_job(
        plan, _manifest(plan), render_dir=tmp_path, media_root=media_root
    )

    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert isinstance(clip.source, AudioBufferSource)
    audio, rate = clip.source.load()
    assert rate == 1000 and len(audio) == 30
    assert isinstance(clip.operations[0], Tempo)
    assert clip.operations[0].factor == pytest.approx(1.25)
    assert isinstance(clip.operations[1], Gain)
    assert clip.operations[1].db == pytest.approx(-3.0)
    assert clip.metadata["renderplan.audio_id"] == external.audio_id
    assert clip.metadata["renderplan.source"] == [external.source[0].to_dict()]


def test_external_audio_traversal_and_unsupported_options_are_rejected(tmp_path):
    from audiocompose.errors import MediaResolutionError

    traversal = _audio("../outside.wav")
    plan = _plan(traversal)
    with pytest.raises(MediaResolutionError, match="beneath"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path, media_root=tmp_path)

    unknown = _audio(options={"unreviewed": True})
    plan = _plan(unknown)
    with pytest.raises(MediaResolutionError, match="unsupported AudioItem option"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path, media_root=tmp_path)


def test_external_audio_uri_requires_explicit_resolver_and_custom_resolver_works(tmp_path):
    from audiocompose import AudioBufferSource
    from audiocompose.errors import MediaResolutionError
    from audiocompose.integrations.media import ResolvedMedia

    external = _audio("fixture:impact")
    plan = _plan(external)
    with pytest.raises(MediaResolutionError, match="no resolver for URI scheme"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path)

    class Resolver:
        request = None

        def resolve(self, request):
            self.request = request
            return ResolvedMedia(
                AudioBufferSource(np.ones(12, dtype=np.float32), 12_000),
                metadata={"provider": "fixture"},
                identity={"asset": "impact"},
            )

    resolver = Resolver()
    job = materialize_renderplan_job(
        plan, _manifest(plan), render_dir=tmp_path, media_resolver=resolver
    )
    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert resolver.request is not None
    assert resolver.request.src == "fixture:impact"
    assert clip.metadata["audiocompose.media.identity"] == {"asset": "impact"}


def test_local_media_symlink_escape_and_missing_root_fail_closed(tmp_path):
    from audiocompose.errors import MediaResolutionError

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    write_wav(outside, np.ones(20, dtype=np.float32), 16_000)
    (root / "link.wav").symlink_to(outside)
    plan = _plan(_audio("link.wav"))
    with pytest.raises(MediaResolutionError, match="symlink"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path, media_root=root)

    plan = _plan(_audio())
    with pytest.raises(MediaResolutionError, match="explicit media_root"):
        materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path)


def test_compose_renderplan_cli_matches_explicit_materialize_and_composition(tmp_path, capsys):
    import json

    from audiocompose import Composer
    from audiocompose.cli import main
    from audiocompose.wav import read_wav

    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    leading = PauseItem(duration_ms=500, reason="sentence").with_identity()
    plan = _plan(leading, speech)
    manifest = _manifest(plan, [artifact])
    plan_path = tmp_path / "chapter.renderplan.toml"
    save_plan(plan, plan_path)
    save_manifest(manifest, tmp_path / "render-manifest.toml")
    output = tmp_path / "chapter.wav"

    assert (
        main(
            [
                "compose-renderplan",
                str(plan_path),
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--sample-rate",
                "16000",
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["composed"] is True
    assert response["plan_id"] == plan.plan_id
    assert response["manifest_id"] == manifest.manifest_id
    assert response["sample_rate"] == 16_000
    assert response["timeline"][0]["renderplan.pause_reason"] == "sentence"
    assert response["timeline"][1]["renderplan.segment_id"] == speech.segment_id

    expected_job = materialize_renderplan_job(
        plan, manifest, render_dir=tmp_path, output_policy=OutputPolicy(sample_rate=16_000)
    )
    expected = Composer().compose(expected_job)
    actual_audio, actual_rate = read_wav(output)
    assert actual_rate == expected.sample_rate
    assert len(actual_audio) == len(expected.audio)
    assert response["frames"] == len(expected.audio)
    assert response["timeline"] == [
        {
            "id": item.id,
            "kind": item.metadata["kind"],
            "start_sample": composed.start_sample,
            "end_sample": composed.end_sample,
            "duration_seconds": composed.duration_samples / expected.sample_rate,
            **{
                key: item.metadata[key]
                for key in (
                    "renderplan.segment_number",
                    "renderplan.segment_id",
                    "renderplan.source",
                    "renderplan.pause_id",
                    "renderplan.pause_reason",
                    "renderplan.pause_source",
                    "renderplan.audio_id",
                    "renderplan.audio.src",
                    "renderplan.audio.options",
                    "render_manifest.wav",
                    "render_manifest.wav_sha256",
                )
                if key in item.metadata
            },
        }
        for item, composed in zip(expected_job.items, expected.items, strict=True)
    ]


def test_renderplan_cli_defaults_to_fallback_and_strict_writes_no_output(tmp_path, capsys):
    import json

    from audiocompose.cli import main
    from audiocompose.wav import read_wav

    speech = _speech(controls=(SpeechControl(1, 3, "emphasis", {}),))
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    plan_path = tmp_path / "emphasis.renderplan.toml"
    save_plan(plan, plan_path)
    save_manifest(_manifest(plan, [artifact]), tmp_path / "render-manifest.toml")

    output = tmp_path / "fallback.wav"
    assert (
        main(
            [
                "compose-renderplan",
                str(plan_path),
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["emphasis_policy"] == "segment_fallback"
    assert response["timeline"][0]["renderplan.emphasis"]["scope"] == "segment_fallback"
    assert response["diagnostics"][0]["code"] == "emphasis.segment_fallback"
    assert output.is_file()
    audible, sample_rate = read_wav(output)
    assert sample_rate == 24_000
    assert np.max(np.abs(audible)) > 0

    strict_output = tmp_path / "strict.wav"
    assert (
        main(
            [
                "compose-renderplan",
                str(plan_path),
                "--render-dir",
                str(tmp_path),
                "--output",
                str(strict_output),
                "--emphasis-policy",
                "strict",
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert "split the speech upstream" in error["errors"][0]["message"]
    assert not strict_output.exists()


def test_compose_renderplan_cli_explains_provenance_without_pause_policy_options(tmp_path, capsys):
    from audiocompose.cli import main

    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    save_plan(plan, tmp_path / "chapter.renderplan.toml")
    save_manifest(_manifest(plan, [artifact]), tmp_path / "render-manifest.toml")
    output = tmp_path / "chapter.wav"

    assert (
        main(
            [
                "compose-renderplan",
                str(tmp_path / "chapter.renderplan.toml"),
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--explain",
            ]
        )
        == 0
    )
    explanation = capsys.readouterr().out
    assert "RenderPlan:" in explanation
    assert speech.segment_id in explanation
    assert "source.txt[12:17]" in explanation

    with pytest.raises(SystemExit):
        main(
            [
                "compose-renderplan",
                "chapter.renderplan.toml",
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--pauses",
                "pause.toml",
            ]
        )


def test_compose_renderplan_cli_defaults_media_root_to_plan_directory(tmp_path):
    from audiocompose.cli import main
    from audiocompose.wav import read_wav

    media = np.full(32, 0.2, dtype=np.float32)
    write_wav(tmp_path / "music.wav", media, 16_000)
    plan = _plan(_audio("music.wav"))
    save_plan(plan, tmp_path / "chapter.renderplan.toml")
    save_manifest(_manifest(plan), tmp_path / "render-manifest.toml")
    output = tmp_path / "chapter.wav"

    assert (
        main(
            [
                "compose-renderplan",
                str(tmp_path / "chapter.renderplan.toml"),
                "--render-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--sample-rate",
                "16000",
            ]
        )
        == 0
    )
    audio, rate = read_wav(output)
    assert rate == 16_000
    assert len(audio) == 32


def test_manifest_render_plan_id_wav_name_and_invalid_wav_are_rejected(tmp_path):
    speech = _speech()
    wav_path, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)

    wrong_plan = RenderManifest(
        render_plan_id=_id("wrong-plan"), speech=(artifact,), complete=True
    ).with_identity()
    with pytest.raises(HandoffValidationError, match="does not match RenderPlan"):
        materialize_renderplan_job(plan, wrong_plan, render_dir=tmp_path)

    object.__setattr__(artifact, "wav", "000002.wav")
    with pytest.raises(HandoffValidationError, match="WAV name"):
        materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)

    wav_path.write_bytes(b"not a WAV")
    broken = _artifact(speech, wav_path)
    with pytest.raises(HandoffValidationError, match="invalid supported WAV"):
        materialize_renderplan_job(plan, _manifest(plan, [broken]), render_dir=tmp_path)


def test_audioitem_pause_speech_order_and_pause_reason_are_literal(tmp_path):
    from audiocompose import Composer

    media_root = tmp_path / "media"
    media_root.mkdir()
    write_wav(media_root / "music.wav", np.full(32, 0.1, dtype=np.float32), 16_000)
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    pause = PauseItem(duration_ms=500, reason="sentence").with_identity()
    other_pause = PauseItem(duration_ms=500, reason="capacity").with_identity()
    external = _audio()
    plan = _plan(external, pause, speech, other_pause)

    job = materialize_renderplan_job(
        plan, _manifest(plan, [artifact]), render_dir=tmp_path, media_root=media_root
    )

    assert [item.metadata["kind"] for item in job.items] == [
        "external_audio",
        "pause",
        "speech",
        "pause",
    ]
    assert isinstance(job.items[1], Silence) and isinstance(job.items[3], Silence)
    assert job.items[1].seconds == job.items[3].seconds == 0.5
    assert job.items[1].metadata["renderplan.pause_reason"] == "sentence"
    assert job.items[3].metadata["renderplan.pause_reason"] == "capacity"
    composition = Composer().compose(job)
    assert [item.item_id for item in composition.items] == [item.id for item in job.items]


def test_zero_pause_has_exactly_zero_timeline_duration(tmp_path):
    from audiocompose import Composer

    pause = PauseItem(duration_ms=0, reason="capacity").with_identity()
    plan = _plan(pause)
    job = materialize_renderplan_job(plan, _manifest(plan), render_dir=tmp_path)

    composed = Composer().compose(job).items[0]
    assert composed.start_sample == composed.end_sample


def test_adapter_has_no_symbolic_pause_policy_surface():
    import inspect

    from audiocompose.integrations import renderplan as adapter

    assert "pause_policy" not in inspect.signature(adapter.materialize_renderplan_job).parameters
    assert not hasattr(adapter, "PausePolicy")
    assert not hasattr(adapter, "load_pause_policy")


def test_different_source_rates_are_normalized_by_composer(tmp_path):
    from audiocompose import Composer

    speech = _speech()
    _, artifact = _rendered(tmp_path, speech, frames=640, sample_rate=16_000)
    media_root = tmp_path / "media"
    media_root.mkdir()
    write_wav(media_root / "music.wav", np.full(320, 0.1, dtype=np.float32), 8_000)
    plan = _plan(speech, _audio())
    job = materialize_renderplan_job(
        plan,
        _manifest(plan, [artifact]),
        render_dir=tmp_path,
        media_root=media_root,
        output_policy=OutputPolicy(sample_rate=24_000),
    )

    composition = Composer().compose(job)

    assert composition.sample_rate == 24_000
    assert [item.source_sample_rate for item in composition.items] == [16_000, 8_000]
    assert len(composition.audio) == 1_920


def test_generic_audiojob_save_load_remains_usable(tmp_path):
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    job = materialize_renderplan_job(plan, _manifest(plan, [artifact]), render_dir=tmp_path)

    saved_path = job.save(tmp_path / "job.audiojob")
    restored = type(job).load(saved_path, verify_sources=True)

    assert restored.source["renderplan.plan_id"] == plan.plan_id
    assert restored.items[0].metadata["renderplan.segment_id"] == speech.segment_id


def test_package_and_adapter_import_without_voice_render_or_utterplan():
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
class BlockProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'voicerender' or fullname.startswith('voicerender.'):
            raise ModuleNotFoundError(fullname)
        if fullname == 'utterplan' or fullname.startswith('utterplan.'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockProducerImports())
import audiocompose
import audiocompose.integrations.renderplan
import audiocompose.integrations.voicerender
assert not any(name == 'voicerender' or name.startswith('voicerender.') for name in sys.modules)
assert not any(name == 'utterplan' or name.startswith('utterplan.') for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_producer_stack_is_an_optional_extra_not_a_base_dependency():
    import importlib

    try:
        tomllib = importlib.import_module("tomllib")
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        tomllib = importlib.import_module("tomli")

    project_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    base_dependencies = project["project"]["dependencies"]
    extra_dependencies = project["project"]["optional-dependencies"]["voicerender"]

    assert any(item.startswith("renderplan>=0.2.1,<0.3") for item in base_dependencies)
    assert not any(
        item.lower().startswith(("voicerender", "utterplan")) for item in base_dependencies
    )
    assert any(item.startswith("utterplan>=0.6.12,<0.7") for item in extra_dependencies)
    assert any(item.startswith("voicerender>=0.2.8,<0.3") for item in extra_dependencies)
    assert (project_path.parent / "audiocompose/integrations/voicerender.py").is_file()


def test_persisted_speech_conditions_verified_samples_without_reloading_wav(
    tmp_path: Path, monkeypatch
) -> None:
    from audiocompose.integrations import renderplan as adapter
    from audiocompose.integrations.renderplan_metadata import (
        AUDIOCOMPOSE_SPEECH_BOUNDARY_INPUT_FRAMES,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_OUTPUT_FRAMES,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_FRAMES,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_SECONDS,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_TRIM_END_SAMPLE,
    )
    from audiocompose.sources import AudioBufferSource, RenderedAudioBuffer

    sample_rate = 1_000
    audio = np.concatenate(
        (
            np.zeros(200, dtype=np.float32),
            np.full(300, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    speech = _speech()
    wav_path = tmp_path / speech.wav
    write_wav(wav_path, audio, sample_rate)
    raw_bytes = wav_path.read_bytes()
    artifact = _artifact(speech, wav_path, frames=len(audio), sample_rate=sample_rate)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])

    original_read_wav = adapter.read_wav
    read_paths = []

    def counted_read_wav(path, *args, **kwargs):
        read_paths.append(Path(path))
        return original_read_wav(path, *args, **kwargs)

    monkeypatch.setattr(adapter, "read_wav", counted_read_wav)
    persisted = materialize_renderplan_job(plan, manifest, render_dir=tmp_path)
    assert read_paths == [wav_path]
    assert wav_path.read_bytes() == raw_bytes
    assert hashlib.sha256(wav_path.read_bytes()).hexdigest() == artifact.sha256.removeprefix(
        "sha256:"
    )

    persisted_clip = persisted.items[0]
    assert isinstance(persisted_clip, AudioClip)
    assert isinstance(persisted_clip.source, AudioBufferSource)
    persisted_audio, actual_rate = persisted_clip.source.load()
    assert actual_rate == sample_rate
    assert len(persisted_audio) == 500
    np.testing.assert_array_equal(persisted_audio, audio[:500])
    assert persisted_clip.metadata["render_manifest.frames"] == 900
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY] == (
        "audiocompose.trailing-silence.v1"
    )
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_INPUT_FRAMES] == 900
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_OUTPUT_FRAMES] == 500
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_FRAMES] == 400
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_REMOVED_SECONDS] == 0.4
    assert persisted_clip.metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_TRIM_END_SAMPLE] == 500

    occurrence = (speech.number, speech.segment_id or speech.computed_id())
    rendered = RenderedAudioBuffer(audio, sample_rate)
    in_memory = materialize_renderplan_job(plan, rendered_speech={occurrence: rendered})
    in_memory_clip = in_memory.items[0]
    assert isinstance(in_memory_clip, AudioClip)
    in_memory_audio, memory_rate = in_memory_clip.source.load()
    assert memory_rate == actual_rate
    np.testing.assert_array_equal(in_memory_audio, persisted_audio)


def test_word_timings_overlapping_removed_tail_are_preserved_verbatim(tmp_path: Path) -> None:
    from audiocompose.sources import RenderedAudioBuffer

    sample_rate = 1_000
    audio = np.concatenate(
        (
            np.zeros(200, dtype=np.float32),
            np.full(300, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    speech = _speech()
    plan = _plan(speech)
    occurrence = (speech.number, speech.segment_id or speech.computed_id())
    timing = {"word": "Hello", "start_sample": 450, "end_sample": 650}
    rendered = RenderedAudioBuffer(audio, sample_rate, word_timings=(timing,))

    job = materialize_renderplan_job(plan, rendered_speech={occurrence: rendered})

    clip = job.items[0]
    assert isinstance(clip, AudioClip)
    assert clip.metadata["renderplan.word_timings"] == [timing]


@pytest.mark.parametrize(
    "timing",
    [
        {"word": "Hello", "start_sample": 501, "end_sample": 600},
        {"word": "Hello", "start_seconds": 0.501, "end_seconds": 0.6},
    ],
)
def test_word_timing_start_beyond_conditioned_endpoint_is_rejected(
    timing: dict[str, object],
) -> None:
    from audiocompose.sources import RenderedAudioBuffer

    audio = np.concatenate(
        (
            np.zeros(200, dtype=np.float32),
            np.full(300, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    speech = _speech()
    plan = _plan(speech)
    occurrence = (speech.number, speech.segment_id or speech.computed_id())
    rendered = RenderedAudioBuffer(audio, 1_000, word_timings=(timing,))

    with pytest.raises(HandoffValidationError, match="exceeds conditioned endpoint"):
        materialize_renderplan_job(plan, rendered_speech={occurrence: rendered})


def test_policy_and_materializer_versions_change_audiojob_identity(
    tmp_path: Path, monkeypatch
) -> None:
    import audiocompose.speech_boundaries as speech_boundaries
    from audiocompose.integrations import renderplan as adapter
    from audiocompose.integrations.renderplan_metadata import (
        AUDIOCOMPOSE_MATERIALIZER,
        AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY,
    )

    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    plan = _plan(speech)
    manifest = _manifest(plan, [artifact])
    original = materialize_renderplan_job(plan, manifest, render_dir=tmp_path)
    assert original.source[AUDIOCOMPOSE_MATERIALIZER] == "renderplan.v2"
    assert original.source[AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY] == (
        "audiocompose.trailing-silence.v1"
    )

    monkeypatch.setattr(adapter, "SPEECH_BOUNDARY_POLICY_ID", "audiocompose.trailing-silence.test")
    monkeypatch.setattr(
        speech_boundaries,
        "SPEECH_BOUNDARY_POLICY_ID",
        "audiocompose.trailing-silence.test",
    )
    changed_policy = materialize_renderplan_job(plan, manifest, render_dir=tmp_path)
    assert changed_policy.source[AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY] == (
        "audiocompose.trailing-silence.test"
    )
    assert changed_policy.items[0].metadata[AUDIOCOMPOSE_SPEECH_BOUNDARY_POLICY] == (
        "audiocompose.trailing-silence.test"
    )

    original_id = AudioJob.load(original.save(tmp_path / "original.audiojob")).job_id
    changed_policy_id = AudioJob.load(
        changed_policy.save(tmp_path / "changed-policy.audiojob")
    ).job_id
    older_materializer = replace(
        original,
        source={**original.source, AUDIOCOMPOSE_MATERIALIZER: "renderplan.v1"},
    )
    older_materializer_id = AudioJob.load(
        older_materializer.save(tmp_path / "older-materializer.audiojob")
    ).job_id
    assert original_id != changed_policy_id
    assert original_id != older_materializer_id


def test_explicit_pause_uses_conditioned_speech_boundary_and_estimate_matches(
    tmp_path: Path,
) -> None:
    from audiocompose import Composer, estimate_composition

    sample_rate = 1_000
    first = _speech(1)
    second = _speech(2, text="World")
    pause = PauseItem(duration_ms=500, reason="sentence").with_identity()
    first_audio = np.concatenate(
        (np.full(200, 0.5, dtype=np.float32), np.zeros(400, dtype=np.float32))
    )
    second_audio = np.full(100, 0.5, dtype=np.float32)
    first_path = tmp_path / first.wav
    second_path = tmp_path / second.wav
    write_wav(first_path, first_audio, sample_rate)
    write_wav(second_path, second_audio, sample_rate)
    plan = _plan(first, pause, second)
    artifacts = [
        _artifact(first, first_path, frames=len(first_audio), sample_rate=sample_rate),
        _artifact(second, second_path, frames=len(second_audio), sample_rate=sample_rate),
    ]

    job = materialize_renderplan_job(
        plan,
        _manifest(plan, artifacts),
        render_dir=tmp_path,
        output_policy=OutputPolicy(sample_rate=sample_rate),
    )
    composition = Composer().compose(job)
    estimate = estimate_composition(job)

    assert estimate.item_output_frames == (200, 500, 100)
    assert estimate.total_output_frames == len(composition.audio) == 800
    assert [(item.start_sample, item.end_sample) for item in composition.items] == [
        (0, 200),
        (200, 700),
        (700, 800),
    ]


def test_external_audio_with_silence_is_not_speech_conditioned(tmp_path: Path) -> None:
    from audiocompose.sources import AudioFileSource

    sample_rate = 1_000
    speech = _speech()
    _, artifact = _rendered(tmp_path, speech)
    media_audio = np.concatenate(
        (
            np.zeros(200, dtype=np.float32),
            np.full(300, 0.5, dtype=np.float32),
            np.zeros(400, dtype=np.float32),
        )
    )
    media_root = tmp_path / "media"
    media_root.mkdir()
    write_wav(media_root / "music.wav", media_audio, sample_rate)
    plan = _plan(speech, _audio("music.wav"))

    job = materialize_renderplan_job(
        plan,
        _manifest(plan, [artifact]),
        render_dir=tmp_path,
        media_root=media_root,
    )

    media_clip = job.items[1]
    assert isinstance(media_clip, AudioClip)
    assert media_clip.metadata["kind"] == "external_audio"
    assert isinstance(media_clip.source, AudioFileSource)
    resolved_audio, resolved_rate = media_clip.source.load()
    assert resolved_rate == sample_rate
    assert len(resolved_audio) == len(media_audio) == 900
    np.testing.assert_array_equal(resolved_audio, media_audio)
