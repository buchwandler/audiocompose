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

    job = load_and_materialize_renderplan_job(plan_path, render_dir=tmp_path)

    assert len(job.items) == 1
    assert isinstance(job.items[0], AudioClip)


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

    assert any(item.startswith("renderplan>=0.1.2,<0.2") for item in base_dependencies)
    assert not any(
        item.lower().startswith(("voicerender", "utterplan")) for item in base_dependencies
    )
    assert any(item.startswith("utterplan>=0.6.3,<0.7") for item in extra_dependencies)
    assert any(item.startswith("voicerender>=0.2.0,<0.3") for item in extra_dependencies)
    assert (project_path.parent / "audiocompose/integrations/voicerender.py").is_file()
