"""Installed-wheel smoke for a real VoiceRender v0.2 producer handoff."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import utterplan
import voicerender
from renderplan import (
    RenderPlan,
    RenderPlanSet,
    RenderPlanSetChapter,
    RenderProfile,
    RoleProfile,
    SourceSpan,
    SpeechControl,
    SpeechItem,
    load_manifest,
    load_plan,
    load_renderplan_set,
    save_plan,
    save_profile,
    save_renderplan_set,
)
from voicerender.engines.api_probe import EngineApiProbe
from voicerender.engines.base import (
    EngineCapabilities,
    EngineDiagnostic,
    EngineSelection,
    EngineSession,
    RenderedSpeech,
    RenderProfileFragment,
    RequestMeasure,
    SpeechRequest,
)
from voicerender.engines.registry import register_adapter
from voicerender.engines.selection import EngineRequest
from voicerender.preflight import preflight_render_plan
from voicerender.renderplan_render import render_renderplan

from audiocompose import (
    BookComposeResult,
    DocumentComposeResult,
    compose_book,
    compose_document,
)
from audiocompose.integrations.renderplan_metadata import (
    RENDER_DIAGNOSTICS_FILENAME,
    VOICERENDER_DEFERRED_CAPABILITIES,
    VOICERENDER_DIAGNOSTICS_SHA256,
)
from audiocompose.integrations.voicerender import (
    EMPHASIS_COMPOSITION_CAPABILITY,
    materialize_voicerender_job,
)
from audiocompose.job import load_job

ENGINE_ID = "audiocompose-installed-wheel-smoke"
TARGET_ID = "installed-wheel-model-v1"


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    assert match is not None, f"unexpected package version: {value}"
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


class _Session:
    def measure(self, request: SpeechRequest) -> RequestMeasure:
        return RequestMeasure(
            fits=True,
            amount=len(request.text),
            maximum=200,
            unit="model_tokens",
            source="audiocompose.installed_smoke.measure",
        )

    def synthesize(self, request: SpeechRequest) -> RenderedSpeech:
        return RenderedSpeech(
            id=request.id,
            audio=[0.0, 0.2, -0.2, 0.1],
            sample_rate=24_000,
        )


class _Adapter:
    id = ENGINE_ID
    package_name = "audiocompose-installed-smoke-fake"

    def __init__(self) -> None:
        self.session = _Session()

    def probe_api(self) -> EngineApiProbe:
        return EngineApiProbe(
            engine=self.id,
            package=self.package_name,
            compatible=True,
            status="ready",
            expected_api_version=1,
            api_version=1,
            contract_source="explicit",
        )

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            id=self.id,
            supports_pronunciation_overrides=True,
            pronunciation_alphabets=frozenset({"ipa"}),
        )

    def discover(self, **_: object) -> tuple[object, ...]:
        return ()

    def resolve(
        self, request: EngineRequest
    ) -> tuple[EngineSelection, tuple[EngineDiagnostic, ...]]:
        return (
            EngineSelection(
                engine=self.id,
                target_id=request.target_id or TARGET_ID,
                language=request.language or "en-US",
                voice=request.voice,
                speaker=request.speaker,
                options=request.options,
            ),
            (),
        )

    def validate_selection(self, selection: EngineSelection) -> tuple[EngineDiagnostic, ...]:
        if selection.target_id != TARGET_ID:
            return (
                EngineDiagnostic(
                    code="installed_smoke.target_invalid",
                    severity="error",
                    message="unexpected fake target",
                ),
            )
        return ()

    def version(self) -> str:
        return "1.0"

    def canonical_synthesis_identity(self, selection: EngineSelection) -> dict[str, object]:
        return {"target_id": selection.target_id, "voice": selection.voice, "revision": "smoke-r1"}

    def renderprofile_fragment(self, selection: EngineSelection) -> RenderProfileFragment:
        return RenderProfileFragment(
            binding={
                "renderer": self.id,
                "target_id": selection.target_id,
                "voice": selection.voice,
                "options": dict(selection.options),
            },
            token_fields=(),
            frontend="installed-smoke",
            pronunciation="installed-smoke",
        )

    @contextmanager
    def open(self, selection: EngineSelection) -> Iterator[EngineSession]:
        del selection
        yield self.session


def _compose_installed_document(root: Path, profile: RenderProfile) -> DocumentComposeResult:
    text = "Installed VoiceRender document-workspace smoke."
    workspace = root / "document-workspace"
    utterplan_dir = workspace / "utterplan"
    utterplan_dir.mkdir(parents=True)
    (workspace / "document.ssmd.md").write_text(text + "\n", encoding="utf-8")
    (workspace / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.document",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "document": {"id": "document-0001", "path": "document.ssmd.md"},
            }
        ),
        encoding="utf-8",
    )

    source_plan = utterplan.FlowPlan(language="en-US", flow=())
    source_path = utterplan_dir / "document.utterplan.toml"
    source_plan.save(source_path)
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"document-smoke--{fingerprint}"
    variant_dir = workspace / "renderplan" / variant
    render_dir = workspace / "voicerender" / variant / "document-0001"
    variant_dir.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")

    speech = SpeechItem(
        number=1,
        text=text,
        language="en-US",
        role="narrator",
        binding={
            "renderer": ENGINE_ID,
            "target_id": TARGET_ID,
            "voice": "smoke-voice",
        },
        controls=(SpeechControl(0, len(text), "emphasis", {"level": "moderate"}),),
        source=(SourceSpan("document.ssmd.md", 0, len(text)),),
    )
    plan = RenderPlan(
        source_plan_id=source_plan.plan_id,
        render_profile_id=profile_id,
        flow=(speech,),
    ).with_identity()
    save_plan(plan, variant_dir / "document.renderplan.toml")
    preflight = preflight_render_plan(
        plan,
        workspace / "preflight.toml",
        emphasis_policy="defer-to-compose",
    )
    assert preflight.status == "ok", preflight.to_dict()
    rendered = render_renderplan(
        plan,
        preflight=preflight,
        output_dir=render_dir,
        cache_dir=workspace / "cache",
        emphasis_policy="defer-to-compose",
    )
    assert not rendered.failed, rendered.error
    assert rendered.manifest.complete
    assert _version_tuple(rendered.manifest.producer.version) >= (0, 2, 4)

    result = compose_document(workspace, variant=variant)
    assert isinstance(result, DocumentComposeResult)
    assert result.status == "written"
    assert result.output.is_file()
    assert result.audiojob.is_dir()
    assert result.report_path.is_file()
    load_job(result.audiojob).validate(verify_sources=True)
    return result


def main() -> None:
    assert Path(sys.prefix).resolve() in Path(__import__("audiocompose").__file__).resolve().parents
    assert Path(sys.prefix).resolve() in Path(voicerender.__file__).resolve().parents
    assert Path(sys.prefix).resolve() in Path(utterplan.__file__).resolve().parents
    assert _version_tuple(version("renderplan")) >= (0, 1, 2)
    assert (0, 6, 4) <= _version_tuple(version("utterplan")) < (0, 7, 0)
    assert _version_tuple(version("voicerender")) >= (0, 2, 4)
    assert _version_tuple(version("voicerender")) < (0, 3, 0)

    register_adapter(_Adapter())
    root = Path(tempfile.mkdtemp(prefix="audiocompose-voicerender-smoke-"))
    utterplan_dir = root / "utterplan"
    chapters_dir = root / "chapters"
    utterplan_dir.mkdir()
    chapters_dir.mkdir()
    (chapters_dir / "chapter-0001.md").write_text("installed smoke\n", encoding="utf-8")

    profile = RenderProfile(
        roles={
            "narrator": RoleProfile(
                language="en-US",
                binding={
                    "renderer": ENGINE_ID,
                    "target_id": TARGET_ID,
                    "voice": "smoke-voice",
                },
            )
        },
        default_role="narrator",
    ).with_identity()
    profile_id = profile.profile_id or profile.computed_id()
    fingerprint = profile_id.removeprefix("sha256:")[:12]
    variant = f"smoke--{fingerprint}"
    variant_dir = root / "renderplan" / variant
    render_dir = root / "voicerender" / variant / "chapter-0001"
    variant_dir.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    save_profile(profile, variant_dir / "renderprofile.toml")

    source_plan = utterplan.FlowPlan(language="en-US", flow=())
    source_path = utterplan_dir / "chapter-0001.utterplan.toml"
    source_plan.save(source_path)
    text = "Installed VoiceRender handoff smoke."
    speech = SpeechItem(
        number=1,
        text=text,
        language="en-US",
        role="narrator",
        binding={
            "renderer": ENGINE_ID,
            "target_id": TARGET_ID,
            "voice": "smoke-voice",
        },
        controls=(
            SpeechControl(0, len(text), "emphasis", {"level": "moderate"}),
            SpeechControl(
                0,
                len(text),
                "pronunciation",
                {"phonemes": "smoke", "alphabet": "ipa", "language": "en-US"},
            ),
        ),
        source=(SourceSpan("chapter-0001.md", 0, len(text)),),
    )
    plan = RenderPlan(
        source_plan_id=source_plan.plan_id,
        render_profile_id=profile_id,
        flow=(speech,),
    ).with_identity()
    plan_path = variant_dir / "chapter-0001.renderplan.toml"
    save_plan(plan, plan_path)

    preflight = preflight_render_plan(
        plan,
        root / "preflight.toml",
        emphasis_policy="defer-to-compose",
    )
    assert preflight.status == "ok", preflight.to_dict()
    rendered = render_renderplan(
        plan,
        preflight=preflight,
        output_dir=render_dir,
        cache_dir=root / "cache",
        emphasis_policy="defer-to-compose",
    )
    assert not rendered.failed, rendered.error
    manifest = load_manifest(rendered.manifest_path)
    assert manifest.complete and manifest.producer.name == "voicerender"
    assert _version_tuple(manifest.producer.version) >= (0, 2, 4)
    diagnostics_path = render_dir / RENDER_DIAGNOSTICS_FILENAME
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["pending_composition"] is True
    assert diagnostics["deferred_controls"][0]["capability"] == EMPHASIS_COMPOSITION_CAPABILITY

    chapter = RenderPlanSetChapter(
        id="chapter-0001",
        source_number=1,
        source="../../chapters/chapter-0001.md",
        utterplan="../../utterplan/chapter-0001.utterplan.toml",
        source_plan_id=source_plan.plan_id,
        renderplan=plan_path.name,
        render_plan_id=plan.plan_id or plan.computed_id(),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ssmdconvert.book",
                "schema_version": 1,
                "ssmd_version": "0.9",
                "chapters": [
                    {
                        "id": chapter.id,
                        "source_number": chapter.source_number,
                        "title": "Installed Smoke",
                        "path": "chapters/chapter-0001.md",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    renderplan_set = RenderPlanSet(
        variant="smoke",
        profile_fingerprint=fingerprint,
        producer_version=version("voicerender"),
        manifest="../../manifest.json",
        utterplan_dir="../../utterplan",
        profile_id=profile_id,
        snapshot="renderprofile.toml",
        source_profile="profile.toml",
        chapters=(chapter,),
    )
    save_renderplan_set(variant_dir / "renderplan-set.toml", renderplan_set)
    loaded_set = load_renderplan_set(variant_dir / "renderplan-set.toml")
    loaded_plan = load_plan(plan_path)
    loaded_manifest = load_manifest(rendered.manifest_path)
    job = materialize_voicerender_job(
        source_path,
        loaded_plan,
        loaded_manifest,
        render_dir=render_dir,
    )
    assert loaded_set.chapters[0].render_plan_id == job.source["renderplan.plan_id"]
    assert job.source[VOICERENDER_DEFERRED_CAPABILITIES] == [EMPHASIS_COMPOSITION_CAPABILITY]
    assert job.source[VOICERENDER_DIAGNOSTICS_SHA256].startswith("sha256:")
    job.validate(verify_sources=True)

    result = compose_book(root, variant=variant)
    assert isinstance(result, BookComposeResult)
    assert result.requested == result.written == 1
    assert result.exit_code == 0
    assert result.report_path.is_file()
    assert (result.output_variant_dir / "chapter-0001.wav").is_file()
    saved_job = load_job(result.output_variant_dir / "chapter-0001.audiojob")
    assert (
        saved_job.source[VOICERENDER_DIAGNOSTICS_SHA256]
        == job.source[VOICERENDER_DIAGNOSTICS_SHA256]
    )
    assert (
        result.outcomes[0].render_diagnostics_sha256 == job.source[VOICERENDER_DIAGNOSTICS_SHA256]
    )
    document_result = _compose_installed_document(root, profile)
    assert document_result.document_id == "document-0001"
    print("installed VoiceRender v0.2.4 artifact -> AudioCompose book and document smoke passed")


if __name__ == "__main__":
    main()
