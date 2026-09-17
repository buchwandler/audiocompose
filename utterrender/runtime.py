from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from ._plan import Plan, Planner, PlannerConfig
from .assembler import assemble
from .assets import AssetProgressCallback
from .calibration import VoiceCalibrationRegistry
from .capabilities import RenderCapabilities
from .effects import AudioSigProsodyProcessor, EmphasisPolicy, apply_emphasis
from .errors import RenderPluginError
from .loudness import LoudnessPolicy, apply_complete_output_loudness
from .model import AudioFragment, RenderResult
from .plugins import (
    KokoroPlugin,
    PiperPlugin,
    PluginRegistry,
    RenderPlugin,
    RenderRequest,
    SegmentRenderContext,
)
from .prosody import resolve_prosody
from .resampling import resample_fragment
from .routing import UnboundVoicePolicy, VoiceRouter
from .voices import VoiceBindings, VoiceInfo
from .wav import ClipPolicy, write_wav


class Renderer:
    """Execute an existing :class:`utterplan.UtterancePlan`."""

    def __init__(
        self,
        *,
        plugins: Sequence[RenderPlugin] | None = None,
        bindings: Mapping[str, str] | None = None,
        default_voice: str | None = None,
        sample_rate: int | None = None,
        auto_language: bool = True,
        unbound_voice_policy: UnboundVoicePolicy = "error",
        preferred_plugins: Sequence[str] = (),
        apply_prosody: bool = True,
        load_entry_points: bool = False,
        calibration: Mapping[str, float] | None = None,
        loudness: LoudnessPolicy | None = None,
        strict_capabilities: bool = True,
        emphasis_policy: EmphasisPolicy = "gain",
        external_audio_resolver: Callable[[Any], AudioFragment | tuple[Any, int]] | None = None,
        external_audio_policy: Literal["error", "alt_text"] = "error",
    ) -> None:
        selected_plugins: Sequence[RenderPlugin] = (
            (KokoroPlugin(), PiperPlugin()) if plugins is None else plugins
        )
        self.registry = PluginRegistry(selected_plugins)
        if load_entry_points:
            self.registry.load_entry_points()
        self.bindings = VoiceBindings(bindings)
        self.router = VoiceRouter(
            registry=self.registry,
            bindings=self.bindings,
            default_voice=default_voice,
            auto_language=auto_language,
            unbound_voice_policy=unbound_voice_policy,
            preferred_plugins=tuple(preferred_plugins),
        )
        self.sample_rate = sample_rate
        self.apply_prosody = apply_prosody
        self.calibration = VoiceCalibrationRegistry(calibration)
        self.loudness = loudness
        self.strict_capabilities = strict_capabilities
        self.emphasis_policy = emphasis_policy
        self.external_audio_resolver = external_audio_resolver
        self.external_audio_policy = external_audio_policy

    def bind(self, logical_voice: str, concrete_voice: str) -> None:
        self.bindings.bind(logical_voice, concrete_voice)

    def plugins(self) -> tuple[str, ...]:
        return self.registry.plugin_ids()

    def models(self, *, language: str | None = None):
        return self.registry.models(language=language)

    def voices(self, *, language: str | None = None, model: str | None = None) -> tuple[VoiceInfo, ...]:
        return self.registry.voices(language=language, model=model)

    def capabilities(self, voice_id: str) -> RenderCapabilities:
        voice = self.registry.voice(voice_id)
        return self.registry.get(voice.plugin).capabilities(voice)

    def ensure_voice(
        self,
        voice_id: str,
        *,
        progress: AssetProgressCallback | None = None,
    ) -> VoiceInfo:
        voice = self.registry.voice(voice_id)
        self.registry.ensure_voice(voice, progress=progress)
        return voice

    def cache_info(self) -> dict[str, int]:
        return self.registry.cache_info()

    def providers(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for plugin_id in self.registry.plugin_ids():
            plugin = self.registry.get(plugin_id)
            provider_info = getattr(plugin, "providers", None)
            result[plugin_id] = provider_info() if callable(provider_info) else getattr(plugin, "provider", None)
        return result

    def refresh(self, plugin_id: str | None = None) -> None:
        self.registry.refresh(plugin_id)

    def calibrate_voice(self, voice_id: str, gain_db: float) -> None:
        self.calibration.set(voice_id, gain_db)

    def _validate_fragment_contract(
        self,
        fragment: AudioFragment,
        capabilities: RenderCapabilities,
    ) -> None:
        invalid_axes = fragment.realized_prosody - capabilities.native_prosody_axes
        if invalid_axes:
            axes = ", ".join(sorted(invalid_axes))
            raise RenderPluginError(f"plugin realized unsupported prosody axes: {axes}")
        if self.strict_capabilities:
            if capabilities.word_timing and not any(
                span.kind == "word" for span in fragment.alignment
            ):
                raise RenderPluginError(
                    "plugin advertises word timing but returned no word alignment"
                )
            if capabilities.phoneme_timing and not any(
                span.kind == "phoneme" for span in fragment.alignment
            ):
                raise RenderPluginError(
                    "plugin advertises phoneme timing but returned no phoneme alignment"
                )
        unsupported = {
            span.kind for span in fragment.alignment if not capabilities.supports_alignment(span.kind)
        }
        if unsupported:
            raise RenderPluginError(
                "plugin returned unsupported alignment kinds: " + ", ".join(sorted(unsupported))
            )

    def render(
        self,
        plan: Plan,
        *,
        segment_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RenderResult:
        plan.validate()
        options = segment_options or {}
        fragments: list[AudioFragment] = []
        for segment in plan.segments:
            voice = self.router.resolve(segment)
            plugin = self.registry.get(voice.plugin)
            prosody = resolve_prosody(segment.directives.prosody)
            request_options = dict(options.get(segment.id, {}))
            directives = segment.directives
            request = RenderRequest(
                segment=segment,
                voice=voice,
                prosody=prosody,
                options=request_options,
                context=SegmentRenderContext.from_plan(plan, segment),
                speaker=request_options.get("speaker", request_options.get("speaker_id")),
                pronunciation=getattr(directives, "pronunciation", None),
                emphasis=getattr(directives, "emphasis", None),
                audio_directive=getattr(directives, "audio", None),
            )
            try:
                if request.audio_directive is not None and self.external_audio_resolver is not None:
                    resolved = self.external_audio_resolver(request.audio_directive)
                    if isinstance(resolved, AudioFragment):
                        fragment = resolved
                    else:
                        audio, rate = resolved
                        fragment = AudioFragment(
                            segment_id=segment.id,
                            audio=audio,
                            sample_rate=int(rate),
                            metadata={"external_audio": request.audio_directive.src},
                        )
                elif request.audio_directive is not None and self.external_audio_policy == "error":
                    raise RenderPluginError(
                        f"external audio has no resolver: {request.audio_directive.src!r}"
                    )
                else:
                    fragment = plugin.render(request)
                emphasis = request.emphasis
                if emphasis is not None:
                    level = getattr(emphasis, "level", emphasis)
                    if isinstance(emphasis, Mapping):
                        level = emphasis.get("level", level)
                    emphasis_gain = {
                        "reduced": -3.0,
                        "moderate": 3.0,
                        "strong": 6.0,
                    }.get(str(level).lower(), 0.0)
                    fragment = apply_emphasis(
                        fragment, emphasis_gain, policy=self.emphasis_policy
                    )
                self._validate_fragment_contract(fragment, plugin.capabilities(voice))
            except RenderPluginError:
                raise
            except Exception as exc:
                raise RenderPluginError(
                    "render failed: "
                    f"plugin={plugin.id} voice={voice.id} segment={segment.id} "
                    f"language={segment.language} cause={exc}"
                ) from exc
            fragment = self.calibration.apply(
                fragment,
                voice.id,
                model_id=voice.model_identity,
            )
            fragments.append(fragment)

        output_rate = self.sample_rate
        if output_rate is None and fragments:
            output_rate = fragments[0].sample_rate
        if output_rate is not None:
            fragments = [resample_fragment(fragment, output_rate) for fragment in fragments]

        processors = (AudioSigProsodyProcessor(),) if self.apply_prosody else ()
        result = assemble(plan, fragments, processors=processors, sample_rate=output_rate, validate_plan=False)
        if self.loudness is not None:
            loudness_result = apply_complete_output_loudness(
                result.audio,
                result.sample_rate,
                self.loudness,
            )
            metadata = dict(result.metadata)
            metadata["utterrender.loudness"] = {
                "applied_gain_db": loudness_result.applied_gain_db,
                "measured_lufs": loudness_result.measured_lufs,
                "target_reached": loudness_result.target_reached,
                "true_peak_dbtp": loudness_result.true_peak_dbtp,
            }
            warnings = result.warnings + ((loudness_result.warning,) if loudness_result.warning else ())
            result = replace(result, audio=loudness_result.audio, metadata=metadata, warnings=warnings)
        return result

    def to_wav(self, plan: Plan, output: str | Path, **kwargs: Any) -> Path:
        clip_policy: ClipPolicy = kwargs.pop("clip_policy", "clamp")
        result = self.render(plan, **kwargs)
        return write_wav(output, result.audio, result.sample_rate, clip_policy=clip_policy)

    def close(self) -> None:
        self.registry.close()

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class TTS:
    """Single high-level facade: text -> utterplan -> utterrender -> audio."""

    def __init__(
        self,
        *,
        renderer: Renderer | None = None,
        planner_config: PlannerConfig | None = None,
        **renderer_kwargs: Any,
    ) -> None:
        self.renderer = renderer or Renderer(**renderer_kwargs)
        self.planner_config = planner_config

    def plan(self, text: str, *, language: str | None = None, input_format: str = "plain") -> Plan:
        if self.planner_config is not None:
            config = self.planner_config
        else:
            if not language:
                raise ValueError("language is required when no planner_config is supplied")
            config = PlannerConfig(language=language, document_format=input_format)
        return Planner(config).plan(text)

    def render_text(self, text: str, *, language: str, input_format: str = "plain") -> RenderResult:
        return self.renderer.render(self.plan(text, language=language, input_format=input_format))

    def to_wav(
        self,
        text: str,
        output: str | Path,
        *,
        language: str,
        input_format: str = "plain",
    ) -> Path:
        plan = self.plan(text, language=language, input_format=input_format)
        return self.renderer.to_wav(plan, output)

    def voices(self, *, language: str | None = None) -> tuple[VoiceInfo, ...]:
        return self.renderer.voices(language=language)

    def ensure_voice(self, voice_id: str) -> VoiceInfo:
        return self.renderer.ensure_voice(voice_id)

    def bind(self, logical_voice: str, concrete_voice: str) -> None:
        self.renderer.bind(logical_voice, concrete_voice)

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> TTS:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
