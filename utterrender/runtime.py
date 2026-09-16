from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._plan import Plan, Planner, PlannerConfig

from .assembler import assemble
from .calibration import VoiceCalibrationRegistry
from .effects import AudioSigProsodyProcessor
from .model import AudioFragment, RenderResult
from .plugins import KokoroPlugin, PiperPlugin, PluginRegistry, RenderPlugin, RenderRequest
from .prosody import resolve_prosody
from .resampling import resample_fragment
from .routing import VoiceRouter
from .voices import VoiceBindings, VoiceInfo
from .wav import write_wav


class Renderer:
    """Execute an existing :class:`utterplan.Plan` through render plugins."""

    def __init__(
        self,
        *,
        plugins: Sequence[RenderPlugin] | None = None,
        bindings: Mapping[str, str] | None = None,
        default_voice: str | None = None,
        sample_rate: int | None = None,
        auto_language: bool = True,
        apply_prosody: bool = True,
        load_entry_points: bool = False,
        calibration: Mapping[str, float] | None = None,
    ) -> None:
        if plugins is None:
            plugins = (KokoroPlugin(), PiperPlugin())
        self.registry = PluginRegistry(plugins)
        if load_entry_points:
            self.registry.load_entry_points()
        self.bindings = VoiceBindings(bindings)
        self.router = VoiceRouter(
            registry=self.registry,
            bindings=self.bindings,
            default_voice=default_voice,
            auto_language=auto_language,
        )
        self.sample_rate = sample_rate
        self.apply_prosody = apply_prosody
        self.calibration = VoiceCalibrationRegistry(calibration)

    def bind(self, logical_voice: str, concrete_voice: str) -> None:
        self.bindings.bind(logical_voice, concrete_voice)

    def voices(self, *, language: str | None = None) -> tuple[VoiceInfo, ...]:
        return self.registry.voices(language=language)

    def ensure_voice(self, voice_id: str) -> VoiceInfo:
        voice = self.registry.voice(voice_id)
        self.registry.get(voice.plugin).ensure_voice(voice)
        return voice

    def calibrate_voice(self, voice_id: str, gain_db: float) -> None:
        self.calibration.set(voice_id, gain_db)

    def render(self, plan: Plan, *, segment_options: Mapping[str, Mapping[str, Any]] | None = None) -> RenderResult:
        plan.validate()
        options = segment_options or {}
        fragments: list[AudioFragment] = []
        for segment in plan.segments:
            voice = self.router.resolve(segment)
            plugin = self.registry.get(voice.plugin)
            prosody = resolve_prosody(segment.directives.prosody)
            request = RenderRequest(
                segment=segment,
                voice=voice,
                prosody=prosody,
                options=dict(options.get(segment.id, {})),
            )
            fragment = plugin.render(request)
            fragment = self.calibration.apply(fragment, voice.id)
            fragments.append(fragment)

        output_rate = self.sample_rate
        if output_rate is None and fragments:
            # Prefer the first routed fragment as the timeline rate; all later
            # fragments are normalized to it.
            output_rate = fragments[0].sample_rate
        if output_rate is not None:
            fragments = [resample_fragment(fragment, output_rate) for fragment in fragments]

        processors = (AudioSigProsodyProcessor(),) if self.apply_prosody else ()
        return assemble(plan, fragments, processors=processors, sample_rate=output_rate)

    def to_wav(self, plan: Plan, output: str | Path, **kwargs: Any) -> Path:
        result = self.render(plan, **kwargs)
        return write_wav(output, result.audio, result.sample_rate)

    def close(self) -> None:
        self.registry.close()

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class TTS:
    """Single high-level facade: text -> utterplan -> utterrender -> audio."""

    def __init__(self, *, renderer: Renderer | None = None, planner_config: PlannerConfig | None = None, **renderer_kwargs: Any) -> None:
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

    def to_wav(self, text: str, output: str | Path, *, language: str, input_format: str = "plain") -> Path:
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

    def __enter__(self) -> "TTS":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
