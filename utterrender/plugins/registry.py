from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal

from ..assets import AssetProgressCallback
from ..errors import RenderPluginError
from ..models import ModelInfo
from ..voices import VoiceInfo
from .base import RenderPlugin

PluginAvailability = Literal["ready", "not_installed", "broken", "unavailable"]


@dataclass(frozen=True, slots=True)
class PluginStatus:
    plugin: str
    status: PluginAvailability
    error: str | None = None


class PluginRegistry:
    def __init__(self, plugins: Iterable[RenderPlugin] = ()) -> None:
        self._plugins: dict[str, RenderPlugin] = {}
        self._models_cache: dict[str | None, tuple[ModelInfo, ...]] = {}
        self._voices_cache: dict[tuple[str, str | None, str | None], tuple[VoiceInfo, ...]] = {}
        self._voice_index: dict[str, VoiceInfo] = {}
        self._statuses: dict[str, PluginStatus] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: RenderPlugin) -> None:
        if not plugin.id:
            raise ValueError("plugin id must not be empty")
        if plugin.id in self._plugins:
            raise ValueError(f"plugin {plugin.id!r} is already registered")
        self._plugins[plugin.id] = plugin
        self._statuses[plugin.id] = PluginStatus(plugin.id, "ready")

    def get(self, plugin_id: str) -> RenderPlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise RenderPluginError(f"unknown render plugin: {plugin_id}") from exc

    def plugin_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def status(self, plugin_id: str) -> PluginStatus:
        self.get(plugin_id)
        return self._statuses[plugin_id]

    def statuses(self) -> tuple[PluginStatus, ...]:
        return tuple(self._statuses[plugin_id] for plugin_id in self.plugin_ids())

    def _record_import_failure(self, plugin: RenderPlugin, exc: ModuleNotFoundError) -> None:
        dependency = getattr(plugin, "dependency_name", None)
        missing = exc.name or "unknown dependency"
        if dependency and missing.split(".", 1)[0] == dependency.split(".", 1)[0]:
            status: PluginAvailability = "not_installed"
            message = f"plugin {plugin.id!r} dependency {dependency!r} is not installed"
        else:
            status = "broken"
            message = f"plugin {plugin.id!r} is missing dependency {missing!r}"
        self._statuses[plugin.id] = PluginStatus(plugin.id, status, message)
        raise RenderPluginError(message) from exc

    def models(self, *, language: str | None = None) -> tuple[ModelInfo, ...]:
        cached = self._models_cache.get(language)
        if cached is not None:
            return cached
        result: list[ModelInfo] = []
        for plugin in self._plugins.values():
            loader = getattr(plugin, "models", None)
            if loader is None:
                continue
            parameters = inspect.signature(loader).parameters
            kwargs = {"language": language} if "language" in parameters else {}
            try:
                result.extend(loader(**kwargs))
            except ModuleNotFoundError as exc:
                self._record_import_failure(plugin, exc)
        values = tuple(sorted(result, key=lambda item: item.id))
        self._models_cache[language] = values
        return values


    def _load_voices(
        self,
        plugin: RenderPlugin,
        *,
        language: str | None,
        model: str | None,
    ) -> tuple[VoiceInfo, ...]:
        kwargs: dict[str, Any] = {}
        parameters = inspect.signature(plugin.voices).parameters
        if "language" in parameters:
            kwargs["language"] = language
        if "model" in parameters:
            kwargs["model"] = model
        try:
            values = tuple(plugin.voices(**kwargs))
        except ModuleNotFoundError as exc:
            self._record_import_failure(plugin, exc)
        self._statuses[plugin.id] = PluginStatus(plugin.id, "ready")
        return values

    def voices(
        self,
        *,
        language: str | None = None,
        model: str | None = None,
    ) -> tuple[VoiceInfo, ...]:
        result: list[VoiceInfo] = []
        for plugin in self._plugins.values():
            key = (plugin.id, language, model)
            voices = self._voices_cache.get(key)
            if voices is None:
                voices = self._load_voices(plugin, language=language, model=model)
                self._voices_cache[key] = voices
                for voice in voices:
                    self._voice_index[voice.id] = voice
            result.extend(voices)
        return tuple(sorted(result, key=lambda item: item.id))

    def voice(self, voice_id: str) -> VoiceInfo:
        if ":" not in voice_id:
            raise RenderPluginError(
                f"voice id must be namespaced as '<plugin>:<voice>', got {voice_id!r}"
            )
        cached = self._voice_index.get(voice_id)
        if cached is not None:
            return cached
        plugin_id, _ = voice_id.split(":", 1)
        self.get(plugin_id)
        for voice in self.voices():
            if voice.id == voice_id:
                return voice
        raise RenderPluginError(f"unknown voice: {voice_id}")

    def refresh(self, plugin_id: str | None = None) -> None:
        if plugin_id is not None:
            self.get(plugin_id)
        self._models_cache.clear()
        if plugin_id is None:
            self._voices_cache.clear()
            self._voice_index.clear()
        else:
            self._voices_cache = {
                key: values
                for key, values in self._voices_cache.items()
                if key[0] != plugin_id
            }
            self._voice_index = {
                voice_id: voice
                for voice_id, voice in self._voice_index.items()
                if voice.plugin != plugin_id
            }

    def ensure_voice(
        self, voice: VoiceInfo, *, progress: AssetProgressCallback | None = None
    ) -> None:
        plugin = self.get(voice.plugin)
        parameters = inspect.signature(plugin.ensure_voice).parameters
        kwargs = {"progress": progress} if "progress" in parameters and progress is not None else {}
        try:
            plugin.ensure_voice(voice, **kwargs)
        except ModuleNotFoundError as exc:
            self._record_import_failure(plugin, exc)
        self.refresh(plugin.id)


    def load_entry_points(self) -> None:
        selected = metadata.entry_points(group="utterrender.plugins")
        for point in selected:
            factory = point.load()
            plugin = factory() if callable(factory) else factory
            self.register(plugin)

    def cache_info(self) -> dict[str, int]:
        return {
            "models": len(self._models_cache),
            "voice_queries": len(self._voices_cache),
            "voices": len(self._voice_index),
        }

    def close(self) -> None:
        failures: list[str] = []
        for plugin in self._plugins.values():
            try:
                plugin.close()
            except Exception as exc:
                failures.append(f"{plugin.id}: {exc}")
        if failures:
            raise RenderPluginError("plugin close failed: " + "; ".join(failures))
