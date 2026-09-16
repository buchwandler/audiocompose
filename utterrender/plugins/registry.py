from __future__ import annotations

from importlib import metadata
from typing import Iterable

from ..errors import RenderPluginError
from ..voices import VoiceInfo
from .base import RenderPlugin


class PluginRegistry:
    def __init__(self, plugins: Iterable[RenderPlugin] = ()) -> None:
        self._plugins: dict[str, RenderPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: RenderPlugin) -> None:
        if not plugin.id:
            raise ValueError("plugin id must not be empty")
        if plugin.id in self._plugins:
            raise ValueError(f"plugin {plugin.id!r} is already registered")
        self._plugins[plugin.id] = plugin

    def get(self, plugin_id: str) -> RenderPlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise RenderPluginError(f"unknown render plugin: {plugin_id}") from exc

    def plugin_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def voices(self, *, language: str | None = None) -> tuple[VoiceInfo, ...]:
        result: list[VoiceInfo] = []
        for plugin in self._plugins.values():
            try:
                result.extend(plugin.voices(language=language))
            except ModuleNotFoundError:
                continue
        return tuple(sorted(result, key=lambda item: item.id))

    def voice(self, voice_id: str) -> VoiceInfo:
        if ":" not in voice_id:
            raise RenderPluginError(
                f"voice id must be namespaced as '<plugin>:<voice>', got {voice_id!r}"
            )
        plugin_id, _ = voice_id.split(":", 1)
        plugin = self.get(plugin_id)
        for voice in plugin.voices():
            if voice.id == voice_id:
                return voice
        raise RenderPluginError(f"unknown voice: {voice_id}")

    def load_entry_points(self) -> None:
        points = metadata.entry_points()
        selected = points.select(group="utterrender.plugins") if hasattr(points, "select") else points.get("utterrender.plugins", ())
        for point in selected:
            factory = point.load()
            plugin = factory() if callable(factory) else factory
            self.register(plugin)

    def close(self) -> None:
        for plugin in self._plugins.values():
            plugin.close()
