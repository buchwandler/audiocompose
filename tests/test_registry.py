from __future__ import annotations

from utterrender import RenderPluginError, VoiceBindings, VoiceInfo
from utterrender.plugins.registry import PluginRegistry


class CatalogPlugin:
    id = "catalog"
    dependency_name = "catalog-runtime"

    def __init__(self) -> None:
        self.voice_calls = 0
        self.ensure_calls = 0
        self.closed = False

    def voices(self, *, language: str | None = None, model: str | None = None):
        self.voice_calls += 1
        del model
        values = (VoiceInfo("catalog:one", "catalog", "one", ("en-us",)),)
        if language is None:
            return values
        return tuple(value for value in values if value.supports_language(language))

    def ensure_voice(self, voice: VoiceInfo, *, progress=None) -> None:
        self.ensure_calls += 1
        if progress is not None:
            progress("ready")
        assert voice.id == "catalog:one"

    def close(self) -> None:
        self.closed = True


def test_voice_language_matching_handles_region_forms() -> None:
    voice = VoiceInfo("x:v", "x", "v", ("de-de",))
    assert voice.supports_language("de-DE")
    assert voice.supports_language("de")
    assert not voice.supports_language("en-us")


def test_voice_bindings() -> None:
    bindings = VoiceBindings({"narrator": "kokoro:af_heart"})
    assert bindings.resolve("narrator") == "kokoro:af_heart"
    bindings.bind("quote", "piper:en_US-lessac-medium")
    assert bindings.to_dict()["quote"].startswith("piper:")


def test_registry_caches_voice_inventory_and_refreshes() -> None:
    plugin = CatalogPlugin()
    registry = PluginRegistry((plugin,))

    assert registry.voices() == registry.voices()
    assert plugin.voice_calls == 1
    assert registry.voice("catalog:one").id == "catalog:one"
    assert registry.cache_info()["voices"] == 1

    registry.refresh("catalog")
    assert registry.voices()[0].id == "catalog:one"
    assert plugin.voice_calls == 2


def test_registry_passes_progress_and_closes_plugins() -> None:
    plugin = CatalogPlugin()
    registry = PluginRegistry((plugin,))
    events: list[str] = []

    registry.ensure_voice(registry.voice("catalog:one"), progress=events.append)
    registry.close()

    assert plugin.ensure_calls == 1
    assert events == ["ready"]
    assert plugin.closed


def test_registry_distinguishes_missing_plugin_dependency() -> None:
    class MissingPlugin(CatalogPlugin):
        id = "missing"

        def voices(self, *, language: str | None = None, model: str | None = None):
            del language, model
            raise ModuleNotFoundError("catalog-runtime", name="catalog-runtime")

    plugin = MissingPlugin()
    registry = PluginRegistry((plugin,))
    try:
        registry.voices()
    except RenderPluginError as exc:
        assert "not installed" in str(exc)
    else:
        raise AssertionError("missing plugin dependency was ignored")
    assert registry.status("missing").status == "not_installed"
