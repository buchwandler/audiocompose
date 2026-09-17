from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import AssetProgressCallback, AssetProgressEvent
from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..models import ModelInfo
from ..voices import VoiceInfo
from .base import RenderRequest


class _LoadedPiperVoice:
    def __init__(self, session: Any, config: Any, frontend: Any) -> None:
        self.session = session
        self.config = config
        self.frontend = frontend
        self.closed = False

    def resolve_speaker(self, value: str | int | None) -> int | None:
        if value is None:
            return getattr(self.config, "default_speaker_id", None)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("speaker must be a name or integer ID")
        if isinstance(value, str):
            try:
                value = self.config.speaker_id_map[value]
            except KeyError as exc:
                raise ValueError(f"unknown speaker {value!r}") from exc
        if not isinstance(value, int):
            raise ValueError("speaker ID must be an integer")
        count = int(getattr(self.config, "num_speakers", 1))
        if value < 0 or value >= count:
            raise ValueError(f"speaker ID {value} is outside 0..{count - 1}")
        return None if count == 1 else value

    def synthesize_ids(self, ids: Sequence[int], *, request: RenderRequest) -> np.ndarray:
        if not ids:
            return np.zeros(0, dtype=np.float32)
        speaker = self.resolve_speaker(request.speaker)
        input_ids = np.asarray([list(ids)], dtype=np.int64)
        scales = np.asarray(
            [
                request.options.get("noise_scale", getattr(self.config, "noise_scale", 0.667)),
                request.options.get(
                    "length_scale",
                    getattr(self.config, "length_scale", 1.0)
                    / request.prosody.rate,
                ),
                request.options.get(
                    "noise_w_scale",
                    getattr(self.config, "noise_w_scale", 0.8),
                ),
            ],
            dtype=np.float32,
        )
        arguments: dict[str, np.ndarray] = {
            "input": input_ids,
            "input_lengths": np.asarray([input_ids.shape[1]], dtype=np.int64),
            "scales": scales,
        }
        if int(getattr(self.config, "num_speakers", 1)) > 1:
            arguments["sid"] = np.asarray([speaker], dtype=np.int64)
        try:
            result = self.session.run(None, arguments)
        except Exception as exc:
            raise RuntimeError("Piper ONNX inference failed") from exc
        if not result:
            raise RuntimeError("Piper ONNX model returned no output")
        audio = np.asarray(result[0], dtype=np.float32).squeeze()
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise RuntimeError(f"Piper model returned invalid waveform shape {audio.shape}")
        return audio

    def close(self) -> None:
        if self.closed:
            return
        close = getattr(self.frontend, "close", None)
        if close is not None:
            close()
        close = getattr(self.session, "close", None)
        if close is not None:
            close()
        self.closed = True


class PiperPlugin:
    """Direct Piper-compatible backend for prepared utterplan segments."""

    id = "piper"
    dependency_name = "piperg2p"

    def __init__(
        self,
        *,
        cache_dir: str | None = None,
        offline: bool | None = None,
        provider: str = "auto",
        provider_options: Mapping[str, Any] | None = None,
        voice_paths: Mapping[str, str | Path | tuple[str | Path, str | Path]] | None = None,
        voice_loader: Callable[[VoiceInfo], Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.offline = offline
        self.provider = provider
        self.provider_options = dict(provider_options or {})
        self.voice_paths = dict(voice_paths or {})
        self.voice_loader = voice_loader
        self._voices: dict[str, Any] = {}

    def _module(self) -> Any:
        import piperg2p

        return piperg2p

    def models(self, *, language: str | None = None):
        del language
        for voice_id, value in self.voice_paths.items():
            model_path = value[0] if isinstance(value, tuple) else value
            yield ModelInfo(
                id=f"piper:{voice_id}",
                plugin=self.id,
                model_id=voice_id,
                languages=(),
                installed=Path(model_path).exists(),
            )

    def voices(self, *, language: str | None = None, model: str | None = None):
        for voice_id, value in self.voice_paths.items():
            if model is not None and model not in {voice_id, f"piper:{voice_id}"}:
                continue
            model_path, config_path = self._paths(value)
            info = self._voice_info(voice_id, model_path, config_path)
            if language is None or info.supports_language(language):
                yield info

    def _paths(
        self,
        value: str | Path | tuple[str | Path, str | Path],
    ) -> tuple[Path, Path]:
        if isinstance(value, tuple):
            model_path, config_path = value
            return Path(model_path), Path(config_path)
        model_path = Path(value)
        return model_path, Path(f"{model_path}.json")

    def _voice_info(self, voice_id: str, model_path: Path, config_path: Path) -> VoiceInfo:
        languages: tuple[str, ...] = ()
        speakers: tuple[str, ...] = ()
        speaker_id_map: Mapping[str, int] = {}
        quality = None
        installed = model_path.exists() and config_path.exists()
        if config_path.exists():
            config = self._module().VoiceConfig.from_json(config_path)
            language = getattr(config, "espeak_voice", None)
            languages = (str(language),) if language else ()
            speaker_id_map = dict(getattr(config, "speaker_id_map", {}))
            speakers = tuple(speaker_id_map)
            quality = voice_id.rsplit("-", 1)[-1]
        return VoiceInfo(
            id=f"piper:{voice_id}",
            plugin=self.id,
            voice_id=voice_id,
            languages=languages,
            model_id=voice_id,
            sample_rate=(int(config.sample_rate) if config_path.exists() else None),
            installed=installed,
            downloadable=not bool(self.offline),
            quality=quality,
            speakers=speakers,
            speaker_id_map=speaker_id_map,
            metadata={"model_path": str(model_path), "config_path": str(config_path)},
        )

    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities:
        return RenderCapabilities(native_rate=True, multi_speaker=bool(voice.speakers))

    def ensure_voice(
        self,
        voice: VoiceInfo,
        *,
        progress: AssetProgressCallback | None = None,
    ) -> None:
        if progress is not None:
            progress(AssetProgressEvent(self.id, "ensure_voice", voice.id, 0, 1))
        self._get_voice(voice)
        if progress is not None:
            progress(AssetProgressEvent(self.id, "ensure_voice", voice.id, 1, 1))

    def _get_voice(self, voice: VoiceInfo) -> Any:
        cached = self._voices.get(voice.id)
        if cached is not None:
            return cached
        if self.voice_loader is not None:
            loaded = self.voice_loader(voice)
            self._voices[voice.id] = loaded
            return loaded
        value = self.voice_paths.get(voice.voice_id)
        if value is None:
            raise FileNotFoundError(f"no local Piper model is configured for {voice.voice_id!r}")
        model_path, config_path = self._paths(value)
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model file does not exist: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Piper voice config does not exist: {config_path}")
        mod = self._module()
        config = mod.VoiceConfig.from_json(config_path)
        frontend = mod.PiperFrontend(config)
        try:
            import onnxruntime

            providers = None if self.provider == "auto" else [self.provider]
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=providers,
                provider_options=[self.provider_options] if providers else None,
            )
        except Exception:
            frontend.close()
            raise
        loaded = _LoadedPiperVoice(session, config, frontend)
        self._voices[voice.id] = loaded
        return loaded

    def _phoneme_ids(self, request: RenderRequest, voice: Any) -> tuple[int, ...] | None:
        value = request.pronunciation
        if value is None:
            return None
        if isinstance(value, Mapping):
            value = value.get("phoneme_ids", value.get("value", value.get("phonemes")))
        else:
            alphabet = getattr(value, "alphabet", "ipa")
            phonemes = getattr(value, "phonemes", None)
            if phonemes is not None:
                value = phonemes
            if alphabet not in {None, "ipa", "espeak"}:
                raise ValueError(f"unsupported Piper pronunciation alphabet {alphabet!r}")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if all(isinstance(item, (int, np.integer)) for item in value):
                return tuple(int(item) for item in value)
            value = " ".join(str(item) for item in value)
        if not isinstance(value, str):
            raise ValueError("Piper pronunciation override must contain phoneme IDs or text")
        return tuple(voice.frontend.encode(value.split()).ids)

    def render(self, request: RenderRequest) -> AudioFragment:
        voice = self._get_voice(request.voice)
        direct_ids = self._phoneme_ids(request, voice)
        if direct_ids is not None:
            audio = voice.synthesize_ids(direct_ids, request=request)
            warnings: tuple[str, ...] = ()
        else:
            annotations: tuple[Any, ...] = ()
            if request.context is not None:
                annotations = tuple(
                    {
                        "start": token.spoken_start - request.segment.spoken_start,
                        "end": token.spoken_end - request.segment.spoken_start,
                        "text": token.text,
                        "lemma": getattr(token, "lemma", None),
                        "language": token.language,
                    }
                    for token in request.context.tokens
                )
            parameters = inspect.signature(voice.frontend.phonemize_prepared).parameters
            result = (
                voice.frontend.phonemize_prepared(request.segment.text, annotations=annotations)
                if "annotations" in parameters
                else voice.frontend.phonemize_prepared(request.segment.text)
            )
            chunks = [
                voice.synthesize_ids(sentence.ids, request=request)
                for sentence in result.sentences
                if sentence.ids
            ]
            audio = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros(0, dtype=np.float32)
            warnings = tuple(getattr(result, "warnings", ()))
        return AudioFragment(
            segment_id=request.segment.id,
            audio=audio,
            sample_rate=int(voice.config.sample_rate),
            realized_prosody=frozenset({"rate"}) if "rate" in request.prosody.requested else frozenset(),
            metadata={"plugin": self.id, "voice": request.voice.id, "warnings": warnings},
        )

    def close(self) -> None:
        for voice in self._voices.values():
            close = getattr(voice, "close", None)
            if close is not None:
                close()
        self._voices.clear()
