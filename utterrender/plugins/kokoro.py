from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..alignment import AudioTextSpan
from ..assets import AssetProgressCallback, AssetProgressEvent
from ..capabilities import RenderCapabilities
from ..model import AudioFragment
from ..models import ModelInfo
from ..voices import VoiceInfo
from .base import RenderRequest
from .kokoro_short_sentence import (
    ShortSentenceConfig,
    cut_linear,
    prepare_short_sentence,
)


class _LoadedKokoro:
    def __init__(self, session: Any, voices: Mapping[str, np.ndarray], sample_rate: int) -> None:
        self.session = session
        self.voices = dict(voices)
        self.sample_rate = sample_rate
        self.has_timing = any(
            str(getattr(output, "name", "")).lower() in {
                "pred_dur", "pred_duration", "durations", "duration",
            }
            for output in (session.get_outputs() if callable(getattr(session, "get_outputs", None)) else ())
        )
        self.closed = False

    def _voice_style(self, name: str, phoneme_count: int) -> np.ndarray:
        try:
            style = np.asarray(self.voices[name], dtype=np.float32)
        except KeyError as exc:
            raise ValueError(f"unknown Kokoro voice {name!r}") from exc
        if style.ndim == 1:
            style = style[None, :]
        if style.ndim == 2:
            index = min(max(phoneme_count - 1, 0), style.shape[0] - 1)
            style = style[index : index + 1, None, :]
        elif style.ndim == 3:
            index = min(max(phoneme_count - 1, 0), style.shape[0] - 1)
            style = style[index : index + 1]
        else:
            raise ValueError(f"invalid Kokoro voice style shape {style.shape}")
        return style

    def synthesize(
        self,
        ids: Sequence[int],
        *,
        voice: str,
        speed: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if not ids:
            return np.zeros(0, dtype=np.float32), None
        inputs = {item.name: item for item in self.session.get_inputs()}
        token_name = "input_ids" if "input_ids" in inputs else "tokens"
        style_name = "ref_s" if "ref_s" in inputs else "style"
        token_dtype = np.int64 if "int64" in inputs[token_name].type else np.int32
        style_dtype = np.float32
        arguments: dict[str, np.ndarray] = {
            token_name: np.asarray([[0, *ids, 0]], dtype=token_dtype),
            style_name: self._voice_style(voice, len(ids)).astype(style_dtype),
        }
        if "speed" in inputs:
            speed_dtype = np.int32 if "int32" in inputs["speed"].type else np.float32
            arguments["speed"] = np.asarray(
                [max(1, round(speed))] if np.issubdtype(speed_dtype, np.integer) else [speed],
                dtype=speed_dtype,
            )
        outputs = self.session.run(None, arguments)
        if not outputs:
            raise RuntimeError("Kokoro ONNX model returned no output")
        audio = np.asarray(outputs[0], dtype=np.float32).squeeze()
        if audio.ndim != 1 or not np.all(np.isfinite(audio)):
            raise RuntimeError(f"Kokoro model returned invalid waveform shape {audio.shape}")
        duration_output = None
        get_outputs = getattr(self.session, "get_outputs", None)
        if callable(get_outputs):
            output_names = get_outputs()
            for index, output in enumerate(output_names):
                if str(getattr(output, "name", "")).lower() in {
                    "pred_dur", "pred_duration", "durations", "duration",
                }:
                    if index < len(outputs):
                        duration_output = np.asarray(outputs[index]).reshape(-1)
                    break
        return audio, duration_output

    def close(self) -> None:
        if self.closed:
            return
        close = getattr(self.session, "close", None)
        if close is not None:
            close()
        self.closed = True


class KokoroPlugin:
    """Direct Kokoro G2P and ONNX backend for prepared utterplan segments."""

    id = "kokoro"
    dependency_name = "kokorog2p"

    def __init__(
        self,
        *,
        offline: bool = False,
        provider: str = "auto",
        provider_options: Mapping[str, Any] | None = None,
        model_paths: Mapping[str, str | Path | tuple[str | Path, str | Path]] | None = None,
        model_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        voice_loader: Callable[[VoiceInfo], Any] | None = None,
        asset_provider: Callable[[VoiceInfo], Any] | None = None,
        short_sentence_config: Any | None = None,
        sample_rate: int = 24000,
    ) -> None:
        self.offline = offline
        self.provider = provider
        self.provider_options = dict(provider_options or {})
        self.model_paths = dict(model_paths or {})
        self.model_profiles = dict(model_profiles or {})
        self.voice_loader = voice_loader or asset_provider
        self.short_sentence_config = short_sentence_config
        self.sample_rate = sample_rate
        self._models: dict[str, Any] = {}

    def _module(self) -> Any:
        import kokorog2p

        return kokorog2p

    def _paths(self, value: str | Path | tuple[str | Path, str | Path]) -> tuple[Path, Path]:
        if isinstance(value, tuple):
            return Path(value[0]), Path(value[1])
        path = Path(value)
        return path, path.with_name("voices.bin")
    def _model_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.model_profiles, *self.model_paths)))

    def models(self, *, language: str | None = None):
        for model_id in self._model_ids():
            profile = self.model_profiles.get(model_id, {})
            languages = tuple(str(item) for item in profile.get("languages", ()))
            if language is not None and languages and not any(
                language.lower().replace("_", "-") == item.lower().replace("_", "-")
                for item in languages
            ):
                continue
            value = self.model_paths.get(model_id)
            model_path = self._paths(value)[0] if value is not None else None
            yield ModelInfo(
                id=f"kokoro:{model_id}",
                plugin=self.id,
                model_id=model_id,
                source=str(profile.get("source", "local")),
                quality=profile.get("quality"),
                languages=languages,
                sample_rate=int(profile.get("sample_rate", self.sample_rate)),
                installed=model_path.exists() if model_path is not None else False,
            )

    def voices(self, *, language: str | None = None, model: str | None = None):
        for model_id in self._model_ids():
            if model is not None and model not in {model_id, f"kokoro:{model_id}"}:
                continue
            profile = self.model_profiles.get(model_id, {})
            value = self.model_paths.get(model_id)
            model_path = voices_path = None
            if value is not None:
                model_path, voices_path = self._paths(value)
            if voices_path is not None and voices_path.exists():
                with np.load(voices_path, allow_pickle=False) as archive:
                    names = tuple(archive.files)
            else:
                names = tuple(str(item) for item in profile.get("voices", ()))
            languages = tuple(str(item) for item in profile.get("languages", ("en-us",)))
            details = profile.get("voice_languages", {})
            for name in names:
                voice_languages = tuple(str(item) for item in details.get(name, languages))
                info = VoiceInfo(
                    id=f"kokoro:{model_id}/{name}",
                    plugin=self.id,
                    voice_id=name,
                    model_id=model_id,
                    languages=voice_languages,
                    sample_rate=int(profile.get("sample_rate", self.sample_rate)),
                    installed=bool(model_path and voices_path and model_path.exists() and voices_path.exists()),
                    downloadable=not self.offline,
                    quality=profile.get("quality"),
                    metadata={
                        "model_path": str(model_path) if model_path else None,
                        "voices_path": str(voices_path) if voices_path else None,
                    },
                )
                if language is None or info.supports_language(language):
                    yield info
    def capabilities(self, voice: VoiceInfo) -> RenderCapabilities:
        loaded = self._models.get(voice.id) or (
            self._models.get(voice.model_id) if voice.model_id is not None else None
        )
        return RenderCapabilities(
            native_rate=True,
            word_timing=bool(getattr(loaded, "has_timing", False)),
        )

    def providers(self) -> Mapping[str, Any]:
        try:
            import onnxruntime
        except ImportError:
            return {"selected": self.provider, "available": ()}
        return {
            "selected": self.provider,
            "available": tuple(onnxruntime.get_available_providers()),
        }

    def ensure_voice(
        self,
        voice: VoiceInfo,
        *,
        progress: AssetProgressCallback | None = None,
    ) -> None:
        if progress is not None:
            progress(AssetProgressEvent(self.id, "ensure_voice", voice.id, 0, 1))
        self._get_model(voice)
        if progress is not None:
            progress(AssetProgressEvent(self.id, "ensure_voice", voice.id, 1, 1))

    def _get_model(self, voice: VoiceInfo) -> Any:
        cached = self._models.get(voice.id)
        if cached is None and voice.model_id is not None:
            cached = self._models.get(voice.model_id)
        if cached is not None:
            return cached
        if self.voice_loader is not None:
            loaded = self.voice_loader(voice)
            self._models[voice.id] = loaded
            if voice.model_id is not None:
                self._models[voice.model_id] = loaded
            return loaded
        value = self.model_paths.get(voice.model_id or "")
        if value is None:
            raise FileNotFoundError(f"no local Kokoro model is configured for {voice.model_id!r}")
        model_path, voices_path = self._paths(value)
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(f"Kokoro model assets are incomplete for {voice.id!r}")
        try:
            import onnxruntime

            providers = None if self.provider == "auto" else [
                {
                    "cpu": "CPUExecutionProvider",
                    "cuda": "CUDAExecutionProvider",
                    "directml": "DmlExecutionProvider",
                    "coreml": "CoreMLExecutionProvider",
                    "openvino": "OpenVINOExecutionProvider",
                }.get(self.provider.lower(), self.provider)
            ]
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=providers,
                provider_options=[self.provider_options] if providers else None,
            )
            with np.load(voices_path, allow_pickle=False) as archive:
                voices = {name: np.asarray(archive[name]) for name in archive.files}
        except Exception:
            raise
        loaded = _LoadedKokoro(session, voices, self.sample_rate)
        self._models[voice.model_id or voice.id] = loaded
        return loaded

    def render(self, request: RenderRequest) -> AudioFragment:
        model = self._get_model(request.voice)
        annotations = ()
        if request.context is not None:
            annotations = request.context.tokens
        g2p = self._module()
        phonemize = g2p.phonemize
        kwargs: dict[str, Any] = {
            "return_ids": True,
            "return_phonemes": True,
            "annotations": annotations,
        }
        pronunciation = request.pronunciation
        if pronunciation is not None:
            alphabet = getattr(pronunciation, "alphabet", None)
            phonemes = getattr(pronunciation, "phonemes", None)
            if isinstance(pronunciation, Mapping):
                alphabet = pronunciation.get("alphabet", alphabet)
                phonemes = pronunciation.get("phonemes", pronunciation.get("value", phonemes))
            if alphabet not in {None, "ipa", "espeak", "kokoro"}:
                raise ValueError(f"unsupported Kokoro pronunciation alphabet {alphabet!r}")
            if not isinstance(phonemes, str) or not phonemes:
                raise ValueError("Kokoro pronunciation override must contain phonemes")
            override_type = getattr(g2p, "OverrideSpan", None)
            if override_type is None:
                raise RuntimeError("kokorog2p does not expose OverrideSpan")
            kwargs["overrides"] = (override_type(0, len(request.segment.text), {"ph": phonemes}),)
        if "target_model" in inspect.signature(phonemize).parameters:
            kwargs["target_model"] = request.voice.model_id
        result = phonemize(request.segment.text, request.segment.language, **kwargs)
        ids = tuple(getattr(result, "token_ids", ()))
        voice_name = request.voice.voice_id
        speed = request.prosody.rate if request.prosody.rate > 0 else 1.0
        short = self.short_sentence_config
        short_result = None
        if isinstance(short, ShortSentenceConfig):
            config = short
            short_result = prepare_short_sentence(
                request.segment.text, len(ids), short,
                language=request.segment.language,
                seed=request.options.get("short_sentence_seed"),
            )
            if short_result.mode == "wrap":
                pretext = short.wrap.phoneme_pretext
                ids = tuple(
                    self._module().phonemes_to_ids(
                        f"{pretext}{result.phonemes or ''}{pretext}",
                        model=request.voice.model_id or "1.0",
                    )
                )
            elif short_result.mode in {"phrase", "randomized-phrase"}:
                contextual = phonemize(
                    short_result.text, request.segment.language,
                    return_ids=True, return_phonemes=True, annotations=annotations,
                )
                ids = tuple(getattr(contextual, "token_ids", ()))
        audio, outputs = model.synthesize(ids, voice=voice_name, speed=speed)
        if short_result is not None and short_result.mode in {"phrase", "randomized-phrase"}:
            try:
                audio = cut_linear(audio, short_result)
            except ValueError:
                short_result = prepare_short_sentence(
                    request.segment.text, 0,
                    ShortSentenceConfig(
                        min_phoneme_length=config.min_phoneme_length,
                        enabled=True,
                        resolve_mode="wrap",
                        wrap=config.wrap,
                    ),
                    language=request.segment.language,
                )
                pretext = config.wrap.phoneme_pretext
                ids = tuple(self._module().phonemes_to_ids(
                    f"{pretext}{result.phonemes or ''}{pretext}",
                    model=request.voice.model_id or "1.0",
                ))
                audio, outputs = model.synthesize(ids, voice=voice_name, speed=speed)
        metadata: dict[str, Any] = {
            "plugin": self.id,
            "voice": request.voice.id,
            "phonemes": result.phonemes,
        }
        if short_result is not None:
            metadata["short_sentence"] = {
                "mode": short_result.mode,
                "fallback": short_result.fallback,
                "target_start": short_result.target_start,
                "target_end": short_result.target_end,
            }
        alignment: list[AudioTextSpan] = []
        if outputs is not None:
            metadata["pred_dur"] = outputs
            tokens = request.context.tokens if request.context is not None else ()
            durations = np.maximum(np.asarray(outputs, dtype=np.float64), 0.0)
            total = float(durations.sum())
            if total > 0 and audio.size and tokens:
                cursor = 0
                token_list = [
                    token for token in tokens
                    if any(character.isalnum() for character in token.text)
                ]
                for index, token in enumerate(token_list):
                    weight = max(1.0, float(len(token.text)))
                    fraction = weight / sum(max(1.0, float(len(item.text))) for item in token_list)
                    end = audio.size if index == len(token_list) - 1 else cursor + round(audio.size * fraction)
                    alignment.append(
                        AudioTextSpan(
                            token.spoken_start,
                            token.spoken_end,
                            cursor,
                            min(audio.size, max(cursor, end)),
                            "word",
                        )
                    )
                    cursor = min(audio.size, max(cursor, end))
        return AudioFragment(
            segment_id=request.segment.id,
            audio=audio,
            sample_rate=int(getattr(model, "sample_rate", request.voice.sample_rate or self.sample_rate)),
            realized_prosody=frozenset({"rate"}) if "rate" in request.prosody.requested else frozenset(),
            alignment=tuple(alignment),
            metadata=metadata,
        )

    def close(self) -> None:
        for model in self._models.values():
            close = getattr(model, "close", None)
            if close is not None:
                close()
        self._models.clear()
