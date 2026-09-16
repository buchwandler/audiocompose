from __future__ import annotations

import argparse
from pathlib import Path

from ._plan import Plan

from .runtime import Renderer, TTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="utterrender")
    sub = parser.add_subparsers(dest="command", required=True)

    voices = sub.add_parser("voices", help="list voices exposed by installed plugins")
    voices.add_argument("--language")

    install = sub.add_parser("install", help="ensure one plugin voice is available locally")
    install.add_argument("voice")

    render = sub.add_parser("render", help="render a .utterplan.json file")
    render.add_argument("plan")
    render.add_argument("output")
    render.add_argument("--voice", required=True)

    say = sub.add_parser("say", help="compile text with utterplan and render it")
    say.add_argument("text")
    say.add_argument("output")
    say.add_argument("--language", required=True)
    say.add_argument("--voice", required=True)
    say.add_argument("--ssmd", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "voices":
        with Renderer(apply_prosody=False) as renderer:
            for voice in renderer.voices(language=args.language):
                print(f"{voice.id}\t{','.join(voice.languages)}\t{voice.quality or '-'}")
        return
    if args.command == "install":
        with Renderer(apply_prosody=False) as renderer:
            renderer.ensure_voice(args.voice)
            print(args.voice)
        return
    if args.command == "render":
        plan = Plan.load(args.plan)
        with Renderer(default_voice=args.voice) as renderer:
            renderer.to_wav(plan, args.output)
        return
    if args.command == "say":
        with TTS(default_voice=args.voice) as tts:
            tts.to_wav(
                args.text,
                Path(args.output),
                language=args.language,
                input_format="ssmd" if args.ssmd else "plain",
            )
