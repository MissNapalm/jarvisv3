#!/usr/bin/env python3
"""Voice assistant for the TV-show library.

Press Enter and speak, or type a command and press Enter. Ctrl+C / "quit" to exit.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvassist.library import Library  # noqa: E402
from tvassist import quick  # noqa: E402
from tvassist.player import Player  # noqa: E402

DIM, BOLD, CYAN, GREEN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[0m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=os.environ.get("TV_LIBRARY", "/mnt/T/TV shows"))
    ap.add_argument("--text", action="store_true", help="keyboard only: no microphone, no speech output")
    ap.add_argument("--no-speak", action="store_true", help="use the microphone but print replies instead of speaking")
    ap.add_argument("--windowed", action="store_true", help="don't start mpv fullscreen")
    ap.add_argument("--backend", default="auto", choices=["auto", "local", "claude"],
                    help="local = free, runs on your GPU via Ollama; claude = Claude API (needs ANTHROPIC_API_KEY); "
                         "auto = claude if a key is set, otherwise local")
    ap.add_argument("--local-model", default="qwen3:14b", help="Ollama model for the local backend")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="Claude effort: low is snappiest for voice (default)")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--speaker", default=os.environ.get("TV_SPEAKER"),
                    help="output device (name or part of it, e.g. 'USB Audio Speakers'); default: system default")
    ap.add_argument("--mic", default=os.environ.get("TV_MIC"),
                    help="microphone (name or part of it, e.g. 'Yeti'); default: system default")
    ap.add_argument("--silence", type=float, default=0.8,
                    help="seconds of silence that end a spoken command (default 0.8; raise if it cuts you off)")
    ap.add_argument("--voice", default=os.path.join(HERE, "voices", "en_US-ryan-high.onnx"))
    ap.add_argument("--scan", action="store_true", help="re-index the library and exit")
    args = ap.parse_args()

    # Running as root inside a desktop session: reach the user's PipeWire and X display.
    # Use whichever runtime dir actually has the PipeWire socket (root's own usually doesn't).
    rt = os.environ.get("XDG_RUNTIME_DIR")
    if not (rt and os.path.exists(os.path.join(rt, "pipewire-0"))) and os.path.exists("/run/user/1000/pipewire-0"):
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
    os.environ.setdefault("DISPLAY", ":0")

    data_dir = os.path.join(HERE, "data")
    os.makedirs(data_dir, exist_ok=True)
    lib = Library(args.library, os.path.join(data_dir, "library.db"))
    print(f"{DIM}Indexing {args.library} ...{RESET}", end=" ", flush=True)
    stats = lib.scan()
    print(f"{DIM}{stats['files']} files in {stats['seconds']}s{RESET}")
    if args.scan:
        print(lib.stats())
        return

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    backend = args.backend if args.backend != "auto" else ("claude" if has_key else "local")
    if backend == "claude" and not has_key:
        sys.exit("Set ANTHROPIC_API_KEY first (get one at https://platform.claude.com/settings/keys), "
                 "or use --backend local.")

    speaker = mic = None
    if not args.text:
        from tvassist import voice
        try:
            if args.speaker:
                speaker, desc = voice.find_device(args.speaker, "Audio/Sink")
                print(f"{DIM}Speakers: {desc}{RESET}")
            if args.mic:
                mic, desc = voice.find_device(args.mic, "Audio/Source")
                print(f"{DIM}Microphone: {desc}{RESET}")
        except ValueError as e:
            sys.exit(f"{e}. See the device list with: wpctl status")
    player = Player(lib, data_dir, fullscreen=not args.windowed,
                    mpv_args=[f"--audio-device=pipewire/{speaker}"] if speaker else None)
    if backend == "claude":
        from tvassist.brain import Brain
        brain = Brain(lib, player, effort=args.effort)
        print(f"{DIM}Brain: Claude API{RESET}")
    else:
        from tvassist.local import LocalBrain
        print(f"{DIM}Brain: {args.local_model} (local, free) - loading ...{RESET}", flush=True)
        brain = LocalBrain(lib, player, model=args.local_model)

    listen = speak = None
    if not args.text:
        from tvassist import voice
        print(f"{DIM}Loading speech models ...{RESET}", flush=True)
        transcribe = voice.Transcriber(args.whisper_model)
        listen = lambda: voice.record_utterance(silence_after=args.silence, device=mic)  # noqa: E731
        if not args.no_speak:
            speak = voice.Speaker(args.voice, device=speaker).say

    def show_tool(name, inp):
        print(f"  {DIM}→ {name}({', '.join(f'{k}={v!r}' for k, v in inp.items())}){RESET}")

    hint = "type a command" if args.text else "press Enter and speak, or type a command"
    print(f"\n{BOLD}TV assistant ready{RESET} ({hint}; 'reset' clears the conversation, 'quit' exits)\n")
    if speak:
        threading.Thread(target=speak, args=("Ready.",), daemon=True).start()

    while True:
        try:
            typed = input(f"{CYAN}›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if typed.lower() in ("quit", "exit"):
            break
        if typed.lower() == "reset":
            brain.reset()
            print(f"{DIM}Conversation cleared.{RESET}")
            continue

        text = typed
        if not text:
            if listen is None:
                continue
            player.duck()
            print(f"{GREEN}● listening…{RESET}", end="\r", flush=True)
            audio = listen()
            if audio is None:
                player.unduck()
                print(f"{DIM}(didn't hear anything){RESET}   ")
                continue
            t = time.time()
            text = transcribe(audio)
            print(f"{BOLD}you:{RESET} {text}  {DIM}({time.time() - t:.1f}s){RESET}")
            if not text:
                player.unduck()
                continue

        t = time.time()
        try:
            reply = quick.handle(text, player)  # instant path for simple playback controls
        except Exception:
            reply = None
        if reply is not None:
            print(f"  {DIM}⚡ quick command{RESET}")
            brain.note(text, reply)
        else:
            try:
                reply = brain.ask(text, on_tool=show_tool)
            except Exception as e:  # keep the assistant alive through API/network errors
                reply = f"Sorry, something went wrong: {e}"
        print(f"{BOLD}assistant:{RESET} {reply}  {DIM}({time.time() - t:.1f}s){RESET}")
        if speak and reply:
            player.duck()
            speak(reply)
        player.unduck()


if __name__ == "__main__":
    main()
