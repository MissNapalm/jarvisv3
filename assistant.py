#!/usr/bin/env python3
"""Voice assistant for the TV-show library.

Say "Hey Jarvis" and then your request, press Enter and speak, or type a command and press Enter.
Ctrl+C / "quit" to exit.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
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
    ap.add_argument("--library", action="append",
                    help="a library folder; repeat for several (default: $TV_LIBRARY, folders separated by ':', "
                         "or /mnt/T/TV shows and /mnt/T/TV)")
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
    ap.add_argument("--no-wake", action="store_true", help='turn off the "Hey Jarvis" wake word')
    ap.add_argument("--wake-sensitivity", type=float, default=float(os.environ.get("TV_WAKE_THRESHOLD", 0.4)),
                    help='wake word score needed, 0-1 (default 0.4; lower = hears quieter voices, '
                         'but more false triggers)')
    ap.add_argument("--silence", type=float, default=0.8,
                    help="seconds of silence that end a spoken command (default 0.8; raise if it cuts you off)")
    ap.add_argument("--voice", default=os.path.join(HERE, "voices", "en_US-ryan-high.onnx"))
    ap.add_argument("--voice-sample", default=os.environ.get("TV_VOICE_SAMPLE"),
                    help="WAV recording (10-30 s) of a voice to clone for replies (needs .venv-voice)")
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
    folders = args.library or os.environ.get("TV_LIBRARY", "/mnt/T/TV shows:/mnt/T/TV").split(":")
    folders = [f for f in folders if f and os.path.isdir(f)]
    if not folders:
        sys.exit("No library folder found; pass --library /path/to/TV")
    lib = Library(folders, os.path.join(data_dir, "library.db"))
    print(f"{DIM}Indexing {', '.join(folders)} ...{RESET}", end=" ", flush=True)
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

    events: queue.Queue = queue.Queue()   # ("text", line) from the keyboard, ("speech", audio) from the mic
    listener = speak = transcribe = None
    if not args.text:
        from tvassist import voice
        print(f"{DIM}Loading speech models ...{RESET}", flush=True)
        transcribe = voice.Transcriber(args.whisper_model)
        if not args.no_speak:
            piper = voice.Speaker(args.voice, device=speaker)
            speak = piper.say
            if args.voice_sample:
                print(f"{DIM}Loading cloned voice from {args.voice_sample} ...{RESET}", flush=True)
                try:
                    speak = voice.CloneSpeaker(args.voice_sample, device=speaker, fallback=piper).say
                except Exception as e:
                    print(f"{DIM}(cloned voice unavailable, using the standard voice: {e}){RESET}")
        chime = voice.make_chime(os.path.join(data_dir, "chime.wav"))

        def on_wake():  # runs on the mic thread the moment "Hey Jarvis" is heard
            subprocess.Popen(["pw-play", *(["--target", speaker] if speaker else []), chime],
                             stderr=subprocess.DEVNULL)
            player.hold()
            listen_started[0] = time.time()
            print(f"\r{GREEN}● Hey Jarvis - listening…{RESET}   ", flush=True)

        try:
            listener = voice.MicListener(events, device=mic, silence_after=args.silence,
                                         wake=not args.no_wake, on_wake=on_wake,
                                         wake_threshold=args.wake_sensitivity)
        except ImportError:
            print(f"{DIM}(wake word unavailable: pip install openwakeword){RESET}")
            listener = voice.MicListener(events, device=mic, silence_after=args.silence, wake=False)

    def keyboard():
        while True:
            line = sys.stdin.readline()
            events.put(("text", line.rstrip("\n")) if line else ("eof", None))
            if not line:
                return

    threading.Thread(target=keyboard, daemon=True).start()

    from tvassist.brain import build_tools
    quick_tools = {t.name: t for t in build_tools(lib, player)}

    def show_tool(name, inp):
        print(f"  {DIM}→ {name}({', '.join(f'{k}={v!r}' for k, v in inp.items())}){RESET}")

    def prompt():
        print(f"{CYAN}›{RESET} ", end="", flush=True)

    if args.text:
        hint = "type a command"
    elif listener and listener.model is not None:
        hint = 'say "Hey Jarvis", press Enter and speak, or type a command'
    else:
        hint = "press Enter and speak, or type a command"
    print(f"\n{BOLD}TV assistant ready{RESET} ({hint}; 'reset' clears the conversation, 'quit' exits)\n")
    if speak:
        threading.Thread(target=speak, args=("Ready.",), daemon=True).start()
    prompt()

    retried = False
    listen_started = [0.0]  # when the current listen began (set on the mic thread)
    while True:
        try:
            kind, payload = events.get()
        except KeyboardInterrupt:
            print()
            break
        if kind in ("eof", "mic_error"):
            if kind == "mic_error":
                print(f"\n{DIM}Microphone stopped: {payload}{RESET}")
            print()
            break

        if kind == "text":
            typed = payload.strip()
            if typed.lower() in ("quit", "exit"):
                break
            if typed.lower() == "reset":
                brain.reset()
                print(f"{DIM}Conversation cleared.{RESET}")
                prompt()
                continue
            if not typed:
                if listener is None:
                    prompt()
                    continue
                player.hold()
                listen_started[0] = time.time()
                print(f"{GREEN}● listening…{RESET}", flush=True)
                listener.request_recording()
                continue
            text = typed
            if listener:
                listener.disarm()
        else:  # speech from the mic (wake word or Enter)
            if payload is None:
                player.release()
                print(f"{DIM}(didn't hear anything){RESET}")
                listener.arm()
                prompt()
                continue
            listened = time.time() - listen_started[0] if listen_started[0] else 0
            t = time.time()
            heard = transcribe(payload)
            text = voice.strip_wake_phrase(heard)
            print(f"{BOLD}you:{RESET} {heard}  {DIM}(listened {listened:.1f}s, transcribed {time.time() - t:.1f}s){RESET}")
            if not text:
                if not retried:  # just "Hey Jarvis" and a pause, or only noise: listen once more for the command
                    retried = True
                    listener.request_recording()
                    print(f"{GREEN}● listening…{RESET}", flush=True)
                    continue
                player.release()
                listener.arm()
                prompt()
                continue
        retried = False
        if re.search(r"\b(pause|stop|hold on|hang on|wait|quit|turn (it )?off)\b", text.lower()):
            player.forget_hold()  # the user wants it paused/stopped: don't resume afterwards
        player.release()  # resume the show we paused to listen (a new show starts playing anyway)

        t = time.time()
        try:  # instant paths: playback controls, and "play/continue <show>" when the show is unambiguous
            reply = quick.handle(text, player)
            if reply is None:
                reply = quick.handle_play(text, lib, quick_tools)
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
        if listener:
            listener.arm()  # only listen for the wake word again once the reply has been spoken
        prompt()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C: exit quietly, whatever it was doing
        print()
