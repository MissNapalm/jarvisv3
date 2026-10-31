#!/usr/bin/env python3
"""Voice assistant for the TV-show library.

Say "Hey Wil" and then your request, press Enter and speak, or type a command and press Enter.
Ctrl+C / "quit" to exit.
"""
from __future__ import annotations

import argparse
import json
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
from tvassist import quick, report  # noqa: E402
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
    ap.add_argument("--think", action="store_true",
                    help="let the local model reason before answering: a little more reliable, several seconds slower")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="Claude effort: low is snappiest for voice (default)")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--speaker", default=os.environ.get("TV_SPEAKER"),
                    help="output device (name or part of it, e.g. 'USB Audio Speakers'); default: system default")
    ap.add_argument("--mic", default=os.environ.get("TV_MIC"),
                    help="microphone (name or part of it, e.g. 'Yeti'); default: system default")
    ap.add_argument("--no-wake", action="store_true", help="turn off the wake word")
    ap.add_argument("--wake-word", default=os.environ.get("TV_WAKE_WORD", "wil"), choices=["wil", "jarvis"],
                    help='"wil" (say "Hey Wil"; spotted by Whisper) or "jarvis" (openWakeWord); default wil')
    ap.add_argument("--wake-sensitivity", type=float, default=float(os.environ.get("TV_WAKE_THRESHOLD", 0.4)),
                    help='wake word score needed, 0-1 (default 0.4; lower = hears quieter voices, '
                         'but more false triggers)')
    ap.add_argument("--silence", type=float, default=0.8,
                    help="seconds of silence that end a spoken command (default 0.8; raise if it cuts you off)")
    ap.add_argument("--voice", default=os.path.join(HERE, "voices", "en_US-ryan-high.onnx"))
    ap.add_argument("--voice-sample", default=os.environ.get("TV_VOICE_SAMPLE"),
                    help="WAV recording (10-30 s) of a voice to clone for replies (needs .venv-voice)")
    ap.add_argument("--scan", action="store_true", help="re-index the library and exit")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TV_UI_PORT", 8765)),
                    help="port for the live web page (default 8765)")
    ap.add_argument("--no-popup", action="store_true",
                    help="don't pop up the little window when you say the wake word (it's always at the --port page)")
    ap.add_argument("--window", action="store_true", help="also open the window on start (it stays open)")
    args = ap.parse_args()

    # Running as root inside a desktop session: reach the user's PipeWire and X display.
    # Use whichever runtime dir actually has the PipeWire socket (root's own usually doesn't).
    rt = os.environ.get("XDG_RUNTIME_DIR")
    if not (rt and os.path.exists(os.path.join(rt, "pipewire-0"))) and os.path.exists("/run/user/1000/pipewire-0"):
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"
    os.environ.setdefault("DISPLAY", ":0")

    data_dir = os.path.join(HERE, "data")
    os.makedirs(data_dir, exist_ok=True)

    # The window first, so there's something to look at (with a loading bar) while the rest starts
    on_idle: list = []  # run whenever it's back to waiting for a command
    on_exit: list = []  # run on the way out
    events: queue.Queue = queue.Queue()   # ("text", line) from the keyboard, ("speech", audio) from the mic
    ui = None
    if not args.scan:
        from tvassist.ui import UI
        ui = UI(events, port=args.port, settings_path=os.path.join(data_dir, "settings.json"))
        # closing the window quits, right away, whatever it's doing (starting up, thinking, talking)
        ui.on_last_window_closed = lambda: shutdown("Window closed")
        if ui.url:
            print(f"{DIM}Web page: {ui.url}{RESET}")
            if args.window:
                open_window(ui.url)
        else:
            print(f"{DIM}(web page unavailable: {ui.error}){RESET}")

    # The cloned voice takes ~20 s to start: begin now, alongside everything else
    clone = None
    if not (args.scan or args.text or args.no_speak) and args.voice_sample:
        from tvassist import voice
        print(f"{DIM}Loading cloned voice from {args.voice_sample} (in the background) ...{RESET}", flush=True)
        voice_done = ui.task("Warming up Wil’s voice", 20)

        def voice_ready(error):
            voice_done()
            if error:
                print(f"\n{DIM}(cloned voice unavailable, using the standard voice: {error}){RESET}", flush=True)
        try:
            clone = voice.CloneSpeaker(args.voice_sample, on_ready=voice_ready)
        except Exception as e:
            voice_ready(str(e))

    # The library folders: chosen in the window on every start (last time's are filled in); from
    # --library, or last time's, when there's no window to ask in
    saved = (ui.settings.get("library") if ui else None) or \
        [f for f in os.environ.get("TV_LIBRARY", "/mnt/T/TV shows").split(":") if f]
    if args.library:
        folders = args.library
    elif ui and ui.url and (args.window or _wait_for(ui.has_viewers, 5)):
        folders = choose_library(ui, saved)  # closing the window meanwhile shuts down (shutdown())
    else:
        folders = saved
    folders = [f for f in folders if f and os.path.isdir(f)]
    if not folders:
        sys.exit("No library folder found; pass --library /path/to/TV")
    lib = Library(folders, os.path.join(data_dir, "library.db"))
    print(f"{DIM}Indexing {', '.join(folders)} ...{RESET}", end=" ", flush=True)
    scan_done = ui.task("Scanning the TV library", 4) if ui else (lambda: None)
    seen = {"files": 0, "folder": "", "at": time.time()}
    scanned = threading.Event()

    def scan_progress(files, folder):
        seen.update(files=files, folder=folder, at=time.time())

    def watch_scan():  # keep the window's loading bar honest: a count while it moves, a warning when stuck
        while not scanned.wait(0.5):
            waiting = time.time() - seen["at"]
            if waiting > 8:
                where = os.path.basename(seen["folder"].rstrip("/")) or seen["folder"]
                ui.task_detail("Scanning the TV library",
                               f"network drive not answering ({waiting:.0f} s) at {where}", stalled=True)
            else:
                ui.task_detail("Scanning the TV library", f"{seen['files']:,} files")
    if ui:
        threading.Thread(target=watch_scan, daemon=True).start()
    stats = lib.scan(progress=scan_progress)
    scanned.set()
    scan_done()
    print(f"{DIM}{stats['files']} files in {stats['seconds']}s{RESET}")
    if args.scan:
        print(lib.stats())
        return

    wake_phrase = "Hey Wil" if args.wake_word == "wil" else "Hey Jarvis"
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
    player = _player[0] = Player(lib, data_dir, fullscreen=not args.windowed,
                    mpv_args=[f"--audio-device=pipewire/{speaker}"] if speaker else None)
    if backend == "claude":
        from tvassist.brain import Brain
        brain = Brain(lib, player, effort=args.effort)
        print(f"{DIM}Brain: Claude API{RESET}")
    else:
        from tvassist.local import LocalBrain
        print(f"{DIM}Brain: {args.local_model} (local, free) - loading ...{RESET}", flush=True)
        brain_done = ui.task("Loading the brain", 4)
        brain = LocalBrain(lib, player, model=args.local_model, think=args.think)
        brain_done()
    ui.attach_player(player)

    listener = speak = transcribe = None
    if not args.text:
        from tvassist import voice
        print(f"{DIM}Loading speech models ...{RESET}", flush=True)
        ears_done = ui.task("Loading speech recognition", 1)
        transcribe = voice.Transcriber(args.whisper_model, hotwords="Hey Wil" if args.wake_word == "wil" else None)
        ears_done()
        if not args.no_speak:
            piper = voice.Speaker(args.voice, device=speaker)
            speak = piper.say
            if clone is not None:  # still warming up, maybe: say() waits for it
                clone.device, clone.fallback = speaker, piper
                speak = clone.say
        chime = voice.make_chime(os.path.join(data_dir, "chime.wav"))
        heard_ping = voice.make_heard_ping(os.path.join(data_dir, "heard.wav"))

        def ping(path):
            subprocess.Popen(["pw-play", *(["--target", speaker] if speaker else []), path],
                             stderr=subprocess.DEVNULL)

        # Wil's head: on screen whenever its checkbox is ticked (drag it anywhere; it remembers where),
        # mouthing along with every reply. On the wake word, unless a window is already open, a little
        # pop-up window too, which closes itself ~15 s after things go quiet.
        win = [None]   # the pop-up window's process
        head = [None]  # the talking head's process
        face_images = [os.path.join(data_dir, f) for f in ("wiltop.png", "wiljaw.png")]  # head, jaw

        def alive(proc):
            return proc is not None and proc.poll() is None

        def show_head():
            if not alive(head[0]) and all(os.path.isfile(f) for f in face_images):
                head[0] = subprocess.Popen(
                    ["/usr/bin/python3", os.path.join(HERE, "tvassist", "face.py"), *face_images, "660",
                     os.path.join(data_dir, "head-position.json")],
                    env={**os.environ, "QT_QPA_PLATFORM": "xcb"}, stdin=subprocess.PIPE, text=True,
                    stdout=subprocess.DEVNULL, stderr=open(os.path.join(data_dir, "face.log"), "w"),
                    start_new_session=True)

        def hide_head():
            if alive(head[0]):
                head[0].terminate()
            head[0] = None

        def close_popup(why=""):
            if alive(win[0]):
                print(f"\n{DIM}(pop-up closed: {why}){RESET}", flush=True)
                win[0].terminate()
            win[0] = None
        ui.on_popup_done = lambda: close_popup("quiet")

        def mouth(levels, fps):  # a reply is starting: move the head's jaw along with it
            if alive(head[0]):
                try:
                    head[0].stdin.write(json.dumps({"say": levels, "fps": fps}) + "\n")
                    head[0].stdin.flush()
                except OSError:
                    pass
        voice.on_speech = mouth
        ui.on_stop = voice.stop_speaking

        def show_popup():
            if args.no_popup or not ui.url or alive(win[0]) or ui.has_viewers():
                return
            proc = win[0] = open_window(ui.url + "?popup=1", size=(380, 520))

            def follow():  # closed by hand
                proc.wait()
                if win[0] is proc:
                    win[0] = None
            threading.Thread(target=follow, daemon=True).start()

        if ui.settings.get("head", True):
            show_head()
        on_exit.append(lambda: (close_popup("quitting"), hide_head()))

        def setting_changed(name, value):  # the head checkbox: show or hide him right away
            if name == "head":
                show_head() if value else hide_head()
        ui.on_setting = setting_changed

        def on_wake():  # runs on the mic thread the moment the wake phrase is heard
            ping(chime)
            show_popup()
            player.hold()
            listen_started[0] = time.time()
            ui.state("listening", f"Heard “{wake_phrase}” — go ahead")
            print(f"\r{GREEN}● {wake_phrase} - listening…{RESET}   ", flush=True)

        try:
            listener = voice.MicListener(events, device=mic, silence_after=args.silence,
                                         wake=not args.no_wake, on_wake=on_wake,
                                         wake_word=args.wake_word, transcriber=transcribe,
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

    if sys.stdin.isatty():  # run as a service there's no keyboard: the web page is the input, keep running
        threading.Thread(target=keyboard, daemon=True).start()

    from tvassist.brain import build_tools
    quick_tools = {t.name: t for t in build_tools(lib, player)}

    def show_tool(name, inp):
        call = f"{name}({', '.join(f'{k}={v!r}' for k, v in inp.items())})"
        ui.message("tool", call)
        print(f"  {DIM}→ {call}{RESET}")

    if args.text:
        idle_hint = "Type a command"
    elif listener and listener.wakes:
        idle_hint = f"Say “{wake_phrase}”, press Talk, or type"
    else:
        idle_hint = "Press Talk or type a command"

    def prompt():  # called whenever it's ready for the next command
        ui.state("idle", idle_hint)
        for f in on_idle:
            f()
        print(f"{CYAN}›{RESET} ", end="", flush=True)

    def answer(reply, secs, **extra):
        ui.message("jarvis", reply, secs=round(secs, 1), **extra)
        print(f"{BOLD}assistant:{RESET} {reply}  {DIM}({secs:.1f}s){RESET}")

    if args.text:
        hint = "type a command"
    elif listener and listener.wakes:
        hint = f'say "{wake_phrase}", press Enter and speak, or type a command'
    else:
        hint = "press Enter and speak, or type a command"
    # The window's "Calibrate mic" button: runs the mic check (between requests, on this thread)
    check = [None]

    def calibrate_button(action):
        if action == "start":
            events.put(("calibrate", None))
        elif action == "skip" and check[0] is not None:
            check[0].skip.set()
    ui.on_calibrate = calibrate_button

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
        if kind == "calibrate":
            if listener is None:
                ui.calibrate("done", "There's no microphone in use (started with --text).", ok=False)
            else:
                from tvassist.calibrate import MicCheck
                listener.disarm()
                ui.state("calibrating", "Follow the steps below")
                check[0] = MicCheck(ui, mic, transcribe, chime=lambda: ping(chime))
                check[0].run()
                check[0] = None
                listener.arm()
            prompt()
            continue
        if kind == "window_closed":
            print(f"\n{DIM}Window closed - shutting down.{RESET}")
            break
        if kind in ("eof", "mic_error"):
            if kind == "mic_error":
                ui.message("info", f"Microphone stopped: {payload}")
                print(f"\n{DIM}Microphone stopped: {payload}{RESET}")
            print()
            break

        if kind == "text":
            typed = payload.strip()
            if typed.lower() in ("quit", "exit"):
                break
            if typed.lower() == "reset":
                brain.reset()
                ui.message("info", "Conversation cleared")
                print(f"{DIM}Conversation cleared.{RESET}")
                prompt()
                continue
            if not typed:
                if listener is None:
                    prompt()
                    continue
                player.hold()
                listen_started[0] = time.time()
                ui.state("listening", "Go ahead")
                print(f"{GREEN}● listening…{RESET}", flush=True)
                listener.request_recording()
                continue
            text = typed
            ui.message("you", text, via="typed")
            if listener:
                listener.disarm()
        else:  # speech from the mic (wake word or Enter)
            if payload is None:
                player.release()
                ui.message("info", "Didn't hear anything")
                print(f"{DIM}(didn't hear anything){RESET}")
                listener.arm()
                prompt()
                continue
            listened = time.time() - listen_started[0] if listen_started[0] else 0
            t = time.time()
            ui.state("transcribing", f"Listened {listened:.1f} s")
            heard = transcribe(payload)
            text = voice.strip_wake_phrase(heard)
            if heard.strip():
                ui.message("you", heard, via="voice", secs=round(time.time() - t, 1))
            print(f"{BOLD}you:{RESET} {heard}  {DIM}(listened {listened:.1f}s, transcribed {time.time() - t:.1f}s){RESET}")
            if not text:
                if not retried:  # just "Hey Wil" and a pause, or only noise: listen once more for the command
                    retried = True
                    listener.request_recording()
                    ui.state("listening", "Go ahead")
                    print(f"{GREEN}● listening…{RESET}", flush=True)
                    continue
                player.release()
                listener.arm()
                prompt()
                continue
            ping(heard_ping)  # only now: it has an actual request, not just "Hey Wil"
        retried = False
        if re.search(r"\b(pause|stop|hold on|hang on|wait|quit|turn (it )?off)\b", text.lower()):
            player.forget_hold()  # the user wants it paused/stopped: don't resume afterwards
        player.release()  # resume the show we paused to listen (a new show starts playing anyway)

        t = time.time()
        reply = None
        ui.state("thinking")
        if speak:
            voice.allow_speech()  # a new request: undo an earlier Stop
        if report.wants_file_update(text):  # "give me a file update": compare with yesterday's files
            print(f"  {DIM}checking files…{RESET}", flush=True)
            reply, details = report.file_update(lib, data_dir)
            print(details)
            brain.note(text, reply)
            answer(reply, time.time() - t)
            if speak:
                ui.state("speaking")
                speak(reply)
            if listener:
                listener.arm()
            prompt()
            continue
        try:  # instant paths: playback controls, and "play/continue <show>" when the show is unambiguous
            reply = quick.handle(text, player)
            if reply is None:
                reply = quick.handle_play(text, lib, quick_tools)
        except Exception:
            reply = None
        quick_reply = reply is not None
        if quick_reply:
            print(f"  {DIM}⚡ quick command{RESET}")
            brain.note(text, reply)
        else:
            try:
                reply = brain.ask(text, on_tool=show_tool)
            except Exception as e:  # keep the assistant alive through API/network errors
                reply = f"Sorry, something went wrong: {e}"
        answer(reply, time.time() - t, quick=quick_reply)
        if speak and reply:
            player.duck()
            ui.state("speaking")
            speak(reply)
        player.unduck()
        if listener:
            listener.arm()  # only listen for the wake word again once the reply has been spoken
        prompt()
    for f in on_exit:
        f()


_player = [None]  # set once it exists, so shutdown() can save where you were


def _descendants() -> list[int]:
    """Every process started from this one, however deep (voice server, head, pop-ups, mpv, mic...)."""
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(d))
    out, todo = [], [os.getpid()]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def shutdown(why: str = "") -> None:
    """Stop everything this started and exit now (from any thread)."""
    import signal
    if why:
        print(f"\n{DIM}{why} - shutting down.{RESET}", flush=True)
    try:
        if _player[0] is not None:
            _player[0].stop()  # saves where you were in the episode
    except Exception:
        pass
    procs = _descendants()
    for pid in procs:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + 2
    while time.time() < end and any(os.path.exists(f"/proc/{p}") for p in procs):
        time.sleep(0.1)
    for pid in _descendants():  # anything that ignored the polite request
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    os._exit(0)


def _wait_for(cond, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def choose_library(ui, saved: list[str]) -> list[str]:
    """Ask in the window which folders are the TV library; blocks until "Use these"."""
    chosen: list = []
    picked = threading.Event()

    def use(folders):
        missing = [f for f in folders if not os.path.isdir(f)]
        if not folders:
            return "Add at least one folder."
        if missing:
            return "Can't find " + ", ".join(missing) + " (is the drive connected?)"
        chosen[:] = folders
        ui.set("library", folders)  # filled in next time
        ui.library("done", folders)
        picked.set()
        return None
    ui.on_library = use
    ui.state("starting", "Choose your TV library")
    ui.library("choose", saved)
    print(f"{DIM}Waiting for the TV library to be chosen in the window ...{RESET}", flush=True)
    picked.wait()
    ui.on_library = None
    return chosen


def open_window(url: str, size: tuple[int, int] = (440, 680)) -> subprocess.Popen:
    """Show the live page in a small app window (Chromium/Chrome --app), else a browser tab."""
    import shutil
    for exe in ("chromium", "google-chrome-stable", "google-chrome", "brave", "microsoft-edge-stable"):
        if shutil.which(exe):
            profile = os.path.join(HERE, "data", "window-profile")  # its own profile: opens as a separate window
            cmd = [exe, f"--app={url}", f"--window-size={size[0]},{size[1]}", f"--user-data-dir={profile}",
                   "--no-first-run", "--no-default-browser-check"]
            break
    else:
        cmd = ["xdg-open", url]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C: exit quietly, whatever it was doing
        print()
    shutdown()  # however it ended ("quit", Ctrl+C, the window): nothing it started is left running
