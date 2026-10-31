"""Live web page for the assistant: what you said, what it answered, what it's doing, and what's playing.

Serves ui.html on http://localhost:<port>/ and pushes updates to it with server-sent events. Text typed
on the page (or the Talk button) goes into the same event queue as the keyboard.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
HISTORY = 300  # messages replayed to a page that connects (or reloads)


class UI:
    def __init__(self, events: queue.Queue, port: int = 8765, player=None, settings_path: str | None = None):
        self.events, self.player = events, player
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._state = {"type": "state", "state": "starting", "detail": "Starting up…", "at": time.time()}
        self._playing: dict = {"type": "playing", "now": None}
        self._loading: dict = {"type": "loading", "tasks": []}
        self._calibrate: dict = {"type": "calibrate", "step": None}
        self._library: dict = {"type": "library", "step": None}
        self._clients: list[queue.Queue] = []
        self._windows = 0          # open main windows (pop-ups not counted)
        self._had_window = False
        self.on_last_window_closed = None  # f(): every main window has been closed
        self.url = None
        self.on_popup_done = None  # called when the pop-up window has gone quiet and wants closing
        self.on_setting = None     # f(name, value) after a setting is changed on the page
        self.on_stop = None        # f(): the Stop button (cut off what it's saying)
        self.on_calibrate = None   # f(action): the mic check's buttons ("skip")
        self.on_library = None     # f(folders) -> error text or None: "Use these" on the library chooser
        self.settings_path = settings_path
        self.settings = {"head": True}  # head: show Wil's head in the screen's corner
        try:
            with open(settings_path) as f:
                self.settings.update(json.load(f))
        except (OSError, TypeError, ValueError):
            pass
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        except OSError as e:  # port taken (another copy running?): carry on without the page
            self.error = str(e)
            return
        self.error = None
        self.server.daemon_threads = True
        self.url = f"http://localhost:{port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        if player is not None:
            self.attach_player(player)

    def attach_player(self, player) -> None:
        """Show what's playing (the player may only exist after the page is up)."""
        self.player = player
        threading.Thread(target=self._watch_player, daemon=True).start()

    def task(self, name: str, expected: float):
        """A startup step for the loading bar; call the returned function when it's done.
        `expected` is roughly how many seconds it takes, so the bar can move while it runs."""
        with self._lock:
            tasks = self._loading["tasks"] + [{"name": name, "expected": expected,
                                               "started": time.time(), "done": False}]
        self._send({"type": "loading", "tasks": tasks})

        def done():
            self.task_detail(name, done=True)
        return done

    def task_detail(self, name: str, detail: str = "", stalled: bool = False, done: bool = False) -> None:
        """Say how a startup step is going ("4,210 files"), or that it's stuck waiting on something."""
        with self._lock:
            tasks = [dict(t, detail=detail, stalled=stalled, done=done or t["done"]) if t["name"] == name else t
                     for t in self._loading["tasks"]]
        self._send({"type": "loading", "tasks": tasks})

    def has_viewers(self) -> bool:
        """Whether a window is open on the page right now."""
        with self._lock:
            return bool(self._clients)

    # ---------- what the assistant reports ----------
    def state(self, state: str, detail: str = "") -> None:
        """idle, listening, transcribing, thinking, speaking (or starting)."""
        self._send({"type": "state", "state": state, "detail": detail, "at": time.time()})

    def message(self, who: str, text: str, **extra) -> None:
        """who: you, jarvis, tool or info. extra: e.g. via="voice", secs=1.2."""
        self._send({"type": "message", "who": who, "text": text, "at": time.time(), **extra})

    def _send(self, msg: dict) -> None:
        with self._lock:
            if msg["type"] == "message":
                self._history = (self._history + [msg])[-HISTORY:]
            elif msg["type"] == "state":
                self._state = msg
            elif msg["type"] == "playing":
                self._playing = msg
            elif msg["type"] == "loading":
                self._loading = msg
            elif msg["type"] == "calibrate":
                self._calibrate = msg
            elif msg["type"] == "library":
                self._library = msg
            elif msg["type"] == "settings":
                self.settings = msg["settings"]
            for q in self._clients:
                q.put(msg)

    def calibrate(self, step: str, text: str, **extra) -> None:
        """The mic check: step is quiet, speak, retry, done, or level (just the meter: extra level=0-1)."""
        if step == "level":  # the meter: frequent, not worth keeping for late joiners
            self._send({"type": "meter", "level": extra.get("level", 0)})
        else:
            self._send({"type": "calibrate", "step": step, "text": text, "at": time.time(), **extra})

    def library(self, step: str, folders: list[str], error: str = "") -> None:
        """The library chooser: step "choose" (waiting for you) or "done"."""
        self._send({"type": "library", "step": step, "error": error,
                    "folders": [{"path": f, "ok": os.path.isdir(f)} for f in folders]})

    @staticmethod
    def browse(start: str = "") -> str | None:
        """KDE's folder picker (or zenity's); the chosen folder, or None if cancelled."""
        start = start if os.path.isdir(start) else os.path.expanduser("~")
        for cmd in (["kdialog", "--title", "Choose a TV library folder", "--getexistingdirectory", start],
                    ["zenity", "--file-selection", "--directory", "--title=Choose a TV library folder",
                     f"--filename={start}/"]):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
            except FileNotFoundError:
                continue
            path = r.stdout.strip()
            return path if r.returncode == 0 and os.path.isdir(path) else None
        return None

    def set(self, name: str, value) -> None:
        self._send({"type": "settings", "settings": {**self.settings, name: value}})
        if self.settings_path:
            try:
                with open(self.settings_path, "w") as f:
                    json.dump(self.settings, f)
            except OSError:
                pass
        if self.on_setting:
            self.on_setting(name, value)

    def _watch_player(self) -> None:
        last = None
        while True:
            try:
                now = self.player.now_playing()
            except Exception:
                now = None
            if now is not None:  # position changes every poll; send whole minutes so it isn't chatty
                now = {k: now[k] for k in ("episode", "position_min", "duration_min", "paused", "volume")}
                now["position_min"] = int(now["position_min"])
            if now != last:
                last = now
                self._send({"type": "playing", "now": now})
            time.sleep(2)

    # ---------- HTTP ----------
    def _handler(self):
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal for the conversation
                pass

            def do_GET(self):
                if self.path.split("?")[0] == "/events":
                    return ui._stream(self, popup="popup=1" in self.path)
                if self.path.split("?")[0] not in ("/", "/index.html"):
                    self.send_error(404)
                    return
                with open(PAGE, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path == "/settings":
                    try:
                        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    except ValueError:
                        data = {}
                    for name, value in data.items():
                        if name in ui.settings and isinstance(value, type(ui.settings[name])):
                            ui.set(name, value)
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/library":
                    try:
                        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    except ValueError:
                        data = {}
                    if data.get("action") == "browse":  # blocks this request until the dialog closes
                        body = json.dumps({"path": ui.browse(str(data.get("start") or ""))}).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    folders = [str(f) for f in data.get("folders") or [] if str(f).strip()]
                    error = ui.on_library(folders) if ui.on_library else "Not choosing a library right now."
                    body = json.dumps({"error": error}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/calibrate":
                    try:
                        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                    except ValueError:
                        data = {}
                    if ui.on_calibrate:
                        ui.on_calibrate(str(data.get("action", "")))
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/stop":
                    if ui.on_stop:
                        ui.on_stop()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path == "/popup-done":  # pages can't always close their own app window
                    if ui.on_popup_done:
                        ui.on_popup_done()
                    self.send_response(204)
                    self.end_headers()
                    return
                if self.path != "/say":
                    self.send_error(404)
                    return
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError:
                    data = {}
                # "" is the same as pressing Enter at the terminal: listen for a command now
                ui.events.put(("text", str(data.get("text", "")).strip()))
                self.send_response(204)
                self.end_headers()

        return Handler

    WINDOW_GRACE = 2.0  # s: a reload reconnects within this

    def _window_gone(self) -> None:
        time.sleep(self.WINDOW_GRACE)
        with self._lock:
            gone = self._windows == 0
        if gone and self.on_last_window_closed:
            self.on_last_window_closed()

    def _stream(self, h: BaseHTTPRequestHandler, popup: bool = False) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        q: queue.Queue = queue.Queue()
        with self._lock:
            backlog = [{"type": "history", "messages": self._history}, self._state, self._playing,
                       {"type": "settings", "settings": self.settings}, self._loading, self._calibrate,
                       self._library]
            self._clients.append(q)
            if not popup:
                self._windows += 1
                self._had_window = True
        try:
            for msg in backlog:
                h.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
            h.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=1)  # often, so a closed window is noticed within a second
                    h.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                except queue.Empty:
                    h.wfile.write(b": keepalive\n\n")
                h.wfile.flush()
        except OSError:  # page closed
            pass
        finally:
            with self._lock:
                self._clients.remove(q)
                if not popup:
                    self._windows -= 1
                    last = self._windows == 0
            if not popup and last:  # closed the window: quit, unless it comes straight back (reload)
                threading.Thread(target=self._window_gone, daemon=True).start()
