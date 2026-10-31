"""Live web page for the assistant: what you said, what it answered, what it's doing, and what's playing.

Serves ui.html on http://localhost:<port>/ and pushes updates to it with server-sent events. Text typed
on the page (or the Talk button) goes into the same event queue as the keyboard.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html")
HISTORY = 300  # messages replayed to a page that connects (or reloads)


class UI:
    def __init__(self, events: queue.Queue, port: int = 8765, player=None):
        self.events, self.player = events, player
        self._lock = threading.Lock()
        self._history: list[dict] = []
        self._state = {"type": "state", "state": "starting", "detail": "Starting up…", "at": time.time()}
        self._playing: dict = {"type": "playing", "now": None}
        self._clients: list[queue.Queue] = []
        self.url = None
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
            threading.Thread(target=self._watch_player, daemon=True).start()

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
            for q in self._clients:
                q.put(msg)

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
                if self.path == "/events":
                    return ui._stream(self)
                if self.path not in ("/", "/index.html"):
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

    def _stream(self, h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        q: queue.Queue = queue.Queue()
        with self._lock:
            backlog = [{"type": "history", "messages": self._history}, self._state, self._playing]
            self._clients.append(q)
        try:
            for msg in backlog:
                h.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
            h.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=15)
                    h.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                except queue.Empty:
                    h.wfile.write(b": keepalive\n\n")
                h.wfile.flush()
        except OSError:  # page closed
            pass
        finally:
            with self._lock:
                self._clients.remove(q)
