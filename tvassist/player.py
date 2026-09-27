"""Drives mpv over its JSON IPC socket and records watch progress."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time

from .library import Episode, Library


class Player:
    def __init__(self, library: Library, data_dir: str, fullscreen: bool = True, mpv_args: list[str] | None = None):
        self.lib = library
        self.mpv_args = mpv_args or []
        self.sock_path = os.path.join(data_dir, "mpv.sock")
        self.fullscreen = fullscreen
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._ducked_from: float | None = None
        self._held = False
        threading.Thread(target=self._track_progress, daemon=True).start()

    # ---------- IPC ----------
    def _cmd(self, *args, timeout: float = 2.0):
        if not self.running():
            raise RuntimeError("Nothing is playing.")
        with self._lock, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(self.sock_path)
            s.sendall((json.dumps({"command": list(args)}) + "\n").encode())
            buf = b""
            while True:  # skip async event lines until the reply arrives
                chunk = s.recv(65536)
                if not chunk:
                    raise RuntimeError("mpv closed the connection")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    msg = json.loads(line)
                    if "error" in msg:
                        if msg["error"] != "success":
                            raise RuntimeError(f"mpv: {msg['error']}")
                        return msg.get("data")

    def get(self, prop: str):
        try:
            return self._cmd("get_property", prop)
        except (RuntimeError, OSError):
            return None

    def set(self, prop: str, value) -> None:
        self._cmd("set_property", prop, value)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------- playback ----------
    def play(self, queue: list[Episode], start: float = 0.0) -> None:
        """Play queue[0] (from `start` seconds); the rest of the queue follows automatically."""
        paths = [self.lib.abspath(e) for e in queue]
        if self.running():
            self._cmd("loadfile", paths[0], "replace", -1, {"start": str(start)} if start else {})
            for p in paths[1:]:
                self._cmd("loadfile", p, "append")
            self._cmd("set_property", "pause", False)
            return
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        args = ["mpv", f"--input-ipc-server={self.sock_path}", "--force-window=immediate",
                "--keep-open=no", "--idle=no", "--sub-auto=fuzzy", "--really-quiet",
                "--hwdec=auto-safe"]  # decode on the GPU: smoother 4K/HEVC, less CPU
        if self.fullscreen:
            args.append("--fs")
        if start:
            args.append(f"--start={start}")
        args += self.mpv_args
        self.proc = subprocess.Popen(args + ["--", *paths], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):  # wait for the socket
            if os.path.exists(self.sock_path):
                break
            time.sleep(0.1)

    def stop(self) -> None:
        if self.running():
            self._save_progress()
            try:
                self._cmd("quit")
            except (RuntimeError, OSError):
                self.proc.terminate()

    def now_playing(self) -> dict | None:
        if not self.running():
            return None
        path = self.get("path")
        ep = self._episode_for(path)
        pos, dur = self.get("time-pos") or 0, self.get("duration") or 0
        return {"episode": ep.label() if ep else os.path.basename(path or ""),
                "episode_id": ep.id if ep else None,
                "position_min": round(pos / 60, 1), "duration_min": round(dur / 60, 1),
                "paused": bool(self.get("pause")), "volume": self.get("volume"),
                "queue_remaining": max(0, (self.get("playlist-count") or 1) - 1 - (self.get("playlist-pos") or 0))}

    # ---------- pausing while the assistant listens ----------
    def hold(self) -> None:
        """Pause the show while listening, so its dialogue can't be mistaken for the user talking."""
        if self.running() and not self.get("pause"):
            self.set("pause", True)
            self._held = True

    def release(self) -> None:
        """Resume a show paused by hold() (no-op if the user's command changed things meanwhile)."""
        if self._held:
            self._held = False
            if self.running():
                try:
                    self.set("pause", False)
                except (RuntimeError, OSError):
                    pass

    def forget_hold(self) -> None:
        """The user asked to pause/stop: leave it that way."""
        self._held = False

    # ---------- volume ducking while the assistant speaks ----------
    def duck(self) -> None:
        if self.running() and self._ducked_from is None and not self.get("pause"):
            vol = self.get("volume")
            if vol is not None:
                self._ducked_from = vol
                self.set("volume", vol * 0.25)

    def unduck(self) -> None:
        if self._ducked_from is not None:
            try:
                self.set("volume", self._ducked_from)
            except (RuntimeError, OSError):
                pass
            self._ducked_from = None

    # ---------- progress tracking ----------
    def _episode_for(self, path: str | None) -> Episode | None:
        if not path:
            return None
        try:
            rel = os.path.relpath(path, self.lib.base)
        except ValueError:
            return None
        return self.lib.by_path(rel)

    def _save_progress(self) -> None:
        ep = self._episode_for(self.get("path"))
        pos, dur = self.get("time-pos"), self.get("duration")
        if ep and pos is not None and dur:
            self.lib.record_progress(ep.path, pos, dur)

    def _track_progress(self) -> None:
        last_path, last_pos, last_dur = None, None, None
        while True:
            time.sleep(5)
            if self.running():
                path, pos, dur = self.get("path"), self.get("time-pos"), self.get("duration")
                if last_path and path != last_path and last_dur:
                    # moved on to the next file: the previous one ran to (near) its end
                    ep = self._episode_for(last_path)
                    if ep and last_pos / last_dur > 0.8:
                        self.lib.record_progress(ep.path, last_dur, last_dur)
                if path and pos is not None and dur:
                    ep = self._episode_for(path)
                    if ep:
                        self.lib.record_progress(ep.path, pos, dur)
                last_path, last_pos, last_dur = path, pos, dur
            elif last_path:
                # mpv exited on its own; if it was near the end of the file it finished
                ep = self._episode_for(last_path)
                if ep and last_dur and last_pos / last_dur > 0.8:
                    self.lib.record_progress(ep.path, last_dur, last_dur)
                last_path = None
