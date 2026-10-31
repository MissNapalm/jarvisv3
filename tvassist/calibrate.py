"""Mic check (the window's "Calibrate mic" button): measure the room, have you say the wake phrase, and set the mic's level.

The window shows each step with a live level meter (ui "calibrate" messages); the terminal gets the same
instructions. PipeWire's mic volume is cubic (twice the setting is ~8x the signal), so the change is the
cube root of the gain needed.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time

import numpy as np

from . import voice

TARGET_PEAK = 0.5            # loudest moments of your voice, as a share of full scale (~ -6 dB): loud, no clipping
MIN_VOL, MAX_VOL = 0.15, 1.0
QUIET_SECONDS = 1.5
PHRASE = "Hey Wil, what's on tonight?"


class Skipped(Exception):
    pass


def source_id(node_name: str | None) -> str:
    """wpctl wants a node id; find the mic's (or use the default mic)."""
    if node_name:
        for n in json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True).stdout):
            if ((n.get("info") or {}).get("props") or {}).get("node.name") == node_name:
                return str(n["id"])
    return "@DEFAULT_AUDIO_SOURCE@"


def get_volume(sid: str) -> float:
    out = subprocess.run(["wpctl", "get-volume", sid], capture_output=True, text=True).stdout  # "Volume: 0.60"
    try:
        return float(out.split()[1])
    except (IndexError, ValueError):
        return 1.0


def set_volume(sid: str, v: float) -> None:
    subprocess.run(["wpctl", "set-volume", sid, f"{v:.2f}"], capture_output=True)


class _Tap:
    """The mic stream, reporting its level to the window as it's read (and noticing Skip)."""

    def __init__(self, stream, report, skip: threading.Event):
        self.stream, self.report, self.skip, self._last = stream, report, skip, 0.0

    def read(self, n: int) -> bytes:
        if self.skip.is_set():
            raise Skipped
        raw = self.stream.read(n)
        now = time.time()
        if raw and now - self._last > 0.08:
            self._last = now
            a = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768
            self.report(float(np.abs(a).max()) if len(a) else 0.0)
        return raw


class MicCheck:
    def __init__(self, ui, mic: str | None, transcriber, chime=None):
        self.ui, self.mic, self.transcriber, self.chime = ui, mic, transcriber, chime or (lambda: None)
        self.skip = threading.Event()  # the window's Skip button

    def _show(self, step: str, text: str, **extra) -> None:
        self.ui.calibrate(step, text, **extra)
        if step != "level":
            print(f"  [mic check] {text}", flush=True)

    def run(self, tries: int = 3) -> None:
        sid = source_id(self.mic)
        proc = subprocess.Popen(["pw-record", "--raw", *(["--target", self.mic] if self.mic else []),
                                 "--rate", str(voice.RATE), "--channels", "1", "--format", "s16", "-"],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        tap = _Tap(proc.stdout, lambda lv: self.ui.calibrate("level", "", level=round(lv, 3)), self.skip)
        try:
            self._run(tap, sid, tries)
        except Skipped:
            self._show("done", "Mic check skipped.", ok=True)
        finally:
            proc.terminate()
            proc.wait()

    def _run(self, tap: _Tap, sid: str, tries: int) -> None:
        self._show("quiet", "Mic check: stay quiet for a moment…")
        raw = b"".join(tap.read(voice.CHUNK * 2) for _ in range(round(QUIET_SECONDS / voice.CHUNK_SEC)))
        noise = float(np.frombuffer(raw, dtype=np.int16).astype(np.float32).std() / 32768)

        for attempt in range(1, tries + 1):
            self.chime()
            self._show("speak", f"Now say: “{PHRASE}”", attempt=attempt)
            audio = voice.record_utterance(source=tap, start_timeout=8, max_seconds=8, silence_after=0.8)
            if audio is None:
                self._show("retry", "I didn't hear anything. Check the mic is on, then try again.")
                continue

            # how loud your voice came in, ignoring a stray click: the 99.5th percentile of the samples
            peak = float(np.percentile(np.abs(audio), 99.5))
            vol = get_volume(sid)
            new = vol
            if peak > 0:
                new = min(MAX_VOL, max(MIN_VOL, vol * (TARGET_PEAK / peak) ** (1 / 3)))
            if abs(new - vol) >= 0.03:
                set_volume(sid, new)

            heard = self.transcriber(audio)
            woke = bool(voice.WAKE_PHRASE.match(heard))
            level = f"Mic level {round(new * 100)}%" + (f" (was {round(vol * 100)}%)" if abs(new - vol) >= 0.03 else "")
            speech = float(audio.std())
            noisy = noise > 0 and speech / noise < 4
            if woke:
                self._show("done", f"All set. I heard “{heard}”. {level}."
                           + (" It's a bit noisy in there, so speak up." if noisy else ""), ok=True)
                return
            if attempt < tries:
                self._show("retry", f"I heard “{heard or '…'}” but not “Hey Wil”. {level}. Let's try once more.")
                time.sleep(1.5)
        self._show("done", "Couldn't confirm “Hey Wil”, but I've set the mic as best I can. "
                           "You can also press Talk or type.", ok=False)
