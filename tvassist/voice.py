"""Microphone capture (PipeWire), speech-to-text (faster-whisper on GPU), text-to-speech (Piper)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import wave

import numpy as np

RATE = 16000
CHUNK = 1600  # 100 ms of 16-bit mono


def find_device(query: str, kind: str) -> tuple[str, str]:
    """Resolve a PipeWire device by (part of) its name or description.
    kind is "Audio/Sink" (speakers) or "Audio/Source" (mics). Returns (node.name, description)."""
    import json
    nodes = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, check=True).stdout)
    q = query.lower()
    for n in nodes:
        props = (n.get("info") or {}).get("props") or {}
        if props.get("media.class") != kind:
            continue
        name, desc = props.get("node.name", ""), props.get("node.description", "")
        if q == name.lower() or q in desc.lower() or q in name.lower():
            return name, desc
    raise ValueError(f"No {'speaker' if kind == 'Audio/Sink' else 'microphone'} matching {query!r}")


def record_utterance(max_seconds: float = 15.0, silence_after: float = 0.8,
                     start_timeout: float = 6.0, on_level=None, source=None,
                     device: str | None = None) -> np.ndarray | None:
    """Record from the default mic until the speaker stops talking.

    Uses a simple energy detector calibrated on the first 300 ms of room noise.
    Returns float32 audio at 16 kHz, or None if nobody spoke.
    """
    # `source` lets tests feed raw 16 kHz s16 audio from a file-like object instead of the mic.
    proc = None if source is not None else subprocess.Popen(
        ["pw-record", *(["--target", device] if device else []),
         "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stream = source if source is not None else proc.stdout
    frames, noise, speaking, silent_for, waited = [], [], False, 0, 0
    try:
        while True:
            raw = stream.read(CHUNK * 2)
            if not raw:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
            frames.append(chunk)
            if len(noise) < 3:
                noise.append(rms)
                continue
            threshold = max(0.012, np.median(noise) * 3.0)
            if on_level:
                on_level(rms, threshold)
            # count whole 100 ms chunks: summing 0.1 floats drifts (8 x 0.1 < 0.8)
            if rms > threshold:
                speaking, silent_for = True, 0
            elif speaking:
                silent_for += 1
                if silent_for >= round(silence_after * 10):
                    break
            else:
                waited += 1
                if waited >= round(start_timeout * 10):
                    return None
                noise.append(rms)
                noise = noise[-20:]
                frames = frames[-5:]  # keep a little pre-roll only
            if len(frames) * 0.1 >= max_seconds:
                break
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
    return np.concatenate(frames) if speaking and frames else None


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", hotwords: str | None = None):
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model, device=device, compute_type=compute)
        self.hotwords = hotwords
        self.model.transcribe(np.zeros(RATE, dtype=np.float32), language="en")  # warm up CUDA kernels

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True,
                                            hotwords=self.hotwords)
        return " ".join(s.text.strip() for s in segments).strip()


class Speaker:
    def __init__(self, voice_path: str, device: str | None = None):
        from piper import PiperVoice
        self.voice = PiperVoice.load(voice_path)
        self.device = device

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with wave.open(path, "wb") as w:
                self.voice.synthesize_wav(text, w)
            r = subprocess.run(["pw-play", *(["--target", self.device] if self.device else []), path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[speech output failed: {r.stderr.strip() or r.returncode}]")
        finally:
            os.unlink(path)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[*_`#>|]", "", text)
    text = re.sub(r"\bS(\d{1,2})E(\d{1,3})\b", lambda m: f"season {int(m[1])} episode {int(m[2])}", text, flags=re.I)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()
