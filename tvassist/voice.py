"""Microphone capture (PipeWire), speech-to-text (faster-whisper on GPU), text-to-speech (Piper)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import wave

import numpy as np

RATE = 16000
CHUNK = 1536  # 96 ms of 16-bit mono: three 512-sample windows for the Silero speech detector
CHUNK_SEC = CHUNK / RATE
MIN_SPEECH_CHUNKS = 3
SPEECH_THRESHOLD = 0.35  # Silero probability; below the usual 0.5 so quiet voices still count


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


_vad_model = None


def speech_prob(chunk: np.ndarray) -> float:
    """Probability that a 96 ms chunk contains speech (Silero VAD, bundled with faster-whisper).
    Unlike a loudness threshold, this ignores hiss, hum, music and a hot mic's background level."""
    global _vad_model
    if _vad_model is None:
        from faster_whisper.vad import get_vad_model
        _vad_model = get_vad_model()
    return float(np.max(_vad_model(chunk[-CHUNK:])))


def record_utterance(max_seconds: float = 15.0, silence_after: float = 0.8,
                     start_timeout: float = 6.0, source=None,
                     device: str | None = None) -> np.ndarray | None:
    """Record until the speaker stops talking; float32 16 kHz audio, or None if nobody spoke.

    Reads from `source` (a raw 16 kHz s16 stream: the shared mic stream, or a file in tests),
    or opens the mic itself when no source is given.
    """
    proc = None if source is not None else subprocess.Popen(
        ["pw-record", "--raw", *(["--target", device] if device else []),
         "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stream = source if source is not None else proc.stdout
    frames, speaking, silent_for, waited, run = [], False, 0, 0, 0
    # count whole chunks rather than summing float seconds (8 x 0.1 < 0.8)
    end_after, give_up, max_chunks = (round(x / CHUNK_SEC) for x in (silence_after, start_timeout, max_seconds))
    try:
        while True:
            raw = stream.read(CHUNK * 2)
            if len(raw) < CHUNK * 2:
                break
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            frames.append(chunk)
            is_speech = speech_prob(chunk) >= SPEECH_THRESHOLD
            run = run + 1 if is_speech else 0
            # speech only "starts" after ~0.3 s in a row, so a leftover syllable (the tail of
            # "Jarvis") or a click can't start and end a recording on its own
            if speaking and is_speech or run >= MIN_SPEECH_CHUNKS:
                speaking, silent_for = True, 0
            elif speaking:
                silent_for += 1
                if silent_for >= end_after:
                    break
            else:
                waited += 1
                if waited >= give_up:
                    return None
                frames = frames[-(4 + MIN_SPEECH_CHUNKS):]  # keep a little pre-roll only
            if len(frames) >= max_chunks:
                break
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()
    return np.concatenate(frames) if speaking and frames else None


# "Hey Wil" (and the ways Whisper spells it), or "Hey Jarvis". "Wil" needs the "hey": a command can
# start with "Will you ...".
WAKE_PHRASE = re.compile(r"^\W*(?:(?:(?:hey|hi|ok|okay|a)\W+)?(?:jarvis|jarvus|travis|jervis)"
                         r"|(?:hey|hay|hi|ok|okay)\W*(?:wil+|wills?|wheel|bill|phil))\b\W*", re.I)


def strip_wake_phrase(text: str) -> str:
    """Drop a leading "Hey Wil," / "Hey Jarvis," that leaked into the recorded command."""
    return WAKE_PHRASE.sub("", text).strip()


def _bells(path: str, notes, seconds: float, decay: float, level: float) -> str:
    """Write overlapping bell tones, notes = [(start s, Hz), ...], to a WAV file."""
    sr = 22050
    audio = np.zeros(int(seconds * sr))
    t = np.arange(len(audio)) / sr
    for start, f in notes:
        tt = np.clip(t - start, 0, None)
        env = np.minimum(1, tt / 0.004) * np.exp(-tt * decay) * (t >= start)  # quick attack, gentle ring-out
        bell = (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 8)
                + 0.08 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt * 12))  # bell-like overtones
        audio += bell * env
    audio *= np.minimum(1, (len(audio) - np.arange(len(audio))) / (0.05 * sr))  # no click at the end
    audio = (audio / np.abs(audio).max() * level * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return path


def make_chime(path: str) -> str:
    """'Hey Jarvis' heard: a soft rising G major arpeggio (G5, B5, D6)."""
    return _bells(path, [(0.0, 783.99), (0.07, 987.77), (0.14, 1174.66)], seconds=0.75, decay=6, level=0.5)


def make_heard_ping(path: str) -> str:
    """Your request heard: a short falling pair (D6, G5), the wake chime's answer."""
    return _bells(path, [(0.0, 1174.66), (0.08, 783.99)], seconds=0.45, decay=10, level=0.4)


class MicListener:
    """Owns the microphone: listens for "Hey Jarvis" and records commands.

    Results go on `events` as ("speech", audio_or_None). A recording starts either when the wake
    word is heard (while armed) or when `request_recording()` is called (the Enter key).
    """
    WAKE_FRAME = 1280  # 80 ms, what openWakeWord expects

    def __init__(self, events, device: str | None = None, silence_after: float = 0.8,
                 wake: bool = True, wake_threshold: float = 0.4, on_wake=None, source=None,
                 wake_word: str = "wil", transcriber: "Transcriber | None" = None):
        """wake_word "jarvis" uses openWakeWord's model; any other ("wil") is spotted by transcribing
        what you say with `transcriber` and checking it starts with the wake phrase."""
        import threading
        self.events, self.silence_after, self.on_wake = events, silence_after, on_wake
        self.wake_threshold = wake_threshold
        self.model = None
        self.transcriber = transcriber if wake and wake_word != "jarvis" else None
        self.wakes = wake and (self.transcriber is not None or wake_word == "jarvis")
        if wake and wake_word == "jarvis":
            import warnings
            import openwakeword
            from openwakeword.model import Model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = Model(wakeword_model_paths=[os.path.join(
                    os.path.dirname(openwakeword.__file__), "resources/models/hey_jarvis_v0.1.onnx")])
        self.armed = threading.Event()
        self.armed.set()
        self._cooldown = 0
        self._reset_pending = False
        self._record = threading.Event()
        # `source` (a raw 16 kHz s16 stream) replaces the mic in tests
        self.stream = source if source is not None else subprocess.Popen(
            ["pw-record", "--raw", *(["--target", device] if device else []),
             "--rate", str(RATE), "--channels", "1", "--format", "s16", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        threading.Thread(target=self._run_transcribed if self.transcriber else self._run, daemon=True).start()

    COOLDOWN_FRAMES = 20  # 1.6 s: the model's feature buffer still holds the last "Hey Jarvis"

    def arm(self) -> None:
        """Listen for the wake word again (call once the assistant has finished replying)."""
        self._reset_pending = True  # done on the listener thread: the model isn't thread-safe
        self.armed.set()

    def disarm(self) -> None:
        self.armed.clear()

    def request_recording(self) -> None:
        self.disarm()
        self._record.set()

    def _capture(self) -> None:
        # commands are short; a cap stops background talk or TV dialogue from keeping it recording
        audio = record_utterance(silence_after=self.silence_after, source=self.stream, max_seconds=8)
        self.events.put(("speech", audio))

    def _run(self) -> None:
        stream = self.stream
        while True:
            if self._record.is_set():
                self._record.clear()
                self._capture()
                continue
            raw = stream.read(self.WAKE_FRAME * 2)  # keep draining even when not armed
            if len(raw) < self.WAKE_FRAME * 2:
                self.events.put(("mic_error", "microphone stream ended"))
                return
            if self.model is None or not self.armed.is_set():
                continue
            if self._reset_pending:
                self._reset_pending = False
                self.model.reset()
                self._cooldown = self.COOLDOWN_FRAMES
            score = max(self.model.predict(np.frombuffer(raw, dtype=np.int16)).values())
            if self._cooldown > 0:  # flush old audio out of the model before trusting it again
                self._cooldown -= 1
                continue
            if score >= self.wake_threshold:
                self.disarm()
                if self.on_wake:
                    self.on_wake()
                self._capture()

    WAKE_WINDOW = 3.0  # seconds: the wake phrase has to open what you say

    def _run_transcribed(self) -> None:
        """Wake word by transcription: catch each stretch of speech, and if it starts with "Hey Wil",
        chime right away (checked every ~0.5 s while you talk) and hand the whole utterance over."""
        stream = self.stream
        end_after = round(self.silence_after / CHUNK_SEC)
        window, max_chunks = round(self.WAKE_WINDOW / CHUNK_SEC), round(10 / CHUNK_SEC)
        pre_roll = 4 + MIN_SPEECH_CHUNKS
        frames, speaking, run, silent, woken = [], False, 0, 0, False
        while True:
            if self._record.is_set():
                self._record.clear()
                self._capture()
                frames, speaking, run, woken = [], False, 0, False
                continue
            raw = stream.read(CHUNK * 2)  # keep draining even when not armed
            if len(raw) < CHUNK * 2:
                self.events.put(("mic_error", "microphone stream ended"))
                return
            if not self.armed.is_set() or self._reset_pending:
                self._reset_pending = False
                frames, speaking, run, woken = [], False, 0, False
                continue
            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
            frames.append(chunk)
            is_speech = speech_prob(chunk) >= SPEECH_THRESHOLD
            run = run + 1 if is_speech else 0
            if not speaking:
                if run < MIN_SPEECH_CHUNKS:
                    frames = frames[-pre_roll:]
                    continue
                speaking, silent = True, 0
            silent = 0 if is_speech else silent + 1
            ended = silent >= end_after or len(frames) >= max_chunks
            if not woken and (ended or (len(frames) <= window + pre_roll and len(frames) % 5 == 0)):
                if WAKE_PHRASE.match(self.transcriber.quick(np.concatenate(frames[:window + pre_roll]))):
                    woken = True
                    if self.on_wake:
                        self.on_wake()
            if ended:
                audio = np.concatenate(frames)
                frames, speaking, run = [], False, 0
                if woken:  # otherwise it was just talk (or the TV): drop it and keep listening
                    woken = False
                    self.disarm()
                    self.events.put(("speech", audio))


class Transcriber:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", hotwords: str | None = None):
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model, device=device, compute_type=compute)
        self.hotwords = hotwords
        self.model.transcribe(np.zeros(RATE, dtype=np.float32), language="en")  # warm up CUDA kernels

    # Whisper invents these from near-silence; on a short clip they almost never mean anything
    PHANTOMS = {"thank you", "thanks", "thank you very much", "thanks for watching", "you", "bye", "okay"}

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True,
                                            hotwords=self.hotwords)
        text = " ".join(s.text.strip() for s in segments).strip()
        if len(audio) < 2.5 * RATE and re.sub(r"[^a-z ]", "", text.lower()).strip() in self.PHANTOMS:
            return ""
        return text


    def quick(self, audio: np.ndarray) -> str:
        """Fast, rough transcript (greedy, no VAD pass): good enough to spot the wake phrase."""
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, without_timestamps=True,
                                            condition_on_previous_text=False, hotwords=self.hotwords)
        return " ".join(s.text.strip() for s in segments).strip()


LIP_FPS = 30
on_speech = None  # f(levels, fps): called as a reply starts playing, with its loudness (0-1) per frame


def speech_levels(path: str, fps: int = LIP_FPS) -> list[float]:
    """Loudness of a 16-bit WAV, one value per 1/fps s, scaled so ordinary speech peaks near 1."""
    with wave.open(path) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    step = max(1, sr // fps)
    rms = np.sqrt(np.mean(a[:len(a) // step * step].reshape(-1, step) ** 2, axis=1))
    if not len(rms) or rms.max() == 0:
        return []
    return [round(float(x), 3) for x in np.clip(rms / np.percentile(rms, 95), 0, 1)]


_hushed = threading.Event()  # set by stop_speaking(): say nothing more until the next request
_playing: "subprocess.Popen | None" = None


def stop_speaking() -> None:
    """Cut off whatever is being said, and skip speaking the reply being worked on."""
    _hushed.set()
    proc = _playing
    if proc is not None and proc.poll() is None:
        proc.terminate()
    if on_speech:
        try:
            on_speech([], LIP_FPS)  # close the mouth
        except Exception:
            pass


def allow_speech() -> None:
    """A new request: replies may be spoken again."""
    _hushed.clear()


def play(path: str, device: str | None = None) -> None:
    """Play a WAV (blocking), telling `on_speech` first so a face can move its mouth along."""
    global _playing
    if _hushed.is_set():
        return
    if on_speech:
        try:
            on_speech(speech_levels(path), LIP_FPS)
        except Exception:
            pass
    proc = _playing = subprocess.Popen(["pw-play", *(["--target", device] if device else []), path],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    _, err = proc.communicate()
    _playing = None
    if proc.returncode != 0 and not _hushed.is_set():
        print(f"[speech output failed: {err.strip() or proc.returncode}]")


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
            play(path, self.device)
        finally:
            os.unlink(path)


class CloneSpeaker:
    """Speaks in a cloned voice via Chatterbox Turbo, which runs in its own environment
    (.venv-voice) as a small server process. Falls back to `fallback` (Piper) if it fails.

    Starting it takes ~20 s (PyTorch, the model, the voice sample), so that happens in the background:
    the constructor returns at once, `on_ready(error or None)` is called when it's up (or failed), and
    say() waits for it if it's still starting."""

    def __init__(self, sample_path: str, device: str | None = None, fallback: "Speaker | None" = None,
                 on_ready=None):
        import json
        import threading
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        py = os.path.join(here, ".venv-voice", "bin", "python")
        if not os.path.exists(py):
            raise RuntimeError("voice cloning isn't installed (.venv-voice missing)")
        if not os.path.exists(sample_path):
            raise RuntimeError(f"voice sample not found: {sample_path}")
        self.device, self.fallback, self._json = device, fallback, json
        # its PyTorch ships its own CUDA libraries; the launcher's LD_LIBRARY_PATH (older cuDNN for
        # Whisper) would make it load the wrong ones
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        log_path = os.path.join(here, "data", "clone_server.log")
        self.proc = subprocess.Popen([py, os.path.join(here, "tvassist", "clone_server.py"), sample_path],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log_path, "w"),
                                     text=True, bufsize=1, env=env)
        self.ready, self.error = threading.Event(), None

        def wait_ready():
            while True:  # skip any library chatter until the server says it's ready
                line = self.proc.stdout.readline()
                if not line:
                    with open(log_path) as f:
                        last = [ln.strip() for ln in f if ln.strip()][-1:] or ["no details"]
                    self.error = f"voice cloning server failed to start ({last[0]}; see {log_path})"
                    break
                if line.startswith("{") and json.loads(line).get("ready"):
                    break
            self.ready.set()
            if on_ready:
                on_ready(self.error)
        threading.Thread(target=wait_ready, daemon=True).start()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        self.ready.wait(180)
        if self.error or not self.ready.is_set():
            if self.fallback:
                self.fallback.say(text)
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            self.proc.stdin.write(self._json.dumps({"text": text, "out": path}) + "\n")
            self.proc.stdin.flush()
            reply = {}
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                if line.startswith("{"):
                    reply = self._json.loads(line)
                    break
            if not reply.get("ok"):
                print(f"[cloned voice failed: {reply.get('error', 'server stopped')}]")
                if self.fallback:
                    self.fallback.say(text)
                return
            play(path, self.device)
        finally:
            os.unlink(path)


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[*_`#>|]", "", text)
    text = re.sub(r"\bS(\d{1,2})E(\d{1,3})\b", lambda m: f"season {int(m[1])} episode {int(m[2])}", text, flags=re.I)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()
