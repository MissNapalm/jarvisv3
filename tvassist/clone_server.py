"""Voice-cloning speech server, run with the separate .venv-voice (PyTorch + Chatterbox).

The assistant starts it once; it loads Chatterbox Turbo, prepares the voice from a sample
recording, then answers one JSON request per line on stdin:
    {"text": "...", "out": "/path/to/output.wav"}  ->  {"ok": true, "seconds": 0.8}
Output audio carries Chatterbox's built-in (inaudible) AI watermark.
"""
import json
import sys
import time
import wave

import numpy as np


def main() -> None:
    sample = sys.argv[1]
    # stdout carries the protocol; send the libraries' own prints/progress bars to stderr
    out, sys.stdout = sys.stdout, sys.stderr

    def reply(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    import torch
    import chatterbox.tts_turbo as turbo
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    # newer librosa/numpy return float64 from resample; the model's STFT needs float32
    _resample = turbo.librosa.resample
    turbo.librosa.resample = lambda *a, **k: _resample(*a, **k).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    model.prepare_conditionals(sample)  # analyse the voice once, reuse it for every reply
    model.generate("Ready.")  # warm up
    reply({"ready": True, "device": device, "sample_rate": model.sr})

    for line in sys.stdin:
        try:
            req = json.loads(line)
            t = time.time()
            wav = model.generate(req["text"])
            audio = np.clip(wav.squeeze(0).cpu().numpy(), -1, 1)
            with wave.open(req["out"], "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(model.sr)
                w.writeframes((audio * 32767).astype(np.int16).tobytes())
            reply({"ok": True, "seconds": round(time.time() - t, 2)})
        except Exception as e:  # report and keep serving
            reply({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
