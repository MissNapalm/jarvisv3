# TV assistant

Voice assistant for a local TV-show library (default `/mnt/T/TV shows` and `/mnt/T/TV`; set `TV_LIBRARY` or repeat `--library`). Shows kept in several folders are merged. A language model understands requests and calls tools: by default the free
local `qwen3:14b` via Ollama on the GPU, or Claude (`claude-opus-5`) when an API key is set. faster-whisper
(`large-v3-turbo`, on the GPU) does speech-to-text; Piper speaks replies; mpv plays episodes.

It only reads the library. Watch history and the index live in `data/library.db`.

## Setup

Linux with PipeWire, mpv and Python 3.10+; an NVIDIA GPU is recommended for Whisper and the local model.

```bash
git clone https://github.com/MissNapalm/jarvisv3.git ~/tv-assistant && cd ~/tv-assistant
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m piper.download_voices --data-dir voices en_US-ryan-high   # the voice (116 MB)
ollama pull qwen3:14b          # free local brain (needs the GPU build of Ollama, e.g. ollama-cuda on Arch)
cp .env.example .env           # optional: speakers, mic, library path, Claude key
```

The Whisper model (~1.6 GB) downloads automatically on first run.

## Run

```bash
~/tv-assistant/tv              # say "Hey Wil, ...", press Enter and talk, or type (free local model)
~/tv-assistant/tv --backend claude   # use Claude instead; key goes in ~/tv-assistant/.env as ANTHROPIC_API_KEY=...
~/tv-assistant/tv --no-wake    # turn off the wake word
~/tv-assistant/tv --wake-word jarvis   # "Hey Jarvis" (openWakeWord) instead of "Hey Wil"
~/tv-assistant/tv --think      # let the local model reason first: a little more reliable, several seconds slower
~/tv-assistant/tv --window     # open the window on start (what wilagent.exe does); it's always at http://localhost:8765
~/tv-assistant/tv --no-popup   # don't pop up a little window on "Hey Wil"
~/tv-assistant/tv --text       # keyboard only
~/tv-assistant/tv --windowed   # mpv in a window instead of fullscreen
~/tv-assistant/tv --silence 1.2   # wait longer before treating a pause as the end of a command (default 0.8 s)
~/tv-assistant/tv --scan       # just re-index and print stats
```

Say **"Hey Wil"** and your request in one go ("Hey Wil, pause"), or "Hey Wil", wait for the chime, then
the request. A rising chime means it heard "Hey Wil"; a short falling ping means it has your request.
"Hey Wil" is spotted by transcribing each stretch of speech with Whisper (it's not one of openWakeWord's
bundled phrases) and it has to open what you say, so "Will you…" or "my friend Bill" don't wake it.
End of speech is detected with Silero VAD, so background noise doesn't keep it listening.

### wilagent.exe and the window

`wilagent.exe` (a shell script, despite the name; put a `.desktop` launcher pointing at it on the desktop or
in `~/.local/share/applications`) starts the assistant in the background with its window, or just opens
the window if it's already running. Closing the window shuts everything down (voice server, head, pop-ups,
mic, mpv; your place in an episode is saved).

The window (http://localhost:8765, a Chromium app window):
- on start, asks which folders are the TV library (last time's are filled in; Add folder… opens KDE's
  folder picker), with a loading bar that shows the file count while scanning and says when the network
  drive stops answering
- shows whether it's ready, listening, thinking or speaking, what you said and what it answered, and what's
  playing; type there, press Talk, or press Stop (or Esc) to cut it off mid-sentence
- 🎙 Calibrate mic: stay quiet a moment, say "Hey Wil, what's on tonight?", and it sets the mic's level
- "Wil's head in the corner": a cut-out head (`data/wiltop.png` + `data/wiljaw.png`, same canvas, kept out
  of git) floating on top of everything, draggable, whose jaw moves with the loudness of each reply
  (`tvassist/face.py`, run with the system Python's PySide6 on XWayland)

It also answers Star Trek trivia and runs a quiz ("quiz me on Star Trek"); for those it thinks before
answering, which is far more accurate with the local model.

Simple playback controls (pause, resume, stop, next/previous, volume, mute, seek, subtitles) are handled
instantly without the model (`tvassist/quick.py`); everything else goes to the model. The library is
re-indexed on every start (~3 s). While you talk or it speaks, mpv's volume is ducked.

## Cloned voice (optional)

Replies can be spoken in a cloned voice with [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox)
(local, GPU, ~0.1-0.5 s per reply; output carries Chatterbox's inaudible AI watermark). Only clone a
voice with its owner's permission. It runs in its own environment because it needs PyTorch:

```bash
python3 -m venv .venv-voice && .venv-voice/bin/python -m pip install chatterbox-tts "setuptools<81"
# a clean 10-30 s recording of the voice, e.g. cut from a longer file:
ffmpeg -ss 94 -t 16 -i recording.wav -ac 1 -ar 24000 data/voice-sample.wav
echo 'TV_VOICE_SAMPLE="'$PWD'/data/voice-sample.wav"' >> .env      # or --voice-sample PATH
```

If it can't start, the assistant falls back to the Piper voice and says why (details in
`data/clone_server.log`).

## Things to say

- "Continue Breaking Bad" / "Play the next episode of Stargate"
- "Play the X-Files episode Squeeze" / "Play season 3 episode 5 of Farscape"
- "Play something random from MASH" / "Surprise me"
- "Pause", "Skip ahead two minutes", "Volume 60", "Next episode", "Subtitles", "Stop"
- "What was I watching?" / "What's new in the library?" / "How big is the library?"
- "Mark season 1 of The Expanse as watched"

## Layout

- `tvassist/library.py`: indexer (filename/folder parsing), search, watch history, next-up logic
- `tvassist/player.py`: mpv over JSON IPC, progress tracking (90%+ counts as watched)
- `tvassist/voice.py`: mic capture with silence detection, Whisper, Piper
- `tvassist/brain.py`: the tools, system prompt, and the Claude brain
- `tvassist/local.py`: the free local brain (Ollama)
- `tvassist/quick.py`: instant handling of simple playback commands
- `tvassist/ui.py`, `tvassist/ui.html`: the live window (local web page, server-sent events)
- `tvassist/face.py`: Wil's talking head; `tvassist/calibrate.py`: the mic check; `wilagent.exe`: the launcher
- `tvassist/report.py`: the "file update" comparison; `snapshot.py`: the daily snapshot job

## File update (are files disappearing?)

Say **"Hey Wil, give me a file update"** (or "are any files missing?", "check the files"). It rescans
the library and compares with the previous day's snapshot: the total, how many files went missing or
appeared, and which shows they belong to. The full list prints on screen and is saved to
`data/file-report-<date>.txt`.

The snapshots come from `snapshot.py`, run daily at 3 AM by a systemd timer (runs after the next boot if
the PC was off; skips, rather than records, a day when the network drive is unreachable). Log:
`data/snapshots.log`. Install:

```bash
sudo cp systemd/tv-snapshot.* /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now tv-snapshot.timer
systemctl list-timers tv-snapshot.timer     # next run
```

## Filename standardization

`standardize_names.py` renames episode files in place to `Show Name - S01E02 - Title.ext` (subtitle/nfo files
sharing the video's name follow along). Folders are never renamed or moved; uncertain files are skipped.

```bash
.venv/bin/python standardize_names.py plan                         # dry run -> rename-backups/plan_*.tsv (+ _skipped.txt)
.venv/bin/python standardize_names.py apply rename-backups/plan_X.tsv
.venv/bin/python standardize_names.py undo  rename-backups/renamed_X.tsv   # restore the old names
```

Every apply writes a snapshot of all filenames plus a line-by-line rename log, in `rename-backups/` and in
`/mnt/T/TV shows/_rename-backup/`. Show names come from the folder, cleaned up; hand fixes go in
`NAME_OVERRIDES` in `tvassist/library.py`.
