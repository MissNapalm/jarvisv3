# TV assistant

Voice assistant for a local TV-show library (default `/mnt/T/TV shows`, set `TV_LIBRARY` or `--library`). A language model understands requests and calls tools: by default the free
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
~/tv-assistant/tv              # say "Hey Jarvis, ...", press Enter and talk, or type (free local model)
~/tv-assistant/tv --backend claude   # use Claude instead; key goes in ~/tv-assistant/.env as ANTHROPIC_API_KEY=...
~/tv-assistant/tv --no-wake    # turn off the "Hey Jarvis" wake word
~/tv-assistant/tv --text       # keyboard only
~/tv-assistant/tv --windowed   # mpv in a window instead of fullscreen
~/tv-assistant/tv --silence 1.2   # wait longer before treating a pause as the end of a command (default 0.8 s)
~/tv-assistant/tv --scan       # just re-index and print stats
```

Say **"Hey Jarvis"** and your request in one go ("Hey Jarvis, pause"); a chime confirms it heard you.
Saying "Hey Jarvis", waiting for the chime and then the request also works. The wake word uses
openWakeWord's bundled model; end of speech is detected with Silero VAD, so background noise doesn't
keep it listening.

Simple playback controls (pause, resume, stop, next/previous, volume, mute, seek, subtitles) are handled
instantly without the model (`tvassist/quick.py`); everything else goes to the model. The library is
re-indexed on every start (~3 s). While you talk or it speaks, mpv's volume is ducked.

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
