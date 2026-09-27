"""Instant handling of simple playback commands, skipping the language model.

Only whole-utterance matches count ("pause", "volume 40", "skip ahead two minutes"); anything
else, or anything said while nothing is playing, returns None and goes to the model.
"""
from __future__ import annotations

import re

from .player import Player

FILLER = re.compile(r"^(?:(?:hey|ok|okay|please|can you|could you|would you|will you|go ahead and|just|and)\s+)+|"
                    r"(?:\s+(?:please|thanks|thank you|now|for me|it|the video|the show|the episode))+$")
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
           "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "half a": 0.5}
NUM = r"(\d+|" + "|".join(sorted(map(re.escape, NUMBERS), key=len, reverse=True)) + r")"


def _num(s: str) -> float:
    return float(s) if s[0].isdigit() else NUMBERS[s]


def normalize(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = re.sub(r"[^\w\s%]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    prev = None
    while prev != t:
        prev, t = t, FILLER.sub("", t).strip()
    return t


def handle(text: str, player: Player) -> str | None:
    """Run the command if it's a simple playback control; return the spoken reply, else None."""
    if not player.running():
        return None
    t = normalize(text)

    def seek(amount: float) -> str:
        player._cmd("seek", amount, "relative")
        secs = abs(int(amount))
        span = f"{secs // 60} minute{'s' if secs // 60 != 1 else ''}" if secs % 60 == 0 else f"{secs} seconds"
        return f"{'Forward' if amount > 0 else 'Back'} {span}."

    if re.fullmatch(r"(pause|pause (it|this|that|playback)|hold on|wait|hang on)", t):
        player.set("pause", True)
        return "Paused."
    if re.fullmatch(r"(resume|unpause|play|continue|keep (going|playing)|go|carry on)", t):
        player.set("pause", False)
        return "Resuming."
    if re.fullmatch(r"(stop|stop (playing|playback|watching)|turn (it )?off|close (it|the player))", t):
        player.stop()
        return "Stopped."
    if re.fullmatch(r"(next|next (episode|one)|skip|skip (this )?(episode|one)|play the next (episode|one))", t):
        player._cmd("playlist-next")
        player.set("pause", False)
        return "Next episode."
    if re.fullmatch(r"(previous|previous (episode|one)|last episode|go back an episode|play the previous (episode|one))", t):
        player._cmd("playlist-prev")
        player.set("pause", False)
        return "Previous episode."
    if re.fullmatch(r"(mute|mute (the )?(sound|audio))", t):
        player.set("mute", True)
        return "Muted."
    if re.fullmatch(r"(unmute|unmute (the )?(sound|audio)|sound on)", t):
        player.set("mute", False)
        return "Unmuted."
    if re.fullmatch(r"(toggle )?(fullscreen|full screen)", t):
        player._cmd("cycle", "fullscreen")
        return "Okay."
    if re.fullmatch(r"(subtitles|subs|captions)( on| off)?|(turn|switch) (on|off) (the )?(subtitles|subs|captions)|"
                    r"(next|change) subtitles?", t):
        if re.search(r"\boff\b", t):
            player.set("sid", "no")
            return "Subtitles off."
        player._cmd("cycle", "sub")
        return "Subtitles on." if player.get("sid") not in (None, False, "no") else "Subtitles off."

    m = re.fullmatch(r"(?:set )?(?:the )?volume (?:(?:up|down) )?(?:to |at )?" + NUM + r"(?: percent|%)?", t)
    if m:
        player._ducked_from = None  # an explicit volume wins over ducking
        vol = max(0, min(130, _num(m[1])))
        player.set("volume", vol)
        return f"Volume {int(vol)}."
    m = re.fullmatch(r"(volume up|louder|turn (it )?up|(a bit |a little )?louder)|(volume down|quieter|softer|turn (it )?down)", t)
    if m:
        base = player._ducked_from if player._ducked_from is not None else (player.get("volume") or 100)
        vol = max(0, min(130, base + (10 if m[1] else -10)))
        player._ducked_from = None
        player.set("volume", vol)
        return f"Volume {int(vol)}."

    m = re.fullmatch(r"(?:skip|jump|go|fast forward|move)(?: ahead| forward)?(?: by)? " + NUM + r" (second|minute)s?", t) or \
        re.fullmatch(r"forward " + NUM + r" (second|minute)s?", t)
    if m:
        return seek(_num(m[1]) * (60 if m[2] == "minute" else 1))
    m = re.fullmatch(r"(?:go back|back|rewind|skip back|jump back)(?: by)? " + NUM + r" (second|minute)s?", t)
    if m:
        return seek(-_num(m[1]) * (60 if m[2] == "minute" else 1))
    if re.fullmatch(r"(skip|jump|go) (ahead|forward)|fast forward", t):
        return seek(30)
    if re.fullmatch(r"(go back|rewind|back up|skip back|jump back)", t):
        return seek(-10)
    return None
