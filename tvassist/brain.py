"""Claude as the assistant's brain: tools over the library + player, with conversation memory."""
from __future__ import annotations

import json
import logging
import random
from typing import Literal

import anthropic
from anthropic import beta_tool

from .library import Library, pretty
from .player import Player

MODEL = "claude-opus-5"
# Tool errors are returned to Claude as is_error results; don't also dump tracebacks into the terminal.
logging.getLogger("anthropic.lib.tools._beta_runner").setLevel(logging.CRITICAL)
MAX_HISTORY_MESSAGES = 60

SYSTEM_TEMPLATE = """You are a voice assistant for a home TV-show library, running on the owner's Linux PC. \
You hear the user through speech recognition and your replies are read aloud by text-to-speech.

How to reply:
- Speak like a person: one or two short sentences, no lists, no markdown, no file paths, no episode IDs.
- Say episodes naturally, e.g. "season 2, episode 5, Grilled".
- Speech recognition garbles names ("stargate s g one", "the x files", "mash"). Map what you hear to the closest \
series below and act; only ask when it's genuinely ambiguous.
- When asked to play something, do it right away with the tools rather than asking for confirmation. \
"Continue", "keep watching" or "next episode of X" means continue_series. Plain "next"/"skip" while \
something is playing means player_control next.
- You can only read the library and control playback. You cannot download, rename, move or delete files; say so if asked.
- Watch history only covers what was played through this assistant.

Star Trek trivia:
- You're also a Star Trek expert: every series and film, characters, actors, ships, episodes, alien \
species and behind-the-scenes facts. Answer Star Trek questions in ONE short sentence, spoken style.
- If asked to quiz them or play Star Trek trivia, ask one question at a time and wait for the answer. Only \
ask about well-known, unambiguous facts you are certain of (who played whom, ship names, species, ranks, \
famous episodes), never trick questions. Then say in one sentence whether they got it (with the right answer \
if not) and the score, and ask the next question. Vary the series; don't repeat. Stop when they say so.
- Don't offer to play episodes during trivia unless they ask.
- If you're not sure of a fact, say so rather than guessing; never invent episode numbers.

The library ({n_series} series). Each line is a spoken name, then the exact series key in brackets to pass to tools:
{series_list}"""


def build_tools(lib: Library, player: Player) -> list:
    def resolve(series: str) -> str:
        if series in lib.all_series():
            return series
        matches = lib.match_series(series, limit=1)
        if not matches:
            raise ValueError(f"No series matching '{series}'. Check the library list.")
        return matches[0]

    def queue_from(series: str, start_id: int, n: int = 25):
        eps = lib.series_episodes(series)
        idx = next((i for i, e in enumerate(eps) if e.id == start_id), 0)
        return eps[idx:idx + n]

    def start(ep, position: float = 0.0) -> str:
        player.play(queue_from(ep.series, ep.id), start=position)
        msg = f"Now playing {ep.label()}"
        if position:
            msg += f", resuming at {int(position // 60)} minutes"
        return msg + "."

    @beta_tool
    def find_series(name: str) -> str:
        """Look up series in the library by (possibly misheard) name. Returns the best matches with
        seasons, episode counts, how many are watched, and what episode is next up.

        Args:
            name: Series name as heard, or a series key from the library list.
        """
        out = []
        for s in lib.match_series(name) if name not in lib.all_series() else [name]:
            eps = lib.series_episodes(s)
            watched = lib.watched_paths(s)
            nxt, pos = lib.next_up(s)
            seasons = sorted({e.season for e in eps if e.season is not None})
            out.append({"series": s, "name": pretty(s), "episodes": len(eps), "seasons": seasons,
                        "watched": len(watched), "next_up": nxt.label() if nxt else "finished",
                        "next_up_episode_id": nxt.id if nxt else None})
        return json.dumps(out or "no matches")

    @beta_tool
    def list_episodes(series: str, season: int | None = None) -> str:
        """List the episodes of a series (optionally one season) with their IDs and watched status.

        Args:
            series: Series key from the library list.
            season: Season number to restrict to.
        """
        s = resolve(series)
        watched = lib.watched_paths(s)
        eps = [e for e in lib.series_episodes(s) if season is None or e.season == season]
        rows = [{"id": e.id, "s": e.season, "e": e.episode, "title": e.title,
                 "watched": e.path in watched} for e in eps[:80]]
        return json.dumps({"series": s, "count": len(eps), "episodes": rows,
                           "truncated": len(eps) > 80})

    @beta_tool
    def search_episodes(query: str, season: int | None = None, episode: int | None = None) -> str:
        """Search every file in the library by words in the show name, folder or episode title.
        Good for "the one where...", episode titles, or specials/movies inside a show folder.

        Args:
            query: Words to match (all must appear), e.g. "breaking bad ozymandias" or "x files pilot".
            season: Only this season number.
            episode: Only this episode number.
        """
        eps = lib.search(query, season, episode)
        return json.dumps([{"id": e.id, "episode": e.label(), "series": e.series} for e in eps] or "no results")

    @beta_tool
    def play_episode(episode_id: int, from_start: bool = False) -> str:
        """Play an episode; the following episodes of the series are queued after it.
        Resumes where the user left off unless from_start is true.

        Args:
            episode_id: Episode ID from another tool's results.
            from_start: Ignore saved progress and start at the beginning.
        """
        ep = lib.episode(episode_id)
        if ep is None:
            raise ValueError("Unknown episode id.")
        prog = lib.progress(ep.path)
        pos = 0.0 if from_start or prog is None or prog["watched"] else max(0.0, prog["position"] - 5)
        return start(ep, pos if pos > 30 else 0.0)

    @beta_tool
    def continue_series(series: str) -> str:
        """Continue watching a series: resume the in-progress episode or play the next unwatched one.

        Args:
            series: Series key from the library list.
        """
        s = resolve(series)
        ep, pos = lib.next_up(s)
        if ep is None:
            return f"The user has finished {pretty(s)}; nothing left to play."
        return start(ep, pos)

    @beta_tool
    def play_random(series: str | None = None) -> str:
        """Play a random episode, preferring unwatched ones. Without a series, picks from the whole library.

        Args:
            series: Series key to pick from; omit for anything in the library.
        """
        pool = lib.series_episodes(resolve(series)) if series else lib.search("", limit=100000)
        watched = {p for s in {e.series for e in pool} for p in lib.watched_paths(s)} if series else set()
        fresh = [e for e in pool if e.path not in watched] or pool
        return start(random.choice(fresh))

    @beta_tool
    def player_control(action: Literal["pause", "resume", "stop", "next", "previous", "seek", "volume",
                                       "subtitles", "audio_track", "fullscreen"],
                       amount: float | None = None) -> str:
        """Control the video player.

        Args:
            action: pause/resume/stop; next/previous episode in the queue; seek by `amount` seconds
                (negative = back); volume sets volume to `amount` percent (0-130); subtitles cycles
                subtitle tracks; audio_track cycles audio tracks; fullscreen toggles fullscreen.
            amount: Seconds for seek, percent for volume.
        """
        if action == "stop":
            player.stop()
            return "Stopped."
        if not player.running():
            return "Nothing is playing."
        if action in ("pause", "resume"):
            player.set("pause", action == "pause")
        elif action in ("next", "previous"):
            player._cmd("playlist-next" if action == "next" else "playlist-prev")
            player.set("pause", False)
        elif action == "seek":
            player._cmd("seek", amount or 30, "relative")
        elif action == "volume":
            player._ducked_from = None  # an explicit volume change wins over ducking
            player.set("volume", max(0, min(130, amount if amount is not None else 100)))
        elif action == "subtitles":
            player._cmd("cycle", "sub")
            return f"Subtitles: {player.get('current-tracks/sub/lang') or player.get('sid')}"
        elif action == "audio_track":
            player._cmd("cycle", "audio")
            return f"Audio: {player.get('current-tracks/audio/lang') or player.get('aid')}"
        elif action == "fullscreen":
            player._cmd("cycle", "fullscreen")
        return json.dumps(player.now_playing())

    @beta_tool
    def now_playing() -> str:
        """What's playing right now, position, volume and how many episodes are queued after it."""
        return json.dumps(player.now_playing() or "Nothing is playing.")

    @beta_tool
    def continue_watching() -> str:
        """Series the user has been watching recently, most recent first, with the next episode for each."""
        return json.dumps(lib.continue_watching() or "No watch history yet.")

    @beta_tool
    def mark_watched(watched: bool, episode_ids: list[int] | None = None, series: str | None = None,
                     season: int | None = None) -> str:
        """Mark episodes watched or unwatched in the assistant's history (never touches the files).
        Give episode_ids, or a series (optionally with a season) to mark all of it.

        Args:
            watched: True to mark watched, False to mark unwatched.
            episode_ids: Specific episode IDs.
            series: Series key; marks the whole series or the given season.
            season: Season number within the series.
        """
        eps = [lib.episode(i) for i in episode_ids or []]
        if series:
            eps += [e for e in lib.series_episodes(resolve(series)) if season is None or e.season == season]
        eps = [e for e in eps if e]
        lib.set_watched([e.path for e in eps], watched)
        return f"Marked {len(eps)} episodes as {'watched' if watched else 'unwatched'}."

    @beta_tool
    def recently_added(limit: int = 8) -> str:
        """Series with the most recently added files.

        Args:
            limit: How many series to return.
        """
        return json.dumps(lib.recently_added(limit))

    @beta_tool
    def file_update() -> str:
        """Check whether any files disappeared: rescans the library and compares the file count and
        list with the previous day's snapshot. Use for "give me a file update", "are files missing?"."""
        from .report import file_update as run
        import os
        spoken, details = run(lib, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
        return spoken + "\n\n" + details[:3000]

    @beta_tool
    def library_stats() -> str:
        """Totals for the library: series, episodes, size on disk, episodes watched."""
        return json.dumps(lib.stats())

    return [find_series, list_episodes, search_episodes, play_episode, continue_series, play_random,
            player_control, now_playing, continue_watching, mark_watched, recently_added, library_stats,
            file_update]


class Brain:
    def __init__(self, lib: Library, player: Player, effort: str = "low"):
        self.client = anthropic.Anthropic()
        self.tools = build_tools(lib, player)
        self.effort = effort
        self.messages: list = []
        series = lib.all_series()
        self.system = SYSTEM_TEMPLATE.format(
            n_series=len(series), series_list="\n".join(f"- {pretty(s)} [{s}]" for s in series))

    def reset(self) -> None:
        self.messages = []

    def note(self, user_text: str, reply: str) -> None:
        """Record a command handled outside the model so later requests have the context."""
        if self.messages and self.messages[-1]["role"] == "user":
            return  # keep roles alternating; drop the note rather than break the history
        self.messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]

    def _trim(self) -> None:
        # Drop the oldest exchanges, cutting only at a plain user text turn so tool_use/tool_result pairs stay intact.
        while len(self.messages) > MAX_HISTORY_MESSAGES:
            cut = next((i for i, m in enumerate(self.messages[1:], 1)
                        if m["role"] == "user" and isinstance(m["content"], str)), None)
            if cut is None:
                break
            self.messages = self.messages[cut:]

    def ask(self, text: str, on_tool=None) -> str:
        self.messages.append({"role": "user", "content": text})
        runner = self.client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=16000,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        last = None
        for message in runner:
            last = message
            # The runner keeps its own copy of the conversation; mirror it so memory carries across turns.
            self.messages.append({"role": "assistant", "content": message.content})
            if on_tool:
                for block in message.content:
                    if block.type == "tool_use":
                        on_tool(block.name, block.input)
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)
        self._trim()
        if last is None:
            return "Sorry, I didn't get a response."
        if last.stop_reason == "refusal":
            return "Sorry, I can't help with that one."
        return " ".join(b.text for b in last.content if b.type == "text").strip()
