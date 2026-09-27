"""Library index + watch history, stored in SQLite.

The media library itself is treated as read-only: nothing here writes, renames
or deletes anything under the library root.
"""
from __future__ import annotations

import difflib
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mkv", ".avi", ".mp4", ".m4v", ".divx", ".mpg", ".mpeg", ".wmv",
              ".ts", ".webm", ".mov", ".ogm", ".flv", ".vob"}
_EXTRA = (r"extras?|featurettes?|deleted[ ._-]?scenes?|behind[ ._-]the[ ._-]scenes|bonus|specials?[ ._-]features?|"
          r"gag[ ._-]?reels?|bloopers?|interviews?|trailers?|xtras|webisodes?|commentar(?:y|ies)")
# A folder counts as extras only when its name starts with one of these words, so
# "Show Season 1-4 + Extras BDRip" is still an episode folder.
EXTRA_DIR = re.compile(rf"^[\d\s._-]*(?:{_EXTRA})\b", re.I)
# In a filename only unambiguous words count ("The Interview" can be a real episode title).
EXTRA_FILE = re.compile(r"\b(extras?|featurettes?|deleted[ ._-]?scenes?|behind[ ._-]the[ ._-]scenes|gag[ ._-]?reel|bloopers?)\b", re.I)
SAMPLE = re.compile(r"(^|[ ._-])sample([ ._-]|$)", re.I)

# Filename episode patterns, tried in order. Each yields (season, episode).
EP_PATTERNS = [
    re.compile(r"\bS(?:E)?(\d{1,2})[ ._-]*x?[ ._-]*E(?:P)?(\d{1,3})", re.I),        # S01E02, SE02xEP05
    re.compile(r"\bSeason[ ._-]*(\d{1,2})[ ._-]*Episode[ ._-]*(\d{1,3})", re.I),    # Season 4 Episode 07
    re.compile(r"(?<![\d])(\d{1,2})x(\d{2,3})(?![\d])", re.I),                        # 1x02
    re.compile(r"(?<![\w])(\d{1,2})e(\d{2})(?![\d])", re.I),                            # 6e03
]
SEASON_DIR = re.compile(r"(?:^|[^a-z])(?:season|series|staffel|saison)[ ._-]*(\d{1,2})(?![\d])(?![ ]*[-&,][ ]*\d)", re.I)
SEASON_DIR_SHORT = re.compile(r"(?:^|[ ._])S(\d{1,2})(?:E\d{0,3}\w*)?(?=[ ._-]|$)(?![ ._]*-[ ._]*S?\d)", re.I)
EP_ONLY = [
    re.compile(r"^(?:ep|e|episode)[ ._-]*(\d{1,3})\b", re.I),                          # Ep-05, e22
    re.compile(r"[ ._-](?:ep|e|episode)[ ._-]*(\d{1,3})\b", re.I),
    re.compile(r"^(\d{1,3})(?:[ ._-]|$|(?=[A-Z][a-z]))"),                              # 03 Jailbait, 12Cyber_Hunt
]
THREE_DIGIT = re.compile(r"[ ._-](\d)(\d{2})(?:[ ._-]|$)")                              # China Beach 315 -
GLUED = re.compile(r"[a-z]0?(\d{1,2})(\d{2})r?$", re.I)                                  # tpz-greysa321


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[\[\(].*?[\]\)]", " ", s)  # drop bracketed release junk
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


RELEASE_JUNK = re.compile(
    r"(\b(season|seasons|series)\b[ ._]*\d.*$|\bS\d{1,2}([ ._-]*-?[ ._-]*S?\d{1,2})?\b.*$|\b(complete|deluxe|dvd|boxset|"
    r"bluray|blu-ray|bdrip|dvdrip|webrip|web-dl|hdtv|x26[45]|hevc|\d{3,4}p|xvid|divx)\b.*$|[+&]\s*extras.*$)", re.I)


# Hand-picked canonical names where folder names are misspelled, shouty or ambiguous (keyed by the cleaned name).
NAME_OVERRIDES = {
    "21 Jump Street.(1987)": "21 Jump Street", "ARROW": "Arrow", "Consantine 2014": "Constantine",
    "DAREDEVIL BORN AGAIN": "Daredevil Born Again", "Dark Matter 2015-2017": "Dark Matter (2015)",
    "Dark Matter 2024": "Dark Matter (2024)", "DAY BREAK": "Day Break", "DEXTER RESURRECTION": "Dexter Resurrection",
    "Dr Who": "Doctor Who", "Euphoria 2019": "Euphoria", "Heroes All Seasons": "Heroes",
    "Heroes Reborn Dark Matters 01-06": "Heroes Reborn Dark Matters", "HOUSE MD": "House M.D.",
    "INCORPORATED -(2016)": "Incorporated", "Knight Rider 1982": "Knight Rider (1982)",
    "Knight Rider 2008": "Knight Rider (2008)", "LEGO MASTERS": "LEGO Masters", "Lexx Seasons": "Lexx",
    "Lost In Space 2018-2021": "Lost in Space (2018)", "MADAM SECRETARY": "Madam Secretary",
    "Marvels Agents of S.H.I.E.L.D.": "Marvel's Agents of S.H.I.E.L.D.", "MIAMI VICE": "Miami Vice",
    "Monarch꞉ Legacy of Monsters (2023)": "Monarch - Legacy of Monsters", "neXt 2020": "neXt",
    "Prison Break 2 0": "Prison Break", "Prison Break (2005)": "Prison Break", "Quantum Leap 2 0": "Quantum Leap (2022)",
    "Reaper 2007": "Reaper", "Seaquest": "seaQuest DSV", "Seaquest DSV": "seaQuest DSV", "SEE": "See",
    "Shogun 2024": "Shogun (2024)", "SOAP": "Soap", "TekwarTV": "TekWar", "The Twilight Zone 1959": "The Twilight Zone (1959)",
    "V 2009": "V (2009)", "War of the Worlds 1988-1990": "War of the Worlds (1988)", "Whiz Kids - TV Series": "Whiz Kids",
    "X-men 97": "X-Men '97", "Star Wars - The Clone Wars": "Star Wars The Clone Wars", "Skeleton Crew": "Star Wars Skeleton Crew",
    "Cyberpunk - Edgerunners": "Cyberpunk Edgerunners", "Babylon 5: B5 Crusade 1-13": "Crusade",
    "Bionic Woman": "Bionic Woman (2007)", "Law and Order": "Law & Order SVU", "Mr Robot": "Mr. Robot",
    "Euphoria": "Euphoria", "Sense8 (2015)": "Sense8", "The Guild (2007)": "The Guild",
    # /mnt/T/TV
    "Star Trek": "Star Trek The Original Series", "Quantum Leap 1989": "Quantum Leap", "Quantum Leap 2022": "Quantum Leap (2022)",
    "Night Court (1984)": "Night Court", "Night Court 1984": "Night Court", "Night Court 2023": "Night Court (2023)",
    "Through the Wormhole With Morgan Freeman": "Through the Wormhole", "Cosmos 1980": "Cosmos (1980)",
    "Cosmos 2019": "Cosmos Possible Worlds", "Cosmos A Space Time Odyssey": "Cosmos A Spacetime Odyssey",
    "Animaniacs 2020": "Animaniacs (2020)", "11 22 63 2016": "11.22.63", "Nana 2006": "Nana",
}

SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "with"}


def title_case(s: str) -> str:
    """Capitalise all-lowercase words ("the walking dead" -> "The Walking Dead"); leave anything already cased."""
    words = s.split(" ")
    out = []
    for i, w in enumerate(words):
        if w.islower() and (i == 0 or w not in SMALL_WORDS):
            w = w[0].upper() + w[1:]
        out.append(w)
    return " ".join(out)


def pretty(series: str) -> str:
    """A speakable name for a series folder path ('.../Star Trek Voyager The Complete series' -> 'Star Trek Voyager')."""
    leaf = series.split("/")[-1]
    s = re.sub(r"\[.*?\]|\{.*?\}", " ", leaf.replace(".(", " ("))
    s = re.sub(r"\((?!(19|20)\d\d\))[^)]*\)", " ", s)       # keep "(2009)"-style years, drop other brackets
    s = s.replace("_", " ")
    # dots are word separators ("Terra.Nova"), except inside acronyms ("M.D.", "S.H.I.E.L.D")
    s = re.sub(r"(?<![A-Za-z])((?:[A-Za-z]\.){2,}[A-Za-z]?\.?)",
               lambda m: m[1].replace(".", "\0") + ("\0" if not m[1].endswith(".") else ""), s)
    s = re.sub(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z0-9])", " ", s).replace("\0", ".")
    s = re.sub(r"\.\.+", ".", s)
    s = RELEASE_JUNK.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -,&+")
    s = s.rstrip(".") if not re.search(r"(?:\b[A-Za-z]\.){2,}$", s) else s
    s = re.sub(r"[\s-]+\bthe$", "", s, flags=re.I).strip(" -") or leaf
    s = title_case(s)
    parts = series.split("/")
    if len(parts) > 1:  # nested: give generic leaves ("Movies", "Specials") their parent's name
        top = pretty(parts[0])
        if not set(norm(top).split()) & set(norm(s).split()):
            s = f"{top}: {s}"
    return NAME_OVERRIDES.get(s, s)


def name_key(name: str) -> str:
    """Compare show names: case and punctuation don't matter, but a year does
    ("Night Court (2023)" != "Night Court"), unlike norm() which drops bracketed text."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower().replace("'", "").replace("’", "")).strip()


def season_of_dir(name: str) -> int | None:
    m = SEASON_DIR.search(name) or SEASON_DIR_SHORT.search(name) or EP_PATTERNS[2].search(name)
    return int(m.group(1)) if m else None  # the last case is a per-episode folder like "Show.1x01.Pilot"


def parse_episode(rel: Path) -> tuple[int | None, int | None, str]:
    """Return (season, episode, title) guessed from a path relative to the show folder."""
    d = parse_episode_detail(rel)
    return d["season"], d["episode"], d["title"]


def parse_episode_detail(rel: Path) -> dict:
    """Like parse_episode, plus how the numbers were found:
    method = explicit (S01E02, 1x02, "Season 1 Episode 2"), folder (episode number + season folder),
    three_digit ("Show 315"), glued ("greysa321"), or None."""
    stem = rel.stem
    folder_season = None
    for part in reversed(rel.parts[:-1]):
        folder_season = season_of_dir(part)
        if folder_season is not None:
            break

    season = episode = method = None
    title_start = 0
    for pat in EP_PATTERNS:
        m = pat.search(stem)
        if m:
            season, episode, title_start, method = int(m.group(1)), int(m.group(2)), m.end(), "explicit"
            break
    if episode is None:
        for pat in EP_ONLY:
            m = pat.search(stem)
            if m and not re.match(r"(19|20)\d\d$", m.group(1)):
                season, episode, title_start, method = folder_season, int(m.group(1)), m.end(), "folder"
                break
    if episode is None:
        m = THREE_DIGIT.search(stem)
        if m and not re.search(r"\d{3,4}p", stem[m.start():m.end() + 1], re.I):
            season, episode, title_start, method = int(m.group(1)), int(m.group(2)), m.end(), "three_digit"
    if episode is None:
        m = GLUED.search(stem)
        if m and folder_season == int(m.group(1)):
            season, episode, title_start, method = int(m.group(1)), int(m.group(2)), len(stem), "glued"
    if season is None:
        season = folder_season

    rest = stem[title_start:] if title_start else stem
    title = re.sub(r"[\[\(].*?[\]\)]", " ", rest)
    title = re.sub(r"\b(1080p|720p|480p|2160p|x26[45]|h\.?26[45]|hevc|bluray|web[- ]?dl|webrip|dvdrip|"
                   r"hdtv|bdrip|aac|ac3|dts|10bit|xvid|divx)\b.*$", "", title, flags=re.I)
    title = re.sub(r"[._]+", " ", title).strip(" -_.")
    return {"season": season, "episode": episode, "title": title, "method": method,
            "folder_season": folder_season, "rest": rest}


def series_root(show_dir: str, rel: Path) -> str:
    """The folder a run of episodes belongs to: the parent of its season folder,
    or the file's own folder when there is no season folder. Lets one top-level
    folder (e.g. 'star trek') hold several distinct series."""
    parts = list(rel.parts[:-1])
    i = len(parts)
    while i > 0 and (season_of_dir(parts[i - 1]) is not None or EXTRA_DIR.match(parts[i - 1])):
        i -= 1
    return "/".join([show_dir, *parts[:i]])


@dataclass
class Episode:
    id: int
    path: str
    show: str
    series: str
    season: int | None
    episode: int | None
    title: str
    is_extra: bool

    def label(self) -> str:
        name = pretty(self.series)
        se = ""
        if self.season is not None and self.episode is not None:
            se = f" S{self.season:02d}E{self.episode:02d}"
        elif self.episode is not None:
            se = f" E{self.episode:02d}"
        return f"{name}{se} - {self.title}".strip(" -")


class Library:
    def __init__(self, roots: str | list[str], db_path: str):
        """`roots`: one or more library folders (e.g. "/mnt/T/TV shows" and "/mnt/T/TV").
        Paths are stored relative to the folder that contains them all, e.g. "TV shows/Airwolf/...";
        shows with the same name in several folders are merged into one series."""
        self.roots = [Path(r) for r in ([roots] if isinstance(roots, str) else roots)]
        self.base = Path(os.path.commonpath([str(r.parent) for r in self.roots]))
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes(
                id INTEGER PRIMARY KEY, path TEXT UNIQUE, show TEXT, series TEXT,
                season INTEGER, episode INTEGER, title TEXT, is_extra INTEGER,
                size INTEGER, mtime REAL, search TEXT);
            CREATE INDEX IF NOT EXISTS ep_show ON episodes(show);
            CREATE INDEX IF NOT EXISTS ep_series ON episodes(series);
            CREATE TABLE IF NOT EXISTS history(
                path TEXT PRIMARY KEY, position REAL, duration REAL,
                watched INTEGER DEFAULT 0, last_played REAL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS snapshots(
                day TEXT PRIMARY KEY, taken REAL, total INTEGER, videos INTEGER, paths BLOB);
        """)
        self._migrate_history()

    @property
    def root(self) -> Path:  # the folder paths are relative to
        return self.base

    def _migrate_history(self) -> None:
        """Older databases stored paths relative to a single library folder; prefix them with
        the folder they're actually in."""
        if self.db.execute("SELECT value FROM meta WHERE key='path_layout'").fetchone():
            return
        with self.db:
            for (path,) in self.db.execute("SELECT path FROM history").fetchall():
                for r in self.roots:
                    if (r / path).exists():
                        new = str((r / path).relative_to(self.base))
                        self.db.execute("UPDATE OR IGNORE history SET path=? WHERE path=?", (new, path))
                        break
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('path_layout', 'relative-to-base')")

    # ---------- indexing ----------
    def scan(self) -> dict:
        t0 = time.time()
        rows = []
        all_files: list[str] = []  # every file, not just videos, for the daily file-count snapshot
        for root in self.roots:
            for show_dir in sorted(os.listdir(root)):
                full = root / show_dir
                if show_dir.startswith((".", "_rename-backup")):
                    continue
                if not full.is_dir():
                    all_files.append(str(full.relative_to(self.base)))
                    continue
                for dirpath, dirnames, filenames in os.walk(full):
                    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                    for fn in filenames:
                        if fn.startswith("."):  # hidden temp files (download ".parts" etc.) come and go
                            continue
                        p = Path(dirpath) / fn
                        all_files.append(str(p.relative_to(self.base)))
                        if p.suffix.lower() not in VIDEO_EXTS or SAMPLE.search(p.stem):
                            continue
                        rel = p.relative_to(full)
                        season, ep, title = parse_episode(rel)
                        extra = any(EXTRA_DIR.match(d) for d in rel.parts[:-1]) or bool(EXTRA_FILE.search(p.stem))
                        try:
                            st = p.stat()
                        except OSError:
                            continue
                        path = str(p.relative_to(self.base))
                        rows.append((path, show_dir, series_root(show_dir, rel), season, ep, title,
                                     int(extra), st.st_size, st.st_mtime, norm(path)))
        # one series per show name: merges a show kept in several folders ("TV shows/Airwolf" and
        # "TV/Airwolf", or "Night.Court.S01" ... "Night.Court.S09") into a single run of episodes
        canonical: dict[str, str] = {}
        for r in rows:
            canonical.setdefault(name_key(pretty(r[2])), r[2])
        rows = [(*r[:2], canonical[name_key(pretty(r[2]))], *r[3:]) for r in rows]
        with self.db:
            self.db.execute("DELETE FROM episodes")
            self.db.executemany(
                "INSERT INTO episodes(path,show,series,season,episode,title,is_extra,size,mtime,search)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('scanned_at', ?)", (str(time.time()),))
        if all_files:  # an empty scan means the drive wasn't reachable: never record that as "all gone"
            self.save_snapshot(all_files, videos=len(rows))
        return {"files": len(rows), "shows": len({r[1] for r in rows}), "seconds": round(time.time() - t0, 1),
                "all_files": len(all_files)}

    # ---------- daily file-count snapshots ("give me a file update") ----------
    KEEP_SNAPSHOTS = 90

    def save_snapshot(self, paths: list[str], videos: int, day: str | None = None) -> None:
        """Record today's file list (the latest scan of the day wins)."""
        import zlib
        day = day or time.strftime("%Y-%m-%d")
        blob = zlib.compress("\n".join(sorted(paths)).encode())
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?)",
                            (day, time.time(), len(paths), videos, blob))
            self.db.execute("DELETE FROM snapshots WHERE day NOT IN "
                            "(SELECT day FROM snapshots ORDER BY day DESC LIMIT ?)", (self.KEEP_SNAPSHOTS,))

    def has_snapshot_today(self) -> bool:
        return self.db.execute("SELECT 1 FROM snapshots WHERE day=?", (time.strftime("%Y-%m-%d"),)).fetchone() is not None

    def _snapshot_paths(self, row) -> set[str]:
        import zlib
        return set(zlib.decompress(row["paths"]).decode().split("\n")) if row["paths"] else set()

    def file_update(self) -> dict:
        """Rescan, then compare today's files with the most recent earlier day's snapshot."""
        reachable = [r for r in self.roots if r.is_dir()]
        if len(reachable) < len(self.roots):
            return {"error": "library drive not reachable",
                    "missing_folders": [str(r) for r in self.roots if not r.is_dir()]}
        stats = self.scan()
        if not stats["all_files"]:
            return {"error": "library drive not reachable (no files found)"}
        today = time.strftime("%Y-%m-%d")
        cur = self.db.execute("SELECT * FROM snapshots WHERE day=?", (today,)).fetchone()
        prev = self.db.execute("SELECT * FROM snapshots WHERE day<? ORDER BY day DESC LIMIT 1", (today,)).fetchone()
        report = {"today": today, "total": cur["total"], "videos": cur["videos"]}
        if prev is None:
            report["previous"] = None
            return report
        now_paths, before = self._snapshot_paths(cur), self._snapshot_paths(prev)
        missing, added = sorted(before - now_paths), sorted(now_paths - before)

        def by_show(paths: list[str]) -> list[tuple[str, int]]:
            counts: dict[str, int] = {}
            for p in paths:
                parts = p.split("/")
                show = parts[1] if len(parts) > 2 else "the main folder"
                counts[show] = counts.get(show, 0) + 1
            return sorted(counts.items(), key=lambda kv: -kv[1])

        report.update({"previous": prev["day"], "previous_total": prev["total"],
                       "previous_videos": prev["videos"], "missing": missing, "added": added,
                       "missing_by_show": by_show(missing), "added_by_show": by_show(added)})
        return report

    def is_indexed(self) -> bool:
        return self.db.execute("SELECT 1 FROM episodes LIMIT 1").fetchone() is not None

    # ---------- lookup ----------
    def _ep(self, row) -> Episode:
        return Episode(row["id"], row["path"], row["show"], row["series"], row["season"],
                       row["episode"], row["title"], bool(row["is_extra"]))

    def abspath(self, ep: Episode) -> str:
        return str(self.base / ep.path)

    def shows(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT show FROM episodes ORDER BY show COLLATE NOCASE")]

    def all_series(self) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT DISTINCT series FROM episodes WHERE is_extra=0 ORDER BY series COLLATE NOCASE")]

    def match_series(self, name: str, limit: int = 5) -> list[str]:
        """Fuzzy-match a spoken show name against every series folder. Series whose name contains
        every word of the query come first (all of them, so "star trek" returns every Star Trek show)."""
        q = norm(name)
        if not q:
            return []
        qtok = set(q.split())
        contains, scored = [], []
        for s in self.all_series():
            n = norm(s.replace("/", " ") + " " + pretty(s))
            leaf = norm(pretty(s))
            tok = set(n.split())
            if qtok <= set(leaf.split()):
                contains.append((0 if leaf == q else 1, leaf, s))
                continue
            overlap = len(qtok & tok) / len(qtok)
            ratio = max(difflib.SequenceMatcher(None, q, leaf).ratio(),
                        difflib.SequenceMatcher(None, q, norm(s.split("/")[0])).ratio())
            scored.append((overlap * 0.6 + ratio * 0.4 + (0.3 if q in n else 0), s))
        scored.sort(key=lambda x: -x[0])
        exact = [s for *_, s in sorted(contains)][:25]
        return exact + [s for sc, s in scored[:max(0, limit - len(exact))] if sc > 0.35]

    def episode(self, ep_id: int) -> Episode | None:
        row = self.db.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        return self._ep(row) if row else None

    def by_path(self, path: str) -> Episode | None:
        row = self.db.execute("SELECT * FROM episodes WHERE path=?", (path,)).fetchone()
        return self._ep(row) if row else None

    def series_episodes(self, series: str, include_extras: bool = False) -> list[Episode]:
        sql = "SELECT * FROM episodes WHERE series=?" + ("" if include_extras else " AND is_extra=0")
        eps = [self._ep(r) for r in self.db.execute(sql, (series,))]
        root_order = {r.name: i for i, r in enumerate(self.roots)}
        eps.sort(key=lambda e: (e.season if e.season is not None else 999,
                                e.episode if e.episode is not None else 9999,
                                root_order.get(e.path.split("/")[0], 99), e.path.lower()))
        # the same episode in two folders: keep one copy (first library folder wins)
        seen, unique = set(), []
        for e in eps:
            k = (e.season, e.episode) if e.season is not None and e.episode is not None else e.path
            if k not in seen:
                seen.add(k)
                unique.append(e)
        return unique

    def search(self, text: str, season: int | None = None, episode: int | None = None,
               include_extras: bool = False, limit: int = 15) -> list[Episode]:
        words = norm(text).split()
        sql, args = "SELECT * FROM episodes WHERE 1=1", []
        for w in words:
            sql += " AND search LIKE ?"
            args.append(f"%{w}%")
        if season is not None:
            sql += " AND season=?"
            args.append(season)
        if episode is not None:
            sql += " AND episode=?"
            args.append(episode)
        if not include_extras:
            sql += " AND is_extra=0"
        sql += " ORDER BY series, season, episode, path LIMIT ?"
        args.append(limit)
        return [self._ep(r) for r in self.db.execute(sql, args)]

    def recently_added(self, limit: int = 10) -> list[dict]:
        rows = self.db.execute("""SELECT series, MAX(mtime) m, COUNT(*) n FROM episodes
            WHERE is_extra=0 GROUP BY series ORDER BY m DESC LIMIT ?""", (limit,))
        return [{"series": r["series"], "added": time.strftime("%Y-%m-%d", time.localtime(r["m"])),
                 "episodes": r["n"]} for r in rows]

    def stats(self) -> dict:
        r = self.db.execute("""SELECT COUNT(DISTINCT show) shows, COUNT(DISTINCT series) series,
            SUM(is_extra=0) episodes, SUM(is_extra) extras, SUM(size) bytes FROM episodes""").fetchone()
        w = self.db.execute("SELECT COUNT(*) FROM history WHERE watched=1").fetchone()[0]
        return {"top_level_folders": r["shows"], "series": r["series"], "episodes": r["episodes"],
                "extras": r["extras"], "total_size_tb": round((r["bytes"] or 0) / 1e12, 2),
                "episodes_watched": w}

    # ---------- history ----------
    def record_progress(self, path: str, position: float, duration: float) -> None:
        watched = 1 if duration and position / duration >= 0.9 else None
        with self.db:
            self.db.execute("""INSERT INTO history(path,position,duration,watched,last_played)
                VALUES(?,?,?,COALESCE(?,0),?)
                ON CONFLICT(path) DO UPDATE SET position=excluded.position, duration=excluded.duration,
                watched=MAX(history.watched, COALESCE(?, history.watched)), last_played=excluded.last_played""",
                            (path, position, duration, watched, time.time(), watched))

    def set_watched(self, paths: list[str], watched: bool) -> None:
        with self.db:
            for p in paths:
                self.db.execute("""INSERT INTO history(path,position,duration,watched,last_played)
                    VALUES(?,0,0,?,?) ON CONFLICT(path) DO UPDATE SET watched=excluded.watched,
                    position=CASE WHEN excluded.watched=1 THEN 0 ELSE history.position END""",
                                (p, int(watched), time.time()))

    def progress(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM history WHERE path=?", (path,)).fetchone()

    def watched_paths(self, series: str) -> set[str]:
        return {r[0] for r in self.db.execute("""SELECT h.path FROM history h JOIN episodes e ON e.path=h.path
            WHERE e.series=? AND h.watched=1""", (series,))}

    def next_up(self, series: str) -> tuple[Episode | None, float]:
        """Episode to continue with in a series, and the resume position in seconds."""
        eps = self.series_episodes(series)
        if not eps:
            return None, 0.0
        last = self.db.execute("""SELECT h.* FROM history h JOIN episodes e ON e.path=h.path
            WHERE e.series=? AND e.is_extra=0 ORDER BY h.last_played DESC LIMIT 1""", (series,)).fetchone()
        if last is None:  # never watched: start at season 1, not with the specials in season 0
            return next((e for e in eps if e.season), eps[0]), 0.0
        idx = next((i for i, e in enumerate(eps) if e.path == last["path"]), None)
        if idx is None:
            return eps[0], 0.0
        if not last["watched"]:
            return eps[idx], last["position"] if last["position"] > 30 else 0.0
        nxt = eps[idx + 1] if idx + 1 < len(eps) else None
        return nxt, 0.0

    def continue_watching(self, limit: int = 8) -> list[dict]:
        rows = self.db.execute("""SELECT e.series, MAX(h.last_played) lp FROM history h
            JOIN episodes e ON e.path=h.path WHERE e.is_extra=0 AND h.last_played IS NOT NULL
            GROUP BY e.series ORDER BY lp DESC LIMIT ?""", (limit,)).fetchall()
        out = []
        for r in rows:
            ep, pos = self.next_up(r["series"])
            out.append({"series": r["series"],
                        "last_watched": time.strftime("%Y-%m-%d", time.localtime(r["lp"])),
                        "next_up": ep.label() if ep else "finished the series",
                        "next_episode_id": ep.id if ep else None,
                        "resume_at_minutes": round(pos / 60, 1) if pos else 0})
        return out
