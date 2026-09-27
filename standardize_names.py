#!/usr/bin/env python3
"""Standardize episode filenames to "Show Name - S01E02 - Title.ext".

  standardize_names.py plan  [--library DIR]              write a rename plan; touches nothing
  standardize_names.py apply PLAN.tsv [--library DIR]      rename per plan, logging every rename as it happens
  standardize_names.py undo  LOG.tsv  [--library DIR]      put every file in the log back to its old name

Only files are renamed, in place (never moved between folders; folders are untouched). Subtitle/nfo/thumb
files that share the video's name are renamed with it. Anything the parser isn't sure of is left alone.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvassist.library import (EXTRA_DIR, EXTRA_FILE, SAMPLE, SMALL_WORDS, VIDEO_EXTS, norm,  # noqa: E402
                              parse_episode_detail, pretty, series_root, title_case)

BACKUP_DIR = os.path.join(HERE, "rename-backups")
LIB_BACKUP_DIRNAME = "_rename-backup"
LIB_NAME = "TV shows"   # a copy of every plan/log is also kept inside the library itself

JUNK = re.compile(
    r"\b(1080p|720p|480p|576p|2160p|4k|uhd|x26[45]|h[ .]?26[45]|hevc|avc|bluray|blu-ray|brrip|bdrip|web[ .-]?dl|webrip|"
    r"web|dvdrip|dvd|hdtv|pdtv|aac\d?|ac3|dts|ddp?\d?|10bit|xvid|divx|amzn|dsnp|hmax|atvp|nf|repack|proper|internal|"
    r"tsv|eztv|rarbg|subs?|multi|ita|eng)\b.*$", re.I)
ILLEGAL = str.maketrans({":": " -", "꞉": " -", "?": "", "*": "", '"': "'", "<": "", ">": "", "|": "-", "/": "-", "\\": "-"})


def clean_title(rest: str, series_name: str, episode: int) -> tuple[str, int | None]:
    """Tidy the text after the episode tag into a title; also detect a second episode number
    ("& 26", "E02") for double episodes."""
    t = re.sub(r"^[\s\]\)\-_.]+", "", rest)
    end_ep = None
    m = re.match(r"^(?:[-&+]|and)?\s*(?:E|Ep|Episode)?\s*(\d{1,3})\b\s*[-._]*\s*", t, re.I)
    if m and int(m.group(1)) in (episode + 1, episode + 2):
        end_ep = int(m.group(1))
        t = t[m.end():]
    t = re.sub(r"^(?:(?:hdtv|pdtv|dvdrip|webrip|web-dl|x264|xvid|720p|1080p)[\s._-]*)+-\s*", "", t, flags=re.I)
    if " " not in t:  # scene-style name: drop the trailing "-GROUP[tag]"
        t = re.sub(r"-[A-Za-z0-9]+(?:\[[^\]]*\])?$", "", t)
    t = re.sub(r"\((?:part|pt\.?)\s*(\d+)\)|\((\d)\)", lambda m: f" Part {m[1] or m[2]} ", t, flags=re.I)
    t = re.sub(r"[\[\{\(].*?[\]\}\)]", " ", t)
    t = re.sub(r"(?<=[A-Za-z0-9'])\.(?=[A-Za-z0-9])", " ", t).replace("_", " ")
    t = JUNK.sub("", t)
    sn = norm(series_name)
    if sn and norm(t).startswith(sn + " "):  # repeated series name at the start
        t = re.sub(r"^\W*" + r"\W+".join([r"\w+"] * len(sn.split())), "", t)
    t = re.sub(r"[ ._-]+S\d{1,2}$", "", t, flags=re.I)
    t = re.sub(r"\.\s+Part ", " Part ", t)
    t = t.translate(ILLEGAL)
    t = re.sub(r"\s+", " ", t).strip(" -_.,")
    if t.islower():
        t = title_case(t)
    elif t and t[0].islower():
        t = t[0].upper() + t[1:]
    if len(t) < 2 or re.fullmatch(r"[\d\s-]+|[A-Za-z]{1,3}", t) or norm(t) == sn:
        t = ""
    return t, end_ep


KNOWN_ABBREVIATIONS = {"sgu": "stargate universe", "b5": "babylon 5", "b5c": "crusade", "bsg": "battlestar galactica"}


def names_other_show(stem: str, rest: str, series_name: str) -> bool:
    """True when the text before the episode tag names a show that isn't this folder's series
    (e.g. "Station.Eleven.S01E10" sitting in the Station 19 folder)."""
    raw = stem[:len(stem) - len(rest)]
    raw = re.sub(r"^[a-z0-9]{2,5}-(?=[a-z])", "", raw)            # "afi-the.flash..." release-group prefix
    prefix = norm(raw)
    prefix = re.sub(r"\b(s\d+e\d+\w*|s\d+|e\d+\w*|\d+e\d+|se\d+xep\d+|\d+x\d+|season|episode|ep|\d+|us|uk)\b", " ", prefix)
    ptoks = [w for w in prefix.split() if w not in SMALL_WORDS and len(w) > 1]
    if not ptoks:
        return False
    stoks = norm(series_name).split()
    sset = set(stoks)
    big = [w for w in stoks if w not in SMALL_WORDS]
    # every contiguous run of series words, as initials ("tos", "tscc") and joined ("babylon5", "shield")
    runs = set()
    for words in (stoks, big):
        for i in range(len(words)):
            for j in range(i + 2, len(words) + 1):
                runs.add("".join(w[0] for w in words[i:j]))
                runs.add("".join(words[i:j]))
    prefix_initials = "".join(w[0] for w in ptoks)                # "emergency room" -> "er"
    if len(ptoks) > 1 and prefix_initials in sset:
        return False
    alias = {"dr": "doctor", "st": "saint"}

    def known(w: str) -> bool:
        return (w in sset or w.rstrip("s") in sset or alias.get(w) in sset or w in runs
                or KNOWN_ABBREVIATIONS.get(w) == " ".join(stoks))
    return not all(known(w) for w in ptoks)


def safe_series(name: str) -> str:
    return re.sub(r"\s+", " ", name.translate(ILLEGAL)).strip()


def plan(root: Path) -> tuple[list[tuple[str, str, str, str]], dict[str, list[str]]]:
    rows: list[tuple[str, str, str, str]] = []   # old, new, kind, how
    skipped: dict[str, list[str]] = defaultdict(list)
    by_dir: dict[Path, list[str]] = defaultdict(list)
    for show_dir in sorted(os.listdir(root)):
        full = root / show_dir
        if not full.is_dir() or show_dir.startswith((".", LIB_BACKUP_DIRNAME)):
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            by_dir[Path(dirpath)] = sorted(filenames)

    for d, files in by_dir.items():
        proposals: dict[str, str] = {}   # old filename -> new filename (videos)
        how: dict[str, str] = {}
        for fn in files:
            p = d / fn
            if p.suffix.lower() not in VIDEO_EXTS or SAMPLE.search(p.stem):
                continue
            show_dir = p.relative_to(root).parts[0]
            rel = p.relative_to(root / show_dir)
            relstr = str(p.relative_to(root))
            if any(EXTRA_DIR.match(x) for x in rel.parts[:-1]) or EXTRA_FILE.search(p.stem):
                skipped["extra/featurette (left as-is)"].append(relstr)
                continue
            info = parse_episode_detail(rel)
            if info["episode"] is None:
                skipped["no episode number found"].append(relstr)
                continue
            if info["season"] is None:
                skipped["episode number but no season"].append(relstr)
                continue
            if info["folder_season"] is not None and info["season"] != info["folder_season"]:
                skipped["season in filename disagrees with folder"].append(relstr)
                continue
            if info["episode"] == 0 and info["method"] != "explicit":
                skipped["episode 0 without explicit SxxE00 tag"].append(relstr)
                continue
            name = safe_series(pretty(series_root(show_dir, rel)))
            if info["method"] in ("explicit", "three_digit") and names_other_show(p.stem, info["rest"], name):
                skipped[f"filename names a different show than its folder (misfiled?)"].append(relstr)
                continue
            title, end_ep = clean_title(info["rest"], name, info["episode"])
            se = f"S{info['season']:02d}E{info['episode']:02d}" + (f"-E{end_ep:02d}" if end_ep else "")
            new = f"{name} - {se}" + (f" - {title}" if title else "") + p.suffix.lower()
            if len(new.encode()) > 240:
                new = f"{name} - {se}{p.suffix.lower()}"
            proposals[fn] = new
            how[fn] = info["method"]

        # collisions: two videos wanting the same name, or a name already taken by an unrelated file
        lower_existing = {f.lower(): f for f in files}
        targets = defaultdict(list)
        for old, new in proposals.items():
            targets[new.lower()].append(old)
        for old, new in list(proposals.items()):
            clash = len(targets[new.lower()]) > 1
            other = lower_existing.get(new.lower())
            if other is not None and other != old and other not in proposals:
                clash = True
            if clash:
                skipped["would collide with another file (duplicate episode?)"].append(str((d / old).relative_to(root)))
                del proposals[old]

        for old, new in sorted(proposals.items()):
            if old == new:
                continue
            rows.append((str((d / old).relative_to(root)), str((d / new).relative_to(root)), "video", how[old]))
            # sidecars: same stem followed by . - or _ (subs, idx/sub, nfo, -thumb.jpg ...)
            old_stem, new_stem = Path(old).stem, Path(new).stem
            for f in files:
                if f == old or Path(f).suffix.lower() in VIDEO_EXTS:
                    continue
                if f.startswith(old_stem) and f[len(old_stem):len(old_stem) + 1] in (".", "-", "_"):
                    nf = new_stem + f[len(old_stem):]
                    if nf.lower() in lower_existing and lower_existing[nf.lower()] != f:
                        continue
                    rows.append((str((d / f).relative_to(root)), str((d / nf).relative_to(root)), "sidecar", ""))
    return rows, skipped


def write_tsv(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("old_path\tnew_path\tkind\thow\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def read_tsv(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return [ln.split("\t") for ln in lines[1:] if ln.strip()]


def copy_into_library(root: Path, path: str) -> None:
    dest = root / LIB_BACKUP_DIRNAME
    dest.mkdir(exist_ok=True)
    with open(path, "rb") as src, open(dest / os.path.basename(path), "wb") as dst:
        dst.write(src.read())


def rename_all(root: Path, pairs: list[tuple[str, str]], log_path: str) -> tuple[int, list[str]]:
    """Rename old->new for each pair, appending each completed rename to log_path immediately."""
    done, problems = 0, []
    with open(log_path, "a", encoding="utf-8") as log:
        if log.tell() == 0:
            log.write("old_path\tnew_path\n")
        for old, new in pairs:
            src, dst = root / old, root / new
            if not src.exists():
                problems.append(f"missing: {old}")
                continue
            if dst.exists() and not (src.parent == dst.parent and src.name.lower() == dst.name.lower()):
                problems.append(f"target exists: {new}")
                continue
            os.rename(src, dst)
            log.write(f"{old}\t{new}\n")
            log.flush()
            os.fsync(log.fileno())
            done += 1
    return done, problems


def update_history(old_new: list[tuple[str, str]]) -> None:
    import sqlite3
    db_path = os.path.join(HERE, "data", "library.db")
    if not os.path.exists(db_path):
        return
    db = sqlite3.connect(db_path)
    with db:
        prefix = LIB_NAME + "/"  # the assistant stores paths as "<library folder>/<show>/..."
        for old, new in old_new:
            db.execute("UPDATE OR IGNORE history SET path=? WHERE path=?", (prefix + new, prefix + old))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["plan", "apply", "undo"])
    ap.add_argument("file", nargs="?", help="plan TSV for apply, rename log TSV for undo")
    ap.add_argument("--library", default="/mnt/T/TV shows")
    ap.add_argument("--no-library-copy", action="store_true", help="don't also copy backups into the library")
    args = ap.parse_args()
    root = Path(args.library)
    global LIB_NAME
    LIB_NAME = root.name
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")

    if args.mode == "plan":
        rows, skipped = plan(root)
        path = os.path.join(BACKUP_DIR, f"plan_{stamp}.tsv")
        write_tsv(path, rows)
        skip_path = os.path.join(BACKUP_DIR, f"plan_{stamp}_skipped.txt")
        with open(skip_path, "w", encoding="utf-8") as f:
            for reason, items in skipped.items():
                f.write(f"## {reason} ({len(items)})\n" + "".join(f"{i}\n" for i in items) + "\n")
        kinds = defaultdict(int)
        for r in rows:
            kinds[r[2] if r[2] == "sidecar" else f"video ({r[3]})"] += 1
        print(f"plan: {path}")
        for k, v in sorted(kinds.items()):
            print(f"  {v:6d} {k}")
        print(f"skipped: {skip_path}")
        for reason, items in skipped.items():
            print(f"  {len(items):6d} {reason}")

    elif args.mode == "apply":
        rows = read_tsv(args.file)
        # 1) snapshot every current filename in the library before touching anything
        snap = os.path.join(BACKUP_DIR, f"all-filenames-before_{stamp}.txt")
        with open(snap, "w", encoding="utf-8") as f:
            for dirpath, _, filenames in os.walk(root):
                for fn in sorted(filenames):
                    f.write(os.path.relpath(os.path.join(dirpath, fn), root) + "\n")
        log = os.path.join(BACKUP_DIR, f"renamed_{stamp}.tsv")
        if not args.no_library_copy:
            copy_into_library(root, snap)
            copy_into_library(root, args.file)
        print(f"snapshot of current names: {snap}")
        print(f"rename log (for undo):     {log}")
        done, problems = rename_all(root, [(r[0], r[1]) for r in rows], log)
        if not args.no_library_copy:
            copy_into_library(root, log)
        update_history([(r[0], r[1]) for r in rows])
        print(f"renamed {done} of {len(rows)} files")
        for p in problems:
            print("  problem:", p)

    elif args.mode == "undo":
        rows = read_tsv(args.file)
        log = os.path.join(BACKUP_DIR, f"undo_{stamp}.tsv")
        done, problems = rename_all(root, [(r[1], r[0]) for r in reversed(rows)], log)
        update_history([(r[1], r[0]) for r in rows])
        print(f"restored {done} of {len(rows)} files (log: {log})")
        for p in problems:
            print("  problem:", p)


if __name__ == "__main__":
    main()
