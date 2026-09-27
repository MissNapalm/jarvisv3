#!/usr/bin/env python3
"""Daily file-count snapshot for the "give me a file update" check.

Counts every file in the library folders and stores today's list in data/library.db, where the
assistant compares it with earlier days. Meant to run once a day (see tv-snapshot.timer); safe to
run by hand any time: the last run of a day replaces that day's snapshot.
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tvassist.library import Library  # noqa: E402


def main() -> int:
    folders = os.environ.get("TV_LIBRARY", "/mnt/T/TV shows").split(":")
    log_path = os.path.join(HERE, "data", "snapshots.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    missing = [f for f in folders if not os.path.isdir(f)]
    if missing:  # network drive offline: don't record that as "every file disappeared"
        log(f"SKIPPED: library folder not reachable: {', '.join(missing)}")
        return 1
    lib = Library(folders, os.path.join(HERE, "data", "library.db"))
    stats = lib.scan()
    if not stats["all_files"]:
        log(f"SKIPPED: no files found in {', '.join(folders)} (drive offline?)")
        return 1
    prev = lib.db.execute("SELECT day, total FROM snapshots WHERE day < ? ORDER BY day DESC LIMIT 1",
                          (time.strftime("%Y-%m-%d"),)).fetchone()
    change = f", {stats['all_files'] - prev['total']:+,} vs {prev['day']}" if prev else ""
    log(f"OK: {stats['all_files']:,} files ({stats['files']:,} videos) in {stats['seconds']}s{change}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
