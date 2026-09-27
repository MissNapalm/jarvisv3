"""'Give me a file update': compare today's library files with the previous day's snapshot."""
from __future__ import annotations

import datetime as dt
import os
import re

from .library import Library, pretty

TRIGGER = re.compile(
    r"\b(file (update|report|check|count|status)|files? update|check (on )?(the |my |our )?files|"
    r"(are |is )?(any |some )?(of )?(the |my |our )?files (missing|disappearing|gone|deleted)|"
    r"how many files|did (we|i) lose (any )?files)\b", re.I)


def wants_file_update(text: str) -> bool:
    return bool(TRIGGER.search(text))


def _when(day: str) -> str:
    d = dt.date.fromisoformat(day)
    if d == dt.date.today() - dt.timedelta(days=1):
        return "yesterday"
    return d.strftime("%B %-d")


def _shows(pairs: list[tuple[str, int]], limit: int = 3) -> str:
    parts = [f"{n} {'in' if show == 'the main folder' else 'from'} "
             f"{show if show == 'the main folder' else pretty(show)}" for show, n in pairs[:limit]]
    more = sum(n for _, n in pairs[limit:])
    if more:
        parts.append(f"{more} elsewhere")
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if parts else ""


def file_update(lib: Library, data_dir: str) -> tuple[str, str]:
    """Returns (short spoken reply, details for the screen). Also saves the details to a file."""
    r = lib.file_update()
    if r.get("error"):
        where = ", ".join(r.get("missing_folders", [])) or "the library"
        return (f"I can't reach the library drive right now, so I can't count the files. "
                f"That's usually the network drive being offline, not files being deleted.",
                f"Not reachable: {where}")
    total = f"{r['total']:,}"
    if r["previous"] is None:
        return (f"There are {total} files in the library right now. This is the first count, "
                f"so I'll be able to compare from tomorrow.", f"{r['total']:,} files ({r['videos']:,} videos)")

    when = _when(r["previous"])
    missing, added = r["missing"], r["added"]
    diff = r["total"] - r["previous_total"]
    if not missing and not added:
        spoken = f"All good. There are {total} files, exactly the same as {when}."
    else:
        change = ("the same number as" if diff == 0 else
                  f"{abs(diff):,} {'fewer' if diff < 0 else 'more'} than")
        spoken = f"There are {total} files, {change} {when}."
        if missing:
            spoken += f" {len(missing):,} file{'s' if len(missing) != 1 else ''} went missing: {_shows(r['missing_by_show'])}."
        if added:
            spoken += f" {len(added):,} new: {_shows(r['added_by_show'])}."
        spoken += " The full list is on screen."

    lines = [f"File update {r['today']} vs {r['previous']}: {r['total']:,} files now "
             f"({r['videos']:,} videos), {r['previous_total']:,} before ({r['previous_videos']:,} videos)."]
    if missing:
        lines.append(f"\nMISSING since {r['previous']} ({len(missing):,}):")
        lines += [f"  - {p}" for p in missing]
    if added:
        lines.append(f"\nNEW since {r['previous']} ({len(added):,}):")
        lines += [f"  + {p}" for p in added]
    details = "\n".join(lines)
    path = os.path.join(data_dir, f"file-report-{r['today']}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(details + "\n")
    preview = "\n".join(lines[:40]) + (f"\n  … {len(lines) - 40} more lines" if len(lines) > 40 else "")
    return spoken, preview + f"\nFull report: {path}"
