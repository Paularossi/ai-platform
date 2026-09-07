"""
scripts/load_roster.py

One-off/rerunnable loader: reads a roster CSV and/or a tutors CSV and
upserts them into Supabase (core.db.load_roster / load_tutors). 
Safe to rerun whenever the lists change.

Usage
-----
    python scripts/load_roster.py --roster data/roster.csv --tutors data/tutors.csv

CSV formats
-----------
roster.csv - header "student_id,tutorial_group,name":
        student_id,tutorial_group,name
        i6123456,1,Alex
        i6123457,2,Bob

tutors.csv - header "tutor_id,name":
        tutor_id,name
        p70080001,Jane Smith
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import ensure_schema, load_roster, load_tutors  # noqa: E402


def _read_roster_csv(path: Path) -> list[tuple[str, int | None, str | None]]:
    rows: list[tuple[str, int | None, str | None]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader, start=2):  # header is line 1
            student_id = row.get("student_id", "").strip()
            if not student_id:
                continue
            raw_group = row.get("tutorial_group", "").strip()
            name = row.get("name", "").strip() or None
            if not raw_group:
                rows.append((student_id, None, name))
                continue
            try:
                rows.append((student_id, int(raw_group), name))
            except ValueError:
                raise ValueError(
                    f"{path}:{line_num}: tutorial_group {raw_group!r} for {student_id} "
                    "isn't a plain number."
                ) from None
    return rows


def _read_tutors_csv(path: Path) -> list[tuple[str, str | None]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            (row["tutor_id"].strip(), row.get("name", "").strip() or None)
            for row in reader
            if row.get("tutor_id", "").strip()
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the student roster and/or tutor list into Supabase.")
    parser.add_argument("--roster", type=Path, help="Path to a roster CSV (student_id, tutorial_group, name).")
    parser.add_argument("--tutors", type=Path, help="Path to a tutors CSV (tutor_id, name).")
    args = parser.parse_args()

    if not args.roster and not args.tutors:
        parser.error("Pass --roster and/or --tutors.")

    ensure_schema()

    if args.roster:
        rows = _read_roster_csv(args.roster)
        load_roster(rows)
        print(f"Loaded {len(rows)} roster entries from {args.roster}.")

    if args.tutors:
        tutors = _read_tutors_csv(args.tutors)
        load_tutors(tutors)
        print(f"Loaded {len(tutors)} tutor entries from {args.tutors}.")


if __name__ == "__main__":
    main()
