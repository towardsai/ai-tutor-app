"""Prep subagent-judge grading inputs from regenerated handgrade sheets.

Not part of the shipped harness; a one-off orchestration helper for the
holistic/faithfulness backfill (kept in-repo so the run is reproducible).

Per run, per item_type the column is treated as all-or-nothing:
- already fully graded (existing judge_filled/handgrade_filled covers every sheet
  row of that type) -> KEEP those grades, do not re-grade;
- otherwise -> ALL sheet rows of that type go to the judge (a single grader per
  column per run, so no human+judge double-counting and clean provenance).

Outputs under runs/_grading/<battery>/:
- chunk_NNN.csv      blinded rows for one grader agent (sheet_row_id + content)
- keep/<run>.csv     existing grades to carry into the final merge
- manifest.json      battery -> {n_chunks, n_rows, runs}
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .grading_merge import DROP_ITEM_TYPES

csv.field_size_limit(10_000_000)

SKIP = {
    "a4_s03_aggressive",
    "a4_s03_fullhist",
    "a4_sessions_prod",
    "smoke_personas",
    "smoke_session_prod",
    "smoke_st_fullhist",
    "smoke_st_prod",
    "ds_smoke_st",
    "e_v2_prod",
}
MAX_ROWS = 40
MAX_CHARS = 300_000
AGENT_COLS = [
    "sheet_row_id",
    "item_type",
    "question",
    "criterion",
    "reference",
    "answer",
]


def battery_type(run_dir: Path) -> str:
    first = json.loads((run_dir / "bundles.jsonl").open().readline())
    return first.get("battery_type")


def existing_filled(run_dir: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for fn in ("judge_filled.csv", "handgrade_filled.csv"):
        p = run_dir / fn
        if p.exists():
            out += [
                r for r in csv.DictReader(p.open()) if (r.get("grade") or "").strip()
            ]
    return out


def plan_run(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (rows_to_grade, rows_to_keep) for one run."""
    sheet = list(csv.DictReader((run_dir / "handgrade_sheet.csv").open()))
    if not sheet:
        return [], []
    sheet_by_type: dict[str, list] = defaultdict(list)
    for r in sheet:
        sheet_by_type[r["item_type"]].append(r)
    filled = existing_filled(run_dir)
    filled_by_type: dict[str, list] = defaultdict(list)
    for r in filled:
        filled_by_type[r["item_type"]].append(r)

    to_grade: list[dict[str, str]] = []
    to_keep: list[dict[str, str]] = []
    for itype, rows in sheet_by_type.items():
        graded = filled_by_type.get(itype, [])
        if len(graded) >= len(rows):  # column already fully graded -> keep
            to_keep.extend(graded)
        else:  # grade the whole column
            to_grade.extend(rows)
    return to_grade, to_keep


def chunk(rows: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    cur: list[dict[str, str]] = []
    cur_chars = 0
    for r in rows:
        rchars = len(r.get("reference") or "") + len(r.get("answer") or "")
        if cur and (len(cur) >= MAX_ROWS or cur_chars + rchars > MAX_CHARS):
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append(r)
        cur_chars += rchars
    if cur:
        chunks.append(cur)
    return chunks


def main() -> None:
    want = sys.argv[1] if len(sys.argv) > 1 else "singleturn"
    # Optional 2nd arg: comma-separated run-name prefixes to include (focus a run
    # on a subset of arms, e.g. "b_se,c_se" for the sessions quality test).
    prefixes = tuple(
        p for p in (sys.argv[2].split(",") if len(sys.argv) > 2 else []) if p
    )
    base = Path("runs")
    out = base / "_grading" / want
    (out / "keep").mkdir(parents=True, exist_ok=True)
    # answers come from bundles (sheet 'answer' is truncated to 4000; fine for grading)
    all_to_grade: list[dict[str, str]] = []
    runs_used: list[str] = []
    by_type = Counter()
    for run_dir in sorted(base.glob("*/")):
        name = run_dir.name
        if name in SKIP or name.startswith("axisa"):
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if not (run_dir / "handgrade_sheet.csv").exists():
            continue
        if battery_type(run_dir) != want:
            continue
        to_grade, to_keep = plan_run(run_dir)
        # Faithfulness is parked (see grading_merge.DROP_ITEM_TYPES): don't spend
        # grading budget on it while the report/merge drop it.
        to_grade = [r for r in to_grade if r["item_type"] not in DROP_ITEM_TYPES]
        if to_keep:
            with (out / "keep" / f"{name}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=to_keep[0].keys())
                w.writeheader()
                w.writerows(to_keep)
        if to_grade:
            runs_used.append(name)
            all_to_grade.extend(to_grade)
            for r in to_grade:
                by_type[r["item_type"].split(":")[0]] += 1

    chunks = chunk(all_to_grade)
    for i, ch in enumerate(chunks):
        with (out / f"chunk_{i:03d}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=AGENT_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(ch)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "battery": want,
                "n_chunks": len(chunks),
                "n_rows": len(all_to_grade),
                "by_item_type": dict(by_type),
                "runs": runs_used,
            },
            indent=2,
        )
    )
    print(f"{want}: {len(all_to_grade)} rows -> {len(chunks)} chunks  {dict(by_type)}")
    print(f"  runs: {len(runs_used)}  out: {out}")


if __name__ == "__main__":
    main()
