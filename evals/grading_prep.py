"""Prep lossless subagent-judge grading inputs from eval bundles.

Not part of the shipped harness; a one-off orchestration helper for the
holistic/faithfulness backfill (kept in-repo so the run is reproducible).

Per run, per item_type the column is treated as all-or-nothing:
- already fully graded (existing judge_filled/handgrade_filled covers every sheet
  row of that type) -> KEEP those grades, do not re-grade;
- otherwise -> ALL sheet rows of that type go to the judge (a single grader per
  column per run, so no human+judge double-counting and clean provenance).

The sheet supplies stable judgment ids and criteria, but its question/answer
columns are previews.  Complete question, answer, and reference content is
strictly rejoined from ``bundles.jsonl`` and the frozen battery before chunks
are written.

Outputs under runs/_grading/<grading-id>/:
- chunk_NNN.csv      blinded rows for one grader agent (sheet_row_id + content)
- keep/<run>.csv     existing grades to carry into the final merge
- manifest.json      battery -> {n_chunks, n_rows, runs}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import tiktoken

from .grading_content import INTEGRITY_COLS, hydrate_full_inputs
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
# Content-only budgets.  The token cap leaves ample room for the grader's
# system prompt, rubric source, CSV syntax, and output in a 100K+ context.
MAX_CHARS = 240_000
MAX_EST_TOKENS = 60_000
ROW_OVERHEAD_TOKENS = 64
AGENT_COLS = [
    "sheet_row_id",
    "item_type",
    "question",
    "criterion",
    "reference",
    "answer",
    "content_chars",
    "estimated_tokens",
    *INTEGRITY_COLS,
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
    previews = list(csv.DictReader((run_dir / "handgrade_sheet.csv").open()))
    sheet = hydrate_full_inputs(run_dir, previews) if previews else []
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


@lru_cache(maxsize=16_384)
def _text_tokens(text: str) -> int:
    # cl100k is an intentionally conservative, locally available planning
    # tokenizer.  The exact subscription grader tokenizer may differ, so the
    # independent character ceiling remains mandatory.
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def row_size(row: dict[str, str]) -> tuple[int, int]:
    content = [
        row.get("item_type") or "",
        row.get("question") or "",
        row.get("criterion") or "",
        row.get("reference") or "",
        row.get("answer") or "",
    ]
    chars = sum(len(value) for value in content)
    tokens = ROW_OVERHEAD_TOKENS + sum(_text_tokens(value) for value in content)
    return chars, tokens


def chunk(
    rows: list[dict[str, str]],
    *,
    max_rows: int = MAX_ROWS,
    max_chars: int = MAX_CHARS,
    max_tokens: int = MAX_EST_TOKENS,
) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    cur: list[dict[str, str]] = []
    cur_chars = 0
    cur_tokens = 0
    for r in rows:
        rchars, rtokens = row_size(r)
        if rchars > max_chars or rtokens > max_tokens:
            raise ValueError(
                f"{r.get('sheet_row_id')}: one lossless judgment exceeds the "
                f"chunk budget ({rchars} chars/{rtokens} tokens; limits "
                f"{max_chars}/{max_tokens}). Increase the explicit limit or "
                "exclude the row; it will not be silently truncated."
            )
        if cur and (
            len(cur) >= max_rows
            or cur_chars + rchars > max_chars
            or cur_tokens + rtokens > max_tokens
        ):
            chunks.append(cur)
            cur, cur_chars, cur_tokens = [], 0, 0
        r["content_chars"] = str(rchars)
        r["estimated_tokens"] = str(rtokens)
        cur.append(r)
        cur_chars += rchars
        cur_tokens += rtokens
    if cur:
        chunks.append(cur)
    return chunks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("battery", nargs="?", default="singleturn")
    # Retain the historical second positional argument.
    parser.add_argument(
        "prefixes",
        nargs="?",
        default="",
        help="comma-separated run-name prefixes",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs"),
        help="directory whose immediate children are run arms",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="grading output directory (default uses battery or run-root name)",
    )
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--max-tokens", type=int, default=MAX_EST_TOKENS)
    return parser.parse_args()


def discover_runs(run_root: Path) -> list[Path]:
    if (run_root / "bundles.jsonl").exists():
        return [run_root]
    return sorted(path for path in run_root.iterdir() if path.is_dir())


def _clear_stale_plan(out: Path) -> None:
    verdicts = sorted(out.glob("verdicts_*.json"))
    if verdicts:
        raise RuntimeError(
            f"{out} already contains {len(verdicts)} verdict files. Use a new "
            "--out directory so a re-plan cannot silently pair old judgments "
            "with new full-input chunks."
        )
    for path in out.glob("chunk_*.csv"):
        path.unlink()
    keep = out / "keep"
    if keep.exists():
        for path in keep.glob("*.csv"):
            path.unlink()


def main() -> None:
    args = _parse_args()
    want = args.battery
    prefixes = tuple(prefix for prefix in args.prefixes.split(",") if prefix)
    base = args.run_root
    output_id = want if base == Path("runs") else base.name
    out = args.out or Path("runs") / "_grading" / output_id
    (out / "keep").mkdir(parents=True, exist_ok=True)
    _clear_stale_plan(out)
    all_to_grade: list[dict[str, str]] = []
    runs_used: list[str] = []
    run_records: list[dict[str, str]] = []
    by_type = Counter()
    for run_dir in discover_runs(base):
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
            run_records.append(
                {
                    "name": name,
                    "path": str(run_dir),
                    "keep_file": f"keep/{name}.csv",
                }
            )
            all_to_grade.extend(to_grade)
            for r in to_grade:
                by_type[r["item_type"].split(":")[0]] += 1

    chunks = chunk(
        all_to_grade,
        max_rows=args.max_rows,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens,
    )
    chunk_records: list[dict[str, int | str]] = []
    for i, ch in enumerate(chunks):
        filename = f"chunk_{i:03d}.csv"
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=AGENT_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(ch)
        chunk_records.append(
            {
                "file": filename,
                "rows": len(ch),
                "content_chars": sum(int(r["content_chars"]) for r in ch),
                "estimated_tokens": sum(int(r["estimated_tokens"]) for r in ch),
            }
        )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 2,
                "grading_id": out.name,
                "battery": want,
                "run_root": str(base),
                "n_chunks": len(chunks),
                "n_rows": len(all_to_grade),
                "by_item_type": dict(by_type),
                "runs": runs_used,
                "run_records": run_records,
                "content_source": "bundles.jsonl+frozen_battery",
                "integrity": {
                    "algorithm": "sha256",
                    "fields": INTEGRITY_COLS,
                },
                "chunk_limits": {
                    "max_rows": args.max_rows,
                    "max_chars": args.max_chars,
                    "max_estimated_tokens": args.max_tokens,
                    "tokenizer": "cl100k_base",
                    "reserved_prompt_headroom_tokens": 10_000,
                },
                "chunks": chunk_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{want}: {len(all_to_grade)} rows -> {len(chunks)} chunks  {dict(by_type)}")
    print(f"  runs: {len(runs_used)}  out: {out}")


if __name__ == "__main__":
    main()
