"""Build and merge the blinded human-grading workbook.

  uv run -m evals.handgrade_workbook build --dirs runs/b_st_* runs/b_pe_* \
      runs/b_se_* --out runs/b_report/workbook.csv
  uv run -m evals.handgrade_workbook merge --workbook runs/b_report/workbook_filled.csv

Build collects the human-judgment rows from each run dir's handgrade_sheet.csv
into one shuffled, blinded workbook: the grader sees question, answer, and
criterion, but never the memory preset, so grades cannot favor a method. A key
map written next to the workbook links each blinded row back to its run dir
for the merge.

Row selection (keeps the pass to a half-day):
- priority 1: ALL session probe rows (the memory metric).
- priority 2: ALL persona llm-check rows.
- priority 3: single-turn key_point + behavior rows for a stratified subset of
  cases (default 12: 6 corpus / 2 redirect / 2 feedback / 2 general), trial 1.

Grade with `pass` or `fail` in the grade column (notes optional). Merge splits
the filled workbook into per-run-dir handgrade_filled.csv files and prints the
evals.grade commands that fold them into grades_merged.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from .common import load_jsonl

VISIBLE_FIELDS = (
    "key",
    "priority",
    "item_type",
    "question",
    "answer",
    "criterion",
    "reference",
    "grade",
    "note",
)


def read_sheet(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "handgrade_sheet.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [dict(row, _dir=str(run_dir)) for row in csv.DictReader(f)]


def singleturn_subset(battery_path: str, per_behavior: dict[str, int]) -> set[str]:
    by_behavior: dict[str, list[str]] = defaultdict(list)
    for case in load_jsonl(battery_path):
        by_behavior[case["expected_behavior"]].append(case["case_id"])
    subset: set[str] = set()
    for behavior, count in per_behavior.items():
        subset.update(sorted(by_behavior.get(behavior, []))[:count])
    return subset


def build(args: argparse.Namespace) -> None:
    per_behavior = {
        "answer_from_corpus": 6,
        "redirect_to_support": 2,
        "acknowledge_feedback": 2,
        "answer_general": 2,
    }
    selected: list[dict[str, str]] = []
    st_subset: set[str] | None = None
    for run_dir in map(Path, args.dirs):
        for row in read_sheet(run_dir):
            item_type = row["item_type"]
            if item_type.startswith("probe:"):
                row["priority"] = "1"
            elif item_type == "persona_llm_check":
                row["priority"] = "2"
            elif item_type in ("key_point", "behavior"):
                if st_subset is None:
                    bundles = load_jsonl(run_dir / "bundles.jsonl")
                    st_subset = singleturn_subset(
                        bundles[0]["battery_path"], per_behavior
                    )
                unit_id, _, rest = row["run_id"].partition("|")
                if unit_id not in st_subset or not rest.endswith("t1"):
                    continue
                row["priority"] = "3"
            else:  # replay rows etc. — not part of this pass
                continue
            selected.append(row)

    random.Random(42).shuffle(selected)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keymap_path = out.with_name(out.stem + "_keymap.csv")
    with (
        open(out, "w", encoding="utf-8") as wf,
        open(keymap_path, "w", encoding="utf-8") as kf,
    ):
        writer = csv.DictWriter(wf, fieldnames=list(VISIBLE_FIELDS))
        writer.writeheader()
        key_writer = csv.writer(kf)
        key_writer.writerow(["key", "run_dir", "sheet_row_id"])
        for index, row in enumerate(selected):
            key = f"g{index:04d}"
            writer.writerow(
                {
                    "key": key,
                    "priority": row["priority"],
                    "item_type": row["item_type"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "criterion": row["criterion"],
                    "reference": row["reference"],
                    "grade": "",
                    "note": "",
                }
            )
            key_writer.writerow([key, row["_dir"], row["sheet_row_id"]])
    counts = defaultdict(int)
    for row in selected:
        counts[row["priority"]] += 1
    print(f"{len(selected)} blinded rows -> {out}")
    print(f"  priority 1 (session probes): {counts['1']}")
    print(f"  priority 2 (persona llm checks): {counts['2']}")
    print(f"  priority 3 (single-turn subset): {counts['3']}")
    print(f"key map (do not open while grading): {keymap_path}")


def merge(args: argparse.Namespace) -> None:
    workbook = Path(args.workbook)
    keymap_path = Path(args.keymap) if args.keymap else None
    if keymap_path is None:
        stem = workbook.stem.replace("_filled", "")
        keymap_path = workbook.with_name(stem + "_keymap.csv")
    with open(keymap_path, encoding="utf-8") as f:
        key_map = {row["key"]: row for row in csv.DictReader(f)}

    by_dir: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    graded = 0
    with open(workbook, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("grade", "").strip():
                continue
            mapping = key_map[row["key"]]
            by_dir[mapping["run_dir"]][mapping["sheet_row_id"]] = row
            graded += 1

    for run_dir, grades in by_dir.items():
        sheet = read_sheet(Path(run_dir))
        out_rows = []
        for sheet_row in sheet:
            filled = grades.get(sheet_row["sheet_row_id"])
            if filled:
                sheet_row["grade"] = filled["grade"].strip().lower()
                sheet_row["note"] = filled.get("note", "")
            sheet_row.pop("_dir", None)
            out_rows.append(sheet_row)
        out_path = Path(run_dir) / "handgrade_filled.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
        print(
            f"{run_dir}: {len(grades)} grade(s) -> {out_path}\n"
            f"  next: uv run -m evals.grade --run {run_dir} "
            f"--handgrades {out_path}"
        )
    print(f"total grades merged: {graded}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    build_p = sub.add_parser("build")
    build_p.add_argument("--dirs", nargs="+", required=True)
    build_p.add_argument("--out", required=True)
    merge_p = sub.add_parser("merge")
    merge_p.add_argument("--workbook", required=True)
    merge_p.add_argument("--keymap", default="")
    args = parser.parse_args()
    if args.cmd == "build":
        build(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
