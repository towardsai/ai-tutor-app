"""Merge subagent-judge verdicts back into runs, and collect uncertain calls.

Companion to grading_prep.py / grade_workflow.js. For one battery:
1. load every runs/_grading/<battery>/verdicts_*.json,
2. per run, write combined_filled.csv = kept existing grades + new judge verdicts,
   then run `evals.grade --handgrades` to regenerate grades_merged.jsonl,
3. flag any expected rows that never got a verdict (chunk agent failed), and
4. write uncertain.md + uncertain.csv = every confidence=low verdict with its
   question/answer/criterion (the human-review queue the user asked for).
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)


def _ah(answer: str) -> str:
    """Short hash of the graded answer text, to disambiguate sheet_row_id.

    sheet_row_id = run_id|item_type|md5(criterion) is NOT globally unique: run_id
    is per-case, so the same question graded under two presets (different run
    dirs) shares an id while having different answers. Keying verdicts by
    (sheet_row_id, answer-hash) restores uniqueness; identical answers collapse
    correctly (same grade).
    """
    return hashlib.md5((answer or "").encode("utf-8")).hexdigest()[:12]


COMBINED_COLS = [
    "sheet_row_id",
    "run_id",
    "battery_type",
    "preset",
    "item_type",
    "criterion",
    "grade",
    "note",
]

# Item types dropped from the merge + review queue. Faithfulness is parked: on
# bundles recorded under the old 6k tool-output cap the judge can't see the
# truncated KB evidence, so it scores capture-completeness not grounding
# (evals.md F23/F24 class). Re-enable after re-recording runs with the raised cap.
DROP_ITEM_TYPES = {"faithfulness"}


def load_verdicts(
    gdir: Path,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """Return (verdicts, content) keyed by (sheet_row_id, answer-hash).

    Recovers the verdict<->row association by POSITION within each chunk (chunk
    row i corresponds to verdict i; order verified identical for all chunks), so
    the per-run answer the verdict actually graded is known and used as the key.
    `content` carries the chunk row (question/criterion/answer) for the report.
    """
    verdicts: dict[tuple[str, str], dict] = {}
    content: dict[tuple[str, str], dict] = {}
    for cf in sorted(gdir.glob("chunk_*.csv")):
        crows = list(csv.DictReader(cf.open()))
        vf = gdir / cf.name.replace("chunk_", "verdicts_").replace(".csv", ".json")
        if not vf.exists():
            continue
        vrows = json.loads(vf.read_text())
        # Match by sheet_row_id with per-id FIFO queues: robust to a short chunk
        # (agent skipped rows), reordering, or within-chunk duplicate ids (a chunk
        # spanning two runs that share a session/turn). A chunk row with no
        # remaining verdict is left unmatched -> surfaces as "missing" in main().
        queues: dict[str, collections.deque] = collections.defaultdict(
            collections.deque
        )
        for v in vrows:
            queues[v["sheet_row_id"]].append(v)
        for r in crows:
            q = queues.get(r["sheet_row_id"])
            if not q:
                continue
            v = q.popleft()
            key = (r["sheet_row_id"], _ah(r.get("answer") or ""))
            verdicts[key] = v
            content[key] = r
    return verdicts, content


def expected_keys(gdir: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for ch in sorted(gdir.glob("chunk_*.csv")):
        for r in csv.DictReader(ch.open()):
            keys.add((r["sheet_row_id"], _ah(r.get("answer") or "")))
    return keys


def note_for(v: dict) -> str:
    return f"[judge:{v.get('confidence', 'high')}] {v.get('reason', '')}".strip()


def main() -> None:
    battery = sys.argv[1] if len(sys.argv) > 1 else "singleturn"
    gdir = Path("runs/_grading") / battery
    manifest = json.loads((gdir / "manifest.json").read_text())
    verdicts, content = load_verdicts(gdir)
    expected = expected_keys(gdir)
    missing = sorted(expected - set(verdicts))
    print(f"{battery}: {len(verdicts)} verdicts, {len(missing)} expected rows missing")

    merged_runs = 0
    key_to_run: dict[tuple[str, str], str] = {}
    for name in manifest["runs"]:
        run_dir = Path("runs") / name
        sheet = list(csv.DictReader((run_dir / "handgrade_sheet.csv").open()))
        combined: list[dict] = []
        # kept existing grades (columns may differ across old files; normalize)
        keep_path = gdir / "keep" / f"{name}.csv"
        if keep_path.exists():
            for r in csv.DictReader(keep_path.open()):
                combined.append({c: r.get(c, "") for c in COMBINED_COLS})
        # new judge verdicts for this run's sheet rows (content key disambiguates
        # the cross-run sheet_row_id collision)
        for r in sheet:
            if r["item_type"] in DROP_ITEM_TYPES:
                continue
            key = (r["sheet_row_id"], _ah(r.get("answer") or ""))
            key_to_run[key] = name
            v = verdicts.get(key)
            if not v:
                continue
            combined.append(
                {
                    "sheet_row_id": r["sheet_row_id"],
                    "run_id": r.get("run_id", ""),
                    "battery_type": r.get("battery_type", battery),
                    "preset": r.get("preset", ""),
                    "item_type": r["item_type"],
                    "criterion": r.get("criterion", ""),
                    "grade": v["grade"],
                    "note": note_for(v),
                }
            )
        cf = run_dir / "combined_filled.csv"
        with cf.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COMBINED_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(combined)
        res = subprocess.run(
            [
                "uv",
                "run",
                "-m",
                "evals.grade",
                "--run",
                str(run_dir),
                "--handgrades",
                str(cf),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            print(f"  MERGE FAILED {name}: {res.stderr.strip()[-300:]}")
        else:
            merged_runs += 1
    print(f"merged {merged_runs}/{len(manifest['runs'])} runs")

    # ---- uncertain queue (the deliverable) --------------------------------
    low = [
        (k, v)
        for k, v in verdicts.items()
        if v.get("confidence") == "low"
        and v.get("item_type", "").split(":")[0] not in DROP_ITEM_TYPES
    ]

    def arm_of(key: tuple[str, str]) -> str:
        return key_to_run.get(key, "?")

    low.sort(key=lambda kv: (kv[1]["item_type"], arm_of(kv[0])))
    with (gdir / "uncertain.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sheet_row_id",
                "run",
                "item_type",
                "judge_grade",
                "judge_reason",
                "question",
                "criterion",
                "answer",
                "your_grade",
            ],
        )
        w.writeheader()
        for key, v in low:
            c = content.get(key, {})
            w.writerow(
                {
                    "sheet_row_id": key[0],
                    "run": arm_of(key),
                    "item_type": v["item_type"],
                    "judge_grade": v["grade"],
                    "judge_reason": v.get("reason", ""),
                    "question": (c.get("question") or "")[:300],
                    "criterion": (c.get("criterion") or "")[:300],
                    "answer": (c.get("answer") or "")[:1200],
                    "your_grade": "",
                }
            )

    lines = [f"# Uncertain judge calls — {battery}", ""]
    lines.append(
        f"{len(low)} low-confidence verdicts of {len(verdicts)} graded. "
        "Fill `your_grade` in uncertain.csv to override.\n"
    )
    by_type = collections.Counter(v["item_type"].split(":")[0] for _, v in low)
    by_run = collections.Counter(arm_of(k) for k, _ in low)
    lines.append(
        "By item type: " + ", ".join(f"{k}={n}" for k, n in sorted(by_type.items()))
    )
    lines.append(
        "By run: " + ", ".join(f"{k}={n}" for k, n in sorted(by_run.items())) + "\n"
    )
    for key, v in low:
        c = content.get(key, {})
        lines.append(
            f"## [{v['item_type']}] judge={v['grade'].upper()} — run `{arm_of(key)}`"
        )
        lines.append(f"- **id**: `{key[0]}`")
        lines.append(f"- **judge reason**: {v.get('reason', '')}")
        lines.append(f"- **Q**: {(c.get('question') or '')[:300]}")
        lines.append(f"- **criterion**: {(c.get('criterion') or '')[:300]}")
        lines.append(f"- **answer**: {(c.get('answer') or '')[:700]}")
        lines.append("")
    (gdir / "uncertain.md").write_text("\n".join(lines) + "\n")
    print(f"uncertain: {len(low)} low-confidence -> {gdir}/uncertain.md (+ .csv)")
    if missing:
        print("MISSING keys (rerun their chunks):", len(missing))


if __name__ == "__main__":
    main()
