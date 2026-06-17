"""Grade saved tutor-turn bundles.

This command does not call the tutor. It reads `bundles.jsonl`, computes every
metric that code can compute, and writes a CSV for judgments that need a
person or validated LLM judge.

  uv run -m evals.grade --run runs/<exp>                 # auto grades + sheet
  uv run -m evals.grade --run runs/<exp> --handgrades runs/<exp>/handgrade_filled.csv

Pure JSON/CSV in and out — no app imports — so old bundles re-grade forever.

Outputs in the run dir:
- grades_auto.jsonl  — one row per bundle row with every code-computable check
  (retrieval recall, behavior proxy checks, citation presence, persona regex
  checks, probe trigger context). See data/eval/README.md for term definitions.
- handgrade_sheet.csv — one row per pending human judgment (key points, session
  probes, replay replies, persona llm-checks). Fill `grade` with pass/fail
  (optionally a note), save as handgrade_filled.csv, re-run with --handgrades.
- grades_merged.jsonl — auto + human grades joined, ready for evals.report.

Behavior checks here are rough proxies, such as "did a course question call
retrieval?" The reportable behavior-accuracy number comes from human grades.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path
from typing import Any

from .common import (
    COMPACTION_SIGNAL_KEYS,
    compaction_active,
    load_jsonl,
    normalize_url,
    write_jsonl,
)

BEHAVIOR_HEURISTICS = {
    "redirect_to_support": r"support|academy team|reach out|contact",
    "acknowledge_feedback": r"thank|feedback|appreciate|noted|passed (this|it) (on|along)",
}
RETRIEVAL_TOOLS = {"retrieve_tutor_context", "run_kb_command"}
# run_kb_command output is raw file text, not structured matches, so recall@shown
# (retrieve_tutor_context only) is blind to KB grounding. KB browses run with cwd
# = data/kb, so raw/docs/<source>/... and raw/courses/<source>/... path fragments
# in the command or its output reveal which corpus source the agent touched.
KB_RAW_SOURCE_RE = re.compile(r"raw/(?:docs|courses)/([A-Za-z0-9_.\-]+)/")


def index_battery(battery: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record in battery:
        for key in ("case_id", "session_id", "replay_id"):
            if key in record:
                index[record[key]] = record
        if "persona_id" in record:
            for question in record["questions"]:
                index[question["question_id"]] = {**question, "_persona": record}
    return index


def kb_browsed_sources(bundle: dict[str, Any]) -> set[str]:
    """source_keys a ``run_kb_command`` touched, parsed from raw/ path fragments
    in the command and its output.

    Lets recall credit KB grounding, which recall@shown cannot see (KB output is
    raw text, not structured matches). Caveats: bundle output is capped at
    6000 chars, so a path past the cap is missed; a broad ``rg`` can surface many
    sources at once (an over-count toward "shown", not "used").
    """
    seen: set[str] = set()
    for call in bundle.get("tool_calls") or []:
        if call.get("tool_name") != "run_kb_command":
            continue
        text = f"{call.get('args_text') or ''}\n{call.get('output_text') or ''}"
        seen.update(KB_RAW_SOURCE_RE.findall(text))
    return seen


def retrieval_metrics(
    bundle: dict[str, Any], source_key: str | None, lesson_url: str | None
) -> dict[str, Any]:
    """Source/lesson hits over the matches from retrieval tool calls.

    Matches are the ranked results the model actually saw. `recall_source`
    and `recall_lesson` answer "was the right source or lesson shown at all?"
    `mrr_lesson` is a ranking score: first result = 1.0, second = 0.5,
    third = 0.33, missing = 0. `recall_anytool_source` is the KB-fair version:
    it also credits a source the agent browsed via run_kb_command, so it is
    comparable across kb_on/kb_off arms (recall_source sees only retrieval).
    """
    matches: list[dict[str, Any]] = []
    called_retrieval = False
    for call in bundle.get("tool_calls") or []:
        if call.get("tool_name") in RETRIEVAL_TOOLS:
            called_retrieval = True
        if call.get("tool_name") == "retrieve_tutor_context":
            matches.extend(call.get("matches") or [])
    lesson = normalize_url(lesson_url)
    source_hit = any(m.get("source_key") == source_key for m in matches)
    lesson_rank = next(
        (
            rank
            for rank, m in enumerate(matches, start=1)
            if lesson and normalize_url(m.get("url")) == lesson
        ),
        0,
    )
    measured = bool(matches or called_retrieval)
    kb_sources = kb_browsed_sources(bundle)
    return {
        "called_retrieval": called_retrieval,
        "retrieved_matches": len(matches),
        "recall_source": source_hit if measured else None,
        "recall_lesson": bool(lesson_rank) if lesson else None,
        "mrr_lesson": (1.0 / lesson_rank) if lesson_rank else 0.0 if lesson else None,
        "recall_anytool_source": (
            (source_hit or source_key in kb_sources)
            if measured and source_key
            else None
        ),
    }


def citation_metrics(
    bundle: dict[str, Any], source_key: str | None, lesson_url: str | None
) -> dict[str, Any]:
    """Did the FINAL answer cite the correct source/lesson?

    Reads `resolved_sources` (the answer's resolved citation cards), which the
    app resolves from BOTH retrieval and KB-browse evidence via kb_manifest.
    Unlike recall@shown (retrieve_tutor_context only), this is tool-agnostic and
    so comparable across kb_on/kb_off arms. None when the case has no ground
    truth source/lesson.
    """
    cited = bundle.get("resolved_sources") or []
    lesson = normalize_url(lesson_url)
    return {
        "cited_correct_source": (
            any(c.get("source_key") == source_key for c in cited)
            if source_key
            else None
        ),
        "cited_correct_lesson": (
            any(normalize_url(c.get("url")) == lesson for c in cited)
            if lesson
            else None
        ),
    }


def behavior_heuristic(expected: str, bundle: dict[str, Any]) -> bool | None:
    """Cheap proxy for behavior routing; None = no code check for this class."""
    answer = bundle.get("answer") or ""
    if expected == "answer_from_corpus":
        return any(
            call.get("tool_name") in RETRIEVAL_TOOLS
            for call in bundle.get("tool_calls") or []
        )
    pattern = BEHAVIOR_HEURISTICS.get(expected)
    if pattern is None:
        return None
    return re.search(pattern, answer, re.IGNORECASE) is not None


def grade_persona_question(question: dict[str, Any], answer: str) -> dict[str, Any]:
    """Apply the battery's self-grading checks (see data/eval/README.md)."""
    results, needs_judgment = [], False
    for check in question.get("checks") or []:
        if check["type"] == "regex_any":
            passed = any(
                re.search(pattern, answer, re.IGNORECASE)
                for pattern in check["patterns"]
            )
            results.append({"type": "regex_any", "passed": passed})
        elif check["type"] == "llm":
            needs_judgment = True
            results.append({"type": "llm", "passed": None})
    anti_hits = [
        pattern
        for pattern in question.get("anti_patterns") or []
        if re.search(pattern, answer, re.IGNORECASE)
    ]
    decided = [r["passed"] for r in results if r["passed"] is not None]
    auto_pass: bool | None
    if anti_hits:
        auto_pass = False
    elif needs_judgment:
        auto_pass = None  # resolved by the hand/judge grade
    else:
        auto_pass = all(decided) if decided else None
    return {
        "checks": results,
        "anti_pattern_hits": anti_hits,
        "needs_judgment": needs_judgment,
        "auto_pass": auto_pass,
    }


def sheet_row(
    *,
    bundle: dict[str, Any],
    item_type: str,
    criterion: str,
    reference: str = "",
) -> dict[str, str]:
    return {
        # Stable hash (NOT builtin hash(), which is salted per process — that
        # made sheet_row_id non-deterministic across runs and silently broke the
        # workbook keymap join whenever a sheet was regenerated).
        "sheet_row_id": f"{bundle['run_id']}|{item_type}"
        f"|{hashlib.md5(criterion.encode('utf-8')).hexdigest()[:8]}",
        "run_id": bundle["run_id"],
        "battery_type": bundle["battery_type"],
        "preset": bundle["preset"],
        "item_type": item_type,
        "question": bundle["query"][:600],
        "answer": (bundle.get("answer") or "")[:4000],
        "criterion": criterion,
        "reference": reference[:2000],
        "grade": "",
        "note": "",
    }


def grade_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    bundles = load_jsonl(run_dir / "bundles.jsonl")
    if not bundles:
        raise SystemExit(f"No bundles in {run_dir}")
    battery = load_jsonl(bundles[0]["battery_path"])
    by_id = index_battery(battery)

    grades: list[dict[str, Any]] = []
    sheet: list[dict[str, str]] = []
    for bundle in bundles:
        record = by_id.get(bundle["unit_id"], {})
        stats = bundle.get("context_stats") or {}
        row: dict[str, Any] = {
            "run_id": bundle["run_id"],
            "unit_id": bundle["unit_id"],
            "battery_type": bundle["battery_type"],
            "preset": bundle["preset"],
            "model": bundle["model"],
            "trial": bundle["trial"],
            "turn_index": bundle.get("turn_index"),
            "error": bundle.get("error"),
            "answer_chars": len(bundle.get("answer") or ""),
            "tool_call_count": len(bundle.get("tool_calls") or []),
            "has_citation": bool(bundle.get("resolved_sources")),
            "ttft_ms": stats.get("ttft_ms"),
            "total_ms": stats.get("total_ms"),
            "input_tokens": stats.get("input_tokens"),
            "output_tokens": stats.get("output_tokens"),
            "est_cost_usd": stats.get("est_cost_usd"),
            "llm_calls": stats.get("llm_calls"),
            "context_tokens_approx": stats.get("context_tokens_approx"),
            "summary_messages": stats.get("summary_messages"),
            "cleared_tool_outputs": stats.get("cleared_tool_outputs"),
            "history_embedding_texts": stats.get("history_embedding_texts"),
            "history_embedding_chars": stats.get("history_embedding_chars"),
        }
        # Surface per-call-view middleware signals (sliding window, truncation,
        # reset, ...) when present; absent means that mechanism did not fire.
        for signal in COMPACTION_SIGNAL_KEYS:
            if signal in stats and signal not in row:
                row[signal] = stats[signal]
        if bundle.get("error"):
            grades.append(row)
            continue

        if bundle["battery_type"] == "singleturn":
            row.update(
                retrieval_metrics(
                    bundle, record.get("source_key"), record.get("lesson_url")
                )
            )
            row.update(
                citation_metrics(
                    bundle, record.get("source_key"), record.get("lesson_url")
                )
            )
            expected = record.get("expected_behavior", "")
            row["expected_behavior"] = expected
            row["behavior_heuristic"] = behavior_heuristic(expected, bundle)
            for point in record.get("key_points") or []:
                sheet.append(
                    sheet_row(
                        bundle=bundle,
                        item_type="key_point",
                        criterion=point,
                        reference=record.get("reference_answer") or "",
                    )
                )
            if expected:
                sheet.append(
                    sheet_row(
                        bundle=bundle,
                        item_type="behavior",
                        criterion=f"Did the tutor do the right thing for "
                        f"'{expected}'? (see README definitions)",
                        reference=record.get("reference_answer") or "",
                    )
                )

        elif bundle["battery_type"] == "sessions":
            probe = next(
                (
                    p
                    for p in record.get("probes", [])
                    if p["turn_index"] == bundle.get("turn_index")
                ),
                None,
            )
            row["is_probe"] = probe is not None
            if probe:
                row["probe_type"] = probe["probe_type"]
                row["probe_tier"] = probe.get("tier") or record.get("tier")
                # Compression context at probe time: a memory eval where the
                # triggers never fired measures nothing (README warning). Uses
                # the generalized check so new mechanisms count too.
                row["compaction_active"] = compaction_active(stats)
                sheet.append(
                    sheet_row(
                        bundle=bundle,
                        item_type=f"probe:{probe['probe_type']}",
                        criterion=f"Expected facts: {probe['expected_facts']} | "
                        f"Rule: {probe['check_note']}",
                    )
                )

        elif bundle["battery_type"] == "personas":
            result = grade_persona_question(record, bundle.get("answer") or "")
            row.update(
                {
                    "persona_id": (record.get("_persona") or {}).get("persona_id"),
                    "auto_pass": result["auto_pass"],
                    "anti_pattern_hits": result["anti_pattern_hits"],
                    "needs_judgment": result["needs_judgment"],
                }
            )
            if result["needs_judgment"] and not result["anti_pattern_hits"]:
                llm_checks = [
                    c["instruction"]
                    for c in record.get("checks", [])
                    if c["type"] == "llm"
                ]
                sheet.append(
                    sheet_row(
                        bundle=bundle,
                        item_type="persona_llm_check",
                        criterion=" | ".join(llm_checks),
                    )
                )

        elif bundle["battery_type"] == "replay":
            sheet.append(
                sheet_row(
                    bundle=bundle,
                    item_type="replay_reply",
                    criterion="Is this reply as helpful and correct as the real "
                    "staff reply? (binary)",
                    reference=record.get("reference_reply") or "",
                )
            )
        grades.append(row)
    return grades, sheet


def merge_handgrades(
    grades: list[dict[str, Any]], filled_csv: Path
) -> list[dict[str, Any]]:
    human: dict[str, list[dict[str, str]]] = {}
    with open(filled_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("grade", "").strip():
                human.setdefault(row["run_id"], []).append(row)
    for grade in grades:
        rows = human.get(grade["run_id"], [])
        if not rows:
            continue
        key_points = [r for r in rows if r["item_type"] == "key_point"]
        if key_points:
            passed = sum(1 for r in key_points if r["grade"].lower() == "pass")
            grade["key_points_passed"] = passed
            grade["key_points_total"] = len(key_points)
        for row in rows:
            verdict = row["grade"].strip().lower() == "pass"
            if row["item_type"] == "behavior":
                grade["behavior_pass"] = verdict
            elif row["item_type"].startswith("probe:"):
                grade["probe_pass"] = verdict
            elif row["item_type"] == "persona_llm_check":
                # Combines with the regex auto result (both must pass).
                grade["auto_pass"] = verdict and grade.get("auto_pass") is not False
            elif row["item_type"] == "replay_reply":
                grade["replay_pass"] = verdict
    return grades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run dir with bundles.jsonl")
    parser.add_argument("--handgrades", default="", help="Filled handgrade CSV")
    args = parser.parse_args()
    run_dir = Path(args.run)

    grades, sheet = grade_run(run_dir)
    write_jsonl(run_dir / "grades_auto.jsonl", grades)
    if sheet:
        with open(run_dir / "handgrade_sheet.csv", "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sheet[0]))
            writer.writeheader()
            writer.writerows(sheet)
    print(
        f"{len(grades)} rows auto-graded; {len(sheet)} human judgments pending "
        f"in handgrade_sheet.csv"
    )
    if args.handgrades:
        merged = merge_handgrades(grades, Path(args.handgrades))
        write_jsonl(run_dir / "grades_merged.jsonl", merged)
        print(f"Merged human grades -> {run_dir}/grades_merged.jsonl")


if __name__ == "__main__":
    main()
