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

# Judge-prompt budget for a faithfulness row's evidence. Bundles capture each
# tool output at run_battery.TOOL_OUTPUT_MAX_CHARS = 40_000 chars (the KB
# shell's own per-call output cap), so one turn's complete evidence can span
# several 40k outputs; 200k covers ~5 max-size calls. This is a GATE, not a
# slice: quality_sheet_rows emits a faithfulness row only when the FULL
# assembled evidence fits (see evidence_fits), so the judge never grades
# silently truncated grounding (the evals.md F23/F24 blind-signal class).
# Worst case per row is ~200k chars (~50k tokens) of judge prompt; typical
# turns are far smaller, and anything larger is excluded, not sliced.
FAITHFULNESS_EVIDENCE_MAX = 200_000

# Holistic = the "would staff approve sending this?" gate (evals/background.md
# Layer 4). Emitted for every single-turn answer; the full rubric lives in
# evals/judge.py RUBRICS["holistic"], this is the at-a-glance criterion.
HOLISTIC_CRITERION = (
    "Would an experienced course tutor approve sending this answer to the "
    "student? Pass only if it is correct, grounded in the right material, and "
    "genuinely helps them learn (builds understanding), with appropriate scope "
    "and tone. (see README / judge rubric)"
)
# Faithfulness = is the answer grounded in the evidence the tutor actually
# retrieved (catches hallucination on corpus questions). Emitted only when the
# turn produced retrieval/KB evidence; rubric in judge.py RUBRICS["faithfulness"].
FAITHFULNESS_CRITERION = (
    "Are the answer's substantive factual/technical claims supported by the "
    "retrieved evidence shown below? Fail on fabricated APIs, parameters, "
    "versions, citations, or course/lesson specifics not in the evidence."
)


def faithfulness_evidence(bundle: dict[str, Any]) -> str:
    """Concatenate the retrieval/KB evidence the tutor saw, for grounding grades.

    Joins each retrieval/KB tool call's command + output (and, for
    `retrieve_tutor_context`, the ranked source titles/URLs) so the judge can
    check the answer's claims against exactly what the tools returned. Returns
    the evidence UNTRUNCATED: whether it fits the judge prompt is a separate
    gate (evidence_fits), so an oversized turn is excluded from faithfulness
    grading rather than judged against a silent slice. Empty when the turn used
    no retrieval/KB tool (e.g. answer_general), which is the signal to skip the
    faithfulness row for that case.
    """
    parts: list[str] = []
    for call in bundle.get("tool_calls") or []:
        name = call.get("tool_name")
        if name not in RETRIEVAL_TOOLS:
            continue
        header = f"[{name}] {(call.get('args_text') or '').strip()}".strip()
        body = (call.get("output_text") or "").strip()
        matches = call.get("matches") or []
        if matches:
            cites = "; ".join(
                f"{m.get('title', '')} <{m.get('url', '')}>" for m in matches[:10]
            )
            body = f"{body}\nSOURCES: {cites}".strip()
        if header or body:
            parts.append(f"{header}\n{body}".strip())
    return "\n\n".join(parts)


def evidence_is_complete(bundle: dict[str, Any]) -> bool:
    """True if every retrieval/KB tool output was captured in full.

    Bundles cap each tool output at ``run_battery.TOOL_OUTPUT_MAX_CHARS`` but
    preserve the true length in ``output_chars``. When a KB-browse output was
    truncated, the judge sees only a slice of what the agent actually read, so a
    faithfulness grade would measure capture-completeness, not grounding (the
    same blind-signal class as evals.md F23/F24). Faithfulness is therefore
    emitted ONLY when nothing the agent grounded on was truncated; after the
    bundle cap is raised and a run is re-recorded, this returns True and the
    column populates automatically.
    """
    for call in bundle.get("tool_calls") or []:
        if call.get("tool_name") not in RETRIEVAL_TOOLS:
            continue
        full = call.get("output_chars")
        kept = len(call.get("output_text") or "")
        if isinstance(full, int) and full > kept:
            return False
    return True


def evidence_fits(evidence: str, max_chars: int = FAITHFULNESS_EVIDENCE_MAX) -> bool:
    """True if the assembled evidence fits the faithfulness judge-prompt budget.

    Companion gate to ``evidence_is_complete``: that one guarantees nothing was
    lost at CAPTURE time, this one guarantees nothing would be lost at JUDGE
    time. A turn whose complete evidence still exceeds the cap is excluded from
    faithfulness grading instead of being judged on truncated grounding.
    """
    return len(evidence) <= max_chars


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
    ``run_battery.TOOL_OUTPUT_MAX_CHARS`` (40k) per call, so a path past the cap
    is missed; a broad ``rg`` can surface many sources at once (an over-count
    toward "shown", not "used").
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
    reference_max: int = 2000,
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
        "reference": reference[:reference_max],
        "grade": "",
        "note": "",
    }


def quality_sheet_rows(
    bundle: dict[str, Any], reference: str = ""
) -> list[dict[str, str]]:
    """Holistic + faithfulness sheet rows for one answered turn.

    Shared by single-turn and session turns: holistic (would staff approve this
    answer?) is emitted for every answered turn; faithfulness only when the turn
    produced retrieval/KB evidence to ground against. ``reference`` is the staff
    answer for single-turn cases (context for holistic) and empty for session
    turns, which have no per-turn gold answer.
    """
    rows = [
        sheet_row(
            bundle=bundle,
            item_type="holistic",
            criterion=HOLISTIC_CRITERION,
            reference=reference,
        )
    ]
    # Faithfulness only when the judge can see 100% of what the agent grounded
    # on: (1) every retrieval/KB output was captured in full at record time
    # (bundles recorded under the old 6k tool-output cap fail this; re-record
    # at the 40k cap to re-enable), and (2) the full assembled evidence fits
    # the judge-prompt budget. A turn failing either is excluded rather than
    # judged on silently truncated evidence, which would score
    # capture-completeness, not grounding (evals.md F23/F24 class).
    evidence = faithfulness_evidence(bundle)
    if evidence and evidence_is_complete(bundle) and evidence_fits(evidence):
        rows.append(
            sheet_row(
                bundle=bundle,
                item_type="faithfulness",
                criterion=FAITHFULNESS_CRITERION,
                reference=evidence,
                reference_max=FAITHFULNESS_EVIDENCE_MAX,
            )
        )
    return rows


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
            "time_to_first_token_ms": stats.get("time_to_first_token_ms"),
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
            # Holistic (whole-answer staff-approval) + faithfulness (grounded in
            # retrieved evidence); staff answer is context for holistic.
            sheet.extend(
                quality_sheet_rows(bundle, record.get("reference_answer") or "")
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
            # Holistic + faithfulness on every answered session turn (no per-turn
            # gold answer, so no reference for holistic). Lets us see whether
            # answer quality degrades on late turns under compaction.
            if bundle.get("answer"):
                sheet.extend(quality_sheet_rows(bundle))

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
            # Holistic + faithfulness on persona answers too (full quality
            # coverage across batteries); the persona profile is the implicit
            # reference, so no staff reference text here.
            if bundle.get("answer"):
                sheet.extend(quality_sheet_rows(bundle))

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
        # Grade source, for honest report labels: the LLM judge prefixes its
        # note with "[judge...]"; human-filled sheets do not. (Part C runs were
        # judge-filled, but the report hardcoded "(human grade)".)
        sources = {
            "judge" if (r.get("note") or "").lstrip().startswith("[judge") else "human"
            for r in rows
        }
        grade["grade_source"] = sources.pop() if len(sources) == 1 else "mixed"

        def _src(row: dict[str, str]) -> str:
            # Per-metric provenance so a run with human key-points but judge-filled
            # holistic/faithfulness still reports each row's true grader (the
            # overall grade_source above goes "mixed" in that case).
            return (
                "judge"
                if (row.get("note") or "").lstrip().startswith("[judge")
                else "human"
            )

        key_points = [r for r in rows if r["item_type"] == "key_point"]
        if key_points:
            passed = sum(1 for r in key_points if r["grade"].lower() == "pass")
            grade["key_points_passed"] = passed
            grade["key_points_total"] = len(key_points)
            kp_sources = {_src(r) for r in key_points}
            grade["key_points_source"] = (
                kp_sources.pop() if len(kp_sources) == 1 else "mixed"
            )
        for row in rows:
            verdict = row["grade"].strip().lower() == "pass"
            if row["item_type"] == "behavior":
                grade["behavior_pass"] = verdict
                grade["behavior_source"] = _src(row)
            elif row["item_type"].startswith("probe:"):
                grade["probe_pass"] = verdict
                grade["probe_source"] = _src(row)
            elif row["item_type"] == "persona_llm_check":
                # Combines with the regex auto result (both must pass).
                grade["auto_pass"] = verdict and grade.get("auto_pass") is not False
                grade["persona_source"] = _src(row)
            elif row["item_type"] == "replay_reply":
                grade["replay_pass"] = verdict
                grade["replay_source"] = _src(row)
            elif row["item_type"] == "holistic":
                grade["holistic_pass"] = verdict
                grade["holistic_source"] = _src(row)
            elif row["item_type"] == "faithfulness":
                grade["faithfulness_pass"] = verdict
                grade["faithfulness_source"] = _src(row)
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
