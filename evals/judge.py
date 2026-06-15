"""LLM judge: auto-fill the human-grading sheet, and validate it against humans.

The judge is the scalable stand-in for hand grading. It reads a run's
``handgrade_sheet.csv`` (the rows ``evals.grade`` left for a person), grades each
with a strong Anthropic model under the same per-item-type rubric a human uses,
and writes ``judge_filled.csv`` in the identical format ``--handgrades`` expects.
So the pipeline is unchanged: ``run_battery -> grade -> judge -> grade
--handgrades -> report``.

A judge number is only reportable after the judge is validated against held-out
human labels. ``validate`` does exactly that: it joins the judge's verdicts to a
human-filled sheet and reports the true-positive and true-negative rates, gating
at >=90% on both (the project methodology in evals.md).

  # grade the pending rows with the judge
  uv run -m evals.judge run --sheet runs/<exp>/handgrade_sheet.csv \
      --out runs/<exp>/judge_filled.csv
  # then fold them in exactly like human grades
  uv run -m evals.grade --run runs/<exp> --handgrades runs/<exp>/judge_filled.csv

  # validate against a human-filled sheet (truth = human)
  uv run -m evals.judge validate --judge runs/<exp>/judge_filled.csv \
      --human runs/<exp>/handgrade_filled.csv

The judge sees only question / answer / criterion / reference (never the preset),
so it cannot favor a memory strategy.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import collections
from pathlib import Path
from typing import Any

# Strong judge by default: validation runs on a small calibration set, and the
# >=90% TP/TN bar is easier to clear with the strongest model. Downgrade to a
# cheaper Anthropic model for bulk Part C grading only if it still clears.
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
CONCURRENCY = 6

PREAMBLE = (
    "You are grading a BLINDED answer from an AI tutor for a course platform "
    "(applied AI, LLMs, RAG, Python). You see only question, answer, criterion, "
    "and an optional staff reference; you do NOT know which system produced the "
    "answer and must not try to infer it. Decide grade=pass|fail, confidence="
    "high|low, and a one-sentence reason. An EMPTY answer fails. Grade ONLY what "
    "the criterion asks, not overall polish. Always give a definite pass/fail; "
    "use confidence=low only for genuinely ambiguous calls."
)

# Same rubrics the human-grading workflow used, keyed by handgrade item_type.
RUBRICS: dict[str, str] = {
    "probe": (
        "ITEM TYPE = session memory probe. The criterion gives 'Expected facts: "
        "[...]' and a 'Rule: ...'. Grade STRICTLY by the Rule. Pass only if the "
        "answer uses/reflects the expected fact(s) as the rule demands. "
        "fact_recall: must use the earlier-stated fact. fact_update: must use the "
        "NEW fact; the OLD pre-update value = fail. anaphora / "
        "anaphora_consistency: recommending the correct, consistent approach is "
        "enough; it need NOT name the specific entity; fail only if it recommends "
        "the wrong approach, contradicts the tutor's earlier advice, or treats an "
        "already-used framework as a new option to switch to. preference_"
        "compliance: must honor the stated preference and avoid the forbidden "
        "patterns in the rule."
    ),
    "persona_llm_check": (
        "ITEM TYPE = persona check. The criterion is a pass/fail instruction "
        "describing what a good answer must do for this user's profile. Pass if "
        "the answer satisfies it; fail otherwise."
    ),
    "key_point": (
        "ITEM TYPE = key point. The criterion is ONE atomic claim a good answer "
        "should contain (the staff 'reference' shows what is correct). PASS if the "
        "answer clearly conveys OR clearly implies the claim's substance (wording "
        "need not match). FAIL only if absent, wrong, contradicted, or merely "
        "weak/ambiguous implication."
    ),
    "behavior": (
        "ITEM TYPE = behavior routing. The criterion asks if the tutor did the "
        "right KIND of thing. redirect_to_support = empathize and point to human "
        "support for a platform/billing/account issue WITHOUT inventing platform "
        "fixes (a long technical fix instead of a redirect = fail); "
        "acknowledge_feedback = thank/acknowledge without promising fixes; "
        "answer_from_corpus = ground the answer in course/docs content; "
        "answer_general = answer from general knowledge without fake citations. "
        "The 'reference' shows the gold staff behavior. Pass if the BEHAVIOR "
        "matches the expected kind even if details differ."
    ),
    "replay_reply": (
        "ITEM TYPE = replay. The criterion asks whether this reply is about as "
        "helpful and correct as the real staff reply (shown as 'reference'). Pass "
        "if it is comparably correct and useful; fail if materially worse, wrong, "
        "or unhelpful."
    ),
}


def rubric_for(item_type: str) -> str:
    if item_type.startswith("probe:"):
        return RUBRICS["probe"]
    return RUBRICS.get(item_type, RUBRICS["key_point"])


def _verdict_model() -> type:
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        grade: str = Field(description="'pass' or 'fail'")
        confidence: str = Field(description="'high' or 'low'")
        reason: str = Field(description="one short sentence")

    return Verdict


def build_prompt(row: dict[str, str]) -> str:
    parts = [
        f"RUBRIC:\n{rubric_for(row['item_type'])}",
        f"\nQUESTION:\n{row.get('question', '')}",
        f"\nCRITERION:\n{row.get('criterion', '')}",
    ]
    if (row.get("reference") or "").strip():
        parts.append(
            f"\nSTAFF REFERENCE (context only, never shown to the tutor):\n{row['reference']}"
        )
    parts.append(f"\nANSWER TO GRADE:\n{row.get('answer') or '(EMPTY)'}")
    return "\n".join(parts)


async def grade_row(
    model: Any, row: dict[str, str], sem: asyncio.Semaphore
) -> dict[str, str]:
    """Return the row with grade/note filled. Empty answers short-circuit to fail."""
    if not (row.get("answer") or "").strip():
        row["grade"] = "fail"
        row["note"] = "[judge:high] empty answer"
        return row
    async with sem:
        for attempt in range(3):
            try:
                verdict = await model.ainvoke(
                    [("system", PREAMBLE), ("human", build_prompt(row))]
                )
                grade = str(verdict.grade).strip().lower()
                row["grade"] = "pass" if grade == "pass" else "fail"
                row["note"] = (
                    f"[judge:{str(verdict.confidence).strip().lower()}] {verdict.reason}"
                )
                return row
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                if attempt == 2:
                    row["grade"] = ""
                    row["note"] = f"[judge:error] {type(exc).__name__}: {exc}"
                    return row
                await asyncio.sleep(2 * (attempt + 1))
    return row


async def run_sheet(sheet: Path, out: Path, model_name: str, concurrency: int) -> None:
    from langchain_anthropic import ChatAnthropic

    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {sheet}")
    # No temperature: newer Anthropic models (e.g. claude-opus-4-8) reject it;
    # structured output keeps verdicts well-formed without it.
    model = ChatAnthropic(model=model_name).with_structured_output(_verdict_model())
    sem = asyncio.Semaphore(concurrency)
    graded = await asyncio.gather(*(grade_row(model, dict(r), sem) for r in rows))
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(graded)
    decided = [r for r in graded if r["grade"]]
    errors = sum(1 for r in graded if not r["grade"])
    npass = sum(1 for r in decided if r["grade"] == "pass")
    print(
        f"judged {len(graded)} rows ({model_name}): {npass} pass / "
        f"{len(decided) - npass} fail, {errors} error -> {out}"
    )


def validate(judge_csv: Path, human_csv: Path) -> None:
    """Truth = human. Report TP/TN rates overall and per item type; gate at 90%."""
    judge = {
        r["sheet_row_id"]: r for r in csv.DictReader(open(judge_csv, encoding="utf-8"))
    }
    human = {
        r["sheet_row_id"]: r
        for r in csv.DictReader(open(human_csv, encoding="utf-8"))
        if (r.get("grade") or "").strip()
    }
    pairs = [
        (
            human[k]["grade"].strip().lower(),
            judge[k]["grade"].strip().lower(),
            judge[k]["item_type"],
        )
        for k in human
        if k in judge and (judge[k].get("grade") or "").strip()
    ]
    if not pairs:
        raise SystemExit("No overlapping graded rows between judge and human sheets.")

    def report(label: str, rows: list[tuple[str, str, str]]) -> bool:
        tp = sum(1 for h, j, _ in rows if h == "pass" and j == "pass")
        fn = sum(1 for h, j, _ in rows if h == "pass" and j == "fail")
        tn = sum(1 for h, j, _ in rows if h == "fail" and j == "fail")
        fp = sum(1 for h, j, _ in rows if h == "fail" and j == "pass")
        tpr = tp / (tp + fn) if (tp + fn) else None
        tnr = tn / (tn + fp) if (tn + fp) else None
        agree = (tp + tn) / len(rows)
        ok = (tpr is None or tpr >= 0.9) and (tnr is None or tnr >= 0.9)

        def pct(x: float | None) -> str:
            return "  n/a" if x is None else f"{x:5.0%}"

        print(
            f"  {label:22} n={len(rows):3}  agree={agree:4.0%}  "
            f"TPR={pct(tpr)}  TNR={pct(tnr)}  {'PASS' if ok else 'FAIL <90%'}"
        )
        return ok

    print(f"Judge vs human ({judge_csv.name} vs {human_csv.name}):")
    overall_ok = report("overall", pairs)
    by_type: dict[str, list] = collections.defaultdict(list)
    for h, j, it in pairs:
        by_type[it.split(":")[0] if it.startswith("probe") else it].append((h, j, it))
    for it in sorted(by_type):
        report(it, by_type[it])
    print(
        f"\nGate (>=90% TPR AND TNR overall): "
        f"{'VALIDATED' if overall_ok else 'NOT validated — refine judge prompt or hand-grade'}"
    )


def main() -> None:
    # The judge does not import app code, so load .env ourselves for the
    # Anthropic key (app.config does the same on import for run_battery).
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="grade a handgrade_sheet.csv with the judge")
    r.add_argument("--sheet", required=True)
    r.add_argument("--out", default="")
    r.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    r.add_argument("--concurrency", type=int, default=CONCURRENCY)
    v = sub.add_parser("validate", help="compare judge verdicts to human labels")
    v.add_argument("--judge", required=True)
    v.add_argument("--human", required=True)
    args = parser.parse_args()

    if args.cmd == "run":
        sheet = Path(args.sheet)
        out = Path(args.out) if args.out else sheet.with_name("judge_filled.csv")
        asyncio.run(run_sheet(sheet, out, args.model, args.concurrency))
    else:
        validate(Path(args.judge), Path(args.human))


if __name__ == "__main__":
    main()
