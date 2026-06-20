"""Compaction study: compaction methods vs keeping everything in context.

The workshop question: when a long context is established up front and then
queried over many turns, is it better to KEEP it all every turn (cached) or to
COMPACT it (summarize / sliding window / selective retrieval / ...)? This is the
Axis-A (memory & context management) question -- NOT the Axis-B "whole doc vs
chunk" knob. We hold retrieval OFF so the agent must answer purely from whatever
each compaction method retained, which isolates the methods and ranks them
against keep-everything.

Setup: turn 0 loads the corpus's largest course lesson into the conversation;
turns 1..N ask questions about it; the same session runs under each memory
preset (the real app middlewares) on Gemini 2.5 Flash with no tools. We then
judge each answer against the full lesson and report cost / tokens / latency /
quality per preset.

  # 1. build the long-context session battery
  uv run --env-file .env -m evals.compaction_study build --questions 15
  # 2. run each preset (no tools, 2.5 Flash) -- see run_compaction_study.sh
  # 3. judge + report from the saved bundles
  uv run --env-file .env -m evals.compaction_study report --runs runs/compaction_*
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
from statistics import mean

from .common import load_jsonl, percentile
from .knowledge_compaction import LESSON_PATH, generate_questions, judge, load_lesson

logger = logging.getLogger("evals.compaction_study")

OUT_DIR = "data/compaction"
SESSION_ID = "longctx_master_ai"
BATTERY_PATH = f"{OUT_DIR}/longctx_session.jsonl"

TURN0_TEMPLATE = (
    "I'm going to ask you a series of questions about the following lesson. "
    "Read it carefully. Answer ONLY from this lesson; if a later question is not "
    "covered by it, say you don't have enough information. Reply 'Ready' once "
    "you've read it.\n\n=== LESSON ===\n{lesson}"
)

# Axis-A memory presets to compare: keep-everything vs each compaction method.
DEFAULT_PRESETS = [
    "full_history",
    "prod",
    "summarization_only",
    "sliding_window",
    "prompt_compression",
    "selective_retention",
    "incontext_history_retrieval",
    "aggressive",
]


def build_battery(lesson_path: str, n_questions: int, out_path: Path) -> None:
    lesson = load_lesson(lesson_path)
    questions = generate_questions(
        lesson, n_questions, Path(OUT_DIR) / "questions.jsonl"
    )
    turns = [TURN0_TEMPLATE.format(lesson=lesson)] + questions
    session = {"session_id": SESSION_ID, "turns": turns}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(session, ensure_ascii=False) + "\n")
    logger.info(
        "Wrote battery: 1 session, turn0 lesson + %d question turns -> %s",
        len(questions),
        out_path,
    )


def _preset_of(run_dir: Path, bundles: list[dict]) -> str:
    if bundles:
        return str(bundles[0].get("preset") or run_dir.name)
    return run_dir.name


def report(run_dirs: list[str], lesson_path: str, out_dir: Path) -> str:
    lesson = load_lesson(lesson_path)
    rows: list[dict] = []
    for rd in run_dirs:
        run_dir = Path(rd)
        bundles = load_jsonl(run_dir / "bundles.jsonl")
        if not bundles:
            continue
        preset = _preset_of(run_dir, bundles)
        # Question turns only (skip turn 0, the lesson-loading turn).
        q_turns = [
            b for b in bundles if (b.get("turn_index") or 0) >= 1 and not b.get("error")
        ]
        passed = 0
        for b in q_turns:
            ok, _ = judge(b["query"], b.get("answer") or "", lesson)
            passed += 1 if ok else 0
        stats = [b.get("context_stats") or {} for b in q_turns]
        in_toks = [s.get("input_tokens") for s in stats if s.get("input_tokens")]
        costs = [s.get("est_cost_usd") for s in stats if s.get("est_cost_usd") is not None]
        ttft = [s.get("ttft_ms") for s in stats if s.get("ttft_ms")]
        compacted = sum(
            1 for s in stats if s.get("summary_messages") or s.get("dropped_messages")
        )
        rows.append(
            {
                "preset": preset,
                "n": len(q_turns),
                "judge_pass": passed,
                "pass_rate": passed / len(q_turns) if q_turns else 0.0,
                "mean_in_tok": mean(in_toks) if in_toks else 0,
                "cum_cost": sum(costs) if costs else 0.0,
                "mean_cost": mean(costs) if costs else 0.0,
                "ttft_p50_ms": percentile(ttft, 50) if ttft else 0,
                "compacted_turns": compacted,
            }
        )

    rows.sort(key=lambda r: -r["pass_rate"])
    lines = [
        "# Compaction study report",
        "",
        f"Lesson: `{lesson_path}` | model: gemini-2.5-flash | no tools "
        "(answer from retained context only) | 1 session.",
        "",
        "| preset | judge pass | mean in tok/turn | session $ | TTFT p50 ms | compacted turns |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['preset']} | {r['judge_pass']}/{r['n']} ({r['pass_rate']:.0%}) | "
            f"{r['mean_in_tok']:.0f} | ${r['cum_cost']:.4f} | {r['ttft_p50_ms']:.0f} | "
            f"{r['compacted_turns']}/{r['n']} |"
        )
    out = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build the long-context session battery.")
    b.add_argument("--lesson", default=LESSON_PATH)
    b.add_argument("--questions", type=int, default=15)
    b.add_argument("--out", default=BATTERY_PATH)

    r = sub.add_parser("report", help="Judge + aggregate per-preset results.")
    r.add_argument("--runs", nargs="+", required=True)
    r.add_argument("--lesson", default=LESSON_PATH)
    r.add_argument("--out", default="runs/compaction_report")

    args = parser.parse_args()
    if args.cmd == "build":
        build_battery(args.lesson, args.questions, Path(args.out))
    elif args.cmd == "report":
        runs = [d for pat in args.runs for d in glob.glob(pat)]
        print(report(runs, args.lesson, Path(args.out)))


if __name__ == "__main__":
    main()
